#!/usr/bin/env bash
# patch_03a: Dataset — базовое хранилище (per-column Arrow IPC, синхронные чанки).
set -euo pipefail

# ---------- inline require_clean_tree ----------
require_clean_tree() {
  local bad_tracked bad_untracked
  bad_tracked="$(git status --porcelain --untracked-files=no)"
  if [[ -n "$bad_tracked" ]]; then
    echo "ERROR: tracked files have uncommitted changes:" >&2
    echo "$bad_tracked" >&2
    return 1
  fi
  bad_untracked=""
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    case "$f" in
      scripts/*|.backups/*|__pycache__/*|*/__pycache__/*|*.pyc|.pytest_cache/*) : ;;
      *) bad_untracked+="  $f"$'\n' ;;
    esac
  done < <(git ls-files --others --exclude-standard)
  if [[ -n "$bad_untracked" ]]; then
    echo "ERROR: untracked outside scripts/, .backups/, caches:" >&2
    echo "$bad_untracked" >&2
    return 1
  fi
  return 0
}
require_clean_tree || exit 1

CUR="$(git rev-parse --abbrev-ref HEAD)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-03a-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_03a"

BR="patch/03a-dataset"
git rev-parse --verify "${BR}" >/dev/null 2>&1 && {
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "switch or delete ${BR}"; exit 1; }
  echo "Resuming on ${BR}"
} || git checkout -b "${BR}"

mkdir -p cadw/dataset
cat > cadw/dataset/__init__.py <<'EOF'
from .dataset import Dataset

__all__ = ["Dataset"]
EOF

cat > cadw/dataset/dataset.py <<'PY'
"""
Dataset — базовое хранилище CADW.

Инвариант: у всех колонок одинаковое число stored batches и одинаковые
num_rows в каждом batch[i] (chunk boundaries). Нарушение ловится при записи.

Layout:
    base_dir/
    ├── columns/
    │   ├── <col1>.arrow     # Arrow IPC File, N record batches
    │   └── <col2>.arrow
    └── dataset_meta.json

Arrow IPC File (не Stream) выбран потому, что позволяет random-access
к batch[i] через ipc.open_file(path).get_batch(i).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

_COLUMNS_DIR = "columns"
_META_FILE = "dataset_meta.json"


class Dataset:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self._columns_dir = self.base_dir / _COLUMNS_DIR
        self._meta_path = self.base_dir / _META_FILE
        self._schema: pa.Schema | None = None
        self._column_paths: dict[str, Path] = {}
        self._chunk_boundaries: list[int] = []
        self._total_rows: int = 0
        self._load_meta_if_exists()

    # ---------------- factories ----------------

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str | Path,
        base_dir: str | Path,
        chunk_size: int = 10_000,
    ) -> "Dataset":
        parquet_path = Path(parquet_path)
        pf = pq.ParquetFile(parquet_path)
        schema = pf.schema_arrow

        ds = cls(base_dir)
        ds._init_empty(schema)

        writers: dict[str, tuple[ipc.RecordBatchFileWriter, pa.OSFile]] = {}
        try:
            for field in schema:
                p = ds._columns_dir / f"{field.name}.arrow"
                sink = pa.OSFile(str(p), "wb")
                w = ipc.new_file(sink, pa.schema([field]))
                writers[field.name] = (w, sink)
                ds._column_paths[field.name] = p

            for rb in pf.iter_batches(batch_size=chunk_size):
                for field in schema:
                    single = pa.RecordBatch.from_arrays(
                        [rb.column(field.name)], names=[field.name]
                    )
                    writers[field.name][0].write_batch(single)
                ds._chunk_boundaries.append(rb.num_rows)
                ds._total_rows += rb.num_rows
        finally:
            for w, sink in writers.values():
                w.close()
                sink.close()

        ds._save_meta()
        return ds

    @classmethod
    def from_batches(
        cls,
        schema: pa.Schema,
        batches_iter: Iterable[pa.RecordBatch],
        base_dir: str | Path,
    ) -> "Dataset":
        ds = cls(base_dir)
        ds._init_empty(schema)

        writers: dict[str, tuple[ipc.RecordBatchFileWriter, pa.OSFile]] = {}
        try:
            for field in schema:
                p = ds._columns_dir / f"{field.name}.arrow"
                sink = pa.OSFile(str(p), "wb")
                w = ipc.new_file(sink, pa.schema([field]))
                writers[field.name] = (w, sink)
                ds._column_paths[field.name] = p

            for rb in batches_iter:
                if rb.schema != schema:
                    rb = rb.select(schema.names).cast(schema)
                for field in schema:
                    single = pa.RecordBatch.from_arrays(
                        [rb.column(field.name)], names=[field.name]
                    )
                    writers[field.name][0].write_batch(single)
                ds._chunk_boundaries.append(rb.num_rows)
                ds._total_rows += rb.num_rows
        finally:
            for w, sink in writers.values():
                w.close()
                sink.close()

        ds._save_meta()
        return ds

    # ---------------- write ----------------

    def add_column(self, name: str, batches_iter: Iterable[pa.RecordBatch]) -> None:
        """Добавить колонку. batches_iter должен давать batches с num_rows,
        совпадающими с self._chunk_boundaries (i-й batch → boundary[i]).
        """
        if self._schema is None:
            raise RuntimeError("Dataset is empty; use from_batches/from_parquet")
        if name in self._column_paths:
            raise ValueError(f"column {name!r} already exists")

        target = self._columns_dir / f"{name}.arrow"
        tmp = target.with_suffix(".tmp.arrow")
        sink = pa.OSFile(str(tmp), "wb")
        writer: ipc.RecordBatchFileWriter | None = None
        field: pa.Field | None = None
        i = 0
        try:
            for rb in batches_iter:
                col = rb.column(0) if rb.num_columns == 1 else rb.column(name)
                if writer is None:
                    field = pa.field(name, col.type)
                    writer = ipc.new_file(sink, pa.schema([field]))
                if i >= len(self._chunk_boundaries):
                    raise ValueError(
                        f"add_column: too many batches "
                        f"(expected {len(self._chunk_boundaries)}, got more)"
                    )
                expected = self._chunk_boundaries[i]
                if rb.num_rows != expected:
                    raise ValueError(
                        f"add_column: batch {i} has {rb.num_rows} rows, "
                        f"expected {expected} (chunk boundaries must match)"
                    )
                single = pa.RecordBatch.from_arrays([col], names=[name])
                writer.write_batch(single)
                i += 1
        finally:
            if writer is not None:
                writer.close()
            sink.close()

        if i != len(self._chunk_boundaries):
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"add_column: got {i} batches, expected {len(self._chunk_boundaries)}"
            )

        tmp.rename(target)
        self._column_paths[name] = target
        self._schema = pa.schema(list(self._schema) + [field])
        self._save_meta()

    # ---------------- read ----------------

    def iter_batches(
        self,
        batch_size: int | None = None,
        columns: Sequence[str] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Yield RecordBatch'и.

        batch_size=None — yield stored batches as-is (одна итерация = одна
        итерация по chunk_boundaries).
        batch_size=N — re-chunk. N > stored: concat; N < stored: split.
        columns — подмножество (порядок важен).
        """
        if not self._column_paths:
            return
        cols = list(columns) if columns is not None else list(self._column_paths.keys())
        for c in cols:
            if c not in self._column_paths:
                raise KeyError(f"column {c!r} not found")

        readers = {c: ipc.open_file(self._column_paths[c]) for c in cols}
        n_batches = readers[cols[0]].num_record_batches

        if batch_size is None:
            for i in range(n_batches):
                arrays = [readers[c].get_batch(i).column(c) for c in cols]
                yield pa.RecordBatch.from_arrays(arrays, names=cols)
            return

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        accum: dict[str, list[pa.Array]] = {c: [] for c in cols}
        accum_n = 0
        for i in range(n_batches):
            stored = {c: readers[c].get_batch(i) for c in cols}
            n = stored[cols[0]].num_rows
            j = 0
            while j < n:
                take = min(batch_size - accum_n, n - j)
                for c in cols:
                    accum[c].append(stored[c].column(c).slice(j, take))
                accum_n += take
                j += take
                if accum_n == batch_size:
                    arrays = [pa.concat_arrays(accum[c]) for c in cols]
                    yield pa.RecordBatch.from_arrays(arrays, names=cols)
                    accum = {c: [] for c in cols}
                    accum_n = 0
        if accum_n > 0:
            arrays = [pa.concat_arrays(accum[c]) for c in cols]
            yield pa.RecordBatch.from_arrays(arrays, names=cols)

    # ---------------- properties ----------------

    @property
    def columns(self) -> list[str]:
        return list(self._column_paths.keys())

    @property
    def schema(self) -> pa.Schema | None:
        return self._schema

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def chunk_boundaries(self) -> list[int]:
        return list(self._chunk_boundaries)

    # ---------------- helpers ----------------

    def _init_empty(self, schema: pa.Schema) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._columns_dir.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._column_paths = {}
        self._chunk_boundaries = []
        self._total_rows = 0

    def _save_meta(self) -> None:
        assert self._schema is not None
        schema_bytes = (
            pa.ipc.serialize_schema(self._schema).serialize().to_pybytes()
        )
        meta = {
            "schema_b64": base64.b64encode(schema_bytes).decode(),
            "column_files": {k: v.name for k, v in self._column_paths.items()},
            "chunk_boundaries": self._chunk_boundaries,
            "total_rows": self._total_rows,
        }
        self._meta_path.write_text(json.dumps(meta, indent=2))

    def _load_meta_if_exists(self) -> None:
        if not self._meta_path.exists():
            return
        meta = json.loads(self._meta_path.read_text())
        schema_bytes = base64.b64decode(meta["schema_b64"])
        self._schema = pa.ipc.read_schema(pa.BufferReader(schema_bytes))
        self._column_paths = {
            k: self._columns_dir / v for k, v in meta["column_files"].items()
        }
        self._chunk_boundaries = list(meta["chunk_boundaries"])
        self._total_rows = int(meta["total_rows"])

    def __repr__(self) -> str:
        return (
            f"Dataset(columns={self.columns}, rows={self._total_rows}, "
            f"batches={len(self._chunk_boundaries)})"
        )
PY

# ---------- export ----------
python3 - <<'PY'
p = "cadw/__init__.py"
s = open(p).read()
if "from .dataset import Dataset" not in s:
    s = s.replace(
        "from .core.memory_guard import MemoryGuard, MemoryLimitExceeded",
        "from .core.memory_guard import MemoryGuard, MemoryLimitExceeded\n"
        "from .dataset import Dataset",
        1,
    )
    s = s.replace(
        '"MemoryGuard", "MemoryLimitExceeded",',
        '"MemoryGuard", "MemoryLimitExceeded", "Dataset",',
        1,
    )
    open(p, "w").write(s)
    print("patched cadw/__init__.py: Dataset exported")
PY

# ---------- tests ----------
cat > tests/test_dataset.py <<'EOF'
"""Тесты Dataset (патч 03a): базовое хранилище per-column Arrow IPC."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import Dataset


def batches_of(table: pa.Table, chunk: int):
    for i in range(0, table.num_rows, chunk):
        yield table.slice(i, chunk).to_batches()[0]


def make_table(n: int = 20) -> pa.Table:
    return pa.table({
        "cat": [f"g{i % 5}" for i in range(n)],
        "val": list(range(n)),
    })


# ---------- from_batches / roundtrip ----------

def test_from_batches_creates_layout(tmp_path):
    ds = Dataset.from_batches(make_table(20).schema, batches_of(make_table(20), 5), tmp_path / "ds")
    assert set(ds.columns) == {"cat", "val"}
    assert ds.total_rows == 20
    assert (tmp_path / "ds" / "columns" / "cat.arrow").exists()
    assert (tmp_path / "ds" / "columns" / "val.arrow").exists()
    assert (tmp_path / "ds" / "dataset_meta.json").exists()


def test_from_batches_roundtrip(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    got = pa.Table.from_batches(list(ds.iter_batches()))
    assert got.num_rows == 20
    assert got.column("cat").to_pylist() == t.column("cat").to_pylist()
    assert got.column("val").to_pylist() == t.column("val").to_pylist()


def test_chunk_boundaries(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 7), tmp_path / "ds")
    # 20 rows, chunk=7 → batches of 7,7,6
    assert ds.chunk_boundaries == [7, 7, 6]


# ---------- from_parquet ----------

def test_from_parquet(tmp_path):
    t = make_table(20)
    src = tmp_path / "src.parquet"
    pq.write_table(t, src)
    ds = Dataset.from_parquet(src, tmp_path / "ds", chunk_size=5)
    assert set(ds.columns) == {"cat", "val"}
    assert ds.total_rows == 20
    assert ds.chunk_boundaries == [5, 5, 5, 5]
    got = pa.Table.from_batches(list(ds.iter_batches()))
    assert got.column("val").to_pylist() == t.column("val").to_pylist()


# ---------- iter_batches ----------

def test_iter_batches_default_yields_stored(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    sizes = [b.num_rows for b in ds.iter_batches()]
    assert sizes == [5, 5, 5, 5]


def test_iter_batches_rechunk_smaller(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    sizes = [b.num_rows for b in ds.iter_batches(batch_size=3)]
    # 20 rows → 3,3,3,3,3,3,2
    assert sizes == [3, 3, 3, 3, 3, 3, 2]
    total = sum(sizes)
    assert total == 20


def test_iter_batches_rechunk_larger(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    sizes = [b.num_rows for b in ds.iter_batches(batch_size=12)]
    assert sizes == [12, 8]


def test_iter_batches_columns_subset(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    for b in ds.iter_batches(columns=["val"]):
        assert b.schema.names == ["val"]


def test_iter_batches_missing_column_raises(tmp_path):
    t = make_table(10)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    with pytest.raises(KeyError):
        list(ds.iter_batches(columns=["nope"]))


# ---------- add_column ----------

def test_add_column_ok(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    new_vals = pa.table({"score": [i * 10 for i in range(20)]})
    ds.add_column("score", batches_of(new_vals, 5))
    assert "score" in ds.columns
    got = pa.Table.from_batches(list(ds.iter_batches()))
    assert got.column("score").to_pylist() == list(range(0, 200, 10))


def test_add_column_wrong_boundary_raises(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    bad = pa.table({"score": list(range(20))})
    with pytest.raises(ValueError, match="chunk boundaries"):
        ds.add_column("score", batches_of(bad, 7))  # 7,7,6 != 5,5,5,5


def test_add_column_too_few_batches_raises(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    short = pa.table({"score": list(range(10))})
    with pytest.raises(ValueError):
        ds.add_column("score", batches_of(short, 5))


# ---------- persistence ----------

def test_reload_from_disk(tmp_path):
    t = make_table(20)
    ds = Dataset.from_batches(t.schema, batches_of(t, 5), tmp_path / "ds")
    ds.add_column("score", batches_of(pa.table({"score": list(range(20))}), 5))

    ds2 = Dataset(tmp_path / "ds")
    assert set(ds2.columns) == {"cat", "val", "score"}
    assert ds2.total_rows == 20
    assert ds2.chunk_boundaries == [5, 5, 5, 5]
    got = pa.Table.from_batches(list(ds2.iter_batches()))
    assert got.column("score").to_pylist() == list(range(20))
EOF

# ---------- DECISIONS §11.1 / §12.2 ----------
python3 - <<'PY'
p = "DECISIONS.md"
s = open(p).read()
if "| `Dataset` (03a)" not in s:
    s = s.replace(
        "| must-fix §11.4 (02a, 02b1, 02b2) | ✅ | 28 |",
        "| must-fix §11.4 (02a, 02b1, 02b2) | ✅ | 28 |\n"
        "| `Dataset` (03a, per-column Arrow IPC, sync batches) | ✅ | 13 |",
        1,
    )
if "§12.2" in s and "3a" not in s.split("§12.2")[0]:
    s = s.replace(
        "### 12.2. Приоритет 1 — фасад",
        "### 12.2. Приоритет 1 — фасад (в работе: 03a — Dataset ✅)",
        1,
    )
open(p, "w").write(s)
print("DECISIONS §11.1 / §12.2 updated")
PY

set +e
uv run pytest -q
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
    echo "ERROR: tests failing (rc=${RC})"
    exit 1
fi

git add -A
git commit -m "03a: Dataset (per-column Arrow IPC, sync batches) + tests (13), export"

echo "==> sha256"
sha256sum cadw/dataset/dataset.py tests/test_dataset.py cadw/__init__.py
echo "==> 03a applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"
echo "    Hard:   git reset --hard ${BK}"