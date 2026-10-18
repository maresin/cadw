#!/usr/bin/env bash
# patch_03b: CadwDataFrame — минимальный фасад над Dataset + примитивы.
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
BK="backup-pre-03b-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_03b"

BR="patch/03b-dataframe"
git rev-parse --verify "${BR}" >/dev/null 2>&1 && {
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "switch or delete ${BR}"; exit 1; }
  echo "Resuming on ${BR}"
} || git checkout -b "${BR}"

# ---------- cadw/dataframe.py ----------
cat > cadw/dataframe.py <<'PY'
"""
CadwDataFrame — публичный фасад.

Контракт «честный фасад»:
  - не реализует алгоритмы обработки данных;
  - не содержит построчных циклов (.as_py() на данных);
  - группировку, сортировку, unique делегирует примитивам
    (groupby_aggregate, external_sort, hash_shuffle);
  - хранит только: Dataset + memory_guard + spill_dir + columns (selection).

См. DECISIONS.md §10.3, §12.2.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from cadw.core.memory_guard import MemoryGuard
from cadw.dataset import Dataset
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ParquetBatchWriter
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.sort import external_sort


class CadwDataFrame:
    # ---------------- construction ----------------

    def __init__(
        self,
        dataset: Dataset,
        *,
        memory_guard: MemoryGuard | None = None,
        spill_dir: str | Path | None = None,
        columns: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError("CadwDataFrame requires a Dataset instance")
        self._dataset = dataset
        self._memory_guard = memory_guard or MemoryGuard(max_memory_mb=100)
        if spill_dir is None:
            spill_dir = Path(tempfile.mkdtemp(prefix="cadw_facade_"))
        self._spill_dir = Path(spill_dir)
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        self._columns = list(columns) if columns is not None else dataset.columns
        for c in self._columns:
            if c not in dataset.columns:
                raise KeyError(f"column {c!r} not found in dataset")

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str | Path,
        base_dir: str | Path,
        *,
        chunk_size: int = 10_000,
        memory_guard: MemoryGuard | None = None,
        spill_dir: str | Path | None = None,
    ) -> "CadwDataFrame":
        ds = Dataset.from_parquet(parquet_path, base_dir, chunk_size=chunk_size)
        return cls(ds, memory_guard=memory_guard, spill_dir=spill_dir)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        *,
        memory_guard: MemoryGuard | None = None,
        spill_dir: str | Path | None = None,
    ) -> "CadwDataFrame":
        return cls(dataset, memory_guard=memory_guard, spill_dir=spill_dir)

    # ---------------- properties ----------------

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def shape(self) -> tuple[int, int]:
        return (self._dataset.total_rows, len(self._columns))

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    # ---------------- reading ----------------

    def to_batches(
        self, batch_size: int | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Поток батчей выбранных колонок. Не материализует."""
        return self._dataset.iter_batches(
            batch_size=batch_size, columns=self._columns
        )

    def head(self, n: int = 5) -> pa.Table:
        """Первые n строк как pa.Table (без .as_py())."""
        if n <= 0 or self._dataset.total_rows == 0:
            schema = pa.schema(
                [self._dataset.schema.field(c) for c in self._columns]
            )
            return pa.Table.from_batches([], schema=schema)
        first = True
        acc: list[pa.RecordBatch] = []
        remaining = n
        for b in self._dataset.iter_batches(
            batch_size=min(n, 1000), columns=self._columns
        ):
            take = min(b.num_rows, remaining)
            acc.append(b.slice(0, take))
            remaining -= take
            first = False
            if remaining <= 0:
                break
        if first:
            schema = pa.schema(
                [self._dataset.schema.field(c) for c in self._columns]
            )
            return pa.Table.from_batches([], schema=schema)
        return pa.Table.from_batches(acc).slice(0, n)

    # ---------------- groupby ----------------

    def groupby(self, by: str | Sequence[str]) -> "_GroupBy":
        by_cols = [by] if isinstance(by, str) else list(by)
        return _GroupBy(self, by_cols)

    def _groupby_impl(
        self,
        by_cols: Sequence[str],
        agg_columns: Sequence[str],
        agg_funcs: Sequence[str],
    ) -> "CadwDataFrame":
        reader_iter = self._dataset.iter_batches(columns=None)
        result = groupby_aggregate(
            reader_iter,
            by_cols,
            agg_columns,
            agg_funcs,
            self._memory_guard,
            self._spill_dir / "groupby",
            log=False,
        )
        return self._wrap_result_table(result)

    # ---------------- unique ----------------

    def unique(self, column: str) -> "CadwDataFrame":
        """Уникальные значения колонки как одноколоночный CadwDataFrame.

        Реализация — композиция примитивов: hash_shuffle по колонке
        (через groupby_aggregate с count=1 как побочным результатом),
        затем отбрасываем count. Никаких построчных .as_py().
        """
        if column not in self._columns:
            raise KeyError(f"column {column!r} not found")
        reader_iter = self._dataset.iter_batches(columns=[column])
        result = groupby_aggregate(
            reader_iter,
            [column],
            [column],
            ["count"],
            self._memory_guard,
            self._spill_dir / "unique",
            log=False,
        )
        values_only = result.select([column])
        return self._wrap_result_table(values_only)

    # ---------------- sort_values ----------------

    def sort_values(
        self,
        by: str | Sequence[str],
        *,
        ascending: bool | Sequence[bool] = True,
    ) -> "CadwDataFrame":
        by_cols = [by] if isinstance(by, str) else list(by)
        for c in by_cols:
            if c not in self._columns:
                raise KeyError(f"sort_values: column {c!r} not found")
        path = external_sort(
            self._dataset.iter_batches(columns=None),
            key_columns=by_cols,
            value_columns=self._columns,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "sort",
            ascending=ascending,
            log=False,
        )
        schema = pa.schema(
            [self._dataset.schema.field(c) for c in self._columns]
        )

        def _iter() -> Iterable[pa.RecordBatch]:
            reader = ArrowIPCBatchReader(str(path), self._columns)
            for b in reader:
                yield b

        new_ds = Dataset.from_batches(schema, _iter(), self._new_dataset_dir())
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- write ----------------

    def write_parquet(
        self, path: str | Path, *, compression: str = "snappy"
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema(
            [self._dataset.schema.field(c) for c in self._columns]
        )
        writer = ParquetBatchWriter(
            str(path), schema, compression=compression
        )
        try:
            for b in self._dataset.iter_batches(columns=self._columns):
                writer.write_batch(b)
        finally:
            writer.close()
        return path

    # ---------------- helpers ----------------

    def _wrap_result_table(self, table: pa.Table) -> "CadwDataFrame":
        new_ds = Dataset.from_batches(
            table.schema,
            iter(table.to_batches()),
            self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def _new_dataset_dir(self) -> Path:
        p = self._spill_dir / "datasets" / f"ds_{uuid.uuid4().hex[:8]}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __repr__(self) -> str:
        return (
            f"CadwDataFrame(rows={self._dataset.total_rows}, "
            f"columns={self._columns})"
        )


class _GroupBy:
    def __init__(self, parent: CadwDataFrame, by_cols: Sequence[str]) -> None:
        self._parent = parent
        self._by = list(by_cols)

    def agg(
        self,
        agg_columns: str | Sequence[str],
        agg_funcs: str | Sequence[str],
    ) -> CadwDataFrame:
        agg_cols = (
            [agg_columns] if isinstance(agg_columns, str) else list(agg_columns)
        )
        funcs = [agg_funcs] if isinstance(agg_funcs, str) else list(agg_funcs)
        return self._parent._groupby_impl(self._by, agg_cols, funcs)
PY

# ---------- export ----------
python3 - <<'PY'
p = "cadw/__init__.py"
s = open(p).read()
if "from .dataframe import CadwDataFrame" not in s:
    s = s.replace(
        "from .dataset import Dataset",
        "from .dataset import Dataset\nfrom .dataframe import CadwDataFrame",
        1,
    )
    s = s.replace(
        '"MemoryGuard", "MemoryLimitExceeded", "Dataset",',
        '"MemoryGuard", "MemoryLimitExceeded", "Dataset", "CadwDataFrame",',
        1,
    )
    open(p, "w").write(s)
    print("patched cadw/__init__.py: CadwDataFrame exported")
PY

# ---------- tests ----------
cat > tests/test_dataframe.py <<'EOF'
"""Тесты CadwDataFrame (патч 03b): честный фасад над Dataset + примитивы."""
import inspect
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cadw.dataframe as df_mod
from cadw import CadwDataFrame, Dataset, MemoryGuard


def make_table(n: int = 20) -> pa.Table:
    return pa.table({
        "cat": [f"g{i % 5}" for i in range(n)],
        "val": list(range(n)),
    })


def write_parquet(tmp_path: Path, n: int = 20) -> Path:
    t = make_table(n)
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    return p


def make_df(tmp_path: Path, n: int = 20, chunk: int = 5) -> CadwDataFrame:
    p = write_parquet(tmp_path, n)
    return CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )


# ---------- construction ----------

def test_from_parquet(tmp_path):
    df = make_df(tmp_path)
    assert df.shape == (20, 2)
    assert set(df.columns) == {"cat", "val"}


def test_from_dataset(tmp_path):
    p = write_parquet(tmp_path)
    ds = Dataset.from_parquet(p, tmp_path / "ds", chunk_size=5)
    df = CadwDataFrame(ds, memory_guard=MemoryGuard(max_memory_mb=50),
                       spill_dir=tmp_path / "spill")
    assert df.shape == (20, 2)


def test_init_requires_dataset():
    with pytest.raises(TypeError):
        CadwDataFrame("not a dataset")


def test_columns_not_found_raises(tmp_path):
    p = write_parquet(tmp_path)
    ds = Dataset.from_parquet(p, tmp_path / "ds", chunk_size=5)
    with pytest.raises(KeyError):
        CadwDataFrame(ds, columns=["nope"])


# ---------- reading ----------

def test_head(tmp_path):
    df = make_df(tmp_path)
    h = df.head(3)
    assert h.num_rows == 3
    assert h.schema.names == ["cat", "val"]


def test_head_zero(tmp_path):
    df = make_df(tmp_path)
    h = df.head(0)
    assert h.num_rows == 0


def test_to_batches(tmp_path):
    df = make_df(tmp_path)
    sizes = [b.num_rows for b in df.to_batches(batch_size=7)]
    assert sum(sizes) == 20
    assert all(b.schema.names == ["cat", "val"] for b in df.to_batches())


# ---------- groupby ----------

def test_groupby_mean(tmp_path):
    df = make_df(tmp_path)
    result = df.groupby("cat").agg("val", "mean")
    assert set(result.columns) == {"cat", "val_mean"}
    # roundtrip через write_parquet + чтение
    out = tmp_path / "agg.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    assert t.num_rows == 5
    vals = {r["cat"]: r["val_mean"] for r in t.to_pylist()}
    # для cat g0: i=0,5,10,15 → mean=7.5
    assert vals["g0"] == 7.5


def test_groupby_sum(tmp_path):
    df = make_df(tmp_path)
    result = df.groupby("cat").agg("val", "sum")
    out = tmp_path / "agg.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    vals = {r["cat"]: r["val_sum"] for r in t.to_pylist()}
    assert vals["g0"] == 0 + 5 + 10 + 15  # 30
    assert vals["g1"] == 1 + 6 + 11 + 16  # 34


def test_groupby_multiple_aggs(tmp_path):
    df = make_df(tmp_path)
    result = df.groupby("cat").agg(["val", "val"], ["sum", "count"])
    out = tmp_path / "agg.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    vals = {r["cat"]: (r["val_sum"], r["val_count"]) for r in t.to_pylist()}
    assert vals["g0"] == (30, 4)


# ---------- unique ----------

def test_unique(tmp_path):
    df = make_df(tmp_path)
    uniq = df.unique("cat")
    assert uniq.columns == ["cat"]
    assert uniq.shape[0] == 5
    out = tmp_path / "u.parquet"
    uniq.write_parquet(out)
    t = pq.read_table(out)
    assert sorted(t.column("cat").to_pylist()) == ["g0", "g1", "g2", "g3", "g4"]


# ---------- sort_values ----------

def test_sort_values_ascending(tmp_path):
    df = make_df(tmp_path)
    sorted_df = df.sort_values("val", ascending=True)
    out = tmp_path / "sorted.parquet"
    sorted_df.write_parquet(out)
    t = pq.read_table(out)
    assert t.column("val").to_pylist() == list(range(20))


def test_sort_values_descending(tmp_path):
    df = make_df(tmp_path)
    sorted_df = df.sort_values("val", ascending=False)
    out = tmp_path / "sorted.parquet"
    sorted_df.write_parquet(out)
    t = pq.read_table(out)
    assert t.column("val").to_pylist() == list(range(19, -1, -1))


def test_sort_values_missing_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError):
        df.sort_values("nope")


# ---------- integration ----------

def test_groupby_chain_write(tmp_path):
    df = make_df(tmp_path)
    out = tmp_path / "result.parquet"
    (df.groupby("cat").agg("val", "mean")).write_parquet(out)
    t = pq.read_table(out)
    assert t.num_rows == 5
    assert t.column("val_mean").to_pylist() is not None


def test_full_pipeline_parquet_to_parquet(tmp_path):
    """Исходный parquet → sort → groupby → parquet."""
    df = make_df(tmp_path)
    pipeline = df.sort_values("val").groupby("cat").agg("val", "mean")
    out = tmp_path / "final.parquet"
    pipeline.write_parquet(out)
    t = pq.read_table(out)
    assert t.num_rows == 5


# ---------- facade is honest ----------

def test_facade_has_no_as_py():
    """CadwDataFrame не должен содержать построчных .as_py() на данных."""
    src = inspect.getsource(df_mod)
    assert ".as_py()" not in src, (
        "CadwDataFrame must not call .as_py() on data rows; "
        "delegate to primitives"
    )


def test_facade_has_no_row_loops():
    """Не должно быть for i in range(batch.num_rows)."""
    src = inspect.getsource(df_mod)
    assert "range(batch.num_rows)" not in src
    assert "range(b.num_rows)" not in src
EOF

# ---------- DECISIONS §11.1 / §12.2 ----------
python3 - <<'PY'
p = "DECISIONS.md"
s = open(p).read()
if "| `CadwDataFrame` (03b)" not in s:
    s = s.replace(
        "| `Dataset` (03a, per-column Arrow IPC, sync batches) | ✅ | 13 |",
        "| `Dataset` (03a, per-column Arrow IPC, sync batches) | ✅ | 13 |\n"
        "| `CadwDataFrame` (03b, минимальный фасад) | ✅ | 17 |",
        1,
    )
if "03a — Dataset ✅" in s and "03b — CadwDataFrame ✅" not in s:
    s = s.replace(
        "### 12.2. Приоритет 1 — фасад (в работе: 03a — Dataset ✅)",
        "### 12.2. Приоритет 1 — фасад (03a ✅, 03b ✅, 03c — в работе)",
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
git commit -m "03b: CadwDataFrame — minimal facade over Dataset (groupby/unique/sort/write/head/to_batches) + tests (17)"

echo "==> sha256"
sha256sum cadw/dataframe.py tests/test_dataframe.py cadw/__init__.py
echo "==> 03b applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"
echo "    Hard:   git reset --hard ${BK}"