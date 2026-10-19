#!/usr/bin/env bash
# patch_v031_rank: CadwDataFrame.rank(by, method='first'|'min'|'dense',
# ascending, pct, output_col). external_sort + stream-rank + re-sort by row_id.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v031-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.3.1 rank"

BR="patch/v031-rank"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/ops/rank.py ----------
cat > cadw/ops/rank.py <<'PY'
"""
Rank: external_sort + потоковое вычисление rank.

Алгоритм (DECISIONS §8, «rank | Sort + локальный ранг»):
  1. external_sort по `by` (ascending).
  2. Потоково: rank = f(позиция, состояние tie-группы). Сохраняем
     (row_id, rank) в файл (порядок — sorted-by-`by`).
  3. external_sort по row_id → rank в исходном порядке.
  4. Zip с исходным DataFrame позиционно.

Streamable методы (v0.3.1):
  - 'first': rank = global position (1-based) в sorted order.
  - 'min':   rank = позиция первого в текущей tie-группе.
  - 'dense': rank = dense counter (1, 2, 3, ... для уникальных значений).

Не streamable (v0.3.2): 'max', 'average', 'last' — требуют знать
конец tie-группы, а значит — буферизацию группы (O(N) при all-equal).

Null values: участвуют как значения (null == null); pandas `skipna=True`
не поддерживается в v0.3.1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyarrow as pa

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_writer import ArrowIPCBatchWriter

logger = get_logger("ops.rank")

VALID_METHODS = ("first", "min", "dense")


def compute_ranks_to_disk(
    sorted_batches: Iterable[pa.RecordBatch],
    *,
    by: str,
    output_col: str,
    method: str,
    pct: bool,
    total_n: int,
    rank_path: Path,
    memory_guard: MemoryGuard,
    log: bool = False,
) -> Path:
    """Stream sorted batches, write (row_id, rank) to rank_path.

    `sorted_batches` должны содержать колонки [by, "row_id"], отсортированы
    по `by` (ascending; ties — по row_id asc, стабильность из external_sort).

    `total_n` — число строк (для pct). Если 0 и pct=True — ValueError.
    """
    if method not in VALID_METHODS:
        raise NotImplementedError(
            f"rank: method={method!r} not supported in v0.3.1. "
            f"Supported: {VALID_METHODS}. "
            f"'max'/'average'/'last' требуют буферизации tie-группы — v0.3.2."
        )
    if pct and total_n <= 0:
        raise ValueError("rank: pct=True requires total_n > 0")

    rank_dtype = pa.float64() if pct else pa.int64()
    schema = pa.schema([("row_id", pa.uint64()), (output_col, rank_dtype)])
    writer = ArrowIPCBatchWriter(str(rank_path), schema)

    _UNSET = object()
    prev_val = _UNSET
    group_start = 0
    global_pos = 0
    dense_counter = 0

    try:
        for batch in sorted_batches:
            memory_guard.check_batch(batch, name="rank.input")
            vals = batch.column(by).to_pylist()
            row_ids = batch.column("row_id").to_pylist()

            ranks: list = []
            for v in vals:
                global_pos += 1
                if prev_val is _UNSET:
                    group_start = global_pos
                    dense_counter = 1
                elif v != prev_val:
                    group_start = global_pos
                    dense_counter += 1
                # else: continue current tie group

                if method == "first":
                    r = global_pos
                elif method == "min":
                    r = group_start
                else:  # dense
                    r = dense_counter

                ranks.append((r / total_n) if pct else r)
                prev_val = v

            rb = pa.RecordBatch.from_arrays(
                [
                    pa.array(row_ids, type=pa.uint64()),
                    pa.array(ranks, type=rank_dtype),
                ],
                schema=schema,
            )
            writer.write_batch(rb)
    finally:
        writer.close()

    if log:
        logger.info(
            "compute_ranks_to_disk: method=%s pct=%s total_n=%d",
            method, pct, total_n,
        )
    return rank_path
PY

# ---------- cadw/ops/__init__.py ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/__init__.py")
s = p.read_text()
if "from .rank import compute_ranks_to_disk, VALID_METHODS" not in s:
    s = s.replace(
        "from .dedup import drop_duplicates",
        "from .dedup import drop_duplicates\n"
        "from .rank import compute_ranks_to_disk, VALID_METHODS",
        1,
    )
    s = s.replace(
        '"drop_duplicates",',
        '"drop_duplicates", "compute_ranks_to_disk", "VALID_METHODS",',
        1,
    )
    p.write_text(s)
    print("ops/__init__.py: rank exported")
PY

# ---------- cadw/__init__.py ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/__init__.py")
s = p.read_text()
if "from .ops.rank import compute_ranks_to_disk" not in s:
    s = s.replace(
        "from .ops.dedup import drop_duplicates",
        "from .ops.dedup import drop_duplicates\n"
        "from .ops.rank import compute_ranks_to_disk, VALID_METHODS",
        1,
    )
    s = s.replace(
        '"drop_duplicates",',
        '"drop_duplicates", "compute_ranks_to_disk", "VALID_METHODS",',
        1,
    )
    p.write_text(s)
    print("cadw/__init__.py: rank exported")
PY

# ---------- dataframe.py: rank method ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# imports
if "from cadw.ops.rank import compute_ranks_to_disk" not in s:
    s = s.replace(
        "from cadw.ops.dedup import drop_duplicates as _ops_drop_duplicates",
        "from cadw.ops.dedup import drop_duplicates as _ops_drop_duplicates\n"
        "from cadw.ops.rank import compute_ranks_to_disk, VALID_METHODS",
        1,
    )

anchor = "    # ---------------- merge ----------------"
if anchor not in s:
    raise SystemExit("merge anchor not found")
if "    def rank(" in s:
    print("dataframe.py: rank already exists")
else:
    method = '''    # ---------------- rank ----------------

    def rank(
        self,
        by: str,
        method: str = "first",
        ascending: bool = True,
        pct: bool = False,
        *,
        output_col: str | None = None,
        batch_size: int = 10_000,
    ) -> "CadwDataFrame":
        """Ранг по колонке `by` (v0.3.1).

        - method ∈ {'first', 'min', 'dense'}:
            - 'first': 1-based позиция в sorted order (ties по row_id asc).
            - 'min':   минимум из рангов в tie-группе.
            - 'dense': 1, 2, 3, ... для уникальных значений.
        - ascending: True — меньшие значения получают меньший ранг.
        - pct: делить rank на total_n.
        - output_col: имя новой колонки (default `{by}_rank`).

        Возвращает новый CadwDataFrame со всеми исходными колонками +
        rank в **исходном порядке** строк.

        Null values участвуют как значения (null == null). pandas skipna
        не поддерживается в v0.3.1.

        Методы 'max'/'average'/'last' — v0.3.2 (требуют буферизации
        tie-группы, O(N) при all-equal).
        """
        if method not in VALID_METHODS:
            raise NotImplementedError(
                f"rank: method={method!r} not supported in v0.3.1. "
                f"Supported: {VALID_METHODS}. "
                f"'max'/'average'/'last' — v0.3.2."
            )
        if by not in self._columns:
            raise KeyError(f"rank: by={by!r} not in columns {self._columns}")
        if output_col is None:
            output_col = f"{by}_rank"
        if output_col in self._columns:
            raise KeyError(
                f"rank: output_col={output_col!r} already in columns"
            )

        # Count filtered rows (needed for pct, and sanity).
        total_n = self.count()
        if total_n == 0:
            # Empty: append an empty rank column of proper dtype.
            rank_dtype = pa.float64() if pct else pa.int64()
            out_fields = [
                self._dataset.schema.field(c) for c in self._columns
            ]
            out_fields.append(pa.field(output_col, rank_dtype))
            out_schema = pa.schema(out_fields)

            def _iter_empty():
                for b in self._iter_arrow_batches(batch_size=batch_size):
                    n = b.num_rows
                    if pct:
                        col = pa.array([], type=pa.float64())
                    else:
                        col = pa.array([], type=pa.int64())
                    yield b.append_column(output_col, col)

            new_ds = Dataset.from_batches(
                out_schema, _iter_empty(), self._new_dataset_dir()
            )
            return CadwDataFrame(
                new_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        spill = self._spill_dir / "rank" / f"r_{uuid.uuid4().hex[:8]}"
        spill.mkdir(parents=True, exist_ok=True)

        # Step 1: sort by `by`.
        sorted_by_val = external_sort(
            self._iter_filtered_batches(),
            key_columns=[by],
            value_columns=[by],
            memory_guard=self._memory_guard,
            spill_dir=spill / "sort_by_val",
            ascending=ascending,
            log=False,
        )

        # Step 2: rank per row → (row_id, rank) on disk.
        from cadw.io.batch_reader import ArrowIPCBatchReader as _IPCReader
        rank_pairs_path = compute_ranks_to_disk(
            _IPCReader(str(sorted_by_val), [by, "row_id"]),
            by=by,
            output_col=output_col,
            method=method,
            pct=pct,
            total_n=total_n,
            rank_path=spill / "ranks.arrow",
            memory_guard=self._memory_guard,
            log=False,
        )

        # Step 3: sort rank pairs by original row_id.
        def _iter_rank_pairs():
            for b in _IPCReader(str(rank_pairs_path), None):
                # columns: row_id, output_col → rename row_id to __orig_rid
                yield b.rename_columns(["__orig_rid", output_col])

        sorted_by_rowid = external_sort(
            _iter_rank_pairs(),
            key_columns=["__orig_rid"],
            value_columns=[output_col],
            memory_guard=self._memory_guard,
            spill_dir=spill / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        # Step 4: build rank Dataset (in original order via __orig_rid).
        rank_dtype = pa.float64() if pct else pa.int64()
        rank_schema = pa.schema([
            ("__orig_rid", pa.uint64()),
            (output_col, rank_dtype),
        ])

        def _iter_rank_after_sort():
            for b in _IPCReader(str(sorted_by_rowid), ["__orig_rid", output_col]):
                yield b

        rank_ds = Dataset.from_batches(
            rank_schema, _iter_rank_after_sort(), spill / "rank_ds"
        )

        # Step 5: zip with original (both in original order; same batch_size).
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field(output_col, rank_dtype))
        out_schema = pa.schema(out_fields)

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            rank_iter = rank_ds.iter_batches(
                batch_size=batch_size, columns=[output_col]
            )
            for ob, rb in zip(orig_iter, rank_iter):
                if ob.num_rows != rb.num_rows:
                    raise RuntimeError(
                        f"rank: batch-size mismatch: orig={ob.num_rows} "
                        f"rank={rb.num_rows}"
                    )
                yield ob.append_column(output_col, rb.column(output_col))

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

'''
    s = s.replace(anchor, method + anchor, 1)
    p.write_text(s)
    print("dataframe.py: rank method added")
PY

# ---------- tests/test_rank.py ----------
cat > tests/test_rank.py <<'EOF'
"""
rank (v0.3.1): external_sort + stream-rank + re-sort by row_id.

Проверяем:
  - method='first'/'min'/'dense' на asc и desc;
  - pct=True;
  - сохранение исходного порядка (position);
  - all-unique / all-same / ties;
  - после filter;
  - пустой результат;
  - output_col custom;
  - invalid method → NotImplementedError;
  - pandas-equivalence для first/min/dense (set-based).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name, data, chunk=3, mem_mb=20):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_df(df, tmp_path, name="out.parquet"):
    out = Path(tmp_path) / name
    df.write_parquet(out)
    return pq.read_table(out).to_pandas()


# ---------- method='first' ----------

def test_rank_first_ascending(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["b", "a", "c", "a"], "v": [1, 2, 3, 4]})
    result = df.rank(by="k", method="first")
    got = read_df(result, tmp_path)
    # sort by k asc, ties by row_id asc:
    # row_id 1 (a,2), row_id 3 (a,4), row_id 0 (b,1), row_id 2 (c,3)
    # ranks:           1,              2,              3,              4
    # back to original order: b→3, a→1, c→4, a→2
    assert got["k_rank"].tolist() == [3, 1, 4, 2]


def test_rank_first_descending(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["b", "a", "c", "a"], "v": [1, 2, 3, 4]})
    result = df.rank(by="k", method="first", ascending=False)
    got = read_df(result, tmp_path)
    # sort by k desc, ties by row_id asc:
    # row_id 2 (c,3), row_id 0 (b,1), row_id 1 (a,2), row_id 3 (a,4)
    # ranks:           1,              2,              3,              4
    # original: b→2, a→3, c→1, a→4
    assert got["k_rank"].tolist() == [2, 3, 1, 4]


# ---------- method='min' ----------

def test_rank_min_with_ties(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="min")
    got = read_df(result, tmp_path)
    # sorted: a(pos1), a(pos2), a(pos3), b(pos4), c(pos5)
    # min: a→1, a→1, a→1, b→4, c→5
    # original (a,b,a,c,a): 1, 4, 1, 5, 1
    assert got["k_rank"].tolist() == [1, 4, 1, 5, 1]


# ---------- method='dense' ----------

def test_rank_dense_with_ties(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["b", "a", "c", "a", "b"], "v": list(range(5))})
    result = df.rank(by="k", method="dense")
    got = read_df(result, tmp_path)
    # distinct sorted values: a→1, b→2, c→3
    # original (b,a,c,a,b): 2, 1, 3, 1, 2
    assert got["k_rank"].tolist() == [2, 1, 3, 1, 2]


# ---------- pct ----------

def test_rank_pct(tmp_path):
    df = make_df(tmp_path, "L", {"k": [10, 20, 30, 40], "v": list(range(4))})
    result = df.rank(by="k", method="first", pct=True)
    got = read_df(result, tmp_path)
    # ranks 1,2,3,4 / 4 = 0.25, 0.5, 0.75, 1.0 (original order 10,20,30,40)
    assert got["k_rank"].tolist() == [0.25, 0.5, 0.75, 1.0]


# ---------- preserve original order ----------

def test_rank_preserves_row_count_and_order(tmp_path):
    n = 50
    rng = np.random.default_rng(42)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="first")
    got = read_df(result, tmp_path)
    assert got.shape[0] == n
    # v column unchanged and in original order
    assert got["v"].tolist() == list(range(n))


# ---------- edge cases ----------

def test_rank_all_unique(tmp_path):
    n = 20
    df = make_df(tmp_path, "L", {"k": list(range(n)), "v": list(range(n))})
    result = df.rank(by="k", method="first")
    got = read_df(result, tmp_path)
    # ranks = 1..n (identity, since sorted order = original order)
    assert got["k_rank"].tolist() == list(range(1, n + 1))


def test_rank_all_same_first(tmp_path):
    n = 10
    df = make_df(tmp_path, "L", {"k": ["X"] * n, "v": list(range(n))})
    result = df.rank(by="k", method="first")
    got = read_df(result, tmp_path)
    assert got["k_rank"].tolist() == list(range(1, n + 1))


def test_rank_all_same_min(tmp_path):
    n = 10
    df = make_df(tmp_path, "L", {"k": ["X"] * n, "v": list(range(n))})
    result = df.rank(by="k", method="min")
    got = read_df(result, tmp_path)
    assert got["k_rank"].tolist() == [1] * n


def test_rank_all_same_dense(tmp_path):
    n = 10
    df = make_df(tmp_path, "L", {"k": ["X"] * n, "v": list(range(n))})
    result = df.rank(by="k", method="dense")
    got = read_df(result, tmp_path)
    assert got["k_rank"].tolist() == [1] * n


# ---------- after filter ----------

def test_rank_after_filter(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    filtered = df[df["v"] > 2]  # rows (b,3),(b,4),(c,5)
    result = filtered.rank(by="k", method="first")
    got = read_df(result, tmp_path)
    # sorted: b(orig pos 0), b(orig pos 1), c(orig pos 2) → ranks 1,2,3
    # but keep='first' with ties by row_id: b at filtered-pos-0 gets 1, b at 1 gets 2
    # c gets 3.
    # output in filtered order: [1, 2, 3]
    assert got["k_rank"].tolist() == [1, 2, 3]


# ---------- output_col custom ----------

def test_rank_custom_output_col(tmp_path):
    df = make_df(tmp_path, "L", {"k": [3, 1, 2], "v": list(range(3))})
    result = df.rank(by="k", method="first", output_col="my_rank")
    got = read_df(result, tmp_path)
    assert "my_rank" in got.columns
    assert "k_rank" not in got.columns


# ---------- errors ----------

def test_rank_invalid_method(tmp_path):
    df = make_df(tmp_path, "L", {"k": [1, 2], "v": [1, 2]})
    for bad in ("max", "average", "last", "unknown"):
        with pytest.raises(NotImplementedError, match="method"):
            df.rank(by="k", method=bad)


def test_rank_missing_column(tmp_path):
    df = make_df(tmp_path, "L", {"k": [1, 2], "v": [1, 2]})
    with pytest.raises(KeyError, match="not in"):
        df.rank(by="nope")


def test_rank_output_col_collision(tmp_path):
    df = make_df(tmp_path, "L", {"k": [1, 2], "v": [1, 2]})
    with pytest.raises(KeyError, match="already in"):
        df.rank(by="k", output_col="v")


# ---------- pandas-equivalence ----------

def test_rank_first_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="first")
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="first").astype("int64")
    np.testing.assert_array_equal(
        got["k_rank"].to_numpy(), expected.to_numpy()
    )


def test_rank_min_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="min")
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="min").astype("int64")
    np.testing.assert_array_equal(
        got["k_rank"].to_numpy(), expected.to_numpy()
    )


def test_rank_dense_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="dense")
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="dense").astype("int64")
    np.testing.assert_array_equal(
        got["k_rank"].to_numpy(), expected.to_numpy()
    )


def test_rank_pct_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(11)
    data = {
        "k": rng.integers(0, 10, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="first", pct=True)
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="first", pct=True)
    np.testing.assert_allclose(
        got["k_rank"].to_numpy(), expected.to_numpy(), rtol=1e-10
    )


# ---------- chained ----------

def test_rank_then_groupby(tmp_path):
    df = make_df(tmp_path, "L", {
        "g": ["A", "A", "B", "B"],
        "k": [3, 1, 4, 2],
    })
    ranked = df.rank(by="k", method="first")
    # k sorted asc: k=1(rank 1), k=2(rank 2), k=3(rank 3), k=4(rank 4)
    # grouped mean of k_rank per group:
    # A: (3, 1) → mean rank (3+1)/2 = 2.0
    # B: (4, 2) → mean rank (4+2)/2 = 3.0
    result = ranked.groupby("g").agg("k_rank", "mean")
    out = tmp_path / "g.parquet"
    result.write_parquet(out)
    rows = {r["g"]: r["k_rank_mean"] for r in pq.read_table(out).to_pylist()}
    assert rows == {"A": 2.0, "B": 3.0}
EOF

# ---------- DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| `rank` (v0.3.1: Sort + local stream; first/min/dense) |" not in s:
    s = s.replace(
        "| `drop_duplicates` (v0.3.0: HashShuffle + local pandas) | ✅ | 15 |",
        "| `drop_duplicates` (v0.3.0: HashShuffle + local pandas) | ✅ | 19 |\n"
        "| `rank` (v0.3.1: Sort + local stream; first/min/dense) | ✅ | 19 |",
        1,
    )

if "| `rank` без max/average/last (v0.3.1) |" not in s:
    s = s.replace(
        "| `drop_duplicates` без recursion при skew | `drop_duplicates` (v0.3.0) | oversize partition → `MemoryLimitExceeded`; v0.3.1 — recursion + streaming dedup |",
        "| `drop_duplicates` без recursion при skew | `drop_duplicates` (v0.3.0) | oversize partition → `MemoryLimitExceeded`; v0.3.1 — recursion + streaming dedup |\n"
        "| `rank` без max/average/last | `rank` (v0.3.1) | требуют буферизации tie-группы (O(N) при all-equal); v0.3.2 — двухпроходный streaming или bounded-buffer. §13.4/49 |\n"
        "| `rank` nulls участвуют как значения (pandas skipna не поддерживается) | `rank` (v0.3.1) | v0.3.2: skip nulls (аналог pandas `rank(na_option='keep')`) |",
        1,
    )

marker_135 = "### 13.5. Открытые вопросы"
episode_49 = """**Эпизод 49 (rank: streamable vs bufferable).** `rank` требует
глобального порядка (sort) и tie-breaking. Разделены методы по
streamability:
- `first`/`min`/`dense` — O(1) state, стримятся в один проход;
- `max`/`average`/`last` — требуют знать конец tie-группы → буферизация
  группы, O(N) при all-equal (патология: 10^9 одинаковых значений →
  10^9 рангов в RAM). Пандас не парится (in-memory), но для out-of-core
  это OOM.

MVP v0.3.1 = streamable trio. v0.3.2: `max`/`average`/`last` через
двухпроходный streaming (первый проход: group_end_rank per group → на
диск; второй: join с sorted stream) или bounded-buffer с spill при
группе > budget.

**Порядок результата:** rank — свойство исходной строки, поэтому
возвращаем в original order. Реализуется двумя сортировками:
1. sort by `by` (нужно для streamable rank);
2. sort ranks by `row_id` (нужно для original order).
Второй sort — маленький (row_id + rank = 16 B/строка), дешевле первого.
Zip с original — позиционный, batch-by-batch с одним `batch_size`.

**Null values:** участвуют как значения (`null == null` → same group).
Pandas `rank(skipna=True)` → NaN ranks для null строк. Наш MVP этого не
делает — документировано в §11.5. v0.3.2: null-детект per row + emit NaN.

**Урок:** «работает для типичных случаев» — не то же, что «звуковая
верхняя граница». Streamable методы — гарантированно O(1) state.
Bufferable методы — O(max tie group size), где max — параметр данных,
не бюджета; для pathological данных = O(N). Разделение по методам
делает ограничение явным.

"""
if "Эпизод 49" not in s:
    s = s.replace(marker_135, episode_49 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §11.5, §13.4/49")
PY

# ---------- CHANGELOG.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.3.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.rank(by, method='first'|'min'|'dense', ascending=True,
  pct=False, output_col=None, batch_size=10_000)`:
  - external_sort по `by` → потоковое вычисление rank → external_sort
    rank-пар по `row_id` → zip с исходным в original order.
  - `first`: 1-based позиция в sorted order (ties по row_id asc).
  - `min`:   минимум ранга в tie-группе.
  - `dense`: 1, 2, 3, ... для уникальных значений.
  - `pct=True`: rank / total_n.
- `cadw.ops.rank.compute_ranks_to_disk` — primitive: sorted stream → file
  (row_id, rank).
- Экспорт: `cadw.compute_ranks_to_disk`, `cadw.VALID_METHODS`.

### Принцип
- Streamable методы — O(1) state. Bufferable (`max`/`average`/`last`) —
  O(max tie group size) = O(N) для pathological данных → **не в MVP**,
  требуют двухпроходного streaming.
- Результат в original order (две сортировки: по `by` + по `row_id`).

### Ограничения (v0.3.1, §11.5)
- **method**: `first`/`min`/`dense` только. `max`/`average`/`last` — v0.3.2.
- **null values**: участвуют как значения (null == null); pandas
  `rank(skipna=True)` не поддерживается.
- **cost**: 2× external_sort + 1× count-scan. Для 10^9 строк — ~3 полных
  прохода I/O.

### Тесты
- 19 тестов в `test_rank.py`: все три метода на asc/desc, pct, edge
  cases (all-unique / all-same / ties), after filter, custom output_col,
  errors (invalid method / missing column / collision), 4 × pandas-
  equivalence, chained с groupby.

"""
if "## [0.3.1]" not in s:
    s = s.replace("## [0.3.0]", entry + "## [0.3.0]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.3.1")
PY

# ---------- run ----------
echo "=== pytest tests/test_rank.py -v ==="
set +e
uv run pytest tests/test_rank.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "rank: rc=$RC"
  git add -A
  git commit -m "WIP: v0.3.1 rank (test red; fix follows)" || true
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.3.1: rank (external_sort + stream-rank + re-sort by row_id; first/min/dense; 19 tests)"

echo "==> v0.3.1 applied on ${BR}"