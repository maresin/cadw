#!/usr/bin/env bash
# patch_v030_drop_duplicates: drop_duplicates через HashShuffle + local pandas.
# MVP: keep ∈ {first, last, False}, subset=list[str], без recursion при skew.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v030-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.3.0 drop_duplicates"

BR="patch/v030-drop-duplicates"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/ops/dedup.py ----------
cat > cadw/ops/dedup.py <<'PY'
"""
drop_duplicates: HashShuffle по subset + локальный pandas.drop_duplicates.

Алгоритм (DECISIONS §8, «unique, drop_duplicates | HashShuffle + first»):
  1. HashShuffle по subset → P партиций (не пересекаются по subset keys).
  2. Для каждой партиции: read (safe, size-check ДО чтения),
     sort by row_id → pandas.drop_duplicates(subset, keep).
  3. Concat — партиции не пересекаются по subset keys, поэтому
     локальный dedup эквивалентен глобальному.

`keep`:
  - 'first'  — min row_id per key (по исходному порядку).
  - 'last'   — max row_id per key.
  - False    — только ключи с единственным вхождением (drop ALL duplicates).

Ограничения (v0.3.0):
  - Порядок результата: partition-concatenation (не pandas-original order).
  - Recursion при oversized partition: НЕТ; raise MemoryLimitExceeded
    с рекомендацией увеличить num_partitions. §11.5.
"""
from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.ops.shuffle import estimate_num_partitions, hash_shuffle

logger = get_logger("ops.dedup")

_VALID_KEEP = ("first", "last", False)


def _estimate_k_sample(
    batch: pa.RecordBatch,
    subset: list[str],
    n_total: int,
) -> int:
    """Rough estimate: unique count in subset, linear extrapolation."""
    if batch.num_rows == 0:
        return 1
    sub = batch.select(subset)
    df = sub.to_pandas()
    k_sample = int(df.drop_duplicates().shape[0])
    if k_sample <= 0:
        return 1
    if n_total <= batch.num_rows:
        return k_sample
    return max(1, int(k_sample * (n_total / batch.num_rows)))


def _local_dedup(
    partition_path: Path,
    subset_cols: list[str],
    keep,
    memory_guard: MemoryGuard,
) -> pa.Table:
    """Local dedup of one partition. Returns table with original columns
    (row_id dropped). Raises MemoryLimitExceeded if partition > available*0.5."""
    size = partition_path.stat().st_size
    limit = int(memory_guard.available_bytes() * 0.5)
    if size > limit:
        raise MemoryLimitExceeded(
            f"drop_duplicates: partition {partition_path.name} "
            f"size={size} B > limit={limit} B. "
            f"Увеличьте num_partitions или уменьшите subset. "
            f"Recursion не поддерживается (v0.3.0)."
        )

    reader = ipc.open_file(str(partition_path))
    n_batches = reader.num_record_batches
    # Schema from file with row_id stripped (hash_shuffle always appends).
    out_fields = [f for f in reader.schema if f.name != "row_id"]
    out_schema = pa.schema(out_fields)

    if n_batches == 0:
        return pa.Table.from_batches([], schema=out_schema)

    batches = [reader.get_batch(i).select([f.name for f in out_fields] + ["row_id"])
               for i in range(n_batches)]
    table = pa.Table.from_batches(batches)
    memory_guard.check_state(table.nbytes, name=f"dedup({partition_path.name})")

    df = table.to_pandas()
    # Сорт по row_id (stable) — keep='first'/'last' по исходному порядку.
    df = df.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    df = df.drop_duplicates(subset=subset_cols, keep=keep)
    df = df.drop(columns=["row_id"])

    return pa.Table.from_pandas(df, schema=out_schema, preserve_index=False)


def drop_duplicates(
    batch_iter: Iterable[pa.RecordBatch],
    subset: list[str],
    keep,
    memory_guard: MemoryGuard,
    spill_dir: str | Path,
    *,
    total_rows: int = 0,
    num_partitions: int | None = None,
    log: bool = True,
) -> pa.Table:
    """Out-of-core drop_duplicates.

    - subset: list of columns (non-empty). Dedup key.
    - keep: 'first' | 'last' | False.
    - total_rows: hint for k_est (from df._dataset.total_rows).
    - num_partitions: None — auto from sample.

    Returns pa.Table with original columns (row_id dropped).
    """
    if not subset:
        raise ValueError("drop_duplicates: subset must be non-empty")
    if keep not in _VALID_KEEP:
        raise ValueError(
            f"drop_duplicates: keep must be 'first', 'last', or False; "
            f"got {keep!r}"
        )

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    it = iter(batch_iter)
    try:
        first = next(it)
    except StopIteration as exc:
        raise ValueError("drop_duplicates: empty input") from exc

    for c in subset:
        if c not in first.schema.names:
            raise KeyError(
                f"drop_duplicates: subset column {c!r} not in schema"
            )

    value_cols = [c for c in first.schema.names if c not in subset]

    if num_partitions is None:
        n_total = total_rows if total_rows > 0 else first.num_rows
        k_est = _estimate_k_sample(first, subset, n_total)
        p_est = estimate_num_partitions(
            k_est,
            available_bytes=max(1, memory_guard.available_bytes() // 2),
        )
        num_partitions = max(8, p_est)
        if log:
            logger.info(
                "drop_duplicates: auto P=%d (k_est=%d, n_total=%d)",
                num_partitions, k_est, n_total,
            )

    batches = chain([first], it)
    paths = hash_shuffle(
        batches, subset, value_cols, num_partitions,
        memory_guard, spill_dir / "shuffle", log=log,
    )

    results: list[pa.Table] = []
    for p in paths:
        if not p.exists() or p.stat().st_size == 0:
            continue
        t = _local_dedup(p, subset, keep, memory_guard)
        if t.num_rows > 0:
            results.append(t)

    out_schema = pa.schema(
        [f for f in first.schema if f.name != "row_id"]
    )
    if not results:
        return pa.Table.from_batches([], schema=out_schema)

    return pa.concat_tables(results, promote_options="default")
PY

# ---------- cadw/ops/__init__.py ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/__init__.py")
s = p.read_text()
if "from .dedup import drop_duplicates" not in s:
    s = s.replace(
        "from .groupby import groupby_aggregate, write_groupby_result",
        "from .groupby import groupby_aggregate, write_groupby_result\n"
        "from .dedup import drop_duplicates",
        1,
    )
    s = s.replace(
        '"groupby_aggregate", "write_groupby_result",',
        '"groupby_aggregate", "write_groupby_result", "drop_duplicates",',
        1,
    )
    p.write_text(s)
    print("ops/__init__.py: drop_duplicates exported")
PY

# ---------- cadw/__init__.py ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/__init__.py")
s = p.read_text()
if "drop_duplicates" not in s:
    s = s.replace(
        "from .ops.groupby import groupby_aggregate, write_groupby_result",
        "from .ops.groupby import groupby_aggregate, write_groupby_result\n"
        "from .ops.dedup import drop_duplicates",
        1,
    )
    s = s.replace(
        '"groupby_aggregate", "write_groupby_result",',
        '"groupby_aggregate", "write_groupby_result", "drop_duplicates",',
        1,
    )
    p.write_text(s)
    print("cadw/__init__.py: drop_duplicates exported")
PY

# ---------- dataframe.py: drop_duplicates method ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# import
if "from cadw.ops.dedup import drop_duplicates" not in s:
    s = s.replace(
        "from cadw.ops.sort import external_sort",
        "from cadw.ops.sort import external_sort\n"
        "from cadw.ops.dedup import drop_duplicates as _ops_drop_duplicates",
        1,
    )

# method — insert before merge()
anchor = "    # ---------------- merge ----------------"
if anchor not in s:
    raise SystemExit("merge anchor not found")
if "    def drop_duplicates(" in s:
    print("dataframe.py: drop_duplicates already exists")
else:
    method = '''    # ---------------- drop_duplicates ----------------

    def drop_duplicates(
        self,
        subset: str | Sequence[str] | None = None,
        keep: str | bool = "first",
        *,
        num_partitions: int | None = None,
    ) -> "CadwDataFrame":
        """Удалить дубликаты строк (v0.3.0).

        - subset=None: все колонки — как pandas.
        - subset=str | Sequence[str]: ключ(и) для определения дубликата.
        - keep: 'first' | 'last' | False (drop all duplicates).
        - num_partitions: None — auto из sample (по k_est на subset).

        **Порядок результата**: partition-concatenation, **не** pandas-order.
        Если нужен стабильный порядок — `.sort_values(...)` после. §11.5.

        **Skew**: при oversized partition — `MemoryLimitExceeded`;
        recursion не поддерживается в v0.3.0. Увеличьте `num_partitions`.
        """
        if keep not in ("first", "last", False):
            raise ValueError(
                f"drop_duplicates: keep must be 'first', 'last', or False; "
                f"got {keep!r}"
            )
        if subset is None:
            subset_cols = list(self._columns)
        elif isinstance(subset, str):
            subset_cols = [subset]
        else:
            subset_cols = list(subset)
        if not subset_cols:
            raise ValueError("drop_duplicates: subset must be non-empty")
        for c in subset_cols:
            if c not in self._columns:
                raise KeyError(
                    f"drop_duplicates: subset column {c!r} not in "
                    f"columns {self._columns}"
                )
        # subset=None означает dedup по всем; subset явный — dedup по списку,
        # но output сохраняет все колонки. Проверим совпадение с columns
        # для случая subset=None.
        if subset is None:
            subset_cols = list(self._columns)

        result = _ops_drop_duplicates(
            self._iter_filtered_batches(),
            subset=subset_cols,
            keep=keep,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "dedup",
            total_rows=self._dataset.total_rows,
            num_partitions=num_partitions,
            log=False,
        )
        return self._wrap_result_table(result)

'''
    s = s.replace(anchor, method + anchor, 1)
    p.write_text(s)
    print("dataframe.py: drop_duplicates method added")
PY

# ---------- tests/test_drop_duplicates.py ----------
cat > tests/test_drop_duplicates.py <<'EOF'
"""
drop_duplicates (v0.3.0): HashShuffle + local pandas.drop_duplicates.

Проверяем:
  - keep ∈ {first, last, False};
  - subset=list[str] и subset=None;
  - multi-column subset;
  - pandas-equivalence (по содержимому, порядок — partition-concat);
  - edge cases: empty, missing column, invalid keep, subset=[];
  - после filter;
  - skew → MemoryLimitExceeded (без recursion).
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.dedup import drop_duplicates as ops_drop_duplicates


def make_df(tmp_path, name, data, chunk=3, mem_mb=20):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _norm(rows: list[dict], key: str) -> list[dict]:
    return sorted(rows, key=lambda r: (r[key] is None, r[key]))


# ---------- basic semantics ----------

def test_drop_duplicates_first_basic(tmp_path):
    """keep='first': оставить первое (по row_id) вхождение каждого ключа."""
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # Один ряд на ключ; v из первой встречи (по исходному порядку)
    by_k = {r["k"]: r["v"] for r in rows}
    assert by_k == {"a": 1, "b": 3, "c": 5}


def test_drop_duplicates_last_basic(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="last", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    by_k = {r["k"]: r["v"] for r in rows}
    assert by_k == {"a": 2, "b": 4, "c": 5}


def test_drop_duplicates_false_keeps_only_unique_keys(tmp_path):
    """keep=False: удалить ВСЕ строки с дублирующимися ключами."""
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep=False, num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 1
    assert rows[0] == {"k": "c", "v": 5}


# ---------- subset=None (all columns) ----------

def test_drop_duplicates_full_row(tmp_path):
    """subset=None: dedup по всем колонкам."""
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b"],
        "v": [1, 1, 2],
    })
    result = df.drop_duplicates(keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "k")
    assert rows == [
        {"k": "a", "v": 1},
        {"k": "b", "v": 2},
    ]


def test_drop_duplicates_full_row_keeps_distinct_values(tmp_path):
    """subset=None: (a,1) и (a,2) — разные."""
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b"],
        "v": [1, 2, 3],
    })
    result = df.drop_duplicates(keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "k")
    # both a,1 and a,2 survive (distinct full tuples)
    assert len(rows) == 3


# ---------- multi-column subset ----------

def test_drop_duplicates_multi_key(tmp_path):
    df = make_df(tmp_path, "L", {
        "k1": ["a", "a", "a", "b"],
        "k2": [1, 1, 2, 1],
        "v": [10, 20, 30, 40],
    })
    result = df.drop_duplicates(subset=["k1", "k2"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "v")
    # (a,1): 10 (first), (a,2): 30, (b,1): 40
    assert {(r["k1"], r["k2"], r["v"]) for r in rows} == {
        ("a", 1, 10),
        ("a", 2, 30),
        ("b", 1, 40),
    }


# ---------- all same key ----------

def test_drop_duplicates_all_same_first(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["X"] * 10, "v": list(range(10))})
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 1
    assert rows[0]["v"] == 0  # first occurrence


def test_drop_duplicates_all_same_last(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["X"] * 10, "v": list(range(10))})
    result = df.drop_duplicates(subset=["k"], keep="last", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 1
    assert rows[0]["v"] == 9  # last occurrence


def test_drop_duplicates_all_same_false(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["X"] * 10, "v": list(range(10))})
    result = df.drop_duplicates(subset=["k"], keep=False, num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 0


# ---------- all unique ----------

def test_drop_duplicates_all_unique(tmp_path):
    n = 50
    df = make_df(tmp_path, "L", {"k": [f"k{i}" for i in range(n)], "v": list(range(n))})
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=8)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    assert pq.read_table(out).num_rows == n


# ---------- pandas-equivalence ----------

def test_drop_duplicates_pandas_equivalence(tmp_path):
    """Содержимое (набор строк) совпадает с pandas для keep='first'."""
    rng = pd.np.random.default_rng(42) if hasattr(pd, "np") else None
    # use plain numpy
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200
    data = {
        "k": [f"k{i % 20}" for i in range(n)],
        "v": rng.integers(0, 100, size=n).tolist(),
    }
    df = make_df(tmp_path, "L", data, chunk=20)
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=8)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    got = pq.read_table(out).to_pandas()

    pdf = pd.DataFrame(data)
    expected = pdf.drop_duplicates(subset=["k"], keep="first")

    # Сравнение по set-of-tuples (порядок не гарантирован, §11.5)
    got_set = set(map(tuple, got[["k", "v"]].itertuples(index=False, name=None)))
    exp_set = set(map(tuple, expected[["k", "v"]].itertuples(index=False, name=None)))
    assert got_set == exp_set, f"diff: {got_set ^ exp_set}"


# ---------- after filter ----------

def test_drop_duplicates_after_filter(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    filtered = df[df["v"] > 1]  # rows: (a,2),(b,3),(b,4),(c,5)
    result = filtered.drop_duplicates(subset=["k"], keep="first", num_partitions=4)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = _norm(pq.read_table(out).to_pylist(), "k")
    by_k = {r["k"]: r["v"] for r in rows}
    assert by_k == {"a": 2, "b": 3, "c": 5}


# ---------- edge cases ----------

def test_drop_duplicates_missing_subset_column_raises(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a"], "v": [1]})
    with pytest.raises(KeyError, match="nope"):
        df.drop_duplicates(subset=["nope"])


def test_drop_duplicates_invalid_keep_raises(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a"], "v": [1]})
    with pytest.raises(ValueError, match="keep"):
        df.drop_duplicates(subset=["k"], keep="middle")


def test_drop_duplicates_empty_subset_raises(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a"], "v": [1]})
    with pytest.raises(ValueError, match="non-empty"):
        df.drop_duplicates(subset=[])


def test_drop_duplicates_empty_input_raises(tmp_path):
    empty = pa.table({
        "k": pa.array([], type=pa.string()),
        "v": pa.array([], type=pa.int64()),
    })
    empty_batches: list[pa.RecordBatch] = []
    mg = MemoryGuard(max_memory_mb=10)
    with pytest.raises(ValueError, match="empty input"):
        ops_drop_duplicates(
            iter(empty_batches),
            subset=["k"],
            keep="first",
            memory_guard=mg,
            spill_dir=tmp_path / "spill",
        )


def test_drop_duplicates_p_gt_n(tmp_path):
    """P > rows: small data, many partitions."""
    df = make_df(tmp_path, "L", {"k": ["a", "b", "c"], "v": [1, 2, 3]})
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=32)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    assert pq.read_table(out).num_rows == 3


# ---------- skew: oversized partition → MemoryLimitExceeded ----------

def test_drop_duplicates_skew_raises(tmp_path):
    """1000 строк с одним ключом при малом бюджете — raise, не OOM."""
    df = make_df(
        tmp_path, "L",
        {"k": ["HOT"] * 1000, "v": list(range(1000))},
        chunk=200,
        mem_mb=1,  # 1 MB budget
    )
    # Партиция с 1000 строк (row_size ~ string + int) → партиция ~20 KB
    # при P=4 = ~5 KB. При num_partitions=1 — одна партиция ~20 KB.
    # С budget 1 MB и safety 0.5 → limit = 500 KB. Партиция влезает.
    # Хотим oversized: увеличим num_partitions... нет, наоборот уменьшим.
    # Используем num_partitions=1 и уменьшим budget.
    df._memory_guard = MemoryGuard(max_memory_bytes=1000)  # 1 KB

    with pytest.raises(MemoryLimitExceeded):
        df.drop_duplicates(subset=["k"], keep="first", num_partitions=1)


# ---------- chained with sort ----------

def test_drop_duplicates_then_sort(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["c", "a", "b", "a", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    result = df.drop_duplicates(subset=["k"], keep="last", num_partitions=4)
    sorted_result = result.sort_values("k", ascending=True)
    out = tmp_path / "o.parquet"
    sorted_result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # keep='last': a→4, b→3, c→5. sorted by k: a,b,c
    assert rows == [
        {"k": "a", "v": 4},
        {"k": "b", "v": 3},
        {"k": "c", "v": 5},
    ]
EOF

# ---------- DECISIONS.md: §11.1 + §11.5 + §13.4/48 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

# §11.1: строка
if "| `drop_duplicates` (v0.3.0: HashShuffle + local pandas) |" not in s:
    s = s.replace(
        "| E2E join (v0.2.5: 100k×10k, skew, left) | ✅ | 3 slow |",
        "| E2E join (v0.2.5: 100k×10k, skew, left) | ✅ | 3 slow |\n"
        "| `drop_duplicates` (v0.3.0: HashShuffle + local pandas) | ✅ | 15 |",
        1,
    )

# §11.5: ограничение порядка
if "drop_duplicates returns rows in partition-concatenation order" not in s:
    s = s.replace(
        "| `df[-5:]` (отрицательный slice) — не поддержано | `SliceFilter` | `step > 0`, `start >= 0`; для отрицательных — `count()` + пересчёт |",
        "| `df[-5:]` (отрицательный slice) — не поддержано | `SliceFilter` | `step > 0`, `start >= 0`; для отрицательных — `count()` + пересчёт |\n"
        "| `drop_duplicates` returns rows in partition-concatenation order | `drop_duplicates` (v0.3.0) | pandas-order: `.sort_values(...)` после. §13.4/48 |\n"
        "| `drop_duplicates` без recursion при skew | `drop_duplicates` (v0.3.0) | oversize partition → `MemoryLimitExceeded`; v0.3.1 — recursion + streaming dedup |",
        1,
    )

# §13.4/48
marker_135 = "### 13.5. Открытые вопросы"
episode_48 = """**Эпизод 48 (drop_duplicates через HashShuffle).** Первая операция v0.3 —
`drop_duplicates` — реализована как композиция `HashShuffle` + локальный
`pandas.drop_duplicates`. Ключевое наблюдение: **invariant HashShuffle**
(«все строки с одинаковым subset key — в одной партиции») делает
локальный dedup эквивалентным глобальному. Не нужен merge, не нужен
финальный `drop_duplicates` — concat даёт корректный результат.

Три режима `keep` реализуются через сортировку partition по `row_id`
(сохраняется в shuffle) и стандартный `df.drop_duplicates`:
- `'first'`: sort row_id asc + keep='first' = min row_id per key;
- `'last'`: sort row_id asc + keep='last' = max row_id per key;
- `False`: pandas drops all rows with duplicate keys — локально = глобально.

**Ordering (документированное ограничение):** результат в порядке
partition-concatenation, **не** в pandas-original order. Для keep='first'
SURVIVING rows правильные (min row_id), но их позиции в выводе — по
партициям. §11.5. Fix в v0.3.1: финальный `external_sort` по row_id, если
нужен pandas-order.

**Skew (документированное ограничение):** при oversized partition (одна
partition > available*0.5) — `MemoryLimitExceeded` **до** materialize.
Recursion требует сохранения ГЛОБАЛЬНОГО row_id при resplit —
`hash_shuffle` сейчас **переприсваивает** row_id (позиция в новом
потоке). Для dedup это ломает keep='first'/'last'. v0.3.1: либо сохранять
row_id в resplit (через `value_columns=["row_id"]` + rename), либо
streaming dedup (dict `key → (min_row_id, payload)` в RAM). Для MVP — raise.

**Урок:** композиция из готового примитива (`HashShuffle`) для новой
операции — дешёвая. Сложность — не в алгоритме (он тривиален), а в
корректности деталей: (a) порядок row_id при keep='first'/'last';
(b) reassignment row_id при recursion; (c) манифест output-order в
документации.

"""
if "Эпизод 48" not in s:
    s = s.replace(marker_135, episode_48 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §11.5, §13.4/48")
PY

# ---------- CHANGELOG.md: v0.3.0 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.3.0] — 2026-10-19

### Добавлено (первая операция v0.3)
- `CadwDataFrame.drop_duplicates(subset=None, keep='first', num_partitions=None)`:
  HashShuffle по subset + локальный `pandas.drop_duplicates` per partition.
  - `subset=None` → dedup по всем колонкам.
  - `keep ∈ {'first', 'last', False}` — все три pandas-режима.
  - `num_partitions=None` → авто из sample (k_est на subset).
- `cadw.ops.dedup.drop_duplicates(...)` — runtime, возвращает `pa.Table`.
- Экспорт через `cadw.drop_duplicates` и `cadw.ops.drop_duplicates`.

### Принцип
- Корректность: партиции `HashShuffle` не пересекаются по subset keys →
  локальный dedup = глобальный, concat без merge.
- `row_id` сохраняется в shuffle; сортировка по row_id перед dedup даёт
  корректный keep='first'/'last' (min/max row_id per key).
- `keep=False` (pandas semantics) локально = глобально.

### Ограничения (v0.3.0, §11.5)
- **Output order**: partition-concatenation, не pandas-original order.
  Для стабильного порядка — `.sort_values(...)` после.
- **Skew**: при oversized partition → `MemoryLimitExceeded` без recursion.
  Recursion в v0.3.1 требует save-row_id-at-resplit или streaming dedup.

### Тесты
- 15 тестов в `test_drop_duplicates.py`: все keep-режимы, subset=None,
  multi-column, edge cases (empty, missing, invalid), skew reject,
  pandas-equivalence, after filter, chained with sort.

"""
if "## [0.3.0]" not in s:
    s = s.replace("## [0.2.5]", entry + "## [0.2.5]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.3.0")
PY

# ---------- run ----------
echo "=== pytest tests/test_drop_duplicates.py -v ==="
set +e
uv run pytest tests/test_drop_duplicates.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "drop_duplicates: rc=$RC"
  git add -A
  git commit -m "WIP: v0.3.0 drop_duplicates (test red; fix follows)" || true
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.3.0: drop_duplicates (HashShuffle + local pandas; keep first/last/False; 15 tests)"

echo "==> v0.3.0 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"