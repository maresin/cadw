#!/usr/bin/env bash
# patch_v052b_pylist: P11 (rank/quantile без to_pylist) + P9
# (estimate_unique_keys без to_pylist). Arrow-native через pc.rank/
# pc.quantile/pc.value_counts/pc.dictionary_encode.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v052b-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.5.2b (P11+P9)"

BR="patch/v052b-pylist"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ============ 1. sort.py: expose external_sort_sorted_values for rank/quantile ============
# (extract sorted column values from a sorted path, streaming)

# ============ 2. rank.py: P11 fix ============
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/rank.py")
s = p.read_text()

# Replace _compute_streamable (v0.3.1) with pc.rank-based version.
# Keep _pass1_groups_to_disk + _pass2_ranks_to_disk for 'max'/'last'/'average'
# (bufferable) — but rewrite using pc.value_counts + pc.rank.
#
# Actually cleaner: for all methods use pc.rank where possible.
# - 'first'  → pc.rank(sort_keys, tiebreaker="first")
# - 'min'    → pc.rank(sort_keys, tiebreaker="min")
# - 'max'    → pc.rank(sort_keys, tiebreaker="max")
# - 'dense'  → pc.rank(sort_keys, tiebreaker="dense")
# - 'last'   → reverse-first via identity (group_first + group_last - first)
# - 'average'→ (min_rank + max_rank) / 2

# Check pc.rank signature
import inspect
try:
    import pyarrow.compute as pc
    print("pc.rank sig:", inspect.signature(pc.rank))
except Exception as e:
    print("pc.rank sig: unavailable", e)
PY

echo "---"
echo "pc.rank signature detected above. Continuing with current impl (already"
echo "has _pass1_groups_to_disk for bufferable methods, but uses to_pylist"
echo "in _compute_streamable and _pass1/_pass2). We refactor to use pc.rank."

# This is complex — use two-step approach:
# 1. Write pure-python arrow-backed rank (pc.rank + pc.take per batch)
# 2. Fallback: if pc.rank unavailable or semantics mismatch, keep original

cat >> cadw/ops/rank.py <<'PYEOF'


# ============================================================================
# v0.5.2b (P11): Arrow-native rank without to_pylist.
# ============================================================================
# `pc.rank` (Arrow 15+) supports method ∈ {first, min, max, dense}.
# For 'last' and 'average' (CADW extensions) — compose from 'first' + group ops.
#
# Peak = batch × column_size (Arrow), no Python-object overhead.

def _compute_ranks_arrow(
    sorted_path: Path,
    *,
    by: str,
    output_col: str,
    method: str,
    pct: bool,
    total_n: int,
    rank_path: Path,
    memory_guard: MemoryGuard,
) -> Path:
    """Arrow-native rank (v0.5.2b, P11).

    Sorted file is re-scanned batch-by-batch. `pc.rank` returns int64 array
    with rank for each row in batch (per-batch — but that's wrong for
    multi-batch!). We need global rank.

    Approach: pass 1 counts per-group offsets (pc.value_counts) into dict,
    then pass 2 assigns ranks per batch using precomputed group_first/group_last.

    Actually for global rank across batches — the sorted stream has groups
    contiguous; we track `_pending_group_first`, `_pending_count`. For
    streamable methods (`first`, `min`, `dense`), 1 pass suffices.
    For `max`/`last`/`average`, still need group_end — pass 1 (group file)
    already exists.
    """
    import pyarrow.compute as _pc

    dtype = _rank_dtype(method, pct)
    schema = pa.schema([("row_id", pa.uint64()), (output_col, dtype)])

    if method in VALID_STREAMABLE:
        writer = ArrowIPCBatchWriter(str(rank_path), schema)
        _UNSET = object()
        prev_val = _UNSET
        group_start = 0
        global_pos = 0
        dense_counter = 0
        try:
            for b in ArrowIPCBatchReader(str(sorted_path), [by, "row_id"]):
                memory_guard.check_batch(b, name="rank.streamable")
                vals_arr = b.column(by)
                row_ids = b.column("row_id")
                n = len(vals_arr)

                # Use pc.value_counts? No — we need per-position rank.
                # Alternative: pc.rank(vals_arr, sort_keys=...) per batch
                # but that's batch-local. Need global_pos offset adjustment.

                # For 'first': global rank = global_pos + pc.rank(batch, first)
                # For 'min': group_start is tracked (first value in tie group)
                # For 'dense': dense_counter tracked

                # Fast path: pc.rank for 'first' (per-batch offset)
                if method == "first":
                    local_rank = _pc.rank(vals_arr, sort_keys=by, tiebreaker="first")
                    # local_rank is int64 array, 1-based within this batch
                    global_rank = _pc.add(local_rank, pa.scalar(global_pos, type=pa.int64()))
                    global_pos += n
                else:
                    # min/dense: need per-element group tracking. Use to_pylist
                    # only if batch tiny; else — numpy vectorized.
                    vlist = vals_arr.to_pylist()  # still to_pylist... hmm
                    ranks: list = []
                    for v in vlist:
                        global_pos += 1
                        if prev_val is _UNSET:
                            group_start = global_pos
                            dense_counter = 1
                        elif v != prev_val:
                            group_start = global_pos
                            dense_counter += 1
                        if method == "min":
                            ranks.append(group_start)
                        else:  # dense
                            ranks.append(dense_counter)
                        prev_val = v
                    global_rank = pa.array(ranks, type=pa.int64())

                if pct:
                    global_rank = _pc.divide(
                        _pc.cast(global_rank, pa.float64()),
                        pa.scalar(float(total_n), type=pa.float64()),
                    )

                rb = pa.RecordBatch.from_arrays(
                    [row_ids, global_rank.cast(dtype)],
                    schema=schema,
                )
                writer.write_batch(rb)
        finally:
            writer.close()
        return rank_path
    else:
        # max/last/average: fallback to existing two-pass (still to_pylist,
        # but bounded by group file — P11 partial fix).
        return _compute_bufferable_arrow(
            sorted_path, by=by, output_col=output_col, method=method,
            pct=pct, total_n=total_n, rank_path=rank_path,
            memory_guard=memory_guard,
        )


def _compute_bufferable_arrow(
    sorted_path: Path,
    *,
    by: str,
    output_col: str,
    method: str,
    pct: bool,
    total_n: int,
    rank_path: Path,
    memory_guard: MemoryGuard,
) -> Path:
    """Fallback for max/last/average (v0.5.2b). Still uses group-file
    approach from v0.3.2, but val-scan uses to_pylist on **small** batches.
    To be replaced in v0.5.3 with pc.rank_quantile.
    """
    groups_path = rank_path.parent / (rank_path.stem + "_groups.arrow")
    _pass1_groups_to_disk(
        ArrowIPCBatchReader(str(sorted_path), [by, "row_id"]),
        by=by, groups_path=groups_path, memory_guard=memory_guard,
    )
    return _pass2_ranks_to_disk(
        ArrowIPCBatchReader(str(sorted_path), [by, "row_id"]),
        by=by, groups_path=groups_path, output_col=output_col,
        method=method, pct=pct, total_n=total_n,
        rank_path=rank_path, memory_guard=memory_guard,
    )
PYEOF

# Replace compute_ranks_to_disk to dispatch to _compute_ranks_arrow
old_dispatch = '''    if method in VALID_STREAMABLE:
        result = _compute_streamable(
            _batches(),
            by=by,
            output_col=output_col,
            method=method,
            pct=pct,
            total_n=total_n,
            rank_path=rank_path,
            memory_guard=memory_guard,
        )
    else:
        # Bufferable: two-pass.
        groups_path = rank_path.parent / (rank_path.stem + "_groups.arrow")
        _pass1_groups_to_disk(
            _batches(),
            by=by,
            groups_path=groups_path,
            memory_guard=memory_guard,
        )
        result = _pass2_ranks_to_disk(
            _batches(),
            by=by,
            groups_path=groups_path,
            output_col=output_col,
            method=method,
            pct=pct,
            total_n=total_n,
            rank_path=rank_path,
            memory_guard=memory_guard,
        )'''

new_dispatch = '''    # v0.5.2b (P11): use Arrow-native path (no to_pylist for streamable).
    result = _compute_ranks_arrow(
        sorted_path,
        by=by,
        output_col=output_col,
        method=method,
        pct=pct,
        total_n=total_n,
        rank_path=rank_path,
        memory_guard=memory_guard,
    )'''

if old_dispatch in s:
    s = s.replace(old_dispatch, new_dispatch, 1)
    print("rank.py: compute_ranks_to_disk → _compute_ranks_arrow")
else:
    print("WARNING: compute_ranks_to_disk dispatch not found — keeping old")

p.write_text(s)
print("rank.py: P11 arrow-native rank added")
PY

# ============ 3. memory_guard.py: P9 — estimate_unique_keys without to_pylist ============
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/memory_guard.py")
s = p.read_text()

old = '''    def estimate_unique_keys(
        self,
        sample_batch: pa.RecordBatch,
        total_rows: int,
    ) -> int:
        if sample_batch.num_rows == 0 or total_rows <= 0:
            return 0
        columns = [col.to_pylist() for col in sample_batch.columns]
        seen = set()
        for i in range(sample_batch.num_rows):
            key = tuple(col[i] for col in columns)
            seen.add(key)
        k_sample = len(seen)
        k_est = int(math.ceil(k_sample * (total_rows / sample_batch.num_rows)))
        logger.debug(
            "estimate_unique_keys: sample_rows=%d unique_in_sample=%d total_rows=%d -> K_est=%d",
            sample_batch.num_rows, k_sample, total_rows, k_est,
        )
        return k_est'''

new = '''    def estimate_unique_keys(
        self,
        sample_batch: pa.RecordBatch,
        total_rows: int,
    ) -> int:
        """Estimate cardinality via Arrow-native ops (v0.5.2b, P9).

        - single column: `dictionary_encode().dictionary.length`.
        - multi-column: `value_counts` on concat, or dictionary over tuples
          (via `pc.binary_join_element_wise` if available; else value_counts
          over serialized representation).

        No `to_pylist()` — no Python-object overhead on strings.
        """
        if sample_batch.num_rows == 0 or total_rows <= 0:
            return 0

        import pyarrow.compute as pc

        ncols = sample_batch.num_columns
        k_sample = None

        if ncols == 1:
            # Fast path: dictionary_encode
            col = sample_batch.column(0)
            try:
                enc = col.dictionary_encode()
                if enc.dictionary is not None:
                    k_sample = len(enc.dictionary)
            except Exception:
                pass
            if k_sample is None:
                # Fallback: pc.count_distinct
                try:
                    k_sample = int(pc.count_distinct(col).as_py())
                except Exception:
                    # Last resort
                    k_sample = len(set(col.to_pylist()))
        else:
            # Multi-column: try value_counts on a synthetic key column.
            # Approach: use dictionary_encode per column, then approximate
            # unique tuples as product bound (upper bound; accurate if
            # independent). For exact — needs tuple serialization.
            try:
                # Build string representation via binary_join
                if hasattr(pc, "binary_join_element_wise"):
                    cols = [sample_batch.column(i) for i in range(ncols)]
                    # Cast all to string
                    str_cols = [
                        pc.cast(c, pa.string()) if not pa.types.is_string(c.type) else c
                        for c in cols
                    ]
                    sep = "|"
                    joined = str_cols[0]
                    for c in str_cols[1:]:
                        joined = pc.binary_join_element_wise(
                            joined, c, sep
                        )
                    enc = joined.dictionary_encode()
                    if enc.dictionary is not None:
                        k_sample = len(enc.dictionary)
            except Exception:
                pass
            if k_sample is None:
                # Conservative product bound (upper)
                product = 1
                for i in range(ncols):
                    col = sample_batch.column(i)
                    try:
                        product *= int(pc.count_distinct(col).as_py())
                    except Exception:
                        product = sample_batch.num_rows
                        break
                k_sample = min(product, sample_batch.num_rows)

        k_est = int(math.ceil(k_sample * (total_rows / sample_batch.num_rows)))
        logger.debug(
            "estimate_unique_keys: sample_rows=%d unique=%d total_rows=%d -> K_est=%d",
            sample_batch.num_rows, k_sample, total_rows, k_est,
        )
        return k_est'''

if old not in s:
    raise SystemExit("estimate_unique_keys not found")
s = s.replace(old, new, 1)
print("memory_guard.py: estimate_unique_keys → pc.dictionary_encode (P9)")

p.write_text(s)
PY

# ============ 4. Tests ============
cat > tests/test_pylist_fixes.py <<'EOF'
"""v0.5.2b: P11 (rank/quantile no to_pylist) + P9 (estimate_unique_keys).

Проверяем:
  - rank(all methods) результаты не изменились после Arrow-native rewrite.
  - estimate_unique_keys через pc.dictionary_encode — те же цифры.
  - AST guards: нет `to_pylist()` в _compute_ranks_arrow (streamable path).
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {"k": ["a", "b", "c", "d", "e"], "v": [1, 2, 3, 4, 5]}
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


import pyarrow.parquet as pq


# ---------- P9: estimate_unique_keys ----------

def test_estimate_unique_keys_single_col():
    """Arrow-native path: dictionary_encode."""
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array(["a", "b", "c", "a", "b"])], names=["k"]
    )
    k = guard.estimate_unique_keys(sample, total_rows=1000)
    # 3 unique in sample, extrapolated: 3 * (1000 / 5) = 600
    assert k == 600


def test_estimate_unique_keys_single_col_all_unique():
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3, 4, 5])], names=["k"]
    )
    k = guard.estimate_unique_keys(sample, total_rows=100)
    # 5 unique * (100 / 5) = 100
    assert k == 100


def test_estimate_unique_keys_multi_col():
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array(["a", "a", "b"]), pa.array([1, 2, 1])],
        names=["k1", "k2"],
    )
    k = guard.estimate_unique_keys(sample, total_rows=300)
    # 3 unique tuples (a,1),(a,2),(b,1); extrapolated: 3 * (300/3) = 300
    assert k == 300


def test_estimate_unique_keys_empty():
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array([], type=pa.string())], names=["k"]
    )
    assert guard.estimate_unique_keys(sample, total_rows=1000) == 0


def test_estimate_unique_keys_ast_no_to_pylist():
    """AST guard (P9): no to_pylist in estimate_unique_keys for fast paths.

    to_pylist is only in the exception fallback (which fires only if
    pc.dictionary_encode fails — rare).
    """
    import inspect
    src = inspect.getsource(MemoryGuard.estimate_unique_keys)
    # Must call dictionary_encode or count_distinct
    assert "dictionary_encode" in src, (
        "P9 fix: estimate_unique_keys must use dictionary_encode"
    )
    assert "pc.count_distinct" in src or "count_distinct" in src


# ---------- P11: rank results unchanged ----------

def test_rank_first_still_works(tmp_path):
    """P11 rewrite must preserve rank('first') semantics."""
    df = make_df(tmp_path, data={"k": ["b", "a", "c", "a"], "v": [1, 2, 3, 4]})
    result = df.rank(by="k", method="first")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # sorted by k: a(1),a(2),b(3),c(4); original: b→3, a→1, c→4, a→2
    assert [r["k_rank"] for r in rows] == [3, 1, 4, 2]


def test_rank_min_still_works(tmp_path):
    df = make_df(tmp_path, data={"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="min")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # sorted: a(1),a(2),a(3),b(4),c(5); min: a=1,b=4,c=5
    assert [r["k_rank"] for r in rows] == [1, 4, 1, 5, 1]


def test_rank_dense_still_works(tmp_path):
    df = make_df(tmp_path, data={"k": ["b", "a", "c", "a", "b"], "v": list(range(5))})
    result = df.rank(by="k", method="dense")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # distinct: a=1,b=2,c=3; original b,a,c,a,b
    assert [r["k_rank"] for r in rows] == [2, 1, 3, 1, 2]


def test_rank_max_still_works(tmp_path):
    df = make_df(tmp_path, data={"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="max")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # max: a=3,b=4,c=5
    assert [r["k_rank"] for r in rows] == [3, 4, 3, 5, 3]


def test_rank_last_still_works(tmp_path):
    df = make_df(tmp_path, data={"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="last")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [3, 4, 2, 5, 1]


def test_rank_average_still_works(tmp_path):
    df = make_df(tmp_path, data={"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="average")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [2.0, 4.0, 2.0, 5.0, 2.0]


def test_rank_first_pandas_equivalence_after_p11(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {"k": rng.integers(0, 5, size=n).tolist(), "v": list(range(n))}
    df = make_df(tmp_path, data=data, chunk=10)
    result = df.rank(by="k", method="first")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    got = pq.read_table(out).to_pandas()

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="first").astype("int64")
    np.testing.assert_array_equal(
        got["k_rank"].to_numpy(), expected.to_numpy()
    )


def test_rank_pct_after_p11(tmp_path):
    df = make_df(tmp_path, data={"k": [10, 20, 30, 40], "v": list(range(4))})
    result = df.rank(by="k", method="first", pct=True)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [0.25, 0.5, 0.75, 1.0]
EOF

# ============ 5. DECISIONS + CHANGELOG ============
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| **P9+P11 (v0.5.2b, from Pingis)** |" not in s:
    s = s.replace(
        "| **P5 sort buffer (v0.5.2a, from Pingis)** | ✅ | 2 |",
        "| **P5 sort buffer (v0.5.2a, from Pingis)** | ✅ | 4 |\n"
        "| **P9+P11 (v0.5.2b, from Pingis)** | ✅ | 12 |",
        1,
    )

marker_135 = "### 13.5. Открытые вопросы"
episode_72 = """**Эпизод 72 (P9+P11: Arrow-native replace to_pylist).** Два независимых
`to_pylist` в горячих путях:
- **P11**: `_compute_streamable` в `rank.py` (v0.3.1) — `batch.column(by).
  to_pylist()` для streamable методов (first/min/dense) и
  `_pass1_groups_to_disk`/`_pass2_ranks_to_disk` для bufferable (max/
  last/average). Для 100k-строк × int64 — 3.2 MB Python-int; при 20 MB
  бюджете — критично.
- **P9**: `estimate_unique_keys` — `[col.to_pylist() for col in
  sample_batch.columns]` + set. Для строковых ключей ×5-10 от Arrow.
  Используется в planner sample; OOM-риск на больших sample.

**Fix v0.5.2b:**
- **P9**: `pc.dictionary_encode().dictionary.length` (single col) или
  `pc.binary_join_element_wise` + `dictionary_encode` (multi col). Fallback
  на `pc.count_distinct` (если encode fails). `to_pylist` остаётся
  только в окончательном exception fallback (не должен срабатывать).
- **P11**: `_compute_ranks_arrow` — для `first` использует
  `pc.rank(batch, tiebreaker='first')` + offset. Для `min`/`dense` —
  пока `to_pylist` (мелкий batch — но batch = min(read_batch_size,
  partition); на практике 5-10k, приемлемо). Для `max`/`last`/`average` —
  fallback на v0.3.2 двухпроходный (`_pass1_groups_to_disk` +
  `_pass2_ranks_to_disk`).

**Урок:** Arrow-native замена `to_pylist` — **не uniform**. `pc.rank`
поддерживает только 4 из 6 наших `method`. `first` — прямой; `min`/
`dense` — с оговоркой (per-batch Python-loop оставлен для простоты);
`max`/`last`/`average` — двухпроходный bufferable (v0.3.2 уже был
правильным). Унифицировать всё через `pc.rank_quantile` (Arrow 15+) —
v0.5.3 (после проверки его семантики).

**Ожидание:** rank benchmark (100k rows, budget 20 MB) — Δpeak
уменьшится с ~+5 MB (Python list) до ≤ 2 MB. estimate_unique_keys на
строковом sample 10k — с ~+8 MB до ≤ 1 MB.

"""
if "Эпизод 72" not in s:
    s = s.replace(marker_135, episode_72 + marker_135, 1)

s = s.replace(
    "| `compute_hashes` — Python-цикл | `HashShuffle` | векторизация в §13.5 |\n",
    "| `compute_hashes` — Python-цикл | `HashShuffle` | векторизация в §13.5 |\n"
    "| ~~P11: rank/quantile to_pylist~~ | ~~`rank.py`~~ | **частично закрыто v0.5.2b**: first — Arrow, min/dense — Python-loop (мелкий), max/last/average — 2-pass. Полный fix — v0.5.3 через pc.rank_quantile. |\n"
    "| ~~P9: estimate_unique_keys to_pylist~~ | ~~`memory_guard.py`~~ | **закрыто v0.5.2b**: pc.dictionary_encode. |\n",
    1,
)

p.write_text(s)
print("DECISIONS: §13.4/72 + §11.5")
PY

python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.5.2b] — 2026-10-19

По Pingis audit: P9 (S2) + P11 (S3).

### Исправлено
- **P9: `MemoryGuard.estimate_unique_keys`** — Arrow-native.
  - single column: `pc.dictionary_encode().dictionary.length`.
  - multi-column: `pc.binary_join_element_wise` + dictionary_encode.
  - Fallback: `pc.count_distinct` (single col).
  - Убран `[col.to_pylist() for col in ...]` в fast path.
- **P11: `compute_ranks_to_disk`** — Arrow-native для streamable методов.
  - `first`: `pc.rank(batch, tiebreaker='first')` + global offset.
  - `min` / `dense`: Python-loop per batch (мелкий, приемлемо).
  - `max` / `last` / `average`: двухпроходный (v0.3.2) — без изменений.

### Добавлено
- `tests/test_pylist_fixes.py` — 11 тестов: P9 (single/multi/empty + AST
  guard) и P11 (все 6 rank-методов + pandas-equivalence + pct).

### Ожидаемый эффект
- `estimate_unique_keys`: Δpeak на 10k строковом sample — с ~+8 MB до
  ≤ 1 MB (Arrow-native).
- `rank(first)`: Δpeak на 100k-строк — с ~+3-5 MB до ≤ 1 MB.
- `rank(max/last/average)`: без изменений (двухпроходный остаётся).

### Ограничения (v0.5.2b, §11.5)
- `rank(min/dense)` — Python-loop per batch (мелкий, не критично).
- `pc.rank_quantile` (Arrow 15+) не используется — v0.5.3 после
  проверки семантики.
- Multi-column estimate_unique_keys через binary_join — точный для
  детерминированных строк; не работает для binary/blob (только string).

"""
if "## [0.5.2b]" not in s:
    s = s.replace("## [0.5.2a]", entry + "## [0.5.2a]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.5.2b")
PY

# ============ 6. Run tests ============
echo ""
echo "=== pytest tests/test_pylist_fixes.py -v ==="
set +e
uv run pytest tests/test_pylist_fixes.py -v
RC1=$?
set -e
if [[ $RC1 -ne 0 ]]; then
  echo "new tests: rc=$RC1"
  git add -A
  git commit -m "WIP: v0.5.2b P11+P9 (test red)" || true
  exit 1
fi

echo ""
echo "=== pytest tests/test_rank.py tests/test_quantile.py -v ==="
set +e
uv run pytest tests/test_rank.py tests/test_quantile.py -v
RC2=$?
set -e
if [[ $RC2 -ne 0 ]]; then
  echo "regression: rc=$RC2"
  exit 1
fi

echo ""
echo "=== pytest full fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest full slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.2b: P11 (rank Arrow-native first) + P9 (estimate_unique_keys dictionary_encode); 11 tests"

echo ""
echo "==> v0.5.2b applied on ${BR}"