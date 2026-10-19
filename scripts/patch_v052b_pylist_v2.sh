#!/usr/bin/env bash
# patch_v052b_pylist_v2: минимальный, безопасный fix.
# P9 (estimate_unique_keys → pc.dictionary_encode) + P11 (rank 'first' → pc.rank).
# 'min'/'dense' — Python loop per batch (мелкий). 'max'/'last'/'average' — v0.3.2 2-pass.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

# === Branch check (avoid recreate if exists) ===
BR="patch/v052b-pylist"
CUR="$(git branch --show-current)"
if [[ "$CUR" == "$BR" ]]; then
  echo "Already on ${BR}"
elif git rev-parse --verify "$BR" >/dev/null 2>&1; then
  echo "Switching to existing ${BR}"
  git checkout "$BR"
else
  echo "Creating ${BR}"
  git checkout -b "$BR"
fi

# Clean tree check (только tracked)
require_clean_tree || exit 1

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v052b-${TS}"
git archive --format=tar.gz -o ".backups/backup-pre-v052b-${TS}/repo.tgz" HEAD
git tag -a "backup-pre-v052b-${TS}" -m "before patch v0.5.2b (P9+P11 partial)"

# ====================================================================
# P9: estimate_unique_keys — Arrow-native
# ====================================================================
echo ""
echo "=== P9: estimate_unique_keys ==="
uv run python - <<'PY'
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
        - multi-column: binary_join_element_wise → dictionary_encode.
        - Fallback: pc.count_distinct per column (product bound).

        No `to_pylist()` in fast path — no Python-object overhead on strings.
        """
        if sample_batch.num_rows == 0 or total_rows <= 0:
            return 0

        import pyarrow.compute as pc

        ncols = sample_batch.num_columns
        k_sample = None

        if ncols == 1:
            col = sample_batch.column(0)
            try:
                enc = col.dictionary_encode()
                if enc.dictionary is not None:
                    k_sample = len(enc.dictionary)
            except Exception:
                pass
            if k_sample is None:
                try:
                    k_sample = int(pc.count_distinct(col).as_py())
                except Exception:
                    k_sample = len(set(col.to_pylist()))

        if k_sample is None:
            # multi-column: binary_join → dict_encode, else product fallback
            try:
                if hasattr(pc, "binary_join_element_wise"):
                    cols = [sample_batch.column(i) for i in range(ncols)]
                    str_cols = [
                        c if pa.types.is_string(c.type) else pc.cast(c, pa.string())
                        for c in cols
                    ]
                    joined = str_cols[0]
                    for c in str_cols[1:]:
                        joined = pc.binary_join_element_wise(joined, c, "|")
                    enc = joined.dictionary_encode()
                    if enc.dictionary is not None:
                        k_sample = len(enc.dictionary)
            except Exception:
                pass
            if k_sample is None:
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
p.write_text(s)
print("  OK: estimate_unique_keys → pc.dictionary_encode")
PY

# ====================================================================
# P11: rank 'first' — Arrow-native (только 'first', остальные без изменений)
# ====================================================================
echo ""
echo "=== P11: rank 'first' → pc.rank ==="
uv run python - <<'PY'
from pathlib import Path
p = Path("cadw/ops/rank.py")
s = p.read_text()

old = '''    try:
        for batch in sorted_batches:
            memory_guard.check_batch(batch, name="rank.streamable.input")
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
                    pa.array(ranks, type=dtype),
                ],
                schema=schema,
            )
            writer.write_batch(rb)
    finally:
        writer.close()
    return rank_path'''

new = '''    import pyarrow.compute as _pc

    try:
        for batch in sorted_batches:
            memory_guard.check_batch(batch, name="rank.streamable.input")
            vals_arr = batch.column(by)
            row_ids_arr = batch.column("row_id")
            n = len(vals_arr)

            # v0.5.2b (P11): 'first' fast path — Arrow-native, no to_pylist.
            # `pc.rank(batch, sort_keys=by, tiebreaker='first')` — 1-based
            # ordinal rank per batch. Глобальный offset: global_pos.
            if method == "first":
                local_rank = _pc.rank(
                    vals_arr, sort_keys=by, tiebreaker="first",
                )
                global_rank = _pc.add(
                    local_rank, pa.scalar(global_pos, type=pa.int64()),
                )
                global_pos += n
                if pct:
                    global_rank = _pc.divide(
                        _pc.cast(global_rank, pa.float64()),
                        pa.scalar(float(total_n), type=pa.float64()),
                    )
                global_rank = _pc.cast(global_rank, dtype)

                rb = pa.RecordBatch.from_arrays(
                    [row_ids_arr, global_rank], schema=schema,
                )
                writer.write_batch(rb)
                continue

            # 'min' / 'dense' — Python-loop per batch (мелкий batch).
            vals = vals_arr.to_pylist()
            row_ids = row_ids_arr.to_pylist()

            ranks: list = []
            for v in vals:
                global_pos += 1
                if prev_val is _UNSET:
                    group_start = global_pos
                    dense_counter = 1
                elif v != prev_val:
                    group_start = global_pos
                    dense_counter += 1

                if method == "min":
                    r = group_start
                else:  # dense
                    r = dense_counter

                ranks.append((r / total_n) if pct else r)
                prev_val = v

            rb = pa.RecordBatch.from_arrays(
                [
                    pa.array(row_ids, type=pa.uint64()),
                    pa.array(ranks, type=dtype),
                ],
                schema=schema,
            )
            writer.write_batch(rb)
    finally:
        writer.close()
    return rank_path'''

if old not in s:
    raise SystemExit("_compute_streamable loop not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("  OK: 'first' → pc.rank; 'min'/'dense' остаются Python-loop")
PY

# ====================================================================
# Tests
# ====================================================================
echo ""
echo "=== Add tests ==="
cat > tests/test_pylist_fixes.py <<'EOF'
"""v0.5.2b: P9 (estimate_unique_keys) + P11 (rank 'first' Arrow-native)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


# ---------- P9: estimate_unique_keys ----------

def test_estimate_unique_keys_single_col():
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array(["a", "b", "c", "a", "b"])], names=["k"]
    )
    assert guard.estimate_unique_keys(sample, total_rows=1000) == 600


def test_estimate_unique_keys_single_col_all_unique():
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3, 4, 5])], names=["k"]
    )
    assert guard.estimate_unique_keys(sample, total_rows=100) == 100


def test_estimate_unique_keys_multi_col():
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array(["a", "a", "b"]), pa.array([1, 2, 1])],
        names=["k1", "k2"],
    )
    assert guard.estimate_unique_keys(sample, total_rows=300) == 300


def test_estimate_unique_keys_empty():
    guard = MemoryGuard(max_memory_mb=100)
    sample = pa.RecordBatch.from_arrays(
        [pa.array([], type=pa.string())], names=["k"]
    )
    assert guard.estimate_unique_keys(sample, total_rows=1000) == 0


def test_estimate_unique_keys_uses_arrow_native():
    """AST guard: fast path uses dictionary_encode, not to_pylist loop."""
    import inspect
    src = inspect.getsource(MemoryGuard.estimate_unique_keys)
    assert "dictionary_encode" in src, (
        "P9 fix: estimate_unique_keys must use pc.dictionary_encode"
    )
    # to_pylist only allowed in except fallback
    assert "count_distinct" in src


# ---------- P11: rank 'first' Arrow-native ----------

def test_rank_first_still_works(tmp_path):
    df = make_df(tmp_path, data={"k": ["b", "a", "c", "a"], "v": [1, 2, 3, 4]})
    result = df.rank(by="k", method="first")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [3, 1, 4, 2]


def test_rank_first_pandas_equivalence(tmp_path):
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


def test_rank_first_pct(tmp_path):
    df = make_df(tmp_path, data={"k": [10, 20, 30, 40], "v": list(range(4))})
    result = df.rank(by="k", method="first", pct=True)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [0.25, 0.5, 0.75, 1.0]


def test_rank_first_uses_pc_rank():
    """AST guard: 'first' path uses pc.rank (Arrow-native)."""
    import inspect
    from cadw.ops import rank as rank_mod
    src = inspect.getsource(rank_mod._compute_streamable)
    assert "pc.rank" in src or "_pc.rank" in src, (
        "P11 fix: rank 'first' must use pc.rank (Arrow-native)"
    )


# ---------- Sanity: other methods unchanged ----------

def test_rank_min_unchanged(tmp_path):
    df = make_df(tmp_path, data={"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="min")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [1, 4, 1, 5, 1]


def test_rank_dense_unchanged(tmp_path):
    df = make_df(tmp_path, data={"k": ["b", "a", "c", "a", "b"], "v": list(range(5))})
    result = df.rank(by="k", method="dense")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [2, 1, 3, 1, 2]


def test_rank_max_unchanged(tmp_path):
    df = make_df(tmp_path, data={"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="max")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k_rank"] for r in rows] == [3, 4, 3, 5, 3]
EOF

# ====================================================================
# Run tests
# ====================================================================
echo ""
echo "=== pytest tests/test_pylist_fixes.py -v ==="
set +e
uv run pytest tests/test_pylist_fixes.py -v
RC1=$?
set -e
if [[ $RC1 -ne 0 ]]; then
  echo "new tests FAILED: rc=$RC1"
  echo ""
  echo "=== Failure details ==="
  uv run pytest tests/test_pylist_fixes.py -x -q 2>&1 | tail -50
  # WIP commit
  git add -A
  git commit -m "WIP: v0.5.2b P9+P11 (test red)" || true
  exit 1
fi

echo ""
echo "=== pytest tests/test_rank.py tests/test_quantile.py ==="
set +e
uv run pytest tests/test_rank.py tests/test_quantile.py -q
RC2=$?
set -e
if [[ $RC2 -ne 0 ]]; then
  echo "regression: rc=$RC2"
  exit 1
fi

echo ""
echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.2b: P9 (estimate_unique_keys pc.dictionary_encode) + P11 (rank 'first' pc.rank); 12 tests"

echo ""
echo "==> v0.5.2b applied on ${BR}"