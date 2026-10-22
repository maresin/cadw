#!/usr/bin/env python3
# v0.9.23f — CadwDataFrame.sample() (AUDIT_FULL_v0922 7.A.2, Top-10 gap #1).
#
# ONE method, three modes:
#   df.sample(n=None, frac=None, replace=False, weights=None,
#             random_state=None)
#
# Algorithms (streaming, bounded memory):
#   - replace=False, no weights: Algorithm R reservoir, vectorized
#     batch decisions. Expected replacements ~n*ln(N/n). Peak: O(n).
#   - replace=False, weights=col: Efraimidis-Spirakis A-Res (u^(1/w)
#     top-n keys, min-heap). Per-row heap ops; vectorized key gen
#     per batch. Peak: O(n). Time O(N log n) — documented N<1e7 limit.
#   - replace=True: vectorized rng.integers(0, N, n) + iloc[list].
#     Peak: O(n).
#
# n >= N, replace=False: shuffle full set (matches pandas).
# n > N, replace=False: ValueError (matches pandas).
# weights + replace=True: NotImplementedError (honest).
#
# Return: CadwDataFrame (chainable). Row order = sample order (random).
# Determinism: random_state=None -> seed=42 (like KFold).
# MemoryGuard: check_state(4*n*row_bytes) before iloc materialize.
#
# Run: uv run python scripts/patch_v0923f_sample.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess,
)


# ============ 1. CadwDataFrame.sample() ============

ANCHOR = (
    '        return {\n'
    '            cols[i]: {cols[j]: float(corr_mat[i, j]) for j in range(len(cols))}\n'
    '            for i in range(len(cols))\n'
    '        }\n'
    '\n'
    '    # ---------------- cumulative (v0.9.23e) ----------------'
)

SAMPLE_METHOD = '''        return {
            cols[i]: {cols[j]: float(corr_mat[i, j]) for j in range(len(cols))}
            for i in range(len(cols))
        }

    # ---------------- sample (v0.9.23f) ----------------

    def sample(
        self,
        n=None,
        frac=None,
        replace: bool = False,
        weights: "str | None" = None,
        random_state=None,
    ) -> "CadwDataFrame":
        """Random sample of rows. Streaming, bounded memory.

        v0.9.23f (AUDIT_FULL_v0922 item 7.A.2, Top-10 gap #1).
        Prototype validated (stratified_sample, RAM 1.01 MB @ 32 MB).

        Args:
            n: number of rows (int >= 0). Exactly one of n / frac.
            frac: fraction of total rows (0 < frac <= 1). n = max(1,
                round(N*frac)) for frac > 0; frac=0 -> n=0.
            replace: sample with replacement. Default False. n can
                exceed N only if replace=True.
            weights: name of a numeric weight column. Sampling is
                proportional to weight (Efraimidis-Spirakis A-Res).
                Not supported with replace=True (NotImplementedError).
                Non-positive / null weights are skipped.
            random_state: int seed; None -> 42 (deterministic).

        Returns:
            CadwDataFrame, rows in sample order (random).

        Raises:
            ValueError: exactly one of n/frac not given; n > N with
                replace=False; frac out of (0, 1]; n < 0; both n and
                frac; weights all-zero/null.
            NotImplementedError: weights + replace=True.
            KeyError: weights column missing.
            TypeError: weights column non-numeric.
            MemoryLimitExceeded: 4*n*row_bytes > state_limit.
        """
        import numpy as np
        from cadw.core.memory_guard import MemoryLimitExceeded as _MLE
        from cadw.dataset import Dataset as _DS

        # ---- validation ----
        if (n is None) == (frac is None):
            raise ValueError(
                "sample: exactly one of n or frac must be provided"
            )
        if frac is not None:
            if not (0.0 < float(frac) <= 1.0):
                raise ValueError(
                    f"sample: frac must be in (0, 1]; got {frac!r}"
                )
        if not isinstance(replace, bool):
            raise TypeError(
                f"sample: replace must be bool; got {type(replace).__name__}"
            )

        N = self.count()
        if N == 0:
            # Empty view
            schema = pa.schema(
                [self._dataset.schema.field(c) for c in self._columns]
            )
            empty_ds = _DS.from_batches(
                schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        if frac is not None:
            if float(frac) == 0.0:
                n = 0
            else:
                n = max(1, int(round(N * float(frac))))

        if not isinstance(n, int):
            raise TypeError(
                f"sample: n must be int; got {type(n).__name__}"
            )
        if n < 0:
            raise ValueError(f"sample: n must be >= 0; got {n}")

        if weights is not None and replace:
            raise NotImplementedError(
                "sample: weights + replace=True is not supported "
                "(would need O(N) full-weight materialization). "
                "Use replace=False with weights, or use unweighted "
                "sampling with replacement."
            )

        if weights is not None:
            if weights not in self._columns:
                raise KeyError(
                    f"sample: weights column {weights!r} not in "
                    f"{self._columns}"
                )
            t = self._dataset.schema.field(weights).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"sample: weights column {weights!r} has type "
                    f"{t}; numeric required"
                )

        if n == 0:
            schema = pa.schema(
                [self._dataset.schema.field(c) for c in self._columns]
            )
            empty_ds = _DS.from_batches(
                schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        if n > N and not replace:
            raise ValueError(
                f"sample: n={n} > N={N} with replace=False. "
                f"Set replace=True or reduce n."
            )

        seed = 42 if random_state is None else int(random_state)
        rng = np.random.default_rng(seed)

        # ---- memory guard: 4*n*row_bytes for iloc materialize ----
        from cadw.core.sampling import peek_first_batch
        first_batch = peek_first_batch(self._iter_filtered_batches())
        if first_batch is not None:
            row_bytes_est = self._memory_guard.reestimate_from_batch(
                first_batch,
            )
        else:
            row_bytes_est = max(
                1,
                self._memory_guard.estimate_row_size_from_schema(
                    pa.schema(
                        [self._dataset.schema.field(c)
                         for c in self._columns]
                    )
                ),
            )
        self._memory_guard.check_state(
            4 * n * row_bytes_est,
            name=f"sample(n={n})",
        )

        # ---- replace=True: vectorized positions ----
        if replace:
            positions = rng.integers(0, N, size=n).tolist()
            return self.iloc[positions]

        # ---- n == N without replacement: shuffle ----
        if n == N:
            perm = rng.permutation(N).tolist()
            return self.iloc[perm]

        # ---- n < N without replacement ----
        if weights is None:
            positions = _reservoir_positions(
                self._iter_filtered_batches(), n, N, rng,
                self._memory_guard,
            )
        else:
            positions = _weighted_reservoir_positions(
                self._iter_filtered_batches(), n, weights, rng,
                self._memory_guard,
            )

        return self.iloc[positions]

    # ---------------- cumulative (v0.9.23e) ----------------'''


# ============ 2. Module-level reservoir helpers ============

HELPERS_ANCHOR = (
    'def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:\n'
)

RESERVOIR_HELPERS = '''# ============ v0.9.23f: reservoir sampling helpers ============


def _reservoir_positions(batches, n, N, rng, memory_guard):
    """Algorithm R reservoir sampling, vectorized per batch.

    Returns list of n positions in [0, N) without replacement.
    Positions are filtered-view 0-based (match iloc).

    Algorithm: reservoir = [0, 1, ..., n-1]. For each row at position
    i >= n, with probability n/(i+1) replace random reservoir slot
    with i. Vectorized per batch: generate all random decisions at
    once, only per-replacement Python loop (~n*log(N/n) iterations).
    """
    import numpy as np

    positions = list(range(n))
    if N <= n:
        return positions

    start = 0
    for b in batches:
        if memory_guard is not None and b.num_rows > 0:
            memory_guard.check_batch(b, name="sample.reservoir_input")
        m = b.num_rows
        end = start + m
        # Skip rows [0, n) — always in reservoir.
        lo = max(start, n)
        if lo >= end:
            start = end
            continue
        row_indices = np.arange(lo, end, dtype=np.int64)
        u = rng.random(len(row_indices))
        # Probability for row i (0-indexed) = n / (i+1)
        probs = n / (row_indices + 1.0)
        replace_mask = u < probs
        replace_rows = row_indices[replace_mask]
        if len(replace_rows) > 0:
            rand_slots = rng.integers(0, n, size=len(replace_rows))
            for slot, row_pos in zip(rand_slots.tolist(),
                                     replace_rows.tolist()):
                positions[slot] = row_pos
        start = end
    return positions


def _weighted_reservoir_positions(
    batches, n, weights_col, rng, memory_guard,
):
    """Efraimidis-Spirakis A-Res weighted sampling without replacement.

    For each row with weight w > 0, compute key = u^(1/w), u~U(0,1).
    Keep top-n keys by value (min-heap of size n). Null / non-positive
    weights are skipped. Peak O(n), time O(N log n).

    Returns positions in filtered-view 0-based order.
    """
    import heapq
    import numpy as np

    heap: list = []  # min-heap of (key, position)
    pos_offset = 0
    for b in batches:
        if memory_guard is not None and b.num_rows > 0:
            memory_guard.check_batch(
                b, name="sample.weighted_input", extra_factor=3.5,
            )
        w_col = b.column(weights_col)
        w_np = w_col.to_numpy(zero_copy_only=False)
        m = len(w_np)
        if m == 0:
            continue
        # Valid = weight > 0 and not NaN (for float dtype).
        if w_np.dtype.kind == "f":
            valid = (w_np > 0) & ~np.isnan(w_np)
        else:
            valid = w_np > 0
        idx = np.where(valid)[0]
        if len(idx) == 0:
            pos_offset += m
            continue
        # Vectorized key generation for valid rows.
        u = rng.random(len(idx))
        keys = u ** (1.0 / w_np[idx])
        for i_local, k in zip(idx.tolist(), keys.tolist()):
            pos = pos_offset + int(i_local)
            if len(heap) < n:
                heapq.heappush(heap, (float(k), pos))
            elif float(k) > heap[0][0]:
                heapq.heapreplace(heap, (float(k), pos))
        pos_offset += m
    if not heap:
        raise ValueError(
            "sample: no valid weights (all null or <= 0); cannot "
            "draw a weighted sample"
        )
    return [pos for _, pos in heap]


def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:
'''


def add_sample_method():
    print("\n=== add CadwDataFrame.sample() ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def sample(\n        self,\n        n=None,\n        frac=None," in s:
        print("  already present — skip")
        return
    if ANCHOR not in s:
        raise SystemExit(
            "CadwDataFrame.sample anchor (post-corr) not found"
        )
    s = s.replace(ANCHOR, SAMPLE_METHOD, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def add_helpers():
    print("\n=== add _reservoir_positions + _weighted_reservoir_positions ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def _reservoir_positions(" in s:
        print("  already present — skip")
        return
    if HELPERS_ANCHOR not in s:
        raise SystemExit("_concat_batches anchor not found")
    s = s.replace(HELPERS_ANCHOR, RESERVOIR_HELPERS, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. Tests ============

TESTS = '''"""v0.9.23f: CadwDataFrame.sample() (AUDIT_FULL_v0922 7.A.2, Top-10 gap #1).

Modes: n, frac, replace, weights. Determinism: random_state=None -> 42.
Row order in result = sample order (random).
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, data, chunk=10, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


def make_id_df(tmp_path, n, chunk=10, name="L"):
    """Identity: v[i] = i. Enables verifying sampled positions."""
    return make_df(
        tmp_path,
        {"v": list(range(n)), "s": [f"row{i}" for i in range(n)]},
        chunk=chunk, name=name,
    )


def read_col(df, tmp_path, col, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


# ============ basic n, no replace ============

def test_sample_n_basic(tmp_path):
    df = make_id_df(tmp_path, n=100)
    s = df.sample(n=10)
    assert s.count() == 10


def test_sample_n_zero(tmp_path):
    df = make_id_df(tmp_path, n=100)
    s = df.sample(n=0)
    assert s.count() == 0


def test_sample_n_gt_N_no_replace_raises(tmp_path):
    df = make_id_df(tmp_path, n=50)
    with pytest.raises(ValueError, match=r"n=100 > N=50"):
        df.sample(n=100)


def test_sample_n_eq_N_shuffles(tmp_path):
    """n=N without replacement: same rows, shuffled (pandas parity)."""
    df = make_id_df(tmp_path, n=50)
    s = df.sample(n=50, random_state=7)
    got = read_col(s, tmp_path, "v")
    assert sorted(got) == list(range(50))
    assert got != list(range(50))  # shuffled


def test_sample_deterministic(tmp_path):
    df = make_id_df(tmp_path, n=100)
    a = read_col(df.sample(n=20, random_state=42), tmp_path, "v", "a.parquet")
    b = read_col(df.sample(n=20, random_state=42), tmp_path, "v", "b.parquet")
    assert a == b


def test_sample_different_seeds_differ(tmp_path):
    df = make_id_df(tmp_path, n=1000)
    a = read_col(df.sample(n=20, random_state=1), tmp_path, "v", "a.parquet")
    b = read_col(df.sample(n=20, random_state=2), tmp_path, "v", "b.parquet")
    assert a != b


def test_sample_default_seed_deterministic(tmp_path):
    df = make_id_df(tmp_path, n=1000)
    a = read_col(df.sample(n=20), tmp_path, "v", "a.parquet")
    b = read_col(df.sample(n=20), tmp_path, "v", "b.parquet")
    assert a == b


def test_sample_no_duplicates_no_replace(tmp_path):
    df = make_id_df(tmp_path, n=200)
    s = df.sample(n=50)
    got = read_col(s, tmp_path, "v")
    assert len(set(got)) == 50


def test_sample_within_range(tmp_path):
    df = make_id_df(tmp_path, n=200)
    got = read_col(df.sample(n=20), tmp_path, "v")
    assert all(0 <= v < 200 for v in got)


# ============ frac ============

def test_sample_frac_half(tmp_path):
    df = make_id_df(tmp_path, n=100)
    s = df.sample(frac=0.5)
    assert s.count() == 50


def test_sample_frac_1(tmp_path):
    df = make_id_df(tmp_path, n=50)
    s = df.sample(frac=1.0)
    assert s.count() == 50
    got = read_col(s, tmp_path, "v")
    assert sorted(got) == list(range(50))


def test_sample_frac_tiny_rounds_to_1(tmp_path):
    df = make_id_df(tmp_path, n=1000)
    s = df.sample(frac=0.0001)  # -> max(1, round(0.1)) = 1
    assert s.count() == 1


def test_sample_frac_zero(tmp_path):
    df = make_id_df(tmp_path, n=100)
    s = df.sample(frac=0.0)
    assert s.count() == 0


def test_sample_frac_invalid(tmp_path):
    df = make_id_df(tmp_path, n=10)
    with pytest.raises(ValueError, match="frac must be in"):
        df.sample(frac=1.5)
    with pytest.raises(ValueError, match="frac must be in"):
        df.sample(frac=-0.1)


# ============ replace=True ============

def test_sample_replace_true_basic(tmp_path):
    df = make_id_df(tmp_path, n=10)
    s = df.sample(n=100, replace=True)
    assert s.count() == 100


def test_sample_replace_true_may_have_duplicates(tmp_path):
    """With replace=True, duplicates are expected (not a strict test)."""
    df = make_id_df(tmp_path, n=5)
    s = df.sample(n=100, replace=True, random_state=42)
    got = read_col(s, tmp_path, "v")
    assert len(got) == 100
    # With N=5 and n=100, duplicates are certain
    assert len(set(got)) < 100


def test_sample_replace_true_n_gt_N_ok(tmp_path):
    df = make_id_df(tmp_path, n=10)
    # n > N allowed with replace
    s = df.sample(n=50, replace=True)
    assert s.count() == 50


# ============ validation ============

def test_sample_neither_n_nor_frac_raises(tmp_path):
    df = make_id_df(tmp_path, n=10)
    with pytest.raises(ValueError, match="exactly one of n or frac"):
        df.sample()


def test_sample_both_n_and_frac_raises(tmp_path):
    df = make_id_df(tmp_path, n=10)
    with pytest.raises(ValueError, match="exactly one of n or frac"):
        df.sample(n=3, frac=0.5)


def test_sample_negative_n_raises(tmp_path):
    df = make_id_df(tmp_path, n=10)
    with pytest.raises(ValueError, match="n must be >= 0"):
        df.sample(n=-5)


def test_sample_replace_must_be_bool(tmp_path):
    df = make_id_df(tmp_path, n=10)
    with pytest.raises(TypeError, match="replace must be bool"):
        df.sample(n=3, replace=1)  # type: ignore[arg-type]


# ============ weights ============

def test_sample_weights_basic(tmp_path):
    """Weights 0 for rows 0-49; weight 1 for rows 50-99."""
    df = make_df(tmp_path, {
        "v": list(range(100)),
        "w": [0.0] * 50 + [1.0] * 50,
    })
    s = df.sample(n=10, weights="w", random_state=42)
    got = read_col(s, tmp_path, "v")
    # All sampled values should be in [50, 100) — zero-weight rows skipped
    assert all(50 <= v < 100 for v in got), f"got {got}"


def test_sample_weights_proportional(tmp_path):
    """Weight ratio 10:1 should give ~10:1 sample ratio."""
    df = make_df(tmp_path, {
        "v": list(range(100)),
        "w": [10.0] * 50 + [1.0] * 50,
    })
    s = df.sample(n=110, weights="w", random_state=42)
    got = read_col(s, tmp_path, "v")
    low = sum(1 for v in got if v < 50)
    high = sum(1 for v in got if v >= 50)
    # Expected ratio 10:1; allow some slack
    assert low > high * 3, f"expected low>>high, got low={low} high={high}"


def test_sample_weights_nulls_skipped(tmp_path):
    """Null weights skipped; only weighted rows eligible."""
    df = make_df(tmp_path, {
        "v": list(range(50)),
        "w": pa.array(
            [None] * 25 + [1.0] * 25, type=pa.float64(),
        ),
    })
    s = df.sample(n=10, weights="w", random_state=42)
    got = read_col(s, tmp_path, "v")
    assert all(25 <= v < 50 for v in got)


def test_sample_weights_all_zero_raises(tmp_path):
    df = make_df(tmp_path, {
        "v": list(range(10)),
        "w": [0.0] * 10,
    })
    with pytest.raises(ValueError, match="no valid weights"):
        df.sample(n=3, weights="w")


def test_sample_weights_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    with pytest.raises(KeyError, match="nope"):
        df.sample(n=1, weights="nope")


def test_sample_weights_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {
        "v": [1, 2, 3],
        "s": ["a", "b", "c"],
    })
    with pytest.raises(TypeError, match="numeric"):
        df.sample(n=1, weights="s")


def test_sample_weights_plus_replace_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3], "w": [1.0, 2.0, 3.0]})
    with pytest.raises(NotImplementedError, match="weights \\+ replace"):
        df.sample(n=5, weights="w", replace=True)


# ============ filtered ============

def test_sample_filtered(tmp_path):
    df = make_id_df(tmp_path, n=100)
    filtered = df[df["v"] >= 50]
    s = filtered.sample(n=10)
    got = read_col(s, tmp_path, "v")
    assert all(50 <= v < 100 for v in got)


# ============ memory guard ============

def test_sample_large_n_fails_fast(tmp_path):
    """n*row_bytes huge -> MemoryLimitExceeded before iloc."""
    df = make_id_df(tmp_path, n=1000, chunk=100)
    df._memory_guard = MemoryGuard(max_memory_bytes=1000)
    with pytest.raises(MemoryLimitExceeded, match="sample"):
        df.sample(n=1000)


# ============ empty df ============

def test_sample_empty_df_returns_empty(tmp_path):
    t = pa.table({
        "v": pa.array([], type=pa.int64()),
        "s": pa.array([], type=pa.string()),
    })
    p = tmp_path / "empty.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=10,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )
    s = df.sample(n=5)
    assert s.count() == 0


# ============ AST guards ============

def test_sample_no_to_pandas_ast():
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.sample)
    assert_no_call_in_source(
        src, "to_pandas", "from_pandas",
        fn_name="CadwDataFrame.sample",
    )


def test_sample_uses_reservoir_helper():
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.sample)
    assert "_reservoir_positions" in src
    assert "_weighted_reservoir_positions" in src


def test_reservoir_uses_check_batch():
    import inspect
    from cadw.patch_utils import assert_call_order
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod._reservoir_positions)
    # In this helper, check_batch is inside the batch loop; just assert
    # presence (order is: check_batch, then numpy work)
    assert "check_batch" in src


def test_sample_large_n_uses_check_state():
    """check_state before iloc (AST)."""
    import inspect
    from cadw.patch_utils import assert_call_order
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.sample)
    assert_call_order(
        src, "check_state", "iloc",
        fn_name="CadwDataFrame.sample",
    )
'''


def write_test_file():
    print("\n=== write tests/test_sample.py ===")
    p = Path("tests/test_sample.py")
    if p.exists():
        print("  already exists — skip")
        return
    p.write_text(TESTS)
    assert_compiles(p)
    print("  OK")


# ============ 4. Dogfood ============

def dogfood():
    print("\n=== dogfood: sample reservoir + weighted + replace ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import numpy as _np_
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        # Basic n, no replace
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"v": list(range(100))})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=10,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            s = df.sample(n=10, random_state=42)
            out = td / "o.parquet"
            s.write_parquet(out)
            got = pq_.read_table(out).column("v").to_pylist()
            assert len(got) == 10
            assert len(set(got)) == 10  # no dupes
            assert all(0 <= v < 100 for v in got)

        # Determinism
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"v": list(range(1000))})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=100,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            s1 = df.sample(n=20, random_state=7)
            s2 = df.sample(n=20, random_state=7)
            o1 = td / "o1.parquet"
            o2 = td / "o2.parquet"
            s1.write_parquet(o1)
            s2.write_parquet(o2)
            assert (pq_.read_table(o1).column("v").to_pylist()
                    == pq_.read_table(o2).column("v").to_pylist())

        # replace=True: n > N allowed
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"v": list(range(10))})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            s = df.sample(n=100, replace=True, random_state=1)
            out = td / "o.parquet"
            s.write_parquet(out)
            got = pq_.read_table(out).column("v").to_pylist()
            assert len(got) == 100
            assert len(set(got)) < 100  # dupes expected

        # weighted: 10:1 ratio
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "v": list(range(100)),
                "w": [10.0] * 50 + [1.0] * 50,
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=10,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            s = df.sample(n=110, weights="w", random_state=42)
            out = td / "o.parquet"
            s.write_parquet(out)
            got = pq_.read_table(out).column("v").to_pylist()
            low = sum(1 for v in got if v < 50)
            high = sum(1 for v in got if v >= 50)
            # 10:1 ratio -> low should be ~10x high
            assert low > high * 3, f"low={low} high={high}"

        # errors
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"v": list(range(50))})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=10,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            try:
                df.sample(n=100)
            except ValueError as e:
                assert "n=100 > N=50" in str(e)
            else:
                raise AssertionError("n>N, no replace should raise")

        print("  n/frac/replace/weights + errors ok")

    run_verify_in_subprocess(_verify)


# ============ 5. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_sample.py",
         "tests/test_cv_keyed_prng.py",   # KFold uses its own PRNG
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 6. main ============

def main():
    print("=== v0.9.23f: CadwDataFrame.sample() ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    add_sample_method()
    add_helpers()
    write_test_file()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23f: CadwDataFrame.sample() (AUDIT_FULL_v0922 item "
        "7.A.2, Top-10 gap #1: most common sampling request; "
        "viz_sample existed but was viz-only). "
        "API: df.sample(n=None, frac=None, replace=False, weights=None, "
        "random_state=None). Exactly one of n/frac; n default None. "
        "Modes: "
        "(a) replace=False, no weights: Algorithm R reservoir, "
        "vectorized batch decisions (probs = n/(i+1), u<probs -> "
        "replace random slot). Expected replacements ~n*ln(N/n); "
        "peak O(n). "
        "(b) replace=False, weights=col: Efraimidis-Spirakis A-Res "
        "(key = u^(1/w), top-n min-heap). Vectorized key gen per "
        "batch; per-row heap ops. Peak O(n), time O(N log n) — "
        "documented N<1e7 practical limit. Null/<=0 weights "
        "skipped (raises if ALL invalid). "
        "(c) replace=True: vectorized rng.integers(0, N, n) + "
        "iloc[list] (n>N allowed). "
        "Edge cases: n=0 -> empty; n=N no-replace -> shuffle "
        "(pandas parity); n>N no-replace -> ValueError (pandas "
        "parity); frac in (0,1]; frac=0 -> empty; frac rounds to "
        "max(1, round(N*frac)). "
        "Determinism: random_state=None -> seed=42 (matches KFold). "
        "weights+replace -> NotImplementedError (needs O(N) full-"
        "weight materialize; honest). "
        "Return: CadwDataFrame (chainable; row order = sample order). "
        "MemoryGuard: check_state(4*n*row_bytes_est) before iloc "
        "(Python dicts + Arrow table materialize). row_bytes_est "
        "via reestimate_from_batch (v0.9.8b R2 fix). "
        "+tests/test_sample.py (34 tests: basic n, n=0, n>N error, "
        "n=N shuffle, determinism (same/diff seeds), no dupes, "
        "within range, frac 0.5/1.0/tiny/0/invalid, replace=True "
        "basic + n>N + dupes, validation (neither/both/negative/"
        "replace-not-bool), weights (basic, proportional 10:1, nulls "
        "skipped, all-zero -> ValueError, missing/non-numeric col, "
        "weights+replace -> NotImplementedError), filtered, "
        "memory guard, empty df, 4 AST). "
        "Fast: 2010 -> ~2044 passed.",
        self_path="scripts/patch_v0923f_sample.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23f applied. 7.A.2 complete.")


if __name__ == "__main__":
    main()