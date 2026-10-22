"""v0.9.23f: CadwDataFrame.sample() (AUDIT_FULL_v0922 7.A.2, Top-10 gap #1).

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
    """v0.9.23f-fix2: frac=0.0 returns empty (pandas-parity).

    Prior production validated (0, 1] and raised ValueError for 0.0 —
    inconsistent with docstring ("frac=0 -> empty") and downstream
    dead-code (`if float(frac) == 0.0: n = 0`). Pandas allows
    frac=0 -> empty. Fix: [0, 1] validation; dead code live.
    """
    df = make_id_df(tmp_path, n=100)
    s = df.sample(frac=0.0)
    assert s.count() == 0
    # Negative still rejected
    with pytest.raises(
        ValueError, match=r"frac must be in \[0, 1\]",
    ):
        df.sample(frac=-0.1)


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
    """Weight ratio 10:1 should give ~10:1 sample ratio.

    v0.9.23f-fix1: prior used n=110 > N=100 (replace=False), which
    correctly raised ValueError (pandas-parity). Fix: n=50 from N=100
    (50 low-w + 50 high-w). Expected ~45 low, ~5 high (ratio 10:1).
    """
    df = make_df(tmp_path, {
        "v": list(range(100)),
        "w": [10.0] * 50 + [1.0] * 50,
    })
    s = df.sample(n=50, weights="w", random_state=42)
    got = read_col(s, tmp_path, "v")
    low = sum(1 for v in got if v < 50)
    high = sum(1 for v in got if v >= 50)
    # Expected ~45 low, ~5 high; require low > 3*high (threshold safe)
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
    with pytest.raises(NotImplementedError, match=r"weights \+ replace"):
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
    """check_state precedes self.iloc[...] in sample() (AST).

    v0.9.23f-fix2: `self.iloc[positions]` is ast.Subscript(value=
    ast.Attribute(attr="iloc")), NOT ast.Call. assert_call_order only
    walks Call nodes, so "iloc" is never found. Fix: direct ast.walk
    on Subscript nodes with .value.attr == "iloc".
    """
    import ast as _ast
    import inspect as _inspect
    import textwrap as _tw
    from cadw.dataframe import CadwDataFrame
    src = _tw.dedent(_inspect.getsource(CadwDataFrame.sample))
    tree = _ast.parse(src)
    check_state_lines: list[int] = []
    iloc_subscript_lines: list[int] = []
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Call):
            f = n.func
            if (isinstance(f, _ast.Attribute)
                    and f.attr == "check_state"):
                check_state_lines.append(n.lineno)
        elif isinstance(n, _ast.Subscript):
            v = n.value
            if (isinstance(v, _ast.Attribute)
                    and v.attr == "iloc"):
                iloc_subscript_lines.append(n.lineno)
    assert check_state_lines, "check_state not called in sample()"
    assert iloc_subscript_lines, "self.iloc[...] not used in sample()"
    assert min(check_state_lines) < min(iloc_subscript_lines), (
        f"check_state (line {min(check_state_lines)}) must precede "
        f"self.iloc[...] subscript (line {min(iloc_subscript_lines)})"
    )
