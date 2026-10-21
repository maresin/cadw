"""v0.9.7b: DescribeOp fused (R5: 109× vs 5-Reduce + sort)."""
from __future__ import annotations

import ast
import inspect
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.describe import DescribeOp, describe_columns
from cadw.patch_utils import (
    parse_method_source, assert_no_call_in_source,
)


def make_df(tmp_path, data, chunk=10, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


# ============ correctness (existing behaviour preserved) ============


def test_describe_basic_5_values(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    desc = df.describe()
    d = desc["x"]
    assert d["count"] == 5
    assert d["mean"] == pytest.approx(3.0)
    assert d["std"] == pytest.approx(np.sqrt(2.5))
    assert d["min"] == 1.0
    assert d["25%"] == pytest.approx(2.0)
    assert d["50%"] == pytest.approx(3.0)
    assert d["75%"] == pytest.approx(4.0)
    assert d["max"] == 5.0


def test_describe_int_input(tmp_path):
    df = make_df(tmp_path, {"x": [1, 2, 3, 4, 5]})
    desc = df.describe()
    d = desc["x"]
    assert d["count"] == 5
    assert d["min"] == 1.0
    assert d["max"] == 5.0


def test_describe_filtered(tmp_path):
    df = make_df(tmp_path, {"x": [1, 2, 3, 4, 5]})
    f = df[df["x"] > 2]
    desc = f.describe()
    d = desc["x"]
    assert d["count"] == 3
    assert d["min"] == 3.0
    assert d["max"] == 5.0


def test_describe_subset(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2], "b": [10, 20], "s": ["x", "y"]})
    desc = df.describe(columns=["a"])
    assert "a" in desc
    assert "b" not in desc
    assert "s" not in desc


def test_describe_all_numeric_default(tmp_path):
    df = make_df(tmp_path, {"i": [1, 2], "f": [1.0, 2.0], "s": ["a", "b"]})
    desc = df.describe()
    assert "i" in desc
    assert "f" in desc
    assert "s" not in desc


def test_describe_missing_column(tmp_path):
    df = make_df(tmp_path, {"x": [1]})
    with pytest.raises(KeyError, match="nope"):
        df.describe(columns=["nope"])


# ============ empty / null ============


def test_describe_empty_column_nan(tmp_path):
    df = make_df(tmp_path, {
        "x": pa.array([], type=pa.float64()),
    })
    desc = df.describe()
    d = desc["x"]
    assert d["count"] == 0
    assert np.isnan(d["mean"])
    assert np.isnan(d["std"])
    assert np.isnan(d["min"])


def test_describe_all_null_column_nan(tmp_path):
    df = make_df(tmp_path, {
        "x": pa.array([None, None, None], type=pa.float64()),
    })
    desc = df.describe()
    d = desc["x"]
    assert d["count"] == 0
    assert np.isnan(d["mean"])


def test_describe_empty_no_numeric_cols(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "b"]})
    assert df.describe() == {}


def test_describe_nulls_skipped(tmp_path):
    df = make_df(tmp_path, {
        "x": pa.array([1.0, None, 3.0, None, 5.0], type=pa.float64()),
    })
    desc = df.describe()
    d = desc["x"]
    assert d["count"] == 3
    assert d["mean"] == pytest.approx(3.0)
    assert d["min"] == 1.0
    assert d["max"] == 5.0


# ============ multi-column ============


def test_describe_multiple_columns(tmp_path):
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [100.0, 200.0, 300.0, 400.0],
    })
    desc = df.describe()
    assert desc["a"]["mean"] == pytest.approx(2.5)
    assert desc["b"]["mean"] == pytest.approx(250.0)
    assert desc["a"]["min"] == 1.0
    assert desc["b"]["max"] == 400.0


def test_describe_multibatch_chunk_1(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}, chunk=1)
    desc = df.describe()
    d = desc["x"]
    assert d["count"] == 5
    assert d["mean"] == pytest.approx(3.0)
    assert d["std"] == pytest.approx(np.sqrt(2.5))


# ============ reservoir semantics ============


def test_describe_reservoir_small_n_exact(tmp_path):
    # N <= reservoir_size → all values in reservoir → exact.
    df = make_df(tmp_path, {"x": list(range(100))})
    d = df.describe(reservoir_size=200)["x"]
    # Exact percentiles of 0..99
    assert d["25%"] == pytest.approx(24.75, abs=1e-9)
    assert d["50%"] == pytest.approx(49.5, abs=1e-9)
    assert d["75%"] == pytest.approx(74.25, abs=1e-9)


def test_describe_reservoir_large_n_approx(tmp_path):
    # N > reservoir_size → approximate; within tolerance.
    rng = np.random.default_rng(0)
    data = rng.normal(size=20000)
    df = make_df(tmp_path, {"x": data.tolist()}, chunk=2000)
    d = df.describe(reservoir_size=1000)["x"]
    # mean/std exact (Welford)
    assert d["mean"] == pytest.approx(float(data.mean()), rel=1e-9)
    assert d["std"] == pytest.approx(float(data.std(ddof=1)), rel=1e-9)
    # quantiles approximate — within 5% of true
    true_q = np.percentile(data, [25, 50, 75])
    assert abs(d["25%"] - true_q[0]) < 0.1
    assert abs(d["50%"] - true_q[1]) < 0.1
    assert abs(d["75%"] - true_q[2]) < 0.1


def test_describe_reservoir_invalid(tmp_path):
    df = make_df(tmp_path, {"x": [1.0]})
    with pytest.raises(ValueError, match="reservoir_size"):
        df.describe(reservoir_size=0)


def test_describe_tight_budget_fail_fast(tmp_path):
    # Large d + large reservoir → state > state_limit → MemoryLimitExceeded
    # BEFORE any batch read (via reduce estimate_state_nbytes).
    df = make_df(tmp_path, {
        "a": list(range(1000)),
        "b": list(range(1000)),
    }, chunk=100, mem_mb=1)
    df._memory_guard = MemoryGuard(max_memory_bytes=50_000)
    with pytest.raises(MemoryLimitExceeded):
        df.describe(reservoir_size=50_000)


# ============ numpy equivalence (exact fields) ============


def test_describe_mean_std_match_numpy(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.normal(size=500)
    df = make_df(tmp_path, {"x": data.tolist()}, chunk=50)
    d = df.describe()["x"]
    assert d["mean"] == pytest.approx(float(data.mean()), rel=1e-12)
    assert d["std"] == pytest.approx(float(data.std(ddof=1)), rel=1e-12)
    assert d["min"] == float(data.min())
    assert d["max"] == float(data.max())


# ============ two-level AST guards (§13.4/190) ============


def test_describe_facade_delegates_to_describe_columns():
    """Facade: describe calls describe_columns; no self.quantile."""
    from cadw.dataframe import CadwDataFrame
    tree = parse_method_source(CadwDataFrame.describe)
    call_names = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "describe_columns" in call_names
    src = inspect.getsource(CadwDataFrame.describe)
    assert_no_call_in_source(src, "quantile", "to_pandas",
                              fn_name="CadwDataFrame.describe")


def test_describe_impl_uses_welford_and_reservoir():
    """Impl: DescribeOp has Welford (M2) + reservoir attributes."""
    src = inspect.getsource(DescribeOp)
    assert "M2" in src
    assert "reservoir" in src
    assert "Algorithm R" in src or "randint" in src


def test_describe_impl_no_to_pandas_ast():
    from cadw.ops import describe as desc_mod
    for fn_name in ("describe_columns",):
        src = inspect.getsource(getattr(desc_mod, fn_name))
        assert_no_call_in_source(src, "to_pandas", fn_name=fn_name)
    # DescribeOp.update method
    src = inspect.getsource(DescribeOp.update)
    assert_no_call_in_source(src, "to_pandas", fn_name="DescribeOp.update")


def test_describe_impl_uses_reduce_op_protocol():
    """DescribeOp is a ReduceOp; describe_columns uses reduce()."""
    from cadw.ops.reduce import ReduceOp
    assert issubclass(DescribeOp, ReduceOp)
    src = inspect.getsource(describe_columns)
    assert "reduce(" in src


# ============ speedup smoke (slow) ============


@pytest.mark.slow
def test_describe_100k_faster_than_manual_5_reduce(tmp_path):
    """100k rows, 2 cols: fused single-pass vs manual 5-Reduce+sort.

    Benchmark reference (audit bench_r5): 109× at N=2M. This test
    asserts only that fused path is faster than manual equivalent
    (each of 5 ReduceOps called separately + df.quantile per col);
    correctness on exact fields covered above.
    """
    from cadw.ops.reduce import (
        reduce, CountOp, MeanOp, StdOp, MinMaxOp,
    )
    rng = np.random.default_rng(0)
    data = rng.normal(size=100_000)
    df = make_df(tmp_path, {
        "a": data.tolist(),
        "b": data.tolist(),
    }, chunk=5000, mem_mb=50)

    # Fused path
    t0 = time.time()
    desc = df.describe()
    t_fused = time.time() - t0
    assert "a" in desc and "b" in desc

    # Manual equivalent (5 Reduce per col + 1 quantile per col)
    t0 = time.time()
    manual = {}
    for c in ("a", "b"):
        n = reduce(df._iter_filtered_batches(), CountOp(c),
                   df._memory_guard, log=False)
        m = reduce(df._iter_filtered_batches(), MeanOp(c),
                   df._memory_guard, log=False)
        s = reduce(df._iter_filtered_batches(), StdOp(c),
                   df._memory_guard, log=False)
        mn = reduce(df._iter_filtered_batches(), MinMaxOp(c, "min"),
                    df._memory_guard, log=False)
        mx = reduce(df._iter_filtered_batches(), MinMaxOp(c, "max"),
                    df._memory_guard, log=False)
        q = df.quantile(c, [0.25, 0.5, 0.75])
        manual[c] = {"count": n, "mean": m, "std": s,
                     "min": mn, "25%": q[0], "50%": q[1],
                     "75%": q[2], "max": mx}
    t_manual = time.time() - t0

    # Exact fields match
    for c in ("a", "b"):
        assert desc[c]["count"] == manual[c]["count"]
        assert desc[c]["mean"] == pytest.approx(manual[c]["mean"], rel=1e-9)
        assert desc[c]["std"] == pytest.approx(manual[c]["std"], rel=1e-9)
        assert desc[c]["min"] == manual[c]["min"]
        assert desc[c]["max"] == manual[c]["max"]

    # Fused should be strictly faster (no 5× re-scan + sort per col)
    assert t_fused < t_manual, (
        f"fused describe ({t_fused:.3f}s) not faster than manual 5-Reduce+sort "
        f"({t_manual:.3f}s)"
    )
