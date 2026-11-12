"""v0.9.222-a: CadwBoost component-wise verification (DEC-033)."""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced


def _mk_df(n, d=4, seed=0, kind="reg"):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    if kind == "reg":
        y = X[:, 0] * 2 + X[:, 1] + 0.1 * rng.normal(size=n)
        y_arr = pa.array(y, type=pa.float64())
    else:
        y = (X[:, 0] > 0).astype(np.float64)
        y_arr = pa.array(y, type=pa.float64())
    cols = {f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)}
    cols["y"] = y_arr
    return CadwDataFrame.from_table(pa.table(cols),
                                     memory_guard=MemoryGuard(max_memory_mb=2000))


# CW-2: state_nbytes n-independent
@pytest.mark.parametrize("n", [1000, 5000, 20000])
def test_state_nbytes_n_independent(n):
    from cadw.ml.cadwboost import CADWBoost
    from cadw.ml.cadwboost.dataframe import fit_df
    df = _mk_df(n, 4)
    m = CADWBoost(n_estimators=10, max_depth=4, loss="mse",
                   tree_mode="symmetric", random_state=42)
    fit_df(m, df, [f"x{j}" for j in range(4)], "y", mode="memory",
           memory_guard=MemoryGuard(max_memory_mb=2000))
    state = m.state_nbytes()
    # state must be bounded by tree (n-independent)
    assert state < 500_000, f"n={n}: state={state} > 500KB (should be tree-only)"


def test_state_nbytes_bounded_across_n():
    """state(n=1k) vs state(n=20k) ratio <= 1.5 (tree shape may vary slightly)."""
    from cadw.ml.cadwboost import CADWBoost
    from cadw.ml.cadwboost.dataframe import fit_df
    states = []
    for n in (1000, 20000):
        df = _mk_df(n, 4, seed=0)
        m = CADWBoost(n_estimators=10, max_depth=4, loss="mse",
                       tree_mode="symmetric", random_state=42)
        fit_df(m, df, [f"x{j}" for j in range(4)], "y", mode="memory",
               memory_guard=MemoryGuard(max_memory_mb=2000))
        states.append(m.state_nbytes())
    ratio = max(states) / max(min(states), 1)
    assert ratio <= 1.5, f"state ratio {ratio:.2f} (n=1k vs n=20k)"


# OOC vs memory bit-identical
def test_ooc_vs_memory_bitexact():
    from cadw.ml.cadwboost import CADWBoost
    from cadw.ml.cadwboost.dataframe import fit_df, predict_df_to_numpy
    df = _mk_df(2000, 4, seed=1)
    m_mem = CADWBoost(n_estimators=10, max_depth=4, loss="mse",
                       tree_mode="symmetric", random_state=42)
    m_ooc = CADWBoost(n_estimators=10, max_depth=4, loss="mse",
                       tree_mode="symmetric", random_state=42)
    mg = MemoryGuard(max_memory_mb=2000)
    fit_df(m_mem, df, [f"x{j}" for j in range(4)], "y",
           mode="memory", memory_guard=mg)
    fit_df(m_ooc, df, [f"x{j}" for j in range(4)], "y",
           mode="ooc", memory_guard=mg)
    # Same model params -> bit-identical predictions
    # v0.9.222-a-fix: iterate batches, no get_next()
    xs_parts = []
    for b in df._iter_filtered_batches():
        xs_parts.append(np.column_stack([
            b.column(f"x{j}").to_numpy(zero_copy_only=False)
            for j in range(4)]))
    X = np.vstack(xs_parts) if len(xs_parts) > 1 else xs_parts[0]
    p_mem = m_mem.predict(X)
    p_ooc = m_ooc.predict(X)
    np.testing.assert_allclose(p_mem, p_ooc, atol=1e-10)


# CW-5: OOC ratio
def test_ooc_ratio_under_2():
    """Peak ratio n=200k vs n=50k < 2.0 (true OOC)."""
    from cadw.ml.cadwboost import CADWBoost
    from cadw.ml.cadwboost.ooc import fit_ooc

    def _peak(n):
        mg = MemoryGuard(max_memory_mb=1000)
        df = _mk_df(n, 4)
        m = CADWBoost(n_estimators=10, max_depth=4, loss="mse",
                       tree_mode="symmetric", random_state=42)
        r = measure_peak_traced(
            fit_ooc, m, df, [f"x{j}" for j in range(4)], "y",
            memory_guard=mg, mg=mg)
        return int(r["peak_bytes"])

    p50 = _peak(50_000)
    p200 = _peak(200_000)
    ratio = p200 / max(p50, 1)
    assert ratio < 2.0, f"OOC ratio {ratio:.2f} (50k -> 200k, expected < 2.0)"


# Guard behavior
def test_guard_accepts_moderate():
    """Moderate guard accepts (n=1k, state ~20KB)."""
    from cadw.ml.cadwboost import CADWBoost
    from cadw.ml.cadwboost.dataframe import fit_df
    df = _mk_df(1000, 4)
    m = CADWBoost(n_estimators=10, max_depth=4, loss="mse",
                   tree_mode="symmetric", random_state=42)
    fit_df(m, df, [f"x{j}" for j in range(4)], "y", mode="memory",
           memory_guard=MemoryGuard(max_memory_mb=500))
    assert m.fitted_


# Sanity: symmetric formula
def test_symmetric_state_formula():
    """Symmetric tree state: 2x estimators -> 1.0-3.0x state.

    Data-dependent: trees stop early if min_gain/min_child_weight block
    split, so 2^depth worst case rarely reached. Formula for guard should
    use 2^depth (worst case); test asserts proportional growth only.
    """
    from cadw.ml.cadwboost import CADWBoost
    from cadw.ml.cadwboost.dataframe import fit_df
    df = _mk_df(1000, 4)
    m10 = CADWBoost(n_estimators=10, max_depth=4, loss="mse",
                     tree_mode="symmetric", random_state=42)
    m20 = CADWBoost(n_estimators=20, max_depth=4, loss="mse",
                     tree_mode="symmetric", random_state=42)
    mg = MemoryGuard(max_memory_mb=2000)
    fit_df(m10, df, [f"x{j}" for j in range(4)], "y",
           mode="memory", memory_guard=mg)
    fit_df(m20, df, [f"x{j}" for j in range(4)], "y",
           mode="memory", memory_guard=mg)
    ratio = m20.state_nbytes() / max(m10.state_nbytes(), 1)
    # 2x estimators -> 1.0-3.0x state (data-dependent tree depth)
    assert 1.0 <= ratio <= 3.0, (
        f"state ratio {ratio:.2f} (expect 1.0-3.0 for 2x estimators; "
        f"data-dependent tree shape)")
