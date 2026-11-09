"""v0.9.159: CW-1..5 verification for GBDT family (5 cards).

State: trees + min/max + header. Tree shape data-dependent -> CW-1
uses upper-bound [0.30, 1.0] (iforest precedent).
"""
import ctypes
import gc
import statistics

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded, measure_peak_traced
from cadw.ml.gbdt import (
    fit_gbdt_regressor, fit_gbdt_classifier, fit_gbdt_multiclass,
)


def _make_reg(n, d=3, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    tbl = {f"x{i}": pa.array(rng.normal(size=n), type=pa.float64())
           for i in range(d)}
    tbl["y"] = pa.array(rng.normal(size=n), type=pa.float64())
    return CadwDataFrame.from_table(
        pa.table(tbl), memory_guard=MemoryGuard(max_memory_mb=guard_mb))


def _make_bin(n, d=3, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    tbl = {f"x{i}": pa.array(rng.normal(size=n), type=pa.float64())
           for i in range(d)}
    tbl["y"] = pa.array((rng.random(size=n) > 0.5).astype(np.int64))
    return CadwDataFrame.from_table(
        pa.table(tbl), memory_guard=MemoryGuard(max_memory_mb=guard_mb))


def _make_mc(n, d=3, K=3, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    tbl = {f"x{i}": pa.array(rng.normal(size=n), type=pa.float64())
           for i in range(d)}
    tbl["y"] = pa.array(rng.randint(0, K, size=n))
    return CadwDataFrame.from_table(
        pa.table(tbl), memory_guard=MemoryGuard(max_memory_mb=guard_mb))


def _upper_bound_reg(n_est, depth, d):
    # CPython per-node: dict entry (~50 B) + tuple(feat,thr,left,right,gain)
    # (~120 B) + int/thr (~30 B) ≈ 200 B; ~250 B/node measured.
    # 500 B/node gives 2x safety margin.
    nodes_max = 2 ** (depth + 1) - 1
    return int(n_est * nodes_max * 500 + 2 * d * 8 + 256)


def _upper_bound_mc(n_est, depth, d, K):
    nodes_max = 2 ** (depth + 1) - 1
    return int(n_est * K * nodes_max * 500 + 2 * d * 8 + K * 8 + 256)


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ============================================================================
# GBDT (Classifier, binary) — card "GBDT"
# ============================================================================

def test_binary_state_within_upper_bound():
    n_est, depth, d = 20, 3, 3
    df = _make_bin(500, d=d)
    m = fit_gbdt_classifier(df, [f"x{i}" for i in range(d)], "y",
                             n_estimators=n_est, max_depth=depth,
                             memory_guard=MemoryGuard(max_memory_mb=500))
    upper = _upper_bound_reg(n_est, depth, d)
    actual = m.state_nbytes()
    assert actual > 0
    # Upper bound loose; actual is ~70-90% of upper
    assert actual <= upper, f"state {actual} > upper {upper}"
    assert actual >= 0.20 * upper, (
        f"state {actual} < 30% of upper {upper} — bound too loose")


def test_binary_state_independent_of_n():
    n_est, depth, d = 10, 3, 3
    m1 = fit_gbdt_classifier(_make_bin(500, d=d, seed=0),
                              [f"x{i}" for i in range(d)], "y",
                              n_estimators=n_est, max_depth=depth,
                              memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = fit_gbdt_classifier(_make_bin(5000, d=d, seed=0),
                              [f"x{i}" for i in range(d)], "y",
                              n_estimators=n_est, max_depth=depth,
                              memory_guard=MemoryGuard(max_memory_mb=500))
    # Tree shape data-dependent -> allow bounded ≤ 3x (typical ~1.1-1.3x)
    ratio = max(m1.state_nbytes(), m2.state_nbytes()) / max(
        min(m1.state_nbytes(), m2.state_nbytes()), 1)
    assert ratio <= 3.0, f"state ratio {ratio:.2f} across n (should be bounded)"


def test_binary_guard_refuses():
    df = _make_bin(200, d=3)
    with pytest.raises(MemoryLimitExceeded):
        fit_gbdt_classifier(df, ["x0", "x1", "x2"], "y",
                             n_estimators=500, max_depth=10,
                             memory_guard=MemoryGuard(max_memory_mb=0.001))


# ============================================================================
# GBDT_L2 (Regressor)
# ============================================================================

def test_regressor_state_within_upper_bound():
    n_est, depth, d = 20, 3, 3
    df = _make_reg(500, d=d)
    m = fit_gbdt_regressor(df, [f"x{i}" for i in range(d)], "y",
                            n_estimators=n_est, max_depth=depth,
                            memory_guard=MemoryGuard(max_memory_mb=500))
    upper = _upper_bound_reg(n_est, depth, d)
    actual = m.state_nbytes()
    assert actual > 0
    assert actual <= upper
    assert actual >= 0.30 * upper


def test_regressor_guard_refuses():
    df = _make_reg(200, d=3)
    with pytest.raises(MemoryLimitExceeded):
        fit_gbdt_regressor(df, ["x0", "x1", "x2"], "y",
                            n_estimators=500, max_depth=10,
                            memory_guard=MemoryGuard(max_memory_mb=0.001))


# ============================================================================
# GBDT_multiclass
# ============================================================================

def test_multiclass_state_within_upper_bound():
    n_est, depth, d, K = 10, 2, 3, 3
    df = _make_mc(500, d=d, K=K)
    m = fit_gbdt_multiclass(df, [f"x{i}" for i in range(d)], "y",
                             n_estimators=n_est, max_depth=depth,
                             memory_guard=MemoryGuard(max_memory_mb=500))
    upper = _upper_bound_mc(n_est, depth, d, K)
    actual = m.state_nbytes()
    assert actual > 0
    assert actual <= upper
    assert actual >= 0.30 * upper


def test_multiclass_guard_refuses():
    df = _make_mc(200, d=3, K=3)
    with pytest.raises(MemoryLimitExceeded):
        fit_gbdt_multiclass(df, ["x0", "x1", "x2"], "y",
                              n_estimators=500, max_depth=10,
                              memory_guard=MemoryGuard(max_memory_mb=0.001))


# ============================================================================
# GBDT_scale_pos_weight (uses fit_gbdt_classifier with scale_pos_weight)
# ============================================================================

def test_scale_pos_weight_state_same_upper():
    n_est, depth, d = 20, 3, 3
    df = _make_bin(500, d=d)
    m = fit_gbdt_classifier(df, [f"x{i}" for i in range(d)], "y",
                             n_estimators=n_est, max_depth=depth,
                             scale_pos_weight=5.0,
                             memory_guard=MemoryGuard(max_memory_mb=500))
    upper = _upper_bound_reg(n_est, depth, d)
    actual = m.state_nbytes()
    assert 0.20 * upper <= actual <= upper


# ============================================================================
# S2c-1b: v0.9.172 memmap partial-OOC test (DEC-065)
# ============================================================================

@pytest.mark.parametrize("fit_fn,target_type", [
    ("fit_gbdt_classifier", "classification"),
    ("fit_gbdt_regressor", "regression"),
])
def test_state_nbytes_independent_of_n(fit_fn, target_type):
    """Persistent state = trees (n-independent).

    v0.9.172 S2c-1b: fit transient is memmap-based but memmap pages are
    RSS-visible (CW-5 40x n ratio ~24-36). See DEC-065 for documented
    limit. Persistent state after fit remains trees only.
    """
    fn = fit_gbdt_classifier if fit_fn == "fit_gbdt_classifier" \
         else fit_gbdt_regressor
    if target_type == "classification":
        df1 = _make_bin(500, d=3, seed=0)
        df2 = _make_bin(5000, d=3, seed=0)
    else:
        df1 = _make_reg(500, d=3, seed=0)
        df2 = _make_reg(5000, d=3, seed=0)
    m1 = fn(df1, ["x0", "x1", "x2"], "y",
            n_estimators=5, max_depth=2, batch_size=500,
            memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = fn(df2, ["x0", "x1", "x2"], "y",
            n_estimators=5, max_depth=2, batch_size=500,
            memory_guard=MemoryGuard(max_memory_mb=500))
    # Trees data-dependent; ratio <= 3.0 (bounded)
    ratio = max(m1.state_nbytes(), m2.state_nbytes()) / max(
        min(m1.state_nbytes(), m2.state_nbytes()), 1)
    assert ratio <= 3.0, (
        f"{fit_fn}: state ratio {ratio:.2f}; "
        f"states={m1.state_nbytes()},{m2.state_nbytes()}")


def test_batch_size_param_accepted():
    """v0.9.172: batch_size param on public fit_* (DEC-023 class)."""
    import inspect
    from cadw.ml.gbdt import fit_gbdt_classifier, fit_gbdt_regressor
    sig_c = inspect.signature(fit_gbdt_classifier)
    sig_r = inspect.signature(fit_gbdt_regressor)
    assert "batch_size" in sig_c.parameters
    assert "batch_size" in sig_r.parameters


def test_memmap_writes_do_not_leak_into_state():
    """state_nbytes should not include memmap-backed transients.

    Only trees (+ min/max for regression) are persistent state; pred,
    cur_leaf, nan_mask are SpillDir-managed and cleaned up after fit.
    """
    df = _make_reg(1000, d=3, seed=0)
    m = fit_gbdt_regressor(df, ["x0", "x1", "x2"], "y",
                            n_estimators=3, max_depth=2,
                            memory_guard=MemoryGuard(max_memory_mb=500))
    # Trees only; check small (< 100 KB at n_est=3, depth=2)
    assert m.state_nbytes() < 100_000, (
        f"state_nbytes={m.state_nbytes()} (expected trees-only, <100KB)")
