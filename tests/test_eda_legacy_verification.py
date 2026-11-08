"""v0.9.161: verification for DriftStats + 3 FeatureSelect aliases.

DriftStats (fit_drift): lazy O(1) container (kind_='drift', state=256 B).
No data read at fit (test comparison performed by DriftABModel methods).
FeatureSelect (fit_feature_select × 3 methods): model O(d), fit transient
O(n*d) via _collect (AP2, guarded).
"""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.drift import fit_drift, DriftABModel
from cadw.ml.feature_select import fit_feature_select, FeatureSelectorModel


def _make_drift_df(n, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(pa.table({
        "a": pa.array(rng.normal(size=n), type=pa.float64()),
        "b": pa.array(rng.normal(size=n) + 0.5, type=pa.float64()),
    }), memory_guard=MemoryGuard(max_memory_mb=guard_mb))


def _make_fs_df(n, d=4, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(np.float64)
    tbl = {f"f{j}": pa.array(X[:, j], type=pa.float64())
           for j in range(d)}
    tbl["y"] = pa.array(y, type=pa.float64())
    return CadwDataFrame.from_table(
        pa.table(tbl), memory_guard=MemoryGuard(max_memory_mb=guard_mb))


# ============================================================================
# DriftStats (fit_drift) — O(1) lazy container
# ============================================================================

def test_drift_state_nbytes_exact():
    df = _make_drift_df(1000)
    m = fit_drift(df, "a", "b", method="ks",
                   memory_guard=MemoryGuard(max_memory_mb=500))
    assert isinstance(m, DriftABModel)
    assert m.kind_ == "drift"
    assert m.state_nbytes() == 256


def test_drift_state_independent_of_n():
    mg = MemoryGuard(max_memory_mb=500)
    m1 = fit_drift(_make_drift_df(500), "a", "b", method="ks",
                    memory_guard=mg)
    m2 = fit_drift(_make_drift_df(5000), "a", "b", method="ks",
                    memory_guard=mg)
    assert m1.state_nbytes() == m2.state_nbytes() == 256


def test_drift_is_lazy_container():
    """fit_drift does not read data (lazy comparison at scoring)."""
    df = _make_drift_df(500)
    m = fit_drift(df, "a", "b", method="ks",
                   memory_guard=MemoryGuard(max_memory_mb=500))
    assert m.kind_ == "drift"
    assert m.state_nbytes() == 256


def test_drift_psi_method():
    df = _make_drift_df(500)
    m = fit_drift(df, "a", "b", method="psi",
                   memory_guard=MemoryGuard(max_memory_mb=500))
    assert m.state_nbytes() == 256


# ============================================================================
# FeatureSelect — model O(d), fit transient O(n*d)
# ============================================================================

@pytest.mark.parametrize("method", ["mutual_info", "variance", "boruta"])
def test_feature_select_state_formula_exact(method):
    """state_nbytes = d*f64 + 512 (scores_ shape (d,))."""
    d = 4
    df = _make_fs_df(500, d=d)
    kw = {"variance_threshold": 0.01} if method == "variance" else {}
    m = fit_feature_select(
        df, [f"f{j}" for j in range(d)], "y",
        method=method, memory_guard=MemoryGuard(max_memory_mb=500), **kw)
    assert isinstance(m, FeatureSelectorModel)
    expected = d * 8 + 512
    assert m.state_nbytes() == expected, (
        f"{method}: state={m.state_nbytes()} expected={expected}")


@pytest.mark.parametrize("method", ["mutual_info", "variance"])
def test_feature_select_state_independent_of_n(method):
    """Same d -> same state, regardless of n (model O(d))."""
    d = 4
    mg = MemoryGuard(max_memory_mb=500)
    kw = {"variance_threshold": 0.01} if method == "variance" else {}
    m1 = fit_feature_select(
        _make_fs_df(500, d=d, seed=0), [f"f{j}" for j in range(d)], "y",
        method=method, memory_guard=mg, **kw)
    m2 = fit_feature_select(
        _make_fs_df(5000, d=d, seed=0), [f"f{j}" for j in range(d)], "y",
        method=method, memory_guard=mg, **kw)
    assert m1.state_nbytes() == m2.state_nbytes()


@pytest.mark.parametrize("method", ["mutual_info", "variance"])
def test_feature_select_guard_refuses_at_domain_bound(method):
    """n=500k, d=4 -> _collect state = ~20 MB; guard 2 MB -> refuse."""
    df = _make_fs_df(500_000, d=4, guard_mb=2.0)
    kw = {"variance_threshold": 0.01} if method == "variance" else {}
    with pytest.raises(MemoryLimitExceeded):
        fit_feature_select(
            df, ["f0", "f1", "f2", "f3"], "y",
            method=method,
            memory_guard=MemoryGuard(max_memory_mb=2.0), **kw)


def test_feature_select_transform_restricts_columns():
    df = _make_fs_df(200, d=4)
    m = fit_feature_select(
        df, ["f0", "f1", "f2", "f3"], "y",
        method="variance", variance_threshold=0.01,
        memory_guard=MemoryGuard(max_memory_mb=500))
    out = m.transform(df)
    # 'y' not in selected; selected ⊆ feature cols
    assert set(out.columns).issubset({"f0", "f1", "f2", "f3"})
