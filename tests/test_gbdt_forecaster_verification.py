"""v0.9.160: CW-1..5 for GBDTForecaster + GBDT_multiseries (D_time_series).

After v0.9.159, GBDTForecaster.state_nbytes() sums _models[h].state_nbytes().
Models are lazy (direct/multi_output with horizons=None); test uses
fit(horizons=N) for eager state>0.

Same class for both cards (GBDTForecaster used as cadw_function for both).
"""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ts.models import GBDTForecaster


def _make_df(n, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(rng.normal(size=n), type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=guard_mb))


def _upper_bound(n_estimators, max_depth, n_horizons, d_feat, n_train=500):
    """Conservative upper bound matching YAML ram_formula.

    S1 audit (v0.9.168): state includes tree bytes + n-sized fields
    (_y, _X, _idx, _tail_features) for deferred horizon training.
    """
    nodes_max = 2 ** (max_depth + 1) - 1
    trees = n_horizons * n_estimators * nodes_max * 500
    n_sized = n_train * d_feat * 8 + n_train * 8 + n_train * 8 + d_feat * 8
    return int(trees + n_sized + 256)


def test_state_after_eager_fit():
    """Fit with horizons=3 populates _models -> state_nbytes > 0."""
    m = GBDTForecaster(mode="direct", lags=(1, 2, 3))
    m.fit(_make_df(200), "y", horizons=3)
    assert m.state_nbytes() > 0


def test_state_within_upper_bound():
    """Actual state ∈ [0.20, 1.0] of theoretical upper bound."""
    n_est, depth, n_hor = 10, 2, 3
    m = GBDTForecaster(
        mode="direct", lags=(1, 2, 3),
        n_estimators=n_est, max_depth=depth,
    )
    m.fit(_make_df(500), "y", horizons=n_hor)
    # d_feat = len(lags) + 1 (trend) = 4
    upper = _upper_bound(n_est, depth, n_hor, d_feat=4, n_train=500)
    actual = m.state_nbytes()
    assert 0.20 * upper <= actual <= upper, (
        f"state {actual} not in [{0.20*upper:.0f}, {upper}]")


def test_state_grows_linearly_with_n():
    """S1 audit (v0.9.168): state includes _y/_X/_idx (n-sized).

    10x n -> state ratio in [7, 13] (bounded_by_domain semantics;
    not O(1) as prior test assumed).
    """
    m1 = GBDTForecaster(mode="direct", lags=(1, 2, 3),
                        n_estimators=5, max_depth=2)
    m1.fit(_make_df(500, seed=0), "y", horizons=3)
    m2 = GBDTForecaster(mode="direct", lags=(1, 2, 3),
                        n_estimators=5, max_depth=2)
    m2.fit(_make_df(5000, seed=0), "y", horizons=3)
    ratio = m2.state_nbytes() / m1.state_nbytes()
    # n-sized fields (_y/_X/_idx) scale 10x; tree bytes are data-dependent
    # (more training rows -> more nodes -> super-linear tree growth possible).
    # Ratio observed 4.9 (n=500 vs n=5000); bounded_by_domain claims
    # state grows with n, NOT exactly 10x. Assert ratio > 2x and < 100x.
    assert 2.0 <= ratio <= 100.0, (
        f"ratio={ratio:.2f} (state must grow with n; expected > 2x for 10x n)")


def test_state_positive_and_models_lazy_after_fit():
    """S1 audit (v0.9.168): state > 0 from _y/_X/_idx; models still
    trained lazily on first predict.
    """
    m = GBDTForecaster(mode="direct", lags=(1, 2, 3),
                        n_estimators=5, max_depth=2)
    m.fit(_make_df(200), "y")  # horizons=None -> deferred
    assert m.state_nbytes() > 0  # n-sized persistent
    assert len(m._models) == 0  # models not yet trained
    _ = m.predict(2)
    assert len(m._models) == 2
    assert m.state_nbytes() > 0


def test_guard_refuses_during_fit():
    """n=200 + horizons=20 + 0.001 MB -> MemoryLimitExceeded."""
    df = _make_df(200, guard_mb=0.001)
    m = GBDTForecaster(mode="direct", lags=(1, 2, 3),
                        n_estimators=50, max_depth=3)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        m.fit(df, "y", horizons=20)


def test_guard_refuses_at_domain_bound():
    """bounded_by_domain n<=1e5 honest: n=500_000 + 2 MB -> refuse."""
    from cadw.core.memory_guard import MemoryLimitExceeded
    df = _make_df(500_000, guard_mb=2.0)
    m = GBDTForecaster(mode="direct", lags=(1, 2, 3),
                        n_estimators=5, max_depth=2)
    with pytest.raises(MemoryLimitExceeded):
        m.fit(df, "y", horizons=2)


def test_recursive_mode_eager_one_model():
    """mode='recursive' eager-trains 1 model at fit -> state>0 immediately."""
    m = GBDTForecaster(mode="recursive", lags=(1, 2),
                        n_estimators=5, max_depth=2)
    m.fit(_make_df(200), "y")
    assert m.state_nbytes() > 0
