"""v0.9.195: SARIMA verification (DEC-033 CW-1..CW-9).

SARIMA refactor: params dispatch + _collect_last_w window truncation.
Auto mode: full y (grid AICc); manual: W_eff (bounded_by_window).
state_nbytes = _y_centered_recent_ + _yd + 256 (+ _saved_y_ legacy).
"""
import warnings

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import (
    MemoryGuard, MemoryLimitExceeded, measure_peak_traced,
)
from cadw.ts.models import SARIMAForecaster


def _series(n, seed=0, s=12):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    y = (100 + 0.5 * t
         + 5 * np.sin(2 * np.pi * t / s)
         + rng.normal(0, 0.5, n))
    return y


def _mk_df(y):
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(y, type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=500))


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# CW-1: state_nbytes exact (manual, save_y=False)
@pytest.mark.parametrize("order,n", [
    ((1, 0, 0), 100), ((1, 0, 0), 500), ((2, 1, 1), 500),
])
def test_state_nbytes_manual_exact(order, n):
    """v0.9.196: state = (p + q + d + s) * 8 + 256, independent of n.

    Refactor: _yd / _y_centered_recent_ released after fit; only
    precomputed predict state kept (p + q + d + s) floats + 256 header.
    """
    y = _series(n, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m = SARIMAForecaster(order=order, auto=False).fit(_mk_df(y), "y")
    p, d, q = order
    s = 0  # no seasonal in these orders
    expected = (p + q + d + s) * 8 + 256
    assert m.state_nbytes() == expected, (
        f"order={order} n={n}: state={m.state_nbytes()} expected={expected}")
    # Confirm O(n) state released
    assert m._yd is None
    assert m._y_centered_recent_ is None


# CW-2: state save_y adds _saved_y_ (n*f64)
def test_state_save_y_adds_saved():
    y = _series(200, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m_no = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(
            _mk_df(y), "y")
        m_sv = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(
            _mk_df(y), "y", save_y=True)
    diff = m_sv.state_nbytes() - m_no.state_nbytes()
    assert diff == int(y.size * 8), (
        f"save_y diff {diff}, expected {y.size * 8}")


# CW-3: manual mode window truncation
def test_manual_max_window_truncates():
    """manual + max_window=100 -> ApproximationInfo + W_eff=100."""
    y = _series(500, seed=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = SARIMAForecaster(order=(1, 0, 0), params="manual").fit(
            _mk_df(y), "y", max_window=100)
    assert m.approximation is not None
    assert any(a.type == "window_truncated" for a in m.approximation)
    assert m.approximation[0].effective == 100
    assert m.n_train_ == 500


# CW-4: bit-identical n<=W_eff
def test_bit_identical_windowed_vs_full():
    """n=100, W=100 -> same predictions as max_window=None."""
    y = _series(100, seed=0)
    m_w = SARIMAForecaster(order=(1, 0, 0), params="manual").fit(
        _mk_df(y), "y", max_window=100)
    m_f = SARIMAForecaster(order=(1, 0, 0), params="manual").fit(
        _mk_df(y), "y", max_window=None)
    np.testing.assert_allclose(m_w.predict(12), m_f.predict(12), atol=1e-9)


# CW-5: auto mode is bounded_state false (full y persistent)
def test_auto_mode_state_o_model():
    """v0.9.196: auto mode also O(model); state independent of n.

    (v0.9.195: auto mode kept _y_centered_recent_=n, bs=false. v0.9.196
    refactor precomputes predict state -> O(model) in both modes.)
    """
    states = []
    for n in (100, 500):
        y = _series(n, seed=0)
        m = SARIMAForecaster(params="auto").fit(_mk_df(y), "y")
        states.append(m.state_nbytes())
        assert m._y_centered_recent_ is None, "auto: O(n) state not released"
        assert m._yd is None, "auto: O(n) state not released"
    # auto grid selects best (p,q,d,s); state small (< 1 KB) and
    # independent of n
    assert all(s < 1024 for s in states), f"states={states} >= 1024"
    assert states[0] == states[1], (
        f"state not n-independent: {states} (same seed, same grid -> same best)")

def test_predict_in_sample_with_df():
    y = _series(100, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(_mk_df(y), "y")
    # df=None (default) should also work (auto mode not used, but manual
    # with W_eff=n=100 gives exact).
    preds_df = m.predict_in_sample(_mk_df(y))
    assert len(preds_df) == 100


# CW-7: d>0 in-sample all NaN
def test_predict_in_sample_d1_nan():
    y = np.cumsum(_series(100, seed=0) - 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m = SARIMAForecaster(order=(0, 1, 0), auto=False).fit(_mk_df(y), "y")
    preds = m.predict_in_sample()
    assert len(preds) == 100
    assert np.all(np.isnan(preds))


# CW-8: auto deprecated warning
def test_auto_deprecated_warning():
    y = _series(100, seed=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SARIMAForecaster(order=(1, 0, 0), auto=False).fit(_mk_df(y), "y")
    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert dep, "expected DeprecationWarning for auto=False"


# CW-9: manual rejects save_y + max_window=int
def test_manual_save_y_int_window_conflict():
    y = _series(100, seed=0)
    with pytest.raises(ValueError, match="save_y=True incompatible"):
        SARIMAForecaster(order=(1, 0, 0), params="manual").fit(
            _mk_df(y), "y", save_y=True, max_window=50)

# CW-10: predict_out_of_sample_ar1 (math correctness of _arma_forecast_short)
def test_predict_out_of_sample_ar1_manual():
    """predict() == manual AR(1) recursion on centered last y."""
    y = _series(100, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(_mk_df(y), "y")
    ar_coef = m._ar[0]
    last_yc = m._last_p_y_centered_[-1]
    expected = []
    prev = last_yc
    for _ in range(12):
        val = ar_coef * prev
        expected.append(val)
        prev = val
    expected_arr = np.array(expected) + m._y_mean_
    np.testing.assert_allclose(m.predict(12), expected_arr, atol=1e-12)


# CW-11: state_nbytes O(model) constant in n
@pytest.mark.parametrize("n", [100, 500, 1000])
def test_state_nbytes_constant_in_n(n):
    """state = (p + q + d + s) * 8 + 256; independent of n."""
    y = _series(n, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(_mk_df(y), "y")
    assert m.state_nbytes() < 512, (
        f"state_nbytes={m.state_nbytes()} >= 512 at n={n} — "
        f"O(model) expected")
    # p=1, q=0, d=0, s=0 -> 8 + 256 = 264
    assert m.state_nbytes() == 8 + 256


# CW-12: predict_in_sample no df -> TypeError
def test_predict_in_sample_no_df_raises():
    """v0.9.196: predict_in_sample() without df + no save_y -> TypeError."""
    y = _series(100, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(_mk_df(y), "y")
    with pytest.raises(TypeError, match="df is required"):
        m.predict_in_sample()


# CW-13: save_y=True enables legacy predict_in_sample()
def test_predict_in_sample_save_y_legacy():
    """save_y=True -> full y persistent -> predict_in_sample() works."""
    y = _series(100, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(
            _mk_df(y), "y", save_y=True)
    preds = m.predict_in_sample()
    assert len(preds) == 100
    assert np.isfinite(preds).sum() > 0
    # state includes _saved_y_ (n * f64)
    assert m.state_nbytes() > 100 * 8

