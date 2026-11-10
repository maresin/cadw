"""v0.9.179: PELT max_window="auto"|int|None + strict."""
import warnings

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.approximation import ApproximationInfo
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.core.warnings import ApproximationWarning
from cadw.ml.pelt import fit_pelt


def _make_df(n, seed=0, guard_mb=100):
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2) + 5])
    tbl = pa.table({"v": pa.array(y, type=pa.float64())})
    return CadwDataFrame.from_table(
        tbl, memory_guard=MemoryGuard(max_memory_mb=guard_mb)
    )


def test_default_auto_no_truncation():
    """n=200, budget=100MB -> W_eff=n -> no approximation."""
    df = _make_df(200)
    m = fit_pelt(df, "v", penalty=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ApproximationWarning)
        m.detect(df)
    assert m.approximation is None


def test_int_within_budget():
    """int W < n, W <= W_max -> no adjustment (but truncated to W)."""
    df = _make_df(200, guard_mb=100)  # W_max large
    m = fit_pelt(df, "v", penalty=10.0, max_window=50)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.detect(df)
    # W_eff=50 < n=200 -> window_truncated, reason=explicit
    assert m.approximation is not None
    assert m.approximation[0].type == "window_truncated"
    assert m.approximation[0].reason == "explicit"
    assert m.approximation[0].effective == 50


def test_int_above_budget_adjusted():
    """int W > W_max (tight guard) -> auto-adjust + warning."""
    df = _make_df(200, guard_mb=1)  # W_max ~8192
    m = fit_pelt(df, "v", penalty=10.0, max_window=20000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.detect(df)
    assert m.approximation is not None
    assert m.approximation[0].type == "window_adjusted"
    assert m.approximation[0].reason in (
        "budget_limit", "n_smaller_than_requested"
    )
    assert m.approximation[0].effective < 20000


def test_int_above_n_small():
    """n_s < W < W_max -> reason='n_smaller_than_requested'."""
    df = _make_df(200, guard_mb=100)  # W_max ~800k
    m = fit_pelt(df, "v", penalty=10.0, max_window=10000)  # n=200
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.detect(df)
    assert m.approximation is not None
    assert m.approximation[0].reason == "n_smaller_than_requested"


def test_none_truncates():
    """max_window=None with tight guard -> W_eff < n + warning."""
    df = _make_df(5000, guard_mb=1)  # W_max ~8192
    m = fit_pelt(df, "v", penalty=10.0, max_window=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.detect(df)
    # n=5000 < W_max=8192 -> no truncation actually
    # Force tighter: n larger than W_max
    df_tight = _make_df(20000, guard_mb=1)  # W_max ~8192, n=20000
    m2 = fit_pelt(df_tight, "v", penalty=10.0, max_window=None)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        m2.detect(df_tight)
    assert m2.approximation is not None
    assert m2.approximation[0].type == "window_truncated"


def test_strict_refuses_above_budget():
    df = _make_df(200, guard_mb=1)  # W_max ~8192
    m = fit_pelt(df, "v", penalty=10.0, max_window=20000, strict=True)
    with pytest.raises(MemoryLimitExceeded) as exc_info:
        m.detect(df)
    e = exc_info.value
    assert e.model == "fit_pelt"
    assert e.params["strict"] is True
    assert len(e.options) >= 2


def test_insufficient_w_max():
    """W_max < 2*min_size -> MemoryLimitExceeded at fit."""
    df = _make_df(100, guard_mb=0.001)  # 1 KB budget
    with pytest.raises(MemoryLimitExceeded) as exc_info:
        fit_pelt(df, "v", penalty=10.0, min_size=5000)  # 2*min_size=10000
    e = exc_info.value
    assert "W_max" in str(e)


def test_invalid_max_window():
    df = _make_df(200)
    with pytest.raises(ValueError, match="max_window must be"):
        fit_pelt(df, "v", max_window="bogus")


def test_int_below_min_size():
    df = _make_df(200)
    with pytest.raises(ValueError, match=r"must be >= 2\*min_size"):
        fit_pelt(df, "v", max_window=1, min_size=2)


def test_approximation_recorded_per_batch():
    """Multi-series: N approximations -> one summary warning."""
    n_per = 50000
    y1 = np.concatenate([np.zeros(n_per // 2), np.ones(n_per - n_per // 2) + 5])
    y2 = np.concatenate([np.zeros(n_per // 2), np.ones(n_per - n_per // 2) - 3])
    tbl = pa.table({
        "id": pa.array([0] * n_per + [1] * n_per, type=pa.int64()),
        "v": pa.array(np.concatenate([y1, y2]), type=pa.float64()),
    })
    df = CadwDataFrame.from_table(
        tbl, memory_guard=MemoryGuard(max_memory_mb=5)
    )
    m = fit_pelt(df, "v", id_col="id", penalty=10.0, max_window=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = m.detect(df)
    # Both series truncated -> 2 approximations, 1 summary warning
    assert len(m.approximation) == 2
    approx_warns = [x for x in w if issubclass(x.category, ApproximationWarning)]
    assert len(approx_warns) == 1  # batch summary
