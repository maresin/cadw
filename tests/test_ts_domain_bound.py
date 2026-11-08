"""v0.9.157: bounded_by_domain verification for single-series TS forecasters.

DEC-011: n is bounded by problem domain (single-series, n<=1e6 typical).
Guard must refuse BEFORE OOM at n=500k with small guard.

5 naive TS: Naive, ETS, AutoETS, Theta, AutoTheta.
"""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ts.models import NaiveForecaster
from cadw.ts.ets import ETSForecaster
from cadw.ts.auto_ets import AutoETSForecaster
from cadw.ts.theta import ThetaForecaster
from cadw.ts.auto_theta import AutoThetaForecaster


def _make_df(n, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    df = CadwDataFrame.from_table(
        pa.table({"y": pa.array(rng.normal(size=n), type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=guard_mb),
    )
    return df


@pytest.mark.parametrize("factory,expected_state_min_mb", [
    (lambda: NaiveForecaster("last"), 4.0),   # n*8 = 4 MB at n=500k
    (lambda: ETSForecaster(),          12.0),  # 3n*8 = 12 MB
    (lambda: AutoETSForecaster(),      12.0),
    (lambda: ThetaForecaster(),         8.0),  # 2n*8 = 8 MB
    (lambda: AutoThetaForecaster(),     8.0),
])
def test_guard_refuses_at_domain_bound(factory, expected_state_min_mb):
    """n=500_000 + guard=2 MB -> MemoryLimitExceeded (state > 2 MB)."""
    n = 500_000
    df = _make_df(n, guard_mb=2.0)  # tight guard on df
    model = factory()
    with pytest.raises(MemoryLimitExceeded):
        model.fit(df, "y")


@pytest.mark.parametrize("factory", [
    lambda: NaiveForecaster("last"),
    lambda: ETSForecaster(),
    lambda: AutoETSForecaster(),
    lambda: ThetaForecaster(),
    lambda: AutoThetaForecaster(),
])
def test_state_nbytes_grows_linearly(factory):
    """State scales ~linearly with n."""
    n1, n2 = 500, 2000  # 4x
    m1 = factory().fit(_make_df(n1, guard_mb=500), "y")
    m2 = factory().fit(_make_df(n2, guard_mb=500), "y")
    s1 = m1.state_nbytes()
    s2 = m2.state_nbytes()
    assert s1 > 0
    assert s2 > 0
    ratio = s2 / s1
    # 4x n -> ~4x state (allow 3.0-5.5 for small header residual)
    assert 3.0 <= ratio <= 5.5, f"ratio={ratio:.2f}, s1={s1}, s2={s2}"


def test_naive_state_formula_exact():
    """Naive state_nbytes == n*f64 + 256."""
    n = 1000
    m = NaiveForecaster("last").fit(_make_df(n, guard_mb=500), "y")
    assert m.state_nbytes() == n * 8 + 256


def test_ets_state_formula_exact():
    """ETS state_nbytes == 3n*f64 + 256 (seasonal config s=12).

    Non-seasonal ETS has smaller season_ (size 0-1), so 3n*f64 is
    an upper bound (safe for guard). For exact-match test, use
    seasonal_period=12 which makes season_ ~n.
    """
    n = 500
    m = ETSForecaster(seasonal_period=12).fit(
        _make_df(n, guard_mb=500), "y")
    # Formule exacte from ets.py:170-172
    # y_.nbytes + fitted_.nbytes + season_.nbytes + 256
    expected = m.y_.nbytes + m.fitted_.nbytes + m.season_.nbytes + 256
    assert m.state_nbytes() == expected
    # For seasonal config, season_ ~n -> ~3n total
    assert expected >= 2 * n * 8 + 256
    assert expected <= 3 * n * 8 + 256


def test_theta_state_formula_exact():
    """Theta state_nbytes == 2n*f64 + 256."""
    n = 500
    m = ThetaForecaster().fit(_make_df(n, guard_mb=500), "y")
    assert m.state_nbytes() == 2 * n * 8 + 256
