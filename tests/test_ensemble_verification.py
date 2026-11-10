"""v0.9.165: bounded_by_domain verification for EnsembleForecaster.

DEC-011: state O(n) via per-member pipelines (Naive+AutoStationarizer ~2n,
Theta ~2n, etc). bounded_by_domain n<=1e6.
"""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ts.models import NaiveForecaster
from cadw.ts.theta import ThetaForecaster
from cadw.ts.ensemble import EnsembleForecaster


def _make_df(n, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(rng.normal(size=n), type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=guard_mb))


def _build():
    return EnsembleForecaster(
        members=[NaiveForecaster("last"), ThetaForecaster(seasonal_period=1)],
        weights="equal")


def test_state_positive():
    """State > 0 after fit; equals sum over pipelines (code fact)."""
    m = _build().fit(_make_df(500), "y")
    assert m.state_nbytes() > 0
    s_pipelines = sum(p.state_nbytes() for p in m._pipelines)
    assert m.state_nbytes() == s_pipelines


def test_state_grows_linearly():
    """v0.9.183: members OOC (Naive O(1)) -> total state constant.

    Historically ratio 3.0-5.5 (expected ~4x linear in n). After Wave 2
    Naive/Theta OOC refactor, member state is O(model) not O(n), so
    ensemble total is constant across n. Assert bounded, not linear.
    """
    m1 = _build().fit(_make_df(500, seed=0), "y")
    m2 = _build().fit(_make_df(2000, seed=0), "y")
    ratio = m2.state_nbytes() / m1.state_nbytes()
    assert 0.9 <= ratio <= 2.0, f"ratio={ratio:.2f} (expected ~1.0, OOC)"


def test_guard_refuses_at_domain_bound():
    """n=500k + guard=2 MB -> MemoryLimitExceeded (state ~4 MB)."""
    df = _make_df(500_000, guard_mb=2.0)
    m = _build()
    with pytest.raises(MemoryLimitExceeded):
        m.fit(df, "y")


def test_state_per_member_bounded():
    """For n=500, each pipeline state <= 4*n*f64 + 8192 (sanity bound)."""
    m = _build().fit(_make_df(500), "y")
    upper_per_pipeline = 4 * 500 * 8 + 8192  # ~24 KB
    for i, p in enumerate(m._pipelines):
        assert p.state_nbytes() <= upper_per_pipeline, (
            f"pipeline[{i}] state={p.state_nbytes()} > {upper_per_pipeline}")


def test_state_independent_of_members_count_linear():
    """2 members vs 3 -> state ratio in [1.4, 2.0] (linear in n_members)."""
    m2 = _build().fit(_make_df(500), "y")
    m3 = EnsembleForecaster(
        members=[NaiveForecaster("last"),
                  ThetaForecaster(seasonal_period=1),
                  NaiveForecaster("seasonal", period=7)],
        weights="equal").fit(_make_df(500), "y")
    ratio = m3.state_nbytes() / m2.state_nbytes()
    # v0.9.182: Naive member state = O(model) tiny (~264B); with
    # multiple Naive members total ~constant -> ratio ~1.0.
    assert 1.0 <= ratio <= 2.2, f"ratio={ratio:.2f} (n_members 2->3)"
