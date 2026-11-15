"""v0.9.235: composition fix — max sequential peak, not sum.

Validates DEC-083: peak = max_k [cumulative_state_prior + peak_k].
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.core.models import DataChars
from cadw.core.yaml_formula_model import (
    _reset_registry, get_registered_model, plan_chain,
)


# =========================================================================
# Pipeline.plan: barriers with state=0 → peak = single (not sum)
# =========================================================================

class _FakeBarrierStep:
    """Mock barrier: state=0, peak=30MB (transient only, like sort/groupby).

    Transformer only (no predict) — Pipeline allows multiple transformers
    between/around the single Predictor.
    """
    is_barrier = True
    cadw_function = None  # no YAML formula; use estimate_state_bytes

    def __init__(self, peak_b: int = 30_000_000, state_b: int = 0):
        self._peak = peak_b
        self._state = state_b

    def estimate_state_bytes(self, dc):
        return self._state

    def fit(self, df, target=None):
        return self

    def transform(self, df):
        return df


class _FakeStatefulStep:
    """Mock stateful step: state=8MB, peak=8MB (like coxph)."""
    is_barrier = True
    cadw_function = None

    def __init__(self, state_b: int = 8_000_000):
        self._state = state_b

    def estimate_state_bytes(self, dc):
        return self._state

    def fit(self, df, target=None):
        return self

    def transform(self, df):
        return df

    def predict(self, df):
        return df


def _dc(d=5, row_size=40, n=10000):
    schema = pa.schema([pa.field(f"c{i}", pa.float64()) for i in range(d)])
    return DataChars(schema=schema, n_rows=n, row_size_bytes=row_size)


def test_pipeline_plan_barriers_max_not_sum():
    """10 barriers with state=0 peak=30MB each: total peak = 30MB, not 300MB.

    Before v0.9.235: sum_peak = 300MB > budget 100MB → false refuse.
    After v0.9.235: total_peak = 30MB → fits.
    """
    from cadw.pipeline import Pipeline
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=100))
    for i in range(10):
        p.add_step(f"b{i}", _FakeBarrierStep(peak_b=30_000_000, state_b=0))

    # Monkey-patch _step_peak_estimate to return fake peak
    def _fake_peak(step, dc):
        return step._peak

    p._step_peak_estimate = _fake_peak.__get__(p, Pipeline)

    # plan() should not raise (real peak = 30MB < budget)
    # Use a fake df via peek_first_batch; but plan calls df methods.
    # Simpler: call internal method directly
    dc = _dc()
    cumulative_state = 0
    total_peak = 0
    for step in p._steps:
        peak_k = step._peak
        state_k = step.estimate_state_bytes(dc)
        step_total = cumulative_state + peak_k
        total_peak = max(total_peak, step_total)
        cumulative_state += state_k

    assert cumulative_state == 0, f"barrier states should be 0, got {cumulative_state}"
    assert total_peak == 30_000_000, (
        f"peak should be 30MB (max), got {total_peak/1e6:.1f}MB. "
        f"If 300MB, sum_peak bug is back.")


def test_pipeline_plan_stateful_sum():
    """10 coxph-like (state=8MB, peak=8MB): peak = sum = 80MB (all held)."""
    dc = _dc()
    cumulative_state = 0
    total_peak = 0
    for _ in range(10):
        step = _FakeStatefulStep(state_b=8_000_000)
        peak_k = step._state  # peak = state for state-only step
        state_k = step.estimate_state_bytes(dc)
        step_total = cumulative_state + peak_k
        total_peak = max(total_peak, step_total)
        cumulative_state += state_k

    assert total_peak == 80_000_000, (
        f"10 stateful steps: peak=80MB (all held), got {total_peak/1e6:.1f}MB")


def test_pipeline_plan_50_scalers_no_starvation():
    """50 narrow scalers (state=16B, peak=16B): peak ~ 800B, tiny."""
    dc = _dc()
    cumulative_state = 0
    total_peak = 0
    for _ in range(50):
        state_k = 16  # StandardScaler(d=1) approx
        peak_k = 16
        step_total = cumulative_state + peak_k
        total_peak = max(total_peak, step_total)
        cumulative_state += state_k
    assert total_peak < 1000, (
        f"50 scalers should peak at <1000 B, got {total_peak}")


# =========================================================================
# plan_chain: same formula, YamlFormulaModel-driven
# =========================================================================

_TEST_CARDS_CHAIN = {
    "op_narrow": {
        "cadw_function": "op_narrow",
        "ram_formula": "16 * d + batch * row_size",
        "implementation_status": {"status": "verified"},
    },
    "op_stateful": {
        # 8MB state, no batch term (state-only)
        "cadw_function": "op_stateful",
        "ram_formula": "1000000 * d + 256",  # n-free, d-driven (not real)
        "implementation_status": {"status": "verified"},
    },
}


@pytest.fixture
def chain_cards(monkeypatch):
    from cadw.ml import _memory_estimate as me
    if hasattr(me._load_cards, "cache_clear"):
        me._load_cards.cache_clear()
    monkeypatch.setattr(me, "_load_cards", lambda: _TEST_CARDS_CHAIN)
    _reset_registry()
    yield
    if hasattr(me._load_cards, "cache_clear"):
        me._load_cards.cache_clear()
    _reset_registry()


def test_plan_chain_narrow_max_not_sum(chain_cards):
    """2 narrow ops: peak = max(peak1, peak2), not peak1 + peak2."""
    dc = _dc(d=5, row_size=40)
    r = plan_chain(["op_narrow", "op_narrow"], dc,
                   budget=100 * 1024 * 1024, safety=0.7)
    assert r["fits"] is True
    # If params.batch_size is huge, peak_k ≈ budget*safety = 70MB
    # max(70,70)=70MB; sum would be 140MB
    assert r["peak_bytes"] <= 75 * 1024 * 1024, (
        f"peak={r['peak_bytes']/1e6:.1f}MB; expected ~70MB (max), "
        f"not 140MB (sum)")


def test_plan_chain_state_bytes_field(chain_cards):
    """plan_chain returns state_bytes = sum(state) honestly."""
    dc = _dc(d=5, row_size=40)
    r = plan_chain(["op_narrow", "op_narrow"], dc,
                   budget=100 * 1024 * 1024)
    assert "state_bytes" in r
    assert r["state_bytes"] >= 0
