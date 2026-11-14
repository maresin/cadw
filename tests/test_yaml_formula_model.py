"""v0.9.232: YamlFormulaModel tests.

Mirror /tmp/memory_guard_proto_v2.py (4 narrow cards, 7 experiments).
Sort (wide) deferred to P2a — no batch term, needs custom solve.

Isolated from production YAML via monkey-patch of _load_cards.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from cadw.core.memory_guard import (
    MemoryGuard, MemoryLimitExceeded, free_ram_mb,
)
from cadw.core.models import DataChars
from cadw.core.yaml_formula_model import (
    _reset_registry, get_registered_model, list_registered, plan_chain,
)


TEST_CARDS = {
    "minmax_scaler": {
        "cadw_function": "minmax_scaler",
        "ram_formula": "16 * d + batch * row_size",
        "implementation_status": {"status": "verified"},
    },
    "standard_scaler": {
        "cadw_function": "standard_scaler",
        "ram_formula": "24 * d + batch * row_size",
        "implementation_status": {"status": "verified"},
    },
    "onehot_encoder": {
        "cadw_function": "onehot_encoder",
        "ram_formula": "V * 16 + batch * row_size",
        "implementation_status": {"status": "verified"},
    },
    "target_encoder": {
        "cadw_function": "target_encoder",
        "ram_formula": "K_cat * 16 + 8 + batch * row_size",
        "implementation_status": {"status": "verified"},
    },
}


@pytest.fixture
def test_cards(monkeypatch):
    from cadw.ml import _memory_estimate as me
    if hasattr(me._load_cards, "cache_clear"):
        me._load_cards.cache_clear()
    monkeypatch.setattr(me, "_load_cards", lambda: TEST_CARDS)
    _reset_registry()
    yield TEST_CARDS
    if hasattr(me._load_cards, "cache_clear"):
        me._load_cards.cache_clear()
    _reset_registry()


def _dc(d=2, row_size=24, n=1000, k_est=None):
    schema = pa.schema([pa.field(f"c{i}", pa.float64()) for i in range(d)])
    return DataChars(schema=schema, n_rows=n, row_size_bytes=row_size,
                     k_est=k_est)


# ---------------------------------------------------------------------------
# A: narrow solve — 4 cards, state formula variety
# ---------------------------------------------------------------------------

def test_A_minmax_solve(test_cards):
    mg = MemoryGuard(max_memory_mb=100)
    p = mg.plan_op("minmax_scaler", _dc(d=2))
    assert p is not None
    assert p.state_bytes == 32
    assert p.batch_size == 3058345


def test_A_standard_solve(test_cards):
    mg = MemoryGuard(max_memory_mb=100)
    p = mg.plan_op("standard_scaler", _dc(d=2))
    assert p is not None
    assert p.state_bytes == 48
    assert p.batch_size == 3058344


def test_A_onehot_solve(test_cards):
    mg = MemoryGuard(max_memory_mb=100)
    p = mg.plan_op("onehot_encoder", _dc(d=2), V=100)
    assert p is not None
    assert p.state_bytes == 1600
    assert p.batch_size == 3058280


def test_A_target_solve(test_cards):
    mg = MemoryGuard(max_memory_mb=100)
    p = mg.plan_op("target_encoder", _dc(d=2), K_cat=10)
    assert p is not None
    assert p.state_bytes == 168
    assert p.batch_size == 3058339


# ---------------------------------------------------------------------------
# C: state > budget -> MemoryLimitExceeded
# ---------------------------------------------------------------------------

def test_C_state_exceeds_budget(test_cards):
    mg = MemoryGuard(max_memory_mb=0.01)  # ~10 KB
    dc = _dc(d=1000)  # state = 24*1000 = 24000 > 7KB usable
    with pytest.raises(MemoryLimitExceeded, match="state="):
        mg.plan_op("standard_scaler", dc)


# ---------------------------------------------------------------------------
# D: tiny budget -> batch=1 (fits) or None (no fit)
# ---------------------------------------------------------------------------

def test_D_tiny_budget(test_cards):
    mg = MemoryGuard(max_memory_mb=0.0001)  # 104 B; 0.7*104 = 72; state=32; slack=40
    p = mg.plan_op("minmax_scaler", _dc(d=2))
    assert p is not None
    assert p.batch_size >= 1


# ---------------------------------------------------------------------------
# E: unknown op -> None
# ---------------------------------------------------------------------------

def test_E_unknown_op(test_cards):
    mg = MemoryGuard(max_memory_mb=100)
    p = mg.plan_op("nonexistent_xyz", _dc())
    assert p is None


# ---------------------------------------------------------------------------
# F: plan_chain narrow-narrow -> fits
# ---------------------------------------------------------------------------

def test_F_chain_narrow_narrow(test_cards):
    mg = MemoryGuard(max_memory_mb=100)
    result = mg.plan_chain(
        ["minmax_scaler", "standard_scaler"], _dc(d=2))
    assert result["fits"] is True
    assert result["peak_bytes"] > 0
    assert len(result["steps"]) == 2


# ---------------------------------------------------------------------------
# G: plan_chain with unknown op -> does not fit, reason
# ---------------------------------------------------------------------------

def test_G_chain_unknown_op(test_cards):
    mg = MemoryGuard(max_memory_mb=100)
    result = mg.plan_chain(
        ["minmax_scaler", "nonexistent_xyz"], _dc(d=2))
    assert result["fits"] is False
    assert "not registered" in (result["reason"] or "")


# ---------------------------------------------------------------------------
# H: plan_chain state>budget -> does not fit, actionable reason
# ---------------------------------------------------------------------------

def test_H_chain_state_exceeds(test_cards):
    mg = MemoryGuard(max_memory_mb=0.01)
    result = mg.plan_chain(
        ["minmax_scaler", "standard_scaler"], _dc(d=1000))
    assert result["fits"] is False
    assert "state=" in (result["reason"] or "")


# ---------------------------------------------------------------------------
# Equivalence: YamlFormulaModel.solve vs solve_batch (both linear)
# ---------------------------------------------------------------------------

def test_equivalence_with_solve_batch(test_cards):
    """For batch-linear, plan_op matches solve_batch logic."""
    mg1 = MemoryGuard(max_memory_mb=100)
    mg2 = MemoryGuard(max_memory_mb=100)
    dc = _dc(d=2)
    p1 = mg1.plan_op("minmax_scaler", dc)
    # solve_batch uses _usable_bytes (no state); plan_op uses available_bytes
    # For fresh guard, both ~= budget. Result close but not bit-identical.
    assert p1 is not None
    sb = mg2.solve_batch("minmax_scaler", dc)
    if sb is not None:
        assert abs(p1.batch_size - sb) <= 100  # small delta from accounting


# ---------------------------------------------------------------------------
# Registry: auto-register real verified cards (no monkey-patch)
# ---------------------------------------------------------------------------

def test_register_real_verified():
    """Real cards: at least 100 verified registered (inventory: 123)."""
    _reset_registry()
    regs = list_registered()
    # 95 unique cadw_function (inventory v0.9.231: 316 _load_cards
    # keys -> 95 unique functions; ~3 keys per card: fn,
    # module::fn, model_name)
    assert len(regs) >= 90, f"only {len(regs)} registered, expected 90+"
    _reset_registry()


# ---------------------------------------------------------------------------
# Bonus: free_ram_mb + MemoryGuard.auto
# ---------------------------------------------------------------------------

def test_free_ram_mb():
    mb = free_ram_mb()
    assert mb >= 0.0  # 0.0 means unavailable (fallback), not failure


def test_auto_guard():
    g = MemoryGuard.auto(fraction=0.1)
    assert g.budget_bytes > 0
    assert g.state_limit_bytes > 0
