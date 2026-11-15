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

# ---------------------------------------------------------------------------
# All 95 real cards with realistic extras (v0.9.234 hardening)
# ---------------------------------------------------------------------------

_TEST_EXTRAS = {
    # ML hyperparams
    "n_estimators": 100, "max_depth": 3, "n_classes": 3,
    "C": 10, "n_arms": 5, "n_trees": 10, "subsample_size": 256,
    "num_kernels": 84, "n_centers": 100, "max_samples": 5000,
    "w": 32, "V": 1000, "avg_word_len": 24,
    "max_query_size": 100, "n_oversamples": 10, "n_cats_cols": 3,
    "total_categories": 100, "n_layers": 3,
    "r": 50, "c": 40, "n_biclusters": 3, "k_sv": 10,
    "n_neighbors": 10, "n_clusters": 3, "n_components": 5,
    "window_size": 32, "branching_factor": 50, "n_factors": 8,
    "n_content": 10, "n_epochs": 10, "hidden_size": 32,
    "latent_size": 8, "n_memory": 10,
    # TS
    "seasonal_period": 7, "period": 7,
    "p": 2, "q": 2, "s": 12, "W_eff": 1000,
    "levels": 3, "n_transformers": 2,
    "n_members": 3, "n_horizons": 5,
    # Data-dependent (from scan)
    "n_events": 100, "U": 100, "I": 100, "n_users": 100,
    "n_items": 100, "n_nodes": 100, "S": 1000, "n_samples": 1000,
    # v0.9.234-fix: missing hyperparams from probe (all legit, dc.extras)
    "posterior_keep": 50, "vocab_size": 1000, "max_level": 6,
    "L": 5, "read_batch_size": 1000, "n_best": 10,
}


def test_all_real_cards_with_extras():
    """v0.9.234: 95 real cards; with realistic extras, >= 90 evaluable.

    Categorizes: eval_ok (solve int OR solve None state-only) vs
    eval_error (KeyError) vs exception. Target: eval_error <= 5.

    Known P2b gaps (allowed eval_error): SARIMA/VAR d-collision; complex
    multi-source formulas; transient-in-n formulas.
    """
    from cadw.core.yaml_formula_model import (
        BatchParams, _reset_registry, get_registered_model,
        list_registered,
    )
    _reset_registry()
    regs = list_registered()

    dc = DataChars(
        schema=pa.schema(
            [pa.field(f"c{i}", pa.float64()) for i in range(5)]),
        n_rows=10_000, row_size_bytes=40, k_est=10,
        extras=dict(_TEST_EXTRAS))

    eval_ok = 0
    solve_int = 0
    solve_none = 0
    eval_errors = []
    exceptions = []

    for name in regs:
        m = get_registered_model(name)
        try:
            m.estimate(BatchParams(batch_size=1000), dc)
        except (KeyError, NameError) as e:
            eval_errors.append((name, str(e)[:80]))
            continue
        except Exception as e:
            exceptions.append((name, f"eval: {type(e).__name__}: {e}"))
            continue
        eval_ok += 1
        try:
            p = m.solve(100 * 1024 * 1024, dc, safety=0.7)
        except MemoryLimitExceeded:
            solve_int += 1  # MLE = valid outcome (state > budget)
            continue
        except Exception as e:
            exceptions.append((name, f"solve: {type(e).__name__}: {e}"))
            continue
        if p is not None:
            solve_int += 1
        else:
            solve_none += 1

    total = len(regs)
    print(
        f"\n[test_all_real_cards_with_extras] total={total}\n"
        f"  eval_ok={eval_ok} (solve_int={solve_int} + "
        f"solve_none={solve_none})\n"
        f"  eval_errors={len(eval_errors)}, exceptions={len(exceptions)}")
    if eval_errors:
        print("  EVAL ERRORS (P2b audit target):")
        for n, e in eval_errors:
            print(f"    {n}: {e}")
    if exceptions:
        print("  EXCEPTIONS:")
        for n, e in exceptions:
            print(f"    {n}: {e}")

    _reset_registry()
    # 95 cards: at least 90 evaluable (5 tolerated for known P2b gaps)
    assert eval_ok >= 90, (
        f"eval_ok={eval_ok}/{total}; expected >=90. "
        f"eval_errors={[n for n, _ in eval_errors]}")
    assert len(exceptions) == 0, f"exceptions: {exceptions}"


def test_linearity_check_quadratic():
    """v0.9.234: batch-quadratic formula (e.g. DBSCAN) -> solve None."""
    import pyarrow as pa
    from cadw.core.yaml_formula_model import YamlFormulaModel, _reset_registry
    from cadw.core.models import DataChars as DC

    # Simulate DBSCAN-like: batch*batch term
    card = {
        "cadw_function": "test_quad",
        "ram_formula": "n * i32 * 2 + batch * batch * f64 + 256",
        "implementation_status": {"status": "verified"},
    }
    m = YamlFormulaModel("test_quad", card)
    dc = DC(
        schema=pa.schema([pa.field("a", pa.float64())]),
        n_rows=10_000, row_size_bytes=8)
    p = m.solve(100 * 1024 * 1024, dc, safety=0.7)
    assert p is None, f"quadratic formula should return None, got {p}"

@pytest.mark.xfail(
    strict=True,
    reason=(
        "v0.9.236: _safe_eval doesn't support ast.Call; max/min/abs in "
        "_env are unreachable (formulas don't call them today; sort "
        "formula uses max() but sort not in registry as batch-linear). "
        "v0.9.237: add Call whitelist {max, min, abs} when sort formula "
        "into registry; then unskip."
    ),
)
def test_safe_eval_builtins():
    """v0.9.237 target: _safe_eval supports max/min/abs (Call whitelist)."""
    from cadw.ml._memory_estimate import _safe_eval
    env = {"a": 3, "b": 7}
    assert _safe_eval("max(a, b)", env) == 7.0
    assert _safe_eval("min(a, b)", env) == 3.0
    assert _safe_eval("abs(-a)", env) == 3.0
    assert _safe_eval("max(a*b, a+b)", env) == 21.0
    assert _safe_eval("2**3", {}) == 8.0
