"""v0.9.232-probe: MemoryGuard API on simple (prototype) cards.

Isolated from production models (30 call sites). Tests MemoryGuard
behavior on 5 simple cards derived from /tmp/lazy_proto.py
(MinMaxScaler, StandardScaler, OneHotEncoder, TargetEncoder, SortStep).

Purpose: discover current MemoryGuard semantics on simple, exact,
fully-controlled formulas BEFORE design unification.

Each test documents expected + observed. Failure = gap for design.
No production code changes in this release.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.core.models import DataChars


# ---------------------------------------------------------------------------
# Test cards — Python dict, injected via monkey-patch of _load_cards.
# No YAML file changes; production loaders untouched.
# ---------------------------------------------------------------------------

TEST_CARDS = {
    "minmax_scaler": {
        # narrow: state = 16*d (lo, hi per column)
        "ram_formula": "16 * d + batch * row_size",
        "bounded_state_dims": {"d": 1},
        "n_scalability": "constant",
        "implementation_status": {"status": "verified",
                                  "verified_in_cadw": True},
    },
    "standard_scaler": {
        # narrow: state = 24*d (mean, M2, n per column)
        "ram_formula": "24 * d + batch * row_size",
        "bounded_state_dims": {"d": 1},
        "n_scalability": "constant",
        "implementation_status": {"status": "verified",
                                  "verified_in_cadw": True},
    },
    "onehot_encoder": {
        # narrow: state = V * 16 (vocab entries, 8+8 per entry)
        "ram_formula": "V * 16 + batch * row_size",
        "bounded_state_dims": {"V": 1},
        "n_scalability": "constant",
        "implementation_status": {"status": "verified",
                                  "verified_in_cadw": True},
    },
    "target_encoder": {
        # narrow: state = K_cat * 16 + 8 (dict + global mean)
        "ram_formula": "K_cat * 16 + 8 + batch * row_size",
        "bounded_state_dims": {"K_cat": 1},
        "n_scalability": "constant",
        "implementation_status": {"status": "verified",
                                  "verified_in_cadw": True},
    },
    "sort_step": {
        # wide: no batch term; 2 branches (Phase1, Phase2 floor)
        "ram_formula": "4 * sb + 3 * K * rbs * (row_size + 16)",
        "bounded_state_dims": {"sb": 1, "K": 1, "rbs": 1},
        "n_scalability": "constant",
        "implementation_status": {"status": "verified",
                                  "verified_in_cadw": True},
    },
}


@pytest.fixture
def test_cards(monkeypatch):
    """Inject test cards via monkey-patch of _load_cards.

    Bypasses YAML files. Production loader untouched after teardown.
    """
    from cadw.ml import _memory_estimate as me
    orig = me._load_cards
    if hasattr(orig, "cache_clear"):
        orig.cache_clear()
    def _fake():
        return TEST_CARDS
    monkeypatch.setattr(me, "_load_cards", _fake)
    yield TEST_CARDS
    if hasattr(orig, "cache_clear"):
        orig.cache_clear()


def _dc(d=2, row_size=24, k_est=None, n_rows=1000):
    return DataChars(
        schema=pa.schema([]), n_rows=n_rows, row_size_bytes=row_size,
        k_est=k_est, avg_key_size_bytes=None,
        size_of_key_entry_bytes=None,
    )


# ---------------------------------------------------------------------------
# 1. Formula evaluation (baseline — does evaluate_formula work at all?)
# ---------------------------------------------------------------------------


def test_evaluate_minmax_simple(test_cards):
    """Formula: 16*d + batch*row_size. d=2, batch=1000, row=24 → 32 + 24000 = 24032."""
    from cadw.ml._memory_estimate import evaluate_formula
    r = evaluate_formula("minmax_scaler", d=2, batch=1000, row_size=24)
    assert r["bytes"] == 24032, f"got {r}"


def test_evaluate_sort_no_batch_term(test_cards):
    """Sort: no batch term (wide). Verify eval works with sb/K/rbs params."""
    from cadw.ml._memory_estimate import evaluate_formula
    r = evaluate_formula("sort_step", sb=16_384, K=8, rbs=1024, row_size=24)
    # 4*16384 + 3*8*1024*40 = 65536 + 983040 = 1048576
    assert r["bytes"] == 1_048_576, f"got {r}"


# ---------------------------------------------------------------------------
# 2. solve_batch inverse for pure-linear (minmax, standard) — should work
# ---------------------------------------------------------------------------


def test_solve_batch_minmax_works(test_cards):
    """Linear formula → solve_batch returns int."""
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc(d=2, row_size=24)
    b = mg.solve_batch("minmax_scaler", dc)
    assert b is not None, "linear formula: solve_batch must not be None"
    usable = mg._usable_bytes()
    expected = (usable - 32) // 24
    assert abs(b - expected) <= 2, f"got {b}, expected ~{expected}"


def test_solve_batch_standard_works(test_cards):
    """Linear formula (state=24*d) → solve_batch returns int."""
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc(d=2, row_size=24)
    b = mg.solve_batch("standard_scaler", dc)
    assert b is not None
    usable = mg._usable_bytes()
    expected = (usable - 48) // 24
    assert abs(b - expected) <= 2, f"got {b}, expected ~{expected}"


# ---------------------------------------------------------------------------
# 3. Sort (wide, no batch term) — what does solve_batch return?
# ---------------------------------------------------------------------------


def test_solve_batch_sort_non_batch(test_cards):
    """Sort has no `batch` in formula (wide, barrier).

    Design: no batch sizing applicable. Expected: None.
    Observed: ?
    """
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc(d=2, row_size=24)
    b = mg.solve_batch("sort_step", dc, budget=100_000_000,
                       sb=16_384, K=8, rbs=1024)
    assert b is None, f"expected None (no batch term), got {b}"


# ---------------------------------------------------------------------------
# 4. Data-dependent params (V, K_cat) — injectable via **extra_params?
# ---------------------------------------------------------------------------


def test_solve_batch_onehot_with_explicit_V(test_cards):
    """Pass V explicitly. Should resolve and give int batch."""
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc(d=2, row_size=24, k_est=100)
    b = mg.solve_batch("onehot_encoder", dc, V=100)
    assert b is not None, "V passed explicitly; expected int batch"
    usable = mg._usable_bytes()
    expected = (usable - 100 * 16) // 24
    assert abs(b - expected) <= 2, f"got {b}, expected ~{expected}"


def test_solve_batch_onehot_without_V(test_cards):
    """V not passed → should raise actionable error (not silent None/wrong)."""
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc(d=2, row_size=24, k_est=100)
    try:
        b = mg.solve_batch("onehot_encoder", dc)
    except (KeyError, ValueError, NameError) as e:
        # acceptable: actionable error
        return
    # If no exception: either None (design: caller must pass V) or int
    # (design: auto-inject from dc.k_est)
    assert b is None or isinstance(b, int), f"got {b!r}"


# ---------------------------------------------------------------------------
# 5. Cross-consistency: get_batch_rows(ef=1) vs solve_batch for linear
# ---------------------------------------------------------------------------


def test_get_batch_rows_ef1_matches_solve_batch_linear(test_cards):
    """For pure-linear: solve_batch == get_batch_rows(ef=1, state=state)."""
    mg1 = MemoryGuard(max_memory_mb=100)
    mg2 = MemoryGuard(max_memory_mb=100)
    dc = _dc(d=2, row_size=24)
    sb = mg1.solve_batch("minmax_scaler", dc)
    if sb is None:
        pytest.skip("solve_batch returned None for linear formula")
    gbr = mg2.get_batch_rows(row_size_bytes=24, expansion_factor=1.0,
                              state_bytes=32)
    assert abs(sb - gbr) <= 5, f"solve_batch={sb} vs gbr={gbr}"


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


def test_state_alone_exceeds_budget(test_cards):
    """StandardScaler state = 24*d. If state > budget → error."""
    mg = MemoryGuard(max_memory_mb=0.01)  # 10 KB budget → ~5 KB usable
    dc = _dc(d=1000, row_size=24)  # 24000 B state > 5 KB
    with pytest.raises((MemoryLimitExceeded, ValueError)):
        mg.solve_batch("standard_scaler", dc)


def test_budget_tiny_solve_batch(test_cards):
    """Tiny budget → None or MemoryLimitExceeded (no batch fits)."""
    mg = MemoryGuard(max_memory_mb=0.001)
    dc = _dc(d=2, row_size=24)
    try:
        b = mg.solve_batch("minmax_scaler", dc)
    except (MemoryLimitExceeded, ValueError):
        return  # raised — acceptable
    assert b is None or b == 0, f"tiny budget → got {b}"


def test_unknown_op_returns_none(test_cards):
    """Unknown op_name → KeyError or None?"""
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc()
    try:
        b = mg.solve_batch("nonexistent_op_xyz", dc)
    except (KeyError, ValueError):
        return  # raised — acceptable
    assert b is None, f"unknown op → got {b}"


# ---------------------------------------------------------------------------
# 7. API surface check (documentation)
# ---------------------------------------------------------------------------


def test_memory_guard_api_surface(test_cards):
    """Document 5 MemoryGuard API methods."""
    methods = ["get_batch_rows", "solve_batch", "plan", "plan_pipeline",
               "estimate_cost"]
    for m in methods:
        assert hasattr(MemoryGuard, m), f"missing {m}"
