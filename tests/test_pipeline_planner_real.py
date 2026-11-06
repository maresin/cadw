"""v0.9.120: P6'-P9' real pipelines — verified models with real ram_formula.

Unlike P6-P8 (synthetic, hardcoded state_bytes), these use
model_spec={"cadw_function": ..., "ram_params": {...}} so the planner
calls _model_step_estimate -> memory_guard.solve_batch -> inverse of
the actual YAML formula.

Verifies:
  - _lookup_card resolves by cadw_function (line 37 of _memory_estimate)
  - solve_batch returns valid batch for batch-linear verified cards
  - state-only cards -> solve_batch None -> fallback to state_bytes
  - planner produces segments + spill boundaries for real chains
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.core.models import DataChars
from cadw.core.planner import OpSpec, plan_pipeline


def _dc(n_rows: int = 10_000, n_cols: int = 20, k_est: int = 100) -> DataChars:
    schema = pa.schema([(f"x{i}", pa.float64()) for i in range(n_cols)])
    return DataChars(
        schema=schema, n_rows=n_rows, row_size_bytes=8 * n_cols,
        k_est=k_est, avg_key_size_bytes=8, size_of_key_entry_bytes=80,
    )


# ---- P6': classification, real fit_logistic ----

def test_p6_real_fit_logistic_no_breakers():
    """filter -> with_column -> fit_logistic. 1 segment."""
    dc = _dc()
    mg = MemoryGuard(max_memory_mb=100)
    ops = [
        OpSpec(name="filter"),
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="fit_logistic", model_spec={
            "cadw_function": "fit_logistic",
            "ram_params": {"d": 20},
        }),
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 3)]
    assert plan.spill_boundaries == []
    assert plan.predicted_outcome == "fits"
    # model step: n_rows=1, row_size = estimated fit peak (solve_batch +
    # evaluate_formula). Must be > 4*d*d*f64 = 12800 (state lower bound).
    assert plan.steps[-1].output_chars.n_rows == 1
    assert plan.steps[-1].output_chars.row_size_bytes >= 4 * 20 * 20 * 8


# ---- P7': regression, sort + real fit_linear ----

def test_p7_real_fit_linear_after_sort():
    """with_column -> sort (breaker) -> fit_linear. 2 segments."""
    dc = _dc()
    mg = MemoryGuard(max_memory_mb=100)
    ops = [
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="sort", spill_after=True),
        OpSpec(name="fit_linear", model_spec={
            "cadw_function": "fit_linear",
            "ram_params": {"d": 20},
        }),
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 2), (2, 3)]
    assert plan.spill_boundaries == [1]
    assert plan.predicted_outcome in ("fits_with_spill", "fits")
    # model step: row_size = estimated fit peak, >= 2*d*d*f64 = 6400.
    assert plan.steps[-1].output_chars.row_size_bytes >= 2 * 20 * 20 * 8


# ---- P8': aggregation, real fit_gaussian_nb ----

def test_p8_real_fit_gaussian_nb_after_groupby():
    """groupby (breaker) -> with_column -> fit_gaussian_nb. 2 segments."""
    dc = _dc(k_est=20)
    mg = MemoryGuard(max_memory_mb=100)
    ops = [
        OpSpec(name="groupby", spill_after=True),
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="fit_gaussian_nb", model_spec={
            "cadw_function": "fit_gaussian_nb",
            "ram_params": {"d": 10, "C": 3},
        }),
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 1), (1, 3)]
    assert plan.spill_boundaries == [0]
    # model step: row_size = estimated fit peak, >= 3*C*d*f64 = 720.
    assert plan.steps[-1].output_chars.row_size_bytes >= 3 * 3 * 10 * 8


# ---- P9': ab_test, real but state-only formula ----

def test_p9_real_fit_ab_test_state_only():
    """sort (breaker) -> fit_ab_test. Formula has no batch term ->
    solve_batch returns None -> fallback to spec.state_bytes."""
    dc = _dc(n_cols=3, n_rows=5_000)
    mg = MemoryGuard(max_memory_mb=100)
    ops = [
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="sort", spill_after=True),
        OpSpec(name="fit_ab_test", model_spec={
            "cadw_function": "fit_ab_test",
            "ram_params": {},
            "state_bytes": 1696,
        }),
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 2), (2, 3)]
    assert plan.spill_boundaries == [1]
    # state-only -> fallback state_bytes
    assert plan.steps[-1].output_chars.row_size_bytes == 1696


# ---- solve_batch on real YAML cards ----

def test_solve_batch_real_cards_linear_ok():
    """fit_logistic/fit_linear/fit_gaussian_nb: batch-linear -> valid batch."""
    dc = _dc()
    mg = MemoryGuard(max_memory_mb=100)
    for fn, params, min_batch in [
        ("fit_logistic", {"d": 20}, 100),
        ("fit_linear", {"d": 20}, 100),
        ("fit_gaussian_nb", {"d": 10, "C": 3}, 100),
    ]:
        b = mg.solve_batch(fn, dc, budget=10_000_000, **params)
        assert b is not None, f"{fn}: solve_batch returned None (unexpected)"
        assert b >= min_batch, f"{fn}: batch={b} < {min_batch}"


def test_solve_batch_real_cards_state_only_none():
    """fit_ab_test: state-only formula -> solve_batch returns None."""
    dc = _dc(n_cols=3, n_rows=5_000)
    mg = MemoryGuard(max_memory_mb=100)
    b = mg.solve_batch("fit_ab_test", dc, budget=10_000_000)
    assert b is None


# ---- coverage: verified YAML cards are resolvable by cadw_function ----

def test_verified_cards_resolvable_by_cadw_function():
    """Every verified card's cadw_function resolves via _lookup_card."""
    from cadw.ml._memory_estimate import _lookup_card, _load_cards
    # _load_cards indexes by 3 keys per card; cadw_function is one.
    cards_by_fn = _load_cards()
    # Spot-check the 4 real-models used in P6'-P9'.
    for fn in ["fit_logistic", "fit_linear", "fit_gaussian_nb", "fit_ab_test"]:
        card = _lookup_card(fn)
        assert card is not None, f"_lookup_card({fn!r}) returned None"
        assert card.get("cadw_function") == fn
        assert card.get("ram_formula"), f"{fn}: no ram_formula"


# ---- coverage gap: known uncarded base algorithms (DEC-045) ----

def test_base_algorithms_coverage_closed():
    """v0.9.128: gap closed in v0.9.123.

    DEC-045 documented fit_kmeans / fit_pca as uncarded (pre-cards era).
    v0.9.123 added B_clustering.B1.KMeans, B1.MiniBatchKMeans,
    C_dim_reduction.C1.PCA. Verify _lookup_card resolves all three.
    """
    from cadw.ml._memory_estimate import _lookup_card
    assert _lookup_card("fit_kmeans") is not None, "B_clustering.B1.KMeans"
    assert _lookup_card("fit_minibatch_kmeans") is not None, "B_clustering.B1.MiniBatchKMeans"
    assert _lookup_card("fit_pca") is not None, "C_dim_reduction.C1.PCA"
    assert _lookup_card("fit_kmeans_ad") is not None, "D_time_series (aliased)"
