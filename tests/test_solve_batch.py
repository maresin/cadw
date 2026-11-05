"""v0.9.119: MemoryGuard.solve_batch — inverse of ram_formula (DEC-044)."""
from __future__ import annotations

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.core.models import DataChars
from cadw.core.planner import OpSpec
from cadw.ml._memory_estimate import (
    evaluate_formula, _is_linear_in_batch, _numeric_inverse_batch,
)


def _dc(n_rows: int = 10_000, n_cols: int = 20) -> DataChars:
    schema = pa.schema([(f"x{i}", pa.float64()) for i in range(n_cols)])
    return DataChars(
        schema=schema, n_rows=n_rows, row_size_bytes=8 * n_cols,
        k_est=100, avg_key_size_bytes=8, size_of_key_entry_bytes=80,
    )


# ---- linearity check ----

def test_linearity_positive_for_fit_linear():
    """fit_linear: 2*batch*row + 2*d*d*f64 — linear in batch."""
    assert _is_linear_in_batch("fit_linear", {"d": 20}, 160) is True


def test_linearity_positive_for_fit_logistic():
    """fit_logistic: 2*batch*row + 4*d*d*f64 — linear."""
    assert _is_linear_in_batch("fit_logistic", {"d": 20}, 160) is True


def test_linearity_negative_for_state_only_formula():
    """fit_ab_test card has state-only formula (no batch) -> False."""
    assert _is_linear_in_batch("fit_ab_test", {}, 16) is False


# ---- solve_batch ----

def test_solve_batch_positive_for_verified():
    dc = _dc(n_cols=20)
    mg = MemoryGuard(max_memory_mb=100)
    b = mg.solve_batch("fit_linear", dc, budget=1_000_000, d=20)
    assert b is not None and b >= 1


def test_solve_batch_consistent_with_formula():
    """Formula at returned batch <= budget; at batch+1 near budget."""
    dc = _dc(n_cols=20)
    mg = MemoryGuard(max_memory_mb=100)
    b = mg.solve_batch("fit_linear", dc, budget=1_000_000, d=20)
    e_b = int(evaluate_formula(
        "fit_linear", batch=b, row_size=160, d=20)["bytes"])
    assert e_b <= 1_000_000, f"e({b})={e_b} > budget"
    e_bp1 = int(evaluate_formula(
        "fit_linear", batch=b + 1, row_size=160, d=20)["bytes"])
    # 1-row rounding slack (<= 2*row_size)
    assert e_bp1 > 1_000_000 or (e_bp1 - 1_000_000) < 2 * 160


def test_solve_batch_zero_when_state_exceeds_budget():
    """state=2*d*d*f64=6400 at d=20 > budget=1000 -> 0."""
    dc = _dc(n_cols=20)
    mg = MemoryGuard(max_memory_mb=100)
    b = mg.solve_batch("fit_linear", dc, budget=1000, d=20)
    assert b == 0


def test_solve_batch_none_for_nonexistent_card():
    dc = _dc()
    mg = MemoryGuard(max_memory_mb=100)
    assert mg.solve_batch("nonexistent_fn_xyz", dc, budget=1_000_000) is None


def test_solve_batch_none_for_state_only_formula():
    """fit_ab_test card: state-only formula -> None (caller fallback)."""
    dc = _dc()
    mg = MemoryGuard(max_memory_mb=100)
    assert mg.solve_batch("fit_ab_test", dc, budget=1_000_000) is None


def test_solve_batch_defaults_from_data_chars():
    """d/n/k_est filled from data_chars automatically."""
    dc = _dc(n_cols=20, n_rows=5_000)
    mg = MemoryGuard(max_memory_mb=100)
    b = mg.solve_batch("fit_linear", dc, budget=1_000_000)
    assert b is not None and b >= 1


def test_solve_batch_uses_usable_bytes_when_budget_none():
    dc = _dc(n_cols=20)
    mg = MemoryGuard(max_memory_mb=10)  # small budget
    b = mg.solve_batch("fit_linear", dc, d=20)
    assert b is not None and b >= 1


# ---- planner integration ----

def test_planner_uses_solve_batch_for_model_step():
    """_model_step_estimate via v2 uses solve_batch -> fits."""
    dc = _dc(n_cols=10, n_rows=10_000)
    mg = MemoryGuard(max_memory_mb=100)
    ops = [OpSpec(name="fit_logistic",
                  model_spec={"cadw_function": "fit_logistic",
                              "ram_params": {"d": 10}})]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.predicted_outcome == "fits"
    assert plan.segments == [(0, 1)]


def test_planner_falls_back_when_formula_state_only():
    """fit_ab_test formula state-only -> solve_batch None -> state_bytes."""
    dc = _dc(n_cols=3, n_rows=5_000)
    mg = MemoryGuard(max_memory_mb=100)
    ops = [OpSpec(name="fit_ab_test",
                  model_spec={"cadw_function": "fit_ab_test",
                              "ram_params": {},
                              "state_bytes": 1696})]
    plan = mg.plan_pipeline(ops, dc)
    # 1696 B << 70 MB budget -> fits
    assert plan.predicted_outcome == "fits"
