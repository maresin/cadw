"""v0.9.8b: R2 — sample-based reestimate (schema underestimate fix)."""
from __future__ import annotations

import inspect

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard


# ============ API: new methods exist ============


def test_reestimate_from_batch_exists():
    g = MemoryGuard(max_memory_mb=10)
    assert hasattr(g, "reestimate_from_batch")
    assert callable(g.reestimate_from_batch)


def test_estimate_row_size_from_schema_with_sample_exists():
    g = MemoryGuard(max_memory_mb=10)
    assert hasattr(g, "estimate_row_size_from_schema_with_sample")
    assert callable(g.estimate_row_size_from_schema_with_sample)


def test_get_batch_rows_accepts_sample_batch():
    g = MemoryGuard(max_memory_mb=10)
    # Signature has sample_batch kwarg
    sig = inspect.signature(g.get_batch_rows)
    assert "sample_batch" in sig.parameters
    assert sig.parameters["sample_batch"].kind is inspect.Parameter.KEYWORD_ONLY


# ============ reestimate_from_batch: correctness ============


def test_reestimate_updates_state():
    g = MemoryGuard(max_memory_mb=10)
    # 100 rows × 200 B string → 20 KB nbytes → 200 B/row
    arr = pa.array(["x" * 200] * 100, type=pa.string())
    rb = pa.RecordBatch.from_arrays([arr], names=["s"])
    before = g.row_size_bytes
    new = g.reestimate_from_batch(rb)
    assert new >= 200  # at least 200 B (strings)
    assert g.row_size_bytes == new
    assert g.row_size_bytes != before


def test_reestimate_empty_batch_raises():
    g = MemoryGuard(max_memory_mb=10)
    empty = pa.RecordBatch.from_arrays(
        [pa.array([], type=pa.string())], names=["s"],
    )
    with pytest.raises(ValueError, match="empty"):
        g.reestimate_from_batch(empty)


def test_reestimate_numeric_consistent_with_schema():
    g = MemoryGuard(max_memory_mb=10)
    # 1000 rows × int64 → ~8000+ B nbytes → ~8 B/row
    arr = pa.array(list(range(1000)), type=pa.int64())
    rb = pa.RecordBatch.from_arrays([arr], names=["x"])
    new = g.reestimate_from_batch(rb)
    assert 8 <= new <= 20  # 8 B + small overhead


# ============ audit case (bench_r2 scale-down) ============


def test_audit_case_schema_underestimates_strings():
    """10 string cols × 200 B/val: schema est << exact (R2)."""
    g = MemoryGuard(max_memory_mb=5)
    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])
    schema_est = g.estimate_row_size_from_schema(schema)
    # Schema est: 10 × 50 = 500 B/row
    assert schema_est == 500

    # Real batch: 10 × 200 B strings
    arrs = [pa.array(["x" * 200] * 100, type=pa.string()) for _ in range(10)]
    real_batch = pa.RecordBatch.from_arrays(arrs, names=schema.names)
    exact_est = g.estimate_row_size_from_batch(real_batch)
    # Exact: ~2000 B/row (10 cols × 200 B + offset overhead)
    assert exact_est > schema_est * 2, (
        f"schema={schema_est}, exact={exact_est}; R2 regression"
    )


def test_audit_case_get_batch_rows_with_sample():
    """get_batch_rows(sample_batch=...) uses exact; result ≤ budget.

    v0.9.8b-fix2: explicit schema baseline (g.estimate_row_size_from_schema)
    to establish R2-bug 500 B/row state. Without it, default _row_size_bytes
    = 80 → rows_bad=22937 (accidental). With it → rows_bad=3670 (documented).
    """
    g = MemoryGuard(max_memory_mb=5)
    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])
    arrs = [pa.array(["x" * 200] * 100, type=pa.string()) for _ in range(10)]
    sample = pa.RecordBatch.from_arrays(arrs, names=schema.names)

    # schema-only baseline: 500 B/row → 3670 rows (R2 bug demo)
    g.estimate_row_size_from_schema(schema)
    rows_bad = g.get_batch_rows(expansion_factor=2.0)

    # exact via sample: 2040 B/row → 899 rows (R2 fix)
    rows_good = g.get_batch_rows(
        expansion_factor=2.0, sample_batch=sample,
    )

    assert rows_good < rows_bad // 2, (
        f"sample-based ({rows_good}) should be < half of schema-based "
        f"({rows_bad}); ratio {rows_bad / rows_good:.2f}"
    )

    # Sanity: actual materialized batch at rows_good fits budget
    actual_batch = pa.RecordBatch.from_arrays(
        [pa.array(["x" * 200] * rows_good, type=pa.string()) for _ in range(10)],
        names=schema.names,
    )
    assert actual_batch.nbytes < 5 * 1024 * 1024, (
        f"materialized {actual_batch.nbytes} B > 5 MB budget"
    )


def test_audit_case_with_sample_helper():
    """estimate_row_size_from_schema_with_sample: exact when sample given."""
    g = MemoryGuard(max_memory_mb=5)
    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])
    arrs = [pa.array(["x" * 200] * 100, type=pa.string()) for _ in range(10)]
    sample = pa.RecordBatch.from_arrays(arrs, names=schema.names)

    est_no_sample = g.estimate_row_size_from_schema_with_sample(schema)
    est_with_sample = g.estimate_row_size_from_schema_with_sample(
        schema, sample,
    )

    assert est_no_sample == 500  # schema fallback
    assert est_with_sample > est_no_sample * 2
    # Also updates state
    assert g.row_size_bytes == est_with_sample


def test_audit_case_no_sample_uses_schema_backward_compat():
    """Without sample_batch, get_batch_rows uses schema/cached (backward compat)."""
    g = MemoryGuard(max_memory_mb=5)
    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])
    g.estimate_row_size_from_schema(schema)
    # Same call as before v0.9.8b: uses cached _row_size_bytes (500)
    rows = g.get_batch_rows(expansion_factor=2.0)
    # 5 MB × 0.7 / (500 × 2) = 3670 (old behavior preserved)
    assert rows == 3670


# ============ AST guards (§13.4/190: methods in impl) ============


def test_reestimate_uses_estimate_row_size_from_batch():
    """Impl: reestimate_from_batch delegates to estimate_row_size_from_batch."""
    src = inspect.getsource(MemoryGuard.reestimate_from_batch)
    assert "estimate_row_size_from_batch" in src


def test_get_batch_rows_handles_sample_batch():
    """Impl: get_batch_rows uses sample_batch when provided."""
    src = inspect.getsource(MemoryGuard.get_batch_rows)
    assert "sample_batch" in src
    assert "estimate_row_size_from_batch" in src


# ============ warning in docstring ============


def test_schema_docstring_has_r2_warning():
    """Docstring of estimate_row_size_from_schema warns about strings."""
    src = inspect.getsource(MemoryGuard.estimate_row_size_from_schema)
    assert "WARNING" in src or "warning" in src.lower()
    assert "string" in src.lower() or "binary" in src.lower()
    assert "reestimate" in src.lower() or "sample" in src.lower()
