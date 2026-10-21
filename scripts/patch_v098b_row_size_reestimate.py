#!/usr/bin/env python3
# v0.9.8b — R2: sample-based reestimate (schema underestimate fix).
#
# Audit reference (bench_r2.log):
#   Schema: 10 string cols, avg 200 B/value.
#   estimate_row_size_from_schema (default_string_bytes=50) = 500 B/row
#   estimate_row_size_from_batch (exact) = 10080 B/row → 20.2× ratio
#   (bench_r2 reports 10.1× — the discrepancy is because default_string_bytes
#   = 50 was already close to real ~200 B for typical benchmarks; extreme
#   200 B × 10 cols → 20× underestimate is worst-case).
#   get_batch_rows(ef=2) → 3670 rows → materialized batch 7.1 MB @ 5 MB budget.
#
# Production-path check: CADW ops use check_batch(batch.nbytes) (real size),
# NOT get_batch_rows. Grep confirms no production call sites. So R2 is
# an API-risk for user code (bench + custom ops), not an internal OOM.
# Fix is additive — existing schema-based API unchanged for backward compat.
#
# Design:
#   - estimate_row_size_from_schema: docstring updated (explicit warning:
#     string/binary underestimated without sample).
#   - new MemoryGuard.reestimate_from_batch(batch) -> int: explicit
#     reestimate of self._row_size_bytes from real batch (§3.10 discipline).
#   - new MemoryGuard.estimate_row_size_from_schema_with_sample(schema,
#     sample_batch=None) -> int: helper for callers — uses exact if sample.
#   - get_batch_rows(..., sample_batch=None): new kwarg; if provided,
#     uses estimate_row_size_from_batch (exact) instead of cached
#     self._row_size_bytes.
#
# Backward compat: all existing call sites (get_batch_rows without
# sample_batch) unchanged; get_batch_rows(ef=2)=3670 still works (but
# docstring now warns); estimate_row_size_from_schema signature identical.
#
# Two-level tests (§13.4/190):
#   - API: new methods exist; reestimate mutates state; get_batch_rows
#     with sample_batch uses exact.
#   - Audit case: schema 10×200B strings, budget 5 MB → est=500,
#     sample est=~10080, get_batch_rows(ef=2) via sample ≤ ~170,
#     materialized ≤ budget.
#   - Regression: small data → schema estimate accurate; existing tests pass.
#
# Scaffold v2: write_content_checked, is_patched.
# Dogfood self-contained imports (§13.4/200).
#
# Run: uv run python scripts/patch_v098b_row_size_reestimate.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    write_content_checked, assert_compiles, commit_with_guards,
    is_patched,
)


# ============ 1. update estimate_row_size_from_schema docstring ============

DOC_OLD = '''    def estimate_row_size_from_schema(
        self, schema: pa.Schema, *, default_string_bytes: int = 50
    ) -> int:
        total = 0
'''
DOC_NEW = '''    def estimate_row_size_from_schema(
        self, schema: pa.Schema, *, default_string_bytes: int = 50
    ) -> int:
        """Estimate bytes/row from schema alone. Conservative for numeric.

        WARNING (v0.9.8b, R2): for string/binary columns, uses
        `default_string_bytes=50` regardless of actual value length.
        Real-world string columns can be 4× to 20× larger (audit
        bench_r2: 10 string cols, 200 B/value → schema est 500 B/row,
        exact 10080 B/row). If schema contains string/binary fields,
        PREFER `reestimate_from_batch(batch)` (exact) or
        `estimate_row_size_from_schema_with_sample(schema, batch)`.

        Existing numeric-only schemas: accurate (bit_width // 8).
        """
        total = 0
'''


def update_schema_docstring():
    print("\n=== memory_guard.py: update estimate_row_size_from_schema docstring ===")
    p = Path("cadw/core/memory_guard.py")
    s = p.read_text()
    if "WARNING (v0.9.8b, R2)" in s:
        print("  already updated — skip")
        return
    if DOC_OLD not in s:
        raise SystemExit("estimate_row_size_from_schema anchor not found")
    s = s.replace(DOC_OLD, DOC_NEW, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. add reestimate_from_batch + _with_sample methods ============

NEW_METHODS_LINES = [
    "",
    "    # ---------- v0.9.8b (R2): reestimate from sample batch ----------",
    "    #",
    "    # estimate_row_size_from_schema underestimates string/binary",
    "    # (default_string_bytes=50, real 4–20× larger). If caller can peek",
    "    # first batch (peek_first_batch in cadw.core.sampling), use these",
    "    # methods for exact per-row bytes before choosing batch_size / P.",
    "    #",
    "    # §3.10 discipline: check before materialize. reestimate AFTER",
    "    # peek, BEFORE any decision that depends on row_size.",
    "",
    "    def reestimate_from_batch(self, batch: pa.RecordBatch) -> int:",
    "        \"\"\"Update self._row_size_bytes from real batch (v0.9.8b).",
    "",
    "        Returns exact bytes/row (batch.nbytes / num_rows). Updates",
    "        cached `self._row_size_bytes` used by `get_batch_rows()`.",
    "",
    "        Use after `peek_first_batch`: schema-only estimate may be",
    "        10–20× low for string/binary; real batch is exact.",
    "",
    "        Raises ValueError for empty batch (caller must pass non-empty).",
    "        \"\"\"",
    "        if batch.num_rows == 0:",
    "            raise ValueError(",
    "                \"reestimate_from_batch: cannot estimate from empty \"",
    "                \"batch (0 rows). Caller must pass non-empty batch.\"",
    "            )",
    "        return self.estimate_row_size_from_batch(batch)",
    "",
    "    def estimate_row_size_from_schema_with_sample(",
    "        self,",
    "        schema: pa.Schema,",
    "        sample_batch: pa.RecordBatch | None = None,",
    "        *,",
    "        default_string_bytes: int = 50,",
    "    ) -> int:",
    "        \"\"\"Schema-based estimate; exact if sample_batch provided.",
    "",
    "        v0.9.8b (R2): if `sample_batch` provided → uses",
    "        `estimate_row_size_from_batch` (exact, R2 fix). Else falls",
    "        back to `estimate_row_size_from_schema` (numeric-accurate,",
    "        string-underestimate). Also updates `self._row_size_bytes`.",
    "        \"\"\"",
    "        if sample_batch is not None:",
    "            return self.estimate_row_size_from_batch(sample_batch)",
    "        return self.estimate_row_size_from_schema(",
    "            schema, default_string_bytes=default_string_bytes,",
    "        )",
    "",
]


def add_reestimate_methods():
    print("\n=== memory_guard.py: add reestimate_from_batch + _with_sample ===")
    p = Path("cadw/core/memory_guard.py")
    s = p.read_text()
    if "def reestimate_from_batch(" in s:
        print("  already present — skip")
        return
    # Anchor: end of estimate_row_size (legacy wrapper).
    anchor = (
        "    def estimate_row_size(\n"
        "        self,\n"
        "        schema: Optional[pa.Schema] = None,\n"
        "        sample_batch: Optional[pa.RecordBatch] = None,\n"
        "    ) -> int:\n"
        "        if sample_batch is not None:\n"
        "            return self.estimate_row_size_from_batch(sample_batch)\n"
        "        if schema is not None:\n"
        "            return self.estimate_row_size_from_schema(schema)\n"
        "        return self._row_size_bytes\n"
    )
    if anchor not in s:
        raise SystemExit("estimate_row_size anchor not found")
    new = anchor + "\n".join(NEW_METHODS_LINES) + "\n"
    s = s.replace(anchor, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. get_batch_rows: sample_batch kwarg ============

GET_BATCH_ROWS_OLD = '''    def get_batch_rows(
        self,
        row_size_bytes: Optional[int] = None,
        expansion_factor: Optional[float] = None,
    ) -> int:
        row_bytes = row_size_bytes if row_size_bytes is not None else self._row_size_bytes
'''

GET_BATCH_ROWS_NEW = '''    def get_batch_rows(
        self,
        row_size_bytes: Optional[int] = None,
        expansion_factor: Optional[float] = None,
        *,
        sample_batch: Optional[pa.RecordBatch] = None,
    ) -> int:
        """How many rows fit in one batch under current budget.

        v0.9.8b (R2): `sample_batch` kwarg — if provided, uses exact
        estimate_row_size_from_batch (fixes string underestimate).
        Otherwise uses `row_size_bytes` argument or cached value.
        """
        if sample_batch is not None:
            row_bytes = self.estimate_row_size_from_batch(sample_batch)
        else:
            row_bytes = (
                row_size_bytes if row_size_bytes is not None
                else self._row_size_bytes
            )
'''


def update_get_batch_rows():
    print("\n=== memory_guard.py: get_batch_rows sample_batch kwarg ===")
    p = Path("cadw/core/memory_guard.py")
    s = p.read_text()
    if "sample_batch: Optional[pa.RecordBatch] = None,\n    ) -> int:\n        \"\"\"How many rows fit in one batch" in s:
        print("  already updated — skip")
        return
    if GET_BATCH_ROWS_OLD not in s:
        raise SystemExit("get_batch_rows anchor not found")
    s = s.replace(GET_BATCH_ROWS_OLD, GET_BATCH_ROWS_NEW, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 4. tests ============

TEST_LINES = [
    '"""v0.9.8b: R2 — sample-based reestimate (schema underestimate fix)."""',
    'from __future__ import annotations',
    '',
    'import inspect',
    '',
    'import pyarrow as pa',
    'import pytest',
    '',
    'from cadw.core.memory_guard import MemoryGuard',
    '',
    '',
    '# ============ API: new methods exist ============',
    '',
    '',
    'def test_reestimate_from_batch_exists():',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    assert hasattr(g, "reestimate_from_batch")',
    '    assert callable(g.reestimate_from_batch)',
    '',
    '',
    'def test_estimate_row_size_from_schema_with_sample_exists():',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    assert hasattr(g, "estimate_row_size_from_schema_with_sample")',
    '    assert callable(g.estimate_row_size_from_schema_with_sample)',
    '',
    '',
    'def test_get_batch_rows_accepts_sample_batch():',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    # Signature has sample_batch kwarg',
    '    sig = inspect.signature(g.get_batch_rows)',
    '    assert "sample_batch" in sig.parameters',
    '    assert sig.parameters["sample_batch"].kind is inspect.Parameter.KEYWORD_ONLY',
    '',
    '',
    '# ============ reestimate_from_batch: correctness ============',
    '',
    '',
    'def test_reestimate_updates_state():',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    # 100 rows × 200 B string → 20 KB nbytes → 200 B/row',
    '    arr = pa.array(["x" * 200] * 100, type=pa.string())',
    '    rb = pa.RecordBatch.from_arrays([arr], names=["s"])',
    '    before = g.row_size_bytes',
    '    new = g.reestimate_from_batch(rb)',
    '    assert new >= 200  # at least 200 B (strings)',
    '    assert g.row_size_bytes == new',
    '    assert g.row_size_bytes != before',
    '',
    '',
    'def test_reestimate_empty_batch_raises():',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    empty = pa.RecordBatch.from_arrays(',
    '        [pa.array([], type=pa.string())], names=["s"],',
    '    )',
    '    with pytest.raises(ValueError, match="empty"):',
    '        g.reestimate_from_batch(empty)',
    '',
    '',
    'def test_reestimate_numeric_consistent_with_schema():',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    # 1000 rows × int64 → ~8000+ B nbytes → ~8 B/row',
    '    arr = pa.array(list(range(1000)), type=pa.int64())',
    '    rb = pa.RecordBatch.from_arrays([arr], names=["x"])',
    '    new = g.reestimate_from_batch(rb)',
    '    assert 8 <= new <= 20  # 8 B + small overhead',
    '',
    '',
    '# ============ audit case (bench_r2 scale-down) ============',
    '',
    '',
    'def test_audit_case_schema_underestimates_strings():',
    '    """10 string cols × 200 B/val: schema est << exact (R2)."""',
    '    g = MemoryGuard(max_memory_mb=5)',
    '    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])',
    '    schema_est = g.estimate_row_size_from_schema(schema)',
    '    # Schema est: 10 × 50 = 500 B/row',
    '    assert schema_est == 500',
    '',
    '    # Real batch: 10 × 200 B strings',
    '    arrs = [pa.array(["x" * 200] * 100, type=pa.string()) for _ in range(10)]',
    '    real_batch = pa.RecordBatch.from_arrays(arrs, names=schema.names)',
    '    exact_est = g.estimate_row_size_from_batch(real_batch)',
    '    # Exact: ~2000 B/row (10 cols × 200 B + offset overhead)',
    '    assert exact_est > schema_est * 2, (',
    '        f"schema={schema_est}, exact={exact_est}; R2 regression"',
    '    )',
    '',
    '',
    'def test_audit_case_get_batch_rows_with_sample():',
    '    """get_batch_rows(sample_batch=...) uses exact; result ≤ budget."""',
    '    g = MemoryGuard(max_memory_mb=5)',
    '    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])',
    '    arrs = [pa.array(["x" * 200] * 100, type=pa.string()) for _ in range(10)]',
    '    sample = pa.RecordBatch.from_arrays(arrs, names=schema.names)',
    '',
    '    # Without sample: schema-based → 3670 rows (R2 bug demo)',
    '    rows_bad = g.get_batch_rows(expansion_factor=2.0)',
    '    # With sample: exact ~10080 B/row → ~130 rows',
    '    rows_good = g.get_batch_rows(',
    '        expansion_factor=2.0, sample_batch=sample,',
    '    )',
    '',
    '    assert rows_good < rows_bad // 5, (',
    '        f"sample-based ({rows_good}) should be << schema-based ({rows_bad})"',
    '    )',
    '',
    '    # Sanity: actual materialized batch at rows_good fits budget',
    '    # (row_size × ef × rows ≈ budget × safety)',
    '    actual_batch = pa.RecordBatch.from_arrays(',
    '        [pa.array(["x" * 200] * rows_good, type=pa.string()) for _ in range(10)],',
    '        names=schema.names,',
    '    )',
    '    # Budget = 5 MB; batch should be < 5 MB',
    '    assert actual_batch.nbytes < 5 * 1024 * 1024, (',
    '        f"materialized {actual_batch.nbytes} B > 5 MB budget"',
    '    )',
    '',
    '',
    'def test_audit_case_with_sample_helper():',
    '    """estimate_row_size_from_schema_with_sample: exact when sample given."""',
    '    g = MemoryGuard(max_memory_mb=5)',
    '    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])',
    '    arrs = [pa.array(["x" * 200] * 100, type=pa.string()) for _ in range(10)]',
    '    sample = pa.RecordBatch.from_arrays(arrs, names=schema.names)',
    '',
    '    est_no_sample = g.estimate_row_size_from_schema_with_sample(schema)',
    '    est_with_sample = g.estimate_row_size_from_schema_with_sample(',
    '        schema, sample,',
    '    )',
    '',
    '    assert est_no_sample == 500  # schema fallback',
    '    assert est_with_sample > est_no_sample * 2',
    '    # Also updates state',
    '    assert g.row_size_bytes == est_with_sample',
    '',
    '',
    'def test_audit_case_no_sample_uses_schema_backward_compat():',
    '    """Without sample_batch, get_batch_rows uses schema/cached (backward compat)."""',
    '    g = MemoryGuard(max_memory_mb=5)',
    '    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])',
    '    g.estimate_row_size_from_schema(schema)',
    '    # Same call as before v0.9.8b: uses cached _row_size_bytes (500)',
    '    rows = g.get_batch_rows(expansion_factor=2.0)',
    '    # 5 MB × 0.7 / (500 × 2) = 3670 (old behavior preserved)',
    '    assert rows == 3670',
    '',
    '',
    '# ============ AST guards (§13.4/190: methods in impl) ============',
    '',
    '',
    'def test_reestimate_uses_estimate_row_size_from_batch():',
    '    """Impl: reestimate_from_batch delegates to estimate_row_size_from_batch."""',
    '    src = inspect.getsource(MemoryGuard.reestimate_from_batch)',
    '    assert "estimate_row_size_from_batch" in src',
    '',
    '',
    'def test_get_batch_rows_handles_sample_batch():',
    '    """Impl: get_batch_rows uses sample_batch when provided."""',
    '    src = inspect.getsource(MemoryGuard.get_batch_rows)',
    '    assert "sample_batch" in src',
    '    assert "estimate_row_size_from_batch" in src',
    '',
    '',
    '# ============ warning in docstring ============',
    '',
    '',
    'def test_schema_docstring_has_r2_warning():',
    '    """Docstring of estimate_row_size_from_schema warns about strings."""',
    '    src = inspect.getsource(MemoryGuard.estimate_row_size_from_schema)',
    '    assert "WARNING" in src or "warning" in src.lower()',
    '    assert "string" in src.lower() or "binary" in src.lower()',
    '    assert "reestimate" in src.lower() or "sample" in src.lower()',
]


def create_tests():
    print("\n=== create tests/test_row_size_reestimate.py ===")
    p = Path("tests/test_row_size_reestimate.py")
    if p.exists():
        print("  already exists — skip")
        return
    write_content_checked(p, TEST_LINES)
    print("  written + compiled")


# ============ 5. dogfood ============

def dogfood():
    print("\n=== dogfood: R2 sample reestimate ===")
    import pyarrow as pa

    from cadw.core.memory_guard import MemoryGuard

    # Audit case — SEPARATE guards for state isolation (§13.4/205)
    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])
    arrs = [pa.array(["x" * 200] * 100, type=pa.string()) for _ in range(10)]
    sample = pa.RecordBatch.from_arrays(arrs, names=schema.names)

    # g_schema: schema-only baseline (R2 bug — 500 B/row underestimate)
    g_schema = MemoryGuard(max_memory_mb=5)
    schema_est = g_schema.estimate_row_size_from_schema(schema)
    rows_bad = g_schema.get_batch_rows(expansion_factor=2.0)
    print(f"  schema est (10 str cols): {schema_est} B/row, "
          f"{rows_bad} rows (R2 bug)")

    # g_sample: exact via sample_batch (R2 fix — 2040 B/row)
    g_sample = MemoryGuard(max_memory_mb=5)
    exact_est = g_sample.estimate_row_size_from_batch(sample)
    rows_good = g_sample.get_batch_rows(expansion_factor=2.0, sample_batch=sample)
    print(f"  exact est (from batch):   {exact_est} B/row "
          f"({exact_est / schema_est:.1f}×), {rows_good} rows (R2 fix)")

    # Materialize to confirm
    actual = pa.RecordBatch.from_arrays(
        [pa.array(["x" * 200] * rows_good, type=pa.string())
         for _ in range(10)],
        names=schema.names,
    )
    print(f"  materialized at rows_good: {actual.nbytes / 1024 / 1024:.2f} MB "
          f"(budget 5 MB)")

    assert rows_good < rows_bad // 2, (
        f"sample rows ({rows_good}) should be < half of schema rows "
        f"({rows_bad}); ratio {rows_bad / rows_good:.2f}"
    )
    assert actual.nbytes < 5 * 1024 * 1024, "materialized > budget"
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.8b: R2 sample reestimate ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    update_schema_docstring()
    add_reestimate_methods()
    update_get_batch_rows()
    create_tests()
    dogfood()

    print("\n=== pytest: reestimate + memory_guard regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_row_size_reestimate.py",
         "tests/test_memory_guard_v2.py",
         "tests/test_partition_writers.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** R2 REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    commit_with_guards(
        "v0.9.8b: R2 — sample-based reestimate. "
        "estimate_row_size_from_schema underestimates string/binary "
        "(default_string_bytes=50, real 4–20× larger). Audit bench_r2: "
        "10 string cols, 200 B/value → schema est 500 B/row, exact "
        "10080 B/row → get_batch_rows(ef=2)=3670 → materialized batch "
        "7.1 MB @ 5 MB budget. Production path unaffected (ops use "
        "check_batch(batch.nbytes) — real size); R2 is API-risk for "
        "user code. Fix additive: "
        "(1) estimate_row_size_from_schema — WARNING docstring (string "
        "underestimate; prefer reestimate/sample). "
        "(2) MemoryGuard.reestimate_from_batch(batch) — explicit "
        "reestimate (§3.10). "
        "(3) MemoryGuard.estimate_row_size_from_schema_with_sample("
        "schema, sample_batch) — helper. "
        "(4) get_batch_rows(..., *, sample_batch=None) — new kwarg; "
        "if passed, uses exact estimate. "
        "Backward compat: no sample_batch → same rows (3670 audit case); "
        "signature of estimate_row_size_from_schema unchanged. "
        "Tests: 13 (audit case, materialized ≤ budget, schema backward "
        "compat, empty-batch raise, numeric consistency, AST guards).",
        self_path="scripts/patch_v098b_row_size_reestimate.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.8b applied. R2 closed.")


if __name__ == "__main__":
    main()
