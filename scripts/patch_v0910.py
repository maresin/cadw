#!/usr/bin/env python3
# v0.9.10 — scaffold v3 (T2.1) + C1 (silent int truncation) +
# U2 (per-column concat) + T2.4 (_merge_fanout §11.5 docstring).
#
# Scope (from prior chat session, agreed):
#   T2.1: patch_utils.apply_or_fail + verify_anchors — promoted from
#     v0.9.9-fix1a/fix1b ad-hoc helpers. 4 consecutive anchor misses
#     in v0.9.9-fix1a/1b/1c/1d (§13.4/160, /161, /206). Fail-fast
#     with context, no silent no-op, idempotent.
#   C1: _per_group_apply — first group int, later float →
#     pa.Table.from_pylist(rows, schema=int64) either truncates
#     (silent corruption) or raises ArrowInvalid (opaque). Fix:
#     widen schema to float64 on fractional-float-in-int-field.
#     §11.5 documented (very large int > 2^53 loses precision).
#   U2: _concat_batches — 3× alloc (from_batches + combine_chunks +
#     to_batches). Fix: per-column pa.concat_arrays (1× alloc).
#   T2.4: _merge_fanout docstring — F5 honest limitation (§11.5).
#     F5 guard only fires for columns new relative to dfs[0]. If
#     dfs[0] adds same name as sibling, base wins silently. Cannot
#     distinguish "was in raw" from "added by dfs[0]" without
#     raw_columns. Fix (backlog): raw_columns kwarg.
#
# Deferred: U1 (test_dt_ceil_date64_raises_init — test creates date32
# though named date64; cosmetic, no data bug), U3 (pyproject version
# "0.1.0" vs CHANGELOG v0.9.9 — not a bug; tag-time concern), PF1/PF2/
# PF3 (Arrow-native heavy keys, pivot, sample — perf, deferred to
# v0.9.11), A1 (dataframe.py split — 5404 lines, deferred), N6/R3.
#
# Run: uv run python scripts/patch_v0910.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ T2.1: apply_or_fail + verify_anchors ============

T21_HELPER_LINES = [
    "",
    "",
    "# ============ v0.9.10 (T2.1): apply_or_fail + verify_anchors ============",
    "#",
    "# Promoted from v0.9.9-fix1a/1b ad-hoc helpers. In v0.9.9-fix1a/b/c/d",
    "# 4 consecutive anchor misses (F9 docstring, F2 anchor, F1 anchor,",
    "# test order). All §13.4/160 (silent no-op) and §13.4/161",
    "# (substring without context). Scaffold v3: fail-fast with context,",
    "# idempotent, no AST fallback that prints diagnostics without warning.",
    "",
    "",
    "def apply_or_fail(path, old, new, *, label, context=10):",
    "    \"\"\"Replace old->new in path; return True if applied, False if already.",
    "",
    "    Fail-fast (SystemExit) if `old` anchor missing — no silent no-op.",
    "    Prints `context` lines around partial-match zone on miss (§13.4/160).",
    "    Idempotent: returns False if `new` already present (no write).",
    "    \"\"\"",
    "    from pathlib import Path as _P",
    "    p = _P(path)",
    "    s = p.read_text()",
    "    if new in s:",
    "        return False",
    "    if old not in s:",
    "        print(f\"  {label}: ANCHOR MISS in {path}\")",
    "        first_line = old.split(chr(10))[0].strip()",
    "        lines = s.split(chr(10))",
    "        for i, line in enumerate(lines):",
    "            if first_line and first_line in line:",
    "                start = max(0, i - 1)",
    "                for j in range(start, min(len(lines), i + context)):",
    "                    print(f\"    L{j + 1:4d}: {lines[j]}\")",
    "                break",
    "        raise SystemExit(f\"{label}: anchor missing in {path}\")",
    "    p.write_text(s.replace(old, new, 1))",
    "    assert_compiles(p)",
    "    return True",
    "",
    "",
    "def verify_anchors(checks, *, fail_fast=True):",
    "    \"\"\"Verify [(path, anchor), ...] all present. Prints all misses.",
    "",
    "    Returns list of (path, anchor) misses. SystemExit(1) if any and",
    "    fail_fast. No write. Use after patches to confirm state before",
    "    dogfood/pytest (§13.4/206: anchor must be exact fragment of new).",
    "    \"\"\"",
    "    import sys as _sys",
    "    from pathlib import Path as _P",
    "    misses = []",
    "    for path, anchor in checks:",
    "        if anchor not in _P(path).read_text():",
    "            misses.append((path, anchor))",
    "    if misses:",
    "        for path, anchor in misses:",
    "            print(f\"  MISSING {path}: {anchor!r}\")",
    "        if fail_fast:",
    "            _sys.exit(1)",
    "    return misses",
]


def t21_add_helpers():
    print("\n=== T2.1: apply_or_fail + verify_anchors ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def apply_or_fail(" in s:
        print("  already present — skip")
        return
    anchor = "\n\ndef _collect_call_linenos("
    if anchor not in s:
        raise SystemExit("T2.1 anchor _collect_call_linenos not found")
    block = "\n".join(T21_HELPER_LINES)
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)
    pu.assert_compiles(p)
    print("  OK")


T21_TEST_LINES = [
    "",
    "",
    "# ============ v0.9.10 (T2.1): apply_or_fail / verify_anchors ============",
    "",
    "",
    "def test_apply_or_fail_basic(tmp_path):",
    "    from cadw.patch_utils import apply_or_fail",
    "    p = tmp_path / 'f.py'",
    "    p.write_text('x = 1\\n')",
    "    applied = apply_or_fail(str(p), 'x = 1', 'x = 2', label='t')",
    "    assert applied is True",
    "    assert p.read_text() == 'x = 2\\n'",
    "",
    "",
    "def test_apply_or_fail_idempotent(tmp_path):",
    "    from cadw.patch_utils import apply_or_fail",
    "    p = tmp_path / 'f.py'",
    "    p.write_text('x = 2\\n')",
    "    applied = apply_or_fail(str(p), 'x = 1', 'x = 2', label='t')",
    "    assert applied is False",
    "    assert p.read_text() == 'x = 2\\n'",
    "",
    "",
    "def test_apply_or_fail_raises_on_miss(tmp_path):",
    "    import pytest",
    "    from cadw.patch_utils import apply_or_fail",
    "    p = tmp_path / 'f.py'",
    "    p.write_text('y = 1\\n')",
    "    with pytest.raises(SystemExit):",
    "        apply_or_fail(str(p), 'x = 1', 'x = 2', label='t')",
    "",
    "",
    "def test_apply_or_fail_prints_context_on_miss(tmp_path, capsys):",
    "    import pytest",
    "    from cadw.patch_utils import apply_or_fail",
    "    p = tmp_path / 'f.py'",
    "    p.write_text('unique_marker_line = 1\\nother = 2\\n')",
    "    with pytest.raises(SystemExit):",
    "        apply_or_fail(",
    "            str(p), 'unique_marker_line = 1\\nwrong_next',",
    "            'never', label='t',",
    "        )",
    "    captured = capsys.readouterr()",
    "    assert 'unique_marker_line' in captured.out",
    "",
    "",
    "def test_verify_anchors_ok(tmp_path):",
    "    from cadw.patch_utils import verify_anchors",
    "    p = tmp_path / 'f.py'",
    "    p.write_text('x = 1\\n')",
    "    misses = verify_anchors([(str(p), 'x = 1')])",
    "    assert misses == []",
    "",
    "",
    "def test_verify_anchors_miss_raises(tmp_path):",
    "    import pytest",
    "    from cadw.patch_utils import verify_anchors",
    "    p = tmp_path / 'f.py'",
    "    p.write_text('x = 1\\n')",
    "    with pytest.raises(SystemExit):",
    "        verify_anchors([(str(p), 'y = 2')])",
    "",
    "",
    "def test_verify_anchors_miss_no_fail_fast(tmp_path):",
    "    from cadw.patch_utils import verify_anchors",
    "    p = tmp_path / 'f.py'",
    "    p.write_text('x = 1\\n')",
    "    misses = verify_anchors([(str(p), 'y = 2')], fail_fast=False)",
    "    assert len(misses) == 1",
    "    assert misses[0][0] == str(p)",
]


def t21_add_tests():
    print("\n=== T2.1: tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_apply_or_fail_basic" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(T21_TEST_LINES))
    pu.assert_compiles(p)
    print("  OK")


# ============ C1: mixed int/float widen in _per_group_apply ============

C1_INSERT_BEFORE_EMIT_GROUP = (
    "    def _maybe_upgrade_for_row(row: dict) -> None:\n"
    "        \"\"\"v0.9.10 (C1): widen int->float64 if row has fractional float.\n"
    "\n"
    "        _build_out_schema infers from first group only. If first is int,\n"
    "        later is float, pa.Table.from_pylist(rows, schema=int64) either\n"
    "        truncates (silent corruption) or raises ArrowInvalid (opaque).\n"
    "        Widen to float64 before flush. Very large int (>2^53) loses\n"
    "        precision on widen — documented in §11.5.\n"
    "        \"\"\"\n"
    "        if not out_schema_box:\n"
    "            return\n"
    "        schema = out_schema_box[0]\n"
    "        upgraded = []\n"
    "        needs = False\n"
    "        for field in schema:\n"
    "            val = row.get(field.name)\n"
    "            if pa.types.is_integer(field.type) and isinstance(val, float):\n"
    "                if val != int(val):\n"
    "                    upgraded.append(pa.field(\n"
    "                        field.name, pa.float64(),\n"
    "                        nullable=field.nullable,\n"
    "                    ))\n"
    "                    needs = True\n"
    "                    continue\n"
    "            upgraded.append(field)\n"
    "        if needs:\n"
    "            out_schema_box[0] = pa.schema(upgraded)\n"
    "\n"
    "    def _emit_row(row: dict) -> None:\n"
    "        \"\"\"Append a row, widening schema if C1 numeric conflict.\"\"\"\n"
    "        nonlocal buf_rows\n"
    "        _maybe_upgrade_for_row(row)\n"
    "        buf_rows.append(row)\n"
    "\n"
    "    def _emit_group(key: tuple, result):\n"
)


def c1_fix():
    print("\n=== C1: _per_group_apply mixed int/float widen ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "v0.9.10 (C1): widen int->float64" in s:
        print("  already fixed — skip")
        return

    # 1. Insert _maybe_upgrade_for_row + _emit_row before _emit_group
    old_emit_def = "    def _emit_group(key: tuple, result):\n"
    if old_emit_def not in s:
        raise SystemExit("C1: _emit_group def anchor not found")
    s = s.replace(
        old_emit_def, C1_INSERT_BEFORE_EMIT_GROUP, 1,
    )

    # 2. Replace 3 buf_rows.append(row) inside _emit_group
    #    Scalar
    old_scalar = (
        "        row[result_col] = _normalize_scalar(result)\n"
        "        buf_rows.append(row)\n"
    )
    new_scalar = (
        "        row[result_col] = _normalize_scalar(result)\n"
        "        _emit_row(row)\n"
    )
    if old_scalar not in s:
        raise SystemExit("C1: scalar buf_rows.append anchor not found")
    s = s.replace(old_scalar, new_scalar, 1)

    #    Dict
    old_dict = (
        "        for k, v in result.items():\n"
        "            row[str(k)] = _normalize_scalar(v)\n"
        "        buf_rows.append(row)\n"
    )
    new_dict = (
        "        for k, v in result.items():\n"
        "            row[str(k)] = _normalize_scalar(v)\n"
        "        _emit_row(row)\n"
    )
    if old_dict not in s:
        raise SystemExit("C1: dict buf_rows.append anchor not found")
    s = s.replace(old_dict, new_dict, 1)

    #    DataFrame
    old_df = (
        "                for gc, v in zip(group_cols, key):\n"
        "                    row.setdefault(gc, v)\n"
        "                buf_rows.append(row)\n"
    )
    new_df = (
        "                for gc, v in zip(group_cols, key):\n"
        "                    row.setdefault(gc, v)\n"
        "                _emit_row(row)\n"
    )
    if old_df not in s:
        raise SystemExit("C1: df buf_rows.append anchor not found")
    s = s.replace(old_df, new_df, 1)

    p.write_text(s)
    pu.assert_compiles(p)
    print("  OK")


C1_TEST_LINES = [
    "",
    "",
    "# ============ v0.9.10 (C1): _per_group_apply mixed int/float widen ============",
    "",
    "",
    "def test_apply_mixed_int_float_widens_schema(tmp_path):",
    "    \"\"\"C1: first group int, later float — schema widens to float64.",
    "",
    "    Prior: pa.Table.from_pylist(rows, schema=int64) truncated",
    "    float(2.5) to int(2), silent corruption. Or raises ArrowInvalid.",
    "    Fix: _maybe_upgrade_for_row widens field to float64 on fractional",
    "    float. int values become float (lossless for |v| < 2^53).",
    "    \"\"\"",
    "    import pyarrow.parquet as pq",
    "    df = make_df(tmp_path, {",
    "        'g': ['a', 'a', 'b', 'b'],",
    "        'v': [1, 2, 3, 4],",
    "    })",
    "    def fn(g):",
    "        if g['g'].to_list()[0] == 'a':",
    "            return 5  # int (first group, schema=int64)",
    "        return 2.5  # float (later group, needs widen)",
    "    result = df.groupby('g').apply(fn)",
    "    out = tmp_path / 'out.parquet'",
    "    result.write_parquet(out)",
    "    rows = {r['g']: r['apply_result'] for r in",
    "            pq.read_table(out).to_pylist()}",
    "    assert rows['a'] == 5.0  # widened to float",
    "    assert rows['b'] == 2.5  # not truncated to 2",
    "",
    "",
    "def test_apply_all_int_stays_int(tmp_path):",
    "    \"\"\"C1 regression: if all groups return int, schema stays int64.",
    "",
    "    No unnecessary widening — only on fractional float.",
    "    \"\"\"",
    "    import pyarrow as pa",
    "    import pyarrow.parquet as pq",
    "    df = make_df(tmp_path, {",
    "        'g': ['a', 'a', 'b', 'b'],",
    "        'v': [1, 2, 3, 4],",
    "    })",
    "    def fn(g):",
    "        return 5  # int for all",
    "    result = df.groupby('g').apply(fn)",
    "    out = tmp_path / 'out.parquet'",
    "    result.write_parquet(out)",
    "    schema = pq.read_schema(out)",
    "    assert schema.field('apply_result').type == pa.int64()",
]


def c1_add_tests():
    print("\n=== C1: tests ===")
    p = Path("tests/test_apply.py")
    s = p.read_text()
    if "def test_apply_mixed_int_float_widens_schema" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(C1_TEST_LINES))
    pu.assert_compiles(p)
    print("  OK")


# ============ U2: _concat_batches per-column ============

def u2_fix():
    print("\n=== U2: _concat_batches per-column pa.concat_arrays ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "v0.9.10 (U2): per-column concat_arrays" in s:
        print("  already fixed — skip")
        return
    old = (
        "def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:\n"
        "    if len(batches) == 1:\n"
        "        return batches[0]\n"
        "    return pa.Table.from_batches(batches).combine_chunks().to_batches()[0]\n"
    )
    new = (
        "def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:\n"
        "    if len(batches) == 1:\n"
        "        return batches[0]\n"
        "    # v0.9.10 (U2): per-column concat_arrays (1× alloc per col)\n"
        "    # vs from_batches + combine_chunks + to_batches (3× alloc).\n"
        "    # Bounded by batch_size (10k rows in _iter_arrow_batches).\n"
        "    arrays = [\n"
        "        pa.concat_arrays([b.column(i) for b in batches])\n"
        "        for i in range(batches[0].num_columns)\n"
        "    ]\n"
        "    return pa.RecordBatch.from_arrays(\n"
        "        arrays, names=batches[0].schema.names,\n"
        "    )\n"
    )
    if old not in s:
        raise SystemExit("U2: _concat_batches anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    pu.assert_compiles(p)
    print("  OK")


# ============ T2.4: _merge_fanout §11.5 docstring ============

def t24_fix():
    print("\n=== T2.4: _merge_fanout §11.5 honest limitation docstring ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()
    if "v0.9.10 (§11.5, C1-adjacent)" in s:
        print("  already fixed — skip")
        return
    old = (
        "    v0.9.1b MVP: row-selection fan-in (where branches drop different rows)\n"
        "    is NOT supported — raises RuntimeError with count mismatch. See §11.5.\n"
        "    \"\"\"\n"
    )
    new = (
        "    v0.9.1b MVP: row-selection fan-in (where branches drop different rows)\n"
        "    is NOT supported — raises RuntimeError with count mismatch. See §11.5.\n"
        "\n"
        "    v0.9.10 (§11.5, C1-adjacent): F5 duplicate-new-column guard\n"
        "    fires only for columns new relative to dfs[0] (base). If\n"
        "    dfs[0] produces a column with the same name as a sibling's\n"
        "    NEW column, dfs[0] silently wins (base_cols_set absorbs it).\n"
        "    Cannot distinguish \"was in raw df\" from \"added by dfs[0]\"\n"
        "    without explicit raw_columns. Fix (backlog v0.9.11):\n"
        "    optional raw_columns kwarg passed by Pipeline.transform →\n"
        "    _merge_fanout. Users hitting this: rename conflicting\n"
        "    features, or wrap in ColumnTransformer.\n"
        "    \"\"\"\n"
    )
    if old not in s:
        raise SystemExit("T2.4: _merge_fanout docstring anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.10 + v0.9.9 F1-F9 on disk ===")
    checks = [
        # T2.1
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "def verify_anchors("),
        # C1
        ("cadw/dataframe.py", "v0.9.10 (C1): widen int->float64"),
        ("cadw/dataframe.py", "_emit_row(row)"),
        ("cadw/dataframe.py", "def _maybe_upgrade_for_row("),
        # U2
        ("cadw/dataframe.py", "v0.9.10 (U2): per-column concat_arrays"),
        # T2.4
        ("cadw/pipeline.py", "v0.9.10 (§11.5, C1-adjacent)"),
        # v0.9.9 F1-F9 (regression — verify not broken)
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
        ("cadw/pipeline.py", "v0.9.9 (N1): use __serialization_name__"),
        ("cadw/accessors/_dt.py", "def _validate_timestamp_only("),
        ("cadw/ops/groupby.py", "v0.9.9 (N3)"),
        ("cadw/pipeline.py", "v0.9.9 (N5)"),
        ("cadw/ml/pca.py", "if TYPE_CHECKING:  # v0.9.9 (B2)"),
        ("cadw/pipeline.py", "# v0.9.9 (B3): for `-> Path` annotation"),
        ("cadw/ml/pca.py", "v0.9.9 (B4)"),
        ("cadw/dataframe.py", "Per-group apply (v0.9.6b2: CADW-native"),
    ]
    pu.verify_anchors(checks)
    print(f"  all {len(checks)} anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.10 via fresh subprocess ===")
    importlib.reload(pu)

    def _verify_all():
        import inspect
        import pyarrow as pa
        # T2.1
        from cadw.patch_utils import apply_or_fail, verify_anchors
        assert callable(apply_or_fail)
        assert callable(verify_anchors)
        # C1: _emit_row exists, _maybe_upgrade_for_row exists
        from cadw.dataframe import CadwDataFrame
        src = inspect.getsource(CadwDataFrame._per_group_apply)
        assert "_maybe_upgrade_for_row" in src
        assert "_emit_row(row)" in src
        assert src.count("_emit_row(row)") >= 3, (
            f"expected >=3 _emit_row calls in _per_group_apply, "
            f"got {src.count('_emit_row(row)')}"
        )
        # U2
        src_concat = inspect.getsource(
            __import__("cadw.dataframe", fromlist=["_concat_batches"])
            ._concat_batches
        )
        assert "pa.concat_arrays" in src_concat
        assert "combine_chunks" not in src_concat
        # T2.4
        from cadw.pipeline import _merge_fanout
        src_mf = inspect.getsource(_merge_fanout)
        assert "v0.9.10 (§11.5" in src_mf
        print("  T2.1, C1, U2, T2.4 all verified")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_apply.py",
         "tests/test_apply_e2e.py",
         "tests/test_pipeline_dag.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGET REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=900,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.10 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    # T2.1 first — scaffold v3 in place
    t21_add_helpers()
    t21_add_tests()

    # C1 — critical correctness
    c1_fix()
    c1_add_tests()

    # U2 — perf, no new test (behavior identical)
    u2_fix()

    # T2.4 — docstring
    t24_fix()

    # Verify via scaffold's own helper
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.10: scaffold v3 + C1 + U2 + T2.4. "
        "T2.1: promoted apply_or_fail + verify_anchors from "
        "v0.9.9-fix1a/b into cadw/patch_utils.py (4 consecutive "
        "anchor misses in v0.9.9 fix chain — §13.4/160/161/206). "
        "Fail-fast with context, idempotent, no AST fallback silent "
        "no-op. C1: _per_group_apply mixed int/float widen. "
        "Prior: first group int → schema=int64; later float "
        "2.5 → pa.Table.from_pylist(rows, schema=int64) truncated "
        "to 2 (silent corruption) or raised ArrowInvalid (opaque). "
        "Fix: _maybe_upgrade_for_row + _emit_row: on fractional "
        "float in int field, widen schema to float64 before flush. "
        "int → float lossless for |v| < 2^53; very large int loses "
        "precision, documented §11.5. U2: _concat_batches per-column "
        "pa.concat_arrays (1× alloc vs 3× for from_batches + "
        "combine_chunks + to_batches). Bounded by batch_size. "
        "T2.4: _merge_fanout docstring — F5 honest limitation "
        "(§11.5): duplicate-new-column guard fires only for "
        "columns new relative to dfs[0]; if base adds same name as "
        "sibling, base wins silently. raw_columns kwarg — backlog "
        "v0.9.11. Deferred: U1 (test_dt_ceil_date64 — cosmetic), "
        "U3 (pyproject version — tag concern), PF1/PF2/PF3 "
        "(Arrow-native heavy keys / pivot / sample — v0.9.11), "
        "A1 (dataframe.py split — 5404 lines), N6, R3.",
        self_path="scripts/patch_v0910.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.10 applied.")


if __name__ == "__main__":
    main()
