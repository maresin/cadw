#!/usr/bin/env python3
# v0.9.10-fix3 — fix2 died on import: cadw.patch_utils triggers
# cadw/__init__ → corrupt dataframe.py (IndentationError). Restore
# BEFORE import. v0.9.10 COMPLETE.
#
# fix2 crash:
#   import cadw.patch_utils as pu          (line 41)
#     → cadw/__init__ → from .dataframe import ...
#     → dataframe.py line 2871 IndentationError
#   restore_dataframe() never ran (inside main(), after import).
#
# §13.4/212 (new): corrupted module state must be restored BEFORE
# the first `import` of that module or its importing chain. Import
# errors are not catchable by later reorder — the crash happens at
# bytecode-level import, not at function call.
#
# Also: fix_apply_or_fail now uses compile-before-write (§13.4/211)
# — same class the fix itself is addressing. Otherwise patching
# patch_utils.py could corrupt it (write-then-compile, fail on
# syntax, file left corrupt).
#
# Run: uv run python scripts/patch_v0910_fix3.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


# ============ 0. Restore corrupt dataframe.py BEFORE importing cadw ============
#
# cadw/__init__.py line 8 imports dataframe; if corrupt, even
# `import cadw.patch_utils` fails at package load. subprocess (git)
# does not import cadw — safe.

print("\n=== pre-import: restore cadw/dataframe.py from git ===")
subprocess.run(
    ["git", "checkout", "--", "cadw/dataframe.py"],
    check=True,
)
_s = Path("cadw/dataframe.py").read_text()
if "Per-group apply (v0.9.6b2: CADW-native" not in _s:
    raise SystemExit("F9 docstring lost on restore (unexpected)")
if "_maybe_upgrade_for_row" in _s:
    raise SystemExit("corrupt C1 insert still present (unexpected)")
print("  restored (F9 intact, C1 removed)")

# Now safe to import cadw chain.
import importlib  # noqa: E402
import cadw.patch_utils as pu  # noqa: E402


# ============ 1. Fix apply_or_fail: compile-before-write (§13.4/211) ============

def fix_apply_or_fail():
    print("\n=== fix apply_or_fail: compile-before-write (§13.4/211) ===")
    old = (
        "    p.write_text(s.replace(old, new, 1))\n"
        "    assert_compiles(p)\n"
        "    return True\n"
    )
    new = (
        "    new_s = s.replace(old, new, 1)\n"
        "    try:\n"
        "        compile(new_s, str(p), 'exec')\n"
        "    except SyntaxError as e:\n"
        "        print(f\"  {label}: COMPILE FAIL for {path}: {e}\")\n"
        "        raise SystemExit(1) from e\n"
        "    p.write_text(new_s)\n"
        "    return True\n"
    )
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if new in s:
        print("  already fixed — skip")
        return
    if old not in s:
        raise SystemExit("apply_or_fail old body not found")
    # compile-before-write (§13.4/211) — this fix itself must not
    # corrupt patch_utils.py on failure.
    new_s = s.replace(old, new, 1)
    try:
        compile(new_s, str(p), "exec")
    except SyntaxError as e:
        raise SystemExit(f"fix_apply_or_fail compile fail: {e}") from e
    p.write_text(new_s)
    pu.assert_compiles(p)
    print("  OK")


ROLLBACK_TEST = [
    "",
    "",
    "def test_apply_or_fail_no_write_on_compile_fail(tmp_path):",
    "    \"\"\"v0.9.10-fix3 (§13.4/211): no write on compile fail.",
    "",
    "    Prior: apply_or_fail wrote file, then assert_compiles → corrupt",
    "    file on bad anchor. Fix: compile in memory, only write on success.",
    "    \"\"\"",
    "    import pytest",
    "    from cadw.patch_utils import apply_or_fail",
    "    p = tmp_path / 'f.py'",
    "    original = 'def f():\\n    x = 1\\n'",
    "    p.write_text(original)",
    "    with pytest.raises(SystemExit):",
    "        apply_or_fail(str(p), '    x = 1\\n', 'x = 1\\n', label='t')",
    "    assert p.read_text() == original",
]


def add_rollback_test():
    print("\n=== add apply_or_fail rollback test ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_apply_or_fail_no_write_on_compile_fail" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(ROLLBACK_TEST))
    pu.assert_compiles(p)
    print("  OK")


# ============ 2. C1 with correct anchors ============

C1_INSERT_OLD = (
    "            buf_rows = []\n"
    "\n"
    "        def _emit_group(key: tuple, result):\n"
)

C1_INSERT_NEW = (
    "            buf_rows = []\n"
    "\n"
    "        def _maybe_upgrade_for_row(row: dict) -> None:\n"
    "            \"\"\"v0.9.10 (C1): widen int->float64 on fractional float.\n"
    "\n"
    "            _build_out_schema infers from first group only. If first is\n"
    "            int, later is float, pa.Table.from_pylist(rows, schema=int64)\n"
    "            either truncates (silent corruption) or raises ArrowInvalid.\n"
    "            Widen to float64 before flush. Lossless for |v| < 2^53.\n"
    "\n"
    "            Limitation: only before first flush (BATCH_ROWS=1000).\n"
    "            If mixed appears after first flush, writer committed to\n"
    "            int64 -> ArrowInvalid. Workaround: return float(x) from fn().\n"
    "            \"\"\"\n"
    "            if not out_schema_box:\n"
    "                return\n"
    "            schema = out_schema_box[0]\n"
    "            upgraded = []\n"
    "            needs = False\n"
    "            for field in schema:\n"
    "                val = row.get(field.name)\n"
    "                if pa.types.is_integer(field.type) and isinstance(val, float):\n"
    "                    if val != int(val):\n"
    "                        upgraded.append(pa.field(\n"
    "                            field.name, pa.float64(),\n"
    "                            nullable=field.nullable,\n"
    "                        ))\n"
    "                        needs = True\n"
    "                        continue\n"
    "                upgraded.append(field)\n"
    "            if needs:\n"
    "                out_schema_box[0] = pa.schema(upgraded)\n"
    "\n"
    "        def _emit_row(row: dict) -> None:\n"
    "            \"\"\"Append a row, widening schema if C1 numeric conflict.\"\"\"\n"
    "            nonlocal buf_rows\n"
    "            _maybe_upgrade_for_row(row)\n"
    "            buf_rows.append(row)\n"
    "\n"
    "        def _emit_group(key: tuple, result):\n"
)


def c1_fix():
    print("\n=== C1: _per_group_apply mixed int/float widen ===")

    pu.apply_or_fail(
        "cadw/dataframe.py", C1_INSERT_OLD, C1_INSERT_NEW,
        label="C1 insert",
    )
    # scalar: 16/16
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "                row[result_col] = _normalize_scalar(result)\n"
            "                buf_rows.append(row)\n"
        ),
        (
            "                row[result_col] = _normalize_scalar(result)\n"
            "                _emit_row(row)\n"
        ),
        label="C1 scalar",
    )
    # dict: 16/20/16
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "                for k, v in result.items():\n"
            "                    row[str(k)] = _normalize_scalar(v)\n"
            "                buf_rows.append(row)\n"
        ),
        (
            "                for k, v in result.items():\n"
            "                    row[str(k)] = _normalize_scalar(v)\n"
            "                _emit_row(row)\n"
        ),
        label="C1 dict",
    )
    # df: 24/28/24
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "                        for gc, v in zip(group_cols, key):\n"
            "                            row.setdefault(gc, v)\n"
            "                        buf_rows.append(row)\n"
        ),
        (
            "                        for gc, v in zip(group_cols, key):\n"
            "                            row.setdefault(gc, v)\n"
            "                        _emit_row(row)\n"
        ),
        label="C1 df",
    )
    print("  C1 OK")


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
    "    \"\"\"",
    "    import pyarrow.parquet as pq",
    "    df = make_df(tmp_path, {",
    "        'g': ['a', 'a', 'b', 'b'],",
    "        'v': [1, 2, 3, 4],",
    "    })",
    "    def fn(g):",
    "        if g['g'].to_list()[0] == 'a':",
    "            return 5  # int (first group)",
    "        return 2.5  # float (later)",
    "    result = df.groupby('g').apply(fn)",
    "    out = tmp_path / 'out.parquet'",
    "    result.write_parquet(out)",
    "    rows = {r['g']: r['apply_result'] for r in",
    "            pq.read_table(out).to_pylist()}",
    "    assert rows['a'] == 5.0",
    "    assert rows['b'] == 2.5",
    "",
    "",
    "def test_apply_all_int_stays_int(tmp_path):",
    "    \"\"\"C1 regression: all-int → schema stays int64 (no widen).\"\"\"",
    "    import pyarrow as pa",
    "    import pyarrow.parquet as pq",
    "    df = make_df(tmp_path, {",
    "        'g': ['a', 'a', 'b', 'b'],",
    "        'v': [1, 2, 3, 4],",
    "    })",
    "    def fn(g):",
    "        return 5",
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


# ============ 3. U2 ============

def u2_fix():
    print("\n=== U2: _concat_batches per-column concat_arrays ===")
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:\n"
            "    if len(batches) == 1:\n"
            "        return batches[0]\n"
            "    return pa.Table.from_batches(batches).combine_chunks().to_batches()[0]\n"
        ),
        (
            "def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:\n"
            "    if len(batches) == 1:\n"
            "        return batches[0]\n"
            "    # v0.9.10 (U2): per-column concat_arrays (1x alloc per col)\n"
            "    # vs from_batches + combine_chunks + to_batches (3x alloc).\n"
            "    # Bounded by batch_size (10k rows in _iter_arrow_batches).\n"
            "    arrays = [\n"
            "        pa.concat_arrays([b.column(i) for b in batches])\n"
            "        for i in range(batches[0].num_columns)\n"
            "    ]\n"
            "    return pa.RecordBatch.from_arrays(\n"
            "        arrays, names=batches[0].schema.names,\n"
            "    )\n"
        ),
        label="U2",
    )
    print("  U2 OK")


# ============ 4. T2.4 ============

def t24_fix():
    print("\n=== T2.4: _merge_fanout §11.5 honest limitation ===")
    pu.apply_or_fail(
        "cadw/pipeline.py",
        (
            "    v0.9.1b MVP: row-selection fan-in (where branches drop different rows)\n"
            "    is NOT supported — raises RuntimeError with count mismatch. See §11.5.\n"
            "    \"\"\"\n"
        ),
        (
            "    v0.9.1b MVP: row-selection fan-in (where branches drop different rows)\n"
            "    is NOT supported — raises RuntimeError with count mismatch. See §11.5.\n"
            "\n"
            "    v0.9.10 (§11.5): F5 duplicate-new-column guard fires only\n"
            "    for columns new relative to dfs[0] (base). If dfs[0]\n"
            "    produces a column with the same name as a sibling's NEW\n"
            "    column, dfs[0] silently wins (base_cols_set absorbs it).\n"
            "    Cannot distinguish \"was in raw df\" from \"added by dfs[0]\"\n"
            "    without explicit raw_columns. Fix (backlog v0.9.11):\n"
            "    optional raw_columns kwarg passed by Pipeline.transform.\n"
            "    Users hitting this: rename conflicting features, or wrap\n"
            "    in ColumnTransformer.\n"
            "    \"\"\"\n"
        ),
        label="T2.4",
    )
    print("  T2.4 OK")


# ============ verify / dogfood / pytest ============

def verify_state():
    print("\n=== verify: v0.9.10 + v0.9.9 ===")
    pu.verify_anchors([
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
        ("cadw/patch_utils.py", "def verify_anchors("),
        ("cadw/dataframe.py", "def _maybe_upgrade_for_row("),
        ("cadw/dataframe.py", "def _emit_row("),
        ("cadw/dataframe.py", "v0.9.10 (U2)"),
        ("cadw/pipeline.py", "raw_columns. Fix (backlog v0.9.11)"),
        # v0.9.9 regression
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
        ("cadw/pipeline.py", "v0.9.9 (N1): use __serialization_name__"),
        ("cadw/accessors/_dt.py", "def _validate_timestamp_only("),
        ("cadw/ops/groupby.py", "v0.9.9 (N3)"),
        ("cadw/pipeline.py", "v0.9.9 (N5)"),
        ("cadw/ml/pca.py", "if TYPE_CHECKING:  # v0.9.9 (B2)"),
        ("cadw/pipeline.py", "# v0.9.9 (B3)"),
        ("cadw/ml/pca.py", "v0.9.9 (B4)"),
        ("cadw/dataframe.py", "Per-group apply (v0.9.6b2: CADW-native"),
    ])
    print("  all anchors present")


def dogfood():
    print("\n=== dogfood: v0.9.10 via subprocess ===")

    def _verify_all():
        import inspect
        from cadw.patch_utils import apply_or_fail, verify_anchors
        assert callable(apply_or_fail) and callable(verify_anchors)
        from cadw.dataframe import CadwDataFrame, _concat_batches
        src = inspect.getsource(CadwDataFrame._per_group_apply)
        assert "_maybe_upgrade_for_row" in src
        n = src.count("_emit_row(row)")
        assert n == 3, f"expected 3 _emit_row calls, got {n}"
        src_concat = inspect.getsource(_concat_batches)
        assert "pa.concat_arrays" in src_concat
        assert "combine_chunks" not in src_concat
        from cadw.pipeline import _merge_fanout
        assert "raw_columns" in inspect.getsource(_merge_fanout)
        print("  T2.1 + C1 + U2 + T2.4 verified")

    pu.run_verify_in_subprocess(_verify_all)


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
        print("\n*** TARGET FAILURE ***")
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


def main():
    print("=== v0.9.10-fix3 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_apply_or_fail()
    importlib.reload(pu)  # pick up compile-before-write
    add_rollback_test()
    c1_fix()
    c1_add_tests()
    u2_fix()
    t24_fix()
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.10-fix3: complete v0.9.10 (C1 + U2 + T2.4). "
        "fix2 died on import: `import cadw.patch_utils` at module "
        "level triggers cadw/__init__ → from .dataframe import "
        "CadwDataFrame → IndentationError (dataframe.py still "
        "corrupt from fix1). restore_dataframe() inside main() "
        "never reached. §13.4/212 (new): corrupted module state "
        "must be restored BEFORE first `import` of that module or "
        "its importing chain — import errors happen at bytecode "
        "load, not function call. Fix3: git checkout at module top, "
        "before `import cadw.patch_utils`; compile-before-write "
        "inside fix_apply_or_fail (§13.4/211 applied to the fix "
        "itself — patching patch_utils.py must not corrupt it); "
        "correct C1 anchors (insert via _flush_buf end 12/12/8, "
        "scalar 16/16, dict 16/20/16, df 24/28/24); U2 per-column "
        "concat_arrays; T2.4 raw_columns §11.5 docstring. C1: "
        "_maybe_upgrade_for_row widens int→float64 on fractional "
        "float. Prior: from_pylist(rows, schema=int64) truncates or "
        "ArrowInvalid. Lossless for |v| < 2^53; BATCH_ROWS limit "
        "documented. Deferred: U1, U3, PF1-3, A1, N6, R3.",
        self_path="scripts/patch_v0910_fix3.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.10-fix3 applied. v0.9.10 COMPLETE.")


if __name__ == "__main__":
    main()
