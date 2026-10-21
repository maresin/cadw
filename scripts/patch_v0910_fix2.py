#!/usr/bin/env python3
# v0.9.10-fix2 — C1 anchor + apply_or_fail compile-before-write.
# v0.9.10 COMPLETE.
#
# v0.9.10-fix1 corrupted cadw/dataframe.py: C1 insert anchor
# "    def _emit_group(key: tuple, result):\n" (4-space) is a
# SUBSTRING of actual 8-space line, so apply_or_fail matched at
# index 4 (mid-indent), replacing the first 4 spaces + def, leaving
# "nonlocal buf_rows" at 0-indent. IndentationError at line 2871.
#
# Root is TWO bugs:
#  (a) Anchor uses 4-space prefix that matches inside 8-space line.
#      Same off-by-4 for scalar (16 vs 12), dict (16/20/16 vs
#      12/16/12), df (24/28/24 vs 20/24/20).
#  (b) apply_or_fail writes file, THEN compiles. On compile fail,
#      file left corrupt. §13.4/211 (new): fail-fast helper that
#      writes-then-checks is not fail-fast — corrupts state.
#
# Fix2:
#  1. git checkout cadw/dataframe.py (restore corrupt C1 insert).
#     F9 committed → intact.
#  2. apply_or_fail: compile new_s IN MEMORY, only write on success.
#     + test_apply_or_fail_no_write_on_compile_fail.
#  3. C1 with correct anchors: insert via `_flush_buf` end context
#     (12-space `buf_rows = []` + 8-space `def _emit_group`),
#     scalar 16/16, dict 16/20/16, df 24/28/24.
#  4. U2 (concat per-column), T2.4 (docstring).
#  5. verify + dogfood + pytest + commit.
#
# Run: uv run python scripts/patch_v0910_fix2.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ 1. Restore corrupt dataframe.py ============

def restore_dataframe():
    print("\n=== restore cadw/dataframe.py from git ===")
    subprocess.run(
        ["git", "checkout", "--", "cadw/dataframe.py"],
        check=True,
    )
    s = Path("cadw/dataframe.py").read_text()
    if "Per-group apply (v0.9.6b2: CADW-native" not in s:
        raise SystemExit("F9 docstring lost on restore (unexpected)")
    if "_maybe_upgrade_for_row" in s:
        raise SystemExit("corrupt C1 insert still present (unexpected)")
    print("  restored (F9 intact, C1 removed)")


# ============ 2. Fix apply_or_fail: compile-before-write ============

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
    p.write_text(s.replace(old, new, 1))
    pu.assert_compiles(p)
    print("  OK (file untouched on bad anchor)")


ROLLBACK_TEST = [
    "",
    "",
    "def test_apply_or_fail_no_write_on_compile_fail(tmp_path):",
    "    \"\"\"v0.9.10-fix2 (§13.4/211): no write on compile fail.",
    "",
    "    Prior: apply_or_fail wrote file, then assert_compiles → corrupt",
    "    file on bad anchor (v0.9.10-fix1: 4-space 'def _emit_group'",
    "    matched inside 8-space line, removed indent).",
    "    Fix: compile in memory, only write on success.",
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


# ============ 3. C1 with correct anchors (16, 16/20/16, 24/28/24) ============

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
    "            int64 → ArrowInvalid. Workaround: return float(x) from fn().\n"
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


# ============ 4. U2 ============

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


# ============ 5. T2.4 ============

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


# ============ verify ============

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
    print("=== v0.9.10-fix2 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    restore_dataframe()
    fix_apply_or_fail()
    add_rollback_test()
    importlib.reload(pu)
    c1_fix()
    c1_add_tests()
    u2_fix()
    t24_fix()
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.10-fix2: complete v0.9.10 (C1 + U2 + T2.4). "
        "v0.9.10-fix1 corrupted cadw/dataframe.py: C1 insert anchor "
        "'    def _emit_group(key: tuple, result):' (4-space) is "
        "substring of actual 8-space line → apply_or_fail matched at "
        "index 4 (mid-indent), leaving 'nonlocal buf_rows' at 0-indent. "
        "Same off-by-4 for scalar (16 vs 12), dict (16/20/16 vs "
        "12/16/12), df (24/28/24 vs 20/24/20). Root §13.4/211 (new): "
        "apply_or_fail wrote file, THEN compiled — on fail, file "
        "corrupt on disk. Fix2: (1) git checkout cadw/dataframe.py "
        "(F9 committed, intact); (2) apply_or_fail compile-before-"
        "write: compile(new_s) in memory, only write on success, "
        "test_apply_or_fail_no_write_on_compile_fail; (3) C1 correct "
        "anchors: insert via _flush_buf end (12-space 'buf_rows = []' "
        "+ 8-space 'def _emit_group'), scalar 16/16, dict 16/20/16, "
        "df 24/28/24; (4) U2 _concat_batches per-column "
        "pa.concat_arrays; (5) T2.4 _merge_fanout §11.5 honest limit "
        "(raw_columns kwarg backlog). C1: _maybe_upgrade_for_row + "
        "_emit_row widen int→float64 on fractional float; prior "
        "truncated (silent) or ArrowInvalid. Lossless for |v| < 2^53; "
        "BATCH_ROWS limitation documented. Deferred: U1 (test "
        "cosmetic), U3 (pyproject version), PF1-PF3 (perf), A1 "
        "(dataframe split), N6, R3.",
        self_path="scripts/patch_v0910_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.10-fix2 applied. v0.9.10 COMPLETE.")


if __name__ == "__main__":
    main()
