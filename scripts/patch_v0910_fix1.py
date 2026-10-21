#!/usr/bin/env python3
# v0.9.10-fix1 — C1 anchor indentation fix + U2 + T2.4. v0.9.10 COMPLETE.
#
# v0.9.10 died at c1_fix() step 2: `C1: scalar buf_rows.append anchor
# not found`. Root: anchors in patch had 8-space indent (`        row[
# result_col]`), but actual code is 12-space (inside `if mode ==
# "scalar":` inside `def _emit_group:` inside `_per_group_apply`).
# Same off-by-4 for dict (8/12/8 vs 12/16/12) and df (16/20/16 vs
# 20/24/20). §13.4/161 (substring anchor without context), now in
# patch body — worse than verify_state.
#
# On disk (T2.1 from v0.9.10 only):
#   apply_or_fail / verify_anchors in patch_utils.py  — present
#   T2.1 tests                                         — present
#   C1                                                 — NOT on disk
#   U2, T2.4                                           — NOT on disk
#   commit                                             — never reached
#
# Irony: v0.9.10 introduces scaffold v3 (apply_or_fail) but does NOT
# use it in c1_fix() — inline `if old not in s: raise SystemExit()`.
# That's exactly the class the helper was built for. fix1 uses
# pu.apply_or_fail() for all replacements — real dogfood of the
# helper, plus correct anchors from snapshot (12/12, 12/16/12,
# 20/24/20).
#
# Run: uv run python scripts/patch_v0910_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
importlib.reload(pu)


# ============ C1: insert _maybe_upgrade_for_row + _emit_row ============

C1_INSERT_BEFORE_EMIT_GROUP = (
    "    def _maybe_upgrade_for_row(row: dict) -> None:\n"
    "        \"\"\"v0.9.10 (C1): widen int->float64 if row has fractional float.\n"
    "\n"
    "        _build_out_schema infers from first group only. If first is int,\n"
    "        later is float, pa.Table.from_pylist(rows, schema=int64) either\n"
    "        truncates (silent corruption) or raises ArrowInvalid (opaque).\n"
    "        Widen to float64 before flush. Very large int (>2^53) loses\n"
    "        precision on widen.\n"
    "\n"
    "        Limitation (backlog v0.9.11): only works before first flush\n"
    "        (BATCH_ROWS=1000). If mixed int/float appears after first\n"
    "        flush, writer is already committed to int64 -> ArrowInvalid.\n"
    "        Workaround: return float(x) from fn() if any group might be\n"
    "        float.\n"
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

    # 1. Insert _maybe_upgrade_for_row + _emit_row before _emit_group
    pu.apply_or_fail(
        "cadw/dataframe.py",
        "    def _emit_group(key: tuple, result):\n",
        C1_INSERT_BEFORE_EMIT_GROUP,
        label="C1 insert",
    )

    # 2. Scalar branch — 12/12 (inside `if mode == "scalar":`)
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "            row[result_col] = _normalize_scalar(result)\n"
            "            buf_rows.append(row)\n"
        ),
        (
            "            row[result_col] = _normalize_scalar(result)\n"
            "            _emit_row(row)\n"
        ),
        label="C1 scalar",
    )

    # 3. Dict branch — 12/16/12 (inside `elif mode == "dict":`)
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "            for k, v in result.items():\n"
            "                row[str(k)] = _normalize_scalar(v)\n"
            "            buf_rows.append(row)\n"
        ),
        (
            "            for k, v in result.items():\n"
            "                row[str(k)] = _normalize_scalar(v)\n"
            "            _emit_row(row)\n"
        ),
        label="C1 dict",
    )

    # 4. DataFrame branch — 20/24/20 (nested loops under `else:`)
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "                    for gc, v in zip(group_cols, key):\n"
            "                        row.setdefault(gc, v)\n"
            "                    buf_rows.append(row)\n"
        ),
        (
            "                    for gc, v in zip(group_cols, key):\n"
            "                        row.setdefault(gc, v)\n"
            "                    _emit_row(row)\n"
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


# ============ U2: _concat_batches per-column ============

def u2_fix():
    print("\n=== U2: _concat_batches per-column pa.concat_arrays ===")
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


# ============ T2.4: _merge_fanout §11.5 docstring ============

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
    print("\n=== verify: v0.9.10 + v0.9.9 F1-F9 ===")
    pu.verify_anchors([
        # T2.1 (from v0.9.10 first run)
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "def verify_anchors("),
        # C1
        ("cadw/dataframe.py", "def _maybe_upgrade_for_row("),
        ("cadw/dataframe.py", "def _emit_row("),
        # U2
        ("cadw/dataframe.py", "v0.9.10 (U2)"),
        # T2.4
        ("cadw/pipeline.py", "raw_columns. Fix (backlog v0.9.11)"),
        # v0.9.9 F1-F9 (regression)
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


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.10 via subprocess ===")

    def _verify_all():
        import inspect
        # T2.1
        from cadw.patch_utils import apply_or_fail, verify_anchors
        assert callable(apply_or_fail)
        assert callable(verify_anchors)
        # C1
        from cadw.dataframe import CadwDataFrame
        src = inspect.getsource(CadwDataFrame._per_group_apply)
        assert "_maybe_upgrade_for_row" in src
        n = src.count("_emit_row(row)")
        assert n >= 3, f"expected >=3 _emit_row calls, got {n}"
        # U2
        from cadw.dataframe import _concat_batches
        src_concat = inspect.getsource(_concat_batches)
        assert "pa.concat_arrays" in src_concat
        assert "combine_chunks" not in src_concat
        # T2.4
        from cadw.pipeline import _merge_fanout
        assert "raw_columns" in inspect.getsource(_merge_fanout)
        print("  T2.1 + C1 + U2 + T2.4 verified")

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


# ============ main ============

def main():
    print("=== v0.9.10-fix1 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    c1_fix()
    c1_add_tests()
    u2_fix()
    t24_fix()
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.10-fix1: complete v0.9.10 (C1 + U2 + T2.4). "
        "v0.9.10 died at c1_fix step 2: scalar anchor had 8-space "
        "indent but actual code has 12 (inside 'if mode == scalar:'). "
        "Same for dict (8/12/8 vs 12/16/12) and df (16/20/16 vs "
        "20/24/20). §13.4/161, first time in patch BODY not verify. "
        "Irony: v0.9.10 introduces scaffold v3 (apply_or_fail) but "
        "c1_fix uses inline 'if old not in s: raise' — exactly the "
        "class the helper prevents. fix1 uses pu.apply_or_fail() "
        "for all replacements with correct anchors. C1: "
        "_maybe_upgrade_for_row + _emit_row widen int->float64 on "
        "fractional-float-in-int-field. Prior: pa.Table.from_pylist"
        "(rows, schema=int64) with float 2.5 → either truncates to "
        "2 (silent) or ArrowInvalid. Fix: schema widen before flush. "
        "Lossless for |v| < 2^53; BATCH_ROWS limitation documented. "
        "U2: _concat_batches per-column pa.concat_arrays (1x alloc "
        "vs 3x). T2.4: _merge_fanout docstring — F5 honest limit: "
        "base_cols_set absorbs columns added by dfs[0]; raw_columns "
        "kwarg backlog v0.9.11. Deferred: U1 (test cosmetic), U3 "
        "(pyproject version), PF1-PF3 (perf), A1 (dataframe split), "
        "N6, R3.",
        self_path="scripts/patch_v0910_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.10-fix1 applied. v0.9.10 COMPLETE.")


if __name__ == "__main__":
    main()
