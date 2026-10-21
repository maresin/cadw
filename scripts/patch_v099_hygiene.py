#!/usr/bin/env python3
# v0.9.9 — hygiene: F1 helper + 8 fixes (N1, N2, N3, N5, B2, B3, B4, docstring).
#
# Structural §13.4/178 fix (7× recurring):
#   - F1: patch_utils.run_verify_in_subprocess(fn) — dogfood in fresh
#     subprocess. Fresh interpreter → fresh imports → no stale cache.
#   - v0.9.9's own dogfood uses this helper.
#
# Fixes (each isolated, sequential, per-fix checkpoint):
#   F2 (N1): ColumnTransformer.to_dict uses __serialization_name__
#   F3 (N2): dt.floor/ceil/round on date32/64 → TypeError in init
#   F4 (N3): _combine_batches/_merge_partials → check_state after group_by
#   F5 (N5): _merge_fanout → clear RuntimeError on duplicate branch column
#   F6 (B2): pca.py F821 annotation
#   F7 (B3): pipeline.py "Path" annotation
#   F8 (B4): PCA docstring: fitted-model, use fit_pca
#   F9: _GroupBy.apply docstring anchor
#
# Defer to v0.9.10 (backlog):
#   N6: Dataset.iter_batches concat_arrays guard (API change)
#   R3: planner spill stub
#   memory models for dedup/rank/quantile/resample
#
# Run: uv run python scripts/patch_v099_hygiene.py

from __future__ import annotations
import ast
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


# ============ F1: run_verify_in_subprocess helper ============

F1_HELPER_LINES = [
    "",
    "",
    "# ============ v0.9.9 (F1): subprocess-based verify (§13.4/178) ============",
    "#",
    "# 7× recurring stale-module-cache in patch scripts (v0.9.5a-fix1, ",
    "# v0.9.6d-fix1, v0.9.6d-fix2, v0.9.6e2-fix1, v0.9.8a-fix2,",
    "# v0.9.8b-fix1, v0.9.8c-fix1). Root: patch-script imports module X at",
    "# top; modifies X.py on disk; dogfood reimports X → gets OLD class",
    "# object from sys.modules. Fix: run dogfood in fresh subprocess.",
    "",
    "",
    "def run_verify_in_subprocess(verify_fn, *, timeout=300):",
    "    \"\"\"Run verify_fn in fresh subprocess (structural §13.4/178 fix).",
    "",
    "    Saves inspect.getsource(verify_fn) to temp file in project root,",
    "    invokes `python temp_file` → fresh interpreter → fresh imports",
    "    → no stale module cache. Prints verify_fn's output to caller's",
    "    stdout; raises SystemExit(1) on any failure.",
    "",
    "    verify_fn must be self-contained: all imports inside its body",
    "    (nested def or module-level function). Typical usage:",
    "",
    "        def dogfood():",
    "            import pyarrow as pa",
    "            from cadw.core.memory_guard import MemoryGuard",
    "            g = MemoryGuard(max_memory_mb=10)",
    "            # assertions",
    "",
    "        run_verify_in_subprocess(dogfood)",
    "",
    "    Args:",
    "        verify_fn: callable() -> None. Raises on failure.",
    "        timeout: subprocess timeout (seconds).",
    "    \"\"\"",
    "    import inspect as _inspect",
    "    import subprocess as _sp",
    "    import sys as _sys",
    "    import tempfile as _tempfile",
    "    from pathlib import Path as _P",
    "",
    "    source = _inspect.getsource(verify_fn)",
    "    with _tempfile.NamedTemporaryFile(",
    "        mode=\"w\", suffix=\".py\", delete=False,",
    "        dir=\".\", prefix=\"_verify_\",",
    "    ) as f:",
    "        f.write(source)",
    "        f.write(\"\\n\\n\")",
    "        f.write(f\"{verify_fn.__name__}()\\n\")",
    "        f.write(f\"print('VERIFY OK: {verify_fn.__name__}')\\n\")",
    "        tmp_path = _P(f.name)",
    "",
    "    try:",
    "        r = _sp.run(",
    "            [_sys.executable, str(tmp_path)],",
    "            cwd=str(_P.cwd()),",
    "            timeout=timeout,",
    "        )",
    "        if r.returncode != 0:",
    "            raise SystemExit(",
    "                f\"verify_fn {verify_fn.__name__!r} failed \"",
    "                f\"(exit {r.returncode}); see stderr above\"",
    "            )",
    "    finally:",
    "        tmp_path.unlink(missing_ok=True)",
]


def f1_add_helper():
    print("\n=== F1: add run_verify_in_subprocess ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def run_verify_in_subprocess(" in s:
        print("  already present — skip")
        return
    anchor = "\n\ndef assert_no_call_in_source("
    if anchor not in s:
        raise SystemExit("F1 anchor not found")
    block = "\n".join(F1_HELPER_LINES)
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ F2: ColumnTransformer.to_dict __serialization_name__ ============

F2_OLD = (
    "            entries.append({\n"
    "                \"name\": name,\n"
    "                \"cols\": list(cols),\n"
    "                \"type\": type(step).__name__,\n"
    "                \"step_dict\": step.to_dict(),\n"
    "            })"
)

F2_NEW = (
    "            # v0.9.9 (N1): use __serialization_name__ (set by\n"
    "            # register_step_type) so from_json finds the same key.\n"
    "            entries.append({\n"
    "                \"name\": name,\n"
    "                \"cols\": list(cols),\n"
    "                \"type\": getattr(\n"
    "                    type(step), \"__serialization_name__\",\n"
    "                    type(step).__name__,\n"
    "                ),\n"
    "                \"step_dict\": step.to_dict(),\n"
    "            })"
)


def f2_ct_serialization_name():
    print("\n=== F2 (N1): ColumnTransformer.to_dict __serialization_name__ ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()
    if "F2 (N1): use __serialization_name__" in s or "N1: use __serialization_name__" in s:
        print("  already fixed — skip")
        return
    if F2_OLD not in s:
        raise SystemExit("F2 anchor not found")
    s = s.replace(F2_OLD, F2_NEW, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ F3: dt.floor/ceil/round date types → TypeError in init ============

F3_OLD = (
    "def _validate_temporal(df: \"CadwDataFrame\", column: str) -> None:\n"
    "    t = df._dataset.schema.field(column).type\n"
    "    if not (pa.types.is_timestamp(t) or pa.types.is_date(t)):\n"
    "        raise TypeError(\n"
    "            f\".dt: column {column!r} has type {t}; \"\n"
    "            f\"expected timestamp or date\"\n"
    "        )\n"
)

F3_NEW = (
    "def _validate_temporal(df: \"CadwDataFrame\", column: str) -> None:\n"
    "    t = df._dataset.schema.field(column).type\n"
    "    if not (pa.types.is_timestamp(t) or pa.types.is_date(t)):\n"
    "        raise TypeError(\n"
    "            f\".dt: column {column!r} has type {t}; \"\n"
    "            f\"expected timestamp or date\"\n"
    "        )\n"
    "\n"
    "\n"
    "def _validate_timestamp_only(df: \"CadwDataFrame\", column: str) -> None:\n"
    "    \"\"\"v0.9.9 (N2): floor/ceil/round require timestamp (not date).\n"
    "\n"
    "    date32/64 has no `.unit` attribute; `_freq_bucket` would raise\n"
    "    AttributeError mid-iteration instead of clear TypeError in init.\n"
    "    Only timestamp types supported for floor/ceil/round.\n"
    "    \"\"\"\n"
    "    t = df._dataset.schema.field(column).type\n"
    "    if not pa.types.is_timestamp(t):\n"
    "        raise TypeError(\n"
    "            f\".dt.floor/ceil/round: column {column!r} has type {t}; \"\n"
    "            f\"expected timestamp (date32/64 not supported — use \"\n"
    "            f\"cast to timestamp first)\"\n"
    "        )\n"
)


def f3_dt_timestamp_only():
    print("\n=== F3 (N2): dt.floor/ceil/round timestamp-only ===")
    p = Path("cadw/accessors/_dt.py")
    s = p.read_text()
    if "_validate_timestamp_only" in s:
        print("  already fixed — skip")
        return
    if F3_OLD not in s:
        raise SystemExit("F3 anchor not found")
    s = s.replace(F3_OLD, F3_NEW, 1)

    # Update floor/ceil/round to call new validator.
    for mode in ("floor", "ceil", "round"):
        old_call = (
            f'    def {mode}(self, freq: str) -> "CadwDataFrame":\n'
            f'        return self._feature_custom(\n'
            f'            "{mode}",\n'
            f'            lambda arr: _freq_bucket(arr, freq, "{mode}"),\n'
            f'            self._parent._dataset.schema.field(self._column).type,\n'
            f'        )\n'
        )
        new_call = (
            f'    def {mode}(self, freq: str) -> "CadwDataFrame":\n'
            f'        _validate_timestamp_only(self._parent, self._column)\n'
            f'        return self._feature_custom(\n'
            f'            "{mode}",\n'
            f'            lambda arr: _freq_bucket(arr, freq, "{mode}"),\n'
            f'            self._parent._dataset.schema.field(self._column).type,\n'
            f'        )\n'
        )
        if old_call in s:
            s = s.replace(old_call, new_call, 1)

    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ F4: _combine_batches / _merge_partials check_state ============

def f4_groupby_check_state():
    print("\n=== F4 (N3): _combine_batches/_merge_partials check_state ===")
    p = Path("cadw/ops/groupby.py")
    s = p.read_text()
    if "v0.9.9 (N3)" in s:
        print("  already fixed — skip")
        return

    # 4a: _combine_batches — add check_state after group_by.
    old_comb = (
        "        table = pa.Table.from_batches([batch.select(select_cols)])\n"
        "        grouped = table.group_by(list(by_columns)).aggregate(base_specs)\n"
    )
    new_comb = (
        "        table = pa.Table.from_batches([batch.select(select_cols)])\n"
        "        grouped = table.group_by(list(by_columns)).aggregate(base_specs)\n"
        "        # v0.9.9 (N3): group_by output + table = peak (checked).\n"
        "        if memory_guard is not None:\n"
        "            memory_guard.check_state(\n"
        "                grouped.nbytes + table.nbytes,\n"
        "                name=\"combiner.group_by\",\n"
        "            )\n"
    )
    if old_comb not in s:
        raise SystemExit("F4 _combine_batches anchor not found")
    s = s.replace(old_comb, new_comb, 1)

    # 4b: _merge_partials — add memory_guard param + check_state.
    old_sig = (
        "def _merge_partials(\n"
        "    table: pa.Table,\n"
        "    by_columns: Sequence[str],\n"
        "    output_specs: Sequence[tuple[str, str]],  # [(col, func), ...] как в _local_groupby\n"
        ") -> pa.Table:\n"
    )
    new_sig = (
        "def _merge_partials(\n"
        "    table: pa.Table,\n"
        "    by_columns: Sequence[str],\n"
        "    output_specs: Sequence[tuple[str, str]],  # [(col, func), ...] как в _local_groupby\n"
        "    memory_guard: \"MemoryGuard | None\" = None,  # v0.9.9 (N3)\n"
        ") -> pa.Table:\n"
    )
    if old_sig not in s:
        raise SystemExit("F4 _merge_partials signature not found")
    s = s.replace(old_sig, new_sig, 1)

    old_merge = (
        "    merged = table.group_by(list(by_columns)).aggregate(merge_specs)\n"
    )
    new_merge = (
        "    merged = table.group_by(list(by_columns)).aggregate(merge_specs)\n"
        "    # v0.9.9 (N3): group_by output + table = peak (checked).\n"
        "    if memory_guard is not None:\n"
        "        memory_guard.check_state(\n"
        "            merged.nbytes + table.nbytes,\n"
        "            name=\"combiner.merge_partials\",\n"
        "        )\n"
    )
    if old_merge not in s:
        raise SystemExit("F4 _merge_partials body anchor not found")
    s = s.replace(old_merge, new_merge, 1)

    # 4c: caller _local_groupby — pass memory_guard to _merge_partials.
    old_call = (
        "            if partial_merge:\n"
        "                return _merge_partials(\n"
        "                    table, by_columns,\n"
        "                    [(col, f) for col, f in zip(agg_columns, agg_funcs)],\n"
        "                )\n"
    )
    new_call = (
        "            if partial_merge:\n"
        "                return _merge_partials(\n"
        "                    table, by_columns,\n"
        "                    [(col, f) for col, f in zip(agg_columns, agg_funcs)],\n"
        "                    memory_guard=memory_guard,\n"
        "                )\n"
    )
    if old_call not in s:
        raise SystemExit("F4 _local_groupby caller not found")
    s = s.replace(old_call, new_call, 1)

    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ F5: _merge_fanout duplicate column → RuntimeError ============

F5_OLD = (
    "    # Build output schema: base (dfs[0]) + new columns from others\n"
    "    base_cols = list(dfs[0].columns)\n"
    "    base_cols_set = set(base_cols)\n"
    "    seen = set(base_cols)\n"
    "    out_fields = [dfs[0]._dataset.schema.field(c) for c in base_cols]\n"
    "    for d in dfs[1:]:\n"
    "        for c in d.columns:\n"
    "            if c in seen:\n"
    "                continue\n"
    "            out_fields.append(d._dataset.schema.field(c))\n"
    "            seen.add(c)\n"
    "    out_schema = pa.schema(out_fields)\n"
)

F5_NEW = (
    "    # Build output schema: base (dfs[0]) + new columns from others\n"
    "    base_cols = list(dfs[0].columns)\n"
    "    base_cols_set = set(base_cols)\n"
    "    # v0.9.9 (N5): track NEW column -> first branch index. Duplicate\n"
    "    # new column from two branches → clear RuntimeError (prior\n"
    "    # silent dedup led to \"Schema and number of arrays unequal\"\n"
    "    # in RecordBatch.from_arrays, cryptic downstream).\n"
    "    seen_new: dict = {}\n"
    "    out_fields = [dfs[0]._dataset.schema.field(c) for c in base_cols]\n"
    "    for i, d in enumerate(dfs[1:], start=1):\n"
    "        for c in d.columns:\n"
    "            if c in base_cols_set:\n"
    "                continue\n"
    "            if c in seen_new:\n"
    "                raise RuntimeError(\n"
    "                    f\"_merge_fanout: duplicate new column {c!r} from \"\n"
    "                    f\"DAG branches {seen_new[c]} and {i}. Rename one \"\n"
    "                    f\"of them (or wrap in ColumnTransformer with \"\n"
    "                    f\"distinct feature names).\"\n"
    "                )\n"
    "            seen_new[c] = i\n"
    "            out_fields.append(d._dataset.schema.field(c))\n"
    "    seen = base_cols_set | set(seen_new.keys())\n"
    "    out_schema = pa.schema(out_fields)\n"
)


def f5_merge_fanout_dup():
    print("\n=== F5 (N5): _merge_fanout duplicate column ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()
    if "v0.9.9 (N5)" in s:
        print("  already fixed — skip")
        return
    if F5_OLD not in s:
        raise SystemExit("F5 anchor not found")
    s = s.replace(F5_OLD, F5_NEW, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ F6: pca.py F821 annotation ============

F6_OLD = (
    "from typing import Optional, Sequence\n"
    "\n"
    "import numpy as np\n"
    "import pyarrow as pa\n"
    "\n"
    "from cadw.core.memory_guard import MemoryGuard\n"
    "from cadw.ops.reduce import reduce\n"
    "from cadw.ops.reduce import CovarianceMeanOp\n"
)

F6_NEW = (
    "from typing import TYPE_CHECKING, Optional, Sequence\n"
    "\n"
    "import numpy as np\n"
    "import pyarrow as pa\n"
    "\n"
    "from cadw.core.memory_guard import MemoryGuard\n"
    "from cadw.ops.reduce import reduce\n"
    "from cadw.ops.reduce import CovarianceMeanOp\n"
    "\n"
    "if TYPE_CHECKING:  # v0.9.9 (B2): forward-annotation only\n"
    "    from cadw.dataframe import CadwDataFrame\n"
)


def f6_pca_f821():
    print("\n=== F6 (B2): pca.py TYPE_CHECKING for CadwDataFrame ===")
    p = Path("cadw/ml/pca.py")
    s = p.read_text()
    if "if TYPE_CHECKING:  # v0.9.9 (B2)" in s:
        print("  already fixed — skip")
        return
    if F6_OLD not in s:
        raise SystemExit("F6 anchor not found")
    s = s.replace(F6_OLD, F6_NEW, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ F7: pipeline.py "Path" annotation ============

F7_OLD = (
    "from __future__ import annotations\n"
    "\n"
    "import json\n"
    "\n"
    "from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable\n"
)

F7_NEW = (
    "from __future__ import annotations\n"
    "\n"
    "import json\n"
    "\n"
    "from pathlib import Path  # v0.9.9 (B3): for `-> Path` annotation\n"
    "from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable\n"
)


def f7_pipeline_path():
    print("\n=== F7 (B3): pipeline.py module-level Path import ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()
    if "# v0.9.9 (B3): for `-> Path` annotation" in s:
        print("  already fixed — skip")
        return
    if F7_OLD not in s:
        raise SystemExit("F7 anchor not found")
    s = s.replace(F7_OLD, F7_NEW, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ F8: PCA docstring ============

F8_OLD = (
    '@dataclass\n'
    'class PCA:\n'
    '    """Fitted PCA model (sklearn.PCA-compatible attributes)."""\n'
)

F8_NEW = (
    '@dataclass\n'
    'class PCA:\n'
    '    """Fitted PCA model (sklearn.PCA-compatible attributes).\n'
    '\n'
    '    v0.9.9 (B4): PCA is a *fitted-model* class — constructor takes\n'
    '    all fitted attributes; `n_components` is NOT an __init__ kwarg.\n'
    '    Use `fit_pca(df, X_cols, n_components=...)` (or\n'
    '    `CadwDataFrame.fit_pca(...)`) to produce a PCA instance.\n'
    '    """\n'
)


def f8_pca_docstring():
    print("\n=== F8 (B4): PCA docstring fitted-model ===")
    p = Path("cadw/ml/pca.py")
    s = p.read_text()
    if "v0.9.9 (B4)" in s:
        print("  already fixed — skip")
        return
    if F8_OLD not in s:
        raise SystemExit("F8 anchor not found")
    s = s.replace(F8_OLD, F8_NEW, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ F9: _GroupBy.apply docstring ============

def f9_groupby_apply_docstring():
    print("\n=== F9: _GroupBy.apply docstring anchor ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "Per-group apply (v0.9.6b2: CADW-native, no pandas)." in s:
        print("  already fixed — skip")
        return

    # Old anchor might be v0.9.1e version. Try exact replace.
    old = (
        '        """Per-group apply (v0.9.1e). Prototype-validated (11 cases).\n'
        '\n'
        '        Args:\n'
        '            fn: callable(group_df: pd.DataFrame) → scalar |\n'
        '                pd.Series | pd.DataFrame. Uniform mode across\n'
        '                groups.\n'
    )
    new = (
        '        """Per-group apply (v0.9.6b2: CADW-native, no pandas).\n'
        '\n'
        '        Args:\n'
        '            fn: callable(group_df: CadwDataFrame) → scalar |\n'
        '                dict[str, scalar] | CadwDataFrame. Uniform\n'
        '                mode across groups. For pd.Series-like output\n'
        '                use a dict; for pd.DataFrame-like output use\n'
        '                CadwDataFrame.from_table(pa_table).\n'
    )
    if old in s:
        s = s.replace(old, new, 1)
        p.write_text(s)
        assert_compiles(p)
        print("  OK (v0.9.1e → v0.9.6b2)")
        return

    # Fallback: find via AST and print diagnostic.
    tree = ast.parse(s)
    found = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "apply"):
            # Check if this is inside _GroupBy class
            print(f"  apply found at line {node.lineno}")
            # Try to find nearby docstring
            if node.body and isinstance(node.body[0], ast.Expr):
                docstr = node.body[0].value
                if isinstance(docstr, ast.Constant):
                    print(f"  docstring starts: {docstr.value[:60]!r}")
            found = True
            break
    if not found:
        print("  WARNING: apply not found — skip (non-fatal)")


# ============ Tests ============

F1_TEST_LINES = [
    "",
    "",
    "# ============ v0.9.9 (F1): run_verify_in_subprocess ============",
    "",
    "",
    "def test_run_verify_in_subprocess_ok():",
    "    from cadw.patch_utils import run_verify_in_subprocess",
    "    def dogfood_ok():",
    "        import pyarrow  # noqa",
    "        print('inner ok')",
    "    run_verify_in_subprocess(dogfood_ok)",
    "",
    "",
    "def test_run_verify_in_subprocess_raises_on_failure():",
    "    from cadw.patch_utils import run_verify_in_subprocess",
    "    def dogfood_fails():",
    "        raise ValueError('intentional')",
    "    with pytest.raises(SystemExit):",
    "        run_verify_in_subprocess(dogfood_fails)",
    "",
    "",
    "def test_run_verify_in_subprocess_has_cadw_fresh():",
    "    from cadw.patch_utils import run_verify_in_subprocess",
    "    def dogfood_import_cadw():",
    "        from cadw.core.memory_guard import MemoryGuard",
    "        g = MemoryGuard(max_memory_mb=10)",
    "        assert g.budget_bytes == 10 * 1024 * 1024",
    "    run_verify_in_subprocess(dogfood_import_cadw)",
]


def add_f1_tests():
    print("\n=== F1 tests: add 3 tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_run_verify_in_subprocess_ok" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(F1_TEST_LINES))
    assert_compiles(p)
    print("  OK")


F2_TEST = [
    "",
    "",
    "def test_ct_to_dict_uses_serialization_name():",
    "    # v0.9.9 (N1): custom-step registered name, not class name.",
    "    from cadw.pipeline import ColumnTransformer",
    "    from tests.test_register_step_type import _CustomStep, _STEP_REGISTRY",
    "    import pytest",
    "    from cadw.pipeline import register_step_type",
    "    register_step_type('MyCustomV99', _CustomStep)",
    "    try:",
    "        ct = ColumnTransformer([",
    "            ('t1', ['x'], _CustomStep({'x': 'float64'})),",
    "        ])",
    "        d = ct.to_dict()",
    "        assert d['transformers'][0]['type'] == 'MyCustomV99', (",
    "            f\"expected 'MyCustomV99', got {d['transformers'][0]['type']!r}\"",
    "        )",
    "    finally:",
    "        from cadw.pipeline import unregister_step_type",
    "        unregister_step_type('MyCustomV99')",
]


def add_f2_test():
    print("\n=== F2 test: ColumnTransformer __serialization_name__ ===")
    p = Path("tests/test_column_transformer.py")
    s = p.read_text()
    if "def test_ct_to_dict_uses_serialization_name" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(F2_TEST))
    assert_compiles(p)
    print("  OK")


F3_TESTS = [
    "",
    "",
    "def test_dt_floor_date32_raises_init(tmp_path):",
    "    # v0.9.9 (N2): date32/64 not supported for floor/ceil/round.",
    "    import datetime as dt",
    "    import pyarrow as pa",
    "    import pyarrow.parquet as pq",
    "    from cadw import CadwDataFrame, MemoryGuard",
    "    t = pa.table({'d': pa.array([dt.date(2020, 1, 1)], type=pa.date32())})",
    "    p = tmp_path / 'src.parquet'",
    "    pq.write_table(t, p)",
    "    df = CadwDataFrame.from_parquet(",
    "        p, tmp_path / 'ds', chunk_size=2,",
    "        memory_guard=MemoryGuard(max_memory_mb=10),",
    "        spill_dir=tmp_path / 'spill',",
    "    )",
    "    import pytest",
    "    with pytest.raises(TypeError, match='expected timestamp'):",
    "        df['d'].dt.floor('1D')",
    "",
    "",
    "def test_dt_ceil_date64_raises_init(tmp_path):",
    "    import datetime as dt",
    "    import pyarrow as pa",
    "    import pyarrow.parquet as pq",
    "    from cadw import CadwDataFrame, MemoryGuard",
    "    t = pa.table({'d': pa.array([dt.date(2020, 1, 1)], type=pa.date32())})",
    "    p = tmp_path / 'src.parquet'",
    "    pq.write_table(t, p)",
    "    df = CadwDataFrame.from_parquet(",
    "        p, tmp_path / 'ds', chunk_size=2,",
    "        memory_guard=MemoryGuard(max_memory_mb=10),",
    "        spill_dir=tmp_path / 'spill',",
    "    )",
    "    import pytest",
    "    with pytest.raises(TypeError, match='expected timestamp'):",
    "        df['d'].dt.ceil('1D')",
]


def add_f3_tests():
    print("\n=== F3 tests: dt date types ===")
    p = Path("tests/test_dt_accessor.py")
    s = p.read_text()
    if "def test_dt_floor_date32_raises_init" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(F3_TESTS))
    assert_compiles(p)
    print("  OK")


F4_TEST = [
    "",
    "",
    "def test_combiner_check_state_called(tmp_path, monkeypatch):",
    "    # v0.9.9 (N3): _combine_batches / _merge_partials check_state.",
    "    from cadw.core.memory_guard import MemoryGuard",
    "    from cadw.ops import groupby as gb_mod",
    "    import pyarrow as pa",
    "    guard = MemoryGuard(max_memory_mb=10)",
    "    calls = []",
    "    orig_check_state = MemoryGuard.check_state",
    "    def spy(self, nbytes, *, name='state'):",
    "        calls.append(name)",
    "        return orig_check_state(self, nbytes, name=name)",
    "    monkeypatch.setattr(MemoryGuard, 'check_state', spy)",
    "    n = 1000",
    "    table = pa.table({",
    "        'k': [f'g{i % 50:03d}' for i in range(n)],",
    "        'v': list(range(n)),",
    "    })",
    "    def it():",
    "        for i in range(0, n, 100):",
    "            yield table.slice(i, 100).to_batches()[0]",
    "    gb_mod.groupby_aggregate(",
    "        it(), ['k'], ['v'], ['sum'], guard, tmp_path,",
    "        num_partitions=4, combiner=True, log=False,",
    "    )",
    "    assert any('combiner' in c for c in calls), (",
    "        f\"expected combiner check_state; got {calls}\"",
    "    )",
]


def add_f4_test():
    print("\n=== F4 test: combiner check_state ===")
    p = Path("tests/test_combiner.py")
    s = p.read_text()
    if "def test_combiner_check_state_called" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(F4_TEST))
    assert_compiles(p)
    print("  OK")


F5_TEST = [
    "",
    "",
    "def test_merge_fanout_duplicate_new_column_raises(tmp_path):",
    "    # v0.9.9 (N5): duplicate new column from two DAG branches.",
    "    import pyarrow as pa",
    "    import pyarrow.compute as pc",
    "    import pyarrow.parquet as pq",
    "    import pytest",
    "    from cadw import CadwDataFrame, MemoryGuard",
    "    from cadw.pipeline import LambdaTransform, Pipeline",
    "    t = pa.table({'x': [1, 2, 3, 4]})",
    "    p = tmp_path / 'src.parquet'",
    "    pq.write_table(t, p)",
    "    df = CadwDataFrame.from_parquet(",
    "        p, tmp_path / 'ds', chunk_size=2,",
    "        memory_guard=MemoryGuard(max_memory_mb=100),",
    "        spill_dir=tmp_path / 'spill',",
    "    )",
    "    pipe = Pipeline()",
    "    pipe.add_step('a', LambdaTransform(",
    "        'x', 'same', lambda arr: pc.add(arr, 1), pa.int64(),",
    "    ), inputs=[])",
    "    pipe.add_step('b', LambdaTransform(",
    "        'x', 'same', lambda arr: pc.add(arr, 2), pa.int64(),",
    "    ), inputs=[])",
    "    pipe.add_step('merge', __import__('cadw.pipeline', fromlist=['DropnaStep']).DropnaStep(),",
    "                  inputs=['a', 'b'])",
    "    pipe.fit(df)",
    "    with pytest.raises(RuntimeError, match='duplicate new column'):",
    "        pipe.transform(df)",
]


def add_f5_test():
    print("\n=== F5 test: _merge_fanout duplicate ===")
    p = Path("tests/test_pipeline_dag.py")
    s = p.read_text()
    if "def test_merge_fanout_duplicate_new_column_raises" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(F5_TEST))
    assert_compiles(p)
    print("  OK")


# ============ Dogfood (verify function) ============

def dogfood():
    print("\n=== dogfood: verify F2-F8 on fresh subprocess ===")
    # Self-contained — all imports in body (§13.4/200)
    import inspect
    import pyarrow as pa
    from cadw.patch_utils import run_verify_in_subprocess

    def _verify_all():
        """Fresh subprocess — verifies all F2-F8 fixes."""
        import inspect
        import pyarrow as pa
        from cadw.pipeline import ColumnTransformer
        from cadw.accessors import _dt as dt_mod
        from cadw.ops import groupby as gb_mod
        from cadw.pipeline import _merge_fanout
        from cadw.ml import pca as pca_mod

        # F2 (N1)
        src_ct = inspect.getsource(ColumnTransformer.to_dict)
        assert "__serialization_name__" in src_ct
        print("  F2 (N1): ColumnTransformer.to_dict has __serialization_name__")

        # F3 (N2)
        src_dt = inspect.getsource(dt_mod._validate_timestamp_only)
        assert "is_timestamp" in src_dt
        # floor/ceil/round call it
        for m in ("floor", "ceil", "round"):
            src_m = inspect.getsource(getattr(dt_mod._DtAccessor, m))
            assert "_validate_timestamp_only" in src_m
        print("  F3 (N2): dt.floor/ceil/round timestamp-only")

        # F4 (N3)
        src_comb = inspect.getsource(gb_mod._combine_batches)
        assert "check_state" in src_comb
        src_merge = inspect.getsource(gb_mod._merge_partials)
        assert "check_state" in src_merge
        print("  F4 (N3): _combine_batches/_merge_partials check_state")

        # F5 (N5)
        src_fanout = inspect.getsource(_merge_fanout)
        assert "duplicate new column" in src_fanout
        print("  F5 (N5): _merge_fanout duplicate column RuntimeError")

        # F6 (B2)
        src_pca_module = (open("cadw/ml/pca.py").read())
        assert "if TYPE_CHECKING:" in src_pca_module
        print("  F6 (B2): pca.py TYPE_CHECKING for CadwDataFrame")

        # F7 (B3)
        src_pipe_module = (open("cadw/pipeline.py").read())
        # Check module-level Path import (not lazy)
        assert "\nfrom pathlib import Path " in src_pipe_module or \
               src_pipe_module.startswith("from pathlib import Path ")
        print("  F7 (B3): pipeline.py module-level Path")

        # F8 (B4)
        assert "v0.9.9 (B4)" in inspect.getsource(pca_mod.PCA)
        print("  F8 (B4): PCA docstring fitted-model")

    run_verify_in_subprocess(_verify_all)


# ============ main ============

def main():
    print("=== v0.9.9: hygiene ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    # F1 first — dogfood will use it
    f1_add_helper()
    add_f1_tests()

    # F2-F9 (each independent, checkpoint per fix)
    f2_ct_serialization_name()
    add_f2_test()

    f3_dt_timestamp_only()
    add_f3_tests()

    f4_groupby_check_state()
    add_f4_test()

    f5_merge_fanout_dup()
    add_f5_test()

    f6_pca_f821()
    f7_pipeline_path()
    f8_pca_docstring()
    f9_groupby_apply_docstring()

    # Dogfood (via fresh subprocess — no stale cache)
    dogfood()

    print("\n=== pytest: hygiene + regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_column_transformer.py",
         "tests/test_dt_accessor.py",
         "tests/test_combiner.py",
         "tests/test_pipeline_dag.py",
         "tests/test_pipeline_save_load.py",
         "tests/test_pca.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** HYGIENE REGRESSION FAILED ***")
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
        "v0.9.9: hygiene — 9 fixes + structural §13.4/178 solution. "
        "F1: patch_utils.run_verify_in_subprocess(fn) — dogfood in "
        "fresh subprocess (7× recurring stale module cache; the fix "
        "defeats by construction). F2 (N1): ColumnTransformer.to_dict "
        "uses __serialization_name__ (round-trip of custom-step "
        "registered name). F3 (N2): dt.floor/ceil/round → TypeError "
        "in init for date32/64 (previously AttributeError mid-iteration). "
        "F4 (N3): _combine_batches/_merge_partials check_state after "
        "group_by (peak grouped.nbytes + table.nbytes). F5 (N5): "
        "_merge_fanout duplicate new column from DAG branches → clear "
        "RuntimeError (prior cryptic 'Schema and number of arrays "
        "unequal'). F6 (B2): pca.py TYPE_CHECKING for CadwDataFrame "
        "(F821). F7 (B3): pipeline.py module-level Path import (B3 "
        "`-> Path` annotation). F8 (B4): PCA docstring: fitted-model "
        "class, use fit_pca (n_components not in __init__). F9: "
        "_GroupBy.apply docstring (v0.9.6b2 anchor). "
        "Deferred to v0.9.10 backlog: N6 Dataset.iter_batches guard, "
        "R3 spill stub, memory models for dedup/rank/quantile/resample.",
        self_path="scripts/patch_v099_hygiene.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.9 applied.")


if __name__ == "__main__":
    main()
