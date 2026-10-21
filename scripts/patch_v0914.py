#!/usr/bin/env python3
# v0.9.14 — final small fixes: F12-6, F12-7, U3, T2.2.
#
# F12-6 (dataframe.py, S3 correctness): _maybe_upgrade_schema handles
#   int→float64 (v0.9.10 C1) but NOT bool→float64. If fn() returns
#   True (bool, schema pa.bool_) first group, then 0.5 (float) later
#   → pa.Table.from_pylist(rows, schema=bool) raises ArrowInvalid
#   (cryptic). Same class as C1. Fix: extend _maybe_upgrade_schema
#   to widen bool→float64 on any non-bool numeric value (parallel
#   to int→float64 logic). Also: any OTHER type mismatch (e.g.
#   string field + numeric later) → try/except around from_pylist
#   raises clear RuntimeError with hint.
#
# F12-7 (dataframe.py, §11.5 doc): _per_group_apply empty-input
#   branch produces flat schema [group_cols..., result_col: float64],
#   but fn's mode (scalar/dict/dataframe) unknown at empty-input time.
#   For dict/dataframe mode with empty input — schema is not accurate
#   (only result_col present, not dict keys / df columns). Documented
#   §11.5, no behavior change.
#
# U3 (pyproject.toml): version "0.1.0" → "0.9.14". Tag-time concern:
#   pip show / metadata was 0.1.0 since initial scaffold, drifted from
#   CHANGELOG 3+ months. No code impact; honest metadata.
#
# T2.2 (patch_utils.py + tests, S3 tooling): assert_no_call_in_source
#   gets optional `receiver_type` kwarg. F12-5 dogfood had to build
#   inline AST walk (Subscript-receiver filter for `X[i].as_py()`
#   vs `Call.as_py()`) because helper matched ANY as_py name.
#   §13.4/217 (precise AST check). New helper use: v0.9.14 dogfood.
#
# Run: uv run python scripts/patch_v0914.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ F12-6a: bool widen in _maybe_upgrade_schema ============

def f12_6a_bool_widen():
    print("\n=== F12-6a: _maybe_upgrade_schema bool→float64 widen ===")
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "            schema = out_schema_box[0]\n"
            "            upgraded = []\n"
            "            needs = False\n"
            "            widen_fields: list = []\n"
            "            for field in schema:\n"
            "                if not pa.types.is_integer(field.type):\n"
            "                    upgraded.append(field)\n"
            "                    continue\n"
            "                widen = False\n"
            "                for row in buf_rows:\n"
            "                    val = row.get(field.name)\n"
            "                    if isinstance(val, float) and val != int(val):\n"
            "                        widen = True\n"
            "                        break\n"
            "                if widen:\n"
        ),
        (
            "            schema = out_schema_box[0]\n"
            "            upgraded = []\n"
            "            needs = False\n"
            "            widen_fields: list = []\n"
            "            for field in schema:\n"
            "                # v0.9.14 (F12-6a): bool field, any non-bool\n"
            "                # numeric → widen to float64. Prior: pa.bool_\n"
            "                # schema + later float → ArrowInvalid (cryptic).\n"
            "                # Same class as C1 (int→float).\n"
            "                if pa.types.is_boolean(field.type):\n"
            "                    widen = False\n"
            "                    for row in buf_rows:\n"
            "                        val = row.get(field.name)\n"
            "                        if isinstance(val, (int, float)) and not isinstance(val, bool):\n"
            "                            widen = True\n"
            "                            break\n"
            "                    if widen:\n"
            "                        upgraded.append(pa.field(\n"
            "                            field.name, pa.float64(),\n"
            "                            nullable=field.nullable,\n"
            "                        ))\n"
            "                        needs = True\n"
            "                        widen_fields.append(field.name)\n"
            "                    else:\n"
            "                        upgraded.append(field)\n"
            "                    continue\n"
            "                if not pa.types.is_integer(field.type):\n"
            "                    upgraded.append(field)\n"
            "                    continue\n"
            "                widen = False\n"
            "                for row in buf_rows:\n"
            "                    val = row.get(field.name)\n"
            "                    if isinstance(val, float) and val != int(val):\n"
            "                        widen = True\n"
            "                        break\n"
            "                if widen:\n"
        ),
        label="F12-6a",
    )
    print("  OK")


# ============ F12-6b: clear RuntimeError on remaining type mismatch ============

def f12_6b_flush_guard():
    print("\n=== F12-6b: clear RuntimeError on non-widenable type mismatch ===")
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "            _maybe_upgrade_schema()  # v0.9.11 (PF4)\n"
            "            if not writer_box:\n"
            "                writer_box.append(\n"
            "                    ArrowIPCBatchWriter(str(applied_path),\n"
            "                                        out_schema_box[0])\n"
            "                )\n"
            "            table = pa.Table.from_pylist(buf_rows,\n"
            "                                         schema=out_schema_box[0])\n"
        ),
        (
            "            _maybe_upgrade_schema()  # v0.9.11 (PF4)\n"
            "            if not writer_box:\n"
            "                writer_box.append(\n"
            "                    ArrowIPCBatchWriter(str(applied_path),\n"
            "                                        out_schema_box[0])\n"
            "                )\n"
            "            # v0.9.14 (F12-6b): non-widenable type mismatch\n"
            "            # (e.g. str field + numeric later) → clear error,\n"
            "            # not cryptic ArrowInvalid from from_pylist.\n"
            "            try:\n"
            "                table = pa.Table.from_pylist(buf_rows,\n"
            "                                             schema=out_schema_box[0])\n"
            "            except (pa.ArrowInvalid, pa.ArrowTypeError) as _e:\n"
            "                raise RuntimeError(\n"
            "                    f\"_per_group_apply: fn() returned \"\n"
            "                    f\"inconsistent types across groups. \"\n"
            "                    f\"Buffered schema={out_schema_box[0]}, \"\n"
            "                    f\"error={_e}. fn() must return the \"\n"
            "                    f\"same type for all groups. Hint: \"\n"
            "                    f\"return float(x) from fn() if numeric \"\n"
            "                    f\"types might mix (int/float/bool); \"\n"
            "                    f\"return None for missing values. §11.5\"\n"
            "                ) from _e\n"
        ),
        label="F12-6b",
    )
    print("  OK")


# ============ F12-7: §11.5 docstring note (empty-input mode) ============

def f12_7_doc():
    print("\n=== F12-7: _per_group_apply §11.5 empty-input mode note ===")
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "        Peak RAM = batch + 2 × max(group_size × row_size) + fn's output.\n"
            "        Bounded per group, independent of N.\n"
        ),
        (
            "        Peak RAM = batch + 2 × max(group_size × row_size) + fn's output.\n"
            "        Bounded per group, independent of N.\n"
            "\n"
            "        Empty-input limitation (§11.5, v0.9.14 F12-7):\n"
            "        when df.count() == 0, fn is never called; output\n"
            "        schema = [group_cols..., result_col: float64]. This\n"
            "        is only accurate for scalar mode. For dict mode\n"
            "        (fn returns dict[str, scalar]) or DataFrame mode\n"
            "        (fn returns CadwDataFrame) with empty input, the\n"
            "        output is the flat result_col schema — not dict\n"
            "        keys / df columns, since mode is unknown at\n"
            "        empty-input time. Caller who needs accurate empty\n"
            "        schema for dict/df mode: use a non-empty check or\n"
            "        synthesize one row.\n"
        ),
        label="F12-7",
    )
    print("  OK")


# ============ U3: pyproject version ============

def u3_pyproject():
    print("\n=== U3: pyproject version 0.1.0 → 0.9.14 ===")
    pu.apply_or_fail(
        "pyproject.toml",
        (
            'name = "cadw-framework"\n'
            'version = "0.1.0"\n'
        ),
        (
            'name = "cadw-framework"\n'
            'version = "0.9.14"\n'
        ),
        label="U3",
    )
    print("  OK")


# ============ T2.2: assert_no_call_in_source receiver_type filter ============

def t2_2_helper():
    print("\n=== T2.2: assert_no_call_in_source receiver_type kwarg ===")
    pu.apply_or_fail(
        "cadw/patch_utils.py",
        (
            "def _collect_call_linenos(tree):\n"
            "    \"\"\"Return {attr_or_name: [lineno, ...]} for all Call nodes.\n"
            "\n"
            "    Handles both `obj.attr(...)` (ast.Attribute, use .attr) and\n"
            "    `bare(...)` (ast.Name, use .id). Docstrings/comments are\n"
            "    invisible to ast.walk over Call nodes (§13.4/161, /183, /185).\n"
            "    \"\"\"\n"
            "    import ast as _ast\n"
            "    out: dict = {}\n"
            "    for node in _ast.walk(tree):\n"
            "        if not isinstance(node, _ast.Call):\n"
            "            continue\n"
            "        f = node.func\n"
            "        if isinstance(f, _ast.Name):\n"
            "            out.setdefault(f.id, []).append(node.lineno)\n"
            "        elif isinstance(f, _ast.Attribute):\n"
            "            out.setdefault(f.attr, []).append(node.lineno)\n"
            "    return out\n"
        ),
        (
            "def _collect_call_linenos(tree, *, receiver_type=None):\n"
            "    \"\"\"Return {attr_or_name: [lineno, ...]} for all Call nodes.\n"
            "\n"
            "    Handles both `obj.attr(...)` (ast.Attribute, use .attr) and\n"
            "    `bare(...)` (ast.Name, use .id). Docstrings/comments are\n"
            "    invisible to ast.walk over Call nodes (§13.4/161, /183, /185).\n"
            "\n"
            "    v0.9.14 (T2.2, §13.4/217): optional `receiver_type` filter\n"
            "    (ast class). When set, only Calls where\n"
            "    `node.func.value` is `isinstance(..., receiver_type)` are\n"
            "    collected. Use ast.Subscript for per-row `X[i].method()`,\n"
            "    ast.Call for aggregate `fn(...).method()`.\n"
            "    \"\"\"\n"
            "    import ast as _ast\n"
            "    out: dict = {}\n"
            "    for node in _ast.walk(tree):\n"
            "        if not isinstance(node, _ast.Call):\n"
            "            continue\n"
            "        f = node.func\n"
            "        if receiver_type is not None:\n"
            "            if not (isinstance(f, _ast.Attribute)\n"
            "                    and isinstance(f.value, receiver_type)):\n"
            "                continue\n"
            "        if isinstance(f, _ast.Name):\n"
            "            out.setdefault(f.id, []).append(node.lineno)\n"
            "        elif isinstance(f, _ast.Attribute):\n"
            "            out.setdefault(f.attr, []).append(node.lineno)\n"
            "    return out\n"
        ),
        label="T2.2 _collect",
    )
    pu.apply_or_fail(
        "cadw/patch_utils.py",
        (
            "def assert_no_call_in_source(src, *names, fn_name=\"<unknown>\"):\n"
            "    \"\"\"AST: raise AssertionError if any Call node matches `name`.\n"
            "\n"
            "    `name` may be a bare function call (`foo()`) or an attribute\n"
            "    call (`obj.foo()`). Comments and docstrings are invisible\n"
            "    (§13.4/161, /183) — only ast.Call nodes are checked. Replaces\n"
            "    `assert 'foo' not in src`, which catches `# No foo() here`\n"
            "    and `Use CadwDataFrame.foo(...)` docstrings.\n"
            "    \"\"\"\n"
            "    import ast as _ast\n"
            "    import textwrap as _tw\n"
            "    try:\n"
            "        tree = _ast.parse(_tw.dedent(src))\n"
            "    except SyntaxError as e:\n"
            "        raise AssertionError(\n"
            "            f\"assert_no_call_in_source({fn_name}): parse error {e}\"\n"
            "        ) from None\n"
            "    calls = _collect_call_linenos(tree)\n"
        ),
        (
            "def assert_no_call_in_source(\n"
            "    src, *names, fn_name=\"<unknown>\", receiver_type=None,\n"
            "):\n"
            "    \"\"\"AST: raise AssertionError if any Call node matches `name`.\n"
            "\n"
            "    `name` may be a bare function call (`foo()`) or an attribute\n"
            "    call (`obj.foo()`). Comments and docstrings are invisible\n"
            "    (§13.4/161, /183) — only ast.Call nodes are checked. Replaces\n"
            "    `assert 'foo' not in src`, which catches `# No foo() here`\n"
            "    and `Use CadwDataFrame.foo(...)` docstrings.\n"
            "\n"
            "    v0.9.14 (T2.2, §13.4/217): optional `receiver_type` ast class.\n"
            "    When set, only Calls where `node.func.value` is instance\n"
            "    of `receiver_type` match. Common: ast.Subscript for\n"
            "    per-row `X[i].method()` (data-loss anti-pattern);\n"
            "    ast.Call for aggregate `fn(...).method()` (legitimate).\n"
            "    \"\"\"\n"
            "    import ast as _ast\n"
            "    import textwrap as _tw\n"
            "    try:\n"
            "        tree = _ast.parse(_tw.dedent(src))\n"
            "    except SyntaxError as e:\n"
            "        raise AssertionError(\n"
            "            f\"assert_no_call_in_source({fn_name}): parse error {e}\"\n"
            "        ) from None\n"
            "    calls = _collect_call_linenos(tree, receiver_type=receiver_type)\n"
        ),
        label="T2.2 assert",
    )
    print("  OK")


# ============ Tests ============

F12_6_TESTS = [
    "",
    "",
    "# ============ v0.9.14 (F12-6): bool→float + clear error on mismatch ============",
    "",
    "",
    "def test_apply_mixed_bool_float_widens(tmp_path):",
    "    \"\"\"F12-6a: first group bool, later float → widen to float64.",
    "",
    "    Prior: schema inferred bool from first group; 0.5 at flush →",
    "    ArrowInvalid (cryptic). Fix: bool field + non-bool numeric →",
    "    widen to float64. 0.5 lossless; True becomes 1.0.",
    "    \"\"\"",
    "    import pyarrow.parquet as pq",
    "    df = make_df(tmp_path, {",
    "        'g': ['a', 'a', 'b', 'b'],",
    "        'v': [1, 2, 3, 4],",
    "    })",
    "    def fn(g):",
    "        if g['g'].to_list()[0] == 'a':",
    "            return True  # bool (first group, schema=bool_)",
    "        return 2.5     # float (later group, needs widen)",
    "    result = df.groupby('g').apply(fn)",
    "    out = tmp_path / 'out.parquet'",
    "    result.write_parquet(out)",
    "    rows = {r['g']: r['apply_result'] for r in",
    "            pq.read_table(out).to_pylist()}",
    "    assert rows['a'] == 1.0",
    "    assert rows['b'] == 2.5",
    "",
    "",
    "def test_apply_mixed_str_int_clear_error(tmp_path):",
    "    \"\"\"F12-6b: str then int → RuntimeError with hint, not cryptic",
    "    ArrowInvalid from from_pylist.\"\"\"",
    "    import pytest",
    "    df = make_df(tmp_path, {",
    "        'g': ['a', 'a', 'b', 'b'],",
    "        'v': [1, 2, 3, 4],",
    "    })",
    "    def fn(g):",
    "        if g['g'].to_list()[0] == 'a':",
    "            return 'hello'  # str",
    "        return 5            # int",
    "    with pytest.raises(RuntimeError, match='inconsistent types'):",
    "        df.groupby('g').apply(fn).write_parquet(tmp_path / 'o.parquet')",
    "",
    "",
    "def test_apply_all_bool_stays_bool(tmp_path):",
    "    \"\"\"F12-6a regression: all-bool → schema stays bool (no widen).\"\"\"",
    "    import pyarrow as pa",
    "    import pyarrow.parquet as pq",
    "    df = make_df(tmp_path, {",
    "        'g': ['a', 'a', 'b', 'b'],",
    "        'v': [1, 2, 3, 4],",
    "    })",
    "    def fn(g):",
    "        return True",
    "    result = df.groupby('g').apply(fn)",
    "    out = tmp_path / 'out.parquet'",
    "    result.write_parquet(out)",
    "    schema = pq.read_schema(out)",
    "    assert schema.field('apply_result').type == pa.bool_()",
]


def add_f12_6_tests():
    print("\n=== F12-6 tests ===")
    p = Path("tests/test_apply.py")
    s = p.read_text()
    if "test_apply_mixed_bool_float_widens" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(F12_6_TESTS))
    pu.assert_compiles(p)
    print("  OK")


T2_2_TESTS = [
    "",
    "",
    "# ============ v0.9.14 (T2.2): assert_no_call_in_source receiver_type ============",
    "",
    "",
    "def test_assert_no_call_in_source_receiver_subscript():",
    "    \"\"\"T2.2: ast.Subscript receiver filter catches X[i].as_py() only.\"\"\"",
    "    import ast",
    "    from cadw.patch_utils import assert_no_call_in_source",
    "    # Per-row: X[i].as_py() — Subscript receiver → should match",
    "    src_per_row = 'def f(x, i):\\n    return x[i].as_py()\\n'",
    "    with pytest.raises(AssertionError, match='as_py'):",
    "        assert_no_call_in_source(",
    "            src_per_row, 'as_py', receiver_type=ast.Subscript,",
    "        )",
    "    # Aggregate: pc.sum(x).as_py() — Call receiver → should NOT match",
    "    src_aggregate = 'def f(x):\\n    return pc.sum(x).as_py()\\n'",
    "    assert_no_call_in_source(  # no raise",
    "        src_aggregate, 'as_py', receiver_type=ast.Subscript,",
    "    )",
    "",
    "",
    "def test_assert_no_call_in_source_receiver_none_unchanged():",
    "    \"\"\"Backward compat: receiver_type=None matches any receiver.\"\"\"",
    "    from cadw.patch_utils import assert_no_call_in_source",
    "    src = 'def f(x):\\n    return pc.sum(x).as_py()\\n'",
    "    with pytest.raises(AssertionError, match='as_py'):",
    "        assert_no_call_in_source(src, 'as_py')",
]


def add_t2_2_tests():
    print("\n=== T2.2 tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "test_assert_no_call_in_source_receiver_subscript" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(T2_2_TESTS))
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.14 + regression ===")
    pu.verify_anchors([
        # v0.9.14
        ("cadw/dataframe.py", "v0.9.14 (F12-6a)"),
        ("cadw/dataframe.py", "v0.9.14 (F12-6b)"),
        ("cadw/dataframe.py", "§11.5, v0.9.14 F12-7"),
        ("pyproject.toml", 'version = "0.9.14"'),
        ("cadw/patch_utils.py", "receiver_type=None"),
        ("cadw/patch_utils.py", "v0.9.14 (T2.2, §13.4/217)"),
        ("tests/test_apply.py", "test_apply_mixed_bool_float_widens"),
        ("tests/test_apply.py", "test_apply_mixed_str_int_clear_error"),
        ("tests/test_patch_utils.py",
         "test_assert_no_call_in_source_receiver_subscript"),
        # v0.9.13 regression
        ("cadw/ops/quantile.py", "v0.9.13 (F12-5, S3 perf)"),
        ("cadw/dataset/row_filter.py", "v0.9.13 (F12-8)"),
        ("tests/test_quantile.py",
         "test_no_shadowed_module_import_in_tests"),
        # v0.9.12 regression
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("cadw/ops/groupby.py", "v0.9.12 (F12-4, PF1)"),
        ("cadw/ops/topk.py", "v0.9.12 (F12-1)"),
        ("cadw/patch_utils.py", "newline=\"\""),
        # v0.9.11/10/9 regression
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
    ])
    print("  all anchors present")


# ============ dogfood (uses new T2.2 helper) ============

def dogfood():
    print("\n=== dogfood: v0.9.14 via subprocess ===")

    def _verify_all():
        import ast as _ast
        import inspect
        import textwrap as _tw
        import pyarrow as pa

        # F12-6a: _maybe_upgrade_schema has bool widen branch
        from cadw.dataframe import CadwDataFrame
        src_pga = _tw.dedent(inspect.getsource(CadwDataFrame._per_group_apply))
        assert "v0.9.14 (F12-6a)" in src_pga
        assert "is_boolean(field.type)" in src_pga
        # F12-6b: clear RuntimeError in flush guard
        assert "v0.9.14 (F12-6b)" in src_pga
        assert "inconsistent types across groups" in src_pga
        # F12-7: §11.5 note present
        assert "§11.5, v0.9.14 F12-7" in src_pga

        # T2.2: helper extended
        from cadw.patch_utils import assert_no_call_in_source, _collect_call_linenos
        # New: receiver_type filter
        src_per_row = "def f(x, i):\n    return x[i].as_py()\n"
        try:
            assert_no_call_in_source(
                src_per_row, "as_py", receiver_type=_ast.Subscript,
            )
            raise AssertionError("T2.2: per-row as_py not flagged")
        except AssertionError as e:
            if "T2.2: per-row" in str(e):
                raise
        # Aggregate NOT flagged by Subscript filter
        src_agg = "def f(x):\n    return pc.sum(x).as_py()\n"
        assert_no_call_in_source(
            src_agg, "as_py", receiver_type=_ast.Subscript,
        )

        # Behavior test F12-6a: mixed bool/float works end-to-end
        import tempfile
        import pyarrow.parquet as pq
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG
        with tempfile.TemporaryDirectory() as td:
            td = _tw  # noqa
            import pathlib as _pl
            td = _pl.Path(td) if False else _pl.Path(tempfile.gettempdir()) / "cadw_v0914_dogfood"
            td.mkdir(exist_ok=True)
            t = pa.table({"g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
            src = td / "src.parquet"
            pq.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=50),
                spill_dir=td / "spill",
            )
            def fn(g):
                return True if g["g"].to_list()[0] == "a" else 2.5
            result = df.groupby("g").apply(fn)
            out = td / "out.parquet"
            result.write_parquet(out)
            rows = {r["g"]: r["apply_result"] for r in pq.read_table(out).to_pylist()}
            assert rows["a"] == 1.0
            assert rows["b"] == 2.5

        # U3: pyproject version
        from pathlib import Path as _P
        pj = _P("pyproject.toml").read_text()
        assert 'version = "0.9.14"' in pj

        print("  F12-6a bool→float widen OK")
        print("  F12-6b clear RuntimeError on str+int OK")
        print("  F12-7 §11.5 note OK")
        print("  T2.2 receiver_type filter (per-row vs aggregate) OK")
        print("  U3 pyproject version 0.9.14 OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_apply.py",
         "tests/test_apply_e2e.py",
         "tests/test_patch_utils.py", "-q"],
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
    print("=== v0.9.14 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    f12_6a_bool_widen()
    f12_6b_flush_guard()
    f12_7_doc()
    u3_pyproject()
    t2_2_helper()
    add_f12_6_tests()
    add_t2_2_tests()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.14: final small fixes — F12-6 (bool→float widen + "
        "clear error on non-widenable mismatch), F12-7 (§11.5 "
        "empty-input mode note), U3 (pyproject version 0.1.0→0.9.14), "
        "T2.2 (assert_no_call_in_source receiver_type kwarg). "
        "F12-6a (dataframe.py, S3): _maybe_upgrade_schema handled "
        "int→float64 (v0.9.10 C1) but NOT bool→float64. fn returning "
        "True (bool schema) then 0.5 → ArrowInvalid at flush. Fix: "
        "extend to widen bool→float64 on any non-bool numeric. "
        "F12-6b (dataframe.py, S3): non-widenable type mismatch "
        "(e.g. str field + numeric later) → try/except around "
        "pa.Table.from_pylist → clear RuntimeError with hint "
        "('return float(x) if numeric types might mix'). Prior: "
        "cryptic ArrowInvalid. F12-7 (dataframe.py, doc): "
        "_per_group_apply empty-input produces flat [group_cols..., "
        "result_col: float64] — accurate only for scalar mode. For "
        "dict/df mode with empty input, schema is not accurate "
        "(mode unknown at empty-input time). §11.5 documented; "
        "workaround: synthesize one row. U3 (pyproject.toml): "
        "version 0.1.0→0.9.14 (drifted 3+ months from CHANGELOG). "
        "T2.2 (patch_utils.py, tooling): assert_no_call_in_source "
        "gets optional receiver_type kwarg. When set (ast.Subscript), "
        "only `X[i].method()` matches (per-row anti-pattern); "
        "`fn(...).method()` (aggregate, legitimate) does not. F12-5 "
        "dogfood in v0.9.13 needed this — built inline AST walk. "
        "§13.4/217 (precise AST check) now tooling-supported. "
        "3 F12-6 tests: mixed bool/float widen, mixed str/int clear "
        "error, all-bool stays bool. 2 T2.2 tests: Subscript receiver "
        "match vs Call receiver non-match, backward-compat default. "
        "v0.9.14 COMPLETE: 14 patches this session (v0.9.8a–v0.9.14) "
        "fixed R1/R2/B1 audit + N1-N5 + B2-B4 + C1 + PF4 + F12-1..8 "
        "+ §13.4/211-220. Deferred: A1 (dataframe.py split 5404 "
        "lines), T2.5/T2.6/T2.7 (tooling), N6 (Dataset.iter_batches "
        "guard, user responsibility), R3 (planner spill stub, v0.2 "
        "backlog). Next: Pingis formal (fresh snapshot + bench "
        "TASK2/TASK3-style) OR A1 split — decision after v0.9.14.",
        self_path="scripts/patch_v0914.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.14 applied.")


if __name__ == "__main__":
    main()
