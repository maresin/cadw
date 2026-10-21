#!/usr/bin/env python3
# v0.9.14-fix3 — bool before int in _infer_arrow_type_from_value.
# v0.9.14 COMPLETE.
#
# v0.9.14-fix2 dogfood green, pytest failed
# test_apply_all_bool_stays_bool:
#   pyarrow.lib.ArrowTypeError: Expected integer, got bool
#   at pa.Table.from_pylist(buf_rows, schema=int64) in
#   _per_group_apply._flush_buf.
#
# Root: _infer_arrow_type_from_value checks int BEFORE bool:
#   if isinstance(v, (np.integer, int)): return pa.int64()
#   if isinstance(v, (np.floating, float)): return pa.float64()
#   if isinstance(v, (np.bool_, bool)): return pa.bool_()  # unreachable
# Python bool is int subclass — isinstance(True, int) = True. So
# _infer_arrow_type_from_value(True) = int64. Then from_pylist
# ([{'x': True}], schema=int64) → ArrowTypeError (Arrow does not
# silently coerce bool→int).
#
# F12-6a (v0.9.14) bool widen branch in _maybe_upgrade_schema was
# DEAD CODE: schema never had pa.bool_ because _infer gave int64.
# Mixed bool/float "worked" coincidentally: True → int64 schema,
# 2.5 float → int branch widen → float64. But all-bool → no widen
# → ArrowTypeError at flush.
#
# §13.4/223 (new): `isinstance(x, int)` includes `bool` (Python
# bool is int subclass). Type-inference chains / dispatchers /
# validators MUST check bool BEFORE int. Same trap: np.bool_ is
# NOT subclass of np.integer (modern numpy), but Python bool IS
# subclass of int — order matters for both isinstance calls.
#
# State on disk (from v0.9.14-fix2, all applied):
#   F12-6a/b, F12-7, T2.2, T2.9 (validate=), U3 — all applied
#   F12-6 tests (3), T2.2 tests (2) — applied
#   test_apply_all_bool_stays_bool — FAILS (this bug)
#   commit — never reached
#
# Fix3:
#   (a) _infer_arrow_type_from_value: bool check BEFORE int, float.
#   (b) Test updates: docstring note §13.4/223; add
#       test_apply_mixed_bool_int_widens (True then 5 → float64).
#   (c) test_infer_arrow_type_from_value_unit — direct unit
#       (True → bool_, 5 → int64, 5.5 → float64, "x" → string).
#   (d) verify + dogfood + pytest + commit.
#
# Run: uv run python scripts/patch_v0914_fix3.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) _infer_arrow_type_from_value reorder ============

def fix_infer_reorder():
    print("\n=== (a) _infer_arrow_type_from_value: bool before int ===")
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "def _infer_arrow_type_from_value(v):\n"
            "    import pyarrow as pa\n"
            "    import numpy as np\n"
            "\n"
            "    if v is None:\n"
            "        return pa.float64()\n"
            "    if isinstance(v, (np.integer, int)):\n"
            "        return pa.int64()\n"
            "    if isinstance(v, (np.floating, float)):\n"
            "        return pa.float64()\n"
            "    if isinstance(v, (np.bool_, bool)):\n"
            "        return pa.bool_()\n"
            "    if isinstance(v, str):\n"
            "        return pa.string()\n"
            "    return pa.float64()\n"
        ),
        (
            "def _infer_arrow_type_from_value(v):\n"
            "    \"\"\"Map a Python scalar to Arrow type (schema inference).\n"
            "\n"
            "    v0.9.14-fix3 (§13.4/223): bool check BEFORE int. Python\n"
            "    bool is int subclass — `isinstance(True, int)` = True,\n"
            "    so int branch captured True → schema int64. Then\n"
            "    `pa.Table.from_pylist([{'x': True}], schema=int64)` →\n"
            "    ArrowTypeError (Expected integer, got bool). Prior:\n"
            "    F12-6a bool widen branch in _maybe_upgrade_schema was\n"
            "    DEAD CODE — schema never had pa.bool_ from _infer.\n"
            "    np.bool_ is not subclass of int/np.integer; for safety\n"
            "    check both bool and np.bool_ first.\n"
            "    \"\"\"\n"
            "    import pyarrow as pa\n"
            "    import numpy as np\n"
            "\n"
            "    if v is None:\n"
            "        return pa.float64()\n"
            "    # §13.4/223: bool BEFORE int (bool is int subclass).\n"
            "    if isinstance(v, (bool, np.bool_)):\n"
            "        return pa.bool_()\n"
            "    if isinstance(v, (np.integer, int)):\n"
            "        return pa.int64()\n"
            "    if isinstance(v, (np.floating, float)):\n"
            "        return pa.float64()\n"
            "    if isinstance(v, str):\n"
            "        return pa.string()\n"
            "    return pa.float64()\n"
        ),
        label="infer reorder",
    )
    print("  OK")


# ============ (b) Update existing test docstring + add bool/int test ============

def update_bool_test_docstring():
    print("\n=== (b) test_apply_all_bool_stays_bool docstring + mixed bool/int test ===")
    pu.apply_or_fail(
        "tests/test_apply.py",
        (
            'def test_apply_all_bool_stays_bool(tmp_path):\n'
            '    """F12-6a regression: all-bool → schema stays bool (no widen)."""\n'
        ),
        (
            'def test_apply_all_bool_stays_bool(tmp_path):\n'
            '    """F12-6a + v0.9.14-fix3 (§13.4/223): all-bool → schema stays bool.\n'
            '\n'
            '    Prior bug: _infer_arrow_type_from_value checked int BEFORE\n'
            '    bool (Python bool is int subclass) → True inferred as int64.\n'
            '    Then from_pylist([{\'x\': True}], schema=int64) →\n'
            '    ArrowTypeError. F12-6a bool widen branch was DEAD CODE.\n'
            '    Fix3: bool first → schema pa.bool_. No widen (all-bool).\n'
            '    """\n'
        ),
        label="test docstring",
    )
    print("  OK")


NEW_TESTS = [
    "",
    "",
    "def test_apply_mixed_bool_int_widens_to_float(tmp_path):",
    "    \"\"\"F12-6a + v0.9.14-fix3: bool first, then int → widen to float64.",
    "",
    "    Schema=bool (True first). buf_rows has True + 5 (int, not bool).",
    "    _maybe_upgrade_schema bool branch: non-bool numeric → widen",
    "    to float64. True → 1.0, 5 → 5.0. Lossless for |v| < 2^53.",
    "    \"\"\"",
    "    import pyarrow.parquet as pq",
    "    df = make_df(tmp_path, {",
    "        'g': ['a', 'a', 'b', 'b'],",
    "        'v': [1, 2, 3, 4],",
    "    })",
    "    def fn(g):",
    "        if g['g'].to_list()[0] == 'a':",
    "            return True  # bool (first group, schema=bool_)",
    "        return 5        # int (later group, non-bool numeric)",
    "    result = df.groupby('g').apply(fn)",
    "    out = tmp_path / 'out.parquet'",
    "    result.write_parquet(out)",
    "    rows = {r['g']: r['apply_result'] for r in",
    "            pq.read_table(out).to_pylist()}",
    "    assert rows['a'] == 1.0",
    "    assert rows['b'] == 5.0",
    "",
    "",
    "def test_infer_arrow_type_from_value_unit():",
    "    \"\"\"v0.9.14-fix3 (§13.4/223): direct unit test for type inference.",
    "",
    "    bool BEFORE int (Python bool is int subclass). Catches the exact",
    "    order bug even without going through apply pipeline.",
    "    \"\"\"",
    "    import pyarrow as pa",
    "    from cadw.dataframe import _infer_arrow_type_from_value",
    "    # bool → bool_ (must be first; not int64)",
    "    assert _infer_arrow_type_from_value(True) == pa.bool_()",
    "    assert _infer_arrow_type_from_value(False) == pa.bool_()",
    "    # int (not bool) → int64",
    "    assert _infer_arrow_type_from_value(5) == pa.int64()",
    "    assert _infer_arrow_type_from_value(0) == pa.int64()",
    "    # float → float64",
    "    assert _infer_arrow_type_from_value(5.5) == pa.float64()",
    "    # str → string",
    "    assert _infer_arrow_type_from_value('x') == pa.string()",
    "    # None → float64 (None widens to float default)",
    "    assert _infer_arrow_type_from_value(None) == pa.float64()",
    "",
    "",
    "def test_infer_arrow_type_from_value_numpy_bool():",
    "    \"\"\"v0.9.14-fix3: np.bool_ also → bool_ (defensive; not int).",
    "",
    "    np.bool_ is not a subclass of int/np.integer in modern numpy,",
    "    but explicit check catches np.bool_(True) → bool_ (not int64).",
    "    \"\"\"",
    "    import numpy as np",
    "    import pyarrow as pa",
    "    from cadw.dataframe import _infer_arrow_type_from_value",
    "    assert _infer_arrow_type_from_value(np.bool_(True)) == pa.bool_()",
    "    assert _infer_arrow_type_from_value(np.bool_(False)) == pa.bool_()",
]


def add_new_tests():
    print("\n=== (b) new tests ===")
    p = Path("tests/test_apply.py")
    s = p.read_text()
    if "test_infer_arrow_type_from_value_unit" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(NEW_TESTS))
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.14-fix3 + regression ===")
    pu.verify_anchors([
        # fix3
        ("cadw/dataframe.py", "v0.9.14-fix3 (§13.4/223)"),
        ("cadw/dataframe.py",
         "§13.4/223: bool BEFORE int (bool is int subclass)"),
        ("tests/test_apply.py", "test_apply_mixed_bool_int_widens_to_float"),
        ("tests/test_apply.py", "test_infer_arrow_type_from_value_unit"),
        ("tests/test_apply.py", "test_infer_arrow_type_from_value_numpy_bool"),
        # fix2 regression (T2.2)
        ("cadw/patch_utils.py", "v0.9.14 (T2.2, §13.4/217)"),
        ("cadw/patch_utils.py", "receiver_type=None"),
        ("tests/test_apply.py", "test_apply_mixed_bool_float_widens"),
        ("tests/test_apply.py", "test_apply_mixed_str_int_clear_error"),
        ("tests/test_apply.py", "test_apply_all_bool_stays_bool"),
        ("tests/test_patch_utils.py",
         "test_assert_no_call_in_source_receiver_subscript"),
        # fix1 regression (T2.9)
        ("cadw/patch_utils.py", "v0.9.14-fix1 (T2.9, §13.4/221)"),
        ("cadw/patch_utils.py", 'elif validate == "toml":'),
        ("pyproject.toml", 'version = "0.9.14"'),
        # v0.9.14 partial (F12-6a/b, F12-7)
        ("cadw/dataframe.py", "v0.9.14 (F12-6a)"),
        ("cadw/dataframe.py", "v0.9.14 (F12-6b)"),
        ("cadw/dataframe.py", "§11.5, v0.9.14 F12-7"),
        # v0.9.13/12/11/10/9 regression
        ("cadw/ops/quantile.py", "v0.9.13 (F12-5, S3 perf)"),
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/patch_utils.py", "def apply_or_fail("),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.14-fix3 via subprocess ===")

    def _verify_all():
        import ast as _ast
        import inspect
        import textwrap as _tw
        import pyarrow as pa

        # Unit: type inference order
        from cadw.dataframe import _infer_arrow_type_from_value
        assert _infer_arrow_type_from_value(True) == pa.bool_()
        assert _infer_arrow_type_from_value(False) == pa.bool_()
        assert _infer_arrow_type_from_value(5) == pa.int64()
        assert _infer_arrow_type_from_value(5.5) == pa.float64()
        assert _infer_arrow_type_from_value("x") == pa.string()

        # AST: bool check before int in _infer source
        src = _tw.dedent(inspect.getsource(_infer_arrow_type_from_value))
        tree = _ast.parse(src)
        # find isinstance calls with tuple containing 'bool'
        bool_isinstance_line = None
        int_isinstance_line = None
        for n in _ast.walk(tree):
            if not isinstance(n, _ast.Call):
                continue
            f = n.func
            if not (isinstance(f, _ast.Name) and f.id == "isinstance"):
                continue
            if len(n.args) < 2:
                continue
            arg2 = n.args[1]
            names = set()
            for elt in _ast.walk(arg2):
                if isinstance(elt, _ast.Name):
                    names.add(elt.id)
            if "bool" in names or "np" in names:
                # distinguish bool vs int
                if "bool" in names and bool_isinstance_line is None:
                    bool_isinstance_line = n.lineno
                if ("int" in names) and int_isinstance_line is None:
                    int_isinstance_line = n.lineno
        assert bool_isinstance_line is not None, "no isinstance(., .bool.) call"
        assert int_isinstance_line is not None, "no isinstance(., .int.) call"
        assert bool_isinstance_line < int_isinstance_line, (
            f"§13.4/223: bool check (line {bool_isinstance_line}) "
            f"must precede int check (line {int_isinstance_line})"
        )

        # End-to-end: all-bool → schema bool
        import tempfile
        import pyarrow.parquet as pq
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG
        with tempfile.TemporaryDirectory() as td:
            import pathlib as _pl
            td = _pl.Path(td)
            t = pa.table({"g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
            src_p = td / "src.parquet"
            pq.write_table(t, src_p)
            df = _CDF.from_parquet(
                src_p, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=50),
                spill_dir=td / "spill",
            )
            def fn(g):
                return True
            result = df.groupby("g").apply(fn)
            out = td / "out.parquet"
            result.write_parquet(out)
            sch = pq.read_schema(out)
            assert sch.field("apply_result").type == pa.bool_(), (
                f"all-bool: schema {sch.field('apply_result').type}, "
                f"expected bool"
            )
            rows = {r["g"]: r["apply_result"]
                    for r in pq.read_table(out).to_pylist()}
            assert rows["a"] is True
            assert rows["b"] is True

        # End-to-end: mixed bool/float widens to float64
        with tempfile.TemporaryDirectory() as td:
            import pathlib as _pl
            td = _pl.Path(td)
            t = pa.table({"g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
            src_p = td / "src.parquet"
            pq.write_table(t, src_p)
            df = _CDF.from_parquet(
                src_p, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=50),
                spill_dir=td / "spill",
            )
            def fn(g):
                return True if g["g"].to_list()[0] == "a" else 2.5
            result = df.groupby("g").apply(fn)
            out = td / "out.parquet"
            result.write_parquet(out)
            rows = {r["g"]: r["apply_result"]
                    for r in pq.read_table(out).to_pylist()}
            assert rows["a"] == 1.0
            assert rows["b"] == 2.5

        print("  _infer unit (bool→bool_, 5→int64, 5.5→float64) OK")
        print("  AST: bool isinstance precedes int isinstance OK")
        print("  E2E all-bool → schema bool (F12-6a) OK")
        print("  E2E mixed bool/float → float64 (F12-6a) OK")

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
    print("=== v0.9.14-fix3 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_infer_reorder()
    update_bool_test_docstring()
    add_new_tests()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.14-fix3: complete v0.9.14 (bool before int in "
        "_infer_arrow_type_from_value). v0.9.14-fix2 dogfood green, "
        "pytest failed test_apply_all_bool_stays_bool: "
        "pyarrow.lib.ArrowTypeError: Expected integer, got bool at "
        "pa.Table.from_pylist(buf_rows, schema=int64). Root: "
        "_infer_arrow_type_from_value checked int BEFORE bool: "
        "`isinstance(True, int)` = True (Python bool is int "
        "subclass) → True inferred as int64. Then from_pylist("
        "[{'x': True}], schema=int64) → ArrowTypeError (Arrow does "
        "not silently coerce bool→int). F12-6a bool widen branch in "
        "_maybe_upgrade_schema was DEAD CODE — schema never had "
        "pa.bool_. Mixed bool/float 'worked' coincidentally "
        "(True→int64 schema, 2.5 float → int branch widen → "
        "float64); all-bool → no widen → ArrowTypeError. §13.4/223 "
        "(new): `isinstance(x, int)` includes bool (Python bool is "
        "int subclass). Type-inference chains / dispatchers / "
        "validators MUST check bool BEFORE int. Same trap: np.bool_ "
        "is NOT subclass of np.integer (modern numpy) but Python "
        "bool IS subclass of int — order matters. Fix3: (a) reorder "
        "_infer_arrow_type_from_value: None → bool/np.bool_ → int → "
        "float → str; (b) update test_apply_all_bool_stays_bool "
        "docstring (now documents the order bug); (c) add "
        "test_apply_mixed_bool_int_widens (True then 5 → float64); "
        "(d) add test_infer_arrow_type_from_value_unit (True→bool_, "
        "5→int64, 5.5→float64, 'x'→string, None→float64) + "
        "test_infer_arrow_type_from_value_numpy_bool (np.bool_ → "
        "bool_); (e) verify + dogfood (unit, AST order check, E2E "
        "all-bool, E2E mixed bool/float) + pytest + commit. "
        "v0.9.14 COMPLETE: F12-6a/b (bool→float widen, clear "
        "RuntimeError on non-widenable mismatch), F12-7 (§11.5 "
        "empty-input note), T2.2 (receiver_type AST filter), T2.9 "
        "(apply_or_fail validate=toml/json/python/none), U3 "
        "(pyproject 0.1.0→0.9.14), fix3 (§13.4/223 bool-before-int "
        "in _infer_arrow_type_from_value). 16 patches this session "
        "(v0.9.8a–v0.9.14-fix3). Deferred: A1 (dataframe.py split "
        "5404 lines), T2.5/T2.6/T2.7 (scaffold v4 tooling), N6 "
        "(Dataset.iter_batches guard, user responsibility), R3 "
        "(planner spill stub, v0.2 backlog). Next: Pingis formal "
        "(fresh snapshot + bench TASK2/TASK3-style).",
        self_path="scripts/patch_v0914_fix3.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.14-fix3 applied. v0.9.14 COMPLETE.")


if __name__ == "__main__":
    main()
