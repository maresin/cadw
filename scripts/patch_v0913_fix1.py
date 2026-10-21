#!/usr/bin/env python3
# v0.9.13-fix1 — direct submodule import for shadowed names.
# v0.9.13 COMPLETE.
#
# v0.9.13 dogfood died:
#   from cadw.ops import quantile as q_mod
#   q_mod._stream_values_at_positions
#   → AttributeError: 'function' object has no attribute
#     '_stream_values_at_positions'
#
# Root: cadw/ops/__init__.py re-exports FUNCTION `quantile`:
#   from .quantile import quantile
# This sets cadw.ops.quantile attribute = function, shadowing the
# submodule. `from cadw.ops import quantile as q_mod` gets the
# FUNCTION (Python prefers package attr over submodule in `from pkg
# import name`). Then q_mod._stream_values_at_positions → AttributeError.
#
# Same bug also in test_quantile_arrow_native_no_per_value_loop
# (added v0.9.13) — would fail on first target run.
#
# §13.4/220 (new): when pkg/__init__.py re-exports a FUNCTION with
# the same name as a SUBMODULE (cadw/ops/__init__.py: `reduce` and
# `quantile`), `from pkg import name` resolves to the FUNCTION. Any
# test/dogfood that needs the MODULE (inspect.getsource, ast.walk,
# AttributeError on `.member`) MUST use DIRECT submodule import:
# `from pkg.name import symbol` (or importlib.import_module).
#
# In CADW: only `reduce` and `quantile` are shadowed (functions in
# cadw/ops/__init__.py with matching submodule name). `sort`,
# `groupby`, `shuffle`, `join`, `dedup`, `rank`, `resample` — no
# same-name function → `from cadw.ops import X` gets module.
#
# Fix1:
#   (a) tests/test_quantile.py::test_quantile_arrow_native_no_per_value_loop
#       → direct submodule import
#   (b) scripts/patch_v0913.py::dogfood → direct submodule import
#       (historical hygiene: idempotent re-run succeeds)
#   (c) fix1's own dogfood → direct submodule import
#   (d) meta-guard test: no `from cadw.ops import <shadowed> as <alias>`
#       in test files (SHADOWED = {'reduce', 'quantile'})
#
# Run: uv run python scripts/patch_v0913_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) fix test_quantile.py import ============

def fix_test_import():
    print("\n=== (a) test_quantile.py: direct submodule import ===")
    pu.apply_or_fail(
        "tests/test_quantile.py",
        (
            "    import ast as _ast\n"
            "    import inspect as _inspect\n"
            "    import textwrap as _tw\n"
            "    from cadw.ops import quantile as q_mod\n"
            "    src = _tw.dedent(_inspect.getsource(q_mod._stream_values_at_positions))\n"
        ),
        (
            "    import ast as _ast\n"
            "    import inspect as _inspect\n"
            "    import textwrap as _tw\n"
            "    # §13.4/220: direct submodule import. `from cadw.ops import\n"
            "    # quantile` gets the FUNCTION (re-exported in\n"
            "    # cadw/ops/__init__.py), shadowing the submodule attribute\n"
            "    # → AttributeError on `.member` access.\n"
            "    from cadw.ops.quantile import _stream_values_at_positions\n"
            "    src = _tw.dedent(_inspect.getsource(_stream_values_at_positions))\n"
        ),
        label="test import",
    )
    print("  OK")


# ============ (b) fix scripts/patch_v0913.py dogfood ============

def fix_scripts_v0913():
    print("\n=== (b) scripts/patch_v0913.py: direct submodule import ===")
    pu.apply_or_fail(
        "scripts/patch_v0913.py",
        (
            "        # F12-5: no `for v in vals`, take() present\n"
            "        from cadw.ops import quantile as q_mod\n"
            "        src_q = _tw.dedent(inspect.getsource(q_mod._stream_values_at_positions))\n"
        ),
        (
            "        # F12-5: no `for v in vals`, take() present\n"
            "        # §13.4/220: direct submodule import (shadowed name).\n"
            "        from cadw.ops.quantile import _stream_values_at_positions\n"
            "        src_q = _tw.dedent(inspect.getsource(_stream_values_at_positions))\n"
        ),
        label="scripts/patch_v0913.py",
    )
    print("  OK")


# ============ (d) meta-guard test ============

META_GUARD_TEST = [
    "",
    "",
    "# ============ v0.9.13-fix1 (§13.4/220): shadowed module imports ============",
    "",
    "",
    "def test_no_shadowed_module_import_in_tests():",
    "    \"\"\"§13.4/220: `from cadw.ops import <shadowed> as <alias>`",
    "    where <shadowed> is a FUNCTION re-exported in",
    "    cadw/ops/__init__.py that shadows its submodule.",
    "",
    "    Only `reduce` and `quantile` are shadowed. `from cadw.ops",
    "    import quantile as q_mod` gets the FUNCTION, not module —",
    "    any `.member` access → AttributeError. Use",
    "    `from cadw.ops.<name> import symbol` for module access.",
    "    \"\"\"",
    "    import ast as _ast",
    "    from pathlib import Path as _P",
    "    SHADOWED = {'reduce', 'quantile'}",
    "    tests_root = _P(__file__).parent",
    "    violations: list[str] = []",
    "    for py in tests_root.glob('test_*.py'):",
    "        if py.name == _P(__file__).name:",
    "            continue",
    "        try:",
    "            tree = _ast.parse(py.read_text())",
    "        except SyntaxError:",
    "            continue",
    "        for node in _ast.walk(tree):",
    "            if not isinstance(node, _ast.ImportFrom):",
    "                continue",
    "            if node.module != 'cadw.ops':",
    "                continue",
    "            for alias in node.names:",
    "                if alias.name in SHADOWED and alias.asname:",
    "                    violations.append(",
    "                        f'{py.name}:{node.lineno}: '",
    "                        f'from cadw.ops import {alias.name} as '",
    "                        f'{alias.asname}'",
    "                    )",
    "    assert not violations, (",
    "        '§13.4/220: shadowed module import — cadw.ops.__init__ '",
    "        're-exports these as FUNCTIONS, shadowing submodules:'",
    "        + chr(10) + chr(10).join(violations)",
    "        + chr(10) + 'Use `from cadw.ops.<name> import <symbol>` '",
    "        + 'for module access.'",
    "    )",
]


def add_meta_guard():
    print("\n=== (d) meta-guard test ===")
    p = Path("tests/test_quantile.py")
    s = p.read_text()
    if "def test_no_shadowed_module_import_in_tests" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(META_GUARD_TEST))
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.13-fix1 + regression ===")
    pu.verify_anchors([
        # fix1a
        ("tests/test_quantile.py",
         "from cadw.ops.quantile import _stream_values_at_positions"),
        ("tests/test_quantile.py", "§13.4/220: direct submodule import"),
        ("tests/test_quantile.py", "test_no_shadowed_module_import_in_tests"),
        # fix1b
        ("scripts/patch_v0913.py",
         "from cadw.ops.quantile import _stream_values_at_positions"),
        # v0.9.13 features
        ("cadw/ops/quantile.py", "v0.9.13 (F12-5, S3 perf)"),
        ("cadw/ops/quantile.py", "_pc.take("),
        ("cadw/dataset/row_filter.py", "v0.9.13 (F12-8)"),
        ("cadw/dataset/row_filter.py", "Asymmetry is by design"),
        # v0.9.12 regression
        ("cadw/ops/topk.py", "v0.9.12 (F12-1)"),
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("cadw/ops/groupby.py", "v0.9.12 (F12-4, PF1)"),
        ("cadw/patch_utils.py", "newline=\"\""),
        ("cadw/patch_utils.py", "v0.9.12 (F12-3, §13.4/211)"),
        # v0.9.11/10/9 regression
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/dataframe.py", "post-flush widen needed"),
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.13-fix1 via subprocess ===")

    def _verify_all():
        import ast as _ast
        import inspect
        import textwrap as _tw
        import pyarrow as pa
        import pyarrow.compute as pc

        # (a) + (c): direct submodule import (shadowed name avoided)
        from cadw.ops.quantile import _stream_values_at_positions
        src_q = _tw.dedent(inspect.getsource(_stream_values_at_positions))
        tree = _ast.parse(src_q)

        per_value_loops = [
            n.lineno for n in _ast.walk(tree)
            if isinstance(n, _ast.For)
            and isinstance(n.iter, _ast.Name)
            and n.iter.id == "vals"
        ]
        assert not per_value_loops, (
            f"F12-5: per-value loop at {per_value_loops}"
        )
        take_calls = [
            n for n in _ast.walk(tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "take"
        ]
        assert take_calls, "F12-5: pc.take not called"

        # Meta-guard behavior: assert no shadowed-import in test_quantile.py
        from pathlib import Path as _P
        tq_src = _P("tests/test_quantile.py").read_text()
        tq_tree = _ast.parse(tq_src)
        violations = []
        for n in _ast.walk(tq_tree):
            if not isinstance(n, _ast.ImportFrom):
                continue
            if n.module != "cadw.ops":
                continue
            for alias in n.names:
                if alias.name in {"reduce", "quantile"} and alias.asname:
                    violations.append(
                        f"line {n.lineno}: {alias.name} as {alias.asname}"
                    )
        assert not violations, (
            f"§13.4/220: shadowed import still present in test_quantile.py: "
            f"{violations}"
        )

        # F12-8 docstring
        from cadw.dataset.row_filter import PredicateFilter
        doc = PredicateFilter.mask_for_batch.__doc__ or ""
        assert "Kleene" in doc
        assert "Asymmetry" in doc or "asymmetric" in doc.lower()

        # Functional sanity: nullable filter + take
        t = pa.table({"v": pa.array([1.0, None, 2.0, None, 3.0])})
        arr = t.column("v")
        non_null = arr.filter(pc.is_valid(arr))
        selected = pc.take(non_null, pa.array([0, 2], type=pa.int64()))
        assert selected.to_pylist() == [1.0, 3.0]

        print("  (a) direct submodule import (test_quantile.py) OK")
        print("  (b) scripts/patch_v0913.py idempotent path fixed")
        print("  (d) meta-guard passes (no shadowed import in tests)")
        print("  F12-5 take, F12-8 Kleene doc — unchanged, still OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_quantile.py",
         "tests/test_row_filter.py",
         "tests/test_dataframe_filters.py",
         "tests/test_arrow_call_memory_guard.py", "-q"],
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
    print("=== v0.9.13-fix1 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_test_import()
    fix_scripts_v0913()
    add_meta_guard()
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.13-fix1: complete v0.9.13 (direct submodule import for "
        "shadowed names). v0.9.13 dogfood died: "
        "`from cadw.ops import quantile as q_mod; "
        "q_mod._stream_values_at_positions` → AttributeError "
        "('function' object has no attribute). Root: "
        "cadw/ops/__init__.py re-exports FUNCTION `quantile` "
        "(`from .quantile import quantile`), which sets package "
        "attribute to function, shadowing submodule. Python "
        "`from pkg import name` prefers pkg attribute over "
        "sys.modules[pkg.name] → gets function, not module. "
        "Same bug also in test_quantile_arrow_native_no_per_value_loop "
        "(added v0.9.13) → would fail on target run. §13.4/220 "
        "(new): when pkg/__init__.py re-exports a FUNCTION with the "
        "same name as a SUBMODULE (cadw: `reduce`, `quantile`), any "
        "test/dogfood that needs the MODULE (inspect.getsource, "
        "ast.walk, AttributeError on `.member`) MUST use DIRECT "
        "submodule import: `from pkg.name import symbol` (or "
        "importlib.import_module). Other names in cadw/ops/__init__ "
        "(`sort`, `groupby`, `shuffle`, `join`, `dedup`, `rank`, "
        "`resample`) have no same-name function → "
        "`from cadw.ops import X` gets module. "
        "Fix1: (a) tests/test_quantile.py::test_quantile_arrow_"
        "native_no_per_value_loop → direct submodule import; "
        "(b) scripts/patch_v0913.py::dogfood → direct submodule "
        "import (idempotent re-run succeeds); (c) fix1's own dogfood "
        "→ direct submodule import; (d) meta-guard "
        "test_no_shadowed_module_import_in_tests (AST walk on all "
        "tests/test_*.py, flag `from cadw.ops import <shadowed> as "
        "<alias>` where shadowed in {'reduce', 'quantile'}). "
        "v0.9.13 COMPLETE: F12-5 quantile Arrow-native stream "
        "(~100-400x for high quantiles); F12-8 PredicateFilter "
        "null semantics docstring (Kleene-drop numeric vs "
        "fill_null(False) string, intentional asymmetry); fix1 "
        "shadowed-module-import fix + meta-guard. Deferred: A1 "
        "(dataframe split 5404 lines), T2.2/T2.5/T2.6/T2.7/T2.8 "
        "(tooling), U3 (pyproject version), N6, R3. Next: Pingis "
        "formal (fresh snapshot + bench) for v0.9.4-v0.9.13 audit.",
        self_path="scripts/patch_v0913_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.13-fix1 applied. v0.9.13 COMPLETE.")


if __name__ == "__main__":
    main()
