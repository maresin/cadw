#!/usr/bin/env python3
# v0.9.15-fix2 — scope T2.7 meta-guard to cadw/ only + warnings.
# v0.9.15 COMPLETE.
#
# v0.9.15-fix1 applied the marker (verify_state green), but dogfood/
# pytest failed on T2.7 test_no_inline_dedent_getsource_outside_helper:
# 13 false-positive violations in scripts/patch_*.py + SyntaxWarning
# from _ast.parse on files with \( regex patterns in docstrings.
#
# Root: T2.7 scanned [cadw/, scripts/]. Historical patch-scripts
# (v0.9.6-v0.9.14) predate dedent_method_source helper — they inline
# `_tw.dedent(_inspect.getsource(...))` legitimately. Those scripts
# are ephemeral (ran once, won't run again). Scanning them:
# (a) 13 false-positive violations, (b) SyntaxWarning from
# `_ast.parse` on files with `\(` regex patterns in docstrings.
#
# §13.4/225 (new): static guards for "reinvented helper" patterns
# must scope to **production code** (`cadw/`), not historical
# artifacts (`scripts/`). Ephemeral one-shot scripts that already
# ran are not subject to forward-looking conventions. For future
# patch-scripts, convention enforced by helper docstring +
# code review (not static scan).
#
# Also: `_ast.parse` on unvetted files emits SyntaxWarning for
# invalid escape sequences in string literals (Python 3.12+).
# Wrap in `warnings.catch_warnings() + simplefilter("ignore",
# SyntaxWarning)` for robustness when scanning arbitrary code.
#
# Fix2:
#   (1) T2.7 test: roots = [cadw/] only. Docstring note §13.4/225.
#   (2) T2.7 test: wrap _ast.parse in warnings suppression.
#   (3) _verify_all dogfood: same scope + warnings.
#   (4) verify + dogfood + pytest + commit.
#
# Run: uv run python scripts/patch_v0915_fix2.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (1) T2.7 scope: cadw/ only ============

def fix_t27_scope():
    print("\n=== (1) T2.7 scope: cadw/ only ===")
    pu.apply_or_fail(
        "tests/test_patch_utils.py",
        "    roots = [_P('cadw'), _P('scripts')]\n",
        (
            "    # v0.9.15-fix2 (§13.4/225): cadw/ only. Scripts/\n"
            "    # skipped — historical patch-scripts (v0.9.6-v0.9.14)\n"
            "    # predate dedent_method_source helper; they inline\n"
            "    # _tw.dedent(_inspect.getsource(...)) legitimately\n"
            "    # (ran once, ephemeral). Future patch-scripts use\n"
            "    # helper by convention (docstring + code review).\n"
            "    roots = [_P('cadw')]\n"
        ),
        label="T2.7 scope",
    )
    print("  OK")


# ============ (2) T2.7 test: warnings suppression in _ast.parse ============

def fix_t27_parse():
    print("\n=== (2) T2.7 test: SyntaxWarning suppression ===")
    pu.apply_or_fail(
        "tests/test_patch_utils.py",
        (
            "            try:\n"
            "                src = py.read_text()\n"
            "                tree = _ast.parse(src)\n"
            "            except (SyntaxError, OSError, UnicodeDecodeError):\n"
            "                continue\n"
        ),
        (
            "            # v0.9.15-fix2 (§13.4/225): SyntaxWarning —\n"
            "            # invalid escape sequences in scanned files.\n"
            "            import warnings as _warn\n"
            "            try:\n"
            "                src = py.read_text()\n"
            "                with _warn.catch_warnings():\n"
            "                    _warn.simplefilter('ignore', SyntaxWarning)\n"
            "                    tree = _ast.parse(src)\n"
            "            except (SyntaxError, OSError, UnicodeDecodeError):\n"
            "                continue\n"
        ),
        label="T2.7 parse warnings",
    )
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.15-fix2 + regression ===")
    pu.verify_anchors([
        # fix2
        ("tests/test_patch_utils.py", "v0.9.15-fix2 (§13.4/225): cadw/ only"),
        ("tests/test_patch_utils.py", "roots = [_P('cadw')]"),
        ("tests/test_patch_utils.py",
         "v0.9.15-fix2 (§13.4/225): SyntaxWarning"),
        # fix1 (marker)
        ("cadw/patch_utils.py", "v0.9.15 (T2.6)"),
        # v0.9.15
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
        ("CHANGELOG.md", "### v0.9.7–v0.9.14: audit round 2"),
        ("cadw/__init__.py", '__version__ = "0.9.14"'),
        ("cadw/patch_utils.py", "def dedent_method_source("),
        ("tests/test_patch_utils.py", "test_dedent_method_source_basic"),
        ("tests/test_patch_utils.py",
         "test_no_inline_dedent_getsource_outside_helper"),
        # v0.9.14-fix3 regression
        ("cadw/dataframe.py", "v0.9.14-fix3 (§13.4/223)"),
        ("tests/test_apply.py", "test_apply_all_bool_stays_bool"),
        ("tests/test_apply.py", "test_infer_arrow_type_from_value_unit"),
        # v0.9.14 regression
        ("cadw/dataframe.py", "v0.9.14 (F12-6a)"),
        ("cadw/dataframe.py", "v0.9.14 (F12-6b)"),
        ("cadw/patch_utils.py", "v0.9.14 (T2.2, §13.4/217)"),
        ("cadw/patch_utils.py", "v0.9.14-fix1 (T2.9, §13.4/221)"),
        ("pyproject.toml", 'version = "0.9.14"'),
        # v0.9.13/12/11/10/9 regression
        ("cadw/ops/quantile.py", "v0.9.13 (F12-5, S3 perf)"),
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/patch_utils.py", "def apply_or_fail("),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.15-fix2 via subprocess ===")

    def _verify_all():
        import ast as _ast
        import warnings as _warn
        from cadw.patch_utils import dedent_method_source
        from cadw.dataframe import CadwDataFrame
        from pathlib import Path as _P

        # v0.9.15: __version__
        import cadw
        assert cadw.__version__ == "0.9.14"

        # CHANGELOG section
        cl = _P("CHANGELOG.md").read_text()
        assert "## [0.9.14] — 2026-10-21" in cl
        assert cl.index("## [Unreleased]") < cl.index("## [0.9.14]")
        assert cl.index("## [0.9.14]") < cl.index("## [0.9.6]")

        # T2.6
        dms_doc = dedent_method_source.__doc__ or ""
        assert "v0.9.15 (T2.6)" in dms_doc
        src, tree = dedent_method_source(CadwDataFrame.head)
        assert src.split("\n")[0].startswith("def ")

        # T2.7 (v0.9.15-fix2): cadw/ only, warnings suppressed
        violations = []
        for py in _P("cadw").rglob("*.py"):
            if py.name == "patch_utils.py":
                continue
            try:
                with _warn.catch_warnings():
                    _warn.simplefilter("ignore", SyntaxWarning)
                    t = _ast.parse(py.read_text())
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue
            for n in _ast.walk(t):
                if not isinstance(n, _ast.Call):
                    continue
                f = n.func
                if not (isinstance(f, _ast.Attribute) and f.attr == "dedent"):
                    continue
                if not n.args:
                    continue
                inner = n.args[0]
                if (isinstance(inner, _ast.Call)
                        and isinstance(inner.func, _ast.Attribute)
                        and inner.func.attr == "getsource"):
                    violations.append(f"{py}:{n.lineno}")
        assert not violations, (
            f"§13.4/225: inline dedent(getsource) in cadw/: {violations}"
        )

        # Sanity: T2.7 source scoped to cadw/ only
        t27_src = _P("tests/test_patch_utils.py").read_text()
        assert "roots = [_P('cadw')]" in t27_src
        assert "roots = [_P('cadw'), _P('scripts')]" not in t27_src
        assert "v0.9.15-fix2 (§13.4/225): cadw/ only" in t27_src

        print("  __version__ = '0.9.14' OK")
        print("  CHANGELOG: 0.9.14 section OK")
        print("  T2.6 dedent_method_source (v0.9.15 marker) OK")
        print("  T2.7 cadw/ only (no violations, no scripts/ scan) OK")
        print("  T2.7 SyntaxWarning suppression OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_patch_utils.py", "-q"],
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
    print("=== v0.9.15-fix2 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_t27_scope()
    fix_t27_parse()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.15-fix2: complete v0.9.15 (T2.7 scope cadw/ only + "
        "SyntaxWarning suppression). v0.9.15-fix1 applied marker "
        "(verify_state green), but dogfood/pytest failed on T2.7 "
        "test_no_inline_dedent_getsource_outside_helper: 13 false-"
        "positive violations in scripts/patch_*.py + SyntaxWarning "
        "from _ast.parse on files with \\( regex patterns in "
        "docstrings. Root: T2.7 scanned [cadw/, scripts/]. "
        "Historical patch-scripts (v0.9.6-v0.9.14) predate the "
        "dedent_method_source helper — they inline "
        "_tw.dedent(_inspect.getsource(...)) legitimately. Those "
        "scripts are ephemeral (ran once, won't run again). "
        "Scanning them: (a) 13 false-positive violations, (b) "
        "SyntaxWarning from _ast.parse on files with \\( regex "
        "patterns in docstrings. §13.4/225 (new): static guards "
        "for 'reinvented helper' patterns must scope to production "
        "code (cadw/), not historical artifacts (scripts/). "
        "Ephemeral one-shot scripts that already ran are not "
        "subject to forward-looking conventions. For future "
        "patch-scripts, convention enforced by helper docstring + "
        "code review (not static scan). Also: _ast.parse on "
        "unvetted files emits SyntaxWarning for invalid escape "
        "sequences in string literals (Python 3.12+). Wrap in "
        "warnings.catch_warnings() + simplefilter('ignore', "
        "SyntaxWarning) for robustness. Fix2: (1) T2.7 test: roots "
        "= [_P('cadw')] only (skip scripts/), docstring note; "
        "(2) T2.7 test: wrap _ast.parse in warnings suppression; "
        "(3) _verify_all dogfood: same scope + warnings; "
        "(4) verify (22 anchors) + dogfood + pytest + commit. "
        "v0.9.15 COMPLETE: CHANGELOG 0.9.14 section, "
        "cadw.__version__='0.9.14', T2.6 dedent_method_source "
        "helper, T2.7 meta-guard (cadw/ only). Deferred: "
        "DECISIONS.md §13.4/211-225 append, A1 dataframe.py split, "
        "release tag v0.9.14 (after v0.9.15).",
        self_path="scripts/patch_v0915_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.15-fix2 applied. v0.9.15 COMPLETE.")


if __name__ == "__main__":
    main()
