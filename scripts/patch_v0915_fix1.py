#!/usr/bin/env python3
# v0.9.15-fix1 — add v0.9.15 version marker to dedent_method_source.
# v0.9.15 COMPLETE.
#
# v0.9.15 applied all 5 steps (log OK×5), but verify_state failed:
#   MISSING cadw/patch_utils.py: 'v0.9.15'
#
# Root: v0.9.15's T2.6 helper (dedent_method_source) docstring
# references §13.4/215/216 but not "v0.9.15" version marker. Prior
# patches (v0.9.7-v0.9.14) put vX.Y.Z in new code blocks — pattern
# broken here. verify_state anchor `("cadw/patch_utils.py",
# "v0.9.15")` demands it.
#
# §13.4/224 (new): patch markers `vX.Y.Z` should appear in every
# new/changed code block (docstring or comment) — the pattern that
# v0.9.7-v0.9.14 followed. This provides (a) grep-grep tracking,
# (b) verify anchor, (c) blame annotation. When a patch introduces
# a new function/block, include the version marker. v0.9.15 T2.6
# broke pattern. Fix: add marker.
#
# State on disk (from v0.9.15, all applied):
#   CHANGELOG.md 0.9.14 section     — applied
#   cadw/__init__.py __version__    — applied
#   patch_utils dedent_method_source — applied (no v0.9.15 marker)
#   T2.6 tests (2)                  — applied
#   T2.7 meta-guard test            — applied
#   verify_state                    — failed on one anchor
#   commit                          — never reached
#
# Fix1:
#   (a) Add `v0.9.15 (T2.6)` marker to dedent_method_source docstring.
#   (b) verify + dogfood + pytest + commit.
#
# Run: uv run python scripts/patch_v0915_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) add version marker to dedent_method_source ============

def add_version_marker():
    print("\n=== (a) dedent_method_source: v0.9.15 marker ===")
    pu.apply_or_fail(
        "cadw/patch_utils.py",
        (
            "def dedent_method_source(fn):\n"
            "    \"\"\"Return (dedented_source, ast.Module) for fn (§13.4/215/216).\n"
            "\n"
            "    inspect.getsource(fn) on a class method / nested def returns\n"
        ),
        (
            "def dedent_method_source(fn):\n"
            "    \"\"\"v0.9.15 (T2.6): return (dedented_src, ast.Module) for fn.\n"
            "\n"
            "    §13.4/215/216: inspect.getsource(fn) on class method /\n"
            "    nested def returns source WITH class-level indent; both\n"
            "    ast.parse and substring checks need dedented form.\n"
            "\n"
            "    inspect.getsource(fn) on a class method / nested def returns\n"
        ),
        label="v0.9.15 marker",
    )
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.15 + fix1 + regression ===")
    pu.verify_anchors([
        # fix1
        ("cadw/patch_utils.py", "v0.9.15 (T2.6)"),
        # v0.9.15
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
        ("CHANGELOG.md", "### v0.9.7–v0.9.14: audit round 2"),
        ("cadw/__init__.py", '__version__ = "0.9.14"'),
        ("cadw/patch_utils.py", "def dedent_method_source("),
        ("cadw/patch_utils.py", "v0.9.15"),
        ("tests/test_patch_utils.py", "test_dedent_method_source_basic"),
        ("tests/test_patch_utils.py",
         "test_no_inline_dedent_getsource_outside_helper"),
        # v0.9.14-fix3 regression
        ("cadw/dataframe.py", "v0.9.14-fix3 (§13.4/223)"),
        ("tests/test_apply.py", "test_apply_all_bool_stays_bool"),
        ("tests/test_apply.py", "test_infer_arrow_type_from_value_unit"),
        # v0.9.14 (F12-6a/b, F12-7, T2.2, T2.9, U3)
        ("cadw/dataframe.py", "v0.9.14 (F12-6a)"),
        ("cadw/dataframe.py", "v0.9.14 (F12-6b)"),
        ("cadw/dataframe.py", "§11.5, v0.9.14 F12-7"),
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
    print("\n=== dogfood: v0.9.15-fix1 via subprocess ===")

    def _verify_all():
        import ast as _ast
        from cadw.patch_utils import dedent_method_source
        from cadw.dataframe import CadwDataFrame

        # v0.9.15: __version__
        import cadw
        assert cadw.__version__ == "0.9.14", (
            f"cadw.__version__ = {cadw.__version__!r}, expected '0.9.14'"
        )

        # CHANGELOG section
        from pathlib import Path as _P
        cl = _P("CHANGELOG.md").read_text()
        assert "## [0.9.14] — 2026-10-21" in cl
        assert "### v0.9.7–v0.9.14: audit round 2" in cl
        assert cl.index("## [Unreleased]") < cl.index("## [0.9.14]")
        assert cl.index("## [0.9.14]") < cl.index("## [0.9.6]")

        # T2.6: helper works + v0.9.15 marker
        import inspect
        dms_doc = dedent_method_source.__doc__ or ""
        assert "v0.9.15 (T2.6)" in dms_doc, (
            f"dedent_method_source docstring missing v0.9.15 marker: "
            f"{dms_doc[:100]!r}"
        )
        src, tree = dedent_method_source(CadwDataFrame.head)
        assert isinstance(src, str) and isinstance(tree, _ast.Module)
        assert src.split("\n")[0].startswith("def ")
        fdefs = [n for n in _ast.walk(tree)
                 if isinstance(n, _ast.FunctionDef) and n.name == "head"]
        assert fdefs, "head not found in parsed tree"

        # T2.7 meta-guard: run over cadw/ + scripts/
        violations = []
        for root in (_P("cadw"), _P("scripts")):
            if not root.exists():
                continue
            for py in root.rglob("*.py"):
                if py.name == "patch_utils.py":
                    continue
                try:
                    src_str = py.read_text()
                    t = _ast.parse(src_str)
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
            f"§13.4/216: inline dedent(getsource(...)) at: {violations}"
        )

        print("  __version__ = '0.9.14' OK")
        print("  CHANGELOG: 0.9.14 section OK")
        print("  T2.6 dedent_method_source (str, tree, v0.9.15 marker) OK")
        print("  T2.7 meta-guard: no inline dedent(getsource) violations OK")

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
    print("=== v0.9.15-fix1 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    add_version_marker()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.15-fix1: complete v0.9.15 (v0.9.15 marker in "
        "dedent_method_source docstring). v0.9.15 applied all 5 "
        "steps (log OK×5) but verify_state failed on one anchor: "
        "MISSING cadw/patch_utils.py: 'v0.9.15'. Root: v0.9.15's "
        "T2.6 helper (dedent_method_source) docstring references "
        "§13.4/215/216 but not 'v0.9.15' version marker. Prior "
        "patches (v0.9.7-v0.9.14) put vX.Y.Z in new code blocks — "
        "pattern broken here. verify_state anchor "
        "(\"cadw/patch_utils.py\", \"v0.9.15\") demands it. §13.4/224 "
        "(new): patch markers vX.Y.Z should appear in every new/"
        "changed code block (docstring or comment) — pattern that "
        "v0.9.7-v0.9.14 followed. Provides (a) grep-grep tracking, "
        "(b) verify anchor, (c) blame annotation. When a patch "
        "introduces a new function/block, include the version "
        "marker. Fix1: (a) dedent_method_source docstring starts "
        "with 'v0.9.15 (T2.6): return (dedented_src, ast.Module)'; "
        "(b) verify (21 anchors, all applied) + dogfood (version, "
        "CHANGELOG order, T2.6 marker, T2.7 no violations) + pytest "
        "+ commit. v0.9.15 COMPLETE: CHANGELOG 0.9.14 section "
        "(covers v0.9.7-v0.9.14, 16 patches, 21 original + 13 "
        "discovered findings, 3 silent data-loss bugs, 13 new "
        "§13.4 meta-lessons), cadw.__version__ = '0.9.14', T2.6 "
        "dedent_method_source helper, T2.7 meta-guard (no inline "
        "dedent-getsource outside helper). Deferred: DECISIONS.md "
        "§13.4/211-223 append, A1 dataframe.py split (5404 lines), "
        "release tag v0.9.14 (after v0.9.15).",
        self_path="scripts/patch_v0915_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.15-fix1 applied. v0.9.15 COMPLETE.")


if __name__ == "__main__":
    main()
