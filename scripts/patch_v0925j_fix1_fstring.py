#!/usr/bin/env python3
# v0.9.25j-fix1 — f-string split across literals: `{` never closed.
#
# STATE: HEAD=v0.9.25i. v0.9.25j aborted at add_methods() via
# apply_or_fail's compile-before-write (§13.4/211):
#
#   cut_qcut_duplicated: COMPILE FAIL for cadw/dataframe.py:
#   f-string: expecting '}' (dataframe.py, line 1798)
#
# 0 mutation of dataframe.py (compile-before-write is a check, not a
# write). 0 mutation of tests/test_v0925j.py (add_tests ran AFTER
# add_methods, never reached). HEAD unchanged.
#
# BUG in scripts/patch_v0925j_cut_qcut_duplicated.py's METHODS_BLOCK:
#   raise ValueError(
#       f"qcut: all values identical ({unique_edges[0] if "
#       f"unique_edges else 'empty'}); cannot create bins"
#   )
#
# The `{` in `({unique_edges[0] if ` opens a replacement field, but
# no `}` closes it in the SAME string literal. Python's f-string
# parser cannot look across literals for `}` — each f-string is
# independently tokenized. Result: SyntaxError "f-string: expecting
# '}'".
#
# §13.4/271 (NEW): f-string brace scope is per-literal, not per-
# concatenation. `f"a{x if " f"y else z}b"` is a SyntaxError, not
# three-piece concatenation. When splitting an f-string across lines
# and the expression contains `if`/`for`/`:` — pre-compute the
# expression into a variable, then use `{var}` in ONE literal.
# Same class as v0.9.19-fix1 (§13.4/230): each list-element or line
# must be a complete syntactic unit for f-strings.
#
# FIX: extract `_first_or_empty = unique_edges[0] if unique_edges
# else "empty"` as a statement before the f-string; then one clean
# f-string `f"qcut: all values identical ({_first_or_empty}); cannot
# create bins"`.
#
# POST-FIX VERIFICATION (§13.4/271 companion): after fix, extract
# METHODS_BLOCK from the fixed script, dedent, `compile()`. This
# checks the INNER string, not just the script's own syntax. Without
# it we'd re-run v0.9.25j, hit the same wall at apply_or_fail, need
# fix2 — costing another cycle. Local pre-flight is cheaper.
#
# §13.4/264 (REPL-first): verify_state does 4 checks:
#   (1) HEAD subject-prefix == v0.9.25i (regex, not substring — §13.4/270)
#   (2) dataframe.py has no cut/qcut/duplicated (0 mutation)
#   (3) script has broken f-string (bug marker)
#   (4) script has fixed f-string ABSENT (fix not yet applied)
#
# §13.4/266 (anchor check): exact 6-line block for broken f-string.
#
# §13.4/267 (mirror-check): only ONE broken f-string in METHODS_BLOCK
# — grep for `f"({` ... `if "` pattern across script; expected 1 hit
# in broken state, 0 after fix. No other identical issue.
#
# §13.4/260/261/262/265 preserved: no subprocess dogfood, no symbol
# moves, typed scalars, no copy-pasted asserts.
#
# Usage:
#   uv run python scripts/patch_v0925j_fix1_fstring.py
#   uv run python scripts/patch_v0925j_cut_qcut_duplicated.py

from __future__ import annotations
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, assert_compiles, check_no_invalid_escape,
)


SCRIPT_PATH = "scripts/patch_v0925j_cut_qcut_duplicated.py"

BROKEN_BLOCK = '''        if len(unique_edges) < 2:
            raise ValueError(
                f"qcut: all values identical ({unique_edges[0] if "
                f"unique_edges else 'empty'}); cannot create bins"
            )'''

FIXED_BLOCK = '''        if len(unique_edges) < 2:
            _first_or_empty = (
                unique_edges[0] if unique_edges else "empty"
            )
            raise ValueError(
                f"qcut: all values identical ({_first_or_empty}); "
                f"cannot create bins"
            )'''


def verify_state():
    print("\n=== verify: HEAD=v0.9.25i (subject-prefix); dataframe.py "
          "0-mutation; broken f-string present ===")

    # (1) HEAD subject-prefix, not substring (§13.4/270)
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    m = re.match(r'^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):', log)
    head_version = m.group(1) if m else None
    if head_version != "v0.9.25i":
        raise SystemExit(
            f"expected HEAD version v0.9.25i; got {head_version!r}"
        )
    print(f"  (1) HEAD subject-prefix = v0.9.25i ✓")

    # (2) dataframe.py has no cut/qcut/duplicated (0-mutation)
    df_src = Path("cadw/dataframe.py").read_text()
    for marker in ("def cut(self", "def qcut(self", "def duplicated(self"):
        if marker in df_src:
            raise SystemExit(
                f"UNEXPECTED: {marker!r} in dataframe.py — v0.9.25j "
                f"should have aborted before write (compile-before-"
                f"write). Partial mutation suspected; inspect."
            )
    print("  (2) dataframe.py: no cut/qcut/duplicated (0 mutation) ✓")

    # (3) broken f-string present in script
    script_src = Path(SCRIPT_PATH).read_text()
    if BROKEN_BLOCK not in script_src:
        raise SystemExit(
            f"broken block not found in {SCRIPT_PATH}. Already fixed "
            f"or script was edited. Inspect."
        )
    print("  (3) broken f-string block present in patch script ✓")

    # (4) fixed block absent (fix not applied yet)
    if FIXED_BLOCK in script_src:
        raise SystemExit(
            "fixed block already present — fix1 misplaced"
        )
    print("  (4) fixed block absent (fix not yet applied) ✓")

    # (5) mirror-check (§13.4/267): exactly 1 broken-f-string pattern
    #     `f"...({" followed by ` if "` across the script's METHODS_BLOCK.
    # Simpler: count occurrences of `({unique_edges[0] if "` — should
    # be exactly 1 in broken state.
    n_broken = script_src.count('({unique_edges[0] if "')
    if n_broken != 1:
        raise SystemExit(
            f"expected exactly 1 broken f-string, found {n_broken}. "
            f"Multiple bugs — fix1 scope too narrow, expand."
        )
    print("  (5) mirror-check: exactly 1 broken f-string (as expected) ✓")

    print("  state OK")


def fix_script():
    print("\n=== apply fix: pre-compute _first_or_empty, one f-string ===")
    applied = apply_or_fail(
        SCRIPT_PATH, BROKEN_BLOCK, FIXED_BLOCK,
        label="qcut_fstring",
    )
    if applied:
        assert_compiles(SCRIPT_PATH)
        check_no_invalid_escape(SCRIPT_PATH)
    print("  applied" if applied else "  already applied — skip")


def verify_inner_compiles():
    """§13.4/271 companion: compile METHODS_BLOCK content AFTER fix.

    Pre-flight for the INNER string, not just the outer script. If
    this fails, we'd hit the same wall in v0.9.25j's apply_or_fail —
    catching it here saves one round-trip.
    """
    print("\n=== post-fix: extract METHODS_BLOCK + compile (inner check) ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_vj_fix_check", SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    block = mod.METHODS_BLOCK
    dedented = textwrap.dedent(block)
    try:
        compile(dedented, "<METHODS_BLOCK>", "exec")
    except SyntaxError as e:
        print(
            f"  STILL BROKEN: {e}\n"
            f"  (line {e.lineno}: {e.text!r})\n"
            f"  Fix1 didn't resolve; inspect _first_or_empty edit."
        )
        raise SystemExit(1)
    print(f"  METHODS_BLOCK compiles clean ({len(dedented.splitlines())} "
          f"lines) ✓")
    # Quick semantic sanity
    if "_first_or_empty" not in dedented:
        raise SystemExit(
            "fixed pattern _first_or_empty not in dedented block — "
            "apply_or_fail replaced wrong fragment?"
        )
    if '{unique_edges[0] if "}' in dedented:
        raise SystemExit(
            "broken pattern still in dedented block — apply_or_fail "
            "under-replaced (partial match?)"
        )
    print("  semantic check: _first_or_empty present, broken pattern "
          "absent ✓")


def main():
    print("=== v0.9.25j-fix1: f-string split across literals ===")
    verify_state()
    fix_script()
    verify_inner_compiles()

    print("\n=== next step ===")
    print("  uv run python scripts/patch_v0925j_cut_qcut_duplicated.py")


if __name__ == "__main__":
    main()