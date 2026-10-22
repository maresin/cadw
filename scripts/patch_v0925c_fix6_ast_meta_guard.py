#!/usr/bin/env python3
# v0.9.25c-fix6 — substring meta-guard replaced with AST.
#
# Bug: v0.9.25c-fix5 added test_no_stale_old_compute_fan_in_formula as
# SUBSTRING check:
#     if "compute_fan_in" in line and ("== 1310" in line or "== 131" in line)
# The meta-guard's own source contains `== 1310` in a condition line,
# and adjacent new test_mustfix_easy docstring contains "(was 1310)".
# dogfood in fix5 caught these as "hits" (false positives). §13.4/161+
# 183: substring guard on code patterns = 4th occurrence in project.
#
# Irony: meta-guard against substring-bugs itself built with substring.
# Fix: AST walk on ast.Compare nodes where left=Call(Name('compute_fan_in'))
# and any comparator=Constant(1310|131). Comments/docstrings invisible.
#
# fix6 also:
#   1. Diagnostic scan: if any REAL stale assertion (AST-level) exists,
#      print file:line and exit. User re-runs with matching anchors.
#   2. Replace substring meta-guard with AST meta-guard.
#   3. Add meta-meta-guard `test_no_substring_meta_guards` to
#      tests/test_patch_utils.py — scans tests/ for substring checks
#      involving patterns that look like code assertions (= = NNN,
#      fn_name, "assert "). Catches future substring meta-guard drift.
#
# Fix-cycle 6/2 on v0.9.25c. Distribution:
#   - cadw/ production: 0/2 (v0.9.25c initial correct)
#   - scripts/collect_project.py: 1/2 (fix4 real bug)
#   - tests/: unbounded but decreasing radius (fix5: 1 stale + meta;
#     fix6: 0 new bugs, only fixes v0.9.25c-fix5's meta-guard)
#
# Run: uv run python scripts/patch_v0925c_fix6_ast_meta_guard.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape, run_verify_in_subprocess,
)


# ============ 0. Verify v0.9.25c + fix1-fix5 state ============

def verify_state():
    print("\n=== verify v0.9.25c + fix1-fix5 state ===")
    checks = [
        # v0.9.25c prod
        ("cadw/ops/sort.py", "parts: list = [self.buffer]"),
        ("cadw/ops/sort.py", "read_batch_size, row_bytes"),
        ("cadw/dataframe.py", "len(uniques) * 180"),
        # fix4 tooling
        ("scripts/collect_project.py", "_split_top_commas"),
        # fix5 test
        ("tests/test_mustfix_easy.py", "== 218"),
        # fix5 meta-guard (substring version — to be replaced)
        ("tests/test_v0925c_phase_a.py",
         "test_no_stale_old_compute_fan_in_formula"),
    ]
    for path, anchor in checks:
        if anchor not in Path(path).read_text():
            raise SystemExit(f"{path}: missing {anchor!r}")
    print("  all anchors present — OK")


# ============ 1. AST diagnostic: any REAL stale assertion? ============

def diagnostic_scan() -> list[str]:
    """AST-walk tests/ for real `compute_fan_in(...) == 1310|131`."""
    import ast as _ast
    from pathlib import Path as _P
    stale_values = {1310, 131}
    violations: list[str] = []
    for py in _P("tests").glob("test_*.py"):
        try:
            tree = _ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Compare):
                continue
            left = node.left
            if not (isinstance(left, _ast.Call)
                    and isinstance(left.func, _ast.Name)
                    and left.func.id == "compute_fan_in"):
                continue
            for comp in node.comparators:
                if (isinstance(comp, _ast.Constant)
                        and isinstance(comp.value, int)
                        and comp.value in stale_values):
                    violations.append(
                        f"{py.name}:{node.lineno}: compute_fan_in(...) == "
                        f"{comp.value}"
                    )
    return violations


def run_diagnostic():
    print("\n=== AST diagnostic: real stale compute_fan_in assertions ===")
    violations = diagnostic_scan()
    if not violations:
        print("  none found — clean")
        return
    print(f"  found {len(violations)} real stale assertion(s):")
    for v in violations:
        print(f"    {v}")
    raise SystemExit(
        "Real stale compute_fan_in assertions found (AST-level, not "
        "comments). Re-run v0.9.25c-fix6 with matching anchors, or fix "
        "manually. Diagnostics above."
    )


# ============ 2. Replace substring meta-guard with AST version ========

OLD_META_GUARD = '''

# ============ v0.9.25c-fix5: meta-guard for stale formula ============


def test_no_stale_old_compute_fan_in_formula():
    """Meta-guard: no test file asserts pre-v0.9.25c formula output.

    Old formula (pre-v0.9.25c): available // (rbs * row_bytes).
    New formula (v0.9.25c): state_limit // (3 * rbs * row_bytes),
    state_limit = available * 0.5.

    Specific hardcoded values to watch:
      - (1_048_576, 100, 8): old=1310, new=218.
      - (1_048_576, 100, 80): old=131, new=21.

    Scans tests/ for any line combining "compute_fan_in" with "== 1310"
    or "== 131" (old formula outputs). Future refactors of the formula
    must update both cadw/ops/sort.py::compute_fan_in AND all tests
    asserting its output. §13.4/144 (stale-test-on-refactor).
    """
    from pathlib import Path as _P
    violations: list[str] = []
    for py in _P("tests").glob("test_*.py"):
        try:
            src = py.read_text()
        except Exception:
            continue
        for i, line in enumerate(src.split("\\n"), 1):
            if "compute_fan_in" not in line:
                continue
            if "== 1310" in line or "== 131" in line:
                violations.append(f"{py.name}:{i}: {line.strip()}")
    assert not violations, (
        "Tests assert pre-v0.9.25c compute_fan_in formula "
        "(= 1310/131). New formula (v0.9.25c): "
        "state_limit // (3*rbs*row_bytes); new values are 218/21. "
        "Fix:\\n" + "\\n".join(violations)
    )
'''

NEW_META_GUARD = '''

# ============ v0.9.25c-fix5 (AST, rewritten v0.9.25c-fix6) ============


def test_no_stale_old_compute_fan_in_formula():
    """AST: no real test asserts pre-v0.9.25c compute_fan_in output.

    Old formula (pre-v0.9.25c): available // (rbs * row_bytes).
    New formula (v0.9.25c): state_limit // (3 * rbs * row_bytes),
    state_limit = available * 0.5.
    Stale output values: 1310 for (1_048_576,100,8); 131 for
    (1_048_576,100,80).

    v0.9.25c-fix6: prior fix5 was a SUBSTRING check ("compute_fan_in" in
    line and "== 1310" in line). It false-positived on (a) its own
    condition line, (b) docstring/comment "(was 1310)" in the fixed
    formula test. §13.4/161+183 — 4th occurrence. AST-based: only
    ast.Compare nodes with left=Call(Name('compute_fan_in')) and
    comparator=Constant(1310|131). Comments/docstrings invisible.
    """
    import ast as _ast
    from pathlib import Path as _P
    stale_values = {1310, 131}
    violations: list[str] = []
    for py in _P("tests").glob("test_*.py"):
        try:
            tree = _ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Compare):
                continue
            left = node.left
            if not (isinstance(left, _ast.Call)
                    and isinstance(left.func, _ast.Name)
                    and left.func.id == "compute_fan_in"):
                continue
            for comp in node.comparators:
                if (isinstance(comp, _ast.Constant)
                        and isinstance(comp.value, int)
                        and comp.value in stale_values):
                    violations.append(
                        f"{py.name}:{node.lineno}: compute_fan_in(...) "
                        f"== {comp.value}"
                    )
    assert not violations, (
        "Real (AST-level) stale compute_fan_in assertions detected. "
        "New values: 218 for (1_048_576,100,8); 21 for "
        "(1_048_576,100,80). Violations: "
        + "; ".join(violations)
    )
'''

def replace_meta_guard():
    print("\n=== replace substring meta-guard with AST version ===")
    applied = apply_or_fail(
        "tests/test_v0925c_phase_a.py", OLD_META_GUARD, NEW_META_GUARD,
        label="meta_guard_ast",
    )
    if applied:
        assert_compiles("tests/test_v0925c_phase_a.py")
        check_no_invalid_escape("tests/test_v0925c_phase_a.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 3. Meta-meta-guard in test_patch_utils.py ============

META_META_GUARD = '''

# ============ v0.9.25c-fix6: meta-meta-guard ============


def test_no_substring_meta_guards_in_tests():
    """Meta-meta-guard: no test uses substring `in` on code-ish
    patterns for assertion of code behavior.

    §13.4/161+183 (4th occurrence v0.9.25c-fix5): a test that does
    `assert "fn_name" in line and "== 1310" in line` cannot distinguish
    the real assertion from its own condition code, from docstrings,
    from comments. This has burned CADW 4× — always via substring
    on code patterns in meta-guards.

    Rule: substring `in` is allowed for:
      - Path / string ops (str content, filenames, config keys).
      - Actual text values (README, log messages).
      - NOT for "is there a call to X" or "does code compare X == N" —
        use ast.parse + ast.walk.

    This guard scans tests/test_*.py for suspicious substring patterns:
    any line where `in` is applied to a variable that ALSO appears
    alongside a code-shaped literal:
      - `assert "..." in line and ... == N` (equality to int literal)
      - `if "fn_name" in line and "== " in line` — this exact
        anti-pattern.
    """
    import re as _re
    from pathlib import Path as _P
    # Narrow signal: code pattern `if "X" in line and "== " in line`
    # — the exact shape of v0.9.25c-fix5's bug. Broader heuristics
    # produce false positives on legitimate text searches.
    pattern = _re.compile(
        r'if\\s+"[^"]+"\\s+in\\s+line\\s+and\\s+"==\\s',
    )
    violations: list[str] = []
    for py in _P("tests").glob("test_*.py"):
        try:
            src = py.read_text()
        except Exception:
            continue
        for i, line in enumerate(src.split("\\n"), 1):
            if pattern.search(line):
                violations.append(f"{py.name}:{i}: {line.strip()}")
    assert not violations, (
        "Substring meta-guard detected (§13.4/161+183, 4th occurrence "
        "v0.9.25c-fix5). Use ast.parse + ast.walk instead.\\n"
        + "\\n".join(violations)
    )
'''

def add_meta_meta_guard():
    print("\n=== add meta-meta-guard to test_patch_utils.py ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "test_no_substring_meta_guards_in_tests" in s:
        print("  already present — skip")
        return
    p.write_text(s + META_META_GUARD)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 4. Dogfood: AST-only (no substring) ============

def dogfood():
    print("\n=== dogfood: AST meta-guard works; no false positives ===")

    def _verify():
        import ast as _ast
        import logging as _logging
        _logging.getLogger("cadw.memory_guard").setLevel(_logging.CRITICAL)
        from pathlib import Path as _P
        from cadw.ops.sort import compute_fan_in

        # (1) Formula is correct.
        assert compute_fan_in(1_048_576, 100, 8) == 218
        assert compute_fan_in(1_048_576, 100, 80) == 21

        # (2) AST scan: no real stale compute_fan_in assertions.
        stale_values = {1310, 131}
        violations = []
        for py in _P("tests").glob("test_*.py"):
            try:
                tree = _ast.parse(py.read_text())
            except SyntaxError:
                continue
            for node in _ast.walk(tree):
                if not isinstance(node, _ast.Compare):
                    continue
                left = node.left
                if not (isinstance(left, _ast.Call)
                        and isinstance(left.func, _ast.Name)
                        and left.func.id == "compute_fan_in"):
                    continue
                for comp in node.comparators:
                    if (isinstance(comp, _ast.Constant)
                            and isinstance(comp.value, int)
                            and comp.value in stale_values):
                        violations.append(
                            f"{py.name}:{node.lineno}"
                        )
        assert not violations, (
            f"AST-detected stale compute_fan_in: {violations}"
        )

        # (3) Substring false-positive NEGATIVE test: a line with
        # both `compute_fan_in` and `== 1310` should NOT be detected
        # by AST (because it's a comment). Construct one on the fly.
        fake_src = (
            "def test_x():\\n"
            "    # compute_fan_in(...) == 1310 (this is a comment)\\n"
            "    pass\\n"
        )
        tree = _ast.parse(fake_src)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Compare):
                raise AssertionError(
                    "AST parser incorrectly detected a Compare node in a "
                    "comment — bug in test harness"
                )

        # (4) Substring NEGATIVE test on real source: line inside AST
        # meta-guard itself has `stale_values = {1310, 131}` — AST
        # sees Set{Constant(1310), Constant(131)}, not Compare. Correct.

        # (5) Confirm meta-guard test PASSES with current files.
        # (Direct AST run — mirrors test_no_stale_old_compute_fan_in_formula.)
        # Already done in (2).

        print(
            "  AST meta-guard: 0 false positives; formula 218/21; "
            "comment-line `compute_fan_in == 1310` correctly ignored"
        )

    run_verify_in_subprocess(_verify)


# ============ 5. Pytest ============

def run_pytest():
    print("\n=== pre-flight: test_facade_has_no_as_py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py", "-q"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** FACADE AS_PY VIOLATION — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: targeted (v0925c + mustfix + patch_utils + collect) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925c_phase_a.py",
         "tests/test_mustfix_easy.py",
         "tests/test_patch_utils.py",
         "tests/test_collect_project_modes.py",
         "tests/test_sort_merge_p_s1.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 6. main ============

def main():
    print("=== v0.9.25c-fix6: AST meta-guard (no substring) ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    run_diagnostic()          # AST scan; exits if real stale found
    replace_meta_guard()      # substring → AST
    add_meta_meta_guard()     # new guard in test_patch_utils.py
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25c: Phase A batch 2 — A.4 + A.5 + A.7 + B.4 "
        "(4 items from docs/roadmap.json). "
        "A.4 (part1_report B1): _RunStream.refill linear concat "
        "(12.9 GB copies @ rbs=200k → 0.13 GB). "
        "A.5 (part1_report B2): encode_categorical in-loop guard. "
        "A.7 (audit §6.6): compute_fan_in uses read_batch_size, NOT "
        "_WRITE_RUN_CHUNK. Formula: state_limit // (3*K*rbs*row). "
        "max_k=256 FD cap. K=8 @ 32MB/8cols/640B/row. "
        "B.4 (docs/roadmap.json): collect_project.py --module / "
        "--phase / --doc / --list-phases. Reads docs/roadmap.json. "
        "fix1 (test/tooling): cleandoc→textwrap.dedent; missing "
        "import subprocess; no-args → legacy snapshot. "
        "fix2 (test-side): logger name 'cadw.memory_guard'. "
        "fix3 (roadmap): A.6 spec narrowed to 12 files; "
        "current_version 0.9.25c; A.4/A.5/A.7 done. "
        "fix4 (REAL production bug v0.9.25c collect_project.py): "
        "_parse_file_spec comma-split BEFORE brace-expansion → "
        "'cadw/ts/{a,b,c}.py' silently dropped. Masked by A.4/A.5/A.7 "
        "planned; unmasked after fix3. Fix: _split_top_commas(s) — "
        "comma only at brace-depth 0. Unit tests for parsing grammar. "
        "fix5 (stale formula test): test_compute_fan_in_formula "
        "asserted pre-v0.9.25c 1310; v0.9.25c new = 218/21. "
        "Meta-guard test_no_stale_old_compute_fan_in_formula added. "
        "fix6 (meta-guard itself was substring, false-positived): "
        "v0.9.25c-fix5 meta-guard used substring `if "
        "'compute_fan_in' in line and '== 1310' in line`. False-"
        "positived on (a) its own condition line, (b) docstring/"
        "comment '= 218 (was 1310)' in the fixed formula test. "
        "§13.4/161+183 — 4th occurrence in project. "
        "Fix: AST-based meta-guard (ast.Compare, left=Call(Name("
        "'compute_fan_in')), comparator=Constant(1310|131)). Comments/"
        "docstrings invisible. Diagnostic scan added to fix6: exits if "
        "REAL stale found (helps user debug before commit). "
        "Also: meta-meta-guard test_no_substring_meta_guards_in_tests "
        "in test_patch_utils.py — regex `if '...' in line and '== '` "
        "catches the exact anti-pattern shape; future meta-guards must "
        "be AST. §13.4/252 (new). "
        "§13.4 meta-lesson (consolidated from 6 cycles v0.9.25c): "
        "(1) formula/grammar refactor → unit-test FIRST, e2e only "
        "for wiring; "
        "(2) grep all callers BEFORE commit — 5s search would have "
        "found test_mustfix_easy.py::test_compute_fan_in_formula; "
        "(3) absence-assert in e2e only if goal unreachable via "
        "planned-items discovery (v0.9.25c A.6 pulled tests/test_"
        "ts_models.py back); "
        "(4) substring meta-guard on code patterns cannot distinguish "
        "itself from its subject — use AST. "
        "Fix-cycle distribution v0.9.25c: cadw/ production 0/2 "
        "(v0.9.25c initial correct since first run); scripts/collect_"
        "project.py 1/2 (fix4 real bug); tests/ unbounded but "
        "decreasing radius: fix1(3)→fix2(2)→fix3(2)→fix4(1+test)→"
        "fix5(stale+meta)→fix6(meta-only, 0 new prod bugs). "
        "Test additions: 12 + 1 meta-guard (fix5) + 1 meta-meta-guard "
        "(fix6) + 1 flipped. "
        "Fast suite: ~2275 → ~2289.",
        self_path="scripts/patch_v0925c_fix6_ast_meta_guard.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25c+fix1-6 applied. Next: v0.9.25d — A.6 "
          "(18 to_pylist extra_factor sites). Production cadw/ only, "
          "no tooling changes.")


if __name__ == "__main__":
    main()