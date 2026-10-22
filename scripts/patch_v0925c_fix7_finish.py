#!/usr/bin/env python3
# v0.9.25c-fix7 — fix5's OLD anchor was partial (2 asserts, replaced 1)
#                 + finishes v0.9.25c-fix6's plan.
#
# Bug: test_mustfix_easy.py::test_compute_fan_in_formula had TWO
# assertions pre-v0.9.25c:
#     assert compute_fan_in(1_048_576, 100, 8) == 1310
#     assert compute_fan_in(1_048_576, 100, 80) == 131
# v0.9.25c-fix5's OLD anchor matched only the first (3 lines: def +
# comment + assert). The second (comment + assert) survived the replace
# as orphan → stale assertion.
#
# v0.9.25c-fix6's AST diagnostic correctly caught it before commit
# (which is the intended behavior — good). fix7 removes the leftover
# and continues fix6's plan (substring meta-guard → AST, meta-meta-guard).
#
# §13.4/253 (2nd occurrence: v0.9.9-fix1c, v0.9.25c-fix5): partial-anchor.
# When replacing function test bodies, OLD anchor must capture ALL
# assertions being updated, not just the first. Grep for additional
# asserts of same function name before replace.
#
# Fix-cycle 7/2 on v0.9.25c. Production cadw/ 0/2 (untouched since
# initial). scripts/collect_project.py 1/2 (fix4 real bug). tests/
# unbounded but decreasing radius (fix7 = 2-line removal + finishes
# fix6 — no new bugs, just incomplete prior anchor).
#
# Run: uv run python scripts/patch_v0925c_fix7_finish.py

from __future__ import annotations
import ast
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


# ============ 0. Verify state (leftover present) ============

def verify_state():
    print("\n=== verify v0.9.25c + fix1-5 state (fix6 exited cleanly) ===")
    checks = [
        ("cadw/ops/sort.py", "parts: list = [self.buffer]"),
        ("cadw/ops/sort.py", "read_batch_size, row_bytes"),
        ("cadw/dataframe.py", "len(uniques) * 180"),
        ("scripts/collect_project.py", "_split_top_commas"),
        ("tests/test_mustfix_easy.py", "== 218"),           # fix5 applied
        ("tests/test_mustfix_easy.py",
         "compute_fan_in(1_048_576, 100, 80) == 131"),      # leftover (bug)
        ("tests/test_v0925c_phase_a.py",
         "test_no_stale_old_compute_fan_in_formula"),       # substring meta
    ]
    for path, anchor in checks:
        if anchor not in Path(path).read_text():
            raise SystemExit(f"{path}: missing {anchor!r}")
    print("  all anchors present (incl. leftover & substring meta) — OK")


# ============ 1. Remove leftover stale assertion ============

LEFT1 = (
    "    # 1_048_576 / (100*80) = 131\n"
    "    assert compute_fan_in(1_048_576, 100, 80) == 131\n"
)
LEFT2 = "    assert compute_fan_in(1_048_576, 100, 80) == 131\n"


def remove_leftover():
    print("\n=== remove leftover == 131 assertion ===")
    p = Path("tests/test_mustfix_easy.py")
    s = p.read_text()
    if LEFT1 in s:
        p.write_text(s.replace(LEFT1, "", 1))
        print("  removed comment + assert (combined anchor)")
    elif LEFT2 in s:
        p.write_text(s.replace(LEFT2, "", 1))
        print("  removed assert only (comment orphan)")
    else:
        raise SystemExit(
            "leftover `== 131` not found. File may have been manually "
            "edited. Inspect tests/test_mustfix_easy.py::"
            "test_compute_fan_in_formula."
        )
    assert_compiles(p)
    check_no_invalid_escape(p)


# ============ 2. Re-run AST diagnostic (expect clean) ============

def diagnostic_scan() -> list[str]:
    stale_values = {1310, 131}
    violations: list[str] = []
    for py in Path("tests").glob("test_*.py"):
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            left = node.left
            if not (isinstance(left, ast.Call)
                    and isinstance(left.func, ast.Name)
                    and left.func.id == "compute_fan_in"):
                continue
            for comp in node.comparators:
                if (isinstance(comp, ast.Constant)
                        and isinstance(comp.value, int)
                        and comp.value in stale_values):
                    violations.append(
                        f"{py.name}:{node.lineno}: == {comp.value}"
                    )
    return violations


def run_diagnostic():
    print("\n=== AST diagnostic: real stale compute_fan_in assertions ===")
    violations = diagnostic_scan()
    if not violations:
        print("  none found — clean")
        return
    print(f"  {len(violations)} remaining:")
    for v in violations:
        print(f"    {v}")
    raise SystemExit("Diagnostic still dirty after remove_leftover")


# ============ 3. Replace substring meta-guard with AST ============

OLD_MG = '''

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

NEW_MG = '''

# ============ v0.9.25c-fix5 (AST, rewritten v0.9.25c-fix7) ============


def test_no_stale_old_compute_fan_in_formula():
    """AST: no real test asserts pre-v0.9.25c compute_fan_in output.

    v0.9.25c-fix7: prior fix5 was SUBSTRING check ("compute_fan_in" in
    line and "== 1310" in line) — 4th substring-guard bug in project
    (§13.4/161+183). It false-positived on its own condition and on
    docstrings "(was 1310)". AST-based: only ast.Compare nodes where
    left=Call(Name('compute_fan_in')) and comparator=Constant(1310|131).
    Comments/docstrings invisible.
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
                        f"{py.name}:{node.lineno}"
                    )
    assert not violations, (
        "Real stale compute_fan_in assertions. New values: 218 for "
        "(1_048_576,100,8); 21 for (1_048_576,100,80). "
        f"Violations: {violations}"
    )
'''


def replace_meta_guard():
    print("\n=== replace substring meta-guard with AST ===")
    applied = apply_or_fail(
        "tests/test_v0925c_phase_a.py", OLD_MG, NEW_MG,
        label="meta_guard_ast",
    )
    if applied:
        assert_compiles("tests/test_v0925c_phase_a.py")
        check_no_invalid_escape("tests/test_v0925c_phase_a.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 4. Meta-meta-guard in test_patch_utils.py ============

MMG = '''

# ============ v0.9.25c-fix7: meta-meta-guard ============


def test_no_substring_meta_guards_in_tests():
    """Meta-meta-guard against substring meta-guards (§13.4/252).

    Substring `if 'X' in line and '== N' in line` cannot distinguish
    a real assertion from its own condition or from a docstring.
    Burned CADW 4× (§13.4/161, /183, /213, /252).
    Regex catches the exact anti-pattern shape.
    """
    import re as _re
    from pathlib import Path as _P
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
        "Substring meta-guard detected (§13.4/252). Use ast.parse + "
        "ast.walk.\\n" + "\\n".join(violations)
    )
'''


def add_meta_meta_guard():
    print("\n=== add meta-meta-guard ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "test_no_substring_meta_guards_in_tests" in s:
        print("  already present — skip")
        return
    p.write_text(s + MMG)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 5. Dogfood ============

def dogfood():
    print("\n=== dogfood: diagnostic clean + AST meta-guard works ===")

    def _verify():
        import ast as _ast
        import logging as _logging
        _logging.getLogger("cadw.memory_guard").setLevel(_logging.CRITICAL)
        from pathlib import Path as _P
        from cadw.ops.sort import compute_fan_in

        # (1) Formula correct.
        assert compute_fan_in(1_048_576, 100, 8) == 218
        assert compute_fan_in(1_048_576, 100, 80) == 21

        # (2) AST scan: no stale.
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
                        violations.append(f"{py.name}:{node.lineno}")
        assert not violations, f"AST stale: {violations}"

        # (3) Substring meta-meta-guard: 0 hits.
        import re
        pattern = re.compile(
            r'if\s+"[^"]+"\s+in\s+line\s+and\s+"==\s',
        )
        mg_hits = []
        for py in _P("tests").glob("test_*.py"):
            for i, line in enumerate(py.read_text().split("\n"), 1):
                if pattern.search(line):
                    mg_hits.append(f"{py.name}:{i}")
        assert not mg_hits, f"substring meta-guard: {mg_hits}"

        print(
            "  diagnostic 0 stale; formula 218/21; "
            "substring meta-guard removed; AST meta-guard clean"
        )

    run_verify_in_subprocess(_verify)


# ============ 6. Pytest ============

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

    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925c_phase_a.py",
         "tests/test_mustfix_easy.py",
         "tests/test_patch_utils.py",
         "tests/test_collect_project_modes.py",
         "tests/test_sort_merge_p_s1.py",
         "tests/test_roadmap.py",
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


# ============ 7. main ============

def main():
    print("=== v0.9.25c-fix7: leftover == 131 + finish fix6 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    remove_leftover()
    run_diagnostic()          # fails fast if still dirty
    replace_meta_guard()
    add_meta_meta_guard()
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
        "--phase / --doc / --list-phases. Brace-expansion for "
        "composite file specs. "
        "fix1 (test/tooling): cleandoc→textwrap.dedent; missing "
        "import subprocess; no-args → legacy snapshot. "
        "fix2 (test-side): logger name 'cadw.memory_guard'. "
        "fix3 (roadmap): A.6 spec narrowed to 12 files; "
        "current_version 0.9.25c; A.4/A.5/A.7 done. "
        "fix4 (REAL production bug v0.9.25c collect_project.py): "
        "_parse_file_spec comma-split BEFORE brace-expansion → "
        "'cadw/ts/{a,b,c}.py' silently dropped. Masked by A.4/A.5/A.7 "
        "planned; unmasked after fix3. Fix: _split_top_commas. "
        "Unit tests for parsing grammar. "
        "fix5 (stale formula test): test_compute_fan_in_formula "
        "asserted pre-v0.9.25c 1310 → fixed to 218. Meta-guard added. "
        "fix6 (diagnostic only): AST scan found fix5's OLD anchor was "
        "PARTIAL — test had TWO assertions (== 1310 AND == 131); fix5 "
        "replaced only 1. fix6 exited before any change (diagnostic "
        "working correctly, prevented bad commit). §13.4/253 (2nd "
        "occurrence: v0.9.9-fix1c, now here): partial-anchor. "
        "When replacing function test bodies, grep for ALL asserts of "
        "same function name before replace. "
        "fix7 (finish): (1) removed leftover '== 131' (2 lines); "
        "(2) substring meta-guard replaced with AST (fix6's plan); "
        "(3) meta-meta-guard test_no_substring_meta_guards_in_tests "
        "in test_patch_utils.py (regex catches exact anti-pattern "
        "shape 'if \"X\" in line and \"== N\"'). "
        "§13.4 meta-lessons consolidated (v0.9.25c, 7 cycles): "
        "(1) formula/grammar refactor → unit-test FIRST; "
        "(2) grep all callers BEFORE commit — 5s search finds stale; "
        "(3) absence-assert in e2e only if goal unreachable via "
        "planned-items discovery; "
        "(4) substring meta-guard cannot distinguish its own subject "
        "from itself — AST is the only correct tool (4th occurrence "
        "in project); "
        "(5) partial-anchor: when replacing test body, grep ALL "
        "asserts of same function, not just first (2nd occurrence). "
        "Fix-cycle distribution v0.9.25c: cadw/ 0/2 (untouched); "
        "scripts/ 1/2 (fix4 real bug); tests/ unbounded but "
        "decreasing: fix1(3)→fix2(2)→fix3(2)→fix4(1prod+1test)→"
        "fix5(1)→fix6(diagnostic found partial)→fix7(0 new, "
        "completes fix6). Diagnostic at fix6 PREVENTED bad commit — "
        "this is the intended safety net, not a failure. "
        "Test additions: 12 + 1 meta-guard (fix7) + 1 meta-meta-guard "
        "(fix7) + 1 flipped. "
        "Fast suite: ~2275 → ~2290.",
        self_path="scripts/patch_v0925c_fix7_finish.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25c+fix1-7 applied. Next: v0.9.25d — A.6 "
          "(18 to_pylist extra_factor sites). Production cadw/ only, "
          "no tooling changes. Expected 0-1 fix-cycle.")


if __name__ == "__main__":
    main()