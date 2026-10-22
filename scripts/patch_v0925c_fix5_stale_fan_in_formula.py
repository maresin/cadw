#!/usr/bin/env python3
# v0.9.25c-fix5 — stale formula test in test_mustfix_easy.py.
#
# test_mustfix_easy.py::test_compute_fan_in_formula asserts the
# pre-v0.9.25c formula:
#     available // (rbs * row_bytes)
# for (1_048_576, 100, 8) == 1310.
#
# v0.9.25c changed to:
#     state_limit // (3 * rbs * row_bytes), state_limit = available * 0.5
# for (1_048_576, 100, 8) == 218.
#
# Production code correct: matches _merge_keys.check_state(
# 3 * K * rbs * row_bytes) <= state_limit. Docstring updated in
# v0.9.25c. Only stale test — missed in v0.9.25c because I grep'd only
# test_sort_merge_p_s1.py (AST test, arg name).
#
# test_mustfix_easy.py has 3 compute_fan_in tests:
#   - formula: FAILS (1310 vs 218)   ← fixed here
#   - minimum_2: passes (=2 either way)
#   - uses_row_size: passes (20 > 2 holds)
#
# fix5 scope:
# 1. Update formula test: 1310 -> 218, add (100, 80) case -> 21.
# 2. Add meta-guard test_no_stale_old_compute_fan_in_formula.
#
# Fix-cycle #5 on v0.9.25c (over budget; production cadw/ = 0 cycles,
# collect_project.py = 1 real bug in fix4; all 5 "cycles" are test-side
# drift from formula/CLI refactor). See §13.4 meta-lesson in commit.
#
# Run: uv run python scripts/patch_v0925c_fix5_stale_fan_in_formula.py

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


# ============ 0. Verify v0.9.25c state ============

def verify_state():
    print("\n=== verify v0.9.25c + fix1-fix4 state ===")
    checks = [
        # v0.9.25c prod anchors
        ("cadw/ops/sort.py", "parts: list = [self.buffer]"),
        ("cadw/ops/sort.py", "read_batch_size, row_bytes"),
        ("cadw/dataframe.py", "len(uniques) * 180"),
        # v0.9.25c tooling fixes
        ("scripts/collect_project.py", "_split_top_commas"),
        # Target stale test present (unfixed) — OK to fail here
        ("tests/test_mustfix_easy.py", "== 1310"),
    ]
    for path, anchor in checks:
        if anchor not in Path(path).read_text():
            raise SystemExit(f"{path}: missing {anchor!r}")
    print("  all anchors + stale test present — OK")


# ============ 1. Fix formula test ============

OLD_FORMULA_TEST = '''def test_compute_fan_in_formula():
    # 1_048_576 / (100*8) = 1310
    assert compute_fan_in(1_048_576, 100, 8) == 1310
'''

NEW_FORMULA_TEST = '''def test_compute_fan_in_formula():
    # v0.9.25c: formula = state_limit // (3 * rbs * row_bytes),
    # state_limit = available * 0.5 (matches _merge_keys.check_state(
    # 3 * K * rbs * row_bytes) <= state_limit). Prior formula (pre-
    # v0.9.25c): available // (rbs * row_bytes) — 6x too permissive
    # for actual merge memory (3x ignored temp buffers × 2 ignored
    # state_safety=0.5). Fix surfaced by audit §6.6 (24s wall @
    # 500k/8cols/32MB).
    # (1_048_576, 100, 8):  524_288 // 2400   = 218 (was 1310)
    # (1_048_576, 100, 80): 524_288 // 24000  = 21  (was 131)
    assert compute_fan_in(1_048_576, 100, 8) == 218
    assert compute_fan_in(1_048_576, 100, 80) == 21
'''

def fix_formula_test():
    print("\n=== fix test_mustfix_easy.py::test_compute_fan_in_formula ===")
    applied = apply_or_fail(
        "tests/test_mustfix_easy.py",
        OLD_FORMULA_TEST, NEW_FORMULA_TEST,
        label="formula_test",
    )
    if applied:
        assert_compiles("tests/test_mustfix_easy.py")
        check_no_invalid_escape("tests/test_mustfix_easy.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 2. Add meta-guard ============

META_GUARD = '''

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

def add_meta_guard():
    print("\n=== add meta-guard to test_v0925c_phase_a.py ===")
    p = Path("tests/test_v0925c_phase_a.py")
    s = p.read_text()
    if "test_no_stale_old_compute_fan_in_formula" in s:
        print("  already present — skip")
        return
    p.write_text(s + META_GUARD)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. Dogfood ============

def dogfood():
    print("\n=== dogfood: formula + meta-guard ===")

    def _verify():
        import logging as _logging
        _logging.getLogger("cadw.memory_guard").setLevel(_logging.CRITICAL)

        from cadw.ops.sort import compute_fan_in

        # New formula: state_limit // (3 * rbs * row_bytes).
        assert compute_fan_in(1_048_576, 100, 8) == 218, (
            f"expected 218, got {compute_fan_in(1_048_576, 100, 8)}"
        )
        assert compute_fan_in(1_048_576, 100, 80) == 21, (
            f"expected 21, got {compute_fan_in(1_048_576, 100, 80)}"
        )
        # Sanity: no stale test line remains.
        from pathlib import Path as _P
        hits = []
        for py in _P("tests").glob("test_*.py"):
            if "compute_fan_in" not in py.read_text():
                continue
            for i, line in enumerate(py.read_text().split("\n"), 1):
                if "compute_fan_in" in line and (
                    "== 1310" in line or "== 131" in line
                ):
                    hits.append(f"{py.name}:{i}")
        assert not hits, f"stale formula lines: {hits}"

        print("  formula = 218/21; no stale == 1310/131 in tests")

    run_verify_in_subprocess(_verify)


# ============ 4. Pytest ============

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

    print("\n=== pytest: targeted (mustfix + v0925c + sort + collect) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_mustfix_easy.py",
         "tests/test_v0925c_phase_a.py",
         "tests/test_collect_project_modes.py",
         "tests/test_roadmap.py",
         "tests/test_sort_merge_p_s1.py",
         "tests/test_merge_arrow.py",
         "tests/test_encode_categorical.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=900,
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


# ============ 5. main ============

def main():
    print("=== v0.9.25c-fix5: stale fan_in formula test ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    fix_formula_test()
    add_meta_guard()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25c: Phase A batch 2 — A.4 + A.5 + A.7 + B.4 "
        "(4 items from docs/roadmap.json). "
        "A.4 (part1_report B1): _RunStream.refill linear concat "
        "(12.9 GB copies @ rbs=200k → 0.13 GB). "
        "A.5 (part1_report B2): encode_categorical in-loop guard "
        "(10M uniques ~600 MB invisible → check_state per batch). "
        "A.7 (audit §6.6): compute_fan_in uses read_batch_size, "
        "NOT _WRITE_RUN_CHUNK. Root cause of 24.4s @ N=500k/8cols/32MB. "
        "Formula: state_limit // (3*K*rbs*row) — matches "
        "_merge_keys.check_state(3*K*rbs*row). max_k=256 FD cap. "
        "Now K=8 @ 32MB/8cols/640B/row. "
        "B.4 (docs/roadmap.json): collect_project.py --module / "
        "--phase / --doc / --list-phases. Reads docs/roadmap.json. "
        "Brace-expansion for composite file specs. "
        "fix1-fix3 (test/tooling): cleandoc→textwrap.dedent; "
        "missing import subprocess; no-args → legacy snapshot "
        "(renamed test); logger name 'cadw.memory_guard'; narrowed "
        "A.6 spec to 12 files; current_version 0.9.25c; A.4/A.5/A.7 "
        "→ done. "
        "fix4 (REAL production bug in v0.9.25c collect_project.py): "
        "_parse_file_spec did spec.split(',') BEFORE brace-expansion. "
        "For 'cadw/ts/{a,b,c}.py' produced ['cadw/ts/{a','b',"
        "'c}.py'] — all silently dropped. Masked by A.4/A.5/A.7 "
        "planned (plain paths) in v0.9.25c; unmasked after fix3 "
        "bookkeeping. Fix: _split_top_commas(s) — split by comma "
        "only at brace-depth 0. Unit tests for _parse_file_spec + "
        "_split_top_commas directly. E2E: --phase A = 14 files "
        "(12 A.6 + A.9 + A.10). "
        "fix5 (stale formula test in test_mustfix_easy.py): "
        "test_compute_fan_in_formula asserted pre-v0.9.25c output "
        "(= 1310 for (1_048_576, 100, 8)). v0.9.25c formula = "
        "state_limit // (3*rbs*row) — new expected 218. Fixed + "
        "added second case (100, 80): 21. Meta-guard "
        "test_no_stale_old_compute_fan_in_formula prevents regression. "
        "§13.4 META-LESSON (hard, v0.9.25c): before committing any "
        "patch that changes a utility function's output, GREP all "
        "test files for direct assertions on that function. "
        "v0.9.25c changed compute_fan_in formula; fix1-4 chased "
        "test bugs by editing e2e assertions one at a time without "
        "grepping all callers. Rule: "
        "  (a) formula/grammar refactor → unit-test in isolation "
        "FIRST, then e2e integration for wiring; "
        "  (b) 'X absent from output' assertion requires X be ONLY "
        "in skipped items AND not reachable via planned-items "
        "discovery (not the case here: tests/test_ts_models.py is "
        "pulled back via A.6 test-discovery); "
        "  (c) e2e can pass accidentally if other planned items "
        "carry same files as the broken code path. "
        "Fix-cycle count: 5/2 on v0.9.25c. All 5 cycles are tests/ "
        "+ scripts/, zero production cadw/ changes. Production cadw/ "
        "fix-cycles: 0/2 (A.4/A.5/A.7 correct since v0.9.25c initial). "
        "Tooling fix-cycles: 1/2 (collect_project.py real bug in "
        "fix4). Test-side drift unbounded but each cycle revealed "
        "deeper layer — exactly what tooling was meant to surface. "
        "Under revised rule (production vs test-side), production "
        "budget OK. "
        "Test additions: 12 new + 1 meta-guard + 1 flipped "
        "(fan_in rbs). "
        "Fast suite: ~2275 → ~2288.",
        self_path="scripts/patch_v0925c_fix5_stale_fan_in_formula.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25c+fix1-5 applied. Phase A remaining: "
          "A.6 (v0.9.25d) — 18 to_pylist sites, production cadw/ only, "
          "not touching collect_project.py. Tooling stable.")


if __name__ == "__main__":
    main()