#!/usr/bin/env python3
# v0.9.25c-fix2 — --phase semantics + logger name (2 test-side bugs).
#
# Bug 1: test_collect_project_phase_mode expected "FILE: cadw/ts/models.py"
#   in --phase A snapshot. But _extract_phase_files correctly skips
#   `done` items (--phase = "current work", not "history"). ts/models.py
#   in phase A is only in done items (A.1, A.2, A.3, A.3b, A.8).
#   Remaining (planned) A work touches ops/sort.py (A.4/A.7/A.9),
#   dataframe.py (A.5/A.6), core/memory_guard.py (A.10). Test was
#   wrong about the contract; code is right.
#
# Bug 2: v0.9.25c-fix1 dogfood suppressed wrong logger name:
#   getLogger("cadw.core.memory_guard") — but memory_guard.py uses
#   get_logger("memory_guard") = getLogger("cadw.memory_guard").
#   So "check_state FAIL: ..." log noise was NOT suppressed in v0.9.25c
#   output. Fix: correct logger name.
#
# This is fix-cycle #2 on v0.9.25c. Both bugs are test-side, not
# production. Per §13.4 budget rule (<=2 fix-cycles), we close here
# without splitting. Meta-lesson: --phase = "current work" contract
# must be asserted with planned-item files, not done-item files.
#
# Run: uv run python scripts/patch_v0925c_fix2_phase_semantics.py

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


# ============ 0. Verify fix1 state ============

def verify_state():
    print("\n=== verify v0.9.25c + fix1 state on disk ===")
    checks = [
        ("tests/test_v0925c_phase_a.py", "textwrap.dedent(inspect.getsource"),
        ("scripts/collect_project.py", "import subprocess\nimport sys as _sys"),
        ("tests/test_collect_project_modes.py",
         "def test_collect_project_no_args_uses_legacy_snapshot"),
        ("scripts/patch_v0925c_fix1_test_bugs.py",
         'getLogger("cadw.core.memory_guard")'),  # bug 2 anchor
    ]
    for path, anchor in checks:
        if anchor not in Path(path).read_text():
            raise SystemExit(
                f"{path}: missing {anchor!r}. Re-run v0.9.25c-fix1."
            )
    print("  all anchors present — OK")


# ============ 1. Fix --phase mode test (semantics) ============

OLD_PHASE_TEST = '''def test_collect_project_phase_mode(tmp_path):
    out = tmp_path / "phase_a.md"
    r = _run("--phase", "A", "--out", str(out))
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    # Phase A items include cadw/ts/models.py and cadw/ops/sort.py
    assert "FILE: cadw/ts/models.py" in text
    assert "FILE: cadw/ops/sort.py" in text
    # Dataframe.py has A.5 (encode_cat)
    assert "FILE: cadw/dataframe.py" in text
'''

NEW_PHASE_TEST = '''def test_collect_project_phase_mode(tmp_path):
    """v0.9.25c-fix2: --phase A = planned items only, not done.

    --phase is for "current work in this phase", not "everything that
    was ever touched by phase A". Done items (A.1-A.3b, A.8) are
    excluded. Remaining A work touches:
      - ops/sort.py (A.4 refill, A.7 fan_in, A.9 dead code)
      - dataframe.py (A.5 encode_cat, A.6 to_pylist sites)
      - core/memory_guard.py (A.10 fallback factor)
    """
    out = tmp_path / "phase_a.md"
    r = _run("--phase", "A", "--out", str(out))
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    # Planned items present.
    assert "FILE: cadw/ops/sort.py" in text
    assert "FILE: cadw/dataframe.py" in text
    assert "FILE: cadw/core/memory_guard.py" in text
    # Done items excluded (A.1-A.3b, A.8 touch cadw/ts/models.py).
    assert "FILE: cadw/ts/models.py" not in text, (
        "--phase A must exclude done items; cadw/ts/models.py is "
        "only in done A.1-A.3b + A.8"
    )
'''

def fix_phase_test():
    print("\n=== fix: test_collect_project_phase_mode (planned-only) ===")
    applied = apply_or_fail(
        "tests/test_collect_project_modes.py",
        OLD_PHASE_TEST, NEW_PHASE_TEST,
        label="phase_test_semantics",
    )
    if applied:
        assert_compiles("tests/test_collect_project_modes.py")
        check_no_invalid_escape("tests/test_collect_project_modes.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 2. Fix logger name in dogfood fixture ============
# The v0.9.25c-fix1 dogfood is baked into patch_v0925c_fix1_test_bugs.py.
# We don't re-run that script, but if user re-runs it (idempotent — fixed
# by is_patched in v0.9.6d), we want correct name. Additionally, the
# CURRENT run of fix2 will do its own dogfood with correct name.
#
# Fix the historical file so the artifact is correct:
OLD_LOGGER_NAME = '_logging.getLogger("cadw.core.memory_guard").setLevel(_logging.CRITICAL)'
NEW_LOGGER_NAME = (
    '# v0.9.25c-fix2: correct logger name is "cadw.memory_guard"\n'
    '        # (get_logger("memory_guard") = getLogger("cadw.memory_guard")).\n'
    '        # Prior v0.9.25c-fix1 used "cadw.core.memory_guard" — no-op.\n'
    '        _logging.getLogger("cadw.memory_guard").setLevel(_logging.CRITICAL)'
)

def fix_logger_name():
    print("\n=== fix: dogfood logger name in patch_v0925c_fix1 ===")
    applied = apply_or_fail(
        "scripts/patch_v0925c_fix1_test_bugs.py",
        OLD_LOGGER_NAME, NEW_LOGGER_NAME,
        label="logger_name",
    )
    if applied:
        assert_compiles("scripts/patch_v0925c_fix1_test_bugs.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 3. Dogfood (clean, correct logger) ============

def dogfood():
    print("\n=== dogfood: --phase semantics + clean logging ===")

    def _verify():
        import logging as _logging
        # Correct logger name (v0.9.25c-fix2).
        _logging.getLogger("cadw.memory_guard").setLevel(_logging.CRITICAL)

        import subprocess as _sp
        import sys as _sys
        import tempfile
        from pathlib import Path as _P

        with tempfile.TemporaryDirectory() as td:
            out = _P(td) / "phase_a.md"
            r = _sp.run(
                [_sys.executable, "scripts/collect_project.py",
                 "--phase", "A", "--out", str(out)],
                cwd=".", capture_output=True, text=True,
            )
            assert r.returncode == 0, r.stderr
            text = out.read_text()
            # Planned A items present.
            assert "FILE: cadw/ops/sort.py" in text
            assert "FILE: cadw/dataframe.py" in text
            assert "FILE: cadw/core/memory_guard.py" in text
            # Done A items excluded.
            assert "FILE: cadw/ts/models.py" not in text, (
                "--phase A must not include done A.1-A.3b files"
            )
            # Size sanity: not the full 27-file snapshot.
            size_kb = out.stat().st_size / 1024
            assert size_kb < 100, (
                f"--phase A produced {size_kb:.1f} KiB — too large, "
                f"likely including done items or wildcards"
            )

        print(
            f"  --phase A: planned-only, "
            f"{size_kb:.1f} KiB (excludes done ts/models.py)"
        )

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

    print("\n=== pytest: targeted (v0925c + collect) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925c_phase_a.py",
         "tests/test_collect_project_modes.py",
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
    print("=== v0.9.25c-fix2: --phase semantics + logger name ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    fix_phase_test()
    fix_logger_name()
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
        "NOT _WRITE_RUN_CHUNK. Root cause of 24.4s @ N=500k/8cols/32MB "
        "— v0.9.23c _merge_keys uses rbs=1000 but fan_in got 50000 → "
        "K=2. Formula: 3*K*rbs*row <= state_limit. max_k=256 FD cap. "
        "Now K=8 @ 32MB/8cols/640B/row. "
        "B.4 (docs/roadmap.json): collect_project.py --module / "
        "--phase / --doc / --list-phases. Reads docs/roadmap.json. "
        "Brace-expansion for composite file specs. "
        "fix1 (3 test/tooling bugs): (1) cleandoc on method source "
        "→ IndentationError; textwrap.dedent. (2) missing "
        "`import subprocess` in collect_project.py. (3) "
        "test_collect_project_no_args_errors asserted error — but "
        "no new-mode flags falls to legacy main() (smart snapshot). "
        "Renamed → _uses_legacy_snapshot. "
        "fix2 (2 test-side bugs, closes fix-cycle #2): "
        "(1) test_collect_project_phase_mode expected cadw/ts/models.py "
        "in --phase A snapshot, but _extract_phase_files correctly "
        "skips done items (--phase = 'current work', not 'history'). "
        "ts/models.py is only in done A.1-A.3b + A.8. "
        "Assert planned-only: ops/sort.py (A.4/A.7/A.9), "
        "dataframe.py (A.5/A.6), core/memory_guard.py (A.10); "
        "assert ts/models.py excluded. "
        "(2) fix1 dogfood suppressed wrong logger name "
        "(cadw.core.memory_guard vs real cadw.memory_guard — "
        "get_logger('memory_guard') = getLogger('cadw.memory_guard')); "
        "log noise now properly suppressed. "
        "Test additions: 10 new + 1 flipped (fan_in rbs). "
        "Dogfood: 4 items + --phase A planned-only verified. "
        "Fix-cycle count: 2/2 on v0.9.25c (§13.4 budget). "
        "Fast suite: ~2275 → ~2285.",
        self_path="scripts/patch_v0925c_fix2_phase_semantics.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25c+fix1+fix2 applied. Next: v0.9.25d "
          "(A.6: 18 to_pylist sites extra_factor + "
          "test_no_cleandoc_on_getsource meta-guard).")


if __name__ == "__main__":
    main()