#!/usr/bin/env python3
# v0.9.26d-fix2-recovery: extend test_roadmap whitelist (minimal, ASCII-only).

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


TESTS = "tests/test_roadmap.py"


def main():
    # 1. Verify HEAD
    head_subj = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head = head_subj.split(":", 1)[0].strip()
    if head != "v0.9.26b":
        raise SystemExit("expected HEAD=v0.9.26b, got " + repr(head))
    print("HEAD verified: " + head)

    # 2. Patch test_roadmap.py
    path = Path(TESTS)
    src = path.read_text()

    old_block = (
        '                assert item["status"] in '
        '("done", "in_progress", "planned"), ('
    )
    if old_block not in src:
        raise SystemExit("old whitelist block not found in " + TESTS)

    new_block = '''                assert item["status"] in (
                    "done", "in_progress", "planned", "deferred"
                ), (
                    f"bad status {item['status']!r} for {item['id']}"
                )
                if item["status"] == "deferred":
                    assert "deferred_to" in item, (
                        f"phase {phase['id']} item {item['id']} "
                        f"is deferred but missing deferred_to"
                    )'''

    src = src.replace(old_block, new_block, 1)
    path.write_text(src)
    compile(src, TESTS, "exec")  # syntax check
    print("patched " + TESTS)

    # 3. Docs tests
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_roadmap.py", "tests/test_changelog.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        raise SystemExit("docs tests failed")

    # 4. Full fast
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast tests failed")

    # 5. Cleanup .bak
    bak = Path("docs/roadmap.json.bak")
    if bak.exists():
        bak.unlink()
        print("removed " + str(bak))

    # 6. Commit (SHORT, ASCII-only)
    msg = (
        "v0.9.26d: Phase A closeout. A.9/A.10 done; A.6 deferred "
        "to Phase G. Test whitelist extended for deferred state. "
        "No production code."
    )
    commit_with_guards(
        msg,
        self_path="scripts/patch_v0926d_fix2_recovery.py",
        check_whitelist=True,
    )
    print("v0.9.26d committed")


if __name__ == "__main__":
    main()