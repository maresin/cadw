#!/usr/bin/env python3
"""v0.7.4c-fix: CHANGELOG guard — skip on non-master branches.

Диагностика: после squash v0.8.2 создали patch/v074c-hexbin от того же
коммита (v0.8.2 tag на HEAD). `git describe --exact-match --match
"v[0-9]*"` succeeds → тест проверяет top CHANGELOG (`[0.7.4c]`) vs tag
(`v0.8.2`) → fail.

Guard fires только в release-контексте: master + semver tag на HEAD.
На patch-ветках — skip (in-progress CHANGELOG для untagged версии —
норма).

Fix: `branch = git branch --show-current; if branch != 'master':
skip(...)`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:25]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def fix_guard():
    print("\n=== tests/test_changelog.py: skip on non-master ===")
    p = Path("tests/test_changelog.py")
    s = p.read_text()

    old = '''def test_changelog_has_latest_tag():
    """Top CHANGELOG entry == tag on HEAD (only checked when HEAD is tagged).

    Why --exact-match: patch-branch workflow has HEAD ahead of latest
    release tag; CHANGELOG legitimately has entries for versions that
    are not yet tagged. Skip is correct — this test fires on release
    commits, not during development.

    Why --match "v[0-9]*": `backup-pre-*` and other non-semver tags
    would otherwise be picked up by `git describe`. Only `vX.Y.Z`.
    """
    r = subprocess.run(
        ["git", "describe", "--tags", "--exact-match",
         "--match", "v[0-9]*"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip("HEAD is not on a semver tag (in development)")'''

    new = '''def test_changelog_has_latest_tag():
    """Top CHANGELOG entry == tag on HEAD (only checked on `master`).

    Why branch check (v0.7.4c-fix): after `git checkout -b patch/...`
    from a tagged commit (e.g., right after squash+tag v0.8.2), HEAD
    still carries the old tag. `git describe --exact-match` then
    succeeds, but CHANGELOG legitimately has an in-progress entry
    (`[0.7.4c]`) that's not yet tagged. Test would fail.

    Guard fires only in release context: **current branch == master**
    AND HEAD is on a semver tag. On patch branches — skip.

    Why --match "v[0-9]*": `backup-pre-*` and other non-semver tags
    would otherwise be picked up by `git describe`. Only `vX.Y.Z`.
    """
    # v0.7.4c-fix: only on `master`
    branch_r = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True,
    )
    current_branch = branch_r.stdout.strip()
    if current_branch != "master":
        pytest.skip(
            f"not on master (current branch: {current_branch or 'detached'}); "
            f"CHANGELOG guard fires only on master (release context)"
        )

    r = subprocess.run(
        ["git", "describe", "--tags", "--exact-match",
         "--match", "v[0-9]*"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip("master HEAD is not on a semver tag (post-merge, pre-tag)")'''

    if old not in s:
        raise SystemExit("guard pattern not found (unexpected state)")

    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: skip on non-master + skip on untagged master")


def main():
    print("=== v0.7.4c-fix ===")
    fix_guard()

    print("\n=== pytest tests/test_changelog.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_changelog.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.4c-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.4c-fix: CHANGELOG guard — skip on non-master branches "
        "(patch branch may have in-progress untagged CHANGELOG entry)"])

    print("\n==> v0.7.4c-fix complete. Готов squash → tag v0.7.4c.")


if __name__ == "__main__":
    main()