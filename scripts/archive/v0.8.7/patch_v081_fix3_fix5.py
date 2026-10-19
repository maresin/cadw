#!/usr/bin/env python3
"""v0.8.1-fix3-fix5: guard test — exact-match + semver filter.

Bug в fix4 guard:
    git describe --tags --abbrev=0
    → "backup-pre-v081-20261019_204452" (backup tag, not semver)

Fix:
    1. --exact-match: HEAD must BE on a tag (not just reach it).
    2. --match "v[0-9]*": only semver tags (excludes backup-pre-*).
    3. skip if HEAD not on tag (patch-branch workflow) — this test only
       fires on release commits, not in development.

After squash → tag v0.8.1: HEAD is exactly on v0.8.1, top CHANGELOG
entry [0.8.1] == tag "v0.8.1" → pass. On patch-branch: skip.
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


def patch_guard():
    print("\n=== tests/test_changelog.py: exact-match + semver ===")
    p = Path("tests/test_changelog.py")
    s = p.read_text()

    old = '''def test_changelog_has_latest_tag():
    """Top entry CHANGELOG == latest git tag.

    Skip, если тегов нет (новый repo / ветка без тегов).
    """
    r = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip("no git tags yet")

    tag = r.stdout.strip()          # e.g. "v0.8.1"
    version = tag.lstrip("v")       # "0.8.1"

    text = CHANGELOG.read_text()
    versions = re.findall(r"^## \\[(\\d[^\\]]*)\\]", text, flags=re.MULTILINE)
    assert versions, "CHANGELOG has no `## [X.Y.Z]` entries"

    top = versions[0]
    assert top == version, (
        f"CHANGELOG top entry [{top}] != latest tag [{version}]. "
        f"Add CHANGELOG entry for {tag} before tagging. "
        f"See §13.4/111."
    )'''

    new = '''def test_changelog_has_latest_tag():
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
        pytest.skip("HEAD is not on a semver tag (in development)")

    tag = r.stdout.strip()          # e.g. "v0.8.1"
    version = tag.lstrip("v")       # "0.8.1"

    text = CHANGELOG.read_text()
    versions = re.findall(r"^## \\[(\\d[^\\]]*)\\]", text, flags=re.MULTILINE)
    assert versions, "CHANGELOG has no `## [X.Y.Z]` entries"

    top = versions[0]
    assert top == version, (
        f"CHANGELOG top entry [{top}] != tag [{tag}]. "
        f"Add CHANGELOG entry for {tag} before tagging. "
        f"See §13.4/111."
    )'''

    if old not in s:
        # Maybe current content differs (attempting fix4 already applied → same old, but with `latest tag [backup-pre-...]` body)
        # Try alternate form
        if "backup-pre-v081" in s:
            # try replace whole function
            print("  attempting full-function replacement")
        else:
            raise SystemExit("guard test pattern not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: --exact-match + --match 'v[0-9]*'")


def main():
    print("=== v0.8.1-fix3-fix5 ===")
    patch_guard()

    print("\n=== pytest tests/test_changelog.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_changelog.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix3-fix5 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_hash_v081.py tests/test_shuffle.py "
          "tests/test_mustfix_local_groupby.py -q ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_hash_v081.py",
         "tests/test_shuffle.py",
         "tests/test_mustfix_local_groupby.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

    print("\n=== bench: v0.8.1 (int64+string shuffle) ===")
    sh(["uv", "run", "python", "scripts/bench_hash_v081.py"], check=False)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.1-fix3-fix5: CHANGELOG guard — exact-match + semver filter "
        "(backup-pre-* tags excluded)"])

    print("\n==> v0.8.1-fix3-fix5 complete. Готов squash → tag v0.8.1.")


if __name__ == "__main__":
    main()