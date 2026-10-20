#!/usr/bin/env python3
"""v0.9.2 release: CHANGELOG bump [Unreleased] → [0.9.2], tag, verify.

WHY release script, not bare `git tag v0.9.2`:
  - test_changelog_has_latest_tag (v0.7.4c-fix, §13.4/111) checks
    versions[0] from `^## \\[\\d...\\]` regex against tag. Currently top
    version = "0.9.1" (v0.9.2x entries live under [Unreleased], which
    does NOT match the regex — 'U' ≠ digit). Bare tag → test fails.
  - Release convention (pandas-style): rename [Unreleased] → [X.Y.Z],
    prepend fresh [Unreleased] for future work.
  - pyproject.toml version stays 0.1.0 (git tags are source of truth;
    pyproject.version unused at runtime — no `__version__` in cadw/).

STEPS:
  1. Rename first `## [Unreleased]` → `## [0.9.2] — YYYY-MM-DD`.
  2. Prepend fresh `## [Unreleased]\\n\\n` for future work.
  3. Commit `v0.9.2: release`.
  4. `git tag -a v0.9.2 -m "..."` on that commit.
  5. Run test_changelog.py — post-tag guard fires; must pass.
  6. Full suite (fast + slow) — sanity.
  7. Print push commands (do NOT auto-push — user decides).

NOT INCLUDED:
  - pyproject.toml version bump (0.1.0 → 0.9.2). Not needed for tag
    guard; would require `uv lock` regen. Out of scope of release tag.
  - push to remote (manual).

GUARDS: full suite + test_changelog.py + tag guard.

Run: uv run python scripts/patch_v092_release.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def bump_changelog(version: str, release_date: str) -> None:
    print(f"\n=== CHANGELOG.md: [Unreleased] → [{version}] ===")
    p = Path("CHANGELOG.md")
    text = p.read_text()

    if f"## [{version}]" in text:
        raise SystemExit(
            f"CHANGELOG already has ## [{version}] — "
            f"previous release left it here? Refusing to double-bump."
        )

    pattern = re.compile(r"^## \[Unreleased\]\s*\n", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise SystemExit(
            "CHANGELOG: no `## [Unreleased]` header. "
            "Run ensure_unreleased_header first (or add manually)."
        )

    # Replace `## [Unreleased]\n` (first occurrence) with:
    #   `## [Unreleased]\n\n## [X.Y.Z] — YYYY-MM-DD\n`
    before = text[:m.start()]
    after = text[m.end():]
    # after starts with `\n` or content like `### Added`
    new_block = (
        f"## [Unreleased]\n\n"
        f"## [{version}] — {release_date}\n"
    )
    # Ensure a blank line before the previous content (which was after
    # the [Unreleased] header — likely `\n### Added` or `\n## [older]`).
    if after and not after.startswith("\n"):
        after = "\n" + after
    new_text = before + new_block + after

    p.write_text(new_text)
    print(f"  bumped to [{version}] — {release_date}")
    print(f"  fresh [Unreleased] header prepended")

    # Sanity: top version-pattern match should be `version`.
    versions = re.findall(
        r"^## \[(\d[^\]]*)\]", new_text, flags=re.MULTILINE,
    )
    if not versions or versions[0] != version:
        raise SystemExit(
            f"CHANGELOG bump sanity failed: versions[0]={versions[:3]}, "
            f"expected {version!r}"
        )
    print(f"  sanity: versions[0]={versions[0]!r}, "
          f"next 3: {versions[1:4]}")


def commit_release(version: str) -> None:
    print(f"\n=== git add/commit: v{version}: release ===")
    sh(["git", "add", "CHANGELOG.md"])
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True,
    )
    other = [
        l for l in r.stdout.splitlines()
        if l.strip() and not l.startswith("A  CHANGELOG.md")
    ]
    if other:
        print(f"  WARNING: other dirty files ({len(other)}):")
        for line in other[:10]:
            print(f"    {line}")
        print("  Committing ONLY CHANGELOG.md; other files stay dirty.")
    sh(["git", "commit", "-m", f"v{version}: release"])


def tag_release(version: str) -> None:
    print(f"\n=== git tag -a v{version} ===")
    # Reject if tag exists locally
    r = subprocess.run(
        ["git", "tag", "--list", f"v{version}"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        raise SystemExit(
            f"tag v{version} already exists. Refusing to overwrite."
        )
    msg = (
        f"v{version}: metrics (regression/classification/roc_auc) + "
        f"CV (KFold/TimeSeriesSplit/cv_score) + scalers "
        f"(Standard/MinMax/MaxAbs/Robust)"
    )
    sh(["git", "tag", "-a", f"v{version}", "-m", msg])


def verify_tag_guard(version: str) -> None:
    print("\n=== verify: test_changelog_has_latest_tag (post-tag) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py::test_changelog_has_latest_tag",
         "tests/test_changelog.py::test_changelog_is_descending",
         "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** TAG GUARD FAILED ***")
        print("  Local tag exists; CHANGELOG is inconsistent.")
        print("  Fix CHANGELOG or `git tag -d v" + version + "` and retry.")
        sys.exit(1)


def full_suite() -> None:
    print("\n=== full suite (fast) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== full suite (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


def main():
    version = "0.9.2"
    print(f"=== v{version} release ===")

    # Pre-flight: on master, clean tree (except expected untracked).
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if branch != "master":
        raise SystemExit(
            f"release must be on master (current: {branch!r})"
        )

    r = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("  WARNING: dirty tree before bump:")
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
        print("  Continuing anyway — release commit will include only CHANGELOG.")

    release_date = date.today().isoformat()

    bump_changelog(version, release_date)
    commit_release(version)
    tag_release(version)
    verify_tag_guard(version)
    full_suite()

    print(f"\n==> v{version} released locally.")
    print("\nNEXT (manual):")
    print(f"  git log --oneline -3           # verify")
    print(f"  git show v{version} --stat     # verify tag")
    print(f"  git push origin master         # optional, when ready")
    print(f"  git push origin v{version}     # optional, when ready")


if __name__ == "__main__":
    main()