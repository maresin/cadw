#!/usr/bin/env python3
# v0.9.3 release: CHANGELOG bump [Unreleased] -> [0.9.3], tag, verify.
#
# Same procedure as v0.9.2 release.
#   - rename first [Unreleased] to [0.9.3] -- YYYY-MM-DD
#   - prepend fresh [Unreleased]
#   - commit CHANGELOG only
#   - annotated tag v0.9.3
#   - verify test_changelog_has_latest_tag (fires post-tag)
#   - full suite fast+slow
#
# Tag covers v0.9.3 window:
#   - K1.1-K1.4 hygiene (merged from pingis/v0.9.2-audit before v0.9.3a)
#   - P-CV1 S1, P-S1 S2, P-D2 S2, C2 S2, C1 S2 (v0.9.3a-e)
#
# Run: uv run python scripts/patch_v093_release.py

from __future__ import annotations
import os, re, subprocess, sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print("  $ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print("    " + line)
    if check and r.returncode != 0:
        print("    STDERR: " + r.stderr.strip()[-1500:])
        sys.exit(r.returncode)
    return r.returncode


def bump_changelog(version, release_date):
    print("\n=== CHANGELOG.md: [Unreleased] -> [" + version + "] ===")
    p = Path("CHANGELOG.md")
    text = p.read_text()

    if "## [" + version + "]" in text:
        raise SystemExit(
            "CHANGELOG already has ## [" + version + "] — "
            "prior release left it here? Refusing to double-bump."
        )

    pattern = re.compile(r"^## \[Unreleased\]\s*\n", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise SystemExit(
            "CHANGELOG: no `## [Unreleased]` header. "
            "Run ensure_unreleased_header first."
        )

    before = text[:m.start()]
    after = text[m.end():]
    new_block = (
        "## [Unreleased]\n\n"
        "## [" + version + "] — " + release_date + "\n"
    )
    if after and not after.startswith("\n"):
        after = "\n" + after
    new_text = before + new_block + after

    p.write_text(new_text)
    print("  bumped to [" + version + "] — " + release_date)

    versions = re.findall(
        r"^## \[(\d[^\]]*)\]", new_text, flags=re.MULTILINE,
    )
    if not versions or versions[0] != version:
        raise SystemExit(
            "CHANGELOG bump sanity failed: versions[0]=" +
            str(versions[:3]) + ", expected " + repr(version)
        )
    print("  sanity: versions[0]=" + repr(versions[0]) +
          ", next 3: " + str(versions[1:4]))


def commit_release(version):
    print("\n=== git add/commit: v" + version + ": release ===")
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
        print("  WARNING: other dirty files (" + str(len(other)) + "):")
        for line in other[:10]:
            print("    " + line)
        print("  Committing ONLY CHANGELOG.md; other files stay dirty.")
    sh(["git", "commit", "-m", "v" + version + ": release"])


def tag_release(version):
    print("\n=== git tag -a v" + version + " ===")
    r = subprocess.run(
        ["git", "tag", "--list", "v" + version],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        raise SystemExit(
            "tag v" + version + " already exists. Refusing to overwrite."
        )
    msg = (
        "v" + version + ": post-audit fixes — "
        "K1.1-K1.4 hygiene (AST qualname, hash warning, join "
        "value_counts cache, roc_auc pos_label validation); "
        "P-CV1 (KFold keyed PRNG O(batch)); "
        "P-S1 (sort merge fan-in); "
        "P-D2 (dedup type-based multiplier); "
        "C2 (StdOp int truncation); "
        "C1 (dead shadowed code); "
        "5 S1/S2 RAM/correctness issues closed"
    )
    sh(["git", "tag", "-a", "v" + version, "-m", msg])


def verify_tag_guard(version):
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
        print("  Fix CHANGELOG or `git tag -d v" + version + "` and retry.")
        sys.exit(1)


def full_suite():
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
    version = "0.9.3"
    print("=== v" + version + " release ===")

    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print("  branch: " + branch)
    if branch != "master":
        raise SystemExit(
            "release must be on master (current: " + repr(branch) + ")"
        )

    r = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("  WARNING: dirty tree before bump:")
        for line in r.stdout.strip().splitlines()[:10]:
            print("    " + line)
        print("  Continuing — release commit will include only CHANGELOG.")

    release_date = date.today().isoformat()

    bump_changelog(version, release_date)
    commit_release(version)
    tag_release(version)
    verify_tag_guard(version)
    full_suite()

    print("\n==> v" + version + " released locally.")
    print("\nNEXT (manual):")
    print("  git --no-pager log --oneline -5      # verify")
    print("  git --no-pager show v" + version + " --stat")
    print("  git push origin master                # optional")
    print("  git push origin v" + version + "      # optional")


if __name__ == "__main__":
    main()
