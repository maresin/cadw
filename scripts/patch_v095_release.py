#!/usr/bin/env python3
# v0.9.5 release: CHANGELOG bump [Unreleased] -> [0.9.5], tag, verify.
# Run: uv run python scripts/patch_v095_release.py

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


def bump(version, release_date):
    print("\n=== CHANGELOG: [Unreleased] -> [" + version + "] ===")
    p = Path("CHANGELOG.md")
    text = p.read_text()
    if "## [" + version + "]" in text:
        raise SystemExit("CHANGELOG already has [" + version + "]")
    pat = re.compile(r"^## \[Unreleased\]\s*\n", re.MULTILINE)
    m = pat.search(text)
    if not m:
        raise SystemExit("no [Unreleased] header")
    before = text[:m.start()]
    after = text[m.end():]
    block = "## [Unreleased]\n\n## [" + version + "] — " + release_date + "\n"
    if after and not after.startswith("\n"):
        after = "\n" + after
    new_text = before + block + after
    p.write_text(new_text)
    versions = re.findall(r"^## \[(\d[^\]]*)\]", new_text, re.MULTILINE)
    if not versions or versions[0] != version:
        raise SystemExit("bump sanity failed: " + str(versions[:3]))
    print("  bumped to [" + version + "]; top 3: " + str(versions[:3]))


def commit_release(version):
    print("\n=== commit v" + version + ": release ===")
    sh(["git", "add", "CHANGELOG.md"])
    sh(["git", "commit", "-m", "v" + version + ": release"])


def tag_release(version):
    print("\n=== tag -a v" + version + " ===")
    r = subprocess.run(["git", "tag", "--list", "v" + version],
                       capture_output=True, text=True)
    if r.stdout.strip():
        raise SystemExit("tag v" + version + " already exists")
    msg = (
        "v" + version + ": hygiene — patch_utils.verify_whitelist "
        "(AST orphan detector for WHITELISTED_FUNCS, recursive "
        "qualname); commit_with_guards(check_whitelist=True) hook; "
        "P-SH2 (uint64 silent fallback -> warning); C6 (AST guard "
        "extends to viz/_aggregate.py + core/memory_guard.py)"
    )
    sh(["git", "tag", "-a", "v" + version, "-m", msg])


def verify_tag_guard():
    print("\n=== test_changelog post-tag ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py::test_changelog_has_latest_tag",
         "tests/test_changelog.py::test_changelog_is_descending", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** TAG GUARD FAILED ***")
        sys.exit(1)


def full_suite():
    print("\n=== fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        sys.exit(1)
    print("\n=== slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        sys.exit(1)


def main():
    version = "0.9.5"
    print("=== v" + version + " release ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())
    if r.stdout.strip() != "master":
        raise SystemExit("must be on master")
    bump(version, date.today().isoformat())
    commit_release(version)
    tag_release(version)
    verify_tag_guard()
    full_suite()
    print("\n==> v" + version + " released locally.")
    print("\nNEXT (manual):")
    print("  git --no-pager log --oneline -5")
    print("  git push origin master")
    print("  git push origin v" + version)


if __name__ == "__main__":
    main()
