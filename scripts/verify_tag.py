#!/usr/bin/env python3
"""Verify git tag integrity per AI_CONSTITUTION AP31 §6.5.

Usage:
    python scripts/verify_tag.py [tag]

Without argument: verifies current working tree (status, version consistency).
With argument: also verifies tag is on HEAD, clean-checkout import works,
and CHANGELOG has entry.

Exit code: 0 if all checks pass, 1 otherwise.

Rationale: v0.9.229 session had TWO AP31 violations (v0.9.229-a missed
4 files; v0.9.229 missed uv.lock). Automate to prevent recurrence.

Sub-tag policy (DEC-078 / P81): all tags require on-HEAD (strict).
Historical sub-tags (e.g. v0.9.229-b1 after v0.9.229-b2 lands) FAIL
on-HEAD by design — intermediate checkpoints, not release tips. Do
not relax to reachability; stale tags are honest signal.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def check(name, ok, detail=""):
    marker = "OK" if ok else "FAIL"
    extra = f"  [{detail[:120]}]" if detail and not ok else ""
    print(f"  [{marker}] {name}{extra}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", nargs="?", default=None,
                        help="tag to verify (omit for working-tree only)")
    args = parser.parse_args()
    repo = Path(__file__).parent.parent
    results = []

    print(f"verify_tag: repo={repo}")
    print(f"  tag={args.tag or '(working-tree only)'}")
    print()

    # Check 1: status clean
    r = run(["git", "status", "--short"], cwd=repo)
    results.append(check("git status --short empty",
                        r.stdout.strip() == "", r.stdout))

    # Check 2: version consistency across 3 files
    r_init = run(["grep", "__version__", "cadw/__init__.py"], cwd=repo)
    ver_init = ""
    if '"' in r_init.stdout:
        ver_init = r_init.stdout.split('"')[1]
    r_py = run(["grep", "^version", "pyproject.toml"], cwd=repo)
    ver_py = ""
    if '"' in r_py.stdout:
        ver_py = r_py.stdout.split('"')[1]
    r_uv = run(["grep", "-A1", 'name = "cadw-framework"', "uv.lock"], cwd=repo)
    ver_uv = ""
    if 'version = "' in r_uv.stdout:
        try:
            ver_uv = r_uv.stdout.split('version = "')[1].split('"')[0]
        except IndexError:
            ver_uv = ""
    consistent = (ver_init == ver_py
                  and (ver_uv == "" or ver_uv == ver_init)
                  and ver_init != "")
    results.append(check(
        "version consistent (init / pyproject / uv.lock)",
        consistent,
        f"init={ver_init}, pyproject={ver_py}, uv_lock={ver_uv}"))

    if args.tag is None:
        print()
        print("PASS" if all(results) else "FAIL")
        return 0 if all(results) else 1

    # Check 3: tag on HEAD
    tag_sha = run(["git", "rev-parse", args.tag], cwd=repo).stdout.strip()
    head_sha = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    results.append(check(
        f"tag {args.tag} on HEAD",
        tag_sha == head_sha and tag_sha != "",
        f"tag={tag_sha[:8]}, HEAD={head_sha[:8]}"))

    # Check 4: CHANGELOG entry exists
    version_only = args.tag.lstrip("v").split("-")[0]
    # Use -F (fixed string): [0.9.229] is literal in CHANGELOG,
    # not regex character class.
    r = run(["grep", "-cF", f"## [{version_only}]", "CHANGELOG.md"], cwd=repo)
    has_entry = r.stdout.strip() not in ("0", "")
    results.append(check(
        f"CHANGELOG has ## [{version_only}]",
        has_entry, r.stdout.strip()))

    # Check 5: clean-checkout import
    with tempfile.TemporaryDirectory() as tmp:
        tmpp = Path(tmp) / "co"
        r = run(["git", "clone", "--quiet", str(repo), str(tmpp)], cwd=repo)
        if r.returncode != 0:
            results.append(check("git clone", False, r.stderr[:200]))
        else:
            r = run(["git", "checkout", "--quiet", args.tag], cwd=tmpp)
            if r.returncode != 0:
                results.append(check("git checkout tag", False,
                                     r.stderr[:200]))
            else:
                env = {**os.environ, "PYTHONPATH": str(tmpp)}
                code = (
                    "import cadw; "
                    f"assert cadw.__version__ == {ver_init!r}, "
                    "f'version {cadw.__version__} != " + ver_init + "'; "
                    "import cadw.ml as m; "
                    "assert 'fit_ordered_probit' in m.__all__; "
                    "print('CLEAN_IMPORT_OK')"
                )
                r = run([sys.executable, "-c", code], cwd=tmpp, env=env)
                ok = (r.returncode == 0
                      and "CLEAN_IMPORT_OK" in r.stdout)
                results.append(check(
                    "clean-checkout import", ok, r.stderr[:200]))

    print()
    print("PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
