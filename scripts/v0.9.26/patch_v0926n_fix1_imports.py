#!/usr/bin/env python3
# v0.9.26n-fix1: fix imports in _aggregations.py.
# §13.4/319 relative import level, §13.4/320 ImportFrom asname.
# Bootstrap-safe (no import cadw).

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

NEW_MOD = "cadw/dataframe/_aggregations.py"
CORE = "cadw/dataframe/_core.py"


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26m":
        raise SystemExit(f"expected HEAD=v0.9.26m, got {head!r}")
    print("HEAD: " + head)

    if not Path(NEW_MOD).exists():
        raise SystemExit(f"{NEW_MOD} missing")
    if not Path(CORE).exists():
        raise SystemExit(f"{CORE} missing")
    print(f"  {NEW_MOD}: present")
    print(f"  {CORE}: present")

    # Expected bad imports
    src = Path(NEW_MOD).read_text()
    bad_patterns = [
        "from _groupby import _GroupBy",       # absolute
        "from cadw.ops.quantile import quantile",  # no asname
    ]
    for pat in bad_patterns:
        if pat in src:
            print(f"  found bad: {pat!r}")


def fix_imports():
    """Fix relative + asname imports in _aggregations.py."""
    print("\n--- fix imports ---")
    p = Path(NEW_MOD)
    src = p.read_text()
    changes = []

    # Fix 1: `from _groupby import _GroupBy` -> `from ._groupby import _GroupBy`
    # (unless it already starts with `.`)
    old1 = "from _groupby import _GroupBy"
    new1 = "from ._groupby import _GroupBy"
    if old1 in src:
        src = src.replace(old1, new1)
        changes.append(f"{old1!r} -> {new1!r}")

    # Fix 2: `from cadw.ops.quantile import quantile` (no as) -> `... as _ops_quantile`
    # Code references `_ops_quantile`. Check what `_core.py` had.
    # Safest: inspect actual usage in _aggregations.py
    if "from cadw.ops.quantile import quantile" in src and "_ops_quantile" in src:
        old2 = "from cadw.ops.quantile import quantile"
        new2 = "from cadw.ops.quantile import quantile as _ops_quantile"
        src = src.replace(old2, new2)
        changes.append(f"{old2!r} -> {new2!r}")

    if not changes:
        print("  no changes needed")
        return 0

    p.write_text(src)
    ast.parse(src, filename=NEW_MOD)
    print(f"  applied {len(changes)} fix(es):")
    for c in changes:
        print(f"    {c}")
    return len(changes)


def audit_other_imports():
    """Check for remaining suspicious absolute relative imports."""
    print("\n--- audit remaining imports ---")
    src = Path(NEW_MOD).read_text()
    tree = ast.parse(src)
    issues = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        # Suspicious: level=0, module has no dot, not stdlib/cadw
        if node.level == 0 and node.module and "." not in node.module:
            # e.g. `from _groupby import` — likely relative
            if node.module.startswith("_"):
                issues.append(
                    f"line {node.lineno}: from {node.module} import "
                    f"(looks relative)"
                )
    if issues:
        print(f"  WARN: {len(issues)} suspicious:")
        for s in issues:
            print(f"    {s}")
    else:
        print("  OK")


def run_tests():
    print("\n--- sanity import ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import cadw; C = cadw.CadwDataFrame; "
         "for n in ('groupby','value_counts','corr','nunique',"
         "'describe','nlargest','nsmallest','isna','isna_sum',"
         "'quantile','count'): "
         "    assert hasattr(C, n), f'{n} missing'; "
         "print('hasattr: OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("sanity import failed")
    print(r.stdout.strip())

    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26n-fix1: fix imports ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()
    n = fix_imports()
    audit_other_imports()

    if n == 0:
        print("\nno changes needed — trying tests anyway")

    run_tests()

    # Whitelist verify
    print("\n--- verify_whitelist ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.patch_utils import verify_whitelist; "
         "verify_whitelist(raise_on_orphan=True); print('WL OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("whitelist verify failed")
    print(r.stdout.strip())

    # Commit
    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26n: extract _aggregations (B.1 step 8, large "
        "mixin) + fix1 (relative import + asname). "
        "groupby/value_counts/corr/nunique/describe/nlargest/"
        "nsmallest/isna/isna_sum/quantile/count (~478 LOC, 11 "
        "symbols) moved verbatim from _core.py to _aggregations.py "
        "as _AggregationsMixin. CadwDataFrame(5 mixins). Post-"
        "write text-based lazy injection (§13.4/303/313/317). "
        "Whitelist class-qualified rewrite (§13.4/318). "
        "fix1: §13.4/319 relative import level (`from _groupby` "
        "-> `from ._groupby`); §13.4/320 ImportFrom asname "
        "(`quantile` -> `quantile as _ops_quantile`). "
        "0 logic change; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26n COMPLETE (fix1)")


if __name__ == "__main__":
    main()