#!/usr/bin/env python3
# v0.9.26n-fix2: verify + proper sanity + tests + commit.
# §13.4/316: python -c for-loop needs newlines, not ';'.

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

NEW_MOD = "cadw/dataframe/_aggregations.py"
CORE = "cadw/dataframe/_core.py"

TARGET_SYMBOLS = [
    "groupby", "value_counts", "corr", "nunique", "describe",
    "nlargest", "nsmallest", "isna", "isna_sum", "quantile", "count",
]


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26m":
        raise SystemExit(f"expected HEAD=v0.9.26m, got {head!r}")
    print("HEAD: " + head)

    # _aggregations.py
    if not Path(NEW_MOD).exists():
        raise SystemExit(f"{NEW_MOD} missing")
    mod_src = Path(NEW_MOD).read_text()
    if "class _AggregationsMixin" not in mod_src:
        raise SystemExit(f"_AggregationsMixin not in {NEW_MOD}")
    # Verify imports fixed
    if "from _groupby import _GroupBy" in mod_src:
        raise SystemExit("still has bad `from _groupby`")
    if "from cadw.ops.quantile import quantile\n" in mod_src:
        raise SystemExit("still has `quantile` without asname")
    if "from ._groupby import _GroupBy" not in mod_src:
        raise SystemExit("missing `from ._groupby import _GroupBy`")
    if "quantile as _ops_quantile" not in mod_src:
        raise SystemExit("missing `quantile as _ops_quantile`")
    ast.parse(mod_src, filename=NEW_MOD)
    print(f"  {NEW_MOD}: imports OK ({len(mod_src.splitlines())} lines)")

    # _core.py
    core_src = Path(CORE).read_text()
    expected = ("class CadwDataFrame(_ExportsMixin, _IteratorsMixin, "
                "_MiscMixin, _IndexingMixin, _AggregationsMixin)")
    if expected not in core_src:
        raise SystemExit(f"_core.py class: expected {expected!r}")
    if "from ._aggregations import _AggregationsMixin" not in core_src:
        raise SystemExit("_core.py: missing _AggregationsMixin import")
    print("  _core.py: 5-mixin inheritance + import OK")

    # Symbols removed from _core
    core_tree = ast.parse(core_src)
    for node in core_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CadwDataFrame":
            names = {
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef,
                                     ast.AsyncFunctionDef))
            }
            still = set(TARGET_SYMBOLS) & names
            if still:
                raise SystemExit(f"_core.py still has: {still}")
            print(f"  _core.py: {len(names)} methods remain")
            break


def main():
    print("=== v0.9.26n-fix2: verify + test + commit ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()

    # Sanity — MULTI-LINE script (§13.4/316 fix)
    print("\n--- sanity (multi-line) ---")
    sanity_script = (
        "import cadw\n"
        "C = cadw.CadwDataFrame\n"
        "names = ['groupby','value_counts','corr','nunique',\n"
        "         'describe','nlargest','nsmallest','isna',\n"
        "         'isna_sum','quantile','count']\n"
        "missing = [n for n in names if not hasattr(C, n)]\n"
        "assert not missing, f'missing: {missing}'\n"
        "print('hasattr: OK', len(names), 'symbols')\n"
    )
    r = subprocess.run(
        ["uv", "run", "python", "-c", sanity_script],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("sanity failed")
    print(r.stdout.strip())

    # Fast suite — AUTHORITATIVE
    print("\n--- fast suite (authoritative) ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")

    # Whitelist
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
        raise SystemExit("whitelist failed")
    print(r.stdout.strip())

    # Commit
    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26n: extract _aggregations (B.1 step 8, large "
        "mixin) + fix1 (imports) + fix2 (sanity). "
        "groupby/value_counts/corr/nunique/describe/nlargest/"
        "nsmallest/isna/isna_sum/quantile/count (~478 LOC, 11 "
        "symbols) moved verbatim from _core.py to _aggregations.py "
        "as _AggregationsMixin. CadwDataFrame(5 mixins). "
        "Post-write text-based lazy injection (§13.4/303/313/317); "
        "whitelist class-qualified (§13.4/318). "
        "fix1: §13.4/319 relative import level + §13.4/320 "
        "ImportFrom asname. fix2: §13.4/316 python -c "
        "for-loop newlines. 0 logic change; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26n COMPLETE (fix1 + fix2)")


if __name__ == "__main__":
    main()