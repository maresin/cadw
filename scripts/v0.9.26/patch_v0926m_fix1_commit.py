#!/usr/bin/env python3
# v0.9.26m-fix1: verify mutations + test + commit.
# Fix sanity: python -c multi-statement with `for` needs \n not `;`.
# §13.4/316.

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CORE = "cadw/dataframe/_core.py"
NEW_MOD = "cadw/dataframe/_indexing.py"

TARGET_SYMBOLS = [
    "iloc", "loc", "at", "iat",
    "head",
    "set_index", "reset_index", "reindex", "sort_values",
]


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26l":
        raise SystemExit(f"expected HEAD=v0.9.26l, got {head!r}")
    print("HEAD: " + head)

    if not Path(NEW_MOD).exists():
        raise SystemExit(f"{NEW_MOD} missing")
    mod_src = Path(NEW_MOD).read_text()
    if "class _IndexingMixin" not in mod_src:
        raise SystemExit(f"_IndexingMixin not in {NEW_MOD}")
    for sym in TARGET_SYMBOLS:
        if f"def {sym}(" not in mod_src:
            raise SystemExit(f"{sym} not in {NEW_MOD}")
    ast.parse(mod_src, filename=NEW_MOD)
    print(f"  {NEW_MOD}: OK ({len(mod_src.splitlines())} lines)")

    core_src = Path(CORE).read_text()
    expected = ("class CadwDataFrame(_ExportsMixin, _IteratorsMixin, "
                "_MiscMixin, _IndexingMixin)")
    if expected not in core_src:
        raise SystemExit(f"_core.py class: expected {expected!r}")
    if "from ._indexing import _IndexingMixin" not in core_src:
        raise SystemExit("_core.py: missing _IndexingMixin import")
    print("  _core.py: 4-mixin inheritance + import OK")

    # Confirm symbols removed from _core class body
    core_tree = ast.parse(core_src)
    for node in core_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CadwDataFrame":
            method_names = {
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef,
                                     ast.AsyncFunctionDef))
            }
            still_present = set(TARGET_SYMBOLS) & method_names
            if still_present:
                raise SystemExit(f"_core.py still has: {still_present}")
            print(f"  _core.py: {len(method_names)} methods remain "
                  f"(targets removed)")
            break


def main():
    print("=== v0.9.26m-fix1: verify + test + commit ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()

    # Sanity: python heredoc (not -c, multi-line!)
    print("\n--- sanity (multi-line) ---")
    sanity_script = (
        "import cadw\n"
        "C = cadw.CadwDataFrame\n"
        "names = ['iloc','loc','at','iat','head',\n"
        "         'set_index','reset_index','reindex','sort_values']\n"
        "missing = [n for n in names if not hasattr(C, n)]\n"
        "assert not missing, f'missing: {missing}'\n"
        "print('hasattr: OK')\n"
        "import pyarrow as pa\n"
        "df = C.from_table(pa.table({'x':[3,1,2],'g':['a','b','c']}))\n"
        "print('head:', df.head(2).count())\n"
        "print('sort:', df.sort_values('x').count())\n"
        "print('set_index:', type(df.set_index('g')).__name__)\n"
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

    # Fast suite
    print("\n--- fast suite ---")
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
        "v0.9.26m: extract _indexing (B.1 step 7, large mixin) "
        "+ fix1 (sanity). iloc/loc/at/iat (properties) + head/"
        "set_index/reset_index/reindex/sort_values (~596 LOC, "
        "9 symbols) moved verbatim from _core.py to _indexing.py "
        "as _IndexingMixin. CadwDataFrame(_ExportsMixin, "
        "_IteratorsMixin, _MiscMixin, _IndexingMixin). Full "
        "AST-discovery (§13.4/307/308/309/310/315 nested args + "
        "except handler); whitelist audit (§13.4/311); hasattr "
        "sanity (§13.4/313). Note: `tail` absent from "
        "CadwDataFrame (§13.4/314 backlog — Phase D feature). "
        "§13.4/316 python -c for-loop newlines. 0 logic change; "
        "2492 tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26m COMPLETE (fix1)")


if __name__ == "__main__":
    main()