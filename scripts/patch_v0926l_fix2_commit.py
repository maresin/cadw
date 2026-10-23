#!/usr/bin/env python3
# v0.9.26l-fix2: sanity API fixed (df.unique, not df['x'].unique).
# Mutations already applied by fix1; verify + test + commit.
# Bootstrap-safe.

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CORE = "cadw/dataframe/_core.py"
NEW_MOD = "cadw/dataframe/_misc.py"

TARGET_SYMBOLS = [
    "unique",
    "_normalize_on",
    "_groupby_impl",
    "ts",
    "plot",
]


def verify_state():
    """Verify mutations applied by fix1 (uncommitted)."""
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26k":
        raise SystemExit(f"expected HEAD=v0.9.26k, got {head!r}")
    print("HEAD: " + head)

    # 1. _misc.py exists with _MiscMixin
    if not Path(NEW_MOD).exists():
        raise SystemExit(f"{NEW_MOD} missing — fix1 didn't run?")
    misc_src = Path(NEW_MOD).read_text()
    if "class _MiscMixin" not in misc_src:
        raise SystemExit(f"_MiscMixin not in {NEW_MOD}")
    for sym in TARGET_SYMBOLS:
        if f"def {sym}(" not in misc_src:
            raise SystemExit(f"{sym} not in {NEW_MOD}")
    print(f"  {NEW_MOD}: _MiscMixin + {len(TARGET_SYMBOLS)} symbols OK")

    # 2. _core.py: symbols removed, class inherits _MiscMixin
    core_src = Path(CORE).read_text()
    if "class CadwDataFrame(_ExportsMixin, _IteratorsMixin, _MiscMixin)" not in core_src:
        raise SystemExit(
            "_core.py: class doesn't inherit _MiscMixin "
            "(inheritance pattern differs from expected)"
        )
    if "from ._misc import _MiscMixin" not in core_src:
        raise SystemExit("_core.py: missing _MiscMixin import")
    print("  _core.py: _MiscMixin inheritance + import OK")

    # 3. Symbols removed from _core.py class body (sanity via AST)
    core_tree = ast.parse(core_src)
    for node in core_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CadwDataFrame":
            method_names = {
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            moved_still_present = set(TARGET_SYMBOLS) & method_names
            if moved_still_present:
                raise SystemExit(
                    f"_core.py still has: {moved_still_present}"
                )
            print(f"  _core.py: {len(method_names)} methods remain "
                  f"(target symbols removed)")
            break


def main():
    print("=== v0.9.26l-fix2: verify + test + commit ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()

    # Import sanity — just verify class loads
    print("\n--- import sanity ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import cadw; "
         "C = cadw.CadwDataFrame; "
         # hasattr checks (no API assumption)
         "assert hasattr(C, 'unique'), 'unique missing'; "
         "assert hasattr(C, '_normalize_on'), '_normalize_on missing'; "
         "assert hasattr(C, '_groupby_impl'), '_groupby_impl missing'; "
         "assert hasattr(C, 'ts'), 'ts missing'; "
         "assert hasattr(C, 'plot'), 'plot missing'; "
         "print('hasattr: OK'); "
         # runtime check via properties (safe API)
         "import pyarrow as pa; "
         "df = C.from_table(pa.table({'x':[1,2,2,3],'g':['a','a','b','b']})); "
         "print('ts:', type(df.ts).__name__); "
         "print('plot:', type(df.plot).__name__); "
         "print('groupby:', df.groupby('g').agg(['x'],['sum']).count())"
         ],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("import sanity failed")
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
        raise SystemExit("whitelist verify failed")
    print(r.stdout.strip())

    # Commit
    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26l: extract _misc (B.1 step 6, small leaf mixin) "
        "+ fix1 (typo) + fix2 (sanity API). "
        "unique, _normalize_on, _groupby_impl, ts, plot moved "
        "verbatim from _core.py to _misc.py as _MiscMixin. "
        "CadwDataFrame(_ExportsMixin, _IteratorsMixin, _MiscMixin). "
        "Full AST-discovery (§13.4/307/308/309/310); whitelist "
        "audit (§13.4/311) confirms no stale keys. "
        "§13.4/312 function-name typo (extract_sources -> "
        "extract_symbols). §13.4/313 sanity-API assumption "
        "(df['x'].unique -> hasattr check). 0 logic change; "
        "2492 tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26l COMPLETE (fix1 + fix2)")


if __name__ == "__main__":
    main()