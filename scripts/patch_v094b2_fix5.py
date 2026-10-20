#!/usr/bin/env python3
# v0.9.4b-2-fix5: qualname requires Class.method prefix.
#
# fix4 inserted "cadw/dataframe.py:describe" (bare method name).
# Guard uses {path}:{ClassName}.{method_name} qualname (K1.1) ->
# violation reported as "CadwDataFrame.describe" -> bare "describe" no match.
#
# Fix: str.replace exact key string. One-line diff.
#
# Run: uv run python scripts/patch_v094b2_fix5.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


def fix_qualname():
    print("\n=== test_arrow_call_memory_guard.py: qualname fix ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    old = '"cadw/dataframe.py:describe"'
    new = '"cadw/dataframe.py:CadwDataFrame.describe"'
    if new in s:
        print("  already fixed - skip")
        return
    if old not in s:
        raise SystemExit("bare 'cadw/dataframe.py:describe' key not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK: " + old + " -> " + new)


def main():
    print("=== v0.9.4b-2-fix5 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_qualname()

    print("\n=== pytest: AST guard ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** AST GUARD STILL RED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    commit_with_guards(
        "v0.9.4b-2: transform/describe/dt/str (fix1-fix5); "
        "whitelist key 'cadw/dataframe.py:CadwDataFrame.describe' "
        "(K1.1 qualname format: Class.method)",
    )
    print("\n==> v0.9.4b-2 complete.")


if __name__ == "__main__":
    main()
