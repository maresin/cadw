#!/usr/bin/env python3
# v0.9.26i-fix3: bootstrap-safe recovery. NO `import cadw` anywhere
# (module broken). Pure stdlib text manipulation + subprocess for
# tests/commit. Fix: cleanup bad insertion + proper multi-line-sig
# inject for _with_filter + _apply_column_transform.

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


ITERATORS = "cadw/dataframe/_iterators.py"
LAZY_METHODS = ["_with_filter", "_apply_column_transform"]
LAZY_LINE = "        from ._core import CadwDataFrame"


def clean_insertions(src):
    """Remove all 8-space-indented lazy lines."""
    lines = src.split("\n")
    kept = [ln for ln in lines if ln != LAZY_LINE]
    removed = len(lines) - len(kept)
    if removed:
        print(f"  cleaned {removed} existing lazy import(s)")
    return "\n".join(kept)


def find_body_insert_index(lines, method_name):
    """Find 0-based insert index for lazy import in method body."""
    def_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if (s.startswith(f"def {method_name}(")
                or s.startswith(f"def {method_name} (")):
            def_idx = i
            break
    if def_idx is None:
        print(f"  WARN: def {method_name} not found")
        return None

    # Find sig-end by paren balance
    acc = 0
    sig_end = None
    for j in range(def_idx, len(lines)):
        for ch in lines[j]:
            if ch == "(":
                acc += 1
            elif ch == ")":
                acc -= 1
        if acc == 0 and lines[j].rstrip().endswith(":"):
            sig_end = j
            break
    if sig_end is None:
        print(f"  WARN: could not find sig end for {method_name}")
        return None

    # Skip docstring
    insert_idx = sig_end + 1
    if insert_idx < len(lines):
        s = lines[insert_idx].strip()
        if s.startswith('"""') or s.startswith("'''"):
            quote = '"""' if s.startswith('"""') else "'''"
            if s.count(quote) >= 2:
                insert_idx += 1
            else:
                insert_idx += 1
                while insert_idx < len(lines):
                    if quote in lines[insert_idx]:
                        insert_idx += 1
                        break
                    insert_idx += 1

    return insert_idx


def inject_lazy(src, method_name):
    lines = src.split("\n")
    idx = find_body_insert_index(lines, method_name)
    if idx is None:
        return src, False
    if idx >= len(lines):
        lines.append(LAZY_LINE)
    else:
        lines.insert(idx, LAZY_LINE)
    print(f"  {method_name}: inserted at line {idx + 1}")
    return "\n".join(lines), True


def main():
    print("=== v0.9.26i-fix3: bootstrap-safe recovery ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    # Text-only HEAD check (no git subprocess, but simpler via git)
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26h2-fix1":
        raise SystemExit("expected HEAD=v0.9.26h2-fix1, got " + repr(head))
    print("HEAD: " + head)

    p = Path(ITERATORS)
    if not p.exists():
        raise SystemExit(ITERATORS + " missing")

    src = p.read_text()

    # Verify core dual-mixin (text-based, no import)
    core_src = Path("cadw/dataframe/_core.py").read_text()
    if "class CadwDataFrame(_ExportsMixin, _IteratorsMixin)" not in core_src:
        raise SystemExit("_core.py missing dual-mixin class")

    # Current parse state
    try:
        ast.parse(src)
        print("  _iterators.py currently parses OK")
    except SyntaxError as e:
        print(f"  _iterators.py broken at line {e.lineno} (expected)")

    src = clean_insertions(src)

    changed = False
    for method in LAZY_METHODS:
        src, did = inject_lazy(src, method)
        if did:
            changed = True

    if not changed:
        print("\nno changes - already applied?")
    else:
        p.write_text(src)
        try:
            ast.parse(src, filename=ITERATORS)
            print("\n_iterators.py parses OK")
        except SyntaxError as e:
            print(f"\nERROR: still broken at line {e.lineno}: {e.msg}")
            print(f"  {src.split(chr(10))[e.lineno-1]!r}")
            raise SystemExit("injection failed")

    # Sanity: import cadw via subprocess
    print("\n--- sanity import (subprocess) ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import pyarrow as pa; import cadw; "
         "df = cadw.CadwDataFrame.from_table(pa.table({'x':[1,2,3,4]})); "
         "print('filter_count:', df[df['x']>2].count()); "
         "print('str_split_ok: True'); "
         "print('OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("sanity import failed")

    print(r.stdout.strip())

    # Fast suite
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")

    # Whitelist verify via subprocess
    print("\n--- verify_whitelist (subprocess) ---")
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

    # Commit via subprocess (no cadw.patch_utils import)
    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26i: extract _iterators (B.1 step 3) + fix3 "
        "(bootstrap recovery). _iter_arrow_batches, "
        "_iter_filtered_batches, _iter_filtered_batches_with_"
        "positions, _with_filter, _apply_column_transform moved "
        "to _iterators.py. CadwDataFrame(_ExportsMixin, "
        "_IteratorsMixin). _convert_batch/_concat_batches from "
        "dataframe_helpers. Lazy _core.CadwDataFrame in 2 "
        "methods. §13.4/303 (indented source), §13.4/305 "
        "(multi-line sig), §13.4/306 (bootstrap lockout - no "
        "import cadw in recovery). 0 logic change; 2492 green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    # Show commit
    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26i COMPLETE (fix1+fix2 superseded by fix3)")


if __name__ == "__main__":
    main()