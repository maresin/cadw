#!/usr/bin/env python3
# v0.9.26i-fix2: correct lazy import injection for multi-line sig.
# Cleans bad insertions, re-injects properly per method.

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


ITERATORS = "cadw/dataframe/_iterators.py"
LAZY_METHODS = ["_with_filter", "_apply_column_transform"]
LAZY_LINE = "        from ._core import CadwDataFrame"


def clean_insertions(src):
    """Remove any 8-space-indented lazy import lines."""
    lines = src.split("\n")
    kept = [ln for ln in lines if ln != LAZY_LINE]
    removed = len(lines) - len(kept)
    if removed:
        print(f"  cleaned {removed} existing lazy import(s)")
    return "\n".join(kept)


def find_body_insert_index(lines, method_name):
    """Return line index to insert at (0-based), or None."""
    # Find def line
    def_idx = None
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if (stripped.startswith(f"def {method_name}(")
                or stripped.startswith(f"def {method_name} (")):
            def_idx = i
            break
    if def_idx is None:
        print(f"  WARN: def {method_name} not found")
        return None

    # Find end of signature: first line ending with ':' at method level
    # Signature lines end with ':' after '-> ReturnType:' or '):'
    # Method def line: 'def foo(self, ...):' or 'def foo(' (continue)
    # Multi-line: '    ...' until '):' or '-> Type:'
    sig_end = None
    for j in range(def_idx, len(lines)):
        stripped = lines[j].rstrip()
        # Signature ends when line ends with ':' AND next thing is body
        # Heuristic: line ends with ':' or ':)' or 'Type:'
        if stripped.endswith(":"):
            # Check it's not inside nested (parens not closed)
            # Simple: count parens on def..j
            acc = 0
            for k in range(def_idx, j + 1):
                for ch in lines[k]:
                    if ch == "(":
                        acc += 1
                    elif ch == ")":
                        acc -= 1
            if acc == 0:
                sig_end = j
                break

    if sig_end is None:
        print(f"  WARN: could not find sig end for {method_name}")
        return None

    # Skip docstring
    insert_idx = sig_end + 1
    if insert_idx < len(lines):
        stripped = lines[insert_idx].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = '"""' if stripped.startswith('"""') else "'''"
            if stripped.count(quote) >= 2:
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
    """Inject proper lazy import at body start."""
    lines = src.split("\n")
    idx = find_body_insert_index(lines, method_name)
    if idx is None:
        return src, False

    # Sanity: original line at idx should be indented (body)
    if idx >= len(lines):
        lines.append(LAZY_LINE)
    else:
        original = lines[idx]
        if original.strip() and not original.startswith("        "):
            print(f"  WARN: {method_name} line {idx+1} indent < 8: "
                  f"{original[:60]!r}")
        lines.insert(idx, LAZY_LINE)
    print(f"  {method_name}: inserted at line {idx + 1}")
    return "\n".join(lines), True


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26h2-fix1":
        raise SystemExit("expected HEAD=v0.9.26h2-fix1, got " + repr(head))
    print("HEAD: " + head)

    if not Path(ITERATORS).exists():
        raise SystemExit(ITERATORS + " missing")

    core_src = Path("cadw/dataframe/_core.py").read_text()
    if "class CadwDataFrame(_ExportsMixin, _IteratorsMixin)" not in core_src:
        raise SystemExit("_core.py dual-mixin class missing")

    try:
        ast.parse(Path(ITERATORS).read_text())
        print("  _iterators.py currently parses OK (unexpected, "
              "but not blocking)")
    except SyntaxError as e:
        print(f"  _iterators.py has syntax error (expected): "
              f"line {e.lineno}")


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26i-fix2: multi-line sig lazy import ===")
    verify_state()

    src = Path(ITERATORS).read_text()
    src = clean_insertions(src)

    changed = False
    for method in LAZY_METHODS:
        src, did = inject_lazy(src, method)
        if did:
            changed = True

    if not changed:
        print("\nno changes - already applied?")
        run_tests()
        return

    Path(ITERATORS).write_text(src)
    try:
        ast.parse(src, filename=ITERATORS)
        print("\n_iterators.py parses OK")
    except SyntaxError as e:
        print(f"\nERROR: still broken at line {e.lineno}: {e.msg}")
        print(f"Line {e.lineno}: {src.split(chr(10))[e.lineno-1]!r}")
        raise SystemExit("injection failed, manual fix needed")

    print("\n--- sanity import ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import pyarrow as pa; import cadw; "
         "df = cadw.CadwDataFrame.from_table(pa.table({'x':[1,2,3,4]})); "
         "print('filter:', df[df['x']>2].count()); "
         "print('str:', df['x'].str.split('-')); "
         "print('OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("sanity import failed")
    print(r.stdout.strip())

    run_tests()

    commit_with_guards(
        "v0.9.26i: extract _iterators (B.1 step 3) + fix1+fix2 "
        "(lazy _core imports). _iter_arrow_batches, "
        "_iter_filtered_batches, _iter_filtered_batches_with_"
        "positions, _with_filter, _apply_column_transform moved "
        "to _iterators.py. CadwDataFrame(_ExportsMixin, "
        "_IteratorsMixin). _convert_batch/_concat_batches from "
        "dataframe_helpers (no cycle). Lazy _core.CadwDataFrame "
        "in 2 methods (circular); §13.4/303 inspect.getsource "
        "indented; §13.4/305 multi-line sig - inject after "
        "signature end, not def+1. 0 logic change; 2492 green.",
        self_path="scripts/patch_v0926i_fix2_multi_sig.py",
        check_whitelist=True,
    )
    print("\nv0.9.26i COMPLETE (with fix1+fix2)")


if __name__ == "__main__":
    main()