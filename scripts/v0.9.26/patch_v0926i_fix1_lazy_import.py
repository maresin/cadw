#!/usr/bin/env python3
# v0.9.26i-fix1: inject lazy _core.CadwDataFrame imports into
# _iterators.py methods. Fix for §13.4/303 (ast.parse on indented
# source from inspect.getsource fails). Text-based injection.

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


ITERATORS = "cadw/dataframe/_iterators.py"

# Methods that construct CadwDataFrame (from failure traceback)
LAZY_METHODS = ["_with_filter", "_apply_column_transform"]


def inject_lazy(src, method_name):
    """Insert `from ._core import CadwDataFrame` after docstring."""
    lines = src.split("\n")

    # Find def line
    def_idx = None
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if (stripped.startswith(f"def {method_name}(")
                or stripped.startswith(f"def {method_name} (")):
            def_idx = i
            break
    if def_idx is None:
        print(f"  WARN: {method_name} def not found")
        return src, False

    # Skip past docstring if present
    insert_at = def_idx + 1
    if insert_at < len(lines):
        stripped = lines[insert_at].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = '"""' if stripped.startswith('"""') else "'''"
            # One-line docstring (starts and ends on same line)
            if stripped.count(quote) >= 2:
                insert_at += 1
            else:
                # Multi-line: scan for closing
                insert_at += 1
                while insert_at < len(lines):
                    if quote in lines[insert_at]:
                        insert_at += 1
                        break
                    insert_at += 1

    # Check if already injected (idempotency)
    for j in range(def_idx, insert_at):
        if "from ._core import CadwDataFrame" in lines[j]:
            print(f"  {method_name}: lazy import already present")
            return src, False

    # Insert lazy import at 8-space indent (method body)
    indent = "        "
    lines.insert(insert_at, f"{indent}from ._core import CadwDataFrame")
    print(f"  {method_name}: lazy import inserted at line {insert_at + 1}")
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
        raise SystemExit(ITERATORS + " missing - v0.9.26i not applied?")
    src = Path(ITERATORS).read_text()
    if "_IteratorsMixin" not in src:
        raise SystemExit("_IteratorsMixin not in _iterators.py")
    if "from ._core import CadwDataFrame" in src:
        print("  WARN: already has lazy import")

    # Sanity: _core.py has been mutated (class inherits both mixins)
    core_src = Path("cadw/dataframe/_core.py").read_text()
    if "class CadwDataFrame(_ExportsMixin, _IteratorsMixin)" not in core_src:
        raise SystemExit(
            "_core.py doesn't have dual-mixin class - state corrupted"
        )
    print("  _core.py: dual-mixin class OK")


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26i-fix1: lazy _core import injection ===")
    verify_state()

    src = Path(ITERATORS).read_text()
    changed = False
    for method in LAZY_METHODS:
        src, did = inject_lazy(src, method)
        if did:
            changed = True

    if not changed:
        print("\nno changes needed - already applied?")
        run_tests()
        return

    Path(ITERATORS).write_text(src)
    ast.parse(src, filename=ITERATORS)
    print("\n_iterators.py updated + parses")

    print("\n--- sanity import ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import cadw; "
         "df = cadw.CadwDataFrame.from_table(__import__('pyarrow').table({'x':[1,2,3,4]})); "
         "print(df[df['x']>2].head(10).count()); "
         "print('_with_filter OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("sanity import failed")
    print(r.stdout.strip())

    run_tests()

    commit_with_guards(
        "v0.9.26i: extract _iterators (B.1 step 3) + fix1 "
        "(lazy _core import via text-based injection). "
        "_iter_arrow_batches, _iter_filtered_batches, "
        "_iter_filtered_batches_with_positions, _with_filter, "
        "_apply_column_transform moved verbatim to _iterators.py. "
        "CadwDataFrame(_ExportsMixin, _IteratorsMixin). "
        "_convert_batch/_concat_batches from dataframe_helpers "
        "(top-level, no cycle). fix1: text-based lazy import "
        "injection (§13.4/303 - ast.parse on indented source "
        "from inspect.getsource fails). 0 logic change; 2492 "
        "tests green.",
        self_path="scripts/patch_v0926i_fix1_lazy_import.py",
        check_whitelist=True,
    )
    print("\nv0.9.26i COMPLETE (with fix1)")


if __name__ == "__main__":
    main()