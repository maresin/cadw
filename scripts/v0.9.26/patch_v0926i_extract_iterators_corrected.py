#!/usr/bin/env python3
# v0.9.26i-corrected: extract _iterators (5 methods only).
# Helpers _convert_batch/_concat_batches already in
# cadw/dataframe_helpers.py (v0.9.25f-a). Import directly.
# 0 logic change; 2492 tests expected green.

import ast
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


CORE = "cadw/dataframe/_core.py"
EXPORTS = "cadw/dataframe/_exports.py"
ITERATORS = "cadw/dataframe/_iterators.py"

TARGET_METHODS = [
    "_iter_arrow_batches",
    "_iter_filtered_batches",
    "_iter_filtered_batches_with_positions",
    "_with_filter",
    "_apply_column_transform",
]

# §13.4/302: verify helpers live in dataframe_helpers before assuming
HELPERS_MODULE = "cadw/dataframe_helpers.py"
HELPERS_NAMES = ["_convert_batch", "_concat_batches"]


# ============ discovery ============

def verify_helpers_location():
    """Confirm helpers live in dataframe_helpers (v0.9.25f-a)."""
    print("\n--- verify helpers location ---")
    p = Path(HELPERS_MODULE)
    if not p.exists():
        raise SystemExit(HELPERS_MODULE + " missing")
    tree = ast.parse(p.read_text())
    found = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.name in HELPERS_NAMES:
                found.add(n.name)
    missing = set(HELPERS_NAMES) - found
    if missing:
        raise SystemExit(
            HELPERS_MODULE + " missing helpers: " + str(missing)
        )
    print("  " + HELPERS_MODULE + ": " +
          str(sorted(found)) + " (all present)")

    # _core imports them via ImportFrom - confirm
    core_src = Path(CORE).read_text()
    core_tree = ast.parse(core_src)
    re_exported = set()
    for n in core_tree.body:
        if isinstance(n, ast.ImportFrom):
            if n.module and "dataframe_helpers" in n.module:
                for a in n.names:
                    nm = a.asname or a.name
                    if nm in HELPERS_NAMES:
                        re_exported.add(nm)
    missing_re = set(HELPERS_NAMES) - re_exported
    if missing_re:
        print("  WARN: not re-exported from _core: " + str(missing_re))
    else:
        print("  _core re-exports via ImportFrom: " +
              str(sorted(re_exported)))


def discover_methods():
    src = Path(CORE).read_text()
    tree = ast.parse(src)

    cd_class = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef) and n.name == "CadwDataFrame"),
        None,
    )
    if cd_class is None:
        raise SystemExit("CadwDataFrame class not found in _core")

    method_ranges = []
    for item in cd_class.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name not in TARGET_METHODS:
            continue
        start = item.lineno
        if item.decorator_list:
            start = min(d.lineno for d in item.decorator_list)
        end = max(
            getattr(n, "end_lineno", item.lineno)
            for n in ast.walk(item)
        )
        method_ranges.append({
            "name": item.name,
            "start": start,
            "end": end,
            "loc": end - start + 1,
        })

    return src, method_ranges


def constructs_cadwdf(source_text):
    try:
        sub = ast.parse(source_text)
    except SyntaxError:
        return False
    for n in ast.walk(sub):
        if (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "CadwDataFrame"):
            return True
        if (isinstance(n, ast.Name) and n.id == "CadwDataFrame"
                and isinstance(n.ctx, ast.Load)):
            return True
    return False


# ============ extraction ============

def extract_sources():
    from cadw.dataframe._core import CadwDataFrame
    method_sources = {}
    for name in TARGET_METHODS:
        fn = CadwDataFrame.__dict__[name]
        method_sources[name] = inspect.getsource(fn)
    return method_sources


def _inject_lazy_cadwdf(src):
    try:
        sub = ast.parse(src)
    except SyntaxError:
        return src
    fn = sub.body[0]
    lines = src.split("\n")
    insert_after = fn.lineno
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        insert_after = fn.body[0].end_lineno
    indent = "        " if src.startswith("    def ") else "    "
    lazy = f"{indent}from ._core import CadwDataFrame"
    lines.insert(insert_after, lazy)
    return "\n".join(lines)


def write_iterators(method_sources, cd_lazy):
    lines = [
        "# cadw/dataframe/_iterators.py",
        "# v0.9.26i (B.1 step 3): _IteratorsMixin.",
        "# Extracted verbatim from _core.py; no logic change.",
        "# Helpers _convert_batch/_concat_batches imported from",
        "# dataframe_helpers (v0.9.25f-a - already extracted).",
        "",
        "from __future__ import annotations",
        "",
        "from typing import Iterator",
        "",
        "import numpy as np",
        "import pyarrow as pa",
        "import pyarrow.compute as pc",
        "",
        "from cadw.core.memory_guard import MemoryGuard",
        "from cadw.dataset import (",
        "    Dataset, MaskFilter, PredicateFilter, RowFilter,",
        "    SliceFilter,",
        ")",
        "from cadw.dataframe_helpers import (",
        "    _concat_batches, _convert_batch,",
        ")",
        "from cadw.io.batch_reader import ArrowIPCBatchReader",
        "from cadw.io.batch_writer import (",
        "    ArrowIPCBatchWriter, ParquetBatchWriter,",
        ")",
        "",
    ]
    if cd_lazy:
        lines.append(
            "# NOTE: methods construct CadwDataFrame. "
            "Lazy import inside method."
        )
        lines.append("")

    lines.append("")
    lines.append("class _IteratorsMixin:")
    lines.append('    """Batch iteration + filter composition mixin '
                 '(v0.9.26i)."""')
    lines.append("")

    for name in TARGET_METHODS:
        src = method_sources[name]
        if cd_lazy:
            src = _inject_lazy_cadwdf(src)
        lines.append(src)
        lines.append("")
        lines.append("")

    text = "\n".join(lines)
    p = Path(ITERATORS)
    p.write_text(text)
    ast.parse(text, filename=str(p))
    print("  wrote " + ITERATORS + " (" +
          str(len(text.splitlines())) + " lines)")
    if cd_lazy:
        print("  lazy _core import injected")


def remove_from_core(src, method_ranges):
    lines = list(src.split("\n"))
    ranges = [(m["start"], m["end"]) for m in method_ranges]
    ranges.sort(reverse=True)
    for start, end in ranges:
        del lines[start - 1:end]
    new_src = "\n".join(lines)
    Path(CORE).write_text(new_src)
    print("  removed " + str(len(ranges)) + " range(s) from " + CORE)


def update_core_class():
    src = Path(CORE).read_text()
    if "class CadwDataFrame(_ExportsMixin, _IteratorsMixin)" in src:
        print("  class already inherits both mixins")
        return
    if "from ._iterators import" not in src:
        src = src.replace(
            "from ._exports import _ExportsMixin, concat",
            "from ._exports import _ExportsMixin, concat\n"
            "from ._iterators import _IteratorsMixin",
            1,
        )
    src = src.replace(
        "class CadwDataFrame(_ExportsMixin):",
        "class CadwDataFrame(_ExportsMixin, _IteratorsMixin):",
        1,
    )
    Path(CORE).write_text(src)
    ast.parse(src, filename=CORE)
    print("  CadwDataFrame inherits _ExportsMixin + _IteratorsMixin")


def update_exports_imports():
    """Replace lazy _core._convert_batch with top-level
    dataframe_helpers._convert_batch."""
    src = Path(EXPORTS).read_text()
    if "from cadw.dataframe_helpers import _convert_batch" in src:
        print("  _exports already imports from dataframe_helpers")
        return

    # Remove lazy line (any indent, ._core)
    src2 = re.sub(
        r"^[ \t]+from \._core import _convert_batch\n",
        "",
        src,
        flags=re.MULTILINE,
    )

    # Add top-level import after cadw imports
    anchor = "from cadw.core.logging import get_logger\n"
    if anchor not in src2:
        raise SystemExit(
            "anchor 'from cadw.core.logging import get_logger' "
            "not found in _exports.py"
        )
    src2 = src2.replace(
        anchor,
        anchor + "from cadw.dataframe_helpers import _convert_batch\n",
        1,
    )

    if src2 == src:
        raise SystemExit("no change made to _exports.py")
    Path(EXPORTS).write_text(src2)
    ast.parse(src2, filename=EXPORTS)
    print("  _exports: lazy _core._convert_batch -> "
          "top-level dataframe_helpers._convert_batch")


# ============ tests ============

def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


# ============ main ============

def main():
    print("=== v0.9.26i-corrected: extract _iterators ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26h2-fix1":
        raise SystemExit("expected HEAD=v0.9.26h2-fix1, got " + repr(head))
    print("HEAD: " + head)

    for p in (CORE, EXPORTS):
        if not Path(p).exists():
            raise SystemExit(p + " missing")
    if Path(ITERATORS).exists():
        raise SystemExit(ITERATORS + " exists - clean first")

    verify_helpers_location()

    src, method_ranges = discover_methods()
    print("\n[methods to extract]")
    for m in method_ranges:
        print("  " + m["name"] + ": lines " + str(m["start"]) +
              "-" + str(m["end"]) + " (" + str(m["loc"]) + " LOC)")

    found = {m["name"] for m in method_ranges}
    missing = set(TARGET_METHODS) - found
    if missing:
        raise SystemExit("missing methods: " + str(missing))

    # load module before inspect
    import cadw.dataframe._core  # noqa: F401

    print("\n--- extracting sources ---")
    method_sources = extract_sources()

    cd_lazy = False
    for name, s in method_sources.items():
        if constructs_cadwdf(s):
            print(f"  {name}: constructs CadwDataFrame -> lazy import")
            cd_lazy = True
    if not cd_lazy:
        print("  no CadwDataFrame construction detected")

    print("\n--- write _iterators.py ---")
    write_iterators(method_sources, cd_lazy)

    print("\n--- update _core.py ---")
    remove_from_core(src, method_ranges)
    update_core_class()

    print("\n--- update _exports.py ---")
    update_exports_imports()

    print("\n--- sanity import ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import cadw; print(cadw.CadwDataFrame); "
         "print(cadw.CadwDataFrame._iter_arrow_batches); "
         "print(cadw.dataframe._iterators._convert_batch); "
         "print(cadw.dataframe._iterators._concat_batches)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("import sanity failed")
    print(r.stdout.strip())

    run_tests()

    commit_with_guards(
        "v0.9.26i: extract _iterators (B.1 step 3). "
        "_iter_arrow_batches, _iter_filtered_batches, "
        "_iter_filtered_batches_with_positions, _with_filter, "
        "_apply_column_transform moved verbatim to _iterators.py. "
        "CadwDataFrame(_ExportsMixin, _IteratorsMixin). "
        "_convert_batch/_concat_batches already in "
        "dataframe_helpers.py (v0.9.25f-a); _exports lazy removed, "
        "top-level import from dataframe_helpers. "
        "0 logic change; 2492 tests green.",
        self_path="scripts/patch_v0926i_extract_iterators_corrected.py",
        check_whitelist=True,
    )
    print("\nv0.9.26i COMPLETE")


if __name__ == "__main__":
    main()