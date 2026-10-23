#!/usr/bin/env python3
# v0.9.26i: extract _iterators (B.1 step 3).
# Move 5 methods + 2 module-level helpers verbatim.
# _exports._convert_batch lazy import -> top-level from _iterators.
# 0 logic change; 2492 tests expected green.

import ast
import inspect
import os
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
TARGET_HELPERS = [
    "_convert_batch",
    "_concat_batches",
]


# ============ discovery ============

def discover():
    src = Path(CORE).read_text()
    tree = ast.parse(src)

    cd_class = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "CadwDataFrame"
    )

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

    helper_ranges = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in TARGET_HELPERS:
            continue
        start = node.lineno
        if node.decorator_list:
            start = min(d.lineno for d in node.decorator_list)
        end = max(
            getattr(n, "end_lineno", node.lineno)
            for n in ast.walk(node)
        )
        helper_ranges.append({
            "name": node.name,
            "start": start,
            "end": end,
            "loc": end - start + 1,
        })

    return {
        "src": src,
        "tree": tree,
        "method_ranges": method_ranges,
        "helper_ranges": helper_ranges,
        "cd_class": cd_class,
    }


def source_imports_needed(source_text, core_tree):
    """AST walk source: which top-level names needed to import?"""
    sub = ast.parse(source_text)
    used = set()
    for n in ast.walk(sub):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            used.add(n.id)

    # local names in source
    local = set()
    for n in ast.walk(sub):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            local.add(n.id)
        if isinstance(n, ast.arg):
            local.add(n.arg)
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                local.add(a.asname or a.name)
        if isinstance(n, ast.Import):
            for a in n.names:
                local.add(a.asname or a.name.split(".")[0])

    # builtins
    import builtins as _b
    bnames = set(dir(_b))

    needed = used - local - bnames - {"self", "cls", "__class__"}
    return needed


def core_top_imports_map(tree):
    """Map name -> 'from module import name' for top-level imports in core."""
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                nm = a.asname or a.name
                imports[nm] = ("from", node.module, a.name)
        elif isinstance(node, ast.Import):
            for a in n.a.names if False else node.names:  # noqa
                nm = a.asname or a.name.split(".")[0]
                imports[nm] = ("import", a.name, None)
    return imports


def constructs_cadwdf(source_text):
    """Does source reference CadwDataFrame(...)?"""
    sub = ast.parse(source_text)
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
    import cadw.dataframe._core as core_mod

    method_sources = {}
    for name in TARGET_METHODS:
        fn = CadwDataFrame.__dict__[name]
        method_sources[name] = inspect.getsource(fn)

    helper_sources = {}
    for name in TARGET_HELPERS:
        fn = getattr(core_mod, name, None)
        if fn is None:
            raise SystemExit(f"helper {name} not found in _core")
        helper_sources[name] = inspect.getsource(fn)

    return method_sources, helper_sources


def write_iterators(method_sources, helper_sources, imports_needed, cd_lazy):
    lines = [
        "# cadw/dataframe/_iterators.py",
        "# v0.9.26i (B.1 step 3): _IteratorsMixin + batch helpers.",
        "# Extracted verbatim from _core.py; no logic change.",
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
        "from cadw.io.batch_reader import ArrowIPCBatchReader",
        "from cadw.io.batch_writer import (",
        "    ArrowIPCBatchWriter, ParquetBatchWriter,",
        ")",
        "",
    ]
    if cd_lazy:
        lines.append(
            "# NOTE: _IteratorsMixin methods construct CadwDataFrame."
        )
        lines.append(
            "# Circular: _core imports _IteratorsMixin. Lazy import "
            "inside method."
        )
        lines.append("")

    # module-level helpers
    for name in TARGET_HELPERS:
        lines.append("")
        lines.append(helper_sources[name])
        lines.append("")

    # mixin class
    lines.append("")
    lines.append("")
    lines.append("class _IteratorsMixin:")
    lines.append('    """Batch iteration + filter composition mixin '
                 '(v0.9.26i)."""')
    lines.append("")

    for name in TARGET_METHODS:
        src = method_sources[name]
        if cd_lazy:
            # inject lazy import right after def line + docstring
            src = _inject_lazy_cadwdf(src)
        lines.append(src)
        lines.append("")
        lines.append("")

    text = "\n".join(lines)
    p = Path(ITERATORS)
    p.write_text(text)
    ast.parse(text, filename=str(p))
    print("  wrote " + ITERATORS + " (" + str(len(text.splitlines())) + " lines)")
    if cd_lazy:
        print("  lazy _core import injected (constructs CadwDataFrame)")


def _inject_lazy_cadwdf(src):
    """Add `from ._core import CadwDataFrame` after def line/docstring."""
    try:
        sub = ast.parse(src)
    except SyntaxError:
        return src
    fn = sub.body[0]
    lines = src.split("\n")

    insert_after = fn.lineno  # 1-based line of 'def'
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        insert_after = fn.body[0].end_lineno

    indent = "        " if src.startswith("    def ") else "    "
    lazy = f"{indent}from ._core import CadwDataFrame"
    lines.insert(insert_after, lazy)
    return "\n".join(lines)


def remove_from_core(info):
    lines = list(info["src"].split("\n"))
    ranges = []
    for m in info["method_ranges"]:
        ranges.append((m["start"], m["end"]))
    for h in info["helper_ranges"]:
        ranges.append((h["start"], h["end"]))
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

    # add import + inheritance
    src = src.replace(
        "from ._exports import _ExportsMixin, concat",
        "from ._exports import _ExportsMixin, concat\n"
        "from ._iterators import (\n"
        "    _IteratorsMixin, _concat_batches, _convert_batch,\n"
        ")",
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


def update_exports():
    """Remove lazy from ._core import _convert_batch; add top-level."""
    src = Path(EXPORTS).read_text()
    if "from ._iterators import _convert_batch" in src:
        print("  _exports already has top-level import")
        return
    # remove lazy line
    import re
    src2 = re.sub(
        r"^[ \t]+from \._core import _convert_batch\n",
        "",
        src,
        flags=re.MULTILINE,
    )
    # add top-level import after cadw imports
    src2 = src2.replace(
        "from cadw.io.batch_writer import (",
        "from ._iterators import _convert_batch\n"
        "from cadw.io.batch_writer import (",
        1,
    )
    if src2 == src:
        print("  WARN: no changes made")
        return
    Path(EXPORTS).write_text(src2)
    ast.parse(src2, filename=EXPORTS)
    print("  _exports: lazy -> top-level _convert_batch")


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
    print("=== v0.9.26i: extract _iterators ===")
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

    info = discover()
    print("\n[methods to extract]")
    for m in info["method_ranges"]:
        print("  " + m["name"] + ": lines " + str(m["start"]) +
              "-" + str(m["end"]) + " (" + str(m["loc"]) + " LOC)")
    print("\n[helpers]")
    for h in info["helper_ranges"]:
        print("  " + h["name"] + ": lines " + str(h["start"]) +
              "-" + str(h["end"]) + " (" + str(h["loc"]) + " LOC)")

    found_methods = {m["name"] for m in info["method_ranges"]}
    missing_m = set(TARGET_METHODS) - found_methods
    if missing_m:
        raise SystemExit("missing methods: " + str(missing_m))
    found_helpers = {h["name"] for h in info["helper_ranges"]}
    missing_h = set(TARGET_HELPERS) - found_helpers
    if missing_h:
        raise SystemExit("missing helpers: " + str(missing_h))

    # load module before inspect
    import cadw.dataframe._core  # noqa: F401

    print("\n--- extracting sources ---")
    method_sources, helper_sources = extract_sources()

    # discovery: does any method construct CadwDataFrame?
    cd_lazy = False
    for name, src in method_sources.items():
        if constructs_cadwdf(src):
            print(f"  {name}: constructs CadwDataFrame -> lazy import")
            cd_lazy = True
    for name, src in helper_sources.items():
        if constructs_cadwdf(src):
            print(f"  {name}: constructs CadwDataFrame -> lazy import")
            cd_lazy = True
    if not cd_lazy:
        print("  no CadwDataFrame construction in extracted code")

    print("\n--- writing _iterators.py ---")
    write_iterators(method_sources, helper_sources, set(), cd_lazy)

    print("\n--- updating _core.py ---")
    remove_from_core(info)
    update_core_class()

    print("\n--- updating _exports.py ---")
    update_exports()

    print("\n--- sanity import ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import cadw; print(cadw.CadwDataFrame); "
         "print(cadw.CadwDataFrame._iter_arrow_batches); "
         "print(cadw.dataframe._iterators._convert_batch)"],
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
        "_apply_column_transform + _convert_batch + "
        "_concat_batches moved verbatim to _iterators.py. "
        "CadwDataFrame(_ExportsMixin, _IteratorsMixin). "
        "_exports: lazy -> top-level _convert_batch (no cycle). "
        "0 logic change; 2492 tests green.",
        self_path="scripts/patch_v0926i_extract_iterators.py",
        check_whitelist=True,
    )
    print("\nv0.9.26i COMPLETE")


if __name__ == "__main__":
    main()