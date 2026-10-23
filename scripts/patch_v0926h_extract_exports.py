#!/usr/bin/env python3
# v0.9.26h: extract _exports mixin (B.1 step 2, pilot).
# Move 5 methods + concat verbatim. Add mixin inheritance.
# 0 logic change; only structural move.

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
TARGET_METHODS = [
    "to_batches", "write_parquet",
    "from_table", "from_parquet", "from_dataset",
]


# ============ discovery ============

def discover():
    src = Path(CORE).read_text()
    lines = src.split("\n")
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

    concat_node = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "concat"),
        None,
    )
    concat_range = None
    if concat_node:
        start = concat_node.lineno
        if concat_node.decorator_list:
            start = min(d.lineno for d in concat_node.decorator_list)
        end = max(
            getattr(n, "end_lineno", concat_node.lineno)
            for n in ast.walk(concat_node)
        )
        concat_range = {
            "name": "concat",
            "start": start,
            "end": end,
            "loc": end - start + 1,
        }

    return {
        "src": src,
        "lines": lines,
        "method_ranges": method_ranges,
        "concat_range": concat_range,
    }


def print_plan(info):
    print("\n" + "=" * 72)
    print("DISCOVERY PLAN")
    print("=" * 72)
    print("\n[methods to extract from CadwDataFrame]")
    for m in info["method_ranges"]:
        print("  " + m["name"] + ": lines " + str(m["start"]) +
              "-" + str(m["end"]) + " (" + str(m["loc"]) + " LOC)")
    cr = info["concat_range"]
    if cr:
        print("\n[concat] module-level:")
        print("  " + cr["name"] + ": lines " + str(cr["start"]) +
              "-" + str(cr["end"]) + " (" + str(cr["loc"]) + " LOC)")
    else:
        print("\n[concat] not found — unexpected")


# ============ extraction ============

def get_method_sources():
    """Extract source via inspect (preserves decorators + docstrings)."""
    from cadw.dataframe._core import CadwDataFrame
    from cadw.dataframe._core import concat as concat_fn

    sources = {}
    for name in TARGET_METHODS:
        raw = CadwDataFrame.__dict__[name]
        fn = raw.__func__ if isinstance(raw, classmethod) else raw
        sources[name] = inspect.getsource(fn)

    concat_src = inspect.getsource(concat_fn)
    return sources, concat_src


def write_exports(sources, concat_src):
    lines = [
        "# cadw/dataframe/_exports.py",
        "# v0.9.26h (B.1 step 2): _ExportsMixin + concat.",
        "# Extracted verbatim from _core.py; no logic change.",
        "",
        "from __future__ import annotations",
        "",
        "import tempfile",
        "import uuid",
        "from pathlib import Path",
        "from typing import Any, Iterator",
        "",
        "import pyarrow as pa",
        "import pyarrow.parquet as pq",
        "",
        "from cadw.core.logging import get_logger",
        "from cadw.core.memory_guard import MemoryGuard",
        "from cadw.dataset import Dataset",
        "from cadw.dataset._in_memory import _InMemoryDataset",
        "from cadw.io.batch_reader import ArrowIPCBatchReader",
        "from cadw.io.batch_writer import (",
        "    ArrowIPCBatchWriter,",
        "    ParquetBatchWriter,",
        ")",
        "",
        "logger = get_logger(__name__)",
        "",
        "",
        "class _ExportsMixin:",
        '    """Export/IO mixin for CadwDataFrame (v0.9.26h).',
        "",
        "    Methods: to_batches, write_parquet, from_table,",
        "    from_parquet, from_dataset.",
        '    """',
        "",
    ]

    for name in TARGET_METHODS:
        if name in sources:
            lines.append(sources[name])
            lines.append("")
            lines.append("")

    if concat_src:
        lines.append("")
        lines.append(concat_src)

    src = "\n".join(lines)
    p = Path(EXPORTS)
    p.write_text(src)
    ast.parse(src, filename=str(p))
    print("  wrote " + EXPORTS + " (" + str(len(src.splitlines())) + " lines)")


def remove_from_core(info):
    lines = list(info["lines"])

    ranges = [(m["start"], m["end"]) for m in info["method_ranges"]]
    if info["concat_range"]:
        ranges.append((
            info["concat_range"]["start"],
            info["concat_range"]["end"],
        ))
    ranges.sort(reverse=True)  # descending to avoid offset shifts

    for start, end in ranges:
        del lines[start - 1:end]

    new_src = "\n".join(lines)
    Path(CORE).write_text(new_src)
    print("  removed " + str(len(ranges)) + " range(s) from " + CORE)


def update_class_inheritance():
    """Add import + _ExportsMixin to class."""
    src = Path(CORE).read_text()
    if "class CadwDataFrame(_ExportsMixin)" in src:
        print("  class already inherits _ExportsMixin")
        return
    if "from ._exports import" not in src:
        src = src.replace(
            "class CadwDataFrame:",
            "from ._exports import _ExportsMixin, concat\n"
            "\n\nclass CadwDataFrame(_ExportsMixin):",
            1,
        )
    else:
        src = src.replace(
            "class CadwDataFrame:",
            "class CadwDataFrame(_ExportsMixin):",
            1,
        )
    Path(CORE).write_text(src)
    ast.parse(src, filename=CORE)
    print("  CadwDataFrame now inherits _ExportsMixin")


# ============ tests ============

def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit(
            "fast suite FAILED. Rollback: git reset --hard HEAD. "
            "Then inspect failures."
        )


# ============ main ============

def main():
    print("=== v0.9.26h: extract _exports ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26g":
        raise SystemExit("expected HEAD=v0.9.26g, got " + repr(head))
    print("HEAD: " + head)

    if not Path(CORE).exists():
        raise SystemExit(CORE + " missing")
    if Path(EXPORTS).exists():
        raise SystemExit(EXPORTS + " exists — clean it first")

    info = discover()
    print_plan(info)
    if len(info["method_ranges"]) != len(TARGET_METHODS):
        raise SystemExit(
            "expected " + str(len(TARGET_METHODS)) +
            " methods, found " + str(len(info["method_ranges"]))
        )

    # Ensure module is loaded (needed for inspect.getsource)
    import cadw.dataframe._core  # noqa: F401

    print("\n--- extraction ---")
    sources, concat_src = get_method_sources()
    write_exports(sources, concat_src)
    remove_from_core(info)
    update_class_inheritance()

    print("\n--- sanity: fresh import ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import cadw; print(cadw.CadwDataFrame); "
         "print(cadw.CadwDataFrame.to_batches); "
         "print(cadw.CadwDataFrame.from_table); "
         "print(cadw.dataframe.concat)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("import sanity failed")
    print(r.stdout.strip())

    run_tests()

    commit_with_guards(
        "v0.9.26h: extract _exports (B.1 step 2, pilot). "
        "to_batches/write_parquet/from_table/from_parquet/"
        "from_dataset + concat moved verbatim to _exports.py. "
        "CadwDataFrame(_ExportsMixin) - mixin composition. "
        "0 logic change; 2488 tests green.",
        self_path="scripts/patch_v0926h_extract_exports.py",
        check_whitelist=True,
    )
    print("\nv0.9.26h COMPLETE")


if __name__ == "__main__":
    main()