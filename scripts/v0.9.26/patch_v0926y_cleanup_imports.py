#!/usr/bin/env python3
# v0.9.26y: import cleanup in _core.py + __init__.py reorganization.
# 13 dead imports removed; dataframe_helpers re-export moved to
# __init__.py; _helpers (5 funcs) kept for lazy imports.
# Bootstrap-safe.

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CORE = "cadw/dataframe/_core.py"
INIT = "cadw/dataframe/__init__.py"


# Dead imports to remove from _core.py (module, kind, names)
DEAD_IMPORTS = [
    ("pyarrow.parquet", "import", ["pq"]),
    ("cadw.core.sampling", "from",
     ["data_chars_for_join", "peek_first_batch"]),
    ("cadw.io.batch_reader", "from", ["ArrowIPCBatchReader"]),
    ("cadw.io.batch_writer", "from",
     ["ArrowIPCBatchWriter", "ParquetBatchWriter"]),
    ("cadw.indexers", "from",
     ["_ILocIndexer", "_LocIndexer", "_AtIndexer", "_IAtIndexer"]),
    ("cadw.ops.groupby", "from", ["groupby_aggregate"]),
    ("cadw.ops.join", "from", ["hash_join", "hash_join_streaming"]),
    ("cadw.ops.sort", "from", ["external_sort"]),
    ("cadw.ops.dedup", "from", ["drop_duplicates"]),
    ("cadw.ops.rank", "from",
     ["compute_ranks_to_disk", "VALID_METHODS"]),
    ("cadw.ops.quantile", "from", ["quantile"]),
    ("cadw.ops.resample", "from",
     ["parse_freq", "resample_to_disk"]),
    ("cadw.dataframe_helpers", "from",
     ["_ArrowTail", "_RollState", "_ExpandingState", "_EwmState",
      "_group_boundaries_arrow", "_convert_batch",
      "_concat_batches", "_empty_value_counts_table",
      "_reservoir_positions", "_weighted_reservoir_positions"]),
]


def verify_lazy_refs():
    """Confirm no other module lazy-imports these from ._core."""
    print("\n--- verify lazy refs ---")
    banned = set()
    for module, _, names in DEAD_IMPORTS:
        for name in names:
            banned.add(name)
    banned.discard("pq")  # not a symbol name
    banned.discard("data_chars_for_join")  # may still be used elsewhere
    banned.discard("peek_first_batch")

    hits = []
    for py in Path("cadw/dataframe").glob("*.py"):
        if py.name in ("_core.py", "__init__.py"):
            continue
        src = py.read_text()
        # Find `from ._core import (...)` blocks
        for m in re.finditer(
            r"from \._core import (\([^)]*\)|[^\n]+)", src
        ):
            block = m.group(1)
            for name in banned:
                # match word boundary
                if re.search(rf"\b{re.escape(name)}\b", block):
                    hits.append((py.name, name))

    if hits:
        print(f"  ⚠ {len(hits)} lazy refs to dead imports:")
        for fn, name in hits:
            print(f"    {fn}: {name}")
        raise SystemExit(
            "dead import still referenced from another module — "
            "cannot remove safely"
        )
    print(f"  ✓ no lazy refs to {len(banned)} names")


def remove_dead_imports():
    """Remove matching ImportFrom/Import statements (whole statement)."""
    print("\n--- remove dead imports ---")
    p = Path(CORE)
    src = p.read_text()
    tree = ast.parse(src)
    lines = src.split("\n")

    to_delete = []
    for module, kind, names in DEAD_IMPORTS:
        found = False
        for node in tree.body:
            if kind == "from" and isinstance(node, ast.ImportFrom):
                if node.module == module:
                    node_names = {a.name for a in node.names}
                    if set(names) <= node_names:
                        start = node.lineno
                        end = getattr(node, "end_lineno", node.lineno)
                        to_delete.append((start, end, f"from {module}"))
                        found = True
                        break
            elif kind == "import" and isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname == names[0] or a.name == module:
                        start = node.lineno
                        end = getattr(node, "end_lineno", node.lineno)
                        to_delete.append(
                            (start, end, f"import {module}")
                        )
                        found = True
                        break
                if found:
                    break
        if not found:
            print(f"  WARN: not found: {kind} {module} {names}")

    to_delete.sort(reverse=True)
    for start, end, desc in to_delete:
        print(f"  remove lines {start}-{end}: {desc}")
        del lines[start - 1:end]

    new_src = "\n".join(lines)
    ast.parse(new_src, filename=CORE)
    Path(CORE).write_text(new_src)
    print(f"  _core.py: {len(new_src.splitlines())} lines")
    return len(to_delete)


def update_init():
    """Reorganize __init__.py: import from actual modules."""
    print("\n--- update __init__.py ---")
    p = Path(INIT)
    src = p.read_text()
    tree = ast.parse(src)

    # Find `from ._core import (...)` — this is our target
    old_node = None
    for node in tree.body:
        if (isinstance(node, ast.ImportFrom)
                and node.module == "_core"
                and node.level == 1):
            old_node = node
            break
    if old_node is None:
        print("  WARN: no `from ._core import` in __init__.py — skip")
        return 0

    start = old_node.lineno
    end = getattr(old_node, "end_lineno", start)
    imported = [a.asname or a.name for a in old_node.names]
    print(f"  current from ._core: {imported}")

    # Categorize
    from_core = []
    from_exports = []
    from_helpers = []
    from_df_helpers = []

    helpers_set = {
        "_apply_filter_with_positions", "_detect_mode",
        "_normalize_scalar", "_infer_arrow_type_from_value",
        "_build_out_schema",
    }
    df_helpers_set = {
        "_ArrowTail", "_RollState", "_ExpandingState", "_EwmState",
        "_group_boundaries_arrow", "_convert_batch",
        "_concat_batches", "_empty_value_counts_table",
        "_reservoir_positions", "_weighted_reservoir_positions",
    }

    for name in imported:
        if name == "CadwDataFrame":
            from_core.append(name)
        elif name == "concat":
            from_exports.append(name)
        elif name in helpers_set:
            from_helpers.append(name)
        elif name in df_helpers_set:
            from_df_helpers.append(name)
        else:
            from_core.append(name)  # fallback

    new_lines = []
    if from_core:
        new_lines.append(
            f"from ._core import {', '.join(from_core)}"
        )
    if from_exports:
        new_lines.append(
            f"from ._exports import {', '.join(from_exports)}"
        )
    if from_helpers:
        new_lines.append("from ._helpers import (")
        for n in from_helpers:
            new_lines.append(f"    {n},")
        new_lines.append(")")
    if from_df_helpers:
        new_lines.append("from cadw.dataframe_helpers import (")
        for n in from_df_helpers:
            new_lines.append(f"    {n},")
        new_lines.append(")")

    print(f"  new imports ({len(new_lines)} lines):")
    for ln in new_lines:
        print(f"    {ln}")

    lines = src.split("\n")
    lines[start - 1:end] = new_lines
    new_src = "\n".join(lines)
    ast.parse(new_src, filename=INIT)
    p.write_text(new_src)
    print("  __init__.py updated")
    return 1


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26y: import cleanup ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26x2":
        raise SystemExit(f"expected HEAD=v0.9.26x2, got {head!r}")
    print("HEAD: " + head)

    if not Path(CORE).exists():
        raise SystemExit(CORE + " missing")
    if not Path(INIT).exists():
        raise SystemExit(INIT + " missing")

    # Safety: verify no lazy refs before removal
    verify_lazy_refs()

    n_removed = remove_dead_imports()
    n_init = update_init()

    print(f"\n  summary: {n_removed} dead imports removed, "
          f"__init__.py reorganized: {n_init}")

    # Sanity
    print("\n--- sanity ---")
    sanity_script = (
        "import cadw\n"
        "assert hasattr(cadw, 'CadwDataFrame')\n"
        "assert hasattr(cadw, 'concat')\n"
        "import cadw.dataframe as d\n"
        "assert hasattr(d, '_ExpandingState')\n"
        "assert hasattr(d, '_RollState')\n"
        "assert hasattr(d, '_group_boundaries_arrow')\n"
        "assert hasattr(d, '_infer_arrow_type_from_value')\n"
        "print('re-exports: OK')\n"
        "C = cadw.CadwDataFrame\n"
        "assert hasattr(C, 'select_dtypes')\n"
        "assert hasattr(C, 'fit_naive_bayes')\n"
        "assert hasattr(C, 'merge')\n"
        "print('CadwDataFrame methods: OK')\n"
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

    run_tests()

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
        "v0.9.26y: _core.py import cleanup + __init__.py "
        "reorganization (B.1 step 20). "
        "13 dead imports removed (cadw.ops.*, cadw.io.*, "
        "cadw.indexers, dataframe_helpers) — all moved to "
        "specific mixins in B.1. __init__.py now imports "
        "_ExpandingState/_RollState/_group_boundaries_arrow "
        "directly from dataframe_helpers, "
        "_infer_arrow_type_from_value from _helpers, concat "
        "from _exports. _helpers (5 funcs) kept in _core.py "
        "for lazy imports from _iterators/_per_group. §13.4/328/"
        "330. _core.py ~558 → ~520 LOC. 0 logic change; tests "
        "green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26y COMPLETE")


if __name__ == "__main__":
    main()