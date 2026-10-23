#!/usr/bin/env python3
# v0.9.26y-fix1: rewrite __init__.py, route each symbol to actual
# source module (§13.4/331). Bootstrap-safe.

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

INIT = "cadw/dataframe/__init__.py"
CORE = "cadw/dataframe/_core.py"


NEW_INIT = '''# cadw/dataframe/__init__.py
# Package facade (v0.9.26y-fix1, post-B.1 split).
# Each re-export routed to actual source module (§13.4/331).

# Primary public surface
from ._core import CadwDataFrame
from ._exports import concat

# Class re-exports
from ._column_proxy import _ColumnProxy
from ._groupby import _GroupBy

# Helpers (v0.9.26w)
from ._helpers import (
    _apply_filter_with_positions,
    _build_out_schema,
    _detect_mode,
    _infer_arrow_type_from_value,
    _normalize_scalar,
)

# Backward-compat re-exports (pre-v0.9.26g surface)
from cadw.core.memory_guard import MemoryGuard
from cadw.core.sampling import data_chars_for_join, peek_first_batch
from cadw.dataframe_helpers import (
    _ArrowTail,
    _EwmState,
    _ExpandingState,
    _RollState,
    _concat_batches,
    _convert_batch,
    _empty_value_counts_table,
    _group_boundaries_arrow,
    _reservoir_positions,
    _weighted_reservoir_positions,
)
from cadw.dataset import (
    CadwIndex,
    Dataset,
    MaskFilter,
    MaterializedIndex,
    MaterializedMultiIndex,
    PredicateFilter,
    RangeIndex,
    RowFilter,
    SliceFilter,
)
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter, ParquetBatchWriter
from cadw.indexers import _AtIndexer, _IAtIndexer, _ILocIndexer, _LocIndexer
from cadw.ops.dedup import drop_duplicates as _ops_drop_duplicates
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.join import hash_join, hash_join_streaming
from cadw.ops.quantile import quantile as _ops_quantile
from cadw.ops.rank import VALID_METHODS, compute_ranks_to_disk
from cadw.ops.resample import parse_freq as _parse_freq
from cadw.ops.resample import resample_to_disk as _resample_to_disk
from cadw.ops.sort import external_sort

__all__ = ["CadwDataFrame", "concat"]
'''


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26x2":
        raise SystemExit(f"expected HEAD=v0.9.26x2, got {head!r}")
    print("HEAD: " + head)

    if not Path(INIT).exists():
        raise SystemExit(INIT + " missing")
    if not Path(CORE).exists():
        raise SystemExit(CORE + " missing")

    # Confirm _core.py cleaned (no ArrowIPCBatchReader)
    core_src = Path(CORE).read_text()
    if "ArrowIPCBatchReader" in core_src:
        raise SystemExit(
            "_core.py still has ArrowIPCBatchReader — cleanup didn't run?"
        )
    print(f"  {CORE}: cleaned (no ArrowIPCBatchReader)")

    # Confirm __init__ broken
    init_src = Path(INIT).read_text()
    if "from ._core import Any" in init_src:
        print(f"  {INIT}: broken (has bad import) — fix needed")
    else:
        print(f"  {INIT}: state unknown — inspecting")
        print(f"  first 5 lines: {init_src.split(chr(10))[:5]}")


def main():
    print("=== v0.9.26y-fix1: rewrite __init__.py ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()

    # Write new __init__
    p = Path(INIT)
    p.write_text(NEW_INIT)
    ast.parse(NEW_INIT, filename=INIT)
    print(f"\n  wrote {INIT} ({len(NEW_INIT.splitlines())} lines)")

    # Sanity import
    print("\n--- sanity ---")
    sanity_script = (
        "import cadw\n"
        "assert hasattr(cadw, 'CadwDataFrame')\n"
        "assert hasattr(cadw, 'concat')\n"
        "import cadw.dataframe as d\n"
        "must_have = [\n"
        "    '_ExpandingState','_RollState','_EwmState',\n"
        "    '_group_boundaries_arrow','_ArrowTail',\n"
        "    '_convert_batch','_concat_batches',\n"
        "    '_reservoir_positions','_weighted_reservoir_positions',\n"
        "    '_infer_arrow_type_from_value','_detect_mode',\n"
        "    '_normalize_scalar','_build_out_schema',\n"
        "    '_apply_filter_with_positions',\n"
        "    '_ColumnProxy','_GroupBy',\n"
        "    'Dataset','CadwIndex','RowFilter','MaskFilter',\n"
        "    'SliceFilter','PredicateFilter',\n"
        "    'MaterializedIndex','MaterializedMultiIndex',\n"
        "    'RangeIndex','MemoryGuard',\n"
        "    'ArrowIPCBatchReader','ArrowIPCBatchWriter',\n"
        "    'ParquetBatchWriter',\n"
        "    '_ILocIndexer','_LocIndexer','_AtIndexer','_IAtIndexer',\n"
        "    'groupby_aggregate','hash_join','hash_join_streaming',\n"
        "    'external_sort','compute_ranks_to_disk','VALID_METHODS',\n"
        "    '_ops_drop_duplicates','_ops_quantile',\n"
        "    '_parse_freq','_resample_to_disk',\n"
        "    'data_chars_for_join','peek_first_batch',\n"
        "]\n"
        "missing = [n for n in must_have if not hasattr(d, n)]\n"
        "assert not missing, f'missing re-exports: {missing}'\n"
        "print('re-exports: OK', len(must_have))\n"
        "C = cadw.CadwDataFrame\n"
        "methods = [\n"
        "    'select_dtypes','fit_naive_bayes','fit_lasso',\n"
        "    'fit_elasticnet','merge','cumsum','value_counts',\n"
        "    'rolling','resample','pivot_table','groupby',\n"
        "    'isin','between','idxmax','idxmin','pct_change',\n"
        "    'ffill','bfill','dropna','fillna','cut','qcut',\n"
        "    'duplicated','astype','drop','rename',\n"
        "    'cumsum','expanding','ewm','shift','diff',\n"
        "    'metrics','encode_categorical',\n"
        "]\n"
        "missing_m = [m for m in methods if not hasattr(C, m)]\n"
        "assert not missing_m, f'missing methods: {missing_m}'\n"
        "print('methods: OK', len(methods))\n"
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
        "v0.9.26y: _core.py import cleanup + __init__.py "
        "reorganization (B.1 step 20) + fix1 (rewrite __init__). "
        "13 dead imports removed from _core.py. fix1 (§13.4/331): "
        "__init__.py fully rewritten to route each symbol to its "
        "actual module (_core/_exports/_column_proxy/_groupby/"
        "_helpers/cadw.*). Auto-generated mirror from v0.9.26g "
        "was broken after split. __all__ = ['CadwDataFrame', "
        "'concat'] preserved. _core.py ~558 → ~530 LOC. "
        "0 logic change; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26y COMPLETE (fix1)")


if __name__ == "__main__":
    main()