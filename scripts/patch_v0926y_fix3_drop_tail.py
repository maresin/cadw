#!/usr/bin/env python3
# v0.9.26y-fix3: drop `tail` from sanity (backlog feature,
# §13.4/314/333). No mutations; re-run sanity + commit.

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26x2":
        raise SystemExit(f"expected HEAD=v0.9.26x2, got {head!r}")
    print("HEAD: " + head)

    core_lines = len(
        Path("cadw/dataframe/_core.py").read_text().splitlines()
    )
    init_lines = len(
        Path("cadw/dataframe/__init__.py").read_text().splitlines()
    )
    print(f"  _core.py: {core_lines} lines")
    print(f"  __init__.py: {init_lines} lines")


def main():
    print("=== v0.9.26y-fix3: drop tail from sanity ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()

    print("\n--- sanity (drop tail) ---")
    sanity_script = (
        "import cadw\n"
        "import pyarrow as pa\n"
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
        # CadwDataFrame methods (no `tail` — §13.4/314 backlog)
        "C = cadw.CadwDataFrame\n"
        "methods = [\n"
        "    'select_dtypes','fit_naive_bayes','fit_lasso',\n"
        "    'fit_elasticnet','merge','cumsum','value_counts',\n"
        "    'rolling','resample','pivot_table','groupby',\n"
        "    'pct_change','ffill','bfill','dropna','fillna',\n"
        "    'cut','qcut','duplicated','astype','drop','rename',\n"
        "    'expanding','ewm','shift','diff',\n"
        "    'metrics','encode_categorical',\n"
        "    'head','sort_values','reindex',\n"
        "    'set_index','reset_index',\n"
        "    'corr','nunique','describe','nlargest','nsmallest',\n"
        "    'isna','isna_sum','quantile','count',\n"
        "    'sample','rank',\n"
        "    'melt','crosstab','drop_duplicates',\n"
        "]\n"
        "missing_m = [m for m in methods if not hasattr(C, m)]\n"
        "assert not missing_m, f'missing methods: {missing_m}'\n"
        "print('DataFrame methods: OK', len(methods))\n"
        # _ColumnProxy methods via instance
        "df = C.from_table(pa.table({'x':[1,5,3]}))\n"
        "proxy = df['x']\n"
        "proxy_methods = ['isin','between','idxmax','idxmin',\n"
        "                 'mean','sum','min','max','std',\n"
        "                 'count','to_list','to_numpy','to_arrow',\n"
        "                 'isnull','notnull']\n"
        "missing_p = [m for m in proxy_methods if not hasattr(proxy, m)]\n"
        "assert not missing_p, f'missing proxy methods: {missing_p}'\n"
        "print('Proxy methods: OK', len(proxy_methods))\n"
        # tail explicitly absent (backlog)
        "assert not hasattr(C, 'tail'), 'tail now exists — update backlog'\n"
        "print('tail absent (backlog §13.4/314): OK')\n"
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
        "reorganization (B.1 step 20) + fix1 (init) + fix2 "
        "(sanity) + fix3 (drop tail). "
        "13 dead imports removed from _core.py. "
        "fix1 (§13.4/331): __init__.py fully rewritten — each "
        "re-export routed to actual source module post-split. "
        "fix2 (§13.4/332): sanity distinguishes CadwDataFrame vs "
        "_ColumnProxy methods (isin/between/idxmax/idxmin on "
        "proxy). fix3 (§13.4/333): drop `tail` from sanity — "
        "backlog feature (§13.4/314, never existed in CADW). "
        "_core.py ~558 → ~530 LOC. 0 logic change; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26y COMPLETE (fix1+2+3)")


if __name__ == "__main__":
    main()