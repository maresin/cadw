#!/usr/bin/env python3
"""v0.7.4b-fix: line_agg — sort by `by` column (line plot needs x-order).

Баг: df.groupby(by).agg() возвращает rows в hash-partition order, не sorted.
Plotly Scatter/Seaborn lineplot/matplotlib plot рисуют «зигзаг» если x не
отсортирован. Test ожидал ['A','B','C'], получил ['C','B','A'].

Fix: line_agg сортирует result по `by` ascending. K строк — прямая
сортировка (Table.sort_by) безопасна.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
        sys.exit(r.returncode)
    return r.returncode


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def patch_line_agg():
    print("\n=== cadw/viz/_aggregate.py: line_agg sorts by `by` ===")
    p = Path("cadw/viz/_aggregate.py")
    s = p.read_text()

    old = '''def line_agg(
    df,
    by: str,
    agg_col: str,
    agg: str = "mean",
) -> pa.Table:
    """Groupby aggregate for line plot. Returns small Table (K rows)."""
    for c in (by, agg_col):
        if c not in df.columns:
            raise KeyError(f"line_agg: column {c!r} not in df")
    if agg not in {"mean", "sum", "count", "min", "max"}:
        raise ValueError(f"line_agg: agg must be sum/mean/count/min/max; got {agg}")
    result_df = df.groupby(by).agg(agg_col, agg)
    # Materialize as pa.Table (K rows — safe)
    batches = list(result_df._iter_filtered_batches())
    if not batches:
        return pa.Table.from_batches([], schema=result_df._dataset.schema)
    return pa.Table.from_batches(batches)'''

    new = '''def line_agg(
    df,
    by: str,
    agg_col: str,
    agg: str = "mean",
) -> pa.Table:
    """Groupby aggregate for line plot, SORTED by `by` ascending.

    v0.7.4b-fix: df.groupby().agg() returns rows in hash-partition order,
    not sorted. Line renderers (plotly Scatter, sns.lineplot,
    matplotlib plot) draw line segments in given x-order — unsorted
    "zig-zag" segments. We sort here (K rows, safe — direct sort).

    Returns small Table (K rows), sorted by `by`.
    """
    for c in (by, agg_col):
        if c not in df.columns:
            raise KeyError(f"line_agg: column {c!r} not in df")
    if agg not in {"mean", "sum", "count", "min", "max"}:
        raise ValueError(f"line_agg: agg must be sum/mean/count/min/max; got {agg}")
    result_df = df.groupby(by).agg(agg_col, agg)
    batches = list(result_df._iter_filtered_batches())
    if not batches:
        return pa.Table.from_batches([], schema=result_df._dataset.schema)
    table = pa.Table.from_batches(batches)
    # Sort by `by` ascending (K rows — small, safe in-memory sort)
    order = table.sort_by([(by, "ascending")])
    return order'''

    s = replace_exact(s, old, new, tag="line-agg-sort")
    p.write_text(s)
    compile_check(p)
    print("  OK: line_agg sorts by `by`")


def main():
    print("=== v0.7.4b-fix ===")
    patch_line_agg()

    print("\n=== pytest tests/test_viz_plotly.py tests/test_viz_seaborn.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_viz_plotly.py",
         "tests/test_viz_seaborn.py",
         "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.4b-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_viz_aggregate.py tests/test_viz_matplotlib.py ===")
    sh(["uv", "run", "pytest",
        "tests/test_viz_aggregate.py",
        "tests/test_viz_matplotlib.py", "-q"])

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.4b-fix: line_agg — sort by `by` ascending (line plot order)"])

    print("\n==> v0.7.4b complete.")


if __name__ == "__main__":
    main()