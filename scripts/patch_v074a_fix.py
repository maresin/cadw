#!/usr/bin/env python3
"""v0.7.4a-fix: exact min/max для hist/hist2d; sample возвращает CadwDataFrame.

Баг 1: _minmax_from_sample берёт first batch (200 строк). HistogramOp
использует np.histogram(range=(lo,hi)) — out-of-range отбрасываются
(9715/10000). hist2d использует clip — не теряет, но искажает.
Fix: MinMaxOp — один Reduce, O(1) state. Exact для обеих.

Баг 2: df.head(n) возвращает pa.Table. sample(head(n)) → pa.Table,
не CadwDataFrame. Fix: обернуть в CadwDataFrame.from_batches.
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


def patch_aggregate():
    print("\n=== cadw/viz/_aggregate.py ===")
    p = Path("cadw/viz/_aggregate.py")
    s = p.read_text()

    # 1. Replace _minmax_from_sample with _minmax_exact
    old_minmax = '''def _minmax_from_sample(
    df, column: str, *, pad_ratio: float = 0.01,
) -> tuple[float, float]:
    """Sample-based min/max from first batch. Fast but approximate.

    Padding ±pad_ratio handles out-of-sample outliers (they bin into
    edge buckets). For exact min/max — use Reduce(MinMaxOp) explicitly.
    """
    first = None
    for b in df._iter_filtered_batches():
        first = b
        break
    if first is None or first.num_rows == 0:
        raise ValueError(f"_minmax_from_sample: empty df for {column!r}")
    arr = first.column(column).to_numpy(zero_copy_only=False)
    if arr.dtype == object:
        vals = np.array(
            [np.nan if v is None else float(v) for v in arr],
            dtype=np.float64,
        )
    else:
        vals = arr.astype(np.float64, copy=False)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        raise ValueError(f"_minmax_from_sample: no valid values in {column!r}")
    lo = float(vals.min())
    hi = float(vals.max())
    if hi == lo:
        hi = lo + 1.0
    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad'''

    new_minmax = '''def _minmax_exact(
    df, column: str, memory_guard: MemoryGuard,
) -> tuple[float, float]:
    """Exact min/max via Reduce(MinMaxOp). One pass, O(1) state.

    v0.7.4a-fix: previous sample-based min/max lost out-of-range points
    (HistogramOp uses np.histogram(range=(lo,hi)) — discards them).
    For viz layer, exact is cheap (~one pass, same cost as histogram) and
    preserves `counts.sum() == non-null rows` invariant.
    """
    from cadw.ops.reduce import MinMaxOp
    if column not in df.columns:
        raise KeyError(f"_minmax_exact: column {column!r} not in df")
    mn = reduce(df._iter_filtered_batches(), MinMaxOp(column, "min"),
                memory_guard, log=False)
    mx = reduce(df._iter_filtered_batches(), MinMaxOp(column, "max"),
                memory_guard, log=False)
    if mn is None or mx is None:
        raise ValueError(
            f"_minmax_exact: no valid values in column {column!r} "
            f"(all null?)"
        )
    lo = float(mn)
    hi = float(mx)
    if hi == lo:
        hi = lo + 1.0
    return lo, hi'''

    s = replace_exact(s, old_minmax, new_minmax, tag="minmax-exact")

    # 2. Update hist to use _minmax_exact
    s = replace_exact(
        s,
        '    lo, hi = _minmax_from_sample(df, column)',
        '    lo, hi = _minmax_exact(df, column, mg)',
        tag="hist-minmax",
    )

    # 3. Update hist2d
    s = replace_exact(
        s,
        '''    x_lo, x_hi = _minmax_from_sample(df, x)
    y_lo, y_hi = _minmax_from_sample(df, y)''',
        '''    x_lo, x_hi = _minmax_exact(df, x, mg)
    y_lo, y_hi = _minmax_exact(df, y, mg)''',
        tag="hist2d-minmax",
    )

    # 4. Fix sample to wrap head(n) as CadwDataFrame
    old_sample = '''def sample(df, n: int = 10_000):
    """Bounded sample: returns df if count() <= n, else df.head(n)."""
    if n < 1:
        raise ValueError(f"sample: n must be >= 1; got {n}")
    cnt = df.count()
    if cnt <= n:
        return df
    return df.head(n)'''
    new_sample = '''def sample(df, n: int = 10_000):
    """Bounded sample: returns df if count() <= n, else df.head(n).

    v0.7.4a-fix: df.head(n) returns pa.Table, not CadwDataFrame.
    Wrap via Dataset.from_batches to keep API consistent.
    """
    if n < 1:
        raise ValueError(f"sample: n must be >= 1; got {n}")
    cnt = df.count()
    if cnt <= n:
        return df
    from cadw.dataframe import CadwDataFrame
    from cadw.dataset import Dataset
    table = df.head(n)
    new_ds = Dataset.from_batches(
        table.schema, iter(table.to_batches()), df._new_dataset_dir(),
    )
    return CadwDataFrame(
        new_ds,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir,
    )'''
    s = replace_exact(s, old_sample, new_sample, tag="sample-wrap")

    p.write_text(s)
    compile_check(p)
    print("  OK: _minmax_exact + sample wraps CadwDataFrame")


def main():
    print("=== v0.7.4a-fix ===")
    patch_aggregate()

    print("\n=== pytest tests/test_viz_aggregate.py tests/test_viz_matplotlib.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_viz_aggregate.py",
         "tests/test_viz_matplotlib.py",
         "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.4a-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.4a-fix: exact min/max via MinMaxOp (hist/hist2d); sample wraps CadwDataFrame"])

    print("\n==> v0.7.4a complete.")


if __name__ == "__main__":
    main()