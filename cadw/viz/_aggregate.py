"""Aggregate layer for viz (v0.7.4a).

Pure functions: input = CadwDataFrame, output = small numpy/Table.
No matplotlib dependency.

- hist     : Reduce(HistogramOp)     → (counts_np, edges_np)
- hist2d   : Reduce(Histogram2DOp)   → (counts_2d, x_edges, y_edges)
- box_stats: df.quantile             → dict
- line_agg : df.groupby().agg        → pa.Table (K rows)
- sample   : df.head(n)              → CadwDataFrame

Peak RAM: bounded by Reduce state (bins, k×d) or by sample size n.
Independent of N (for hist/hist2d), or O(n) small (for sample).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.histogram import Histogram2DOp
from cadw.ops.reduce import HistogramOp, reduce


def _minmax_exact(
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
    return lo, hi


def hist(
    df,
    column: str,
    bins: int = 50,
    *,
    memory_guard: MemoryGuard | None = None,
):
    """1D histogram → (counts: np.ndarray, edges: np.ndarray)."""
    if column not in df.columns:
        raise KeyError(f"hist: column {column!r} not in df")
    if bins < 1:
        raise ValueError(f"hist: bins must be >= 1; got {bins}")
    mg = memory_guard or df._memory_guard
    lo, hi = _minmax_exact(df, column, mg)
    op = HistogramOp(column, bins=bins, min_val=lo, max_val=hi)
    counts = reduce(df._iter_filtered_batches(), op, mg, log=False)
    edges = np.linspace(lo, hi, bins + 1)
    return counts.astype(np.int64), edges


def hist2d(
    df,
    x: str,
    y: str,
    bins: tuple[int, int] = (200, 200),
    *,
    memory_guard: MemoryGuard | None = None,
):
    """2D histogram → (counts: (bx, by) np.ndarray, x_edges, y_edges)."""
    for c in (x, y):
        if c not in df.columns:
            raise KeyError(f"hist2d: column {c!r} not in df")
    bx, by = int(bins[0]), int(bins[1])
    if bx < 1 or by < 1:
        raise ValueError(f"hist2d: bins must be >= 1; got {bins}")
    mg = memory_guard or df._memory_guard
    x_lo, x_hi = _minmax_exact(df, x, mg)
    y_lo, y_hi = _minmax_exact(df, y, mg)
    op = Histogram2DOp(
        x, y,
        bins_x=bx, bins_y=by,
        x_min=x_lo, x_max=x_hi, y_min=y_lo, y_max=y_hi,
    )
    result = reduce(df._iter_filtered_batches(), op, mg, log=False)
    return result["counts"], result["x_edges"], result["y_edges"]


def box_stats(
    df,
    column: str,
    *,
    quantiles: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict:
    """Boxplot stats via df.quantile. Returns dict {min, q1, median, q3, max}."""
    if column not in df.columns:
        raise KeyError(f"box_stats: column {column!r} not in df")
    if tuple(quantiles) != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError(
            f"box_stats: quantiles must be exactly (0, .25, .5, .75, 1); "
            f"got {tuple(quantiles)}"
        )
    q = df.quantile(column, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(q[0]),
        "q1": float(q[1]),
        "median": float(q[2]),
        "q3": float(q[3]),
        "max": float(q[4]),
    }


def line_agg(
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
    return order


def sample(df, n: int = 10_000, *, random: bool = False):
    """Bounded sample (v0.7.4c: +random).

    - random=False (default, fast): df.head(n). Bounded I/O (first n).
      **May be biased** if data is ordered (e.g. time-sorted parquet) —
      samples only prefix. Use for quick preview.
    - random=True: uniform random sample of n rows. Full O(N) scan
      (count + iloc[random_indices] → reads to max position). O(n) memory.
      Use for viz correctness on arbitrary/ordered data.

    Returns CadwDataFrame either way (count() <= n → returns df unchanged).
    """
    if n < 1:
        raise ValueError(f"sample: n must be >= 1; got {n}")
    cnt = df.count()
    if cnt <= n:
        return df
    if random:
        import numpy as _np
        rng = _np.random.default_rng(42)
        idx = _np.sort(rng.choice(cnt, size=n, replace=False))
        # df.iloc[list] → CadwDataFrame (v0.4.0b, filter-compat v0.4.0d)
        return df.iloc[idx.tolist()]
    # head (fast, may be biased)
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
    )
