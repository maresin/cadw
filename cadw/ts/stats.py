# CADW time-series window statistics (v0.9.24).
#
# tsfresh-subset window aggregations reimplemented on Arrow + numpy with
# explicit RAM bounds. One streaming pass per group; a single deque of
# size max(windows) per group is the only per-group state.
#
# RAM formula table (per group; n_groups=1 if ungrouped):
#
#   stat                     per-group state
#   ------------------------ -------------------------
#   mean/std/sum/min/max     O(max_window x 8 B)   (shared deque)
#   median/q10/q25/q50/q75/q90   O(max_window x 8 B)
#   skew/kurt                O(max_window x 8 B)
#   energy/abs_energy/cid    O(max_window x 8 B)
#   crossings/slope/entropy  O(max_window x 8 B)
#   count_above_mean         O(max_window x 8 B)
#   autocorr_lagK            O(max_window x 8 B)
#   ------------------------ -------------------------
#   TOTAL (all stats share one deque)  O(max_window x n_groups x 8 B)
#   per-batch pylist: O(batch x 3.5 x (1 + has_group_col) x 8 B)
#
# Window is a deque of the last max(windows) values per group; every
# statistic reads only the last w values. Computation per output point
# is O(w log w) for quantiles, O(w) otherwise -- independent of N.
#
# All statistics return NaN when the valid (non-null) count in the
# window is below min_periods, or when the statistic is undefined
# (e.g. skew with <3 points or std==0).
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import pyarrow as pa

from .features import (
    _BATCH, _PYLIST_FACTOR, _check, _finalize, _new_schema_with_extra,
    _sort_for_groups,
)


_ALL_STATS = frozenset({
    "mean", "std", "sum", "min", "max", "median",
    "skew", "kurt",
    "q10", "q25", "q50", "q75", "q90",
    "abs_energy", "energy", "cid",
    "crossings", "slope", "entropy", "count_above_mean",
})


def _parse_stat(s):
    if s.startswith("autocorr_lag"):
        try:
            k = int(s[len("autocorr_lag"):])
        except ValueError:
            raise ValueError(f"rolling_stats: invalid stat {s!r}")
        if k < 1:
            raise ValueError(
                f"rolling_stats: autocorr lag must be >= 1; got {k}"
            )
        return ("autocorr", k)
    if s not in _ALL_STATS:
        raise ValueError(
            f"rolling_stats: unknown stat {s!r}; valid={sorted(_ALL_STATS)} "
            f"or 'autocorr_lagK'"
        )
    return (s, None)


def _compute_stat(name, k, arr_w):
    valid = arr_w[~np.isnan(arr_w)]
    if name == "mean":
        return float(valid.mean()) if len(valid) else None
    if name == "std":
        return float(valid.std(ddof=1)) if len(valid) > 1 else None
    if name == "sum":
        return float(valid.sum()) if len(valid) else None
    if name == "min":
        return float(valid.min()) if len(valid) else None
    if name == "max":
        return float(valid.max()) if len(valid) else None
    if name == "median":
        return float(np.median(valid)) if len(valid) else None
    if name == "skew":
        if len(valid) < 3:
            return None
        s = valid.std(ddof=0)
        if s == 0:
            return None
        return float(np.mean(((valid - valid.mean()) / s) ** 3))
    if name == "kurt":
        if len(valid) < 4:
            return None
        s = valid.std(ddof=0)
        if s == 0:
            return None
        return float(np.mean(((valid - valid.mean()) / s) ** 4) - 3.0)
    if name in ("q10", "q25", "q50", "q75", "q90"):
        if not len(valid):
            return None
        q = int(name[1:]) / 100.0
        return float(np.percentile(valid, q * 100.0))
    if name == "abs_energy":
        return float(np.sum(valid ** 2)) if len(valid) else None
    if name == "energy":
        d = np.diff(arr_w)
        d = d[~np.isnan(d)]
        return float(np.sum(d ** 2)) if len(d) else None
    if name == "cid":
        d = np.diff(arr_w)
        d = d[~np.isnan(d)]
        return float(np.sqrt(np.sum(d ** 2))) if len(d) else None
    if name == "crossings":
        if len(valid) < 2:
            return None
        m = valid.mean()
        sg = np.sign(arr_w - m)
        sg = sg[~np.isnan(sg)]
        if len(sg) < 2:
            return None
        return float(np.sum(sg[:-1] * sg[1:] < 0))
    if name == "slope":
        mask = ~np.isnan(arr_w)
        if mask.sum() < 2:
            return None
        x = np.arange(len(arr_w), dtype=np.float64)[mask]
        y = arr_w[mask]
        xm = x.mean()
        denom = np.sum((x - xm) ** 2)
        if denom == 0:
            return None
        return float(np.sum((x - xm) * (y - y.mean())) / denom)
    if name == "entropy":
        if not len(valid):
            return None
        hist, _ = np.histogram(valid, bins=10)
        p = hist / hist.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))
    if name == "count_above_mean":
        if not len(valid):
            return None
        return float(np.sum(valid > valid.mean()))
    if name == "autocorr":
        if len(arr_w) <= k:
            return None
        a = arr_w[:-k]
        b = arr_w[k:]
        mask = ~(np.isnan(a) | np.isnan(b))
        a = a[mask]
        b = b[mask]
        if len(a) < 2:
            return None
        sa = a.std(ddof=0)
        sb = b.std(ddof=0)
        if sa == 0 or sb == 0:
            return None
        return float(
            np.sum((a - a.mean()) * (b - b.mean())) / (len(a) * sa * sb)
        )
    return None


def rolling_stats(
    df, column, windows, *,
    stats=("mean", "std"),
    group_col=None, time_col=None, min_periods=None,
):
    """Rolling window statistics for time-series feature extraction.

    - windows: int | Sequence[int], each >= 1.
    - stats: subset of:
        mean std sum min max median skew kurt
        q10 q25 q50 q75 q90
        abs_energy energy cid crossings slope entropy count_above_mean
        autocorr_lagK (K >= 1, K < window)
    - group_col: per-entity windows (no cross-entity leakage).
    - time_col: sort key for reproducibility (stable within group).
    - min_periods: int or None (=w). Minimum non-null count in window.

    Output: original + one column per (window, stat):
      {column}_roll{w}_{stat}
      {column}_roll{w}_autocorr_lag{K}

    RAM: O(max_window x n_groups x 8 B) state; per-batch pylist factor
    2.5 + 3.5 x (1 + has_group_col). Independent of N.
    """
    if isinstance(windows, int):
        windows = (windows,)
    windows = sorted({int(w) for w in windows})
    if not windows:
        raise ValueError("rolling_stats: windows is empty")
    if min(windows) < 1:
        raise ValueError(
            f"rolling_stats: windows must be >= 1; got {windows}"
        )
    if column not in df._columns:
        raise KeyError(f"rolling_stats: column {column!r} not in df")
    col_type = df._dataset.schema.field(column).type
    if not (
        pa.types.is_integer(col_type) or pa.types.is_floating(col_type)
    ):
        raise TypeError(
            f"rolling_stats: column {column!r} has type {col_type}; "
            f"numeric required"
        )
    if group_col is not None and group_col not in df._columns:
        raise KeyError(
            f"rolling_stats: group_col {group_col!r} not in df"
        )
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"rolling_stats: time_col {time_col!r} not in df")

    stats = list(stats)
    parsed = [_parse_stat(s) for s in stats]
    # validate autocorr lag < min(window)
    min_w = windows[0]
    for name, k in parsed:
        if name == "autocorr" and k >= min_w:
            raise ValueError(
                f"rolling_stats: autocorr lag {k} >= min(window) {min_w}"
            )

    if min_periods is None:
        mp_map = {w: w for w in windows}
    elif isinstance(min_periods, int):
        mp_map = {w: min_periods for w in windows}
    else:
        raise TypeError(
            f"rolling_stats: min_periods must be int or None; "
            f"got {type(min_periods).__name__}"
        )

    extras = []
    for w in windows:
        for s, (name, k) in zip(stats, parsed):
            if name == "autocorr":
                extras.append((f"{column}_roll{w}_autocorr_lag{k}", pa.float64()))
            else:
                extras.append((f"{column}_roll{w}_{name}", pa.float64()))
    out_schema = _new_schema_with_extra(df, extras)

    sorted_df = _sort_for_groups(df, group_col, time_col)
    n_cols = len(sorted_df._columns)
    max_w = windows[-1]
    extra_factor = 2.5 + _PYLIST_FACTOR * (1 + (group_col is not None))

    def gen():
        carry = {}
        for batch in sorted_df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch,
                   f"ts.rolling_stats.{column}", extra_factor)
            n = batch.num_rows
            vals = batch.column(column).to_pylist()
            grps = (
                batch.column(group_col).to_pylist()
                if group_col is not None else None
            )

            outs = {}
            for w in windows:
                for s in stats:
                    outs[(w, s)] = [None] * n

            for i in range(n):
                g = grps[i] if grps is not None else None
                dq = carry.get(g)
                if dq is None:
                    dq = deque(maxlen=max_w)
                    carry[g] = dq
                v = vals[i]
                dq.append(float(v) if v is not None else np.nan)
                # Materialize once per i: a numpy view over the deque.
                # Partial windows (len(dq) < w) are valid whenever
                # min_periods allows fewer than w points -- pandas
                # rolling(min_periods) semantics.
                full = np.fromiter(dq, dtype=np.float64, count=len(dq))
                for w in windows:
                    arr_w = full[-w:]  # partial if len(dq) < w
                    valid_count = int(np.sum(~np.isnan(arr_w)))
                    if valid_count < mp_map[w]:
                        continue
                    for s, (name, k) in zip(stats, parsed):
                        outs[(w, s)][i] = _compute_stat(name, k, arr_w)

            arrays = [batch.column(c) for c in range(n_cols)]
            for w in windows:
                for s in stats:
                    arrays.append(pa.array(outs[(w, s)], type=pa.float64()))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

    return _finalize(df, out_schema, gen)


def _pearson_xcorr(xs, ys, lag):
    # Pearson correlation between xs[t] and ys[t-lag].
    # lag > 0: y lags x (corr(x[t], y[t-lag])); lag = 0: concurrent.
    if lag == 0:
        a, b = xs, ys
    elif lag > 0:
        a = xs[lag:]
        b = ys[:-lag]
    else:
        L = -lag
        a = xs[:-L]
        b = ys[L:]
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return None
    sa = a.std(ddof=0)
    sb = b.std(ddof=0)
    if sa == 0 or sb == 0:
        return None
    return float(np.sum((a - a.mean()) * (b - b.mean())) / (len(a) * sa * sb))


def rolling_crosscorr(
    df, col_x, col_y, windows, *,
    lags=(0,),
    group_col=None, time_col=None, min_periods=None,
):
    """Rolling Pearson cross-correlation between two series.

    Output columns: {col_x}__{col_y}_roll{w}_xcorr_lag{K} for each
    (w, K). Pearson between x[t] and y[t-K]:
      K = 0 -> concurrent;
      K > 0 -> y lags x (x[t] correlates with y[t-K]);
      K < 0 -> x lags y.

    Compute bound: O(N x |lags| x max_window) -- each output point
    scans one window of size w for one lag. Windows are numpy views
    over a shared deque; no per-point list materialization.

    RAM formula (per group; n_groups=1 if ungrouped):
      - shared deque of (x, y) pairs: O(max_window x 16 B)
      - per-batch pylist overhead: explicit extra_factor =
        3.5 + 3.5 x (2 + has_group_col)  (three pylist columns:
        x, y, optional group_col)
      - total state: O(max_window x n_groups x 16 B); independent of N.

    Valid only when non-null paired count in the window >= min_periods
    (default min_periods = w).
    """
    if isinstance(windows, int):
        windows = (windows,)
    windows = sorted({int(w) for w in windows})
    if not windows:
        raise ValueError("rolling_crosscorr: windows is empty")
    if min(windows) < 1:
        raise ValueError(
            f"rolling_crosscorr: windows must be >= 1; got {windows}"
        )
    if isinstance(lags, int):
        lags = (lags,)
    lags = sorted({int(k) for k in lags})
    min_w = windows[0]
    for K in lags:
        if abs(K) >= min_w:
            raise ValueError(
                f"rolling_crosscorr: |lag| {abs(K)} >= min(window) {min_w}"
            )
    if col_x not in df._columns:
        raise KeyError(f"rolling_crosscorr: {col_x!r} not in df")
    if col_y not in df._columns:
        raise KeyError(f"rolling_crosscorr: {col_y!r} not in df")
    if col_x == col_y:
        raise ValueError(
            "rolling_crosscorr: col_x and col_y must differ; "
            "use rolling_stats(autocorr_lagK) for self-correlation"
        )
    for c in (col_x, col_y):
        t = df._dataset.schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"rolling_crosscorr: column {c!r} has type {t}; "
                f"numeric required"
            )
    if group_col is not None and group_col not in df._columns:
        raise KeyError(
            f"rolling_crosscorr: group_col {group_col!r} not in df"
        )
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"rolling_crosscorr: time_col {time_col!r} not in df")

    if min_periods is None:
        mp_map = {w: w for w in windows}
    elif isinstance(min_periods, int):
        mp_map = {w: min_periods for w in windows}
    else:
        raise TypeError(
            f"rolling_crosscorr: min_periods must be int or None; "
            f"got {type(min_periods).__name__}"
        )

    extras = []
    for w in windows:
        for K in lags:
            extras.append((
                f"{col_x}__{col_y}_roll{w}_xcorr_lag{K}", pa.float64()
            ))
    out_schema = _new_schema_with_extra(df, extras)

    sorted_df = _sort_for_groups(df, group_col, time_col)
    n_cols = len(sorted_df._columns)
    max_w = windows[-1]
    extra_factor = 3.5 + _PYLIST_FACTOR * (2 + (group_col is not None))

    def gen():
        carry = {}
        for batch in sorted_df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch,
                   f"ts.rolling_crosscorr.{col_x}__{col_y}", extra_factor)
            n = batch.num_rows
            xs = batch.column(col_x).to_pylist()
            ys = batch.column(col_y).to_pylist()
            grps = (
                batch.column(group_col).to_pylist()
                if group_col is not None else None
            )
            outs = {}
            for w in windows:
                for K in lags:
                    outs[(w, K)] = [None] * n

            for i in range(n):
                g = grps[i] if grps is not None else None
                dq = carry.get(g)
                if dq is None:
                    dq = deque(maxlen=max_w)
                    carry[g] = dq
                xv = float(xs[i]) if xs[i] is not None else np.nan
                yv = float(ys[i]) if ys[i] is not None else np.nan
                dq.append((xv, yv))
                # Materialize paired arrays once per i.
                arr = np.fromiter(
                    (v for pair in dq for v in pair),
                    dtype=np.float64, count=2 * len(dq),
                ).reshape(-1, 2)
                arr_x = arr[:, 0]
                arr_y = arr[:, 1]
                for w in windows:
                    ax = arr_x[-w:]
                    ay = arr_y[-w:]
                    valid = int(np.sum(~(np.isnan(ax) | np.isnan(ay))))
                    if valid < mp_map[w]:
                        continue
                    for K in lags:
                        outs[(w, K)][i] = _pearson_xcorr(ax, ay, K)

            arrays = [batch.column(c) for c in range(n_cols)]
            for w in windows:
                for K in lags:
                    arrays.append(pa.array(outs[(w, K)], type=pa.float64()))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

    return _finalize(df, out_schema, gen)
