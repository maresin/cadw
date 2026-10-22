# CADW time-series feature engineering (v0.9.24).
#
# Streaming, batch-wise feature construction for TS forecasting. All
# functions append new columns to a CadwDataFrame (matching the .dt
# accessor convention) and preserve or re-establish row order.
#
# RAM strategy: O(batch) state per function; carry buffers between
# batches bounded by max_lag x n_groups x row_bytes. Every batch goes
# through MemoryGuard.check_batch with an explicit extra_factor that
# accounts for pylist overhead (3.5x per pylist column, AUDIT_FULL 3.2).
#
# Group-aware semantics: group_col=None -> positional over
# _iter_arrow_batches (matches df.shift). group_col given -> data is
# sorted by [group_col, time_col] before streaming so carry buffers
# never cross group boundaries; output row order equals sorted order.
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from cadw.dataset import Dataset

_BATCH = 10_000
_PYLIST_FACTOR = 3.5   # AUDIT_FULL 3.2: pylist overhead vs Arrow, per col


def _check(guard, batch, name, extra_factor):
    if guard is None:
        return
    guard.check_batch(batch, name=name, extra_factor=extra_factor)


def _new_schema_with_extra(df, extras):
    fields = []
    for c in df._columns:
        f = df._dataset.schema.field(c)
        fields.append(pa.field(c, f.type, nullable=True))
    for name, typ in extras:
        fields.append(pa.field(name, typ, nullable=True))
    return pa.schema(fields)


def _finalize(df, out_schema, gen):
    from cadw.dataframe import CadwDataFrame
    new_ds = Dataset.from_batches(out_schema, gen(), df._new_dataset_dir())
    return CadwDataFrame(
        new_ds, memory_guard=df._memory_guard, spill_dir=df._spill_dir,
    )


def _sort_for_groups(df, group_col, time_col):
    keys = []
    if group_col is not None:
        keys.append(group_col)
    if time_col is not None:
        keys.append(time_col)
    if not keys:
        return df
    return df.sort_values(keys)


def _sort_multi(df, keys):
    # Per-column ascending sort [(col, ascending), ...].
    # df.sort_values only supports scalar ascending (list -> truthy bug),
    # so call external_sort directly with a per-column ascending sequence.
    from cadw.ops.sort import external_sort
    from cadw.io.batch_reader import ArrowIPCBatchReader
    from cadw.dataframe import CadwDataFrame
    cols = [c for c, _ in keys]
    asc = [a for _, a in keys]
    path = external_sort(
        df._iter_filtered_batches(),
        key_columns=cols,
        value_columns=df._columns,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir / "sort",
        ascending=asc,
        log=False,
    )
    schema = pa.schema(
        [df._dataset.schema.field(c) for c in df._columns]
    )

    def _iter():
        for b in ArrowIPCBatchReader(str(path), df._columns):
            yield b

    new_ds = Dataset.from_batches(schema, _iter(), df._new_dataset_dir())
    return CadwDataFrame(
        new_ds, memory_guard=df._memory_guard, spill_dir=df._spill_dir,
    )


def _shift_stream(
    df, column, ks, *, group_col, time_col, fill_value, direction,
    _already_sorted=False, _out_prefix=None,
):
    if column not in df._columns:
        raise KeyError(f"lag_features: column {column!r} not in df")
    col_type = df._dataset.schema.field(column).type
    if group_col is not None and group_col not in df._columns:
        raise KeyError(
            f"lag_features: group_col {group_col!r} not in df"
        )
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"lag_features: time_col {time_col!r} not in df")

    max_k = max(ks)
    prefix = _out_prefix or f"{column}_{direction}"
    extras = [(f"{prefix}{k}", col_type) for k in ks]
    out_schema = _new_schema_with_extra(df, extras)

    if _already_sorted:
        sorted_df = df
    else:
        sorted_df = _sort_for_groups(df, group_col, time_col)

    n_cols = len(sorted_df._columns)
    extra_factor = 2.0 + _PYLIST_FACTOR * (1 + (group_col is not None))

    def gen():
        carry = {}
        for batch in sorted_df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch,
                   f"ts.{direction}.{column}", extra_factor)
            n = batch.num_rows
            vals = batch.column(column).to_pylist()
            if group_col is not None:
                grps = batch.column(group_col).to_pylist()
            else:
                grps = None

            out = {k: [None] * n for k in ks}
            if grps is None:
                hist = carry.get(None)
                if hist is None:
                    hist = deque(maxlen=max_k)
                    carry[None] = hist
                for i in range(n):
                    for k in ks:
                        out[k][i] = hist[-k] if len(hist) >= k else fill_value
                    hist.append(vals[i])
            else:
                for i in range(n):
                    g = grps[i]
                    hist = carry.get(g)
                    if hist is None:
                        hist = deque(maxlen=max_k)
                        carry[g] = hist
                    for k in ks:
                        out[k][i] = hist[-k] if len(hist) >= k else fill_value
                    hist.append(vals[i])

            arrays = [batch.column(c) for c in range(n_cols)]
            for k in ks:
                arrays.append(pa.array(out[k], type=col_type))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

    return _finalize(df, out_schema, gen)


def lag_features(
    df, column, lags=(1,),
    *, group_col=None, time_col=None, fill_value=None,
):
    # Append {column}_lag{k} for each k in lags (k >= 1).
    lags = sorted({int(k) for k in lags})
    if not lags:
        raise ValueError("lag_features: lags is empty")
    if min(lags) < 1:
        raise ValueError(f"lag_features: lags must be >= 1; got {lags}")
    return _shift_stream(
        df, column, lags,
        group_col=group_col, time_col=time_col,
        fill_value=fill_value, direction="lag",
    )


def lead_features(
    df, column, leads=(1,),
    *, group_col=None, time_col=None, fill_value=None,
):
    # Append {column}_lead{k} for each k in leads (k >= 1).
    # Implementation: reverse time order, lag, rename, restore order.
    leads = sorted({int(k) for k in leads})
    if not leads:
        raise ValueError("lead_features: leads is empty")
    if min(leads) < 1:
        raise ValueError(f"lead_features: leads must be >= 1; got {leads}")
    if time_col is None:
        raise ValueError("lead_features: time_col required")
    keys = []
    if group_col is not None:
        keys.append((group_col, True))
    keys.append((time_col, False))
    rev = _sort_multi(df, keys)
    lagged = _shift_stream(
        rev, column, leads,
        group_col=group_col, time_col=time_col,
        fill_value=fill_value, direction="lag",
        _already_sorted=True,
    )
    rename_map = {f"{column}_lag{k}": f"{column}_lead{k}" for k in leads}
    back_keys = ([(group_col, True)] if group_col is not None else []) + [(time_col, True)]
    fwd = _sort_multi(lagged, back_keys)
    return fwd.rename(rename_map)


def diff_features(
    df, column, lags=(1,), *,
    group_col=None, time_col=None, seasonal_periods=None,
):
    # Append {column}_diff{k} (and {column}_seasonal_diff{s}_{k}).
    # Computes cur - lag(k) directly with its own carry buffer; no
    # intermediate lag columns are materialized.
    # RAM: O(max_lag) ungrouped / O(max_lag x n_groups) grouped carry,
    # plus one batch worth of pylist per column (factor 3.5 per col).
    if isinstance(lags, int):
        lags = (lags,)
    lags = sorted({int(k) for k in lags})
    if not lags:
        raise ValueError("diff_features: lags is empty")
    if min(lags) < 1:
        raise ValueError(f"diff_features: lags must be >= 1; got {lags}")
    if column not in df._columns:
        raise KeyError(f"diff_features: column {column!r} not in df")
    col_type = df._dataset.schema.field(column).type
    if not (pa.types.is_integer(col_type) or pa.types.is_floating(col_type)):
        raise TypeError(
            f"diff_features: column {column!r} has type {col_type}; "
            f"numeric required"
        )
    if group_col is not None and group_col not in df._columns:
        raise KeyError(
            f"diff_features: group_col {group_col!r} not in df"
        )
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"diff_features: time_col {time_col!r} not in df")

    sps = tuple(seasonal_periods) if seasonal_periods else ()
    all_ks = set(lags)
    for sp in sps:
        for k in lags:
            all_ks.add(int(sp) * int(k))
    all_ks = sorted(all_ks)
    max_k = all_ks[-1]

    extras = [(f"{column}_diff{k}", col_type) for k in lags]
    for sp in sps:
        for k in lags:
            extras.append((f"{column}_seasonal_diff{sp}_{k}", col_type))
    out_schema = _new_schema_with_extra(df, extras)

    sorted_df = _sort_for_groups(df, group_col, time_col)
    n_cols = len(sorted_df._columns)
    extra_factor = 2.0 + _PYLIST_FACTOR * (1 + (group_col is not None))

    def gen():
        carry = {}
        for batch in sorted_df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch,
                   f"ts.diff.{column}", extra_factor)
            n = batch.num_rows
            vals = batch.column(column).to_pylist()
            grps = (
                batch.column(group_col).to_pylist()
                if group_col is not None else None
            )
            out = {key: [None] * n for key, _ in extras}
            for i in range(n):
                g = grps[i] if grps is not None else None
                hist = carry.get(g)
                if hist is None:
                    hist = deque(maxlen=max_k)
                    carry[g] = hist
                cur = vals[i]
                for k in all_ks:
                    if len(hist) >= k and cur is not None and hist[-k] is not None:
                        diff = float(cur) - float(hist[-k])
                        for reg_k in lags:
                            if reg_k == k:
                                out[f"{column}_diff{reg_k}"][i] = diff
                        for sp in sps:
                            if sp * 1 == k or any(sp * rk == k for rk in lags):
                                for rk in lags:
                                    if sp * rk == k:
                                        out[f"{column}_seasonal_diff{sp}_{rk}"][i] = diff
                hist.append(cur)
            arrays = [batch.column(c) for c in range(n_cols)]
            for key, typ in extras:
                arrays.append(pa.array(out[key], type=typ))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

    return _finalize(df, out_schema, gen)


# ==================== calendar / cyclical / fourier ====================

_CALENDAR_PC = {
    "year": pc.year,
    "month": pc.month,
    "day": pc.day,
    "hour": pc.hour,
    "minute": pc.minute,
    "second": pc.second,
    "weekday": pc.day_of_week,
    "day_of_year": pc.day_of_year,
    "quarter": pc.quarter,
    "week_of_year": pc.week,
}


def _validate_temporal(df, column):
    t = df._dataset.schema.field(column).type
    if not (pa.types.is_timestamp(t) or pa.types.is_date(t)):
        raise TypeError(
            f"ts: column {column!r} has type {t}; expected timestamp or date"
        )


def calendar_features(
    df, ts_col, features=("year", "month", "day", "weekday", "hour"),
) -> "CadwDataFrame":
    # Append {ts_col}_{feature} for each feature (Arrow pc.* kernels).
    # RAM: O(batch) state; pure vectorized per batch; no carry buffer.
    if ts_col not in df._columns:
        raise KeyError(f"calendar_features: {ts_col!r} not in df")
    _validate_temporal(df, ts_col)
    for f in features:
        if f not in _CALENDAR_PC:
            raise ValueError(
                f"calendar_features: unknown feature {f!r}; "
                f"valid={sorted(_CALENDAR_PC)}"
            )
    extras = [(f"{ts_col}_{f}", pa.int64()) for f in features]
    out_schema = _new_schema_with_extra(df, extras)
    n_cols = len(df._columns)

    def gen():
        for batch in df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch, f"ts.calendar.{ts_col}", 2.0)
            arr = batch.column(ts_col)
            arrays = [batch.column(c) for c in range(n_cols)]
            for f in features:
                arrays.append(
                    pc.cast(_CALENDAR_PC[f](arr), pa.int64())
                )
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
    return _finalize(df, out_schema, gen)


def cyclical_features(
    df, ts_col, features=("month", "weekday", "hour"),
) -> "CadwDataFrame":
    # Append {ts_col}_{f}_sin / _cos for each feature.
    # Period: month=12, weekday=7, hour=24, day_of_year=365, quarter=4.
    # RAM: O(batch) state; vectorized; no carry buffer.
    periods = {
        "month": 12, "weekday": 7, "hour": 24,
        "day_of_year": 366, "quarter": 4, "week_of_year": 53,
    }
    if ts_col not in df._columns:
        raise KeyError(f"cyclical_features: {ts_col!r} not in df")
    _validate_temporal(df, ts_col)
    for f in features:
        if f not in periods:
            raise ValueError(
                f"cyclical_features: unknown feature {f!r}; "
                f"valid={sorted(periods)}"
            )
    extras = []
    for f in features:
        extras.append((f"{ts_col}_{f}_sin", pa.float64()))
        extras.append((f"{ts_col}_{f}_cos", pa.float64()))
    out_schema = _new_schema_with_extra(df, extras)
    n_cols = len(df._columns)

    def gen():
        for batch in df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch, f"ts.cyclical.{ts_col}", 2.5)
            arr = batch.column(ts_col)
            arrays = [batch.column(c) for c in range(n_cols)]
            for f in features:
                p = periods[f]
                if f == "weekday":
                    base = pc.cast(pc.day_of_week(arr), pa.int64())
                elif f == "day_of_year":
                    base = pc.cast(pc.day_of_year(arr), pa.int64())
                elif f == "week_of_year":
                    base = pc.cast(pc.week(arr), pa.int64())
                else:
                    base = pc.cast(_CALENDAR_PC[f](arr), pa.int64())
                ang = pc.multiply(
                    pc.cast(base, pa.float64()),
                    pa.scalar(2.0 * np.pi / p, type=pa.float64()),
                )
                arrays.append(pc.sin(ang))
                arrays.append(pc.cos(ang))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
    return _finalize(df, out_schema, gen)


def fourier_features(
    df, time_col, period, order, prefix=None,
) -> "CadwDataFrame":
    # Append {prefix}_sin{i} / {prefix}_cos{i} for i=1..order.
    # Uses time_col as numeric seconds (int64) or timestamp cast to sec.
    # RAM: O(batch) state; vectorized; no carry buffer.
    if time_col not in df._columns:
        raise KeyError(f"fourier_features: {time_col!r} not in df")
    if int(order) < 1:
        raise ValueError(f"fourier_features: order must be >= 1; got {order}")
    if period <= 0:
        raise ValueError(f"fourier_features: period must be > 0; got {period}")
    order = int(order)
    prefix = prefix or f"{time_col}_fourier"
    extras = []
    for i in range(1, order + 1):
        extras.append((f"{prefix}_sin{i}", pa.float64()))
        extras.append((f"{prefix}_cos{i}", pa.float64()))
    out_schema = _new_schema_with_extra(df, extras)
    n_cols = len(df._columns)
    col_type = df._dataset.schema.field(time_col).type
    is_ts = pa.types.is_timestamp(col_type)

    def gen():
        for batch in df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch, f"ts.fourier.{time_col}", 2.5)
            arr = batch.column(time_col)
            if is_ts:
                factor = {"s": 1, "ms": 1_000, "us": 1_000_000,
                          "ns": 1_000_000_000}[arr.type.unit]
                ints = pc.cast(arr, pa.int64())
                secs = pc.divide(ints, pa.scalar(factor, type=pa.int64()))
                t = pc.cast(secs, pa.float64())
            else:
                t = pc.cast(arr, pa.float64())
            arrays = [batch.column(c) for c in range(n_cols)]
            for i in range(1, order + 1):
                ang = pc.multiply(
                    t, pa.scalar(2.0 * np.pi * i / period, type=pa.float64())
                )
                arrays.append(pc.sin(ang))
                arrays.append(pc.cos(ang))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
    return _finalize(df, out_schema, gen)


def holiday_features(df, ts_col) -> "CadwDataFrame":
    # Append {ts_col}_is_weekend and {ts_col}_is_month_start/end.
    # RAM: O(batch); vectorized pc kernels; no carry buffer.
    if ts_col not in df._columns:
        raise KeyError(f"holiday_features: {ts_col!r} not in df")
    _validate_temporal(df, ts_col)
    extras = [
        (f"{ts_col}_is_weekend", pa.bool_()),
        (f"{ts_col}_is_month_start", pa.bool_()),
        (f"{ts_col}_is_month_end", pa.bool_()),
    ]
    out_schema = _new_schema_with_extra(df, extras)
    n_cols = len(df._columns)
    is_ts = pa.types.is_timestamp(df._dataset.schema.field(ts_col).type)

    def gen():
        for batch in df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch, f"ts.holiday.{ts_col}", 2.0)
            arr = batch.column(ts_col)
            wd = pc.cast(pc.day_of_week(arr), pa.int64())
            is_weekend = pc.is_in(
                wd, value_set=pa.array([5, 6], type=pa.int64())
            )
            d = pc.cast(pc.day(arr), pa.int64())
            is_month_start = pc.equal(d, pa.scalar(1, type=pa.int64()))
            # month_end: day(arr + 1 unit) == 1 (safe for ts); for date
            # fallback: day + 1 equals 1 -> never; use day==28/29/30/31
            # heuristic via month comparison.
            if is_ts:
                try:
                    nd = pc.cast(pc.day(pc.add(arr, pa.scalar(1, type=arr.type))),
                                 pa.int64())
                    is_month_end = pc.equal(nd, pa.scalar(1, type=pa.int64()))
                except Exception:
                    is_month_end = pc.fill_null(
                        pc.equal(pc.cast(pc.month(arr), pa.int64()),
                                 pa.scalar(-1, type=pa.int64())),
                        False,
                    )
            else:
                nd = pc.cast(pc.day(pc.add(arr, pa.scalar(1, type=pa.int32()))),
                             pa.int64())
                is_month_end = pc.equal(nd, pa.scalar(1, type=pa.int64()))
            arrays = [batch.column(c) for c in range(n_cols)]
            arrays.extend([is_weekend, is_month_start, is_month_end])
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
    return _finalize(df, out_schema, gen)


# ==================== convenience facade ====================

def make_features(
    df, *, target_col, ts_col,
    lags=(1, 7), diffs=(1,), seasonal_periods=None,
    calendar=("month", "weekday", "hour"),
    cyclical=("month", "weekday", "hour"),
    fourier=(24, 7), fourier_order=3,
    group_col=None,
    holiday=True,
) -> "CadwDataFrame":
    # Chain feature builders in the canonical order: calendar -> cyclical
    # -> fourier -> holiday -> lag -> diff. Returns df with all appended
    # columns; row order is the group-sorted order when group_col given.
    out = df
    if calendar:
        out = calendar_features(out, ts_col, calendar)
    if cyclical:
        out = cyclical_features(out, ts_col, cyclical)
    if fourier:
        for p in fourier:
            out = fourier_features(out, ts_col, period=p,
                                   order=fourier_order,
                                   prefix=f"{ts_col}_f{p}")
    if holiday:
        out = holiday_features(out, ts_col)
    if lags:
        out = lag_features(out, target_col, lags,
                            group_col=group_col, time_col=ts_col)
    if diffs:
        out = diff_features(out, target_col, diffs,
                             group_col=group_col, time_col=ts_col,
                             seasonal_periods=seasonal_periods)
    return out
