# CADW time-series backtesting (v0.9.24).
#
# Two split strategies:
#   - expanding: train on [0, t_i), test on [t_i, t_i+horizon)
#     fold sizes grow: train = init_train + i*step
#   - sliding: fixed train size; train = [t_i - train_size, t_i)
#
# Metrics implemented (formulas):
#   MAE    = mean(|y - f|)
#   RMSE   = sqrt(mean((y - f)^2))
#   MAPE   = mean(|y - f| / |y|)  * 100 ; NaN if any |y|==0
#   sMAPE  = mean(2|y - f| / (|y| + |f|)) * 100 ; NaN if y+f==0 for all
#   MASE   = MAE / scale ; scale = mean(|y[t]-y[t-m]|) over the training
#            series, m = seasonal_period (default 1 => mean|dy|). MASE
#            < 1 beats naive random walk of lag m.
#   pinball(q) = mean( max(q*(y-f), (q-1)*(y-f)) ) for forecast f that is
#            the q-quantile. Only computed when quantiles is not None and
#            the forecaster exposes a predict(steps, return_quantiles=q)
#            or an internally stored quantile vector; otherwise a warning
#            is emitted (per fold) and no pinball row is produced.
#
# RAM: O(n_folds x batch) at any instant; results accumulate scalars
# only. No fold-level arrays are stored across folds. Forecaster state
# is clone-and-fit per fold (never reused).
from __future__ import annotations

import copy
import warnings
from typing import Sequence

import numpy as np


_DEFAULT_METRICS = ("mae", "rmse", "mape", "smape", "mase")


def _clone_forecaster(f):
    # Deep copy is safe for our forecasters: numpy arrays and plain
    # dicts only, no open file handles or Arrow datasets.
    return copy.deepcopy(f)


def _mae(y, f):
    return float(np.mean(np.abs(y - f)))


def _rmse(y, f):
    return float(np.sqrt(np.mean((y - f) ** 2)))


def _mape(y, f):
    if np.any(y == 0):
        return float("nan")
    return float(np.mean(np.abs(y - f) / np.abs(y)) * 100.0)


def _smape(y, f):
    denom = np.abs(y) + np.abs(f)
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y[mask] - f[mask]) / denom[mask]) * 100.0)


def _mase(y, f, train, m=1):
    if len(train) <= m:
        return float("nan")
    scale = float(np.mean(np.abs(train[m:] - train[:-m])))
    if scale == 0 or not np.isfinite(scale):
        return float("nan")
    return float(np.mean(np.abs(y - f)) / scale)


def _pinball(y, f, q):
    resid = y - f
    return float(np.mean(np.maximum(q * resid, (q - 1.0) * resid)))


def _folds_expanding(n, horizon, n_folds, initial_train, step):
    # Yields (train_end, test_end). Train = [0, train_end); test =
    # [train_end, test_end).  test length == horizon.
    out = []
    te = initial_train
    for _ in range(n_folds):
        test_end = te + horizon
        if test_end > n:
            break
        out.append((te, test_end))
        te = te + step
    return out


def _folds_sliding(n, horizon, n_folds, train_size, step):
    out = []
    te = train_size
    for _ in range(n_folds):
        test_end = te + horizon
        if test_end > n:
            break
        out.append((te, test_end))
        te = te + step
    return out


def _predict_quantiles(model, steps, q):
    # Best-effort support for quantile forecasters. If the model has a
    # 'predict_quantile' or 'predict' with return_quantiles kwarg, use
    # it. Otherwise return None (caller emits a warning per fold).
    for name in ("predict_quantile", "predict_quantiles"):
        if hasattr(model, name):
            try:
                return np.asarray(getattr(model, name)(steps, q), dtype=np.float64)
            except Exception:
                return None
    try:
        return np.asarray(model.predict(steps, return_quantiles=q), dtype=np.float64)
    except TypeError:
        return None


def backtest_forecaster(
    df, forecaster, *,
    target_col, time_col,
    group_col=None,
    horizon=1, n_folds=5,
    strategy="expanding",
    train_size=None,
    step=None,
    metrics=_DEFAULT_METRICS,
    quantiles=None,
    seasonal_period=1,
):
    """Backtest a CADW forecaster on a CadwDataFrame.

    Returns (summary_df, folds_df) as plain list-of-dicts (callers
    wrap in a DataFrame if desired; CADW DataFrame construction from
    these small tables is cheap).

    summary_df rows: {group, metric, mean, std, n_folds}
    folds_df rows:   {group, fold, metric, value, n_train, n_test}
    group value is None for single-series.

    strategy:
      'expanding' (default): initial_train = train_size or
        max(n // (n_folds + 1), horizon + 1); step default = horizon.
      'sliding': train_size REQUIRED; step default = horizon.

    Forecaster is CLONED per fold; fitted state does not leak.

    Groups with fewer than (train_size_required + horizon x n_folds)
    rows are skipped and a warning row {group, metric='SKIPPED',
    value=reason} is appended to folds_df.
    """
    if horizon < 1:
        raise ValueError(f"backtest: horizon must be >= 1; got {horizon}")
    if n_folds < 1:
        raise ValueError(f"backtest: n_folds must be >= 1; got {n_folds}")
    if strategy not in ("expanding", "sliding"):
        raise ValueError(
            f"backtest: strategy must be 'expanding' or 'sliding'; "
            f"got {strategy!r}"
        )
    if strategy == "sliding" and train_size is None:
        raise ValueError(
            "backtest: strategy='sliding' requires train_size"
        )
    metrics = tuple(metrics)
    for m in metrics:
        if m not in ("mae", "rmse", "mape", "smape", "mase"):
            raise ValueError(
                f"backtest: unknown metric {m!r}; valid: "
                f"mae, rmse, mape, smape, mase"
            )
    if target_col not in df._columns:
        raise KeyError(f"backtest: target_col {target_col!r} not in df")
    if time_col not in df._columns:
        raise KeyError(f"backtest: time_col {time_col!r} not in df")
    if group_col is not None and group_col not in df._columns:
        raise KeyError(f"backtest: group_col {group_col!r} not in df")

    # Partition into groups (streaming -> per-group rows in memory;
    # backtest series are typically small enough that the whole target
    # column for one group fits, but we still route extraction through
    # the MemoryGuard in _extract_series via the forecaster's fit).
    from .models import _extract_series

    group_values = [None]
    if group_col is not None:
        seen = []
        for batch in df._iter_arrow_batches(batch_size=10_000):
            vals = batch.column(group_col).to_pylist()
            for v in vals:
                if v not in seen:
                    seen.append(v)
        group_values = seen

    folds_rows = []
    summary_rows = []

    for gv in group_values:
        y = _extract_series(
            df, target_col, time_col=time_col,
            group_col=group_col, group_value=gv,
        )
        y = y[~np.isnan(y)]
        n = len(y)
        min_required = (
            (train_size if train_size else max(horizon + 1, n // (n_folds + 1)))
            + horizon * n_folds
        )
        if n < min_required:
            folds_rows.append({
                "group": gv, "fold": -1, "metric": "SKIPPED",
                "value": f"insufficient rows (n={n} < {min_required})",
                "n_train": 0, "n_test": 0,
            })
            warnings.warn(
                f"backtest: group {gv!r} skipped (n={n} < {min_required})",
                stacklevel=2,
            )
            continue

        if strategy == "expanding":
            init_train = (
                train_size if train_size is not None
                else max(horizon + 1, n // (n_folds + 1))
            )
            st = step if step is not None else horizon
            folds = _folds_expanding(n, horizon, n_folds, init_train, st)
        else:
            st = step if step is not None else horizon
            folds = _folds_sliding(n, horizon, n_folds, train_size, st)

        if not folds:
            folds_rows.append({
                "group": gv, "fold": -1, "metric": "SKIPPED",
                "value": "no valid folds", "n_train": 0, "n_test": 0,
            })
            warnings.warn(
                f"backtest: group {gv!r} produced no folds", stacklevel=2,
            )
            continue

        per_metric = {m: [] for m in metrics}
        pinball_supported = True
        if quantiles is not None:
            per_metric_q = {q: [] for q in quantiles}
        for fi, (te, test_end) in enumerate(folds):
            if strategy == "expanding":
                y_train = y[:te]
            else:
                y_train = y[te - train_size:te]
            y_test = y[te:test_end]
            n_train = len(y_train)
            n_test = len(y_test)

            # Clone + fit fresh per fold.
            model = _clone_forecaster(forecaster)
            # Fit expects a CadwDataFrame; build a tiny table per fold.
            import pyarrow as pa
            from cadw import CadwDataFrame
            train_df = CadwDataFrame.from_table(
                pa.table({
                    "__y__": pa.array(y_train, type=pa.float64()),
                })
            )
            model.fit(train_df, "__y__")
            fc = np.asarray(model.predict(horizon), dtype=np.float64)
            if len(fc) != horizon:
                fc = fc[:horizon]

            for m in metrics:
                if m == "mae":
                    v = _mae(y_test, fc)
                elif m == "rmse":
                    v = _rmse(y_test, fc)
                elif m == "mape":
                    v = _mape(y_test, fc)
                elif m == "smape":
                    v = _smape(y_test, fc)
                elif m == "mase":
                    v = _mase(y_test, fc, y_train, seasonal_period)
                folds_rows.append({
                    "group": gv, "fold": fi, "metric": m,
                    "value": v, "n_train": n_train, "n_test": n_test,
                })
                per_metric[m].append(v)

            if quantiles is not None and pinball_supported:
                for q in quantiles:
                    qfc = _predict_quantiles(model, horizon, q)
                    if qfc is None:
                        warnings.warn(
                            f"backtest: forecaster {type(forecaster).__name__} "
                            f"does not support quantile forecasts; "
                            f"pinball metrics skipped",
                            stacklevel=2,
                        )
                        pinball_supported = False
                        break
                    if len(qfc) != horizon:
                        qfc = qfc[:horizon]
                    v = _pinball(y_test, qfc, q)
                    folds_rows.append({
                        "group": gv, "fold": fi, "metric": f"pinball_{q}",
                        "value": v, "n_train": n_train, "n_test": n_test,
                    })
                    per_metric_q[q].append(v)

        for m, vals in per_metric.items():
            arr = np.asarray(vals, dtype=np.float64)
            valid = arr[~np.isnan(arr)]
            summary_rows.append({
                "group": gv, "metric": m,
                "mean": float(valid.mean()) if len(valid) else float("nan"),
                "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
                "n_folds": len(vals),
            })
        if quantiles is not None and pinball_supported:
            for q, vals in per_metric_q.items():
                arr = np.asarray(vals, dtype=np.float64)
                valid = arr[~np.isnan(arr)]
                summary_rows.append({
                    "group": gv, "metric": f"pinball_{q}",
                    "mean": float(valid.mean()) if len(valid) else float("nan"),
                    "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
                    "n_folds": len(vals),
                })

    return summary_rows, folds_rows


__all__ = ["backtest_forecaster"]
