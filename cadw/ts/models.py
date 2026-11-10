# CADW time-series forecasters (v0.9.24).
#
# Three forecasters on a common API:
#   - NaiveForecaster(strategy='last'|'seasonal'|'drift')
#   - SARIMAForecaster(order=None, seasonal_order=None, s=None, auto=True)
#   - GBDTForecaster(mode='direct'|'recursive'|'multi_output', ...)
#
# Common API contract (ForecasterBase):
#   fit(df, target_col, *, time_col=None, group_col=None, **kwargs)
#   predict(steps)   -> np.ndarray (len=steps), 1-D, finite point forecasts
#   predict_in_sample() -> np.ndarray (len=n_train)
# Predict before fit raises RuntimeError. steps < 1 raises ValueError.
#
# SARIMA: CSS (conditional sum of squares) fit via iterative Hannan-
# Rissanen. Supports regular ARIMA(p,d,q) and optional seasonal
# (P,D,Q,s). Auto-ARIMA grid (AIC):
#   - order=None and auto=True:
#       p,q in 0..2, d in 0..1  -> 18 combos
#       if seasonal (s and seasonal_order=None):
#           P,Q,D in 0..1 -> 8x more combos (144 total)
#       if seasonal_order=None and s=None: seasonal=False (18 combos)
#   - order/seasonal_order explicitly set: auto disabled, single fit.
# AIC = n_obs * ln(sigma2) + 2 * n_params. Bounded grid prevents O(N) flail.
#
# GBDT: three modes; default is 'direct' (per plan):
#   - 'recursive': one model, feeding predictions back as inputs.
#     Propagates error across steps; keep for parity, not recommended.
#   - 'direct' (default): h independent models, one per horizon. No
#     error propagation, no feature build leakage.
#   - 'multi_output': h independent models predicting vector jointly
#     (CADW GBDT has no native multi-target; this is 'multi-output
#     (h-independent)'. Explicit, documented).
#
# All forecasters route per-batch data through MemoryGuard via
# make_features / _extract_series helpers. GBDT additionally uses
# cadw.ml.gbdt (which routes Reduce ops through the guard).
from __future__ import annotations
from cadw.core.approximation import ApproximationMixin

from typing import Optional

import numpy as np
import pyarrow as pa

from .features import _BATCH, _check, make_features


def _extract_series(df, target_col, *, time_col=None, group_col=None,
                     group_value=None):
    # Extract target as float64 numpy array; optionally filter to one
    # group, optionally sort by time_col. Streams batches, no full-frame
    # materialization.
    #
    # v0.9.90-S5: two paths.
    #   - No group filter: pre-allocate np.empty(df.count(), float64),
    #     fill per batch. Peak = n*8 bytes (not n*~36 via Python list);
    #     guard.check_state(n*8) fires before alloc.
    #   - Group filter: Python-list path retained (per-series n is
    #     small; filtered count would require two-pass scan). Guarded
    #     per batch by _check.
    if target_col not in df._columns:
        raise KeyError(f"target column {target_col!r} not in df")
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"time_col {time_col!r} not in df")
    if group_col is not None and group_col not in df._columns:
        raise KeyError(f"group_col {group_col!r} not in df")

    if time_col is not None:
        keys = ([group_col] if group_col is not None else []) + [time_col]
        df = df.sort_values(keys)

    filter_active = (group_col is not None and group_value is not None)

    if not filter_active:
        n_total = int(df.count())
        if n_total > 0 and df._memory_guard is not None:
            df._memory_guard.check_state(
                n_total * 8, name="ts.extract_series")
        arr = np.empty(n_total, dtype=np.float64)
        off = 0
        for batch in df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch, "ts.extract_series", 3.5)
            vals = batch.column(target_col).to_pylist()
            n_b = len(vals)
            if off + n_b > n_total:
                # df.count() and iterate() disagree — fall back safely
                # by trimming to first n_total (defensive).
                n_b = max(0, n_total - off)
                vals = vals[:n_b]
            # Fill slice with float(v) or nan
            slice_ = arr[off:off + n_b]
            for i, v in enumerate(vals):
                slice_[i] = float(v) if v is not None else np.nan
            off += n_b
            if off >= n_total:
                break
        if off < n_total:
            # Fewer rows than count() claimed (filter mismatch inside
            # df?) — trim.
            arr = arr[:off]
        return arr

    # group-filter path (Python-list, per-series small)
    out = []
    for batch in df._iter_arrow_batches(batch_size=_BATCH):
        _check(df._memory_guard, batch, "ts.extract_series", 3.5)
        vals = batch.column(target_col).to_pylist()
        gvals = batch.column(group_col).to_pylist()
        out.extend(
            float(v) if v is not None else np.nan
            for v, g in zip(vals, gvals) if g == group_value)
    return np.asarray(out, dtype=np.float64)


def _check_df(df, target_col, time_col=None, group_col=None):
    if target_col not in df._columns:
        raise KeyError(f"target column {target_col!r} not in df")
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"time_col {time_col!r} not in df")
    if group_col is not None and group_col not in df._columns:
        raise KeyError(f"group_col {group_col!r} not in df")


def _iter_target_batches(
    df, target_col, *, time_col=None, group_col=None, group_value=None,
    batch_size=None,
):
    """v0.9.186 (DEC-070): streaming target extraction — yields per-batch
    float64 arrays.

    Companion to ``_extract_series`` (which materializes full O(n) array).
    Use when caller has an O(1)/O(model) accumulator (e.g. Naive's last_k
    deque + drift_mean). For full-array consumers (stl_decompose,
    _ses_fit's `fitted`), use ``_extract_series``.

    Parameters
    ----------
    df : CadwDataFrame
    target_col : str
    time_col : str | None
        If set, sorts df by time_col (external sort, OOC-safe).
    group_col : str | None
    group_value : Any | None
        If both set, filters df to that group. Falls back to non-streaming
        ``_extract_series`` (small per-series).
    batch_size : int | None
        Passed to ``_iter_filtered_batches``.

    Yields
    ------
    np.ndarray, shape (b,) — one batch (float64); b <= batch_size.
    """
    if target_col not in df._columns:
        raise KeyError(f"target column {target_col!r} not in df")
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"time_col {time_col!r} not in df")
    if group_col is not None and group_col not in df._columns:
        raise KeyError(f"group_col {group_col!r} not in df")

    if time_col is not None:
        keys = ([group_col] if group_col is not None else []) + [time_col]
        df = df.sort_values(keys)

    if group_col is not None and group_value is not None:
        # Group-filtered: fall back to non-streaming (small per-series).
        yield _extract_series(
            df, target_col, time_col=None,
            group_col=group_col, group_value=group_value)
        return

    mg = getattr(df, "_memory_guard", None)

    # v0.9.186 (DEC-023 fix): derive batch_size from memory_guard when None.
    # MemoryGuard returns ~1.3M for single-f64 (8 B/row) at 500 MB budget;
    # that's >> n and yields 1 batch = n (full materialization). Cap at
    # 10_000 to force streaming multi-batch (100+ batches for n=1e6).
    if batch_size is None:
        derived = None
        if mg is not None:
            try:
                derived = mg.get_batch_rows(expansion_factor=3.5)
            except (AttributeError, TypeError):
                derived = None
        if derived and derived > 0:
            batch_size = min(int(derived), 10_000)
        else:
            batch_size = 10_000

    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None:
            mg.check_batch(
                b, name="ts.iter_target_batches", extra_factor=3.5)
        arr = b.column(target_col).to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False)
        yield arr


def _collect_last_w(
    df, target_col, W, *, time_col=None, group_col=None,
    group_value=None, batch_size=None,
):
    """v0.9.189 (DEC-072): streaming buffer of last W observations.

    Ring buffer, O(W) RAM. Returns (y_used, n_total, lost_range):
    - n_total <= W: y_used = full y (bit-identical to _extract_series),
      lost_range = None.
    - n_total > W: y_used = last W observations (truncation);
      lost_range = (0, n_total - W).

    Callers: ETS/AutoETS (HW + grid search on last W),
    Theta/OTM s>1 (stl_decompose on last W).

    YAML flip to bounded_state:true + n_scalability:bounded_by_window
    uses this primitive; W_eff from guard.get_window_size() (typically
    target_fraction=0.25, unit_bytes = state_bytes + batch transient).
    """
    W = int(W)
    if W < 2:
        raise ValueError(f"_collect_last_w: W must be >= 2, got {W}")

    buf = np.zeros(W, dtype=np.float64)
    filled = 0
    head = 0
    n_total = 0

    for arr in _iter_target_batches(
            df, target_col, time_col=time_col,
            group_col=group_col, group_value=group_value,
            batch_size=batch_size):
        if arr.size == 0:
            continue
        # Fast path: if batch fully fills remaining buffer and no overflow
        for v in arr:
            buf[head] = float(v)
            head = (head + 1) % W
            if filled < W:
                filled += 1
            n_total += 1

    if filled < W:
        y_used = buf[:filled].copy()
    else:
        # Linearize ring: [head:W] + [0:head]
        y_used = np.concatenate([buf[head:], buf[:head]])

    lost = (0, n_total - filled) if n_total > filled else None
    return y_used, n_total, lost


class ForecasterBase(ApproximationMixin):
    """Abstract forecaster contract (v0.9.24).

    Subclasses MUST implement fit, predict, predict_in_sample. All
    three methods conform to the same semantics:
      fit(df, target_col, *, time_col=None, group_col=None):
        consume a CadwDataFrame, extract target (and features for GBDT),
        set self._is_fitted = True.
      predict(steps): return 1-D np.ndarray, len=steps, finite point
        forecasts at horizons 1..steps.
      predict_in_sample(): return 1-D np.ndarray, len=n_train, model
        in-sample prediction.

    Pre-conditions:
      - fit required before predict / predict_in_sample; otherwise
        RuntimeError.
      - steps must be int >= 1; otherwise ValueError.
    """

    def __init__(self):
        self._is_fitted = False

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        raise NotImplementedError

    def predict(self, steps):
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}.predict called before fit"
            )
        if not isinstance(steps, int):
            raise TypeError(
                f"{type(self).__name__}.predict: steps must be int; "
                f"got {type(steps).__name__}"
            )
        if steps < 1:
            raise ValueError(
                f"{type(self).__name__}.predict: steps must be >= 1; "
                f"got {steps}"
            )
        return self._predict(steps)

    def predict_in_sample(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}.predict_in_sample called before fit"
            )
        return self._predict_in_sample()

    def _predict(self, steps):
        raise NotImplementedError

    def _predict_in_sample(self):
        raise NotImplementedError


# ==================== NaiveForecaster ====================

class NaiveForecaster(ForecasterBase):
    """Naive baselines.

    strategies:
      'last'      - y_{T+h} = y_T
      'seasonal'  - y_{T+h} = y_{T+h - s} (s=period; s required)
      'drift'     - y_{T+h} = y_T + h * mean(diff(y))

    RAM: O(N) for storing the training series (needed for in-sample
    and drift mean); no MemoryGuard allocation beyond the extraction
    buffer, which is streamed and checked. O(1) at predict time.
    """

    def __init__(self, strategy="last", period=None):
        super().__init__()
        if strategy not in ("last", "seasonal", "drift"):
            raise ValueError(
                f"NaiveForecaster: strategy must be 'last', "
                f"'seasonal', or 'drift'; got {strategy!r}"
            )
        if strategy == "seasonal":
            if period is None or int(period) < 1:
                raise ValueError(
                    "NaiveForecaster: strategy='seasonal' requires "
                    "period >= 1"
                )
        self.strategy = strategy
        self.period = int(period) if period is not None else None
        # v0.9.182 OOC state
        self.last_k_ = None
        self.drift_mean_ = None
        self._saved_y_ = None
        self.target_col_ = None
        self.time_col_ = None

    def fit(self, df, target_col, *, time_col=None, group_col=None,
            save_y=False):
        """Fit Naive baseline.

        v0.9.186 (DEC-070): streaming fit — O(batch + period) RAM.
        - No _extract_series full-array materialization in fit.
        - Tail buffer keeps last `s` values (period for seasonal; 1 otherwise).
        - Drift mean from streaming diff accumulation (NaN-safe).
        - save_y=True opt-in: accumulates full y for legacy no-df
          predict_in_sample() (adds O(n) state).
        """
        _check_df(df, target_col, time_col, group_col)
        self.target_col_ = target_col
        self.time_col_ = time_col

        s = self.period if self.strategy == "seasonal" else 1

        tail_buf: list = []  # last up to s values (Python list; s small)
        n_total = 0
        diff_sum = 0.0
        diff_count = 0
        prev_val = None
        saved_parts = [] if save_y else None

        for arr in _iter_target_batches(
                df, target_col, time_col=time_col, batch_size=None):
            if arr.size == 0:
                continue
            # Cross-batch diff (NaN-safe)
            if prev_val is not None and not (
                    np.isnan(prev_val) or np.isnan(arr[0])):
                diff_sum += float(arr[0]) - prev_val
                diff_count += 1
            # Within-batch diff (NaN-safe; uses sum(diff)=last-first)
            if arr.size > 1:
                diffs = np.diff(arr)
                mask = ~np.isnan(diffs)
                diff_sum += float(np.sum(diffs[mask]))
                diff_count += int(mask.sum())
            prev_val = float(arr[-1])
            # Tail buffer (last up to s values) — avoid tolist() on full
            # batch when batch >= s (10k Python floats = 640 KB transient).
            if arr.size >= s:
                tail_buf = arr[-s:].tolist()  # only last s (~64 B for s=1)
            else:
                tail_buf.extend(arr.tolist())
                if len(tail_buf) > s:
                    tail_buf = tail_buf[-s:]
            n_total += arr.size
            if save_y:
                saved_parts.append(arr)

        if n_total < 2:
            raise ValueError(
                "NaiveForecaster: need at least 2 observations to fit"
            )

        last_k_arr = np.array(tail_buf, dtype=np.float64)

        if self.strategy == "last":
            self.last_k_ = last_k_arr[-1:].copy()
            self.drift_mean_ = None
        elif self.strategy == "seasonal":
            if n_total < s:
                raise ValueError(
                    f"NaiveForecaster seasonal: need len(y) >= period; "
                    f"len(y)={n_total}, s={s}"
                )
            self.last_k_ = last_k_arr.copy()
            self.drift_mean_ = None
        else:  # drift
            self.last_k_ = last_k_arr[-1:].copy()
            self.drift_mean_ = (diff_sum / diff_count
                                if diff_count > 0 else 0.0)

        self._saved_y_ = (
            np.concatenate(saved_parts)
            if save_y and saved_parts else None
        )
        self._is_fitted = True
        return self

    def state_nbytes(self):
        """v0.9.182 OOC: O(period) + optional legacy _saved_y_."""
        if not self._is_fitted:
            return 0
        k = self.last_k_.nbytes if self.last_k_ is not None else 0
        ys = self._saved_y_.nbytes if self._saved_y_ is not None else 0
        return int(k + ys + 256)

    def _predict(self, steps):
        """v0.9.182: uses last_k_ (O(period)), not full y_."""
        if self.strategy == "last":
            return np.full(steps, self.last_k_[0], dtype=np.float64)
        if self.strategy == "seasonal":
            s = self.period
            # last_k_ = y[-s:] = [y[T-s], ..., y[T-1]]
            # out[h] = y[T-s + (h mod s)]
            out = np.empty(steps, dtype=np.float64)
            for h in range(steps):
                out[h] = self.last_k_[h % s]
            return out
        # drift
        return (self.last_k_[0]
                + self.drift_mean_ * np.arange(1, steps + 1,
                                                dtype=np.float64))

    def predict_in_sample(self, df=None):
        """v0.9.182: OOC default requires df (or save_y=True legacy)."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}.predict_in_sample called before fit"
            )
        if df is not None:
            y = _extract_series(df, self.target_col_,
                                 time_col=self.time_col_)
            return self._predict_in_sample_from_y(y)
        if self._saved_y_ is not None:
            return self._predict_in_sample_from_y(self._saved_y_)
        raise TypeError(
            f"{type(self).__name__}.predict_in_sample: df is required "
            f"(OOC default since v0.9.182). Use fit(..., save_y=True) for "
            f"legacy no-df calls (adds O(n) persistent state)."
        )

    def _predict_in_sample_from_y(self, y):
        """Recompute in-sample on arbitrary y (bit-identical pre-v0.9.182)."""
        T = len(y)
        out = np.full(T, np.nan, dtype=np.float64)
        if self.strategy == "last":
            out[1:] = y[:-1]
        elif self.strategy == "seasonal":
            s = self.period
            if T > s:
                out[s:] = y[:-s]
        else:
            diffs = np.diff(y)
            diffs = diffs[~np.isnan(diffs)]
            drift = float(diffs.mean()) if len(diffs) else 0.0
            out[1:] = y[:-1] + drift
        return out



# ==================== SARIMA (CSS / Hannan-Rissanen) ====================

def _difference(y, d):
    """Apply d-th order regular differencing. Non-finite leading rows
    of order k are NaN."""
    out = y.copy()
    for _ in range(d):
        new = np.full_like(out, np.nan)
        new[1:] = out[1:] - out[:-1]
        out = new
    return out


def _seasonal_difference(y, D, s):
    out = y.copy()
    for _ in range(D):
        new = np.full_like(out, np.nan)
        new[s:] = out[s:] - out[:-s]
        out = new
    return out


def _integrate_back(y_forecast, y_levels, d, D, s, seasonal_period_last):
    """Undo differencing on forecasts.

    ORDER: reverse of _prep (which does d regular THEN D seasonal).
    So integration order is D seasonal FIRST, then d regular.
    v0.9.25a fix: prior code did d→D — structurally wrong for D>0
    (produced forecasts that were monotone-inflated by the seasonal
    level carried through regular cumsum). Not caught by tests because
    every SARIMA test in v0.9.24 had D=0.

    y_levels[lvl] = last observed value at lvl-th regular-diff level:
      lvl=0 -> y_raw[-1]            (original scale, centered)
      lvl=1 -> diff(y_raw)[-1]      (1st regular diff)
      lvl=2 -> diff(diff(y_raw))[-1] etc.
    seasonal_period_last = last s values of _difference(y_raw, d) —
      i.e., the series BEFORE seasonal diff, AFTER regular diff.
    v0.9.25a fix: prior caller passed y_raw[-s:] — wrong for d>0
    (y_raw has not been regular-differenced yet; _seasonal_difference
    is applied to diff(y_raw), so the head s values must come from
    diff(y_raw), not y_raw).
    """
    f = np.asarray(y_forecast, dtype=np.float64)
    # Undo seasonal differencing FIRST (reverse of _prep order).
    for _ in range(D):
        out = np.empty_like(f)
        for h in range(len(f)):
            if h < s:
                out[h] = f[h] + seasonal_period_last[h]
            else:
                out[h] = f[h] + out[h - s]
        f = out
    # Then undo regular differencing (d levels, outer→inner).
    for lvl in range(d):
        base = y_levels[lvl] if lvl < len(y_levels) else 0.0
        if not np.isfinite(base):
            base = 0.0
        f = np.cumsum(f) + base
    return f


def _ar_fit(y, p):
    """Fit AR(p) to y via OLS on available rows. Returns (coef, resid)."""
    n = len(y)
    if n <= p:
        raise ValueError(f"AR fit: n={n} <= p={p}")
    Y = y[p:]
    X = np.empty((n - p, p), dtype=np.float64)
    for i in range(p):
        X[:, i] = y[p - i - 1:n - i - 1]
    # drop rows with NaN
    mask = ~(np.isnan(Y) | np.isnan(X).any(axis=1))
    Y = Y[mask]
    X = X[mask]
    if len(Y) < p + 1:
        raise ValueError(f"AR fit: insufficient rows after NaN drop ({len(Y)})")
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = np.full(n, np.nan, dtype=np.float64)
    resid[mask.nonzero()[0] + p] = Y - X @ beta
    return beta, resid


def _hannan_rissanen(y, p, q, ar_m=None):
    """Hannan-Rissanen two-step ARMA(p,q) fit.

    Step 1: fit a high-order AR(m) to get residual estimates.
    Step 2: regress y_t on y_{t-1..p} and e_{t-1..q} via OLS.
    Returns (ar_coefs, ma_coefs, sigma2).
    """
    if ar_m is None:
        ar_m = max(p, q) + p
        if ar_m < 1:
            ar_m = 1
    # Step 1
    _, e = _ar_fit(y, ar_m)
    # Step 2: OLS on (y lags, e lags)
    n = len(y)
    start = max(p, q) + ar_m  # ensure e available for all q lags
    Y = y[start:]
    X = np.empty((n - start, p + q), dtype=np.float64)
    for i in range(p):
        X[:, i] = y[start - i - 1:n - i - 1]
    for i in range(q):
        X[:, p + i] = e[start - i - 1:n - i - 1]
    mask = ~(np.isnan(Y) | np.isnan(X).any(axis=1))
    Y = Y[mask]
    X = X[mask]
    if len(Y) < p + q + 2:
        raise ValueError(
            f"Hannan-Rissanen: insufficient rows ({len(Y)}); "
            f"reduce orders or increase N"
        )
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    ar = beta[:p] if p else np.empty(0)
    ma = beta[p:] if q else np.empty(0)
    resid = Y - X @ beta
    sigma2 = float(np.mean(resid ** 2))
    return ar, ma, sigma2


def _arma_forecast(y, ar, ma, sigma2, steps, in_sample=False):
    """Recursive ARMA forecast using the fitted coefficients."""
    p = len(ar)
    q = len(ma)
    n = len(y)
    if in_sample:
        # one-step-ahead predictions in-sample (best-effort OLS fit)
        out = np.full(n, np.nan, dtype=np.float64)
        # Step 1 residuals from a high-order AR fit as approximate e
        m = max(p, q) + p if (p or q) else 1
        if n > m:
            _, e = _ar_fit(y, m)
        else:
            e = np.full(n, 0.0)
        for t in range(m, n):
            s = 0.0
            for i in range(p):
                if t - i - 1 >= 0 and not np.isnan(y[t - i - 1]):
                    s += ar[i] * y[t - i - 1]
            for i in range(q):
                if t - i - 1 >= 0 and not np.isnan(e[t - i - 1]):
                    s += ma[i] * e[t - i - 1]
            out[t] = s
        return out
    # Out-of-sample recursive forecast
    y_ext = list(y)
    e_ext = [0.0] * n
    # Initialize e_ext via in-sample residuals
    m = max(p, q) + p if (p or q) else 1
    if n > m:
        _, e = _ar_fit(y, m)
        for i, v in enumerate(e):
            if not np.isnan(v):
                e_ext[i] = v
    fc = np.empty(steps, dtype=np.float64)
    for h in range(steps):
        s = 0.0
        for i in range(p):
            s += ar[i] * y_ext[-i - 1]
        for i in range(q):
            s += ma[i] * e_ext[-i - 1]
        fc[h] = s
        y_ext.append(s)
        e_ext.append(0.0)  # MA residual for future = 0 in expectation
    return fc


class SARIMAForecaster(ForecasterBase):
    """SARIMA(p,d,q)(P,D,Q,s) via CSS Hannan-Rissanen.

    - order=(p,d,q); seasonal_order=(P,D,Q) or None; s=int or None.
    - auto=True and (order is None or seasonal_order is None):
        grid: p,q in 0..2, d in 0..1;
        seasonal (if s not None): P,Q,D in 0..1.
        Select by AIC = n_obs * ln(sigma2) + 2*n_params.
    - auto=False: requires order (and seasonal_order if s set).

    WARNING: AIC can under-difference trending series. On a random
    walk the level-stationary ARMA fit often has lower AIC than the
    integrated (d=1) fit, so auto=True may pick d=0. For clearly
    trending or unit-root series, set order=(p,1,q) explicitly. This
    is a documented limitation of AIC-based selection, not a bug in
    the CSS fit. If a level-stationary fit is chosen on trending
    data, forecasts will mean-revert toward the fitted intercept.
    """

    handles_trend: bool = True
    handles_seasonality: bool = True
    handles_multi_seasonality: bool = False

    DEFAULT_P_GRID = (0, 1, 2)
    DEFAULT_D_GRID = (0, 1)
    DEFAULT_Q_GRID = (0, 1, 2)
    DEFAULT_SP_GRID = (0, 1)
    DEFAULT_SD_GRID = (0, 1)
    DEFAULT_SQ_GRID = (0, 1)

    def __init__(
        self, order=None, seasonal_order=None, s=None, auto=True,
    ):
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.s = s
        self.auto = bool(auto)
        if order is not None and (not isinstance(order, tuple) or len(order) != 3):
            raise ValueError("order must be a 3-tuple (p,d,q)")
        if seasonal_order is not None and (
            not isinstance(seasonal_order, tuple) or len(seasonal_order) != 3
        ):
            raise ValueError(
                "seasonal_order must be a 3-tuple (P,D,Q)"
            )
        if seasonal_order is not None and s is None:
            raise ValueError(
                "SARIMA: seasonal_order requires s (seasonal period)"
            )
        if s is not None and (not isinstance(s, int) or s < 2):
            raise ValueError(f"SARIMA: s must be int >= 2; got {s}")
        self._fit_info = {}

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        if len(y) < 4:
            raise ValueError("SARIMA: need at least 4 observations")
        self.y_raw_ = y.copy()
        # v0.9.25a: center to remove the mean. Hannan-Rissanen OLS has
        # no intercept; fitting on raw y with mean≠0 makes ARMA predict
        # toward 0 (e.g. ARIMA(0,0,0) on series with mean=100 predicted
        # all zeros). Centering is exact: add _y_mean back after
        # integration. For d>0 the integration level already captures
        # the mean; the +_y_mean at the end is still correct (same
        # offset in centered space).
        y_clean = y[~np.isnan(y)]
        self._y_mean = float(y_clean.mean()) if len(y_clean) else 0.0
        self._y_centered_ = y - self._y_mean
        self._fit(self._y_centered_)
        self._is_fitted = True
        return self

    def _prep(self, y, d, D, s):
        yd = _difference(y, d) if d else y.copy()
        if D and s is not None:
            yd = _seasonal_difference(yd, D, s)
        # drop NaNs from head; keep length bookkeeping
        return yd

    def _fit_one(self, y, p, d, q, P, D, Q, s):
        yd = self._prep(y, d, D, s)
        # Fit regular ARMA on yd.
        try:
            ar, ma, sigma2 = _hannan_rissanen(yd, p + P * (s if s else 0), q + Q * (s if s else 0))
        except Exception as e:
            return None, None, None, f"{type(e).__name__}: {e}"
        # v0.9.25a-fix1: allow sigma2 == 0 (perfect fit is valid —
        # e.g. exactly linear series with d=1,D=1 gives all-zero
        # twice-differenced and sigma2=0). Reject only negative or
        # non-finite. AIC uses log(max(sigma2, 1e-300)) so it stays
        # finite for sigma2=0 (log(0)=-inf produced non-comparable
        # -inf across candidates, even though -inf is a valid "best"
        # signal).
        if sigma2 < 0 or not np.isfinite(sigma2):
            return None, None, None, f"sigma2={sigma2!r}"
        n_valid = int(np.sum(~np.isnan(yd)))
        n_params = p + q + P + Q + 1
        sigma2_safe = max(sigma2, 1e-300)
        aic = n_valid * np.log(sigma2_safe) + 2 * n_params
        return (ar, ma, sigma2, yd), aic, sigma2, None

    def _fit(self, y):
        s = self.s
        if not self.auto or self.order is not None:
            p, d, q = self.order or (0, 0, 0)
            P, D, Q = self.seasonal_order or (0, 0, 0)
            res, aic, sigma2, err = self._fit_one(y, p, d, q, P, D, Q, s)
            if res is None:
                raise ValueError(
                    f"SARIMA fit failed for (p,d,q)(P,D,Q)=({p},{d},{q})"
                    f"({P},{D},{Q}): {err}"
                )
            ar, ma, sigma2_, yd = res
            self._ar = ar
            self._ma = ma
            self._sigma2 = sigma2_
            self._order_ = (p, d, q)
            self._seasonal_order_ = (P, D, Q)
            self._yd = yd
            self._fit_info = {
                "aic": aic,
                "sigma2": sigma2_,
                "order": (p, d, q),
                "seasonal_order": (P, D, Q),
            }
            return
        # auto: grid search
        best = None
        best_aic = np.inf
        best_info = None
        seasonal_grid = (
            [(0, 0, 0)] if s is None
            else [(P, D, Q)
                  for P in self.DEFAULT_SP_GRID
                  for D in self.DEFAULT_SD_GRID
                  for Q in self.DEFAULT_SQ_GRID]
        )
        for p in self.DEFAULT_P_GRID:
            for d in self.DEFAULT_D_GRID:
                for q in self.DEFAULT_Q_GRID:
                    for P, D, Q in seasonal_grid:
                        res, aic, sigma2, err = self._fit_one(
                            y, p, d, q, P, D, Q, s,
                        )
                        if res is None:
                            continue
                        if aic < best_aic:
                            best_aic = aic
                            best = res
                            best_info = {
                                "aic": aic,
                                "sigma2": sigma2,
                                "order": (p, d, q),
                                "seasonal_order": (P, D, Q),
                            }
        if best is None:
            raise ValueError(
                "SARIMA auto: all candidate fits failed; try reducing grid"
            )
        self._ar, self._ma, self._sigma2, self._yd = best
        self._order_ = best_info["order"]
        self._seasonal_order_ = best_info["seasonal_order"]
        self._fit_info = best_info

    def state_nbytes(self):
        """SARIMA persistent state: y_raw_ + _y_centered_ (2n*f64) + coefs+256.

        y_raw_ and _y_centered_ are O(n); coefs (ar_, ma_, intercept_,
        sigma2_) are O(p+q+~10) -> covered by +256.
        """
        if self.y_raw_ is None:
            return 0
        n = int(self.y_raw_.size)
        return int(2 * n * 8 + 256)

    def _predict(self, steps):
        yd = self._yd
        valid = ~np.isnan(yd)
        if not valid.any():
            raise RuntimeError("SARIMA: no valid observations after differencing")
        y_clean = yd[valid]
        fc_diff = _arma_forecast(y_clean, self._ar, self._ma, self._sigma2, steps)
        p, d, q = self._order_
        P, D, Q = self._seasonal_order_
        s = self.s
        if d or D:
            # v0.9.25a: build y_levels from centered series at each
            # regular-diff level. For d=2+, prior code used 0.0 for
            # levels >0 — structurally wrong (should be last diff value).
            levels = []
            yd_for_levels = self._y_centered_
            for lvl in range(d):
                v = float(yd_for_levels[-1]) if len(yd_for_levels) else 0.0
                levels.append(v)
                yd_for_levels = _difference(yd_for_levels, 1)
            # v0.9.25a: seasonal_period_last from _difference(y_centered, d)
            # — the series BEFORE seasonal diff, AFTER regular diff.
            # Prior code took self.y_raw_[-s:] — wrong for d>0.
            if D and s:
                yd_for_seas = (
                    _difference(self._y_centered_, d) if d
                    else self._y_centered_.copy()
                )
                seas_last = yd_for_seas[-s:] if len(yd_for_seas) >= s                     else np.zeros(s, dtype=np.float64)
            else:
                seas_last = np.empty(0, dtype=np.float64)
            fc = _integrate_back(
                fc_diff, tuple(levels), d, D, s, seas_last,
            )
        else:
            fc = fc_diff
        # v0.9.25a: restore the mean (centered in fit).
        return (fc + self._y_mean).astype(np.float64)

    def _predict_in_sample(self):
        """In-sample prediction (v0.9.25a contract fix).

        ForecasterBase.predict_in_sample contract: len=n_train, values
        in original y scale.

        - d=0, D=0: returns one-step-ahead predictions in y scale.
        - d>0 or D>0: returns all-NaN. In-sample prediction at original
          y scale for a differenced model is not well-defined (each
          point would require its own integration base; the diff-head
          is not invertible in-sample without leakage). Documented
          limitation; TemporalHierarchy._fit_one already falls back to
          OLS when predict_in_sample returns all-NaN.
        """
        p, d, q = self._order_
        P, D, Q = self._seasonal_order_
        n = len(self.y_raw_)
        if d or D:
            return np.full(n, np.nan)
        yd = self._yd
        valid = ~np.isnan(yd)
        if not valid.any():
            return np.full(n, np.nan)
        y_clean = yd[valid]
        fc_diff = _arma_forecast(
            y_clean, self._ar, self._ma, self._sigma2, 0, in_sample=True,
        )
        if len(fc_diff) != n:
            return np.full(n, np.nan)
        return (fc_diff + self._y_mean).astype(np.float64)


# ==================== GBDTForecaster ====================

def _build_lag_matrix(y, lags, seasonal_periods=(), *, memory_guard=None):
    """Return (X, valid_idx) where X is (n - max_lag, n_features).

    Features: y_lag1..y_lagp, plus y_lag(s*k) for seasonal s.
    All features are NaNs for t < max_lag; those rows are dropped.
    Trend row index (t) is added as 'trend' column.

    v0.9.90-S5: optional memory_guard. Guard before np.column_stack:
    peak = (n - max_lag) * (n_lags + 1) * 8.
    """
    n = len(y)
    all_lags = list(lags)
    for s in seasonal_periods:
        for k in lags:
            all_lags.append(s * k)
    all_lags = sorted(set(int(k) for k in all_lags))
    max_lag = max(all_lags)
    if n <= max_lag + 1:
        raise ValueError(
            f"GBDTForecaster: need n > max_lag + 1; n={n}, "
            f"max_lag={max_lag}"
        )
    if memory_guard is not None:
        n_rows = n - max_lag
        est = n_rows * (len(all_lags) + 1) * 8
        memory_guard.check_state(est, name="gbdt_forecaster.lag_matrix")
    idx = np.arange(max_lag, n)
    cols = []
    for k in all_lags:
        cols.append(y[idx - k])
    cols.append(idx.astype(np.float64))  # trend
    X = np.column_stack(cols)
    return X, idx, all_lags


class _NumpyReg:
    """Minimal wrapper: fit GBDT on numpy X, y using cadw.ml.fit_gbdt_regressor.

    Builds a temporary CadwDataFrame internally. This is where every
    allocation is routed through MemoryGuard.
    """

    def __init__(self, memory_guard=None):
        self.guard = memory_guard
        self.model = None
        self.feature_names = None

    def fit(self, X, y, feature_names, *, n_estimators=50, max_depth=3, learning_rate=0.1):
        from cadw import CadwDataFrame
        import pyarrow as pa
        cols = {"__target__": pa.array(y, type=pa.float64())}
        for i, name in enumerate(feature_names):
            cols[name] = pa.array(X[:, i], type=pa.float64())
        df = CadwDataFrame.from_table(pa.table(cols))
        from cadw.ml.gbdt import fit_gbdt_regressor
        model = fit_gbdt_regressor(
            df, feature_names, "__target__",
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, memory_guard=self.guard,
        )
        self.model = model
        self.feature_names = list(feature_names)
        return self

    def state_nbytes(self) -> int:
        """Delegate to wrapped GBDTRegressor (self.model)."""
        if self.model is None:
            return 0
        if hasattr(self.model, "state_nbytes"):
            return int(self.model.state_nbytes())
        return 0

    def predict(self, X):
        # GBDTRegressor.predict expects X numpy (n, d).
        return self.model.predict(X)


class GBDTForecaster(ForecasterBase):
    """Gradient-boosted trees forecaster (three modes).

    Parameters:
      mode='direct' (default): h independent models, one per horizon.
        No error propagation; features come from training data only.
        In direct mode horizons are DETERMINED BY predict(steps):
          - fit(df, ..., horizons=None) defers all training until the
            first predict(steps) call, at which point exactly `steps`
            horizon models are trained (skforecast/Nixtla-style).
          - fit(df, ..., horizons=K) trains K horizon models eagerly;
            predict(steps) with steps > K raises ValueError.
        No silent extrapolation. A caller that asks for 20 steps gets
        20 distinct horizon models (or a clear error).
      mode='recursive': one model predicting 1 step ahead; predictions
        are fed back as features. Propagates error; use for parity.
      mode='multi_output': h independent models called jointly; the
        label 'multi_output (h-independent)' is explicit because CADW
        GBDT has no native multi-target. Same deferred training as
        'direct'.
      lags=(1, 2, 3): base lags.
      seasonal_periods=(7,) or (): seasonal lags = s * lags.
      n_estimators=50, max_depth=3, learning_rate=0.1: GBDT hyperparams.

    RAM: O(N x max_lag) numpy feature matrix for the training data +
    per-batch MemoryGuard checks in the GBDT Reduce passes.
    """

    def __init__(
        self, mode="direct", lags=(1, 2, 3), seasonal_periods=(),
        n_estimators=50, max_depth=3, learning_rate=0.1,
        differencing="none", decompose="none",
        decompose_periods=None,
    ):
        super().__init__()
        if mode not in ("direct", "recursive", "multi_output"):
            raise ValueError(
                f"GBDTForecaster: mode must be 'direct', 'recursive', "
                f"or 'multi_output'; got {mode!r}"
            )
        # v0.9.83: capability-based transforms (GBDT can't extrapolate
        # trend; external differencing/STL via AutoForecastPipeline
        # machinery).
        if differencing not in ("none", "auto", None):
            if not (isinstance(differencing, int) and differencing > 0):
                raise ValueError(
                    f"GBDTForecaster: differencing={differencing!r} "
                    f"must be 'none', 'auto', None, or positive int"
                )
        if decompose not in ("none", "stl", "mstl", None):
            raise ValueError(
                f"GBDTForecaster: decompose={decompose!r} "
                f"must be 'none', 'stl', 'mstl', or None"
            )
        if decompose in ("stl", "mstl") and not decompose_periods:
            raise ValueError(
                f"GBDTForecaster: decompose={decompose!r} requires "
                f"decompose_periods non-empty"
            )
        if decompose == "mstl" and len(decompose_periods) < 2:
            raise ValueError(
                f"GBDTForecaster: decompose='mstl' requires "
                f">= 2 decompose_periods; got {len(decompose_periods)}"
            )
        self.mode = mode
        self.lags = tuple(int(k) for k in lags)
        self.seasonal_periods = tuple(int(s) for s in seasonal_periods)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self._models = None
        self._feature_names = None
        self._tail_features = None
        self._y = None
        self._X = None
        self._idx = None
        self._all_lags = None
        self._n_horizons = 0
        self._is_fitted = False
        # v0.9.83: capability-based transforms
        self.differencing = differencing
        self.decompose = decompose
        self.decompose_periods = (
            list(decompose_periods) if decompose_periods else None)
        self._inner_pipeline = None
        # Instance-level capability flags (override class defaults — GBDT
        # has no class-level flags, so getattr(...) would return False).
        # When transforms are configured, GBDT reports handles_* = True so
        # an outer AutoForecastPipeline will not double-wrap.
        has_trend = (
            differencing == "auto"
            or (isinstance(differencing, int) and differencing > 0))
        has_season = decompose in ("stl", "mstl")
        self.handles_trend = has_trend
        self.handles_seasonality = has_season
        self.handles_multi_seasonality = (decompose == "mstl")

    def _build_inner_pipeline(self):
        """Construct TransformedTargetForecaster wrapping a copy of self.

        v0.9.83: lazy import of transforms to avoid circular dependency
        (transforms.py imports ForecasterBase from models.py).
        Returns None if no transforms configured.
        """
        from cadw.ts.transforms import (
            AutoStationarizer, Differencer, MSTLTransformer,
            STLTransformer, TransformedTargetForecaster,
        )
        import copy
        transformers = []
        if self.decompose == "stl":
            transformers.append(
                STLTransformer(int(self.decompose_periods[0])))
        elif self.decompose == "mstl":
            transformers.append(
                MSTLTransformer(list(self.decompose_periods)))
        if self.differencing == "auto":
            transformers.append(
                AutoStationarizer(seasonal_period=None))
        elif isinstance(self.differencing, int) and self.differencing > 0:
            transformers.append(Differencer(lags=self.differencing))
        if not transformers:
            return None
        inner = copy.copy(self)
        inner.differencing = "none"
        inner.decompose = "none"
        inner.decompose_periods = None
        inner._inner_pipeline = None
        inner._models = None
        inner._y = None
        inner._X = None
        inner._idx = None
        inner._tail_features = None
        inner._is_fitted = False
        return TransformedTargetForecaster(transformers, inner)

    def fit(
        self, df, target_col, *, time_col=None, group_col=None,
        horizons=None,
    ):
        # v0.9.83: if transforms configured, delegate to inner pipeline
        if (self.differencing not in ("none", None)
                or self.decompose not in ("none", None)):
            if self._inner_pipeline is None:
                self._inner_pipeline = self._build_inner_pipeline()
            if self._inner_pipeline is not None:
                self._inner_pipeline.fit(
                    df, target_col, time_col=time_col,
                    group_col=group_col)
                self._is_fitted = True
                return self
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        if len(y) < 5:
            raise ValueError("GBDTForecaster: need at least 5 observations")
        self._y = y
        self._memory_guard = df._memory_guard
        X, idx, all_lags = _build_lag_matrix(
            y, self.lags, self.seasonal_periods,
            memory_guard=df._memory_guard,
        )
        self._feature_names = [f"y_lag{k}" for k in all_lags] + ["trend"]
        self._all_lags = all_lags
        self._X = X
        self._idx = idx
        self._tail_features = X[-1].copy()
        self._models = []
        self._n_horizons = 0
        self._is_fitted = True
        # Eager training if horizons given (recursive always needs 1).
        if self.mode == "recursive":
            self._train_n_horizons(1)
        elif horizons is not None:
            if not isinstance(horizons, int) or horizons < 1:
                raise ValueError(
                    f"GBDTForecaster.fit: horizons must be int >= 1; "
                    f"got {horizons!r}"
                )
            self._train_n_horizons(int(horizons))
        return self

    def _train_n_horizons(self, n_hor):
        """Train (or re-use) exactly n_hor horizon models.

        For direct/multi_output: one model per horizon h=1..n_hor.
        For recursive: only horizon 1 (one-step model).
        """
        if self._n_horizons >= n_hor:
            return  # already have enough
        y = self._y
        X = self._X
        idx = self._idx
        for h in range(self._n_horizons + 1, n_hor + 1):
            yh = np.full(len(idx), np.nan)
            for j in range(len(idx)):
                t = idx[j]
                if t + h <= len(y) - 1:
                    yh[j] = y[t + h]
            mask = ~np.isnan(yh) & ~np.isnan(X).any(axis=1)
            if mask.sum() < 3:
                raise ValueError(
                    f"GBDTForecaster: not enough rows for horizon {h} "
                    f"(have {int(mask.sum())}, need 3). Reduce lags or "
                    f"provide a longer training series."
                )
            reg = _NumpyReg(memory_guard=self._memory_guard)
            reg.fit(
                X[mask], yh[mask], self._feature_names,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
            )
            self._models.append(reg)
        self._n_horizons = n_hor

    def _build_features_for_step(self, y_hist):
        arr = np.empty(len(self._all_lags) + 1, dtype=np.float64)
        for i, k in enumerate(self._all_lags):
            arr[i] = y_hist[-k]
        arr[-1] = float(len(y_hist))
        return arr

    def predict(self, steps):
        # v0.9.83: in transform mode, delegate to inner pipeline (which
        # holds fit state on the inner GBDT copy); naked mode calls
        # self._predict (existing code path).
        # v0.9.84-fix1: explicit _is_fitted and steps > 0 validation
        # (original GBDTForecaster.predict did NOT validate; tests
        # test_predict_before_fit_raises and test_steps_validation
        # exercise this via the base-class contract).
        if self._inner_pipeline is not None:
            return self._inner_pipeline.predict(steps)
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                f"{type(self).__name__}.predict called before fit")
        if not isinstance(steps, int):
            raise TypeError(
                f"GBDTForecaster.predict: steps must be int, "
                f"got {type(steps).__name__}")
        if steps < 1:
            raise ValueError(
                f"GBDTForecaster.predict: steps must be >= 1, "
                f"got {steps}")
        return self._predict(steps)

    def predict_in_sample(self):
        # v0.9.83: pipeline predict_in_sample has a length mismatch
        # (GBDT returns NaN-padded n-length; transforms were fit on
        # n - d). Not supported in transform mode; naked mode works
        # normally.
        # v0.9.84-fix2: explicit _is_fitted check (test_predict_before_
        # fit_raises expects RuntimeError when called before fit).
        if self._inner_pipeline is not None:
            raise NotImplementedError(
                "GBDTForecaster.predict_in_sample with transforms "
                "(differencing/decompose): not supported due to length "
                "mismatch. Use naked GBDT for in-sample, or call "
                "pipeline.forecaster.predict_in_sample() directly to "
                "inspect the inner model's in-sample residual on the "
                "transformed series.")
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                f"{type(self).__name__}.predict_in_sample called "
                f"before fit")
        return self._predict_in_sample()

    def state_nbytes(self):
        """Persistent state: per-horizon GBDT models + lag matrices.

        Delegates to _inner_pipeline (differencing/decompose) when present.
        Otherwise counts:
          - sum of per-horizon GBDTRegressor.state_nbytes()
          - _y        (n,) f64     -- needed for recursive mode + deferred horizons
          - _X        (n, d) f64   -- lag matrix; needed by _train_n_horizons
          - _idx      (n,) int64   -- origin indices; needed by _train_n_horizons
          - _tail_features (d,) f64
          - +256 header
        S1 audit (2026-11-09): previous version under-reported by ~3x
        (n=1000) to ~2500x (n=1e6, d=4) — treated as bounded_state bug.
        """
        if self._inner_pipeline is not None:
            return int(self._inner_pipeline.state_nbytes())
        total = 256
        # n-sized persistent fields
        for attr in ("_y", "_X", "_idx", "_tail_features"):
            v = getattr(self, attr, None)
            if v is not None and hasattr(v, "nbytes"):
                total += int(v.nbytes)
        # per-horizon fitted models
        models = getattr(self, "_models", None) or []
        for m in models:
            if hasattr(m, "state_nbytes"):
                total += int(m.state_nbytes())
        return int(total)

    def _predict(self, steps):
        if self.mode == "recursive":
            if self._n_horizons < 1:
                self._train_n_horizons(1)
            y_hist = list(self._y)
            out = np.empty(steps, dtype=np.float64)
            model = self._models[0]
            for h in range(steps):
                x = self._build_features_for_step(y_hist).reshape(1, -1)
                p = float(model.predict(x)[0])
                out[h] = p
                y_hist.append(p)
            return out
        # direct / multi_output: guarantee exactly `steps` horizon models.
        self._train_n_horizons(int(steps))
        out = np.empty(steps, dtype=np.float64)
        tail = self._tail_features.reshape(1, -1)
        for h in range(steps):
            out[h] = float(self._models[h].predict(tail)[0])
        return out

    def _predict_in_sample(self):
        y = self._y
        n = len(y)
        max_lag = max(self._all_lags)
        out = np.full(n, np.nan, dtype=np.float64)
        if n <= max_lag:
            return out
        if self._n_horizons < 1:
            self._train_n_horizons(1)
        X, idx, _ = _build_lag_matrix(y, self.lags, self.seasonal_periods)
        preds = self._models[0].predict(X)
        out[idx] = preds
        return out




# ==================== forecaster registry ====================

def get_forecaster(name, **kwargs):
    """Return a fresh forecaster by short name ('naive', 'sarima', 'gbdt').

    Registry used by CadwDataFrame.ts.fit_forecaster and the accessor.
    """
    n = str(name).lower()
    if n == 'naive':
        return NaiveForecaster(**kwargs)
    if n == 'sarima':
        return SARIMAForecaster(**kwargs)
    if n == 'gbdt':
        return GBDTForecaster(**kwargs)
    raise ValueError(
        f"get_forecaster: unknown name {name!r}; "
        f"valid: 'naive', 'sarima', 'gbdt'"
    )


__all__ = [
    "ForecasterBase",
    "NaiveForecaster",
    "SARIMAForecaster",
    "GBDTForecaster",
    "get_forecaster",
]

