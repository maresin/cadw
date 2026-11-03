"""CUSUM: cumulative sum control chart for changepoint detection.

Two-sided CUSUM for detecting mean shifts:
  target = mean(x)
  s_pos[t] = max(0, s_pos[t-1] + x[t] - target - k)
  s_neg[t] = max(0, s_neg[t-1] - (x[t] - target) - k)
  score[t] = max(s_pos[t], s_neg[t])
  anomaly if score > h

k (slack) typically 0.5 * shift_size, h (decision) typically 3-5 * std.
Reference: Kats (facebookresearch/Kats) cusum_detection (Gaussian CUSUM
with iterative mean estimation).

Pure numpy — no sklearn/scipy/torch/kats in cadw/ (AST-guard).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pyarrow as pa

from cadw.core.spill import AutoArrayResult, SpillArray


@dataclass
class CUSUMResult:
    """CUSUM fit result (v0.9.113 dual).

    decision_scores / labels: np.ndarray (legacy / output="memory") or
    AutoArrayResult (df + output="auto"|"disk", is_spilled=True).
    State during fit is O(1) (s_pos_cur, s_neg_cur + Welford n/mean/M2);
    output is O(n) -> SpillArray.
    """
    decision_scores: Union[np.ndarray, AutoArrayResult]
    labels: Union[np.ndarray, AutoArrayResult]
    threshold: float
    k: float
    h: float
    target_mean: float
    target_std: float
    contamination: Optional[float]
    is_spilled: bool = False


def _probe_memory_limit_mb(guard) -> float | None:
    def _is_pos(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    MB = ("max_memory_mb", "_max_memory_mb", "limit_mb", "budget_mb",
          "_budget_mb", "max_mb", "_max_mb", "memory_limit_mb",
          "_memory_limit_mb")
    for a in MB:
        v = getattr(guard, a, None)
        if _is_pos(v):
            return float(v)
    BYTES = ("max_memory_bytes", "_max_memory_bytes", "max_bytes",
             "_max_bytes", "limit_bytes", "_limit_bytes", "budget_bytes",
             "_budget_bytes")
    for a in BYTES:
        v = getattr(guard, a, None)
        if _is_pos(v):
            return float(v) / (1024.0 * 1024.0)
    try:
        d = guard.__dict__
    except AttributeError:
        return None
    for k, v in d.items():
        kl = k.lower()
        if not _is_pos(v):
            continue
        if "mb" in kl and any(t in kl for t in ("mem", "budget", "limit", "max")):
            return float(v)
    return None


def fit_cusum(
    df_or_series,
    value_col: Optional[str] = None,
    *,
    k: float = 0.5,
    h: float = 5.0,
    contamination: Optional[float] = None,
    guard=None,
    output: str = "auto",
    batch_size: "int | None" = None,
) -> CusumResult:
    """Two-sided CUSUM changepoint detection (v0.9.113 dual dispatch).

    df path (new, OOC-capable):
        fit_cusum(df, value_col, output="auto")   -> SpillArray output
        fit_cusum(df, value_col, output="memory") -> np.ndarray output
        fit_cusum(df, value_col, output="disk")   -> forced spill

    Legacy np.ndarray path (back-compat, unchanged):
        fit_cusum(series)   # in-memory, equivalent to output="memory"

    `contamination` requires output="memory" (percentile over full array).
    """
    is_df = (hasattr(df_or_series, "_iter_filtered_batches")
             and hasattr(df_or_series, "_columns"))
    if is_df:
        if value_col is None:
            raise ValueError(
                "fit_cusum: value_col required when df is passed")
        if value_col not in df_or_series._columns:
            raise KeyError(
                f"fit_cusum: column {value_col!r} not in {df_or_series._columns}")
        return _fit_cusum_df(
            df_or_series, value_col,
            k=k, h=h, contamination=contamination,
            guard=guard, output=output,
            batch_size=batch_size,
        )
    if value_col is not None:
        raise ValueError(
            "fit_cusum: value_col only used with df input; omit for np.ndarray")
    return _fit_cusum_series(
        df_or_series, k=k, h=h,
        contamination=contamination, guard=guard,
    )


def _fit_cusum_df(df, value_col, *, k, h, contamination, guard, output, batch_size=None):
    """df streaming CUSUM: state O(1), output O(n) -> SpillArray."""
    if output not in ("memory", "auto", "disk"):
        raise ValueError(
            f"output must be 'memory'|'auto'|'disk'; got {output!r}")
    if output != "memory" and contamination is not None:
        raise NotImplementedError(
            f"fit_cusum: contamination={contamination!r} requires "
            f"output='memory' (percentile needs full scores array); "
            f"got output={output!r}")

    mg = guard if guard is not None else getattr(df, "_memory_guard", None)

    if batch_size is None:
        try:
            schema = df._dataset.schema
            row_size = mg.estimate_row_size_from_schema(schema)
            state_bytes = 64
            batch_size = mg.get_batch_rows(
                row_size_bytes=row_size,
                expansion_factor=3.0,
                state_bytes=state_bytes,
            )
        except Exception:
            batch_size = 10_000

    # Pass 1: Welford mean/std (numerically stable, O(1) state).
    wn = 0
    wmean = 0.0
    wM2 = 0.0
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="cusum.batch_welford", extra_factor=2.0)
        x_b = b.column(value_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        nb = len(x_b)
        if nb == 0:
            continue
        batch_mean = float(x_b.mean())
        batch_M2 = float(((x_b - batch_mean) ** 2).sum())
        # Parallel Welford merge
        total_n = wn + nb
        delta = batch_mean - wmean
        wmean = wmean + delta * nb / total_n
        wM2 = wM2 + batch_M2 + delta * delta * wn * nb / total_n
        wn = total_n

    if wn < 3:
        raise ValueError(f"series too short for CUSUM (min 3), got {wn}")

    target = wmean
    sigma = math.sqrt(wM2 / wn) if wn > 0 else 0.0
    if sigma < 1e-12:
        sigma = 1e-12

    k_scaled = float(k)
    h_scaled = float(h)

    # Output buffers
    if output == "memory":
        all_scores = []
        scores_buf = None
        labels_buf = None
    else:
        spill_ratio = 0.0 if output == "disk" else 0.5
        scores_buf = SpillArray(
            guard=mg, width=1, dtype=np.float64,
            name="cusum.scores", spill_ratio=spill_ratio)
        labels_buf = SpillArray(
            guard=mg, width=1, dtype=np.float64,
            name="cusum.labels", spill_ratio=spill_ratio)

    # Pass 2: streaming recurrence.
    s_pos_cur = 0.0
    s_neg_cur = 0.0
    t_offset = 0
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="cusum.batch_recur", extra_factor=3.0)
        x_b = b.column(value_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        nb = len(x_b)
        if nb == 0:
            continue
        z_b = (x_b - target) / sigma
        scores_b = np.empty(nb, dtype=np.float64)
        for i in range(nb):
            zt = z_b[i]
            if t_offset + i == 0:
                s_pos_cur = max(0.0, zt - k_scaled)
                s_neg_cur = max(0.0, -zt - k_scaled)
            else:
                s_pos_cur = max(0.0, s_pos_cur + zt - k_scaled)
                s_neg_cur = max(0.0, s_neg_cur - zt - k_scaled)
            scores_b[i] = max(s_pos_cur, s_neg_cur)
            if s_pos_cur > h_scaled:
                s_pos_cur = 0.0
            if s_neg_cur > h_scaled:
                s_neg_cur = 0.0
        t_offset += nb

        if output == "memory":
            all_scores.append(scores_b)
        else:
            scores_buf.write_array(scores_b.reshape(-1, 1))
            labels_buf.write_array(
                (scores_b > h_scaled).astype(np.float64).reshape(-1, 1))

    if output == "memory":
        scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
        if contamination is not None:
            threshold = float(np.percentile(
                scores, 100.0 * (1.0 - contamination)))
            if threshold <= 0:
                threshold = h_scaled
        else:
            threshold = h_scaled
        labels = (scores > threshold).astype(int)
        return CUSUMResult(
            decision_scores=scores, labels=labels,
            threshold=threshold, k=k, h=h,
            target_mean=target, target_std=sigma,
            contamination=contamination, is_spilled=False,
        )
    return CUSUMResult(
        decision_scores=scores_buf.finalize(),
        labels=labels_buf.finalize(),
        threshold=float(h_scaled), k=k, h=h,
        target_mean=target, target_std=sigma,
        contamination=contamination, is_spilled=True,
    )


def _fit_cusum_series(
    series: np.ndarray,
    *,
    k: float = 0.5,
    h: float = 5.0,
    contamination: float | None = None,
    guard=None,
) -> CUSUMResult:
    """Two-sided CUSUM changepoint detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
        1D univariate time series.
    k : float, default=0.5
        Slack (allowance) parameter in std units. k=0.5 * shift_size / std
        is the standard choice.
    h : float, default=5.0
        Decision threshold in std units. h=4-5 gives ARL0 ~ 168-465
        for in-control detection.
    contamination : float | None, default=None
        If set, threshold = percentile(scores, 100*(1-c)) — overrides h.
    guard : MemoryGuard | None

    Returns
    -------
    CUSUMResult
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < 3:
        raise ValueError(f"series too short for CUSUM (min 3), got {n}")

    if guard is not None:
        # v0.9.109a: honest O(n) state estimate. Fit allocates 5-6 O(n)
        # arrays: z (8n), s_pos (8n), s_neg (8n), scores (8n),
        # labels (8n), plus temp bool for `scores > threshold` (~n).
        # Peak ~ 48n = 6 * 8n bytes. Prior 8*n*3 = 24n underestimated
        # by 2x (silent OOM at tight budgets). O(n) state means OOC
        # impossible -> YAML draft + DEC-030 (SpillArray rewrite, like
        # fit_zms).
        est_bytes = 8 * n * 6
        if hasattr(guard, "check_state") and callable(guard.check_state):
            guard.check_state(est_bytes, name="cusum.state")
        else:
            # Duck-typed guard (max_memory_mb only) — legacy fallback.
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                if est_bytes > limit_mb * 1024 * 1024:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"CUSUM estimated {est_bytes} bytes > "
                        f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    target = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-12:
        sigma = 1e-12

    # Standardise: k and h are interpreted in std units
    z = (x - target) / sigma
    k_scaled = float(k)
    h_scaled = float(h)

    s_pos = np.zeros(n, dtype=np.float64)
    s_neg = np.zeros(n, dtype=np.float64)
    s_pos_cur = 0.0
    s_neg_cur = 0.0
    for t in range(n):
        if t == 0:
            s_pos_cur = max(0.0, z[t] - k_scaled)
            s_neg_cur = max(0.0, -z[t] - k_scaled)
        else:
            s_pos_cur = max(0.0, s_pos_cur + z[t] - k_scaled)
            s_neg_cur = max(0.0, s_neg_cur - z[t] - k_scaled)
        # Record PRE-reset score (used for labels & output)
        s_pos[t] = s_pos_cur
        s_neg[t] = s_neg_cur
        # Standard CUSUM reset: alarm at score > h -> return to 0
        if s_pos_cur > h_scaled:
            s_pos_cur = 0.0
        if s_neg_cur > h_scaled:
            s_neg_cur = 0.0

    scores = np.maximum(s_pos, s_neg)

    if contamination is not None:
        threshold = float(np.percentile(
            scores, 100.0 * (1.0 - contamination)))
        if threshold <= 0:
            threshold = h_scaled
    else:
        threshold = h_scaled

    labels = (scores > threshold).astype(int)

    return CUSUMResult(
        decision_scores=scores,
        labels=labels,
        threshold=threshold,
        k=k,
        h=h,
        target_mean=target,
        target_std=sigma,
        contamination=contamination,
    )

