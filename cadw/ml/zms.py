"""ZMS: rolling z-score anomaly detector.

For each timestamp t with trailing window [t-w+1, t]:
  mu = mean(window)
  sigma = std(window)
  score[t] = |x[t] - mu| / sigma   (sigma floored to eps)
  anomaly if score > threshold

Simplest univariate point-anomaly detector after Hampel. Reference:
Merlion.models.anomaly.zms.ZMS (with TemporalResample + LagTransform
preprocessing stripped for native CADW use).

Pure numpy — no sklearn/scipy/merlion/torch in cadw/ (AST-guard).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pyarrow as pa

from cadw.core.spill import AutoArrayResult, SpillArray


@dataclass
class ZMSResult:
    """ZMS fit result (v0.9.113 dual).

    `decision_scores` / `labels`: np.ndarray (in-memory, legacy np.ndarray
    input or output="memory") or AutoArrayResult (df input,
    output="auto"|"disk"). `is_spilled` is True only for the latter;
    use `.to_numpy()` (raises if spilled) or iterate for streaming.
    """
    decision_scores: Union[np.ndarray, AutoArrayResult]
    labels: Union[np.ndarray, AutoArrayResult]
    threshold: float
    window_size: int
    contamination: Optional[float]
    is_spilled: bool = False

    def state_nbytes(self) -> int:
        """v0.9.194 (S1.5): honest memory state for DEC-033 CW-1.

        Memory path (np.ndarray or InMemoryArrayResult):
            256 + n*8 (decision_scores) + n*8 (labels)
        Spilled path (OnDiskArrayResult): 256 (state O(1), disk O(n))
        """
        sz = 256
        if self.is_spilled:
            return sz
        for arr in (self.decision_scores, self.labels):
            if isinstance(arr, np.ndarray):
                sz += int(arr.nbytes)
            elif hasattr(arr, "to_numpy"):
                try:
                    sz += int(arr.to_numpy().nbytes)
                except Exception:
                    pass
        return int(sz)


def _resolve_chunk_nw_zms(w: int, guard=None) -> int:
    """v0.9.91-S6a: chunk size so peak RAM stays O(chunk*w).

    Bytes per window: view (w*8) + mean+std temp (~w*8). Total 2*w*8.
    """
    default_target = 8 * 1024 * 1024
    target_bytes = default_target
    if guard is not None:
        sl = getattr(guard, "state_limit_bytes", None)
        if isinstance(sl, (int, float)) and sl > 0:
            target_bytes = min(int(target_bytes), max(1, int(sl) // 8))
    bytes_per_window = 8 * max(int(w), 1) * 2
    n = max(100, target_bytes // bytes_per_window)
    return min(10_000, n)


# v0.9.130 (DEC-054): _probe_memory_limit_mb -> 
# cadw.core.memory_guard.probe_limit_mb (single source of truth).
from cadw.core.memory_guard import probe_limit_mb as _probe_memory_limit_mb


def fit_zms(
    df_or_series,
    value_col: Optional[str] = None,
    *,
    window_size: int = 50,
    threshold: float = 3.0,
    contamination: Optional[float] = None,
    guard=None,
    output: str = "auto",
    batch_size: "int | None" = None,
) -> ZMSResult:
    """Rolling z-score anomaly detection (v0.9.113 dual dispatch).

    df path (new, OOC-capable):
        fit_zms(df, value_col, output="auto")   -> SpillArray output
        fit_zms(df, value_col, output="memory") -> np.ndarray output
        fit_zms(df, value_col, output="disk")   -> forced spill

    Legacy np.ndarray path (back-compat, unchanged):
        fit_zms(series)   # in-memory, equivalent to output="memory"

    `contamination` requires output="memory" (percentile over full array).
    """
    is_df = (hasattr(df_or_series, "_iter_filtered_batches")
             and hasattr(df_or_series, "_columns"))
    if is_df:
        if value_col is None:
            raise ValueError(
                "fit_zms: value_col required when df is passed")
        if value_col not in df_or_series._columns:
            raise KeyError(
                f"fit_zms: column {value_col!r} not in {df_or_series._columns}")
        return _fit_zms_df(
            df_or_series, value_col,
            window_size=window_size, threshold=threshold,
            contamination=contamination, guard=guard, output=output,
            batch_size=batch_size,
        )
    if value_col is not None:
        raise ValueError(
            "fit_zms: value_col only used with df input; omit for np.ndarray")
    return _fit_zms_series(
        df_or_series,
        window_size=window_size, threshold=threshold,
        contamination=contamination, guard=guard,
    )


def _fit_zms_df(df, value_col, *, window_size, threshold, contamination,
                guard, output, batch_size=None):
    """df streaming ZMS; SpillArray output (unless output='memory')."""
    w = int(window_size)
    if w < 1:
        raise ValueError(f"window_size must be >= 1; got {w}")
    if output not in ("memory", "auto", "disk"):
        raise ValueError(
            f"output must be 'memory'|'auto'|'disk'; got {output!r}")
    if output != "memory" and contamination is not None:
        raise NotImplementedError(
            f"fit_zms: contamination={contamination!r} requires "
            f"output='memory' (percentile needs full array); "
            f"got output={output!r}")

    mg = guard if guard is not None else getattr(df, "_memory_guard", None)

    if batch_size is None:
        try:
            schema = df._dataset.schema
            row_size = mg.estimate_row_size_from_schema(schema)
            state_bytes = (w - 1) * 8 * 2
            batch_size = mg.get_batch_rows(
                row_size_bytes=row_size,
                expansion_factor=3.0,
                state_bytes=state_bytes,
            )
        except Exception:
            batch_size = max(w * 20, 10_000)

    if output == "memory":
        all_scores = []
        all_labels = []
        scores_buf = None
        labels_buf = None
    else:
        spill_ratio = 0.0 if output == "disk" else 0.5
        scores_buf = SpillArray(
            guard=mg, width=1, dtype=np.float64,
            name="zms.scores", spill_ratio=spill_ratio)
        labels_buf = SpillArray(
            guard=mg, width=1, dtype=np.float64,
            name="zms.labels", spill_ratio=spill_ratio)

    tail = None
    total_n = 0
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="zms.batch", extra_factor=3.0)
        x_b = b.column(value_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        if x_b.size == 0:
            continue
        if tail is None:
            tail = np.full(w - 1, x_b[0], dtype=np.float64)
        x_full = np.concatenate([tail, x_b])
        n_cur = len(x_b)
        view = np.lib.stride_tricks.sliding_window_view(x_full, w)
        chunk = _resolve_chunk_nw_zms(w, mg)
        mu = np.empty(n_cur, dtype=np.float64)
        sigma = np.empty(n_cur, dtype=np.float64)
        for i in range(0, n_cur, chunk):
            end = min(i + chunk, n_cur)
            nb = end - i
            if mg is not None and hasattr(mg, "check_state"):
                mg.check_state(nb * w * 8 * 2, name="zms.chunk")
            blk = np.ascontiguousarray(view[i:end])
            mu[i:end] = blk.mean(axis=1)
            sigma[i:end] = blk.std(axis=1)
        sigma = np.where(sigma < 1e-12, 1e-12, sigma)
        scores = np.abs(x_b - mu) / sigma

        if output == "memory":
            all_scores.append(scores)
            all_labels.append((scores > float(threshold)).astype(int))
        else:
            scores_buf.write_array(scores.reshape(-1, 1))
            labels_buf.write_array(
                (scores > float(threshold)).astype(np.float64).reshape(-1, 1))

        tail = x_full[-(w - 1):] if w > 1 else np.array([], dtype=np.float64)
        total_n += n_cur

    if total_n < w:
        raise ValueError(f"series length {total_n} < window_size {w}")

    if output == "memory":
        scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
        labels = (np.concatenate(all_labels) if all_labels
                  else np.zeros(0, dtype=int))
        if contamination is not None:
            thr = float(np.percentile(
                scores, 100.0 * (1.0 - contamination)))
            if thr <= 0:
                thr = float(threshold)
            labels = (scores > thr).astype(int)
        else:
            thr = float(threshold)
        return ZMSResult(
            decision_scores=scores, labels=labels,
            threshold=thr, window_size=w,
            contamination=contamination, is_spilled=False,
        )
    return ZMSResult(
        decision_scores=scores_buf.finalize(),
        labels=labels_buf.finalize(),
        threshold=float(threshold), window_size=w,
        contamination=contamination, is_spilled=True,
    )


def _fit_zms_series(
    series: np.ndarray,
    *,
    window_size: int = 50,
    threshold: float = 3.0,
    contamination: float | None = None,
    guard=None,
) -> ZMSResult:
    """Rolling z-score anomaly detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
    window_size : int, default=50
        Trailing window length.
    threshold : float, default=3.0
        Z-score threshold. If `contamination` is set, used only as
        fallback when percentile-derived threshold is 0.
    contamination : float | None, default=None
        If set, threshold = percentile(scores, 100*(1-c)).
    guard : MemoryGuard | None

    Returns
    -------
    ZMSResult
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"series length {n} < window_size {window_size}")

    # v0.9.91-S6a: legacy fallback for duck-typed guards without
    # check_state (max_memory_mb only). Real MemoryGuard uses chunked
    # check_state below.
    has_check_state = (
        guard is not None and hasattr(guard, "check_state")
        and callable(guard.check_state)
    )
    if guard is not None and not has_check_state:
        est_bytes = int(n * window_size * 8 * 2)
        limit_mb = _probe_memory_limit_mb(guard)
        if limit_mb is not None and limit_mb > 0:
            if est_bytes > limit_mb * 1024 * 1024:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"ZMS estimated {est_bytes} bytes > "
                    f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    # Trailing windows: pad window_size-1 with edge value
    padded = np.concatenate([np.full(window_size - 1, x[0]), x])
    view = np.lib.stride_tricks.sliding_window_view(padded, window_size)
    n_w = view.shape[0]
    chunk = _resolve_chunk_nw_zms(window_size, guard)

    mu = np.empty(n_w, dtype=np.float64)
    sigma = np.empty(n_w, dtype=np.float64)
    for i in range(0, n_w, chunk):
        end = min(i + chunk, n_w)
        nb = end - i
        if has_check_state:
            guard.check_state(
                nb * window_size * 8 * 2, name="zms.chunk")
        blk = np.ascontiguousarray(view[i:end])
        mu[i:end] = blk.mean(axis=1)
        sigma[i:end] = blk.std(axis=1)
    sigma = np.where(sigma < 1e-12, 1e-12, sigma)

    scores = np.abs(x - mu) / sigma

    if contamination is not None:
        thr = float(np.percentile(
            scores, 100.0 * (1.0 - contamination)))
        if thr <= 0:
            thr = float(threshold)
    else:
        thr = float(threshold)

    labels = (scores > thr).astype(int)

    return ZMSResult(
        decision_scores=scores,
        labels=labels,
        threshold=thr,
        window_size=window_size,
        contamination=contamination,
    )

