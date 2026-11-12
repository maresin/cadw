"""Hampel filter: rolling median + MAD anomaly detector.

For each timestamp t with window [t-half, t+half]:
  med = median(window); mad = median(|window - med|)
  sigma = 1.4826 * mad; score[t] = |x[t] - med| / sigma
  anomaly if score > n_sigma (default 3.0)

v0.9.217: df path with output='auto' (default) streams batch-by-batch;
SpillArray for scores/labels; fit transient O(batch + w); O(1) state
spilled. Legacy output='memory' preserves bit-identical O(n) path.
ndarray input: only 'memory'|'auto' (both in-memory); 'disk' raises.

Reference: sktime.transformations.series.outlier_detection.HampelDetector.
Pure numpy — no sklearn/scipy/aeon/torch in cadw/ (AST-guard).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from cadw.core.spill import SpillArray, AutoArrayResult


@dataclass
class HampelResult:
    decision_scores: "np.ndarray | AutoArrayResult"
    labels: "np.ndarray | AutoArrayResult"
    threshold: float
    window_size: int
    n_sigma: float
    contamination: float | None
    is_spilled: bool = False

    def state_nbytes(self) -> int:
        """v0.9.217: O(1) when spilled (256 header); O(n) legacy memory path."""
        if self.is_spilled:
            return 256
        return int(self.decision_scores.nbytes + self.labels.nbytes + 8 + 256)


def _resolve_chunk_nw_hampel(w: int, guard=None) -> int:
    default_target = 8 * 1024 * 1024
    target_bytes = default_target
    if guard is not None:
        sl = getattr(guard, "state_limit_bytes", None)
        if isinstance(sl, (int, float)) and sl > 0:
            target_bytes = min(int(target_bytes), max(1, int(sl) // 8))
    bytes_per_window = 8 * max(int(w), 1) * 2
    n = max(100, target_bytes // bytes_per_window)
    return min(10_000, n)


from cadw.core.memory_guard import probe_limit_mb as _probe_memory_limit_mb


def fit_hampel(
    df_or_series,
    value_col=None,
    *,
    window_size: int = 7,
    n_sigma: float = 3.0,
    contamination: float | None = None,
    guard=None,
    output: str = "auto",
    batch_size: "int | None" = None,
) -> HampelResult:
    """Hampel filter (v0.9.217 dual-dispatch + streaming).

    df path:
      output='auto' (default) -> streaming; SpillArray; state O(1) spilled.
      output='memory'         -> legacy bit-identical; state O(n).
      output='disk'           -> forced spill.

    ndarray path: only output='memory'|'auto' (both in-memory);
      output='disk' raises NotImplementedError.

    contamination requires output='memory' (percentile needs full array).
    """
    if output not in ("memory", "auto", "disk"):
        raise ValueError(
            f"fit_hampel: output must be 'memory'|'auto'|'disk'; got {output!r}")
    is_df = (hasattr(df_or_series, "_iter_filtered_batches")
             and hasattr(df_or_series, "_columns"))
    if is_df:
        if value_col is None:
            raise ValueError(
                "fit_hampel: value_col required when df is passed")
        if value_col not in df_or_series._columns:
            raise KeyError(
                f"fit_hampel: column {value_col!r} not in "
                f"{df_or_series._columns}")
        return _fit_hampel_df(
            df_or_series, value_col,
            window_size=window_size, n_sigma=n_sigma,
            contamination=contamination, guard=guard,
            output=output, batch_size=batch_size)
    if value_col is not None:
        raise ValueError(
            "fit_hampel: value_col only used with df input; omit for np.ndarray")
    if output == "disk":
        raise NotImplementedError(
            "fit_hampel: output='disk' requires df input "
            "(ndarray is already in RAM)")
    return _fit_hampel_series(
        df_or_series, window_size=window_size, n_sigma=n_sigma,
        contamination=contamination, guard=guard)


def _fit_hampel_series(series, *, window_size, n_sigma, contamination, guard):
    return _fit_hampel_series_impl(
        series, window_size=window_size, n_sigma=n_sigma,
        contamination=contamination, guard=guard)


def _fit_hampel_df(df, value_col, *, window_size, n_sigma, contamination,
                   guard, output, batch_size=None):
    if contamination is not None and output != "memory":
        raise NotImplementedError(
            f"fit_hampel: contamination={contamination!r} requires "
            f"output='memory' (percentile needs full scores array); "
            f"got output={output!r}")
    mg = guard if guard is not None else getattr(df, "_memory_guard", None)

    if output == "memory":
        parts = []
        for b in df._iter_filtered_batches(batch_size=batch_size):
            if mg is not None and hasattr(mg, "check_batch"):
                mg.check_batch(b, name="hampel.batch", extra_factor=2.0)
            x_b = b.column(value_col).cast(
                pa.float64(), safe=False).to_numpy(zero_copy_only=False)
            if x_b.size > 0:
                parts.append(x_b)
        if not parts:
            raise ValueError(f"fit_hampel: column {value_col!r} is empty")
        series = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return _fit_hampel_series_impl(
            series, window_size=window_size, n_sigma=n_sigma,
            contamination=contamination, guard=mg)

    return _fit_hampel_df_streaming(
        df, value_col, window_size=window_size, n_sigma=n_sigma,
        guard=mg, output=output, batch_size=batch_size)


def _fit_hampel_df_streaming(df, value_col, *, window_size, n_sigma,
                              guard, output, batch_size=None):
    """v0.9.217: streaming; next_x = real future (may be shorter than half).

    Invariant: after processing, emit one batch per iteration; pending_future
    drains to empty; last batch (empty next_x) edge-pads with x_b[-1].
    Per-window median is chunk-independent -> bit-identical to memory path.
    """
    mg = guard
    n_total = int(df.count())
    if n_total < 3:
        raise ValueError(f"series too short for Hampel (min 3), got {n_total}")
    if window_size < 3:
        raise ValueError(f"window_size must be >= 3, got {window_size}")
    if window_size % 2 == 0:
        window_size += 1
    if window_size > n_total:
        window_size = n_total if n_total % 2 == 1 else n_total - 1
    half = window_size // 2
    threshold = float(n_sigma)

    if mg is not None and hasattr(mg, "check_state"):
        mg.check_state(256 + half * 8 * 4, name="hampel.streaming.state")

    spill_ratio = 0.0 if output == "disk" else 0.5
    scores_buf = SpillArray(
        guard=mg, width=1, dtype=np.float64,
        name="hampel.scores", spill_ratio=spill_ratio)
    labels_buf = SpillArray(
        guard=mg, width=1, dtype=np.float64,
        name="hampel.labels", spill_ratio=spill_ratio)

    if batch_size is None:
        try:
            if mg is not None and hasattr(mg, "estimate_row_size_from_schema"):
                row_size = mg.estimate_row_size_from_schema(df._dataset.schema)
                batch_size = mg.get_batch_rows(
                    row_size_bytes=row_size,
                    expansion_factor=3.0,
                    state_bytes=4 * window_size * 8)
            else:
                batch_size = 1000
        except Exception:
            batch_size = 1000
    batch_size = max(1, min(int(batch_size), n_total))

    it = iter(df._iter_filtered_batches(batch_size=batch_size))
    pending_future = np.empty(0, dtype=np.float64)
    prev_tail = np.empty(0, dtype=np.float64)
    n_written = 0
    exhausted = False

    def _pull_one():
        nonlocal pending_future, exhausted
        if exhausted:
            return False
        try:
            nb = next(it)
        except StopIteration:
            exhausted = True
            return False
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(nb, name="hampel.batch", extra_factor=2.0)
        arr = nb.column(value_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        if arr.size > 0:
            pending_future = (arr if len(pending_future) == 0
                              else np.concatenate([pending_future, arr]))
        return True

    def _fill_until(min_len):
        while len(pending_future) < min_len and not exhausted:
            if not _pull_one():
                break

    # Prime: batch_size + half (lookahead)
    _fill_until(batch_size + half)

    while len(pending_future) > 0:
        n_take = min(batch_size, len(pending_future))
        current_x = pending_future[:n_take].copy()
        pending_future = (pending_future[n_take:].copy()
                          if n_take < len(pending_future)
                          else np.empty(0, dtype=np.float64))

        # Ensure we have real future >= half OR exhausted
        _fill_until(half)

        # next_x = real future (may be empty; never None)
        next_x = (pending_future[:half].copy()
                  if len(pending_future) > 0 else
                  np.empty(0, dtype=np.float64))

        _process_hampel_chunk(
            current_x, prev_tail, next_x, half, threshold, window_size,
            scores_buf, labels_buf, mg)
        n_written += len(current_x)

        combined = (np.concatenate([prev_tail, current_x])
                    if len(prev_tail) > 0 else current_x)
        prev_tail = (combined[-half:].copy()
                     if len(combined) >= half else combined.copy())

    if n_written != n_total:
        raise RuntimeError(
            f"hampel streaming: wrote {n_written}, expected {n_total}")

    return HampelResult(
        decision_scores=scores_buf.finalize(),
        labels=labels_buf.finalize(),
        threshold=threshold, window_size=window_size,
        n_sigma=n_sigma, contamination=None, is_spilled=True)


def _process_hampel_chunk(x_b, prev_tail, next_x, half, threshold, window_size,
                           scores_buf, labels_buf, mg):
    """Process one batch with real future context (next_x is ndarray, may be empty).

    - prefix: prev_tail (real past), or edge-pad(x[0]) for first batch.
    - suffix: next_x (real future) if any, else edge-pad(x_b[-1]) for last batch.
    - next_x short (< half): pad with next_x[-1] (= last real future = df[-1]
      if this is the last non-empty batch; or a mid-sequence value if data
      continues — but we _fill_until(half) before this call, so len(next_x) <
      half IMPLIES iterator exhausted -> next_x[-1] is df[-1]).
    """
    n_chunk = len(x_b)
    if n_chunk == 0:
        return

    # Prefix (half elements) — real past or edge from x[0]
    if len(prev_tail) >= half:
        prefix = prev_tail[-half:].astype(np.float64, copy=False)
    elif len(prev_tail) > 0:
        pad_left = half - len(prev_tail)
        prefix = np.concatenate([
            np.full(pad_left, prev_tail[0], dtype=np.float64),
            prev_tail.astype(np.float64, copy=False)])
    else:
        prefix = np.full(half, x_b[0], dtype=np.float64)

    # Suffix (half elements) — real future or edge from x_b[-1]
    if len(next_x) >= half:
        suffix = next_x[:half].astype(np.float64, copy=False)
    elif len(next_x) > 0:
        # Short real future: pad with LAST REAL future value (= df[-1] because
        # _fill_until(half) already exhausted iterator before calling us)
        pad_right = half - len(next_x)
        suffix = np.concatenate([
            next_x.astype(np.float64, copy=False),
            np.full(pad_right, next_x[-1], dtype=np.float64)])
    else:
        # No future: last batch, edge pad with x_b[-1] (= df[-1])
        suffix = np.full(half, x_b[-1], dtype=np.float64)

    extended = np.concatenate([prefix, x_b, suffix])
    view = np.lib.stride_tricks.sliding_window_view(extended, window_size)
    if view.shape[0] != n_chunk:
        raise RuntimeError(
            f"hampel streaming: view {view.shape[0]} != {n_chunk}")

    chunk_size = _resolve_chunk_nw_hampel(window_size, mg)
    chunk_size = max(1, min(chunk_size, n_chunk))

    for i in range(0, n_chunk, chunk_size):
        end = min(i + chunk_size, n_chunk)
        nb = end - i
        if mg is not None and hasattr(mg, "check_state"):
            mg.check_state(
                nb * window_size * 8 * 2, name="hampel.chunk")
        blk = np.ascontiguousarray(view[i:end])
        m_blk = np.median(blk, axis=1)
        mad_blk = np.median(np.abs(blk - m_blk[:, None]), axis=1)
        sigma_blk = 1.4826 * mad_blk
        sigma_blk = np.where(sigma_blk < 1e-12, 1e-12, sigma_blk)
        scores_blk = np.abs(x_b[i:end] - m_blk) / sigma_blk
        labels_blk = (scores_blk > threshold).astype(np.float64)
        scores_buf.write_array(scores_blk.reshape(-1, 1))
        labels_buf.write_array(labels_blk.reshape(-1, 1))


def _fit_hampel_series_impl(
    series: np.ndarray, *, window_size: int = 7, n_sigma: float = 3.0,
    contamination: float | None = None, guard=None,
) -> HampelResult:
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < 3:
        raise ValueError(f"series too short for Hampel (min 3), got {n}")
    if window_size < 3:
        raise ValueError(f"window_size must be >= 3, got {window_size}")
    if window_size % 2 == 0:
        window_size += 1
    if window_size > n:
        window_size = n if n % 2 == 1 else n - 1

    has_check_state = (
        guard is not None and hasattr(guard, "check_state")
        and callable(guard.check_state))
    if guard is not None and not has_check_state:
        est_bytes = int(n * window_size * 8 * 2)
        limit_mb = _probe_memory_limit_mb(guard)
        if limit_mb is not None and limit_mb > 0:
            if est_bytes > limit_mb * 1024 * 1024:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"Hampel estimated {est_bytes} bytes > "
                    f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    half = window_size // 2
    padded = np.pad(x, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window_size)
    n_w = view.shape[0]
    chunk = _resolve_chunk_nw_hampel(window_size, guard)

    if has_check_state:
        guard.check_state(int(5 * n_w * 8 + 256), name="hampel.output")

    med = np.empty(n_w, dtype=np.float64)
    mad = np.empty(n_w, dtype=np.float64)
    for i in range(0, n_w, chunk):
        end = min(i + chunk, n_w)
        nb = end - i
        if has_check_state:
            guard.check_state(
                nb * window_size * 8 * 2, name="hampel.chunk")
        blk = np.ascontiguousarray(view[i:end])
        m_blk = np.median(blk, axis=1)
        med[i:end] = m_blk
        mad[i:end] = np.median(np.abs(blk - m_blk[:, None]), axis=1)

    sigma = 1.4826 * mad
    sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    scores = np.abs(x - med) / sigma

    if contamination is not None:
        threshold = float(np.percentile(
            scores, 100.0 * (1.0 - contamination)))
        if threshold <= 0:
            threshold = float(n_sigma)
    else:
        threshold = float(n_sigma)

    labels = (scores > threshold).astype(int)

    return HampelResult(
        decision_scores=scores, labels=labels,
        threshold=threshold, window_size=window_size,
        n_sigma=n_sigma, contamination=contamination, is_spilled=False)
