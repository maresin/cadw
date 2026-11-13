"""Kaplan-Meier estimator (product-limit) — out-of-core (v0.9.143, DEC-058).

Promoted from internal audit prototype. Pipeline:
  1. Count total rows (single pass, O(batch)).
  2. external_sort by time_col via cadw.ops.sort (spills runs to disk).
  3. Streaming pass over sorted runs -> step function S(t) via
     _KMAccumulator. Buffer (n_unique, 5) starts in RAM; spills to disk
     when n_unique * 40 > state_limit_bytes.

bounded_state: true — either in-RAM step function (n_unique small) or
SpillDir-backed (n_unique > cap). n-independent via external_sort +
spill. State_nbytes non-spilled: n_unique * 5 * f64 + 128.
Spilled: 256 B (Path + SpillDir refs + header).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import (
    SpillDir, array_schema, array_to_record_batch, record_batch_to_numpy,
    spill_array_writer, spill_array_reader,
)
from cadw.ops.sort import external_sort, read_sorted


_HEADER_BYTES = 128
_SPILLED_HEADER_BYTES = 256


class _KMAccumulator:
    """Streaming Kaplan-Meier step accumulator (see impl docstring)."""

    def __init__(self, n_total: int, mg: MemoryGuard, sdir: SpillDir,
                 max_buf_rows: int = 1024):
        self.n_total = int(n_total)
        self.mg = mg
        self.sdir = sdir
        # v0.9.229-f (P25c): cap buf at min(max_buf_rows, state_limit//40).
        # Was max(1024, state_limit//40) -> buf preallocated to state_limit
        # (~100-250MB at 500MB budget) regardless of n. Now 40KB default;
        # spills after max_buf_rows uniques. Peak becomes constant in n.
        self.cap = min(max_buf_rows, max(1, int(mg.state_limit_bytes // 40)))
        # G2 guard: cap*40 <= state_limit_bytes by construction, but AST
        # gate requires explicit check_state before np.empty.
        mg.check_state(self.cap * 5 * 8, name="km.buf")
        self.buf = np.empty((self.cap, 5), dtype=np.float64)
        self.n_buf = 0
        self.spill_path: Optional[Path] = None
        self.spill_writer = None
        self._spill_cm = None       # v0.9.229-f2: context manager handle
        self._spilled = False       # v0.9.229-f2: spill happened (finalize signal)
        self.n_spilled = 0
        self.S = 1.0
        self.greenwood = 0.0
        self.seen_before = 0
        self.cur_t: Optional[float] = None
        self.cur_d = 0
        self.cur_c = 0
        self.n_unique = 0
        self.median_survival: Optional[float] = None

    def _emit(self, t: float, n_at_risk: int, d: int):
        if n_at_risk <= 0 or d < 0:
            return
        if d > 0:
            frac = d / n_at_risk
            self.S *= (1.0 - frac)
            if n_at_risk > d:
                self.greenwood += d / (n_at_risk * (n_at_risk - d))
        row = (float(t), float(self.S), float(self.greenwood),
               float(n_at_risk), float(d))
        self.n_unique += 1
        if self.median_survival is None and self.S <= 0.5:
            self.median_survival = float(t)
        if self.spill_writer is None:
            if self.n_buf < self.cap:
                self.buf[self.n_buf] = row
                self.n_buf += 1
                return
            self._open_spill()
        self._write_one(row)

    def _open_spill(self):
        if self._spilled:
            return
        self.spill_path = self.sdir.path("km_steps.arrow")
        self._spill_cm = spill_array_writer(
            self.spill_path, array_schema(5, np.float64))
        # v0.9.229-f2: capture yielded writer. Before: __enter__() return
        # value was discarded -> 'self.spill_writer' was the
        # _GeneratorContextManager, which lacks write_batch() ->
        # AttributeError on first spill. Latent since v0.9.116 because
        # cap=max(1024, state_limit//40)=6.5M at 500MB budget: n_unique
        # <= 6.5M never spilled, so path never executed in tests.
        self.spill_writer = self._spill_cm.__enter__()
        chunk = 4096
        for i in range(0, self.n_buf, chunk):
            self.spill_writer.write_batch(
                array_to_record_batch(self.buf[i:i + chunk]))
        self.n_spilled = self.n_buf
        self.n_buf = 0
        self.buf = None
        self._spilled = True

    def _write_one(self, row):
        arr = np.asarray(row, dtype=np.float64).reshape(1, 5)
        self.spill_writer.write_batch(array_to_record_batch(arr))
        self.n_spilled += 1

    def consume_sorted_batches(self, batches, time_col: str = "t",
                               event_col: str = "e") -> None:
        for batch in batches:
            t_arr = np.asarray(batch.column(time_col).to_numpy(
                zero_copy_only=False), dtype=np.float64)
            e_arr = np.asarray(batch.column(event_col).to_numpy(
                zero_copy_only=False), dtype=np.float64) != 0.0
            n = t_arr.shape[0]
            if n == 0:
                continue
            change_idx = np.flatnonzero(np.diff(t_arr) != 0) + 1
            starts = np.concatenate(([0], change_idx))
            ends = np.concatenate((change_idx, [n]))
            for s, e in zip(starts, ends):
                t_val = float(t_arr[s])
                d = int(e_arr[s:e].sum())
                c = int((e - s) - d)
                if self.cur_t is None:
                    self.cur_t = t_val; self.cur_d = d; self.cur_c = c
                elif t_val == self.cur_t:
                    self.cur_d += d; self.cur_c += c
                else:
                    n_at_risk = self.n_total - self.seen_before
                    self._emit(self.cur_t, n_at_risk, self.cur_d)
                    self.seen_before += self.cur_d + self.cur_c
                    self.cur_t = t_val; self.cur_d = d; self.cur_c = c
        if self.cur_t is not None:
            n_at_risk = self.n_total - self.seen_before
            self._emit(self.cur_t, n_at_risk, self.cur_d)
            self.seen_before += self.cur_d + self.cur_c
            self.cur_t = None

    def close(self):
        if self._spill_cm is not None:
            self._spill_cm.__exit__(None, None, None)
            self._spill_cm = None
            self.spill_writer = None

    def finalize(self):
        if self._spilled:
            return {
                "times": None, "survival": None, "greenwood": None,
                "n_at_risk": None, "n_events": None,
                "spill_path": self.spill_path,
                "n_unique": self.n_unique,
                "median_survival": self.median_survival,
            }
        return {
            "times": self.buf[:self.n_buf, 0].copy(),
            "survival": self.buf[:self.n_buf, 1].copy(),
            "greenwood": self.buf[:self.n_buf, 2].copy(),
            "n_at_risk": self.buf[:self.n_buf, 3].astype(np.int64),
            "n_events": self.buf[:self.n_buf, 4].astype(np.int64),
            "spill_path": None,
            "n_unique": self.n_unique,
            "median_survival": self.median_survival,
        }


@dataclass
class KaplanMeierModel:
    times: Optional[np.ndarray]
    survival: Optional[np.ndarray]
    greenwood: Optional[np.ndarray]
    n_at_risk: Optional[np.ndarray]
    n_events: Optional[np.ndarray]
    n_total: int
    n_unique: int
    median_survival: Optional[float]
    spill_path: Optional[Path] = None
    spill_dir: Optional[SpillDir] = None

    def state_nbytes(self) -> int:
        if self.spill_path is not None:
            return _SPILLED_HEADER_BYTES
        return int(self.n_unique * 5 * 8 + _HEADER_BYTES)

    def _iter_step_chunks(self):
        if self.spill_path is None:
            yield np.column_stack([
                self.times, self.survival, self.greenwood,
                self.n_at_risk, self.n_events]).astype(np.float64)
            return
        with spill_array_reader(self.spill_path) as r:
            for rb in r:
                yield record_batch_to_numpy(rb)

    def survival_at(self, t: float) -> float:
        last = 1.0
        if self.spill_path is None:
            idx = np.searchsorted(self.times, t, side="right") - 1
            if idx < 0:
                return 1.0
            return float(self.survival[idx])
        for chunk in self._iter_step_chunks():
            if chunk.shape[0] == 0:
                continue
            mask = chunk[:, 0] <= t
            if not mask.any():
                break
            last_idx = int(np.flatnonzero(mask)[-1])
            if chunk[last_idx, 0] > t:
                break
            last = float(chunk[last_idx, 1])
            if chunk[last_idx, 0] == t:
                break
        return last

    def survival_curve(self) -> Iterator[tuple[float, float]]:
        if self.spill_path is None:
            for i in range(self.times.shape[0]):
                yield float(self.times[i]), float(self.survival[i])
            return
        for chunk in self._iter_step_chunks():
            for row in chunk:
                yield float(row[0]), float(row[1])

    def cleanup(self) -> None:
        if self.spill_dir is not None:
            self.spill_dir.__exit__(None, None, None)
            self.spill_dir = None


def _count_total(df, time_col: str, mg: MemoryGuard,
                 batch_size: int | None) -> int:
    total = 0
    for batch in df._iter_filtered_batches(batch_size=batch_size):
        total += batch.num_rows
        mg.check_batch(batch, name="km.count_batch")
    return total


def fit_kaplan_meier(
    df, *,
    time_col: str,
    event_col: str,
    memory_guard=None,
    batch_size: int | None = None,
    spill_dir: Path | None = None,
    max_buf_rows: int = 1024,             # NEW v0.9.229-f (P25c)
    sort_buffer_bytes: int = 16_384,      # NEW v0.9.229-f (P25c, 16 KB)
) -> KaplanMeierModel:
    """Fit Kaplan-Meier on right-censored survival data.

    Input columns: time_col (positive float, event or censoring time),
    event_col (0/1 indicator; 1=event, 0=censored). Any other columns
    are passed through external_sort but not used.

    OOC: external_sort + streaming accumulator. Buffer spills to disk
    when n_unique > max_buf_rows (default 1024). State (non-spilled):
    n_unique * 5 * f64 + 128; (spilled): 256 B.

    v0.9.229-f (P25c + DEC-023): external_sort(max_buffer_bytes=
    sort_buffer_bytes) bounds Phase 1 at 4*sort_buffer_bytes (default
    64KB). Accumulator buf capped at max_buf_rows (40KB). batch_size
    auto-derived from sort_buffer_bytes (not whole-n single batch).
    Peak = 4*sort_buffer_bytes + max_buf_rows*5*f64 +
    4*batch*row_size + 512 ~= 168KB constant (independent of n).
    Prior: Phase 1 O(0.2*budget), buf prealloc O(state_limit), batch=n.
    """
    mg = memory_guard or MemoryGuard(max_memory_mb=512)

    # v0.9.229-f (DEC-023 + P25c): batch must fit one Phase-1 sort run.
    # At batch_size=None single batch = whole n -> O(n) peak from
    # _count_total + external_sort factory (v0.9.138/139/114 precedent).
    if batch_size is None:
        try:
            _row_size = mg.estimate_row_size_from_schema(df._dataset.schema)
        except Exception:
            _row_size = 16
        _rows_per_run = max(1, sort_buffer_bytes // max(1, _row_size))
        batch_size = min(_rows_per_run, 10_000)

    factory = lambda: df._iter_filtered_batches(batch_size=batch_size)
    n_total = _count_total(df, time_col, mg, batch_size)
    if n_total == 0:
        raise ValueError("fit_kaplan_meier: empty input")

    if spill_dir is not None:
        sdir = SpillDir(guard=mg, name="kaplan_meier", path=spill_dir)
    else:
        sdir = SpillDir(guard=mg, name="kaplan_meier")
    sdir.__enter__()

    try:
        sorted_path = external_sort(
            factory(),
            key_columns=[time_col],
            value_columns=[time_col, event_col],
            memory_guard=mg,
            spill_dir=sdir.root,
            max_buffer_bytes=sort_buffer_bytes,   # NEW v0.9.229-f (P25c)
            log=False,
        )
        acc = _KMAccumulator(n_total=n_total, mg=mg, sdir=sdir,
                             max_buf_rows=max_buf_rows)  # NEW v0.9.229-f
        try:
            acc.consume_sorted_batches(
                read_sorted(sorted_path, columns=[time_col, event_col]),
                time_col=time_col, event_col=event_col,
            )
        finally:
            acc.close()
        out = acc.finalize()
    except BaseException:
        sdir.__exit__(None, None, None)
        raise

    return KaplanMeierModel(
        times=out["times"], survival=out["survival"],
        greenwood=out["greenwood"], n_at_risk=out["n_at_risk"],
        n_events=out["n_events"], n_total=int(n_total),
        n_unique=int(out["n_unique"]),
        median_survival=out["median_survival"],
        spill_path=out["spill_path"], spill_dir=sdir,
    )
