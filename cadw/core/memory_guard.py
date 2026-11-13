# cadw/core/memory_guard.py
"""
MemoryGuard: бюджет памяти и проверки перед операциями.
Все размеры — в байтах.

Контракт (см. DECISIONS.md §9):
  - available_bytes()                    — текущий бюджет
  - get_batch_rows(row_bytes, ef)        — верхняя граница строк в батче
  - min_batch_rows()                     — нижняя граница (soft)
  - check_batch(batch)                   — фактический nbytes <= available
  - check_state(nbytes)                  — nbytes <= available * state_safety
  - check_partition_buffers(P, buf)      — P*buf <= available * buffer_safety
  - estimate_row_size_from_batch(batch)  — из первого батча
  - estimate_row_size_from_schema(schema)
  - estimate_unique_keys(sample, total)  — для выбора P
  - estimate_state_size(op, schema, d)   — до запуска (cov, histogram)
  - log_step(...)                        — подробное логирование
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import pyarrow as pa

from cadw.core.logging import get_logger

from cadw.core.models import DataChars, Params, CostEstimate, get_model


logger = get_logger("memory_guard")

# v0.9.34 (B4): utf8 bloat factor for the len(set(...)) fallback.
_UNIQUE_FALLBACK_FACTOR = 4


def _fallback_unique_count(col) -> int:
    """v0.9.34 (B4): utf8 bloat factor for the short-string set fallback."""
    return len(set(col.to_pylist())) * _UNIQUE_FALLBACK_FACTOR


def measure_peak(func, *args, **kwargs):
    """Dev calibration tool: run func, measure peak RAM by 3 sources.

    Combines three independent measurements:
      - tracemalloc: Python + numpy allocs (excluding mmap)
      - pyarrow default memory pool: Arrow internal buffers
      - resource.getrusage: OS-level RSS peak (includes mmap, page cache)

    NOT for runtime use (1-10s overhead, page-cache effects). Use for:
      - verification tests: test_*_formula_accuracy
      - calibration scripts: scripts/calibrate_memory.py (planned v0.9.103)
      - ad-hoc debugging: `measure_peak(fit_lda, df=df, ...)`

    Linux/macOS only (uses `resource`; not available on Windows).
    RSS is peak-only (not current); for isolated measurements run in a
    subprocess. See docs/ml_tools/MEMORY_AND_TIME_ESTIMATION.md §3.

    Returns
    -------
    dict with keys:
        py_bytes:    tracemalloc peak (Python + numpy non-mmap)
        arrow_bytes: Arrow pool delta during run
        rss_bytes:   OS RSS peak delta (KB on Linux → bytes here)
        total_bytes: max(rss_bytes, py_bytes + arrow_bytes) — recommended
        result:      passthrough return of func(*args, **kwargs)
    """
    import tracemalloc
    import pyarrow as pa

    pool0 = pa.default_memory_pool().bytes_allocated()

    rss0 = 0
    try:
        import resource
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, AttributeError):
        pass

    tracemalloc.start()
    try:
        result = func(*args, **kwargs)
        py_peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    arrow_delta = max(0, pa.default_memory_pool().bytes_allocated() - pool0)

    rss_delta_bytes = 0
    try:
        import resource
        rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        delta_raw = rss_end - rss0
        # Linux ru_maxrss is KB; macOS is bytes.
        if delta_raw > 1_000_000_000:
            rss_delta_bytes = int(delta_raw)
        else:
            rss_delta_bytes = int(max(0, delta_raw) * 1024)
    except (ImportError, AttributeError):
        pass

    return {
        "py_bytes": int(py_peak),
        "arrow_bytes": int(arrow_delta),
        "rss_bytes": int(rss_delta_bytes),
        "total_bytes": max(rss_delta_bytes, int(py_peak) + int(arrow_delta)),
        "result": result,
    }


def _read_vmrss() -> int:
    """Current VmRSS in bytes from /proc/self/status. 0 if unavailable."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0


def _read_pss() -> int:
    """Current Pss in bytes from /proc/self/smaps_rollup. 0 if unavailable.

    Pss (proportional set size) accounts for shared pages proportionally:
    if a page is shared by N processes, each contributes 1/N. For a single
    process this is RSS minus shared .so pages (libc, python), so Pss delta
    is the *unique* memory cost of an operation. This is what matters for
    "does this fit in budget" question.
    """
    try:
        with open('/proc/self/smaps_rollup') as f:
            for line in f:
                if line.startswith('Pss:'):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0


def measure_peak_pss(
    func,
    *args,
    interval_s: float = 0.0001,
    **kwargs,
):
    """Measure incremental Pss peak during func() via background sampler.

    Unlike `measure_peak` (which uses ru_maxrss delta = peak-since-process-
    start and fails silently when preloaded peak >= fit peak — see DEC-031,
    DEC-032), this samples CURRENT Pss every interval_s and reports peak-base.

    Why this matters:
      - ru_maxrss delta = 0 whenever baseline (parquet load, prewarm scan,
        df materialization) raises ru_maxrss above fit peak. That is the
        common case for ML fits where state is small relative to data.
      - Pss sampler detects small deltas (>= ~50 KB) regardless of baseline
        because it compares peak vs current at start, not peak-since-start.

    Parameters
    ----------
    func : callable
        The function to measure.
    *args
        Positional arguments to func.
    interval_s : float, default 1e-4
        Sampling interval in seconds (100us default). Lower = higher
        detection floor but more CPU overhead.
    **kwargs
        Keyword arguments to func.

    Returns
    -------
    dict with keys:
        pss_bytes:    peak Pss delta in bytes (recommended for verification)
        vm_bytes:     peak VmRSS delta in bytes (for cross-check)
        samples:      number of samples taken
        result:       passthrough return of func(*args, **kwargs)

    Notes
    -----
    Linux-only (`/proc/self/smaps_rollup`, `/proc/self/status`). On other
    platforms returns pss_bytes=0, vm_bytes=0.

    Detection floor: ~50-100 KB (from .so page-fault noise). For sub-100KB
    signals this tool is not reliable; fall back to py_bytes from
    measure_peak (tracemalloc) if signal is Python-only.

    Sampling misses short spikes if func has a transient that lasts <
    interval_s. For typical ML fits (batch loops, per-arm sweeps) each
    phase lasts ms, so 100us sampling catches it. For one-shot numpy ops
    that may last <1ms, increase interval or use larger magnitudes.

    Uses sys.setswitchinterval to ensure the sampler thread gets CPU time
    between numpy allocations (numpy releases GIL for big ops).
    """
    import sys as _sys
    import threading
    import time
    import gc

    pss0 = _read_pss()
    vm0 = _read_vmrss()
    if pss0 == 0 and vm0 == 0:
        # /proc unavailable: fall through, return 0 measurement.
        return {"pss_bytes": 0, "vm_bytes": 0, "samples": 0,
                "result": func(*args, **kwargs)}

    gc.collect()
    pss_base = _read_pss()
    vm_base = _read_vmrss()

    pss_peak = [pss_base]
    vm_peak = [vm_base]
    stop = threading.Event()
    samples = [0]

    def _sampler():
        while not stop.is_set():
            p = _read_pss()
            v = _read_vmrss()
            if p > pss_peak[0]:
                pss_peak[0] = p
            if v > vm_peak[0]:
                vm_peak[0] = v
            samples[0] += 1
            time.sleep(interval_s)

    old_iv = _sys.getswitchinterval()
    _sys.setswitchinterval(max(interval_s / 2.0, 1e-5))
    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    try:
        result = func(*args, **kwargs)
    finally:
        stop.set()
        t.join(timeout=2.0)
        _sys.setswitchinterval(old_iv)

    return {
        "pss_bytes": max(0, pss_peak[0] - pss_base),
        "vm_bytes": max(0, vm_peak[0] - vm_base),
        "samples": samples[0],
        "result": result,
    }


_PAGE_SIZE = 4096


def _init_page_size() -> int:
    global _PAGE_SIZE
    try:
        import os as _os
        _PAGE_SIZE = int(_os.sysconf('SC_PAGE_SIZE'))
    except Exception:
        pass
    return _PAGE_SIZE


_init_page_size()


def _read_vmrss_fast() -> int:
    """VmRSS in bytes from /proc/self/statm (~2-5 us per call).

    /proc/self/statm is one line of 7 ints (size, resident, shared, text,
    lib, data, dt). Resident in field[1]. This is 100x faster than
    /proc/self/status (parse) and 1000x faster than /proc/self/smaps_rollup
    (walk VMAs), enabling ~200us sampling intervals needed to catch per-batch
    transients in ML fits.
    """
    try:
        with open('/proc/self/statm', 'rb') as f:
            parts = f.read().split()
        return int(parts[1]) * _PAGE_SIZE
    except Exception:
        return 0


def measure_peak_traced(
    func,
    *args,
    mg=None,
    interval_s: float = 0.0002,
    **kwargs,
):
    """Like measure_peak, but with fast VmRSS sampler + phase markers.

    Difference from measure_peak (DEC-031/032 issue): uses CURRENT RSS
    delta (peak - base), not ru_maxrss delta. Catches fit peak even when
    preload peak (parquet load) >= fit peak — the common case for ML fits
    where state is small relative to data.

    Difference from measure_peak_pss: /proc/self/statm (~2us) vs
    /proc/self/smaps_rollup (~3ms). Sampling interval 200us -> ~5000
    samples during 1s fit, vs 2-155 for Pss. Peak is caught.

    Phase markers: if `mg` is passed and `mg.mark(name)` is called inside
    func, the RSS at each mark is recorded. This lets us decompose the
    peak: which phase has how many bytes. Critical for "is formula wrong"
    diagnosis (R4: find missing term, not bump coefficient).

    Returns
    -------
    dict:
        peak_bytes:    peak RSS - base RSS, bytes (VmRSS delta)
        baseline:      RSS at start, bytes
        phase_events:  list of (name, rss_at_mark - baseline), ordered
        peak_phase:    name of phase where peak was reached (or None)
        samples:       n samples taken
        result:        func(*args, **kwargs)
    """
    import sys as _sys
    import threading
    import time as _time
    import gc as _gc

    if _read_vmrss_fast() == 0:
        # /proc unavailable: run without measurement.
        return {"peak_bytes": 0, "baseline": 0, "phase_events": [],
                "peak_phase": None, "samples": 0,
                "result": func(*args, **kwargs)}

    _gc.collect()
    baseline = _read_vmrss_fast()
    peak = [baseline]
    peak_name = [None]
    samples = [0]
    traced = []

    if mg is not None:
        mg._traced = traced

    stop = threading.Event()

    def _sampler():
        while not stop.is_set():
            v = _read_vmrss_fast()
            if v > peak[0]:
                peak[0] = v
            samples[0] += 1
            _time.sleep(interval_s)

    old_iv = _sys.getswitchinterval()
    _sys.setswitchinterval(max(interval_s / 2.0, 1e-5))
    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    try:
        result = func(*args, **kwargs)
    finally:
        stop.set()
        t.join(timeout=2.0)
        _sys.setswitchinterval(old_iv)
        if mg is not None:
            mg._traced = None

    # Post-process phase markers: for each, was it at peak?
    phase_events = [(name, rss - baseline) for name, rss in traced]
    max_phase_rss = baseline
    for name, rss in traced:
        if rss > max_phase_rss:
            max_phase_rss = rss
            peak_name[0] = name

    return {
        "peak_bytes": max(0, peak[0] - baseline),
        "baseline": baseline,
        "phase_events": phase_events,
        "peak_phase": peak_name[0],
        "samples": samples[0],
        "result": result,
    }


class MemoryLimitExceeded(RuntimeError):
    """Превышен лимит памяти (v0.9.176: enriched with actionable fields).

    Backward-compatible: ``MemoryLimitExceeded("msg")`` still works.
    Enriched constructor accepts optional fields for R2-compliant refuse
    (DEC-067: model, params, min_required_bytes, available_bytes, options).

    Attributes (all optional, default None/[]):
        model             — fit function name (e.g., "fit_gbdt_classifier")
        params            — user-supplied params dict (the ones that didn't fit)
        min_required_bytes — minimum RAM for best variant (analytical)
        available_bytes   — current MemoryGuard.available_bytes()
        options           — actionable suggestions list[str]
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        params: dict | None = None,
        min_required_bytes: int | None = None,
        available_bytes: int | None = None,
        options: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.params = params
        self.min_required_bytes = min_required_bytes
        self.available_bytes = available_bytes
        self.options = list(options) if options else []


class MemoryGuard:
    def __init__(
        self,
        max_memory_bytes: Optional[int] = None,
        *,
        max_memory_mb: Optional[float] = None,
        safety: float = 0.7,
        state_safety: float = 0.5,
        buffer_safety: float = 0.3,
        min_batch_rows: int = 1000,
        default_expansion_factor: float = 2.0,
        # v0.9.19: closed-loop measurement (opt-in, defaults off).
        runtime_monitor: bool = False,
        calibrate: bool = False,
        rss_soft_factor: float = 1.2,
        rss_hard_factor: float = 1.8,
    ) -> None:
        if max_memory_bytes is not None:
            budget = int(max_memory_bytes)
        elif max_memory_mb is not None:
            budget = int(max_memory_mb * 1024 * 1024)
        else:
            raise ValueError("Укажите max_memory_bytes или max_memory_mb")
        if budget <= 0:
            raise ValueError("Бюджет должен быть > 0")

        self._budget = budget
        self._safety = safety
        self._state_safety = state_safety
        self._traced: 'list | None' = None
        self._buffer_safety = buffer_safety
        self._min_batch_rows = max(1, int(min_batch_rows))
        self._default_ef = default_expansion_factor

        self._reserved = 0
        # Консервативный дефолт для legacy-методов; перезаписывается estimate_*
        self._row_size_bytes: int = 80

        # v0.9.19: closed-loop memory measurement (opt-in).
        self._runtime_monitor = bool(runtime_monitor)
        self._calibrate = bool(calibrate)
        self._rss_soft_factor = float(rss_soft_factor)
        self._rss_hard_factor = float(rss_hard_factor)
        # Lazy baselines (initialized on first _get_pool_delta / _get_rss_delta)
        self._pool: Any = None
        self._pool_baseline: int = 0
        self._process: Any = None
        self._rss_baseline: int = 0
        # Calibration samples: {op_name: [{predicted, arrow, rss}, ...]}
        self._calibration_samples: dict[str, list[dict]] = {}

        logger.info(
            "MemoryGuard: budget=%.2f MiB safety=%.2f state=%.2f buffer=%.2f "
            "min_rows=%d ef_default=%.2f",
            budget / (1024 * 1024), safety, state_safety, buffer_safety,
            min_batch_rows, default_expansion_factor,
        )

    # ---------- бюджет ----------

    def mark(self, name: str) -> None:
        """Phase annotation for measure_peak_traced.

        No-op outside tracing (mg._traced is None). Inside tracing, appends
        (name, current_vmrss) to the trace list. Overhead in hot loop when
        not tracing: ~20 ns per call (attribute lookup + branch).
        """
        traced = self._traced
        if traced is None:
            return
        traced.append((name, _read_vmrss_fast()))

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def state_limit_bytes(self) -> int:
        """Current state_size limit: `available_bytes() × state_safety`.

        Public since v0.8.11b — callers (e.g., `merge()` strict-mode
        decision) need to compare against state_limit without going
        through `check_state` (which raises).
        """
        return self._state_limit()

    def available_bytes(self) -> int:
        return max(0, self._budget - self._reserved)

    def _usable_bytes(self) -> int:
        return int(self.available_bytes() * self._safety)

    def _state_limit(self) -> int:
        return int(self.available_bytes() * self._state_safety)

    def _buffer_limit(self) -> int:
        return int(self.available_bytes() * self._buffer_safety)

    # ---------- оценка размера строки ----------

    def estimate_row_size_from_batch(self, batch: pa.RecordBatch) -> int:
        if batch.num_rows == 0:
            raise ValueError("Нельзя оценить размер строки по пустому батчу")
        size = max(1, math.ceil(batch.nbytes / batch.num_rows))
        self._row_size_bytes = size
        logger.debug("row_size=batch: nbytes=%d rows=%d -> %d B",
                     batch.nbytes, batch.num_rows, size)
        return size

    def estimate_row_size_from_schema(
        self, schema: pa.Schema, *, default_string_bytes: int = 50
    ) -> int:
        """Estimate bytes/row from schema alone. Conservative for numeric.

        WARNING (v0.9.8b, R2): for string/binary columns, uses
        `default_string_bytes=50` regardless of actual value length.
        Real-world string columns can be 4× to 20× larger (audit
        bench_r2: 10 string cols, 200 B/value → schema est 500 B/row,
        exact 10080 B/row). If schema contains string/binary fields,
        PREFER `reestimate_from_batch(batch)` (exact) or
        `estimate_row_size_from_schema_with_sample(schema, batch)`.

        Existing numeric-only schemas: accurate (bit_width // 8).
        """
        total = 0
        for field in schema:
            t = field.type
            if pa.types.is_boolean(t):
                total += 1
            elif pa.types.is_integer(t) or pa.types.is_floating(t):
                bit_width = getattr(t, "bit_width", 64)
                total += max(1, bit_width // 8)
            elif pa.types.is_string(t) or pa.types.is_binary(t):
                total += default_string_bytes
            elif pa.types.is_timestamp(t) or pa.types.is_date(t):
                total += 8
            else:
                total += 8
        total = max(1, total)
        self._row_size_bytes = total
        logger.debug("row_size=schema: %d B/row (%d fields)", total, len(schema))
        return total

    # legacy-совместимость
    def estimate_row_size(
        self,
        schema: Optional[pa.Schema] = None,
        sample_batch: Optional[pa.RecordBatch] = None,
    ) -> int:
        if sample_batch is not None:
            return self.estimate_row_size_from_batch(sample_batch)
        if schema is not None:
            return self.estimate_row_size_from_schema(schema)
        return self._row_size_bytes

    # ---------- v0.9.8b (R2): reestimate from sample batch ----------
    #
    # estimate_row_size_from_schema underestimates string/binary
    # (default_string_bytes=50, real 4–20× larger). If caller can peek
    # first batch (peek_first_batch in cadw.core.sampling), use these
    # methods for exact per-row bytes before choosing batch_size / P.
    #
    # §3.10 discipline: check before materialize. reestimate AFTER
    # peek, BEFORE any decision that depends on row_size.

    def reestimate_from_batch(self, batch: pa.RecordBatch) -> int:
        """Update self._row_size_bytes from real batch (v0.9.8b).

        Returns exact bytes/row (batch.nbytes / num_rows). Updates
        cached `self._row_size_bytes` used by `get_batch_rows()`.

        Use after `peek_first_batch`: schema-only estimate may be
        10–20× low for string/binary; real batch is exact.

        Raises ValueError for empty batch (caller must pass non-empty).
        """
        if batch.num_rows == 0:
            raise ValueError(
                "reestimate_from_batch: cannot estimate from empty "
                "batch (0 rows). Caller must pass non-empty batch."
            )
        return self.estimate_row_size_from_batch(batch)

    def estimate_row_size_from_schema_with_sample(
        self,
        schema: pa.Schema,
        sample_batch: pa.RecordBatch | None = None,
        *,
        default_string_bytes: int = 50,
    ) -> int:
        """Schema-based estimate; exact if sample_batch provided.

        v0.9.8b (R2): if `sample_batch` provided → uses
        `estimate_row_size_from_batch` (exact, R2 fix). Else falls
        back to `estimate_row_size_from_schema` (numeric-accurate,
        string-underestimate). Also updates `self._row_size_bytes`.
        """
        if sample_batch is not None:
            return self.estimate_row_size_from_batch(sample_batch)
        return self.estimate_row_size_from_schema(
            schema, default_string_bytes=default_string_bytes,
        )


    @property
    def row_size_bytes(self) -> int:
        return self._row_size_bytes

    # ---------- расчёт размера батча ----------

    def get_batch_rows(
        self,
        row_size_bytes: Optional[int] = None,
        expansion_factor: Optional[float] = None,
        *,
        sample_batch: Optional[pa.RecordBatch] = None,
        state_bytes: int = 0,
    ) -> int:
        """How many rows fit in one batch under current budget.

        v0.9.8b (R2): `sample_batch` kwarg — if provided, uses exact
        estimate_row_size_from_batch (fixes string underestimate).
        Otherwise uses `row_size_bytes` argument or cached value.

        v0.9.106 (DEC-022): `state_bytes` — deduct algorithm's persistent
        state (O(d²) for linear/logistic, O(C·d) for NB, etc.) before
        sizing batch. Without this, at large d state dominates and
        check_batch / check_state overcommit (silent OOM).

        Backward compat: state_bytes=0 → identical to previous behavior.

        Raises MemoryLimitExceeded if state_bytes >= usable_bytes
        (state alone exceeds the batch budget — reduce d or increase
        budget; no batch size can fix this).
        """
        if sample_batch is not None:
            row_bytes = self.estimate_row_size_from_batch(sample_batch)
        else:
            row_bytes = (
                row_size_bytes if row_size_bytes is not None
                else self._row_size_bytes
            )
        ef = expansion_factor if expansion_factor is not None else self._default_ef
        if row_bytes <= 0 or ef <= 0:
            raise ValueError("row_size_bytes и expansion_factor должны быть > 0")
        if state_bytes < 0:
            raise ValueError("state_bytes must be >= 0")

        usable = self._usable_bytes() - int(state_bytes)
        if usable <= 0:
            logger.error(
                "get_batch_rows FAIL: state_bytes=%d >= usable_bytes=%d "
                "(state alone exceeds budget)",
                state_bytes, self._usable_bytes(),
            )
            raise MemoryLimitExceeded(
                f"state_bytes {state_bytes} B >= usable_bytes "
                f"{self._usable_bytes()} B. No batch size fits. "
                f"Reduce d / C, or increase budget."
            )
        raw = int(usable / (row_bytes * ef))

        if raw >= self._min_batch_rows:
            rows = raw
        elif raw >= 1:
            rows = raw
            logger.warning(
                "batch_rows=%d < min_batch_rows=%d (usable=%d B, row=%d B, ef=%.2f). "
                "min_batch_rows недостижим при текущем бюджете.",
                rows, self._min_batch_rows, usable, row_bytes, ef,
            )
        else:
            rows = 1
            logger.warning(
                "Бюджета не хватает даже на 1 строку: usable=%d B, row=%d B, ef=%.2f. "
                "Возможен MemoryLimitExceeded.",
                usable, row_bytes, ef,
            )

        logger.debug("get_batch_rows: usable=%d row=%d ef=%.2f -> %d rows",
                     usable, row_bytes, ef, rows)
        return rows

    def get_window_size(
        self,
        *,
        unit_bytes: int,
        name: str = "window",
        target_fraction: float = 0.25,
    ) -> int:
        """Largest W s.t. W * unit_bytes <= state_limit * target_fraction.

        v0.9.176 (DEC-067/068): bound on algorithm **state** (window size for
        MLE-forecasters, max_window for pelt, etc.).

        Different from ``get_batch_rows`` (I/O granularity, ~1-10k rows).
        This returns algorithmic state capacity (~10^5-10^7 observations
        depending on budget).

        Parameters
        ----------
        unit_bytes : int
            Bytes per unit of window (e.g., 8 for f64 series).
        name : str
            Diagnostic label.
        target_fraction : float
            Fraction of state_limit to allocate for window. Default 0.25
            (leaves 0.75 for other state + batch + temp arrays).

        Returns
        -------
        int
            Window size (>= 2). Returns 2 if state_limit is trivially small
            (guard refuses with min_history check elsewhere).
        """
        if unit_bytes < 1:
            raise ValueError(f"unit_bytes must be >= 1; got {unit_bytes}")
        if not (0 < target_fraction < 1):
            raise ValueError(
                f"target_fraction must be in (0, 1); got {target_fraction}"
            )
        target = int(self._state_limit() * target_fraction)
        return max(2, target // unit_bytes)

    def min_batch_rows(self) -> int:
        return self._min_batch_rows

    def default_expansion_factor(self) -> float:
        return self._default_ef

    # ---------- проверки ----------

    def check_batch(
        self,
        batch: pa.RecordBatch,
        *,
        name: str = "batch",
        extra_factor: float = 1.0,
    ) -> None:
        """Проверяет, что батч (с учётом extra_factor) влезает в available_bytes.

        extra_factor > 1.0 используется, когда операция создаёт временный
        overhead поверх batch.nbytes: например, compute_hashes делает
        to_pylist() (Python-int ~28 B на значение), что для int-колонок
        даёт x3-x5. Передавайте консервативный коэффициент (3.0-5.0),
        чтобы check_batch отражал реальный пик RAM.
        """
        if extra_factor < 1.0:
            raise ValueError("extra_factor must be >= 1.0")
        size = int(batch.nbytes * extra_factor)
        limit = self.available_bytes()
        if size > limit:
            logger.error(
                "check_batch FAIL: %s nbytes=%d extra_factor=%.2f effective=%d > available=%d",
                name, batch.nbytes, extra_factor, size, limit,
            )
            raise MemoryLimitExceeded(
                f"{name}: {size} B (nbytes={batch.nbytes} * {extra_factor}) > available {limit} B"
            )
        logger.debug(
            "check_batch OK: %s nbytes=%d extra_factor=%.2f effective=%d available=%d",
            name, batch.nbytes, extra_factor, size, limit,
        )

    def check_state(self, state_bytes: int, *, name: str = "state") -> None:
        limit = self._state_limit()
        if state_bytes > limit:
            logger.error("check_state FAIL: %s nbytes=%d > limit=%d",
                         name, state_bytes, limit)
            raise MemoryLimitExceeded(f"{name}: {state_bytes} B > state_limit {limit} B")
        logger.debug("check_state OK: %s nbytes=%d limit=%d", name, state_bytes, limit)

    def check_partition_buffers(self, num_partitions: int, buffer_bytes: int) -> None:
        if num_partitions < 1:
            raise ValueError("num_partitions должен быть >= 1")
        if buffer_bytes < 1:
            raise ValueError("buffer_bytes должен быть >= 1")
        total = num_partitions * buffer_bytes
        limit = self._buffer_limit()
        if total > limit:
            logger.error("check_partition_buffers FAIL: P=%d buf=%d total=%d limit=%d",
                         num_partitions, buffer_bytes, total, limit)
            raise MemoryLimitExceeded(
                f"Партиционные буферы: {num_partitions}*{buffer_bytes}={total} B > limit {limit} B"
            )
        logger.debug("check_partition_buffers OK: P=%d buf=%d total=%d limit=%d",
                     num_partitions, buffer_bytes, total, limit)

    def recommended_partition_buffer_bytes(self, num_partitions: int) -> int:
        if num_partitions < 1:
            raise ValueError
        return max(1, self._buffer_limit() // num_partitions)

    # ---------- v0.9.8a (R1): real per-writer check ----------
    #
    # check_partition_buffers(P, recommended_partition_buffer_bytes(P))
    # is vacuous: recommended(P) = buffer_limit // P → product
    # ≤ buffer_limit by construction. Real ArrowIPCBatchWriter
    # footprint (audit bench_r1: 1.2–4.4 KB/writer measured at
    # P=100..5000) is not visible to that check → 1e6 writers
    # could allocate ~1.2 GB while passing.
    #
    # check_partition_writers uses a measured conservative
    # baseline (2048 B/writer). Threshold shared with buffer_safety
    # (0.3 × available_bytes), same as check_partition_buffers.

    def check_partition_writers(
        self,
        num_partitions: int,
        *,
        bytes_per_writer: int = 2048,
    ) -> None:
        """Verify P × bytes_per_writer fits buffer budget.

        v0.9.8a (R1): check_partition_buffers is vacuous by
        construction (`P × (limit // P) ≤ limit`); this check
        uses measured per-writer ArrowIPC writer footprint.

        bytes_per_writer default = 2048 B (conservative median
        of audit bench_r1: 1.2 KB @ P=5000, 1.9 KB @ P=1000,
        4.4 KB @ P=100; median 2 KB with margin).

        Limit = buffer_safety × available_bytes (same as
        check_partition_buffers). Raises MemoryLimitExceeded.
        """
        if num_partitions < 1:
            raise ValueError("num_partitions must be >= 1")
        if bytes_per_writer < 1:
            raise ValueError("bytes_per_writer must be >= 1")
        total = num_partitions * bytes_per_writer
        limit = self._buffer_limit()
        if total > limit:
            logger.error(
                "check_partition_writers FAIL: P=%d × %d B/writer "
                "= %d B > limit=%d B (per-writer overhead, "
                "audit bench_r1)",
                num_partitions, bytes_per_writer, total, limit,
            )
            raise MemoryLimitExceeded(
                f"Partition writers: P={num_partitions} × "
                f"{bytes_per_writer} B/writer = {total} B > "
                f"limit {limit} B. Reduce num_partitions or "
                f"increase budget. (Real ArrowIPCBatchWriter "
                f"footprint; see bench_r1.)"
            )
        logger.debug(
            "check_partition_writers OK: P=%d × %d B = %d ≤ %d",
            num_partitions, bytes_per_writer, total, limit,
        )

    # ---------- v0.9.19: closed-loop memory measurement ----------
    #
    # Tier 1 (pool): pa.default_memory_pool().bytes_allocated() is an
    # exact C++ atomic counter for Arrow allocations. ~100ns to read.
    # Delta since MemoryGuard.__init__ = bytes held by CADW's Arrow use.
    #
    # Tier 2 (RSS): psutil.Process().memory_info().rss is approximate
    # (includes all memory: Arrow + numpy + Python + shared libs).
    # ~10us per read; only sampled (rss_sample_every). Catches unmodeled
    # allocations (to_pylist Python objects, np.zeros, foreign libs).
    #
    # Tier 3 (calibration): per-op {predicted, arrow, rss} sample log.
    # Compare actual to predicted — C = actual/predicted. Warn on C > 2.
    #
    # All baselines are lazy: initialized on first call, so
    # MemoryGuard() with both flags off has zero overhead.

    def _get_pool_delta(self) -> int:
        """Bytes allocated by Arrow since guard creation (or last baseline).

        Lazily initializes pa.default_memory_pool() baseline on first call.
        Cheap (~100ns): atomic read of C++ counter.
        """
        if self._pool is None:
            self._pool = pa.default_memory_pool()
            self._pool_baseline = self._pool.bytes_allocated()
        return self._pool.bytes_allocated() - self._pool_baseline

    def _get_rss_delta(self) -> int:
        """Bytes of RSS delta since guard creation.

        Lazily initializes psutil.Process() baseline on first call.
        ~10us per read (system call). Sampled, not per-batch.
        """
        if self._process is None:
            import psutil
            self._process = psutil.Process()
            self._rss_baseline = self._process.memory_info().rss
        return self._process.memory_info().rss - self._rss_baseline

    @contextmanager
    def calibrate(
        self,
        op_name: str,
        predicted_bytes: int = 0,
        sample_peak: bool = False,
        sample_interval_ms: float = 10.0,
    ) -> Iterator[None]:
        """Measure pool + RSS delta for op; warn on formula drift.

        v0.9.19: context manager wraps op execution. On entry, captures
        pool/RSS baselines. On exit, computes deltas and:
          - if calibrate=True: records {predicted, arrow, rss} sample;
            warns (log WARNING) if actual/predicted > 2.0 (drift signal).
          - if runtime_monitor=True: raises MemoryLimitExceeded if RSS
            delta > rss_hard_factor × budget (safety net for unmodeled
            allocations). Warns if > rss_soft_factor × budget.

        predicted_bytes=0 means "no formula" (measure only). Caller may
        pass model estimate to get actionable drift signal.

        Both flags off → yield only, zero overhead.

        v0.9.20: `sample_peak=True` — start background thread that polls
        pa.default_memory_pool().bytes_allocated() and psutil RSS every
        `sample_interval_ms` (default 10ms), tracking max. Records
        peak_absolute (peak above Arrow zero, comparable to formula §9.2.N
        which predicts absolute peak) AND peak_delta (peak above op-start
        baseline). Use when validating formula (absolute) or detecting
        unmodeled temp buffers (delta). No overhead when calibrate=False
        — thread only spawns when sample_peak=True AND calibrate=True.
        """
        if not (self._calibrate or self._runtime_monitor):
            yield
            return

        # v0.9.19-fix2: direct lazy init — do NOT call _get_*_delta.
        # Those are caller-facing measurement APIs that read from
        # self._pool / self._process on EVERY call, even when baseline
        # is already set. Calling them here consumed 1 extra read from
        # each of pool/process, breaking start/end measurement:
        # start=clamped[last], end=clamped[last] → delta=0. §13.4/231.
        if self._pool is None:
            self._pool = pa.default_memory_pool()
            self._pool_baseline = self._pool.bytes_allocated()
        if self._process is None:
            import psutil
            self._process = psutil.Process()
            self._rss_baseline = self._process.memory_info().rss

        pool_start = self._pool.bytes_allocated()
        rss_start = self._process.memory_info().rss

        # v0.9.20: optional peak sampling via background thread.
        # Runs only when sample_peak=True AND (calibrate or monitor).
        # Thread spawn ~100us; polls every sample_interval_ms.
        peak_pool_box = [pool_start]
        peak_rss_box = [rss_start]
        stop_evt = None
        sampler_thread = None
        if sample_peak:
            import threading
            stop_evt = threading.Event()
            interval_s = max(0.001, float(sample_interval_ms) / 1000.0)

            def _sampler():
                while not stop_evt.is_set():
                    try:
                        p = self._pool.bytes_allocated()
                        r = self._process.memory_info().rss
                        if p > peak_pool_box[0]:
                            peak_pool_box[0] = p
                        if r > peak_rss_box[0]:
                            peak_rss_box[0] = r
                    except Exception:
                        pass
                    stop_evt.wait(interval_s)

            sampler_thread = threading.Thread(
                target=_sampler, daemon=True,
                name=f"cadw_peak_sampler_{op_name}",
            )
            sampler_thread.start()

        try:
            yield
        finally:
            if stop_evt is not None:
                stop_evt.set()
                if sampler_thread is not None:
                    sampler_thread.join(timeout=2.0)

            pool_end = self._pool.bytes_allocated()
            rss_end = self._process.memory_info().rss
            net_pool_delta = pool_end - pool_start
            net_rss_delta = rss_end - rss_start

            if sample_peak:
                peak_pool_abs = peak_pool_box[0]
                peak_rss_abs = peak_rss_box[0]
                peak_pool_delta = peak_pool_abs - pool_start
                peak_rss_delta = peak_rss_abs - rss_start
            else:
                peak_pool_abs = pool_end
                peak_rss_abs = rss_end
                peak_pool_delta = net_pool_delta
                peak_rss_delta = net_rss_delta

            # Tier 2 (runtime_monitor): hard safety net via peak RSS
            # when sampled (more honest), fallback net RSS otherwise.
            if self._runtime_monitor:
                effective_rss = peak_rss_delta if sample_peak else net_rss_delta
                soft = int(self._budget * self._rss_soft_factor)
                hard = int(self._budget * self._rss_hard_factor)
                if effective_rss > hard:
                    raise MemoryLimitExceeded(
                        f"{op_name}: RSS delta {effective_rss} B > "
                        f"{self._rss_hard_factor:.1f}× budget ({hard} B). "
                        f"Unmodeled allocation (numpy/pandas/foreign lib?). "
                        f"Arrow delta {peak_pool_delta if sample_peak else net_pool_delta} B. "
                        f"See §9.6."
                    )
                elif effective_rss > soft:
                    logger.warning(
                        "RSS soft-drift: op=%s delta=%d B > %.1f× budget "
                        "(soft=%d B, hard=%d B). Arrow delta %d B.",
                        op_name, effective_rss, self._rss_soft_factor,
                        soft, hard,
                        peak_pool_delta if sample_peak else net_pool_delta,
                    )

            # Tier 3 (calibrate): log sample, warn on formula drift
            # v0.9.20-fix3 (§13.4/235): pass peak DELTA (above op-start)
            # not peak ABS. Formula §9.2.N predicts delta (batch + state);
            # actual must be delta too. Absolute pool/RSS include
            # pre-existing allocations (all batches, interpreter) → C
            # absurd (e.g. 161 MB / 5 KB = 32210).
            if self._calibrate:
                self._record_calibration(
                    op_name,
                    predicted_bytes,
                    peak_pool_delta,
                    peak_rss_delta,
                    net_pool_delta,
                    net_rss_delta,
                    sample_mode=sample_peak,
                )

    def _record_calibration(
        self,
        op_name: str,
        predicted: int,
        peak_pool_delta: int,
        peak_rss_delta: int,
        net_pool_delta: int,
        net_rss_delta: int,
        *,
        sample_mode: bool = False,
    ) -> None:
        """Append calibration sample; warn on C = peak_delta/predicted > 2.

        v0.9.20-fix3 (§13.4/235): peak_pool/peak_rss entry fields are
        DELTAS above op-start (peak_abs - start), not absolute.
        Formula §9.2.N predicts delta (batch + state); comparison must
        be delta/delta, not absolute/predicted. v0.9.20 compared peak_abs
        (e.g. 161 MB interpreter RSS) to predicted (5 KB) → C=32210.
        """
        entry = {
            "predicted": int(predicted),
            "peak_pool": int(peak_pool_delta),
            "peak_rss": int(peak_rss_delta),
            "net_pool_delta": int(net_pool_delta),
            "net_rss_delta": int(net_rss_delta),
            "sample_mode": "peak" if sample_mode else "delta",
        }
        self._calibration_samples.setdefault(op_name, []).append(entry)

        if predicted > 0:
            c_peak = peak_pool_delta / predicted
            c_rss = peak_rss_delta / predicted
            logger.info(
                "calibrate op=%s mode=%s predicted=%d peak_pool_delta=%d "
                "(C=%.2f) peak_rss_delta=%d (C=%.2f) net_pool_delta=%d",
                op_name, entry["sample_mode"], predicted, peak_pool_delta,
                c_peak, peak_rss_delta, c_rss, net_pool_delta,
            )
            if c_peak > 2.0 or c_rss > 2.0:
                logger.warning(
                    "calibrate DRIFT: op=%s predicted=%d C_peak=%.2f "
                    "C_rss=%.2f (threshold 2.0; consider fixing formula "
                    "or updating §9.2.N calibration)",
                    op_name, predicted, c_peak, c_rss,
                )
        else:
            logger.info(
                "calibrate op=%s mode=%s predicted=unknown "
                "peak_pool_delta=%d peak_rss_delta=%d net_pool_delta=%d",
                op_name, entry["sample_mode"], peak_pool_delta,
                peak_rss_delta, net_pool_delta,
            )

    @property
    def calibration_samples(self) -> dict[str, list[dict]]:
        """Read-only copy of calibration samples.

        {op_name: [{"predicted": int, "arrow": int, "rss": int}, ...]}

        Empty if calibrate=False. Accumulated across all calibrate() calls.
        """
        return {k: list(v) for k, v in self._calibration_samples.items()}

    def reset_calibration(self) -> None:
        """Clear accumulated calibration samples (between bench runs)."""
        self._calibration_samples = {}

    # ---------- оценка числа уникальных ключей ----------

    def estimate_unique_keys(
        self,
        sample_batch: pa.RecordBatch,
        total_rows: int,
    ) -> int:
        """Estimate cardinality via Arrow-native ops (v0.5.2b, P9).

        - single column: `dictionary_encode().dictionary.length`.
        - multi-column: binary_join_element_wise → dictionary_encode.
        - Fallback: pc.count_distinct per column (product bound).

        No `to_pylist()` in fast path — no Python-object overhead on strings.
        """
        if sample_batch.num_rows == 0 or total_rows <= 0:
            return 0

        import pyarrow.compute as pc

        ncols = sample_batch.num_columns
        k_sample = None

        if ncols == 1:
            col = sample_batch.column(0)
            try:
                enc = col.dictionary_encode()
                if enc.dictionary is not None:
                    k_sample = len(enc.dictionary)
            except Exception:
                pass
            if k_sample is None:
                try:
                    k_sample = int(pc.count_distinct(col).as_py())
                except Exception:
                    # v0.9.34 (B4): apply utf8 bloat factor (part1_report
                    # §2.3). Python str objects + set slots ~4x raw bytes
                    # for short strings; matches the encode_categorical
                    # 180 B/entry heuristic. Prior code returned the raw
                    # count, underestimating downstream state by ~4x.
                    k_sample = _fallback_unique_count(col)

        if k_sample is None:
            # multi-column: binary_join → dict_encode, else product fallback
            try:
                if hasattr(pc, "binary_join_element_wise"):
                    cols = [sample_batch.column(i) for i in range(ncols)]
                    str_cols = [
                        c if pa.types.is_string(c.type) else pc.cast(c, pa.string())
                        for c in cols
                    ]
                    joined = str_cols[0]
                    for c in str_cols[1:]:
                        joined = pc.binary_join_element_wise(joined, c, "|")
                    enc = joined.dictionary_encode()
                    if enc.dictionary is not None:
                        k_sample = len(enc.dictionary)
            except Exception:
                pass
            if k_sample is None:
                product = 1
                for i in range(ncols):
                    col = sample_batch.column(i)
                    try:
                        product *= int(pc.count_distinct(col).as_py())
                    except Exception:
                        product = sample_batch.num_rows
                        break
                k_sample = min(product, sample_batch.num_rows)

        k_est = int(math.ceil(k_sample * (total_rows / sample_batch.num_rows)))
        logger.debug(
            "estimate_unique_keys: sample_rows=%d unique=%d total_rows=%d -> K_est=%d",
            sample_batch.num_rows, k_sample, total_rows, k_est,
        )
        return k_est

    # ---------- оценка размера состояния ----------

    def estimate_state_size(
        self,
        op_name: str,
        schema: Optional[pa.Schema] = None,
        *,
        d: Optional[int] = None,
        bins: int = 255,
    ) -> int:
        name = op_name.lower()
        if name in {"count", "sum", "min", "max", "mean"}:
            return 16
        if name in {"var", "std"}:
            return 24
        if name in {"cov", "corr"}:
            dim = d if d is not None else (len(schema) if schema is not None else 1)
            return 8 + 8 * dim + 8 * dim * dim
        if name == "histogram":
            return 8 * bins + 16
        # v0.9.25e (A.10): fallback scales with state_limit, not fixed
        # 1 MiB. Prior: fixed 1 MiB → false raise at tight budgets
        # (1 MiB > 250 KB state_limit at 500 KB budget) and false pass
        # at large budgets (1 MiB < 2% of state_limit at 100 MB).
        # Scaled fallback: unknown op "moderate" relative to available
        # memory. An op whose init() OOMs despite passing this fallback
        # is the op's responsibility (§11.5 documented limitation) —
        # override ReduceOp.estimate_state_nbytes() for exact bound.
        fallback = int(self._state_limit() * 0.1)
        logger.warning(
            "estimate_state_size: unknown op %r, fallback = "
            "10%% of state_limit (%d B). Override "
            "ReduceOp.estimate_state_nbytes() on the op for exact bound.",
            op_name, fallback,
        )
        return fallback

    # ---------- резервирование ----------

    def reserve(self, nbytes: int) -> None:
        if nbytes < 0:
            raise ValueError
        if self._reserved + nbytes > self._budget:
            raise MemoryLimitExceeded(
                f"Резервирование {nbytes} B превышает бюджет "
                f"(reserved={self._reserved}, budget={self._budget})"
            )
        self._reserved += nbytes

    def release(self, nbytes: int) -> None:
        self._reserved = max(0, self._reserved - nbytes)

    # ---------- логирование ----------

    def _load_throughput_rates(self) -> dict:
        """Read rates from throughput.yaml (default profile). v0.9.119."""
        from pathlib import Path as _P
        try:
            import yaml as _yaml
        except ImportError:
            return {}
        _YAML_DIR = _P(__file__).resolve().parent.parent.parent / "docs" / "ml_tools"
        cp = _YAML_DIR / "throughput.yaml"
        if not cp.exists():
            return {}
        try:
            data = _yaml.safe_load(cp.read_text(encoding="utf-8"))
        except Exception:
            return {}
        profiles = (data or {}).get("profiles") or {}
        default = profiles.get("default") or {}
        return dict(default.get("rates") or {})

    @staticmethod
    def _classify_time(seconds: float) -> str:
        if seconds < 1.0:
            return "fast"
        if seconds < 60.0:
            return "medium"
        if seconds < 600.0:
            return "slow"
        return "very_slow"

    def solve_batch(
        self,
        op_name: str,
        data_chars: DataChars,
        *,
        budget: Optional[int] = None,
        **extra_params,
    ) -> "int | None":
        """Inverse of ram_formula w.r.t. batch (DEC-044, AP23).

        Returns largest batch b s.t. ram_formula(op_name, batch=b,
        row_size=..., **extra_params) <= budget. Fills d, n, k_est, k, K
        from data_chars if absent in extra_params. Caller supplies C,
        K_topics, w, etc. explicitly via **extra_params.

        Returns None if:
          - card has no ram_formula (draft/legacy)
          - formula is batch-independent (e.g. state-only)
          - formula is non-linear in batch
        In all None cases caller falls back to expansion_factor +
        LegacyFormulaWarning.
        """
        from cadw.ml._memory_estimate import (
            _lookup_card, _is_linear_in_batch, _numeric_inverse_batch,
        )
        card = _lookup_card(op_name)
        if card is None or not card.get("ram_formula"):
            return None
        budget = int(budget) if budget is not None else int(self._usable_bytes())
        row_size = max(1, int(data_chars.row_size_bytes))
        params = dict(extra_params)
        params.setdefault("d", len(data_chars.schema))
        params.setdefault("n", int(data_chars.n_rows))
        if data_chars.k_est is not None:
            kk = int(data_chars.k_est)
            params.setdefault("k_est", kk)
            params.setdefault("k", kk)
            params.setdefault("K", kk)
        if not _is_linear_in_batch(op_name, params, row_size):
            return None
        return _numeric_inverse_batch(op_name, params, budget, row_size)

    def estimate_cost(
        self,
        op_name: str,
        data_chars: DataChars,
        *,
        op_context: Optional[dict] = None,
    ) -> CostEstimate:
        """Estimated RAM + time for op_name (v0.9.119, AP22).

        Uses existing plan() + estimate() for RAM; adds time from
        throughput.yaml rate (conservative, LegacyFormulaWarning if
        absent). Returns CostEstimate with fits_ram + time_class.
        """
        import warnings as _w
        from cadw.core.warnings import LegacyFormulaWarning

        model = get_model(op_name)
        params = model.solve(
            self.available_bytes(), data_chars,
            safety=self._safety, op_context=op_context,
        )
        if params is None:
            return CostEstimate(
                op_name=op_name, est_ram_bytes=0, fits_ram=False,
                est_ops=0, est_time_seconds=0.0,
                time_class="unknown", reason="does_not_fit",
            )
        est_ram = model.estimate(params, data_chars)
        fits_ram = int(est_ram) <= int(self.available_bytes() * self._safety)

        est_time_model = model.estimate_time(params, data_chars)
        est_ops_default = int(data_chars.n_rows)
        if est_time_model is not None:
            est_ops = est_ops_default
            est_time = float(est_time_model)
            time_class = self._classify_time(est_time)
        else:
            rates = self._load_throughput_rates()
            entry = rates.get(op_name) or {}
            rate = entry.get("rate_per_sec")
            est_ops = est_ops_default
            if rate and float(rate) > 0:
                est_time = float(est_ops) / float(rate)
                time_class = self._classify_time(est_time)
            else:
                _w.warn(
                    f"estimate_cost({op_name!r}): no throughput rate; "
                    f"time_class=unknown (LegacyFormulaWarning, AP22)",
                    LegacyFormulaWarning, stacklevel=2,
                )
                est_time = 0.0
                time_class = "unknown"

        return CostEstimate(
            op_name=op_name,
            est_ram_bytes=int(est_ram),
            fits_ram=fits_ram,
            est_ops=int(est_ops),
            est_time_seconds=float(est_time),
            time_class=time_class,
        )

    def plan(
        self,
        op_name: str,
        data_chars: DataChars,
        *,
        op_context: Optional[dict] = None,
    ) -> Params:
        """Запланировать одну операцию (§9.2). Raises MemoryLimitExceeded при None."""
        model = get_model(op_name)
        params = model.solve(
            self.available_bytes(),
            data_chars,
            safety=self._safety,
            op_context=op_context,
        )
        if params is None:
            raise MemoryLimitExceeded(
                f"plan({op_name!r}): no params exist. {model.describe()}"
            )
        return params

    def plan_pipeline(self, ops, data_chars: DataChars, *, allow_spill: bool = True):
        """Запланировать цепочку (§9.3). Raises PipelineMemoryError.

        `allow_spill` — no-op в v0.1.x (spill_boundaries reserved for v0.2).
        """
        from cadw.core.planner import plan_pipeline
        return plan_pipeline(
            ops,
            data_chars,
            self.available_bytes(),
            safety=self._safety,
            allow_spill=allow_spill,
            memory_guard=self,
        )

    def log_step(
        self,
        step: str,
        *,
        batch_id: Optional[int] = None,
        n_rows: Optional[int] = None,
        batch_bytes: Optional[int] = None,
        state_bytes: Optional[int] = None,
    ) -> None:
        logger.info(
            "step=%s batch_id=%s rows=%s batch_bytes=%s state_bytes=%s available=%d",
            step, batch_id, n_rows, batch_bytes, state_bytes, self.available_bytes(),
        )

    # ---------- legacy (DEPRECATED) ----------
    # Оставлены для совместимости с текущим кодом. При переходе на новые
    # примитивы (Reduce/HashShuffle) эти методы будут удалены.

    def get_read_batch_size(self) -> int:
        """DEPRECATED: используйте get_batch_rows(expansion_factor=1.0)."""
        limit = int(self._budget * 0.10)
        return max(1, limit // max(1, self._row_size_bytes))

    def get_write_batch_size(self) -> int:
        """DEPRECATED."""
        limit = int(self._budget * 0.20)
        return max(1, limit // max(1, self._row_size_bytes))

    def get_max_dict_elements(self) -> int:
        """DEPRECATED: будет заменено байтовым лимитом в HashPartitioner."""
        limit = int(self._budget * 0.60)
        return max(1, limit // 80)



# ---- probe_limit_mb (canonical, v0.9.130 DEC-054) ----

def measure_peak_pool(fn, *args, sample_ms: float = 5.0, **kwargs):
    """Measure peak Arrow pool bytes AND peak VmRSS during fn(*args, **kwargs).

    v0.9.230 (DEC-081): pool_peak is live Arrow alloc (authoritative for
    framework RAM claims). rss_peak is VmRSS delta (includes OS page cache
    for spill files, glibc arena fragmentation, mmap'd IPC sinks — typically
    4-32x above pool for external_sort workloads).

    Returns:
      {"result": fn result, "pool_peak": int, "pool_after": int,
       "rss_peak": int}

    Usage:
      r = measure_peak_pool(fit_kaplan_meier, df, time_col="t", ...)
      assert r["pool_peak"] < budget  # honest framework claim
      # r["rss_peak"] documents actual machine RAM (overreporting, disclosed)
    """
    import gc, threading, time
    import pyarrow as pa

    pool = pa.default_memory_pool()
    base_pool = pool.bytes_allocated()
    base_rss = _read_vmrss_fast()
    peak_pool = [0]
    peak_rss = [0]
    stop = [False]

    def _sampler():
        while not stop[0]:
            try:
                cur = pool.bytes_allocated()
                if cur > peak_pool[0]:
                    peak_pool[0] = cur
                rss = _read_vmrss_fast()
                if rss > peak_rss[0]:
                    peak_rss[0] = rss
            except Exception:
                pass
            time.sleep(sample_ms / 1000.0)

    gc.collect()
    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        stop[0] = True
        t.join(timeout=1)
    return {
        "result": result,
        "pool_peak": peak_pool[0] - base_pool,
        "pool_after": pool.bytes_allocated() - base_pool,
        "rss_peak": peak_rss[0] - base_rss,
    }


def probe_limit_mb(guard) -> float | None:
    """Probe memory limit in MB from a duck-typed guard.

    Canonical implementation. Prior to v0.9.130 had 16 duplicate
    copies in cadw/ml/*.py (autoencoder, binseg, hampel, zms, stray,
    bocpd, sub_pca, eif, tspq, s2g, patch_memory_ad, cusum,
    kmeans_ad, kshape, dwt_mlead, linear).

    Returns
    -------
    None if guard is None or no limit attribute is found;
    float MB if guard exposes a positive max_memory_mb / limit_mb /
    budget_bytes / etc. Checks attribute names first, then
    guard.__dict__ for any *mb* key matching mem/budget/limit/max.
    """
    if guard is None:
        return None

    def _is_pos(v):
        return (isinstance(v, (int, float))
                and not isinstance(v, bool) and v > 0)

    MB = ("max_memory_mb", "_max_memory_mb", "limit_mb", "budget_mb",
          "_budget_mb", "max_mb", "_max_mb", "memory_limit_mb",
          "_memory_limit_mb")
    for a in MB:
        v = getattr(guard, a, None)
        if _is_pos(v):
            return float(v)

    BYTES = ("max_memory_bytes", "_max_memory_bytes", "max_bytes",
             "_max_bytes", "limit_bytes", "_limit_bytes",
             "budget_bytes", "_budget_bytes")
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
        if _is_pos(v) and "mb" in kl and any(
                t in kl for t in ("mem", "budget", "limit", "max")):
            return float(v)
    return None
