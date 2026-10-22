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

from cadw.core.models import DataChars, Params, get_model


logger = get_logger("memory_guard")


class MemoryLimitExceeded(RuntimeError):
    """Превышен лимит памяти."""


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
    ) -> int:
        """How many rows fit in one batch under current budget.

        v0.9.8b (R2): `sample_batch` kwarg — if provided, uses exact
        estimate_row_size_from_batch (fixes string underestimate).
        Otherwise uses `row_size_bytes` argument or cached value.
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

        usable = self._usable_bytes()
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
                    k_sample = len(set(col.to_pylist()))

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