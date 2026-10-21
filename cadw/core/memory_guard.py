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
from typing import Optional

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

    @property
    def row_size_bytes(self) -> int:
        return self._row_size_bytes

    # ---------- расчёт размера батча ----------

    def get_batch_rows(
        self,
        row_size_bytes: Optional[int] = None,
        expansion_factor: Optional[float] = None,
    ) -> int:
        row_bytes = row_size_bytes if row_size_bytes is not None else self._row_size_bytes
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
        logger.warning("estimate_state_size: неизвестная операция '%s', "
                       "консервативно 1 MiB", op_name)
        return 1024 * 1024

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