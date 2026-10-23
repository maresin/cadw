# cadw/dataframe/_join.py
# v0.9.26u (B.1 step 15): _JoinMixin.
# Symbol: merge.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import pyarrow as pa
from cadw.core.sampling import (
    data_chars_for_join,
    peek_first_batch,
)
from cadw.dataset import Dataset
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.ops.join import hash_join, hash_join_streaming
from typing import Sequence


class _JoinMixin:
    """Join / merge mixin (v0.9.26u)."""
    def merge(
        self,
        right: "CadwDataFrame",
        on: str | Sequence[str],
        how: str = "inner",
        *,
        suffixes: tuple[str, str] = ("_x", "_y"),
        num_partitions: int | None = None,
        use_planner: bool = True,
    ) -> "CadwDataFrame":
        """Hash join (v0.9.23j). how ∈ {'inner', 'left',
        'outer', 'anti', 'semi'}.

        - how='inner': совпадающие ключи (left ∪ right, on-coalesced).
        - how='left':  все left + right-колонки (null при нет совпадения).
        - how='outer': left ∪ right, обе стороны null-padded.
        - how='anti':  left без совпадения в right (SQL anti join).
        - how='semi':  left с совпадением в right (SQL semi join);
                       output = left-колонки only.

        NULL-семантика для anti/semi: null-ключ слева никогда
        не совпадает (SQL); anti emit'ит, semi — нет.
        - on: имена общих колонок в обеих сторонах.
        - suffixes: для пересекающихся не-on колонок.
        - num_partitions: число партиций HashShuffle; None — из
          JoinMemoryModel (если use_planner), иначе дефолт 8.
        - use_planner: использовать JoinMemoryModel.solve() для выбора
          num_partitions / max_output_rows. Если
          False — runtime использует старые дефолты (0.3 * available / 256).
        """
        from ._core import CadwDataFrame, logger
        if how not in ("inner", "left", "outer", "anti", "semi"):
            raise NotImplementedError(
                f"merge: how={how!r}; supported: inner, left, "
                f"outer, anti, semi. 'right' — v0.9.23j."
            )
        if not isinstance(right, CadwDataFrame):
            raise TypeError("merge: right must be CadwDataFrame")
        on_list = [on] if isinstance(on, str) else list(on)
        for c in on_list:
            if c not in self.columns:
                raise KeyError(f"merge: on column {c!r} not in self")
            if c not in right.columns:
                raise KeyError(f"merge: on column {c!r} not in right")

        left_schema = pa.schema(
            [self._dataset.schema.field(c) for c in self.columns]
        )
        right_schema = pa.schema(
            [right._dataset.schema.field(c) for c in right.columns]
        )

        # --- Planner: sample first batch, plan, override defaults. ---
        max_output_rows: int | None = None

        if use_planner:
            left_sample = peek_first_batch(self._iter_filtered_batches())
            right_sample = peek_first_batch(right._iter_filtered_batches())

            # Пустая сторона: короткое замыкание (runtime обработает).
            if left_sample is not None and right_sample is not None:
                left_chars = data_chars_for_join(
                    left_sample, self._dataset.total_rows,
                    on_list, self._memory_guard,
                )
                right_chars = data_chars_for_join(
                    right_sample, right._dataset.total_rows,
                    on_list, self._memory_guard,
                )
                # v0.9.23i-fix2: JoinMemoryModel.solve supports only
                # 'inner'|'left'. Map anti/semi/outer -> 'left' for
                # planning (planner build=right, probe=left; same class).
                # Actual execution dispatches to anti_join_streaming /
                # hash_join_streaming(how='left') appropriately.
                _planner_how = (
                    "inner" if how == "inner" else "left"
                )
                params = self._memory_guard.plan(
                    "join", left_chars,
                    op_context={
                        "right_chars": right_chars,
                        "how": _planner_how,
                    },
                )
                if num_partitions is None:
                    num_partitions = params.num_partitions
                max_output_rows = params.max_output_rows

                # v0.8.10-pre (P-A3 addendum): whole-operation RSS WARNING.
                # Per-batch check in _process_partition_pair enforces
                # per-batch memory, but does NOT see Arrow pool init +
                # hash-table build (~40 MB floor, measured across 3 runs
                # @ N=1M: 114.7 / 142.8 / 156.8 MB). Warn if worst-case
                # whole-operation RSS may exceed state_limit. Do not raise
                # (would break existing tests with 20 MB budget — bumping
                # budgets to 100 MB is v0.8.11 work).
                from cadw.ops.join import (
                    _JOIN_FIXED_OVERHEAD_BYTES,
                    _JOIN_STRICT_BUDGET_THRESHOLD_BYTES,
                )
                # v0.9.23i-fix2: predicted_output per how.
                # anti/semi: ≤ left_n (per-probe-batch guard sufficient,
                # whole-op overhead only — 40 MB fixed).
                # outer: left_join (left_n*avg_matches) + anti_right
                # (≤ right_n) rows.
                # inner/left: left_n*avg_matches (unchanged).
                if how in ("anti", "semi"):
                    predicted_output = 0
                elif how == "outer":
                    predicted_output = (
                        left_chars.n_rows
                        * params.avg_matches
                        * params.output_row_size
                        + right_chars.n_rows * params.output_row_size
                    )
                else:
                    predicted_output = (
                        left_chars.n_rows
                        * params.avg_matches
                        * params.output_row_size
                    )
                total_rss_estimate = (
                    _JOIN_FIXED_OVERHEAD_BYTES + 5 * predicted_output
                )
                # v0.8.11b: strict when budget is large enough that the
                # fixed 40 MB overhead is meaningful. Threshold:
                # budget >= 4×overhead (160 MB) AND
                # state_limit >= overhead (available >= 2×overhead).
                # Below: warn only — raise would fire trivially for
                # tiny joins (40 MB overhead >> state_limit at small
                # budgets). See §9.2.6.
                state_limit = self._memory_guard.state_limit_bytes
                budget = self._memory_guard.budget_bytes
                strict_mode = (
                    budget >= _JOIN_STRICT_BUDGET_THRESHOLD_BYTES
                    and state_limit >= _JOIN_FIXED_OVERHEAD_BYTES
                )
                if strict_mode:
                    self._memory_guard.check_state(
                        total_rss_estimate,
                        name="join.whole_operation_rss",
                    )
                elif total_rss_estimate > state_limit:
                    logger.warning(
                        "merge: predicted whole-operation RSS delta "
                        "~%.0f MiB (output %.1f MiB + %.0f MiB Arrow "
                        "pool) exceeds state_limit (%.0f MiB) but "
                        "budget %.0f MiB < %.0f MiB strict threshold. "
                        "Per-batch guards still enforce; actual peak "
                        "may exceed budget. Bump MemoryGuard budget "
                        "to enable strict enforcement. See §9.2.6.",
                        total_rss_estimate / (1024 * 1024),
                        predicted_output / (1024 * 1024),
                        _JOIN_FIXED_OVERHEAD_BYTES / (1024 * 1024),
                        state_limit / (1024 * 1024),
                        budget / (1024 * 1024),
                        _JOIN_STRICT_BUDGET_THRESHOLD_BYTES / (1024 * 1024),
                    )
            else:
                # Одна из сторон пуста — planner не нужен; hash_join
                # короткозамкнёт. num_partitions не важен.
                if num_partitions is None:
                    num_partitions = 8
        else:
            if num_partitions is None:
                num_partitions = 8

        # v0.5.1 (P2): use streaming path — output written to disk
        # batch-by-batch, never accumulated in RAM.
        # v0.9.23i: dispatch outer/anti/semi via dedicated functions.
        # v0.9.23i-fix2: only anti (semi in v0.9.23j; outer via
        # left+anti composition inside dispatch, no direct call).
        from cadw.ops.join import anti_join_streaming as _anti
        if how == "outer":
            # v0.9.23i: outer = left_join(left, right) ∪ anti_join(
            # right, left)_padded. Two passes (right is re-iterable via
            # right._iter_filtered_batches()).
            left_schema_out, left_path = hash_join_streaming(
                self._iter_filtered_batches(),
                right._iter_filtered_batches(),
                left_schema, right_schema, on=on_list,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / "join" / "outer_left",
                how="left", suffixes=suffixes,
                num_partitions=num_partitions,
                max_output_rows=max_output_rows,
            )
            anti_schema, anti_path = _anti(
                right._iter_filtered_batches(),
                self._iter_filtered_batches(),
                right_schema, left_schema, on=on_list,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / "join" / "outer_anti",
                num_partitions=num_partitions,
            )
            # Build final schema (left_join schema — superset of anti-
            # right padded). Write concat: left_path rows as-is +
            # anti_path rows with null left cols + right values.
            out_schema = left_schema_out
            out_path = self._spill_dir / "join" / "outer_output.arrow"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            from cadw.io.batch_writer import ArrowIPCBatchWriter as _W
            writer = _W(str(out_path), out_schema)
            on_set = set(on_list)
            overlap = {
                c for c in left_schema.names if c not in on_set
            } & {
                c for c in right_schema.names if c not in on_set
            }
            try:
                # Left rows (from left join)
                for b in ArrowIPCBatchReader(str(left_path), None):
                    if b.num_rows > 0:
                        writer.write_batch(b)
                # Anti-right rows padded with null left cols
                for b in ArrowIPCBatchReader(str(anti_path), None):
                    n = b.num_rows
                    if n == 0:
                        continue
                    arrays = []
                    for c in on_list:
                        arrays.append(b.column(c))
                    for f in left_schema:
                        if f.name in on_set:
                            continue
                        nm = (
                            f.name + suffixes[0]
                            if f.name in overlap else f.name
                        )
                        arrays.append(pa.nulls(n, type=f.type))
                    for f in right_schema:
                        if f.name in on_set:
                            continue
                        nm = (
                            f.name + suffixes[1]
                            if f.name in overlap else f.name
                        )
                        arrays.append(b.column(f.name))
                    writer.write_batch(pa.RecordBatch.from_arrays(
                        arrays, schema=out_schema,
                    ))
            finally:
                writer.close()
                left_path.unlink(missing_ok=True)
                anti_path.unlink(missing_ok=True)
        elif how == "anti":
            # v0.9.23i: anti join = left rows without match in right.
            out_schema, out_path = _anti(
                self._iter_filtered_batches(),
                right._iter_filtered_batches(),
                left_schema, right_schema, on=on_list,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / "join" / "anti",
                num_partitions=num_partitions,
            )
        elif how == "semi":
            # v0.9.23j: semi join = left rows with match in right.
            from cadw.ops.join import semi_join_streaming as _semi
            out_schema, out_path = _semi(
                self._iter_filtered_batches(),
                right._iter_filtered_batches(),
                left_schema, right_schema, on=on_list,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / "join" / "semi",
                num_partitions=num_partitions,
            )
        else:
            out_schema, out_path = hash_join_streaming(
                self._iter_filtered_batches(),
                right._iter_filtered_batches(),
                left_schema,
                right_schema,
                on=on_list,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / "join",
                how=how,
                suffixes=suffixes,
                num_partitions=num_partitions,
                max_output_rows=max_output_rows,
            )
        try:
            new_ds = Dataset.from_batches(
                out_schema,
                ArrowIPCBatchReader(str(out_path), None),
                self._new_dataset_dir(),
            )
        finally:
            out_path.unlink(missing_ok=True)
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

