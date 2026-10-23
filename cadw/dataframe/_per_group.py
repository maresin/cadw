# cadw/dataframe/_per_group.py
# v0.9.26s (B.1 step 13): _PerGroupMixin.
# Symbols: _shift_on, _diff_on, _per_group_shift,
#   _per_group_diff, _per_group_apply.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import pyarrow as pa
from cadw.dataframe_helpers import (
    _ArrowTail,
    _group_boundaries_arrow,
)
from cadw.dataset import Dataset
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.sort import external_sort


class _PerGroupMixin:
    """Per-group + sort-aware column ops (v0.9.26s)."""
    def _shift_on(
        self,
        k: int,
        fill_value,
        *,
        columns: "str | Sequence[str] | None",
        on_cols: list[str],
    ) -> "CadwDataFrame":
        """Sort-aware shift on `on_cols` (v0.8.2, multi v0.8.6).

        Algorithm:
          1. external_sort by [on_cols...] (row_id tie-break from heap)
             → sorted IPC. Lexicographic composite sort.
          2. Read sorted stream, apply stateful shift (deque per column,
             O(k) memory). Output (row_id, shifted_cols) → IPC.
          3. external_sort by __orig_rid → original order.
          4. Zip with original.

        Peak RAM: batch + O(k × #cols). Independent of N.
        `on_cols` pre-normalized by caller (v0.8.6).
        """
        from ._core import CadwDataFrame
        from collections import deque

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"shift: column {c!r} not in {self._columns}"
                )
        shift_cols = set(cols)

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in shift_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        # 1. Sort by [on_cols...] (lexicographic)
        all_cols = list(self._columns)
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=on_cols,
            value_columns=all_cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "shift" / "sort_by_on",
            ascending=True,
            log=False,
        )

        # 2. Shift on sorted stream
        sorted_reader = ArrowIPCBatchReader(str(sorted_path), None)
        shift_schema = pa.schema(
            [pa.field("row_id", pa.uint64())]
            + [
                pa.field(c, self._dataset.schema.field(c).type, nullable=True)
                for c in cols
            ]
        )
        shift_path = (
            self._spill_dir / "shift" / "shifted_by_on.arrow"
        )
        shift_path.parent.mkdir(parents=True, exist_ok=True)

        def _fill(size: int, type_: pa.DataType) -> pa.Array:
            if fill_value is None:
                return pa.nulls(size, type=type_)
            return pa.array([fill_value] * size, type=type_)

        writer = ArrowIPCBatchWriter(str(shift_path), shift_schema)
        # v0.9.18 (AUDIT §1.1): Arrow-native tail buffer (same fix as shift).
        bufs: dict[str, _ArrowTail] = {c: _ArrowTail(k) for c in cols}
        try:
            for b in sorted_reader:
                self._memory_guard.check_batch(
                    b, name="shift_on.sorted_input"
                )
                n = b.num_rows
                arrays: list = [b.column("row_id")]
                for c in cols:
                    col_vals = b.column(c)
                    prev_arr = bufs[c].get()
                    prev_len = len(prev_arr) if prev_arr is not None else 0
                    fill_n = max(0, k - prev_len)
                    parts = []
                    if fill_n > 0:
                        parts.append(_fill(fill_n, col_vals.type))
                    if prev_arr is not None:
                        parts.append(prev_arr)
                    if n > k:
                        parts.append(col_vals.slice(0, n - k))
                    out = (
                        pa.concat_arrays(parts) if parts
                        else pa.array([], type=col_vals.type)
                    )
                    if len(out) > n:
                        out = out.slice(0, n)
                    elif len(out) < n:
                        out = pa.concat_arrays([
                            out,
                            pa.nulls(n - len(out), type=col_vals.type),
                        ])
                    arrays.append(out)
                    bufs[c].push(col_vals)
                rb = pa.RecordBatch.from_arrays(arrays, schema=shift_schema)
                writer.write_batch(rb)
        finally:
            writer.close()

        # 3. Sort back by __orig_rid.
        # external_sort ALWAYS appends its own "row_id" column. Our stream
        # already has "row_id" (= original position) → two row_ids would
        # trigger ArrowInvalid: Multiple matches for FieldRef.Name(row_id).
        # Rename "row_id" → "__orig_rid" so external_sort's appended row_id
        # is a different column. Sort by __orig_rid ascending = original order.
        def _iter_shifted():
            for b in ArrowIPCBatchReader(str(shift_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        sorted_rowid_path = external_sort(
            _iter_shifted(),
            key_columns=["__orig_rid"],
            value_columns=cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "shift" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        # 4. Zip (re-chunk to same batch_size)
        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            shifted_reader = ArrowIPCBatchReader(
                str(sorted_rowid_path), cols,
            )
            accum: list[pa.RecordBatch] = []
            accum_n = 0
            shifted_gen = iter(shifted_reader)
            for ob in orig_iter:
                target_n = ob.num_rows
                while accum_n < target_n:
                    try:
                        nb = next(shifted_gen)
                    except StopIteration:
                        raise RuntimeError(
                            "shift: shifted stream exhausted before "
                            "original (batch mismatch)"
                        )
                    accum.append(nb)
                    accum_n += nb.num_rows
                combined = pa.Table.from_batches(accum)
                take = combined.slice(0, target_n)
                rest = combined.slice(
                    target_n, combined.num_rows - target_n,
                )
                accum = list(rest.to_batches()) if rest.num_rows else []
                accum_n = combined.num_rows - target_n
                # v0.8.2-fix4: (a) REPLACE shifted columns, not append —
                # out_schema has self._columns (shift replaces column value,
                # doesn't add new column; matches v0.6.2b semantics);
                # (b) Table.column() returns ChunkedArray; combine_chunks()
                # + to_batches()[0] → RecordBatch with Array columns.
                take_batch = take.combine_chunks().to_batches()[0]
                arrays = []
                for c in self._columns:
                    if c in shift_cols:
                        arrays.append(take_batch.column(c))
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def _diff_on(
        self,
        *,
        k: int,
        columns: "str | Sequence[str] | None",
        on: "str | Sequence[str]",
    ) -> "CadwDataFrame":
        """Sort-aware diff: pc.subtract(col, shifted(col)) with on=col sort.

        v0.8.2 (single), v0.8.6 (multi). Delegates to _shift_on + subtract
        per batch. No extra sort needed (reuses shift's 2 sorts).
        """
        from ._core import CadwDataFrame
        # v0.8.6: validate on early (before numeric-type check on columns).
        on_cols = self._normalize_on(on, fn_name="diff")
        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"diff: column {c!r} not in {self._columns}"
                )
        # Validate numeric for diff
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"diff: column {c!r} has type {t}; numeric required"
                )
        diff_cols = set(cols)

        # Shift with fill_value=None (nulls for first k rows)
        shifted_df = self._shift_on(
            k, None, columns=cols, on_cols=on_cols,
        )

        # Zip original + shifted, subtract
        import pyarrow.compute as _pc
        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in diff_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        batch_size = 10_000

        def _iter_diffed():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            shifted_iter = shifted_df._iter_arrow_batches(
                batch_size=batch_size,
            )
            for ob, sb in zip(orig_iter, shifted_iter):
                if ob.num_rows != sb.num_rows:
                    raise RuntimeError(
                        f"diff_on: batch size mismatch "
                        f"(orig={ob.num_rows}, shifted={sb.num_rows})"
                    )
                arrays = []
                for c in self._columns:
                    if c in diff_cols:
                        arrays.append(
                            _pc.subtract(ob.column(c), sb.column(c))
                        )
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_diffed(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def _per_group_shift(
        self,
        group_cols: list[str],
        on_cols: list[str],
        k: int,
        fill_value,
        columns: "str | Sequence[str] | None",
    ) -> "CadwDataFrame":
        """Composite-sort per-group shift (validated by proto_per_group.py).

        Algorithm (11 cases, A==B):
          1. external_sort by [*group_cols, *on_cols] — contiguous groups.
          2. Stream sorted; single deque per value col; clear on group change.
          3. external_sort by __orig_rid.
          4. Zip: replace shifted cols, keep originals.

        Peak RAM = batch + O(k × #cols). Independent of N and G.
        """
        from ._core import CadwDataFrame
        from collections import deque

        if not group_cols:
            raise ValueError("_per_group_shift: group_cols must be non-empty")
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"_per_group_shift: k must be int >= 1; got {k!r}")
        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_shift: group column {gc!r} not in {self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_shift: on column {oc!r} not in {self._columns}"
                )

        if columns is None:
            cols = [c for c in self._columns if c not in group_cols]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c in group_cols:
                raise ValueError(
                    f"_per_group_shift: group column {c!r} cannot be shifted"
                )
            if c not in self._columns:
                raise KeyError(
                    f"_per_group_shift: column {c!r} not in {self._columns}"
                )
        if not cols:
            raise ValueError("_per_group_shift: no value columns to shift")
        shift_cols = set(cols)

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in shift_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir()
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        sort_keys = list(group_cols) + list(on_cols)
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=sort_keys,
            value_columns=self._columns,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_shift" / "sort_by_keys",
            ascending=True,
            log=False,
        )

        applied_schema = pa.schema(
            [pa.field("row_id", pa.uint64())]
            + [
                pa.field(c, self._dataset.schema.field(c).type, nullable=True)
                for c in cols
            ]
        )
        applied_path = self._spill_dir / "per_group_shift" / "applied.arrow"
        applied_path.parent.mkdir(parents=True, exist_ok=True)

        writer = ArrowIPCBatchWriter(str(applied_path), applied_schema)
        bufs = {c: deque(maxlen=k) for c in cols}
        _SENTINEL = object()
        prev_group = _SENTINEL
        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name="per_group_shift.sorted"
                , extra_factor=7.0)
                n = b.num_rows
                if n == 0:
                    continue
                # v0.9.21 (§1.4): Arrow-native group boundaries in shift.
                # Prior: group_arrs = {gc: b.column(gc).to_pylist()...}
                # + per-row tuple(group_arrs[gc][i]) = O(n × K_cols)
                # Python ops + n × K_cols Python objects (28 B each).
                # Now: _group_boundaries_arrow = O(n × K_cols) C++ ops
                # + O(n) transient bool masks. Peak Python per batch =
                # max_group_size × cols (transient, GC'd between
                # groups), not n × cols.
                boundaries = _group_boundaries_arrow(b, group_cols)

                # v0.9.21-fix1: batch-level key via 1-row to_pylist
                # (bounded #batches × K_cols, NOT per-group as_py).
                batch_first_key = tuple(
                    b.column(gc).slice(0, 1).to_pylist()[0]
                    for gc in group_cols
                )
                if (prev_group is _SENTINEL
                        or batch_first_key != prev_group):
                    if prev_group is not _SENTINEL:
                        for dq in bufs.values():
                            dq.clear()

                all_shifted_arrow = {c: [] for c in cols}
                for gi in range(len(boundaries) - 1):
                    start = boundaries[gi]
                    end = boundaries[gi + 1]
                    if gi > 0:
                        # New group within batch — reset state (no key
                        # tracking; batch-boundary key handles carryover).
                        for dq in bufs.values():
                            dq.clear()
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_shifted = {c: [None] * group_len for c in cols}
                    for i in range(group_len):
                        for c in cols:
                            if len(bufs[c]) < k:
                                group_shifted[c][i] = fill_value
                            else:
                                group_shifted[c][i] = bufs[c][0]
                            bufs[c].append(gvals[c][i])
                    for c in cols:
                        all_shifted_arrow[c].append(
                            pa.array(
                                group_shifted[c],
                                type=b.schema.field(c).type,
                            )
                        )
                arrays = [b.column("row_id")]
                for c in cols:
                    arrays.append(pa.concat_arrays(all_shifted_arrow[c]))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
                # v0.9.21-fix1: track last row's key for next batch's
                # cross-batch continuity (bounded 1-row to_pylist).
                prev_group = tuple(
                    b.column(gc).slice(n - 1, 1).to_pylist()[0]
                    for gc in group_cols
                )
        finally:
            writer.close()

        def _iter_applied():
            for b in ArrowIPCBatchReader(str(applied_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        resorted_path = external_sort(
            _iter_applied(),
            key_columns=["__orig_rid"],
            value_columns=cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_shift" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            sorted_reader = ArrowIPCBatchReader(str(resorted_path), cols)
            accum: list = []
            accum_n = 0
            gen = iter(sorted_reader)
            for ob in orig_iter:
                target_n = ob.num_rows
                while accum_n < target_n:
                    try:
                        nb = next(gen)
                    except StopIteration:
                        raise RuntimeError(
                            "per_group_shift: sorted stream exhausted before "
                            "original"
                        )
                    accum.append(nb)
                    accum_n += nb.num_rows
                combined = pa.Table.from_batches(accum)
                take = combined.slice(0, target_n)
                rest = combined.slice(
                    target_n, combined.num_rows - target_n,
                )
                accum = list(rest.to_batches()) if rest.num_rows else []
                accum_n = combined.num_rows - target_n
                take_batch = take.combine_chunks().to_batches()[0]
                arrays = []
                for c in self._columns:
                    if c in shift_cols:
                        arrays.append(take_batch.column(c))
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def _per_group_diff(
        self,
        group_cols: list[str],
        on_cols: list[str],
        k: int,
        columns: "str | Sequence[str] | None",
    ) -> "CadwDataFrame":
        """Per-group diff — delegate to shift then subtract."""
        from ._core import CadwDataFrame
        import pyarrow.compute as _pc

        if not isinstance(k, int) or k < 1:
            raise ValueError(f"_per_group_diff: k must be int >= 1; got {k!r}")

        if columns is None:
            cols = [c for c in self._columns if c not in group_cols]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c in group_cols:
                raise ValueError(
                    f"_per_group_diff: group column {c!r} cannot be diffed"
                )
            if c not in self._columns:
                raise KeyError(
                    f"_per_group_diff: column {c!r} not in {self._columns}"
                )
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"_per_group_diff: column {c!r} has type {t}; "
                    f"numeric required"
                )
        if not cols:
            raise ValueError("_per_group_diff: no value columns")

        diff_cols = set(cols)

        shifted_df = self._per_group_shift(
            group_cols, on_cols, k, None, cols,
        )

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in diff_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        batch_size = 10_000

        def _iter_diffed():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            shifted_iter = shifted_df._iter_arrow_batches(batch_size=batch_size)
            for ob, sb in zip(orig_iter, shifted_iter):
                if ob.num_rows != sb.num_rows:
                    raise RuntimeError(
                        f"_per_group_diff: batch size mismatch "
                        f"(orig={ob.num_rows}, shifted={sb.num_rows})"
                    )
                arrays = []
                for c in self._columns:
                    if c in diff_cols:
                        arrays.append(_pc.subtract(ob.column(c), sb.column(c)))
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_diffed(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def _per_group_apply(
        self,
        group_cols: list[str],
        on_cols: list[str],
        fn,
        result_col: str,
        memory_guard: "MemoryGuard",
    ) -> "CadwDataFrame":
        """Composite-sort apply(fn) on each group. v0.9.6b2 CADW-native.

        Args:
            group_cols: group keys. Non-empty.
            on_cols: within-group sort keys (empty → row_id order).
            fn: callable(group_df: CadwDataFrame) → scalar | dict[str,scalar]
                | CadwDataFrame. Uniform mode across groups.
            result_col: column name for scalar mode.
            memory_guard: per-group memory check before fn().

        Peak RAM = batch + 2 × max(group_size × row_size) + fn's output.
        Bounded per group, independent of N.

        Empty-input limitation (§11.5, v0.9.14 F12-7):
        when df.count() == 0, fn is never called; output
        schema = [group_cols..., result_col: float64]. This
        is only accurate for scalar mode. For dict mode
        (fn returns dict[str, scalar]) or DataFrame mode
        (fn returns CadwDataFrame) with empty input, the
        output is the flat result_col schema — not dict
        keys / df columns, since mode is unknown at
        empty-input time. Caller who needs accurate empty
        schema for dict/df mode: use a non-empty check or
        synthesize one row.

        No pandas. For multi-metric per-group output use a dict
        (`{"mean": ..., "max": ...}`); for multi-row output use
        `CadwDataFrame.from_table(pa_table)`.

        fn's own alloc is user responsibility (§3.9 extended).
        """
        from ._core import CadwDataFrame, _build_out_schema, _detect_mode, _normalize_scalar
        import pyarrow as pa
        from cadw.io.batch_reader import ArrowIPCBatchReader
        from cadw.io.batch_writer import ArrowIPCBatchWriter
        from cadw.core.memory_guard import MemoryLimitExceeded

        if not group_cols:
            raise ValueError(
                "_per_group_apply: group_cols must be non-empty"
            )
        if not callable(fn):
            raise TypeError(
                f"_per_group_apply: fn must be callable; "
                f"got {type(fn).__name__}"
            )
        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_apply: group column {gc!r} not in "
                    f"{self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_apply: on column {oc!r} not in "
                    f"{self._columns}"
                )
        if not isinstance(result_col, str) or not result_col:
            raise ValueError(
                f"_per_group_apply: result_col must be non-empty str; "
                f"got {result_col!r}"
            )

        if self.count() == 0:
            out_fields = [
                self._dataset.schema.field(c) for c in group_cols
            ]
            out_fields.append(pa.field(result_col, pa.float64(), nullable=True))
            out_schema = pa.schema(out_fields)
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        sort_keys = list(group_cols) + list(on_cols)
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=sort_keys,
            value_columns=self._columns,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_apply" / "sort_by_keys",
            ascending=True,
            log=False,
        )

        out_schema_box: list = []
        out_mode_box: list = []
        buf_rows: list = []
        BATCH_ROWS = 1000
        applied_path = (
            self._spill_dir / "per_group_apply" / "applied.arrow"
        )
        applied_path.parent.mkdir(parents=True, exist_ok=True)
        writer_box: list = []

        def _flush_buf():
            # v0.9.22 (§13.4/244, AUDIT_ML §1.7): buf_rows bounded
            # at BATCH_ROWS=1000 (see _emit_row) → ~140 KB at K=5.
            # Real guard is _flush_group at 0.3·available. No
            # check_batch needed — bounded by construction.
            nonlocal buf_rows
            if not buf_rows:
                return
            _maybe_upgrade_schema()  # v0.9.11 (PF4)
            if not writer_box:
                writer_box.append(
                    ArrowIPCBatchWriter(str(applied_path),
                                        out_schema_box[0])
                )
            # v0.9.14 (F12-6b): non-widenable type mismatch
            # (e.g. str field + numeric later) → clear error,
            # not cryptic ArrowInvalid from from_pylist.
            try:
                table = pa.Table.from_pylist(buf_rows,
                                             schema=out_schema_box[0])
            except (pa.ArrowInvalid, pa.ArrowTypeError) as _e:
                raise RuntimeError(
                    f"_per_group_apply: fn() returned "
                    f"inconsistent types across groups. "
                    f"Buffered schema={out_schema_box[0]}, "
                    f"error={_e}. fn() must return the "
                    f"same type for all groups. Hint: "
                    f"return float(x) from fn() if numeric "
                    f"types might mix (int/float/bool); "
                    f"return None for missing values. §11.5"
                ) from _e
            for b in table.to_batches():
                writer_box[0].write_batch(b)
            buf_rows = []

        def _maybe_upgrade_schema() -> None:
            """v0.9.11 (PF4): widen int->float64 on fractional float.

            Deferred from per-row (_maybe_upgrade_for_row) to
            once-per-flush. Per-row allocation of `upgraded` list
            was O(N*K) — for 10M groups x K=10, ~100M list allocs
            + ~1B attr checks. Now: one scan of buf_rows
            (<= BATCH_ROWS=1000 rows) per flush. ~1000x fewer allocs,
            ~10x fewer attr checks.

            Checks each integer field across all buf_rows; if any
            fractional float found -> widen to float64. Lossless
            for |v| < 2^53.

            Limitation (§11.5): only before first flush. Once writer
            committed (writer_box non-empty), post-flush widen raises
            clear RuntimeError (prior: silent truncation or cryptic
            ArrowInvalid at write_batch). Workaround: return
            float(x) from fn() if any group might be fractional.
            """
            if not out_schema_box:
                return
            schema = out_schema_box[0]
            upgraded = []
            needs = False
            widen_fields: list = []
            for field in schema:
                # v0.9.14 (F12-6a): bool field, any non-bool
                # numeric → widen to float64. Prior: pa.bool_
                # schema + later float → ArrowInvalid (cryptic).
                # Same class as C1 (int→float).
                if pa.types.is_boolean(field.type):
                    widen = False
                    for row in buf_rows:
                        val = row.get(field.name)
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            widen = True
                            break
                    if widen:
                        upgraded.append(pa.field(
                            field.name, pa.float64(),
                            nullable=field.nullable,
                        ))
                        needs = True
                        widen_fields.append(field.name)
                    else:
                        upgraded.append(field)
                    continue
                if not pa.types.is_integer(field.type):
                    upgraded.append(field)
                    continue
                widen = False
                for row in buf_rows:
                    val = row.get(field.name)
                    if isinstance(val, float) and val != int(val):
                        widen = True
                        break
                if widen:
                    upgraded.append(pa.field(
                        field.name, pa.float64(),
                        nullable=field.nullable,
                    ))
                    needs = True
                    widen_fields.append(field.name)
                else:
                    upgraded.append(field)
            if needs:
                if writer_box:
                    raise RuntimeError(
                        f"_per_group_apply: post-flush widen needed "
                        f"for {widen_fields}, but writer already "
                        f"committed. Workaround: return float(x) "
                        f"from fn() if any group might be fractional. "
                        f"See §11.5."
                    )
                out_schema_box[0] = pa.schema(upgraded)

        def _emit_row(row: dict) -> None:
            """Append row; widen check deferred to _flush_buf (PF4)."""
            nonlocal buf_rows
            buf_rows.append(row)

        def _emit_group(key: tuple, result):
            nonlocal buf_rows

            mode = _detect_mode(result)
            if not out_mode_box:
                out_mode_box.append(mode)
                out_schema_box.append(
                    _build_out_schema(group_cols, self, mode,
                                      result, result_col)
                )
            elif mode != out_mode_box[0]:
                raise RuntimeError(
                    f"_per_group_apply: mixed return modes across "
                    f"groups: first group returned {out_mode_box[0]!r}, "
                    f"group {key!r} returned {mode!r}. fn() must "
                    f"return same type for all groups."
                )

            if mode == "scalar":
                row = {gc: v for gc, v in zip(group_cols, key)}
                row[result_col] = _normalize_scalar(result)
                _emit_row(row)
            elif mode == "dict":
                row = {gc: v for gc, v in zip(group_cols, key)}
                for k, v in result.items():
                    row[str(k)] = _normalize_scalar(v)
                _emit_row(row)
            else:  # CadwDataFrame
                for b in result._iter_filtered_batches():
                    for rec in b.to_pylist():
                        row = dict(rec)
                        for gc, v in zip(group_cols, key):
                            row.setdefault(gc, v)
                        _emit_row(row)
            if len(buf_rows) >= BATCH_ROWS:
                _flush_buf()

        _UNSET = object()
        prev_key = _UNSET
        group_buf: list = []
        max_group_limit = int(
            memory_guard.available_bytes() * 0.3
        )

        def _flush_group():
            nonlocal group_buf
            if not group_buf:
                return
            est_pre = sum(b.nbytes for b in group_buf) * 2
            if est_pre > max_group_limit:
                raise MemoryLimitExceeded(
                    f"_per_group_apply: group size ~{est_pre // 2} B "
                    f"× 2 (materialize + overhead) = {est_pre} B > "
                    f"0.3 × avail = {max_group_limit} B. Split by "
                    f"more group cols or use groupby().agg() if fn "
                    f"reduces. See §11.5."
                )
            memory_guard.check_state(
                est_pre, name="per_group_apply.group",
            )
            gt = pa.Table.from_batches(group_buf)
            est_actual = gt.nbytes * 2
            if est_actual > max_group_limit:
                raise MemoryLimitExceeded(
                    f"_per_group_apply: actual group size "
                    f"{gt.nbytes} B × 2 = {est_actual} B > 0.3 × avail "
                    f"= {max_group_limit} B (pre-estimate was "
                    f"{est_pre} B; buffer overhead). See §11.5."
                )
            # v0.9.6b2: CADW-native group materialize (in-memory).
            gdf = CadwDataFrame.from_table(
                gt,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )
            result = fn(gdf)
            _emit_group(tuple(prev_key), result)
            group_buf.clear()

        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                memory_guard.check_batch(
                    b, name="per_group_apply.sorted"
                , extra_factor=7.0)
                n = b.num_rows
                if n == 0:
                    continue
                group_vals = {
                    gc: b.column(gc).to_pylist() for gc in group_cols
                }
                start = 0
                for i in range(1, n):
                    gk = tuple(group_vals[gc][i] for gc in group_cols)
                    prev_gk = tuple(
                        group_vals[gc][i - 1] for gc in group_cols
                    )
                    if gk != prev_gk:
                        gk_start = tuple(
                            group_vals[gc][start] for gc in group_cols
                        )
                        piece = b.slice(start, i - start)
                        if gk_start != prev_key:
                            if prev_key is not _UNSET:
                                _flush_group()
                            prev_key = gk_start
                        group_buf.append(piece)
                        start = i
                gk_start = tuple(
                    group_vals[gc][start] for gc in group_cols
                )
                piece = b.slice(start, n - start)
                if gk_start != prev_key:
                    if prev_key is not _UNSET:
                        _flush_group()
                    prev_key = gk_start
                group_buf.append(piece)

            _flush_group()
            _flush_buf()
        finally:
            if writer_box:
                writer_box[0].close()

        if not out_schema_box:
            raise RuntimeError(
                "_per_group_apply: no groups produced output — "
                "unreachable after empty check"
            )

        def _iter_applied():
            for b in ArrowIPCBatchReader(str(applied_path), None):
                yield b

        new_ds = Dataset.from_batches(
            out_schema_box[0], _iter_applied(),
            self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

