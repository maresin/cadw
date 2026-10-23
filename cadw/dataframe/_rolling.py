# cadw/dataframe/_rolling.py
# v0.9.26q (B.1 step 11): _RollingMixin.
# Symbols: rolling, _rolling_on, _per_group_rolling,
#   expanding, ewm.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import pyarrow as pa
from cadw.dataframe_helpers import (
    _EwmState,
    _ExpandingState,
    _RollState,
    _group_boundaries_arrow,
)
from cadw.dataset import Dataset
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.sort import external_sort


class _RollingMixin:
    """Rolling/expanding/ewm mixin (v0.9.26q)."""
    def rolling(
        self,
        window: int,
        agg: str,
        *,
        min_periods: "int | None" = None,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Rolling window aggregation (v0.6.2c; sort-aware v0.8.4; multi v0.8.6).

        - window: int >= 1.
        - agg: {'sum', 'mean', 'min', 'max', 'count'}.
        - min_periods=None -> window.
        - columns=None -> all numeric columns.
        - on: positional if None; sort-aware if str/Sequence
          (`on=["ts", "sensor"]` — lexicographic composite sort, NOT
          per-group; for per-group see v0.8.5a).

        Output: original + `{col}_rolling_{agg}` (APPEND).
        Nulls skip min/max/sum/mean but **consume window slot** (positional,
        matches pandas skipna=True). Peak O(window × #cols).
        """
        from ._core import CadwDataFrame
        from collections import deque

        if not isinstance(window, int):
            raise TypeError(
                f"rolling: window must be int; got {type(window).__name__}"
            )
        if window < 1:
            raise ValueError(f"rolling: window must be >= 1; got {window}")
        if agg not in {"sum", "mean", "min", "max", "count"}:
            raise ValueError(
                f"rolling: agg must be one of sum/mean/min/max/count; "
                f"got {agg!r}"
            )
        if min_periods is None:
            mp = window
        elif not isinstance(min_periods, int):
            raise TypeError(
                f"rolling: min_periods must be int; "
                f"got {type(min_periods).__name__}"
            )
        elif not (1 <= min_periods <= window):
            raise ValueError(
                f"rolling: min_periods must be in [1, window={window}]; "
                f"got {min_periods}"
            )
        else:
            mp = min_periods

        # v0.8.4: sort-aware; v0.8.6: multi-column on.
        if on is not None:
            on_cols = self._normalize_on(on, fn_name="rolling")
            return self._rolling_on(
                window=window, agg=agg, min_periods=mp,
                columns=columns, on_cols=on_cols,
            )

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"rolling: column {c!r} not in {self._columns}"
                )

        # Validate numeric for sum/mean/min/max (count works for any type).
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"rolling: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        # Output schema: original cols + rolling output per (col, agg).
        out_fields = []
        for c in self._columns:
            out_fields.append(self._dataset.schema.field(c))
        out_type_by_col: dict[str, pa.DataType] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:  # min/max
                ot = in_type
            out_name = f"{c}_rolling_{agg}"
            out_type_by_col[c] = ot
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter_rolling():
            states = {c: _RollState(window, mp, agg) for c in cols}
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="rolling.input", extra_factor=3.5)
                n = b.num_rows
                out_arrays = []
                for c in self._columns:
                    out_arrays.append(b.column(c))
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    rolling_vals = []
                    for v in src_vals:
                        rv = state.push(v)
                        rolling_vals.append(rv)
                    out_arrays.append(
                        pa.array(rolling_vals, type=out_type_by_col[c])
                    )
                yield pa.RecordBatch.from_arrays(
                    out_arrays, schema=out_schema
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_rolling(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def _rolling_on(
        self,
        *,
        window: int,
        agg: str,
        min_periods: int,
        columns: "str | Sequence[str] | None",
        on_cols: list[str],
    ) -> "CadwDataFrame":
        """Sort-aware rolling on `on_cols` (v0.8.4; multi v0.8.6).

        Validated by prototype (scripts/proto_rolling_on.py):
          A (sort_values+rolling+sort_values) ==
          B (external_sort + _RollState + external_sort) ==
          C (pure Python deque).

        Algorithm (identical to v0.8.4, multi-key `on_cols`):
          1. external_sort by [on_cols...] (row_id tie-break from heap).
          2. _RollState.push per row (per column) → IPC
             (row_id, {col}_rolling_{agg}).
          3. external_sort by __orig_rid (§13.4/122).
          4. Zip original + rolling cols (APPEND).

        Peak RAM: batch + O(window × #cols). Independent of N.
        `on_cols` pre-normalized by caller (v0.8.6).
        """
        from ._core import CadwDataFrame
        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"rolling: column {c!r} not in {self._columns}"
                )

        # Validate numeric (agg != count requires numeric; same v0.6.2c)
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"rolling: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        # Output schema: original + {col}_rolling_{agg} (APPEND semantics).
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_type_by_col: dict[str, tuple[str, pa.DataType]] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:  # min/max
                ot = in_type
            out_name = f"{c}_rolling_{agg}"
            out_type_by_col[c] = (out_name, ot)
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)
        rolling_out_cols = [out_type_by_col[c][0] for c in cols]

        # 1. Sort by `on` (row_id tie-break from external_sort's heap).
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=on_cols,
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "rolling" / "sort_by_on",
            ascending=True,
            log=False,
        )

        # 2. Roll on sorted stream, write (row_id, {col}_rolling_{agg}).
        rolling_schema_fields: list[pa.Field] = [
            pa.field("row_id", pa.uint64()),
        ]
        for c in cols:
            out_name, ot = out_type_by_col[c]
            rolling_schema_fields.append(
                pa.field(out_name, ot, nullable=True)
            )
        rolling_schema = pa.schema(rolling_schema_fields)

        rolling_path = self._spill_dir / "rolling" / "rolling_by_on.arrow"
        rolling_path.parent.mkdir(parents=True, exist_ok=True)

        writer = ArrowIPCBatchWriter(str(rolling_path), rolling_schema)
        states = {c: _RollState(window, min_periods, agg) for c in cols}
        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name="rolling_on.sorted_input"
                , extra_factor=7.0)
                rid_list = b.column("row_id").to_pylist()
                arrays: list[pa.Array] = [
                    pa.array(rid_list, type=pa.uint64()),
                ]
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    rolled = [state.push(v) for v in src_vals]
                    out_name, ot = out_type_by_col[c]
                    arrays.append(pa.array(rolled, type=ot))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        arrays, schema=rolling_schema,
                    )
                )
        finally:
            writer.close()

        # 3. Sort back by __orig_rid. external_sort appends its own
        # "row_id"; input already has "row_id" (original pos) → duplicate
        # → ArrowInvalid. Rename to "__orig_rid" (§13.4/122).
        def _iter_rolled():
            for b in ArrowIPCBatchReader(str(rolling_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        sorted_rowid_path = external_sort(
            _iter_rolled(),
            key_columns=["__orig_rid"],
            value_columns=rolling_out_cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "rolling" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        # 4. Zip original + rolling cols (APPEND; rolling doesn't replace).
        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            rolled_reader = ArrowIPCBatchReader(
                str(sorted_rowid_path), rolling_out_cols,
            )
            accum: list[pa.RecordBatch] = []
            accum_n = 0
            rolled_gen = iter(rolled_reader)
            for ob in orig_iter:
                target_n = ob.num_rows
                while accum_n < target_n:
                    try:
                        nb = next(rolled_gen)
                    except StopIteration:
                        raise RuntimeError(
                            "rolling: rolled stream exhausted before "
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
                take_batch = take.combine_chunks().to_batches()[0]
                arrays = [ob.column(c) for c in self._columns]
                for out_name in rolling_out_cols:
                    arrays.append(take_batch.column(out_name))
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


    def _per_group_rolling(
        self,
        group_cols: list[str],
        on_cols: list[str],
        window: int,
        agg: str,
        min_periods: "int | None",
        columns: "str | Sequence[str] | None",
    ) -> "CadwDataFrame":
        """Composite-sort per-group rolling — APPEND semantics."""
        from ._core import CadwDataFrame
        if not group_cols:
            raise ValueError("_per_group_rolling: group_cols must be non-empty")
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"_per_group_rolling: window must be int >= 1; got {window!r}"
            )
        if agg not in {"sum", "mean", "min", "max", "count"}:
            raise ValueError(
                f"_per_group_rolling: agg must be one of sum/mean/min/max/"
                f"count; got {agg!r}"
            )
        if min_periods is None:
            mp = window
        elif not isinstance(min_periods, int):
            raise TypeError(
                f"_per_group_rolling: min_periods must be int; "
                f"got {type(min_periods).__name__}"
            )
        elif not (1 <= min_periods <= window):
            raise ValueError(
                f"_per_group_rolling: min_periods must be in [1, window="
                f"{window}]; got {min_periods}"
            )
        else:
            mp = min_periods

        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_rolling: group column {gc!r} not in "
                    f"{self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_rolling: on column {oc!r} not in "
                    f"{self._columns}"
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
                    f"_per_group_rolling: group column {c!r} cannot be rolled"
                )
            if c not in self._columns:
                raise KeyError(
                    f"_per_group_rolling: column {c!r} not in {self._columns}"
                )
        if not cols:
            raise ValueError("_per_group_rolling: no value columns")
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"_per_group_rolling: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_type_by_col: dict[str, tuple[str, pa.DataType]] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:
                ot = in_type
            out_name = f"{c}_rolling_{agg}"
            out_type_by_col[c] = (out_name, ot)
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)
        rolling_out_cols = [out_type_by_col[c][0] for c in cols]

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
            spill_dir=self._spill_dir / "per_group_rolling" / "sort_by_keys",
            ascending=True,
            log=False,
        )

        applied_schema = pa.schema(
            [pa.field("row_id", pa.uint64())]
            + [
                pa.field(
                    out_type_by_col[c][0],
                    out_type_by_col[c][1],
                    nullable=True,
                )
                for c in cols
            ]
        )
        applied_path = self._spill_dir / "per_group_rolling" / "applied.arrow"
        applied_path.parent.mkdir(parents=True, exist_ok=True)

        writer = ArrowIPCBatchWriter(str(applied_path), applied_schema)
        states = {c: _RollState(window, mp, agg) for c in cols}
        _SENTINEL = object()
        prev_group = _SENTINEL
        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name="per_group_rolling.sorted"
                , extra_factor=7.0)
                n = b.num_rows
                if n == 0:
                    continue
                # v0.9.21 (§1.4): Arrow-native group boundaries in rolling
                # (same rationale as _per_group_shift).
                boundaries = _group_boundaries_arrow(b, group_cols)

                # v0.9.21-fix1: batch-level key via 1-row to_pylist.
                batch_first_key = tuple(
                    b.column(gc).slice(0, 1).to_pylist()[0]
                    for gc in group_cols
                )
                if (prev_group is _SENTINEL
                        or batch_first_key != prev_group):
                    if prev_group is not _SENTINEL:
                        for st in states.values():
                            st.reset()

                all_rolled_arrow = {c: [] for c in cols}
                for gi in range(len(boundaries) - 1):
                    start = boundaries[gi]
                    end = boundaries[gi + 1]
                    if gi > 0:
                        # New group within batch — reset state.
                        for st in states.values():
                            st.reset()
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_rolled = {c: [None] * group_len for c in cols}
                    for i in range(group_len):
                        for c in cols:
                            group_rolled[c][i] = states[c].push(gvals[c][i])
                    for c in cols:
                        out_name, ot = out_type_by_col[c]
                        all_rolled_arrow[c].append(
                            pa.array(group_rolled[c], type=ot)
                        )
                arrays = [b.column("row_id")]
                for c in cols:
                    arrays.append(pa.concat_arrays(all_rolled_arrow[c]))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
                # v0.9.21-fix1: track last row's key for next batch.
                prev_group = tuple(
                    b.column(gc).slice(n - 1, 1).to_pylist()[0]
                    for gc in group_cols
                )
        finally:
            writer.close()

        def _iter_rolled():
            for b in ArrowIPCBatchReader(str(applied_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        resorted_path = external_sort(
            _iter_rolled(),
            key_columns=["__orig_rid"],
            value_columns=rolling_out_cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_rolling" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            sorted_reader = ArrowIPCBatchReader(
                str(resorted_path), rolling_out_cols,
            )
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
                            "per_group_rolling: sorted stream exhausted "
                            "before original"
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
                arrays = [ob.column(c) for c in self._columns]
                for out_name in rolling_out_cols:
                    arrays.append(take_batch.column(out_name))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def expanding(
        self,
        agg: str,
        *,
        min_periods: int = 1,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Cumulative (growing) window aggregation (v0.8.0).

        Equivalent to `rolling(window=n_so_far, agg)` but O(1) state
        without deque. For min/max, cumulative min/max is monotone —
        no monotonic deque needed (unlike rolling).

        - agg ∈ {'sum', 'mean', 'min', 'max', 'count'}.
        - min_periods=1: first non-null row → non-null output.
        - columns=None: all numeric columns (count works on any type).
        - Nulls: skip; null row → null output.

        Output: original columns + `{col}_expanding_{agg}` per column.

        Not supported: `center`, `on=col` (sort-first via `.sort_values`).
        """
        from ._core import CadwDataFrame
        if agg not in {"sum", "mean", "min", "max", "count"}:
            raise ValueError(
                f"expanding: agg must be one of sum/mean/min/max/count; "
                f"got {agg!r}"
            )
        if min_periods < 1:
            raise ValueError(
                f"expanding: min_periods must be >= 1; got {min_periods}"
            )
        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"expanding: column {c!r} not in {self._columns}"
                )
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"expanding: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        out_fields = [
            self._dataset.schema.field(c) for c in self._columns
        ]
        out_type_by_col: dict[str, pa.DataType] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:  # min/max
                ot = in_type
            out_name = f"{c}_expanding_{agg}"
            out_type_by_col[c] = ot
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter_expanding():
            states = {c: _ExpandingState(agg, min_periods) for c in cols}
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="expanding.input", extra_factor=3.5)
                out_arrays = [b.column(c) for c in self._columns]
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    vals = [state.push(v) for v in src_vals]
                    out_arrays.append(
                        pa.array(vals, type=out_type_by_col[c])
                    )
                yield pa.RecordBatch.from_arrays(
                    out_arrays, schema=out_schema
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_expanding(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def ewm(
        self,
        *,
        alpha: "float | None" = None,
        span: "float | None" = None,
        halflife: "float | None" = None,
        com: "float | None" = None,
        adjust: bool = True,
        min_periods: int = 0,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Exponentially weighted mean (v0.8.0).

        Exactly one of `alpha`, `span`, `halflife`, `com` must be given.

        - alpha ∈ (0, 1]: direct smoothing factor.
        - span >= 1: alpha = 2 / (span + 1).
        - halflife > 0: alpha = 1 - exp(-ln2 / halflife).
        - com >= 0: alpha = 1 / (1 + com).

        - adjust=True (pandas default): y_t = s_num/s_den where
            s_num = (1-α)·s_num + x_t,  s_den = (1-α)·s_den + 1.
        - adjust=False: y_t = (1-α)·y_{t-1} + α·x_t,  y_0 = x_0.
        - min_periods: minimum non-null rows before non-null output.
        - columns=None: all numeric columns.

        Null semantics: null row → null output; state preserved (skip).
        This matches `ignore_na=True` in pandas. `ignore_na=False` is NOT
        implemented.

        Output: original columns + `{col}_ewm` per column.

        **Not supported**: `bias=True` correction for `adjust=False`
        (rarely used); `ignore_na=False`.
        """
        from ._core import CadwDataFrame
        # Exactly one parameter
        given = sum(p is not None for p in (alpha, span, halflife, com))
        if given != 1:
            raise ValueError(
                f"ewm: exactly one of alpha/span/halflife/com required; "
                f"got {given}"
            )
        if alpha is not None:
            if not (0.0 < alpha <= 1.0):
                raise ValueError(
                    f"ewm: alpha must be in (0, 1]; got {alpha}"
                )
            a = float(alpha)
        elif span is not None:
            if span < 1:
                raise ValueError(f"ewm: span must be >= 1; got {span}")
            a = 2.0 / (float(span) + 1.0)
        elif halflife is not None:
            if halflife <= 0:
                raise ValueError(
                    f"ewm: halflife must be > 0; got {halflife}"
                )
            import math
            a = 1.0 - math.exp(-math.log(2.0) / float(halflife))
        else:  # com
            if com < 0:
                raise ValueError(f"ewm: com must be >= 0; got {com}")
            a = 1.0 / (1.0 + float(com))

        if min_periods < 0:
            raise ValueError(
                f"ewm: min_periods must be >= 0; got {min_periods}"
            )
        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"ewm: column {c!r} not in {self._columns}"
                )
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"ewm: column {c!r} has type {t}; numeric required"
                )

        out_fields = [
            self._dataset.schema.field(c) for c in self._columns
        ]
        for c in cols:
            out_fields.append(
                pa.field(f"{c}_ewm", pa.float64(), nullable=True)
            )
        out_schema = pa.schema(out_fields)

        def _iter_ewm():
            states = {c: _EwmState(a, adjust, min_periods) for c in cols}
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="ewm.input", extra_factor=3.5)
                out_arrays = [b.column(c) for c in self._columns]
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    vals = [state.push(v) for v in src_vals]
                    out_arrays.append(
                        pa.array(vals, type=pa.float64())
                    )
                yield pa.RecordBatch.from_arrays(
                    out_arrays, schema=out_schema
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_ewm(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

