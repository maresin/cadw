# cadw/dataframe/_cumulative.py
# v0.9.26r (B.1 step 12): _CumulativeMixin.
# Symbols: cumsum, cumprod, cummax, cummin,
#   _cumulative_impl, _per_group_cumulative, rank, sample.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import uuid
import pyarrow as pa
from cadw.dataframe_helpers import (
    _ExpandingState,
    _group_boundaries_arrow,
    _reservoir_positions,
    _weighted_reservoir_positions,
)
from cadw.dataset import Dataset
from cadw.ops.rank import (
    VALID_METHODS,
    compute_ranks_to_disk,
)
from cadw.ops.sort import external_sort


class _CumulativeMixin:
    """Cumulative + rank + sample mixin (v0.9.26r)."""
    def cumsum(self, columns=None, *, on=None) -> "CadwDataFrame":
        """Cumulative sum. Global (row_id order) or sort-aware if on=col.

        v0.9.23e (AUDIT_FULL_v0922 item 7.A.6). APPEND `{col}_cumsum`.
        Nulls preserved (null row -> null output; state frozen, matches
        expanding semantics).
        """
        return self._cumulative_impl("sum", columns, on)


    def cumprod(self, columns=None, *, on=None) -> "CadwDataFrame":
        """Cumulative product. See `cumsum`. APPEND `{col}_cumprod`."""
        return self._cumulative_impl("prod", columns, on)


    def cummax(self, columns=None, *, on=None) -> "CadwDataFrame":
        """Cumulative max. See `cumsum`. APPEND `{col}_cummax`."""
        return self._cumulative_impl("max", columns, on)


    def cummin(self, columns=None, *, on=None) -> "CadwDataFrame":
        """Cumulative min. See `cumsum`. APPEND `{col}_cummin`."""
        return self._cumulative_impl("min", columns, on)


    def _cumulative_impl(
        self, op: str, columns, on,
    ) -> "CadwDataFrame":
        """Internal: cumulative sum/prod/max/min (global, not grouped).

        op ∈ {'sum', 'prod', 'max', 'min'}. Streaming per batch; on=None
        (row_id order) or on=col (sort-aware via external_sort + sort
        back). Output: APPEND `{col}_{op}` names to match expanding/ewm:
          sum -> `{col}_cumsum`, prod -> `{col}_cumprod`,
          max -> `{col}_cummax`, min -> `{col}_cummin`.
        """
        from ._core import CadwDataFrame
        import pyarrow.compute as _pc
        from cadw.io.batch_reader import ArrowIPCBatchReader as _IPCReader
        from cadw.io.batch_writer import ArrowIPCBatchWriter as _W
        from cadw.ops.sort import external_sort as _ext_sort

        _SUFFIX = {"sum": "cumsum", "prod": "cumprod",
                   "max": "cummax", "min": "cummin"}
        suffix = _SUFFIX[op]

        if columns is None:
            cols = [c for c in self._columns
                    if pa.types.is_integer(
                        self._dataset.schema.field(c).type)
                    or pa.types.is_floating(
                        self._dataset.schema.field(c).type)]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError(
                f"{suffix}: no numeric columns selected"
            )
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"{suffix}: column {c!r} not in {self._columns}"
                )
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"{suffix}: column {c!r} has type {t}; numeric required"
                )

        out_cols = {c: f"{c}_{suffix}" for c in cols}
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        for c in cols:
            # Same dtype as input (int -> int, float -> float) to
            # preserve identity; mean is the only op that promotes to
            # float64. cum* are all identity-preserving.
            in_t = self._dataset.schema.field(c).type
            out_fields.append(pa.field(out_cols[c], in_t, nullable=True))
        out_schema = pa.schema(out_fields)

        # Sort-aware path
        if on is not None:
            on_cols = self._normalize_on(on, fn_name=suffix)
            sorted_path = _ext_sort(
                self._iter_filtered_batches(),
                key_columns=on_cols,
                value_columns=list(self._columns),
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / suffix / "sort_by_on",
                ascending=True,
                log=False,
            )
            applied_path = (
                self._spill_dir / suffix / "cumulative.arrow"
            )
            applied_path.parent.mkdir(parents=True, exist_ok=True)
            applied_schema = pa.schema(
                [pa.field("row_id", pa.uint64())]
                + [out_schema.field(out_cols[c]) for c in cols]
            )
            writer = _W(str(applied_path), applied_schema)
            states = {c: _ExpandingState(op, 1) for c in cols}
            try:
                for b in _IPCReader(str(sorted_path), None):
                    self._memory_guard.check_batch(
                        b, name=f"{suffix}.sorted_input",
                        extra_factor=3.5,  # 1 pylist per column
                    )
                    rid_arr = b.column("row_id")
                    out_arrays: list = [rid_arr]
                    for c in cols:
                        vals = b.column(c).to_pylist()
                        state = states[c]
                        rolled = [state.push(v) for v in vals]
                        out_arrays.append(
                            pa.array(rolled, type=out_schema.field(
                                out_cols[c]).type)
                        )
                    writer.write_batch(pa.RecordBatch.from_arrays(
                        out_arrays, schema=applied_schema,
                    ))
            finally:
                writer.close()

            # Sort back by __orig_rid
            def _iter_applied():
                for b in _IPCReader(str(applied_path), None):
                    names = b.schema.names
                    rid_idx = names.index("row_id")
                    new_names = (
                        ["__orig_rid"]
                        + [n for i, n in enumerate(names) if i != rid_idx]
                    )
                    yield b.rename_columns(new_names)

            resorted_path = _ext_sort(
                _iter_applied(),
                key_columns=["__orig_rid"],
                value_columns=list(out_cols.values()),
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / suffix / "sort_by_rowid",
                ascending=True,
                log=False,
            )

            out_col_list = list(out_cols.values())
            batch_size = 10_000

            def _iter_zipped():
                orig_iter = self._iter_arrow_batches(
                    batch_size=batch_size)
                sorted_reader = _IPCReader(str(resorted_path), out_col_list)
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
                                f"{suffix}: sorted stream exhausted "
                                f"before original"
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
                    for oc in out_col_list:
                        arrays.append(take_batch.column(oc))
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

        # Streaming path (no on): global cumulative, row_id order
        def _iter_cumulative():
            states = {c: _ExpandingState(op, 1) for c in cols}
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name=f"{suffix}.input", extra_factor=3.5,
                )
                out_arrays = [b.column(c) for c in self._columns]
                for c in cols:
                    state = states[c]
                    vals = b.column(c).to_pylist()
                    cum_vals = [state.push(v) for v in vals]
                    out_arrays.append(
                        pa.array(cum_vals, type=out_schema.field(
                            out_cols[c]).type)
                    )
                yield pa.RecordBatch.from_arrays(
                    out_arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_cumulative(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def _per_group_cumulative(
        self,
        group_cols: list[str],
        on_cols: list[str],
        op: str,
        columns,
    ) -> "CadwDataFrame":
        """Composite-sort per-group cumulative (sum/prod/max/min).

        Algorithm (mirrors _per_group_shift, v0.9.23e):
          1. external_sort by [*group_cols, *on_cols] — contiguous groups
             (on_cols empty -> [*group_cols]).
          2. Stream sorted; single _ExpandingState per col; reset on
             group change.
          3. external_sort by __orig_rid.
          4. Zip: APPEND {col}_{op} (not replace, unlike shift).

        Peak RAM = batch + O(K_groups * n_cols * 8). Independent of N.
        """
        from ._core import CadwDataFrame
        import pyarrow.compute as _pc
        from cadw.io.batch_reader import ArrowIPCBatchReader as _IPCReader
        from cadw.io.batch_writer import ArrowIPCBatchWriter as _W
        from cadw.ops.sort import external_sort as _ext_sort

        _SUFFIX = {"sum": "cumsum", "prod": "cumprod",
                   "max": "cummax", "min": "cummin"}
        suffix = _SUFFIX[op]

        if not group_cols:
            raise ValueError(
                f"_per_group_cumulative: group_cols must be non-empty"
            )
        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_cumulative: group column {gc!r} not in "
                    f"{self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_cumulative: on column {oc!r} not in "
                    f"{self._columns}"
                )
        if columns is None:
            cols = [c for c in self._columns if c not in group_cols
                    and (pa.types.is_integer(
                        self._dataset.schema.field(c).type)
                        or pa.types.is_floating(
                            self._dataset.schema.field(c).type))]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError(
                f"_per_group_cumulative: no numeric columns selected"
            )
        for c in cols:
            if c in group_cols:
                raise ValueError(
                    f"_per_group_cumulative: group column {c!r} cannot "
                    f"be accumulated"
                )
            if c not in self._columns:
                raise KeyError(
                    f"_per_group_cumulative: column {c!r} not in "
                    f"{self._columns}"
                )
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"_per_group_cumulative: column {c!r} has type {t}; "
                    f"numeric required"
                )

        out_cols = {c: f"{c}_{suffix}" for c in cols}
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        for c in cols:
            in_t = self._dataset.schema.field(c).type
            out_fields.append(
                pa.field(out_cols[c], in_t, nullable=True)
            )
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        sort_keys = list(group_cols) + list(on_cols)
        sorted_path = _ext_sort(
            self._iter_filtered_batches(),
            key_columns=sort_keys,
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / suffix / "sort_by_keys",
            ascending=True,
            log=False,
        )

        applied_schema = pa.schema(
            [pa.field("row_id", pa.uint64())]
            + [out_schema.field(out_cols[c]) for c in cols]
        )
        applied_path = self._spill_dir / suffix / "cumulative.arrow"
        applied_path.parent.mkdir(parents=True, exist_ok=True)

        writer = _W(str(applied_path), applied_schema)
        states = {c: _ExpandingState(op, 1) for c in cols}
        _SENTINEL = object()
        prev_group = _SENTINEL
        try:
            for b in _IPCReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name=f"per_group_{suffix}.sorted",
                    extra_factor=3.5,
                )
                n = b.num_rows
                if n == 0:
                    continue
                boundaries = _group_boundaries_arrow(b, group_cols)
                batch_first_key = tuple(
                    b.column(gc).slice(0, 1).to_pylist()[0]
                    for gc in group_cols
                )
                if (prev_group is _SENTINEL
                        or batch_first_key != prev_group):
                    if prev_group is not _SENTINEL:
                        for st in states.values():
                            st.reset()
                all_cum_arrow = {c: [] for c in cols}
                for gi in range(len(boundaries) - 1):
                    start = boundaries[gi]
                    end = boundaries[gi + 1]
                    if gi > 0:
                        for st in states.values():
                            st.reset()
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_cum = {c: [None] * group_len for c in cols}
                    for i in range(group_len):
                        for c in cols:
                            group_cum[c][i] = states[c].push(gvals[c][i])
                    for c in cols:
                        all_cum_arrow[c].append(pa.array(
                            group_cum[c],
                            type=out_schema.field(out_cols[c]).type,
                        ))
                arrays = [b.column("row_id")]
                for c in cols:
                    arrays.append(pa.concat_arrays(all_cum_arrow[c]))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
                prev_group = tuple(
                    b.column(gc).slice(n - 1, 1).to_pylist()[0]
                    for gc in group_cols
                )
        finally:
            writer.close()

        def _iter_applied():
            for b in _IPCReader(str(applied_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        out_col_list = list(out_cols.values())
        resorted_path = _ext_sort(
            _iter_applied(),
            key_columns=["__orig_rid"],
            value_columns=out_col_list,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / suffix / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            sorted_reader = _IPCReader(str(resorted_path), out_col_list)
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
                            f"per_group_{suffix}: sorted stream "
                            f"exhausted before original"
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
                for oc in out_col_list:
                    arrays.append(take_batch.column(oc))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def rank(
        self,
        by: str,
        method: str = "first",
        ascending: bool = True,
        pct: bool = False,
        *,
        output_col: str | None = None,
        batch_size: int = 10_000,
    ) -> "CadwDataFrame":
        """Ранг по колонке `by` (v0.3.1).

        - method ∈ {'first', 'min', 'dense', 'max', 'last', 'average'}:
            - 'first':   1-based позиция в sorted order (ties по row_id asc).
            - 'min':     минимум из рангов в tie-группе.
            - 'dense':   1, 2, 3, ... для уникальных значений.
            - 'max':     максимум из рангов в tie-группе.
            - 'last':    reverse of 'first' (CADW extension, НЕ в pandas).
            - 'average': (min + max) / 2 (float).
        - ascending: True — меньшие значения получают меньший ранг.
        - pct: делить rank на total_n.
        - output_col: имя новой колонки (default `{by}_rank`).

        Возвращает новый CadwDataFrame со всеми исходными колонками +
        rank в **исходном порядке** строк.

        Null values участвуют как значения (null == null). pandas skipna
        не поддерживается в v0.3.1.

        Методы 'max'/'last'/'average' реализованы двухпроходным
        streaming (O(1) RAM): pass 1 → group file, pass 2 → join.
        """
        from ._core import CadwDataFrame
        if method not in VALID_METHODS:
            raise NotImplementedError(
                f"rank: method={method!r} not supported. "
                f"Supported: {VALID_METHODS}."
            )
        if by not in self._columns:
            raise KeyError(f"rank: by={by!r} not in columns {self._columns}")
        if output_col is None:
            output_col = f"{by}_rank"
        if output_col in self._columns:
            raise KeyError(
                f"rank: output_col={output_col!r} already in columns"
            )

        # Count filtered rows (needed for pct, and sanity).
        total_n = self.count()
        if total_n == 0:
            # Empty: append an empty rank column of proper dtype.
            rank_dtype = pa.float64() if (pct or method == "average") else pa.int64()
            out_fields = [
                self._dataset.schema.field(c) for c in self._columns
            ]
            out_fields.append(pa.field(output_col, rank_dtype))
            out_schema = pa.schema(out_fields)

            def _iter_empty():
                for b in self._iter_arrow_batches(batch_size=batch_size):
                    n = b.num_rows
                    if pct:
                        col = pa.array([], type=pa.float64())
                    else:
                        col = pa.array([], type=pa.int64())
                    yield b.append_column(output_col, col)

            new_ds = Dataset.from_batches(
                out_schema, _iter_empty(), self._new_dataset_dir()
            )
            return CadwDataFrame(
                new_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        spill = self._spill_dir / "rank" / f"r_{uuid.uuid4().hex[:8]}"
        spill.mkdir(parents=True, exist_ok=True)

        # Step 1: sort by `by`.
        sorted_by_val = external_sort(
            self._iter_filtered_batches(),
            key_columns=[by],
            value_columns=[by],
            memory_guard=self._memory_guard,
            spill_dir=spill / "sort_by_val",
            ascending=ascending,
            log=False,
        )

        # Step 2: rank per row → (row_id, rank) on disk.
        # compute_ranks_to_disk принимает Path (не iterable) — bufferable
        # методы требуют двух проходов по sorted_by.
        from cadw.io.batch_reader import ArrowIPCBatchReader as _IPCReader
        rank_pairs_path = compute_ranks_to_disk(
            sorted_by_val,
            by=by,
            output_col=output_col,
            method=method,
            pct=pct,
            total_n=total_n,
            rank_path=spill / "ranks.arrow",
            memory_guard=self._memory_guard,
            log=False,
        )

        # Step 3: sort rank pairs by original row_id.
        def _iter_rank_pairs():
            for b in _IPCReader(str(rank_pairs_path), None):
                # columns: row_id, output_col → rename row_id to __orig_rid
                yield b.rename_columns(["__orig_rid", output_col])

        sorted_by_rowid = external_sort(
            _iter_rank_pairs(),
            key_columns=["__orig_rid"],
            value_columns=[output_col],
            memory_guard=self._memory_guard,
            spill_dir=spill / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        # Step 4: build rank Dataset (in original order via __orig_rid).
        rank_dtype = pa.float64() if (pct or method == "average") else pa.int64()
        rank_schema = pa.schema([
            ("__orig_rid", pa.uint64()),
            (output_col, rank_dtype),
        ])

        def _iter_rank_after_sort():
            for b in _IPCReader(str(sorted_by_rowid), ["__orig_rid", output_col]):
                yield b

        rank_ds = Dataset.from_batches(
            rank_schema, _iter_rank_after_sort(), spill / "rank_ds"
        )

        # Step 5: zip with original (both in original order; same batch_size).
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field(output_col, rank_dtype))
        out_schema = pa.schema(out_fields)

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            rank_iter = rank_ds.iter_batches(
                batch_size=batch_size, columns=[output_col]
            )
            for ob, rb in zip(orig_iter, rank_iter):
                if ob.num_rows != rb.num_rows:
                    raise RuntimeError(
                        f"rank: batch-size mismatch: orig={ob.num_rows} "
                        f"rank={rb.num_rows}"
                    )
                yield ob.append_column(output_col, rb.column(output_col))

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def sample(
        self,
        n=None,
        frac=None,
        replace: bool = False,
        weights: "str | None" = None,
        random_state=None,
    ) -> "CadwDataFrame":
        """Random sample of rows. Streaming, bounded memory.

        v0.9.23f (AUDIT_FULL_v0922 item 7.A.2, Top-10 gap #1).
        Prototype validated (stratified_sample, RAM 1.01 MB @ 32 MB).

        Args:
            n: number of rows (int >= 0). Exactly one of n / frac.
            frac: fraction of total rows [0, 1]. n = max(1,
                round(N*frac)) for frac > 0; frac=0 -> empty.
            replace: sample with replacement. Default False. n can
                exceed N only if replace=True.
            weights: name of a numeric weight column. Sampling is
                proportional to weight (Efraimidis-Spirakis A-Res).
                Not supported with replace=True (NotImplementedError).
                Non-positive / null weights are skipped.
            random_state: int seed; None -> 42 (deterministic).

        Returns:
            CadwDataFrame, rows in sample order (random).

        Raises:
            ValueError: exactly one of n/frac not given; n > N with
                replace=False; frac out of [0, 1]; n < 0; both n and
                frac; weights all-zero/null.
            NotImplementedError: weights + replace=True.
            KeyError: weights column missing.
            TypeError: weights column non-numeric.
            MemoryLimitExceeded: 4*n*row_bytes > state_limit.
        """
        from ._core import CadwDataFrame
        import numpy as np
        from cadw.core.memory_guard import MemoryLimitExceeded as _MLE
        from cadw.dataset import Dataset as _DS

        # ---- validation ----
        if (n is None) == (frac is None):
            raise ValueError(
                "sample: exactly one of n or frac must be provided"
            )
        if frac is not None:
            # v0.9.23f-fix2: [0, 1] (pandas-parity; frac=0 -> empty).
            if not (0.0 <= float(frac) <= 1.0):
                raise ValueError(
                    f"sample: frac must be in [0, 1]; got {frac!r}"
                )
        if not isinstance(replace, bool):
            raise TypeError(
                f"sample: replace must be bool; got {type(replace).__name__}"
            )

        N = self.count()
        if N == 0:
            # Empty view
            schema = pa.schema(
                [self._dataset.schema.field(c) for c in self._columns]
            )
            empty_ds = _DS.from_batches(
                schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        if frac is not None:
            if float(frac) == 0.0:
                n = 0
            else:
                n = max(1, int(round(N * float(frac))))

        if not isinstance(n, int):
            raise TypeError(
                f"sample: n must be int; got {type(n).__name__}"
            )
        if n < 0:
            raise ValueError(f"sample: n must be >= 0; got {n}")

        if weights is not None and replace:
            raise NotImplementedError(
                "sample: weights + replace=True is not supported "
                "(would need O(N) full-weight materialization). "
                "Use replace=False with weights, or use unweighted "
                "sampling with replacement."
            )

        if weights is not None:
            if weights not in self._columns:
                raise KeyError(
                    f"sample: weights column {weights!r} not in "
                    f"{self._columns}"
                )
            t = self._dataset.schema.field(weights).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"sample: weights column {weights!r} has type "
                    f"{t}; numeric required"
                )

        if n == 0:
            schema = pa.schema(
                [self._dataset.schema.field(c) for c in self._columns]
            )
            empty_ds = _DS.from_batches(
                schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        if n > N and not replace:
            raise ValueError(
                f"sample: n={n} > N={N} with replace=False. "
                f"Set replace=True or reduce n."
            )

        seed = 42 if random_state is None else int(random_state)
        rng = np.random.default_rng(seed)

        # ---- memory guard: 4*n*row_bytes for iloc materialize ----
        from cadw.core.sampling import peek_first_batch
        first_batch = peek_first_batch(self._iter_filtered_batches())
        if first_batch is not None:
            row_bytes_est = self._memory_guard.reestimate_from_batch(
                first_batch,
            )
        else:
            row_bytes_est = max(
                1,
                self._memory_guard.estimate_row_size_from_schema(
                    pa.schema(
                        [self._dataset.schema.field(c)
                         for c in self._columns]
                    )
                ),
            )
        self._memory_guard.check_state(
            4 * n * row_bytes_est,
            name=f"sample(n={n})",
        )

        # ---- replace=True: vectorized positions ----
        if replace:
            positions = rng.integers(0, N, size=n).tolist()
            return self.iloc[positions]

        # ---- n == N without replacement: shuffle ----
        if n == N:
            perm = rng.permutation(N).tolist()
            return self.iloc[perm]

        # ---- n < N without replacement ----
        if weights is None:
            positions = _reservoir_positions(
                self._iter_filtered_batches(), n, N, rng,
                self._memory_guard,
            )
        else:
            positions = _weighted_reservoir_positions(
                self._iter_filtered_batches(), n, weights, rng,
                self._memory_guard,
            )

        return self.iloc[positions]

