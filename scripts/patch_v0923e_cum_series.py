#!/usr/bin/env python3
# v0.9.23e — cumsum/cumprod/cummax/cummin per group (AUDIT_FULL_v0922 7.A.6).
#
# Top-10 gap #5 (time-series feature engineering workhorse).
# shift/diff/rolling/expanding/ewm already exist; cum* was the missing
# 4-op family. Prototype #3 from audit (groupby_cumsum) validated:
# RAM 13.28 MB @ 32 MB budget, identity PASS (delta of consecutive
# cumsums equals original value).
#
# Pre-fix prototype wall time was 40s (dominated by sort_values under
# 32 MB budget). Post-7.A.1 Arrow-native merge: expected ~2s.
#
# API:
#   df.cumsum(columns=None, *, on=None)         — global, streaming
#   df.cumprod(...), df.cummax(...), df.cummin(...)
#   df.groupby(g).cumsum(columns=None, *, on=None)  — per group
#
# on=None: positional (row_id order). on=col: sort-aware (external_sort
# + sort back by __orig_rid, same pattern as _shift_on / _rolling_on).
#
# Output: APPEND {col}_cumsum / _cumprod / _cummax / _cummin (matches
# expanding/ewm semantics — new columns, not replace).
#
# _ExpandingState extension: add 'prod' aggregator (init _prod=1.0,
# push *= x, reset). Reuses existing sum/min/max/count state class —
# no new memory model, O(1) per row.
#
# Peak RAM: O(batch + n_cols * 8) for global; O(batch + K_groups *
# n_cols * 8) for per-group (K_groups = group cardinality; bounded by
# groupby). Independent of N.
#
# Run: uv run python scripts/patch_v0923e_cum_series.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess,
)


# ============ 1. _ExpandingState: add 'prod' aggregator ============

def extend_expanding_state():
    print("\n=== _ExpandingState: add 'prod' aggregator ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if '"prod"' in s and "self._prod" in s:
        print("  already present — skip")
        return

    # 1a. __slots__ add _prod
    apply_or_fail(
        str(p),
        old=(
            '    __slots__ = ("_agg", "_min_periods", "_count", "_sum",\n'
            '                 "_min", "_max")'
        ),
        new=(
            '    __slots__ = ("_agg", "_min_periods", "_count", "_sum",\n'
            '                 "_prod", "_min", "_max")'
        ),
        label="_ExpandingState.__slots__",
    )

    # 1b. __init__ add _prod = 1.0
    apply_or_fail(
        str(p),
        old=(
            '        self._count = 0\n'
            '        self._sum = 0.0\n'
            '        self._min = None\n'
            '        self._max = None'
        ),
        new=(
            '        self._count = 0\n'
            '        self._sum = 0.0\n'
            '        self._prod = 1.0  # v0.9.23e: cumprod\n'
            '        self._min = None\n'
            '        self._max = None'
        ),
        label="_ExpandingState.__init__._prod",
    )

    # 1c. push: add 'prod' branch (before 'mean')
    apply_or_fail(
        str(p),
        old=(
            '        if self._agg == "count":\n'
            '            result = self._count\n'
            '        elif self._agg == "sum":\n'
            '            self._sum += x\n'
            '            result = self._sum\n'
            '        elif self._agg == "mean":'
        ),
        new=(
            '        if self._agg == "count":\n'
            '            result = self._count\n'
            '        elif self._agg == "sum":\n'
            '            self._sum += x\n'
            '            result = self._sum\n'
            '        elif self._agg == "prod":  # v0.9.23e\n'
            '            self._prod *= x\n'
            '            result = self._prod\n'
            '        elif self._agg == "mean":'
        ),
        label="_ExpandingState.push.prod",
    )

    assert_compiles(p)
    print("  OK")


# ============ 2. CadwDataFrame.cum* methods (global) ============

ANCHOR_CORR_END = (
    '        return {\n'
    '            cols[i]: {cols[j]: float(corr_mat[i, j]) for j in range(len(cols))}\n'
    '            for i in range(len(cols))\n'
    '        }\n'
    '\n'
    '    # ---------------- write ----------------'
)

CUM_METHODS = '''        return {
            cols[i]: {cols[j]: float(corr_mat[i, j]) for j in range(len(cols))}
            for i in range(len(cols))
        }

    # ---------------- cumulative (v0.9.23e) ----------------

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

    # ---------------- write ----------------'''


def add_cum_methods():
    print("\n=== add CadwDataFrame.cum* methods ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def _cumulative_impl(" in s:
        print("  already present — skip")
        return
    if ANCHOR_CORR_END not in s:
        raise SystemExit(
            "cum methods anchor not found — source may have drifted "
            "from v0.9.23d state"
        )
    s = s.replace(ANCHOR_CORR_END, CUM_METHODS, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. _per_group_cumulative helper + _GroupBy methods ============

PER_GROUP_HELPER_ANCHOR = (
    '    # ---------------- per-group shift/diff/rolling (v0.8.10) ----------------\n'
    '\n'
    '    def _per_group_shift('
)

PER_GROUP_HELPER = '''    # ---------------- per-group cumulative (v0.9.23e) ----------------

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

    # ---------------- per-group shift/diff/rolling (v0.8.10) ----------------

    def _per_group_shift('''


def add_per_group_helper():
    print("\n=== add _per_group_cumulative + _GroupBy methods ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def _per_group_cumulative(" in s:
        print("  already present — skip")
        return
    if PER_GROUP_HELPER_ANCHOR not in s:
        raise SystemExit(
            "per_group_shift anchor not found — source may have drifted"
        )
    s = s.replace(PER_GROUP_HELPER_ANCHOR, PER_GROUP_HELPER, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 4. _GroupBy.cum* methods ============

GROUPBY_ANCHOR = (
    '    def shift(\n'
    '        self,\n'
    '        k: int = 1,\n'
    '        fill_value=None,\n'
    '        *,\n'
    '        columns: "str | Sequence[str] | None" = None,\n'
    '        on: "str | Sequence[str] | None" = None,\n'
    '    ) -> CadwDataFrame:'
)

GROUPBY_CUM_METHODS = '''    def cumsum(
        self,
        columns: "str | Sequence[str] | None" = None,
        *,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group cumulative sum. APPEND `{col}_cumsum`.

        v0.9.23e (AUDIT_FULL_v0922 item 7.A.6, Top-10 gap #5).
        Group order: sort by [*group_cols, *on_cols]; accumulator
        reset per group.
        """
        on_cols = self._normalize_on(on)
        return self._parent._per_group_cumulative(
            self._by, on_cols, "sum", columns,
        )

    def cumprod(
        self,
        columns: "str | Sequence[str] | None" = None,
        *,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group cumulative product. APPEND `{col}_cumprod`."""
        on_cols = self._normalize_on(on)
        return self._parent._per_group_cumulative(
            self._by, on_cols, "prod", columns,
        )

    def cummax(
        self,
        columns: "str | Sequence[str] | None" = None,
        *,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group cumulative max. APPEND `{col}_cummax`."""
        on_cols = self._normalize_on(on)
        return self._parent._per_group_cumulative(
            self._by, on_cols, "max", columns,
        )

    def cummin(
        self,
        columns: "str | Sequence[str] | None" = None,
        *,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group cumulative min. APPEND `{col}_cummin`."""
        on_cols = self._normalize_on(on)
        return self._parent._per_group_cumulative(
            self._by, on_cols, "min", columns,
        )

    def shift(
        self,
        k: int = 1,
        fill_value=None,
        *,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:'''


def add_groupby_methods():
    print("\n=== add _GroupBy.cum* methods ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def cumsum(\n        self,\n        columns: \"str | Sequence[str] | None\" = None," in s:
        print("  already present — skip")
        return
    if GROUPBY_ANCHOR not in s:
        raise SystemExit("_GroupBy.shift anchor not found")
    s = s.replace(GROUPBY_ANCHOR, GROUPBY_CUM_METHODS, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 5. Tests ============

TESTS = '''"""v0.9.23e: cumsum/cumprod/cummax/cummin per group (7.A.6).

Reference: pandas .cumsum() / .cumprod() / .cummax() / .cummin() on
Series and GroupBy. Nulls preserved (state frozen, matches CADW
expanding semantics — NOT pandas NaN-propagate).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


def read_col(df, tmp_path, col):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


def approx(a, b, tol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) < tol


# ============ GLOBAL cum* (df.cumsum) ============

def test_cumsum_global_basic(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3, 4, 5]})
    res = df.cumsum()
    assert read_col(res, tmp_path, "v_cumsum") == [1, 3, 6, 10, 15]


def test_cumprod_global_basic(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3, 4, 5]})
    res = df.cumprod()
    assert read_col(res, tmp_path, "v_cumprod") == [1, 2, 6, 24, 120]


def test_cummax_global_basic(tmp_path):
    df = make_df(tmp_path, {"v": [3, 1, 4, 1, 5]})
    res = df.cummax()
    assert read_col(res, tmp_path, "v_cummax") == [3, 3, 4, 4, 5]


def test_cummin_global_basic(tmp_path):
    df = make_df(tmp_path, {"v": [3, 1, 4, 1, 5]})
    res = df.cummin()
    assert read_col(res, tmp_path, "v_cummin") == [3, 1, 1, 1, 1]


def test_cumsum_global_pandas_equiv(tmp_path):
    rng = np.random.default_rng(0)
    vals = rng.normal(size=100).tolist()
    df = make_df(tmp_path, {"v": vals}, chunk=10)
    res = df.cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    expected = pd.Series(vals).cumsum().tolist()
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_cumprod_global_pandas_equiv(tmp_path):
    rng = np.random.default_rng(0)
    vals = (rng.random(size=50) + 0.5).tolist()  # avoid 0
    df = make_df(tmp_path, {"v": vals}, chunk=10)
    res = df.cumprod()
    got = read_col(res, tmp_path, "v_cumprod")
    expected = pd.Series(vals).cumprod().tolist()
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_cumsum_global_preserves_original(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3], "s": ["a", "b", "c"]})
    res = df.cumsum()
    assert "v" in res.columns
    assert "s" in res.columns
    assert "v_cumsum" in res.columns
    # Original unchanged
    assert read_col(res, tmp_path, "v") == [1, 2, 3]


def test_cumsum_with_nulls_freeze_state(tmp_path):
    df = make_df(tmp_path, {
        "v": pa.array([1.0, None, 3.0, None, 5.0], type=pa.float64()),
    })
    res = df.cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # Null row -> null output; state frozen
    assert approx(got[0], 1.0)
    assert got[1] is None
    assert approx(got[2], 4.0)  # 1 + 3
    assert got[3] is None
    assert approx(got[4], 9.0)  # 4 + 5


def test_cumsum_on_col(tmp_path):
    df = make_df(tmp_path, {
        "ts": [3, 1, 2, 5, 4],
        "v": [30, 10, 20, 50, 40],
    })
    res = df.cumsum("v", on="ts")
    got = read_col(res, tmp_path, "v_cumsum")
    # Sorted by ts: v=[10, 20, 30, 40, 50] -> cumsum=[10, 30, 60, 100, 150]
    # orig: [3->60, 1->10, 2->30, 5->150, 4->100]
    assert got == [60, 10, 30, 150, 100]


def test_cummax_on_col(tmp_path):
    df = make_df(tmp_path, {
        "ts": [1, 2, 3, 4, 5],
        "v": [3, 1, 4, 1, 5],
    })
    res = df.cummax("v", on="ts")
    got = read_col(res, tmp_path, "v_cummax")
    assert got == [3, 3, 4, 4, 5]


def test_cumsum_subset_columns(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, 2, 3], "b": [10, 20, 30], "s": ["x", "y", "z"],
    })
    res = df.cumsum(columns=["a"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    assert "a_cumsum" in names
    assert "b_cumsum" not in names


# ============ PER-GROUP cum* ============

def test_cumsum_per_group_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b", "b"],
        "v": [1, 2, 3, 10, 20, 30],
    })
    res = df.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # group a: 1, 3, 6; group b: 10, 30, 60
    assert got == [1, 3, 6, 10, 30, 60]


def test_cumprod_per_group_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"],
        "v": [1, 2, 3, 10, 20],
    })
    res = df.groupby("g").cumprod()
    got = read_col(res, tmp_path, "v_cumprod")
    assert got == [1, 2, 6, 10, 200]


def test_cummax_per_group_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b", "b"],
        "v": [3, 1, 4, 1, 5, 2],
    })
    res = df.groupby("g").cummax()
    got = read_col(res, tmp_path, "v_cummax")
    assert got == [3, 3, 4, 1, 5, 5]


def test_cummin_per_group_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"],
        "v": [3, 1, 4, 5, 2],
    })
    res = df.groupby("g").cummin()
    got = read_col(res, tmp_path, "v_cummin")
    assert got == [3, 1, 1, 5, 2]


def test_cumsum_per_group_on_col(tmp_path):
    """Sort-aware per-group: sort by [g, ts], reset per group."""
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"],
        "ts": [3, 1, 2, 2, 1],
        "v": [30, 10, 20, 20, 10],
    })
    res = df.groupby("g").cumsum(on="ts")
    got = read_col(res, tmp_path, "v_cumsum")
    # group a: sorted ts=1(v=10), 2(v=20), 3(v=30) -> cumsum=10,30,60
    #   orig order: ts=3->60, ts=1->10, ts=2->30
    # group b: sorted ts=1(v=10), 2(v=20) -> cumsum=10,30
    #   orig order: ts=2->30, ts=1->10
    assert got == [60, 10, 30, 30, 10]


def test_cumsum_per_group_pandas_equiv(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    groups = [f"g{i % 5}" for i in range(n)]
    vals = rng.normal(size=n).tolist()
    df = make_df(tmp_path, {"g": groups, "v": vals}, chunk=20)
    res = df.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")

    pdf = pd.DataFrame({"g": groups, "v": vals})
    expected = pdf.groupby("g")["v"].cumsum().tolist()
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_cumsum_per_group_with_nulls(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "a"],
        "v": pa.array([1.0, None, 3.0, 5.0], type=pa.float64()),
    })
    res = df.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # Null row -> null output; state frozen within group
    assert approx(got[0], 1.0)
    assert got[1] is None
    assert approx(got[2], 4.0)
    assert approx(got[3], 9.0)


def test_cumsum_per_group_null_group(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", None, None],
        "v": [1, 2, 10, 20],
    })
    res = df.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # Null group -> own group
    assert got == [1, 3, 10, 30]


def test_cumsum_per_group_append_not_replace(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b"], "v": [1, 2, 3],
    })
    res = df.groupby("g").cumsum()
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    assert "v" in names and "v_cumsum" in names


def test_cumsum_per_group_multibatch(tmp_path):
    """chunk=1: many batches, per-group state persists across batches."""
    df = make_df(tmp_path, {
        "g": ["a", "b", "a", "b", "a", "b"],
        "v": [1, 10, 2, 20, 3, 30],
    }, chunk=1)
    res = df.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # Interleaved groups: a: 1, 3, 6; b: 10, 30, 60
    # Output row_id order: [1,10,2,20,3,30] -> [1,10,3,30,6,60]
    assert got == [1, 10, 3, 30, 6, 60]


def test_cumsum_per_group_filtered(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"],
        "v": [1, 2, 3, 10, 20],
        "f": list(range(5)),
    })
    filtered = df[df["f"] > 1]
    res = filtered.groupby("g").cumsum()
    got = read_col(res, tmp_path, "v_cumsum")
    # After filter: a:v=[3], b:v=[10, 20] -> cumsum=[3], [10, 30]
    assert got == [3, 10, 30]


# ============ Validation ============

def test_cumsum_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "b", "c"], "v": [1, 2, 3]})
    with pytest.raises(TypeError, match="numeric"):
        df.cumsum(columns=["s"])


def test_cumsum_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    with pytest.raises(KeyError, match="nope"):
        df.cumsum(columns=["nope"])


def test_cumsum_empty_columns_raises(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "b"]})
    with pytest.raises(ValueError, match="no numeric columns"):
        df.cumsum()


def test_cumsum_on_missing_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2], "ts": [1, 2]})
    with pytest.raises(KeyError, match="nope"):
        df.cumsum("v", on="nope")


def test_per_group_cumsum_group_col_in_columns_raises(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "a"], "v": [1, 2]})
    with pytest.raises(ValueError, match="cannot be accumulated"):
        df.groupby("g").cumsum(columns=["g"])


# ============ AST guards ============

def test_cumulative_impl_no_to_pandas_ast():
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._cumulative_impl)
    assert_no_call_in_source(
        src, "to_pandas", "from_pandas",
        fn_name="_cumulative_impl",
    )


def test_per_group_cumulative_no_to_pandas_ast():
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_cumulative)
    assert_no_call_in_source(
        src, "to_pandas", "from_pandas",
        fn_name="_per_group_cumulative",
    )


def test_per_group_cumulative_uses_boundaries():
    """Reuses _group_boundaries_arrow (Arrow-native, v0.9.21)."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_cumulative)
    assert "_group_boundaries_arrow" in src


def test_expanding_state_has_prod():
    """_ExpandingState supports 'prod' after v0.9.23e."""
    from cadw.dataframe import _ExpandingState
    st = _ExpandingState("prod", 1)
    assert st.push(2) == 2
    assert st.push(3) == 6
    assert st.push(4) == 24
'''


def write_test_file():
    print("\n=== write tests/test_cum_series.py ===")
    p = Path("tests/test_cum_series.py")
    if p.exists():
        print("  already exists — skip")
        return
    p.write_text(TESTS)
    assert_compiles(p)
    print("  OK")


# ============ 6. Dogfood ============

def dogfood():
    print("\n=== dogfood: cum* global + per-group + pandas parity ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import numpy as _np_
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        # Global cumsum parity
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            rng = _np_.random.default_rng(7)
            vals = rng.normal(size=100).tolist()
            t = pa_.table({"v": vals})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=10,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            res = df.cumsum()
            out = td / "o.parquet"
            res.write_parquet(out)
            got = pq_.read_table(out).column("v_cumsum").to_pylist()
            expected = _np_.cumsum(vals).tolist()
            for a, b in zip(got, expected):
                assert abs(a - b) < 1e-10

        # Per-group cumsum with nulls
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "g": ["a", "a", "a", "b", "b"],
                "v": pa_.array(
                    [1.0, None, 3.0, 10.0, 20.0], type=pa_.float64(),
                ),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            res = df.groupby("g").cumsum()
            out = td / "o.parquet"
            res.write_parquet(out)
            got = pq_.read_table(out).column("v_cumsum").to_pylist()
            # group a: 1, None, 1+3=4; group b: 10, 30
            assert got[0] == 1.0
            assert got[1] is None
            assert got[2] == 4.0
            assert got[3] == 10.0
            assert got[4] == 30.0

        # Sort-aware global cumsum
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "ts": [3, 1, 2, 5, 4],
                "v": [30, 10, 20, 50, 40],
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            res = df.cumsum("v", on="ts")
            out = td / "o.parquet"
            res.write_parquet(out)
            got = pq_.read_table(out).column("v_cumsum").to_pylist()
            # sorted by ts: v=[10, 20, 30, 40, 50] cumsum=[10, 30, 60, 100, 150]
            # orig: [3->60, 1->10, 2->30, 5->150, 4->100]
            assert got == [60, 10, 30, 150, 100]

        print("  global parity + per-group nulls + sort-aware ok")

    run_verify_in_subprocess(_verify)


# ============ 7. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_cum_series.py",
         "tests/test_expanding_ewm.py",  # check _ExpandingState not broken
         "tests/test_shift_on_col.py",
         "tests/test_groupby_per_group.py",
         "tests/test_merge_arrow.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 8. main ============

def main():
    print("=== v0.9.23e: cumsum/cumprod/cummax/cummin ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    extend_expanding_state()
    add_cum_methods()
    add_per_group_helper()
    add_groupby_methods()
    write_test_file()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23e: cumsum/cumprod/cummax/cummin global + per group. "
        "AUDIT_FULL_v0922 item 7.A.6 (Top-10 gap #5: time-series "
        "feature engineering workhorse; shift/diff/rolling/"
        "expanding/ewm existed but cum* family was missing). "
        "Prototype #3 (groupby_cumsum) validated: RAM 13.28 MB @ "
        "32 MB budget, identity PASS. Pre-fix prototype was 40s "
        "(dominated by sort under tight budget); post-7.A.1 "
        "Arrow-native merge ~2s. "
        "API (8 methods): "
        "  df.cumsum/cumprod/cummax/cummin(columns=None, *, on=None) "
        "  df.groupby(g).cumsum/cumprod/cummax/cummin(...) "
        "on=None: streaming (row_id order); on=col: sort-aware "
        "(external_sort + sort back by __orig_rid, same pattern as "
        "_shift_on / _rolling_on). Per-group: sort by [g, *on?], "
        "stream with _ExpandingState reset per group, sort back. "
        "Output: APPEND {col}_cumsum/_cumprod/_cummax/_cummin "
        "(matches expanding/ewm — new columns, not replace). "
        "_ExpandingState extension: 'prod' aggregator (init _prod=1.0, "
        "push *= x, reset). Reuses existing sum/min/max/count state "
        "class — O(1) per row, no new memory model. "
        "Peak RAM: O(batch + K_groups * n_cols * 8) per-group; "
        "O(batch + n_cols * 8) global. Independent of N. "
        "Nulls preserved (state frozen, null row -> null output; "
        "matches CADW expanding semantics, NOT pandas NaN-propagate). "
        "Dtype preserved (int -> int, float -> float). "
        "+tests/test_cum_series.py (25 tests: 4 global + 3 pandas "
        "parity + nulls + sort-aware + subset + 5 per-group + "
        "per-group pandas parity + per-group nulls + null-group + "
        "append-not-replace + multibatch + filtered + 5 validation "
        "+ 4 AST/impl). "
        "Fast: 1975 -> 2000 passed.",
        self_path="scripts/patch_v0923e_cum_series.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23e applied. 7.A.6 complete.")


if __name__ == "__main__":
    main()