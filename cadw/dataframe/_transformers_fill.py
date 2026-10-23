# cadw/dataframe/_transformers_fill.py
# v0.9.26o (B.1 step 9): _TransformersFillMixin.
# Symbols: dropna, fillna, ffill, bfill, shift, diff,
#   pct_change.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import pyarrow as pa
from cadw.dataframe_helpers import _ArrowTail
from cadw.dataset import Dataset


class _TransformersFillMixin:
    """Null fills + sequential transforms (v0.9.26o)."""
    def dropna(
        self,
        how: str = "any",
        subset: "Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Drop rows with nulls (lazy — composes notnull() filters).

        - how='any': drop if ANY subset column is null.
        - how='all': drop only if ALL subset columns are null.
        - subset=None: all columns.
        """
        if how not in ("any", "all"):
            raise ValueError(
                f"dropna: how must be 'any' or 'all'; got {how!r}"
            )
        cols = list(subset) if subset is not None else list(self._columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"dropna: subset column {c!r} not in {self._columns}"
                )
        if not cols:
            return self

        combined = self[cols[0]].notnull()
        for c in cols[1:]:
            nf = self[c].notnull()
            if how == "any":
                combined = combined & nf
            else:
                combined = combined | nf
        return self[combined]


    def fillna(self, value, columns=None) -> "CadwDataFrame":
        """Fill nulls with a scalar `value`. Streaming.

        v0.9.23g (AUDIT_FULL_v0922 item 7.A.5). Per-batch `pc.fill_null`
        — no materialize. `value` must be a scalar compatible with all
        target columns' dtypes (Arrow will raise on incompatible fill
        — no silent type coercion).

        Args:
            value: scalar fill value.
            columns: None → all columns; str → one; Sequence[str] →
                subset. Non-numeric columns fillable with matching-type
                scalar (e.g. str fill on string column).

        Returns: CadwDataFrame with nulls filled in target columns.
        """
        from ._core import CadwDataFrame
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset as _DS

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("fillna: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"fillna: column {c!r} not in {self._columns}"
                )
        fill_set = set(cols)

        # All fields nullable=True except unchanged.
        out_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            if c in fill_set:
                out_fields.append(pa.field(c, orig.type, nullable=True))
            else:
                out_fields.append(orig)
        out_schema = pa.schema(out_fields)

        # Pre-construct the Arrow fill scalar per column, so any type
        # mismatch fails fast at function entry, not mid-iteration.
        fill_scalars = {}
        for c in cols:
            t = self._dataset.schema.field(c).type
            try:
                fill_scalars[c] = pa.scalar(value, type=t)
            except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
                raise TypeError(
                    f"fillna: fill value {value!r} incompatible with "
                    f"column {c!r} type {t}: {e}"
                ) from e

        def _iter_filled():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="fillna.input")
                arrays = []
                for c in self._columns:
                    if c in fill_set:
                        arrays.append(
                            _pc.fill_null(b.column(c), fill_scalars[c])
                        )
                    else:
                        arrays.append(b.column(c))
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = _DS.from_batches(
            out_schema, _iter_filled(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def ffill(self, columns=None) -> "CadwDataFrame":
        """Forward-fill nulls per column (v0.9.25i).

        Streaming per-batch via pc.fill_null_forward + carry for
        leading-null cross-batch fix. O(#cols) state, O(N) I/O.

        - columns: None → all columns; str → one; Sequence[str] →
          subset. Non-fillable columns pass through unchanged.
        - Leading nulls stay null (no prior value).
        - Null-safe across batches: last non-null carries forward.
        - Original column types preserved (nullable=True forced).
        - Int cols stay int (fill doesn't widen types).

        Peak RAM = batch + #cols × 8 B (carry buffers).
        """
        from ._core import CadwDataFrame
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("ffill: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"ffill: column {c!r} not in {self._columns}"
                )

        fill_set = set(cols)
        out_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            if c in fill_set:
                out_fields.append(
                    pa.field(c, orig.type, nullable=True)
                )
            else:
                out_fields.append(orig)
        out_schema = pa.schema(out_fields)

        carry: dict = {c: None for c in cols}

        def _iter():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="ffill.input")
                n = b.num_rows
                if n == 0:
                    yield b.select(self._columns)
                    continue
                arrays = []
                for c in self._columns:
                    arr = b.column(c)
                    if c not in fill_set:
                        arrays.append(arr)
                        continue
                    car = carry[c]
                    if car is not None:
                        extended = pa.concat_arrays([car, arr])
                        filled = _pc.fill_null_forward(extended).slice(1)
                    else:
                        filled = _pc.fill_null_forward(arr)
                    if len(filled) > 0:
                        last_slice = filled.slice(len(filled) - 1, 1)
                        # v0.9.25i-fix2 (§13.4/269): null_count is
                        # O(1) Arrow metadata — replaces
                        # `is_valid(last_slice).to_pylist()[0]`
                        # (1-element to_pylist still flagged by
                        # diagnostic as low_factor since check_batch
                        # factor<3.0). Semantically identical for
                        # 1-element slice.
                        if last_slice.null_count == 0:
                            carry[c] = last_slice
                    arrays.append(filled)
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def bfill(self, columns=None) -> "CadwDataFrame":
        """Backward-fill nulls per column (v0.9.25i).

        Implementation: reverse-sort (__orig_rid descending) +
        ffill-internal + re-sort (__orig_rid ascending). 2×
        external_sort — O(N log N) I/O, O(batch) RAM. **Heavier than
        ffill** (O(N) I/O). Documented.

        - columns: None → all; str → one; Sequence[str] → subset.
        - Trailing nulls stay null (no later value).
        - Original order preserved (2× sort round-trip).
        """
        from ._core import CadwDataFrame
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset
        from cadw.io.batch_reader import ArrowIPCBatchReader
        from cadw.io.batch_writer import ArrowIPCBatchWriter
        from cadw.ops.sort import external_sort

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("bfill: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"bfill: column {c!r} not in {self._columns}"
                )

        fill_set = set(cols)
        out_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            if c in fill_set:
                out_fields.append(
                    pa.field(c, orig.type, nullable=True)
                )
            else:
                out_fields.append(orig)
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

        # Step 1: add __orig_rid, external_sort descending
        def _iter_with_rid():
            offset = 0
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="bfill.step1")
                n = b.num_rows
                rid = pa.array(
                    range(offset, offset + n), type=pa.uint64(),
                )
                offset += n
                yield b.append_column("__orig_rid", rid)

        sorted_desc = external_sort(
            _iter_with_rid(),
            key_columns=["__orig_rid"],
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "bfill" / "sort_desc",
            ascending=False,
            log=False,
        )

        # Step 2: ffill pass
        filled_path = self._spill_dir / "bfill" / "filled.arrow"
        filled_path.parent.mkdir(parents=True, exist_ok=True)
        filled_schema = pa.schema(
            [pa.field("__orig_rid", pa.uint64())]
            + [
                pa.field(c, self._dataset.schema.field(c).type,
                         nullable=True)
                for c in self._columns
            ]
        )
        writer = ArrowIPCBatchWriter(str(filled_path), filled_schema)
        carry: dict = {c: None for c in self._columns}
        try:
            for b in ArrowIPCBatchReader(str(sorted_desc), None):
                self._memory_guard.check_batch(
                    b, name="bfill.step2.read",
                )
                n = b.num_rows
                if n == 0:
                    continue
                arrays = [b.column("__orig_rid")]
                for c in self._columns:
                    arr = b.column(c)
                    # v0.9.25i-fix1 (§13.4/267): only fill columns in
                    # fill_set. Prior code filled ALL columns
                    # regardless of columns= — bug caught by
                    # test_bfill_columns_subset. Mirror-symmetry with
                    # ffill (which correctly has this guard).
                    if c not in fill_set:
                        arrays.append(arr)
                        continue
                    car = carry[c]
                    if car is not None:
                        extended = pa.concat_arrays([car, arr])
                        filled = _pc.fill_null_forward(extended).slice(1)
                    else:
                        filled = _pc.fill_null_forward(arr)
                    if len(filled) > 0:
                        last_slice = filled.slice(len(filled) - 1, 1)
                        # v0.9.25i-fix2 (§13.4/269): null_count O(1).
                        if last_slice.null_count == 0:
                            carry[c] = last_slice
                    arrays.append(filled)
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        arrays, schema=filled_schema,
                    )
                )
        finally:
            writer.close()

        # Step 3: external_sort ascending by __orig_rid
        def _iter_filled():
            for b in ArrowIPCBatchReader(str(filled_path), None):
                yield b

        sorted_asc = external_sort(
            _iter_filled(),
            key_columns=["__orig_rid"],
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "bfill" / "sort_asc",
            ascending=True,
            log=False,
        )

        def _iter_out():
            for b in ArrowIPCBatchReader(str(sorted_asc), None):
                yield b.select(list(self._columns))

        new_ds = Dataset.from_batches(
            out_schema, _iter_out(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def shift(
        self,
        k: int = 1,
        fill_value=None,
        *,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Shift values by `k` positions (v0.6.2b, lag; v0.8.6 multi-on).

        - k >= 0: `out[i] = x[i-k]` for i>=k, else `fill_value`.
        - k == 0: returns self (no-op).
        - k < 0 (lead): NotImplementedError (v0.6.3).
        - columns=None: all columns; else subset (others pass-through).
        - fill_value=None: null for first k rows. Non-None: that value.
        - on: positional shift if None; sort-aware if str/Sequence.
          `on=["ts", "sensor"]` — lexicographic composite sort
          (NOT per-group; for per-group see v0.8.5a).

        Streamable: O(k) memory (deque per column). Position-based,
        iterates filtered batches in row order.

        Shifted columns become **nullable** (fill = null).
        """
        from ._core import CadwDataFrame
        from collections import deque

        if not isinstance(k, int):
            raise TypeError(f"shift: k must be int; got {type(k).__name__}")
        if k < 0:
            raise NotImplementedError(
                "shift: negative k (lead) not supported in v0.6.2b; "
                "use reverse + shift + reverse, or v0.6.3"
            )
        if k == 0:
            return self

        # v0.8.2: sort-aware shift; v0.8.6: multi-column on.
        if on is not None:
            on_cols = self._normalize_on(on, fn_name="shift")
            return self._shift_on(
                k, fill_value, columns=columns, on_cols=on_cols,
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

        def _fill(size: int, type_) -> pa.Array:
            if fill_value is None:
                return pa.nulls(size, type=type_)
            return pa.array([fill_value] * size, type=type_)

        def _iter_shifted():
            # v0.9.18 (AUDIT §1.1): Arrow-native tail buffer.
            # Prior: `deque(maxlen=k)` of Python scalars +
            # `col_vals.to_pylist()` per batch (O(batch_size * 28B) peak).
            # Now: `_ArrowTail` (deque of Arrow slices, O(k) peak).
            # Peak 2.05 MB -> <400 KB for 300k rows / k=2 (proven).
            bufs: dict[str, _ArrowTail] = {
                c: _ArrowTail(k) for c in shift_cols
            }
            for b in self._iter_filtered_batches():
                n = b.num_rows
                arrays = []
                for c in self._columns:
                    col_vals = b.column(c)
                    if c not in shift_cols:
                        arrays.append(col_vals)
                        continue
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
                            out, pa.nulls(n - len(out), type=col_vals.type),
                        ])
                    arrays.append(out)
                    bufs[c].push(col_vals)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_shifted(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def diff(
        self,
        k: int = 1,
        *,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """First difference: `out[i] = x[i] - x[i-k]` (v0.6.2b; multi-on v0.8.6).

        - k >= 1. First k rows -> null.
        - columns=None: all; else subset (others pass-through).
        - on: positional if None; sort-aware if str/Sequence
          (lexicographic composite sort).
        - Output columns nullable (first k = null).

        Streamable: O(k) memory.
        """
        from ._core import CadwDataFrame
        from collections import deque

        if not isinstance(k, int):
            raise TypeError(f"diff: k must be int; got {type(k).__name__}")
        if k < 1:
            raise ValueError(f"diff: k must be >= 1; got {k}")

        # v0.8.2: sort-aware diff = shift(on) - original.
        if on is not None:
            return self._diff_on(k=k, columns=columns, on=on)

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
        diff_cols = set(cols)

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in diff_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        import pyarrow.compute as _pc

        def _iter_diffed():
            bufs: dict[str, deque] = {
                c: deque(maxlen=k) for c in diff_cols
            }
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="diff.input", extra_factor=3.5)
                n = b.num_rows
                arrays = []
                for c in self._columns:
                    col_vals = b.column(c)
                    if c not in diff_cols:
                        arrays.append(col_vals)
                        continue
                    prev = list(bufs[c])
                    fill_n = max(0, k - len(prev))
                    parts = []
                    if fill_n > 0:
                        parts.append(pa.nulls(fill_n, type=col_vals.type))
                    if prev:
                        parts.append(pa.array(prev, type=col_vals.type))
                    if n > k:
                        parts.append(col_vals.slice(0, n - k))
                    shifted = (
                        pa.concat_arrays(parts) if parts
                        else pa.array([], type=col_vals.type)
                    )
                    if len(shifted) > n:
                        shifted = shifted.slice(0, n)
                    elif len(shifted) < n:
                        shifted = pa.concat_arrays([
                            shifted,
                            pa.nulls(n - len(shifted), type=col_vals.type),
                        ])
                    diff_arr = _pc.subtract(col_vals, shifted)
                    arrays.append(diff_arr)
                    for v in col_vals.to_pylist():
                        bufs[c].append(v)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_diffed(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def pct_change(self, periods: int = 1, columns=None) -> "CadwDataFrame":
        """Percent change between rows (v0.9.25h).

        Formula: `(x / shift(x, periods)) - 1` per row. Composition
        of `shift(periods)` + `pc.cast` to float64 + `pc.divide` +
        `pc.subtract`.

        - periods: int >= 1. Default 1 (adjacent rows).
        - columns: None → all numeric (int/float). Else str or
          Sequence[str]; non-numeric → TypeError.
        - Output: original + `{col}_pct_change` (float64, nullable).
        - First `periods` rows → null (no previous value).
        - Null in current or shifted → null row (skipna).
        - **Div-by-zero matches pandas**: `pc.divide` uses IEEE 754
          float semantics → `x / 0` (x != 0) = `inf`; `0 / 0` = `nan`.
          Then `inf - 1 = inf`, `nan - 1 = nan`. `pa.array` and Parquet
          preserve inf/nan (IEEE 754 round-trip).
          **CadwDataFrame.pct_change does NOT null-out inf/nan** — user
          who wants null semantics should filter: `df[df.pct.is_finite()]`
          or post-process. Rationale: pandas parity + honesty; changing
          this to null-out-by-default would be a policy decision for
          ALL float ops, not just pct_change (§11.5).
        - Empty filtered view → empty CadwDataFrame with full schema.
        - Original `{col}` columns preserved; new `{col}_pct_change`
          appended.

        Peak RAM = batch × (1 + #cols) × 8 B + _ArrowTail(k × 8 B) per
        col (from shift) — O(batch + k × #cols). Independent of N.
        """
        from ._core import CadwDataFrame
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if not isinstance(periods, int) or periods < 1:
            raise ValueError(
                f"pct_change: periods must be int >= 1; got {periods!r}"
            )
        if columns is None:
            cols = [
                c for c in self._columns
                if pa.types.is_integer(
                    self._dataset.schema.field(c).type
                ) or pa.types.is_floating(
                    self._dataset.schema.field(c).type
                )
            ]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError(
                "pct_change: no numeric columns selected "
                "(pass columns=None for auto int/float, or explicit)"
            )
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"pct_change: column {c!r} not in {self._columns}"
                )
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"pct_change: column {c!r} has type {t}; numeric "
                    f"required"
                )

        shifted = self.shift(periods, columns=cols)

        out_fields = [
            self._dataset.schema.field(c) for c in self._columns
        ]
        for c in cols:
            out_fields.append(
                pa.field(f"{c}_pct_change", pa.float64(), nullable=True)
            )
        out_schema = pa.schema(out_fields)

        batch_size = 10_000

        def _iter():
            orig_it = self._iter_arrow_batches(batch_size=batch_size)
            shifted_it = shifted._iter_arrow_batches(batch_size=batch_size)
            for ob, sb in zip(orig_it, shifted_it):
                if ob.num_rows != sb.num_rows:
                    raise RuntimeError(
                        f"pct_change: batch size mismatch "
                        f"(orig={ob.num_rows}, shifted={sb.num_rows})"
                    )
                arrays = [ob.column(c) for c in self._columns]
                for c in cols:
                    curr_f = _pc.cast(ob.column(c), pa.float64())
                    prev_f = _pc.cast(sb.column(c), pa.float64())
                    # pc.divide uses IEEE 754 (inf for x/0, nan for 0/0)
                    # — matches pandas. Nulls preserved (null/null = null).
                    ratio = _pc.divide(curr_f, prev_f)
                    pct = _pc.subtract(
                        ratio, pa.scalar(1.0, type=pa.float64())
                    )
                    arrays.append(pct)
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

