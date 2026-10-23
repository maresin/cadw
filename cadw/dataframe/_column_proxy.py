# cadw/dataframe/_column_proxy.py
# v0.9.26j (B.1 step 4): _ColumnProxy class.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import pyarrow as pa
from cadw.dataset import (
    MaterializedIndex,
    MaterializedMultiIndex,
    PredicateFilter,
)

class _ColumnProxy:
    """df['col']: comparisons → PredicateFilter; .dt/.str → accessors."""
    __slots__ = ("_column", "_parent")

    def __init__(self, column: str, parent: "CadwDataFrame") -> None:
        self._column = column
        self._parent = parent

    def _pred(self, op: str, value) -> PredicateFilter:
        return PredicateFilter(self._column, op, value)

    def __gt__(self, v): return self._pred("gt", v)
    def __ge__(self, v): return self._pred("ge", v)
    def __lt__(self, v): return self._pred("lt", v)
    def __le__(self, v): return self._pred("le", v)
    def __eq__(self, v): return self._pred("eq", v)  # type: ignore[override]
    def __ne__(self, v): return self._pred("ne", v)  # type: ignore[override]
    def __hash__(self): return id(self)

    def isnull(self) -> PredicateFilter:
        """Null mask. Use in `df[df['a'].isnull()]`."""
        return PredicateFilter(self._column, "is_null", None)

    def notnull(self) -> PredicateFilter:
        """Not-null mask. Use in `df[df['a'].notnull()]`."""
        return PredicateFilter(self._column, "not_null", None)

    def isin(self, values) -> PredicateFilter:
        """Set membership (v0.9.25g). `df[df['a'].isin([1, 2, 3])]`.

        - values: iterable of scalars; coerced to column type per batch.
        - Empty iterable → all False.
        - Null row matches only if None is in values (pandas semantics).
        - Non-coercible type → TypeError at first batch.
        """
        return PredicateFilter(self._column, "isin", values)

    def between(self, left, right, *, inclusive: str = "both"):
        """Range predicate (v0.9.25g). `df[df['a'].between(0, 10)]`.

        - inclusive ∈ {'both' (default), 'neither', 'left', 'right'}.
        - Null row not matched (Kleene drop).
        - left > right → all False (no argument reorder).
        """
        return PredicateFilter(
            self._column, "between", (left, right, inclusive),
        )

    def idxmax(self):
        """Index label of first row with maximum value (v0.9.25h).

        - Returns index LABEL (RangeIndex → int; MaterializedIndex →
          original label; MultiIndex → tuple). Not positional.
        - First occurrence in row_id order (pandas `idxmax` semantics).
        - Nulls skipped. All-null column → ValueError.
        - Empty filtered view → ValueError.

        Two-pass: ReduceOp(MinMaxOp, 'max') for the max value, then
        stream to find first matching position via pc.equal +
        pc.indices_nonzero. Peak = batch + O(1) state.
        """
        from cadw.ops.reduce import MinMaxOp, reduce
        import pyarrow.compute as _pc

        max_val = reduce(
            self._parent._iter_filtered_batches(),
            MinMaxOp(self._column, "max"),
            self._parent._memory_guard,
            log=False,
        )
        if max_val is None:
            raise ValueError(
                f"idxmax: column {self._column!r} has no non-null values"
            )
        for batch, positions in (
            self._parent._iter_filtered_batches_with_positions()
        ):
            arr = batch.column(self._column)
            eq = _pc.equal(arr, pa.scalar(max_val, type=arr.type))
            eq = _pc.fill_null(eq, pa.scalar(False, type=pa.bool_()))
            nonzero = _pc.indices_nonzero(eq)
            if len(nonzero) > 0:
                local_pos = int(nonzero[0])
                raw_pos = int(positions[local_pos])
                return self._index_at_raw_pos(raw_pos)
        raise RuntimeError("idxmax: value not found (unreachable)")

    def idxmin(self):
        """Index label of first row with minimum value (v0.9.25h).

        Mirror of `idxmax` (see docstring). Nulls skipped.
        """
        from cadw.ops.reduce import MinMaxOp, reduce
        import pyarrow.compute as _pc

        min_val = reduce(
            self._parent._iter_filtered_batches(),
            MinMaxOp(self._column, "min"),
            self._parent._memory_guard,
            log=False,
        )
        if min_val is None:
            raise ValueError(
                f"idxmin: column {self._column!r} has no non-null values"
            )
        for batch, positions in (
            self._parent._iter_filtered_batches_with_positions()
        ):
            arr = batch.column(self._column)
            eq = _pc.equal(arr, pa.scalar(min_val, type=arr.type))
            eq = _pc.fill_null(eq, pa.scalar(False, type=pa.bool_()))
            nonzero = _pc.indices_nonzero(eq)
            if len(nonzero) > 0:
                local_pos = int(nonzero[0])
                raw_pos = int(positions[local_pos])
                return self._index_at_raw_pos(raw_pos)
        raise RuntimeError("idxmin: value not found (unreachable)")

    def _index_at_raw_pos(self, raw_pos: int):
        """Look up index label at raw (unfiltered) position.

        Delegates to MaterializedIndex.__getitem__ (in
        cadw/dataset/index.py) — avoids .as_py() in facade (AST
        guard test_facade_has_no_as_py).
        """
        if self._parent._index_kind == "range":
            return self._parent._index_start + int(raw_pos)
        if self._parent._index_kind == "materialized":
            idx = MaterializedIndex(self._parent._index_values)
            return idx[raw_pos]
        if self._parent._index_kind == "multi":
            idx = MaterializedMultiIndex(self._parent._index_levels)
            return idx[raw_pos]
        raise RuntimeError(
            f"idxmax/idxmin: unknown index_kind "
            f"{self._parent._index_kind!r}"
        )

    @property
    def dt(self):
        """Datetime accessor. Requires timestamp or date column."""
        from cadw.accessors import _DtAccessor
        return _DtAccessor(self._parent, self._column)

    @property
    def str(self):
        """String accessor. Requires string or binary column."""
        from cadw.accessors import _StrAccessor
        return _StrAccessor(self._parent, self._column)


    # ---------- scalar aggregators (v0.9.6b1) ----------
    #
    # Composition of existing ReduceOp. O(1) state, bounded by
    # MemoryGuard. No pandas. This is the CADW-native replacement for
    # `g["col"].mean()` etc. inside per-group apply.

    def mean(self) -> float:
        from cadw.ops.reduce import MeanOp, reduce
        return reduce(
            self._parent._iter_filtered_batches(),
            MeanOp(self._column),
            self._parent._memory_guard,
            log=False,
        )

    def sum(self) -> float:
        from cadw.ops.reduce import SumOp, reduce
        return reduce(
            self._parent._iter_filtered_batches(),
            SumOp(self._column),
            self._parent._memory_guard,
            log=False,
        )

    def min(self):
        from cadw.ops.reduce import MinMaxOp, reduce
        return reduce(
            self._parent._iter_filtered_batches(),
            MinMaxOp(self._column, "min"),
            self._parent._memory_guard,
            log=False,
        )

    def max(self):
        from cadw.ops.reduce import MinMaxOp, reduce
        return reduce(
            self._parent._iter_filtered_batches(),
            MinMaxOp(self._column, "max"),
            self._parent._memory_guard,
            log=False,
        )

    def std(self, ddof: int = 1) -> float:
        from cadw.ops.reduce import StdOp, reduce
        return reduce(
            self._parent._iter_filtered_batches(),
            StdOp(self._column, ddof=ddof),
            self._parent._memory_guard,
            log=False,
        )

    def count(self) -> int:
        """Non-null count (skipna=True)."""
        from cadw.ops.reduce import CountOp, reduce
        return reduce(
            self._parent._iter_filtered_batches(),
            CountOp(self._column),
            self._parent._memory_guard,
            log=False,
        )

    # ---------- bounded materialize (v0.9.6b1) ----------
    #
    # check_state BEFORE materialize (§3.10). n = upper bound via
    # parent.count(); over-estimate acceptable (conservative guard).

    def _estimate_materialize_bytes(self) -> int:
        """v0.9.18 (AUDIT §1.5): sample-based per-row estimate.

        Prior: `n * 50` under-estimates strings 1.2×-200× — 1M rows
        × 1KB strings passes 50 MB guard but allocates 1.05 GB (OOM).
        Fix: sample first batch, use nbytes/num_rows + 49 (CPython
        str overhead). Numeric keeps 50 B/row conservative default.
        """
        n = self._parent.count()
        if n == 0:
            return 0
        ftype = self._parent._dataset.schema.field(self._column).type
        if (pa.types.is_string(ftype) or pa.types.is_large_string(ftype)
                or pa.types.is_binary(ftype)
                or pa.types.is_large_binary(ftype)):
            # Sample first batch to estimate avg row bytes
            avg_bytes = 64  # fallback if empty
            for b in self._parent._iter_filtered_batches():
                if b.num_rows > 0:
                    # batch.nbytes includes string data + 4B offsets
                    avg_bytes = max(64, int(b.nbytes / b.num_rows) + 50)
                    break
            return n * avg_bytes
        # Numeric / temporal / bool — 50 B/row is generous
        return n * 50

    def to_list(self) -> list:
        """Materialize column as Python list. Bounded by MemoryGuard.

        v0.9.18 (AUDIT §1.5): sample-based estimate. check_state
        BEFORE materialize.
        """
        n = self._parent.count()
        est = self._estimate_materialize_bytes()
        self._parent._memory_guard.check_state(
            est, name=f"to_list({self._column})",
        )
        out: list = []
        for b in self._parent._iter_filtered_batches():
            out.extend(b.column(self._column).to_pylist())
        return out

    def to_numpy(self):
        """Materialize column as numpy array. Bounded by MemoryGuard.

        Numeric: zero-copy from Arrow buffer (via to_numpy(zero_copy_
        only=False)). Strings: object array. Peak ≈ n × element_size
        + n × 8 (offset for variable types). check_state BEFORE.
        """
        import numpy as np
        n = self._parent.count()
        ftype = self._parent._dataset.schema.field(self._column).type
        # Conservative element size: 8 for numeric, 32 for string/other.
        elem_size = 8 if (
            pa.types.is_integer(ftype) or pa.types.is_floating(ftype)
            or pa.types.is_boolean(ftype)
            or pa.types.is_timestamp(ftype) or pa.types.is_date(ftype)
            or pa.types.is_time(ftype)
        ) else 32
        self._parent._memory_guard.check_state(
            n * elem_size, name=f"to_numpy({self._column})",
        )
        chunks: list = []
        for b in self._parent._iter_filtered_batches():
            chunks.append(
                b.column(self._column).to_numpy(zero_copy_only=False)
            )
        if not chunks:
            return np.array([])
        return np.concatenate(chunks)

    def to_arrow(self) -> pa.Array:
        """Materialize column as single Arrow Array. Bounded.

        Peak ≈ n × element_size. check_state BEFORE materialize.
        """
        n = self._parent.count()
        ftype = self._parent._dataset.schema.field(self._column).type
        elem_size = 8 if (
            pa.types.is_integer(ftype) or pa.types.is_floating(ftype)
            or pa.types.is_boolean(ftype)
            or pa.types.is_timestamp(ftype) or pa.types.is_date(ftype)
            or pa.types.is_time(ftype)
        ) else 50
        self._parent._memory_guard.check_state(
            n * elem_size, name=f"to_arrow({self._column})",
        )
        chunks: list = []
        for b in self._parent._iter_filtered_batches():
            chunks.append(b.column(self._column))
        if not chunks:
            return pa.array([], type=ftype)
        if len(chunks) == 1:
            return chunks[0]
        return pa.concat_arrays(chunks)

    def __repr__(self) -> str:
        return f"ColumnProxy({self._column!r})"
