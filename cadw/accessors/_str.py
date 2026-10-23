"""String accessor (v0.9.0a) — Arrow-native string ops.

Feature ops (len/lower/upper/replace/strip) return new CadwDataFrame
with `{column}_{op}`. Predicate ops (contains/starts_with/ends_with)
return MaskFilter for use in `df[mask]` — pandas-compatible primary use.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from cadw.dataframe import CadwDataFrame


def _validate_string(df: "CadwDataFrame", column: str) -> None:
    t = df._dataset.schema.field(column).type
    if not (pa.types.is_string(t) or pa.types.is_large_string(t)
            or pa.types.is_binary(t) or pa.types.is_large_binary(t)):
        raise TypeError(
            f".str: column {column!r} has type {t}; expected string or binary"
        )


class _StrAccessor:
    """Pandas-style `df['s'].str.*`."""

    __slots__ = ("_parent", "_column")

    def __init__(self, parent: "CadwDataFrame", column: str) -> None:
        self._parent = parent
        self._column = column
        _validate_string(parent, column)

    def _feature(self, name: str, fn, out_type) -> "CadwDataFrame":
        return self._parent._apply_column_transform(
            self._column, name, fn, out_type, accessor="str",
        )

    @property
    def len(self) -> "CadwDataFrame":
        return self._feature("len", lambda arr: pc.utf8_length(arr),
                             pa.int64())

    @property
    def lower(self) -> "CadwDataFrame":
        return self._feature("lower", lambda arr: pc.utf8_lower(arr),
                             pa.string())

    @property
    def upper(self) -> "CadwDataFrame":
        return self._feature("upper", lambda arr: pc.utf8_upper(arr),
                             pa.string())

    @property
    def strip(self) -> "CadwDataFrame":
        return self._feature(
            "strip", lambda arr: pc.utf8_trim_whitespace(arr),
            pa.string(),
        )

    @property
    def lstrip(self) -> "CadwDataFrame":
        return self._feature(
            "lstrip", lambda arr: pc.utf8_ltrim_whitespace(arr),
            pa.string(),
        )

    @property
    def rstrip(self) -> "CadwDataFrame":
        return self._feature(
            "rstrip", lambda arr: pc.utf8_rtrim_whitespace(arr),
            pa.string(),
        )

    def starts_with(self, pat: str) -> "MaskFilter":
        from cadw.dataset.row_filter import PredicateFilter
        return PredicateFilter(self._column, "starts_with", pat)

    def ends_with(self, pat: str) -> "MaskFilter":
        from cadw.dataset.row_filter import PredicateFilter
        return PredicateFilter(self._column, "ends_with", pat)

    def replace(self, pat: str, repl: str, *,
                regex: bool = False) -> "CadwDataFrame":
        # Appends {col}_replaced. regex=False default (literal).
        if regex:
            fn = lambda arr: pc.replace_substring_regex(arr, pat, repl)
        else:
            fn = lambda arr: pc.replace_substring(arr, pat, repl)
        return self._feature("replaced", fn, pa.string())

    def slice(
        self, start: int, stop: "int | None" = None,
    ) -> "CadwDataFrame":
        """Slice string characters (v0.9.25k).

        Args:
            start: start index (0-based; negative → from end).
            stop: stop index (exclusive; None → end of string).
                Negative → from end (pandas str.slice parity).

        Returns: augmented CadwDataFrame with `{col}_slice` string
        column (nullable=True). Null input → null output. OOB
        (start >= len) → empty string (matches pc.utf8_slice_codeunits).

        Peak RAM = batch + O(batch). Independent of N.
        """
        import pyarrow.compute as _pc
        if not isinstance(start, int) or isinstance(start, bool):
            raise TypeError(
                f"str.slice: start must be int; got "
                f"{type(start).__name__}"
            )
        if stop is not None and (
            not isinstance(stop, int) or isinstance(stop, bool)
        ):
            raise TypeError(
                f"str.slice: stop must be int or None; got "
                f"{type(stop).__name__}"
            )

        def _fn(arr):
            # Cast to standard string for consistent output type
            # (large_string cast down; binary rejected at _validate_string
            # for len/lower; utf8 kernels may accept binary but we're
            # conservative — cast to string).
            arr_str = _pc.cast(arr, pa.string())
            sliced = _pc.utf8_slice_codeunits(arr_str, start, stop)
            return _pc.cast(sliced, pa.string())

        return self._parent._apply_column_transform(
            self._column, "slice", _fn, pa.string(), accessor="str",
        )

    def cat(self, other: str, *, sep: str = "") -> "CadwDataFrame":
        """Concatenate with another string column (v0.9.25k).

        Args:
            other: another column name in the same DataFrame.
            sep: separator string between values (default '').

        Returns: augmented CadwDataFrame with `{col}_cat_{other}` string
        column (nullable=True). Null in either side → null
        (pc.binary_join_element_wise null propagation; pandas
        `Series.str.cat` parity).

        Both source columns must be string or large_string. Output is
        standard `pa.string()` (large_string cast down).

        Note: scalar concatenation (`str.cat('-suffix')`) NOT supported
        — use `.replace('', '-suffix')` or custom transform. §11.5.

        Peak RAM = batch × 2 (source + destination). Independent of N.
        """
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if not isinstance(other, str):
            raise TypeError(
                f"str.cat: other must be str; got "
                f"{type(other).__name__}"
            )
        if not isinstance(sep, str):
            raise TypeError(
                f"str.cat: sep must be str; got {type(sep).__name__}"
            )
        if other not in self._parent._columns:
            raise KeyError(
                f"str.cat: other column {other!r} not in "
                f"{self._parent._columns}"
            )
        if other == self._column:
            raise ValueError(
                "str.cat: cannot concatenate column with itself"
            )

        for c in (self._column, other):
            t = self._parent._dataset.schema.field(c).type
            if not (
                pa.types.is_string(t) or pa.types.is_large_string(t)
            ):
                raise TypeError(
                    f"str.cat: column {c!r} has type {t}; "
                    f"string/large_string required"
                )

        out_col = f"{self._column}_cat_{other}"
        if out_col in self._parent._columns:
            raise KeyError(
                f"str.cat: output column {out_col!r} already exists"
            )

        out_fields = [
            self._parent._dataset.schema.field(c)
            for c in self._parent._columns
        ]
        # §13.4/272: nullable=True (Dataset.from_batches singleton
        # rebuild is nullable=True; mismatch → ArrowInvalid at runtime).
        out_fields.append(pa.field(out_col, pa.string(), nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter():
            for b in self._parent._iter_arrow_batches():
                self._parent._memory_guard.check_batch(
                    b, name="str.cat.input", extra_factor=3.0,
                )
                a_arr = _pc.cast(b.column(self._column), pa.string())
                o_arr = _pc.cast(b.column(other), pa.string())
                joined = _pc.binary_join_element_wise(
                    a_arr, o_arr, sep,
                )
                arrays = [
                    b.column(c) for c in self._parent._columns
                ]
                arrays.append(joined)
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._parent._new_dataset_dir(),
        )
        from cadw.dataframe import CadwDataFrame as _CDF
        return _CDF(
            new_ds,
            memory_guard=self._parent._memory_guard,
            spill_dir=self._parent._spill_dir,
        )

    def count(self, pat: str) -> "CadwDataFrame":
        """Count non-overlapping occurrences of `pat` (v0.9.25k).

        Args:
            pat: non-empty literal pattern string (not regex).

        Returns: augmented CadwDataFrame with `{col}_count` int64
        column (nullable=True). Null input → null output.

        For regex counting: use pc.count_substring_regex via custom
        transform or backlog v0.9.26 `str.count_regex`. §11.5.

        Peak RAM = batch + O(batch). Independent of N.
        """
        import pyarrow.compute as _pc
        if not isinstance(pat, str):
            raise TypeError(
                f"str.count: pat must be str; got "
                f"{type(pat).__name__}"
            )
        if not pat:
            raise ValueError("str.count: pat must be non-empty")

        def _fn(arr):
            # Cast to string for consistent output; count_substring may
            # return int32 (PyArrow 26) → cast to int64 (pandas parity).
            arr_str = _pc.cast(arr, pa.string())
            cnt = _pc.count_substring(arr_str, pat)
            return _pc.cast(cnt, pa.int64())

        return self._parent._apply_column_transform(
            self._column, "count", _fn, pa.int64(), accessor="str",
        )

    def split(
        self, pat: str, *, max_splits: "int | None" = None,
        reverse: bool = False,
    ) -> "CadwDataFrame":
        """Split string into list of substrings (v0.9.25l).

        Args:
            pat: non-empty literal separator (not regex).
            max_splits: max splits per value (None → unlimited).
            reverse: split from end (only with max_splits).

        Returns: augmented CadwDataFrame with `{col}_split`
        list<string> column (nullable=True). Null → null.
        Empty string → [''] (one-element list, pandas Series.str.split
        parity).

        For regex split: use pc.split_pattern_regex via custom
        transform (backlog `str.rsplit_regex`, v0.9.26, §11.5).

        Peak RAM = batch × O(avg_parts). Independent of N.
        """
        import pyarrow.compute as _pc
        if not isinstance(pat, str):
            raise TypeError(
                f"str.split: pat must be str; got {type(pat).__name__}"
            )
        if not pat:
            raise ValueError("str.split: pat must be non-empty")
        if max_splits is not None and (
            not isinstance(max_splits, int)
            or isinstance(max_splits, bool)
            or max_splits < 1
        ):
            raise ValueError(
                f"str.split: max_splits must be int >= 1 or None; got "
                f"{max_splits!r}"
            )
        if not isinstance(reverse, bool):
            raise TypeError(
                f"str.split: reverse must be bool; got "
                f"{type(reverse).__name__}"
            )

        def _fn(arr):
            arr_str = _pc.cast(arr, pa.string())
            return _pc.split_pattern(
                arr_str,
                pattern=pat,
                max_splits=(max_splits if max_splits is not None else -1),
                reverse=reverse,
            )

        return self._parent._apply_column_transform(
            self._column, "split", _fn,
            pa.list_(pa.string()), accessor="str",
        )

    def contains(self, pat: str) -> "MaskFilter":
        """Substring match → MaskFilter (for `df[mask]`).

        Like pandas `Series.str.contains(pat, regex=False, na=False)`
        — null → False.
        """
        from cadw.dataset.row_filter import PredicateFilter
        return PredicateFilter(self._column, "contains", pat)
