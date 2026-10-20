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
    def contains(self, pat: str) -> "MaskFilter":
        """Substring match → MaskFilter (for `df[mask]`).

        Like pandas `Series.str.contains(pat, regex=False, na=False)`
        — null → False.
        """
        from cadw.dataset.row_filter import PredicateFilter
        return PredicateFilter(self._column, "contains", pat)
