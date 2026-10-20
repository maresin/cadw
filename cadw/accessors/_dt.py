"""Datetime accessor (v0.9.0a) — Arrow-native temporal ops.

Each accessor call returns a new CadwDataFrame with `{column}_{feature}`
appended (original columns preserved). Bounded per-batch, O(batch),
zero pandas, zero `.as_py()`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from cadw.dataframe import CadwDataFrame


# feature name → (arrow fn, output_type)
_DT_FEATURES: dict[str, tuple] = {
    "year": (pc.year, pa.int64()),
    "month": (pc.month, pa.int64()),
    "day": (pc.day, pa.int64()),
    "hour": (pc.hour, pa.int64()),
    "minute": (pc.minute, pa.int64()),
    "second": (pc.second, pa.int64()),
    "weekday": (pc.day_of_week, pa.int64()),   # Monday=0..Sunday=6
    "day_of_year": (pc.day_of_year, pa.int64()),
    "quarter": (pc.quarter, pa.int64()),
}


def _validate_temporal(df: "CadwDataFrame", column: str) -> None:
    t = df._dataset.schema.field(column).type
    if not (pa.types.is_timestamp(t) or pa.types.is_date(t)):
        raise TypeError(
            f".dt: column {column!r} has type {t}; "
            f"expected timestamp or date"
        )


class _DtAccessor:
    """Pandas-style `df['ts'].dt.*`. Each property = new CadwDataFrame."""

    __slots__ = ("_parent", "_column")

    def __init__(self, parent: "CadwDataFrame", column: str) -> None:
        self._parent = parent
        self._column = column
        _validate_temporal(parent, column)

    def _feature(self, name: str) -> "CadwDataFrame":
        fn, ot = _DT_FEATURES[name]
        return self._parent._apply_column_transform(
            self._column, name, lambda arr: fn(arr), ot, accessor="dt",
        )

    @property
    def year(self) -> "CadwDataFrame":
        return self._feature("year")

    @property
    def month(self) -> "CadwDataFrame":
        return self._feature("month")

    @property
    def day(self) -> "CadwDataFrame":
        return self._feature("day")

    @property
    def hour(self) -> "CadwDataFrame":
        return self._feature("hour")

    @property
    def minute(self) -> "CadwDataFrame":
        return self._feature("minute")

    @property
    def second(self) -> "CadwDataFrame":
        return self._feature("second")

    @property
    def weekday(self) -> "CadwDataFrame":
        return self._feature("weekday")

    @property
    def day_of_year(self) -> "CadwDataFrame":
        return self._feature("day_of_year")

    @property
    def quarter(self) -> "CadwDataFrame":
        return self._feature("quarter")
