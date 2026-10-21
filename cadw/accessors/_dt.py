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


def _validate_timestamp_only(df: "CadwDataFrame", column: str) -> None:
    """v0.9.9 (N2): floor/ceil/round require timestamp (not date).

    date32/64 has no `.unit` attribute; `_freq_bucket` would raise
    AttributeError mid-iteration instead of clear TypeError in init.
    Only timestamp types supported for floor/ceil/round.
    """
    t = df._dataset.schema.field(column).type
    if not pa.types.is_timestamp(t):
        raise TypeError(
            f".dt.floor/ceil/round: column {column!r} has type {t}; "
            f"expected timestamp (date32/64 not supported — use "
            f"cast to timestamp first)"
        )


def _freq_bucket(arr, freq_str, mode):
    # Bucketize timestamp array via int64 math.
    # mode in {"floor", "ceil", "round"} (round half-up).
    from cadw.ops.resample import parse_freq, _TS_UNIT_FACTOR
    if not pa.types.is_timestamp(arr.type):
        raise TypeError(
            "_freq_bucket: expected timestamp; got "
            + str(arr.type)
        )
    freq_sec = parse_freq(freq_str)
    factor = _TS_UNIT_FACTOR[arr.type.unit]
    int_ns = pc.cast(arr, pa.int64())
    secs = pc.cast(
        pc.floor(pc.divide(int_ns,
                        pa.scalar(factor, type=pa.int64()))),
        pa.int64(),
    )
    fq = pa.scalar(freq_sec, type=pa.int64())
    div = pc.divide(secs, fq)
    base = pc.multiply(div, fq)
    if mode == "floor":
        bucket = base
    elif mode == "ceil":
        rem = pc.subtract(secs, base)
        add = pc.if_else(
            pc.greater(rem, pa.scalar(0, type=pa.int64())),
            fq,
            pa.scalar(0, type=pa.int64()),
        )
        bucket = pc.add(base, add)
    elif mode == "round":
        half = freq_sec // 2
        shifted = pc.add(secs, pa.scalar(half, type=pa.int64()))
        bucket = pc.multiply(pc.divide(shifted, fq), fq)
    else:
        raise ValueError("_freq_bucket: bad mode " + repr(mode))
    scaled = pc.multiply(bucket, pa.scalar(factor, type=pa.int64()))
    return pc.cast(scaled, arr.type)

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

    def _feature_custom(self, name, fn, out_type):
        # v0.9.4b-2-fix1: _feature(name) uses static _DT_FEATURES;
        # _feature_custom(name, fn, out_type) allows per-call fn+type
        # (floor/ceil/round/week_of_year). Symmetric to str._feature.
        return self._parent._apply_column_transform(
            self._column, name, fn, out_type, accessor="dt",
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

    def floor(self, freq: str) -> "CadwDataFrame":
        _validate_timestamp_only(self._parent, self._column)
        return self._feature_custom(
            "floor",
            lambda arr: _freq_bucket(arr, freq, "floor"),
            self._parent._dataset.schema.field(self._column).type,
        )

    def ceil(self, freq: str) -> "CadwDataFrame":
        _validate_timestamp_only(self._parent, self._column)
        return self._feature_custom(
            "ceil",
            lambda arr: _freq_bucket(arr, freq, "ceil"),
            self._parent._dataset.schema.field(self._column).type,
        )

    def round(self, freq: str) -> "CadwDataFrame":
        _validate_timestamp_only(self._parent, self._column)
        return self._feature_custom(
            "round",
            lambda arr: _freq_bucket(arr, freq, "round"),
            self._parent._dataset.schema.field(self._column).type,
        )

    @property
    def week_of_year(self) -> "CadwDataFrame":
        # ISO week (1-53). For US week numbering -- backlog.
        return self._feature_custom(
            "week_of_year",
            lambda arr: pc.iso_week(arr),
            pa.int16(),
        )
    @property
    def quarter(self) -> "CadwDataFrame":
        return self._feature("quarter")
