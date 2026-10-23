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

    def strftime(self, format: str) -> "CadwDataFrame":
        """Format timestamp as string (v0.9.25l).

        Args:
            format: strftime format string (e.g. '%Y-%m-%d %H:%M').
                Default pc.strftime format is '%Y-%m-%dT%H:%M:%S'
                if not provided; we require explicit format for
                readability.

        Returns: augmented CadwDataFrame with `{col}_strftime` string
        column (nullable=True). Null → null.

        Naive timestamp → interpreted as UTC (PyArrow convention).
        Timestamp with tz → formatted in that tz (pc.strftime uses
        local wall clock of the tz).

        Locale: 'C' (default). For other locales, use pc.strftime
        directly. §11.5.
        """
        import pyarrow.compute as _pc
        if not isinstance(format, str):
            raise TypeError(
                f"dt.strftime: format must be str; got "
                f"{type(format).__name__}"
            )
        if not format:
            raise ValueError("dt.strftime: format must be non-empty")

        def _fn(arr):
            return _pc.strftime(arr, format=format)

        return self._parent._apply_column_transform(
            self._column, "strftime", _fn, pa.string(), accessor="dt",
        )

    def tz_convert(self, tz: str) -> "CadwDataFrame":
        """Convert tz-aware timestamp to target timezone (v0.9.25l).

        Args:
            tz: target timezone string (e.g. 'America/New_York').

        Returns: augmented CadwDataFrame with `{col}_tz_convert`
        timestamp-with-target-tz column (nullable=True). Null → null.

        Requires source column to be tz-aware (timestamp with tz).
        For naive → tz-aware, use pc.assume_timezone directly
        (backlog `dt.tz_localize`, §11.5).

        Implementation: pc.cast(tz-aware→tz-aware). If PyArrow 26
        does not support this cast (verify_state REPL check #7),
        raises NotImplementedError with clear message. Honest
        limitation — no silent fallback to Python loop (violates
        §3.10 out-of-core).
        """
        import pyarrow.compute as _pc
        if not isinstance(tz, str):
            raise TypeError(
                f"dt.tz_convert: tz must be str; got {type(tz).__name__}"
            )
        if not tz:
            raise ValueError("dt.tz_convert: tz must be non-empty")

        src_type = self._parent._dataset.schema.field(
            self._column
        ).type
        if not pa.types.is_timestamp(src_type):
            raise TypeError(
                f"dt.tz_convert: column {self._column!r} has type "
                f"{src_type}; timestamp required"
            )
        if src_type.tz is None:
            raise TypeError(
                f"dt.tz_convert: column {self._column!r} is naive "
                f"(no tz); use pc.assume_timezone directly for "
                f"naive→tz-aware (backlog dt.tz_localize, §11.5)"
            )

        target_type = pa.timestamp(src_type.unit, tz=tz)

        # Probe: is this cast supported by PyArrow?
        try:
            _probe = _pc.cast(
                pa.array(
                    [None], type=pa.timestamp(src_type.unit, tz="UTC"),
                ),
                target_type,
            )
        except (pa.ArrowNotImplementedError, pa.ArrowInvalid) as e:
            raise NotImplementedError(
                f"dt.tz_convert: pc.cast between tz-aware timestamps "
                f"is not supported by this PyArrow build ({type(e).__name__}: "
                f"{str(e)[:120]}). Backlog v0.9.26. Workaround: use "
                f"pc.assume_timezone after converting to naive UTC via "
                f"pc.cast(arr, pa.timestamp(unit)) — this yields UTC "
                f"naive (not local), which is correct for the "
                f"interpretation step."
            ) from e

        out_fields = [
            self._parent._dataset.schema.field(c)
            for c in self._parent._columns
        ]
        out_fields.append(
            pa.field(f"{self._column}_tz_convert", target_type,
                     nullable=True)
        )
        out_schema = pa.schema(out_fields)

        def _iter():
            for b in self._parent._iter_arrow_batches():
                self._parent._memory_guard.check_batch(
                    b, name="dt.tz_convert.input", extra_factor=2.0,
                )
                converted = _pc.cast(b.column(self._column), target_type)
                arrays = [
                    b.column(c) for c in self._parent._columns
                ]
                arrays.append(converted)
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        from cadw.dataset import Dataset as _DS
        from cadw.dataframe import CadwDataFrame as _CDF
        new_ds = _DS.from_batches(
            out_schema, _iter(), self._parent._new_dataset_dir(),
        )
        return _CDF(
            new_ds,
            memory_guard=self._parent._memory_guard,
            spill_dir=self._parent._spill_dir,
        )
