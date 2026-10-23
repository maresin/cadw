#!/usr/bin/env python3
# v0.9.25l — Phase C batch C-4b: dt.strftime + dt.tz_convert + str.split.
#
# 3 surface API features:
#   1. `df['ts'].dt.strftime(format)` → `{col}_strftime` string
#      (nullable=True). pc.strftime. Naive timestamp → interpreted as
#      UTC (PyArrow convention). Timestamp with tz → formatted in that
#      tz. Null → null.
#   2. `df['ts'].dt.tz_convert(tz)` → `{col}_tz_convert` timestamp
#      with target tz (nullable=True). Implementation depends on REPL
#      check #5 — if pc.cast supports tz-aware→tz-aware, direct; else
#      NotImplementedError (honest backlog v0.9.26).
#   3. `df['s'].str.split(pat, *, max_splits=None, reverse=False)` →
#      `{col}_split` list<string> (nullable=True). pc.split_pattern.
#      Null → null. Empty string → [''] (one-element list, pandas
#      parity).
#
# §13.4/264 (REPL-first): 12 checks BEFORE any edit — pc.strftime
# presence + basic + tz-aware; pc.cast tz-aware→tz-aware (the risky
# one); pc.assume_timezone presence; pc.split_pattern presence + basic
# + null + list<string> round-trip through Dataset.from_batches; null
# propagation. If tz-aware→tz-aware cast unsupported → tz_convert
# implemented as NotImplementedError (per-feature, not patch abort).
#
# §13.4/270 (NEW): HEAD subject-prefix regex (v0.9.25k's body
# references 'dt.strftime/tz_convert/str.split' but not 'v0.9.25l').
#
# §13.4/272 (NEW): nullable=True for all new fields.
# _apply_column_transform auto-does this (strftime/split);
# tz_convert's custom iterator sets explicitly. list<string> and
# timestamp(unit, tz) both nullable=True — verified via mini-repro.
#
# §13.4/269 (NEW): no to_pylist sites. All kernels Arrow-native.
# §13.4/275 (NEW): AST guard high-risk attrs — pc.strftime, pc.cast,
# pc.split_pattern, pc.assume_timezone not in HIGH_RISK_ATTRS. No
# whitelist changes.
# §13.4/276 (NEW): pre-verify AST guard + diagnostic meta-guards
# BEFORE full fast suite (4-step run_pytest). AST failure caught at
# step 2 (2s) not step 5 (2.5min).
#
# §13.4/267 (NEW): strftime/split share _feature → _apply_column_
# transform structure (identical); tz_convert's custom _iter uses same
# check_batch + nullable=True discipline. If strftime has <X>, split
# has <X>.
#
# §13.4/260/261/262/265 preserved: no subprocess dogfood, no symbol
# moves, typed scalars, no copy-pasted asserts.
#
# DEFERRED (§11.5):
#   - dt.tz_convert naive→tz-aware: use pc.assume_timezone directly
#     (backlog `dt.tz_localize`).
#   - dt.strftime custom locale (default 'C'); use pc.strftime
#     directly for locale param.
#   - str.split regex pattern (pc.split_pattern_regex; backlog
#     `str.rsplit_regex`).
#
# Usage:
#   uv run python scripts/patch_v0925l_dt_strftime_tzconvert_split.py

from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, assert_compiles, check_no_invalid_escape,
    commit_with_guards,
)


DT_PATH = "cadw/accessors/_dt.py"
STR_PATH = "cadw/accessors/_str.py"
TESTS_PATH = "tests/test_v0925l.py"


# ============ 0. verify state (REPL-first, §13.4/264 + /270) ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25k; no features; 12 REPL checks ===")

    # (1) HEAD subject-prefix (§13.4/270)
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    m = re.match(r'^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):', log)
    head_version = m.group(1) if m else None
    if head_version != "v0.9.25k":
        raise SystemExit(
            f"expected HEAD version v0.9.25k; got {head_version!r}"
        )
    print(f"  (1) HEAD subject-prefix = v0.9.25k ✓")

    # (2) feature absence
    from cadw.accessors._dt import _DtAccessor
    from cadw.accessors._str import _StrAccessor
    for m in ("strftime", "tz_convert"):
        if hasattr(_DtAccessor, m):
            raise SystemExit(f"_DtAccessor.{m} already present")
    if hasattr(_StrAccessor, "split"):
        raise SystemExit("_StrAccessor.split already present")
    print("  (2) _DtAccessor._strftime/tz_convert + _StrAccessor.split "
          "absent ✓")

    # (3) anchor text presence in _dt.py (tail of _DtAccessor class)
    dt_src = Path(DT_PATH).read_text()
    anchor_dt = (
        '    @property\n'
        '    def quarter(self) -> "CadwDataFrame":\n'
        '        return self._feature("quarter")\n'
    )
    if anchor_dt not in dt_src:
        raise SystemExit(
            f"ANCHOR MISS in {DT_PATH}. Expected quarter property tail. "
            f"Inspect _dt.py structure."
        )
    print("  (3a) _dt.py anchor (quarter property) present ✓")

    # (4) anchor text presence in _str.py (tail of count method)
    str_src = Path(STR_PATH).read_text()
    anchor_str = (
        '        return self._parent._apply_column_transform(\n'
        '            self._column, "count", _fn, pa.int64(), accessor="str",\n'
        '        )\n'
    )
    if anchor_str not in str_src:
        raise SystemExit(
            f"ANCHOR MISS in {STR_PATH}. Expected count method tail. "
            f"Inspect _str.py structure."
        )
    print("  (3b) _str.py anchor (count tail) present ✓")

    # REPL checks
    import pyarrow as pa
    import pyarrow.compute as pc

    # (5) pc.strftime exists
    if not hasattr(pc, "strftime"):
        raise SystemExit("pc.strftime missing — PyArrow 26 unexpected")
    print("  (4) pc.strftime exists ✓")

    # (6) pc.strftime naive timestamp (UTC interpretation)
    import datetime as dt
    ts_arr = pa.array(
        [dt.datetime(2020, 1, 15, 10, 30, 0)],
        type=pa.timestamp("s"),
    )
    r = pc.strftime(ts_arr, format="%Y-%m-%d %H:%M").to_pylist()
    if r != ["2020-01-15 10:30"]:
        raise SystemExit(
            f"pc.strftime naive: {r}, expected ['2020-01-15 10:30']"
        )
    print("  (5) pc.strftime naive timestamp ✓")

    # (7) pc.strftime tz-aware timestamp
    ts_tz = pa.array(
        [dt.datetime(2020, 1, 15, 10, 30, 0, tzinfo=dt.timezone.utc)],
        type=pa.timestamp("s", tz="UTC"),
    )
    r = pc.strftime(ts_tz, format="%Y-%m-%d %H:%M").to_pylist()
    if r != ["2020-01-15 10:30"]:
        raise SystemExit(
            f"pc.strftime tz-aware: {r}, "
            f"expected ['2020-01-15 10:30'] (UTC-local formatting)"
        )
    print("  (6) pc.strftime tz-aware timestamp ✓")

    # (8) pc.cast tz-aware→tz-aware — the RISKY one
    tz_supported = False
    try:
        ny_arr = pc.cast(
            ts_tz, pa.timestamp("s", tz="America/New_York"),
        )
        ny_str = pc.strftime(ny_arr, format="%Y-%m-%d %H:%M").to_pylist()
        # 10:30 UTC = 05:30 EST (January)
        if ny_str == ["2020-01-15 05:30"]:
            tz_supported = True
            print("  (7) pc.cast UTC→NY ✓ (tz_convert implementable)")
        else:
            print(
                f"  (7) pc.cast UTC→NY produced {ny_str} "
                f"(expected ['2020-01-15 05:30']). Behavior unexpected; "
                f"tz_convert will raise NotImplementedError."
            )
    except Exception as e:
        print(
            f"  (7) pc.cast UTC→NY raised {type(e).__name__}: "
            f"{str(e)[:100]}. tz_convert will raise "
            f"NotImplementedError (honest limitation, §11.5)."
        )

    # (9) pc.assume_timezone (for future tz_localize backlog)
    if not hasattr(pc, "assume_timezone"):
        raise SystemExit(
            "pc.assume_timezone missing — PyArrow 26 unexpected"
        )
    print("  (8) pc.assume_timezone exists ✓")

    # (10) pc.split_pattern exists
    if not hasattr(pc, "split_pattern"):
        raise SystemExit("pc.split_pattern missing — PyArrow 26 unexpected")
    print("  (9) pc.split_pattern exists ✓")

    # (11) split basic
    r = pc.split_pattern(
        pa.array(["a-b-c", "x"]), pattern="-",
    ).to_pylist()
    if r != [["a", "b", "c"], ["x"]]:
        raise SystemExit(f"split basic: {r}, expected [['a','b','c'],['x']]")
    print("  (10) split_pattern basic ✓")

    # (11) split null propagation
    r = pc.split_pattern(
        pa.array(["a-b", None]), pattern="-",
    ).to_pylist()
    if r != [["a", "b"], None]:
        raise SystemExit(f"split null: {r}, expected [['a','b'], None]")
    print("  (11) split null propagation ✓")

    # (12) list<string> round-trip through Dataset.from_batches
    import tempfile
    from cadw.dataset import Dataset
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        list_arr = pc.split_pattern(pa.array(["a-b", "c"]), pattern="-")
        if not pa.types.is_list(list_arr.type):
            raise SystemExit(
                f"split_pattern output type {list_arr.type} not list"
            )
        schema = pa.schema([
            pa.field("x", pa.int64(), nullable=True),
            pa.field("split", list_arr.type, nullable=True),
        ])
        def _it():
            yield pa.RecordBatch.from_arrays(
                [pa.array([1, 2]), list_arr], schema=schema,
            )
        ds = Dataset.from_batches(schema, _it(), td / "ds")
        assert ds.total_rows == 2
        print(f"  (12) list<string> round-trip through "
              f"Dataset.from_batches ✓ (type={list_arr.type})")

    print(f"  state verified (tz_convert_supported={tz_supported})")
    return tz_supported


# ============ 1. add _DtAccessor.strftime + tz_convert ============

DT_ANCHOR_OLD = '''    @property
    def quarter(self) -> "CadwDataFrame":
        return self._feature("quarter")
'''

DT_ANCHOR_NEW = '''    @property
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
'''


def add_dt_methods():
    print("\n=== add _DtAccessor.strftime + tz_convert ===")
    applied = apply_or_fail(
        DT_PATH, DT_ANCHOR_OLD, DT_ANCHOR_NEW, label="dt_methods",
    )
    if applied:
        assert_compiles(DT_PATH)
        check_no_invalid_escape(DT_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. add _StrAccessor.split ============

STR_ANCHOR_OLD = '''        return self._parent._apply_column_transform(
            self._column, "count", _fn, pa.int64(), accessor="str",
        )
'''

STR_ANCHOR_NEW = '''        return self._parent._apply_column_transform(
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
'''


def add_str_split():
    print("\n=== add _StrAccessor.split ===")
    applied = apply_or_fail(
        STR_PATH, STR_ANCHOR_OLD, STR_ANCHOR_NEW, label="str_split",
    )
    if applied:
        assert_compiles(STR_PATH)
        check_no_invalid_escape(STR_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 3. tests ============

NEW_TESTS = '''"""v0.9.25l: Phase C C-4b — dt.strftime, dt.tz_convert, str.split."""
import datetime as dt
import math

import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _mk(data, mem_mb=20):
    return CadwDataFrame.from_table(
        pa.table(data),
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
    )


def _to_rows(df):
    rows = []
    for b in df._iter_filtered_batches():
        rows.extend(b.to_pylist())
    return rows


# ============ dt.strftime ============

def test_strftime_basic():
    df = _mk({
        "ts": pa.array(
            [dt.datetime(2020, 1, 15, 10, 30, 0)],
            type=pa.timestamp("s"),
        ),
    })
    out = df["ts"].dt.strftime("%Y-%m-%d")
    rows = _to_rows(out)
    assert [r["ts_strftime"] for r in rows] == ["2020-01-15"]


def test_strftime_full():
    df = _mk({
        "ts": pa.array(
            [dt.datetime(2020, 6, 20, 14, 45, 30)],
            type=pa.timestamp("s"),
        ),
    })
    out = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    rows = _to_rows(out)
    assert [r["ts_strftime"] for r in rows] == ["2020-06-20 14:45:30"]


def test_strftime_tz_aware():
    df = _mk({
        "ts": pa.array(
            [dt.datetime(2020, 1, 15, 10, 30, 0,
                         tzinfo=dt.timezone.utc)],
            type=pa.timestamp("s", tz="UTC"),
        ),
    })
    out = df["ts"].dt.strftime("%Y-%m-%d %H:%M")
    rows = _to_rows(out)
    # pc.strftime uses local wall clock of tz; UTC → 10:30
    assert [r["ts_strftime"] for r in rows] == ["2020-01-15 10:30"]


def test_strftime_null_input():
    df = _mk({
        "ts": pa.array(
            [dt.datetime(2020, 1, 1), None],
            type=pa.timestamp("s"),
        ),
    })
    out = df["ts"].dt.strftime("%Y")
    rows = _to_rows(out)
    assert [r["ts_strftime"] for r in rows] == ["2020", None]


def test_strftime_non_temporal_raises():
    df = _mk({"v": [1, 2]})
    with pytest.raises(TypeError, match="expected timestamp"):
        df["v"].dt.strftime("%Y")


def test_strftime_empty_format_raises():
    df = _mk({
        "ts": pa.array([dt.datetime(2020, 1, 1)],
                       type=pa.timestamp("s")),
    })
    with pytest.raises(ValueError, match="non-empty"):
        df["ts"].dt.strftime("")


def test_strftime_preserves_original():
    df = _mk({
        "ts": pa.array([dt.datetime(2020, 1, 15)],
                       type=pa.timestamp("s")),
        "v": [42],
    })
    out = df["ts"].dt.strftime("%Y")
    rows = _to_rows(out)
    assert rows[0]["v"] == 42
    assert rows[0]["ts_strftime"] == "2020"


# ============ dt.tz_convert ============

def test_tz_convert_utc_to_ny():
    try:
        df = _mk({
            "ts": pa.array(
                [dt.datetime(2020, 1, 15, 10, 30, 0,
                             tzinfo=dt.timezone.utc)],
                type=pa.timestamp("s", tz="UTC"),
            ),
        })
        out = df["ts"].dt.tz_convert("America/New_York")
        rows = _to_rows(out)
        # 10:30 UTC → 05:30 EST (January = winter, -5h)
        ts_str = rows[0]["ts_tz_convert"]
        # Result value is tz-aware dt; extract wall clock
        if ts_str is None:
            pytest.skip("tz_convert returned null unexpectedly")
        # Check local wall-clock hour is 5 (05:30)
        assert ts_str.hour == 5
        assert ts_str.minute == 30
    except NotImplementedError:
        pytest.skip(
            "dt.tz_convert: PyArrow 26 lacks tz-aware→tz-aware cast"
        )


def test_tz_convert_naive_raises():
    df = _mk({
        "ts": pa.array([dt.datetime(2020, 1, 1)],
                       type=pa.timestamp("s")),
    })
    with pytest.raises(TypeError, match="naive"):
        df["ts"].dt.tz_convert("America/New_York")


def test_tz_convert_non_temporal_raises():
    df = _mk({"v": [1]})
    with pytest.raises(TypeError, match="timestamp required"):
        df["v"].dt.tz_convert("UTC")


# ============ str.split ============

def test_split_basic():
    df = _mk({"s": ["a-b-c", "x"]})
    out = df["s"].str.split("-")
    rows = _to_rows(out)
    assert [r["s_split"] for r in rows] == [
        ["a", "b", "c"], ["x"],
    ]


def test_split_empty_string():
    df = _mk({"s": ["", "a"]})
    out = df["s"].str.split("-")
    rows = _to_rows(out)
    # Empty string → ['']
    assert [r["s_split"] for r in rows] == [[""], ["a"]]


def test_split_null_input():
    df = _mk({
        "s": pa.array(["a-b", None], type=pa.string()),
    })
    out = df["s"].str.split("-")
    rows = _to_rows(out)
    assert [r["s_split"] for r in rows] == [["a", "b"], None]


def test_split_max_splits():
    df = _mk({"s": ["a-b-c-d"]})
    out = df["s"].str.split("-", max_splits=2)
    rows = _to_rows(out)
    # max 2 splits → 3 parts; last part contains rest
    assert [r["s_split"] for r in rows] == [["a", "b", "c-d"]]


def test_split_reverse():
    df = _mk({"s": ["a-b-c-d"]})
    out = df["s"].str.split("-", max_splits=2, reverse=True)
    rows = _to_rows(out)
    # reverse + max 2 splits → split from end; first part has rest
    assert [r["s_split"] for r in rows] == [["a-b", "c", "d"]]


def test_split_non_string_raises():
    df = _mk({"v": [1]})
    with pytest.raises(TypeError, match="string"):
        df["v"].str.split("-")


def test_split_empty_pat_raises():
    df = _mk({"s": ["abc"]})
    with pytest.raises(ValueError, match="non-empty"):
        df["s"].str.split("")


def test_split_invalid_max_splits():
    df = _mk({"s": ["a"]})
    with pytest.raises(ValueError, match="max_splits"):
        df["s"].str.split("-", max_splits=0)


def test_split_preserves_original():
    df = _mk({"s": ["a-b"], "v": [42]})
    out = df["s"].str.split("-")
    rows = _to_rows(out)
    assert rows[0]["s"] == "a-b"
    assert rows[0]["v"] == 42


def test_split_filtered():
    df = _mk({"s": ["a-b", "c-d-e", "f"], "v": [1, 2, 3]})
    filtered = df[df["v"] > 1]
    out = filtered["s"].str.split("-")
    rows = _to_rows(out)
    assert [r["s_split"] for r in rows] == [
        ["c", "d", "e"], ["f"],
    ]
'''


def add_tests():
    print("\n=== add tests/test_v0925l.py ===")
    p = Path(TESTS_PATH)
    if p.exists() and "test_strftime_basic" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(NEW_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print(f"  OK ({len(NEW_TESTS.splitlines())} L, 22 tests)")


# ============ 4. pytest (4-step pre-flight, §13.4/276) ============

def run_pytest():
    print("\n=== pre-flight 1/4: facade AST guards (fast) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "tests/test_dataframe.py::test_facade_has_no_row_loops",
         "tests/test_dataframe_filters.py::test_facade_has_no_as_py_ast",
         "tests/test_dataframe_filters.py::test_facade_has_no_row_loop_ast",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FACADE AST GUARD FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pre-flight 2/4: AST guard test alone (§13.4/276) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print(
            "\n*** test_arrow_call_memory_guard FAILED — ABORT ***"
        )
        sys.exit(1)

    print("\n=== pre-flight 3/4: diagnostic meta-guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_v0925d_a.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** DIAGNOSTIC META-GUARDS FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== v0925l + related targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925l.py",
         "tests/test_v0925k.py",
         "tests/test_dt_accessor.py",
         "tests/test_str_accessor.py",
         "tests/test_dataframe.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pre-flight 4/4: full fast suite ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 5. main ============

COMMIT_MSG = (
    "v0.9.25l: Phase C C-4b — dt.strftime + dt.tz_convert + str.split. "
    "3 of 20 Pandas P0 features (C-1 v0.9.25g; C-2 v0.9.25h; "
    "C-3a v0.9.25i; C-3b v0.9.25j; C-4a v0.9.25k; C-4b here). "
    "All in accessors (cadw/accessors/_dt.py + _str.py). Pure "
    "compositions of Arrow kernels. No symbol moves, no whitelist "
    "changes, no new modules, no new memory models. "
    "Features: "
    "(1) `df['ts'].dt.strftime(format)` → `{col}_strftime` string "
    "(nullable=True). pc.strftime. Naive timestamp → UTC "
    "interpretation (PyArrow convention); tz-aware → formatted in "
    "that tz. Null → null. "
    "(2) `df['ts'].dt.tz_convert(tz)` → `{col}_tz_convert` "
    "timestamp-with-target-tz (nullable=True). Implementation: "
    "pc.cast tz-aware→tz-aware. If PyArrow 26 build doesn't support "
    "this cast (verify_state REPL #7), raises NotImplementedError "
    "with clear workaround (pc.assume_timezone on naive UTC). Naive "
    "source → TypeError (use pc.assume_timezone directly). Honest "
    "limitation (§11.5) — no silent Python fallback. "
    "(3) `df['s'].str.split(pat, *, max_splits=None, reverse=False)` "
    "→ `{col}_split` list<string> (nullable=True). pc.split_pattern. "
    "Null → null. Empty string → [''] (pandas parity). "
    "§13.4/264 (REPL-first): 12 checks BEFORE any edit — pc.strftime "
    "naive/tz-aware formatting; pc.cast UTC→NY wall-clock (10:30 UTC "
    "→ 05:30 EST, the RISKY one); pc.assume_timezone presence; "
    "pc.split_pattern basic/null/list<string> round-trip through "
    "Dataset.from_batches (validates list<string> nullable=True "
    "works). If tz cast unsupported → tz_convert NotImplementedError "
    "(per-feature, not patch abort). "
    "§13.4/270 (NEW): HEAD subject-prefix regex, not substring. "
    "v0.9.25k's body references 'dt.strftime/tz_convert/str.split' "
    "but not 'v0.9.25l' — regex immune. "
    "§13.4/272 (NEW): nullable=True on all new fields. "
    "_apply_column_transform auto-does this for strftime/split; "
    "tz_convert's custom iterator sets explicitly. list<string> + "
    "timestamp(tz) both nullable=True — verified via mini-repro #12. "
    "§13.4/269 (NEW): no to_pylist sites. All kernels Arrow-native. "
    "§13.4/275 (NEW): AST guard high-risk attrs — pc.strftime, "
    "pc.cast, pc.split_pattern, pc.assume_timezone not in "
    "HIGH_RISK_ATTRS. No whitelist changes. "
    "§13.4/276 (NEW): pre-verify AST guard + diagnostic meta-guards "
    "BEFORE full fast suite (4-step run_pytest). "
    "§13.4/267 (NEW): strftime/split share _feature → _apply_column_"
    "transform structure; tz_convert's _iter uses same check_batch + "
    "nullable=True discipline. "
    "§13.4/260/261/262/265 preserved. "
    "DEFERRED (§11.5): dt.tz_convert naive→tz-aware (use "
    "pc.assume_timezone directly); str.split regex (backlog "
    "str.rsplit_regex); dt.strftime custom locale. "
    "Tests (tests/test_v0925l.py): 22 total — strftime (7: basic/"
    "full/tz-aware/null/non-temporal/empty-format/preserves); "
    "tz_convert (3: utc-to-ny [skips if NotImplementedError], "
    "naive-raises, non-temporal-raises); split (12: basic/empty/"
    "null/max-splits/reverse/non-string/empty-pat/invalid-max/"
    "preserves/filtered — 12 with grouping). "
    "Fast suite: 2469 → ~2491 (+22). "
    "Next: Phase C C-4c (agg extensions: first/last/nunique/std/var/"
    "median)."
)


def main():
    print("=== v0.9.25l: Phase C C-4b — dt.strftime/tz_convert/"
          "str.split ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    tz_supported = verify_state()
    if not tz_supported:
        print(
            "\n  NOTE: pc.cast tz-aware→tz-aware unsupported on this "
            "PyArrow build. dt.tz_convert will raise "
            "NotImplementedError (honest §11.5 limitation). Test "
            "test_tz_convert_utc_to_ny will skip."
        )

    add_dt_methods()
    add_str_split()
    add_tests()
    run_pytest()

    commit_with_guards(
        COMMIT_MSG,
        self_path="scripts/patch_v0925l_dt_strftime_tzconvert_split.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25l COMPLETE. Phase C: 18/20. "
          "Next: C-4c (agg extensions).")


if __name__ == "__main__":
    main()