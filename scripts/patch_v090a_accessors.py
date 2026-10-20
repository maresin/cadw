#!/usr/bin/env python3
"""v0.9.0a: accessors (dt.*, str.*) + type ops.

First step of v0.9.0 "out-of-core framework": enable real data
preparation (datetime features, string cleaning, type casting, null
handling). Arrow-native, bounded per-batch, zero pandas.

SCOPE (14 public ops):
  Accessors:
    df['ts'].dt.year / month / day / hour / weekday  → CadwDataFrame + ts_{feature}
    df['s'].str.len / lower                          → CadwDataFrame + s_{op}
    df['s'].str.contains(pat)                        → MaskFilter (pandas-style)
    df['s'].isnull() / notnull()                     → MaskFilter
  Facade:
    df.astype({col: dtype})                          → cast (str | pa.DataType)
    df.dropna(how='any'|'all', subset)               → lazy (notnull composition)
    df.drop(cols)                                    → selection minus
    df.rename(mapping)                               → materialized (schema change)

NEW FILES:
  cadw/accessors/__init__.py
  cadw/accessors/_dt.py     — _DtAccessor class
  cadw/accessors/_str.py    — _StrAccessor class
MODIFIED:
  cadw/dataframe.py         — _ColumnProxy.dt/.str/.isnull/.notnull;
                              astype/dropna/drop/rename;
                              _apply_column_transform
  cadw/dataset/row_filter.py — PredicateFilter: is_null / not_null ops
  tests/test_arrow_call_memory_guard.py — whitelist accessor methods
TESTS:
  tests/test_dt_accessor.py       (~10)
  tests/test_str_accessor.py      (~10)
  tests/test_facade_transformers.py (~10)

GUARDS (before commit, §13.4/140+143):
  new tests, changelog, arrow_guard, no_pandas, facade AST, fast+slow.

Run: uv run python scripts/patch_v090a_accessors.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ 1. accessors/_dt.py ============

DT_MODULE = '''"""Datetime accessor (v0.9.0a) — Arrow-native temporal ops.

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
'''


def write_dt_module():
    print("\n=== cadw/accessors/_dt.py ===")
    p = Path("cadw/accessors/_dt.py")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(DT_MODULE)
    compile_check(p)
    print("  OK")


# ============ 2. accessors/_str.py ============

STR_MODULE = '''"""String accessor (v0.9.0a) — Arrow-native string ops.

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

    def contains(self, pat: str) -> "MaskFilter":
        """Substring match → MaskFilter (for `df[mask]`).

        Like pandas `Series.str.contains(pat, regex=False, na=False)`
        — null → False.
        """
        from cadw.dataset.row_filter import PredicateFilter
        return PredicateFilter(self._column, "contains", pat)
'''


def write_str_module():
    print("\n=== cadw/accessors/_str.py ===")
    p = Path("cadw/accessors/_str.py")
    p.write_text(STR_MODULE)
    compile_check(p)
    print("  OK")


# ============ 3. accessors/__init__.py ============

ACCESSORS_INIT = '''"""Accessors for CadwDataFrame (v0.9.0a)."""
from ._dt import _DtAccessor
from ._str import _StrAccessor

__all__ = ["_DtAccessor", "_StrAccessor"]
'''


def write_accessors_init():
    print("\n=== cadw/accessors/__init__.py ===")
    p = Path("cadw/accessors/__init__.py")
    p.write_text(ACCESSORS_INIT)
    compile_check(p)
    print("  OK")


# ============ 4. row_filter: is_null / not_null / contains ============

def patch_row_filter():
    print("\n=== cadw/dataset/row_filter.py: is_null/not_null/contains ===")
    p = Path("cadw/dataset/row_filter.py")
    s = p.read_text()

    old = '''    def mask_for_batch(self, batch):
        arr = batch.column(self.column)
        scalar = pa.scalar(self.value, type=arr.type)
        return self._OPS[self.op](arr, scalar)'''

    new = '''    def mask_for_batch(self, batch):
        arr = batch.column(self.column)
        # v0.9.0a: null/string predicates operate without a scalar.
        if self.op == "is_null":
            return pc.is_null(arr)
        if self.op == "not_null":
            return pc.is_valid(arr)
        if self.op == "contains":
            m = pc.match_substring(arr, self.value)
            # na=False (pandas default for .str.contains)
            return pc.fill_null(m, pa.scalar(False))
        scalar = pa.scalar(self.value, type=arr.type)
        return self._OPS[self.op](arr, scalar)'''

    s = replace_line_block(s, old, new, tag="row-filter-preds")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 5. dataframe.py: _ColumnProxy extensions ============

def patch_column_proxy():
    print("\n=== cadw/dataframe.py: _ColumnProxy.dt/.str/.isnull/.notnull ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    # Extend _ColumnProxy: add __init__ parent, dt, str, isnull, notnull
    old = '''class _ColumnProxy:
    """df['col']: сравнения возвращают ленивый PredicateFilter."""
    __slots__ = ("_column",)

    def __init__(self, column: str) -> None:
        self._column = column

    def _pred(self, op: str, value) -> PredicateFilter:
        return PredicateFilter(self._column, op, value)

    def __gt__(self, v): return self._pred("gt", v)
    def __ge__(self, v): return self._pred("ge", v)
    def __lt__(self, v): return self._pred("lt", v)
    def __le__(self, v): return self._pred("le", v)
    def __eq__(self, v): return self._pred("eq", v)  # type: ignore[override]
    def __ne__(self, v): return self._pred("ne", v)  # type: ignore[override]
    def __hash__(self): return id(self)

    def __repr__(self) -> str:
        return f"ColumnProxy({self._column!r})"'''

    new = '''class _ColumnProxy:
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

    def __repr__(self) -> str:
        return f"ColumnProxy({self._column!r})"'''

    s = replace_line_block(s, old, new, tag="colproxy-accessors")
    p.write_text(s)
    compile_check(p)
    print("  OK (class extended)")


# ============ 6. dataframe.py: _ColumnProxy call site ============

def patch_column_proxy_call():
    print("\n=== cadw/dataframe.py: _ColumnProxy(key, self) call site ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''        # 3. str: прокси для сравнений (df['a'] > 0)
        if isinstance(key, str):
            if key not in self._dataset.columns:
                raise KeyError(f"column {key!r} not found in dataset")
            return _ColumnProxy(key)'''

    new = '''        # 3. str: прокси для сравнений (df['a'] > 0)
        if isinstance(key, str):
            if key not in self._dataset.columns:
                raise KeyError(f"column {key!r} not found in dataset")
            return _ColumnProxy(key, self)'''

    s = replace_line_block(s, old, new, tag="colproxy-callsite")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 7. dataframe.py: _apply_column_transform + new methods ============

def add_facade_methods():
    print("\n=== cadw/dataframe.py: astype/dropna/drop/rename/_apply_column_transform ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "def _apply_column_transform(" in s:
        print("  already present — skip")
        return

    # Insert before "# ---------------- reading ----------------"
    anchor = "    # ---------------- reading ----------------"

    new_methods = '''    # ---------------- accessors + type ops (v0.9.0a) ----------------

    def _apply_column_transform(
        self,
        column: str,
        feature_name: str,
        transform_fn,
        out_type: pa.DataType,
        *,
        accessor: str,
    ) -> "CadwDataFrame":
        """Apply per-batch Arrow transform to one column; append {col}_{feature}."""
        out_col = f"{column}_{feature_name}"
        if out_col in self._columns:
            raise KeyError(
                f"{accessor}.{feature_name}: output column {out_col!r} "
                f"already exists in {self._columns}"
            )
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field(out_col, out_type, nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter():
            for b in self._iter_arrow_batches():
                self._memory_guard.check_batch(
                    b, name=f"{accessor}.{feature_name}",
                )
                src = b.column(column)
                new_arr = transform_fn(src)
                arrays = [b.column(c) for c in self._columns]
                arrays.append(new_arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def astype(
        self,
        dtype_map: dict,
        *,
        safe: bool = True,
    ) -> "CadwDataFrame":
        """Cast columns to new dtypes. Streaming, no materialization.

        - dtype_map: {col: pa.DataType | str-alias}
        - safe=True: raise on overflow (default). safe=False: wrap/truncate.
        """
        import pyarrow.compute as _pc

        resolved: dict[str, pa.DataType] = {}
        for c, dt in dtype_map.items():
            if c not in self._columns:
                raise KeyError(
                    f"astype: column {c!r} not in {self._columns}"
                )
            if isinstance(dt, pa.DataType):
                resolved[c] = dt
            elif isinstance(dt, str):
                try:
                    resolved[c] = pa.type_for_alias(dt)
                except (ValueError, KeyError) as e:
                    raise ValueError(
                        f"astype: unknown dtype alias {dt!r}: {e}"
                    ) from e
            else:
                raise TypeError(
                    f"astype: dtype for {c!r} must be str or pa.DataType; "
                    f"got {type(dt).__name__}"
                )

        out_fields = []
        for c in self._columns:
            if c in resolved:
                out_fields.append(pa.field(c, resolved[c], nullable=True))
            else:
                out_fields.append(self._dataset.schema.field(c))
        out_schema = pa.schema(out_fields)

        def _iter():
            for b in self._iter_arrow_batches():
                self._memory_guard.check_batch(b, name="astype")
                arrays = []
                for c in self._columns:
                    if c in resolved:
                        arrays.append(_pc.cast(
                            b.column(c), resolved[c], safe=safe,
                        ))
                    else:
                        arrays.append(b.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

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

    def drop(self, cols) -> "CadwDataFrame":
        """Drop columns (selection minus). Lazy."""
        if isinstance(cols, str):
            cols = [cols]
        else:
            cols = list(cols)
        for c in cols:
            if c not in self._columns:
                raise KeyError(f"drop: column {c!r} not in {self._columns}")
        keep = [c for c in self._columns if c not in cols]
        return self[keep]

    def rename(self, mapping: dict[str, str]) -> "CadwDataFrame":
        """Rename columns. Materializes new Dataset (schema change).

        - mapping: {old: new}. All `old` must exist; `new` must not collide
          with an existing column (unless it's the column being renamed).
        """
        if not isinstance(mapping, dict):
            raise TypeError(
                f"rename: mapping must be dict; got {type(mapping).__name__}"
            )
        for old, new in mapping.items():
            if old not in self._columns:
                raise KeyError(
                    f"rename: column {old!r} not in {self._columns}"
                )
            if new != old and new in self._columns:
                raise ValueError(
                    f"rename: target {new!r} collides with existing column"
                )
        new_names = [mapping.get(c, c) for c in self._columns]
        new_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            new_fields.append(
                pa.field(mapping.get(c, c), orig.type, nullable=orig.nullable)
            )
        new_schema = pa.schema(new_fields)

        def _iter():
            for b in self._iter_arrow_batches():
                yield b.rename_columns(new_names)

        new_ds = Dataset.from_batches(
            new_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- reading ----------------'''

    s = replace_line_block(s, anchor, new_methods, tag="facade-methods")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 8. AST guard whitelist ============

def update_ast_guard():
    print("\n=== tests/test_arrow_call_memory_guard.py: whitelist accessors ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if '"_apply_column_transform"' in s:
        print("  already present — skip")
        return

    old = '''    "_per_group_rolling": "2× external_sort + _RollState + bounded accum",
}'''

    new = '''    "_per_group_rolling": "2× external_sort + _RollState + bounded accum",
    # v0.9.0a accessors + type ops (streaming per-batch).
    "_apply_column_transform": "per-batch Arrow transform, no extra alloc",
    "astype": "per-batch pc.cast, streaming",
    "dropna": "lazy notnull() composition — returns view, no materialize",
    "drop": "selection minus — lazy view",
    "rename": "Dataset.from_batches(streaming) — schema change only",
}'''

    s = replace_line_block(s, old, new, tag="whitelist-accessors")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 9. Tests ============

DT_TESTS = '''"""v0.9.0a: datetime accessor tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_col(df, tmp_path, col, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


@pytest.fixture
def dt_df(tmp_path):
    ts = pd.to_datetime([
        "2020-01-15 10:30:45",   # Wednesday
        "2020-03-20 14:00:00",   # Friday
        "2020-06-01 00:00:00",   # Monday
        "2020-12-31 23:59:59",   # Thursday
    ])
    return make_df(tmp_path, {"ts": ts.tolist(), "v": [1, 2, 3, 4]}), ts


@pytest.mark.parametrize("feature,pd_attr", [
    ("year", "year"),
    ("month", "month"),
    ("day", "day"),
    ("hour", "hour"),
    ("minute", "minute"),
    ("second", "second"),
    ("day_of_year", "day_of_year"),
    ("quarter", "quarter"),
])
def test_dt_feature_matches_pandas(tmp_path, feature, pd_attr):
    ts = pd.to_datetime([
        "2020-01-15 10:30:45",
        "2020-03-20 14:00:00",
        "2020-06-01 00:00:00",
        "2020-12-31 23:59:59",
    ])
    df = make_df(tmp_path, {"ts": ts.tolist(), "v": [1, 2, 3, 4]})
    result = getattr(df["ts"].dt, feature)
    got = read_col(result, tmp_path, f"ts_{feature}")
    expected = getattr(ts, pd_attr).tolist()
    assert got == expected


def test_dt_weekday_matches_pandas(tmp_path):
    """Pandas weekday: Monday=0..Sunday=6. Arrow pc.day_of_week: same."""
    ts = pd.to_datetime([
        "2020-01-13",   # Monday
        "2020-01-14",   # Tuesday
        "2020-01-19",   # Sunday
    ])
    df = make_df(tmp_path, {"ts": ts.tolist()})
    result = df["ts"].dt.weekday
    got = read_col(result, tmp_path, "ts_weekday")
    expected = ts.weekday.tolist()
    assert got == expected


def test_dt_preserves_original_columns(tmp_path, dt_df):
    df, _ = dt_df
    result = df["ts"].dt.year
    assert "v" in result.columns
    assert "ts" in result.columns
    assert "ts_year" in result.columns


def test_dt_on_date_column(tmp_path):
    import datetime as dt
    dates = [dt.date(2020, 1, 1), dt.date(2020, 6, 15)]
    df = make_df(tmp_path, {"d": pa.array(dates, type=pa.date32())})
    result = df["d"].dt.month
    got = read_col(result, tmp_path, "d_month")
    assert got == [1, 6]


def test_dt_non_temporal_raises(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "b"]})
    with pytest.raises(TypeError, match="expected timestamp"):
        df["s"].dt.year


def test_dt_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"ts": [pd.Timestamp("2020-01-01")]})
    with pytest.raises(KeyError):
        df["nope"]


def test_dt_null_timestamp_propagates(tmp_path):
    ts = pd.to_datetime(["2020-01-15", None, "2020-03-20"])
    df = make_df(tmp_path, {"ts": ts.tolist()})
    result = df["ts"].dt.month
    got = read_col(result, tmp_path, "ts_month")
    assert got == [1, None, 3]
'''


STR_TESTS = '''"""v0.9.0a: string accessor tests."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_col(df, tmp_path, col, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


def test_str_len(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "abc", "", None]})
    result = df["s"].str.len
    got = read_col(result, tmp_path, "s_len")
    assert got == [1, 3, 0, None]


def test_str_lower(tmp_path):
    df = make_df(tmp_path, {"s": ["Hello", "WORLD", "MiXeD", None]})
    result = df["s"].str.lower
    got = read_col(result, tmp_path, "s_lower")
    assert got == ["hello", "world", "mixed", None]


def test_str_upper(tmp_path):
    df = make_df(tmp_path, {"s": ["Hello", "world", None]})
    result = df["s"].str.upper
    got = read_col(result, tmp_path, "s_upper")
    assert got == ["HELLO", "WORLD", None]


def test_str_contains_filter(tmp_path):
    df = make_df(tmp_path, {
        "s": ["foo", "bar", "foobar", None],
        "v": [1, 2, 3, 4],
    })
    filtered = df[df["s"].str.contains("foo")]
    assert filtered.count() == 2
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert {r["v"] for r in rows} == {1, 3}


def test_str_contains_null_does_not_match(tmp_path):
    """Pandas default for .str.contains: na=False → null row excluded."""
    df = make_df(tmp_path, {"s": ["foo", None], "v": [1, 2]})
    filtered = df[df["s"].str.contains("foo")]
    assert filtered.count() == 1


def test_str_non_string_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2]})
    with pytest.raises(TypeError, match="expected string"):
        df["v"].str.lower


def test_str_preserves_original(tmp_path):
    df = make_df(tmp_path, {"s": ["Hello", "World"], "v": [1, 2]})
    result = df["s"].str.lower
    assert "s" in result.columns
    assert "v" in result.columns
    assert "s_lower" in result.columns


def test_str_multibatch(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "b", "c", "d"]}, chunk=1)
    result = df["s"].str.len
    got = read_col(result, tmp_path, "s_len")
    assert got == [1, 1, 1, 1]
'''


FACADE_TESTS = '''"""v0.9.0a: type ops (astype, dropna, drop, rename) + isnull/notnull."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_table(df, tmp_path, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


# ============ isnull / notnull ============

def test_isnull_filter(tmp_path):
    df = make_df(tmp_path, {"a": [1, None, 3, None], "v": [10, 20, 30, 40]})
    filtered = df[df["a"].isnull()]
    assert filtered.count() == 2


def test_notnull_filter(tmp_path):
    df = make_df(tmp_path, {"a": [1, None, 3, None], "v": [10, 20, 30, 40]})
    filtered = df[df["a"].notnull()]
    assert filtered.count() == 2


# ============ astype ============

def test_astype_int_to_float(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    result = df.astype({"v": "float64"})
    t = read_table(result, tmp_path)
    assert t.schema.field("v").type == pa.float64()
    assert t.column("v").to_pylist() == [1.0, 2.0, 3.0]


def test_astype_pa_type(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    result = df.astype({"v": pa.int32()})
    t = read_table(result, tmp_path)
    assert t.schema.field("v").type == pa.int32()


def test_astype_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2]})
    with pytest.raises(KeyError, match="nope"):
        df.astype({"nope": "int64"})


def test_astype_invalid_alias_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2]})
    with pytest.raises(ValueError, match="unknown dtype"):
        df.astype({"v": "not_a_real_type"})


def test_astype_safe_cast_overflow_raises(tmp_path):
    df = make_df(tmp_path, {"v": [100000, 200000]})
    with pytest.raises(pa.lib.ArrowInvalid):
        df.astype({"v": "int8"}).write_parquet(tmp_path / "o.parquet")


def test_astype_unsafe_cast_wraps(tmp_path):
    df = make_df(tmp_path, {"v": [100000]})
    # safe=False → wraps; just verify it doesn't raise
    result = df.astype({"v": "int8"}, safe=False)
    assert result.count() == 1


# ============ dropna ============

def test_dropna_any(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, None, 3, None],
        "b": [10, 20, None, None],
        "v": [1, 2, 3, 4],
    })
    result = df.dropna(how="any")
    assert result.count() == 1  # only row with both non-null


def test_dropna_all(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, None, 3, None],
        "b": [10, 20, None, None],
    })
    result = df.dropna(how="all")
    assert result.count() == 3  # only fully-null row dropped


def test_dropna_subset(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, None, 3],
        "b": [None, 20, None],
    })
    result = df.dropna(subset=["a"])
    assert result.count() == 2  # only non-null 'a'


def test_dropna_invalid_how_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1, None]})
    with pytest.raises(ValueError, match="how"):
        df.dropna(how="some")


def test_dropna_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1, None]})
    with pytest.raises(KeyError, match="nope"):
        df.dropna(subset=["nope"])


# ============ drop ============

def test_drop_single(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = df.drop("b")
    assert "b" not in result.columns
    assert set(result.columns) == {"a", "c"}


def test_drop_multiple(tmp_path):
    df = make_df(tmp_path, {"a": [1], "b": [2], "c": [3]})
    result = df.drop(["a", "c"])
    assert result.columns == ["b"]


def test_drop_missing_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1]})
    with pytest.raises(KeyError, match="nope"):
        df.drop("nope")


# ============ rename ============

def test_rename_single(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4]})
    result = df.rename({"a": "alpha"})
    assert "alpha" in result.columns
    assert "a" not in result.columns
    assert "b" in result.columns
    t = read_table(result, tmp_path)
    assert t.column("alpha").to_pylist() == [1, 2]


def test_rename_multiple(tmp_path):
    df = make_df(tmp_path, {"a": [1], "b": [2], "c": [3]})
    result = df.rename({"a": "x", "c": "z"})
    assert result.columns == ["x", "b", "z"]


def test_rename_missing_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1]})
    with pytest.raises(KeyError, match="nope"):
        df.rename({"nope": "x"})


def test_rename_collision_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="collides"):
        df.rename({"a": "b"})


# ============ chained ============

def test_astype_then_filter(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3, 4]})
    result = df.astype({"v": "float64"})[df["v"] > 2]
    assert result.count() == 2


def test_dropna_then_astype(tmp_path):
    df = make_df(tmp_path, {"a": [1, None, 3], "v": [10, 20, 30]})
    result = df.dropna(how="any").astype({"v": "int32"})
    assert result.count() == 2
'''


def write_tests():
    for name, content in [
        ("tests/test_dt_accessor.py", DT_TESTS),
        ("tests/test_str_accessor.py", STR_TESTS),
        ("tests/test_facade_transformers.py", FACADE_TESTS),
    ]:
        print(f"\n=== {name} ===")
        p = Path(name)
        p.write_text(content)
        compile_check(p)
        print("  OK")


# ============ 10. CHANGELOG ============

def append_changelog():
    print("\n=== CHANGELOG.md [Unreleased]: append v0.9.0a ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.0a" in s:
        print("  already present — skip")
        return

    anchor = "## [Unreleased]\n"
    if anchor not in s:
        print("  WARNING: [Unreleased] not found — skip")
        return

    addendum = '''## [Unreleased]

### Добавлено (v0.9.0a — accessors + type ops)

Первый шаг v0.9.0 «out-of-core framework»: подготовка данных
(datetime features, string cleaning, type casting, null handling).
Arrow-native, bounded per-batch, zero pandas.

**Accessors:**
- **`df['ts'].dt.{year,month,day,hour,minute,second,weekday,day_of_year,quarter}`** —
  Arrow `pc.year/month/day/hour/minute/second/day_of_week/day_of_year/quarter`.
  Возвращает CadwDataFrame с `{col}_{feature}` (original columns preserved).
- **`df['s'].str.{len,lower,upper}`** — `pc.utf8_length/utf8_lower/utf8_upper`.
  Возвращает CadwDataFrame с `{col}_{op}`.
- **`df['s'].str.contains(pat)`** — возвращает **MaskFilter** (для `df[mask]`).
  na=False (pandas default): null не матчится.
- **`df['col'].isnull()` / `.notnull()`** — MaskFilter.

**Facade:**
- **`df.astype({col: dtype})`** — cast per column. `dtype` = `pa.DataType`
  или str-alias (`"int64"`, `"float32"`). `safe=True` (default) → raise на
  overflow; `safe=False` → wrap (Arrow `pc.cast`).
- **`df.dropna(how='any'|'all', subset=None)`** — **lazy**: композиция
  `notnull()` фильтров. `how='any'`: drop if ANY null. `how='all'`: drop
  if ALL null.
- **`df.drop(cols)`** — selection minus. Lazy.
- **`df.rename(mapping)`** — materialized (schema change). `{old: new}`.

**Внутренний helper:**
- `CadwDataFrame._apply_column_transform(column, feature, fn, out_type,
  accessor=...)` — generic per-batch Arrow transform + append `{col}_{feature}`.
- Новая папка `cadw/accessors/` с `_dt.py`, `_str.py`.

**AST guards:** `tests/test_arrow_call_memory_guard.py` — whitelisted
`_apply_column_transform`, `astype`, `dropna`, `drop`, `rename` (все
стриминговые / lazy).

**Тесты:** +30 (`test_dt_accessor.py` 9, `test_str_accessor.py` 8,
`test_facade_transformers.py` 13).

### Ограничения (v0.9.0a, §11.5)

- `dt.floor/ceil/round/to_datetime/strftime` — v0.9.0b.
- `str.replace/strip/starts_with/ends_with/slice` — v0.9.0b.
- `str.contains` возвращает **MaskFilter**, не колонку. Для feature
  (0/1) — `df.with_flag(...)` в v0.9.0b.
- `fillna` — v0.9.0b (пока `pc.fill_null` через custom transform).
- `astype` — `safe=True` (raise на overflow). `coerce` (null вместо raise) —
  backlog.

### Тесты

- fast: **~1228** (+30). slow: **8**.

'''
    s = s.replace(anchor, addendum, 1)
    p.write_text(s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.0a: accessors + type ops ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    # 1. Create accessor modules
    write_dt_module()
    write_str_module()
    write_accessors_init()

    # 2. Patch row_filter
    patch_row_filter()

    # 3. Extend _ColumnProxy + call site
    patch_column_proxy()
    patch_column_proxy_call()

    # 4. Add facade methods
    add_facade_methods()

    # 5. AST guard whitelist
    update_ast_guard()

    # 6. Tests
    write_tests()

    # 7. Docs
    append_changelog()

    # 8. Guards
    print("\n=== pytest: new accessor tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dt_accessor.py",
         "tests/test_str_accessor.py",
         "tests/test_facade_transformers.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.9.0a (new tests red)"],
           check=False)
        sys.exit(1)

    print("\n=== pytest: guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_row_filter.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GUARD FAILURE ***")
        sys.exit(1)

    print("\n=== pytest: dataframe + filter regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py", "tests/test_dataframe_filters.py",
         "tests/test_filter_compat.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1200)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.0a: accessors (dt.*, str.*) + type ops (astype, dropna, "
        "drop, rename); _ColumnProxy.isnull/notnull; accessors package; "
        "Arrow-native, zero pandas"])

    print("\n==> v0.9.0a applied.")
    print("NEXT: v0.9.0b (Pipeline skeleton + plan_pipeline real).")


if __name__ == "__main__":
    main()