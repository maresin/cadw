#!/usr/bin/env python3
# v0.9.4b-2: _GroupBy.transform(str) + describe + dt/str extras.
#
# _GroupBy.transform(agg_columns, agg_funcs):
#   Aggregate + left-merge back. Composition of _groupby_impl + merge.
#   LIMITATION: row order = partition-concat (merge semantics), not
#   original. Documented 11.5. For pandas order: add __orig_rid + sort.
# describe(): numeric summary (count/mean/std/min/quartiles/max).
# dt.floor/ceil/round(freq) + week_of_year (Arrow compute + int64 math).
# str.strip/lstrip/rstrip/starts_with/ends_with/replace.
# PredicateFilter extended: starts_with/ends_with ops.
#
# Run: uv run python scripts/patch_v094b2_extras.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    write_content, assert_compiles, commit_with_guards,
)


# ============ 1. PredicateFilter: starts_with / ends_with ============

def patch_predicate_filter():
    print("\n=== row_filter.py: starts_with / ends_with ===")
    p = Path("cadw/dataset/row_filter.py")
    s = p.read_text()
    if '"starts_with"' in s:
        print("  already present - skip")
        return
    old_ops = (
        '    _OPS_NO_SCALAR = frozenset({"is_null", "not_null", "contains"})'
    )
    new_ops = (
        '    _OPS_NO_SCALAR = frozenset({'
        '"is_null", "not_null", "contains", '
        '"starts_with", "ends_with"})'
    )
    if old_ops not in s:
        raise SystemExit("_OPS_NO_SCALAR anchor not found")
    s = s.replace(old_ops, new_ops, 1)

    old_body = (
        '        if self.op == "contains":\n'
        '            m = pc.match_substring(arr, self.value)\n'
        '            # na=False (pandas default for .str.contains)\n'
        '            return pc.fill_null(m, pa.scalar(False))\n'
    )
    new_body = old_body + (
        '        if self.op == "starts_with":\n'
        '            m = pc.starts_with(arr, self.value)\n'
        '            return pc.fill_null(m, pa.scalar(False))\n'
        '        if self.op == "ends_with":\n'
        '            m = pc.ends_with(arr, self.value)\n'
        '            return pc.fill_null(m, pa.scalar(False))\n'
    )
    if old_body not in s:
        raise SystemExit("mask_for_batch contains-anchor not found")
    s = s.replace(old_body, new_body, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. _StrAccessor: strip / starts_with / replace ============

def patch_str_accessor():
    print("\n=== accessors/_str.py: strip/starts_with/replace ===")
    p = Path("cadw/accessors/_str.py")
    s = p.read_text()
    if "def starts_with(self, pat" in s:
        print("  already present - skip")
        return
    anchor = '    def contains(self, pat: str) -> "MaskFilter":\n'
    if anchor not in s:
        raise SystemExit("contains anchor not found in _str.py")

    L = []
    L.append('    @property')
    L.append('    def strip(self) -> "CadwDataFrame":')
    L.append('        return self._feature(')
    L.append('            "strip", lambda arr: pc.utf8_trim_whitespace(arr),')
    L.append('            pa.string(),')
    L.append('        )')
    L.append('')
    L.append('    @property')
    L.append('    def lstrip(self) -> "CadwDataFrame":')
    L.append('        return self._feature(')
    L.append('            "lstrip", lambda arr: pc.utf8_ltrim_whitespace(arr),')
    L.append('            pa.string(),')
    L.append('        )')
    L.append('')
    L.append('    @property')
    L.append('    def rstrip(self) -> "CadwDataFrame":')
    L.append('        return self._feature(')
    L.append('            "rstrip", lambda arr: pc.utf8_rtrim_whitespace(arr),')
    L.append('            pa.string(),')
    L.append('        )')
    L.append('')
    L.append('    def starts_with(self, pat: str) -> "MaskFilter":')
    L.append('        from cadw.dataset.row_filter import PredicateFilter')
    L.append('        return PredicateFilter(self._column, "starts_with", pat)')
    L.append('')
    L.append('    def ends_with(self, pat: str) -> "MaskFilter":')
    L.append('        from cadw.dataset.row_filter import PredicateFilter')
    L.append('        return PredicateFilter(self._column, "ends_with", pat)')
    L.append('')
    L.append('    def replace(self, pat: str, repl: str, *,')
    L.append('                regex: bool = False) -> "CadwDataFrame":')
    L.append('        # Appends {col}_replaced. regex=False default (literal).')
    L.append('        if regex:')
    L.append('            fn = lambda arr: pc.replace_substring_regex(arr, pat, repl)')
    L.append('        else:')
    L.append('            fn = lambda arr: pc.replace_substring(arr, pat, repl)')
    L.append('        return self._feature("replaced", fn, pa.string())')
    L.append('')
    block = "\n".join(L)
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. _DtAccessor: floor/ceil/round/week_of_year ============

def patch_dt_accessor():
    print("\n=== accessors/_dt.py: floor/ceil/round/week_of_year ===")
    p = Path("cadw/accessors/_dt.py")
    s = p.read_text()
    if "def floor(self, freq" in s:
        print("  already present - skip")
        return
    anchor = '    @property\n    def quarter(self) -> "CadwDataFrame":\n'
    if anchor not in s:
        raise SystemExit("quarter anchor not found in _dt.py")

    # Helper at module level first.
    helper_anchor = 'class _DtAccessor:\n'
    if helper_anchor not in s:
        raise SystemExit("_DtAccessor class anchor not found")
    helper_L = []
    helper_L.append('def _freq_bucket(arr, freq_str, mode):')
    helper_L.append('    # Bucketize timestamp array via int64 math.')
    helper_L.append('    # mode in {"floor", "ceil", "round"} (round half-up).')
    helper_L.append('    from cadw.ops.resample import parse_freq, _TS_UNIT_FACTOR')
    helper_L.append('    if not pa.types.is_timestamp(arr.type):')
    helper_L.append('        raise TypeError(')
    helper_L.append('            "_freq_bucket: expected timestamp; got "')
    helper_L.append('            + str(arr.type)')
    helper_L.append('        )')
    helper_L.append('    freq_sec = parse_freq(freq_str)')
    helper_L.append('    factor = _TS_UNIT_FACTOR[arr.type.unit]')
    helper_L.append('    int_ns = pc.cast(arr, pa.int64())')
    helper_L.append('    secs = pc.cast(')
    helper_L.append('        pc.floor(pc.divide(int_ns,')
    helper_L.append('                        pa.scalar(factor, type=pa.int64()))),')
    helper_L.append('        pa.int64(),')
    helper_L.append('    )')
    helper_L.append('    fq = pa.scalar(freq_sec, type=pa.int64())')
    helper_L.append('    div = pc.divide(secs, fq)')
    helper_L.append('    base = pc.multiply(div, fq)')
    helper_L.append('    if mode == "floor":')
    helper_L.append('        bucket = base')
    helper_L.append('    elif mode == "ceil":')
    helper_L.append('        rem = pc.subtract(secs, base)')
    helper_L.append('        add = pc.if_else(')
    helper_L.append('            pc.greater(rem, pa.scalar(0, type=pa.int64())),')
    helper_L.append('            fq,')
    helper_L.append('            pa.scalar(0, type=pa.int64()),')
    helper_L.append('        )')
    helper_L.append('        bucket = pc.add(base, add)')
    helper_L.append('    elif mode == "round":')
    helper_L.append('        half = freq_sec // 2')
    helper_L.append('        shifted = pc.add(secs, pa.scalar(half, type=pa.int64()))')
    helper_L.append('        bucket = pc.multiply(pc.divide(shifted, fq), fq)')
    helper_L.append('    else:')
    helper_L.append('        raise ValueError("_freq_bucket: bad mode " + repr(mode))')
    helper_L.append('    scaled = pc.multiply(bucket, pa.scalar(factor, type=pa.int64()))')
    helper_L.append('    return pc.cast(scaled, arr.type)')
    helper_L.append('')
    helper_L.append('')
    helper_block = "\n".join(helper_L)
    s = s.replace(helper_anchor, helper_block + helper_anchor, 1)

    # Methods before quarter.
    L = []
    L.append('    def floor(self, freq: str) -> "CadwDataFrame":')
    L.append('        return self._feature(')
    L.append('            "floor",')
    L.append('            lambda arr: _freq_bucket(arr, freq, "floor"),')
    L.append('            self._parent._dataset.schema.field(self._column).type,')
    L.append('        )')
    L.append('')
    L.append('    def ceil(self, freq: str) -> "CadwDataFrame":')
    L.append('        return self._feature(')
    L.append('            "ceil",')
    L.append('            lambda arr: _freq_bucket(arr, freq, "ceil"),')
    L.append('            self._parent._dataset.schema.field(self._column).type,')
    L.append('        )')
    L.append('')
    L.append('    def round(self, freq: str) -> "CadwDataFrame":')
    L.append('        return self._feature(')
    L.append('            "round",')
    L.append('            lambda arr: _freq_bucket(arr, freq, "round"),')
    L.append('            self._parent._dataset.schema.field(self._column).type,')
    L.append('        )')
    L.append('')
    L.append('    @property')
    L.append('    def week_of_year(self) -> "CadwDataFrame":')
    L.append('        # ISO week (1-53). For US week numbering -- backlog.')
    L.append('        return self._feature(')
    L.append('            "week_of_year",')
    L.append('            lambda arr: pc.iso_week(arr),')
    L.append('            pa.int16(),')
    L.append('        )')
    L.append('')
    block = "\n".join(L)
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 4. _GroupBy.transform + describe ============

def patch_transform():
    print("\n=== dataframe.py: _GroupBy.transform ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def transform(self, agg_columns, agg_funcs" in s:
        print("  already present - skip")
        return
    anchor = (
        '    def _normalize_on(self, on) -> list:\n'
        '        if on is None:\n'
        '            return []\n'
        '        if isinstance(on, str):\n'
        '            return [on]\n'
        '        return list(on)\n'
    )
    if anchor not in s:
        raise SystemExit("_normalize_on anchor not found")
    L = []
    L.append('')
    L.append('    def transform(self, agg_columns, agg_funcs):')
    L.append('        # Aggregate + broadcast back to original rows via')
    L.append('        # left-merge on group keys. Same signature as agg().')
    L.append('        #')
    L.append('        # Returns CadwDataFrame with all original columns + new')
    L.append('        # {col}_{func} columns (like pandas assign of transform).')
    L.append('        #')
    L.append('        # LIMITATION (§11.5): row order in the result is')
    L.append('        # partition-concatenation (df.merge semantics), not original.')
    L.append('        # For pandas-compatible order, add __orig_rid and sort_values')
    L.append('        # after (documented; v0.9.5 backlog for streaming order-preserving)')
    L.append('        agg_cols = (')
    L.append('            [agg_columns] if isinstance(agg_columns, str)')
    L.append('            else list(agg_columns)')
    L.append('        )')
    L.append('        funcs = (')
    L.append('            [agg_funcs] if isinstance(agg_funcs, str)')
    L.append('            else list(agg_funcs)')
    L.append('        )')
    L.append('        if len(agg_cols) != len(funcs):')
    L.append('            raise ValueError(')
    L.append('                "transform: agg_columns and agg_funcs must have "')
    L.append('                "same length; got " + str(len(agg_cols)) + " vs "')
    L.append('                + str(len(funcs))')
    L.append('            )')
    L.append('        agg_df = self._parent._groupby_impl(self._by, agg_cols, funcs)')
    L.append('        return self._parent.merge(agg_df, on=self._by, how="left")')
    L.append('')
    block = "\n".join(L)
    s = s.replace(anchor, anchor + block, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def patch_describe():
    print("\n=== dataframe.py: describe ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def describe(self, columns=None):" in s:
        print("  already present - skip")
        return
    anchor = '        return {"total": total, "per_column": per_col}\n'
    if anchor not in s:
        raise SystemExit("memory_usage return anchor not found")
    L = []
    L.append('')
    L.append('    def describe(self, columns=None):')
    L.append('        # Numeric summary: count, mean, std, min, 25%, 50%,')
    L.append('        # 75%, max per column. Returns dict {col: {stat: value}}.')
    L.append('        #')
    L.append('        # columns=None: all numeric columns (int/float).')
    L.append('        #')
    L.append('        # Cost: per column -- 5 Reduce passes (count/mean/std/')
    L.append('        # min/max) + 1 external_sort (quantiles). For d numeric')
    L.append('        # cols, O(d) sorts total. See 11.6 for a fused DescribeOp')
    L.append('        # (single-pass optimization) backlog.')
    L.append('        from cadw.ops.reduce import (')
    L.append('            reduce, CountOp, MeanOp, StdOp, MinMaxOp,')
    L.append('        )')
    L.append('        if columns is None:')
    L.append('            cols = []')
    L.append('            for c in self._columns:')
    L.append('                t = self._dataset.schema.field(c).type')
    L.append('                if (pa.types.is_integer(t)')
    L.append('                        or pa.types.is_floating(t)):')
    L.append('                    cols.append(c)')
    L.append('        elif isinstance(columns, str):')
    L.append('            cols = [columns]')
    L.append('        else:')
    L.append('            cols = list(columns)')
    L.append('        for c in cols:')
    L.append('            if c not in self._columns:')
    L.append('                raise KeyError(')
    L.append('                    "describe: column " + repr(c) + " not in "')
    L.append('                    + str(self._columns)')
    L.append('                )')
    L.append('        result = {}')
    L.append('        for c in cols:')
    L.append('            count = reduce(')
    L.append('                self._iter_filtered_batches(), CountOp(c),')
    L.append('                self._memory_guard, log=False,')
    L.append('            )')
    L.append('            mean = reduce(')
    L.append('                self._iter_filtered_batches(), MeanOp(c),')
    L.append('                self._memory_guard, log=False,')
    L.append('            )')
    L.append('            std = reduce(')
    L.append('                self._iter_filtered_batches(), StdOp(c),')
    L.append('                self._memory_guard, log=False,')
    L.append('            )')
    L.append('            mn = reduce(')
    L.append('                self._iter_filtered_batches(),')
    L.append('                MinMaxOp(c, "min"),')
    L.append('                self._memory_guard, log=False,')
    L.append('            )')
    L.append('            mx = reduce(')
    L.append('                self._iter_filtered_batches(),')
    L.append('                MinMaxOp(c, "max"),')
    L.append('                self._memory_guard, log=False,')
    L.append('            )')
    L.append('            q = self.quantile(c, [0.25, 0.5, 0.75])')
    L.append('            result[c] = {')
    L.append('                "count": count,')
    L.append('                "mean": mean,')
    L.append('                "std": std,')
    L.append('                "min": mn,')
    L.append('                "25%": q[0],')
    L.append('                "50%": q[1],')
    L.append('                "75%": q[2],')
    L.append('                "max": mx,')
    L.append('            }')
    L.append('        return result')
    L.append('')
    block = "\n".join(L)
    s = s.replace(anchor, anchor + block, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 5. Tests ============

TESTS = [
    "# v0.9.4b-2: transform, describe, dt/str extras.",
    "",
    "from pathlib import Path",
    "",
    "import pandas as pd",
    "import pyarrow as pa",
    "import pyarrow.parquet as pq",
    "import pytest",
    "",
    "from cadw import CadwDataFrame, MemoryGuard",
    "",
    "",
    "def make_df(tmp_path, data, chunk=10, name=\"L\", mem_mb=50):",
    "    t = pa.table(data)",
    "    p = Path(tmp_path) / f\"{name}.parquet\"",
    "    pq.write_table(t, p)",
    "    return CadwDataFrame.from_parquet(",
    "        p, Path(tmp_path) / f\"ds_{name}\", chunk_size=chunk,",
    "        memory_guard=MemoryGuard(max_memory_mb=mem_mb),",
    "        spill_dir=Path(tmp_path) / f\"spill_{name}\",",
    "    )",
    "",
    "",
    "def to_pd(df, tmp_path, name=\"out.parquet\"):",
    "    out = tmp_path / name",
    "    df.write_parquet(out)",
    "    return pq.read_table(out).to_pandas()",
    "",
    "",
    "# ============ _GroupBy.transform ============",
    "",
    "def test_transform_mean_basic(tmp_path):",
    "    df = make_df(tmp_path, {",
    "        \"g\": [\"a\", \"a\", \"b\", \"b\"],",
    "        \"v\": [1.0, 3.0, 10.0, 20.0],",
    "    }, name=\"a\")",
    "    out = df.groupby(\"g\").transform(\"v\", \"mean\")",
    "    pd_out = to_pd(out, tmp_path)",
    "    assert \"v_mean\" in pd_out.columns",
    "    assert \"v\" in pd_out.columns",
    "    assert len(pd_out) == 4",
    "    grouped = pd_out.groupby(\"g\")[\"v_mean\"].first().to_dict()",
    "    assert grouped[\"a\"] == 2.0",
    "    assert grouped[\"b\"] == 15.0",
    "",
    "",
    "def test_transform_sum_basic(tmp_path):",
    "    df = make_df(tmp_path, {",
    "        \"g\": [\"x\", \"x\", \"y\"],",
    "        \"v\": [1, 2, 5],",
    "    }, name=\"a\")",
    "    out = df.groupby(\"g\").transform(\"v\", \"sum\")",
    "    pd_out = to_pd(out, tmp_path)",
    "    grouped = pd_out.groupby(\"g\")[\"v_sum\"].first().to_dict()",
    "    assert grouped[\"x\"] == 3",
    "    assert grouped[\"y\"] == 5",
    "",
    "",
    "def test_transform_with_filter(tmp_path):",
    "    df = make_df(tmp_path, {",
    "        \"g\": [\"a\", \"a\", \"a\", \"b\", \"b\"],",
    "        \"v\": [1.0, 2.0, 3.0, 10.0, 20.0],",
    "    }, name=\"a\")",
    "    f = df[df[\"v\"] > 2]",
    "    out = f.groupby(\"g\").transform(\"v\", \"mean\")",
    "    pd_out = to_pd(out, tmp_path)",
    "    assert len(pd_out) == 3",
    "    grouped = pd_out.groupby(\"g\")[\"v_mean\"].first().to_dict()",
    "    assert grouped[\"a\"] == 3.0",
    "    assert grouped[\"b\"] == 15.0",
    "",
    "",
    "def test_transform_length_mismatch(tmp_path):",
    "    df = make_df(tmp_path, {\"g\": [\"a\"], \"v\": [1.0]}, name=\"a\")",
    "    with pytest.raises(ValueError, match=\"same length\"):",
    "        df.groupby(\"g\").transform([\"v\"], [\"mean\", \"sum\"])",
    "",
    "",
    "# ============ describe ============",
    "",
    "def test_describe_basic(tmp_path):",
    "    df = make_df(tmp_path, {\"x\": [1.0, 2.0, 3.0, 4.0, 5.0]}, name=\"a\")",
    "    desc = df.describe()",
    "    assert \"x\" in desc",
    "    d = desc[\"x\"]",
    "    assert d[\"count\"] == 5",
    "    assert d[\"mean\"] == pytest.approx(3.0)",
    "    assert d[\"std\"] == pytest.approx(1.5811388300841898, rel=1e-9)",
    "    assert d[\"min\"] == 1.0",
    "    assert d[\"25%\"] == pytest.approx(2.0)",
    "    assert d[\"50%\"] == pytest.approx(3.0)",
    "    assert d[\"75%\"] == pytest.approx(4.0)",
    "    assert d[\"max\"] == 5.0",
    "",
    "",
    "def test_describe_filtered(tmp_path):",
    "    df = make_df(tmp_path, {\"x\": [1, 2, 3, 4, 5]}, name=\"a\")",
    "    f = df[df[\"x\"] > 2]",
    "    desc = f.describe()",
    "    assert desc[\"x\"][\"count\"] == 3",
    "    assert desc[\"x\"][\"min\"] == 3.0",
    "    assert desc[\"x\"][\"max\"] == 5.0",
    "",
    "",
    "def test_describe_subset(tmp_path):",
    "    df = make_df(tmp_path, {",
    "        \"a\": [1, 2], \"b\": [10, 20], \"s\": [\"x\", \"y\"],",
    "    }, name=\"a\")",
    "    desc = df.describe(columns=[\"a\"])",
    "    assert \"a\" in desc",
    "    assert \"b\" not in desc",
    "    assert \"s\" not in desc",
    "",
    "",
    "def test_describe_all_numeric_default(tmp_path):",
    "    df = make_df(tmp_path, {",
    "        \"i\": [1, 2], \"f\": [1.0, 2.0], \"s\": [\"a\", \"b\"],",
    "    }, name=\"a\")",
    "    desc = df.describe()",
    "    assert \"i\" in desc",
    "    assert \"f\" in desc",
    "    assert \"s\" not in desc",
    "",
    "",
    "def test_describe_missing_column(tmp_path):",
    "    df = make_df(tmp_path, {\"x\": [1]}, name=\"a\")",
    "    with pytest.raises(KeyError, match=\"nope\"):",
    "        df.describe(columns=[\"nope\"])",
    "",
    "",
    "# ============ dt.floor/ceil/round/week_of_year ============",
    "",
    "def test_dt_floor_hour(tmp_path):",
    "    ts = pd.to_datetime([",
    "        \"2020-01-01 10:30:00\", \"2020-01-01 10:45:00\",",
    "    ])",
    "    df = make_df(tmp_path, {\"ts\": pa.array(ts)}, name=\"a\")",
    "    out = df[\"ts\"].dt.floor(\"1h\")",
    "    pd_out = to_pd(out, tmp_path)",
    "    assert [str(t) for t in pd_out[\"ts_floor\"].tolist()] == [",
    "        \"2020-01-01 10:00:00\", \"2020-01-01 10:00:00\",",
    "    ]",
    "",
    "",
    "def test_dt_ceil_hour(tmp_path):",
    "    ts = pd.to_datetime([",
    "        \"2020-01-01 10:30:00\", \"2020-01-01 10:00:00\",",
    "    ])",
    "    df = make_df(tmp_path, {\"ts\": pa.array(ts)}, name=\"a\")",
    "    out = df[\"ts\"].dt.ceil(\"1h\")",
    "    pd_out = to_pd(out, tmp_path)",
    "    assert [str(t) for t in pd_out[\"ts_ceil\"].tolist()] == [",
    "        \"2020-01-01 11:00:00\", \"2020-01-01 10:00:00\",",
    "    ]",
    "",
    "",
    "def test_dt_round_hour(tmp_path):",
    "    ts = pd.to_datetime([",
    "        \"2020-01-01 10:20:00\", \"2020-01-01 10:40:00\",",
    "    ])",
    "    df = make_df(tmp_path, {\"ts\": pa.array(ts)}, name=\"a\")",
    "    out = df[\"ts\"].dt.round(\"1h\")",
    "    pd_out = to_pd(out, tmp_path)",
    "    assert [str(t) for t in pd_out[\"ts_round\"].tolist()] == [",
    "        \"2020-01-01 10:00:00\", \"2020-01-01 11:00:00\",",
    "    ]",
    "",
    "",
    "def test_dt_week_of_year(tmp_path):",
    "    # 2020-01-01 is ISO week 1; 2020-12-31 is ISO week 53.",
    "    ts = pd.to_datetime([\"2020-01-01\", \"2020-12-31\"])",
    "    df = make_df(tmp_path, {\"ts\": pa.array(ts)}, name=\"a\")",
    "    out = df[\"ts\"].dt.week_of_year",
    "    pd_out = to_pd(out, tmp_path)",
    "    assert pd_out[\"ts_week_of_year\"].tolist() == [1, 53]",
    "",
    "",
    "def test_dt_floor_non_timestamp_raises(tmp_path):",
    "    df = make_df(tmp_path, {\"x\": [1, 2]}, name=\"a\")",
    "    with pytest.raises(TypeError, match=\"expected timestamp\"):",
    "        df[\"x\"].dt.floor(\"1h\")",
    "",
    "",
    "# ============ str.strip / starts_with / replace ============",
    "",
    "def test_str_strip(tmp_path):",
    "    df = make_df(tmp_path, {\"s\": [\"  hi  \", \"a\\tb\"]}, name=\"a\")",
    "    out = df[\"s\"].str.strip",
    "    pd_out = to_pd(out, tmp_path)",
    "    assert pd_out[\"s_strip\"].tolist() == [\"hi\", \"a\\tb\"]",
    "",
    "",
    "def test_str_starts_with_mask(tmp_path):",
    "    df = make_df(tmp_path, {",
    "        \"s\": [\"foo_bar\", \"baz_qux\", \"foo_baz\", None],",
    "        \"v\": [1, 2, 3, 4],",
    "    }, name=\"a\")",
    "    f = df[df[\"s\"].str.starts_with(\"foo\")]",
    "    pd_out = to_pd(f, tmp_path)",
    "    assert pd_out[\"v\"].tolist() == [1, 3]",
    "",
    "",
    "def test_str_ends_with_mask(tmp_path):",
    "    df = make_df(tmp_path, {",
    "        \"s\": [\"foo_bar\", \"baz_qux\", \"baz_bar\", None],",
    "        \"v\": [1, 2, 3, 4],",
    "    }, name=\"a\")",
    "    f = df[df[\"s\"].str.ends_with(\"bar\")]",
    "    pd_out = to_pd(f, tmp_path)",
    "    assert pd_out[\"v\"].tolist() == [1, 3]",
    "",
    "",
    "def test_str_replace_literal(tmp_path):",
    "    df = make_df(tmp_path, {\"s\": [\"a_b_c\", \"x_y\"]}, name=\"a\")",
    "    out = df[\"s\"].str.replace(\"_\", \"-\")",
    "    pd_out = to_pd(out, tmp_path)",
    "    assert pd_out[\"s_replaced\"].tolist() == [\"a-b-c\", \"x-y\"]",
    "",
    "",
    "def test_str_replace_regex(tmp_path):",
    "    df = make_df(tmp_path, {\"s\": [\"ab12cd\", \"ef34\"]}, name=\"a\")",
    "    out = df[\"s\"].str.replace(\"[0-9]+\", \"#\", regex=True)",
    "    pd_out = to_pd(out, tmp_path)",
    "    assert pd_out[\"s_replaced\"].tolist() == [\"ab#cd\", \"ef#\"]",
    "",
]


def write_tests():
    print("\n=== tests/test_dataframe_v094b2.py (new) ===")
    p = Path("tests/test_dataframe_v094b2.py")
    write_content(p, TESTS)
    assert_compiles(p)
    print("  OK")


# ============ 6. CHANGELOG + DECISIONS 11.5 ============

def append_changelog():
    print("\n=== CHANGELOG v0.9.4b-2 ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.4b-2 - transform" in s:
        print("  already present - skip")
        return
    from cadw.patch_utils import append_unreleased_entry
    L = [
        "### Добавлено (v0.9.4b-2 - transform / describe / dt / str)",
        "",
        "- **`df.groupby(g).transform(agg_columns, agg_funcs)`** —",
        "  aggregate + left-merge back. Same signature as `agg()`. Result:",
        "  all original columns + new `{col}_{func}` columns. Row order =",
        "  partition-concat (merge semantics) — documented 11.5. For",
        "  pandas order, add `__orig_rid` + `sort_values()` after.",
        "- **`df.describe(columns=None)`** — numeric summary dict",
        "  `{col: {count/mean/std/min/25%/50%/75%/max}}`. Default: all",
        "  numeric cols. Cost: 5 Reduce passes + 1 sort per column.",
        "- **`df['ts'].dt.floor(freq)` / `ceil(freq)` / `round(freq)`** —",
        "  bucketize timestamp. `freq` = `'5s'`, `'1h'`, `'1D'` etc.",
        "  Appends `{col}_floor` / `_ceil` / `_round` (same dtype).",
        "- **`df['ts'].dt.week_of_year`** — ISO week (1-53). int16.",
        "- **`df['s'].str.strip/lstrip/rstrip`** — whitespace trim.",
        "  Appends `{col}_strip` / `_lstrip` / `_rstrip`.",
        "- **`df['s'].str.starts_with(pat)` / `.ends_with(pat)`** — return",
        "  `MaskFilter` for `df[mask]`. `na=False` (pandas default).",
        "- **`df['s'].str.replace(pat, repl, regex=False)`** — literal",
        "  substring replace (default). `regex=True` for regex. Appends",
        "  `{col}_replaced`.",
        "- **`PredicateFilter` extended** with `starts_with`, `ends_with`",
        "  ops (per-batch Arrow `pc.starts_with` / `pc.ends_with`).",
        "",
        "### Ограничения (v0.9.4b-2, 11.5)",
        "",
        "- **`transform` row order** = merge partition-concat, not",
        "  original. Streaming order-preserving transform → v0.9.5 backlog.",
        "- **`describe` cost**: O(d) sorts for quantiles. Fused `DescribeOp`",
        "  (single pass, no separate min/max) → 11.6 backlog.",
        "- **`dt.floor/ceil/round`** assumes post-1970 timestamps (positive",
        "  int64). Pre-1970 edge cases (negative int) diverge from pandas",
        "  floor semantics. `dt.round` = round-half-up.",
        "- **`dt.week_of_year`** = ISO week. US week numbering → backlog.",
        "- **`str.replace`** default `regex=False` (literal). pandas default",
        "  is `regex=True`; CADW chose literal for predictability.",
        "",
        "### Тесты (v0.9.4b-2)",
        "",
        "+22 fast (`test_dataframe_v094b2.py`): transform (4), describe (4),",
        "dt.floor/ceil/round/week_of_year (5), str.strip/starts_with/",
        "ends_with/replace (5). Plus 4 error cases.",
        "",
        "- fast: **~1640** (+22). slow: **20** (unchanged).",
        "",
    ]
    entry = "\n".join(L)
    p.write_text(append_unreleased_entry(s, entry))
    print("  OK")


def append_decisions_11_5():
    print("\n=== DECISIONS 11.5: transform order + describe cost ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.4b-2 (transform / describe)" in s:
        print("  already present - skip")
        return
    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        raise SystemExit("DECISIONS 11.5 not found")
    L = [
        "**v0.9.4b-2 (transform / describe):**",
        "- `df.groupby(g).transform(cols, funcs)` - aggregate + left-merge.",
        "  Row order = merge partition-concat, NOT original (df.merge()",
        "  semantics). For pandas-compatible order: add `__orig_rid`",
        "  column before transform, sort_values after. Streaming",
        "  order-preserving transform - v0.9.5 backlog.",
        "- `describe()` cost: 5 Reduce passes + 1 external_sort per",
        "  numeric column. Fused `DescribeOp` (count/mean/std/min/max",
        "  in one pass) - 11.6 backlog.",
        "- `dt.floor/ceil/round` - int64 arithmetic; post-1970 assumption.",
        "  Pre-1970 timestamps (negative int64) may diverge from pandas.",
        "- `dt.week_of_year` - ISO week (1-53), not US week numbering.",
        "",
    ]
    note = "\n".join(L)
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK")


def main():
    print("=== v0.9.4b-2 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    patch_predicate_filter()
    patch_str_accessor()
    patch_dt_accessor()
    patch_transform()
    patch_describe()
    write_tests()
    append_changelog()
    append_decisions_11_5()

    print("\n=== pytest: new v094b2 tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_dataframe_v094b2.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: transform/describe/dt/str regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_groupby_v2.py",
         "tests/test_dt_accessor.py",
         "tests/test_str_accessor.py",
         "tests/test_row_filter.py",
         "tests/test_quantile.py",
         "tests/test_reduce.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    commit_with_guards(
        "v0.9.4b-2: _GroupBy.transform (agg+merge; order=partition, "
        "11.5); describe (count/mean/std/min/q/max); dt.floor/ceil/round"
        "(freq) + week_of_year; str.strip/lstrip/rstrip/starts_with/"
        "ends_with/replace; PredicateFilter starts_with/ends_with ops",
    )
    print("\n==> v0.9.4b-2 applied.")


if __name__ == "__main__":
    main()
