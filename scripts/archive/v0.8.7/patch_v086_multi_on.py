#!/usr/bin/env python3
"""v0.8.6: multi-column `on` for shift/diff/rolling.

CHANGE:
  - `shift(k, on=str | Sequence[str] | None)`
  - `diff(k, on=str | Sequence[str] | None)`
  - `rolling(window, agg, on=str | Sequence[str] | None)`
  - `_shift_on(..., on_cols: list[str])` (pre-normalized)
  - `_rolling_on(..., on_cols: list[str])`
  - `_diff_on(..., on: str | Sequence[str])` — normalizes, then
    delegates to `_shift_on(..., on_cols)`.
  - New helper `CadwDataFrame._normalize_on(on, fn_name)`.

SEMANTICS:
  `on=["ts", "sensor"]` = lexicographic composite sort.
  Sort by (ts, sensor, row_id tie-break), apply shift/rolling per row,
  sort back by row_id.

  NOT per-group. For per-sensor windows: `groupby('sensor').shift(1,
  on='ts')` (v0.8.5a+, не реализовано).

DESIGN:
  `_normalize_on` validates + returns `list[str]`. Called by dispatch
  (`shift`/`rolling`) OR by `_diff_on` (which needs to check `on` before
  numeric-type checks on `columns`). Redundant double-call in
  `shift` → `_shift_on` avoided by passing `on_cols` list.

  Existing AST guards (`test_shift_on_second_sort_renames_row_id` etc.)
  continue to pass — second `external_sort` still uses
  `key_columns=["__orig_rid"]` (literal list); first uses `on_cols`
  (variable, `keys=None` in AST, guarded by the `[-1]`-only checks).

Run: uv run python scripts/patch_v086_multi_on.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:25]:
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


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if len(starts) == 0:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ Fix 1: _normalize_on helper + shift/diff/rolling signatures ============

def patch_normalize_helper():
    print("\n=== cadw/dataframe.py: _normalize_on helper ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "def _normalize_on(" in s:
        print("  already present — skip")
        return

    anchor = '''    # ---------------- shift / diff ----------------

    def shift('''
    if anchor not in s:
        raise SystemExit("shift/diff section anchor not found")

    helper = '''    def _normalize_on(
        self, on: "str | Sequence[str]", *, fn_name: str,
    ) -> list[str]:
        """Normalize `on` to list[str], validate each column exists.

        v0.8.6: multi-column `on` (`on=["ts", "sensor"]`) — lexicographic
        composite sort. Accepts str (backward compat) or Sequence[str].
        """
        if isinstance(on, str):
            on_cols = [on]
        else:
            on_cols = list(on)
        if not on_cols:
            raise ValueError(f"{fn_name}: on must be non-empty")
        for c in on_cols:
            if c not in self._columns:
                raise KeyError(
                    f"{fn_name}: on column {c!r} not in {self._columns}"
                )
        return on_cols

'''
    s = s.replace(anchor, helper + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: _normalize_on added")


# ============ Fix 2: shift signature + dispatch ============

def patch_shift_dispatch():
    print("\n=== cadw/dataframe.py: shift — signature + dispatch ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old_sig = '''        columns: "str | Sequence[str] | None" = None,
        on: "str | None" = None,
    ) -> "CadwDataFrame":
        """Shift values by `k` positions (v0.6.2b, lag).

        - k >= 0: `out[i] = x[i-k]` for i>=k, else `fill_value`.
        - k == 0: returns self (no-op).
        - k < 0 (lead): NotImplementedError (v0.6.3).
        - columns=None: all columns; else subset (others pass-through).
        - fill_value=None: null for first k rows. Non-None: that value.

        Streamable: O(k) memory (deque per column). Position-based,
        iterates filtered batches in row order.

        Shifted columns become **nullable** (fill = null).
        """'''
    new_sig = '''        columns: "str | Sequence[str] | None" = None,
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
        """'''
    s = replace_line_block(s, old_sig, new_sig, tag="shift-sig")

    old_dispatch = '''        # v0.8.2: sort-aware shift.
        if on is not None:
            return self._shift_on(
                k, fill_value, columns=columns, on=on,
            )'''
    new_dispatch = '''        # v0.8.2: sort-aware shift; v0.8.6: multi-column on.
        if on is not None:
            on_cols = self._normalize_on(on, fn_name="shift")
            return self._shift_on(
                k, fill_value, columns=columns, on_cols=on_cols,
            )'''
    s = replace_line_block(s, old_dispatch, new_dispatch, tag="shift-dispatch")

    p.write_text(s)
    compile_check(p)
    print("  OK: shift — signature + dispatch")


# ============ Fix 3: _shift_on signature ============

def patch_shift_on_sig():
    print("\n=== cadw/dataframe.py: _shift_on — on_cols ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''    def _shift_on(
        self,
        k: int,
        fill_value,
        *,
        columns: "str | Sequence[str] | None",
        on: str,
    ) -> "CadwDataFrame":
        """Sort-aware shift on `on` column (v0.8.2).

        Algorithm:
          1. external_sort by [on, row_id] → sorted IPC.
          2. Read sorted stream, apply stateful shift (deque per column,
             O(k) memory). Output (row_id, shifted_cols) → IPC.
          3. external_sort by row_id → original order.
          4. Zip with original.

        Peak RAM: batch + O(k × #cols). Independent of N.
        """
        from collections import deque

        if on not in self._columns:
            raise KeyError(
                f"shift: on column {on!r} not in {self._columns}"
            )

        if columns is None:'''
    new = '''    def _shift_on(
        self,
        k: int,
        fill_value,
        *,
        columns: "str | Sequence[str] | None",
        on_cols: list[str],
    ) -> "CadwDataFrame":
        """Sort-aware shift on `on_cols` (v0.8.2, multi v0.8.6).

        Algorithm:
          1. external_sort by [on_cols...] (row_id tie-break from heap)
             → sorted IPC. Lexicographic composite sort.
          2. Read sorted stream, apply stateful shift (deque per column,
             O(k) memory). Output (row_id, shifted_cols) → IPC.
          3. external_sort by __orig_rid → original order.
          4. Zip with original.

        Peak RAM: batch + O(k × #cols). Independent of N.
        `on_cols` pre-normalized by caller (v0.8.6).
        """
        from collections import deque

        if columns is None:'''
    s = replace_line_block(s, old, new, tag="shift-on-sig")

    old_sort = '''        # 1. Sort by [on, row_id]
        all_cols = list(self._columns)
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=[on],
            value_columns=all_cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "shift" / "sort_by_on",
            ascending=True,
            log=False,
        )'''
    new_sort = '''        # 1. Sort by [on_cols...] (lexicographic)
        all_cols = list(self._columns)
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=on_cols,
            value_columns=all_cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "shift" / "sort_by_on",
            ascending=True,
            log=False,
        )'''
    s = replace_line_block(s, old_sort, new_sort, tag="shift-on-sort")

    p.write_text(s)
    compile_check(p)
    print("  OK: _shift_on — on_cols")


# ============ Fix 4: _diff_on ============

def patch_diff_on():
    print("\n=== cadw/dataframe.py: diff signature + _diff_on ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old_diff_sig = '''        columns: "str | Sequence[str] | None" = None,
        on: "str | None" = None,
    ) -> "CadwDataFrame":
        """First difference: `out[i] = x[i] - x[i-k]` (v0.6.2b).

        - k >= 1. First k rows -> null.
        - columns=None: all; else subset (others pass-through).
        - Output columns nullable (first k = null).

        Streamable: O(k) memory.
        """'''
    new_diff_sig = '''        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """First difference: `out[i] = x[i] - x[i-k]` (v0.6.2b; multi-on v0.8.6).

        - k >= 1. First k rows -> null.
        - columns=None: all; else subset (others pass-through).
        - on: positional if None; sort-aware if str/Sequence
          (lexicographic composite sort).
        - Output columns nullable (first k = null).

        Streamable: O(k) memory.
        """'''
    s = replace_line_block(s, old_diff_sig, new_diff_sig, tag="diff-sig")

    old_diff_disp = '''        # v0.8.2: sort-aware diff = shift(on) - original.
        if on is not None:
            return self._diff_on(k=k, columns=columns, on=on)'''
    new_diff_disp = '''        # v0.8.2: sort-aware diff; v0.8.6: multi-column on.
        if on is not None:
            return self._diff_on(k=k, columns=columns, on=on)'''
    # No change needed, but confirm block exists
    if old_diff_disp not in s:
        print("  WARNING: diff dispatch block not found (unchanged)")

    old_diff_on = '''    def _diff_on(
        self,
        *,
        k: int,
        columns: "str | Sequence[str] | None",
        on: str,
    ) -> "CadwDataFrame":
        """Sort-aware diff: pc.subtract(col, shifted(col)) with on=col sort.

        v0.8.2. Delegates to _shift_on + subtract per batch. No extra
        sort needed (reuses shift's 2 sorts).
        """
        if columns is None:'''
    new_diff_on = '''    def _diff_on(
        self,
        *,
        k: int,
        columns: "str | Sequence[str] | None",
        on: "str | Sequence[str]",
    ) -> "CadwDataFrame":
        """Sort-aware diff: pc.subtract(col, shifted(col)) with on=col sort.

        v0.8.2 (single), v0.8.6 (multi). Delegates to _shift_on + subtract
        per batch. No extra sort needed (reuses shift's 2 sorts).
        """
        # v0.8.6: validate on early (before numeric-type check on columns).
        on_cols = self._normalize_on(on, fn_name="diff")
        if columns is None:'''
    s = replace_line_block(s, old_diff_on, new_diff_on, tag="diff-on-sig")

    old_shift_call = '''        # Shift with fill_value=None (nulls for first k rows)
        shifted_df = self._shift_on(
            k, None, columns=cols, on=on,
        )'''
    new_shift_call = '''        # Shift with fill_value=None (nulls for first k rows)
        shifted_df = self._shift_on(
            k, None, columns=cols, on_cols=on_cols,
        )'''
    s = replace_line_block(s, old_shift_call, new_shift_call, tag="diff-on-shift-call")

    p.write_text(s)
    compile_check(p)
    print("  OK: _diff_on — on_cols")


# ============ Fix 5: rolling signature + _rolling_on ============

def patch_rolling():
    print("\n=== cadw/dataframe.py: rolling + _rolling_on ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old_sig = '''        columns: "str | Sequence[str] | None" = None,
        on: "str | None" = None,
    ) -> "CadwDataFrame":
        """Rolling window aggregation (v0.6.2c).

        - window: int >= 1.
        - agg: {'sum', 'mean', 'min', 'max', 'count'}.
              count = non-null count in window.
        - min_periods=None -> window. First (min_periods-1) rows = null.
        - columns=None -> all numeric columns.

        Output: original columns + `{col}_rolling_{agg}` for each column.

        Incremental streaming: O(1) per row for sum/mean/count; O(1)
        amortized for min/max via monotonic deque. Peak O(window × #cols).

        skips nulls (pandas default). First rows (before min_periods)
        receive null.

        Not in v0.6.2c: `on=col` (sort-based; use .sort_values(...) first),
        expanding, ewm, std/var/median/quantile.
        """'''
    new_sig = '''        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Rolling window aggregation (v0.6.2c; sort-aware v0.8.4; multi v0.8.6).

        - window: int >= 1.
        - agg: {'sum', 'mean', 'min', 'max', 'count'}.
        - min_periods=None -> window.
        - columns=None -> all numeric columns.
        - on: positional if None; sort-aware if str/Sequence
          (`on=["ts", "sensor"]` — lexicographic composite sort, NOT
          per-group; for per-group see v0.8.5a).

        Output: original + `{col}_rolling_{agg}` (APPEND).
        Nulls skip min/max/sum/mean but **consume window slot** (positional,
        matches pandas skipna=True). Peak O(window × #cols).
        """'''
    s = replace_line_block(s, old_sig, new_sig, tag="rolling-sig")

    old_disp = '''        # v0.8.4: sort-aware rolling (same 2-sort pattern as _shift_on).
        if on is not None:
            return self._rolling_on(
                window=window, agg=agg, min_periods=mp,
                columns=columns, on=on,
            )'''
    new_disp = '''        # v0.8.4: sort-aware; v0.8.6: multi-column on.
        if on is not None:
            on_cols = self._normalize_on(on, fn_name="rolling")
            return self._rolling_on(
                window=window, agg=agg, min_periods=mp,
                columns=columns, on_cols=on_cols,
            )'''
    s = replace_line_block(s, old_disp, new_disp, tag="rolling-disp")

    old_on_sig = '''    def _rolling_on(
        self,
        *,
        window: int,
        agg: str,
        min_periods: int,
        columns: "str | Sequence[str] | None",
        on: str,
    ) -> "CadwDataFrame":
        """Sort-aware rolling on `on` column (v0.8.4).

        Validated by prototype (scripts/proto_rolling_on.py):
          A (sort_values+rolling+sort_values) ==
          B (external_sort + _RollState + external_sort) ==
          C (pure Python deque).

        Algorithm:
          1. external_sort by [on] (row_id tie-break from heap).
          2. _RollState.push per row (per column) → IPC
             (row_id, {col}_rolling_{agg}).
          3. external_sort by __orig_rid (rename, avoid duplicate
             row_id — §13.4/122).
          4. Zip original + rolling cols (APPEND; rolling adds new cols).

        Peak RAM: batch + O(window × #cols). Independent of N.
        """
        if on not in self._columns:
            raise KeyError(
                f"rolling: on column {on!r} not in {self._columns}"
            )

        if columns is None:'''
    new_on_sig = '''    def _rolling_on(
        self,
        *,
        window: int,
        agg: str,
        min_periods: int,
        columns: "str | Sequence[str] | None",
        on_cols: list[str],
    ) -> "CadwDataFrame":
        """Sort-aware rolling on `on_cols` (v0.8.4; multi v0.8.6).

        Validated by prototype (scripts/proto_rolling_on.py):
          A (sort_values+rolling+sort_values) ==
          B (external_sort + _RollState + external_sort) ==
          C (pure Python deque).

        Algorithm (identical to v0.8.4, multi-key `on_cols`):
          1. external_sort by [on_cols...] (row_id tie-break from heap).
          2. _RollState.push per row (per column) → IPC
             (row_id, {col}_rolling_{agg}).
          3. external_sort by __orig_rid (§13.4/122).
          4. Zip original + rolling cols (APPEND).

        Peak RAM: batch + O(window × #cols). Independent of N.
        `on_cols` pre-normalized by caller (v0.8.6).
        """
        if columns is None:'''
    s = replace_line_block(s, old_on_sig, new_on_sig, tag="rolling-on-sig")

    old_sort = '''        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=[on],
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "rolling" / "sort_by_on",
            ascending=True,
            log=False,
        )'''
    new_sort = '''        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=on_cols,
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "rolling" / "sort_by_on",
            ascending=True,
            log=False,
        )'''
    s = replace_line_block(s, old_sort, new_sort, tag="rolling-on-sort")

    p.write_text(s)
    compile_check(p)
    print("  OK: rolling + _rolling_on — on_cols")


# ============ Fix 6: tests ============

def add_shift_multi_tests():
    print("\n=== tests/test_shift_on_col.py: multi-key tests ===")
    p = Path("tests/test_shift_on_col.py")
    s = p.read_text()
    if "test_shift_on_multi_key" in s:
        print("  already present — skip")
        return
    extra = '''

# ---------- v0.8.6: multi-column on ----------

def test_shift_on_multi_key(tmp_path):
    """on=['ts','sensor'] — lexicographic composite sort."""
    df = make_df(tmp_path, data={
        "ts": [2, 1, 1, 2],
        "sensor": ["a", "a", "b", "b"],
        "v": [40, 10, 20, 30],
    })
    res = df.shift(1, on=["ts", "sensor"], columns=["v"])
    # Sorted by (ts, sensor): row1(1,a)=10, row2(1,b)=20, row0(2,a)=40,
    # row3(2,b)=30. shift(1): [None, 10, 20, 40].
    # Back to original row_id order [0,1,2,3]: [20, None, 10, 40]
    vals = read_col(res, tmp_path, "v")
    assert vals == [20, None, 10, 40]


def test_diff_on_multi_key(tmp_path):
    df = make_df(tmp_path, data={
        "ts": [2, 1, 1, 2],
        "sensor": ["a", "a", "b", "b"],
        "v": [40, 10, 20, 30],
    })
    res = df.diff(1, on=["ts", "sensor"], columns=["v"])
    # Sorted v: [10, 20, 40, 30]. diff(1): [None, 10, 20, -10].
    # Back: [20, None, 10, -10]
    vals = read_col(res, tmp_path, "v")
    assert vals == [20, None, 10, -10]


def test_shift_on_single_element_list(tmp_path):
    """on=['ts'] behaves like on='ts'."""
    df = make_df(tmp_path)
    r1 = df.shift(1, on="ts", columns=["v"])
    r2 = df.shift(1, on=["ts"], columns=["v"])
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    v1 = read_col(r1, tmp_path / "r1", "v")
    v2 = read_col(r2, tmp_path / "r2", "v")
    assert v1 == v2


def test_shift_on_empty_list_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        df.shift(1, on=[], columns=["v"])


def test_shift_on_multi_key_stable_tiebreak(tmp_path):
    """Equal on-values: row_id tie-break keeps original order."""
    df = make_df(tmp_path, data={
        "ts": [1, 1, 2],
        "sensor": ["a", "a", "b"],
        "v": [10, 20, 30],
    })
    res = df.shift(1, on=["ts", "sensor"], columns=["v"])
    # Sorted by (ts, sensor, row_id): row0(1,a)=10, row1(1,a)=20,
    # row2(2,b)=30. shift(1): [None, 10, 20].
    vals = read_col(res, tmp_path, "v")
    assert vals == [None, 10, 20]
'''
    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: +5 shift/diff multi tests")


def add_rolling_multi_tests():
    print("\n=== tests/test_rolling_on_col.py: multi-key test ===")
    p = Path("tests/test_rolling_on_col.py")
    s = p.read_text()
    if "test_rolling_on_multi_key" in s:
        print("  already present — skip")
        return
    extra = '''

# ---------- v0.8.6: multi-column on ----------

def test_rolling_on_multi_key(tmp_path):
    """on=['ts','sensor'] — lexicographic composite sort."""
    df = make_df(tmp_path, data={
        "ts": [2, 1, 1, 2],
        "sensor": ["a", "a", "b", "b"],
        "v": [40, 10, 20, 30],
    })
    res = df.rolling(2, "sum", on=["ts", "sensor"], columns=["v"])
    # Sorted v: [10, 20, 40, 30]. Rolling sum w2 (min_periods=2):
    # [None, 30, 60, 70]. Back to orig [0,1,2,3]: [60, None, 30, 70]
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [60, None, 30, 70]


def test_rolling_on_empty_list_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        df.rolling(2, "sum", on=[], columns=["v"])


def test_rolling_on_multi_key_min_periods_1(tmp_path):
    df = make_df(tmp_path, data={
        "ts": [2, 1, 1, 2],
        "sensor": ["a", "a", "b", "b"],
        "v": [40, 10, 20, 30],
    })
    res = df.rolling(2, "sum", on=["ts", "sensor"], columns=["v"],
                     min_periods=1)
    # Sorted v: [10, 20, 40, 30]. Rolling sum w2 (min_periods=1):
    # [10, 30, 60, 70]. Back to orig: [60, 10, 30, 70]
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [60, 10, 30, 70]
'''
    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: +3 rolling multi tests")


# ============ Fix 7: CHANGELOG + DECISIONS ============

def insert_changelog():
    print("\n=== CHANGELOG.md: prepend [0.8.6] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "## [0.8.6]" in s:
        print("  already present — skip")
        return
    entry = """## [0.8.6] — 2026-10-19

### Добавлено
- **Multi-column `on`** для `shift` / `diff` / `rolling`:
  `on=["ts", "sensor"]` (Sequence[str]).
  - **Лексикографический composite sort**, не per-group. Сортировка по
    `[on_cols..., row_id]`, применение shift/rolling per row, sort-back
    по `__orig_rid` (как в v0.8.2 / v0.8.4).
  - Per-group — `groupby('sensor').shift(1, on='ts')` (v0.8.5a+).
- `CadwDataFrame._normalize_on(on, fn_name)` — helper: `str | Sequence[str]`
  → `list[str]` + validation.
- `_shift_on(..., on_cols: list[str])` / `_rolling_on(..., on_cols:
  list[str])` — принимают pre-normalized list; `external_sort` получает
  `key_columns=on_cols`.

### Backward compat
- `on="ts"` — как раньше (single-element normalization).
- `on=["ts"]` — идентично `on="ts"`.
- `on=[]` — `ValueError: {fn}: on must be non-empty`.

### Ограничения (v0.8.6, §11.5)
- Multi-column `on` — **composite sort**, не per-group. Для per-group
  windows — `groupby(...).shift/rolling(..., on=...)` (v0.8.5a).
- `_iter_shifted` / `_iter_zipped` не дедуплицируются — дублирование
  остаётся (refactor в backlog).
- Stability with duplicate `on` values — tie-break by row_id (heap),
  как в v0.8.2/0.8.4.

### Тесты
- +5 в `test_shift_on_col.py`: multi-key shift/diff, single-element-list
  equivalence, empty-list ValueError, stable tie-break.
- +3 в `test_rolling_on_col.py`: multi-key sum, min_periods=1,
  empty-list ValueError.
- AST guards (v0.8.2/0.8.4) — unaffected (first sort uses `on_cols`
  variable; second still literal `["__orig_rid"]`).
- fast: **~1064 passed**. slow: **8 passed**.

"""
    m = re.search(r'^## \[', s, re.MULTILINE)
    if not m:
        raise SystemExit("no `## [` in CHANGELOG")
    s = s[:m.start()] + entry + s[m.start():]
    p.write_text(s)
    print("  OK: [0.8.6]")


def patch_decisions():
    print("\n=== DECISIONS.md: §11.1 + §11.5 + §13.4/131 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "| Multi-column `on` (v0.8.6) |" not in s:
        s = s.replace(
            "| `rolling(on=col)` (v0.8.4: sort-aware, prototype-validated) | ✅ | 18 |",
            "| `rolling(on=col)` (v0.8.4: sort-aware, prototype-validated) | ✅ | 18 |\n"
            "| Multi-column `on` (v0.8.6: shift/diff/rolling) | ✅ | 8 |",
            1,
        )

    if "| `on` multi-column — composite sort, not per-group (v0.8.6)" not in s:
        s = s.replace(
            "| `rolling` on=col — single column (v0.8.4)",
            "| `on` multi-column — composite sort, not per-group (v0.8.6) | `_shift_on` / `_rolling_on` | `on=[\"ts\",\"sensor\"]` = lexicographic; per-group — `groupby(...)` (v0.8.5a+) |\n"
            "| `rolling` on=col — single column (v0.8.4)",
            1,
        )

    if "Эпизод 131" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 131 (multi-column `on`: composite sort, not per-group).**
v0.8.6 добавляет `on=["ts", "sensor"]` для shift/diff/rolling. Технически
это малая правка — `external_sort` уже принимает `key_columns:
Sequence[str]` и корректно обрабатывает multi-key (per-column `_Key` +
row_id tie-break в heap). Единственное — `_shift_on`/`_rolling_on`
передавали `key_columns=[on]` (single literal); заменили на `on_cols`
(pre-normalized list).

**Семантика (задокументирована, важно не путать):**
- `on=["ts", "sensor"]` — **лексикографический composite sort**.
  Все строки сортируются по (ts, sensor, row_id), затем shift/rolling
  per row, sort-back.
- **НЕ per-group.** Для независимых окон по каждому sensor:
  `groupby('sensor').shift(1, on='ts')` (v0.8.5a+, не реализовано).

Пример: ts=[2,1,1,2], sensor=[a,a,b,b], v=[40,10,20,30]. Sorted v:
[10 (1,a), 20 (1,b), 40 (2,a), 30 (2,b)]. shift(1): [None, 10, 20, 40].
Back to orig: [20, None, 10, 40]. Если бы пользователь ожидал
per-sensor, он получил бы [None, 10, None, 20] — **не наш случай**.

**`_normalize_on` helper** — `str | Sequence[str]` → `list[str]` +
validation (`non-empty`, each in `self._columns`). Вызывается в dispatch
(`shift`/`rolling`) и в `_diff_on` (до numeric-type check на columns,
чтобы `on=[]` давал `ValueError` не `TypeError`).

**AST guards v0.8.2/0.8.4 unaffected:** second `external_sort` всё ещё
`key_columns=["__orig_rid"]` (literal), первый — `key_columns=on_cols`
(variable, `keys=None` в AST). Тест `test_shift_on_second_sort_renames_
row_id` смотрит только `ext_calls[-1]` — passes.

**Урок (малые фичи):** composite sort для multi-key — уже работал в
`external_sort`, единственная ошибка была в caller'е (`[on]` вместо
`on_cols`). Это **не** повод для prototype-first (§13.4/123) — нет
нового алгоритма. Но важен **честный docstring**: "composite sort, not
per-group" — иначе пользователь неправильно поймёт семантику.

**Backlog:** per-group shift/rolling (v0.8.5a+) — **другая** фича:
`HashShuffle` по group, `external_sort` по `on` внутри group, application,
concat. Требует prototype-first (как v0.8.2/v0.8.4).

**Следующее:** v0.8.3 `resample(freq)` — prototype-first, prototype
`scripts/proto_resample.py` (30 строк, 1 agg, bucket boundaries).

'''
        s = s.replace(marker, episode + marker, 1)

    p.write_text(s)
    print("  OK: §11.1, §11.5, §13.4/131")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v086-multi-on"

    print("=== v0.8.6: multi-column on for shift/diff/rolling ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v086-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v086-{ts_str}",
        "-m", "before patch v0.8.6"])

    patch_normalize_helper()
    patch_shift_dispatch()
    patch_shift_on_sig()
    patch_diff_on()
    patch_rolling()
    add_shift_multi_tests()
    add_rolling_multi_tests()
    insert_changelog()
    patch_decisions()

    print("\n=== pytest tests/test_shift_on_col.py tests/test_rolling_on_col.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_shift_on_col.py",
         "tests/test_rolling_on_col.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.6 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest rolling/shift/expanding regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_rolling.py",
         "tests/test_shift_diff.py",
         "tests/test_shift_on_col.py",
         "tests/test_rolling_on_col.py",
         "tests/test_expanding_ewm.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

    print("\n=== pytest tests/test_changelog.py -v (patch branch — skip latest) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_changelog.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** CHANGELOG FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.6: multi-column on for shift/diff/rolling "
        "(lexicographic composite sort; §13.4/131)"])

    print("\n==> v0.8.6 applied on", branch)


if __name__ == "__main__":
    main()