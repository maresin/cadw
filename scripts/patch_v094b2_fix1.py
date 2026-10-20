#!/usr/bin/env python3
# v0.9.4b-2-fix1: _DtAccessor._feature signature; use _feature_custom.
# Run: uv run python scripts/patch_v094b2_fix1.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


def add_feature_custom():
    print("\n=== _dt.py: add _feature_custom ===")
    p = Path("cadw/accessors/_dt.py")
    s = p.read_text()
    if "def _feature_custom(self" in s:
        print("  already present - skip")
        return
    anchor = (
        '    def _feature(self, name: str) -> "CadwDataFrame":\n'
        '        fn, ot = _DT_FEATURES[name]\n'
        '        return self._parent._apply_column_transform(\n'
        '            self._column, name, lambda arr: fn(arr), ot, accessor="dt",\n'
        '        )\n'
    )
    if anchor not in s:
        raise SystemExit("_feature anchor not found")
    L = []
    L.append('')
    L.append('    def _feature_custom(self, name, fn, out_type):')
    L.append('        # v0.9.4b-2-fix1: _feature(name) uses static _DT_FEATURES;')
    L.append('        # _feature_custom(name, fn, out_type) allows per-call fn+type')
    L.append('        # (floor/ceil/round/week_of_year). Symmetric to str._feature.')
    L.append('        return self._parent._apply_column_transform(')
    L.append('            self._column, name, fn, out_type, accessor="dt",')
    L.append('        )')
    L.append('')
    block = "\n".join(L)
    s = s.replace(anchor, anchor + block, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def fix_dt_methods():
    print("\n=== _dt.py: floor/ceil/round/week_of_year -> _feature_custom ===")
    p = Path("cadw/accessors/_dt.py")
    s = p.read_text()

    # floor
    old_floor = (
        '    def floor(self, freq: str) -> "CadwDataFrame":\n'
        '        return self._feature(\n'
        '            "floor",\n'
        '            lambda arr: _freq_bucket(arr, freq, "floor"),\n'
        '            self._parent._dataset.schema.field(self._column).type,\n'
        '        )'
    )
    new_floor = (
        '    def floor(self, freq: str) -> "CadwDataFrame":\n'
        '        return self._feature_custom(\n'
        '            "floor",\n'
        '            lambda arr: _freq_bucket(arr, freq, "floor"),\n'
        '            self._parent._dataset.schema.field(self._column).type,\n'
        '        )'
    )
    if new_floor in s:
        print("  floor already fixed")
    elif old_floor in s:
        s = s.replace(old_floor, new_floor, 1)
    else:
        raise SystemExit("floor anchor not found")

    # ceil
    old_ceil = (
        '    def ceil(self, freq: str) -> "CadwDataFrame":\n'
        '        return self._feature(\n'
        '            "ceil",\n'
        '            lambda arr: _freq_bucket(arr, freq, "ceil"),\n'
        '            self._parent._dataset.schema.field(self._column).type,\n'
        '        )'
    )
    new_ceil = (
        '    def ceil(self, freq: str) -> "CadwDataFrame":\n'
        '        return self._feature_custom(\n'
        '            "ceil",\n'
        '            lambda arr: _freq_bucket(arr, freq, "ceil"),\n'
        '            self._parent._dataset.schema.field(self._column).type,\n'
        '        )'
    )
    if new_ceil in s:
        print("  ceil already fixed")
    elif old_ceil in s:
        s = s.replace(old_ceil, new_ceil, 1)
    else:
        raise SystemExit("ceil anchor not found")

    # round
    old_round = (
        '    def round(self, freq: str) -> "CadwDataFrame":\n'
        '        return self._feature(\n'
        '            "round",\n'
        '            lambda arr: _freq_bucket(arr, freq, "round"),\n'
        '            self._parent._dataset.schema.field(self._column).type,\n'
        '        )'
    )
    new_round = (
        '    def round(self, freq: str) -> "CadwDataFrame":\n'
        '        return self._feature_custom(\n'
        '            "round",\n'
        '            lambda arr: _freq_bucket(arr, freq, "round"),\n'
        '            self._parent._dataset.schema.field(self._column).type,\n'
        '        )'
    )
    if new_round in s:
        print("  round already fixed")
    elif old_round in s:
        s = s.replace(old_round, new_round, 1)
    else:
        raise SystemExit("round anchor not found")

    # week_of_year
    old_week = (
        '    @property\n'
        '    def week_of_year(self) -> "CadwDataFrame":\n'
        '        # ISO week (1-53). For US week numbering -- backlog.\n'
        '        return self._feature(\n'
        '            "week_of_year",\n'
        '            lambda arr: pc.iso_week(arr),\n'
        '            pa.int16(),\n'
        '        )'
    )
    new_week = (
        '    @property\n'
        '    def week_of_year(self) -> "CadwDataFrame":\n'
        '        # ISO week (1-53). For US week numbering -- backlog.\n'
        '        return self._feature_custom(\n'
        '            "week_of_year",\n'
        '            lambda arr: pc.iso_week(arr),\n'
        '            pa.int16(),\n'
        '        )'
    )
    if new_week in s:
        print("  week_of_year already fixed")
    elif old_week in s:
        s = s.replace(old_week, new_week, 1)
    else:
        raise SystemExit("week_of_year anchor not found")

    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def main():
    print("=== v0.9.4b-2-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    add_feature_custom()
    fix_dt_methods()

    print("\n=== pytest: v094b2 tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_dataframe_v094b2.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: dt/str regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dt_accessor.py",
         "tests/test_str_accessor.py",
         "tests/test_row_filter.py",
         "-q"],
        timeout=300,
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
        "v0.9.4b-2-fix1: _DtAccessor._feature(name) static; new "
        "_feature_custom(name, fn, out_type) for floor/ceil/round/"
        "week_of_year (fixes 4 tests); str._feature was already 3-arg",
    )
    print("\n==> v0.9.4b-2-fix1 applied.")


if __name__ == "__main__":
    main()
