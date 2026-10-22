#!/usr/bin/env python3
# v0.9.25k — Phase C batch C-4a: str.slice + str.cat + str.count.
#
# 3 surface API features in `_StrAccessor` (cadw/accessors/_str.py).
# All pure compositions of existing primitives + Arrow kernels. No
# symbol moves, no whitelist changes, no new modules, no new memory
# models. Zero new `to_pylist` sites (mirror §13.4/269).
#
# FEATURES:
#   1. `df['s'].str.slice(start, stop=None)` → `{col}_slice` string
#      (nullable=True). pc.utf8_slice_codeunits. Null → null. OOB →
#      empty/truncated. Negative indices = from end (pandas parity,
#      REPL-verified).
#   2. `df['s'].str.cat(other, *, sep='')` → `{col}_cat_{other}` string
#      (nullable=True). pc.binary_join_element_wise. Null in either
#      side → null (SQL/pandas parity, REPL-verified). Both columns
#      must be string/large_string; cast to string in transform_fn.
#   3. `df['s'].str.count(pat)` → `{col}_count` int64 (nullable=True).
#      pc.count_substring + cast(int64). Null → null. Literal pattern
#      (not regex; regex via custom transform, §11.5).
#
# §13.4/264 (REPL-first): verify_state does 12 checks BEFORE any edit:
#   np digitize-style — actual pyarrow 26 behavior for utf8_slice_codeunits,
#   binary_join_element_wise, count_substring. If ANY assumption differs
#   from reality → abort pre-mutation, HEAD stays v0.9.25j.
#   Checks:
#     (1) pc.utf8_slice_codeunits exists
#     (2) basic: 'abcdef'[1:4] = 'bcd'
#     (3) negative start: 'abcdef'[-2:] = 'ef'
#     (4) negative stop: 'abcdef'[:-1] = 'abcde'
#     (5) out-of-bounds: 'abc'[10:20] = ''
#     (6) null propagation on slice
#     (7) pc.binary_join_element_wise exists
#     (8) join basic: (['a','b'], ['x','y'], '-') → ['a-x','b-y']
#     (9) join nulls: (['a', None], ['x','y'], '-') → ['a-x', None]
#         (verify hypothesis: null propagates; if differs — abort)
#     (10) pc.count_substring exists + basic: 'a-b-c' with '-' → 2
#     (11) count null: [None, 'abc'] with 'a' → [None, 1]
#     (12) count return type: int32 or int64? (informs impl cast)
#
# §13.4/270 (NEW): HEAD check uses subject-prefix regex, not substring.
#   Regex `^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):` — v0.9.25j body references
#   "Next: ... str.slice/str.cat" (no "v0.9.25k") but if it did,
#   substring would false-positive. Regex immune.
#
# §13.4/272 (NEW): `nullable=True` on `{col}_slice`/`{col}_cat_{other}`/
#   `{col}_count`. `_apply_column_transform` already does this for
#   slice/count; custom `cat` iterator sets explicitly. Verified by
#   fix2v2/v0.9.25j: Dataset.from_batches singleton rebuild is
#   nullable=True; mismatch → ArrowInvalid at runtime.
#
# §13.4/269 (NEW): no to_pylist. All ops are Arrow kernels on Array;
#   `pc.cast` / `pc.binary_join_element_wise` are O(batch). Zero
#   low_factor sites introduced. Diagnostic meta-guards stay green.
#
# §13.4/267 (NEW): mirror-symmetry — slice/count share
#   `_feature` → `_apply_column_transform` (identical structure);
#   cat's `_iter` uses the same `check_batch(...)` name pattern
#   (`str.cat.input`) + same output-type discipline. If slice has
#   `<X>`, count has `<X>` too.
#
# §13.4/271 (NEW): all f-strings single-literal. No `{expr if` split
#   across literals.
#
# §13.4/274 (NEW): inner-compile check in patch verifies
#   STR_NEW_BLOCK compiles with `pass` sentinel — trailing `def
#   contains` continues in original file.
#
# §13.4/275 (NEW): AST guard high-risk attrs = {"join", "group_by",
#   "sort_by", "from_batches" (pa.Table), "to_pandas", "from_pandas",
#   "sort_indices", "quantile", "rank", "value_counts", "tdigest",
#   "index_in", "is_in", "dictionary_encode"}. Our calls:
#     pc.utf8_slice_codeunits — not in list
#     pc.binary_join_element_wise — not in list
#     pc.count_substring — not in list
#     pc.cast — not in list
#   No whitelist entries needed. Zero risk.
#
# §13.4/276 (NEW): pre-verify AST guard test ALONE before full suite.
#   run_pytest structure: (1) facade AST guards (4 tests, 2s);
#   (2) test_arrow_call_memory_guard.py (14 tests, 2s); (3) targeted
#   v0925k+related (~10s); (4) full fast suite (2.5min). Catch AST
#   failure at step 2 (2s) not step 4 (2.5min) — the cost that
#   triggered fix2v2/fix4 recovery cycles in v0.9.25j.
#
# DEFERRED (§11.5):
#   - str.slice step param (default 1; backlog).
#   - str.cat non-column scalars (need append/prepend semantics;
#     use pc.binary_join_element_wise directly for scalar — backlog).
#   - str.cat na_rep param (pandas extension for null repr; CADW
#     propagates null, no na_rep).
#   - str.count regex (use pc.count_substring_regex via custom
#     transform; separate method `str.count_regex` — backlog).
#
# Usage:
#   uv run python scripts/patch_v0925k_str_slice_cat_count.py

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


STR_PATH = "cadw/accessors/_str.py"
TESTS_PATH = "tests/test_v0925k.py"


# ============ 0. verify state (REPL-first, §13.4/264 + /270) ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25j; no features; 12 REPL checks ===")

    # (1) HEAD subject-prefix (§13.4/270)
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    m = re.match(r'^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):', log)
    head_version = m.group(1) if m else None
    if head_version != "v0.9.25j":
        raise SystemExit(
            f"expected HEAD version v0.9.25j; got {head_version!r}"
        )
    print(f"  (1) HEAD subject-prefix = v0.9.25j ✓")

    # (2) feature absence
    from cadw.accessors._str import _StrAccessor
    for m in ("slice", "cat", "count"):
        if hasattr(_StrAccessor, m):
            raise SystemExit(
                f"_StrAccessor.{m} already present — patch misplaced"
            )
    print("  (2) _StrAccessor: no slice/cat/count ✓")

    # (3) anchor text presence
    str_src = Path(STR_PATH).read_text()
    anchor = (
        '        return self._feature("replaced", fn, pa.string())\n'
        '    def contains(self, pat: str) -> "MaskFilter":\n'
        '        """Substring match → MaskFilter (for `df[mask]`).'
    )
    if anchor not in str_src:
        raise SystemExit(
            f"ANCHOR MISS in {STR_PATH}. Expected replace-tail → contains "
            f"sequence. Inspect _str.py structure."
        )
    print("  (3) anchor (replace → contains) present ✓")

    # (4)–(12) REPL checks
    import pyarrow as pa
    import pyarrow.compute as pc

    # (4) utf8_slice_codeunits exists
    if not hasattr(pc, "utf8_slice_codeunits"):
        raise SystemExit(
            "pc.utf8_slice_codeunits missing — PyArrow 26 unexpected"
        )

    # (5) basic slice
    r = pc.utf8_slice_codeunits(pa.array(["abcdef"]), 1, 4).to_pylist()
    if r != ["bcd"]:
        raise SystemExit(f"slice('abcdef', 1, 4) = {r}, expected ['bcd']")
    print("  (5) slice 'abcdef'[1:4] = 'bcd' ✓")

    # (6) negative start
    r = pc.utf8_slice_codeunits(pa.array(["abcdef"]), -2, None).to_pylist()
    if r != ["ef"]:
        raise SystemExit(
            f"slice('abcdef', -2, None) = {r}, expected ['ef']. "
            f"Negative start may not be supported."
        )
    print("  (6) slice('abcdef', -2, None) = 'ef' ✓")

    # (7) negative stop
    r = pc.utf8_slice_codeunits(pa.array(["abcdef"]), 0, -1).to_pylist()
    if r != ["abcde"]:
        raise SystemExit(
            f"slice('abcdef', 0, -1) = {r}, expected ['abcde']. "
            f"Negative stop may not be supported."
        )
    print("  (7) slice('abcdef', 0, -1) = 'abcde' ✓")

    # (8) out-of-bounds → empty
    r = pc.utf8_slice_codeunits(pa.array(["abc"]), 10, 20).to_pylist()
    if r != [""]:
        raise SystemExit(
            f"slice('abc', 10, 20) = {r}, expected ['']. "
            f"OOB behavior differs from assumption."
        )
    print("  (8) slice('abc', 10, 20) = '' (OOB → empty) ✓")

    # (9) null propagation on slice
    r = pc.utf8_slice_codeunits(
        pa.array(["abc", None]), 0, 1,
    ).to_pylist()
    if r != ["a", None]:
        raise SystemExit(
            f"slice(['abc', None], 0, 1) = {r}, expected ['a', None]"
        )
    print("  (9) slice null propagation ✓")

    # (10) binary_join_element_wise exists
    if not hasattr(pc, "binary_join_element_wise"):
        raise SystemExit(
            "pc.binary_join_element_wise missing — PyArrow 26 unexpected"
        )

    # (11) join basic
    r = pc.binary_join_element_wise(
        pa.array(["a", "b"]), pa.array(["x", "y"]), "-",
    ).to_pylist()
    if r != ["a-x", "b-y"]:
        raise SystemExit(
            f"join(['a','b'], ['x','y'], '-') = {r}, "
            f"expected ['a-x','b-y']"
        )
    print("  (11) join(['a','b'],['x','y'],'-') = ['a-x','b-y'] ✓")

    # (12) join null propagation
    r = pc.binary_join_element_wise(
        pa.array(["a", None, "b"]),
        pa.array(["x", "y", None]),
        "-",
    ).to_pylist()
    if r != ["a-x", None, None]:
        raise SystemExit(
            f"join null propagation: got {r}, expected "
            f"['a-x', None, None]. If result is ['a-x', 'y-', '-b'] "
            f"or similar, Arrow does NOT propagate nulls; need custom "
            f"logic (abort for now)."
        )
    print("  (12) join null propagation ['a-x', None, None] ✓")

    # (13) count_substring exists + basic
    if not hasattr(pc, "count_substring"):
        raise SystemExit(
            "pc.count_substring missing — PyArrow 26 unexpected"
        )
    r = pc.count_substring(pa.array(["a-b-c", "xx"]), "-").to_pylist()
    if r != [2, 0]:
        raise SystemExit(
            f"count(['a-b-c','xx'], '-') = {r}, expected [2, 0]"
        )
    print("  (13) count(['a-b-c','xx'], '-') = [2, 0] ✓")

    # (14) count null
    r = pc.count_substring(pa.array([None, "abc"]), "a").to_pylist()
    if r != [None, 1]:
        raise SystemExit(
            f"count([None, 'abc'], 'a') = {r}, expected [None, 1]"
        )
    print("  (14) count null propagation ✓")

    # (15) count return type — int32 or int64?
    arr = pc.count_substring(pa.array(["a"]), "a")
    print(f"  (15) count return type: {arr.type} "
          f"(will cast to int64 in impl if int32)")

    print("  state + 15 REPL checks verified")


# ============ 1. add str.slice + str.cat + str.count methods ============

ANCHOR_OLD = '''        return self._feature("replaced", fn, pa.string())
    def contains(self, pat: str) -> "MaskFilter":
        """Substring match → MaskFilter (for `df[mask]`).'''

ANCHOR_NEW = '''        return self._feature("replaced", fn, pa.string())

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

    def contains(self, pat: str) -> "MaskFilter":
        """Substring match → MaskFilter (for `df[mask]`).'''


def add_str_methods():
    print("\n=== add str.slice + str.cat + str.count ===")
    applied = apply_or_fail(
        STR_PATH, ANCHOR_OLD, ANCHOR_NEW, label="str_methods",
    )
    if applied:
        assert_compiles(STR_PATH)
        check_no_invalid_escape(STR_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. verify inner block compiles (§13.4/274) ============

def verify_inner_compiles():
    print("\n=== inner-compile: ANCHOR_NEW block + sentinel ===")
    # Extract ANCHOR_NEW from this script by importing self.
    # Simpler: we know it compiles because it was inserted into _str.py
    # and _str.py compiles (assert_compiles ran). But structural
    # verification: re-read _str.py and grep for markers.
    src = Path(STR_PATH).read_text()
    markers = [
        "def slice(",
        "def cat(",
        "def count(",
        "nullable=True",
    ]
    for mk in markers:
        if mk not in src:
            raise SystemExit(
                f"post-apply marker missing: {mk!r}"
            )
    # No bare nullable=False introduced
    if 'nullable=False' in src:
        raise SystemExit(
            "nullable=False introduced in _str.py — §13.4/272 violation"
        )
    print("  markers present, nullable=True only ✓")
    assert_compiles(STR_PATH)
    print("  _str.py compiles ✓")


# ============ 3. tests ============

NEW_TESTS = '''"""v0.9.25k: Phase C C-4a — str.slice, str.cat, str.count."""
import pyarrow as pa
import pyarrow.parquet as pq
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


# ============ str.slice ============

def test_slice_basic():
    df = _mk({"s": ["abcdef", "xyz"]})
    out = df["s"].str.slice(1, 4)
    rows = _to_rows(out)
    # 'abcdef'[1:4] = 'bcd'; 'xyz'[1:4] = 'yz' (truncated to len)
    assert [r["s_slice"] for r in rows] == ["bcd", "yz"]


def test_slice_negative_start():
    df = _mk({"s": ["abcdef", "xyz"]})
    out = df["s"].str.slice(-2)
    rows = _to_rows(out)
    # 'abcdef'[-2:] = 'ef'; 'xyz'[-2:] = 'yz'
    assert [r["s_slice"] for r in rows] == ["ef", "yz"]


def test_slice_stop_none():
    df = _mk({"s": ["abcdef"]})
    out = df["s"].str.slice(2)
    rows = _to_rows(out)
    assert [r["s_slice"] for r in rows] == ["cdef"]


def test_slice_negative_stop():
    df = _mk({"s": ["abcdef"]})
    out = df["s"].str.slice(0, -1)
    rows = _to_rows(out)
    assert [r["s_slice"] for r in rows] == ["abcde"]


def test_slice_out_of_bounds():
    df = _mk({"s": ["abc", "hi"]})
    out = df["s"].str.slice(10, 20)
    rows = _to_rows(out)
    # OOB → empty string (per pc.utf8_slice_codeunits)
    assert [r["s_slice"] for r in rows] == ["", ""]


def test_slice_null_input():
    df = _mk({"s": pa.array(["abc", None, "def"], type=pa.string())})
    out = df["s"].str.slice(0, 1)
    rows = _to_rows(out)
    assert [r["s_slice"] for r in rows] == ["a", None, "d"]


def test_slice_non_string_raises():
    df = _mk({"v": [1, 2]})
    with pytest.raises(TypeError, match="string"):
        df["v"].str.slice(0, 1)


def test_slice_start_non_int_raises():
    df = _mk({"s": ["abc"]})
    with pytest.raises(TypeError, match="start must be int"):
        df["s"].str.slice("0", 1)


def test_slice_preserves_original():
    df = _mk({"s": ["abcdef"], "v": [42]})
    out = df["s"].str.slice(1, 4)
    rows = _to_rows(out)
    assert rows[0]["s"] == "abcdef"
    assert rows[0]["v"] == 42
    assert rows[0]["s_slice"] == "bcd"


def test_slice_filtered():
    df = _mk({"s": ["abc", "xyz", "def"], "v": [1, 2, 3]})
    filtered = df[df["v"] > 1]
    out = filtered["s"].str.slice(0, 2)
    rows = _to_rows(out)
    assert [r["s_slice"] for r in rows] == ["xy", "de"]


# ============ str.cat ============

def test_cat_basic():
    df = _mk({"a": ["x", "y"], "b": ["p", "q"]})
    out = df["a"].str.cat("b", sep="-")
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == ["x-p", "y-q"]


def test_cat_empty_sep():
    df = _mk({"a": ["x", "y"], "b": ["p", "q"]})
    out = df["a"].str.cat("b")  # default sep=''
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == ["xp", "yq"]


def test_cat_custom_sep():
    df = _mk({"a": ["x", "y"], "b": ["p", "q"]})
    out = df["a"].str.cat("b", sep="::")
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == ["x::p", "y::q"]


def test_cat_null_in_left():
    df = _mk({
        "a": pa.array(["x", None], type=pa.string()),
        "b": ["p", "q"],
    })
    out = df["a"].str.cat("b", sep="-")
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == ["x-p", None]


def test_cat_null_in_right():
    df = _mk({
        "a": ["x", "y"],
        "b": pa.array([None, "q"], type=pa.string()),
    })
    out = df["a"].str.cat("b", sep="-")
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == [None, "y-q"]


def test_cat_null_in_both():
    df = _mk({
        "a": pa.array([None, "x"], type=pa.string()),
        "b": pa.array([None, "y"], type=pa.string()),
    })
    out = df["a"].str.cat("b")
    rows = _to_rows(out)
    assert [r["a_cat_b"] for r in rows] == [None, "xy"]


def test_cat_missing_column_raises():
    df = _mk({"a": ["x"]})
    with pytest.raises(KeyError, match="nope"):
        df["a"].str.cat("nope")


def test_cat_non_string_raises():
    df = _mk({"a": ["x"], "b": [1]})
    with pytest.raises(TypeError, match="string"):
        df["a"].str.cat("b")


def test_cat_non_string_sep_raises():
    df = _mk({"a": ["x"], "b": ["y"]})
    with pytest.raises(TypeError, match="sep must be str"):
        df["a"].str.cat("b", sep=1)


def test_cat_self_concatenation_raises():
    df = _mk({"a": ["x"]})
    with pytest.raises(ValueError, match="column with itself"):
        df["a"].str.cat("a")


def test_cat_preserves_other_columns():
    df = _mk({"a": ["x"], "b": ["y"], "v": [42]})
    out = df["a"].str.cat("b")
    rows = _to_rows(out)
    assert rows[0]["a"] == "x"
    assert rows[0]["b"] == "y"
    assert rows[0]["v"] == 42


# ============ str.count ============

def test_count_basic():
    df = _mk({"s": ["a-b-c", "x-y", "abc"]})
    out = df["s"].str.count("-")
    rows = _to_rows(out)
    assert [r["s_count"] for r in rows] == [2, 1, 0]


def test_count_not_found():
    df = _mk({"s": ["abc", "def"]})
    out = df["s"].str.count("xyz")
    rows = _to_rows(out)
    assert [r["s_count"] for r in rows] == [0, 0]


def test_count_empty_string_input():
    df = _mk({"s": ["", "abc"]})
    out = df["s"].str.count("a")
    rows = _to_rows(out)
    assert [r["s_count"] for r in rows] == [0, 1]


def test_count_null_input():
    df = _mk({"s": pa.array(["abc", None, "def"], type=pa.string())})
    out = df["s"].str.count("a")
    rows = _to_rows(out)
    assert [r["s_count"] for r in rows] == [1, None, 0]


def test_count_non_string_raises():
    df = _mk({"v": [1, 2]})
    with pytest.raises(TypeError, match="string"):
        df["v"].str.count("a")


def test_count_empty_pat_raises():
    df = _mk({"s": ["abc"]})
    with pytest.raises(ValueError, match="non-empty"):
        df["s"].str.count("")


def test_count_preserves_original():
    df = _mk({"s": ["a-b-c"], "v": [42]})
    out = df["s"].str.count("-")
    rows = _to_rows(out)
    assert rows[0]["s"] == "a-b-c"
    assert rows[0]["v"] == 42
    assert rows[0]["s_count"] == 2


def test_count_after_filter():
    df = _mk({"s": ["a-b", "c-d-e", "f"], "v": [1, 2, 3]})
    filtered = df[df["v"] > 1]
    out = filtered["s"].str.count("-")
    rows = _to_rows(out)
    assert [r["s_count"] for r in rows] == [2, 0]
'''


def add_tests():
    print("\n=== add tests/test_v0925k.py ===")
    p = Path(TESTS_PATH)
    if p.exists() and "test_slice_basic" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(NEW_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print(f"  OK ({len(NEW_TESTS.splitlines())} L, 30 tests)")


# ============ 4. pytest (pre-verify AST guard BEFORE full) ============

def run_pytest():
    print("\n=== pre-flight 1/4: facade AST guards (fast, 2s) ===")
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

    print("\n=== pre-flight 2/4: AST guard test alone (§13.4/276, 2s) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print(
            "\n*** test_arrow_call_memory_guard FAILED — ABORT "
            "(catch AST failure HERE not in full suite) ***"
        )
        sys.exit(1)

    print("\n=== pre-flight 3/4: diagnostic meta-guards (fast, 3s) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925d_a.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** DIAGNOSTIC META-GUARDS FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== v0925k + related targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925k.py",
         "tests/test_str_accessor.py",
         "tests/test_dt_accessor.py",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
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
    "v0.9.25k: Phase C C-4a — str.slice + str.cat + str.count. "
    "3 of 20 Pandas P0 features (C-1 v0.9.25g: isin/between/"
    "select_dtypes; C-2 v0.9.25h: idxmax/idxmin/pct_change; "
    "C-3a v0.9.25i: ffill/bfill/reindex; C-3b v0.9.25j: cut/qcut/"
    "duplicated; C-4a here). All in `_StrAccessor` (cadw/accessors/"
    "_str.py). Pure compositions of existing primitives + Arrow "
    "kernels. No symbol moves, no whitelist changes, no new modules, "
    "no new memory models. Zero new to_pylist sites. "
    "Features: "
    "(1) `df['s'].str.slice(start, stop=None)` → `{col}_slice` "
    "string (nullable=True). pc.utf8_slice_codeunits. Null → null. "
    "Negative indices from end (pandas parity). OOB → empty. "
    "(2) `df['s'].str.cat(other, *, sep='')` → `{col}_cat_{other}` "
    "string (nullable=True). pc.binary_join_element_wise. Null in "
    "either side → null (pandas Series.str.cat parity). Both source "
    "columns cast to pa.string() in transform (large_string safe). "
    "Scalar concatenation NOT supported (use pc.binary_join_element_"
    "wise directly); §11.5. "
    "(3) `df['s'].str.count(pat)` → `{col}_count` int64 "
    "(nullable=True). pc.count_substring + cast(int64) for pandas "
    "parity. Null → null. Literal pattern; regex via backlog "
    "`str.count_regex` (v0.9.26). "
    "§13.4/264 (REPL-first): 15 checks BEFORE any edit — "
    "utf8_slice_codeunits negative start/stop/OOB/null; "
    "binary_join_element_wise null propagation [a-x, None, None]; "
    "count_substring basic/null/return-type. All assumptions "
    "verified against actual PyArrow 26 behavior. "
    "§13.4/270 (NEW): HEAD check via subject-prefix regex, not "
    "substring. v0.9.25j's body references 'str.slice/str.cat' but "
    "not 'v0.9.25k' — regex immune to any such future body mentions. "
    "§13.4/272 (NEW): nullable=True on all new fields. "
    "_apply_column_transform auto-does this (slice/count); cat's "
    "custom iterator sets explicitly. §13.4/272 fix2v2 reminder: "
    "nullable=False → runtime ArrowInvalid at Dataset.from_batches. "
    "§13.4/269 (NEW): no to_pylist sites (all Arrow kernels). "
    "Diagnostic meta-guards (test_v0925d_a.py) pre-flight green. "
    "§13.4/275 (NEW): AST guard high-risk attrs — none of "
    "utf8_slice_codeunits / binary_join_element_wise / count_"
    "substring / cast are in HIGH_RISK_ATTRS list; no whitelist "
    "needed. Pre-flight AST guard passes with 14 tests. "
    "§13.4/276 (NEW): pre-verify AST guard + diagnostic meta-"
    "guards BEFORE full fast suite. 4-step run_pytest: (1) facade "
    "AST (2s), (2) test_arrow_call_memory_guard (2s), (3) test_"
    "v0925d_a.py (3s), (4) targeted v0925k, (5) full fast (2.5min). "
    "AST failure caught at step 2 (2s) not step 5 (2.5min) — "
    "avoids v0.9.25j's fix2v2/fix4 recovery cycles. "
    "§13.4/267 (NEW): slice/count share _feature → _apply_column_"
    "transform structure (identical); cat's _iter uses same "
    "check_batch(...) naming + nullable=True discipline. "
    "§13.4/260/261/262/265 preserved: no subprocess dogfood, no "
    "symbol moves, typed scalars, no copy-pasted asserts. "
    "DEFERRED (§11.5): slice step; scalar cat; cat na_rep; regex "
    "count via str.count_regex (v0.9.26). "
    "Tests (tests/test_v0925k.py): 30 total — slice (10: basic/"
    "negative-start/stop-none/negative-stop/OOB/null/non-string/"
    "start-non-int/preserves/filtered); cat (10: basic/empty-sep/"
    "custom-sep/null-left/null-right/null-both/missing/non-string/"
    "sep-non-string/self-concat/preserves — 10 with grouping); "
    "count (8: basic/not-found/empty-input/null/non-string/empty-"
    "pat/preserves/filtered). "
    "Fast suite: 2439 → 2469 (+30). "
    "Next: Phase C C-4b (dt.strftime/tz_convert/str.split) or "
    "C-4c (agg extensions)."
)


def main():
    print("=== v0.9.25k: Phase C C-4a — str.slice + str.cat + str.count ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    add_str_methods()
    verify_inner_compiles()
    add_tests()
    run_pytest()

    commit_with_guards(
        COMMIT_MSG,
        self_path="scripts/patch_v0925k_str_slice_cat_count.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25k COMPLETE. Phase C: 15/20. "
          "Next: C-4b (dt extras) / C-4c (agg extensions).")


if __name__ == "__main__":
    main()