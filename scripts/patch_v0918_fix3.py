#!/usr/bin/env python3
# v0.9.18-fix3 — index.py numpy paths: import + type safety + AST whitelist.
# v0.9.18 COMPLETE.
#
# v0.9.18-fix2 pytest (fast) 14 failures:
#   (a) test_no_unguarded_high_risk_arrow_calls — dictionary_encode()
#       in _string_hash_column without preceding check_batch (§3.11
#       AST guard). Root: v0.9.18 §1.3 added dict_encode without
#       whitelisting; caller chain hash_shuffle._process →
#       check_batch(extra_factor=3.0) → compute_hashes →
#       _hash_column → _string_hash_column → dict_encode. AST
#       guard sees only local scope; cross-function chain invisible.
#       Fix: whitelist cadw/ops/shuffle.py:_string_hash_column with
#       rationale (dict_encode O(K + batch) ≤ 3×batch.nbytes
#       covered by caller check_batch).
#   (b) 13 test failures: NameError: name 'np' is not defined in
#       MaterializedIndex.is_monotonic_increasing/decreasing. Root:
#       v0.9.18 §1.6 used np.all/np.flatnonzero but index.py has
#       no `import numpy as np` at module level. Fix: add it.
#   (c) test_loc_tuple_on_single_index_raises_keyerror — expected
#       KeyError, got no raise. Root: get_positions_all numpy fast
#       path does `np_arr == ("a", 1)` → numpy coerces tuple to
#       string array, matches first element → false positive
#       position [0], no KeyError. Fix: numpy only for numeric/
#       temporal types + scalar labels.
#
# §13.4/229 (new): numpy element-wise `arr == label` coerces label
# to array dtype. For string index + int label, `np.array(["1"]) ==
# 1` → [True] (str coercion). Prior to_pylist + `"1" == 1` → False
# (type strict). Numpy fast path only safe for same-type scalar
# comparisons; for string/object use Python loop. Same class as
# §13.4/223 (bool before int) + §13.4/226 (nullable int → float64):
# numpy implicit dtype coercion loses Arrow type strictness.
#
# Bonus latent bug (not in tests, but real): has_duplicates v0.9.18
# uses count_distinct default mode='only_valid' — nulls counted
# separately from values → mixed null/value dup detection wrong.
# Fix: mode='all' (counts null as a value, matches pandas
# Index.has_duplicates).
#
# Run: uv run python scripts/patch_v0918_fix3.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) AST whitelist for _string_hash_column ============

def whitelist_string_hash():
    print("\n=== (a) whitelist _string_hash_column dictionary_encode ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if "cadw/ops/shuffle.py:_string_hash_column" in s:
        print("  already whitelisted — skip")
        return
    anchor = '''    "cadw/ops/sort.py:_write_run":
        "buf ≤ 0.2×avail; sort_indices per batch",'''
    if anchor not in s:
        raise SystemExit("whitelist anchor not found")
    new_entry = '''
    "cadw/ops/shuffle.py:_string_hash_column":
        "dict_encode O(K + batch) where K = uniques-in-batch; caller "
        "hash_shuffle._process check_batch(extra_factor=3.0) covers "
        "peak ≤ 3×batch.nbytes; §13.4/226/227 null edges handled",'''
    s = s.replace(anchor, anchor + new_entry, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (b) import numpy in index.py ============

def add_numpy_import():
    print("\n=== (b) index.py: import numpy as np ===")
    p = Path("cadw/dataset/index.py")
    s = p.read_text()
    if "\nimport numpy as np\n" in s:
        print("  already imported — skip")
        return
    anchor = "import pyarrow as pa\n"
    if anchor not in s:
        raise SystemExit("index.py pa import anchor not found")
    new = "import numpy as np\nimport pyarrow as pa\n"
    s = s.replace(anchor, new, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (c) get_positions_all: numpy only numeric/temporal scalar ============

def fix_get_positions_all_safety():
    print("\n=== (c) get_positions_all: numpy only numeric+scalar ===")
    p = Path("cadw/dataset/index.py")
    s = p.read_text()
    if "# §13.4/229" in s and "numeric + scalar" in s:
        print("  already fixed — skip")
        return
    old_block = '''    def get_positions_all(self, label) -> list[int]:
        """All 0-based positions matching `label` (v0.4.1).

        v0.9.18 (AUDIT §1.6): numpy `flatnonzero` for numeric
        indexes (peak O(n) bool mask, not O(n) Python objects).
        """
        import numpy as _np
        try:
            np_arr = self._values.to_numpy(zero_copy_only=False)
            return _np.flatnonzero(np_arr == label).tolist()
        except (TypeError, ValueError):
            # Fallback for object / unhashable
            v = self._values.to_pylist()
            positions: list[int] = []
            for i, x in enumerate(v):
                try:
                    if x == label:
                        positions.append(i)
                except TypeError:
                    pass
            return positions'''

    new_block = '''    def get_positions_all(self, label) -> list[int]:
        """All 0-based positions matching `label` (v0.4.1).

        v0.9.18 (AUDIT §1.6): numpy `flatnonzero` for numeric
        indexes (peak O(n) bool mask, not O(n) Python objects).

        v0.9.18-fix3 (§13.4/229): numpy fast path restricted to
        numeric + scalar labels. For string/object arrays, numpy
        coerces label to array dtype (`np.array(["1"]) == 1` →
        [True] str coercion), losing Arrow type strictness. For
        tuple labels on single index, numpy coerces tuple to str
        array → false positives. Fallback: Python loop (type-strict,
        matches pre-v0.9.18 semantics).
        """
        import numpy as _np
        t = self._values.type
        is_scalar = not isinstance(label, (tuple, list))
        is_numeric = (
            pa.types.is_integer(t) or pa.types.is_floating(t)
        )
        if is_scalar and is_numeric:
            try:
                np_arr = self._values.to_numpy(zero_copy_only=False)
                if np_arr.dtype != object:
                    return _np.flatnonzero(np_arr == label).tolist()
            except (TypeError, ValueError):
                pass
        # Fallback: Python loop (type-strict, handles strings,
        # nulls, non-scalar labels; matches pandas label semantics)
        v = self._values.to_pylist()
        positions: list[int] = []
        for i, x in enumerate(v):
            try:
                if x == label:
                    positions.append(i)
            except TypeError:
                pass
        return positions'''

    if old_block not in s:
        raise SystemExit("get_positions_all block anchor not found")
    s = s.replace(old_block, new_block, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (d) has_duplicates: mode='all' (null as value) ============

def fix_has_duplicates_null_semantics():
    print("\n=== (d) has_duplicates: count_distinct mode='all' ===")
    p = Path("cadw/dataset/index.py")
    s = p.read_text()
    if "mode='all'" in s and "has_duplicates" in s.split("mode='all'")[1][:500]:
        print("  already fixed — skip")
        return
    old_block = '''    def has_duplicates(self) -> bool:
        """True iff at least one label appears more than once.

        v0.9.18 (AUDIT §1.6): count_distinct (Arrow-native hash,
        O(K) peak not O(N)).
        """
        import pyarrow.compute as _pc
        n = len(self._values)
        if n <= 1:
            return False
        try:
            distinct = int(_pc.count_distinct(self._values).as_py())
            return distinct < n
        except (TypeError, AttributeError):
            # Fallback for non-hashable types
            v = self._values.to_pylist()
            seen: list = []
            for x in v:
                if x in seen:
                    return True
                seen.append(x)
            return False'''

    new_block = '''    def has_duplicates(self) -> bool:
        """True iff at least one label appears more than once.

        v0.9.18 (AUDIT §1.6): count_distinct (Arrow-native hash,
        O(K) peak not O(N)).

        v0.9.18-fix3: mode='all' — count null as a value (matches
        pandas Index.has_duplicates). v0.9.18 default mode=
        'only_valid' skipped nulls → mixed null/value duplicates
        detected wrong (e.g. [1, None, 2] gave True incorrectly).
        """
        import pyarrow.compute as _pc
        n = len(self._values)
        if n <= 1:
            return False
        try:
            distinct = int(
                _pc.count_distinct(self._values, mode='all').as_py()
            )
            return distinct < n
        except (TypeError, AttributeError, ValueError):
            # Fallback for non-hashable types
            v = self._values.to_pylist()
            seen: list = []
            for x in v:
                if x in seen:
                    return True
                seen.append(x)
            return False'''

    if old_block not in s:
        raise SystemExit("has_duplicates block anchor not found")
    s = s.replace(old_block, new_block, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (e) __iter__ null-aware ============

def fix_iter_null_aware():
    print("\n=== (e) __iter__: null-aware numpy/fallback ===")
    p = Path("cadw/dataset/index.py")
    s = p.read_text()
    if "if self._values.null_count > 0:\n            yield from self._values.to_pylist()" in s:
        print("  already fixed — skip")
        return
    old_block = '''    def __iter__(self) -> Iterator:
        # v0.9.18 (AUDIT §1.6): numpy iteration (zero-copy for numeric).
        # Prior `to_pylist()` materialized full Python list (40 MB / 1M ints).
        # Now O(1) peak — one numpy scalar at a time.
        np_arr = self._values.to_numpy(zero_copy_only=False)
        for x in np_arr:
            yield x.item() if hasattr(x, "item") else x'''

    new_block = '''    def __iter__(self) -> Iterator:
        # v0.9.18 (AUDIT §1.6): numpy iteration (zero-copy for numeric).
        # Prior `to_pylist()` materialized full Python list (40 MB / 1M ints).
        # Now O(1) peak — one numpy scalar at a time.
        # v0.9.18-fix3: null-containing falls back to to_pylist (numpy
        # gives NaN for nulls, not None — semantic difference).
        if self._values.null_count > 0:
            yield from self._values.to_pylist()
            return
        np_arr = self._values.to_numpy(zero_copy_only=False)
        for x in np_arr:
            yield x.item() if hasattr(x, "item") else x'''

    if old_block not in s:
        raise SystemExit("__iter__ block anchor not found")
    s = s.replace(old_block, new_block, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (f) regression tests ============

def add_regression_tests():
    print("\n=== (f) test_loc.py: get_positions_all type safety tests ===")
    p = Path("tests/test_loc.py")
    s = p.read_text()
    if "test_get_positions_all_string_index_type_strict" in s:
        print("  already present — skip")
        return
    old = '''def test_get_positions_all_missing():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert idx.get_positions_all(99) == []'''
    new = '''def test_get_positions_all_missing():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert idx.get_positions_all(99) == []


def test_get_positions_all_string_index_type_strict():
    """v0.9.18-fix3 (§13.4/229): string index + int label → no match."""
    idx = MaterializedIndex(pa.array(["1", "2", "3"]))
    assert idx.get_positions_all(1) == []
    assert idx.get_positions_all("1") == [0]


def test_get_positions_all_tuple_label_single_index():
    """v0.9.18-fix3 (§13.4/229): non-scalar label on single index → no match."""
    idx = MaterializedIndex(pa.array(["a", "b", "c"]))
    assert idx.get_positions_all(("a", 1)) == []
    assert idx.get_positions_all(["a"]) == []'''
    if old not in s:
        raise SystemExit("test_loc.py anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.18-fix3 + regression ===")
    pu.verify_anchors([
        # fix3
        ("tests/test_arrow_call_memory_guard.py",
         "cadw/ops/shuffle.py:_string_hash_column"),
        ("cadw/dataset/index.py", "import numpy as np"),
        ("cadw/dataset/index.py", "§13.4/229"),
        ("cadw/dataset/index.py", "mode='all'"),
        ("tests/test_loc.py", "test_get_positions_all_string_index_type_strict"),
        ("tests/test_loc.py", "test_get_positions_all_tuple_label_single_index"),
        # regression
        ("cadw/ops/shuffle.py", "§13.4/227"),
        ("cadw/ops/shuffle.py", "§13.4/226"),
        ("cadw/ops/shuffle.py", "v0.9.18 (AUDIT §1.3)"),
        ("cadw/dataframe.py", "class _ArrowTail:"),
        ("cadw/ops/rank.py", "v0.9.18 (AUDIT §1.2)"),
        ("cadw/dataset/index.py", "v0.9.18 (AUDIT §1.6)"),
        ("cadw/patch_utils.py", "v0.9.15 (T2.6)"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
        ("cadw/__init__.py", '__version__ = "0.9.14"'),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
    ])
    print("  all anchors present")


# ============ dogfood (happy path only, §13.4/228) ============

def dogfood():
    print("\n=== dogfood: v0.9.18-fix3 (happy path only, §13.4/228) ===")

    def _verify_all():
        # Happy path only — edge cases in pytest.
        import pyarrow as pa
        from cadw.dataset.index import MaterializedIndex

        # Numeric: numpy fast path (null-free)
        idx_num = MaterializedIndex(pa.array([1, 2, 3, 4, 5]))
        assert idx_num.is_monotonic_increasing()
        assert not idx_num.is_monotonic_decreasing()
        assert idx_num.get_positions_all(3) == [2]
        assert not idx_num.has_duplicates()

        # String: Python fallback path (type-strict)
        idx_str = MaterializedIndex(pa.array(["a", "b", "c"]))
        assert idx_str.is_monotonic_increasing()
        assert idx_str.get_positions_all("b") == [1]
        # Type-strict: int label against string index → no match
        assert idx_str.get_positions_all(1) == []
        # Non-scalar (tuple): no match on single index
        assert idx_str.get_positions_all(("a", 1)) == []

        print("  numeric monotonic/positions/dup OK")
        print("  string monotonic/positions OK")
        print("  type-strict + non-scalar label OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_loc.py", "tests/test_loc_multi.py",
         "tests/test_filter_compat.py", "tests/test_cadw_index.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_hash_v081.py", "-q"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.18-fix3 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    whitelist_string_hash()
    add_numpy_import()
    fix_get_positions_all_safety()
    fix_has_duplicates_null_semantics()
    fix_iter_null_aware()
    add_regression_tests()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.18-fix3: complete v0.9.18 (3 latent bugs in "
        "v0.9.18 numpy/dict-encode paths). v0.9.18-fix2 green in "
        "hash tests but pytest fast revealed 14 failures. (a) "
        "test_no_unguarded_high_risk_arrow_calls: dictionary_encode "
        "in _string_hash_column without preceding check (§3.11 AST "
        "guard). Root: v0.9.18 §1.3 added dict_encode without "
        "whitelisting; caller chain hash_shuffle._process → "
        "check_batch(extra_factor=3.0) → compute_hashes → "
        "_hash_column → _string_hash_column → dict_encode. AST "
        "guard sees only local scope; cross-function chain "
        "invisible. Fix: whitelist cadw/ops/shuffle.py:"
        "_string_hash_column (dict_encode O(K + batch) ≤ "
        "3×batch.nbytes covered by caller check_batch). (b) 13 "
        "failures: NameError: name 'np' is not defined in "
        "MaterializedIndex.is_monotonic_increasing/decreasing. "
        "Root: v0.9.18 §1.6 used np.all/np.flatnonzero but "
        "index.py had no `import numpy as np`. Fix: add it. (c) "
        "test_loc_tuple_on_single_index_raises_keyerror: expected "
        "KeyError, got no raise. Root: get_positions_all numpy "
        "fast path did `np_arr == (\"a\", 1)` → numpy coerced "
        "tuple to string array, matched first element → false "
        "positive position [0], no KeyError. §13.4/229 (new): "
        "numpy element-wise `arr == label` coerces label to array "
        "dtype. For string index + int label, `np.array([\"1\"]) "
        "== 1` → [True] (str coercion), prior to_pylist + "
        "`\"1\" == 1` → False (type strict). Numpy fast path only "
        "safe for same-type scalar comparisons; for string/object "
        "use Python loop. Same class as §13.4/223 (bool→int) + "
        "§13.4/226 (nullable int → float64): numpy implicit dtype "
        "coercion loses Arrow type strictness. Fix (c): "
        "get_positions_all numpy fast path restricted to numeric + "
        "scalar; Python-loop fallback for strings + non-scalar "
        "labels (pre-v0.9.18 type-strict semantics). Bonus latent "
        "bug: has_duplicates v0.9.18 used count_distinct default "
        "mode='only_valid' — nulls counted separately from values "
        "→ [1, None, 2] gave True incorrectly. Fix: mode='all' "
        "(counts null as value, matches pandas Index.has_duplicates). "
        "Also: __iter__ null-containing falls back to to_pylist "
        "(numpy gives NaN not None for nulls). Plus 2 new regression "
        "tests: string index + int label (type strict), tuple label "
        "on single index. v0.9.18 COMPLETE: fix failing slow test + "
        "§1.1 shift tail, §1.2 rank numpy, §1.3 string hash "
        "dict-encode, §1.5 to_list guard, §1.6 index numpy; fix1 "
        "(null indices float64), fix2 (all-null empty dict), fix3 "
        "(np import + str type-strict + has_dup mode='all' + iter "
        "null-aware + AST whitelist).",
        self_path="scripts/patch_v0918_fix3.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.18-fix3 applied. v0.9.18 COMPLETE.")


if __name__ == "__main__":
    main()
