#!/usr/bin/env python3
# v0.9.18-fix1 — _string_hash_column: null indices float64 → int64.
# v0.9.18 COMPLETE.
#
# v0.9.18 applied all 8 steps (log OK ×8), verify_state green, but
# dogfood failed:
#   h = _string_hash_column(arr)
#   IndexError: arrays used as indices must be of integer type
#
# Root: pyarrow DictionaryArray.indices.to_numpy(zero_copy_only=False)
# on null-containing array returns **float64** (NaN for nulls), not
# int64. Prior code did `uniq_hashes[indices[non_null]]` — float64
# array indexing → IndexError.
#
# §13.4/226 (new): pyarrow nullable integer Array.to_numpy(
# zero_copy_only=False) → float64 (NaN for nulls), not int64.
# For integer indexing: `arr.fill_null(0).to_numpy(...).astype(int64)`
# + separate null_mask via `arr.is_null().to_numpy()`.
#
# Also caught: test_hash_v081.py::
# test_own_hash_cross_batch_string_with_and_without_nulls would have
# failed too (b2 = ["a", None] hits same path). Patch failed at
# dogfood before pytest ran, so tests never saw it.
#
# Fix1:
#   (a) _string_hash_column: null_mask via is_null(), safe_indices
#       via fill_null(0).astype(int64).
#   (b) Add regression test for dict-encode path with nulls.
#   (c) verify + dogfood + pytest + commit.
#
# Run: uv run python scripts/patch_v0918_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) fix _string_hash_column null handling ============

def fix_string_hash_nulls():
    print("\n=== (a) _string_hash_column: null indices float64 → int64 ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()
    if "§13.4/226" in s and "safe_indices" in s:
        print("  already fixed — skip")
        return

    old_block = '''    K = len(uniques)
    uniq_hashes = np.empty(K, dtype=np.uint64)
    for i, v in enumerate(uniques.to_pylist()):
        if v is None:
            uniq_hashes[i] = _NULL_SENTINEL
        else:
            data = v.encode("utf-8") if isinstance(v, str) else v
            uniq_hashes[i] = xxhash.xxh64(data, seed=0).intdigest()

    indices = enc.indices.to_numpy(zero_copy_only=False)
    out = np.full(n, _NULL_SENTINEL, dtype=np.uint64)
    non_null = indices >= 0
    out[non_null] = uniq_hashes[indices[non_null]]
    return out'''

    new_block = '''    K = len(uniques)
    uniq_hashes = np.empty(K, dtype=np.uint64)
    for i, v in enumerate(uniques.to_pylist()):
        if v is None:
            uniq_hashes[i] = _NULL_SENTINEL
        else:
            data = v.encode("utf-8") if isinstance(v, str) else v
            uniq_hashes[i] = xxhash.xxh64(data, seed=0).intdigest()

    # v0.9.18-fix1 (§13.4/226): null-containing DictionaryArray.indices
    # to_numpy(zero_copy_only=False) → float64 (NaN), not int64. Prior
    # `uniq_hashes[indices[non_null]]` → IndexError. Fix: null_mask via
    # is_null(), safe_indices via fill_null(0).astype(int64).
    idx_arr = enc.indices
    null_mask = idx_arr.is_null().to_numpy(zero_copy_only=False)
    safe_indices = (
        idx_arr.fill_null(0)
        .to_numpy(zero_copy_only=False)
        .astype(np.int64, copy=False)
    )
    out = uniq_hashes[safe_indices]
    out[null_mask] = _NULL_SENTINEL
    return out'''

    if old_block not in s:
        raise SystemExit("_string_hash_column null block anchor not found")
    s = s.replace(old_block, new_block, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (b) regression test ============

def add_regression_test():
    print("\n=== (b) test_hash_v081.py: dict-encode path with nulls ===")
    p = Path("tests/test_hash_v081.py")
    s = p.read_text()
    if "test_string_hash_dict_encode_path_with_nulls" in s:
        print("  already present — skip")
        return
    marker = "\n\n# ============ v0.9.6a: own XXH64 hash, no pandas ============"
    if marker not in s:
        raise SystemExit("test file marker not found")
    new_test = '''

# ============ v0.9.18-fix1 (§13.4/226): dict-encode null path ============


def test_string_hash_dict_encode_path_with_nulls():
    """Dictionary-encode fast path with nulls must not IndexError.

    §13.4/226: DictionaryArray.indices.to_numpy(zero_copy_only=False)
    returns float64 (NaN for nulls), not int64. Prior
    `uniq_hashes[indices[non_null]]` → IndexError: arrays used as
    indices must be integer type. Fix: fill_null(0).astype(int64) +
    separate is_null mask.
    """
    from cadw.ops.shuffle import _string_hash_column, _NULL_SENTINEL
    arr = pa.array(["a", "b", "a", None, "b", None], type=pa.string())
    h = _string_hash_column(arr)
    assert h[0] == h[2]          # "a" == "a"
    assert h[1] == h[4]          # "b" == "b"
    assert h[0] != h[1]          # "a" != "b"
    assert h[3] == h[5] == _NULL_SENTINEL  # nulls coalesce
    assert _NULL_SENTINEL not in h[[0, 1, 2, 4]]


def test_string_hash_all_nulls():
    """All-null array → all _NULL_SENTINEL, no crash."""
    from cadw.ops.shuffle import _string_hash_column, _NULL_SENTINEL
    arr = pa.array([None, None, None], type=pa.string())
    h = _string_hash_column(arr)
    assert all(int(v) == int(_NULL_SENTINEL) for v in h)


def test_string_hash_empty_array():
    """Empty array → empty result."""
    from cadw.ops.shuffle import _string_hash_column
    arr = pa.array([], type=pa.string())
    h = _string_hash_column(arr)
    assert len(h) == 0


def test_string_hash_no_nulls_no_regression():
    """No-null path unchanged (all unique values)."""
    from cadw.ops.shuffle import _string_hash_column
    arr = pa.array(["a", "b", "c"], type=pa.string())
    h = _string_hash_column(arr)
    assert len(set(h.tolist())) == 3

'''
    s = s.replace(marker, new_test + marker, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.18-fix1 + regression ===")
    pu.verify_anchors([
        # fix1
        ("cadw/ops/shuffle.py", "§13.4/226"),
        ("cadw/ops/shuffle.py", "safe_indices"),
        ("tests/test_hash_v081.py",
         "test_string_hash_dict_encode_path_with_nulls"),
        # v0.9.18
        ("tests/test_category_beyond_memory.py",
         "def test_groupby_1m_unique_1mb_succeeds_or_raises"),
        ("tests/test_category_beyond_memory.py",
         "def test_groupby_explicit_huge_p_writer_check"),
        ("cadw/dataframe.py", "class _ArrowTail:"),
        ("cadw/ops/rank.py", "v0.9.18 (AUDIT §1.2)"),
        ("cadw/ops/shuffle.py", "v0.9.18 (AUDIT §1.3)"),
        ("cadw/dataframe.py", "_estimate_materialize_bytes"),
        ("cadw/dataset/index.py", "v0.9.18 (AUDIT §1.6)"),
        # v0.9.17/16/15/14 regression
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный"),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        ("cadw/patch_utils.py", "v0.9.15 (T2.6)"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
        ("cadw/__init__.py", '__version__ = "0.9.14"'),
        ("cadw/dataframe.py", "v0.9.14-fix3 (§13.4/223)"),
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.18-fix1 ===")

    def _verify_all():
        import pyarrow as pa
        import numpy as np
        from cadw.ops.shuffle import _string_hash_column, _NULL_SENTINEL

        # Regression: null-containing dict-encode path
        arr = pa.array(["a", "b", "a", None, "b", None], type=pa.string())
        h = _string_hash_column(arr)
        assert h[0] == h[2], f"hash(a) mismatch: {h}"
        assert h[1] == h[4], f"hash(b) mismatch: {h}"
        assert h[0] != h[1], f"hash(a) == hash(b): {h}"
        assert h[3] == h[5], f"null hash mismatch: {h}"
        assert int(h[3]) == int(_NULL_SENTINEL)

        # All-null
        arr_null = pa.array([None, None, None], type=pa.string())
        h_null = _string_hash_column(arr_null)
        assert all(int(v) == int(_NULL_SENTINEL) for v in h_null)

        # No-null path
        arr_clean = pa.array(["a", "b", "c"], type=pa.string())
        h_clean = _string_hash_column(arr_clean)
        assert len(set(h_clean.tolist())) == 3

        # Empty
        arr_empty = pa.array([], type=pa.string())
        assert len(_string_hash_column(arr_empty)) == 0

        print("  dict-encode with nulls: no IndexError OK")
        print("  all-null, empty, no-null paths OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: hash test ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_hash_v081.py", "-q"],
        timeout=600,
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
    print("=== v0.9.18-fix1 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_string_hash_nulls()
    add_regression_test()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.18-fix1: complete v0.9.18 (null indices in "
        "_string_hash_column). v0.9.18 applied all 8 steps (log "
        "OK ×8), verify_state green, but dogfood failed: "
        "_string_hash_column(arr) with nulls → IndexError: arrays "
        "used as indices must be integer type. Root: pyarrow "
        "DictionaryArray.indices.to_numpy(zero_copy_only=False) on "
        "null-containing array returns float64 (NaN for nulls), "
        "not int64. Prior `uniq_hashes[indices[non_null]]` → "
        "IndexError. §13.4/226 (new): pyarrow nullable integer "
        "Array.to_numpy(zero_copy_only=False) → float64 (NaN for "
        "nulls), not int64. For integer indexing: "
        "arr.fill_null(0).to_numpy(...).astype(int64) + separate "
        "null_mask via arr.is_null().to_numpy(). Also caught: "
        "test_own_hash_cross_batch_string_with_and_without_nulls "
        "would have failed too (b2=[a, None] hits same path); patch "
        "failed at dogfood before pytest ran, so tests never saw "
        "it. Fix1: (a) _string_hash_column: null_mask via is_null(), "
        "safe_indices via fill_null(0).astype(int64); (b) 4 new "
        "regression tests (dict-encode path with nulls, all-null, "
        "empty, no-null-no-regression); (c) verify (17 anchors) + "
        "dogfood + pytest + commit. v0.9.18 COMPLETE: fix failing "
        "slow test + §1.1 shift/_shift_on Arrow-native tail, §1.2 "
        "rank numpy, §1.3 string hash dict-encode, §1.4 to_list "
        "guard sample-based, §1.6 index numpy.",
        self_path="scripts/patch_v0918_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.18-fix1 applied. v0.9.18 COMPLETE.")


if __name__ == "__main__":
    main()
