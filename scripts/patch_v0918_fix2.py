#!/usr/bin/env python3
# v0.9.18-fix2 — _string_hash_column: all-null edge (empty dict).
# v0.9.18 COMPLETE.
#
# v0.9.18-fix1 applied cleanly (log OK ×2), verify_state green, but
# dogfood failed on all-null edge:
#   arr_null = pa.array([None, None, None], type=pa.string())
#   h_null = _string_hash_column(arr_null)
#   IndexError: index 0 is out of bounds for axis 0 with size 0
#
# Root: pyarrow dictionary_encode() on all-null array → empty
# dictionary (K=0), indices all-null. fill_null(0) → [0,0,0], но
# uniq_hashes = np.empty(0) → safe_indices[0] = 0 → IndexError.
#
# §13.4/227 (new): pyarrow dictionary_encode() on all-null array
# → empty dictionary (K=0). fill_null(0) при K=0 → out-of-bounds.
# Special-case K == 0 → np.full(n, _NULL_SENTINEL).
#
# Meta-lesson §13.4/228 (new): dogfood должен покрывать те же edge
# cases, что pytest. При fix1 dogfood упал на no-null edge до того,
# как pytest успел запуститься; тот же паттерн повторился на all-null
# edge в fix2-итерации. Правило: dogfood = быстрый smoke (happy
# path) + только инварианты, требующие subprocess (fresh imports).
# Edge cases — pytest. Иначе dogfood-to-pytest pipeline даёт
# multiple red herrings.
#
# Fix2:
#   (a) _string_hash_column: K==0 → early return all-sentinel.
#   (b) verify + dogfood (только happy path!) + pytest + commit.
#
# Run: uv run python scripts/patch_v0918_fix2.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) fix K==0 edge ============

def fix_all_null_dict():
    print("\n=== (a) _string_hash_column: K==0 (all-null) early return ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()
    if "§13.4/227" in s and "if K == 0:" in s:
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

    new_block = '''    K = len(uniques)
    # v0.9.18-fix2 (§13.4/227): all-null array → empty dictionary (K=0).
    # fill_null(0) then indexes empty uniq_hashes → IndexError. Early
    # return all-sentinel — every row is null by construction.
    if K == 0:
        return np.full(n, _NULL_SENTINEL, dtype=np.uint64)

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
        raise SystemExit("_string_hash_column K==0 anchor not found")
    s = s.replace(old_block, new_block, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.18-fix2 + regression ===")
    pu.verify_anchors([
        # fix2
        ("cadw/ops/shuffle.py", "§13.4/227"),
        ("cadw/ops/shuffle.py", "if K == 0:"),
        # fix1
        ("cadw/ops/shuffle.py", "§13.4/226"),
        ("cadw/ops/shuffle.py", "safe_indices"),
        ("tests/test_hash_v081.py",
         "test_string_hash_dict_encode_path_with_nulls"),
        ("tests/test_hash_v081.py", "test_string_hash_all_nulls"),
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
        # v0.9.17/16/15/14
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный"),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        ("cadw/patch_utils.py", "v0.9.15 (T2.6)"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
        ("cadw/__init__.py", '__version__ = "0.9.14"'),
        ("cadw/dataframe.py", "v0.9.14-fix3 (§13.4/223)"),
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
    ])
    print("  all anchors present")


# ============ dogfood: HAPPY PATH ONLY (see §13.4/228) ============

def dogfood():
    print("\n=== dogfood: v0.9.18-fix2 (happy path only, §13.4/228) ===")

    def _verify_all():
        # §13.4/228: dogfood = happy-path smoke only. Edge cases
        # (all-null, empty, null-containing) live in pytest. Prior
        # v0.9.18 / fix1 chains both failed at dogfood before pytest
        # could run — same class of red herring. Fix2: minimal.
        import pyarrow as pa
        from cadw.ops.shuffle import _string_hash_column

        # Happy path: no nulls, 3 distinct values → 3 distinct hashes
        arr = pa.array(["a", "b", "c"], type=pa.string())
        h = _string_hash_column(arr)
        assert len(h) == 3
        assert len(set(h.tolist())) == 3
        print("  no-null path: 3 distinct values → 3 distinct hashes OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: hash test (full edge coverage) ===")
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
    print("=== v0.9.18-fix2 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_all_null_dict()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.18-fix2: complete v0.9.18 (all-null in "
        "_string_hash_column). v0.9.18-fix1 applied cleanly (log "
        "OK ×2), verify_state green, but dogfood failed on all-null "
        "edge: pa.array([None]*3).dictionary_encode() → empty "
        "dictionary (K=0), fill_null(0) → [0,0,0], uniq_hashes "
        "size 0 → IndexError: index 0 out of bounds. §13.4/227 "
        "(new): pyarrow dictionary_encode() on all-null array → "
        "empty dictionary; fill_null(0) при K=0 → out-of-bounds; "
        "special-case K==0 → np.full(n, _NULL_SENTINEL). "
        "§13.4/228 (new): dogfood должен покрывать те же edge "
        "cases, что pytest — при fix1 dogfood упал на null-containing "
        "edge до того как pytest запустился; тот же паттерн "
        "повторился на all-null edge в fix2. Правило: dogfood = "
        "быстрый happy-path smoke (subprocess для fresh imports); "
        "edge cases — pytest. Fix2: (a) _string_hash_column: K==0 "
        "→ early return all-sentinel; (b) dogfood reduced к "
        "no-null happy path only (§13.4/228); (c) verify (20 "
        "anchors) + dogfood + pytest + commit. v0.9.18 COMPLETE: "
        "fix failing slow test + §1.1 shift/_shift_on Arrow-native "
        "tail, §1.2 rank numpy, §1.3 string hash dict-encode, "
        "§1.4 to_list guard sample-based, §1.6 index numpy; "
        "fix1 (null indices float64), fix2 (all-null empty dict).",
        self_path="scripts/patch_v0918_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.18-fix2 applied. v0.9.18 COMPLETE.")


if __name__ == "__main__":
    main()
