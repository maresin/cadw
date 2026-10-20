#!/usr/bin/env python3
"""v0.9.2e prototype: KFold + TimeSeriesSplit fold IDs.

WHY PROTOTYPE (§13.4/123): fold assignment — algorithmic. Verify:
  1. No overlap between test folds (each row exactly once across tests).
  2. Fold sizes correct (KFold: floor/ceil; TSS: increasing train).
  3. TSS: no future leakage (max(train_pos) < min(test_pos) per split).
  4. Edge cases (k=0, k>n, n<k+1) raise.

Run: uv run python scripts/proto_cv.py
Expected: PASS 8/8.
"""
from __future__ import annotations

import numpy as np


def kfold_ids(n: int, k: int, *, shuffle: bool = False, seed: int = 42):
    """Return (n,) int array of fold IDs in [0, k).

    shuffle=False: fold_id[i] = i % k  (deterministic, O(1) per row).
    shuffle=True: O(N) permutation materialized once.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if not shuffle:
        return np.arange(n) % k
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ids = np.empty(n, dtype=np.int64)
    ids[perm] = np.arange(n) % k
    return ids


def tss_splits(n: int, k: int):
    """Standard TimeSeriesSplit: split i test = [fs*(i+1), fs*(i+2)),
    train = [0, fs*(i+1)). Last split's test gets the remainder.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if n < k + 1:
        raise ValueError(f"n={n} < k+1={k+1}")
    fold_size = n // (k + 1)
    splits = []
    for i in range(k):
        train_end = fold_size * (i + 1)
        test_end = fold_size * (i + 2) if i < k - 1 else n
        splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return splits


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  {status} {name}" + (f"  {detail}" if detail else ""))
    return cond


def main():
    print("=== proto CV: KFold + TimeSeriesSplit ===\n")
    all_pass = True

    # 1. KFold 100/5
    ids = kfold_ids(100, 5)
    sizes = [int((ids == f).sum()) for f in range(5)]
    all_pass &= check("1. KFold 100/5 sizes",
                      sizes == [20, 20, 20, 20, 20], f"sizes={sizes}")

    # 2. KFold 103/5 (uneven)
    ids = kfold_ids(103, 5)
    sizes = [int((ids == f).sum()) for f in range(5)]
    all_pass &= check("2. KFold 103/5 uneven",
                      sizes == [21, 21, 21, 20, 20], f"sizes={sizes}")

    # 3. KFold overlap (union of test-fold positions == all)
    ids = kfold_ids(100, 5)
    union = np.sort(np.concatenate([np.where(ids == f)[0] for f in range(5)]))
    all_pass &= check("3. KFold no overlap",
                      np.array_equal(union, np.arange(100)), f"n={len(union)}")

    # 4. TSS test sizes
    splits = tss_splits(100, 5)
    test_sizes = [len(test) for _, test in splits]
    all_pass &= check("4. TSS test sizes",
                      test_sizes == [16, 16, 16, 16, 20], f"sizes={test_sizes}")

    # 5. TSS no future leakage
    leak_free = True
    for i, (train, test) in enumerate(splits):
        if len(train) and len(test) and max(train) >= min(test):
            leak_free = False
            print(f"    LEAK split {i}: train_max={max(train)} >= test_min={min(test)}")
    all_pass &= check("5. TSS no future leakage", leak_free)

    # 6. TSS train grows monotonically
    train_sizes = [len(train) for train, _ in splits]
    grows = all(train_sizes[i] < train_sizes[i+1] for i in range(len(train_sizes)-1))
    all_pass &= check("6. TSS train grows",
                      grows, f"sizes={train_sizes}")

    # 7. KFold shuffle: same sizes, different assignment
    ids_n = kfold_ids(100, 5, shuffle=False)
    ids_s = kfold_ids(100, 5, shuffle=True, seed=42)
    sizes_s = [int((ids_s == f).sum()) for f in range(5)]
    different = not np.array_equal(ids_n, ids_s)
    all_pass &= check("7. KFold shuffle (sizes + different)",
                      sizes_s == [20, 20, 20, 20, 20] and different,
                      f"sizes={sizes_s} diff={different}")

    # 8. Edge cases raise
    edge_ok = True
    for bad in [lambda: kfold_ids(10, 0),
                lambda: kfold_ids(10, 11),
                lambda: tss_splits(3, 5)]:
        try:
            bad()
            edge_ok = False
        except ValueError:
            pass
    all_pass &= check("8. Edge cases raise", edge_ok)

    print()
    if all_pass:
        print("ALL PASS — CV fold assignment validated.")
        return 0
    print("SOME FAILED — do NOT implement patch.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
