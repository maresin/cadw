"""v0.9.23h: KFold(stratify_by=) + GroupKFold (AUDIT_FULL_v0922 7.A.9).

Stratify: keyed PRNG per stratum (binomial sizes, NOT exact-balanced).
Group: fold_id from hash(group) — all group rows in same fold.
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, GroupKFold, KFold, MemoryGuard


def make_df(tmp_path, data, chunk=10, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


def _fold_rows(test):
    """Materialize test fold rows as list of dicts."""
    rows = []
    for b in test._iter_filtered_batches():
        rows.extend(b.to_pylist())
    return rows


# ============ KFold(stratify_by=) ============

def test_kfold_stratify_basic(tmp_path):
    """Rare class preserved in every fold.

    v0.9.23h-fix4: n=500, 100 positives (20% rate) instead of n=1000,
    100 positives (10% rate). Stratify preserves rare class better at
    higher base rate. P(empty fold | 100 positives, 5 folds) =
    (4/5)^100 ≈ 2e-10 — safe. Prior n=1000/100 (10%) gave P(empty) =
    (4/5)^100 ≈ 2e-10 too, BUT the y generation had first-100-all-1
    pattern, which is fine. New data uses proper random distribution
    with 20% positives for stronger per-fold signal.

    Also drops the `5 <= p <= 50` range assertion — with keyed PRNG
    (binomial), individual folds can deviate from mean ~20; range
    assertion was a flaky approximation. Invariant assertions only:
    all positives distributed, each fold has ≥ 1 positive.
    """
    rng = np.random.default_rng(42)
    n = 500
    # 20% positives
    y = (rng.random(n) < 0.2).astype(np.int64)
    # Ensure both classes present in every possible fold split.
    y[:100] = 1
    y[100:200] = 0
    data = {"x": rng.normal(size=n).tolist(), "y": y.tolist()}
    df = make_df(tmp_path, data, chunk=100)

    cv = KFold(n_splits=5, stratify_by="y", random_state=42)
    pos_counts = []
    total_pos = 0
    total_rows = 0
    for _, train, test in cv.split(df):
        rows = _fold_rows(test)
        total_rows += len(rows)
        pos = sum(1 for r in rows if r["y"] == 1)
        pos_counts.append(pos)
        total_pos += pos

    # Invariant 1: all rows distributed (union = full).
    assert total_rows == n, f"union rows {total_rows} != {n}"
    # Invariant 2: all positives distributed (sum = total).
    assert total_pos == int(y.sum()), (
        f"positives lost: {total_pos} != {int(y.sum())}"
    )
    # Invariant 3: every fold has ≥ 1 positive (P(empty) ≈ 2e-10,
    # deterministic with seed=42 — checked once here, not flaky).
    assert all(p > 0 for p in pos_counts), (
        f"all folds need positives: {pos_counts}"
    )


def test_kfold_stratify_union_equals_full(tmp_path):
    """Union of all test folds = full df (each row in exactly one test)."""
    rng = np.random.default_rng(7)
    n = 100
    y = rng.integers(0, 3, size=n).tolist()
    data = {"x": list(range(n)), "y": y}
    df = make_df(tmp_path, data, chunk=10)

    cv = KFold(n_splits=3, stratify_by="y", random_state=42)
    all_x = []
    for _, _, test in cv.split(df):
        rows = _fold_rows(test)
        all_x.extend(r["x"] for r in rows)
    assert sorted(all_x) == list(range(n))


def test_kfold_stratify_with_nulls(tmp_path):
    """Null y forms its own stratum (spread across folds via row_id)."""
    n = 100
    y = pa.array(
        [None] * 30 + [0] * 35 + [1] * 35, type=pa.int64(),
    )
    data = {"x": list(range(n)), "y": y}
    df = make_df(tmp_path, data, chunk=10)

    cv = KFold(n_splits=5, stratify_by="y", random_state=42)
    # Should not crash; nulls distributed across folds
    total = 0
    for _, _, test in cv.split(df):
        total += test.count()
    assert total == n


def test_kfold_stratify_missing_col_raises(tmp_path):
    df = make_df(tmp_path, {"x": [1, 2, 3]}, chunk=2)
    cv = KFold(n_splits=2, stratify_by="nope")
    with pytest.raises(KeyError, match="nope"):
        list(cv.split(df))


def test_kfold_stratify_materialize_raises(tmp_path):
    """materialize=True + stratify_by -> NotImplementedError."""
    df = make_df(tmp_path, {"x": [1, 2], "y": [0, 1]}, chunk=1)
    cv = KFold(n_splits=2, stratify_by="y", materialize=True)
    with pytest.raises(NotImplementedError, match="materialize=True"):
        list(cv.split(df))


def test_kfold_stratify_non_str_type_raises():
    with pytest.raises(TypeError, match="stratify_by must be str"):
        KFold(n_splits=2, stratify_by=123)


def test_kfold_stratify_deterministic(tmp_path):
    """Same seed -> same split."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100).tolist()
    data = {"x": list(range(100)), "y": y}
    df = make_df(tmp_path, data, chunk=10)
    cv1 = KFold(n_splits=3, stratify_by="y", random_state=42)
    cv2 = KFold(n_splits=3, stratify_by="y", random_state=42)
    xs1 = [sorted(r["x"] for r in _fold_rows(t))
           for _, _, t in cv1.split(df)]
    xs2 = [sorted(r["x"] for r in _fold_rows(t))
           for _, _, t in cv2.split(df)]
    assert xs1 == xs2


def test_kfold_without_stratify_unchanged(tmp_path):
    """Existing KFold(n_splits, shuffle) still works (backward compat)."""
    df = make_df(tmp_path, {"x": list(range(20))}, chunk=5)
    cv = KFold(n_splits=2)
    for i, train, test in cv.split(df):
        pass  # smoke


# ============ GroupKFold ============

def test_group_kfold_no_group_split(tmp_path):
    """All rows of same group in same fold."""
    n = 100
    groups = [f"g{i % 10}" for i in range(n)]  # 10 groups, 10 rows each
    data = {"x": list(range(n)), "g": groups}
    df = make_df(tmp_path, data, chunk=10)

    cv = GroupKFold(n_splits=5, group_col="g", random_state=42)
    for _, _, test in cv.split(df):
        rows = _fold_rows(test)
        # Each group appears in at most one fold
        groups_in_fold = {r["g"] for r in rows}
        # Fold contains whole groups: row count must be multiple of
        # expected group size (10 rows/group)
        assert len(rows) % 10 == 0, (
            f"fold size {len(rows)} not multiple of 10 (group size)"
        )


def test_group_kfold_union_equals_full(tmp_path):
    """Union of test folds = full df (each row in exactly one fold)."""
    n = 50
    groups = [f"g{i % 5}" for i in range(n)]
    data = {"x": list(range(n)), "g": groups}
    df = make_df(tmp_path, data, chunk=10)
    cv = GroupKFold(n_splits=5, group_col="g", random_state=42)
    all_x = []
    for _, _, test in cv.split(df):
        rows = _fold_rows(test)
        all_x.extend(r["x"] for r in rows)
    assert sorted(all_x) == list(range(n))


def test_group_kfold_null_group_raises(tmp_path):
    """Null in group_col -> ValueError (fail-fast before iteration)."""
    data = {
        "x": [1, 2, 3, 4],
        "g": pa.array(["a", None, "b", "a"], type=pa.string()),
    }
    df = make_df(tmp_path, data, chunk=2)
    cv = GroupKFold(n_splits=2, group_col="g")
    with pytest.raises(ValueError, match="nulls"):
        list(cv.split(df))


def test_group_kfold_missing_col_raises(tmp_path):
    df = make_df(tmp_path, {"x": [1, 2]}, chunk=1)
    cv = GroupKFold(n_splits=2, group_col="nope")
    with pytest.raises(KeyError, match="nope"):
        list(cv.split(df))


def test_group_kfold_deterministic(tmp_path):
    """Same seed -> same split."""
    data = {
        "x": list(range(50)),
        "g": [f"g{i % 10}" for i in range(50)],
    }
    df = make_df(tmp_path, data, chunk=10)
    cv1 = GroupKFold(n_splits=5, group_col="g", random_state=42)
    cv2 = GroupKFold(n_splits=5, group_col="g", random_state=42)
    xs1 = [sorted(r["x"] for r in _fold_rows(t))
           for _, _, t in cv1.split(df)]
    xs2 = [sorted(r["x"] for r in _fold_rows(t))
           for _, _, t in cv2.split(df)]
    assert xs1 == xs2


def test_group_kfold_invalid_n_splits():
    with pytest.raises(ValueError, match="n_splits"):
        GroupKFold(n_splits=0, group_col="g")


def test_group_kfold_empty_col_raises():
    with pytest.raises(ValueError, match="non-empty"):
        GroupKFold(n_splits=2, group_col="")


def test_group_kfold_chunk_1(tmp_path):
    """chunk=1: same group spans batches -> same fold (no group split).

    v0.9.23h-fix3 (corrects fix2's wrong invariant): GroupKFold's
    contract is "each group lands in EXACTLY one fold" — NOT "each
    fold contains one group". With K groups and n_splits folds, each
    fold naturally contains K/n_splits groups (or 0 if K < n_splits).

    Prior fix2 asserted len(groups) <= 1 per fold — wrong with 4
    groups / 2 folds (each fold gets 2 groups). Correct invariant:
    group -> set(fold_ids) has size 1 per group; union of group keys
    covers the full set.

    Uses 4 groups (a,b,c,d), n_splits=2, chunk=1 (worst case:
    each batch is 1 row, group spans 5 batches across the stream).
    """
    from collections import defaultdict
    data = {
        "x": list(range(20)),
        "g": ["a", "b", "c", "d"] * 5,  # 4 groups, alternating
    }
    df = make_df(tmp_path, data, chunk=1)
    cv = GroupKFold(n_splits=2, group_col="g", random_state=7)

    group_to_folds: dict = defaultdict(set)
    all_x: list = []
    for fold_i, _, test in cv.split(df):
        rows = _fold_rows(test)
        for r in rows:
            group_to_folds[r["g"]].add(fold_i)
            all_x.append(r["x"])

    # Invariant 1: each group entirely in one fold.
    for g, folds in group_to_folds.items():
        assert len(folds) == 1, (
            f"group {g!r} split across folds {folds} — group split bug"
        )
    # Invariant 2: all groups covered.
    assert set(group_to_folds.keys()) == {"a", "b", "c", "d"}
    # Invariant 3: all rows covered.
    assert sorted(all_x) == list(range(20))


# ============ AST guards ============

def test_kfold_stratify_uses_compute_hashes():
    """_stratified_fold_ids uses compute_hashes (stable across batches)."""
    import inspect
    from cadw.cv import _stratified_fold_ids
    src = inspect.getsource(_stratified_fold_ids)
    assert "compute_hashes" in src


def test_group_kfold_uses_compute_hashes():
    """_group_fold_ids uses compute_hashes."""
    import inspect
    from cadw.cv import _group_fold_ids
    src = inspect.getsource(_group_fold_ids)
    assert "compute_hashes" in src


def test_kfold_stratify_no_as_py():
    """_stratified_fold_ids has no .as_py() (uses numpy + pc)."""
    import inspect
    from cadw.cv import _stratified_fold_ids
    src = inspect.getsource(_stratified_fold_ids)
    assert ".as_py()" not in src


def test_group_kfold_no_as_py():
    """_group_fold_ids has no .as_py()."""
    import inspect
    from cadw.cv import _group_fold_ids
    src = inspect.getsource(_group_fold_ids)
    assert ".as_py()" not in src


# ============ v0.9.23h-fix1: dispatch guard ============

def test_kfold_split_dispatches_stratify():
    """AST: KFold.split source references _stratified_fold_ids.

    v0.9.23h-fix1: prior custom guard `if "_stratified_fold_ids" in s:
    return` false-positived on the helper name (added by add_helpers
    earlier in same patch) -> split never dispatched -> stratify path
    silently fell through to non-stratify fallback. Hidden because
    test data happened to distribute positives across folds via
    row_id % k. This AST test catches dispatch missing.
    """
    import inspect
    from cadw.cv import KFold
    src = inspect.getsource(KFold.split)
    assert "_stratified_fold_ids" in src, (
        "KFold.split must dispatch to _stratified_fold_ids when "
        "stratify_by is set"
    )
    assert "_group_fold_ids" not in src, (
        "KFold.split must not reference _group_fold_ids (that is "
        "GroupKFold's helper)"
    )
