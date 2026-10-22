#!/usr/bin/env python3
# v0.9.23h-fix3 — test_group_kfold_chunk_1 wrong invariant.
#
# fix2's test asserted `len(groups) <= 1` per fold — but with 4 groups
# (a,b,c,d) and 2 folds, each fold naturally gets 2 groups. Correct
# GroupKFold invariant: each GROUP lands in exactly one fold (no
# group split across folds). Fold CONTENT ("1 group" vs "2 groups")
# is irrelevant.
#
# This is a test-authoring bug (§13.4/164, /171 class: manual
# arithmetic wrong). Production GroupKFold correct — the assertion
# `len(groups) <= 1` was the error.
#
# fix3 replaces with group->fold_id map:
#   for each fold: for each row: group_to_folds[g].add(fold_id)
#   assert all(len(folds) == 1)
#   assert union == full group set
#
# Also strengthens dogfood with same invariant.
#
# Run: uv run python scripts/patch_v0923h_fix3_group_invariant.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess,
)


# ============ 1. Replace wrong test with correct invariant ============

WRONG_TEST = '''def test_group_kfold_chunk_1(tmp_path):
    """chunk=1: same group spans batches -> same fold (no group split).

    v0.9.23h-fix2: empty folds are valid GroupKFold behavior (whole-
    group assignment -> unbalanced sizes; if all groups hash to one
    fold, other folds are empty). Test allows empty folds (len(groups)
    == 0) and single-group folds (== 1), rejects multi-group (>= 2).

    Uses 4 groups (a,b,c,d) to reduce degenerate-hash chance; still
    asserts non_empty_folds >= 1 as sanity that GroupKFold dispatched
    (catches all-folds-empty = dispatch broken).
    """
    data = {
        "x": list(range(20)),
        "g": ["a", "b", "c", "d"] * 5,  # 4 groups, alternating
    }
    df = make_df(tmp_path, data, chunk=1)
    cv = GroupKFold(n_splits=2, group_col="g", random_state=7)
    non_empty_folds = 0
    for _, _, test in cv.split(df):
        rows = _fold_rows(test)
        groups = {r["g"] for r in rows}
        # Empty (0 groups) or single-group (1) fold — valid.
        # Multi-group (>= 2) fold — group split bug.
        assert len(groups) <= 1, (
            f"fold contains mixed groups {groups} — group split bug"
        )
        if rows:
            non_empty_folds += 1
    # Sanity: at least one fold got rows (dispatch works; if all
    # empty, GroupKFold is broken).
    assert non_empty_folds >= 1, (
        "all folds empty — GroupKFold dispatch broken"
    )'''

CORRECT_TEST = '''def test_group_kfold_chunk_1(tmp_path):
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
    assert sorted(all_x) == list(range(20))'''


def fix_test():
    print("\n=== replace fix2's wrong invariant with correct ===")
    p = Path("tests/test_cv_stratified.py")
    applied = apply_or_fail(
        str(p), WRONG_TEST, CORRECT_TEST,
        label="group_kfold_chunk_1_invariant",
    )
    if applied:
        assert_compiles(p)
        print("  applied")
    else:
        print("  already correct — skip")


# ============ 2. Dogfood with correct invariant ============

def dogfood():
    print("\n=== dogfood: group->fold invariant + empty folds valid ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        from collections import defaultdict
        import numpy as _np_
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import (
            CadwDataFrame as _CDF, GroupKFold as _GKF,
            MemoryGuard as _MG,
        )

        # (a) chunk=1: 4 groups, 2 folds -> each group in one fold
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "x": pa_.array(_np_.arange(20).astype(_np_.int64)),
                "g": pa_.array(["a", "b", "c", "d"] * 5),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src, row_group_size=1)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            cv = _GKF(n_splits=2, group_col="g", random_state=7)
            group_to_folds: dict = defaultdict(set)
            all_x: list = []
            for fold_i, _, test in cv.split(df):
                rows = []
                for b in test._iter_filtered_batches():
                    rows.extend(b.to_pylist())
                for r in rows:
                    group_to_folds[r["g"]].add(fold_i)
                    all_x.append(r["x"])
            # Each group in exactly one fold
            for g, folds in group_to_folds.items():
                assert len(folds) == 1, (
                    f"group {g!r} split across {folds}"
                )
            # All groups covered
            assert set(group_to_folds.keys()) == {"a", "b", "c", "d"}
            # All rows covered
            assert sorted(all_x) == list(range(20))

        # (b) n_splits > #groups: empty folds valid (must not crash)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "x": pa_.array(_np_.arange(10).astype(_np_.int64)),
                "g": pa_.array(["a", "b"] * 5),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            cv = _GKF(n_splits=5, group_col="g", random_state=7)
            sizes = []
            for _, _, test in cv.split(df):
                sizes.append(test.count())
            assert sum(sizes) == 10, f"union rows: {sum(sizes)}"
            # 2 groups, 5 folds -> at least 3 empty folds (pigeonhole)
            assert sizes.count(0) >= 3, (
                f"expected >=3 empty folds: sizes={sizes}"
            )

        # (c) null group raises
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "x": [1, 2, 3, 4],
                "g": pa_.array(["a", None, "b", "a"], type=pa_.string()),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            cv = _GKF(n_splits=2, group_col="g")
            try:
                list(cv.split(df))
            except ValueError as e:
                assert "nulls" in str(e)
            else:
                raise AssertionError("null group must raise")

        # (d) stratify still works (regression for fix1 dispatch)
        import inspect
        from cadw.cv import KFold as _KF
        assert "_stratified_fold_ids" in inspect.getsource(_KF.split), (
            "KFold.split stratify dispatch broken (fix1 regression)"
        )

        # (e) stratify: positives distributed across folds
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            rng = _np_.random.default_rng(0)
            n = 200
            y = _np_.zeros(n, dtype=_np_.int64)
            pos_idx = rng.choice(n, size=10, replace=False)
            y[pos_idx] = 1
            t = pa_.table({
                "x": pa_.array(rng.normal(size=n)),
                "y": pa_.array(y),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=20,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            cv = _KF(n_splits=5, stratify_by="y", random_state=42)
            pos_per_fold = []
            for _, _, test in cv.split(df):
                rows = []
                for b in test._iter_filtered_batches():
                    rows.extend(b.to_pylist())
                pos_per_fold.append(sum(1 for r in rows if r["y"] == 1))
            assert sum(pos_per_fold) == 10
            assert all(p > 0 for p in pos_per_fold), pos_per_fold

        print(
            "  group->fold invariant + empty folds + null + "
            "stratify dispatch + stratify positives ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 3. Tests ============

def run_pytest():
    print("\n=== pre-flight: test_facade_has_no_as_py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "-q"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** FACADE AS_PY VIOLATION — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_cv_stratified.py",
         "tests/test_cv.py",
         "tests/test_cv_keyed_prng.py",
         "tests/test_cv_e2e.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 4. main ============

def main():
    print("=== v0.9.23h-fix3: group->fold invariant ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_test()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23h: KFold(stratify_by=) + GroupKFold (AUDIT_FULL_v0922 "
        "item 7.A.9, Top-10 gap #3). "
        "Two thin wrappers over existing streaming KFold (v0.9.3a): "
        "stratify_by=col (keyed PRNG per stratum, binomial sizes, "
        "O(batch) state, null-y own stratum); GroupKFold (fold_id "
        "from hash(group_key), all group rows in one fold, O(batch) "
        "state, null group -> ValueError). "
        "_attach_column_with_fn fn(offset, batch) instead of fn("
        "offset, n_b). Export GroupKFold from cadw. "
        "fix1 (2 production bugs + 1 hidden-test-gap): "
        "(1) GroupKFold _iter_filtered_batches(columns=[...]) no "
        "such kwarg; iterate all, select per batch. "
        "(2) add_stratify_branch custom guard `if _stratified_fold_"
        "ids in s: return` false-positived on helper name (§13.4/182 "
        "3rd occurrence in series); fix: apply_or_fail directly. "
        "(3) Hidden test gap: dogfood + test_kfold_stratify_basic "
        "passed with bug (2) because data layout distributed "
        "positives via row_id % k fallback. Fix: test_kfold_split_"
        "dispatches_stratify (AST) + dogfood asserts same. "
        "fix2 (test-only, then fix3 corrects fix2's wrong invariant): "
        "fix2 changed test_group_kfold_chunk_1 to assert len(groups) "
        "<= 1 per fold — but with 4 groups / 2 folds, each fold "
        "gets 2 groups (CORRECT GroupKFold behavior). Wrong "
        "invariant (§13.4/164/171 test-authoring class). "
        "fix3 replaces with correct invariant: group -> set(fold_"
        "ids) has size 1 (no group split across folds) + union of "
        "groups covers full set + union of rows covers all N. "
        "Dogfood also strengthened with group->fold map. "
        "Structural lesson: GroupKFold's contract is 'group lands "
        "in EXACTLY one fold', NOT 'fold contains one group'. "
        "K groups / n_splits folds -> K/n_splits groups per fold "
        "(or 0 if K < n_splits). Test design must check the "
        "correct invariant. "
        "+tests/test_cv_stratified.py (20 tests: stratify basic, "
        "union coverage, null-y stratum, missing col, materialize+"
        "stratify raises, determinism, non-str type, dispatch AST; "
        "group no-split, union, null group raises, missing col, "
        "determinism, invalid n_splits, empty col, chunk=1 group->"
        "fold invariant; 4 helper AST guards). "
        "Pre-flight test_facade_has_no_as_py runs BEFORE full pytest. "
        "Fast: ~2123 passed.",
        self_path="scripts/patch_v0923h_fix3_group_invariant.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23h-fix3 applied. 7.A.9 COMPLETE.")


if __name__ == "__main__":
    main()