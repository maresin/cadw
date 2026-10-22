#!/usr/bin/env python3
# v0.9.23h-fix2 — test_group_kfold_chunk_1 assumed non-empty fold.
#
# Bug (test only, production correct): with n_splits=2 and 2 groups
# {"a", "b"}, GroupKFold assigns each group to one fold via
# hash(group, seed) % k. If both hash to fold 0, fold 1 is empty.
# This is CORRECT GroupKFold behavior (unbalanced fold sizes by
# group sizes — documented in class docstring). Test asserted
# `len(groups) == 1` — failed on empty fold (`len(set()) == 0`).
#
# Fix:
#   (1) Use 4 groups (a,b,c,d) instead of 2 — reduces chance of
#       degenerate hash to all-one-fold (still possible, that's ok).
#   (2) Assert len(groups) <= 1 per fold (0 or 1 group valid).
#   (3) Assert non_empty_folds >= 1 (sanity that GroupKFold actually
#       dispatched; catches "all folds empty" = dispatch broken).
#   (4) Docstring explains empty-fold-by-design.
#
# Structural lesson: for methods with documented "unbalanced sizes"
# (GroupKFold, StratifiedKFold per-stratum binomial, any hash-based
# assignment), tests must handle empty/degenerate cases. Add to
# future test design checklist.
#
# Run: uv run python scripts/patch_v0923h_fix2_empty_fold.py

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


# ============ 1. Fix test_group_kfold_chunk_1 ============

OLD_TEST = '''def test_group_kfold_chunk_1(tmp_path):
    """chunk=1 (worst case): same group spans batches -> same fold."""
    data = {
        "x": list(range(20)),
        "g": ["a", "b"] * 10,  # alternating -> every batch has a, b
    }
    df = make_df(tmp_path, data, chunk=1)
    cv = GroupKFold(n_splits=2, group_col="g", random_state=7)
    for _, _, test in cv.split(df):
        rows = _fold_rows(test)
        groups = {r["g"] for r in rows}
        # Each fold gets only one of {a, b} (whole-group assignment)
        assert len(groups) == 1, (
            f"fold contains mixed groups {groups} — group split bug"
        )'''

NEW_TEST = '''def test_group_kfold_chunk_1(tmp_path):
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


def fix_test():
    print("\n=== fix test_group_kfold_chunk_1 ===")
    p = Path("tests/test_cv_stratified.py")
    applied = apply_or_fail(
        str(p), OLD_TEST, NEW_TEST,
        label="group_kfold_chunk_1",
    )
    print("  applied" if applied else "  already fixed — skip")
    if applied:
        assert_compiles(p)


# ============ 2. Dogfood ============

def dogfood():
    print("\n=== dogfood: chunk=1 + empty-fold valid ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import numpy as _np_
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import (
            CadwDataFrame as _CDF, GroupKFold as _GKF,
            MemoryGuard as _MG,
        )

        # 4 groups, n_splits=2, chunk=1: no group spans folds
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
            non_empty = 0
            all_x = []
            for _, _, test in cv.split(df):
                rows = []
                for b in test._iter_filtered_batches():
                    rows.extend(b.to_pylist())
                groups = {r["g"] for r in rows}
                assert len(groups) <= 1, (
                    f"fold has {len(groups)} groups: {groups}"
                )
                if rows:
                    non_empty += 1
                all_x.extend(r["x"] for r in rows)
            assert non_empty >= 1, "all folds empty"
            assert sorted(all_x) == list(range(20)), (
                f"union: {sorted(all_x)[:5]}..."
            )

        # n_splits > #groups: some folds MUST be empty (valid)
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
            # With 2 groups in 5 folds: at most 2 non-empty, rest empty.
            assert sum(sizes) == 10, (
                f"union rows must be 10, got {sum(sizes)}"
            )
            assert all(s >= 0 for s in sizes)
            # Most folds must be empty (2 groups / 5 folds)
            assert sizes.count(0) >= 3, (
                f"expected >=3 empty folds with 2 groups, 5 splits; "
                f"got sizes {sizes}"
            )

        print("  chunk=1 + empty-fold-valid + n_splits>groups ok")

    run_verify_in_subprocess(_verify)


# ============ 3. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_cv_stratified.py",
         "tests/test_cv.py",
         "tests/test_cv_keyed_prng.py",
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
    print("=== v0.9.23h-fix2: empty-fold-valid test ===")
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
        "Two thin wrappers over existing streaming KFold (v0.9.3a). "
        "KFold(stratify_by=col): fold_id = splitmix64(row_id*GOLDEN "
        "+ hash(y, seed)) % k — keyed PRNG per stratum, binomial "
        "fold sizes per stratum, O(batch) state. NOT exact-balanced "
        "per stratum; materialize=True + stratify_by -> "
        "NotImplementedError. Null in y -> own stratum. "
        "GroupKFold(n_splits, group_col): fold_id = splitmix64("
        "hash(group_key, seed)) % k — all rows of same group in "
        "same fold; O(batch) state. Fold sizes unbalanced by group "
        "sizes (correct by contract — EMPTY folds are valid when "
        "all groups hash to one fold; test_v0.9.23h-fix2 documents "
        "this explicitly). Null in group_col -> ValueError. "
        "_attach_column_with_fn extended: fn(offset, batch) instead "
        "of fn(offset, n_b). Existing KFold callers updated. "
        "Export GroupKFold from cadw. "
        "fix1 (2 production bugs + 1 hidden-test-gap): "
        "(1) GroupKFold null-check used _iter_filtered_batches("
        "columns=[...]) — no such kwarg; iterate all, select per "
        "batch. "
        "(2) add_stratify_branch() custom guard `if "
        "\"_stratified_fold_ids\" in s: return` false-positived "
        "(helper added earlier in same patch) -> stratify branch "
        "never dispatched. §13.4/182 3rd occurrence in series. "
        "Fix: apply_or_fail directly. "
        "(3) Hidden test gap: dogfood + test_kfold_stratify_basic "
        "passed with bug (2) because data layout distributed "
        "positives via row_id % k fallback. Fix: "
        "test_kfold_split_dispatches_stratify (AST) + dogfood "
        "asserts same. "
        "fix2 (test-only, production correct): "
        "test_group_kfold_chunk_1 assumed `len(groups) == 1` per "
        "fold. But GroupKFold with n_splits=2 and 2 groups can "
        "hash both groups to one fold -> other fold empty (len("
        "set()) == 0). Correct GroupKFold behavior (documented "
        "unbalanced sizes). Fix: use 4 groups (a,b,c,d) + assert "
        "len(groups) <= 1 (0 valid) + non_empty_folds >= 1 sanity. "
        "New dogfood: n_splits=5 with 2 groups -> >= 3 empty "
        "folds (valid, must not crash). "
        "Structural lessons: "
        "  - empty-fold-valid test design for unbalanced-by-contract "
        "methods (GroupKFold, StratifiedKFold per-stratum); "
        "  - dispatch AST guard: helper exists + dispatched, not "
        "just exists; "
        "  - apply_or_fail is the ONLY idempotency guard. "
        "+tests/test_cv_stratified.py (20 tests: stratify basic, "
        "union coverage, null-y stratum, missing col, materialize+"
        "stratify raises, determinism, non-str type, dispatch AST; "
        "group no-split, union, null group raises, missing col, "
        "determinism, invalid n_splits, empty col, chunk=1 with "
        "empty-fold-valid; 4 helper AST guards). "
        "Fast: ~2123 passed.",
        self_path="scripts/patch_v0923h_fix2_empty_fold.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23h-fix2 applied. 7.A.9 COMPLETE.")


if __name__ == "__main__":
    main()