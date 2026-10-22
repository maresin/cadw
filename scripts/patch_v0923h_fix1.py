#!/usr/bin/env python3
# v0.9.23h-fix1 — 2 bugs + 1 hidden-test-gap in v0.9.23h.
#
# (1) GroupKFold.split(): _iter_filtered_batches() has no `columns`
#     kwarg -> TypeError. Fix: iterate all columns; select group_col
#     per batch.
#
# (2) add_stratify_branch() custom guard false-positive (§13.4/182,
#     3rd occurrence in series: v0.9.23g-fix3, v0.9.23g-fix4,
#     v0.9.23h): `if "_stratified_fold_ids" in s: return` — helper
#     added by add_helpers() -> guard fires -> branch never inserted
#     into split(). Fix: use apply_or_fail directly (it IS the
#     idempotency guard, returns False if `new` already present,
#     scoped to exact old/new block).
#
# (3) Hidden test gap: dogfood passed stratify section accidentally
#     (data layout distributed positives across folds via row_id % k
#     fallback). test_kfold_stratify_basic in pytest would also pass
#     for that data. Fix: AST assertion that KFold.split source
#     contains _stratified_fold_ids (catches dispatch missing).
#     + dogfood asserts same via inspect.getsource.
#
# No new features, no behavior change — only fixes what should have
# been there.
#
# Run: uv run python scripts/patch_v0923h_fix1.py

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


# ============ 1. GroupKFold null-check: no columns= kwarg ============

OLD_NULL_CHECK = '''        # Pre-check: group_col has no nulls (fail before iteration).
        for b in df._iter_filtered_batches(columns=[self.group_col]):
            if b.column(self.group_col).null_count > 0:
                raise ValueError(
                    f"GroupKFold: group_col {self.group_col!r} has "
                    f"nulls; groups must be non-null. Drop nulls "
                    f"before split (df[df['{self.group_col}'].notnull()])."
                )'''

NEW_NULL_CHECK = '''        # Pre-check: group_col has no nulls (fail before iteration).
        # v0.9.23h-fix1: _iter_filtered_batches() has no `columns`
        # kwarg -> iterate all, select group_col per batch.
        for b in df._iter_filtered_batches():
            if b.column(self.group_col).null_count > 0:
                raise ValueError(
                    f"GroupKFold: group_col {self.group_col!r} has "
                    f"nulls; groups must be non-null. Drop nulls "
                    f"before split (df[df['{self.group_col}'].notnull()])."
                )'''


def fix_group_null_check():
    print("\n=== GroupKFold null-check: no columns= kwarg ===")
    p = Path("cadw/cv.py")
    applied = apply_or_fail(
        str(p), OLD_NULL_CHECK, NEW_NULL_CHECK,
        label="group_null_check",
    )
    print("  applied" if applied else "  already fixed — skip")
    if applied:
        assert_compiles(p)


# ============ 2. add_stratify_branch: apply_or_fail, no custom guard ============

OLD_SPLIT_BLOCK = '''        n = df.count()
        if self.n_splits > n:
            raise ValueError(
                f"KFold: n_splits={self.n_splits} > n={n}"
            )

        if self.materialize:'''

NEW_SPLIT_BLOCK = '''        n = df.count()
        if self.n_splits > n:
            raise ValueError(
                f"KFold: n_splits={self.n_splits} > n={n}"
            )

        # v0.9.23h: stratify_by dispatch. Streaming only; exact-
        # balanced + stratify (materialize=True) deferred.
        if self.stratify_by is not None:
            if self.materialize:
                raise NotImplementedError(
                    "KFold: materialize=True + stratify_by is "
                    "not supported (exact-balanced per stratum "
                    "requires full y in memory; v0.9.23h uses "
                    "streaming keyed-PRNG stratify instead). "
                    "Use materialize=False + stratify_by, or "
                    "materialize=True without stratify_by."
                )
            if self.stratify_by not in df.columns:
                raise KeyError(
                    f"KFold: stratify_by column "
                    f"{self.stratify_by!r} not in {df.columns}"
                )
            k = self.n_splits
            seed = (_DEFAULT_SEED if self.random_state is None
                    else int(self.random_state))
            strat_col = self.stratify_by
            def _fn(offset: int, b: pa.RecordBatch) -> pa.Array:
                n_b = b.num_rows
                row_ids = np.arange(offset, offset + n_b,
                                    dtype=np.uint64)
                ids = _stratified_fold_ids(
                    row_ids, b, strat_col, k, seed,
                )
                return pa.array(ids, type=pa.int32())
            augmented = _attach_column_with_fn(
                df, "__fold", pa.int32(), _fn,
            )
            base_cols = list(df.columns)
            for i in range(self.n_splits):
                train = augmented[augmented["__fold"] != i][base_cols]
                test = augmented[augmented["__fold"] == i][base_cols]
                yield i, train, test
            return

        if self.materialize:'''


def fix_stratify_branch():
    print("\n=== split(): add stratify branch (apply_or_fail, no guard) ===")
    p = Path("cadw/cv.py")
    applied = apply_or_fail(
        str(p), OLD_SPLIT_BLOCK, NEW_SPLIT_BLOCK,
        label="stratify_branch",
    )
    print("  applied" if applied else "  already present — skip")
    if applied:
        assert_compiles(p)


# ============ 3. Test: AST dispatch assertion ============

def add_dispatch_test():
    print("\n=== add test: KFold.split dispatches stratify ===")
    p = Path("tests/test_cv_stratified.py")
    s = p.read_text()
    if "test_kfold_split_dispatches_stratify" in s:
        print("  already present — skip")
        return
    extra = '''

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
'''

    p.write_text(s + extra)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 4. Dogfood ============

def dogfood():
    print("\n=== dogfood: stratify dispatch + group + null ===")

    def _verify():
        import inspect
        import tempfile
        from pathlib import Path as _P
        import numpy as _np_
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import (
            CadwDataFrame as _CDF, GroupKFold as _GKF,
            KFold as _KF, MemoryGuard as _MG,
        )

        # (a) AST dispatch guard
        assert "_stratified_fold_ids" in inspect.getsource(_KF.split), (
            "KFold.split must dispatch to _stratified_fold_ids"
        )

        # (b) stratify: exact fold sizes vary (binomial) but per-fold
        # positive count > 0 and total = 25 (all positives distributed).
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            rng = _np_.random.default_rng(0)
            n = 500
            y = _np_.zeros(n, dtype=_np_.int64)
            pos_idx = rng.choice(n, size=25, replace=False)
            y[pos_idx] = 1
            t = pa_.table({
                "x": pa_.array(rng.normal(size=n)),
                "y": pa_.array(y),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=50,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            cv = _KF(n_splits=5, stratify_by="y", random_state=42)
            pos_per_fold = []
            total_rows = 0
            for _, _, test in cv.split(df):
                rows = []
                for b in test._iter_filtered_batches():
                    rows.extend(b.to_pylist())
                total_rows += len(rows)
                pos_per_fold.append(sum(1 for r in rows if r["y"] == 1))
            assert total_rows == n
            assert sum(pos_per_fold) == 25, pos_per_fold
            assert all(p > 0 for p in pos_per_fold), pos_per_fold

        # (c) group: chunk=1, no group in two folds
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            n = 60
            groups = [f"g{i % 6}" for i in range(n)]
            t = pa_.table({
                "x": pa_.array(_np_.arange(n).astype(_np_.int64)),
                "g": pa_.array(groups),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src, row_group_size=1)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            cv = _GKF(n_splits=3, group_col="g", random_state=42)
            all_x = []
            for _, _, test in cv.split(df):
                rows = []
                for b in test._iter_filtered_batches():
                    rows.extend(b.to_pylist())
                all_x.extend(r["x"] for r in rows)
            assert sorted(all_x) == list(range(n))

        # (d) null group raises
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

        # (e) backward compat: KFold no stratify
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"x": list(range(20))})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=5,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            cv = _KF(n_splits=2)
            for _, _, _ in cv.split(df):
                pass

        print("  dispatch + stratify + group + null + backward compat ok")

    run_verify_in_subprocess(_verify)


# ============ 5. Tests ============

def run_pytest():
    # Pre-flight: facade as_py
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


# ============ 6. main ============

def main():
    print("=== v0.9.23h-fix1: stratify dispatch + group null ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_group_null_check()
    fix_stratify_branch()
    add_dispatch_test()
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
        "NotImplementedError. Null in y -> own stratum (row_id "
        "mixing spreads across folds, groupby dropna=False parity). "
        "GroupKFold(n_splits, group_col): fold_id = splitmix64("
        "hash(group_key, seed)) % k — all rows of same group in "
        "same fold; O(batch) state; fold sizes unbalanced by group "
        "sizes (correct by contract). Null in group_col -> "
        "ValueError (fail-fast before iteration). "
        "_attach_column_with_fn extended: fn(offset, batch) instead "
        "of fn(offset, n_b) — stratify needs y column per batch; "
        "existing KFold callers updated in same patch. "
        "Export GroupKFold from cadw (public API). "
        "fix1 (2 production bugs + 1 hidden-test-gap): "
        "(1) GroupKFold.split(): _iter_filtered_batches() has no "
        "`columns` kwarg -> TypeError; iterate all columns, select "
        "group_col per batch. "
        "(2) add_stratify_branch() custom guard "
        "`if \"_stratified_fold_ids\" in s: return` false-positived "
        "(helper added earlier in same patch by add_helpers) -> "
        "stratify branch never inserted into split() -> stratify "
        "silently fell through to non-stratify fallback. §13.4/182 "
        "3rd occurrence in series (v0.9.23g-fix3, v0.9.23g-fix4, "
        "v0.9.23h). Fix: apply_or_fail directly — it IS the "
        "idempotency guard. "
        "(3) Hidden test gap: dogfood and test_kfold_stratify_basic "
        "both passed with bug (2) because data layout distributed "
        "positives across folds via row_id % k fallback. Fix: "
        "test_kfold_split_dispatches_stratify (AST: split source "
        "contains _stratified_fold_ids) + dogfood asserts same via "
        "inspect.getsource. Would have caught bug (2) at first run. "
        "+tests/test_cv_stratified.py (19 tests: stratify basic, "
        "union coverage, null-y stratum, missing col, materialize+"
        "stratify raises, determinism, non-str type, dispatch AST; "
        "group no-split, union, null group raises, missing col, "
        "determinism, invalid n_splits, empty col, chunk=1 worst "
        "case; 4 helper AST guards). "
        "Pre-flight: test_facade_has_no_as_py runs BEFORE full "
        "pytest (new rule from v0.9.23g). "
        "Process note: 2nd fix-cycle on v0.9.23h (at §13.4/238 "
        "threshold). Root cause of both cycles: the same recurring "
        "trap (`if marker in s: skip`) that v0.9.23g lesson said to "
        "avoid — but I wrote it again. apply_or_fail is the guard; "
        "custom guards are wrong. Structural fix in v0.9.23h: the "
        "dispatch AST test catches future dispatch regressions. "
        "Fast: 2104 -> ~2123 passed.",
        self_path="scripts/patch_v0923h_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23h-fix1 applied. 7.A.9 complete.")


if __name__ == "__main__":
    main()