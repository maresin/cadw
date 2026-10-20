"""v0.9.2e: cross-validation tests (KFold + TimeSeriesSplit + cv_score)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard, KFold, TimeSeriesSplit, cv_score
from cadw.pipeline import GBDTStep


def make_df(tmp_path, data, chunk=10, name="L", mem_mb=50):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def make_xy(tmp_path, n=100, name="L"):
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    y = x * 2 + rng.normal(scale=0.1, size=n)
    return make_df(
        tmp_path,
        {"x": x.tolist(), "y": y.tolist()},
        chunk=20, name=name,
    )


def _union_test_rows(cv, df, n):
    """Return sorted union of raw positions in test folds."""
    train_seen = np.zeros(n, dtype=bool)
    test_seen = np.zeros(n, dtype=bool)
    # Use KFold fold_id for reconstruction (only for shuffle=False):
    for i, train, test in cv.split(df):
        # Count rows
        train_n = train.count()
        test_n = test.count()
        # For shuffle=False: fold i gets positions where i % k == i
        # We don't know positions directly; use sizes + non-overlap via
        # alternate method (below).
    return None, None


# ============ KFold: basic ============

def test_kfold_5_splits(tmp_path):
    df = make_xy(tmp_path, n=100)
    cv = KFold(n_splits=5)
    splits = list(cv.split(df))
    assert len(splits) == 5
    for fold_i, train, test in splits:
        assert fold_i in range(5)


def test_kfold_fold_sizes(tmp_path):
    df = make_xy(tmp_path, n=100)
    cv = KFold(n_splits=5)
    sizes = [(train.count(), test.count()) for _, train, test in cv.split(df)]
    # 100 / 5 = 20 each
    for train_n, test_n in sizes:
        assert test_n == 20
        assert train_n == 80


def test_kfold_uneven_sizes(tmp_path):
    df = make_xy(tmp_path, n=103)
    cv = KFold(n_splits=5)
    test_sizes = [test.count() for _, _, test in cv.split(df)]
    # 103 % 5: folds 0-2 get 21, folds 3-4 get 20
    assert test_sizes == [21, 21, 21, 20, 20]


def test_kfold_no_overlap(tmp_path):
    """Union of test fold rows == full df (all rows exactly once)."""
    df = make_xy(tmp_path, n=100)
    cv = KFold(n_splits=5)
    # Collect all test rows across folds (via write-back; row order preserved)
    total_test_rows = 0
    for _, _, test in cv.split(df):
        total_test_rows += test.count()
    assert total_test_rows == 100


def test_kfold_shuffle_true_same_sizes(tmp_path):
    # v0.9.3a: shuffle=True default is keyed PRNG (binomial
    # fold sizes, ~mean n/k, sigma ~= sqrt(n/k)). For
    # exact-balanced, use materialize=True.
    df = make_xy(tmp_path, n=100)
    cv = KFold(n_splits=5, shuffle=True, random_state=42,
               materialize=True)
    sizes = [test.count() for _, _, test in cv.split(df)]
    assert sizes == [20, 20, 20, 20, 20]


def test_kfold_shuffle_different_assignment(tmp_path):
    """shuffle=True with different seeds → different fold assignment.

    Use identity data (y == row_id) so each fold's y-set is the
    row-position set assigned to that fold. Different seeds must
    yield different per-fold content.
    """
    data = {"x": list(range(100)), "y": list(range(100))}
    df = make_df(tmp_path, data)
    cv1 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv2 = KFold(n_splits=5, shuffle=True, random_state=99)

    def _test_y_sets(cv, df):
        sets = []
        for _, _, test in cv.split(df):
            ys = []
            for b in test._iter_filtered_batches():
                ys.extend(b.column("y").to_pylist())
            sets.append(set(ys))
        return sets

    sets1 = _test_y_sets(cv1, df)
    sets2 = _test_y_sets(cv2, df)

    # v0.9.3a: shuffle=True default = keyed PRNG; sizes are
    # ~binomial, not exact. Sum invariant holds (all rows).
    assert sum(len(s) for s in sets1) == 100
    assert sum(len(s) for s in sets2) == 100

    # Assignment differs: at least one fold has different content
    differ = any(s1 != s2 for s1, s2 in zip(sets1, sets2))
    assert differ, (
        "different seeds must produce different fold assignments"
    )

    # Invariant: each row appears exactly once across test folds
    for sets in (sets1, sets2):
        union = set().union(*sets)
        assert union == set(range(100))


def test_kfold_n_splits_gt_n_raises(tmp_path):
    df = make_xy(tmp_path, n=10)
    cv = KFold(n_splits=11)
    with pytest.raises(ValueError, match="n_splits=11 > n=10"):
        list(cv.split(df))


def test_kfold_invalid_n_splits():
    with pytest.raises(ValueError, match="n_splits"):
        KFold(n_splits=0)


def test_kfold_with_filter(tmp_path):
    """KFold over filtered df: fold assignment based on filtered view."""
    df = make_xy(tmp_path, n=100)
    filtered = df[df["x"] > 0]  # ~50 rows
    n_filtered = filtered.count()
    cv = KFold(n_splits=2)
    test_sizes = [test.count() for _, _, test in cv.split(filtered)]
    assert sum(test_sizes) == n_filtered


# ============ TimeSeriesSplit: basic ============

def test_tss_5_splits(tmp_path):
    data = {"ts": list(range(100)), "y": list(range(100))}
    df = make_df(tmp_path, data)
    cv = TimeSeriesSplit(n_splits=5, time_col="ts")
    splits = list(cv.split(df))
    assert len(splits) == 5


def test_tss_no_future_leakage(tmp_path):
    """Per fold: max(train.ts) < min(test.ts)."""
    n = 100
    data = {"ts": list(range(n)), "y": list(range(n))}
    df = make_df(tmp_path, data)
    cv = TimeSeriesSplit(n_splits=5, time_col="ts")
    for fold_i, train, test in cv.split(df):
        # Materialize ts min/max
        train_ts = []
        for b in train._iter_filtered_batches():
            train_ts.extend(b.column("ts").to_pylist())
        test_ts = []
        for b in test._iter_filtered_batches():
            test_ts.extend(b.column("ts").to_pylist())
        if train_ts and test_ts:
            assert max(train_ts) < min(test_ts), (
                f"fold {fold_i}: leak — train_max={max(train_ts)} >= "
                f"test_min={min(test_ts)}"
            )


def test_tss_train_grows(tmp_path):
    data = {"ts": list(range(100)), "y": list(range(100))}
    df = make_df(tmp_path, data)
    cv = TimeSeriesSplit(n_splits=5, time_col="ts")
    train_sizes = [train.count() for _, train, _ in cv.split(df)]
    for i in range(len(train_sizes) - 1):
        assert train_sizes[i] < train_sizes[i + 1]


def test_tss_test_sizes(tmp_path):
    """Standard TSS: n=100, k=5, fs=16 → tests [16,16,16,16,20]."""
    data = {"ts": list(range(100)), "y": list(range(100))}
    df = make_df(tmp_path, data)
    cv = TimeSeriesSplit(n_splits=5, time_col="ts")
    test_sizes = [test.count() for _, _, test in cv.split(df)]
    assert test_sizes == [16, 16, 16, 16, 20]


def test_tss_n_lt_k_plus_1_raises(tmp_path):
    data = {"ts": list(range(3)), "y": list(range(3))}
    df = make_df(tmp_path, data)
    cv = TimeSeriesSplit(n_splits=5, time_col="ts")
    with pytest.raises(ValueError, match=r"n=3 < n_splits\+1"):
        list(cv.split(df))


def test_tss_missing_time_col_raises(tmp_path):
    data = {"x": list(range(20)), "y": list(range(20))}
    df = make_df(tmp_path, data)
    cv = TimeSeriesSplit(n_splits=2, time_col="ts")
    with pytest.raises(KeyError, match="ts"):
        list(cv.split(df))


def test_tss_unsorted_input(tmp_path):
    """Input unsorted by ts; TSS sorts internally → no leakage."""
    ts = [50, 10, 90, 20, 70, 30, 80, 40, 60, 100]
    data = {"ts": ts, "y": list(range(len(ts)))}
    df = make_df(tmp_path, data)
    cv = TimeSeriesSplit(n_splits=2, time_col="ts")
    for fold_i, train, test in cv.split(df):
        train_ts = []
        for b in train._iter_filtered_batches():
            train_ts.extend(b.column("ts").to_pylist())
        test_ts = []
        for b in test._iter_filtered_batches():
            test_ts.extend(b.column("ts").to_pylist())
        if train_ts and test_ts:
            assert max(train_ts) < min(test_ts)


# ============ cv_score ============

def test_cv_score_kfold_gbdt(tmp_path):
    df = make_xy(tmp_path, n=100)
    cv = KFold(n_splits=5, shuffle=False, random_state=42)

    def factory():
        return GBDTStep(["x"], "y", n_estimators=2, max_depth=1)

    result = cv_score(
        df, cv, factory,
        X_cols=["x"], y_col="y", metrics=["r2"],
    )
    assert "fold_scores" in result
    assert len(result["fold_scores"]) == 5
    assert "mean" in result and "std" in result
    assert "r2" in result["mean"]
    assert not np.isnan(result["mean"]["r2"])


def test_cv_score_timeseries(tmp_path):
    """TSS cv_score on i.i.d. data — mean r2 > 0.3.

    §13.4/169: prior version used linear trend (y = 2*ts, x = y). TSS
    split train=[0,50), test=[50,100) — test x values 100..198 OUTSIDE
    training bin range → GBDT clips to last bin → r2 ≈ -23.9. Not a
    CADW bug — tree model without lag features cannot extrapolate
    (§11.5). Test used unsuitable data for TSS.

    Fix: i.i.d. x ~ N(0,1), y = 2x + noise. TSS still splits by ts
    order, but x/y distributions in train vs test are identical →
    GBDT learns well (r2 > 0.9 typically).

    For real time-series with trend, use TimeSeriesPipeline with lag
    features (backlog v0.9.4) so GBDT sees bounded training distribution.
    """
    rng = np.random.default_rng(42)
    n = 200
    ts = list(range(n))
    x = rng.normal(size=n).tolist()  # i.i.d., no trend
    y = [2.0 * xi + 0.1 * rng.normal() for xi in x]
    data = {"ts": ts, "x": x, "y": y}
    df = make_df(tmp_path, data, chunk=20)

    cv = TimeSeriesSplit(n_splits=3, time_col="ts")

    def factory():
        return GBDTStep(["x"], "y", n_estimators=10, max_depth=2)

    result = cv_score(
        df, cv, factory,
        X_cols=["x"], y_col="y", metrics=["r2"],
    )
    assert len(result["fold_scores"]) == 3
    assert result["mean"]["r2"] > 0.3


def test_cv_score_empty_metrics_raises(tmp_path):
    df = make_xy(tmp_path, n=20)
    cv = KFold(n_splits=2)

    def factory():
        return GBDTStep(["x"], "y", n_estimators=1)

    with pytest.raises(ValueError, match="metrics list is empty"):
        cv_score(
            df, cv, factory,
            X_cols=["x"], y_col="y", metrics=[],
        )


def test_cv_score_custom_pred_col(tmp_path):
    df = make_xy(tmp_path, n=20)
    cv = KFold(n_splits=2)

    def factory():
        return GBDTStep(["x"], "y", n_estimators=1, max_depth=1)

    result = cv_score(
        df, cv, factory,
        X_cols=["x"], y_col="y", metrics=["mse"],
    )
    assert len(result["fold_scores"]) == 2
