"""Tests for D.1.1 Isolation Forest (v0.9.31)."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.ml.iforest import fit_isolation_forest, IsolationForestModel



def _scores_to_numpy(df):
    """Extract 'score' column from score_samples CadwDataFrame as 1-D numpy.

    v0.9.95: iforest.score_samples now returns CadwDataFrame (unified
    return type). Tests that check numeric values need numpy.
    """
    import numpy as np
    parts = [b.column("score").to_numpy(zero_copy_only=False)
             for b in df._iter_filtered_batches()]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _df(X):
    cols = {f"x{i}": pa.array(X[:, i], type=pa.float64()) for i in range(X.shape[1])}
    return CadwDataFrame.from_table(pa.table(cols))


def test_fit_smoke_and_state_independent_of_n():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(2000, 3))
    m = fit_isolation_forest(_df(X), ["x0", "x1", "x2"], n_trees=20,
                            subsample_size=128, random_state=0)
    assert isinstance(m, IsolationForestModel)
    # state must be bounded by n_trees * subsample (2*subsample nodes * 40 B)
    assert m.state_nbytes() < 20 * 2 * 128 * 40
    assert m.n_trees_ == 20


def test_obvious_outlier_gets_top_score():
    # Standard IForest params (subsample=256, n_trees=100) per the paper.
    rng = np.random.RandomState(1)
    X = np.vstack([rng.normal(0, 1, size=(200, 2)), np.array([[1000.0, 1000.0]])])
    m = fit_isolation_forest(_df(X), ["x0", "x1"], n_trees=100,
                            subsample_size=256, random_state=0)
    s = _scores_to_numpy(m.score_samples(_df(X)))
    # The extreme outlier must be in the top-5 highest-scored rows.
    top5 = set(np.argsort(-s)[:5].tolist())
    assert len(X) - 1 in top5, f"outlier not in top-5; top5={top5}"


def test_determinism_same_seed():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(300, 2))
    m1 = fit_isolation_forest(_df(X), ["x0", "x1"], n_trees=10,
                             subsample_size=64, random_state=42)
    m2 = fit_isolation_forest(_df(X), ["x0", "x1"], n_trees=10,
                             subsample_size=64, random_state=42)
    s1 = _scores_to_numpy(m1.score_samples(_df(X))); s2 = _scores_to_numpy(m2.score_samples(_df(X)))
    assert np.allclose(s1, s2)


def test_chunk_size_invariance_for_fit():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(500, 2))
    tbl = pa.table({f"x{i}": pa.array(X[:, i]) for i in range(2)})
    from cadw.dataset import Dataset
    import tempfile, pyarrow.parquet as pq
    path = tempfile.mktemp(suffix=".parquet")
    pq.write_table(tbl, path, row_group_size=1)
    df1 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1)
    df2 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1000)
    m1 = fit_isolation_forest(df1, ["x0", "x1"], n_trees=10,
                             subsample_size=128, random_state=7)
    m2 = fit_isolation_forest(df2, ["x0", "x1"], n_trees=10,
                             subsample_size=128, random_state=7)
    # Reservoir sample is order-dependent; compare score aggregates
    s1 = _scores_to_numpy(m1.score_samples(_df(X))); s2 = _scores_to_numpy(m2.score_samples(_df(X)))
    assert abs(np.nanmean(s1) - np.nanmean(s2)) < 0.1


def test_single_row_score_is_finite():
    X = np.array([[1.0, 2.0]])
    m = fit_isolation_forest(_df(np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])),
                            ["x0", "x1"], n_trees=5, subsample_size=3, random_state=0)
    s = _scores_to_numpy(m.score_samples(_df(X)))
    assert np.isfinite(s[0])


def test_constant_feature_no_crash():
    X = np.column_stack([np.zeros(100), np.arange(100, dtype=float)])
    m = fit_isolation_forest(_df(X), ["x0", "x1"], n_trees=10,
                            subsample_size=64, random_state=0)
    s = _scores_to_numpy(m.score_samples(_df(X)))
    assert np.all(np.isfinite(s))


def test_nulls_excluded_from_score():
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [2.0, 4.0], [3.0, 5.0]])
    m = fit_isolation_forest(_df(np.nan_to_num(X, nan=0.0)), ["x0", "x1"],
                            n_trees=5, subsample_size=3, random_state=0)
    s = _scores_to_numpy(m.score_samples(_df(X)))
    assert np.isnan(s[1])
    assert np.isfinite(s[0]) and np.isfinite(s[2]) and np.isfinite(s[3])


def test_predict_threshold_controls_count():
    """v0.9.95: predict returns CadwDataFrame (column "is_outlier")."""
    rng = np.random.RandomState(5)
    X = rng.normal(size=(1000, 2))
    m = fit_isolation_forest(_df(X), ["x0", "x1"], n_trees=20,
                            subsample_size=128, contamination=0.1,
                            random_state=0)
    t_loose = m.predict(_df(X), threshold=0.4)
    t_tight = m.predict(_df(X), threshold=0.7)
    # CadwDataFrame, extract bool column via iter
    n_loose = sum(
        sum(b.column("is_outlier").to_pylist())
        for b in t_loose._iter_filtered_batches())
    n_tight = sum(
        sum(b.column("is_outlier").to_pylist())
        for b in t_tight._iter_filtered_batches())
    assert n_loose >= n_tight


def test_ref_vs_sklearn_auroc():
    """On labelled synthetic anomalies AUROC >= 0.75 (paper claims 0.85+)."""
    from sklearn.ensemble import IsolationForest as SKIF
    rng = np.random.RandomState(6)
    n_norm, n_anom = 500, 20
    X_norm = rng.normal(0, 1, size=(n_norm, 2))
    X_anom = rng.uniform(5, 7, size=(n_anom, 2))
    X = np.vstack([X_norm, X_anom])
    y = np.array([0]*n_norm + [1]*n_anom)
    m = fit_isolation_forest(_df(X), ["x0", "x1"], n_trees=100,
                            subsample_size=256, random_state=0)
    s = _scores_to_numpy(m.score_samples(_df(X)))
    sk = SKIF(n_estimators=100, random_state=0, contamination="auto").fit(X)
    ss = -sk.score_samples(X)  # negate: sklearn returns lower = more anomalous
    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, _ = spearmanr(s, ss)
    assert rho >= 0.5, f"expected rho>=0.5 got {rho:.3f}"
    # AUROC on label
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y, s)
    assert auroc >= 0.85, f"AUROC {auroc:.3f} < 0.85"


def test_state_ram_independent_of_n():
    """State bytes must not grow with N."""
    rng = np.random.RandomState(7)
    X_small = rng.normal(size=(100, 2))
    X_large = rng.normal(size=(5000, 2))
    m_small = fit_isolation_forest(_df(X_small), ["x0", "x1"],
                                  n_trees=10, subsample_size=64, random_state=0)
    m_large = fit_isolation_forest(_df(X_large), ["x0", "x1"],
                                  n_trees=10, subsample_size=64, random_state=0)
    # state must be bounded by n_trees * 2 * subsample_size nodes * 40 B
    bound = 10 * 2 * 64 * 40  # = 51200
    assert m_large.state_nbytes() <= bound
    assert m_small.state_nbytes() <= bound


def test_non_numeric_column_raises():
    tbl = pa.table({"x0": pa.array([1.0, 2.0]), "s": pa.array(["a", "b"])})
    df = CadwDataFrame.from_table(tbl)
    with pytest.raises(TypeError):
        fit_isolation_forest(df, ["x0", "s"], n_trees=5, subsample_size=2)


def test_too_few_rows_raises():
    tbl = pa.table({"x0": pa.array([1.0])})
    df = CadwDataFrame.from_table(tbl)
    with pytest.raises(ValueError):
        fit_isolation_forest(df, ["x0"], n_trees=5, subsample_size=256)
