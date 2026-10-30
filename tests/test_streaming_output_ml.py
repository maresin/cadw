"""S4: output= on 9 ML predict/transform/score functions."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.core.spill import (
    AutoArrayResult, AutoResult, InMemoryArrayResult, InMemoryResult,
    OnDiskArrayResult, OnDiskResult,
)



def _scores_col_to_numpy(df, col="score"):
    """Extract column from producer CadwDataFrame as 1-D numpy.

    v0.9.95: producers return CadwDataFrame; old tests assumed
    np.ndarray / AutoArrayResult. Helper bridges for numeric checks.
    """
    import numpy as np
    parts = [b.column(col).to_numpy(zero_copy_only=False)
             for b in df._iter_filtered_batches()]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _mk_df(cols_data):
    return CadwDataFrame.from_table(pa.table(cols_data))


def _iforest_model(n=300, seed=0):
    from cadw.ml.iforest import fit_isolation_forest
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    df = _mk_df({"x0": pa.array(X[:, 0]), "x1": pa.array(X[:, 1])})
    return fit_isolation_forest(df, ["x0", "x1"], n_trees=10,
                                subsample_size=64, random_state=0), df


def _chains_model(n=200, seed=0):
    from cadw.ml.chains import fit_chains
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    a = (X[:, 0] > 0).astype(float)
    b = (rng.uniform(size=n) < 0.8).astype(float) * a + (1 - a) * (rng.uniform(size=n) < 0.2)
    b = b.astype(float)
    df = _mk_df({"x0": pa.array(X[:, 0]), "x1": pa.array(X[:, 1]),
                 "a": pa.array(a), "b": pa.array(b)})
    m = fit_chains(df, ["x0", "x1"], ["a", "b"], chain=True,
                   n_estimators=5)
    return m, df


def _ordinal_model(n=200, seed=0):
    from cadw.ml.ordinal import fit_ordinal_logit
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = np.digitize(X[:, 0] + 0.3 * rng.normal(size=n),
                    [-0.5, 0.5, 1.5]).astype(float)
    df = _mk_df({"x0": pa.array(X[:, 0]), "x1": pa.array(X[:, 1]),
                 "y": pa.array(y)})
    m = fit_ordinal_logit(df, ["x0", "x1"], "y")
    return m, df


def _tfidf_model(n=50, seed=0):
    from cadw.ml.tfidf import fit_tfidf
    rng = np.random.RandomState(seed)
    vocab = ["apple", "banana", "cherry", "dog", "engine"]
    docs = [" ".join(rng.choice(vocab, size=4, replace=True)) for _ in range(n)]
    df = _mk_df({"text": pa.array(docs)})
    m = fit_tfidf(df, "text", vocab_size=50)
    return m, df


def _randsvd_model(n=100, seed=0, d=5):
    from cadw.ml.randsvd import fit_randomized_svd
    rng = np.random.RandomState(seed)
    Z = rng.normal(size=(n, 2))
    X = Z @ rng.normal(size=(2, d)) + 0.05 * rng.normal(size=(n, d))
    cols = {f"x{j}": pa.array(X[:, j]) for j in range(d)}
    df = _mk_df(cols)
    m = fit_randomized_svd(df, [f"x{j}" for j in range(d)], n_components=2)
    return m, df


def _gmm_model(n=200, seed=0):
    from cadw.ml.gmm import fit_gmm
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    df = _mk_df({"x0": pa.array(X[:, 0]), "x1": pa.array(X[:, 1])})
    m = fit_gmm(df, ["x0", "x1"], n_components=2, random_state=0,
                max_samples=200)
    return m, df


# ─── iforest ──────────────────────────────────────────────────────

def test_iforest_auto_small_inmemory():
    """v0.9.95: auto returns CadwDataFrame; small data → in-memory
    backing (_InMemoryDataset), no spill."""
    from cadw import CadwDataFrame
    from cadw.dataset._in_memory import _InMemoryDataset
    m, df = _iforest_model()
    r = m.score_samples(df, output="auto")
    assert isinstance(r, CadwDataFrame)
    assert isinstance(r._dataset, _InMemoryDataset)  # in-memory, no spill
    assert r.count() == df.count()
    assert r.columns == ["score"]


def test_iforest_disk_spills():
    """v0.9.95: disk returns CadwDataFrame with disk-backed Dataset
    (forced spill)."""
    from cadw import CadwDataFrame
    from cadw.dataset._in_memory import _InMemoryDataset
    m, df = _iforest_model()
    r = m.score_samples(df, output="disk")
    assert isinstance(r, CadwDataFrame)
    assert not isinstance(r._dataset, _InMemoryDataset)  # disk-backed
    assert r.count() == df.count()
    # Values preserved through spill round-trip
    vals = _scores_col_to_numpy(r, "score")
    assert vals.shape == (df.count(),)


def test_iforest_auto_equals_memory():
    """v0.9.95: both output= modes return CadwDataFrame with same
    values; difference is only storage strategy."""
    m, df = _iforest_model()
    mem = m.score_samples(df, output="memory")
    auto = m.score_samples(df, output="auto")
    mem_vals = _scores_col_to_numpy(mem, "score")
    auto_vals = _scores_col_to_numpy(auto, "score")
    np.testing.assert_array_equal(mem_vals, auto_vals)


# ─── chains ───────────────────────────────────────────────────────

def test_chains_proba_auto_disk():
    """v0.9.95: unified CadwDataFrame."""
    from cadw import CadwDataFrame
    from cadw.dataset._in_memory import _InMemoryDataset
    m, df = _chains_model()
    a = m.predict_proba(df, output="auto")
    assert isinstance(a, CadwDataFrame)
    K = len(a.columns)
    assert K >= 1 and all(c.startswith("prob_") for c in a.columns)
    d = m.predict_proba(df, output="disk")
    assert isinstance(d, CadwDataFrame)
    assert not isinstance(d._dataset, _InMemoryDataset)


def test_chains_predict_auto_disk():
    from cadw import CadwDataFrame
    from cadw.dataset._in_memory import _InMemoryDataset
    m, df = _chains_model()
    a = m.predict(df, output="auto")
    assert isinstance(a, CadwDataFrame)
    assert all(c.startswith("pred_") for c in a.columns)
    d = m.predict(df, output="disk")
    assert isinstance(d, CadwDataFrame)
    assert not isinstance(d._dataset, _InMemoryDataset)


def test_chains_auto_equals_memory():
    """v0.9.95: extract per-column via _iter_filtered_batches."""
    m, df = _chains_model()
    mem_df = m.predict_proba(df, output="memory")
    auto_df = m.predict_proba(df, output="auto")
    for col in mem_df.columns:
        mem = _scores_col_to_numpy(mem_df, col)
        got = _scores_col_to_numpy(auto_df, col)
        np.testing.assert_allclose(mem, got)


# ─── ordinal ──────────────────────────────────────────────────────

def test_ordinal_proba_auto_disk():
    """v0.9.95: unified CadwDataFrame."""
    from cadw import CadwDataFrame
    from cadw.dataset._in_memory import _InMemoryDataset
    m, df = _ordinal_model()
    a = m.predict_proba(df, output="auto")
    assert isinstance(a, CadwDataFrame)
    assert all(c.startswith("prob_") for c in a.columns)
    d = m.predict_proba(df, output="disk")
    assert isinstance(d, CadwDataFrame)
    assert not isinstance(d._dataset, _InMemoryDataset)


def test_ordinal_predict_auto_disk():
    from cadw import CadwDataFrame
    m, df = _ordinal_model()
    a = m.predict(df, output="auto")
    assert isinstance(a, CadwDataFrame)
    assert a.columns == ["pred"]
    mem_df = m.predict(df, output="memory")
    assert isinstance(mem_df, CadwDataFrame)
    mem_vals = _scores_col_to_numpy(mem_df, "pred")
    got_vals = _scores_col_to_numpy(a, "pred")
    np.testing.assert_array_equal(mem_vals, got_vals)


# ─── tfidf ────────────────────────────────────────────────────────

def test_tfidf_dense_auto_disk():
    m, df = _tfidf_model()
    a = m.transform_dense(df, text_col="text", output="auto")
    assert isinstance(a, AutoArrayResult)
    d = m.transform_dense(df, text_col="text", output="disk")
    assert isinstance(d, OnDiskArrayResult)
    d.close()


def test_tfidf_dense_auto_equals_memory():
    m, df = _tfidf_model()
    mem = m.transform_dense(df, text_col="text", output="memory")
    auto = m.transform_dense(df, text_col="text", output="auto")
    got = np.concatenate([a for a in auto], axis=0)
    np.testing.assert_allclose(mem, got)


# ─── randsvd ──────────────────────────────────────────────────────

def test_randsvd_auto_disk():
    m, df = _randsvd_model()
    a = m.transform(df, output="auto")
    assert isinstance(a, AutoArrayResult)
    d = m.transform(df, output="disk")
    assert isinstance(d, OnDiskArrayResult)
    d.close()


def test_randsvd_auto_equals_memory():
    m, df = _randsvd_model()
    mem = m.transform(df, output="memory")
    auto = m.transform(df, output="auto")
    got = np.concatenate([a for a in auto], axis=0)
    np.testing.assert_allclose(mem, got)


# ─── gmm (pa.Table path) ─────────────────────────────────────────

def test_gmm_score_auto_disk():
    """v0.9.95: unified CadwDataFrame; output= is storage only."""
    from cadw import CadwDataFrame
    from cadw.dataset._in_memory import _InMemoryDataset
    m, df = _gmm_model()
    a = m.score_samples(df, output="auto")
    assert isinstance(a, CadwDataFrame)
    assert a.columns == ["log_likelihood"]
    d = m.score_samples(df, output="disk")
    assert isinstance(d, CadwDataFrame)
    assert not isinstance(d._dataset, _InMemoryDataset)  # disk-backed


def test_gmm_predict_auto_disk():
    """v0.9.95: unified CadwDataFrame."""
    from cadw import CadwDataFrame
    from cadw.dataset._in_memory import _InMemoryDataset
    m, df = _gmm_model()
    a = m.predict(df, output="auto")
    assert isinstance(a, CadwDataFrame)
    assert a.columns == ["cluster"]
    d = m.predict(df, output="disk")
    assert isinstance(d, CadwDataFrame)
    assert not isinstance(d._dataset, _InMemoryDataset)


def test_gmm_auto_equals_memory():
    """v0.9.95: both modes CadwDataFrame; extract column via iter."""
    m, df = _gmm_model()
    mem_df = m.score_samples(df, output="memory")
    auto_df = m.score_samples(df, output="auto")
    mem = _scores_col_to_numpy(mem_df, "log_likelihood")
    got = _scores_col_to_numpy(auto_df, "log_likelihood")
    np.testing.assert_allclose(mem, got, equal_nan=True)


# ─── output= validation ─────────────────────────────────────────

def test_output_invalid_iforest():
    m, df = _iforest_model()
    with pytest.raises(ValueError, match="output must be"):
        m.score_samples(df, output="garbage")


def test_output_invalid_chains():
    m, df = _chains_model()
    with pytest.raises(ValueError, match="output must be"):
        m.predict_proba(df, output="garbage")
    with pytest.raises(ValueError, match="output must be"):
        m.predict(df, output="garbage")


def test_output_invalid_ordinal():
    m, df = _ordinal_model()
    with pytest.raises(ValueError, match="output must be"):
        m.predict_proba(df, output="garbage")
    with pytest.raises(ValueError, match="output must be"):
        m.predict(df, output="garbage")


def test_output_invalid_tfidf():
    m, df = _tfidf_model()
    with pytest.raises(ValueError, match="output must be"):
        m.transform_dense(df, text_col="text", output="garbage")


def test_output_invalid_randsvd():
    m, df = _randsvd_model()
    with pytest.raises(ValueError, match="output must be"):
        m.transform(df, output="garbage")


def test_output_invalid_gmm():
    m, df = _gmm_model()
    with pytest.raises(ValueError, match="output must be"):
        m.score_samples(df, output="garbage")
    with pytest.raises(ValueError, match="output must be"):
        m.predict(df, output="garbage")
