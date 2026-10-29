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
    m, df = _iforest_model()
    r = m.score_samples(df, output="auto")
    assert isinstance(r, AutoArrayResult)
    assert isinstance(r, InMemoryArrayResult)
    assert len(r) == df.count()


def test_iforest_disk_spills():
    m, df = _iforest_model()
    r = m.score_samples(df, output="disk")
    assert isinstance(r, OnDiskArrayResult)
    assert r.is_spilled
    with pytest.raises(MemoryLimitExceeded):
        r.to_numpy()
    r.close()


def test_iforest_auto_equals_memory():
    m, df = _iforest_model()
    mem = m.score_samples(df, output="memory")
    auto = m.score_samples(df, output="auto")
    got = np.concatenate([a for a in auto], axis=0).ravel()
    np.testing.assert_array_equal(mem, got)


# ─── chains ───────────────────────────────────────────────────────

def test_chains_proba_auto_disk():
    m, df = _chains_model()
    a = m.predict_proba(df, output="auto")
    assert isinstance(a, AutoArrayResult)
    d = m.predict_proba(df, output="disk")
    assert isinstance(d, OnDiskArrayResult)
    d.close()


def test_chains_predict_auto_disk():
    m, df = _chains_model()
    a = m.predict(df, output="auto")
    assert isinstance(a, AutoArrayResult)
    d = m.predict(df, output="disk")
    assert isinstance(d, OnDiskArrayResult)
    d.close()


def test_chains_auto_equals_memory():
    m, df = _chains_model()
    mem = m.predict_proba(df, output="memory")
    auto = m.predict_proba(df, output="auto")
    got = np.concatenate([a for a in auto], axis=0)
    np.testing.assert_allclose(mem, got)


# ─── ordinal ──────────────────────────────────────────────────────

def test_ordinal_proba_auto_disk():
    m, df = _ordinal_model()
    a = m.predict_proba(df, output="auto")
    assert isinstance(a, AutoArrayResult)
    d = m.predict_proba(df, output="disk")
    assert isinstance(d, OnDiskArrayResult)
    d.close()


def test_ordinal_predict_auto_disk():
    m, df = _ordinal_model()
    a = m.predict(df, output="auto")
    assert isinstance(a, AutoArrayResult)
    # verify class values match memory
    mem = m.predict(df, output="memory")
    got = np.concatenate([a for a in a], axis=0).ravel()
    np.testing.assert_array_equal(mem, got)


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
    m, df = _gmm_model()
    a = m.score_samples(df, output="auto")
    assert isinstance(a, AutoResult)
    d = m.score_samples(df, output="disk")
    assert isinstance(d, OnDiskResult)
    with pytest.raises(MemoryLimitExceeded):
        d.to_table()
    d.close()


def test_gmm_predict_auto_disk():
    m, df = _gmm_model()
    a = m.predict(df, output="auto")
    assert isinstance(a, AutoResult)
    d = m.predict(df, output="disk")
    assert isinstance(d, OnDiskResult)
    d.close()


def test_gmm_auto_equals_memory():
    m, df = _gmm_model()
    mem = m.score_samples(df, output="memory").column("log_likelihood").to_pylist()
    auto = m.score_samples(df, output="auto")
    got = [r["log_likelihood"] for b in auto for r in b.to_pylist()]
    np.testing.assert_allclose(np.array(mem), np.array(got),
                                equal_nan=True)


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
