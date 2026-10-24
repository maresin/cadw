"""Tests for D.6.9 streaming TF-IDF (v0.9.41)."""
from __future__ import annotations
import inspect
import math
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import tfidf as tmod
from cadw.ml.tfidf import fit_tfidf, TfidfModel


def _corpus(n=200, seed=0):
    rng = np.random.RandomState(seed)
    vocab = ["apple", "banana", "cherry", "dog", "engine", "field"]
    docs = []
    for _ in range(n):
        k = rng.randint(3, 7)
        docs.append(" ".join(rng.choice(vocab, size=k, replace=True)))
    return docs


def _df(docs):
    return CadwDataFrame.from_table(pa.table({"text": pa.array(docs)}))


def test_reference_sklearn_tfidf_exact():
    """Our TF-IDF must match sklearn's smoothed idf and l2 norm."""
    sk = pytest.importorskip("sklearn.feature_extraction.text")
    docs = _corpus(seed=0)
    m = fit_tfidf(_df(docs), "text", vocab_size=50, sublinear_tf=True, norm="l2")
    ours = m.transform_dense(_df(docs), text_col="text", norm="l2")
    # sklearn with the SAME tokenizer and the same vocabulary
    vec = sk.TfidfVectorizer(
        tokenizer=lambda s: [t for t in s.lower().split() if len(t) > 1],
        token_pattern=None, vocabulary=m.vocab_, sublinear_tf=True,
        norm="l2", smooth_idf=True, use_idf=True)
    X = vec.fit_transform(docs).toarray()
    assert ours.shape == X.shape
    assert np.allclose(ours, X, atol=1e-9), np.max(np.abs(ours - X))


def test_idf_formula_smoothed():
    docs = _corpus(n=100, seed=1)
    m = fit_tfidf(_df(docs), "text", vocab_size=50)
    n = len(docs)
    dfv = {w: sum(1 for d in docs if w in d.split()) for w in m.vocab_}
    for w, j in zip(m.vocab_, range(len(m.vocab_))):
        ref = math.log((1.0 + n) / (1.0 + dfv[w])) + 1.0
        assert abs(m.idf_[j] - ref) < 1e-9


def test_rare_words_get_higher_idf():
    docs = _corpus(seed=2)
    m = fit_tfidf(_df(docs), "text", vocab_size=50)
    j = {w: i for i, w in enumerate(m.vocab_)}
    dfa = sum(1 for d in docs if "apple" in d.split())
    dfd = sum(1 for d in docs if "dog" in d.split())
    if dfa != dfd:
        rarer = "apple" if dfa < dfd else "dog"
        common = "dog" if rarer == "apple" else "apple"
        assert m.idf_[j[rarer]] > m.idf_[j[common]]


def test_sublinear_tf_vs_linear():
    docs = ["apple apple apple banana", "banana banana cherry"]
    a = fit_tfidf(_df(docs), "text", vocab_size=10, sublinear_tf=True, norm=None)
    b = fit_tfidf(_df(docs), "text", vocab_size=10, sublinear_tf=False, norm=None)
    ja = a._index["apple"]
    va = a.transform_dense(_df(docs), text_col="text", norm=None)[0, ja]
    vb = b.transform_dense(_df(docs), text_col="text", norm=None)[0, a._index["apple"]]
    assert abs(va - (1.0 + math.log(3.0)) * a.idf_[ja]) < 1e-9
    assert abs(vb - 3.0 * a.idf_[ja]) < 1e-9


def test_l2_norm_is_unit():
    docs = _corpus(seed=3)
    m = fit_tfidf(_df(docs), "text", vocab_size=50, norm="l2")
    X = m.transform_dense(_df(docs), text_col="text", norm="l2")
    nz = X[np.any(X != 0, axis=1)]
    assert np.allclose(np.linalg.norm(nz, axis=1), 1.0, atol=1e-9)


def test_min_df_filters():
    docs = ["rare one", "common frequent", "common frequent",
            "common frequent"]
    m = fit_tfidf(_df(docs), "text", vocab_size=50, min_df=2)
    assert "common" in m.vocab_ and "frequent" in m.vocab_
    assert "rare" not in m.vocab_
    assert "one" not in m.vocab_


def test_determinism():
    docs = _corpus(seed=4)
    a = fit_tfidf(_df(docs), "text", vocab_size=50)
    b = fit_tfidf(_df(docs), "text", vocab_size=50)
    assert a.vocab_ == b.vocab_
    assert np.array_equal(a.idf_, b.idf_)


def test_edge_guards():
    docs = _corpus(n=50, seed=5)
    with pytest.raises(ValueError, match="vocab_size"):
        fit_tfidf(_df(docs), "text", vocab_size=1)
    with pytest.raises(ValueError, match="min_df"):
        fit_tfidf(_df(docs), "text", min_df=0)
    with pytest.raises(ValueError, match="norm"):
        fit_tfidf(_df(docs), "text", norm="l3")
    with pytest.raises(KeyError):
        fit_tfidf(_df(docs), "nope")
    with pytest.raises(ValueError, match="fewer than 2"):
        fit_tfidf(_df(["x"] * 5), "text")


def test_ram_guard_raises_when_tight():
    docs = _corpus(n=100, seed=6)
    mg = MemoryGuard(max_memory_bytes=1024)
    with pytest.raises(MemoryLimitExceeded):
        fit_tfidf(_df(docs), "text", memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    docs = _corpus(n=300, seed=7)
    df = _df(docs)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "tf.parquet")
    pq.write_table(tbl, path, row_group_size=41)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=64)
    a = fit_tfidf(df, "text", vocab_size=50)
    b = fit_tfidf(dmb, "text", vocab_size=50)
    assert a.vocab_ == b.vocab_
    assert np.allclose(a.idf_, b.idf_, atol=1e-9)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(tmod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
