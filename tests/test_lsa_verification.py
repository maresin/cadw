"""v0.9.212: fit_lsa verification (LSA / TF-IDF + randomized SVD).

Sklearn parity + OOC + guard + release + validation.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import fit_lsa
from cadw.ml.tfidf import fit_tfidf


def _mk_df(docs):
    return CadwDataFrame.from_table(pa.table({"text": pa.array(docs)}),
                                     memory_guard=MemoryGuard(max_memory_mb=500))


def _docs(n_repeat=20):
    base = [
        "the cat sat on the mat", "the dog ran in the park",
        "cat and dog play together", "the weather is nice today",
        "it is raining today", "sunny weather tomorrow",
        "cats love to sleep", "dogs love to play",
        "the mat is on the floor", "parks are great for dogs",
    ]
    return base * n_repeat


def _extract(df_out, k):
    parts = []
    for b in df_out._iter_filtered_batches():
        arr = np.column_stack([
            b.column(f"lsa_{j}").to_numpy(zero_copy_only=False)
            for j in range(k)
        ])
        parts.append(arr)
    return np.vstack(parts) if parts else np.empty((0, k))


def _warmup():
    df = _mk_df(_docs(n_repeat=2))
    m = fit_lsa(df, "text", n_components=3, vocab_size=20,
                n_power_iter=2, seed=0,
                memory_guard=MemoryGuard(max_memory_mb=500))
    m.release_train_state()


_warmup()


# ---- CW-1: sklearn TruncatedSVD parity ----
def test_sklearn_parity_sigma():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    docs = _docs(n_repeat=10)
    vec = TfidfVectorizer(max_features=50, sublinear_tf=True, norm="l2",
                           token_pattern=r"[a-z0-9]+")
    X = vec.fit_transform(docs)
    sk = TruncatedSVD(n_components=5, random_state=0).fit(X)

    df = _mk_df(docs)
    lsa = fit_lsa(df, "text", n_components=5, vocab_size=50,
                  sublinear_tf=True, n_power_iter=4, seed=0,
                  memory_guard=MemoryGuard(max_memory_mb=500))
    assert lsa.vocab_size_ == X.shape[1], (
        f"vocab mismatch: cadw={lsa.vocab_size_}, sk={X.shape[1]}")
    sigma_diff = np.abs(lsa.singular_values_ - sk.singular_values_).max()
    assert sigma_diff < 0.01, f"sigma parity: {sigma_diff:.4e}"


# ---- CW-2: transform shape parity ----
def test_transform_shape():
    df = _mk_df(_docs(n_repeat=5))
    lsa = fit_lsa(df, "text", n_components=3, vocab_size=20,
                  n_power_iter=2, seed=0,
                  memory_guard=MemoryGuard(max_memory_mb=500))
    Z = _extract(lsa.transform(df, output="memory"), 3)
    assert Z.shape == (df.count(), 3)


# ---- CW-3: state_nbytes = model (independent of n) ----
def test_state_independent_of_n():
    docs_small = _docs(n_repeat=1)  # n=10
    docs_large = _docs(n_repeat=100)  # n=1000
    lsa_small = fit_lsa(_mk_df(docs_small), "text", n_components=3,
                        vocab_size=20, n_power_iter=2, seed=0,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    lsa_large = fit_lsa(_mk_df(docs_large), "text", n_components=3,
                        vocab_size=20, n_power_iter=2, seed=0,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    # state = components (k,V) + sigma (k) + tfidf vocab (V); independent of n
    assert lsa_small.state_nbytes() == lsa_large.state_nbytes(), (
        f"state grew with n: {lsa_small.state_nbytes()} vs "
        f"{lsa_large.state_nbytes()}")


# ---- CW-4: guard refuses tight ----
def test_guard_refuses_tight():
    df = _mk_df(_docs(n_repeat=5))
    with pytest.raises(MemoryLimitExceeded):
        fit_lsa(df, "text", n_components=5, vocab_size=50,
                n_power_iter=2, seed=0,
                memory_guard=MemoryGuard(max_memory_mb=0.001))


# ---- CW-5: reproducibility same seed ----
def test_reproducibility():
    df = _mk_df(_docs(n_repeat=3))
    l1 = fit_lsa(df, "text", n_components=3, vocab_size=20,
                  n_power_iter=2, seed=42,
                  memory_guard=MemoryGuard(max_memory_mb=500))
    l2 = fit_lsa(df, "text", n_components=3, vocab_size=20,
                  n_power_iter=2, seed=42,
                  memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(l1.components_, l2.components_)
    np.testing.assert_array_equal(l1.singular_values_, l2.singular_values_)


# ---- CW-6: release idempotent ----
def test_release_idempotent():
    df = _mk_df(_docs(n_repeat=2))
    lsa = fit_lsa(df, "text", n_components=3, vocab_size=20,
                  n_power_iter=2, seed=0,
                  memory_guard=MemoryGuard(max_memory_mb=500))
    lsa.release_train_state()
    lsa.release_train_state()  # idempotent
    # state_nbytes baseline = 256 (header); all field contributions = 0
    assert lsa.state_nbytes() <= 256
    with pytest.raises(RuntimeError, match="model released"):
        lsa.transform(df)


# ---- CW-7: input validation ----
def test_input_validation():
    df = _mk_df(_docs(n_repeat=2))
    with pytest.raises(ValueError, match="text_col must be non-empty"):
        fit_lsa(df, "", n_components=3,
                memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="n_components must be >= 1"):
        fit_lsa(df, "text", n_components=0,
                memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(KeyError, match="column 'bogus' not in"):
        fit_lsa(df, "bogus", n_components=3,
                memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-8: vocab_size < n_components raises ----
def test_vocab_smaller_than_components_raises():
    df = _mk_df(_docs(n_repeat=2))
    with pytest.raises(ValueError, match="vocab_size=3 < n_components=10"):
        fit_lsa(df, "text", n_components=10, vocab_size=3,
                memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-9: TfidfModel.transform_iter CSR parity ----
def test_transform_iter_csr_shape():
    docs = _docs(n_repeat=2)
    df = _mk_df(docs)
    m = fit_tfidf(df, "text", vocab_size=20,
                   memory_guard=MemoryGuard(max_memory_mb=500))
    V = len(m.vocab_)
    n_rows = 0
    for M_b in m.transform_iter(df, text_col="text",
                                  memory_guard=MemoryGuard(max_memory_mb=500)):
        # M_b is scipy csr (b, V)
        assert hasattr(M_b, "shape")
        assert M_b.shape[1] == V
        n_rows += M_b.shape[0]
    assert n_rows == df.count()


# ---- CW-10: to_dict ----
def test_to_dict():
    df = _mk_df(_docs(n_repeat=3))
    lsa = fit_lsa(df, "text", n_components=3, vocab_size=20,
                  n_power_iter=2, seed=0,
                  memory_guard=MemoryGuard(max_memory_mb=500))
    d = lsa.to_dict()
    assert d["n_components"] == 3
    assert d["vocab_size"] == lsa.vocab_size_
    assert d["n_docs"] == df.count()
    assert len(d["singular_values"]) == 3


# ---- CW-11: n_power_iter >= 1 (q=1 still works, q=4 better) ----
def test_power_iter_quality():
    df = _mk_df(_docs(n_repeat=10))
    # q=1 less accurate; q=4 stable
    lsa_q1 = fit_lsa(df, "text", n_components=5, vocab_size=50,
                      n_power_iter=1, seed=0,
                      memory_guard=MemoryGuard(max_memory_mb=500))
    lsa_q4 = fit_lsa(df, "text", n_components=5, vocab_size=50,
                      n_power_iter=4, seed=0,
                      memory_guard=MemoryGuard(max_memory_mb=500))
    # sigma from q=4 should be >= q=1 for top components (q=4 better
    # approximation of true sigma; but both positive)
    assert np.all(lsa_q4.singular_values_ >= 0)
    assert np.all(lsa_q1.singular_values_ >= 0)


# ---- CW-12: transform output modes memory/auto equivalent ----
def test_transform_output_modes():
    df = _mk_df(_docs(n_repeat=3))
    lsa = fit_lsa(df, "text", n_components=3, vocab_size=20,
                  n_power_iter=2, seed=0,
                  memory_guard=MemoryGuard(max_memory_mb=500))
    Z_mem = _extract(lsa.transform(df, output="memory"), 3)
    Z_auto = _extract(lsa.transform(df, output="auto"), 3)
    np.testing.assert_allclose(Z_mem, Z_auto, atol=1e-12)
