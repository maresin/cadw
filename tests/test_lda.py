"""Tests for D.6.6 online variational Bayes LDA (v0.9.41)."""
from __future__ import annotations
import inspect
import math
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import lda as lmod
from cadw.ml.lda import fit_lda, LdaModel, tokenize


TOPICS = {
    0: ["apple", "banana", "cherry", "grape"],
    1: ["engine", "brake", "wheel", "clutch"],
    2: ["guitar", "piano", "violin", "drum"],
}


def _docs(n=600, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 3, size=n)
    out = []
    for i in range(n):
        words = list(rng.choice(TOPICS[int(y[i])], size=6, replace=True))
        out.append(" ".join(words))
    return out, y


def _df(docs, extra=None):
    d = {"text": pa.array(docs)}
    if extra is not None:
        for k, v in extra.items():
            d[k] = pa.array(v)
    return CadwDataFrame.from_table(pa.table(d))


def _nmi(y, lab):
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(y, lab))


def test_tokenizer_contract():
    assert tokenize("Apple, BANANA! x") == ["apple", "banana"]
    assert tokenize(None) == []
    assert tokenize("") == []


def test_synthetic_recovers_topics():
    docs, y = _docs(seed=0)
    m = fit_lda(_df(docs), "text", n_topics=3, vocab_size=50, max_iter=6, seed=0)
    g = m.transform(_df(docs), text_col="text")
    assert g.shape == (len(docs), 3)
    assert _nmi(y, np.argmax(g, axis=1)) > 0.85


def test_reference_sklearn_lda_comparable():
    skl = pytest.importorskip("sklearn.decomposition")
    docs, y = _docs(n=400, seed=1)
    m = fit_lda(_df(docs), "text", n_topics=3, vocab_size=50, seed=0)
    g = m.transform(_df(docs), text_col="text")
    ours = _nmi(y, np.argmax(g, axis=1))
    # same bag-of-words, sklearn LDA reference
    V = {w: i for i, w in enumerate(m.vocab_)}
    X = np.zeros((len(docs), len(m.vocab_)))
    for i, t in enumerate(docs):
        for tok in tokenize(t):
            j = V.get(tok)
            if j is not None:
                X[i, j] += 1
    ref = skl.LatentDirichletAllocation(
        n_components=3, max_iter=20, learning_method="batch",
        random_state=0).fit_transform(X)
    theirs = _nmi(y, np.argmax(ref, axis=1))
    assert ours >= theirs * 0.9, (ours, theirs)


def test_components_are_distributions():
    docs, _ = _docs(n=200, seed=2)
    m = fit_lda(_df(docs), "text", n_topics=3, vocab_size=40, max_iter=2)
    assert np.all(m.components_ >= 0)
    assert np.allclose(m.components_.sum(axis=1), 1.0)
    assert m.components_.shape[0] == 3


def test_perplexity_finite_and_better_than_uniform():
    docs, _ = _docs(n=300, seed=3)
    m = fit_lda(_df(docs), "text", n_topics=3, vocab_size=40, max_iter=5)
    p = m.perplexity(_df(docs), text_col="text")
    assert math.isfinite(p) and p > 0
    # A fitted model must beat the uniform-topic model: predict every word
    # with the vocabulary's marginal frequency instead of via topics.
    # Uniform-over-vocabulary prediction has perplexity exactly V.
    V = len(m.vocab_)
    assert p < V, (p, V)


def test_determinism():
    docs, _ = _docs(n=200, seed=4)
    a = fit_lda(_df(docs), "text", n_topics=3, vocab_size=40, max_iter=3, seed=0)
    b = fit_lda(_df(docs), "text", n_topics=3, vocab_size=40, max_iter=3, seed=0)
    assert np.array_equal(a.components_, b.components_)
    assert a.vocab_ == b.vocab_


def test_edge_guards():
    docs, _ = _docs(n=100, seed=5)
    with pytest.raises(ValueError, match="n_topics"):
        fit_lda(_df(docs), "text", n_topics=0)
    with pytest.raises(ValueError, match="vocab_size"):
        fit_lda(_df(docs), "text", vocab_size=1)
    with pytest.raises(ValueError, match="max_iter"):
        fit_lda(_df(docs), "text", max_iter=0)
    with pytest.raises(KeyError):
        fit_lda(_df(docs), "nope")
    with pytest.raises(ValueError, match="fewer than 2"):
        fit_lda(_df(["x"] * 10), "text")


def test_ram_guard_raises_when_tight():
    docs, _ = _docs(n=300, seed=6)
    mg = MemoryGuard(max_memory_bytes=1024)
    with pytest.raises(MemoryLimitExceeded):
        fit_lda(_df(docs), "text", n_topics=3, memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    docs, _ = _docs(n=300, seed=7)
    df = _df(docs)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "t.parquet")
    pq.write_table(tbl, path, row_group_size=47)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=64)
    a = fit_lda(df, "text", n_topics=3, vocab_size=40, max_iter=3)
    b = fit_lda(dmb, "text", n_topics=3, vocab_size=40, max_iter=3)
    assert np.allclose(a.components_, b.components_, atol=1e-9)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(lmod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src

def test_lda_e_step_sparse_matches_dense():
    """v0.9.87-P5a-fix-lda-a: sparse E-step bit-identical to dense.

    Independent reference implementation (dense, small V) vs production
    _e_step dispatched via CSR. Zero positions contribute exact 0, so
    fixed-point sum order matches (CSR indices sorted).
    """
    from scipy.sparse import csr_matrix
    from cadw.ml.lda import LdaModel, _digamma

    rng = np.random.RandomState(42)
    D, V, K = 20, 50, 3
    # Sparse-ish counts (70% zero)
    C_dense = rng.poisson(lam=1.2, size=(D, V)).astype(np.float64)
    C_dense = C_dense * (rng.rand(D, V) < 0.7)
    beta = rng.dirichlet(np.ones(V), size=K)
    beta /= beta.sum(axis=1, keepdims=True)
    alpha = 0.1

    # Reference dense E-step (explicit, matches original v0.9.42 math)
    Elogbeta_ref = _digamma(beta) - _digamma(beta.sum(axis=1))[:, None]
    gamma_ref = np.full((D, K), alpha + C_dense.sum(axis=1)[:, None] / K)
    for _ in range(20):
        Elogtheta = _digamma(gamma_ref) - _digamma(
            gamma_ref.sum(axis=1))[:, None]
        logp = Elogtheta[:, None, :] + Elogbeta_ref.T[None, :, :]
        logp -= logp.max(axis=2, keepdims=True)
        p = np.exp(logp)
        p /= p.sum(axis=2, keepdims=True)
        new = alpha + (C_dense[:, :, None] * p).sum(axis=1)
        delta = np.max(np.abs(new - gamma_ref))
        gamma_ref = new
        if delta < 1e-4:
            break

    model = LdaModel(beta, [], {"alpha": alpha})
    gamma_sp, _ = model._e_step(csr_matrix(C_dense), alpha)
    assert np.allclose(gamma_sp, gamma_ref, atol=1e-12), (
        f"sparse vs dense E-step: max diff "
        f"{np.max(np.abs(gamma_sp - gamma_ref))}")

def test_lda_large_vocab_no_oom():
    """v0.9.87-P5a-fix-lda-a: true large-V fit (no silent OOM).

    Corpus: 200 docs x 250 unique words each = 50000 unique tokens
    total -> V=50000 (not capped; vocab_size=50000 matches actual).
    D per batch = 200 (single-batch df).

    Old dense path: phi = (D, V, K) = 200 * 50000 * 10 * 8 = 800 MB
    per E-step call. Guard saw only batch text (~50 KB), no raise
    -> silent 800 MB alloc.

    New sparse path: phi per doc = nnz * K * 8 = 250 * 10 * 8 = 20 KB.
    Whole fit under 100 MB guard.
    """
    import psutil, os

    n_docs = 200
    words_per_doc = 250  # 200 * 250 = 50000 unique
    docs = [" ".join(f"w{i*words_per_doc + j}"
                     for j in range(words_per_doc))
            for i in range(n_docs)]
    df = _df(docs)

    mg = MemoryGuard(max_memory_mb=100)
    proc = psutil.Process(os.getpid())
    base = proc.memory_info().rss

    m = fit_lda(df, "text", n_topics=10, vocab_size=50000,
                max_iter=2, n_restarts=1, seed=0, memory_guard=mg)
    peak_delta_mb = (proc.memory_info().rss - base) / 1e6

    assert m.fitted_
    # V is actual unique count (50000) — vocab_size cap does not
    # artificially inflate; corpus spans exactly 50000 tokens.
    assert m.components_.shape == (10, 50000), (
        f"expected (10, 50000), got {m.components_.shape}; "
        f"corpus should have 50000 unique words")
    # Dense phi (800 MB) would blow this; sparse path ~100-150 MB.
    assert peak_delta_mb < 300, (
        f"peak +{peak_delta_mb:.0f} MB; dense phi would be ~800 MB, "
        f"sparse should be under 200 MB")

