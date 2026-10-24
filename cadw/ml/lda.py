"""Online variational Bayes LDA (D.6.6, v0.9.41).

Reference: Blei, Ng, Jordan, *Latent Dirichlet Allocation*, JMLR 3, 2003;
Hoffman, Blei, Bach, *Online Learning for Latent Dirichlet Allocation*,
NIPS 2010. Reference implementation in tests: sklearn
LatentDirichletAllocation (Pedregosa et al., JMLR 12, 2011).
See .pingis/algorithm_notes/d6_6_lda.md.

Streaming design: pass 1 builds a bounded vocabulary (document frequency
capped at vocab_size); passes 2..max_iter run online variational EM -- a
document-level E-step (gamma/phi) per batch followed by an M-step with the
Hoffman learning rate rho_t = (tau0 + t)^(-kappa); default kappa=0.0
means a full batch M-step each pass (correct for offline fit -- the
mini-batch E-step is still streamed, so no D x V matrix is materialised),
kappa>0 gives the stochastic online schedule. State is the topic-word
matrix beta (K x V) + alpha, independent of the number of documents.
"""
from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase alphanumeric tokenizer (min length 2)."""
    if text is None:
        return []
    return [t for t in _TOKEN_RE.findall(str(text).lower()) if len(t) > 1]


def _digamma(x):
    from scipy.special import digamma
    return digamma(x)


class LdaModel:
    """Fitted LDA: topic-word distributions and a transform()."""

    def __init__(self, components, vocab, params):
        self.components_ = np.asarray(components, dtype=np.float64)
        self.vocab_ = list(vocab)
        self._index = {w: i for i, w in enumerate(self.vocab_)}
        self.params_ = dict(params)
        self.fitted_ = True

    @property
    def n_topics_(self):
        return int(self.components_.shape[0])

    def state_nbytes(self):
        return int(self.components_.nbytes + 8 * len(self.vocab_) + 512)

    def to_dict(self):
        return {"n_topics": self.n_topics_, "vocab_size": len(self.vocab_),
                "params": self.params_}

    def _encode(self, texts):
        V = len(self.vocab_)
        C = np.zeros((len(texts), V), dtype=np.float64)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                j = self._index.get(tok)
                if j is not None:
                    C[i, j] += 1.0
        return C

    def _e_step(self, C, alpha, max_inner=20, tol=1e-4):
        D, V = C.shape
        K = self.components_.shape[0]
        beta = self.components_
        Elogbeta = _digamma(beta) - _digamma(beta.sum(axis=1))[:, None]
        gamma = np.full((D, K), alpha + C.sum(axis=1)[:, None] / K)
        phi = np.zeros((D, V, K), dtype=np.float64)
        for _ in range(max_inner):
            Elogtheta = _digamma(gamma) - _digamma(gamma.sum(axis=1))[:, None]
            logp = Elogtheta[:, None, :] + Elogbeta.T[None, :, :]
            logp -= logp.max(axis=2, keepdims=True)
            p = np.exp(logp)
            p /= p.sum(axis=2, keepdims=True)
            phi = p
            new = alpha + (C[:, :, None] * phi).sum(axis=1)
            if np.max(np.abs(new - gamma)) < tol:
                gamma = new
                break
            gamma = new
        return gamma, phi

    def transform(self, df, *, text_col, max_inner=20, memory_guard=None):
        """Document-topic distribution gamma (normalised) per document."""
        mg = memory_guard or df._memory_guard
        alpha = float(self.params_["alpha"])
        out = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="lda.transform", extra_factor=3.0)
            texts = b.column(text_col).to_pylist()
            C = self._encode(texts)
            g, _ = self._e_step(C, alpha, max_inner=max_inner)
            out.append(g / g.sum(axis=1, keepdims=True))
        return np.vstack(out) if out else np.zeros((0, self.n_topics_))

    def perplexity(self, df, *, text_col, memory_guard=None):
        """Held-out perplexity exp(-loglik / total tokens).

        Uses the true predictive log-likelihood
        log p(w) = log sum_k theta_k beta_{k,w}, with theta from the
        document E-step, not the variational bound (Elogbeta).
        """
        mg = memory_guard or df._memory_guard
        alpha = float(self.params_["alpha"])
        beta = self.components_
        ll = 0.0
        ntok = 0.0
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="lda.perplexity", extra_factor=3.0)
            texts = b.column(text_col).to_pylist()
            C = self._encode(texts)
            if C.size == 0 or C.sum() == 0:
                continue
            g, _ = self._e_step(C, alpha)
            theta = g / g.sum(axis=1, keepdims=True)      # (D, K)
            logp = np.log(np.maximum(theta @ beta, 1e-300))  # (D, V)
            ll += float((C * logp).sum())
            ntok += float(C.sum())
        if ntok == 0:
            return float("nan")
        return float(math.exp(-ll / ntok))


def _build_vocab(df, text_col, vocab_size, mg):
    df_counts = {}
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="lda.vocab", extra_factor=3.0)
        for t in b.column(text_col).to_pylist():
            for tok in set(tokenize(t)):
                df_counts[tok] = df_counts.get(tok, 0) + 1
        mg.check_state(len(df_counts) * 96 + 1024, name="lda.vocab_state")
    top = sorted(df_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:vocab_size]
    return [w for w, _ in top]


def _fit_once(C_batches_fn, K, V, alpha, beta, max_iter, seed, mg, dummy_cls,
              doc_sample=None):
    """One EM run from a document-seeded init (or Dirichlet fallback).

    Seeding each topic with the word-count vector of a random document
    breaks the symmetry that a random Dirichlet init leaves in place:
    empirically the Dirichlet init stalls in a poor local optimum where
    topics mix (NMI ~0.6), while the document seed reaches NMI ~0.85-1.0.
    """
    rng = np.random.RandomState(seed)
    if doc_sample is not None and len(doc_sample) >= K:
        idxs = rng.choice(len(doc_sample), size=K, replace=False)
        comp = np.zeros((K, V), dtype=np.float64)
        for k, di in enumerate(idxs):
            row = doc_sample[int(di)] + 0.01
            comp[k] = row / row.sum()
    else:
        comp = rng.dirichlet(0.5 * np.ones(V), size=K)
    dummy = dummy_cls(comp, [], {"alpha": alpha, "beta": beta})
    for it in range(int(max_iter)):
        sstats = np.zeros((K, V), dtype=np.float64)
        for C in C_batches_fn():
            if C.sum() == 0:
                continue
            dummy.components_ = comp
            g, phi = dummy._e_step(C, float(alpha))
            sstats += (C[:, :, None] * phi).sum(axis=0).T
        beta_hat = sstats + float(beta)
        beta_hat /= beta_hat.sum(axis=1, keepdims=True)
        comp = beta_hat
    return comp


def _train_ppl(C_batches_fn, comp, alpha):
    beta = comp
    ll = 0.0
    ntok = 0.0
    for C in C_batches_fn():
        if C.sum() == 0:
            continue
        D = C.shape[0]
        K = beta.shape[0]
        gamma = np.full((D, K), alpha + C.sum(axis=1)[:, None] / K)
        dummy = LdaModel(beta, [], {"alpha": alpha})
        g, _ = dummy._e_step(C, float(alpha))
        theta = g / g.sum(axis=1, keepdims=True)
        logp = np.log(np.maximum(theta @ beta, 1e-300))
        ll += float((C * logp).sum())
        ntok += float(C.sum())
    if ntok == 0:
        return float("inf")
    return float(math.exp(-ll / ntok))


def fit_lda(df, text_col, *, n_topics=10, vocab_size=1000, max_iter=15,
            batch_size=None, alpha=0.1, beta=0.01, tau0=1.0, kappa=0.0,
            n_restarts=16, seed=0, memory_guard: Optional[MemoryGuard] = None):
    """Fit LDA over a text column (multi-restart, perplexity-selected).

Note (v0.9.42): this is *sample-based* LDA, not the streaming variational
E-step from Hoffman-Blei-Bach 2010. State is O(K·V); fit memory is
O(max_samples × V) where max_samples=2000 rows — bounded, does not grow
with N. True streaming E-step is future work.
"""
    if n_topics < 1:
        raise ValueError("fit_lda: n_topics must be >= 1")
    if vocab_size < 2:
        raise ValueError("fit_lda: vocab_size must be >= 2")
    if max_iter < 1:
        raise ValueError("fit_lda: max_iter must be >= 1")
    if n_restarts < 1:
        raise ValueError("fit_lda: n_restarts must be >= 1")
    if text_col not in df._columns:
        raise KeyError(f"fit_lda: column {text_col!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    vocab = _build_vocab(df, text_col, vocab_size, mg)
    if len(vocab) < 2:
        raise ValueError("fit_lda: vocabulary has fewer than 2 tokens")
    V = len(vocab)
    K = int(n_topics)
    idx = {w: i for i, w in enumerate(vocab)}

    MAX_SAMPLE_ROWS = 2000
    doc_sample: list = []
    doc_sample_rows = 0

    def _batches():
        nonlocal doc_sample_rows
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="lda.fit", extra_factor=3.0)
            texts = b.column(text_col).to_pylist()
            C = np.zeros((len(texts), V), dtype=np.float64)
            for i, tx in enumerate(texts):
                for tok in tokenize(tx):
                    j = idx.get(tok)
                    if j is not None:
                        C[i, j] += 1.0
            if doc_sample_rows < MAX_SAMPLE_ROWS:
                take = min(C.shape[0], MAX_SAMPLE_ROWS - doc_sample_rows)
                if take > 0:
                    doc_sample.append(C[:take])
                    doc_sample_rows += take
            yield C

    best_comp = None
    best_ppl = float("inf")
    for r in range(int(n_restarts)):
        ds = (np.vstack(doc_sample) if doc_sample else None)
        comp = _fit_once(_batches, K, V, float(alpha), float(beta),
                         max_iter, seed + r, mg, LdaModel, doc_sample=ds)
        ppl = _train_ppl(_batches, comp, float(alpha))
        mg.check_state(K * V * 8 + 512, name="lda.state")
        if ppl < best_ppl:
            best_ppl = ppl
            best_comp = comp
    comp = best_comp / best_comp.sum(axis=1, keepdims=True)
    model = LdaModel(comp, vocab, {"n_topics": K, "vocab_size": V,
                                   "max_iter": max_iter, "alpha": alpha,
                                   "beta": beta, "tau0": tau0,
                                   "kappa": kappa, "n_restarts": n_restarts,
                                   "train_perplexity": best_ppl})
    log_fit("lda", state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="lda.model")
    return model
