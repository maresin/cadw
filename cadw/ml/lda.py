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
import shutil
import tempfile
from pathlib import Path
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


def _tokenize_to_csr(texts, idx, V):
    """Sparse CSR doc-term count matrix (v0.9.87-P5a-fix-lda-a).

    Memory: O(nnz + D), independent of V. Replaces the dense
    (D, V) intermediate that OOM'd at V >= 10k.
    """
    from scipy.sparse import csr_matrix
    D = len(texts)
    rows, cols, vals = [], [], []
    for i, tx in enumerate(texts):
        counts = {}
        for tok in tokenize(tx):
            j = idx.get(tok)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            rows.append(i)
            cols.append(j)
            vals.append(float(c))
    if not vals:
        return csr_matrix((D, V), dtype=np.float64)
    return csr_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64),
          np.asarray(cols, dtype=np.int64))),
        shape=(D, V))


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
        """Sparse CSR doc-term counts (v0.9.87-P5a-fix-lda-a)."""
        V = len(self.vocab_)
        return _tokenize_to_csr(texts, self._index, V)

    def _e_step(self, C, alpha, max_inner=20, tol=1e-4, return_phi=False):
        """Sparse E-step (v0.9.87-P5a-fix-lda-a).

        Accepts CSR or dense C; converts dense to CSR first. Per-doc
        fixed-point touches only non-zero vocab positions. Mathematically
        identical to dense path (zero C[d,v] contributes exact 0 to
        gamma). Returns:
          return_phi=False -> (gamma, None)
          return_phi=True  -> (gamma, list[(nz_v, p_nnz)])
        """
        from scipy.sparse import csr_matrix, issparse
        if not issparse(C):
            C = csr_matrix(C)
        D, V = C.shape
        K = self.components_.shape[0]
        beta = self.components_
        Elogbeta = _digamma(beta) - _digamma(beta.sum(axis=1))[:, None]  # (K,V)
        Elogbeta_T = Elogbeta.T  # (V, K)
        row_sum = np.asarray(C.sum(axis=1)).ravel()  # (D,)
        gamma = np.full((D, K), float(alpha))
        gamma += row_sum[:, None] / K
        phi_rows = [] if return_phi else None
        for _ in range(max_inner):
            Elogtheta = _digamma(gamma) - _digamma(gamma.sum(axis=1))[:, None]
            new_gamma = np.full((D, K), float(alpha))
            cur_phi = [] if return_phi else None
            for d in range(D):
                s = C.indptr[d]
                e = C.indptr[d + 1]
                if s == e:
                    if return_phi:
                        cur_phi.append(
                            (np.zeros(0, dtype=np.int64),
                             np.zeros((0, K), dtype=np.float64)))
                    continue
                nz_v = C.indices[s:e]
                nz_c = C.data[s:e]
                logp = Elogtheta[d][None, :] + Elogbeta_T[nz_v]  # (nnz,K)
                logp -= logp.max(axis=1, keepdims=True)
                p = np.exp(logp)
                p /= p.sum(axis=1, keepdims=True)
                new_gamma[d] += nz_c @ p
                if return_phi:
                    cur_phi.append((nz_v, p))
            delta = float(np.max(np.abs(new_gamma - gamma))) if D > 0 else 0.0
            gamma = new_gamma
            if return_phi:
                phi_rows = cur_phi
            if delta < tol:
                break
        return gamma, phi_rows

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
        K = beta.shape[0]
        ll = 0.0
        ntok = 0.0
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="lda.perplexity", extra_factor=3.0)
            texts = b.column(text_col).to_pylist()
            C = self._encode(texts)  # CSR
            mg.check_state(
                C.nnz * K * 8 * 6 + C.shape[0] * K * 8 + 1024,
                name="lda.perplexity_batch_state")
            if C.nnz == 0:
                continue
            g, _ = self._e_step(C, alpha)  # dispatches sparse
            theta = g / g.sum(axis=1, keepdims=True)  # (D, K)
            # v0.9.87-P5a-fix-lda-a: logp on non-zeros only.
            for d in range(C.shape[0]):
                s = C.indptr[d]
                e = C.indptr[d + 1]
                if s == e:
                    continue
                nz_v = C.indices[s:e]
                nz_c = C.data[s:e]
                logp_nz = np.log(
                    np.maximum(theta[d] @ beta[:, nz_v], 1e-300))
                ll += float((nz_c * logp_nz).sum())
                ntok += float(nz_c.sum())
        if ntok == 0:
            return float("nan")
        return float(math.exp(-ll / ntok))


def _build_vocab(df, text_col, vocab_size, mg):
    df_counts = {}
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="lda.vocab", extra_factor=3.0)
        # v0.9.90-S3 (G2): pre-check BEFORE dict grows. Worst case
        # this batch adds b.num_rows new unique tokens.
        mg.check_state(
            (len(df_counts) + b.num_rows) * 96 + 1024,
            name="lda.vocab_state")
        for t in b.column(text_col).to_pylist():
            for tok in set(tokenize(t)):
                df_counts[tok] = df_counts.get(tok, 0) + 1
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
            if C.nnz == 0:
                continue
            dummy.components_ = comp
            g, phi_rows = dummy._e_step(
                C, float(alpha), return_phi=True)
            # v0.9.87-P5a-fix-lda-a: sparse M-step. Accumulate
            # (v, nz_c * phi_doc) contributions and scatter-add once.
            all_v = []
            all_contrib = []
            for d, (nz_v, p) in enumerate(phi_rows):
                if len(nz_v) == 0:
                    continue
                nz_c = C.data[C.indptr[d]:C.indptr[d + 1]]
                all_v.append(nz_v)
                all_contrib.append(nz_c[:, None] * p)
            if all_v:
                all_v = np.concatenate(all_v)
                all_contrib = np.concatenate(all_contrib)
                np.add.at(sstats.T, all_v, all_contrib)
        beta_hat = sstats + float(beta)
        beta_hat /= beta_hat.sum(axis=1, keepdims=True)
        comp = beta_hat
    return comp


def _train_ppl(C_batches_fn, comp, alpha):
    beta = comp
    ll = 0.0
    ntok = 0.0
    for C in C_batches_fn():
        if C.nnz == 0:
            continue
        D = C.shape[0]
        K = beta.shape[0]
        dummy = LdaModel(beta, [], {"alpha": alpha})
        g, _ = dummy._e_step(C, float(alpha))
        theta = g / g.sum(axis=1, keepdims=True)
        # v0.9.87-P5a-fix-lda-a: logp on non-zeros only (CSR).
        for d in range(D):
            s = C.indptr[d]
            e = C.indptr[d + 1]
            if s == e:
                continue
            nz_v = C.indices[s:e]
            nz_c = C.data[s:e]
            logp_nz = np.log(
                np.maximum(theta[d] @ beta[:, nz_v], 1e-300))
            ll += float((nz_c * logp_nz).sum())
            ntok += float(nz_c.sum())
    if ntok == 0:
        return float("inf")
    return float(math.exp(-ll / ntok))


def _estimate_k_tokens(df, text_col, sample_size=2000):
    """Heuristic token-cardinality estimate for num_partitions sizing.

    Sample first `sample_size` rows, count distinct tokens; extrapolate
    to total_rows. Over-estimates for natural corpora (Heaps' law is
    sublinear), which is safe for K-spill (extra disk, bounded RAM).
    """
    distinct = set()
    sampled = 0
    total = 0
    for b in df._iter_filtered_batches():
        total += b.num_rows
        if sampled >= sample_size:
            continue
        take = min(b.num_rows, sample_size - sampled)
        texts = b.column(text_col).to_pylist()[:take]
        for t in texts:
            for tok in set(tokenize(t)):
                distinct.add(tok)
        sampled += take
    if sampled == 0:
        return 0, total
    return int(len(distinct) * (total / sampled)), total


def _build_vocab_k_spill(df, text_col, vocab_size, mg, spill_dir,
                          num_partitions, min_df=1):
    """Bounded-RAM vocab via hash_shuffle (v0.9.116 K-spill, AP5).

    Returns (vocab, dfv) — vocab list[str], dfv list[int] (document freq
    per token in same order). min_df=1 gives LDA behavior; tfidf passes
    its own min_df.

    Pass 1: tokenize → hash to P partitions (Arrow IPC on disk).
    Pass 2: per-partition Counter → top-vocab_size.
    Pass 3: merge per-partition tops → global top-vocab_size.

    Exact (not approximate): token pruned in its partition implies
    >= vocab_size tokens in that same partition have count >= its
    count, so its global rank is > vocab_size (partitions are disjoint
    by hash — global count = partition count).

    RAM: max(K_raw/P * 96, vocab_size * 96) + P * vocab_size * 96.
    Disk: O(K_raw * avg_token_len + P * vocab_size * 96).
    """
    from cadw.ops.shuffle import hash_shuffle
    from cadw.io.batch_reader import ArrowIPCBatchReader

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    _any_tokens = [False]

    def _token_batches():
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="lda.kspill.batch", extra_factor=2.0)
            toks = []
            for t in b.column(text_col).to_pylist():
                for tok in set(tokenize(t)):
                    toks.append(tok)
            if toks:
                _any_tokens[0] = True
                yield pa.RecordBatch.from_arrays(
                    [pa.array(toks, type=pa.string())],
                    names=["token"])

    try:
        partitions = hash_shuffle(
            _token_batches(),
            key_columns=["token"],
            value_columns=[],
            num_partitions=num_partitions,
            memory_guard=mg,
            spill_dir=spill_dir / "shuffle",
            log=False,
        )
    except ValueError:
        if not _any_tokens[0]:
            return [], []
        raise

    per_partition_top = []
    for p_path in partitions:
        counter = {}
        reader = ArrowIPCBatchReader(str(p_path), columns=["token"])
        try:
            for b in reader:
                for tok in b.column("token").to_pylist():
                    counter[tok] = counter.get(tok, 0) + 1
                mg.check_state(
                    len(counter) * 96 + 1024,
                    name="lda.kspill.pcount")
        finally:
            reader.close()
        if min_df > 1:
            counter = {k: v for k, v in counter.items() if v >= min_df}
        top_p = sorted(counter.items(),
                       key=lambda kv: (-kv[1], kv[0]))[:vocab_size]
        per_partition_top.append(top_p)
        del counter

    merged = {}
    for top_p in per_partition_top:
        for tok, cnt in top_p:
            merged[tok] = merged.get(tok, 0) + cnt

    top = sorted(merged.items(),
                 key=lambda kv: (-kv[1], kv[0]))[:vocab_size]
    return [w for w, _ in top], [c for _, c in top]


def fit_lda(df, text_col, *, n_topics=10, vocab_size=1000, max_iter=15,
            batch_size=None, alpha=0.1, beta=0.01, tau0=1.0, kappa=0.0,
            n_restarts=16, seed=0, memory_guard: Optional[MemoryGuard] = None,
            spill_dir=None, num_partitions=None):
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
    # v0.9.116-lda-kspill: K-spill if spill_dir or num_partitions given.
    # Default (both None) → legacy path (back-compat, bit-identical).
    if spill_dir is None and num_partitions is None:
        vocab = _build_vocab(df, text_col, vocab_size, mg)
    else:
        _cleanup_spill = spill_dir is None
        _spill_dir = spill_dir if spill_dir is not None else tempfile.mkdtemp(
            prefix="lda_kspill_")
        try:
            if num_partitions is None:
                _k_est, _ = _estimate_k_tokens(df, text_col)
                if _k_est <= 0:
                    num_partitions = 2
                else:
                    from cadw.ops.shuffle import estimate_num_partitions
                    num_partitions = max(2, estimate_num_partitions(
                        _k_est, mg.state_limit_bytes))
            vocab, _ = _build_vocab_k_spill(
                df, text_col, vocab_size, mg, _spill_dir, num_partitions)
        finally:
            if _cleanup_spill:
                shutil.rmtree(_spill_dir, ignore_errors=True)
    if len(vocab) < 2:
        raise ValueError("fit_lda: vocabulary has fewer than 2 tokens")
    V = len(vocab)
    K = int(n_topics)
    idx = {w: i for i, w in enumerate(vocab)}

    MAX_SAMPLE_ROWS = 2000
    doc_sample: list = []
    doc_sample_rows = 0

    # v0.9.87-P5a-fix-lda-a-fix1: cap doc_sample TOTAL, not per-take.
    # Pre-fix bug: per-take cap allowed doc_sample to grow to
    # MAX_SAMPLE_ROWS=2000 rows; at V=50000 that is 800 MB silent OOM.
    # Post-fix: total capped at state_limit // (2 * row_bytes), so
    # doc_sample <= state_limit / 2 (half for it, half for other state).
    max_sample_rows = MAX_SAMPLE_ROWS
    if V > 0:
        budget_rows = mg.state_limit_bytes // (V * 8 * 2)
        max_sample_rows = min(max_sample_rows, max(0, budget_rows))

    def _batches():
        nonlocal doc_sample_rows
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="lda.fit", extra_factor=3.0)
            texts = b.column(text_col).to_pylist()
            C = _tokenize_to_csr(texts, idx, V)  # sparse
            mg.check_state(
                C.nnz * K * 8 * 6 + C.shape[0] * K * 8 + 1024,
                name="lda.fit_batch_state")
            if doc_sample_rows < max_sample_rows:
                take = min(C.shape[0], max_sample_rows - doc_sample_rows)
                if take > 0:
                    doc_sample.append(C[:take].toarray())
                    doc_sample_rows += take
            yield C

    best_comp = None
    best_ppl = float("inf")
    for r in range(int(n_restarts)):
        if doc_sample:
            n_ds = sum(d.shape[0] for d in doc_sample)
            mg.check_state(n_ds * V * 8 + 1024,
                           name="lda.doc_sample")
            ds = np.vstack(doc_sample)
        else:
            ds = None
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
