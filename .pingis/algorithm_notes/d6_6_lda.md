# D.6.6 Online variational Bayes LDA

## Paper
Blei, Ng, Jordan, *Latent Dirichlet Allocation*, JMLR 3, 2003.
Hoffman, Blei, Bach, *Online Learning for Latent Dirichlet Allocation*,
NIPS 2010. Reference implementation in tests: sklearn
`LatentDirichletAllocation` (Pedregosa et al., JMLR 12, 2011).

## Docs
- `cadw.ml.lda.fit_lda(df, text_col, n_topics=10, vocab_size=1000,
  max_iter=15, alpha=0.1, beta=0.01, n_restarts=16, seed=0)`
- `CadwDataFrame.fit_lda(...)`.
- Model: `transform(df, text_col)` -> normalised document-topic gamma,
  `perplexity(df, text_col)`, `components_`, `vocab_`.

## Design
- Pass 1: bounded document-frequency vocabulary (top `vocab_size`, tie by
  token) — O(V) state.
- Document-level variational E-step (gamma/phi, per-document fixed point)
  streamed per batch; the M-step accumulates topic-word sufficient
  statistics (K x V) per epoch. No D x V matrix is materialised; the
  in-memory batch is O(batch x V) and passes check_batch(extra_factor=3.0).
- Topic symmetry is broken by a **document-seeded init**: each topic starts
  as the word-count vector of a random (bounded 2000-row) document sample.
  A random Dirichlet init stalls in a mixed-topic local optimum (NMI ~0.6);
  the document seed reaches NMI 0.85-1.0. Multi-restart selection by
  training perplexity (default 16 restarts).
- `rho_t = (tau0 + t)^(-kappa)`; default kappa=0 gives a full batch M-step
  per epoch (correct for offline fit), kappa>0 is the stochastic online
  schedule.
- log_fit emitted per fit (iterative model).

## Benchmark (`.pingis/benches/lda.json`, 3 topics, vocab 12)
| n | fit_s | perplexity | peak_mb | state_b |
|---|---|---|---|---|
| 2 000 | 4.43 | 4.130 | 147.4 | 896 |
| 10 000 | 21.19 | 4.131 | 298.6 | 896 |
| 50 000 | 104.26 | 4.131 | 1 319.7 | 896 |
Reference: sklearn LDA on the identical bag-of-words. Test tolerance:
our NMI >= 0.9 x sklearn's on the same BoW (sklearn reaches 1.0, ours
reaches 1.0 at the defaults; the >= 0.9 x bar is the honest regression
bound). Perplexity is computed with the true predictive likelihood
log sum_k theta_k beta_kw, not the variational bound.

## RAM model
State = K*V*8 bytes + vocabulary (8*V) + 512 overhead (896 B measured for
K=3, V=12). Fit holds one in-memory batch (O(batch*V)); the multi-restart
loop keeps only the best K x V beta.
