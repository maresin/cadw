# J3 — Topic modeling

*Section:* J — Natural language processing

**Description:** Discover latent topics in a document collection; infer per-document mixtures.

**Real-life examples:**
- Editorial topic tagging
- Corpus exploration
- Keyword discovery in tickets

## Benchmarks
- **20Newsgroups** — https://scikit-learn.org/stable/datasets/real_world.html
  - metrics: perplexity, coherence_c_v
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: perplexity, fit_time_s, peak_rss_mb, vocab_size

## Models
### LDA — baseline / tier1_core

**Roles:** probabilistic, representation
**Status:** ✓ cadw, ✓ agent

**Purpose:** Blei 2003 - latent Dirichlet allocation: each document is a mixture of K topics, each topic a distribution over words. Inductive bias: exchangeable bag-of-words with Dirichlet priors - interpretable, but ignores word order and struggles on short texts.

**Use cases:**
- Editorial topic tagging
- Corpus exploration on 20NG
- Keyword discovery in support tickets

**Config:** `{'n_topics': 'int', 'alpha': 'float', 'beta': 'float', 'max_iter': 'int'}`

**CADW:** `cadw.ml.lda.fit_lda`

**Benchmarks:**
- **[literature]** 20Newsgroups → perplexity=null
  - algorithm: `LDA (Blei et al. 2003)` (paper)
  - number: https://scikit-learn.org/stable/datasets/real_world.html
  - code/paper: https://aclanthology.org/W04-3252/
  - note: no_public_benchmark: 20NG has no live public perplexity/coherence leaderboard; per-paper numbers use different vocab/preprocessing
- **[cadw_measured]** cadw_internal → perplexity=4.131 ± 0.0005 (n=3)
  - algorithm: `cadw.ml.lda` (code)
  - number: .pingis/benches/lda.json
  - code/paper: cadw/ml/lda.py
  - note: CADW-native: perplexity=4.130/4.131/4.131 @n=2k/10k/50k, k=3 (scripts/bench_lda.py, .pingis/benches/lda.json); synthetic corpus; std=half-range.
- **[cadw_measured]** cadw_internal → fit_time_s=102.892 ± 49.2 (n=3)
  - algorithm: `cadw.ml.lda` (code)
  - number: .pingis/benches/lda.json
  - code/paper: cadw/ml/lda.py
  - note: CADW-native: fit_s=4.491/21.382/102.892 @n=2k/10k/50k (scripts/bench_lda.py, .pingis/benches/lda.json); std=half-range; hardware-specific.

### NMF — baseline / tier2_wrapper

**Roles:** representation
**Status:** desc

**Purpose:** Lee & Seung 1999 - non-negative matrix factorization of the doc-term matrix. Inductive bias: additive parts-based topics with sparsity enforced by non-negativity; no probabilistic likelihood, so perplexity is undefined.

**Use cases:**
- Parts-based topic discovery
- Topic tagging with sparse vocabularies
- Document clustering

**Config:** `{'n_components': 'int', 'init': 'string', 'max_iter': 'int'}`

**Benchmarks:**
- **[literature]** 20Newsgroups → perplexity=null
  - algorithm: `sklearn.decomposition.NMF` (docs)
  - number: https://scikit-learn.org/stable/datasets/real_world.html
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
  - note: no_public_benchmark: NMF has no likelihood/perplexity; 20NG has no live public coherence leaderboard either

### BERTopic — sota / tier3_export

**Roles:** representation, nonlinear_interactions
**Status:** desc, export

**Purpose:** Grootendorst 2022 - pipeline of PLM embeddings + UMAP + HDBSCAN + c-TF-IDF for topic representation. Inductive bias: contextual embeddings + density clustering; SOTA coherence on short texts, but torch-only via to_batches(format='torch').

**Use cases:**
- Short-text topic discovery (tweets, tickets)
- Large-scale corpus exploration
- Dynamic topic modeling over time

**Config:** `{'embedding_model': 'string', 'umap_n_neighbors': 'int', 'min_cluster_size': 'int'}`

**Benchmarks:**
- **[literature]** 20Newsgroups → coherence_c_v=null
  - algorithm: `BERTopic (Grootendorst 2022)` (docs)
  - number: https://scikit-learn.org/stable/datasets/real_world.html
  - code/paper: https://maartengr.github.io/BERTopic/index.html
  - note: no_public_benchmark: 20NG has no live public coherence leaderboard; BERTopic paper reports its own c_v on a different corpus subset

## Metrics
- **perplexity** (anchor, lower)
  - Held-out per-word perplexity; canonical probabilistic topic-model fit.
  - *special:* Only defined for probabilistic models (LDA, not NMF/BERTopic). Vocab and preprocessing change the number - compare only within the same pipeline.
- **coherence_c_v** (diagnostic, higher)
  - Human-interpretability proxy for top-word sets.
  - *special:* Not comparable to perplexity; report alongside it, never combined.
- **fit_time_s** (diagnostic, lower)
  - Wall-clock fit time.
  - *special:* Hardware-specific.

## Ensemble rules
- **[probabilistic]** → LDA (max_per_role=1)
  - Only CADW-verified topic model; NMF and BERTopic are reference-only until implemented, so they are excluded from ensembles.
