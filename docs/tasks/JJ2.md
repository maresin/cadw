# J2 — Text embeddings

*Section:* J — Natural language processing

**Description:** Map documents to dense vectors so that semantic similarity becomes geometric.

**Real-life examples:**
- Semantic search
- Duplicate detection
- Ticket clustering

## Benchmarks
- **STS-B** — https://huggingface.co/datasets/sentence-transformers/stsb
  - metrics: spearman_cosine, pearson_cosine
- **MTEB** — https://huggingface.co/spaces/mteb/leaderboard
  - metrics: nDCG_at_10, mean_average_precision, main_score

## Models
### LSA_TFIDF_SVD — baseline / tier1_core

**Roles:** representation
**Status:** ✓ cadw, ✓ agent

**Purpose:** Latent Semantic Analysis: TF-IDF doc-term matrix reduced by randomized SVD to k dense factors. Inductive bias: linear low-rank co-occurrence structure - captures topical synonymy without supervision; cannot encode word order or polysemy.

**Use cases:**
- Topic clustering of ticket corpora
- Baseline document similarity search
- Pre-embedding feature compression

**Config:** `{'n_components': 'int', 'n_iter': 'int'}`

**CADW:** `cadw.ml.randsvd.fit_randomized_svd`

**Benchmarks:**
- **[literature]** STS-B → spearman_cosine=null
  - algorithm: `TF-IDF + randomized SVD (LSA)` (docs)
  - number: https://huggingface.co/datasets/sentence-transformers/stsb
  - code/paper: https://scikit-learn.org/stable/modules/decomposition.html
  - note: no_public_benchmark: STS-B GLUE LB ranks only fine-tuned PLMs; unsupervised LSA has no public row

### Word2Vec — baseline / tier2_wrapper

**Roles:** representation
**Status:** desc

**Purpose:** Mikolov 2013 - shallow predictive embedding from local context windows; document vectors obtained by pooling. Inductive bias: distributional similarity from local co-occurrence; no context-dependent polysemy.

**Use cases:**
- Domain word vectors on small corpora
- Cheap semantic features for classifiers
- Nearest-neighbour term expansion

**Config:** `{'vector_size': 'int', 'window': 'int', 'sg': 'int', 'epochs': 'int'}`

**Benchmarks:**
- **[literature]** STS-B → spearman_cosine=null
  - algorithm: `Word2Vec (Mikolov et al. 2013)` (paper)
  - number: https://huggingface.co/datasets/sentence-transformers/stsb
  - code/paper: https://arxiv.org/abs/1301.3781
  - note: no_public_benchmark: Word2Vec was never entered on the STS-B GLUE leaderboard; paper reports word-similarity tasks, not STS-B

### SentenceTransformer — sota / tier3_export

**Roles:** representation, nonlinear_interactions
**Status:** desc, export

**Purpose:** Reimers & Gurevych 2019 - siamese BERT fine-tuned contrastively to produce sentence embeddings directly. Inductive bias: contextual, order-aware semantics from large corpora; SOTA on STS/MTEB, torch-only via to_batches(format='torch').

**Use cases:**
- Semantic search / RAG retrieval
- Duplicate question detection
- Cross-lingual matching

**Config:** `{'base_model': 'string', 'pooling': 'string', 'max_seq_length': 'int'}`

**Benchmarks:**
- **[literature]** MTEB → main_score=null
  - algorithm: `Sentence-BERT (Reimers & Gurevych 2019)` (paper)
  - number: https://huggingface.co/spaces/mteb/leaderboard
  - code/paper: https://arxiv.org/abs/1908.10084
  - note: fetch_ok_no_number: MTEB Space HTML fetched 2026-10-24 but the table is client-side rendered; no per-model number extractable
- **[literature]** STS-B → spearman_cosine=null
  - algorithm: `Sentence-BERT (Reimers & Gurevych 2019)` (paper)
  - number: https://huggingface.co/datasets/sentence-transformers/stsb
  - code/paper: https://arxiv.org/abs/1908.10084
  - note: no_public_benchmark: STS-B GLUE LB not fetchable; S-BERT paper reports STS-B dev Spearman but that is a paper number, not a live LB row

## Metrics
- **spearman_cosine** (anchor, higher)
  - Rank correlation between cosine similarity and human similarity.
  - *special:* Only comparable on the same STS split; unsupervised and fine-tuned embeddings are NOT comparable.
- **nDCG_at_10** (diagnostic, higher)
  - Retrieval quality in MTEB task families.
  - *special:* MTEB aggregates heterogeneous tasks; report per-family, never as one scalar.
- **fit_time_s** (diagnostic, lower)
  - Wall-clock fit/encode time.
  - *special:* Hardware-specific.
- **main_score** (diagnostic, higher)
  - MTEB aggregate main score; leaderboard-defined.
  - *special:* Task-subset dependent.

## Ensemble rules
- **[representation]** → LSA_TFIDF_SVD (max_per_role=1)
  - Only CADW-verified representation model here; concatenating with a PLM embedding requires to_batches(format='torch') and is out of scope for native CADW.
