# J1 — Text classification

*Section:* J — Natural language processing

**Description:** Assign a label to a text document (sentiment, topic, intent).

**Real-life examples:**
- Sentiment analysis
- Support-ticket routing
- Spam / abuse detection

## Benchmarks
- **SST-2** — https://nlp.stanford.edu/sentiment/
  - metrics: accuracy
- **cadw_internal_nlp** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: fit_time_s, peak_rss_mb, vocab_size

## Models
### TFIDF_LogReg — baseline / tier1_core

**Roles:** linear_signal, representation
**Status:** ✓ cadw, ✓ agent

**Purpose:** TF-IDF bag-of-words features fed to a linear logistic classifier. Inductive bias: additive linear signal over sparse lexical n-grams - strong on topic/sentiment tasks with n in the thousands; loses word order and long-range context.

**Use cases:**
- Support-ticket intent routing
- Sentiment on product reviews
- Abuse / spam detection

**Config:** `{'max_features': 'int', 'ngram_range': 'list', 'C': 'float'}`

**CADW:** `cadw.ml.tfidf.fit_tfidf`

**Benchmarks:**
- **[literature]** SST-2 → accuracy=null
  - algorithm: `sklearn TfidfVectorizer + LogisticRegression` (docs)
  - number: https://huggingface.co/datasets/nyu-mll/glue
  - code/paper: https://scikit-learn.org/stable/modules/feature_extraction.html
  - note: no_public_benchmark: SST-2 GLUE live leaderboard not fetchable (PWC redirect to HF trending; gluebenchmark.com unreachable); TF-IDF+LR has no public per-model SST-2 row
- **[cadw_measured]** cadw_internal → fit_time_s=1.281 ± 0.628 (n=3)
  - algorithm: `cadw.ml.tfidf` (code)
  - number: .pingis/benches/tfidf.json
  - code/paper: cadw/ml/tfidf.py
  - note: CADW-native: fit_s=0.026/0.258/1.281 @n=10k/100k/500k (scripts/bench_tfidf.py, .pingis/benches/tfidf.json); std=half-range; hardware-specific.
- **[cadw_measured]** cadw_internal → peak_rss_mb=229.6 ± 114.45 (n=3)
  - algorithm: `cadw.ml.tfidf` (code)
  - number: .pingis/benches/tfidf.json
  - code/paper: cadw/ml/tfidf.py
  - note: CADW-native: peak_mb=0.7/45.1/229.6 @n=10k/100k/500k (scripts/bench_tfidf.py, .pingis/benches/tfidf.json); std=half-range; hardware-specific.

### ComplementNB — baseline / tier1_core

**Roles:** probabilistic
**Status:** ✓ agent, desc

**Purpose:** Complement NB on TF-IDF (Rennie 2003). Inductive bias: per-class independent word likelihoods with a complement correction for imbalance. Very fast; weakest when word interactions matter. Reference impl in .pingis/impl/J/complement_nb.py (smoke 9/9, REPORT.md §6).

**Use cases:**
- Fast spam baseline
- Long-tail topic tagging
- Low-resource text classification

**Config:** `{'alpha': 'float'}`

**Benchmarks:**
- **[literature]** SST-2 → accuracy=null
  - algorithm: `sklearn.naive_bayes.ComplementNB` (docs)
  - number: https://huggingface.co/datasets/nyu-mll/glue
  - code/paper: https://scikit-learn.org/stable/modules/naive_bayes.html
  - note: no_public_benchmark: SST-2 GLUE live LB not fetchable; ComplementNB has no public SST-2 row

### LinearSVC — baseline / tier2_wrapper

**Roles:** linear_signal
**Status:** ✓ agent, desc

**Purpose:** L2-regularized linear SVM on TF-IDF (Joachims 1998). Inductive bias: maximum-margin linear separator over sparse features - strong on topic tasks; no calibrated probabilities out of the box.

**Use cases:**
- Topic classification on 20NG
- Reuters news routing
- Intent detection on transcripts

**Config:** `{'C': 'float', 'max_iter': 'int'}`

**Benchmarks:**
- **[literature]** 20Newsgroups → accuracy=null
  - algorithm: `sklearn.svm.LinearSVC` (docs)
  - number: https://scikit-learn.org/stable/datasets/real_world.html
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  - note: no_public_benchmark: no live 20NG leaderboard; sklearn dataset page has splits but no per-model accuracies

### BERT — sota / tier3_export

**Roles:** representation, nonlinear_interactions
**Status:** export

**Purpose:** Devlin 2019 - bidirectional transformer pre-trained with masked LM, fine-tuned per task. Inductive bias: contextual token representations with self-attention over the full sequence - SOTA on SST-2/GLUE; torch-only, reached via to_batches(format='torch').

**Use cases:**
- GLUE benchmark fine-tuning
- Review sentiment at scale
- Multi-lingual intent detection

**Config:** `{'model_size': 'string', 'max_len': 'int', 'lr': 'float'}`

**Benchmarks:**
- **[literature]** SST-2 → accuracy=null
  - algorithm: `BERT (Devlin et al. 2019)` (paper)
  - number: https://huggingface.co/datasets/nyu-mll/glue
  - code/paper: https://arxiv.org/abs/1810.04805
  - note: no_public_benchmark: SST-2 GLUE live LB not fetchable (PWC redirect, gluebenchmark.com unreachable); paper numbers are not a live LB

## Metrics
- **accuracy** (anchor, higher)
  - Correct-class fraction; canonical classification metric.
  - *special:* Threshold 0.5; meaningful only when classes are balanced (SST-2 yes; 20NG/Reuters no - prefer f1_macro).
- **f1_macro** (diagnostic, higher)
  - Macro-averaged F1; robust to class imbalance.
  - *special:* Not comparable to accuracy numerically.
- **fit_time_s** (diagnostic, lower)
  - Wall-clock fit time.
  - *special:* Hardware-specific.
- **peak_rss_mb** (diagnostic, lower)
  - CADW measured peak RSS during fit; not comparable to external.
  - *special:* tracemalloc-based.

## Ensemble rules
- **[linear_signal, probabilistic]** → TFIDF_LogReg (max_per_role=1)
  - LogReg (margin-based) and ComplementNB (generative) give complementary lexical priors; average predicted probabilities on the same TF-IDF matrix.
