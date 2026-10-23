#!/usr/bin/env python3
# v0.9.33: docs — ML task taxonomy + bug/algorithm backlog + PINGIS tasks.
# Docs + data (roadmap.json Phase I). 0 production code.

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

ROADMAP = "docs/roadmap.json"
PYPROJECT = "pyproject.toml"
INIT = "cadw/__init__.py"
CONTEXT = "CONTEXT.md"
TODAY = "2026-10-23"
TARGET_VERSION = "0.9.33"


# ============ verify ============

def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.32":
        raise SystemExit(f"expected HEAD=v0.9.32, got {head!r}")
    print(f"HEAD: {head}")

    if Path("docs/ML_TASKS_TAXONOMY.md").exists():
        raise SystemExit("docs/ML_TASKS_TAXONOMY.md exists — already applied?")
    if Path("docs/BUG_BACKLOG.md").exists():
        raise SystemExit("docs/BUG_BACKLOG.md exists")

    for f in ("docs/roadmap.json", "CONTEXT.md", "pyproject.toml"):
        if not Path(f).exists():
            raise SystemExit(f"{f} missing")
    print("  all target files present")


# ============ 1. ML_TASKS_TAXONOMY.md ============

ML_TAXONOMY = """# CADW — ML Tasks Taxonomy (navigator)

**Version:** 0.9.33 (2026-10-23)
**Purpose:** master navigator per ML task type. For each type —
recommended 3 tiers of algorithms (baseline / best-in-class /
ensemble candidates) and CADW implementation status.

**Reading order:**
1. This file — pick task type, see recommended tiers + CADW status
2. `docs/roadmap_research/part2_taxonomy.md` — full taxonomy from
   PINGIS research (54 types, exhaustive)
3. `docs/ALGORITHM_BACKLOG.md` — pending algorithms with test/bench
   criteria
4. `docs/tasks/XX_<type>.md` — detailed recipe (Phase E, filled as
   algorithms land)

**Legend:**
- ✓ available now (v0.9.33)
- ⏳ pending (see ALGORITHM_BACKLOG, item D.X.Y)
- ✗ out of scope (torch-only, no out-of-core fit)

---

## Master navigation table

| Chapter | Task type | Baseline | Best-in-class | Ensemble candidates |
|---|---|---|---|---|
| **A1** | Binary classification | LogisticRegression | GBDT (LightGBM-like) | LogReg + GBDT + NB |
| **A2** | Multiclass | Multinomial logreg | GBDT multiclass | GBDT + NB + linear |
| **A3** | Regression | Ridge (L2) | GBDT (L2) | Ridge + GBDT + Tweedie |
| **A4** | Multilabel | OvR binary | Classifier chains | OvR GBDT + chain |
| **A5** | Ordinal classification | mord OrdinalRidge | GBDT-on-thresholds | OrdLogit + GBDT |
| **A6** | Imbalanced classification | Weighted Logistic | Cost-sens GBDT | Cost-GBDT + balanced bag + calib |
| **A7** | Quantile / interval regression | Linear QuantReg | GBDT (pinball) | GBDT + linear + NGBoost |
| **B1** | Flat clustering | k-means | HDBSCAN | k-means + HDBSCAN + GMM |
| **B2** | Hierarchical clustering | Ward linkage | HDBSCAN (via B3) | BIRCH→linkage |
| **B3** | Streaming clustering | MiniBatchKMeans | BIRCH CF-tree | BIRCH + MBKM two-stage |
| **C1** | Linear dim reduction | PCA (cov+eigh) | Randomized SVD | PCA + rand-proj + AE |
| **C2** | Nonlinear dim reduction | t-SNE (sample) | UMAP (sample+stream) | PCA + UMAP |
| **C3** | Autoencoder | (torch batching) | VAE / TabNet-lite | AE + PCA |
| **D1** | TS forecast (univariate) | Naive / ETS | SARIMA / GBDT-lag | GBDT + SARIMA + ETS |
| **D2** | TS forecast (multi) | GBDT per series | VAR / TFT | GBDT + VAR |
| **D3** | TS hierarchical | Bottom-up | MinT-WLS | MinT + bottom-up |
| **D4** | TS classification | Interval features | MiniRocket | MiniRocket + interval + GBDT |
| **D5** | Change-point detection | Binary segmentation | PELT (online) | PELT + Bayesian |
| **E1** | Unsup anomaly | z-score / IForest | Isolation Forest | IForest + LOF + z |
| **E2** | Supervised anomaly | Weighted GBDT | Cost-GBDT + threshold | GBDT + LogReg |
| **E3** | TS anomaly | Z-score over window | Matrix Profile | MP + IForest + PELT |
| **F1** | Collaborative recsys | Item-item KNN | ALS (implicit) | ALS + KNN + GBDT-ranker |
| **F2** | Hybrid recsys | Linear mix | MF + content GBDT | ALS + content + GBDT |
| **G1** | Learning to Rank | Pairwise logistic | LambdaMART | LM + linear + listwise |
| **H1** | Causal effect | Propensity matching | T/S-learner on GBDT | DML + T + forest |
| **H2** | Uplift modeling | T-learner | Uplift trees (Qini) | Uplift + X-learner |
| **I1** | Survival analysis | Kaplan-Meier | CoxPH / RSF | CoxPH + GBDT-surv |
| **J1** | Text classification | TF-IDF + LogReg | MiniLM + linear | TFIDF + embed + BERT |
| **J2** | NER / seq tagging | (torch only) | (torch only) | (out of scope) |
| **J3** | Text embeddings | word2vec (sample) | sentence-transformers | Embed + GBDT |
| **J4** | Topic modeling | NMF | LDA (Reduce EM) | LDA + BERTopic (sample) |
| **J5** | Generation / translation | (torch only) | LLM | (out of scope) |
| **K1** | Image classification | (torch only) | (torch only) | (out of scope) |
| **K2** | Object detection | (torch only) | (torch only) | (out of scope) |
| **K3** | Segmentation | (torch only) | (torch only) | (out of scope) |
| **K4** | Vision retrieval | (torch backbone + ANN) | (torch backbone + ANN) | (out of scope) |
| **L1** | Graph ML (node/graph) | Label propagation | (torch GNN) | (Reduce label prop) |
| **L2** | Graph embeddings | Node2vec (sample+stream) | DeepWalk | (Reduce walks + embed) |
| **M1** | RL / bandits | ε-greedy | LinUCB (contextual) | UCB + LinUCB |
| **N1** | Density / VAE / flows | Histogram | GMM (Reduce EM) | GMM + KDE |
| **N2** | Diffusion / LLM | (torch only) | LLM | (out of scope) |
| **O1** | EDA | CADW describe/hist | ydata-profiling (sample) | — |
| **O2** | Imputation | mean / median | IterativeImputer | mean + IForest-based |
| **O3** | Feature engineering | Polys via expr | Target encoding (OOF) | freq + onehot + target |
| **O4** | Feature selection | variance threshold | Perm importance / Boruta | MI + Boruta + GBDT-imp |
| **O5** | Data validation | Schema check | Statistical (KS/drift) | schema + drift |
| **O6** | Sampling | Random / stratified | Reservoir / balanced | strat + reservoir |
| **O7** | Hyperparameter search | Grid | Bayesian (Optuna, external) | grid + random + bayes |
| **O8** | Explainability | GBDT gain | SHAP (sample) / perm | gain + perm + SHAP |
| **O9** | Fairness | (pending) | DP/EO/TPR-gap (Reduce) | — |
| **O10** | Drift detection | Mean shift | KS/PSI/Chi2 | KS + PSI + Chi2 |
| **O11** | A/B testing | t-test | CUPED | t + MW + CUPED |
| **O12** | Online learning | partial_fit | SGDClassifier / online GBDT | — |
| **O13** | Geospatial | hexbin | h3-aggregations | h3 + hexbin |

---

## Detailed sections (available now + imminent)

### A1. Binary classification

**Description:** predict P(y=1|x) with y ∈ {0, 1}.

**Typical use cases (exhaustive):**
Credit scoring, churn prediction, fraud detection, CTR prediction,
medical diagnosis, spam detection, defect detection, recommendation
relevance, A/B conversion, ad targeting, loan default, propensity
modeling (H1), uplift (H2), anomaly supervised (E2).

**3 tiers:**
- **Baseline:** LogisticRegression (L2) — fast, interpretable
- **Best:** GBDT (LightGBM-style leaf-wise) — SOTA on tabular Kaggle
- **Ensemble:** LogReg (linear) + GBDT (trees) + NB (probabilistic)

**CADW status:**
- ✓ `fit_logistic` — available
- ✓ `fit_gbdt_classifier` (binary) — v0.7.5
- ✓ `fit_gaussian_nb`, `fit_bernoulli_nb`
- ✓ `fit_gbdt_classifier(scale_pos_weight=...)` — v0.9.32 (imbalanced)
- ⏳ sample_weight in `fit_logistic` — D.1.3 extension (patch pending)
- ⏳ calibration (Platt / isotonic) — D.4.6
- ⏳ mutual_info / Boruta — D.6.1

### A2. Multiclass classification

Same tiers as A1, but softmax.

**CADW status:**
- ✓ `fit_gbdt_classifier` (K >= 3, softmax) — v0.7.8
- ✓ `fit_gaussian_nb`
- ⏳ classifier chains (for multilabel) — D.5.2

### A3. Regression

**Description:** predict y ∈ ℝ.

**Typical use cases:** price, demand, delivery time, insurance
claims, sales forecast, energy load, ETA, revenue, salary.

**3 tiers:**
- **Baseline:** Ridge (L2, closed-form)
- **Best:** GBDT (L2)
- **Ensemble:** Ridge (linear extrapolation) + GBDT (nonlinear) +
  GBDT Tweedie (count/tail targets)

**CADW status:**
- ✓ `fit_linear`, `fit_lasso`, `fit_elasticnet`
- ✓ `fit_gbdt_regressor` (L2)
- ⏳ pinball loss (A7) — D.6
- ⏳ Tweedie / Poisson — backlog

### A6. Imbalanced classification

**Description:** binary or multiclass with heavy class imbalance
(0.1% - 5% positive).

**Typical use cases:** fraud, rare disease, defect detection, churn
of high-value clients, click-through-rate on rare event.

**3 tiers:**
- **Baseline:** LogisticRegression (class_weight='balanced')
- **Best:** Cost-sensitive GBDT (`scale_pos_weight=neg/pos`)
- **Ensemble:** Cost-GBDT + Balanced bagging (undersampling + GBDT) +
  Calibrated Logistic

**CADW status:**
- ✓ `fit_gbdt_classifier(scale_pos_weight=...)` — v0.9.32
- ✓ `predict(X, threshold=...)` — v0.9.31
- ⏳ sample_weight in Logistic — D.1.3 extension
- ⏳ calibration (Platt / isotonic) — D.4.6
- ⏳ balanced bagging — via KFold composition

### B1. Flat clustering

**Typical use cases:** customer segmentation, product grouping,
embedding compression, geographic zones, document grouping.

**3 tiers:**
- **Baseline:** k-means
- **Best:** HDBSCAN (density-based, arbitrary shapes)
- **Ensemble:** k-means (spherical) + HDBSCAN (density) + GMM (soft)

**CADW status:**
- ✓ `fit_kmeans` — v0.7.1
- ⏳ MiniBatchKMeans / GMM (EM) — D.5.3
- ⏳ BIRCH CF-tree — D.5.4
- ⏳ HDBSCAN — T3 (sample+stream recipe, Phase H)

### C1. Linear dimensionality reduction

**Typical use cases:** visualization, noise removal, feature
compression, whitening, outlier detection (reconstruction error).

**3 tiers:**
- **Baseline:** PCA (cov + eigh)
- **Best:** Randomized SVD (Halko et al., Halko 2011) — O(N·d·k)
- **Ensemble:** PCA + random projection + AE latent

**CADW status:**
- ✓ `fit_pca` — v0.7.2
- ⏳ randomized SVD — D.6.4
- ⏳ AE — F.3 (Light NNs)

### D1. TS forecast (univariate)

**Typical use cases:** demand, traffic, sales, energy, temperature,
server load, web sessions.

**3 tiers:**
- **Baseline:** Naive / ETS (Holt-Winters)
- **Best:** SARIMA / GBDT-lag
- **Ensemble:** GBDT-lag (direct) + SARIMA (seasonal) + ETS (smooth)

**CADW status:**
- ✓ `NaiveForecaster`, `SARIMAForecaster`, `GBDTForecaster`
- ✓ `TemporalHierarchy` (MinT-WLS, D3)
- ✓ `backtest_forecaster` (CV)
- ⏳ ETS — D.4.3
- ⏳ Theta-method — D.4.4
- ⏳ VAR(p) — D.4.5

### D4. TS classification

**Typical use cases:** ECG arrhythmias, activity recognition,
equipment fault, gesture from sensors, industrial diagnostics.

**3 tiers:**
- **Baseline:** interval stats (mean/std/slope) + GBDT
- **Best:** MiniRocket (Dempster et al. 2021, O(N))
- **Ensemble:** MiniRocket + interval features + GBDT

**CADW status:**
- ✓ interval features (`cadw.ts.stats` tsfresh-subset)
- ⏳ MiniRocket streaming — D.4.1
- ⏳ N-BEATS / 1D-CNN — F.2

### E1. Unsupervised anomaly

**Typical use cases:** fraud detection, network intrusion, sensor
faults, data quality, IoT monitoring.

**3 tiers:**
- **Baseline:** z-score / IQR
- **Best:** Isolation Forest (batched, ψ=256)
- **Ensemble:** IForest (global) + LOF (local) + z-score
  (interpretable)

**CADW status:**
- ✓ `fit_iforest` — v0.9.31 (D.1.1)
- ⏳ LOF — T3 (sample)
- ⏳ One-Class SVM — T4 (sample)
- ⏳ AE-reconstruction — F.3

### E2. Supervised anomaly

Same as A6 (imbalanced) — rare events with labels.

**CADW status:**
- ✓ `fit_gbdt_classifier(scale_pos_weight=...)` — v0.9.32
- ✓ `predict(X, threshold=...)` — v0.9.31

### E3. TS anomaly

**Typical use cases:** monitoring, IoT, metrics, industrial sensors.

**3 tiers:**
- **Baseline:** z-score over rolling window
- **Best:** Matrix Profile (STUMPY-like, O(N log N))
- **Ensemble:** MP + IForest + PELT (change-point)

**CADW status:**
- ⏳ Matrix Profile — D.1.2
- ⏳ PELT — D.4.2
- ✓ IForest on lag features (via composition)

### O1-O11. Data Science (EDA / FE / validation)

**Covered by:**
- O1 EDA: ✓ `df.describe()`, `hist`, `hist2d`, `quantile`,
  `value_counts`
- O2 Cleaning: ✓ `fillna`, `dropna`, `ffill`, `bfill`; ⏳
  IterativeImputer — N34
- O3 FE: ✓ `assign` (planned C.1); ⏳ encoders (D.6.9, C.2-C.4)
- O4 Selection: ✓ `compute_feature_importance`; ✓ perm importance
  (v0.9.31); ⏳ mutual_info / Boruta — D.6.1
- O5 Validation: ✓ schema via pyarrow; ⏳ drift — D.6.2
- O6 Sampling: ✓ KFold, GroupKFold, TimeSeriesSplit
- O7 Hyperparam: ⏳ external (Optuna) — via to_batches
- O8 XAI: ✓ perm importance; ⏳ SHAP (sample, Phase H)
- O9 Fairness: ⏳ fairness metrics — D.1.4
- O10 Drift: ⏳ KS/PSI/Chi2 — D.6.2
- O11 A/B: ⏳ t/MW/CUPED — D.6.2

---

## Pending types (see ALGORITHM_BACKLOG)

- **A4/A5** (multilabel, ordinal) — D.5.2, D.5.1
- **A7** (quantile reg) — GBDT pinball pending
- **B2/B3** (hierarchical / BIRCH) — D.5.4, T1 recipe
- **C2** (t-SNE/UMAP) — T2 recipe
- **D2** (multi TS) — D.4.5 VAR
- **D5** (change-point) — D.4.2 PELT
- **F1/F2** (recsys) — D.2.2 ALS, D.2.3 hybrid
- **G1** (LTR) — D.2.1 LambdaMART
- **H1/H2** (causal/uplift) — D.3.2, D.3.3
- **I1** (survival) — D.3.1 CoxPH
- **J1/J3/J4** (text) — D.6.9 TF-IDF, N11
- **M1** (bandits) — D.6.7 LinUCB
- **N1** (density) — ✓ GMM v0.9.31 (D.6.8)
- **O12/O13** (online / geospatial) — N37, D.6.5

Types **out of scope** (torch-only, no out-of-core fit):
J2, J5, K1-K4, L1, L2, N2.

These CAN still be used via `df.to_batches(format='torch')` —
recipes in `docs/roadmap_research/part2_todo_roadmap.md §T5-T10`.
"""


# ============ 2. BUG_BACKLOG.md ============

BUG_BACKLOG = """# CADW — Bug backlog (v0.9.33)

**Sources:** `.pingis/audits/AUDIT_REPORT_v0930.md`,
`.pingis/audits/AUDIT_v0931.md`,
`docs/roadmap_research/part1_report.md` §1-§2.

**Patch:** v0.9.34 — one bugfix pass, no deferral. If any fix
requires split → D.x.y-1, D.x.y-2 sub-patches, still no deferral.

| ID | Priority | Bug | File | Patch |
|---|---|---|---|---|
| **B1** | **P0** | `_RunStream.refill` quadratic `pa.concat_tables([buffer, chunk])` — до 320 GB copies @ rbs=1M | `cadw/ops/sort.py` | v0.9.34 |
| **B2** | **P0** | `encode_categorical` accumulates Python-set of uniques (O(K), ~600 MB @ 10M uniq) | `cadw/dataframe/_ml_bindings.py` | v0.9.34 |
| **B3** | P1 | ROC-AUC pass2 row-wise `zip(probas, ys)` (CPU-linear) | `cadw/ops/metrics.py:639` | v0.9.34 |
| **B7** | P1 | `_string_hash_column` Python-loop over dict (O(K) per batch) | `cadw/ops/shuffle.py:177` | v0.9.34 |
| **A9** | P1 | ML log wiring — 3/6 models (linear / pca / naive_bayes not wired) | `cadw/ml/{linear,pca,naive_bayes}.py` | v0.9.34 |
| **B4** | P2 | `memory_guard.py:700` fallback `len(set(...))` without extra_factor | `cadw/core/memory_guard.py:700` | v0.9.34 |
| **A10** | P2 | `external_sort` vs `hash_shuffle` empty-input contract not documented | docstrings | v0.9.34 |
| **B6** | P2 | README version drift (v0.8.8 / 0.9.24 / `__version__`) | `README.md` | v0.9.34 |

---

## Per-bug detail

### B1 (P0) — Quadratic `_RunStream.refill`

- **Repro:** 2000 chunks of 1 row: repeated concat = 0.078 s;
  `concat_tables(parts_list)` = 0.0003 s (~250× + superlinear).
- **Contract:** reference `docs/roadmap_research/part1_report.md §2.1`.
- **Impact:** at rbs=200k: 12.9 GB copies; @1M: up to 320 GB (est).
  Peak RSS correct, but I/O pressure → page-cache swap.
- **Fix:** linear buffer via `parts_list` + single `concat_tables`.
- **Test:** `tests/test_refill_linear.py` — bench 2000 chunks; assert
  time ≤ 2× single concat; assert copies ≤ rbs.
- **Bench artifact:** `.pingis/benches/v0934_b1_refill.json`.

### B2 (P0) — `encode_categorical` Python-set

- **Repro:** 10M unique strings — Python-set ~600 MB, not seen by
  MemoryGuard (per-batch ~130 KB).
- **Contract:** `check_state()` before Pass1.
- **Fix:** `estimate_k_sample_arrow` → `check_state(K_est ×
  (value_bytes + 50))`; or Arrow dictionary IPC + join (Pass1 O(1)
  RAM).
- **Test:** `tests/test_encode_categorical_high_card.py` — 100k
  uniques in 10 MB budget → raises `MemoryLimitExceeded`.
- **Bench artifact:** `.pingis/benches/v0934_b2_encode.json`.

### B3 (P1) — ROC-AUC pass2 Python loop

- **Repro:** 500k rows: 0.073 s Python loop vs 0.168 s Arrow
  group_by (but +72 MB, K=N).
- **Fix:** vectorize via `pc` primitives; or keep if not bottleneck —
  **but** must document *why* kept (memory tradeoff).
- **Test:** `tests/test_roc_auc_streaming.py` — 1M rows bench;
  memory ≤ current; time ≤ current.
- **Bench artifact:** `.pingis/benches/v0934_b3_rocauc.json`.

### B7 (P1) — `_string_hash_column` Python loop

- **Repro:** O(K) Python objects per batch on dict keys.
- **Fix:** `dictionary_encode` + hash dict once (Arrow native).
- **Test:** `tests/test_string_hash_native.py` — output equal to
  current; faster on high K.
- **Bench artifact:** `.pingis/benches/v0934_b7_hash.json`.

### A9 (P1) — ML log wiring (3 remaining)

- **Contract:** uniform `[fit] <model> | iter=N/M | loss=X |
  state_bytes=B | rss=Y MiB` via `cadw.ml._logging.log_fit`.
- **Fix:** call `log_fit` in `fit_linear` (one-shot fit — no iter;
  log final only), `fit_pca`, `fit_gaussian_nb` / `fit_multinomial_nb`
  / `fit_bernoulli_nb`.
- **Test:** `tests/test_ml_logs_uniform.py` — grep `log_fit` in all 6
  `cadw/ml/*.py`; run each fit with `CADW_LOG=1`; parse stdout format.

### B4 (P2) — `memory_guard` fallback factor

- **Fix:** add `extra_factor=4` in `len(set(...))` fallback (part1
  §2.3 factor for utf8).
- **Test:** inline test verifies factor.

### A10 (P2) — empty-input contract

- **Fix:** docstrings for `external_sort`, `hash_shuffle`:
  both raise `ValueError` on empty input.
- **Test:** explicit `pytest.raises(ValueError)`.

### B6 (P2) — README version drift

- **Fix:** `README.md` — pull version from `cadw.__version__`
  (single source).
- **Test:** grep that README contains `__version__` string (no
  hardcoded number).
"""


# ============ 3. ALGORITHM_BACKLOG.md ============

ALGO_BACKLOG = """# CADW — Algorithm backlog (v0.9.33)

**Sources:** `docs/roadmap_research/part2_todo_roadmap.md` (NEEDS
N1-N40, TODO T1-T12),
`docs/roadmap_research/consolidated_roadmap.md` §6 (Phase D),
`.pingis/algorithm_notes/algorithm_notes.md`.

**Already done:** 4 (D.1.1 IForest v0.9.31, D.1.3 cost-sensitive
GBDT v0.9.32, D.6.3 perm importance v0.9.31, D.6.8 GMM density
v0.9.31).

**Pending:** 24 items. All must be implemented — see
`.pingis/tasks/TASK_v0935_algorithms.md` for exact assignment.

---

## Pending table

| ID | Priority | Complexity | Name | State | Test criterion |
|---|---|---|---|---|---|
| D.1.2 | P0 | M | Matrix Profile streaming | O(window) | finds 5 anomalies on synthetic |
| D.1.4 | P1 | S | fairness metrics DP/EO/TPR-gap | O(K) | = fairlearn on reference |
| D.2.1 | P0 | M | LambdaMART lambdarank (GBDT) | O(model) | NDCG@10 on LETOR-like |
| D.2.2 | P0 | M | ALS (implicit) iterative Reduce | O((U+V)·k) | RMSE/NDCG ~ implicit ALS |
| D.2.3 | P1 | M | hybrid MF + content | O(model) | NDCG > ALS-only |
| D.3.1 | P0 | M | CoxPH partial likelihood + Sort | O(d) | c-index ~ lifelines |
| D.3.2 | P0 | M | T/S-learner + propensity | O(model) | ATE within CI on synthetic |
| D.3.3 | P1 | M | uplift trees (Qini) | O(trees) | Qini-coef > T-learner |
| D.4.1 | P0 | M | MiniRocket streaming | O(#kernels) fixed | accuracy within 2% of reference |
| D.4.2 | P0 | M | PELT online change-point | O(window) | detects 3 points on synthetic |
| D.4.3 | P1 | M | ETS (Holt-Winters) | O(k) seasonal | MASE ~ statsmodels |
| D.4.4 | P1 | S | Theta-method | O(1) | MASE on M3-like |
| D.4.5 | P1 | M | VAR(p) | O(p·k²) | RMSE ~ statsmodels |
| D.4.6 | P1 | M | calibration Platt/isotonic | O(d) / O(N_bins) | Brier/ECE not worse |
| D.5.1 | P1 | M | ordinal logistic cumulative link | O(d·K) | MAE ~ mord OrdinalRidge |
| D.5.2 | P1 | S | classifier chains | O(K·model) | subset accuracy ≥ OvR |
| D.5.3 | P1 | M | MiniBatchKMeans / GMM (EM) | O(K·d) / O(K·d²) | inertia/likelihood ~ sklearn |
| D.5.4 | P1 | L | BIRCH CF-tree streaming | O(#CF nodes) | = sklearn Birch on sample |
| D.6.1 | P1 | M | mutual_info / variance / Boruta | O(d) | top-k = sklearn |
| D.6.2 | P1 | M | drift (KS/PSI/Chi2), A/B (t/MW/CUPED) | O(1) | p-value ~ scipy |
| D.6.4 | P1 | M | randomized SVD | O(N·d·k) | explained var ~ sklearn PCA |
| D.6.5 | P1 | S | geo-aggregations h3/hexbin | O(1) | map = h3-equivalent |
| D.6.6 | P1 | L | LDA (variational/EM) | O(K·V) | perplexity ~ gensim sample |
| D.6.7 | P1 | S | LinUCB contextual bandit | O(d²·arms) | regret ~ baseline |
| D.6.9 | P0 | M | TF-IDF on disk | O(V) | linear classifier = sklearn |

---

## Universal requirements per algorithm

**Location:**
- Model class + `fit_<name>` in `cadw/ml/<name>.py` (models) or
  `cadw/ops/<name>.py` (primitives).
- Mixin method `df.fit_<name>(...)` appended to
  `cadw/dataframe/_ml_bindings.py` (§13.4/329 append pattern).

**API contract:**
- `class <Name>Model` with `.predict(df)`, `.predict_proba(df)` if
  applicable; `.to_dict()` for serialization.
- `fit_<name>(df, X_cols, y_col, *, ..., memory_guard=None) -> <Name>Model`.
- All numeric inputs via `_stack_numeric` / `_column_to_f64`
  (batched).
- Every `check_batch` call must include `extra_factor ≥ 2.0`
  (measured for the specific op).

**RAM constraints:**
- State O(1) / O(d²) / O(model) — no materialization of full dataset.
- `batch_memory_factor()` override if per-batch peak > yielded
  `batch.nbytes` (F9 pattern, v0.9.32).
- `check_state(est_state_bytes)` preflight before any O(n) allocation.

**Tests (≥8 per algorithm):**
1. Synthetic known-output test.
2. Reference-comparison vs sklearn/statsmodels/lifelines (accuracy
   within documented tolerance).
3. Edge: empty input, 1 row, all null, all same.
4. Determinism: same seed → same result.
5. Multi-batch: chunk=1 vs chunk=1000 → same result.
6. RAM: peak RSS via `psutil` ≤ 2× `MemoryGuard.pred_total`.
7. Guard: high-cardinality / large-N → `MemoryLimitExceeded` if over
   budget (not crash).
8. Source contract: `inspect.getsource` matches design docstring
   (semantic).

**Bench (mandatory):**
- `scripts/bench_<name>.py` — N=1M synthetic; peak RSS, time vs
  reference.
- Output: `.pingis/benches/<name>.json` with raw numbers.

**Docs:**
- `.pingis/algorithm_notes/<name>.md` — design rationale, sources
  (paper + docs + benchmarks), edge cases, deferral reasons (if any
  — must be explicit time estimate).
- `docs/ML_TASKS_TAXONOMY.md` — update status from ⏳ to ✓.
- `docs/tasks/XX_<name>.md` (optional Phase E) — recipe.

**Logs (iterative models):**
- `log_fit` from `cadw.ml._logging` per iteration; gated by
  `CADW_LOG=1`.
- Format: `[fit] <name> | iter=N/M | loss=X.XXXXXX | state_bytes=B |
  rss=Y MiB`.

**Version:** each algorithm = 1 patch (or D.x.y-1 / D.x.y-2 if >500
LOC). Version bumps +0.0.1.

**No refusal:** PINGIS implements all 24 items. If a specific item
needs an external package > 10 MB (e.g. `h3`), PINGIS documents the
requirement in the patch message + proposes dependency; user
approves/rejects. But default: implement natively.

**Substitution allowed** with justification:
- If a better algorithm exists (paper + benchmark), PINGIS may
  substitute, but must:
  1. Cite source (paper + docs + benchmark).
  2. Show benchmark vs original plan.
  3. Prove RAM class equal or better.
  4. Same or better test coverage.

**Addition allowed** with justification:
- PINGIS may add algorithms not in the list if they unlock a task
  type or improve an existing one. Must be justified in patch
  message + `.pingis/experiments/<name>.md`.

---

## Out-of-scope (T5-T10)

Explicitly NOT implemented (torch-only or degenerate):

- **T5** NCF / SASRec — torch batching only (via
  `to_batches(format='torch')`).
- **T6** BERT / LLM fine-tune — torch.
- **T7** CV (vision models) — torch.
- **T8** GNN — torch (classic Reduce label-prop OK, but not GNN).
- **T9** RL (non-bandit) — torch.
- **T10** Generative (VAE/GAN/diffusion/LLM) — torch.

Recipes for out-of-scope tasks live in
`docs/roadmap_research/part2_todo_roadmap.md §3`.
"""


# ============ 4. PINGIS task v0.9.34 ============

TASK_V0934 = """# TASK v0.9.34 — Bug fix pass (8 items, no deferral)

**Branch:** `pingis/dev-audit` (same as v0.9.32)
**Baseline:** HEAD = `38c8eb7 v0.9.32`, tests: 2557 passed, 2 skipped.
**Patch:** **one commit** `v0.9.34: bugfix pass (8 items)`.
**Rejection/deferral:** NOT ALLOWED. Each item is either fixed or
rejected with **explicit proof** (repro + measurement showing it's not
a bug). "wontfix" without measurement → rejected commit.

---

## Bug list (8 items)

Exact list from `docs/BUG_BACKLOG.md`. Full repro + contract per bug
is there.

| # | ID | Priority | Bug | File |
|---|---|---|---|---|
| 1 | B1 | P0 | Quadratic `_RunStream.refill` | `cadw/ops/sort.py` |
| 2 | B2 | P0 | `encode_categorical` Python-set | `cadw/dataframe/_ml_bindings.py` |
| 3 | B3 | P1 | ROC-AUC pass2 Python loop | `cadw/ops/metrics.py:639` |
| 4 | B7 | P1 | `_string_hash_column` Python loop | `cadw/ops/shuffle.py:177` |
| 5 | A9 | P1 | ML log wiring (3/6) | `cadw/ml/{linear,pca,naive_bayes}.py` |
| 6 | B4 | P2 | `memory_guard` fallback factor | `cadw/core/memory_guard.py:700` |
| 7 | A10 | P2 | empty-input contract docs | docstrings |
| 8 | B6 | P2 | README version drift | `README.md` |

---

## Requirements per bug

For each item:

1. **Repro first.** Write `tests/test_v0934_<id>_<name>.py` (or
   inline test) with the failing case from `docs/BUG_BACKLOG.md`.
   Run it. Confirm it fails **for the documented reason**.

2. **Fix.** Apply the documented fix (see `BUG_BACKLOG.md` per-item
   section).

3. **Test green.** After fix, run test — must pass.

4. **Fast suite.** `uv run pytest -q -m "not slow"` → **all green**
   (2557+ passed, no regressions).

5. **Bench artifact.** For B1/B2/B3/B7: `scripts/bench_v0934_<id>.py`
   → output `.pingis/benches/v0934_<id>.json` with before/after
   numbers (time, peak RSS, copies).

6. **Commit message.** One final commit includes per-bug one paragraph
   (what was bug, what fixed, numbers before/after).

---

## Reporting structure

All artifacts under `.pingis/`:

```
.pingis/
├── audits/
│   └── v0934_bugfix_report.md         ← summary: 8 items status
├── benches/
│   ├── v0934_b1_refill.json
│   ├── v0934_b2_encode.json
│   ├── v0934_b3_rocauc.json
│   └── v0934_b7_hash.json
└── experiments/
    └── v0934_notes.md                  ← edge cases / surprises
```

`v0934_bugfix_report.md` format:

```markdown
# v0.9.34 — Bug fix report

| ID | Status | Fix | Before | After | Test |
|---|---|---|---|---|---|
| B1 | fixed | linear parts_list | 12.9 GB copies | 0.13 GB copies | pass |
| B2 | fixed | K_est + check_state | OOM @ 10M uniq | MemoryLimitExceeded early | pass |
| B3 | fixed / rejected (measurement) | ... | ... | ... | pass |
...
```

---

## Anti-patterns (auto-reject)

- **Skipping test.** "Test would fail without fix, but I trust it" →
  reject.
- **Wontfix without measurement.** "This is fine" → reject unless
  measured (RSS/time/bug-free).
- **Partial fix with deferral.** "Fixed B1, B2 partial, rest next
  patch" → reject. Fix all 8, or provide explicit rejection proof
  for the deferred ones.
- **Changing test expectations to pass.** Test suite is contract.
- **Silent expansion of WHITELISTED_FUNCS.** No — each new entry must
  be justified inline.

---

## Acceptance

- All 8 bugs fixed (or rejected with measurement).
- Fast suite: 2557+ passed, 0 failed.
- Slow suite: 25+ passed, 0 failed.
- Bench artifacts present in `.pingis/benches/`.
- One commit `v0.9.34: bugfix pass (8 items)` on `pingis/dev-audit`.
- Report `v0934_bugfix_report.md` complete.

After acceptance → proceed to TASK_v0935 (algorithms).
"""


# ============ 5. PINGIS task v0.9.35 ============

TASK_V0935 = """# TASK v0.9.35+ — Algorithm implementation (24 items, no deferral)

**Branch:** `pingis/dev-audit`
**Baseline:** HEAD = `v0.9.34` (after bug pass).
**Patches:** 24 individual patches (D.1.2, D.1.4, D.2.1, ..., D.6.9).
Version bumps: v0.9.35 → v0.9.36+ (each patch +0.0.1).
**Rejection/deferral:** Only for **explicitly out-of-scope** T5-T10
(torch-only, listed in `docs/ALGORITHM_BACKLOG.md`). Everything else
must be implemented.

---

## Algorithm list (24 items)

Exact list from `docs/ALGORITHM_BACKLOG.md`. Full requirements
(API, RAM, tests, bench, docs) — same file, "Universal requirements".

| # | ID | Priority | Complexity | Name | Patch |
|---|---|---|---|---|---|
| 1 | D.1.2 | P0 | M | Matrix Profile streaming | v0.9.35 |
| 2 | D.1.4 | P1 | S | fairness metrics DP/EO/TPR-gap | v0.9.35 |
| 3 | D.2.1 | P0 | M | LambdaMART lambdarank | v0.9.35 |
| 4 | D.2.2 | P0 | M | ALS (implicit) iterative Reduce | v0.9.36 |
| 5 | D.2.3 | P1 | M | hybrid MF + content | v0.9.36 |
| 6 | D.3.1 | P0 | M | CoxPH partial likelihood + Sort | v0.9.36 |
| 7 | D.3.2 | P0 | M | T/S-learner + propensity | v0.9.37 |
| 8 | D.3.3 | P1 | M | uplift trees (Qini) | v0.9.37 |
| 9 | D.4.1 | P0 | M | MiniRocket streaming | v0.9.37 |
| 10 | D.4.2 | P0 | M | PELT online change-point | v0.9.38 |
| 11 | D.4.3 | P1 | M | ETS (Holt-Winters) | v0.9.38 |
| 12 | D.4.4 | P1 | S | Theta-method | v0.9.38 |
| 13 | D.4.5 | P1 | M | VAR(p) | v0.9.38 |
| 14 | D.4.6 | P1 | M | calibration Platt/isotonic | v0.9.39 |
| 15 | D.5.1 | P1 | M | ordinal logistic | v0.9.39 |
| 16 | D.5.2 | P1 | S | classifier chains | v0.9.39 |
| 17 | D.5.3 | P1 | M | MiniBatchKMeans / GMM (EM) | v0.9.39 |
| 18 | D.5.4 | P1 | L | BIRCH CF-tree streaming | v0.9.40 |
| 19 | D.6.1 | P1 | M | mutual_info / variance / Boruta | v0.9.40 |
| 20 | D.6.2 | P1 | M | drift + A/B tests | v0.9.40 |
| 21 | D.6.4 | P1 | M | randomized SVD | v0.9.40 |
| 22 | D.6.5 | P1 | S | geo-aggregations h3/hexbin | v0.9.40 |
| 23 | D.6.6 | P1 | L | LDA | v0.9.41 |
| 24 | D.6.7 | P1 | S | LinUCB contextual bandit | v0.9.41 |
| 25 | D.6.9 | P0 | M | TF-IDF on disk | v0.9.41 |

**Batching:** S items (D.1.4, D.4.4, D.5.2, D.6.5, D.6.7) — 2-3 per
patch. M items — 1-2 per patch. L items (D.5.4, D.6.6) — 1 per patch,
may split D.x.y-1/D.x.y-2.

**Verify:** before starting each patch — `git log --oneline -1` =
previous version, run baseline fast suite (~170s), note counts.

---

## Requirements per algorithm

Exact same as `docs/ALGORITHM_BACKLOG.md` "Universal requirements":

1. Model + fit function in `cadw/ml/<name>.py` or
   `cadw/ops/<name>.py`.
2. Mixin method in `cadw/dataframe/_ml_bindings.py` (§13.4/329
   append pattern).
3. ≥8 tests (synthetic, reference, edge, determinism, multi-batch,
   RAM, guard, source contract).
4. Bench script `scripts/bench_<name>.py` — 1M rows, peak RSS vs
   reference; artifact `.pingis/benches/<name>.json`.
5. `.pingis/algorithm_notes/<name>.md` — design + sources +
   benchmarks.
6. Update `docs/ML_TASKS_TAXONOMY.md` — status ⏳ → ✓.
7. Update `docs/roadmap.json` — D.x.y → done.
8. Uniform ML log (`log_fit` helper) if iterative.
9. CHANGELOG entry (### v0.9.XX).

---

## Special requirements

### Research first (per algorithm)

Before writing code:

1. **Read the paper.** arXiv / journal. Cite key equations in
   `.pingis/algorithm_notes/<name>.md`.
2. **Read reference library source** (sklearn, statsmodels, lifelines,
   etc.). Note API contract and any surprising edge cases.
3. **Fetch benchmarks.** Kaggle, papers with code, UCR archive, etc.
   Record in algorithm_notes.
4. **Design RAM model.** State size, per-batch peak, `check_batch`
   factors. Document before coding.
5. **Pick primitive composition.** Reduce / HashShuffle / Sort —
   which fits, which doesn't.

### Sources to check (mandatory)

| Task class | Must read |
|---|---|
| ML models | sklearn docs + paper + Kaggle solution |
| TS forecast | statsmodels docs + M4/M5 comp writeups |
| Anomaly | sklearn docs + ODDS benchmark + paper |
| Survival | lifelines + scikit-survival + paper |
| Causal | EconML + CausalML docs + papers |
| Recsys | implicit lib + Kaggle RecSys comps |
| LTR | LightGBM lambdarank docs + LETOR paper |
| TS class | sktime + UCR archive + MiniRocket paper |
| Drift | evidently + alibi-detect + KS/PSI papers |
| Text | sklearn TfidfVectorizer + sklearn source |

If a public benchmark exists (KDD Cup, Kaggle, UCR, M4/M5, LETOR,
ODD, MovieLens) — cite the number in algorithm_notes. Never invent
benchmarks — if no verified source, say "no verified citation".

### Substitution / addition (allowed, justified)

- **Substitution:** if a better algorithm exists (paper + benchmark +
  RAM class equal-or-better), substitute. Must include:
  - Original plan vs substitute name
  - Paper source
  - Benchmark comparing
  - RAM model comparison
- **Addition:** may add algorithms not in list if they unlock a task
  type. Justify in `.pingis/experiments/<name>.md`.

### External dependencies

- If an algorithm requires a package > 10 MB (e.g. `h3`, `stumpy`):
  - Default: implement natively (preferred per project constraint).
  - If native is impractical: propose dependency in the patch message;
    user approves/rejects. Do not silently install.
- Small deps (< 10 MB, pure Python, MIT/BSD/Apache) may be added if
  the alternative would be > 300 LOC of custom code.

---

## Reporting structure

```
.pingis/
├── algorithm_notes/
│   ├── d1_2_matrix_profile.md
│   ├── d1_4_fairness.md
│   └── ... (24 files)
├── benches/
│   ├── d1_2_matrix_profile.json
│   └── ... (24 files)
├── experiments/
│   ├── v0935_summary.md           ← 24-item progress table
│   └── <name>.md                  ← per-algorithm edge cases
└── audits/
    └── v0935_algo_report.md       ← final summary (when all done)
```

`v0935_summary.md`:
```markdown
# v0.9.35+ — Algorithm progress

| ID | Version | Status | Tests | Bench (RSS/time) | Reference |
|---|---|---|---|---|---|
| D.1.2 | v0.9.35 | ✓ | 9 | 45 MB / 2.1 s | stumpy: 42 MB / 1.9 s |
| D.1.4 | v0.9.35 | ✓ | 8 | 2 MB / 0.05 s | fairlearn: 2 MB / 0.04 s |
| ... |
| D.6.9 | v0.9.41 | ✓ | 10 | 120 MB / 8 s | sklearn: 130 MB / 6 s |
```

---

## Anti-patterns (auto-reject)

- **Skip tests.** Every algorithm ≥8 tests, no exceptions.
- **`pytest.skip` / `pytest.mark.skip`** unless external tool absent
  AND noted in report.
- **Fake benchmark.** Numbers from empty data, single batch, or
  cherry-picked.
- **No reference comparison.** "I trust my implementation" → reject.
- **Silent algorithm substitution.** Replacing plan without
  justification + benchmark → revert.
- **`import pandas` in `cadw/`.** Only in tests.
- **`.to_pylist()` on batch without `check_batch(extra_factor≥2)`**
  → reject (AST guard).
- **Materializing full dataset.** Peak RSS > 2× `pred_total` → reject.
- **Partial deliverable.** "D.1.2 mostly works" → reject. Complete or
  split into sub-patches (`D.x.y-1`, `D.x.y-2`) with **both landed**.
- **Hardcoded test expectations matched to current code.** Not
  contract-driven. Use `sklearn`/`statsmodels` reference.

---

## Acceptance (per patch)

Each patch:

- All algorithm's tests pass.
- Fast suite green (2557+ passed).
- Slow suite green (25+ passed).
- Bench artifact under `.pingis/benches/`.
- Algorithm_notes under `.pingis/algorithm_notes/`.
- ML log uniform (if iterative).
- `docs/ML_TASKS_TAXONOMY.md` status update.
- `docs/roadmap.json` item → done.
- CHANGELOG entry.
- One commit.

## Acceptance (whole task)

When all 24 items done:

- Fast suite: 2557+ 24×8 = **~2750 passed**.
- Slow suite: 25+ passed.
- All `D.x.y` items done in `docs/roadmap.json`.
- `.pingis/audits/v0935_algo_report.md` complete.
- Phase D → `completed` in `docs/roadmap.json`.
- Tag `v0.9.41` (or actual final version).

After acceptance → Phase E (recipes) + Phase G (tech debt close) per
`docs/roadmap_research/consolidated_roadmap.md`.
"""


# ============ helpers ============

def write_file(path, content):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    print(f"  wrote {path} ({len(content.splitlines())} lines)")


def update_roadmap():
    print("\n--- update roadmap.json ---")
    data = json.loads(Path(ROADMAP).read_text())

    # Phase I: Bug backlog
    if not any(p.get("id") == "I" for p in data["phases"]):
        phase_i = {
            "id": "I",
            "name": "Bug backlog (part1_report §1)",
            "description": "Bug fixes from part1_report.md audit — no "
                           "deferral allowed. Fix or reject with "
                           "measurement.",
            "status": "in_progress",
            "version_range": "v0.9.34",
            "items": [
                {"id": "I.1", "name": "B1 — _RunStream.refill quadratic concat",
                 "patch": "v0.9.34", "status": "planned",
                 "complexity": "S", "file": "cadw/ops/sort.py",
                 "source": "part1_report.md B1 (P0)"},
                {"id": "I.2", "name": "B2 — encode_categorical O(K) Python-set",
                 "patch": "v0.9.34", "status": "planned",
                 "complexity": "S",
                 "file": "cadw/dataframe/_ml_bindings.py",
                 "source": "part1_report.md B2 (P0)"},
                {"id": "I.3", "name": "B3 — ROC-AUC pass2 Python loop",
                 "patch": "v0.9.34", "status": "planned",
                 "complexity": "S", "file": "cadw/ops/metrics.py",
                 "source": "part1_report.md B3 (P1)"},
                {"id": "I.4", "name": "B7 — _string_hash_column O(K) Python",
                 "patch": "v0.9.34", "status": "planned",
                 "complexity": "S", "file": "cadw/ops/shuffle.py",
                 "source": "part1_report.md B7 (P1)"},
                {"id": "I.5", "name": "A9 — ML log wiring (3 remaining)",
                 "patch": "v0.9.34", "status": "planned",
                 "complexity": "S",
                 "file": "cadw/ml/{linear,pca,naive_bayes}.py",
                 "source": "AUDIT_v0931 A9 (P1)"},
                {"id": "I.6", "name": "B4 — memory_guard fallback factor",
                 "patch": "v0.9.34", "status": "planned",
                 "complexity": "S", "file": "cadw/core/memory_guard.py",
                 "source": "part1_report.md B4 (P2)"},
                {"id": "I.7", "name": "A10 — empty-input contract docs",
                 "patch": "v0.9.34", "status": "planned",
                 "complexity": "S", "file": "docstrings",
                 "source": "AUDIT_v0930 A10 (P2)"},
                {"id": "I.8", "name": "B6 — README version drift",
                 "patch": "v0.9.34", "status": "planned",
                 "complexity": "S", "file": "README.md",
                 "source": "part1_report.md B6 (P2)"},
            ],
        }
        data["phases"].append(phase_i)
        print("  added Phase I (8 bug items)")

    # Phase D: ensure status=in_progress, count done/pending
    phd = next(p for p in data["phases"] if p["id"] == "D")
    if phd.get("status") == "planned":
        phd["status"] = "in_progress"
        phd["started_at"] = "2026-10-23"
        print("  D.status → in_progress")

    done = [i["id"] for i in phd["items"] if i.get("status") == "done"]
    pending = [i["id"] for i in phd["items"] if i.get("status") != "done"]
    print(f"  D items: {len(done)} done, {len(pending)} pending")

    data["updated"] = TODAY
    Path(ROADMAP).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )


def update_context():
    print("\n--- update CONTEXT.md ---")
    p = Path(CONTEXT)
    src = p.read_text()
    if "docs/ML_TASKS_TAXONOMY.md" in src:
        print("  already linked")
        return
    # Append navigation section
    add = """

## Documentation navigation (v0.9.33)

- **`docs/ML_TASKS_TAXONOMY.md`** — master navigator: 54 ML task
  types → 3-tier algorithms (baseline / best / ensemble) + CADW
  status. **Start here** when picking a model.
- **`docs/BUG_BACKLOG.md`** — bug backlog (Phase I), 8 items, exact
  patches.
- **`docs/ALGORITHM_BACKLOG.md`** — pending Phase D, 24 items, with
  universal test/bench requirements.
- **`docs/ROADMAP.md`** — consolidated phase roadmap (A-I).
- **`docs/roadmap_research/`** — PINGIS research (part1/part2/
  consolidated roadmap).
- **`.pingis/tasks/`** — PINGIS work assignments (TASK_v0934 bugs,
  TASK_v0935 algorithms).
- **`.pingis/audits/`** — PINGIS audit reports.
- **`.pingis/algorithm_notes/`** — per-algorithm design notes.
- **`.pingis/benches/`** — raw bench numbers.

## Current state (v0.9.33)

- Bugs backlog: 8 items → TASK_v0934 (bug pass, no deferral).
- Algorithm backlog: 24 items → TASK_v0935 (algorithm pass, no
  deferral).
- PINGIS workspace: `.pingis/` — all artifacts centralised.
"""
    p.write_text(src.rstrip() + add)
    print(f"  appended navigation section")


def bump_version():
    print(f"\n--- version → {TARGET_VERSION} ---")
    pp = Path(PYPROJECT)
    src = pp.read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', src, re.M)
    old = m.group(1) if m else "?"
    src = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\1"{TARGET_VERSION}"',
        src, count=1, flags=re.M,
    )
    pp.write_text(src)
    print(f"  pyproject.toml: {old} → {TARGET_VERSION}")

    init = Path(INIT)
    src = init.read_text()
    src = re.sub(
        r'__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{TARGET_VERSION}"',
        src, count=1,
    )
    init.write_text(src)

    import importlib
    import cadw
    importlib.reload(cadw)
    if cadw.__version__ != TARGET_VERSION:
        raise SystemExit(
            f"reload: {cadw.__version__} != {TARGET_VERSION}"
        )
    print(f"  ✓ cadw.__version__ == {TARGET_VERSION}")


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


# ============ main ============

def main():
    print("=== v0.9.33: docs taxonomy + backlog + PINGIS tasks ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"branch: {br}")

    verify_state()

    print("\n--- write docs ---")
    write_file("docs/ML_TASKS_TAXONOMY.md", ML_TAXONOMY)
    write_file("docs/BUG_BACKLOG.md", BUG_BACKLOG)
    write_file("docs/ALGORITHM_BACKLOG.md", ALGO_BACKLOG)

    print("\n--- write PINGIS tasks ---")
    write_file(".pingis/tasks/TASK_v0934_bugs.md", TASK_V0934)
    write_file(".pingis/tasks/TASK_v0935_algorithms.md", TASK_V0935)

    update_roadmap()
    update_context()
    bump_version()
    run_tests()

    print("\n--- verify_whitelist ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.patch_utils import verify_whitelist; "
         "verify_whitelist(raise_on_orphan=True); print('WL OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("whitelist failed")
    print(r.stdout.strip())

    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.33: docs — ML task taxonomy + bug/algorithm backlog "
        "+ PINGIS tasks. Docs + roadmap.json data only; 0 production "
        "code change. "
        "New docs: "
        "(1) docs/ML_TASKS_TAXONOMY.md — master navigator for 54 ML "
        "task types (A1-O13). 3 tiers per type: baseline / "
        "best-in-class / ensemble candidates. Detailed sections for "
        "types available today (A1/A2/A3/A6/B1/C1/D1/D4/E1/E2/E3/"
        "O1-O11); stub sections for pending types link to "
        "ALGORITHM_BACKLOG D.x.y. Out-of-scope types (J2/J5/K1-K4/"
        "L1/L2/N2) explicitly marked torch-only. "
        "(2) docs/BUG_BACKLOG.md — 8-item bug backlog (B1/B2/B3/B7/"
        "A9/B4/A10/B6) with per-item repro, contract, fix proposal, "
        "test requirement, bench artifact — mirrors roadmap Phase I. "
        "(3) docs/ALGORITHM_BACKLOG.md — 24 pending Phase D items "
        "with state class, priority, complexity, test criterion, "
        "bench requirement, universal implementation contract. "
        "(4) .pingis/tasks/TASK_v0934_bugs.md — PINGIS bug pass "
        "assignment. Exact 8-item list, no deferral allowed, "
        "rejection only with measurement proof. Reporting structure "
        "under .pingis/audits + .pingis/benches + .pingis/experiments. "
        "(5) .pingis/tasks/TASK_v0935_algorithms.md — PINGIS algorithm "
        "pass assignment. 24 items in 7 patch groups (v0.9.35–v0.9.41). "
        "Universal requirements: model+fit function, mixin method, "
        "≥8 tests, bench artifact, algorithm notes, uniform ML log, "
        "roadmap/taxonomy update, CHANGELOG. Substitution allowed "
        "with benchmark + citation; addition allowed with "
        "justification. Only T5-T10 (torch-only) explicitly "
        "out-of-scope. "
        "Roadmap.json: added Phase I (bug backlog, 8 items), "
        "Phase D.status → in_progress. CONTEXT.md: docs navigation "
        "section added. Version bump 0.9.32 → 0.9.33. "
        "Tests: 2557 passed, 2 skipped (fast)."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)
    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"\n{log}")
    print("\nv0.9.33 COMPLETE. PINGIS ready for TASK_v0934.")


if __name__ == "__main__":
    main()