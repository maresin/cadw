# CADW — ML Tasks Taxonomy (navigator)

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
| **O9** | Fairness | DP/EO/TPR-gap | DF-based (group sums) | ✓ v0.9.35 (D.1.4) |
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
- ✓ ETS — v0.9.38 (D.4.3)
- ✓ Theta-method — v0.9.38 (D.4.4)
- ✓ VAR(p) — v0.9.38 (D.4.5)

### D4. TS classification

**Typical use cases:** ECG arrhythmias, activity recognition,
equipment fault, gesture from sensors, industrial diagnostics.

**3 tiers:**
- **Baseline:** interval stats (mean/std/slope) + GBDT
- **Best:** MiniRocket (Dempster et al. 2021, O(N))
- **Ensemble:** MiniRocket + interval features + GBDT

**CADW status:**
- ✓ interval features (`cadw.ts.stats` tsfresh-subset)
- ✓ MiniRocket streaming — v0.9.37 (D.4.1)
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
- ✓ Matrix Profile — v0.9.35 (D.1.2)
- ✓ PELT — v0.9.38 (D.4.2)
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
- O9 Fairness: ✓ `df.fairness_metrics` — v0.9.35 (D.1.4)
- O10 Drift: ⏳ KS/PSI/Chi2 — D.6.2
- O11 A/B: ⏳ t/MW/CUPED — D.6.2

---

## Pending types (see ALGORITHM_BACKLOG)

- **A4/A5** (multilabel, ordinal) — D.5.2, D.5.1
- **A7** (quantile reg) — GBDT pinball pending
- **B2/B3** (hierarchical / BIRCH) — D.5.4, T1 recipe
- **C2** (t-SNE/UMAP) — T2 recipe
- **D2** (multi TS) — ✓ VAR v0.9.38 (D.4.5)
- **D5** (change-point) — ✓ PELT v0.9.38 (D.4.2)
- **F1/F2** (recsys) — ✓ ALS v0.9.36 (D.2.2), ✓ hybrid MF v0.9.36 (D.2.3)
- **G1** (LTR) — ✓ LambdaMART v0.9.35 (D.2.1)
- **H1/H2** (causal/uplift) — ✓ T/S-learner v0.9.37 (D.3.2), ✓ uplift v0.9.37 (D.3.3)
- **I1** (survival) — ✓ CoxPH v0.9.36 (D.3.1)
- **J1/J3/J4** (text) — D.6.9 TF-IDF, N11
- **M1** (bandits) — D.6.7 LinUCB
- **N1** (density) — ✓ GMM v0.9.31 (D.6.8)
- **O12/O13** (online / geospatial) — N37, D.6.5

Types **out of scope** (torch-only, no out-of-core fit):
J2, J5, K1-K4, L1, L2, N2.

These CAN still be used via `df.to_batches(format='torch')` —
recipes in `docs/roadmap_research/part2_todo_roadmap.md §T5-T10`.
