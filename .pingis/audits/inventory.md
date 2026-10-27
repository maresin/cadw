# cadw Inventory — ML models (auto-generated)

ML modules: 47

| file | LOC | summary | guard | fit_funcs | classes |
|---|---|---|---|---|---|
| als.py | 207 | Implicit-feedback ALS (D.2.2, Phase D, v0.9.36). | check_batch | fit_als | ALSModel |
| anomaly_pipeline.py | 286 | AnomalyPipeline: detrend / deseasonalize before anomaly detection. | delegate |  | AnomalyPipelineResult, AnomalyPipeline |
| autoencoder.py | 280 | Autoencoder (MLP): reconstruction-based DL anomaly detection. | delegate | fit_autoencoder | AutoencoderResult |
| binseg.py | 197 | BinarySegmentation: greedy top-down change-point detection (L2 cost). | delegate | fit_binseg | BinSegResult |
| birch.py | 235 | BIRCH CF-tree streaming clustering (D.5.4, v0.9.40). | check_batch | fit_birch | _CF, _Node, BIRCHModel |
| bocpd.py | 333 | BOCPD: Bayesian Online Change-Point Detection (Student-t likelihood). | delegate | fit_bocpd | BOCPDResult |
| calibration.py | 251 | Probability calibration: Platt scaling and isotonic regression (D.4.6, | check_batch | fit_calibration | CalibratedModel |
| chains.py | 171 | Classifier chains for multi-label classification (D.5.2, v0.9.39). | check_batch | fit_chains | ChainsModel |
| coxph.py | 220 | Cox proportional hazards (D.3.1, Phase D, v0.9.36). | check_batch | fit_coxph | CoxPHModel |
| cusum.py | 160 | CUSUM: cumulative sum control chart for changepoint detection. | delegate | fit_cusum | CUSUMResult |
| drift.py | 181 | Distribution drift and A/B tests (D.6.2, v0.9.40). | delegate | fit_drift, fit_ab_test | DriftABModel |
| dwt_mlead.py | 294 | DWT-MLEAD: Multilevel DWT-based anomaly detection (Thill, Konen, Baeck | delegate | fit_dwt_mlead | DwtMleadResult |
| eif.py | 360 | Extended Isolation Forest (EIF) anomaly detection. | delegate | fit_eif | EIFResult, _EIFTree, _EIFForest |
| fairness.py | 178 | Fairness metrics (D.1.4, Phase D, v0.9.35). | check_batch |  | FairnessReport |
| feature_select.py | 239 | Feature selection: mutual information, variance, Boruta (D.6.1). | check_batch | fit_feature_select | FeatureSelectorModel |
| gbdt.py | 1531 | Gradient Boosting Decision Trees — regression + classification (v0.7.5 | delegate | fit_gbdt_regressor, fit_gbdt_classifier, fit_gbdt_multiclass | _MinMaxOp, _AllNodeHistOp, _AssignLeavesOp, _LeafValuesOp, GBDTRegressor, GBDTClassifier, GBDTMulticlassClassifier |
| geo.py | 163 | Geospatial aggregations and distance (D.6.5, v0.9.40). | check_batch | fit_geo_agg | GeoAggModel |
| gmm.py | 216 | GMM density estimation (D.6.8, Phase D, v0.9.31). | check_batch | fit_gmm | GMMModel |
| hampel.py | 140 | Hampel filter: rolling median + MAD anomaly detector. | delegate | fit_hampel | HampelResult |
| hybrid_mf.py | 147 | Hybrid matrix factorization + content features (D.2.3, v0.9.36). | delegate | fit_hybrid_mf | HybridMFModel |
| iforest.py | 323 | Isolation Forest (D.1.1, Phase D, v0.9.31). | check_batch | fit_isolation_forest | _Tree, IsolationForestModel |
| importance.py | 155 | Permutation Feature Importance (D.6.3, Phase D, v0.9.31). | check_batch |  |  |
| kmeans.py | 374 | k-means clustering (v0.7.1). | check_batch | fit_kmeans | KMeansStepOp, KMeansModel |
| kmeans_ad.py | 228 | KMeansAD: k-means clustering anomaly detection on sliding windows. | delegate | fit_kmeans_ad | KMeansADResult |
| kshape.py | 380 | KShapeAD: k-Shape clustering-based TS anomaly detection. | delegate | fit_kshape | KShapeResult |
| lambdamart.py | 290 | LambdaMART / lambdarank (D.2.1, Phase D, v0.9.35). | check_batch | fit_lambdamart | _Node, LambdaMARTModel |
| lda.py | 273 | Online variational Bayes LDA (D.6.6, v0.9.41). | check_batch | fit_lda | LdaModel |
| linear.py | 470 | Linear/Ridge regression (v0.7.0a). | delegate | fit_linear, fit_lasso, fit_elasticnet | XTXOp, LinearModel |
| linucb.py | 141 | LinUCB disjoint linear bandit (D.6.7, v0.9.41). | check_batch | fit_linucb | LinUCBModel |
| logistic.py | 412 | Logistic regression (v0.7.0b). | delegate | fit_logistic | LogisticGradOp, LogisticModel |
| matrix_profile.py | 157 | Matrix Profile (D.1.2, Phase D, v0.9.35) — streaming z-normalized. | check_batch | fit_matrix_profile | MatrixProfileModel |
| meta_learners.py | 221 | T-learner / S-learner meta-learners for the ATE (D.3.2, v0.9.37). | check_batch | fit_meta_learner | MetaLearnerModel |
| minibatch.py | 337 | MiniBatch KMeans and streaming EM GMM (D.5.3, v0.9.39). | check_batch | fit_minibatch_kmeans, fit_minibatch_gmm | MiniBatchKMeansModel, MiniBatchGMMModel |
| minirocket.py | 181 | MiniRocket-style random convolution features (D.4.1, v0.9.37). | check_batch | fit_minirocket | MiniRocketModel |
| naive_bayes.py | 703 | Naive Bayes (v0.7.6) — Gaussian, Multinomial, Bernoulli. | delegate | fit_gaussian_nb, fit_multinomial_nb, fit_bernoulli_nb | _GaussianStatsOp, _MultinomialStatsOp, _BernoulliStatsOp, GaussianNB, MultinomialNB, BernoulliNB, _MultinomialPredictMixin |
| ordinal.py | 161 | Ordinal logistic regression, cumulative-link model (D.5.1, v0.9.39). | check_batch | fit_ordinal_logit | OrdinalLogitModel |
| patch_memory_ad.py | 351 | PatchMemoryAD: PaAno-inspired patch+memory-bank anomaly detection. | delegate | fit_patch_memory_ad | PatchMemoryADResult |
| pca.py | 335 | Principal Component Analysis (v0.7.2). | delegate | fit_pca | PCA |
| pelt.py | 195 | PELT change-point detection (D.4.2, v0.9.38). | check_batch | fit_pelt | PELTModel |
| randsvd.py | 136 | Randomized SVD / truncated PCA (D.6.4, v0.9.40). | check_batch | fit_randomized_svd | RandomizedSVDModel |
| s2g.py | 371 | S2G (Series2Graph): graph-based subsequence anomaly detection. | delegate | fit_s2g | S2GResult |
| stray.py | 253 | STRAY: streaming TS anomaly detection via robust KNN + EVT. | delegate | fit_stray | STRAYResult |
| sub_pca.py | 298 | Sub-PCA: windowed PCA reconstruction-error anomaly detection. | delegate | fit_sub_pca | SubPCAResult |
| tfidf.py | 128 | Streaming TF-IDF vectorizer (D.6.9, v0.9.41). | check_batch | fit_tfidf | TfidfModel |
| tspq.py | 294 | T-squared + Q (Hotelling + Squared Prediction Error) anomaly detection | delegate | fit_t2q | T2QResult |
| uplift.py | 381 | Uplift trees with Qini evaluation (D.3.3, v0.9.42). | check_batch | fit_uplift | _CausalTree, UpliftModel |
| zms.py | 129 | ZMS: rolling z-score anomaly detector. | delegate | fit_zms | ZMSResult |# cadw Inventory — capabilities & RAM entry points

## Layers (from README/CONTEXT)

- Storage (Parquet/IPC) → Dataset → Access (RowFilter, MemoryGuard, Planner) → Primitives (Reduce, HashShuffle, Sort) → Operations (groupby, join, dedup, rank, quantile, resample, sort, pivot, topk, metrics) → DataFrame (CadwDataFrame) → Export (to_batches, ML, viz)

## Core RAM controls

- `cadw/core/memory_guard.py` — MemoryGuard: check_batch, check_state, check_partition_buffers, check_partition_writers, estimate_state_size, log_step, get_read/write_batch_size.

## ML modules (47) — guard status

check_batch called directly: 24 modules. Delegate to reduce()/guard: 23 modules.

Windowed TS anomaly models (risk: materialize O(n*w) sliding windows via .copy()):
- sub_pca._prepare_windows / autoencoder / kmeans_ad (import _prepare_windows)
- kshape._sliding_windows, s2g, patch_memory_ad, dwt_mlead._sliding_windows
- eif._make_windows, tspq._make_windows

See inventory table below for per-module detail.