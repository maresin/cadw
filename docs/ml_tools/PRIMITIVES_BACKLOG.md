# PRIMITIVES_BACKLOG — forward scan of non-implemented cards

**Version:** 1.0 (2026-11-08, v0.9.136)
**Purpose:** identify which primitive functions the 162 non-implemented
cards would need if implemented, cluster them, prioritize by consumer
count. Referenced by DEC-056 (AP10 amendment: forward extraction allowed
when prospective consumers >= 2).
**Hierarchy:** Tier 4 (reference). Complements `DECISION_LOG` DEC-056,
`AI_CONSTITUTION` §5 (AP10 amendment), `PRIMITIVES_AND_PRINCIPLES.md`.

**Status:** living document. Update as primitives are implemented.

**Counts:** 20 P1 clusters (~30 functions; some clusters are
families of 2-3 variants: L3 kernels ×3, L6 laplacian+eigsh,
L9 pinball loss+grad, L12 consensus+coassoc, L13 ranknet+listnet+
lambdarank, L14 grid+random, L18 ks+chi2+psi). 15 P2 clusters
(~20 functions). 20 P3 clusters (~20 functions, opportunistic).

**Where dependencies live:** each card's `depends_on` field lists
primitive paths + `depends_on_backlog: PRIMITIVES_BACKLOG §<cluster>`
+ `depends_on_status: planned`. Discoverable via:
  `grep "depends_on_status: planned" docs/ml_tools/*.yaml` -> 54 cards
  `grep "§L1" docs/ml_tools/*.yaml` -> 8 cards (knn_query consumers)
**Scope:** 162 cards (status=None: described_only=125, not_impl=106,
inference-only=6, last-priority=5, overlap makes counts non-additive).

---

## How this was made

1. Probe of all `status: None` cards via YAML scan.
2. Per card: infer primitives from `purpose` text + `algorithm_source_url`.
3. Cluster by (primitive function name), count **distinct consumers**.
4. Priority by consumer count:
   - **P1** = 3+ distinct cards use the same primitive
   - **P2** = 2 distinct cards
   - **P3** = 1 card (opportunistic, extract only if a 2nd caller
     appears naturally)
5. Home module choice:
   - `cadw/ops/linalg` — distances, kernels, matrix ops
   - `cadw/ops/stats` — statistical tests, moments, quantiles
   - `cadw/ops/numeric` — elementwise numeric (sigmoid, erfinv, log1p)
   - `cadw/ops/graph` (new) — MST, laplacian, spectral
   - `cadw/ops/sampling` (new) — reservoir, stratified, weighted, SMOTE
   - `cadw/ops/itemset` (new) — Apriori, FP-tree, PrefixSpan
   - `cadw/ops/optim` — SGD (existing) + new: adam, leapfrog, bayes-opt
   - `cadw/ops/proba` (new) — viterbi, forward-backward, HMC accept
   - `cadw/ml/_nn_init` — Xavier (existing) + he, orthogonal
   - `cadw/ml/_gbdt_common` (new) — pinball / lambda / listnet grad
   - `cadw/ml/_causal_common` (new) — DR score, ED divergence, honest split
   - `cadw/ml/_survival_common` (new) — log-rank, Breslow (coxph has partial)
   - `cadw/ts/_smoothing` — SES/Holt (deferred, see DEC-054 K3b)
   - `cadw/ts/_decomposition` — STL (existing)

**Non-goals:** don't implement all. Prioritize by P1 in v0.9.137+.
P2/P3 wait for second consumer to justify extraction (AP10).

---

## Already implemented (v0.9.132-135)

| Primitive | Module | Current consumers | Prospective |
|---|---|---|---|
| `sigmoid` | `cadw.ops.numeric` | gbdt, calibration, ordinal, +2 | +4-6 |
| `erfinv` | `cadw.ops.numeric` | otm, theta | +3 |
| `ols_fit` | `cadw.ops.linalg` | otm, theta | +2 |
| `sq_dists` | `cadw.ops.linalg` | patch_memory_ad | +12 (P1) |
| `znormalize_rows` | `cadw.ops.stats` | autoencoder, patch_memory_ad | +5 |
| `arm_stats` | `cadw.ops.stats` (v0.9.143) | bandits ×3 | +2 (future policy variations) |
| `nmf_mu` | `cadw.ops.linalg` (v0.9.143) | NMF ×2 | +1 (Top2Vec) |
| `reservoir_sample` + `batch_to_features` | `cadw.ops.sampling` (v0.9.143-b5) | KernelDensity, ReservoirSample | +3 (SMOTE, OHE) |
| `knn_query` | `cadw.ops.linalg` (v0.9.144) | LOF, DBSCAN, HDBSCAN, SMOTE, ML-kNN, ItemKNN, KNNImputer, EuclideanKMeansZ | +0 (implemented) |
| `core_distances` | `cadw.ops.linalg` (v0.9.144) | HDBSCAN, DBSCAN, LOF | +0 (implemented) |
| `pairwise_kernel` (rbf/poly/laplacian) | `cadw.ops.linalg` (v0.9.145) | Spectral ×5, KernelPCA, SparsePCA, GPLVM, GP reg/forecaster, GP-EI, BayesianOpt_GP | +0 (implemented) |
| `chol_solve` | `cadw.ops.linalg` (v0.9.145) | GP reg/forecaster, GP-EI, BayesianOpt_GP, GPLVM | +0 (implemented) |
| `eigh_topk` | `cadw.ops.linalg` (v0.9.146) | GPR, Spectral ×5, KernelPCA, GPLVM | +4 (randomized_svd, Consensus ×3) |
| `graph_laplacian` | `cadw.ops.graph` (v0.9.147) | Spectral ×5, LaplacianEigenmaps | +0 (implemented) |
| `eigsh_topk` | `cadw.ops.graph` (v0.9.147) | Spectral ×5 | +0 (implemented, wraps eigh_topk) |
| `mst` | `cadw.ops.graph` (v0.9.147) | HDBSCAN ×2, Agglomerative single | +1 (STRAY) |
| `normalize_weights` | `cadw.ops.stats` (v0.9.148) | ML-kNN, SMOTE, WeightedSample, +1 | +0 (implemented) |
| `apriori` + `eclat` + `support_count` + `candidate_generation` | `cadw.ops.itemset` (v0.9.149) | Apriori, Eclat | +0 (implemented) |
| `fp_growth` + `prefix_span` + `gsp` | `cadw.ops.itemset` (v0.9.150) | FP-growth, PrefixSpan, GSP | +0 (implemented) |
| `pinball_loss` + `pinball_grad` | `cadw.ml._gbdt_common` (v0.9.151) | LinearQuantReg, GBDT_pinball, CQR | +0 (implemented) |
| `adaboost_weight` | `cadw.ml._gbdt_common` (v0.9.151) | RUSBoost | +0 (implemented) |
| `cumulative_logit_prob` | `cadw.ml._gbdt_common` (v0.9.151) | OrderedProbitBayesian | +0 (implemented) |
| `dr_score` + `aipw_ate` | `cadw.ml._causal_common` (v0.9.152) | DR_learner, CausalForest ×2, DR_learner_uplift | +0 (implemented) |
| `ks_2samp` + `psi` + `chi2_categorical` | `cadw.ops.stats` (v0.9.153) | EvidentlyDrift, ADWIN, SequentialTest | +0 (implemented) |
| `ranknet_grad` + `listnet_grad` + `lambdarank_grad` | `cadw.ml._ltr_common` (v0.9.154) | PairwiseLogistic, ListNet, LambdaMART | +0 (implemented) |
| `logrank` | `cadw.ml._survival_common` (v0.9.155) | RandomSurvivalForest | +0 (implemented) |
| `grid_search` + `random_search` | `cadw.ops.optim` (v0.9.156) | GridSearch, RandomizedSearch | +1 (OTM/ETS/SARIMA migration) |
| `acf` | `cadw.ops.stats` | AutoStationarizer | +5 (P1) |
| `stl_decompose` | `cadw.ts._decomposition` | otm, theta, transforms, anomaly_pipeline | +2 |
| `_xavier_init` | `cadw.ml._nn_init` | autoencoder, patch_memory_ad | +3 |
| `sgd_streaming` | `cadw.ops.optim` | logistic, linear, ordinal (multinomial) | +4 |

---

## P1 — 3+ prospective consumers (implement first)

### L8 (implemented, v0.9.149-150)
- `cadw.ops.itemset`: support_count, candidate_generation,
  apriori, eclat, fp_growth, prefix_span, gsp.
- Consumers: Apriori, Eclat, FP-growth, PrefixSpan, GSP (5/6
  R_association cards; HMM pending — model, not primitive).

### L12. `_consensus_matrix` + `_coassociation`
- **Consumers**: KMeansHDBSCAN_consensus, EvidenceAccumulation, CSPA
- **Home**: `cadw.ops.linalg.consensus_matrix(labels_list)` (co-association
  matrix from multiple clusterings), then spectral eigendecomp (L6).
- **Effort**: 1 session.

### L20. `_cumulative_logit` (proportional-odds / threshold model)
- **Consumers**: GBDT_threshold, OrderedProbitBayesian, OrdinalLogit
  (currently in `cadw/ml/ordinal.py` — migrate after extract)
- **Home**: `cadw.ml._gbdt_common.cumulative_logit_prob(eta, theta)`.
- **Effort**: 0.3 session.

---

## P2 — 2 prospective consumers (implement alongside the 2nd consumer)

| Primitive | Home | Consumers | Effort |
|---|---|---|---|
| `_mutual_reachability` | `cadw.ops.graph` | HDBSCAN, OPTICS | 0.5 |
| `_linkage_matrix` (single/complete/average) | `cadw.ops.graph` | Agglomerative, BIRCH (CF-tree) | 1 |
| `_rand_svd` | `cadw.ops.linalg` | randomized_svd, t-SNE init | 1 |
| `_als_step` (enhanced for implicit) | `cadw.ops.linalg` | SVDpp, Hybrid_MF_GBDT | 0.5 (ALS exists) |
| `_softmax_ce_grad` | `cadw.ml._gbdt_common` | ListNet, LambdaMART | 0.5 |
| `_prox_l1` (soft-threshold) | `cadw.ops.numeric` | SparsePCA, NMF-sparse | 0.2 |
| `_prefix_projection` | `cadw.ops.itemset` | PrefixSpan, GSP | 1 |
| `_leapfrog` + `_hmc_accept` | `cadw.ops.proba` | NUTS, HMC (if added) | 1 |
| `_mean_field_update` | `cadw.ops.proba` | MeanFieldVI, SVI | 0.5 |
| `_sprt_threshold` (Wald SPRT) | `cadw.ops.stats` | SequentialTest, SPRT-based drift | 0.3 |
| `_ed_divergence` | `cadw.ml._causal_common` | CausalTree, CausalForest | 0.3 |
| `_honest_split` | `cadw.ml._causal_common` | CausalTree, CausalForest | 0.5 |
| `_kmeans_plus_plus` | `cadw.ml._common` (from kmeans.py) | MiniBatchKMeans, BisectingKMeans | 0.2 |
| `_adaboost_misclassified` | `cadw.ml._gbdt_common` | RUSBoost, GBDT_scale_pos_weight | 0.3 |
| `_matrix_completion_sgd` | `cadw.ops.linalg` | SVDpp, Hybrid_MF_GBDT | 1 |

---

## P3 — 1 prospective consumer (opportunistic; extract on 2nd caller)

| Primitive | Card | Note |
|---|---|---|
| `_t_sne_gradient` | t-SNE | 1 card, no second consumer yet |
| `_umap_fuzzy_graph` + `_umap_layout` | UMAP | 1 card, algorithm complex |
| `_viterbi` + `_forward_backward` | HMM | 1 card |
| `_exp_histogram` | ADWIN | 1 card |
| `_shap_tree_path` | SHAPTreeExplainer | 1 card, large scope |
| `_anchor_explain` | Anchors | 1 card |
| `_fairness_reductions` | FairlearnParity | 1 card |
| `_plaid_update` | PlaidModel | 1 card |
| `_bimax_recursive` | Bimax | 1 card |
| `_smote_interp` | SMOTE | 1 card |
| `_mice_regress` | IterativeImputer | 1 card |
| `_kbins_quantile` | KBinsDiscretizer | 1 card |
| `_poly_terms` | PolynomialFeatures | 1 card |
| `_target_encode_fold` | TargetEncoder | 1 card |
| `_one_hot` | OneHotEncoder | 1 card (encode_categorical exists) |
| `_count_sketch` | Word2Vec | 1 card (inference-only, skip) |
| `_bert_tokenize` | SentenceTransformer | 1 card, inference-only, **skip** |
| `_diffusion_reverse` | TabDDPM, TabSyn | 2 cards but same NN scope, **P3-skip** (S_generative last-priority) |
| `_ctgan_conditional_gen` | CTGAN, CTAB-GAN+, TVAE | 3 cards, but NN-heavy, **P3-skip** |
| `_gbdt_full_grad_hess` | CADWBoost | 1 card (sandbox), **skip** (gbdt.py has this) |

**P3-skip rationale:** generative (S_generative) = last-priority
(DEC-050 weak machine); inference-only cards = `to_batches` export
only (no fit); NN-based cards = external library export only.

---

## Coverage map: card → primitives

### A_classification (6 non-impl)
- `GaussianProcessRegressor` (A3): L3, L4, L19
- `ML-kNN` (A4): L1, L17
- `OrderedProbitBayesian` (A5): L20, + `sigmoid` (have)
- `RUSBoost` (A6): L16, L17, P2 `_adaboost_misclassified`
- `CQR` (A7): L9, + conformal (individual)
- `GBDT_pinball` (A7): L9, + `gbdt.py` extension
- `LinearQuantReg` (A7): L9

### B_clustering (13 non-impl)
- `BisectingKMeans`, `KMeansOnly` (B1/B6): `kmeans.py` + P2 `_kmeans_plus_plus`
- `GaussianMixtureClustering` (B1/B2): `gmm.py` (have)
- `SpectralClustering` (B1/B2/B3): L3, L6, L19
- `DBSCAN` (B2): L1, L2, `_region_query` (inline)
- `HDBSCAN` (B2): L1, L2, L5, P2 `_mutual_reachability`
- `AgglomerativeClustering` (B3): P2 `_linkage_matrix`
- `BIRCH` (B3): `.pingis/impl/` (already implemented, promotion v0.9.131 pending)
- `KShape`, `EuclideanKMeansZ`, `ShapeGMM` (B4): `kshape.py` (have) + `znormalize_rows` (have)
- `SpectralCoclustering`, `SpectralBiclustering` (B5): L3, L6, L19
- `PlaidModel`, `Bimax` (B5): P3 individual
- `KMeansHDBSCAN_consensus`, `EvidenceAccumulation`, `CSPA` (B6): L12

### C_dim_reduction (8 non-impl)
- `TruncatedSVD`, `randomized_svd` (C1): P2 `_rand_svd`
- `KernelPCA`, `SparsePCA` (C1): L3, P2 `_prox_l1`
- `NMF` (C2/J3): L7
- `ALS` (C2) — implemented in F_recsys
- `t-SNE`, `UMAP` (C3): P3 individual
- `GPLVM` (C3): L3, L4, L19
- `Boruta`, `SelectKBest`, `PermutationImportance` (C4): `feature_select.py` (have), `importance.py` (have)

### D_time_series (5 non-impl)
- `GaussianProcessForecaster` (D2): L3, L4
- `HIVE-COTE` (D4): P3-skip (ensemble, `tier: ensemble`)
- 3 pretrained Chronos: **skip** (inference-only)

### E_anomaly (3 non-impl)
- `LOF` (E1): L1, L2
- `GMMAnomaly` (E1): `gmm.py` (have)
- `CalibratedClassifierCV` (E2): `calibration.py` (have) + wrapper (individual)
- `GBDT_scale_pos_weight` (E2): L16, L17

### F_recsys (3 non-impl)
- `ItemKNN` (F1): L1
- `GBDT_ALS_embedding`, `Hybrid_MF_GBDT` (F1/F2): `als.py` (have) + gbdt stacking
- `SVDpp` (F2): P2 `_matrix_completion_sgd`

### G_ltr (3 non-impl)
- `PairwiseLogistic`, `LambdaMART`, `ListNet` (G1): L13, P2 `_softmax_ce_grad`

### H_causal (4 non-impl)
- `DR_learner`, `DR_learner_uplift` (H1/H2): L11
- `CausalForest`, `CausalTree` (H1/H2): L11, P2 `_ed_divergence`, `_honest_split`
- `T_learner_uplift` (H2): `meta_learners.py` (have)

### I_survival (5 non-impl)
- `KaplanMeier`, `WeibullAFT`: `.pingis/impl/` (promotion v0.9.131)
- `RandomSurvivalForest`, `DeepSurv`, `DeepHit`: L10 (last two inference-only)

### J_nlp (5 non-impl)
- `ComplementNB`: `naive_bayes.py` (have)
- `LinearSVC`: `logistic.py` extension (individual)
- `Word2Vec`, `SentenceTransformer`: **skip** (inference-only, NN)
- `LDA_embedding`, `NMF` (J2/J3): L7
- `Top2Vec` (J3): L7 + P3 (embedding step external)

### M_bandits (3 non-impl)
- `EpsilonGreedy`, `UCB1`, `ThompsonSampling` (M1): L15
- `NeuralUCB` (M1): **skip** (NN, requires torch)

### N_density (2 non-impl)
- `KernelDensity`: `.pingis/impl/` (promotion v0.9.131)
- `HDBSCAN`: L1, L2, L5

### O_eda (21 non-impl)
- `Describe`, `Hist`, `ValueCounts`: `dataframe.py` (have)
- `SimpleImputer`: `fillna` (have)
- `KNNImputer`: L1
- `IterativeImputer`: P3 `_mice_regress`
- `OneHotEncoder`: `encode_categorical` (have) + P3 `_one_hot`
- `KBinsDiscretizer`: P3 `_kbins_quantile`
- `PolynomialFeatures`: P3 `_poly_terms`
- `TargetEncoder`: P3 `_target_encode_fold`
- `MutualInfoSelect`, `VarianceThreshold`, `BorutaSelect`: `feature_select.py` (have)
- `PanderaSchema`, `GreatExpectationsSuite`: **skip** (external libs)
- `StratifiedSplit`, `ReservoirSample`, `SMOTE`, `WeightedSample`: `cv.py` (stratified exists) + L17 + P3 `_smote_interp`
- `GridSearch`, `RandomizedSearch`, `OptunaTPE`: L14
- `PermutationImportance`: `importance.py` (have)
- `SHAPTreeExplainer`, `Anchors`: P3 individual
- `FairnessMetrics`: `fairness.py` (have)
- `FairlearnParity`: P3 `_fairness_reductions`
- `DriftStats`, `EvidentlyDrift`: L18
- `ADWIN`: P3 `_exp_histogram` + L18
- `SequentialTest`: P2 `_sprt_threshold`

### R_association (5 non-impl)
- `Apriori`, `Eclat`, `FP-growth`: L8
- `GSP`, `PrefixSpan`: P2 `_prefix_projection`
- `HMM`: P3 `_viterbi` + `_forward_backward`

### S_generative (6 non-impl) — **skip all** (last-priority DEC-050)
- TVAE, CTGAN, CTAB-GAN+, TabNet, TabDDPM, TabSyn, TabDDPM_classifier

### T_optimization (6 non-impl)
- `GP-EI`, `SMAC`: L3, L4 + `_ei_acquisition` (individual)
- `TPE`: P3 individual (kernel density per split)
- `NelderMead`, `CMA-ES`, `BayesianOpt_GP`: T2 — bayes-opt uses L3+L4

### V_uncertainty (2 non-impl)
- `SplitConformal`, `CVplus`: individual (conformal is small algorithm)

### W_probabilistic (5 non-impl)
- `MeanFieldVI`, `BlackBoxVI`, `SVI`: P2 `_mean_field_update`
- `MetropolisHastings`, `NUTS`: P2 `_leapfrog` + `_hmc_accept`

---

## Implementation roadmap

### v0.9.137 — P1 group A: distances + kernels
- L1 `_knn_query` (1 sess)
- L2 `_core_distances` (0.5 sess)
- L3 `_rbf_kernel` (1 sess)
- L4 `_chol_solve` (0.5 sess)
- **Total: 3 sessions**. Unblocks: LOF, DBSCAN, HDBSCAN, SMOTE, GPR, GP forecaster, spectral clustering (L6 pending), GPLVM, KernelPCA, GP-EI, BayesianOpt, KNNImputer.

### v0.9.138 — P1 group B: graph + eigsh + MST
- L5 `_mst` (1 sess)
- L6 `_graph_laplacian` + `_eigsh_topk` (1-2 sess)
- L19 `_eigen_decomp_topk` (0.5 sess)
- **Total: 2-3 sessions**. Unblocks: HDBSCAN dendrogram, Agglomerative single, Spectral ×3, Consensus ×3.

### v0.9.139 — P1 group C: factorization + itemset
- L7 `_nmf_mu` (1 sess)
- L8 `_itemset_mining` (2 sess)
- **Total: 3 sessions**. Unblocks: NMF, Top2Vec, LDA_embedding, Apriori, Eclat, FP-growth.

### v0.9.140 — P1 group D: losses + stats
- L9 `_pinball_loss` (0.5)
- L10 `_survival_logrank` (1)
- L11 `_doubly_robust_score` (0.5)
- L13 `_ltr_gradients` (1)
- L14 `_grid_search` (1)
- L15 `_arm_stats` (0.5)
- L16 `_adaboost_weight_update` (0.3)
- L17 `_fuzzy_cluster_weights` (0.2)
- L18 `_ks_stat` (0.5)
- L20 `_cumulative_logit` (0.3)
- **Total: 5-6 sessions**. Unblocks: quantile reg, survival, LTR, bandits, drift.

### v0.9.141+ — P2 groups (one at a time, alongside 2nd consumer)
Each P2 item lands when its second consumer is scheduled for implementation.

### P3 — opportunistic only
Extract only when a 2nd consumer appears naturally. Do not pre-empt.

---

## AP10 amendment (per DEC-056, applied v0.9.134)

**Old AP10:** "Don't extract a helper at 1 site."

**New AP10 (v0.9.134):** "Don't extract at 1 site UNLESS >= 2 prospective
consumers exist (documented here). When prospective >= 2, extraction at
1 site IS allowed and encouraged — forward investment, not violation."

**Application:**
- v0.9.134 extracted `_sq_dists` (12 prospective) and `_acf` (5
  prospective) at 1 current site each — legitimate under new AP10.
- P3 items (1 prospective, no second on roadmap) — stay inline per
  old AP10 heuristic.

---

## Maintenance

When a P1/P2 primitive is implemented:
1. Move item from backlog to "Already implemented" table.
2. Update affected cards' `depends_on` (add dotted path).
3. Remove any local duplicate in implementing module (AST migration).
4. Update "Implementation roadmap" — mark done.

When a new described_only card is added (post-freeze DEC-052: needs
DEC rationale), add to "Coverage map" and update consumer counts for
its primitives.

---

**End of PRIMITIVES_BACKLOG v1.0.**
