# INVENTORY — sections I / N / J / M / O

Total cards: 56

## Status summary
- verified_in_cadw: 20
- described_only: 36

## Section I  (6 cards) — {'verified_in_cadw': 1, 'described_only': 5}

| subtype | model | tier | tier_level | status | cadw_module | cadw_fn |
|---|---|---|---|---|---|---|
| I1 | CoxPH | baseline | tier1_core | verified_in_cadw | cadw.ml.coxph | fit_coxph |
| I1 | KaplanMeier | baseline | tier1_core | described_only |  |  |
| I1 | WeibullAFT | baseline | tier1_core | described_only |  |  |
| I1 | RandomSurvivalForest | sota | tier3_export | described_only |  |  |
| I1 | DeepSurv | sota | tier3_export | described_only |  |  |
| I1 | DeepHit | sota | tier3_export | described_only |  |  |

## Section N  (5 cards) — {'verified_in_cadw': 2, 'described_only': 3}

| subtype | model | tier | tier_level | status | cadw_module | cadw_fn |
|---|---|---|---|---|---|---|
| N1 | GaussianMixture | baseline | tier1_core | verified_in_cadw | cadw.ml.gmm | fit_gmm |
| N1 | KernelDensity | baseline | tier2_wrapper | described_only |  |  |
| N1 | BIRCH | baseline | tier1_core | verified_in_cadw | cadw.ml.birch | fit_birch |
| N1 | HDBSCAN | sota | tier2_wrapper | described_only |  |  |
| N1 | NormalizingFlow | sota | tier3_export | described_only |  |  |

## Section J  (10 cards) — {'verified_in_cadw': 3, 'described_only': 7}

| subtype | model | tier | tier_level | status | cadw_module | cadw_fn |
|---|---|---|---|---|---|---|
| J1 | TFIDF_LogReg | baseline | tier1_core | verified_in_cadw | cadw.ml.tfidf | fit_tfidf |
| J1 | ComplementNB | baseline | tier1_core | described_only |  |  |
| J1 | LinearSVC | baseline | tier2_wrapper | described_only |  |  |
| J1 | BERT | sota | tier3_export | described_only |  |  |
| J2 | LSA_TFIDF_SVD | baseline | tier1_core | verified_in_cadw | cadw.ml.randsvd | fit_randomized_svd |
| J2 | Word2Vec | baseline | tier2_wrapper | described_only |  |  |
| J2 | SentenceTransformer | sota | tier3_export | described_only |  |  |
| J3 | LDA | baseline | tier1_core | verified_in_cadw | cadw.ml.lda | fit_lda |
| J3 | NMF | baseline | tier2_wrapper | described_only |  |  |
| J3 | BERTopic | sota | tier3_export | described_only |  |  |

## Section M  (5 cards) — {'verified_in_cadw': 1, 'described_only': 4}

| subtype | model | tier | tier_level | status | cadw_module | cadw_fn |
|---|---|---|---|---|---|---|
| M1 | LinUCB | baseline | tier1_core | verified_in_cadw | cadw.ml.linucb | fit_linucb |
| M1 | EpsilonGreedy | baseline | tier1_core | described_only |  |  |
| M1 | UCB1 | baseline | tier1_core | described_only |  |  |
| M1 | ThompsonSampling | sota | tier2_wrapper | described_only |  |  |
| M1 | NeuralUCB | sota | tier3_export | described_only |  |  |

## Section O  (30 cards) — {'verified_in_cadw': 13, 'described_only': 17}

| subtype | model | tier | tier_level | status | cadw_module | cadw_fn |
|---|---|---|---|---|---|---|
| O1 | Describe | baseline | tier1_core | verified_in_cadw | cadw.dataframe | describe |
| O1 | Hist | baseline | tier1_core | verified_in_cadw | cadw.plot_accessor | hist |
| O1 | ValueCounts | baseline | tier1_core | verified_in_cadw | cadw.dataframe | value_counts |
| O1 | YDataProfiling | sota | tier2_wrapper | described_only |  |  |
| O2 | SimpleImputer | baseline | tier1_core | verified_in_cadw | cadw.dataframe | fillna |
| O2 | KNNImputer | baseline | tier1_core | described_only |  |  |
| O2 | IterativeImputer | sota | tier2_wrapper | described_only |  |  |
| O3 | OneHotEncoder | baseline | tier1_core | described_only |  |  |
| O3 | KBinsDiscretizer | baseline | tier1_core | described_only |  |  |
| O3 | PolynomialFeatures | sota | tier1_core | described_only |  |  |
| O4 | MutualInfoSelect | baseline | tier1_core | verified_in_cadw | cadw.ml.feature_select | fit_feature_select |
| O4 | VarianceThreshold | baseline | tier1_core | verified_in_cadw | cadw.ml.feature_select | fit_feature_select |
| O4 | BorutaSelect | sota | tier2_wrapper | verified_in_cadw | cadw.ml.feature_select | fit_feature_select |
| O5 | PanderaSchema | baseline | tier2_wrapper | described_only |  |  |
| O5 | GreatExpectationsSuite | sota | tier3_export | described_only |  |  |
| O6 | StratifiedSplit | baseline | tier1_core | described_only |  |  |
| O6 | ReservoirSample | baseline | tier1_core | described_only |  |  |
| O6 | SMOTE | sota | tier2_wrapper | described_only |  |  |
| O7 | GridSearch | baseline | tier1_core | described_only |  |  |
| O7 | RandomizedSearch | baseline | tier1_core | described_only |  |  |
| O7 | OptunaTPE | sota | tier2_wrapper | described_only |  |  |
| O8 | PermutationImportance | baseline | tier1_core | verified_in_cadw | cadw.ml.importance | permutation_importance |
| O8 | SHAPTreeExplainer | sota | tier3_export | described_only |  |  |
| O9 | FairnessMetrics | baseline | tier1_core | verified_in_cadw | cadw.ml.fairness | fairness_metrics |
| O9 | FairlearnParity | sota | tier2_wrapper | described_only |  |  |
| O10 | DriftStats | baseline | tier1_core | verified_in_cadw | cadw.ml.drift | fit_drift |
| O10 | EvidentlyDrift | sota | tier3_export | described_only |  |  |
| O11 | ProportionsZTest | baseline | tier1_core | verified_in_cadw | cadw.ml.drift | fit_ab_test |
| O11 | MannWhitneyU | baseline | tier1_core | verified_in_cadw | cadw.ml.drift | fit_ab_test |
| O11 | CUPED | sota | tier1_core | verified_in_cadw | cadw.ml.drift | fit_ab_test |
