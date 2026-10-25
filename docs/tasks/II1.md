# I1 — Time-to-event (Cox / AFT / deep)

*Section:* I — Survival analysis

**Description:** Predict time-to-event under right-censoring; rank patients by hazard.

**Real-life examples:**
- Clinical prognosis (cancer, cardiovascular)
- Customer churn with unobserved exits
- Equipment failure / predictive maintenance

## Benchmarks
- **CoxPH_Survival** — https://github.com/sebp/scikit-survival/tree/master/sksurv/datasets
  - metrics: c_index, ibs, brier_time

## Models
### CoxPH — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** Semi-parametric proportional-hazards model; additive log-hazard in features. Inductive bias: linear log-risk with time-independent covariate effects — strong when effects are multiplicative and n is moderate.

**Use cases:**
- Clinical risk scoring on tabular covariates
- Churn with censoring
- Reliability / field-failure modeling

**Config:** `{'max_iter': 'int', 'tol': 'float'}`

**CADW:** `cadw.ml.coxph.fit_coxph`

**Benchmarks:**
- **[literature]** CoxPH_Survival → c_index=null
  - algorithm: `lifelines.CoxPHFitter` (docs)
  - number: https://github.com/sebp/scikit-survival/tree/master/sksurv/datasets
  - code/paper: https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html
  - note: no_public_benchmark: sksurv datasets page (fetched 2026-10-24) lists datasets but no C-index leaderboard
- **[literature]** CoxPH_Survival → ibs=null
  - algorithm: `lifelines.CoxPHFitter` (docs)
  - number: https://github.com/sebp/scikit-survival/tree/master/sksurv/datasets
  - code/paper: https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html
  - note: no_public_benchmark: no public integrated-Brier leaderboard for CoxPH on sksurv collection
- **[cadw_reproduced]** cadw_internal → c_index=0.6755 (n=1)
  - algorithm: `cadw.ml.coxph` (code)
  - reproduces: `lifelines.CoxPHFitter`
  - number: .pingis/benches/reproduction/coxph.json
  - code/paper: cadw/ml/coxph.py
  - note: M2 parity on synthetic_survival_like DGP; cadw=0.675451, ref(lifelines)=0.675451, verdict=consistent (tol=0.10)
- **[cadw_measured]** cadw_internal → c_index=0.7229 ± 0.0011 (n=2)
  - algorithm: `cadw.ml.coxph` (code)
  - number: .pingis/benches/coxph.json
  - code/paper: cadw/ml/coxph.py
  - note: CADW-native: c_index=0.7229 @n=5000 and 0.7244 @n=50000 (coxph.json); std=half-range; NOT comparable to external LB.

### KaplanMeier — baseline / tier1_core

**Roles:** probabilistic
**Status:** ✓ agent, desc

**Purpose:** Non-parametric product-limit estimator of the marginal survival curve. Inductive bias: no covariate effects — only the pooled survival function, so it is the canonical right-censored baseline that any covariate model must improve.

**Use cases:**
- Unadjusted survival by trial arm
- Baseline churn / retention curves
- Reliability curves on censored field data

**Config:** `{'confidence_interval': 'string'}`

**Benchmarks:**
- **[literature]** CoxPH_Survival → ibs=null
  - algorithm: `lifelines.KaplanMeierFitter` (docs)
  - number: https://github.com/sebp/scikit-survival/tree/master/sksurv/datasets
  - code/paper: https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html
  - note: no_public_benchmark: KM is a marginal-curve baseline; no public IBS leaderboard on sksurv collection

### WeibullAFT — baseline / tier1_core

**Roles:** linear_signal, probabilistic
**Status:** ✓ agent, desc

**Purpose:** Parametric accelerated-failure-time model with Weibull baseline. Inductive bias: log-time is linear in covariates plus a Weibull noise shape — stronger than Cox when the Weibull assumption holds; misspecified when hazard is non-monotone.

**Use cases:**
- Industrial reliability with monotone hazard
- Clinical survival when Weibull fits
- Small-n trial survival with strong parametric prior

**Config:** `{'fit_intercept': 'bool'}`

**Benchmarks:**
- **[literature]** CoxPH_Survival → c_index=null
  - algorithm: `lifelines.WeibullAFTFitter` (docs)
  - number: https://github.com/sebp/scikit-survival/tree/master/sksurv/datasets
  - code/paper: https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html
  - note: no_public_benchmark: no public C-index leaderboard for WeibullAFT on sksurv collection

### RandomSurvivalForest — sota / tier3_export

**Roles:** nonlinear_interactions
**Status:** desc, export

**Purpose:** Ishwaran & Kogalur ensemble of survival trees (log-rank splits). Inductive bias: piecewise-constant hazard with nonlinear interactions and no proportional-hazards assumption; strong on heterogeneous tabular cohorts.

**Use cases:**
- Clinical risk on heterogeneous cohorts
- Predictive maintenance with mixed signals
- Genomic survival (high-d, small-n)

**Config:** `{'n_estimators': 'int', 'min_samples_split': 'int', 'max_depth': 'int'}`

**Benchmarks:**
- **[literature]** CoxPH_Survival → c_index=null
  - algorithm: `sksurv.ensemble.RandomSurvivalForest` (docs)
  - number: https://github.com/sebp/scikit-survival/tree/master/sksurv/datasets
  - code/paper: https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.RandomSurvivalForest.html
  - note: no_public_benchmark: sksurv docs page (fetched 2026-10-24) has no aggregated C-index table

### DeepSurv — sota / tier3_export

**Roles:** nonlinear_interactions, representation
**Status:** desc, export

**Purpose:** Katzman 2018 — Cox proportional hazards with an MLP as the log-risk function. Inductive bias: learned nonlinear covariate representation feeding the Cox partial likelihood; torch-only, reachable only via to_batches(format='torch').

**Use cases:**
- Large-cohort clinical risk
- Genomic survival prediction
- Censored CTR / churn with high-d features

**Config:** `{'hidden_sizes': 'list', 'dropout': 'float', 'lr': 'float'}`

**Benchmarks:**
- **[literature]** CoxPH_Survival → c_index=null
  - algorithm: `DeepSurv (Katzman et al. 2018)` (paper)
  - number: https://github.com/sebp/scikit-survival/tree/master/sksurv/datasets
  - code/paper: https://arxiv.org/abs/1606.08131
  - note: no_public_benchmark: DeepSurv reports per-dataset C-index on its own splits; no unified LB on sksurv collection

### DeepHit — sota / tier3_export

**Roles:** nonlinear_interactions, representation
**Status:** desc, export

**Purpose:** Lee 2018 — deep network predicting a discrete-time cause-specific hazard distribution directly, no PH assumption. Inductive bias: fully nonparametric hazard curve learned by NN + ranking loss; SOTA when hazards cross, torch-only.

**Use cases:**
- Competing-risks clinical survival
- Multi-cause churn with non-PH effects
- Non-proportional time-to-failure

**Config:** `{'hidden_sizes': 'list', 'num_durations': 'int', 'alpha': 'float', 'beta': 'float'}`

**Benchmarks:**
- **[literature]** CoxPH_Survival → c_index=null
  - algorithm: `DeepHit (Lee et al. 2018)` (paper)
  - number: https://github.com/sebp/scikit-survival/tree/master/sksurv/datasets
  - code/paper: https://arxiv.org/abs/1801.05599
  - note: no_public_benchmark: DeepHit reports on its own synthetic + METABRIC splits; no public LB on sksurv collection

## Metrics
- **c_index** (anchor, higher)
  - Concordance of risk ranking under censoring; the primary survival metric.
  - *special:* Depends on censoring rate; only compare within the same dataset. Prefer time-dependent AUC when censoring is heavy.
- **ibs** (diagnostic, lower)
  - Integrated Brier score — calibration + discrimination, censoring-adjusted.
  - *special:* Not comparable to c_index numerically; report side-by-side.
- **brier_time** (diagnostic, lower)
  - Time-resolved Brier at fixed horizons (1y, 5y, ...).
  - *special:* Requires pre-specified horizons; not summed across time.

## Ensemble rules
- **[linear_signal, probabilistic]** → CoxPH (max_per_role=2)
  - CoxPH (semi-parametric PH) and WeibullAFT (parametric AFT) give complementary hazard parameterizations; average the predicted risk ranking when both verified in cadw.
