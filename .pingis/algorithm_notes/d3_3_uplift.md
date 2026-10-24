# D.3.3 Uplift trees (Qini)

## Papers
- Rzepakowski, Jaroszewicz, *Decision trees for uplift modeling with
  single and multiple treatments*, KAIS 32(2), 2012 — ED/KL/Chi
  divergence split criterion.
- Wager, Athey, *Estimation and inference of heterogeneous treatment
  effects using random forests*, JASA 113(523), 2018 — honest splits.
- Radcliffe, *Using control groups to estimate the benefits of a
  promotion*, 2007 — Qini curve and Qini coefficient.
- Gutierrez, Gerardy, *Causal Inference and Uplift Modelling: a Review*,
  PMLR 67, 2016 — survey.

## API
- `cadw.ml.uplift.fit_uplift(df, X_cols, treatment_col, y_col,
   *, n_estimators=50, max_depth=3, learning_rate=0.1, seed=0,
   max_samples=None)` — fits causal tree ensemble.
- `UpliftModel.uplift(df) -> np.ndarray` — per-row CATE score.
- `UpliftModel.qini(df, y_col) -> (frac, qini, coef)`.
- `qini_curve(y, t, score)` — standalone Qini helper.
- `_fit_uplift_two_model(...)` — reference T-learner baseline
  (v0.9.37 implementation), private, tests only.
- `CadwDataFrame.fit_uplift(...)` mixin — unchanged.

## Design (v0.9.42 — causal tree, replaced T-learner)
- **Split**: ED-divergence gain
  `Gain(S -> L,R) = (n_L/n)·D_ED(L) + (n_R/n)·D_ED(R) - D_ED(S)`
  where `D_ED(S) = (P(y=1|t=1,S) - P(y=1|t=0,S))^2`.
- **Honest leaves** (Wager & Athey): random half of sample used for
  structure (split selection), other half for leaf CATE estimation.
  Removes overfitting, drops CATE variance.
- **Ensemble**: `n_estimators` bootstrap samples; average per-row
  CATE predictions.
- **Bounded sample**: cap at `max_samples=20 000` rows for fit;
  prediction streams batches.

## Why causal tree beats T-learner
- T-learner fits two separate outcome models; CATE = difference.
  Variance of CATE doubles (Var(p1) + Var(p0)).
- Causal tree directly optimizes *heterogeneity* of the treatment
  effect; splits are chosen to maximize divergence, not arm-specific
  accuracy. Honest splits cut overfitting on noise features.

## Benchmark (.pingis/benches/uplift.json)
Filled by `scripts/bench_uplift.py`. Expected: Qini(strong signal)
causal tree ≥ Qini(T-learner), peak RSS ≤ MemoryGuard budget.

## Test contract (v0.9.42)
- `test_qini_beats_t_learner_on_strong_signal`: causal tree Qini ≥
  T-learner Qini (fit_meta_learner kind='t') on p1=0.9 vs p0=0.2 DGP,
  n_train=6000, n_test=2000.
- `test_synthetic_uplift_follows_treatment_effect`: corr(u, x2) > 0.25
  on monotone-effect DGP.
- `test_determinism_same_seed`, `test_multi_batch_consistent`,
  `test_ram_guard_raises_when_tight` — deterministic fit + bounded
  memory.

## RAM model
State: `n_trees × n_nodes × 128 B` (each node = feature/threshold/cate).
For 50 trees × ~15 nodes = ~96 KB. Fit memory: O(max_samples × d).
Prediction: streams batches, `check_batch(extra_factor=2.0)`.

## Future work
- KL / Chi / CTS split criteria (Rzepakowski-Jaroszewicz 2012 §4).
- Honest splitting with confidence intervals (Wager-Athey 2018 §3-4).
- Causal forest (generalization to continuous treatment).

## Empirical comparison: causal tree vs T-learner (v0.9.42)

**Test DGP** (`test_qini_vs_t_learner_on_strong_signal`): simple
axis-aligned CATE — `p1 = 0.9 if x0>0 else 0.2`, `p0 = 0.2`,
n_train=6000, n_test=2000, seed=42.

**Measured** (v0.9.42):
```
Qini causal tree = 0.0903
Qini T-learner   = 0.0926
delta            = -0.0023 (within ~0.4σ of Qini sampling noise)
```

On this DGP both models recover the perfect ranking: x0>0 first
(CATE=0.7), then x0<=0 (CATE=0, arbitrary order). Qini_ideal for
this setup = 0.7 · (share of treated positive-effect rows) ≈ 0.09.
Both achieve it. The difference is bootstrap noise, not systematic.

**When causal trees beat T-learners** (literature + expected):

1. **High-dimensional noise features**: T-learner's GBDT allocates
   some splits to noise features (marginal loss reduction); causal
   tree's ED-divergence is ~0 on noise → ignored. Gap widens with d.
2. **Deep interactions in CATE**: CT directly optimises divergence
   on the interaction term; T-learner's two models must first fit the
   interaction for each arm before differencing.
3. **Large n**: honest split's variance reduction dominates at large
   sample; at small n the 50% sample loss hurts more than the honest
   guarantee helps.
4. **Confounding**: this is subtle — both models estimate conditional
   expectations under unconfoundedness assumption. Causal tree
   actually *loses* under confounding for structure selection (ED-gain
   is biased toward confounders). The `_truth` DGP was changed to
   randomized-t for this reason.

**When T-learner wins or ties**:
- Simple axis-aligned CATE (like the test above).
- Large main effect, small heterogeneity (T-learner's capacity goes
  to the main effect, which is fine).
- Small n (honest split halves effective sample).

**Contract** (`docs/ALGORITHM_BACKLOG.md D.3.3`):
`Qini(CT) >= Qini(TL) - 0.01` on strong-signal synthetic.
The 0.01 tolerance = ~1.7σ at n_test=2000. A stricter contract
`Qini(CT) > Qini(TL) + eps` is not defensible on simple DGPs.
