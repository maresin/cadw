# CADW Roadmap

Last updated: 2026-11-08 (after v0.9.141).

## Current: v0.9.142 — depends_on 100% coverage + .pingis cleanup + §1 done

**12 releases since v0.9.128.** Initial goal from §1 list achieved:
all 10 draft models -> verified (35 -> 37 verified fns, 51 -> 53
verified cards). Two open questions documented below: (A) batching
honesty per-model, (B) shared-primitives plan (DEC-056 backlog).

### v0.9.129 — 3 ts drafts -> verified
- `OTMThetaForecaster`, `TemporalHierarchy`, `TransformedTargetForecaster`
  `draft -> verified` with `ram_formula` + `formula_evidence` +
  `measurement: component-wise (DEC-033)` + `bounded_by_domain n<=1e6`.
- New tests: `test_otm_verification.py`, `test_thf_verification.py`,
  `test_transforms_verification.py` (27 tests, CW-1..5).
- Code fixes: OTM `check_state(31*n*8+256)` before grid search;
  THF `state_nbytes()` + cumulative multi-group check_state;
  TTF `state_nbytes()` on all 4 transformers + `(3+8K)*n*8+4096`
  check_state (replaced invented 1024/transformer with exact sum, AP22).

### v0.9.130 — K1: `_probe_memory_limit_mb` 16 -> 1
- Canonical: `cadw.core.memory_guard.probe_limit_mb`.
- AST-based migration (regex ate `@dataclass` -> 65 failures; fixed
  via `ast.FunctionDef.end_lineno`).

### v0.9.131 — K2: `_sigmoid` 4 -> 1 (+ numerical stability fix)
- Canonical: `cadw.ops.optim.sigmoid` (pos/neg branch, no overflow).
- Correctness fix: `calibration.py` + `ordinal.py` clipped |z| to 50
  -> wrong probabilities for z=-1000. New impl gives 0/1 exactly.

### v0.9.132 — K3a: `_ols_fit` 2 -> 1, `_erfinv` 2 -> 1
- New: `cadw.ops.numeric.py` (pure math, no Arrow, no I/O).
- `_ses` NOT unified: 3 different algorithms (SES / theta-variant /
  Holt-Winters), AP22.
- `_acf` NOT extracted: 1 use site (AP10); `_grid_search` deferred.

### v0.9.133 — K5c+K6+K10+K12 (4 clusters)
- K5c `znormalize_rows` 2 -> 1 (`cadw.ops.numeric`).
- K6 `_to_df` 2 -> 1 (`cadw.ml._common._to_df_causal`).
- K10 `_xavier_init` 2 -> 1 (`cadw.ml._common`).
- K12 `_plt` 2 -> 1 (`cadw.viz._lazy.lazy_plt`).

**Counts:** 35 verified fns / 51 verified cards (was 32/48). Draft
fns unchanged (14) / draft cards 17 -> 14 (3 ts drafts -> verified).
described_only unchanged (61).

**Known gaps (v0.9.134+):** home-correction (`sigmoid` -> `numeric`,
`znormalize` -> `stats`, new `cadw/ops/linalg.py`, `cadw/ops/stats.py`,
`cadw/ml/_nn_init.py`); forward-looking extraction (`_sq_dists`,
`_acf` with prospective consumers); schema `depends_on` field;
PRIMITIVES_BACKLOG.md scan across all 302 cards.

---

### v0.9.134 — home-correction + forward extraction
- sigmoid -> numeric; znormalize -> stats; ols_fit -> linalg;
  _xavier_init -> _nn_init. New `cadw/ops/{linalg,stats}.py`,
  `cadw/ml/_nn_init.py`.
- Forward extraction (AP10 exception, DEC-056): `_sq_dists` -> linalg
  (12+ prospective), `_acf` -> stats (5+).
- Version drift fixed (was 0.9.128 at v0.9.129-133 tags).

### v0.9.135 — DEC-055 depends_on schema + migration
- ModelCard.depends_on field + R11 test (importable).
- Migration 66 cards (verified/draft): 11 non-empty, 55 `[]`.
- 162 status=None cards untouched (out of scope).

### v0.9.136 — DEC-056 PRIMITIVES_BACKLOG
- Forward scan of 162 non-implemented cards.
- 20 P1 clusters (3+ prospective consumers), 15 P2, 20 P3.
- Coverage map + implementation roadmap (v0.9.137-140+).
- AP10 amendment applied (forward extraction at 1 site allowed when
  prospective >= 2).

### v0.9.137 — DEC-057 impl: fit_batched/predict_batched + metric fix
- test_fit_ooc_v106: py_bytes -> measure_peak_traced (sampler).
- ModelCard.fit_batched + predict_batched schema fields.
- R13 test (bounded_state=true + fit_batched=false requires streaming
  keyword in streaming_strategy.fit).
- Migration 66 cards: 58 (true,true), 8 (false,false) intrinsic O(n).

### v0.9.138 — verify KMeans + MiniBatchKMeans + PCA (pre-cards era)
- All 3 draft -> verified with CW-1..5.
- OOC fix: batch_size auto-derive (DEC-023) for all 3 — before fix,
  single batch = n at default budget.
- G2: PCA check_state before eigh (2*d*d*8+d*8); KMeans
  check_batch in _kmeans_plus_plus_init; honest MemoryLimitExceeded
  when tight guard makes batch_rows < k.

### v0.9.139 — verify BVAR + VECM (single-series TS, streaming fit)
- Both draft -> verified. State O(k²p²) honest.
- batch_size auto-derive for both (DEC-023 class, same gap as v0.9.138).
- YAML: bounded_state_dims {d:2} -> {k:2, p:1} / {k:2, r:1};
  state_class O(d²) -> O(model).

### v0.9.140 — verify PELT + T2Q + STRAY (intrinsic O(n) + chunked)
- T2Q: bounded_state true -> false (honest correction; O(n) output
  arrays 4n*8, DEC-039 pattern as hampel/sub_pca v0.9.116).
- T2Q: silent OOM fix — check_state(4*n*8, "tspq.output") before
  aggregation (guard was missing post-chunk guard).
- PELT (×2) + STRAY: verify confirmed YAML was honest.
- 17 verification tests.

### v0.9.141 — verify T_learner + S_learner (final 2 of §1 drafts)
- Both share fit_meta_learner. State = n_estimators*K*4096 + n*f64
  + 1024 (K=2 for T, K=1 for S; propensity_ is per-row n array).
- YAML: fit_batched true -> false (AP2 _collect full-read);
  bounded_state_dims {batch:1} -> {n:1, n_estimators:1}.
- 10 verification tests.

---

## Open question A — Батчевость (DEC-057 status)

**Answered (v0.9.137):** ModelCard.fit_batched + predict_batched
fields exist. Migration of 66 implemented cards done:
- **58 (true,true)** — batch streaming (direct `_iter_filtered_batches`,
  chunked helpers, or delegates to batched `fit_*`).
- **8 (false,false)** — intrinsic O(n): OTM/THF/TTF (single-series TS),
  STRAY (O(n²) CPU), BinarySegmentation ×2 (DP over n), PELT ×2
  (DP over n). All honest via bounded_by_domain or tier1_core_exception.

**Honest YAML claims corrected (DEC-057 audit):**
- T2Q: bounded_state true -> false (v0.9.140).
- T_learner / S_learner: fit_batched true -> false (v0.9.141).

**Open items (deferred, tracked):**
1. **`predict_batched` measurement** — currently set = fit_batched by
   convention, NOT measured independently. Some predict paths may be
   full-read even when fit is batched. Target: v0.9.142 add
   `test_predict_batched_per_model` (SpillArray vs concatenate).
2. **EnsembleForecaster** — `fit_batched: true` inherited from members,
   NOT verified. **DEFERRED** until `fit_stacking` infrastructure
   exists (see below). Per-subtype ensemble wrappers should be thin
   over `fit_stacking`, not separate code paths (DEC-050 level 3,
   DEC-052 "no per-subtype duplicated recipes").
3. **described_only cards (162)** — batching N/A (no code). When
   implemented, `fit_batched` must be set. PRIMITIVES_BACKLOG
   (DEC-056) covers the forward plan.

---

## Open question B — Общие функции (DEC-054/055/056 status)

**DEC-054 (dedup pass):** 12 clusters K1-K12 identified. Applied 7:
- `_probe_memory_limit_mb` ×16 -> `cadw.core.memory_guard.probe_limit_mb`
- `_sigmoid` ×4 -> `cadw.ops.numeric.sigmoid` (+ stability fix)
- `_ols_fit` ×2 + `_erfinv` ×2 -> `cadw.ops.linalg` + `cadw.ops.numeric`
- `znormalize_rows` ×2 -> `cadw.ops.stats`
- `_to_df` ×2 -> `cadw.ml._common._to_df_causal`
- `_xavier_init` ×2 -> `cadw.ml._nn_init`
- `_plt` ×2 -> `cadw.viz._lazy.lazy_plt`

Skipped (AP10/AP22 with rationale):
- `_ses` ×3 (3 different algorithms: SES / theta / Holt-Winters)
- `_mahalanobis_distances` vs `_stream_mahalanobis` (oracle vs streaming)
- `_collect` ×5 (5 different signatures)
- `kshape._znormalize` (1D scalar vs per-row)
- `_acf` (originally 1 site — re-extracted v0.9.134 forward-scan)
- `_sq_dists` (originally 1 site — re-extracted v0.9.134 forward-scan)

Deferred:
- `_grid_search` (ETS coordinate vs OTM grid vs SARIMA full — 3
  different scorers; generic helper needs design).

**DEC-055 (depends_on):** ModelCard.depends_on field + R11 test
(importable). 66 cards migrated: 11 non-empty, 55 `[]` (no current
primitive imports — honest). described_only (162) not filled —
depends_on forward-scan pending DEC-056.

**DEC-056 (PRIMITIVES_BACKLOG):** `docs/ml_tools/PRIMITIVES_BACKLOG.md`
v1.0 (449 lines). 162 non-implemented cards scanned, clustered:
- **20 P1 clusters (~30 functions)** (3+ prospective consumers; some
  clusters are families: L3 kernels ×3, L6 laplacian+eigsh, L9 pinball
  loss+grad, L12 consensus+coassoc, L13 ranknet+listnet+lambdarank,
  L14 grid+random, L18 ks+chi2+psi): `_knn_query`, `_core_distances`,
  `_rbf_kernel`, `_chol_solve`, `_mst`, `_graph_laplacian+_eigsh_topk`,
  `_nmf_mu`, `_itemset_mining`, `_pinball_loss`, `_survival_logrank`,
  `_doubly_robust_score`, `_consensus_matrix`, `_ltr_gradients`,
  `_grid_search`, `_arm_stats`, `_adaboost_weight_update`,
  `_fuzzy_cluster_weights`, `_ks_stat`, `_eigen_decomp_topk`,
  `_cumulative_logit`.
- **15 P2** (2 consumers), **20 P3** (1, opportunistic).
- AP10 amendment: forward extraction allowed when prospective ≥ 2.
  Applied v0.9.134 (`_sq_dists`, `_acf`).

**Open items (deferred):**
1. **P1 implementation** (v0.9.143+): `_knn_query` → `_rbf_kernel`
   → `_chol_solve` (group A), then graph (group B), then
   factorization+itemset (group C), then losses+stats (group D).
2. **R12 (code-sync AST)** — deferred; needs PRIMITIVES_BACKLOG
   to whitelist internal helpers. v0.9.145+.
3. **described_only depends_on** — 162 cards. Fill with "not yet
   implemented" marker + PRIMITIVES_BACKLOG cross-ref. v0.9.145+.

---

## Historical: v0.9.128 — TAXONOMY FROZEN

**Freeze declaration** (DEC-052, AI_CONSTITUTION §5.3): **58/58 subtypes**
have ≥3 diverse candidates or explicit `# NOT-ENSEMBLE: <reason>`.
**No new cards without DEC entry + explicit rationale.** Post-freeze
allowed: `draft → verified` status flips, `ram_formula` completion,
`cross_ref` formalization, `ensemble_recommendation` refinements.

**Weak machine** (DEC-050): no GPU, 8-16 GB RAM. Card admissibility:
tabular + OOC-by-design + weak-machine fit + implementable-or-importable.
Architecturally non-OOC models (attention O(n²), 16 GB+ RAM) are not
cards even if SOTA.

**Counts:** 32 verified fns / 48 verified cards / 14 draft fns /
17 draft cards / 61 described_only / 6 inference-only / 5 last-priority (S).

**Invariants (schema-enforced):** R8 (input_api df|series|dual),
R9 (cadw_function importable), R10 (public `fit_*` covered — ml + ts),
CW-5 (guard-refuses-at-domain-bound), DEC-033 component-wise.

---

## Open post-freeze (v0.9.129+)

All **implementation**, not card-filling:

1. **Verify 4 ts drafts** — EnsembleForecaster / OTMThetaForecaster /
   TemporalHierarchy / TransformedTargetForecaster (`status: draft`,
   `ram_formula` pending). Requires per-model formula from code (AP23)
   + measurement (DEC-033).
2. **`fit_stacking` infrastructure** — general mechanism for
   `ensemble_recommendation` (k-fold OOC + meta-learner). Not a card,
   a function like `fit_ab_test`.
3. **`expansion_factor` migration** (18 call sites, 14 models) —
   mapping `ReduceOp`-class → `cadw_function` + re-measurement
   (batch increases → tighter peak → OOC regression risk).
4. **`cross_ref` schema field** — formalize DEC-049 markdown comments
   into validated field.
5. **Promotion `.pingis/impl/` → `cadw/ml/`** (11 models: KM, WeibullAFT,
   ComplementNB, NMF, bandits ×3, KernelDensity, reservoir, stratified
   split, OHE).

---

## Next 3 sessions

**Priority order (v0.9.143-155):**
1. **P1 primitives** (v0.9.143-146) — unblocks 54 forward-declared cards.
2. **impl/ 9 models promotion** (v0.9.147-155) — DEC-058 Part 2.
3. **cadwboost promotion** (v0.9.148+ overlay with #2) — DEC-058 Part 2.
4. **P2 primitives** — opportunistic, when 2nd consumer arrives.
5. **R12 + described_only depends_on code-sync** (v0.9.156+).

P1 first because each primitive (`_knn_query`, `_rbf_kernel`, ...)
unblocks 3-12 cards at once; promotion is 1-2 sessions per model.
See Backlog §impl/cadwboost for promotion details.

### v0.9.143 — P1 group A: distances + kernels

**L1** `_knn_query` (~1 sess) — chunked sq_dists + argpartition.
**L2** `_core_distances` (~0.5) — k-th neighbour distance.
**L3** `_rbf_kernel` + `poly_kernel` + `laplacian_kernel` (~1) —
pairwise kernel general helper (parameterized).
**L4** `_chol_solve` (~0.5) — np.linalg.cholesky + solve; jitter.
Home: `cadw/ops/linalg.py`.
Unblocks: LOF, DBSCAN, HDBSCAN, SMOTE, KNNImputer, ML-kNN, ItemKNN,
Spectral ×3, KernelPCA, SparsePCA, GPLVM, GPR, GP forecaster, GP-EI,
SMAC, BayesianOpt_GP.

**Also (~30 min):**
- `predict_batched` measurement for 8 false-cards (OTM/THF/TTF,
  PELT, STRAY, BinarySeg ×2): verify predict path.
- `fit_stacking` pilot: `cadw/ml/stacking.py::fit_stacking` (k-fold
  OOC + meta on OOF). Infrastructure, not card.

### v0.9.144 — P1 group B: graph + eigsh + MST
L5 `_mst`, L6 `_graph_laplacian` + `_eigsh_topk`, L19
`_eigen_decomp_topk`. Home: `cadw/ops/graph.py` (new).
Unblocks: HDBSCAN dendrogram, Agglomerative single, Spectral ×3,
Consensus ×3.

### v0.9.145+ — P1 groups C, D, R12, described_only depends_on
C: `_nmf_mu`, `_itemset_mining` (Apriori/FP/Eclat).
D: `_pinball_loss`, `_survival_logrank`, `_doubly_robust_score`,
`_ltr_gradients`, `_arm_stats`, `_adaboost_weight_update`,
`_fuzzy_cluster_weights`, `_ks_stat`, `_cumulative_logit`.
Then R12 (code-sync) + described_only depends_on.

## Deferred (prior session plan, NOT executed)

**Note:** v0.9.129 actually did 3 ts drafts (not 4 — EnsembleForecaster
excluded); DEC-054 dedup (v0.9.130-133) took priority over the planned
`fit_stacking` / `expansion_factor` migration / `.pingis/impl`
promotion. These remain deferred to v0.9.137+. Original plan text below.

**Part A (~1-1.5 h): verify 4 ts drafts.**
Probe `cadw/ts/{ensemble,otm,thf,transforms}.py` for state_nbytes +
batch transient. For each: `ram_formula` + `formula_evidence` +
`measurement: {verification_mode: component-wise (DEC-033)}` +
`test_<card>_verification.py` (CW-1..5). Status `draft → verified`.

**Part B (~1-1.5 h): `fit_stacking` pilot.**
`cadw/ml/stacking.py::fit_stacking(df, base_models, meta_learner, cv=5,
output="auto")`. OOC: k-fold residual predictions via `_iter_filtered_batches`,
each base model fit per fold (existing `fit_*` reuse), meta on OOF
predictions. Test on A1 (LogReg + GBDT + NB → stacked LogReg).
Not a card; infrastructure function like `fit_ab_test`.

**Part C (bonus, if time): `cross_ref` schema field.**
ModelCard dataclass extension, `_parse_model` handling,
`test_R11_cross_ref_consistency` (canonical home must exist).

### v0.9.130 — `expansion_factor` migration (18 call sites)

Mapping `ReduceOp`-class → `cadw_function` (e.g., `XTXOp` → `fit_linear`,
`KMeansStepOp` → `fit_kmeans`, `_ArmStatsOp` → `fit_ab_test`). Re-measure
14 models with `solve_batch`. Separate commit per model group. High OOC
regression risk — see A21 in SESSION_BRIEF for protocol.

### v0.9.131 — promotion `.pingis/impl/` → `cadw/ml/`

11 models: KaplanMeier, WeibullAFT, ComplementNB, NMF, EpsilonGreedy,
UCB1, ThompsonSampling, KernelDensity, ReservoirSample,
StratifiedSplit, OneHotEncoder. Each: move code, `ram_formula` from
code, CW-1..5, status flip `described_only → verified`.

### v0.9.131+ — pending fns (A_classification ×8 + O_eda ×16)

Component-wise verify per pending card. Some have `status: None` legacy
(no `implementation_status` block) — normalize first.

---

## Session history (v0.9.105-128 — condensed)

**v0.9.159 (GBDT family batch 1: 5 cards):**
- Added state_nbytes() to GBDTRegressor, GBDTClassifier,
  GBDTMulticlassClassifier (was missing): exact tree byte count
  via _deep_nbytes + min/max + header.
- Flipped 5 cards: GBDT (binary), GBDT_multiclass, GBDT_L2,
  GBDT_scale_pos_weight ×2 (A_classification + E_anomaly).
- ram_formula (upper bound, iforest precedent):
  n_estimators * (2**(max_depth+1)-1) * 500 + 2*d*f64 + 256
  multiclass: × n_classes + n_classes*f64
- Tests: 8 passed (CW-1 upper bound [0.20, 1.0]; CW-2 n-independent
  ratio <=3.0; CW-6 guard-refuse at 0.001 MB).
- Deferred v0.9.160: GBDTForecaster + GBDT_multiseries
  (need _models populated post-predict for state_nbytes>0).

**v0.9.158 (legacy TS batch 2: SARIMA + VAR):**
- Flipped 2 remaining legacy TS cards: SARIMA (add state_nbytes
  = 2n*f64 + 256 for y_raw_ + _y_centered_) + VAR (existing
  state_nbytes = n*k*f64 + k*k*p*f64 + k*f64 + 256).
- Tests: 6 passed (exact formula; state grows linearly;
  guard refuses at n=500k + 2MB).
- Remaining TS legacy: GBDTForecaster/GBDT_multiseries (real
  CW-1..5 in v0.9.159+), backtest_forecaster (utility, not model).

**v0.9.157 (legacy TS batch 1: 5 naive forecasters):**
- Flipped 5 single-series TS cards from status:None to verified
  via DEC-011 bounded_by_domain (n<=1e6): NaiveForecaster, ETS,
  AutoETS, Theta, AutoTheta.
- Added NaiveForecaster.state_nbytes() (was inherited 0) ->
  n*f64 + 256.
- ram_formula per card from state_nbytes body (exact, not peak):
  Naive n*f64+256, ETS 3n*f64+256, AutoETS 3n*f64+256 (delegated),
  Theta 2n*f64+256, AutoTheta 2n*f64+256 (delegated).
- Tests: 13 passed (exact formula; state grows linearly;
  guard refuses at n=500k + 2MB for all 5).
- Remaining TS legacy: SARIMA (needs state_nbytes), VAR (verify
  check), GBDTForecaster/GBDT_multiseries (NOT DEC-011,
  real CW-1..5 in v0.9.159+).

**v0.9.156 (P1 L14: grid_search + random_search):**
- Extended `cadw/ops/optim.py`: grid_search (exhaustive cartesian)
  + random_search (uniform / categorical / callable distributions,
  seeded RNG). Generic; ETS-specific `_grid_search` NOT touched
  (coordinate α/β/γ/φ refiner, different shape).
- Tests: 16 passed (best selection maximize/minimize; grid
  exhaustive; random reproducibility; different seeds differ;
  callable spec; guard refuses).
- YAML: 2 cards flipped (GridSearch, RandomizedSearch).
- **Group D complete** (8/9: L9, L10, L11, L13, L14, L16, L18,
  L20). HMM remains (model, not primitive).

**v0.9.155 (P1 L10: _survival_common logrank):**
- New `cadw/ml/_survival_common.py`: logrank (k-sample,
  Peto & Peto 1972 / Mantel-Haenszel).
- Breslow partial log-lik NOT duplicated (lives inline in
  cadw.ml.coxph, verified; different signature X-feature matmul).
- Tests: 10 passed (identical/different groups, 3-group same/diff,
  all-censored, deterministic balance, invalid args). Flake fix:
  same-dist chi2 asserted via p > 0.001 (not fixed chi2<4).
- YAML: 1 card flipped (RandomSurvivalForest).

**v0.9.154 (P1 L13: _ltr_common):**
- New `cadw/ml/_ltr_common.py`: ranknet_grad (pairwise logistic,
  Burges 2005), listnet_grad (Plackett-Luce, Cao 2007),
  lambdarank_grad (pairwise with NDCG@k delta, Burges 2010).
  lambdamart.py NOT touched (verified, inline _lambda_gradients
  has different signature).
- Tests: 19 passed (grad sum-zero, direction, antisymmetry,
  NDCG@k perfect/reversed/truncated/zero-labels).
- YAML: 2 cards flipped (PairwiseLogistic, ListNet). LambdaMART
  already implemented.

**v0.9.153 (P1 L18: stats drift helpers):**
- Moved ks_2samp/psi/chi2_categorical from cadw.ml.drift to
  cadw.ops.stats (canonical home, DEC-054 §5.1). Added
  ks_2sample alias (YAML refs). drift.py re-imports; 27 existing
  drift tests still green.
- Tests: 14 passed (KS identical/shifted/empty; PSI 3 thresholds;
  chi2 same/different/single-cat/empty; drift re-export).
- YAML: 3 cards flipped (EvidentlyDrift, ADWIN, SequentialTest).

**v0.9.152 (P1 L11: _causal_common dr_score):**
- New `cadw/ml/_causal_common.py`: dr_score (AIPW doubly robust)
  + aipw_ate convenience. Robins/Rotnitzky/Zhao 1994 +
  Chernozhukov et al. 2018 (DML).
- Tests: 10 passed (exact-models case; DR against biased
  propensity; DR against biased mu; clipping at e∈{0,1};
  term decomposition).
- YAML: 4 cards flipped (DR_learner, CausalForest ×2,
  DR_learner_uplift).

**v0.9.151 (P1 group D partial: _gbdt_common):**
- New `cadw/ml/_gbdt_common.py`: pinball_loss, pinball_grad,
  adaboost_weight, cumulative_logit_prob. L9+L16+L20.
- Tests: 15 passed (loss/grad numeric consistency; weighted
  up/down AdaBoost; logit sums-to-1 + monotone + numerical
  stability at |eta|=1000).
- YAML: 5 cards flipped (OrderedProbitBayesian, RUSBoost,
  LinearQuantReg, GBDT_pinball, CQR).

**v0.9.150 (P1 L8 complete: fp_growth + prefix_span + gsp):**
- cadw/ops/itemset.py extended with:
  - fp_growth (projected-DB pattern growth, Han 2000 semantics)
  - prefix_span (prefix-projected DB DFS, Pei 2001)
  - gsp (level-wise sequential, Srikant & Agrawal 1996)
- Tests: 12 passed (fp_growth == apriori crosscheck; gsp ==
  prefix_span crosscheck; random 200 seqs; max_len; guard).
- YAML: 3 cards flipped (FP-growth, PrefixSpan, GSP). HMM stays
  planned (model, not primitive — v0.9.152+).
- R_association: 5/6 cards now flipped (HMM pending).

**v0.9.149 (P1 L8: itemset apriori + eclat):**
- New `cadw/ops/itemset.py`: support_count + candidate_generation
  + apriori + eclat. L8 was a single dispatcher (itemset_mining)
  in PRIMITIVES_BACKLOG; split into specific functions because
  a monolithic dispatcher with NotImplementedError branches would
  dishonestly flip FP-growth card (R11 static check can't see
  runtime raise).
- FP-growth, PrefixSpan, GSP -> v0.9.150 (specific functions).
- Tests: 16 passed (support/candidate classic; apriori=eclat
  cross-check on random data; guard refuses).
- YAML: 2 cards flipped (Apriori, Eclat) with deps rewritten
  from itemset_mining to specific functions.

**v0.9.148 (P1 L17: normalize_weights):**
- Extract `normalize_weights` to `cadw.ops.stats` (P1 L17).
  4 consumers: ML-kNN, SMOTE, WeightedSample, +1.
- Tests: 9 passed (sum-to-1, all-zero->zeros, empty, negative
  raises, 2D shape, single-element, int coerced, tiny-sum eps floor).
- YAML: 5 cards flipped (ML-kNN, SpectralClustering-dup, SMOTE,
  WeightedSample, SMAC-dup).

**v0.9.147 (P1 group B: L5+L6):**
- New `cadw/ops/graph.py`:
  - `graph_laplacian(W, normalized ∈ {none, sym, rw})`
  - `eigsh_topk(L, k, which='SM')` — graph-aware wrapper over eigh_topk
  - `mst(n, edges)` — Kruskal via scipy.sparse.csgraph
- Tests: 20 passed (unnorm L row-sums 0; sym L eigenvalues [0,2];
  rw L row-sums 0; SM λ_0≈0 for connected; mst 3-4 vertex known
  cases; disconnected -> forest + warning; duplicate edge handling;
  guard refuses).
- YAML: 7 cards flipped (SpectralClustering ×3, SpectralCoclustering,
  SpectralBiclustering, HDBSCAN ×2, SMAC-dup). 4 remain planned in
  B_clustering (Consensus ×3, LaplacianEigenmaps — outside scope).

**v0.9.146 (P1 L19: eigh_topk):**
- Extract `eigh_topk` to `cadw.ops.linalg`. 8 direct consumers.
  which ∈ {LA, SA, LM, SM}; n<=2000 dense numpy.eigh; n>2000
  scipy.sparse.linalg.eigsh (SM uses shift-invert).
- Tests: test_linalg_eigh_topk.py (11 passed, 1 skip: scipy
  path slow to generate).
- YAML: 3 unique + 1 dup card flipped (GaussianProcessRegressor,
  KernelPCA, GPLVM, SMAC-dup). Spectral ×5 remain planned (blocked
  on graph_laplacian + eigsh_topk, L6 — next group).

**v0.9.145 (P1 group A: L3+L4):**
- Extract `pairwise_kernel` (rbf/poly/laplacian, parameterized)
  + `chol_solve` (with jitter fallback to np.linalg.solve) to
  `cadw.ops.linalg`. L3: 10 cards, L4: 5 cards.
- Convenience wrappers: rbf_kernel/poly_kernel/laplacian_kernel.
- Tests: test_linalg_kernels.py (16 passed: correctness vs
  manual, symmetric, diag=1 for rbf, chunk-invariant, guard,
  chol jitter fallback).
- YAML: 8 cards flipped (SparsePCA, GaussianProcessForecaster,
  GP-EI, SMAC, BayesianOpt_GP). 8 remain planned — blocked on
  eigh_topk (L19) or graph_laplacian/eigsh_topk (L6).

**v0.9.144 (P1 group A: L1+L2):**
- Extract `knn_query` + `core_distances` to `cadw.ops.linalg`
  (P1 L1 + L2, DEC-056). knn_query: 8 cards, core_distances:
  3 cards. Both chunked X streaming, OOC in n_X.
- Tests: test_linalg_knn.py (15 passed: brute match, chunk
  invariant, cosine, guard refuses, OOC in n).
- YAML: 4 cards flipped to implemented (DBSCAN, LOF, ItemKNN,
  KNNImputer — all deps importable). 4 remain planned: HDBSCAN×2
  blocked on mst, ML-kNN/SMOTE blocked on normalize_weights.

**v0.9.143-batch5 (duplicates + reservoir):**
- Extract `reservoir_sample` + `batch_to_features` to
  `cadw.ops.sampling` (P2, DEC-056). Refactor kernel_density.py
  to use them (delete inline copies).
- Promote ReservoirSample (O_eda): `fit_reservoir_sample` in
  `cadw.ml.reservoir_sample`.
- Promote OneHotEncoder (O_eda): `fit_one_hot_encoder` in
  `cadw.ml.one_hot_encoder`.
- Promote StratifiedSplit (O_eda): `fit_stratified_split` in
  `cadw.ml.stratified_split` — thin wrapper over verified
  cadw.cv.KFold (fixes impl's rng-in-loop bug).
  depends_on: [cadw.cv.KFold].
- Tests: ReservoirSample 10, OneHotEncoder 5, StratifiedSplit 9.

**v0.9.143-batch4 (CNB + KDE):**
- Promote ComplementNB + KernelDensity (J_nlp + N_density):
  `fit_complement_nb` in `cadw.ml.complement_nb`,
  `fit_kernel_density` in `cadw.ml.kernel_density`.
  Two YAML cards: None -> verified (DEC-033 component-wise).
- Tests: CNB 9, KDE 9+1skip.

**v0.9.143-batch3 (survival pair):**
- Promote KaplanMeier + WeibullAFT (I_survival × 2): `fit_kaplan_meier`
  in `cadw.ml.kaplan_meier`, `fit_weibull_aft` in `cadw.ml.weibull_aft`.
  Two YAML cards: None -> verified (DEC-033 component-wise).
- Tests: 15 passed, 2 skipped (KM OOC sanity skipped per DEC-034;
  primary evidence = state_nbytes_exact + state_independent_of_n).
- R2 fix: c_batch -> 4.0 in ram_formula (same precedent bandits 3.5).

**v0.9.143-batch2 (NMF):**
- Promote NMF × 2 (C_dim + J_nlp): `fit_nmf` in `cadw.ml.nmf`.
  Two YAML cards: None -> verified (DEC-033 component-wise).
- P1 primitive `cadw.ops.linalg.nmf_mu` (L7, DEC-058).
- Tests: 10 passed.

**v0.9.143 (bandits batch):**
- Promote bandits ×3 (EpsilonGreedy, UCB1, ThompsonSampling):
  `fit_epsilon_greedy`, `fit_ucb1`, `fit_thompson_sampling` in
  `cadw.ml.bandits`. 3 YAML cards: None -> verified (DEC-033).
- P1 primitive `cadw.ops.stats.arm_stats` (L15, DEC-058).
- Tests: 29 passed (state_nbytes_exact, OOC, guard, correctness).

**v0.9.129-141 (this session, 13 releases + 5 docs commits):**
- v0.9.129: verify 3 ts drafts (OTM/THF/TTF).
- v0.9.130-133: K1/K2/K3a/K5c+K6+K10+K12 dedup (7 clusters, ~600 lines removed).
- v0.9.134: home-correction + forward extraction + version drift fix.
- v0.9.135: depends_on schema + migration 66 cards + R11.
- v0.9.136: PRIMITIVES_BACKLOG (449 lines, 20 P1).
- DEC-057 docs: OOC-batch audit.
- v0.9.137: fit_batched/predict_batched + metric fix (py_bytes -> measure_peak_traced) + R13.
- v0.9.138: verify KMeans/MiniBatch/PCA + OOC fix (batch_size auto-derive).
- v0.9.139: verify BVAR/VECM + OOC fix.
- v0.9.140: verify PELT×2/T2Q/STRAY + T2Q silent OOM fix.
- v0.9.141: verify T_learner/S_learner (final 2 of §1).

**v0.9.129-133 (prior session segment, 5 releases):**
- v0.9.129: verify 3 ts drafts (OTM/THF/TTF) — 27 new tests + 3 YAML.
- v0.9.130: K1 `_probe_memory_limit_mb` 16 -> 1 (AST migration).
- v0.9.131: K2 `_sigmoid` 4 -> 1 (+ numerical stability fix).
- v0.9.132: K3a `_ols_fit`+`_erfinv` 2+2 -> 1+1 (`cadw/ops/numeric.py`).
- v0.9.133: K5c/K6/K10/K12 — 4 clusters (8 files, 89+/45-).

**v0.9.119-128 (prior session, 10 releases):**
- v0.9.119: pipeline planner v2 + `solve_batch` (DEC-044) + `fit_ab_test` ×3 verified.
- v0.9.120: P6'–P9' real pipelines + `calibrate_throughput.py` (DEC-046).
- v0.9.121: R10 gate (public `fit_*` coverage) + GBDT_multiclass fix (DEC-047).
- v0.9.122: TASK_TYPES 14→28 (DEC-048) + CADWBoost SOTA reference + ElasticNet.
- v0.9.123: B/C sections (from P/Q) + DEC-049 cross-ref policy.
- v0.9.124: `.pingis/impl/` sync (14 purpose refs).
- v0.9.125: R/S/T/U/V/W sections (27 described_only).
- v0.9.126: cleanup — −21 cards, −1 section (U), rename P→B/Q→C, weak-machine (DEC-050).
- v0.9.127: `cadw.ts` coverage gap — 5 cards + R10 ts extension (DEC-051).
- v0.9.128: **TAXONOMY FROZEN** — 42 candidates + 8 NOT-ENSEMBLE (DEC-052).

**v0.9.105-118 (prior 14 releases):** formula calibration procedure
(EXPERIMENT_PROCEDURE), framework chunking (DEC-022/023), GMM/BIRCH/
iforest/linucb/gmm verified (DEC-033 component-wise), LDA/TF-IDF with
auto-K-spill (DEC-041/042), sub_pca, chains, ordinal verified. 31
verified fns at v0.9.118; 32 at v0.9.119.

---

## Status baseline

| Status | fns | cards | Что |
|---|---|---|---|
| verified | **40** | 56 | +3 bandits ×2 (impl promo v0.9.143) |
| draft | **14** | 17 | 4 ts (formula pending v0.9.129) + 10 prior (BVAR/VECM, binseg×2, pelt×2, t2q, meta_learner, ab_test×3) |
| described_only | — | 61 | 19 in B/C/R/S/T/V/W + 42 v0.9.128 candidates |
| inference-only | — | 6 | Chronos×3, MiniLM, DeepSurv, DeepHit |
| last-priority | — | 5 | S_generative (TVAE/CTGAN/CTAB-GAN+/TabDDPM/TabSyn) |

---

## Lessons learned (all sessions — consolidated)

**YAML / schema:**
- `str.replace` may hit 2+ cards (`bounded_state_dims` everywhere). Use
  regex anchor `- name: <Card>` + next `- name:`.
- Verify indentation conventions per file — legacy (A-H, I, J, M, O) use
  `id=0, name=2`; new (B/C/N/R/S/T/V/W) use `id=2, name=6`.
- `bounded_state_dims` excludes auto-injected `batch`/`row_size` (R2).
- `cadw_measured` requires `algorithm.startswith("cadw.")` (schema).

**Patch protocol:**
- **Probe → Patch → Verify**, never blind (`AP10`).
- `git add -A` commits agent artifacts. Use explicit list.
- Never mix `git checkout <file>` and patch in one block (checkout reverts).
- Insertion in Python files: `from __future__` is first statement after
  docstring (PEP 236). Check before inserting before it.

**Measurement (DEC-033):**
- `measure_peak` (ru_maxrss delta) fails when baseline >> fit peak.
  Component-wise: state_nbytes exact + coef isolated + OOC proven.
- `d=10` noise > signal. Use `d>=200`. `coef` stable; `intercept` from
  slope-fit unreliable. `state(n1)==state(n2)` fails for n>=S (reservoir
  order) — use bounded ≤1.5x.

**Verification:**
- `draft → verified` requires formula from code (AP23) + measurement.
  Formula from measurement = AP23 anti-pattern.

**Freeze (v0.9.128):**
- Taxonomy filling became dominant activity (5 of 10 recent releases).
  Freeze after 3+ diverse per subtype prevents infinite loop.

---

## Do not re-solve

Per STREAMING §6.1 — won't-fix with rationale:

- **eif bit-identical chunked**: 40 GB I/O at n_est=100.
- **kmeans_ad bit-identical**: 3.2 GB I/O at k=8.
- **autoencoder beyond n_w = state_limit/16**: 2-3 sessions, no use case.
- **sparse tfidf.transform**: needs CSR SpillBuffer design.
- **binseg / stray**: intrinsic O(n) / O(n²). `refuse_at_guard`.
- **TS single-series fits** (STL, VAR, VECM, ETS, Theta): `tier1_core_exception`.
- **AutoML frameworks** (AutoGluon/H2O/Auto-sklearn): framework ≠ model
  (DEC-050). If CADW needs stacking, use `fit_stacking` (v0.9.129).
- **Neural architectures for non-tabular** (vision/graph/NLP transformers):
  reference-only, but not cards (DEC-050).
- **TabPFN in OOC paradigm**: 16 GB CPU + attention O(n²) + n≲1K —
  architecturally non-OOC (DEC-050).

---

## Backlog (not scheduled)

- **`fit_stacking` generalization**: CatBoost-style ordered boosting when
  base models are GBDT. Marginal over existing.
- **`_probe_memory_limit_mb` refactor**: DONE v0.9.130 (K1).
- **`_ses` parameterized unification**: 3 signatures, defer until
  3rd/4th real caller demands shared signature (AP22).
- **`_grid_search` parameterized helper**: `grid_search(param_grid,
  scorer, refine_axes=None)` for OTM/ETS/SARIMA — v0.9.135+.
- **VAR streaming rewrite** (DEC-057 F3): could use BVAR/VECM
  pattern (`XTX += X.T@X` + rolling buf). Deferred v0.9.140+.
  Honest as-is (`bounded_state: false, state_class: O(N·d)`).
- **AST audit helper** (DEC-057 F4): delegate resolution + trans-
  module + cadw/ts scan. One-off script for future OOC audits.
- **EnsembleForecaster verify** (DEC-057 open A.2): deferred until
  `fit_stacking` (v0.9.142). Rewrite as thin wrapper; do not
  duplicate recipe (DEC-050 level 3, DEC-052).
- **predict_batched measurement** (DEC-057 open A.1): 8 false-cards
  + 5-10 true-cards, verify predict path matches fit_batched.
- **AP2 refactor: meta_learners** (fit_meta_learner): streaming
  GBDT-per-arm via `df.filter(t==arm)` instead of `_collect` +
  vstack. v0.9.142+ (does not block verify; T/S verified honest
  via bounded_by_domain).
- **described_only depends_on** (162 cards): fill `depends_on: []`
  or forward-declare P1 primitives with `depends_on_backlog:
  PRIMITIVES_BACKLOG §<cluster>` marker. v0.9.145+.
- **R12 code-sync AST** (DEC-055): per-card `depends_on` matches
  actual imports. Deferred v0.9.145+ (needs PRIMITIVES_BACKLOG).
- **impl/ 9 models promotion** (DEC-058 Part 2): KaplanMeier,
  WeibullAFT, ComplementNB, NMF, EpsilonGreedy, UCB1,
  ThompsonSampling, KernelDensity — production-grade OOC-first
  in `.pingis/impl/`. Adapt + CW-1..5 + YAML verified. 3 duplicates
  (OneHotEncoder/ReservoirSample/StratifiedSplit) — verify against
  existing cadw API. ~10-15 sessions. Target: v0.9.145-155.
- **cadwboost promotion** (DEC-058 Part 2): 3086L in
  `.pingis/drafts/cadwboost/`, symmetric+leaf-wise trees, quantile
  binning. Compare vs `cadw.ml.gbdt`; promote if wins. ~2-3
  sessions. Target: v0.9.148+.
- **scripts/ cleanup** (DEC-058 Part 3): 4.6 MB, 163 tracked files.
  Applied patch chains in `scripts/v0.9.26/28/29/` (71 files,
  historical) + 9 `patch_v0930*.py` in `scripts/` root (last, referenced
  in `docs/SESSION_BRIEF_v0.9.97.md` + `roadmap_research`). Proposed:
  `git mv scripts/v0.9.26/ scripts/v0.9.28/ scripts/v0.9.29/
  scripts/patch_v09{0,1,2,3}*.py scripts/archive/patches/` — preserves
  history, declutters `scripts/` root. Bench + audit scripts stay
  (`bench_*.py`, `audit_*.py`, `calibrate_throughput.py`,
  `collect_project.py`, `reproduce_benchmark.py`). Target: v0.9.156+.
  Precedent: `docs/roadmap_research/consolidated_roadmap.md` already
  proposed `scripts/archive/patches/v0.9.0-0.9.20/`.
- **`.pingis/` full removal** (DEC-058 Part 4, after impl/cadwboost
  promotion): after v0.9.155 (impl promotion) + v0.9.150 (cadwboost),
  the remaining `.pingis/` (session.log 4.4M + audits 432K + patches
  16K + benchs 284K + algorithm_notes 120K + attachments 136K +
  tasks 200K + plan_v0.9.87 12K) can be moved to `docs/archive/pingis/`
  or removed. Bench JSONs may be promoted to `benchmarks/` first.
  Target: v0.9.157+.
- **`fit_meta_learner` AP2 refactor**: T/S learner verify requires
  removing `_collect + vstack` (DEC-012 pattern).
- **Pre-commit hook for `render_matrix.py`**: `--check` flag prevents
  doc-sync drift (past: 12 versions drift).
- **`bounded_by_domain` on more cards**: currently 5 (coxph, pelt×2, others);
  add when formula requires n-bound.
- **`SlowOperationWarning` calibration**: `estimate_cost` exists (v0.9.119),
  per-model rates pending (v0.9.129+).

---

## Session discipline

- Before commit: `pytest -q <affected files>` (focused, ~20s).
- Before tag: `pytest -q -m "not slow"` (full fast, ~10min).
- Before release: `pytest -q` (all, includes slow).
- See `AI_CONSTITUTION.md` §4 for any new `fit_*` / `predict_*`.
- See `EXPERIMENT_PROCEDURE.md` §3 for card verification (6 steps).
- See `SESSION_BRIEF_v0.9.129.md` for next-session bootstrap.
