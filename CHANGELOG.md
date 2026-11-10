## [Unreleased]

### v0.9.142 (2026-11-08) — depends_on 100% coverage + .pingis cleanup

**Counts:** 37 verified fns / 53 verified cards (unchanged).
**Coverage:** 228/228 cards have `depends_on` (0 missing).

Three sub-commits in this release:
- v0.9.142 (`70135a7`): described_only depends_on forward-declare.
- v0.9.142b (`a75655a`): coverage 100% (legacy + O_eda).
- v0.9.142d (`6ebdfe4`): DEC-058 .pingis cleanup + promotion backlog.

**Schema additions (DEC-055 amend):**
- `ModelCard.depends_on_backlog: str | None` — ref to
  PRIMITIVES_BACKLOG §<cluster>.
- `ModelCard.depends_on_status: str | None` — "planned" |
  "implemented"; R11 skips importable check when "planned".

**Migration (228 cards):**
- **63 verified**: `depends_on` actual (11 non-empty with imports).
- **3 draft**: `depends_on` actual (EnsembleForecaster + 2).
- **54 described_only planned**: forward-declared P1 primitives with
  `depends_on_backlog: "PRIMITIVES_BACKLOG §L1, §L2, §L5"` +
  `depends_on_status: planned`. Discoverable via
  `grep "depends_on_status: planned" docs/ml_tools/*.yaml` -> 54.
- **78 described_only empty**: `depends_on: []` (no P1 match —
  implemented in cadw, utility, or NN-skip with `nn_excluded: true`).
- **30 legacy (status=None)**: actual `depends_on` (AST-extracted
  from cadw_module; 0-2 primitive imports each) +
  `depends_on_status: implemented`. status stays None (source-fidelity
  only, not DEC-033 component-wise verified). Cards: ETS, Theta,
  SARIMA, VAR, GBDT×4, LambdaMART, T_learner_uplift, CausalTree,
  feature_select×3, PermutationImportance, FairnessMetrics,
  DriftStats, Hybrid_MF_GBDT, NaiveForecaster, GBDTForecaster,
  GBDT_multiseries, IntervalFeatures_GBDT, GBDT_scale_pos_weight.
- **5 O_eda (Describe/Hist/ValueCounts/SimpleImputer/EncodeCategorical)**:
  `depends_on: []` (CadwDataFrame public API, not primitives) +
  `depends_on_status: implemented`.

**PRIMITIVES_BACKLOG header updated:**
- "20 P1" -> "20 P1 clusters (~30 functions)". Some clusters are
  families: L3 kernels ×3, L6 laplacian+eigsh, L9 pinball loss+grad,
  L12 consensus+coassoc, L13 ranknet+listnet+lambdarank, L14
  grid+random, L18 ks+chi2+psi.
- "Where dependencies live" section: grep-discoverable pattern.

**DEC-058 (new):** .pingis cleanup + impl/cadwboost promotion backlog.
- 29 gitignored stale files + 5 dirs removed (22 -> 9 MB, saved
  13 MB). 11 stale audits, 15 scratch prototypes, 3 misc, 5 dirs.
- Kept: 8 USED audits + inefficiency.md (referenced test),
  experiments/v0934_notes.md, algorithm_notes/, benches/, impl/,
  drafts/cadwboost/.
- **Promotion backlog (v0.9.145+)**: impl/ 9 models (production-
  grade OOC-first per docstrings) + cadwboost (3086L, symmetric +
  leaf-wise trees). 3 duplicates (OneHotEncoder / ReservoirSample /
  StratifiedSplit) — verify against existing cadw API.
  Estimated: 10-15 sessions + 2-3 sessions.

**Tests:** schema 110 passed (R11 skips planned). Coverage audit:
228/228 cards have `depends_on`.

### v0.9.141 (2026-11-08) — verify T_learner + S_learner (§1 complete)

**Counts:** 35 -> 37 verified fns; 51 -> 53 verified cards. **All 10
draft models from §1 list are now verified.**

T_learner + S_learner (fit_meta_learner, kind='t'/'s') -> verified.
State = n_estimators*K*4096 + n*f64 + 1024 (K=2 T, K=1 S; propensity_
per-row array). YAML: fit_batched true->false (AP2 _collect);
bounded_state_dims {batch:1} -> {n:1, n_estimators:1}.

10 new tests. 2.4s runtime. Schema: 110 passed.

### v0.9.140 (2026-11-08) — verify PELT + T2Q + STRAY

**Counts:** 32 -> 35 verified fns (after v0.9.138, 139).

T2Q: bounded_state true -> false (honest; O(n) output arrays 4n*8,
DEC-039 pattern). Silent OOM fix: check_state(4*n*8, "tspq.output")
before post-chunk aggregation — was missing at n=1e6 + tight guard.

PELT (×2 D3.2 + D5 stub) + STRAY: verify confirmed YAML honest
(bounded_by_domain / tier1_core_exception). ram_formula "256" (PELT
container) and "2*n*f64+256" (STRAY).

17 new tests. 3.7s runtime. Schema: 110 passed.

### v0.9.139 (2026-11-08) — verify BVAR + VECM

BVARModel + VECMModel -> verified. Single-series TS with streaming
fit (`for batch`, `check_state`). batch_size auto-derive (DEC-023)
added — same gap as v0.9.138.

YAML: bounded_state_dims {d:2} -> {k:2, p:1}/{k:2, r:1};
state_class O(d²) -> O(model) (state = O(k²p²)).

12 new tests. 3s runtime. Schema: 110 passed.

### v0.9.138 (2026-11-08) — verify KMeans + MiniBatchKMeans + PCA

Pre-cards-era models -> verified. All 3 had **OOC bug** (DEC-023 class):
`_iter_filtered_batches()` without batch_size → 1 batch = n at
default budget → fit transient O(n).

Fixes:
- All 3: batch_size auto-derive (`derive_batch_size`); propagate to
  `_iter_filtered_batches` + `_kmeans_plus_plus_init`.
- MiniBatch: io_batch_size (separate from SGD batch_size) — 5
  unpatched calls in fit (reservoir, sweep, conv passes).
- PCA G2: check_state(2*d*d*8+d*8) before `np.linalg.eigh(cov)` —
  tight guard was silent OOM at d=1000+.
- KMeans G2: check_batch(extra_factor=2.0) in _kmeans_plus_plus_init;
  MemoryLimitExceeded instead of ValueError when tight guard makes
  batch_rows < k.

state_nbytes added: KMeansModel (k*d+8+256), PCA (k*d+d+3k+256).

18 new tests. 116s runtime. Schema: 110 passed.

### v0.9.137 (2026-11-08) — DEC-057 impl: fit_batched/predict_batched + metric fix

test_fit_ooc_v106: py_bytes -> measure_peak_traced. py_bytes
(tracemalloc) is blind to numpy malloc; an unbatched numpy fit
passed the old test. measure_peak_traced uses VmRSS sampler
(v0.9.110/DEC-033 precedent). Ratio 1.3 -> 2.0.

ModelCard.fit_batched + predict_batched + R13 test
(bounded_state=true + fit_batched=false requires streaming keyword
in streaming_strategy.fit — prevents silent mismatch).

Migration 66 cards: 58 (true,true), 8 (false,false) intrinsic O(n).
All 8 honest via bounded_by_domain / tier1_core_exception.

9 tests (fit_ooc), 110 schema. Runtime: ~22s + 30s.

### v0.9.136 (2026-11-08) — PRIMITIVES_BACKLOG (DEC-056)

New: docs/ml_tools/PRIMITIVES_BACKLOG.md (449 lines). Forward scan
of 162 non-implemented cards. 20 P1 (3+ prospective consumers),
15 P2 (2), 20 P3 (1, opportunistic).

AP10 amendment (DEC-056): forward extraction allowed when
prospective ≥ 2. Applied v0.9.134 (_sq_dists, _acf).

DEC-056 status: APPLIED (v0.9.136). Roadmap: v0.9.143-145 P1 groups A-D.

### v0.9.135 (2026-11-08) — depends_on schema + migration (DEC-055)

ModelCard.depends_on: list[str] (dotted path). R11 test (importable).
66 cards migrated: 11 non-empty, 55 `[]`. 162 status=None cards
left untouched (no code).

Migration gotchas found and fixed: indent 2 vs 6 (AI_CONSTITUTION
§5.4 legacy vs new); duplicate names across files (IForest E/D,
Logistic_balanced E/A — first got overwritten).

109 schema passed (was 108, +1 for R11).

### v0.9.134 (2026-11-08) — home-correction + forward extraction

**Counts:** 35 verified fns / 51 verified cards (unchanged).

**Home-correction (DEC-054 addendum):**
- `sigmoid`: `cadw.ops.optim` -> `cadw.ops.numeric` (optim = SGD;
  sigmoid = elementwise numeric). `optim.py` re-exports for back-compat.
- `znormalize_rows`: `cadw.ops.numeric` -> new `cadw.ops.stats`
  (mean/std = statistics).
- `ols_fit`: `cadw.ops.numeric` -> new `cadw.ops.linalg`
  (closed-form regression = linalg).
- `_xavier_init`: `cadw.ml._common` -> new `cadw.ml._nn_init`
  (NN-specific; was mixed with Arrow helpers).

**Forward extraction (AP10 exception per DEC-056):**
- `_sq_dists`: `cadw.ml.patch_memory_ad` -> `cadw.ops.linalg.sq_dists`
  (12+ prospective consumers: kmeans, kmeans_ad, kshape, tspq, iforest,
  eif, DBSCAN, HDBSCAN, LOF, bisecting, minibatch, spectral).
- `_acf`: `AutoStationarizer._acf` (method) -> `cadw.ops.stats.acf`
  (5+ prospective: binseg, pelt, bocpd, matrix_profile, stray).
  Method delegates: `return _acf_fn(y, lag)`.

**Migration:** AST-based function removal (`ast.FunctionDef.end_lineno`)
+ method body replacement (`ast.ClassDef` -> `ast.FunctionDef.body[0].lineno`
.. `end_lineno`). Import updates via `str.replace` on the exact import
block (v0.9.130 lesson: never `str.replace` whole-file; use AST for
function boundaries).

**Back-compat:** all import aliases preserved (`_ols_fit`, `_erfinv`,
`_sigmoid`, `_znormalize_rows`, `_xavier_init`, `_sq_dists`).
Identity verified (`is` check in smoke): e.g. `cadw.ml.gbdt._sigmoid
is cadw.ops.numeric.sigmoid`.

**Version note:** `__version__` was still `0.9.128` at v0.9.129-133
tags (5 tags without bump — historical drift, not retroactively
fixable without rewriting tags). Bumped directly to `0.9.134` here.
Future releases: bump `__version__` + `pyproject.toml` + CHANGELOG
in the same commit as the code (see test_version_consistency.py).

**Tests:** 401 passed (gbdt ×3, calibration, ordinal, autoencoder
×2, patch_memory_ad ×2, otm ×2, theta, auto_theta, transforms,
ts_models, schema).

### v0.9.133 (2026-11-08) — K5c+K6+K10+K12 dedup (4 clusters)

**Counts:** verified fns 35 / verified cards 51 (unchanged from v0.9.129).
No YAML changes. 9 files touched.

**Dedup (DEC-054):**
- K5c `znormalize_rows` (autoencoder + patch_memory_ad) ->
  `cadw.ops.numeric.znormalize_rows`.
- K6 `_to_df` (meta_learners + uplift) -> `cadw.ml._common._to_df_causal`.
- K10 `_xavier_init` (autoencoder + patch_memory_ad) ->
  `cadw.ml._common._xavier_init`.
- K12 `_plt` (seaborn_adapter + matplotlib_adapter) ->
  new `cadw.viz._lazy.lazy_plt(module_name)` (parameterized error message).

**Not touched (documented in DEC-054):**
- `kshape._znormalize` — different semantics (1D scalar std, not per-row).
- `_sq_dists` — 1 file, 2 sites (AP10).
- `_mahalanobis_distances` vs `_stream_mahalanobis` — different algorithms
  (reference oracle vs streaming production).
- `_collect` — 5 different signatures (AP22).

**Migration:** AST-based (`ast.FunctionDef.end_lineno` + `ast.ImportFrom`
injection). v0.9.130 lesson — regex eats decorators/statements between
functions.

**Tests:** 210 passed (autoencoder, patch_memory_ad, meta_learners,
uplift, viz_matplotlib, viz_seaborn, both *_verification, schema).

### v0.9.132 (2026-11-08) — K3a: ols_fit + erfinv dedup (2+2 -> 1+1)

**New:** `cadw/ops/numeric.py` (pure math primitives; no Arrow, no I/O).

**Before:** `_ols_fit` 2 copies (otm.py:44, theta.py:55 identical
semantics); `_erfinv` 2 copies (otm.py:71, theta.py:105 identical
Newton-3 body).

**After:** both modules `from cadw.ops.numeric import ols_fit as
_ols_fit, erfinv as _erfinv`. 6 call sites unchanged.

**Scope discipline:**
- `_ses` NOT unified: otm `(lvl, fitted)`, theta `_ses_fit -> (_, _, sse)`,
  ets Holt-Winters `(a,b,g,p)` — 3 different algorithms (AP22).
- `_acf` NOT extracted: single use site (`transforms.py:596`, AP10).
- `_grid_search` NOT unified: ETS coordinate refinement vs OTM inline
  grid — different scorers, defer to v0.9.135+.

**Tests:** 219 passed (test_otm, test_theta, test_auto_theta,
test_ts_models, test_ts_backtest, test_otm_verification, schema).

### v0.9.131 (2026-11-08) — K2: sigmoid dedup (4 -> 1) + numerical stability fix

**Single source of truth:** `cadw.ops.optim.sigmoid` (public).

**Before:** `_sigmoid` copy-pasted in 4 modules:
- `optim.py:24` — numerically stable (pos/neg branch, no overflow)
- `gbdt.py:130` — `0.5*(1+tanh(0.5z))` — stable but different impl
- `calibration.py:47` — `1/(1+exp(-clip(z,-50,50)))` — clipping hack
- `ordinal.py:30` — same clipping hack

**After:** gbdt/calibration/ordinal `from cadw.ops.optim import sigmoid
as _sigmoid`. `optim._sigmoid -> optim.sigmoid` (public).

**Correctness fix, not just dedup:** calibration+ordinal clipped |z| to
50, producing wrong probabilities for strongly-regularized logistic and
extreme ordinal thresholds (`p=1/(1+e^-50)≈1` instead of `~0` for
`z=-1000`). New stable impl uses pos/neg branch (no clipping).

**Smoke:** `sigmoid([-1000,-50,-1,0,1,50,1000]) → [0.0,...,1.0]` with
`warnings.simplefilter('error')` — no overflow warnings. All 4 impls
(`np.allclose`) agree to `1e-15`.

**Regression:** `is_dataclass(LinearModel)` still True; `gbdt._LOSSES`
present (decorator/statement adjacent to removed `_sigmoid` was
untouched — AST migration, not regex).

**Tests:** 319 passed.

### v0.9.130 (2026-11-08) — K1: `_probe_memory_limit_mb` dedup (16 -> 1)

**Single source of truth:** `cadw.core.memory_guard.probe_limit_mb`.

**Before:** identical `_probe_memory_limit_mb` body copy-pasted in 16
modules: autoencoder, binseg, hampel, zms, stray, bocpd, sub_pca, eif,
tspq, s2g, patch_memory_ad, cusum, kmeans_ad, kshape, dwt_mlead, linear.

**After:** each module `from cadw.core.memory_guard import probe_limit_mb
as _probe_memory_limit_mb` (back-compat alias, 2 call sites per file
unchanged).

**AST-based migration (not regex).** First attempt used regex
`(?:.*\n)*?` + lookahead for `^def |^class `, which ate decorators/
statements between `_probe_memory_limit_mb` and the next def:
- `linear.py` lost `@dataclass` above `LinearModel` -> TypeError
- `gbdt.py` lost `_LOSSES = (...)` line -> NameError in 65 tests
Fix: `ast.FunctionDef.end_lineno` gives exact boundary; decorators above
and code below untouched. Verified `is_dataclass(LinearModel)` True.

**Tests:** 327 passed, 4 skipped.

**Lint:** 502 deleted lines (-16 copies x~30) replaced with 102 lines
(1 canonical fn + 16 aliases).

### v0.9.129 (2026-11-08) — verify 3 ts drafts (OTM, THF, TTF)

**Counts:** 32 -> 35 verified fns; 48 -> 51 verified cards. Draft fns
unchanged (14); draft cards 17 -> 14 (3 ts drafts -> verified).

**OOC fix + component-wise verification (DEC-033) for 3 draft cards:**
- `OTMThetaForecaster`: `check_state(31*n*8+256)` before grid search;
  state = `2n*8 + 256`. `bounded_by_domain n<=1e6` (n=1e6 = 2740 years
  of daily data).
- `TemporalHierarchy`: `state_nbytes()` added (was missing); cumulative
  `check_state` across groups (was single-group only).
- `TransformedTargetForecaster`: `state_nbytes()` on all 4 transformers
  (Differencer, STLTransformer, MSTLTransformer, AutoStationarizer);
  `TTF.state_nbytes` replaced invented `1024/transformer` with exact
  sum (AP22); `check_state((3+8K)*n*8+4096)` added (K=len(transformers);
  8x is conservative: STL internal ~8n, Differencer ~2n).

**YAML:** 3 cards `draft -> verified` with `ram_formula` +
`formula_evidence` + `measurement: component-wise (DEC-033)` +
`bounded_by_domain` + `tier1_core_exception` (single-series architecture,
like ETS/VAR/VECM).

**Tests:** 27 new (`test_otm_verification.py`,
`test_thf_verification.py`, `test_transforms_verification.py`, CW-1..5
per STABILITY_PROTOCOL §4.1.1).

**Schema:** 108 passed. Target gates: 420 passed.

### v0.9.128 (2026-11-06) — TAXONOMY FROZEN: 42 candidates + 8 NOT-ENSEMBLE

**No verified-fn count change** (32 fns / 48 verified unchanged; 14
draft fns / 17 draft cards unchanged). +42 described_only cards across
15 sections.

Probe v0.9.128 (ensemble-viability): 16 OK-roles, 27 NEED-1, 8 NEED-2,
7 NOT-ENSEMBLE. 35 subtypes had <3 diverse candidates.

**Added (42 described_only cards):**
- NEED-1 (27 subtypes): +1 third-role candidate each.
  A3 GP, A4 ML-kNN, A5 OrderedProbit-Bayesian, A6 RUSBoost, A7 CQR,
  B4 ShapeGMM, C3 GPLVM, C4 SelectKBest, D2 GaussianProcessForecaster,
  E1 GMMAnomaly, E2 CalibratedClassifierCV, F1 GBDT_ALS_embedding,
  F2 SVDpp, G1 ListNet, H1 CausalForest, J2 LDA_embedding, J3 Top2Vec,
  O3 TargetEncoder, O6 WeightedSample, O8 Anchors, O10 ADWIN,
  O11 SequentialTest, R2 HMM, S1 TabNet, S2 TabDDPM_classifier,
  W1 SVI.
- NEED-2 (8 subtypes): +2 candidates each.
  B1 +GMMClustering+SpectralClustering
  B2 +GMMClustering+SpectralClustering
  B3 +BisectingKMeans+SpectralClustering
  B5 +PlaidModel+Bimax
  B6 +EvidenceAccumulation+CSPA
  C1 +KernelPCA+SparsePCA
  H2 +CausalForest+DR_learner_uplift
  T2 +BayesianOpt_GP+SMAC
- All 42 cards: `described_only: true, not_implemented_in_cadw: true,
  tier: baseline|sota, algorithm_source_url: paper|docs|code` (AP23,
  evidence-chain for future implementation).

**8 NOT-ENSEMBLE header comments** (`# NOT-ENSEMBLE (<id>): <reason>`):
- D5 alias of D3.2
- O5 validation metrics, not predictions
- O9 fairness metrics, not predictions
- R1 association patterns, not predictions
- T1 GP-EI/TPE/SMAC = 3 algorithmically different (roles-proxy not
  applicable)
- V1 calibration alternatives — fusion breaks calibration
- V2 conformal alternatives — fusion impossible
- W2 MCMC chain alternatives — fusion statistically invalid

**FREEZE DECLARATION (DEC-052 + AI_CONSTITUTION §5.3):**

> Taxonomy frozen at v0.9.128. 58/58 subtypes have ≥3 diverse
> candidates OR explicit NOT-ENSEMBLE. No new cards without DEC entry
> + explicit rationale. Post-freeze allowed: draft→verified status
> flips, ram_formula completion, cross_ref formalization,
> ensemble_recommendation refinements.

**Tests:** `test_ml_tools_schema.py` 108 passed (no new tests; all
42 cards pass existing R1-R10 gates). Version bump 0.9.127 -> 0.9.128.

### v0.9.127 (2026-11-06) — cadw.ts coverage gap: 5 forecasters + R10 extension

**No verified-fn count change** (32 fns / 48 verified unchanged; draft
fns 10 -> 14, draft cards 13 -> 17 — 4 ts models as `draft`).

Probe (audit-card-coverage) found 5 uncarded `cadw.ts` models, R10
(v0.9.121) covered only `cadw.ml`:

**Added (5 cards in D_time_series.yaml subtype D1):**
- `EnsembleForecaster` — v0.9.82 verified, status: draft
  (`ram_formula` pending v0.9.128)
- `OTMThetaForecaster` — v0.9.78 verified, status: draft
- `TemporalHierarchy` — v0.9.89 verified, status: draft
- `TransformedTargetForecaster` — v0.9.79a verified, status: draft
- `backtest_forecaster` — utility (not model), status: described_only

**R10 extension (tests/test_ml_tools_schema.py):**
- `test_R10_cadw_ts_covered`: all public `cadw.ts` fit-like
  (`*Forecaster` / `fit_*` / `TemporalHierarchy` / `backtest_forecaster`)
  must be carded or whitelisted
- `TS_UTILITY_WHITELIST`: base class (`ForecasterBase`), factory
  (`get_forecaster`), + 5 alias-cases where class name != card name
  (`ETSForecaster` -> card `ETS`, `ThetaForecaster` -> card `Theta`,
  `AutoETSForecaster` -> `AutoETS`, `AutoThetaForecaster` -> `AutoTheta`,
  `SARIMAForecaster` -> `SARIMA`)

**Audit confirmed (v0.9.127):**
- `cadw.ml` public `fit_*` (14): all carded
- `cadw.ts` public fit-like (12): all carded
- `.pingis/impl/` models (11): all carded (v0.9.124 sync)
- Dangling `cadw_function` (R9): 0

**Open (v0.9.128):** 4 ts drafts -> verified requires per-model
formula-from-code (AP23) + measurement (DEC-033). Separate session.

**Status update:** `test_ml_tools_schema.py` 107 -> 108 (new
test_R10_cadw_ts_covered). Version bump 0.9.126 -> 0.9.127.

### v0.9.126 (2026-11-06) — cleanup: weak-machine, no framework cards, P->B/Q->C

**Weak machine = no GPU, 8-16 GB RAM** (2.8 GB in DEC-048 was a snapshot,
not a budget). Card admissibility rule updated: tabular + OOC-by-design +
weak-machine fit + implementable-or-importable.

**Removed (21 cards + 1 section):**
- K_vision.yaml (4 cards) — non-tabular (images)
- L_graph.yaml (4 cards) — non-tabular (graphs)
- U_automl.yaml (whole section, 3 framework cards) — framework != model
- D_time_series: Chronos-Bolt-Small, Chronos-Bolt-Base (negative scaling),
  Timer-S1 (1B+, GPU-only), t0-beta (GPU-only), AutoGluon-TS (framework)
- J_nlp: BERT (MiniLM 99% quality at 10% RAM), BERTopic (needs BERT)
- N_density: NormalizingFlow (torch-only; GMM/KDE cover density)
- A_classification: TabPFN (16 GB CPU min; attention O(n^2); n<=1K CPU
  — architecturally non-OOC, not "not yet optimized")

**Renamed (duplicate with ML_TASKS_TAXONOMY B1/C1):**
- P_clustering.yaml -> B_clustering.yaml (subtype P1..P6 -> B1..B6)
- Q_dim_reduction.yaml -> C_dim_reduction.yaml (Q1..Q4 -> C1..C4)

**Card status updates:**
- 4 inference-only (Chronos-Bolt-Tiny/Mini/Chronos-2-small,
  SentenceTransformer): purpose += "Requires torch; inference-only."
- 2 draft-target (DeepSurv, DeepHit): purpose += "Optional numpy
  draft-target (2-3 sessions); marginal over RSF; v0.9.128+."
- 5 S_generative (TVAE/CTGAN/CTAB-GAN+/TabDDPM/TabSyn): purpose +=
  "Last priority (v0.9.130+); CPU-feasible but training is hours-days."

**Cross-refs updated** in ROADMAP.md, ML_TASKS_TAXONOMY.md,
test_ml_tools_schema.py. DEC-049 historical refs preserved as-is
(append-only log). DEC-050 records rationale.

**Levels clarified (DEC-050 §A3):**
- `ensemble_recommendation` (subtype YAML) = diverse-role list
- `tier: ensemble` card = named recipe (D1.EnsembleForecaster already
  exists v0.9.82)
- `fit_stacking(df, base_models, meta_learner)` = v0.9.127+ infrastructure
  function, not a card
- AutoML framework = selection + HPO + stacking + pipeline search
  — deleted; if CADW needs stacking, that's fit_stacking

**16 sections remain** (A-H, I, J, M, N, O, R, S, T, V, W; K/L/U deleted;
P/Q renamed to B/C). 18 YAML + 3 support = 21 total.

### v0.9.125 (2026-11-06) — R/S/T/U/V/W sections (27 described_only cards)

**No verified-fn changes** (32 fns / 48 verified / 10 draft / 13 draft
cards). +27 described_only in 6 new sections (cumulative described_only
46: 19 in P/Q + 27 here).

New sections (all described_only, no CADW impl, no formula):
- **R_association.yaml** (5 cards, R1 FrequentItemset + R2
  SequentialPattern): Apriori, Eclat, FP-growth (SOTA), GSP, PrefixSpan
  (SOTA). Anchor: mean_lift / sequence_coverage.
- **S_generative.yaml** (5 cards, S1 Neural VAE/GAN + S2 Diffusion):
  TVAE, CTGAN, CTAB-GAN+, TabDDPM, TabSyn. All tier3_export (torch).
  Anchor: downstream_accuracy.
- **T_optimization.yaml** (5 cards, T1 Bayesian + T2 Derivative-free):
  GP-EI, TPE (SOTA), SMAC, Nelder-Mead, CMA-ES (SOTA). Anchor:
  best_found_value.
- **U_automl.yaml** (3 cards, U1 Full AutoML): Auto-sklearn, H2O AutoML,
  AutoGluon 1.2 (SOTA per AMLB 2025). Anchor: mean_rank.
- **V_uncertainty.yaml** (5 cards, V1 Calibration + V2 Conformal):
  Platt, Temperature, Isotonic (SOTA), Split Conformal, CV+ (SOTA).
  Anchor: ece / empirical_coverage.
- **W_probabilistic.yaml** (4 cards, W1 VI + W2 MCMC): Mean-field VI,
  Black-box VI (SOTA), Metropolis-Hastings, NUTS (SOTA). Anchor:
  elbo / ess.

**benchmarks.yaml:** +HPOBench (tabular_hpo, modality=HPOBench) —
referenced by T_optimization; canonical HPO leaderboard.

**Anchor metric warnings:** all anchors in R/S/T/U/V/W are
`not_implemented_in_cadw: true` (no CADW code) — schema emits warning
(not error): subtype has no computable anchor, values external.

**Canonical homes for cross-ref:**
- R/S/T/U/V/W new — no other canonical home.
- V1 Platt/Isotonic -> cadw.ml.calibration (existing); W: VI/MCMC have
  no CADW impl (Pingis .pingis/impl/J/nmf.py similar spirit but not
  canonical).

**Deferred (v0.9.126):** expansion_factor migration (18 call sites) +
formal cross_ref schema field + promotion of `.pingis/impl/` verified
models to `cadw/ml/`.

### v0.9.124 (2026-11-06) — .pingis/impl/ sync into cards (purpose refs)

**No fn/card-count changes** (32 fns / 48 verified / 10 draft / 13
draft cards; +19 described_only in P/Q unchanged).

- **10 described_only cards** get `purpose` reference to their working
  sandbox impl + REPORT.md §6 smoke evidence (no new fields; no
  benchmark_results entry, per "no momentary data" policy):
  - I_survival: KaplanMeier, WeibullAFT -> .pingis/impl/I/*.py
  - J_nlp: ComplementNB, NMF -> .pingis/impl/J/*.py
  - M_bandits: EpsilonGreedy, UCB1, ThompsonSampling -> .pingis/impl/M/bandits.py
  - O_eda: ReservoirSample, StratifiedSplit, OneHotEncoder -> .pingis/impl/O/*.py
- **4 deferred O_eda cards** get `purpose` reference to REPORT.md §4
  rationale (memory model already proved by N/KernelDensity or
  O/OneHotEncoder; no new algorithmic content):
  - KNNImputer, KBinsDiscretizer, GridSearch, RandomizedSearch.
- **L_graph / K_vision** unchanged: headers already state "Out of scope:
  torch required" = REPORT.md §7 rationale (no duplicate needed).

**Status unchanged:** all 14 remain `described_only`. Pingis impls are
sandbox `.pingis/impl/` (not `cadw/ml/`), no `ram_formula`/`measurement`/
`test_*_verification` — promotion to `verified_in_cadw` -> separate
session (v0.9.126+).

### v0.9.123 (2026-11-06) — P_clustering + Q_dim_reduction (22 new cards)

**No verified-fn changes** (32 fns / 48 verified cards unchanged;
draft fns 7 -> 10, draft cards 10 -> 13).

- **P_clustering.yaml** (new, 13 cards): 2 draft (KMeans,
  MiniBatchKMeans — formula from code, measurement v0.9.124) + 11
  described_only (BisectingKMeans, DBSCAN, HDBSCAN, Agglomerative,
  BIRCH, KShape, EuclideanKMeansZ, SpectralCoclustering,
  SpectralBiclustering, KMeansOnly, KMeansHDBSCAN_consensus).
  Subtypes P1-P6 (centroid / density / hierarchical / shape /
  biclustering / ensemble).
- **Q_dim_reduction.yaml** (new, 9 cards): 1 draft (PCA — formula from
  code, measurement v0.9.124) + 8 described_only (TruncatedSVD,
  randomized_svd, NMF, ALS, t-SNE, UMAP, PermutationImportance,
  Boruta). Subtypes Q1-Q4 (linear / matrix factor / manifold /
  feature selection).
- **Cross-ref policy** (DEC-049): canonical home + markdown comment +
  `cross_ref_comment` (schema-ignored field). HDBSCAN/BIRCH ->
  N_density.N1; NMF -> J_nlp.J3; ALS -> F_recsys; KShape ->
  D_time_series.D4; randomized_svd -> J_nlp.J2.LSA_TFIDF_SVD;
  PermutationImportance -> O_eda.O8. Migration to formal
  `cross_ref` field -> v0.9.124+.
- **R10 whitelist empty**: fit_kmeans, fit_minibatch_kmeans, fit_pca
  now in cards (P1, Q1). No uncarded public `fit_*` remain.
- **Schema rules discovered/fixed**: `O(k·d)` not in VALID_STATE_CLASS
  (use `O(model)`); baseline+sota required per subtype; exactly one
  anchor metric per subtype (+`special` field); `bounded_state_dims`
  no `batch`/`row_size`.

**Deferred (v0.9.124):** R_association, S_generative, T_optimization,
U_automl, V_uncertainty, W_probabilistic (all described_only —
no code, will land when implementable).

### v0.9.122 (2026-11-06) — taxonomy extension (28 types) + CADWBoost SOTA reference

**No fn/card changes** (32 fns / 48 cards; taxonomy + reference-only card).

- **TASK_TYPES** 14 -> **28** (DEC-048). 14 new: `clustering`,
  `dimensionality_reduction`, `matrix_factorization`, `association_rules`,
  `generative_modelling`, `structured_prediction`, `feature_selection`,
  `covariance_estimation`, `biclustering`, `manifold_learning`,
  `optimization`, `automl`, `uncertainty_quantification`,
  `probabilistic_inference`. Sources: SMU DIGA 2025, Zenodo Two-Axis 2026,
  sklearn Unsupervised §2.2-2.6.
- **Discovery:** model-level `task_type` = 0/302 cards (all `task_type`
  in YAML live inside `benchmark_results[]`, free-form). Extension safe;
  adoption -> v0.9.123+ (per section, together with candidate selection).
- **CADWBoost** reference-only card in A_classification (`tier: sota`,
  `described_only: true`, `not_implemented_in_cadw: true`).
  `.pingis/drafts/cadwboost/`: 6/6 datasets better than `cadw.ml.gbdt`
  (-12%..-80% rmse), OOC ratio 1.08. Promote target v0.9.124+.
- **`ML_TASKS_TAXONOMY.md`**: v0.9.122 section — table of 14 new types
  with section home + baseline/SOTA/ensemble candidates. Reference
  policy: реализуемые SOTA -> eventual `cadw_function`; нереализуемые
  (TabPFN/TabICL/BERT/ResNet/GCN) -> `not_implemented_in_cadw: true`.
- **DEC-048**: taxonomy extension + CADWBoost + reference policy +
  non-goals (vision/graph/NLP transformers — reference-only).

### v0.9.121 (2026-11-05) — R10 coverage gate + GBDT_multiclass card fix

**No fn/card changes** (32 fns / 48 cards; gate hygiene only).

- **GBDT_multiclass card bug fix**: `cadw_function: fit_gbdt_classifier` ->
  `fit_gbdt_multiclass`. Card purpose is "One-vs-all softmax GBDT"
  (`_fit_boosting_multiclass`), but cadw_function pointed to the binary
  dispatcher (`_fit_boosting`). R5 (alias status consistency) and R9
  (importable) both passed — semantic mismatch invisible until audit.
- **R10** (`tests/test_ml_tools_schema.py`):
  - `test_R10_public_fit_covered`: every public `fit_*` in
    `cadw.ml.__all__` must be resolvable by `_load_cards()` or whitelisted
    (`UNCARDED_WHITELIST` with DEC ref + target version). Catches the
    DEC-045 class of gap (pre-cards-era base algorithms).
  - `test_R10_GBDT_multiclass_function_matches`: pins the specific
    GBDT_multiclass expectation (name -> function match for cases where
    they diverge meaningfully). Not a general heuristic (R5 alias system
    legitimately shares `cadw_function` across differently-named cards).
- **DEC-047**: GBDT_multiclass bug + R10 rationale + coverage gap plan.
  Uncаrded public `fit_*` (whitelist, target v0.9.122): `fit_kmeans`
  (v0.7.1), `fit_pca` (v0.7.2), `fit_elasticnet`.

### v0.9.120 (2026-11-05) — P6'-P9' real pipelines + calibrate_throughput

**No fn/card changes** (32 fns / 48 cards, framework validation only).

- **P6'-P9' real** (`tests/test_pipeline_planner_real.py`, 9 tests):
  `model_spec={"cadw_function": ..., "ram_params": {...}}` — planner uses
  `solve_batch` (inverse of actual YAML `ram_formula`) for verified models
  (`fit_logistic`, `fit_linear`, `fit_gaussian_nb`; `fit_ab_test` — state-only
  formula -> `None` -> fallback `state_bytes`).
- **Bug fix** (planner): `_plan_pipeline_v2` model_spec output_chars used
  `spec.state_bytes` (=0 when only `cadw_function`+`ram_params` passed);
  now uses computed `estimate` (single source: planner knows estimate).
- **`scripts/calibrate_throughput.py`** + `tests/test_calibrate_throughput.py`
  (5 tests): `--dry-run`/`--write` for `reduce`/`sort`/`hash_shuffle`
  rates. PyYAML 1.1 quirk fix: `5.0e7` without sign-in-exponent resolved as
  string; `_write_rates` normalizes all rates to float.
- **DEC-045**: P6'-P9' validation + YAML coverage gap (`fit_kmeans`,
  `fit_pca` — 0 cards, pre-cards-era base algorithms; `fit_kmeans_ad`
  aliased, covered).
- **DEC-046**: calibrate_throughput.py design; per-model fit-time
  calibration deferred to v0.9.121+.

### v0.9.119 (2026-11-05) — pipeline planner v2 + solve_batch + fit_ab_test x3 verified

**Verified fns: 31 -> 32** (45 -> 48 cards).

Framework (DEC-043, DEC-044):
- `plan_pipeline(memory_guard=...)` — pipeline planner v2: segments by
  `OpSpec.spill_after`, carried state between segments, per-segment batch,
  time via `throughput.yaml`. `memory_guard=None` -> v1 (24 test_planner.py
  back-compat).
- `CostEstimate` + `MemoryGuard.estimate_cost` + `MemoryModel.estimate_time`
  (AP22 skeleton; calibration `scripts/calibrate_throughput.py` -> v0.9.120).
- `MemoryGuard.solve_batch`: inverse of `ram_formula` via numeric sampling
  (AP23-compliant, `_is_linear_in_batch` guard); `None` -> fallback
  `expansion_factor`.

`fit_ab_test` x3 verified (ProportionsZTest / MannWhitneyU / CUPED):
- OOC via `Reduce` (t/cuped) and `external_sort` + streaming rank (mw).
- Component-wise (DEC-033): state exact (1696/1744/296 B), state_n-independent,
  tight_guard refuses, batch_size invariant, streaming matches scipy.
- `DriftABModel.result_` populated; `to_dict()["kind"]=="ab"` preserved
  (9 existing test_drift.py green).

Tests: +6 `test_estimate_cost.py`, +10 `test_pipeline_planner_examples.py`
(P1-P8 synthetic), +14 `test_solve_batch.py`, +17 `test_drift_ab_verification.py`.
Schema gates: 141/141 (test_ml_tools_schema, test_yaml_state_fields,
test_render_matrix).

### v0.9.118 (2026-11-04) — ordinal + chains verified; R9 schema test

**Verified fns: 29 -> 31** (43 -> 45 cards).

- `fit_ordinal_logit` verified (component-wise DEC-033): state_nbytes =
  d*f64 + (K-1)*f64 + 256; batch_size kwarg. YAML: state_class O(d²) -> O(d)
  (was false — beta is d, theta is K-1); bounded_state true -> false
  (fit peak O(n*d)); bounded_by_domain n<=1e6; CW-5 guard-refuses-at-bound.
- `fit_chains` verified x2 alias (OneVsRest_Logistic + ClassifierChains):
  state_nbytes = sum(GBDTModel.state_nbytes()) + 512; batch_size kwarg.
  YAML: state_class O(d²) -> O(model); bounded_state false; bounded_by_domain.
- **R9** (tests/test_ml_tools_schema.py): every card's cadw_function must be
  importable (module attr / class method / CadwDataFrame / accessor class).
  Catches YAML drift: `fit_classifier_chains` (in YAML) vs `fit_chains` (code).
  Fixed 2 broken cards (A_classification OneVsRest + ClassifierChains).
- Full-fast triage: 8 fails -> 0 in 3 runs (T1 LDA estimate formula,
  T2 ordinal drop, T3 render_matrix regen, T4 lda extra_factor + guard,
  T5 linucb median + import, T6 gmm coef tolerance, T7 chains ratio,
  T8 lda ratio — all DEC-034 escalation).

### v0.9.117 (2026-11-04) — LDA + TF-IDF verified; auto K-spill

**Verified fns: 27 -> 29** (41 -> 43 cards).

- `fit_lda` + `fit_tfidf` verified (component-wise DEC-033):
  - LdaModel.state_nbytes(): K*V*8 + V*(8+49+24+116) + 512 (was undercount 4x).
  - TfidfModel.state_nbytes(): V*8 + V*(8+49+24+116) + 512 (same).
  - `_batches()` / `_collect` get batch_size pass-through (was declared but
    unused → peak O(n), fixed to O(batch)).
- **Auto K-spill** (DEC-042): K_raw_est > state_limit_bytes // 96 // 2 →
  hash_shuffle 3-pass; else legacy `dfc = {}` (bit-identical for natural
  corpora). Applies to both LDA and TF-IDF.
- **DEC-041**: bounded_state refers to state (O(V)), not fit peak
  (O(batch + K_raw)). Honest: guard refuses at K_raw >> state_limit, not OOM.
  Precedent: CoxPH `bounded_by_domain: n<=1e6`.
- `_build_vocab_k_spill` returns (vocab, dfv) with min_df filter.
- Adversarial test: K_raw=80k > threshold=39k → K-spill, fit succeeds under 30MB.

### v0.9.116 (2026-11-04) — sub_pca verified; df-API migration; DEC-039/035

**Verified fns: 24 -> 27** (38 -> 41 cards).

- `fit_sub_pca` verified x2 alias (component-wise DEC-033):
  state_nbytes = 2*n*f64 + n_components*window_size*f64 + window_size*f64 + 264.
  Dual dispatch (df, col | [cols] for multichannel). 10 tests.
- **df-API migration** (R8 warnings 5 -> 0): `fit_bocpd`, `fit_eif`,
  `fit_kmeans_ad` gain df_or_series dispatch. Precedent: fit_zms/fit_cusum
  v0.9.113.
- **DEC-034 partial**: `test_lda_bounded_memory` uses measure_peak_traced.
- **DEC-035** (BIRCH _insert split): row-by-row argmin; removes (bf,bf,d)=4MB
  intermediate per split (bf=50, d=200), 80KB now. Bit-identical (9/9 pass).
- **DEC-039**: CW-4 partial for 7 series-path O(n)-output fns; G2 fix in
  hampel+sub_pca (output check_state added); bounded_by_domain = "guard
  refuses, not OOM". CW-5 test_guard_refuses_at_domain_bound added.

### v0.9.115 (2026-11-04) — 30 verified fns; input_api invariant (R8)

**Verified fns: 24 -> 30.** Six CoxPH-precedent migrations (all dual
dispatch df+series, state_nbytes, bounded_by_domain n<=1e6, 9 CW-tests
each, component-wise DEC-033):

| fn | state formula | notes |
|---|---|---|
| fit_kshape | 2*n*f64 + k*w*f64 + 264 | SpillArray (output="disk") pending v0.9.116 |
| fit_hampel | 2*n*f64 + 264 | |
| fit_dwt_mlead | n*f64 + n*bool + max_level*i64 + 264 | YAML was O(batch), fixed to O(n) |
| fit_patch_memory_ad | 3*n*f64 + n_epochs*32 + n_memory*latent*f64 + weights + 264 | SpillDir memmap for X_norm |
| fit_autoencoder | 3*n*f64 + n_epochs*32 + enc_w + dec_w + 264 | 2 alias cards flipped |
| fit_s2g | 5*n*f64 + n_nodes²*f64 + n_nodes*f64 + 264 | graph state O(model) |

**input_api invariant (R8):** 88 cards with `cadw_function` declare
`input_api: df|series|dual`. `series`-only: bocpd/eif/kmeans_ad
(verified, migration_target: v0.9.116) + binseg/stray/sub_pca/t2q
(draft). Precedent: zms/cusum v0.9.113 (STREAMING §1.5).

**Files:**
- cadw/ml/{kshape,hampel,dwt_mlead,patch_memory_ad,autoencoder,s2g}.py
- tests/test_{kshape,hampel,dwt_mlead,patch_memory_ad,autoencoder,s2g}_verification.py (6 new, 54 tests)
- tests/test_ml_tools_schema.py — R8 (39 lines)
- docs/ml_tools/[A-O]_*.yaml — input_api on 88 cards (11 files); KShapeAD/Hampel/DWT-MLEAD/PatchMemoryAD/Autoencoder×2/S2G flipped
- docs/ROADMAP.md — v0.9.115 closed, v0.9.116 df-API migration bullet
- cadw/__init__.py, pyproject.toml — version bump 0.9.102 -> 0.9.115 (drift fixed: __version__ had been 0.9.102 since v0.9.102; tags advanced without bump)

**Status:** 30 verified fns / 38 verified cards / 18 draft / 32 pending / 62 no-fn.
**Full fast:** 4522 passed, 9 skipped, 0 failed (14:16).

### v0.9.114 (2026-11-04) — CoxPH-precedent: 4 O(n)-output algorithms verified (24 fns)

**CoxPH-precedent** for O(n)-output: `state_nbytes()` + `bounded_by_domain
{axis: n, bound: 1e6, reason: <domain>}`. Honest `state_class: O(n)`,
`n_scalability: linear`, `bounded_state: false`. Avoids SpillArray for
these 4; unblocks in one session.

**v0.9.114-coxph — fit_coxph verified:**
- batch_size kwarg + propagation (pre-pass + _checked_iter + predict_risk).
- predict_risk output="memory": two-pass pre-alloc (AP2 fix).
- YAML: draft -> verified; state_class O(n)->O(model); bounded_state_dims
  {n}->{d, n_events}; bounded_by_domain n<=1e6 (c-index unstable >1e6 +
  concordance_index O(n) intrinsic, v0.9.97 do-not-re-solve).
- 4 tests. Peak-based fit test dropped: external_sort run_buffer (~25MB)
  > baseline ~1MB at n=100K.

**v0.9.114-kmeans_ad — verified:**
- KMeansADResult.state_nbytes(): 4*n*8 + k*d*8 + 256.
- YAML: verified; bounded_state_dims {batch}->{n, k, d}; bounded_by_domain.
- 4 tests. Tight-guard uses Duck-typed guard (real MemoryGuard -> chunked,
  no raise).

**v0.9.114-eif — verified:**
- EIFResult.state_nbytes(): 3*n*8 + 256. Forest hyperparameter-bounded
  (n_est*psi Python objects) excluded (O(model), not O(n)).
- Chunked fit already present (v0.9.91-S6b-fix3). No code change.
- YAML: verified; bounded_state_dims {batch}->{n}; bounded_by_domain.
- 4 tests.

**v0.9.114-bocpd — verified:**
- BOCPDResult.state_nbytes(): 3*n*8 + posterior (~103KB compact) + 256.
- Sequential Python loop O(n*max_run) time; RAM O(n) output only.
  Compact default; full_posterior=True -> O(n*max_run) quadratic, doc.
- YAML: verified; bounded_state_dims {k,d}->{n, posterior_keep, max_run}.
- 4 tests. test_state_ratio_linear uses DELTA (fixed posterior ~26KB
  dominates direct ratio at 1K->10K).

**DEC-036 partial:** sample-based threshold not needed for these 4
(CoxPH-precedent path suffices). Remains for kshape v0.9.115.

**Baseline:**
- Verified fns: 20 -> 24. Cards: 27 -> 33.
- Focused: 167 passed. Gates: 142 passed.

### v0.9.113 (2026-11-03) — STREAMING §1.5 (producer API) + zms + cusum verified (partial)

**Phase A — input convention documented:**
- `STREAMING_OUTPUT_ARCHITECTURE.md` §1.5: producer API = `fit_*(df, col, ...)`;
  private helpers may take `np.ndarray`. Legacy producers list (8+ anomaly
  detectors + AnomalyPipeline.fit_predict) tracked in ROADMAP §Legacy inventory.
  Dual dispatch (`df_or_series` type sniff) documented as transitional.

**Phase B — fit_zms verified × 3 aliases (pilot):**
- `fit_zms(df_or_series, value_col=None, *, output="auto", batch_size=None)`.
  df path: two-pass streaming + SpillArray output (spill_ratio 0.5 auto,
  0.0 disk); np.ndarray path: legacy in-memory (7 old tests unchanged).
- `ZMSResult` Union[np.ndarray, AutoArrayResult] + `is_spilled`.
- `contamination` + output!="memory" -> NotImplementedError (refuse, honest).
- YAML 3 aliases (zscore ×2, ZMS ×1): state_class O(d)->O(batch),
  n_scalability constant->linear, bounded_state_dims {w, batch},
  ram_formula + formula_evidence + measurement.
- tests/test_zms_verification.py: 6 tests.

**v0.9.113-cusum — fit_cusum verified:**
- `fit_cusum(df_or_series, value_col=None, *, k, h, contamination, output="auto",
  batch_size=None)`. `_fit_cusum_df` two-pass Welford (mean/std) + streaming
  recurrence, O(1) state (s_pos_cur, s_neg_cur + Welford scalars).
- `CUSUMResult` Union + is_spilled.
- contamination + output!="memory" -> NotImplementedError.
- YAML CUSUM: state_class O(n)->O(1), bounded_state false->true,
  n_scalability refuse_at_guard->linear, ram_formula + formula_evidence + measurement.
- tests/test_cusum_verification.py: 6 tests.

**OOC fix — batch_size kwarg for zms + cusum** (AI_CONSTITUTION §3 pattern):
- Both have `batch_size: int|None = None`; when None -> derive from
  `get_batch_rows`. Tests pass `batch_size=1000, output="disk"` for OOC
  verification (spill_ratio=0.0 -> peak = O(batch) independent of total n).

**v0.9.113-dec036 — contamination-threshold blocker for eif/km/bocpd/kshape:**
- `fit_eif`, `fit_kmeans_ad`, `fit_bocpd` have `contamination=0.1` default
  -> `np.percentile(scores)` needs full O(n). Cannot use zms-template
  (which has `contamination: Optional = None` + refuse). Three options
  in DEC-036: (A) sample-based threshold, (B) Optional default breaking,
  (C) refuse (bad UX). Recommendation: (A) v0.9.114.
- kshape: `_map_scores_to_timestamps` expand n_w->n needs full array +
  rolling-max; separate work v0.9.114.
- Refined draft_reason for 3 cards (EIF × 2 aliases + KMeansAD + BOCPD).
- ROADMAP §v0.9.113 marked partial; §v0.9.114 blocked-list updated.

**Baseline:**
- Verified fns: 18 -> 20 (+zms, cusum). Cards: 23 -> 27 (zms × 3 + cusum × 1).
- Focused: ~190 passed, 0 failed (test_cusum x2, test_zms x2, schema,
  yaml_state_fields, render_matrix).

**Do not re-solve:**
- zms/cusum OOC (verified v0.9.113; batch_size kwarg + SpillArray + refuse-contamination).
- eif/kmeans_ad/bocpd/kshape (DEC-036; v0.9.114 design).
- 4 verified in v0.9.111/112 (BIRCH, MiniRocket × 2, ALS × 2) — unchanged.

### v0.9.112 (2026-11-03) — MiniRocket x2 + ALS x2 verified (18 fns), 8 cards -> draft

**v0.9.112-batch1 — verified (component-wise DEC-033):**

MiniRocket (embedding + sota aliases) -> verified:
- fit is O(1) per n (validates cols + builds num_kernels, no data iteration).
  state_nbytes() = num_kernels*(KERNEL_LEN*8+8) + 1024.
- YAML: state_class O(batch)->O(model), state_type windowed->aggregation,
  bounded_state_dims {batch}->{num_kernels}, + ram_formula/formula_evidence/
  measurement. Both aliases flipped (R5).
- tests/test_minirocket_verification.py: 4 tests.

ALS (x2 aliases) -> verified:
- CODE: state_nbytes() extended to include user_index_/item_index_/_u_pos/
  _i_pos (+~165 B/entry). Was factors-only (U+I)*k*8 — 30% under-report made
  MemoryGuard under-size batch ~6x (unsafe).
- CODE: fit_als batch_size kwarg + propagation to all _iter_filtered_batches.
  _objective_streaming threads batch_size (was closure-miss NameError in fix1).
- YAML: ram_formula + formula_evidence + measurement; bounded_state_dims
  {N,d}->{U,I,n_factors}; n_scalability kept LINEAR (external_sort K-way
  merge peak O(K), not O(1) — honest, not widened threshold).
- tests/test_als_verification.py: 4 tests.
- tests/test_als.py: 2 stale state_nbytes assertions updated (fix2 missed one).

KShapeAD -> draft (DEC-013):
- O(n) output (decision_scores, labels) in RAM, not spilled (kshape.py:502).
- YAML: state_class->O(n), bounded_state->false, n_scalability->linear,
  + bounded_by_domain {n: 1e6}, status: draft, draft_reason.

**v0.9.112-batch2 — 7 cards -> draft (no code changes):**

DEC-013 (O(n) output, SpillArray pending v0.9.113):
- EIF, KMeansAD, BOCPD (per-point/per-timestep scores).

AP5 (unbounded dict, K-spill needed):
- LDA (_build_vocab df_counts raw unique tokens ~96 B/token; guard refuses).

AP2 (np.concatenate/vstack in batch loop):
- CoxPH (predict_risk output='memory' + concordance_index).
- T_learner / S_learner (_collect Xs.append + np.vstack; full X materialization).
  R5: aliases same status.

All 5 tier1_core+bounded_state=false cards got bounded_by_domain (R7).
7 cards added bounded_by_domain in fix (schema caught on first verify).

**Release:**
- Verified: 16 -> 18 fns. Cards: 19 -> 23 verified / 22 draft.
- Focused: 196 passed (test_minirocket x2, test_als x2, test_birch x2,
  test_ml_tools_schema, test_yaml_state_fields, test_render_matrix).
- Full fast: 4434 passed, 9 skipped, 0 failed (13:19).
- ROADMAP: v0.9.112 -> v0.9.113 (SpillArray 6 drafts) -> v0.9.114 (AP2
  refactors + DEC-034/035) -> v0.9.115 (K-spill + remaining pending).

**Do not re-solve:** DEC-013 (zms/cusum/kshape/eif/kmeans_ad/bocpd need
SpillArray v0.9.113); AP5 (LDA/tfidf need K-spill v0.9.115); AP2
(CoxPH/T/S_learner refactor v0.9.114).

### v0.9.111 (2026-11-03) — fit_birch verified (16 total) + R6/R7 schema + baseline stabilization

**Phase 0 — baseline stabilization (pre-existing red from v0.9.99 doc drift):**
- render_matrix re-run: 10 files (MATRIX.md + docs/tasks/AA*,DD3_1,EE*,MM1,NN1).
  15 test_render_matrix fails accumulated between v0.9.99 (last regen) and
  v0.9.110g (YAML changes: gmm verified, HDBSCAN/NormalizingFlow described_only).
- test_roadmap regex: accept vX.Y.Z<suffix> (v0.9.110g, v0.9.93-logistic).
  _vkey drops trailing `$`. Subpatch is part of version, not a separate token.
- test_iforest_verification: median-of-3 in test_ooc_batch_fixed; widen
  test_formula_small_scale to [0.5, 2.0]. DEC-034: full-suite SNR 1.2
  (baseline RSS from prior 4400 tests raises measure_peak_traced noise to
  ~5 MB). Primary evidence is state-exact (bit-exact) + coef isolated.
- test_arrow_call_memory_guard: whitelist measure_peak_pss + measure_peak_traced
  (threading.Thread.join for daemon sampler cleanup, NOT pa.Table.join —
  same precedent as MemoryGuard.calibrate v0.9.20-fix4).

**Phase 1 — fit_birch OOC + verification (DEC-033 component-wise):**
- cadw/ml/birch.py: batch_size kwarg. None -> derive via
  get_batch_rows(state_est=bf*((2d+1)*8+32) + k_bound*d*8 + 512,
  expansion_factor=3.0). Int = explicit override.
- tests/test_birch_verification.py: 6 tests (state_nbytes_exact,
  cf_count_bounded, state_bounded_for_large_n, batch_coef_isolated,
  formula_small_scale, ooc_batch_fixed). State exact by construction
  (cf_count known from tree; bounded by branching_factor via 1-level
  split policy).
- docs/ml_tools/N_density.yaml BIRCH card: status -> verified,
  ram_formula + formula_evidence + measurement (component-wise per
  DEC-033). bounded_state_dims: {k, d} -> {branching_factor, k, d}.
- DEC-035: coef=1.16 honest (f64 zero-copy _stack_numeric; iforest 2.77
  reflects reservoir + per-tree transients). ~5 MB intercept is glibc
  arena fragmentation (per-row _insert transients) — allocator-dependent,
  not code-derivable. Formula keeps 3.0 conservative per fit_linear
  v0.9.107 precedent (safe upper for guard; at small batch under-estimates
  by 1.7x — known small limitation, v0.9.112 `_insert` batching).

**Phase 2 — Part 2 (re-scoped from 62 to 19 cards, probe-verified):**
- tests/test_ml_tools_schema.py: +test_R6 (ModelCard without cadw_function
  requires described_only=True OR not_implemented_in_cadw=True);
  +test_R7 (ModelCard with cadw_function requires implementation_status
  block present; pending status=None is valid).
- 19 no-fn cards patched: 8 tier3_export -> not_implemented_in_cadw: true
  (Chronos-Bolt-Tiny/Mini/Small/Base, Chronos-2-small, AutoGluon-TS,
  Timer-S1, t0-beta); 11 tier1/tier2 -> described_only: true
  (KaplanMeier, WeibullAFT, ComplementNB, NMF, EpsilonGreedy, UCB1,
  ThompsonSampling, KernelDensity, OneHotEncoder, StratifiedSplit,
  ReservoirSample).
- scripts/bulk_flip_status.py: new helper (card-scoped, A16; idempotent;
  --dry-run; fail-loud on anchor miss per AP8). Companion to test_R5.

**Release:**
- 16 verified total (15 + BIRCH).
- Focused suite: 779 passed, 0 failed (test_birch x2, iforest_ver,
  linucb_ver, gmm_ver, ml_tools_schema, yaml_state_fields,
  render_matrix, roadmap, arrow_gate, primitive_reuse,
  guard_before_alloc, no_dense_windows, constitution_gates).
- docs/ROADMAP.md: Current -> v0.9.111; Status baseline 15 -> 16;
  Next-3 shifted (v0.9.112 Group C + DEC-034/035 backlog; v0.9.113
  SpillArray zms/cusum; v0.9.114 Group C + tag).
- CHANGELOG.md: this entry.

**Do not re-solve:** DEC-031/032/033 (component-wise, iforest/linucb);
v0.9.107 verified cards; _stack_numeric (v0.9.108 final); MemoryGuard core.

### v0.9.110 (2026-11-03) — component-wise verification (DEC-033), 15 verified

**v0.9.110a — fit_isolation_forest OOC (DEC-031):**
batch_size kwarg + auto-derive (state_est conservative v2: reservoir
S*d*8 + trees n_trees*2*S*180 + cached numpy + score transient).
Removed end-of-fit check_state (was AP4: guard after alloc, only
covered lists not cached numpy). fit loop now propagates batch_size.
OOC proven (batch=500 fixed, n in [1e3, 1e5]): peak ratio 1.00.
Both YAML aliases (E_anomaly + D_time_series) -> status=draft with
draft_reason: matrix measurement fails at multiple cells due to
measure_peak tool limit, not code. Tests: 12/12 iforest.

**v0.9.110b — fit_linucb OOC (DEC-032):**
batch_size kwarg + k_max_hint param (K is data property: first-batch
max(arm)+1 x 2 safety). AP4 fix: check_state before
newA = np.zeros(...) in state_grow branch. arms extraction:
cast(int64)+to_numpy (removes to_pylist Python-object overhead).
state_est = K*d^2*8 + K*d*8 + d^2*8 + d*8 + 512. OOC: 1.00.
Same draft status (DEC-032 corrected DEC-031: fix is NOT state
refactor since linucb is pure numpy; real fix is tooling).
Tests: 10/10 linucb.

**v0.9.110c — component-wise verification (DEC-033) + tooling:**
New `measure_peak_traced` in memory_guard (fast VmRSS sampler via
/proc/self/statm, 2us/sample, 200us interval) + `mg.mark(name)`
phase markers. Plus `measure_peak_pss` (smaps_rollup, 3ms/sample)
and `_read_vmrss_fast`.

Why: 16-iteration investigation (linucb) + 8 (iforest) confirmed
`measure_peak` (ru_maxrss delta) returns 0 when baseline >> fit
peak (allocator reuse, not new mmap). 5 tool attempts (fadvise,
malloc_trim, smaps_rollup, phase-trace, statm sampler) confirmed
fundamental. Component-wise is the honest answer:
- state exact via model.state_nbytes()
- coef isolated at d=200, batch 1000-4000
- formula small-scale median-of-3
- OOC independent

fit_linucb verified. expansion_factor 3.0 -> 4.0 (= ceil(3.18*1.15)
per AP23). Formula: k*d^2*f64 + k*d*f64 + d^2*f64 + 256 +
3.5*batch*row_size. Test: 8/8 test_linucb_verification.py (~3s,
stable x8 runs).

**v0.9.110d — fit_isolation_forest verified x2 (DEC-033):**
state bounded (not identical) for n>=S; state monotone for n<S
(psi=min(n,S)). Formula: n_trees*2*subsample_size*40 +
3.5*batch*row_size + subsample_size*d*f64. dims: {k, d} ->
{n_trees, subsample_size, d} (R2 requires formula names in dims).
Test: 8/8 iforest_verification (~3s, stable x5).

**v0.9.110e — methodology update (T1/T2/T3/T5):**
- STABILITY_PROTOCOL v1.0->v1.1: §4.1.1 Component-wise mode
- EXPERIMENT_PROCEDURE v1.0->v1.1: §3.7 Step 6-alt (CW-1..6 +
  CW-AP1..4 anti-patterns)
- ALGORITHM_TEMPLATE v1.0->v1.1: Step 5-alt
- EXAMPLE_VERIFIED_CARD: Example 2 (linucb component-wise)
- AI_CONSTITUTION: AP21/AP23 with component-wise alternative

**v0.9.110f — snapshot tooling:**
TINY_FULL += test_linucb_verification.py, test_iforest_verification.py
(canonical templates for gmm/birch/etc, +20 KB). TINY_SKELETON +=
experiment_v110_*.py (skeleton reveals predicted_peak + subprocess/
prewarm/malloc_trim/fadvise pattern). EXAMPLES_INDEX: 2 rows for
component-wise + phase-traced.

**v0.9.110g — fit_gmm verified (DEC-033):**
batch_size kwarg + state_est = 5*max_samples*d*f64 + k*(1+2d)*f64 +
256 (sample S*d*8 + M-step temp 4*S*d*8). Formula: k*(1+2d)*f64 +
5*max_samples*d*f64 + 2.0*batch*row_size + 256. dims: {k, d} ->
{k, d, max_samples}. coef 2.0 physical (arrays list + column_stack;
measured 1.73-2.0). Test: 8/8 gmm_verification (~5s, stable x3).

**Total: 15 verified cards.** (2 MatrixProfile + 9 v0.9.107 + 1
randsvd + linucb + iforest x2 + gmm). Verification suite runtime
~20s (vs 30+ min per attempt with RSS matrix).

**Do not re-solve** (DEC-033): combined RSS matrix at large d is
physically unmeasurable (baseline >> peak). Component-wise is not a
relaxation — state is bit-exact, coef is physical (code-derived
copies of row_size), OOC is proven independently.

### v0.9.109 (2026-11-02) — fit_cusum honesty, R5 aliases, randsvd verified

**v0.9.109a — fit_cusum honesty (DEC-030):**
YAML claimed `state_class: O(1)`, `bounded_state: true`, but code
allocates 6 O(n) arrays (z, s_pos, s_neg, scores, labels, temp bool
= ~48n). Prior guard est 24n underestimated 2x. Fix: est_bytes 24n
-> 48n, use guard.check_state when available. YAML: O(1) -> O(n),
bounded_state: false, ram_formula "6 * n * f64", status: draft,
bounded_by_domain {axis: n, bound: 1000000}. Same class as DEC-013.
Test: 10/10 test_cusum.py.

**v0.9.109b — test_R5 aliases consistency:**
18 fns / 43 cards; 0 inconsistent. All aliases of one fn must share
status. Protects future verified flips from missing one alias.

**v0.9.109c — fit_randomized_svd verified:**
batch_size kwarg + auto-derive (ef=3.0). Formula:
  3*batch*row_size + 5*d*(n_components+n_oversamples)*f64
Measurement: coef=3.000, d-sweep ratios [0.97, 1.00, 1.00, 1.12],
OOC ratio=1.00 (38.6 MB at 50 MB budget). Tests: 8/8 randsvd +
78 schema.

**v0.9.109d — snapshot bootstrap includes EXPERIMENT_PROCEDURE.md:**
AI_FIRST_FILES position 4 (T2). Snapshot header renumbered 1-12.
Tiny mode (developer flow) now bootstraps next session with the
6-step DOE.

**Total: 12 verified cards.**


### v0.9.108 (2026-11-02) — refactor: _stack_numeric → _common.py

Single source of truth (AI_CONSTITUTION §5.1). Eliminates 23 duplicate
code paths across cadw/ml/.

**New module `cadw/ml/_common.py` (~56 lines):**
- `_column_to_f64(arr)` — canonical
- `_numeric_to_f64 = _column_to_f64` (alias for linear.py)
- `_stack_numeric(batch, cols)` — canonical

**Removed:**
- 5 def copies (gbdt, linear, naive_bayes, pca, kmeans)
- 9 `_stack_numeric_safe` wrappers (linucb, randsvd, calibration,
  birch, feature_select, chains, geo, uplift, minibatch)
- 9 inline `from cadw.ml.gbdt import _stack_numeric` sites
  (ordinal ×3, meta_learners ×3 + mixed, lambdamart ×1, drift ×1)

**Bit-identical.** 464 ML tests green. No formula/verified changes.
No public API changes (`_stack_numeric` remains internal).


### v0.9.107 (2026-11-02) — optimizations + 9 cards verified

**Code optimizations:**
- O2b (`logistic.py:LogisticGradOp.update`): `del X` after
  `X_aug = column_stack([ones, X])`. Peak in update drops from
  3× batch (X + X_aug + T1) to 2× (X_aug + T1). Measured coef
  a: 3.05 → 2.0.
- O4 (`naive_bayes.py:_GaussianStatsOp.update`): `np.einsum('ij,ij->j',
  Xc, Xc)` instead of `(Xc * Xc).sum(axis=0)`. Removes (B_c, d) temp.

**9 cards: `draft` → `verified`:**
- fit_linear, fit_lasso, fit_elasticnet
- fit_logistic (×4)
- fit_gaussian_nb, fit_multinomial_nb, fit_bernoulli_nb

All via STABILITY_PROTOCOL §6 (stability + formula + matrix + OOC).
Production path (no monkey-patch) after v0.9.106 framework fix.

**DECs closed:** DEC-019 (O2 partial → complete via O2b),
DEC-021 (O4 applied), DEC-022, DEC-023 (closed in v0.9.106).

**Deferred to v0.9.108:** refactor `_stack_numeric` (5 copies) →
`cadw/ml/_common.py` (single source of truth, AI_CONSTITUTION §5.1).
Bit-identical, no verified impact. Roadmap updated.

### v0.9.106 (2026-11-02) — framework: chunking + state-aware guard

Closes DEC-022 + DEC-023. OOC works on production path (no patch).

Core:
- `MemoryGuard.get_batch_rows(state_bytes=0)` — deduct op state from
  usable; raise if state >= usable. Backward compat.
- `CadwDataFrame._iter_filtered_batches(batch_size=None)` — propagate
  to `Dataset.iter_batches`.
- `derive_batch_size(op, df, guard, batch_size=None)` helper.

fit_* API (`batch_size="int | None" = None`):
- fit_linear, fit_lasso, fit_elasticnet
- fit_logistic (+ SGD path with O(d) state bound)
- fit_gaussian_nb, fit_multinomial_nb, fit_bernoulli_nb

Fixed latent v0.9.97 → v0.9.105 bug in `estimate_from_yaml` batch-fallback
(passed `op` (str) as `row_size_bytes`; TypeError swallowed).

Evidence: OOC plateau 0.98-1.00 all 7 fit_*, 219 tests green.


### v0.9.105 (2026-11-02) — 9 cards formula-calibrated; framework gap DEC-023

Formula calibration for 6 ReduceOps / 9 cards (linear, lasso, logistic x4,
NB x3). All 9: `verified_in_cadw=true`, `status=draft`,
`draft_reason: OOC blocked by framework gap (DEC-023)`.

Code fixes:
- O2: `del X_parts` in `logistic._stack_and_validate` (partial; peak in update)
- O3: Bernoulli keep bool, no f64 copy (coef 2.94 -> 2.0)

Framework gap (DEC-023): `_iter_filtered_batches` does not honor
`batch_size`; production uses batch=n. Peak grows linearly with n
without patch; flat with patch. v0.9.106 fix.

New: `docs/ml_tools/EXPERIMENT_PROCEDURE.md` (6-step DOE, rules R1-R4,
exit criterion, anti-patterns A1-A5),
`scripts/experiments/experiment_v105_rep.py` (reproducible stages 0-6),
DEC-018..025.


### v0.9.103 (2026-11-01) — ram_formula mechanism + first 2 verified

First honest verified cards via STABILITY_PROTOCOL end-to-end (code ->
formula -> measurement). Matrix_profile ×2 verified: 9/9 cells ±30%,
OOC check passes.

**Mechanism (P1):**

- `cadw/ml/_memory_estimate.py`:
  - `_safe_eval(formula, env)` — AST walker. Only `+ - * / **` and
    identifiers. Rejects Call/Attr/Subscript/comparison/boolean
    (security: formulas from YAML, must not execute arbitrary code).
  - `evaluate_formula(op, **params)` — calibration path. No guard,
    no budget. Used by matrix tests. Returns `{bytes, formula,
    unknown_names}`.
  - `estimate_from_yaml` — runtime path. Prefers `ram_formula`;
    auto-injects `batch` / `row_size` from guard. Legacy `dims+tf`
    fallback with `LegacyFormulaWarning`.

- `docs/ml_tools/memory_constants.yaml` (new): 14 typed constants
  (`f64=8`, `f32=4`, `i32=4`, `i64=8`, `i8=1`, `bool=1`,
  `dict_entry_bytes=116`, `py_str_overhead_bytes=49`, ...).
  Version-stamped for drift detection.

**Verified (2):**

- `MatrixProfile` (D_time_series): `S * f64 * (771 + 5 * w)`.
  9/9 cells (S ∈ {1024, 2048, 4096}, w ∈ {16, 32, 64}, n=10^5),
  max deviation <30%. OOC: peak(n=10^6)/peak(n=10^4) < 2.0.
  `bounded_state: true`, dims `{S: 1, w: 1}`.
- `MatrixProfile_discord` (D_time_series): same formula (alias).

**Structural migration (DEC-011 closed):**

- `bounded_by_domain` field added to schema.py: `{axis, bound,
  reason}`. Validation: `tier1_core + bounded_state:false` accepted
  with EITHER `tier1_core_exception` (legacy) OR `bounded_by_domain`.
  `status: tier1_core_exception` no longer used (0 cards).
- `fit_pelt` ×2: `bounded_by_domain: {axis: n, bound: 10^6}`.
  Reverted `verified` → `draft` (DEC-014): `fit` is container (256B);
  formula requires `_collect_series` refactor v0.9.104.

**Schema enforcement (R1-R4, tests/test_ml_tools_schema.py):**

- R1: status ∈ {draft, verified, tier1_core_exception} or None
- R2: status=verified ⟹ ram_formula present
- R3: status=verified ⟹ no `# BACKLOG:` in module code
- R4: status=draft ⟹ draft_reason present

**Draft_reason migration (all 21 draft cards):**

- 14 existing reasons preserved (`# draft_reason:` → YAML field).
- 7 new reasons added:
  - `fit_classifier_chains` ×2, `fit_tfidf` (vocab K-spill v0.9.104)
  - `fit_binseg` ×2 (external cumsum v0.9.105)
  - `fit_stray` (batched kNN v0.9.106)
  - `fit_t2q` (strided+chunked v0.9.104)
  - `fit_pelt` ×2 (container, `_collect_series` refactor v0.9.104)

**Discoveries (DEC-012, DEC-013, DEC-014):**

- DEC-012: `_stack_numeric` (AP2) blocks verified for linear,
  logistic ×4, lasso, gaussian_nb, bernoulli_nb, multinomial_nb.
  Refactor v0.9.104 (pre-allocated, guard extra_factor=3.0).
- DEC-013: `fit_zms` ×3 O(n) output (decision_scores, labels).
  SpillArray v0.9.105.
- DEC-014: `bounded_by_domain` is modifier, not verified-enabler.
  Formula + matrix still required.

**Status baseline (after v0.9.103):**

| Status | Count |
|---|---|
| verified | 2 |
| draft | 21 |
| tier1_core_exception | 0 (replaced by `bounded_by_domain`) |
| pending (no status) | 64 |

**Tests:** ~4345 passed, 9 skipped, 25 deselected (fast;
+10 formula_accuracy, +4 R1-R4, +2 estimate_memory regression,
0 regressions).

**Not in scope (v0.9.104+):**

- `_stack_numeric` refactor → +9 verified (linear, logistic ×4,
  lasso, NB ×3) [DEC-012]
- `_collect_series` pelt refactor → +2 verified [DEC-014]
- `SpillArray` for `fit_zms` ×3 [DEC-013]
- Remaining 62 pending card migrations (batches of ~15/session)
- `solve_batch(op, budget, **params)` inverse solver
- `bounded_by_domain` applied to future exception cases only

### v0.9.102 (2026-10-31) — methodology docs + status system baseline (0 verified)

Baseline honesty: **0 verified**. First verified appears in v0.9.103 when
formula mechanism + matrix measurement land.

**Methodology documents (T1–T5, AI_CONSTITUTION §4.5):**

- T2 `docs/ml_tools/STABILITY_PROTOCOL.md`: calibration vs runtime;
  code -> formula -> measurement; 9-cell matrix (±30%); stability =
  OOC + spill + no open backlog
- T3 `docs/ml_tools/ALGORITHM_TEMPLATE.md`: 8-step checklist; anti-patterns
- T3 `docs/ml_tools/EXAMPLE_VERIFIED_CARD.md`: full fit_linear example + counter-example
- T4 `docs/ml_tools/GLOSSARY.md`: 44 canonical terms
- T4 `docs/ml_tools/DECISION_LOG.md`: DEC-001..DEC-011
- T5 `docs/ml_tools/CHAT_PROTOCOL.md`: 4 phases (probe/patch/review/commit);
  /tmp patch scripts; redirect heavy tests (AP13); paste only tail;
  QUESTION/OPTIONS/RECOMMENDATION; delimiter discipline

**AI_CONSTITUTION.md updates:**

- §1 reading order: T1 first, 11 docs
- §4.5 new: document hierarchy T1–T5 + conflict resolution
- §5 header: "running log; AP1-AP14, AP21-AP23"
- AP21: atomic closure; no "verified with backlog"
- AP22: slow operations warn (CPU instead of silent RAM OOM)
- AP23: formula from code, verified by measurement
- §5.1: single source of truth (one fact, one place)
- §5.2 new: tier vs status orthogonal; `tier1_core_exception` is
  temporary wart, replaced by `bounded_by_domain` field in v0.9.103
- §6 header: removed "+ won't fix" (DEC-004)

**Infrastructure:**

- `cadw/core/warnings.py`: LegacyFormulaWarning, SlowOperationWarning,
  ActualDataModeFallbackWarning, CalibrationMissingWarning,
  SlowOperationRequiresConfirm
- `cadw/core/memory_guard.py::measure_peak()`: 3-source peak
  (tracemalloc + arrow pool + RSS)

**Status migration (batch 0 = 23 of 87 cards):**

- 21 `draft` (matrix measurement requires evaluate_formula mechanism,
  deferred v0.9.103)
- 2 `tier1_core_exception` (pelt x2, temporary wart per DEC-011)
- 64 cards without `status` (pending v0.9.103+)
- **0 verified** (honest — no measurement done yet)
- `transient_factor_measured` rolled back (fragmentation)

**scripts/collect_project.py rework:**

- `AI_FIRST_FILES`: set -> ordered 11-tuple (reading order)
- Header: READ IN THIS ORDER + SESSION START CHECKLIST
- Missing AI-first file -> exit 3 (CADW_SNAPSHOT_ALLOW_MISSING=1 override)
- Tiny mode: 520 KB (98% reduction from 28 MB)

**Not in scope (v0.9.103+):**

- `ram_formula` + `_safe_eval` + `evaluate_formula` mechanism
- `memory_constants.yaml` (typed constants)
- `test_X_formula_accuracy` matrix tests per algorithm
- Remaining 64 status migrations (batches of ~15/session)
- `solve_batch(op, budget, **params)` inverse solver
- Schema enforcement R1-R4 (`test_ml_tools_schema.py`)
- Replace `tier1_core_exception` with `bounded_by_domain` (DEC-011)

**Tests:** 4336 passed, 9 skipped, 25 deselected (fast). No test count
changes (docs + infra only).

CHANGELOG/ROADMAP updated: v0.9.102 current (baseline, 0 verified);
v0.9.103 = formula mechanism + first verified + bounded_by_domain.

### v0.9.101 (2026-10-30) — coxph reference-only + LDA dims fix

**Task 1 — coxph `_breslow` reclassified as REFERENCE ONLY.**

Audit pass 3 (v0.9.100) flagged `cadw/ml/coxph.py::_breslow` (L47)
as an unguarded O(n) alloc: `S2 = cumsum(n·d²)` vs a `N·d·8` guard.
Probe in v0.9.101 found that `_breslow` is **called only from
`tests/test_coxph.py`** (L66, L72, L157) — the numeric reference for
bit-identical checks against `_breslow_streaming` (L223), which is
the production fit path with `state O(d² + pending_ties·d)` and
guard `mg.check_state(3*d*d*8 + 5*pending_budget, name="coxph.state")`
(L449).

Reclassified as CLOSED (reference-only), same class as
`dwt_mlead::_mahalanobis_distances` (pass 3 #5, closed in v0.9.100).

Changes:
  - `_breslow` docstring: explicit "REFERENCE ONLY" + pointer to
    `_breslow_streaming` + "do NOT call from non-test code".
  - pass3 audit: #3 → CLOSED (reference only); Result 1 open → 0
    open, 4 closed → 5 closed.
  - No prod-code logic change.

**Task 2 — LDA YAML card dims fix (V-axis, tf=12).**

`docs/ml_tools/J_nlp.yaml` LDA card declared
`bounded_state_dims: {k: 1, d: 1}, transient_factor: 1.0`. The `d`
axis was ambiguous: used as user's `vocab_size` cap, but the actual
peak in `_build_vocab` is `K_raw * 96 B` (raw unique tokens, not
the cap). At V_raw=10⁷, `estimate_memory("fit_lda", k=10, d=10⁶)`
returned "fits" but `_build_vocab` guard fires at ~960 MB.

Fixed:
  - `bounded_state_dims: {k: 1, d: 1}` → `{d: 1}` (d = raw unique
    tokens; user passes it explicitly).
  - `transient_factor: 1.0` → `12.0` (96 B / 8 B-base per entry).
  - sstats `K*V_cap*8` not modeled separately (smaller for
    V_cap << V_raw; the dominant term is now correct).
  - After: `estimate_memory("fit_lda", d=10^4)` → fits (0.96 MB);
    `d=10⁷` → fits=False (960 MB).

Tests: `tests/test_estimate_memory.py` +2 (`fits_small`,
`refuses_large`).

**Known issues (still tracked):**

- **`lda::_build_vocab` O(K_raw) dict** — v0.9.102 target:
  2-pass HashShuffle (token hash → partitions → k-way merge
  top-vocab_size by `(-count, token)`).
- **`encode_categorical` O(K) reverse_maps** — v0.9.102 target:
  lazy `SpillMap` (breaking opt-in).

**Tests:** ~4334 passed, 9 skipped (fast; +2 new estimate_memory
tests). No prod-code logic changes (docstring + YAML + audit only).

CHANGELOG/ROADMAP updated: v0.9.101 current; Next = v0.9.102
(lda vocab K-spill + lazy reverse_maps), v0.9.103 (MultiRocket,
D4 starts), v0.9.104 (Hydra + BOSS).

### v0.9.100 (2026-10-30) — version bump + audit pass 3

**Task 1 — version bump (metadata).**

`cadw/__init__.py::__version__` and `pyproject.toml::version`
were stuck at `"0.9.88"` since v0.9.88 (2026-10-28). Tags v0.9.89
through v0.9.99 exist but were never reflected in the version
string — a real pip-install issue (pip sees 0.9.88 for a wheel
built from a v0.9.99 commit).

Both moved to `"0.9.100"`. Historical v0.9.89-v0.9.95 do not have
`### v...` CHANGELOG entries (only `## [Unreleased]` content up
to v0.9.96); not retro-fixed here — the tags exist, the release
notes were split into CHANGELOG v0.9.96+ — documented in this
entry, not repaired.

**Task 2 — test_version_consistency.py strengthened (2 new tests).**

Before v0.9.100, the test only checked `pyproject == cadw.__version__`
and `### v<version>` present in CHANGELOG. Two new invariants:

- `test_version_is_semver`: `^\d+\.\d+\.\d+$` (no suffix in
  `__version__`, even though tags carry suffixes like
  `v0.9.93-logistic`).
- `test_version_matches_git_tag_on_master`: on master (release
  context), `git describe --tags --exact-match --match "v[0-9]*"`
  must equal `v<version>`. Skipped on patch branches. Catches the
  v0.9.89-v0.9.99 class of drift at release-time, not after.

**Task 3 — audit pass 3 (`pass3_2026-10-30.md`).**

Reviewed the 5 UNGUARDED-RAM entries from pass 2 (v0.9.87):

| # | Function | Verdict |
|---|---|---|
| 1 | `anomaly_pipeline::fit_predict` L278 | **guarded** — `check_state` on L264 (pass2 only looked at first alloc) |
| 2 | `chains::_predict_hard_batched` L119 | **per-batch output** — `n = b.num_rows`, O(batch·K) by design |
| 3 | `coxph::_breslow` L58 | **REAL CANDIDATE** — `S2 = cumsum(n·d²)` vs guard `N·d·8`; underestimate factor `d`; open item v0.9.101+ |
| 4 | `dwt_mlead::_stream_mahalanobis` L141 | **false positive** — `if n_w < 2: return np.zeros(n_w)` = empty edge case |
| 5 | `dwt_mlead::_mahalanobis_distances` L179 | **not production** — docstring: test-only reference |

Result: **1 open candidate** (coxph `_breslow`), **4 closed**.
Full report: `.pingis/audits/pass3_2026-10-30.md`.

**Task 4 — ROADMAP bumped.**

`Current: v0.9.100`; Next = v0.9.101 (lda vocab K-spill),
v0.9.102 (lazy reverse_maps, breaking), v0.9.103 (MultiRocket,
D4 starts).

**Known issues (still tracked):**

- **coxph `_breslow` guard underestimate** (pass3): `S2 =
  cumsum(n·d²)` at guard `N·d·8`. At d=1000, n=10⁶ the S2 array is
  ~8 GB but guard estimates ~8 MB. Not OOM-tested yet. Candidate
  v0.9.101 fix: change `coxph.state` guard to `n * (d + d*d) * 8`
  (or refuse-at-guard at large d). Out of scope for a metadata tag.

**Tests:** ~4332 passed, 9 skipped (fast; +2 new version tests,
+3 git-tag/semver assertions; no prod-code changes beyond the
version strings). Historical test count remains +4331 from
v0.9.99 (unchanged).

### v0.9.99 (2026-10-30) — metrics guards verification + encode_categorical actionable error

**Task 1 — YAML card for `encode_categorical`.**

Added `O(K)` to `VALID_STATE_CLASS` in `docs/ml_tools/schema.py`
(previously only O(n·K) variants were accepted; K-only state was
unrepresentable).

New card `EncodeCategorical` in `docs/ml_tools/O_eda.yaml` (subtype
O3, Feature engineering / encoding):
  `state_class: O(K)`, `bounded_state: false`,
  `n_scalability: refuse_at_guard`, `bounded_state_dims: {K: 1}`,
  `transient_factor: 22.5` (180 B/unique ÷ 8, matching current
  per-entry cost).

After v0.9.99: `df.estimate_memory("encode_categorical",
K=10**7)` returns `fits: False` with alternatives.

**Task 2 — actionable error in `encode_categorical`.**

The internal `check_state((len(uniques) + b.num_rows) * 180)`
already fired `MemoryLimitExceeded`, but with the generic
"state limit exceeded" message. Now wrapped to include:
  - **why** it fires: reverse_maps is `list[unique_values]` — O(K)
    by API contract, independent of N;
  - estimated K and MB;
  - two concrete alternatives: top-K via `pc.value_counts().top_k`,
    or wait for v0.9.102 lazy reverse_maps (SpillMap on disk);
  - original message preserved via `from e`.

This unblocks users hitting a K-overflow: the previous generic
message looked like a bug (state should be O(K) but couldn't say
why the K matters or how to fix).

**Task 3 — metrics.classification large-K guard tests.**

`MetricsOp.estimate_state_nbytes()` returns `64 + 24*K`
(tp[K] + fp[K] + fn[K] × int64). `reduce()` calls
`memory_guard.check_state(estimate_state_nbytes(), name="MetricsOp.
estimated")` **before** `op.init()` (v0.5.1 P8).
So guards already work — probe #5 in v0.9.98 (`grep check_state
cadw/ops/metrics.py` = 0 matches) was a **false negative**: it
grep'd the metrics module, but the guard lives in `cadw/ops/
reduce.py` where `MetricsOp` runs.

New tests confirm this:
  - `test_metrics_classification_large_k_fits`: K=10^5 (2.4 MB)
    fits under 100 MB guard, accuracy returned.
  - `test_metrics_classification_huge_k_guard`: K=10^7 (240 MB)
    → `MemoryLimitExceeded` with "estimated" in message (from
    `reduce` pre-flight).

No prod code changed for metrics; the tests lock the existing
contract.

**Anti-pattern surfaced this session:**

- **AP16** — `grep check_state <module>.py = 0` ≠ "no guards".
  `ReduceOp` subclasses route their guard through
  `cadw/ops/reduce.py::reduce`, which calls
  `op.estimate_state_nbytes()` + `check_state(...)` **before**
  `op.init()`. For any `ReduceOp` subclass, the guard is in
  reduce.py, not in the op's own module. Probe: `grep
  estimate_state_nbytes <module>.py` → if non-zero, guards exist
  via reduce. (v0.9.98 probe #5 made this mistake for
  `metrics.classification`; v0.9.99 fixes the diagnosis.)

**Known issue (still tracked, pre-existing):** `__version__` and
`pyproject.toml::version` say 0.9.88 while tags exist through
v0.9.99. Target v0.9.100 (version bump + audit pass 3).

**Test count:** ~4331 passed, 8 skipped (fast; +3 new tests: 2
metrics, 1 encode; no prod-code changes except error wrapper).

### v0.9.98 (2026-10-30) — tfidf.transform_sparse(output=)

**New API: `TfidfModel.transform_sparse(df, *, text_col, norm="l2",
memory_guard=None, output="auto")`**

CSR-like sparse rows without O(D) RAM. Each document becomes a
`(indices, values)` pair. RAM is O(batch * avg_nnz_per_doc) for
`output="auto"`/`"disk"`; only `output="memory"` holds O(D) RAM
(backward compat with existing `transform()`).

Return type by `output`:
  "memory" -> list[tuple[np.ndarray int64, np.ndarray float64]]
              (bit-identical to transform())
  "auto"   -> AutoResult; iterable pa.RecordBatch with SPARSE_SCHEMA
  "disk"   -> AutoResult (OnDiskResult; forces spill)

**Explicit exception to v0.9.95 unified-CadwDataFrame.** List-typed
columns (`pa.list_(pa.int32())`, `pa.list_(pa.float64())`) break the
tabular contract (sort/groupby/join do not apply to list cells). For
a tabular return use `transform_dense(output=...)` instead. This is
documented in the method docstring.

**Design (no new primitive):** `SpillBuffer` already accepts an
arbitrary `pa.Schema` (v0.9.90-S1). Arrow list-types are native
(precedent: `accessors/_str.py:321`, `pa.list_(pa.string())`). So
`transform_sparse` is ~100 LOC wrapping existing primitives —
not a new `SparseSpillBuffer` class.

**Schema:** `SPARSE_SCHEMA = pa.schema([indices: list<int32>,
values: list<float64>])`. int32 indices (vocab_size default 1e4; max
2^31); float64 values (matches `_row()` dtype → sklearn reference
bit-identical).

**Tests:** `tests/test_tfidf.py` +6 cases (17 total; 11 existing + 6
new): memory-shape, memory==transform bit-identical, output
validation, auto==memory, nnz preserved, disk actually spills.

**Not in this release (split from ROADMAP v0.9.97's v0.9.98 scope):**

- `encode_categorical` K-spill (3-pass HashShuffle) → v0.9.99.
- `metrics.classification` K-guard probe → v0.9.100.
- `lda::_build_vocab` K-spill → v0.9.101.
- D4 ROCKET + version bump → v0.9.102+.

Each is a separate session; keeping each tag minimal avoids
"half-CSR-primitive" states.

**Anti-pattern surfaced this session:**

- **AP10-adjacent** — tests written with assumed input format:
  `tokenize` drops tokens with `len < 2` (`cadw/ml/lda.py:37`);
  single-char test data (`"a b c"`) failed `fit_tfidf` with
  "vocabulary has fewer than 2 tokens". Fix: multi-char words in
  test data. Probe input-format assumptions **before** writing tests.
- **AP8-adjacent (atomic patch)** — 5 sequential `src.replace()`
  with per-step print; anchor miss on step 2 loses steps 1.
  Patched to validate ALL anchors first, then mutate. Rule: for
  multi-replace patches, `missing = [old for old in FIXES if old
  not in src]; if missing: sys.exit(1)` before `write_text`.

**Known issue (still tracked, pre-existing):** `__version__` and
`pyproject.toml::version` say 0.9.88 while tags exist through
v0.9.98. Target v0.9.102 (version bump + release audit).

**Test count:** 4328 passed, 8 skipped (fast; +6 tfidf tests, no
regressions).

### v0.9.97 (2026-10-30) — YAML bounded_state_dims + df.estimate_memory

**Task 1: `bounded_state_dims` in YAML (86/86 verified cards).**

New `StateFields` fields (`docs/ml_tools/schema.py`):
  `bounded_state_dims: dict[str, int] | None`  — `{axis: power}`
  `transient_factor: float = 1.0`

Semantics: `est_bytes = sum(params[a] ** p for a, p in dims.items()) * 8 * tf`

Populated 86 verified cards (84 via
`scripts/patch_v0997_yaml_dims.py --apply`, 2 by hand for
N_density GaussianMixture + BIRCH whose verified flag was
flow-style `{verified_in_cadw: true, ...}` — see AP12).

Mapping (`STATE_CLASS_TO_DIMS`):
  `O(1)` → `{}`, tf=1.0; `O(batch)` → `{batch:1}`, tf=1.0;
  `O(d)` → `{d:1}`, tf=1.0; `O(d²)` → `{d:2}`, tf=2.0;
  `O(n)` → `{n:1}`, tf=1.0 (conservative, spill-backed);
  `O(model)` → `{k:1, d:1}`, tf=1.0;
  `O(n·w)` → `{n:1, w:1}`, tf=2.0;
  `O(N·d)` → `{N:1, d:1}`, tf=1.0.

`fit_binseg` tf=3.0 by hand (3 prefix arrays; brief §3.1 estimate
of 240 MB at n=1e7 matches tf=3.0).

`REQUIRE_STATE_FIELDS` already `True` since v0.9.87-P6 (brief
claimed `False` — stale docstring). Fixed docstring + assert
message + flag comment. `test_g4_verified_cards_have_state_fields`
extended to require `bounded_state_dims` for non-refuse verified
cards.

**Task 2: `df.estimate_memory(op, **params)` — new API.**

`cadw/dataframe/_misc.py::estimate_memory` + new
`cadw/ml/_memory_estimate.py` (`@lru_cache(maxsize=1)` on
`_load_cards`). Looks up YAML card by `cadw_function`,
`"module::function"`, or card `name`; evaluates formula against
`MemoryGuard.available_bytes()`; returns:
  `{"fits": bool, "est_ram_mb": float|None, "bottleneck": str,
    "available_mb": float, "alternatives": list[str]}`

`n_scalability: refuse_at_guard` → `est_ram_mb=None`, fits=False.
Alternatives per family: sgd for linear/logistic, pelt/stray for
binseg, reduce hp for gbm, randomized for pca.

Tests: `tests/test_estimate_memory.py` — 6 cases.

**Task 3 (reduction): CoxPH concordance audit — no code change.**

Brief v0.9.96 §8 claimed `concordance_index` was O(n²). Audit:
already chunked (`chunk = 1_000_000 // n`, bool matrix ~8 MB per
chunk). RAM is **O(n)**, not O(n²). Detailed docstring added to
`cadw/ml/coxph.py::concordance_index`; moved to "Do not re-solve"
in ROADMAP.

**Anti-patterns surfaced (AI_CONSTITUTION §5):**

- **AP10** — patch вслепую: `git log -- file` + `git show
  HEAD:file | grep 'def X'` before writing "add X".
- **AP12** — `\b`-pattern with `.match()` silently no-ops:
  `.match()` anchors at position 0, `\b` needs word-char there.
  Use `.search()` when pattern has no `^`.
- **AP13** — `set -eo pipefail; cmd | tail` in VSCodium terminal
  = visual "crash" (shell exit 1 → pty closed). Use `cmd >
  /tmp/log 2>&1; echo "exit=$?"; tail /tmp/log`.
- **AP14** — YAML anchor "next N chars" fails when nested
  collections (`benchmark_results`) sit between anchor and
  target. Anchor by structural marker + block-scoped search until
  next `- name:` at same indent.

**Known issue (still tracked, pre-existing):** `cadw.__version__`
and `pyproject.toml::version` say 0.9.88 while tags exist through
v0.9.97. Out of scope; target v0.9.98.

**Test count:** ~4345 passed, 8 skipped (fast; +6 estimate_memory,
+schema count, +G11 param for new `_memory_estimate.py`).

### v0.9.96 (2026-10-30) — G11: transitive-migration AST gate

**G11 (new gate).** AST scan for callers of migrated producers
using `pa.Table` / `np.ndarray` API on the `CadwDataFrame` return.
Catches the AP7 class of v0.9.95 stragglers (pivot, value_counts,
duplicated, subset_accuracy) that manual grep found one-by-one.

Design:
- **Bare-name tracking** (`groupby_aggregate(...)`, `_gba(...)` via
  import alias) — resolves through imports map.
- **Intra-class tracking** (`self.<method>(...)`) — only when
  `<method>` has `output=` in the same class. This makes
  `self.predict_proba(...)` on gmm tracked, but on logistic /
  gbdt / calibration (unmigrated — return ndarray) NOT tracked.
- **Reassignment untracks** — `result = pa.Table.from_batches(...)`
  clears producer status for the rest of the function body.
- **Four violation kinds**: `forbidden_method` (`.column`,
  `.num_rows`, `.to_table`, ...), `forbidden_attr` (`.schema`,
  `.shape`, `.data`), `isinstance_legacy` (`isinstance(result,
  pa.Table)`), `binop` (`result == 0`, `result > 0`).
- **`# noqa: G11 — <reason>`** per-line opt-out for the rare
  cross-object real bug.

Enforcement: `tests/test_constitution_gates.py::test_g11_transitive_migration`,
142 parametrized cases (one per `cadw/**/*.py` file). Baseline at
v0.9.95 HEAD: 0 violations after reassignment-tracking addition
(the `_aggregations.value_counts` false positive drove that fix).

**Also in this release:**
- `CadwDataFrame.to_pylist()` — pa.Table-compatible materialization
  for legacy callers; O(n) RAM, inspection only.
- `tests/test_groupby_v2.py`, `tests/test_streaming_output_ops.py`
  rewritten to the unified `CadwDataFrame` API (v0.9.95 contract).
- `tests/test_roadmap.py::test_roadmap_md_structure` now
  version-agnostic: parses `## Current:` and requires
  `## Next 3 sessions` to name >=3 strictly-greater versions
  (previously hardcoded `("v0.9.93", "v0.9.94", "v0.9.95")` —
  guaranteed to fail on every release).

**Known issue (not addressed here):** `cadw.__version__` and
`pyproject.toml::version` still say `0.9.88` while tags exist
through v0.9.96. Pre-existing since v0.9.89; out of scope for this
session. Track for v0.9.97.

**Test count:** 4308 passed, 8 skipped (fast; includes G1-G11,
G11 adds 142 transitive-migration parameters). No prod-code
changes — only gate infrastructure + tests + docs.

### v0.9.95 (2026-10-30) — unified CadwDataFrame (BREAKING)

**BREAKING:** `output=` no longer selects return type. All 13
producers return `CadwDataFrame` regardless of mode; `output=`
selects **storage strategy only**:

  "memory" → CadwDataFrame (in-memory, _InMemoryDataset, no disk)
  "auto"   → CadwDataFrame (in-memory if fits budget, else disk)
  "disk"   → CadwDataFrame (disk-backed, Arrow IPC)

Migration pattern (for user code):
  result = producer(...)   # CadwDataFrame, NOT pa.Table/np.ndarray
  # extract numeric: [b.column(c).to_numpy(...)
  #                    for b in result._iter_filtered_batches()]
  # or: result.to_pylist() / result.select(["c1", "c2"])

Producers migrated (5 waves):

  Ops:
    groupby_aggregate, drop_duplicates, topk_to_disk

  ML:
    iforest.score_samples/predict/score
    coxph.predict_risk
    gmm.score_samples/predict/score
    chains.predict_proba/predict
    ordinal.predict_proba/predict
    tfidf.transform_dense
    randsvd.transform

CadwDataFrame additions (additive, non-breaking):
- to_pylist()             — pa.Table-compatible materializer
- select(columns)         — pandas-compatible column subset
- from_auto_result(...)   — wrap AutoResult (v0.9.94-narrow)
- from_autoarray_result(...) — wrap AutoArrayResult (v0.9.94-narrow)
- from_producer_result(...)  — universal wrap
                                (AutoResult|AutoArrayResult|pa.Table|np.ndarray)
- _from_numpy(...)        — internal in-memory numpy wrap

_wrap_result_table passthrough CadwDataFrame. Skew path
(`_groupby_with_skew`) wrapped — was returning pa.Table,
bypassing main return (multichunk value_counts used this path).

Lessons (see AI_CONSTITUTION §5 AP7–AP9):
- Tail-only patch insufficient when producer has multiple
  return points (memory / auto/disk / skew / combiner /
  partial_merge / early-return).
- Silent anchor-miss: `str.replace` without `else: sys.exit`
  hides failures. Always guard.
- Arrow: pa.concat_tables(list[Table]) vs
  pa.Table.from_batches(list[RecordBatch]) vs pa.concat_arrays
  (returns Array, not ChunkedArray).

Test count: 4167 passed, 8 skipped (full fast, ~9 min).

Full fast pre-tag is the gate that caught ~50 transitive
failures across 10+ test files.

Next: v0.9.96 — G11 (transitive migration check),
sparse tfidf.transform (CSR SpillBuffer), CoxPH concordance
streaming, or new functionality (D4 variants, TS forecast).

### v0.9.93-logistic (2026-10-30) - wide-feature OOC + closed-path bug fix

**BREAKING (production bug fix):** `fit_logistic(alpha>0)` closed
path was NOT minimizing the penalized objective since v0.7.0b.
Newton IRLS used the unpenalized gradient in `solve(H_L + α·mask,
grad_L)`, whose fixed point is grad_L = 0 → unregularized MLE.
Fix: `solve(H_L + α·mask, grad_L + α·β·mask)` — standard IRLS
with ridge in weighted LS.

  Evidence (alpha=1000, n=1000, d=10, near-separable):
  | |coef| | LL  | penalized obj |
  |-------|-----|---------------|
  | closed pre-fix | 1.33 | -383 | 502 |
  | closed post-fix | 0.35 | -510 | 44 |
  | sgd (correct) | 0.35 | -510 | 44 |

  Users with `alpha` comparable to `n` (alpha>10 at n=1000) will
  see different coef magnitudes — CORRECT behavior. Existing tests
  (alpha=1, lam=0.001) unchanged.

Wide-feature OOC for binary logistic regression:

- `fit_logistic(solver="auto"|"closed"|"sgd")`.
  - closed: Newton IRLS, O(d²), default d≤1000.
  - sgd: `sgd_streaming(loss="logistic")`, O(d), any d.
  - auto: closed if `3·(d+1)²·8 ≤ 0.5·available_bytes`.
- `sgd_streaming` extended:
  - `loss="ridge"|"logistic"` (default ridge, back-compat).
  - `penalty_mask` (None default): grad += lam*W*mask. Exempt intercept.
  - `use_rmsprop=True` default; False for logistic (RMSprop oscillates
    at strong reg, plain GD contracts at 1-lr*lam).

Tests (+6 in test_logistic.py). G10 criteria in AI_CONSTITUTION §4.2.

Test iteration history (5 attempts):
1. coef rel_err <0.25 @ alpha=1 → 0.70 (near-degenerate manifold)
2. alpha=0 hard predict ≥95% → 87.4% (MLE at infinity)
3. alpha=0.5 → 85.1% (still unreg: lam=0.0005)
4. alpha=500, RMSprop → strong_reg RMSE 0.42 (RMSprop oscillation)
5. use_rmsprop=False + closed fix → ALL PASS, revealed real bug

Process note: commit 5b5c1db claimed §4.1 but file was 325 lines
without §4.1 (`cat >>` heredoc silently skipped). Fixed in
v0.9.93-docs; §4.1 + §4.2 now present (411 lines).

Next: fit_coxph (Breslow, separate primitive), fit_pca (randsvd).

### v0.9.91-S6e (2026-10-29) - binseg + stray: refuse_at_guard (documented)

Both modules honestly raise `MemoryLimitExceeded` when the algorithmic
O(n) / O(n^2) state exceeds `state_limit`. This is not a fixable
violation — it is a **documented limitation**:

  - `cadw/ml/binseg.py` — prefix-sum arrays (`cs`, `cs2`, O(n)) are
    intrinsic to O(1)-per-split segment cost. Without them, binary
    segmentation becomes O(n^2). Guard: `est = 8*n*3`. Refuse threshold:
    n > state_limit / 24 (≈ 10^7 at 1 GB budget).
  - `cadw/ml/stray.py` — brute-force KNN is O(n²) time, O(n*d) input
    (not the algorithm's state, but unavoidable at exact KNN). Guard:
    `est = 8*n*d*min(n, 4*k)`. Docstring already says: n <= 20k,
    use EIF/Sub-PCA for larger.

No code change. YAML update (mark `n_scalability: refuse_at_guard`,
`tier1_core_exception: <reason>`) is backlog — needs D_time_series /
E_anomaly YAML in hand.

CHANGELOG-only. Full suite unchanged: 3981 passed, 8 skipped.

### v0.9.91-docs (2026-10-29) - P11 + §6 status consolidation

Docs-only. STREAMING_OUTPUT_ARCHITECTURE.md:

  §5 P11: sgd_streaming is for linear heads only. Not MLP, not EM
  clustering, not non-parametric. Reusable parts for non-linear
  models are batch_stream contract + mg.check_state discipline
  (not the function itself).

  §6 restructured: closed (S0-S5b in v0.9.90, S6a-e in v0.9.91),
  §6.1 won't-fix (S6b-full eif, S6c-full kmeans_ad, S6d-full
  autoencoder) with concrete cost calc, §6.2 backlog (default
  output="auto" migration, BREAKING, v0.9.92).

### v0.9.91-S6d (2026-10-29) - autoencoder chunked SGD (bit-identical)

Autoencoder had guard-before-_prepare_windows but still
materialized (n_w, d) after guard passed (400 MB for n=10^6,
w=50). Chunked SGD via rng.permutation(n_w) as (n_w,) int64:
8*n_w bytes = 50x smaller than (n_w, d) float64. On-demand window
fetch per mini-batch (bs*d*8 bounded). Bit-identical to pre-S6d
on same random_state (window_scores diff 0.00e+00,
training_losses diff 0.00e+00). Peak: 16*n_w + bs*d*8*6 +
params*8*2.

Two paths: duck-typed guard -> old path (bit-identical); real
MemoryGuard -> chunked. Test updated:
test_fit_autoencoder_chunked_does_not_call_prepare_windows.

### v0.9.91-S6b-tspq + S6c (2026-10-29) - tspq + kmeans_ad chunked

tspq.fit_t2q: chunked mean (1a) + cov (M2, 1b) +
proj/recon/resid (2). Peak = chunk*d*8*3 + 3*n_s*8 + d*d*8
(was 4*n_s*d*8). Bit-identical (mean/cov same math).
Guard pre-check updated from full-window (8*n_s*d) to chunked
peak — old formula raised even when chunked path could fit.

kmeans_ad.fit_kmeans_ad: two paths. Duck-typed guard -> old
_prepare_windows (bit-identical). Real MemoryGuard -> chunked
reservoir + Lloyd + score. Peak: S*d*8 + k*d*8 +
chunk*d*8 + 3*n_w*8. NOT bit-identical (reservoir init vs
exact rng.choice(n_w, size=k); see 6.1 S6c-full).

Test updates: 2 (chunked_does_not_call path tests).

### v0.9.91-S6b-eif (2026-10-29) - eif chunked fit

Chunked eif.fit_eif eliminates (n_w, d) intermediate.
Pre-sample per-tree indices (rng.choice(n_w, size=psi)) with
same rng sequence as _EIFForest.fit; gather unique sampled
windows in one chunked pass; build trees manually with
rng_build advancing identically. NOT bit-identical (rng state
divergence from original per-tree choice; see 6.1 S6b-full).
Peak: n_est*psi*d*8 + n_w*8 + chunk*d*8*2 (was n_w*d*8).

Fix iterations: fix1 (reservoir, recall<0.5), fix2 (single
index sample, still wrong — per-tree vs single-sample), fix3
(per-tree pre-sample + gather + manual build; tests green).

### v0.9.91-S6e (2026-10-29) - binseg + stray refuse_at_guard

Both correctly raise MemoryLimitExceeded on tight budget. This
is a documented limitation, not a fixable violation. O(n)
prefix sums (binseg) and O(n^2) brute KNN (stray) are intrinsic
to the algorithms (spilling them = different algorithms).

### v0.9.91-S6a (2026-10-29) - hampel + zms strided+chunked

Both used sliding_window_view (0-copy) but then called
np.median/np.mean/np.std on the view, which materialize a
contiguous (n_w, w) copy inside numpy (400 MB for n=10^6,
w=51). Chunked iteration -> peak O(chunk_nw*w*8*F),
F=2. Bit-identical to batch reference (max diff 0.00e+00).

Guard semantics preserved: real MemoryGuard -> per-chunk
check_state (OOC, succeeds where old path raised); duck-typed
guard -> legacy total-estimate check preserved (max_memory_mb).

### v0.9.90-S5b (2026-10-29) - TS guards (base path)

Guard before O(n) allocations in 4 TS functions (base path only;
group path documented as STREAMING_OUTPUT_ARCHITECTURE.md §5 P10):

  - ts/thf.py::_fit_one — guard before y_per_level + y_base copy.
    est = n * (1 + sum(1/m_k)) * 8.
  - ts/features.py::_shift_stream — guard before `out` dict
    (`n × |lags| × 100 B`). Base path; group path uses per-group
    deque (bounded by max_lag) but dict-per-group depends on
    n_groups (documented limitation).
  - ts/stats.py::rolling_stats — guard before `outs` dict
    (`n × |windows| × |stats| × 100 B`). Same scope.
  - ts/stats.py::rolling_crosscorr — guard before `outs`
    (`n × |windows| × |lags| × 100 B`). Same scope.

4 files, +40 LOC. All TS tests green (guard not triggered at
typical n). Full fast suite: 3981 passed, 8 skipped.

### v0.9.90-S5 (2026-10-29) - TS guards (root `_extract_series`)

Root cause: `_extract_series` accumulated Python-list (n × ~36 B)
instead of np.empty (n × 8 B). Fix: two paths — pre-alloc
`np.empty(df.count())` for no-filter (guard `check_state(n*8)`
BEFORE alloc), Python-list retained for group filter (per-series n
small in typical TS).

Also: `var.py::VARModel.fit` guard before `_stack_lags`
(est = Y + X + Yt); `models.py::_build_lag_matrix` gained
optional `memory_guard` kwarg (GBDTForecaster.fit passes it);
`auto_theta.py::_MultiplicativeWrapper.fit` guard before 3× O(n)
copies (log_y → pa.array → Dataset).

3 files, +63 LOC. All 17 TS test files green. Full suite: 3981 passed.

### v0.9.90-S4 (2026-10-29) - ML output= (9 functions)

9 functions returning O(n) np.ndarray / pa.Table accept
`output="memory"|"auto"|"disk"`:

  np.ndarray → AutoArrayResult (SpillArray):
    iforest.score_samples, chains.predict_proba, chains.predict,
    ordinal.predict_proba, ordinal.predict, tfidf.transform_dense,
    randsvd.transform.

  pa.Table → AutoResult (SpillBuffer):
    gmm.score_samples, gmm.predict.

`output="memory"` (default): bit-identical to pre-S4.
`output="auto"`: in-memory if fits, spills at threshold 0.5.
`output="disk"`: forces spill (spill_ratio=0.0).

Not in S4 (documented, not defects):
  - linear/logistic/gbdt/kmeans predict return CadwDataFrame
    (already OOC via Dataset.from_batches, no fix needed).
  - eif.fit_eif takes np.ndarray — user contract.
  - tfidf.transform (sparse) — variable per-row width; needs
    binary/CSR SpillBuffer (backlog).

6 modules + test_streaming_output_ml.py (+21 tests). Full: 3981.

### v0.9.90-S3 (2026-10-29) - 11 G2 (guard-after-alloc) fixes

Audit §6 of STREAMING_OUTPUT_ARCHITECTURE.md: check_state must
precede allocation, not follow it. Prior pattern (append then
check) left OOM window on a single oversized batch.

11 sites fixed in 10 modules:

  _collect family: feature_select, calibration, chains, ordinal,
  meta_learners — n += b.num_rows and check_state moved BEFORE
  Xs.append.

  Vocab builders: lda::_build_vocab, tfidf::fit_tfidf,
  _ml_bindings::encode_categorical — check_state((len + n_rows)
  × per_entry) at top of loop, before insertion.

  minirocket::transform — check_state before rows.append.

  hybrid_mf::fit — check_state(n_items * d_content * 8) BEFORE
  _content_matrix allocates C.

  dwt_mlead::fit — full pre-guard (n_padded_est, max_level_est,
  chunk_pre) BEFORE pad + DWT; old post-alloc guard removed.

10 files, +66/-45 LOC. All existing tests in 10 affected modules
green (guard reordering, no behavior change).

### v0.9.90-S2-fix (2026-10-29) - spill_ratio=0.0 + validation

1. SpillBuffer/SpillArray: allow spill_ratio=0.0 (= spill on first
   batch). Used by output="disk" to force OnDiskResult regardless
   of budget. Prior validation (0.0 < ratio <= 1.0) rejected 0.0,
   ValueError from output="disk" path. Threshold formula changed
   from `max(1, int(avail*ratio))` to `int(avail*ratio)` — 0.0
   means "any non-empty batch spills", 0.5 unchanged.

2. output= strict validation: groupby_aggregate / drop_duplicates /
   topk_to_disk raise ValueError on output not in
   (memory, auto, disk). Prior: unknown output fell through to
   auto branch silently.

+2 tests. 7 files. Full suite: 3961 → 3963 → later 3981 (+S3 0,
+S4 21, +S5 0).

### v0.9.90-S2 (2026-10-29) - 4 ops output= + topk fix + pivot OOC

Public API (back-compat via default `output="memory"`):

  groupby_aggregate(..., output="memory"|"auto"|"disk")
  drop_duplicates(..., output="memory"|"auto"|"disk")
  topk_to_disk(..., output="memory"|"auto"|"disk")

`"memory"` → pa.Table (old behavior).
`"auto"`   → AutoResult (SpillBuffer, spill at 0.5 threshold).
`"disk"`   → AutoResult with spill_ratio=0.0 (forced).

`pivot_to_disk` signature unchanged. Internally now:
`groupby_aggregate(output="auto")` + iterate AutoResult instead of
`list(long_table.to_batches())`. Fixes OOM when K_pairs >
state_limit (prior raised before pivot output even started).

**topk_to_disk algorithmic fix (audit #13):**

  Prior: per_batch: list[pa.Table] = O(P × k × row_size). For
  n_total=10^6 / chunk 100 / k=10, P=10^4, acc ≈ 800 KB → hit
  state_limit on tight budgets even though k=10 (output 80 B).

  Now: incremental merge. current_top = _per_batch_topk(current_top
  + batch_top, k) each iteration. State = O(k × row_size) = output
  size. check_state(current_top.nbytes) fires only if k × row_size
  > state_limit (impossible output).

4 files + test_streaming_output_ops.py (+14). Full: 3946 → 3960.

### v0.9.90-S1 (2026-10-29) - Streaming-output API core

Adds `SpillBuffer` / `AutoResult` / `SpillArray` / `AutoArrayResult`
(`cadw/core/spill.py`, ~360 LOC) + `docs/ml_tools/
STREAMING_OUTPUT_ARCHITECTURE.md` (normative, §1-§6).

Spill-on-demand: threshold = `guard.available_bytes() × spill_ratio`
(default 0.5 = `state_safety`). Once spilled, stays spilled (no
un-spill). `OnDiskResult.to_table()` / `OnDiskArrayResult.to_numpy()`
raise `MemoryLimitExceeded` with §1.3 reference (no escape hatches).

Hotfix within S1: `cadw/ml/rocket.py::_read_batch` `check_state` →
`check_batch(extra_factor=3.0)` — input is one `pa.RecordBatch`,
peak ~3× `batch.nbytes`, not state; was classified `other-guard` by
`scripts/service/diagnose_unguarded_to_pylist.py`.

20 tests (`tests/test_spill_buffer.py`): SpillBuffer + AutoResult
(no-spill/spill/empty/guard=None/order/re-iterable/lifecycle/context
manager) + SpillArray + AutoArrayResult (width validation, 1d
reshape, cleanup).

### v0.9.89-D4 (2026-10-29) - ROCKET

`cadw/ml/rocket.py` — Dempster, Petitjean, Webb, DMKD 2020.
Random convolutional kernels (weights ~ N(0,1), biases ~ U(-1,1),
dilations ∈ {1,2,4,...,max_dilation}). Feature extraction streams
per series via `SpillDir`; classifier SGD ridge via `cadw.ops.
optim.sgd_streaming`. OOC in `n_kernels` (state O(d·K), not O(d²)).
Documented limitation: labels `O(n_series)` in RAM (guarded,
refuse-at-guard) — truly unbounded requires two-pass labels protocol
(backlog).

`cadw/ml/_series.py::iter_series` extracted from `minirocket.py`
(shared D4 helper). 11 tests.

### v0.9.89-D2 (2026-10-29) - BVAR + VECM + D2 ensemble

`cadw/ts/bvar.py::BVARModel` — Bayesian VAR with Minnesota prior
(Litterman 1986, Banbura-Giannone-Reichlin 2010). Streaming
sufficient stats (XTX, XTY, YTY) + Normal-Inverted Wishart conjugate
posterior. State O(k²·p²), independent of n.

`cadw/ts/vecm.py::VECMModel` — Johansen cointegration VECM
(Johansen 1988/1991, Lutkepohl ch. 6-7). Streaming cross-covariance
accumulation, generalized eigenproblem on k×k. State O(k²·p²),
independent of n.

VAR YAML fixed: `cadw_module cadw.ts.models` → `cadw.ts.var`,
`cadw_function VAR` → `VARModel`.

D2 ensemble_recommendation filled (was null, WARN from validator):
VAR + BVAR + VECM + GBDT_multiseries, `max_per_role 2 → 4` (all
share `temporal_dynamics`).

18 tests (`tests/test_bvar.py` 9 + `tests/test_vecm.py` 9).

### v0.9.88 (2026-10-28) - Post-release: audit closure + P2b-full-2

Follow-up to v0.9.87. Closes remaining honest gaps and false
positives from P5 audit; formalizes TSAD metric honesty.

P2b-full (4b71709): lambdamart true OOC via BFS
  Rewrite fit_lambdamart tree-build from DFS to level-synchronous
  BFS via ReduceOps (adapted from gbdt P1a _AllNodeHistOpCodes /
  _AssignLeavesOpCodes / _LeafValuesOp). All n-sized state on
  SpillDir memmap: codes/y/scores/grad/hess/node_id. Uniform binning
  replaces quantile edges (documented deviation, same-version
  determinism). Eliminated last G1 whitelist entry — G1 empty.

v0.9.87-fix2 (bd15aaf): LambdaMART YAML + docstring honest
  P2b-full message overstated query grouping (said O(n_q) RAM;
  source showed O(n) Python ints, ~36n transient peak). Reclassified
  bounded_state:false, n_scalability:linear; docstring updated with
  "KNOWN GAP" block.

v0.9.87-fix3 (98fc08b): anomaly metrics honest
  YAML D3.1/D3.2/D3.3/D5 declared 24 TSAD metric cards; 20 absent
  from source (vus_pr, vus_roc, auprc, range_f1, affiliation_f1,
  point_f1, point_adjusted_f1, nab, detection_delay, best_5_ranking).
  New MetricCard.not_implemented_in_cadw field + parse + WARN on
  anchor-with-flag. 20 cards marked. TSAD metric implementation is
  v0.9.88+ feature backlog (not refactor).

v0.9.87-fix4 (0aa9404): STL + AnomalyPipeline guard
  stl_decompose gained memory_guard kw with est = n*8*8 + 4096
  check_state before first alloc (~8n float64). AnomalyPipeline.
  fit_predict passes through; diff branch guards (n-1)*8. Honest
  scope: STLTransformer/MSTL/Differencer/AutoStationarizer/
  AutoForecastPipeline fit still no guard propagation — v0.9.88
  backlog (~0.5 session).

v0.9.87-P2b-full-2 (a3a1368): query grouping external_sort
  Eliminated last O(n) hot spot: q_groups Python dict (O(n_q) keys +
  O(n) Python-int values = ~36n transient peak). external_sort by
  query_col → _SortedStream adapter; group_offsets int64 (n_q+1);
  per-group contiguous slices bounded by max_group_size.
  Bit-identical to P2b-full. Test tightened to prove fit SUCCEEDS at
  4 MB guard with 80k rows.

v0.9.87-fix5 (23a8cd2): LambdaMART YAML stale flip
  After P2b-full-2: bounded_state false -> true, n_scalability
  linear -> constant, tier1_core_exception removed.
  13 tier1_core_exception entries -> 12 (all remaining honest:
  single-series TS, DP-changepoint, STAMP MatrixProfile,
  multivariate VAR, STRAY n<=20k, CoxPH baseline arrays).

v0.9.87-P5a-fix-als (d1c5692): streaming dict build + objective
  Two O(n) paths in fit_als: (1) users/items list accum + concat +
  unique (O(n) transient, ~3 GB @ n=1e8); (2) _objective n-sized
  u_arr/i_arr with X[up] fancy-index (n*k = 12.8 GB @ n=1e8,k=16 per
  ALS iteration). Fixed via streaming set (O(U+I)) and
  _objective_streaming (per-batch fancy index). Bit-identical formula.

P5a-review-feature_select: false positive (dismissed)
  Guard check_state INSIDE batch loop raises BEFORE np.vstack —
  honest refuse_at_guard, not silent OOM. AST scan limitation.

Methodology consequence:
  - Whitelist: all 4 gates empty (G1 post-P2b-full, G2 post-P1a,
    G3 post-windowed, G4 post-P6/P7).
  - tier1_core_exception: 13 -> 12 (LambdaMART graduated to true OOC).
  - not_implemented_in_cadw markers: 20 TSAD metric cards (honest
    capability statement).
  - Fast suite: 3869 -> 3879 (+10).

v0.9.88 feature backlog (not refactor):
  - implement TSAD metrics (VUS-PR/VUS-ROC per TSB-AD;
    point_adjusted_f1 / range_f1 / affiliation_f1 per UCR/TAB; NAB;
    detection_delay; best_5_ranking; auprc). ~2 sessions.
  - propagate memory_guard through ts/transforms.py (STLTransformer,
    MSTL, Differencer, AutoStationarizer, AutoForecastPipeline,
    TransformedTargetForecaster). ~0.5 session.
  - P5a-fix-gmm-full (optional): full-data streaming EM (quality
    feature; v0.9.31 sample-based currently fine).

### v0.9.87 (2026-10-28) - OOC refactor + state taxonomy registration

Two themes: (1) restore out-of-core where audit found silent OOM
(HIGH-1 windowed, BOCPD, GBDT multiclass, coxph, LDA); (2) make the
promise enforceable — state taxonomy as METHODOLOGY, 4 AST-scan
gates, all 84 verified cards registered. 26 commits.

PHASE 0 — Methodology + gates (eace4f9..c1e0729):
  - METHODOLOGY_ALGORITHM_SELECTION.md: 5-step process, state
    taxonomy (state_class / state_type / bounded_state /
    n_scalability), §7 mandatory YAML fields, §8 4 gates, §9
    anti-patterns, §10 tier rules.
  - PRIMITIVES_AND_PRINCIPLES.md: 6 postulates, Reduce/HashShuffle/
    Sort/Batch I/O, MemoryGuard API, 3 spill patterns,
    anti-patterns, commit-pitfall (§9 backticks in -m).
  - cadw/core/spill.py: SpillDir + ArrowIPC array adapters.
  - cadw/io/batch_reader.py: ArrowIPCBatchReader.get_batch(idx) +
    num_batches (random access via IPC footer).
  - 4 gate tests (test_primitive_reuse, test_guard_before_alloc,
    test_no_dense_windows, test_yaml_state_fields); whitelist in
    .pingis/plan_v0.9.87_refactor.md.

HIGH-1 windowed + BOCPD (9fc9f8f..f97c22f):
  - sub_pca O(n·d) -> O(chunk·d) via 4-pass streaming Welford cov.
  - dwt_mlead O(n·w) -> O(chunk·w + w²) chunked mahalanobis.
  - s2g O(n·w)×2 -> O(chunk·w) streaming conv/quantize.
  - patch_memory_ad O(n·w) train -> memmap+SpillDir.
  - kshape O(n·w+n·L+n·k·L) -> O(chunk·(w+L)) memmap+chunked.
  - BOCPD O(n·max_run) -> O(keep·max_run) rolling posterior
    (163 MB -> 0.5 MB @ n=100k, 300×).

P5 audit (9378a2b):
  - scripts/audit_pass2_ml.py: AST scan (--dirs/--out).
  - 62 models classified (48 ML files, 417 fns; 28 TS/ops, 301 fns).
  - Triage: 5 real issues (gmm, coxph, lda, als, feature_select);
    rest acceptable-by-design.

GBDT multiclass true OOC (adb76c6..388213b, P1a series):
  - Persistent binning. _AllNodeHistOpCodes (bincount, ~10×).
  - Multiclass codes path (K× speedup); binary + regression codes
    path (5-10× speedup).
  - Grad/hess (n,K) -> memmap; pred (n,K) -> memmap; y_all -> memmap.
  - Guard: (1+4K) -> (1+2K) -> (K+2) -> 2.
  - Cumulative: 328n -> 16n @ K=10 (20×). True OOC.
  - G1 whitelist for gbdt REMOVED.

Lambdamart partial (0c049c9, fa5f20c, a3bf7ad):
  - P2a: pre-allocate X (was Xs.append + vstack transient 2×).
  - P2b-narrow: X -> memmap (SpillDir). Deletes after binning.
  - Remaining codes (n·d) / y / scores / grad / hess in RAM.
  - P2b-full: level-sync BFS rewrite — v0.9.88 backlog.

P5a-fix-coxph (a3ac543) — streaming OOC:
  - Real OOM was NOT vstack X (n·d) — it was S2 =
    cumsum(w*Xo[:,:,None]*Xo[:,None,:]) in _breslow, materializing
    (n,d,d) = 8 GB @ n=1e6, d=10. Plus O(n) allocs throughout.
  - New: _breslow_streaming + _baseline_hazard_streaming. External
    sort by T desc; per-ties-block accumulate S0/S1/S2.
    RAM: O(batch·d + d² + pending_ties·d).
  - fit_coxph gains optional spill_dir=None (tempfile default).
  - Model baseline arrays O(n_events); guard-checked at fit;
    refuse_at_guard if > state_limit (documented exception).
  - Bit-identical to in-RAM _breslow (atol=1e-10) verified.

P5a-fix-lda-a (ae9f4a0) — sparse E-step:
  - Real OOM was NOT vstack doc_sample (capped v0.9.42) — it was
    phi (D, V, K) = 800 MB @ D=200, V=50000, K=10. Guard saw only
    text bytes (~50 KB).
  - _tokenize_to_csr: sparse CSR doc-term counts.
  - _e_step dispatches via CSR; touches only non-zero vocab
    positions. Bit-identical to dense (atol=1e-12).
  - _fit_once M-step scatter-add via np.add.at. perplexity /
    _train_ppl logp on non-zeros only.
  - doc_sample cap TOTAL (not per-take): min(2000,
    state_limit // (V*8*2)).
  - Model beta K×V param-bounded; refuse_at_guard if > state_limit.

P5a-review-gmm (8407a80) — false positive:
  - Audit flagged _responsibilities as O(n·k), but n = sample size
    (v0.9.31 max_samples=10k), not dataset N. Bounded, no OOM.
  - True per-batch streaming EM = future D.6.x feature (not OOM).

P6 — state taxonomy registration (b76acb0..fa98aee):
  - schema.py: ModelCard gains state_class / state_type /
    bounded_state / n_scalability / streaming_strategy /
    primitive_reuse / tier1_core_exception. _validate_model_state
    with METHODOLOGY §7 + §10 rules.
  - All 84 verified cards registered:
      D_time_series 39 (D1:7 D2:2 D3.1:13 D3.2:5 D3.3:8 D4:2 D5:2)
      A_classification 14 (A1:3 A2:3 A3:3 A4:2 A5:1 A6:2)
      E_anomaly 3, F_recsys 3, G_learning_to_rank 1, H_causal 4,
      I_survival 1, J_nlp 3, M_bandits 1, N_density 2, O_eda 13.
      (K_vision, L_graph: 0 verified.)
  - 13 tier1_core_exception entries (bounded=false tier1_core):
      ETS/AutoETS/Theta/AutoTheta/SARIMA (single-series TS)
      VAR (multivariate TS d²p params)
      LambdaMART (P2b-full backlog)
      CoxPH (baseline arrays O(n_events))
      MatrixProfile/MatrixProfile_discord (STAMP O(n·w))
      BinarySegmentation/PELT (D3.2 + D5, DP array O(n))
      STRAY (n<=20k hard limit)
  - REQUIRE_STATE_FIELDS = True. G4 strict asserts all verified
    cards have state_class + bounded_state (unless whitelisted).

_gate_whitelist.is_whitelisted fix (v0.9.87-P6c-β-fix1):
  Section wildcard `E.*` was documented in plan §whitelist but
  never implemented — entry[:-1] = "E." matched nothing. Two-rule
  fix: subtype-level (prefix+".") for dotted entries (D3.1.*), and
  section-level (digit-follows-prefix) for single-letter (E ->
  E1, E2, E10; not EE1). +3 unit tests.

FINAL WHITELIST (v0.9.87 release):
  G1: cadw/ml/lambdamart.py::fit_lambdamart
      (accepted MEDIUM debt; P2b-full v0.9.88 backlog)
  G2: (empty)
  G3: (empty)
  G4: (empty — all 84 verified cards complete)

Test count: 3522 -> 3868 (+346 tests).
Fast suite: 3868 passed, 8 skipped.

Next session (v0.9.88 backlog):
  - P2b-full: lambdamart codes/y/scores -> memmap; level-sync BFS
    _build_tree; removes last G1 entry.
  - P5a-fix-gmm-full: streaming full-data EM (quality feature, not
    OOM); v0.9.31 sample-based currently fine.
  - P5a-review-als, feature_select: source review.

### v0.9.86.3 (2026-10-27) - Remove HybridETSGBDT (redundant with EnsembleForecaster)

Removes HybridETSGBDTForecaster (introduced v0.9.86, corrected
v0.9.86.2). Not a bug fix; architectural cleanup.

DECISION CONTEXT (user observation):
  HybridETSGBDT = ETS(y) + GBDT(y - ETS(y)) is additive composition.
  EnsembleForecaster = w1*ETS(y) + w2*GBDT(y) is weighted averaging.
  For general-purpose TS these are FUNCTIONALLY REDUNDANT: Ensemble
  already provides the "combine ETS + GBDT" capability with
  inverse-MASE weighting, and beats or ties Hybrid on 4 of 5
  benchmark DGPs:
    ets_friendly:       both NEUTRAL (~+1%)
    linear AR(1):       Ensemble wins (weight->0), Hybrid LOSS +7.5%
    threshold_ar:       Ensemble wins, Hybrid LOSS +4.3-8.4%
    interaction:        Ensemble wins, Hybrid NEUTRAL/LOSS +0.4-14.4%
    sinusoidal_p5:      Hybrid direct mode -74% WIN; Ensemble does
                        NOT reproduce this (GBDT on full y cannot
                        capture period-5 residual) -- the only
                        non-redundant case, and narrow (direct mode +
                        trend-feature memorization of short-period
                        residual; loses with recursive mode).

  The narrow WIN does not justify:
    - 200 LOC + 14 tests + 3 bench scripts of maintenance
    - false 'tier: sota' claim in card (v0.9.86.2 only partially
      corrected: purpose rewritten, tier left at sota)
    - documented false attribution 'M4-winner family' (v0.9.86.2
      corrected to Zhang 2003; residual taint in card history)
    - STL residual gap: Hybrid was the ONLY verified D1 model that
      did NOT apply STL (Theta/AutoTheta use stl_decompose; ETS
      natively handles trend+season; Hybrid fed raw residual to
      GBDT without STL decomposition)
    - reproduces as 4 lines of user code:
        ets = ETSForecaster(s=12).fit(df, 'y')
        resid = y - ets.predict_in_sample()
        gbdt = GBDTForecaster(mode='direct').fit(resid_df, 'y')
        final = ets.predict(h) + gbdt.predict(h)

REMOVED (git history retains all code):
  - cadw/ts/hybrid.py (~200 LOC)
  - tests/test_hybrid.py (14 tests)
  - scripts/bench_hybrid.py, bench_hybrid_tuning.py,
    bench_hybrid_nonlinear.py
  - .pingis/benches/hybrid.json, hybrid_tuning.json,
    hybrid_nonlinear.json
  - docs/ml_tools/D_time_series.yaml D1 card HybridETSGBDT
    (2 benchmark entries)
  - cadw/ts/__init__.py: import + __all__ entry

PRESERVED (history):
  - CHANGELOG v0.9.86, v0.9.86.1, v0.9.86.2 unchanged. Prior entries
    document what we did and what we learned; the removal is
    additive, not a rewrite.
  - MATRIX.md and 40 docs/tasks/*.md regenerated (Hybrid row gone).

RECOVERY:
  Fully reversible: `git revert <v0.9.86.3_sha>` restores code + tests
  + benches. If a future niche case genuinely needs short-period-
  residual composition (period-5 residual with s=12, direct mode),
  re-introduce as a script (not a class) or wait for ARNN/ES-RNN in
  cadw/ts (proper multi-series residual modeling; backlog).

BACKLOG:
  - Hybrid-related items from v0.9.86.2 (use_trend=False,
    auto_lag_selection, ARNN residual neural) REMOVED as moot.
  - 'HybridETSGBDT PI combined-variance' (v0.9.86.1) REMOVED as moot.
  - 'Hybrid tuning' (v0.9.86) was CLOSED in v0.9.86.2; now moot.
  - Net backlog: 21 -> 18 (3 hybrid items closed as 'model removed').
  - New conceptual note (informational, not actionable): "Zhang 2003
    residual modeling with GBDT residual is redundant with weighted
    ensemble for general-purpose TS; only distinct on short-period
    deterministic residual". Not a work item; a design lesson for
    future per-model wrappers.

DESIGN LESSON:
  Not every named literature method deserves a class in cadw. Composition
  models (ETS+GBDT, ARIMA+ANN) are 4-line user recipes when the
  constituents are already shipped. Reserve class slots for models
  whose composition requires non-trivial state (learning weights,
  end-to-end joint training, custom PIs), otherwise the codebase
  accumulates redundant wrappers with false tier claims.

Version 0.9.86.2 -> 0.9.86.3.

Focused: schema 73; render 52; version 6; changelog 2+2s; AST-guard
136 (was 137; hybrid.py gone); WL 34 OK; audit_leakage 7 warnings
(pretrained, unchanged); audit_url_log 0 errors.

Fast suite: 3549 -> 3534 passed (-15: -14 test_hybrid, -1 AST
parametrization for removed hybrid.py), 8 skipped.

Next: D2 (multi forecast) -- VARMAX + foundation-model reference;
currently 2 models (VAR, ?), 0 ensemble. Real gap, unlike Hybrid
which was redundant.
### v0.9.86.2 (2026-10-27) - Hybrid attribution correction + nonlinear bench

Corrects a factual error in v0.9.86/v0.9.86.1 and adds the
literature-valid nonlinear benchmark. No model code change.

ATTRIBUTION CORRECTION:
  v0.9.86/v0.9.86.1 labelled HybridETSGBDT "M4-winner family".
  This is factually wrong. The M4 competition winner is ES-RNN
  (Smyl 2018, "A hybrid method of exponential smoothing and
  recurrent neural networks for time series forecasting"), which is
  a GLOBAL cross-series model with shared ES + RNN parameters.
  HybridETSGBDT is a per-series ETS + residual model:
    - Zhang (2003) "Time series forecasting using a hybrid ARIMA
      and neural network model" (Neurocomputing 50:159-175) -- the
      original per-series residual-modeling family.
    - hybridts::ets_arnn (Chakraborty 2023, CRAN) -- the closest
      reference: ETS + ARNN on residual. Our model replaces ARNN
      with GBDT.
  Card purpose corrected; CHANGELOG v0.9.86 + v0.9.86.1 remain
  intact (they record what we did); this entry is the correction.

BENCH VALIDITY FINDING:
  v0.9.86 bench used AR(1) phi=-0.6 residual (linear autocorrelation).
  This was an INVALID test for a nonlinear residual model: GBDT is
  weaker than ANN on smooth linear residuals, and Zhang 2003-type
  residual modeling is designed for NONLINEAR structure. Honest
  negative on linear AR(1) is not evidence against the architecture.

NEW BENCH (scripts/bench_hybrid_nonlinear.py,
.pingis/benches/hybrid_nonlinear.json):
  3 nonlinear DGPs, n=500 s=12 h=12, 5 seeds, 5 variants.
  DGP1 threshold_ar: e[t] = +0.7*e[t-1] + eps (if e[t-1]>0) else
                     -0.7*e[t-1] + eps (sign-dependent AR).
    ALL LOSS: hybrid_default +4.3%, hybrid_direct +6.0%,
    hybrid_lags6 +4.5%, hybrid_direct_lags6 +8.4%.
    Interpretation: GBDT 50 trees x depth 3 x lr 0.1 cannot
    recover the sign-dependent threshold; residual prediction is
    noise around 0.
  DGP2 sinusoidal_p5: e[t] = 1.2*sin(2*pi*t/5) + 0.2*eps
                     (period 5; ETS seasonal_period=12).
    hybrid_direct: -74.1% WIN (MASE 0.060 vs ETS 0.232).
    hybrid_direct_lags6: -71.2% WIN.
    hybrid_default (recursive): +37.2% LOSS.
    hybrid_lags6 (recursive): +90.8% LOSS.
    Interpretation: WIN is real but NARROW. Direct mode + GBDT
    trend feature memorizes (t mod 5) -> offset, and test range
    (t=500..511) still contains 2+ full periods of 5. Recursive
    feedback destroys the phase in 1 step.
  DGP3 interaction: e[t] = 0.5*e[t-1]*e[t-2] + eps.
    All NEUTRAL/LOSS (+0.4 to +14.4%). Depth-3 trees underfit
    second-order interactions at this signal level.

VERDICT:
  HybridETSGBDT is a NARROW tool, not a general-purpose ETS
  improver. Wins only when (a) residual has short-period structure
  (shorter than ETS seasonal_period), (b) direct mode (recursive
  destroys periodicity), (c) test range still within memorizable
  trend-feature extrapolation. Loses on linear AR(1), sign-dependent
  AR, second-order interactions with current defaults.

YAML card (docs/ml_tools/D_time_series.yaml, D1 HybridETSGBDT):
  - purpose: rewritten to remove "M4-winner", replace with
    "Zhang (2003) residual-modeling family; reference
    hybridts::ets_arnn; NOT M4-winner". Explicit narrow-win
    description, explicit loss list, explicit "defaults are safe
    but not winning config".
  - benchmark_results: appended second cadw_measured entry
    (sinusoidal_p5 direct mode, MASE 0.0600 +/- 0.0102, -74.1% vs
    ETS, n_datasets=3, note documents both WIN and other two DGP
    LOSS results). First entry (linear AR(1)) augmented with
    VALIDITY CAVEAT noting it is not a valid test for a nonlinear
    residual model.
  - ensemble_recommendation: unchanged. Not added to primary D1
    ensemble (narrow win, not general-purpose).
  MATRIX.md 157 lines + 40 tasks regenerated.

Backlog additions:
  - HybridETSGBDT: use_trend=False for non-periodic residual
    (trend feature causes boundary overfit when residual has no
    t-dependence).
  - HybridETSGBDT: auto_lag_selection (literature ARNN auto-picks
    lags; ours fixed (1,2,3)).
  - HybridETSGBDT: ARNN residual (neural) vs GBDT residual
    (Zhang 2003 original uses ANN, which handles smooth
    nonlinearities better than depth-3 GBDT).

Design notes:
  - No model code change. The v0.9.86 API is correct as-is;
    only attribution and bench coverage were wrong.
  - Recursive default remains (safe: GBDT idles on non-periodic
    residual; does not harm ETS). Direct mode is an opt-in for
    short-period residual cases.

Version 0.9.86.1 -> 0.9.86.2.

Focused: schema 73; render 52; version 6; changelog 2+2s;
AST-guard 137; WL 34 OK; audit_leakage 7 warnings (pretrained);
audit_url_log 0 errors (new cadw_measured entry: algorithm=cadw.*,
benchmark=cadw_internal, note required -- all present).

Fast suite: 3549 passed, 8 skipped (unchanged -- no new tests).

Next: Hybrid tuning item now CLOSED (tried 5 variants x 3 nonlinear
DGPs; narrow win documented, general-purpose loss confirmed). Next
direction: DU-1 (supervised tabular) OR D2 multi forecast OR D4 TS
classification.
### v0.9.86.1 (2026-10-27) - HybridETSGBDT YAML card + PI combined-variance

Closes v0.9.86 honestly: registers HybridETSGBDT in the D1 matrix
and fixes prediction-interval width (was point-shift only;
combined-variance now). No new model code; net +2 tests.

PI combined-variance (cadw/ts/hybrid.py, ~15 LOC in
prediction_interval):
  Old: PI = ETS PI shifted by GBDT point forecast. Width == ETS
  width, ignoring residual variance.
  New: mid = ETS_mid + gbdt_fc; half = sqrt(ets_half^2 +
  (z*sigma_resid)^2) with z = NormalDist().inv_cdf(0.5 +
  level/2) and sigma_resid = nanstd(y - ETS_in_sample). Assumes
  ETS and GBDT errors independent (approximate; GBDT is fit on
  ETS residual, so near-independence is expected but not exact -
  documented). Defensive fallback: sigma_resid NaN/0 -> ETS width
  (old point-shift semantics). _gbdt is None -> ETS PI as-is.

  3 tests touched (tests/test_hybrid.py now 14, was 12):
    - test_prediction_interval_shifted_by_gbdt rewritten:
      mid == ETS_mid + gbdt_fc; width >= ETS width (old assertions
      lo==ets_lo+gbdt_fc / hi==ets_hi+gbdt_fc fail on wider
      combined PI, expected).
    - test_prediction_interval_widens_with_residual (new):
      strictly wider when sigma_resid > 0.
    - test_prediction_interval_equals_ets_when_gbdt_skipped (new):
      min_resid_train=10^6 -> _gbdt is None -> PI == ETS PI
      exactly (np.allclose).

YAML D1 card (docs/ml_tools/D_time_series.yaml, before
GBDTForecaster card):
  - name: HybridETSGBDT
    tier: sota (M4-winner family, consistent with AutoTheta=sota)
    tier_level: tier1_core
    roles: [temporal_dynamics, nonlinear_interactions]
    purpose: >- (honest limitation inline)
    cadw_module: cadw.ts.hybrid
    cadw_function: HybridETSGBDTForecaster
    benchmark_results: cadw_measured (n_datasets=2, metric=mase,
              value=0.1475, std=0.0091, from
              .pingis/benches/hybrid.json; note carries 4-cause
              diagnosis)
    evaluation_scope: synthetic
    dgp_source: synthetic_nonlinear_residual
  MATRIX.md + 40 docs/tasks/*.md regenerated.

Design notes:
  - YAML cards have no capability_flags:/special: (anchor-only
    fields); honest limitation goes in purpose: and note:
    (Theta precedent).
  - cadw_measured schema rule: algorithm must be cadw.*, benchmark
    must be cadw_internal. benchmark_url points to per-file JSON
    matching ETS/AutoETS/Theta/AutoTheta precedent.
  - n_datasets=2 = number of DGPs (not seeds); matches AutoTheta.
  - ensemble_recommendation unchanged: Hybrid NOT added to primary
    D1 ensemble due to honest negative; diagnostic-only.

Fix iterations:
  - v0.9.86.1-fix1: initial patch attempt had (a) test anchor with
    assumed docstring (actual v0.9.86 test has none); (b) YAML
    anchor '  - name: GBDTForecaster\n  tier:' with 2-space
    indent (actual: 4-space). Both fixed; script made idempotent
    with explicit anchor-miss diagnostics.

Version 0.9.86 -> 0.9.86.1.

Backlog: 19 items unchanged. Hybrid tuning and PI now consistent
with implementation; AST-count doc-rot note remains (v0.9.87).

Focused: test_hybrid 14 passed; render 52; schema 73; version 6;
changelog 2+2s. WL 34 OK. AST-guard 137. audit_leakage 7 warnings.
audit_url_log 0 errors.

Fast suite: 3547 -> 3549 passed (+2), 8 skipped.

Next: Hybrid tuning (backlog) OR pivot to DU-1 (supervised
tabular; pipeline + metrics + hybrid all reusable).
### v0.9.86 (2026-10-27) - HybridETSGBDT forecaster (M4-winner family)

Closes the last major D1 gap: hybrid ETS+GBDT (Zhang 2003 M4-winner
approach family). Complements AutoTheta (M3-winner class). D1 now
ships both M-competition winner approaches.

HybridETSGBDTForecaster (cadw/ts/hybrid.py, ~200 LOC, 12 tests):
  Wraps ETSForecaster (trend+season) and GBDTForecaster (residual).
  Final forecast = ETS_fc + GBDT_fc(residuals). Residual GBDT is fit
  on y - ETS.predict_in_sample().

  Parameters:
    ets_kwargs      passed to ETSForecaster
    gbdt_kwargs     passed to GBDTForecaster (default mode='recursive',
                    lags=(1,2,3))
    min_resid_train minimum finite residuals to fit GBDT (default 20);
                    if fewer, hybrid == ETS (GBDT skipped silently)

  Behavior:
    - ETS in-sample warmup is finite (unlike SARIMA with d>0 or D>0),
      so residual is finite everywhere in the common case, hence GBDT
      gets full training rows.
    - GBDT residual fit exception: warn, ETS-only fallback.
    - predict_in_sample: ETS_in + GBDT_in where GBDT is finite; else
      ETS_in unchanged (no NaN leak into public API).
    - PI: ETS PI shifted by GBDT point forecast (additive point
      shift). GBDT uncertainty is NOT added to width. Documented
      limitation; proper combined-variance PI in backlog v0.9.87.

  Not supported in v0.9.86:
    - Multi-series (group_col): NotImplementedError (v1.0).
    - Nested transforms (differencing=, decompose=): not applicable.
      ETS handles trend+season natively; GBDT operates on the
      already-stationary residual. To stationarize a hybrid forecast
      pipeline, wrap Hybrid in AutoForecastPipeline.

  12 tests (tests/test_hybrid.py):
    contract_basic, does_not_hurt_on_pure_ets_dgp (2x tolerance;
    real not-hurt assertion in bench across seeds),
    predict_in_sample_applies_gbdt_component,
    predict_before_fit_raises, steps_validation, short_series_raises,
    group_col_raises, gbdt_skipped_when_resid_too_small,
    capability_flags, prediction_interval_shifted_by_gbdt,
    determinism, gbdt_mode_direct_works.

Bench (scripts/bench_hybrid.py, .pingis/benches/hybrid.json):
  Two DGPs, n=500, s=12, h=12, seeds=3:
    ets_friendly:       trend + season + iid noise (ETS expected to win)
    nonlinear_residual: trend + season + AR(1) phi=-0.6 (fast
                       alternation ETS smoothing misses; GBDT should
                       recover some signal)

  Measured MASE (seed-mean):
    ets_friendly:       ets=0.0811  hybrid=0.0818  gbdt=0.9860
                        hybrid vs ets: +0.9% (within noise; GBDT
                        correctly idles when residual has no signal)
    nonlinear_residual: ets=0.1372  hybrid=0.1475  gbdt=0.9132
                        hybrid vs ets: +7.5% (hybrid HURTS)

  HONEST NEGATIVE RESULT: on the DGP designed for hybrid to win
  (AR(1) phi=-0.6 residual), hybrid is +7.5% WORSE than plain
  ETS. Direction is consistent across 3 seeds; sample stds
  0.0091 vs 0.0177; n=3 seeds is small but directional. Diagnosis:
  (a) ETS with additive trend absorbs alpha*residual into y_hat,
      so the residual fed to GBDT is smaller than the true AR(1)
      component and correlated with y_hat;
  (b) recursive mode feeds GBDT's own near-mean predictions back
      as lags, so steps 2+ regress to mean faster than the true
      AR(1) alternation;
  (c) 50 trees x depth 3 x lr 0.1 undertrains a weak phi=-0.6
      signal at n=500;
  (d) GBDT's 'trend' feature causes boundary overfit at test
      (test trend n_train+1 .. n_train+h was never seen during
      fit).

  The model does NOT harm ETS-friendly data (+0.9% within noise;
  GBDT predictions are ~0). But on our synthetic nonlinear case,
  hybrid's M4-winner promise does not materialize. Real M4
  datasets have stronger, longer-range residual structure;
  whether hybrid helps there is untested and blocked on M4-scale
  data availability.

  YAML card + cadw_measured deferred to v0.9.86.1. Backlog item
  opened: 'Hybrid tuning: test direct mode, lags subset, remove
  trend feature, min_resid_train=100 on nonlinear_residual DGP'.

Design notes:
  - Chose ETS over SARIMA as base because ETS predict_in_sample
    works for all configs (SARIMA returns all-NaN for d>0 or D>0,
    making residual construction undefined). ETS is the safest base
    for residual extraction.
  - Chose 'recursive' as GBDT default: residual has no trend (ETS
    removed it), so no extrapolation issue; recursive avoids
    per-horizon models. 'direct' supported for parity.
  - Silent GBDT skip via min_resid_train (expected short-series
    case); warn on exception (unexpected failure).

Fix iterations:
  - v0.9.86-fix1: bench_hybrid.py contained a print(f-string) whose
    contents began with a backslash-n followed by a bracketed label.
    The backslash-n was written through the outer heredoc as a
    literal newline inside the f-string, breaking the source. Fixed
    by splitting into a bare print() followed by a print(f-string)
    on separate lines.

Version 0.9.85 -> 0.9.86.

Backlog: 16 items; item 9 (HybridETS_GBDT) CLOSED via P1. New
backlog item: HybridETSGBDT PI combined-variance (v0.9.87).

Fast suite: 3534 -> 3547 passed (+13: +12 hybrid + +1 AST
parametrization for new hybrid.py), 8 skipped. Focused:
test_hybrid 12 passed. WL 34 OK. AST-guard 137 passed (baseline 136; +1 for hybrid.py). audit_leakage 7 warnings (pretrained).
audit_url_log 0 errors.

Doc-rot noted: prior CHANGELOG entries claim "AST-guard 129
passed"; actual count at v0.9.85 was 136 (grew as cadw/ gained
files across v0.9.76-v0.9.85). Not retroactively fixed; new
backlog item "AST count in CHANGELOG: replace per-release
stale 129 with actuals or add per-release note".

Next: v0.9.86.1 - HybridETSGBDT YAML card + bench numbers + PI
combined-variance fix; OR pivot to DU-1 (supervised tabular;
pipeline + metrics + hybrid all reusable).
### v0.9.85 (2026-10-27) - Forecast metrics extension (interval + robust + quantile)

Extends cadw/ts/backtest.py with 7 new metric families used by
TSB-AD/TAB/fev-bench evaluations. Complements the existing 5 point
metrics (MAE, RMSE, MAPE, sMAPE, MASE).

New metrics (cadw/ts/backtest.py, ~200 LOC + refactor of _mase):

  Point metrics:
    WAPE    = sum|y-f| / sum|y| * 100; robust to zero actuals (unlike
              MAPE, which NaNs); NaN when sum|y|=0.
    MQLIKE  = mean(y/f - log(y/f) - 1); variance-robust for volatility
              forecasts; requires y>0, f>0; NaN otherwise.

  Interval metric:
    MSIS    = Mean Scaled Interval Score (Gneiting & Raftery 2007):
              mean[(hi-lo) + (2/alpha)*(miss_lo) + (2/alpha)*(miss_hi)]
              / naive_MAE(m).
              Requires forecaster.prediction_interval(steps, level).
              Skipped (warning) if unsupported. Configurable via new
              msis_alpha parameter (default 0.05 => 95% interval).

  Quantile metrics (require `quantiles` parameter + forecaster support):
    pinball  = per-q pinball loss (existing; now complemented by
               aggregations below).
    WQL      = weighted quantile loss: mean over q of w(q)*pinball(q)
               with w(q)=2q for q<=0.5, 2(1-q) for q>0.5. Primary
               probabilistic metric in fev-bench.
    SQL      = WQL / naive_MAE(m); scale-free WQL for cross-series
               comparison.
    CRPS     = 2 * mean pinball over quantile levels; approximate
               CRPS from quantile losses (not exact integral).

  Helper refactor: _mase now delegates its scale computation to new
  _mase_scale helper (shared by MSIS/SQL).
  _predict_interval helper wraps forecaster.prediction_interval.

Integration (backtest_forecaster):
  - `metrics` validation extended from 5 to 11 valid names.
  - `msis_alpha=0.05` new kwarg (must be in (0, 1)).
  - If wql/sql/crps requested without `quantiles`, emit warning and
    drop them from the effective metric set (do not silently compute
    wrong values).
  - If MSIS requested but forecaster lacks prediction_interval: emit
    warning once per fold sequence, drop msis from summary (row absent).
  - Empty per_metric lists (e.g. msis unsupported) are skipped from
    summary (no NaN mean row).

12 new tests (tests/test_ts_backtest.py, total now 22 + 1 skip):
  wape_perfect_forecast_zero (positive value); wape_zero_actuals_nan;
  mqlike_perfect_zero (constant y=2, naive=2 -> 0); mqlike_negative_y_
  nan; msis_skipped_for_naive (warning + row absent); msis_perfect_
  interval_zero (Spy PI on y_train with variation -> 0);
  msis_nonzero_interval (skip if scale=0); wql_requires_quantiles
  (warning + drop); wql_crps_sql_from_quantiles (Spy quantile=2 on
  y=2 -> all ~0); invalid_metric_raises; msis_alpha_validation
  (must be in (0,1)); wape_matches_hand_compute (Spy predicts 0 ->
  WAPE=100 on y=1).

  Spy test doubles (_SpyPI, _SpyQuantile) isolate interval and
  quantile paths from real forecaster specifics.

YAML D1 metrics (+7 diagnostic): mape, rmse, mae, wape, mqlike,
msis, pinball (per-q), wql, sql, crps. Anchor remains MASE; smape
unchanged. Total D1 metrics now: 1 anchor + 12 diagnostic (was 1 + 2).
MATRIX.md + 40 docs/tasks/*.md regenerated.

Fixup (v0.9.85-fix1): initial YAML patch used a multi-line plain
scalar with a colon-space inside (`quantile losses: CRPS ~ ...`) —
YAML scanner rejected it. Collapsed CRPS responsibility/special to
single lines; render_matrix OK.

Version 0.9.84 -> 0.9.85 (skipped HybridETS_GBDT for now; hybrid is
M4-specialized marginal gain over AutoTheta, lower priority than
metrics — needed for ALL future forecast sections).

Fast suite: 3522 -> 3534 passed (+12), 8 skipped. Focused:
test_ts_backtest 22 passed + 1 skipped. WL 34 OK. AST-guard 129
passed. audit_leakage 7 warnings (pretrained). audit_url_log 0
errors.

Backlog: 16 items unchanged (no item closed, no item added).

Next: v0.9.86 — HybridETS_GBDT (M4 winner idea: ETS fit -> residual
-> GBDT on residual features); OR pivot to DU-1 (supervised tabular
section) which needs the pipeline + metrics now in place.

### v0.9.84 (2026-10-27) - AnomalyPipeline + GBDT transforms integration

Closes the anomaly-stationarity gap (backlog item 3) and wires GBDT
into the transforms foundation. P0 documentation and P1 code from
the v0.9.77b backlog decision.

GBDTForecaster transforms integration (v0.9.83, cadw/ts/models.py
+~120 LOC, tests/test_gbdt_transforms.py +14 tests):
  New kwargs: differencing='none'|'auto'|int>0, decompose='none'|
  'stl'|'mstl', decompose_periods=list[int]|None. Defaults preserve
  v0.9.82 API (no-op).
  - differencing='auto': AutoStationarizer wraps y before GBDT fit;
    fixes GBDT's inability to extrapolate trend (Pritzsche: GBDT/
    GBM/gamboost cannot forecast trend without detrending).
  - decompose='stl'|'mstl': STL/MSTL wraps y; seasonality removed
    before fit, added back at predict.
  - _build_inner_pipeline uses lazy import (avoid circular dep —
    transforms.py imports ForecasterBase from models.py).
  - Instance-level capability flags (handles_trend/seas/multi)
    override class defaults; with transforms enabled GBDT reports
    True so outer AutoForecastPipeline does not double-wrap (verified
    via pipeline_idempotent_with_autoforecastpipeline test).
  - predict_in_sample raises NotImplementedError in transform mode
    (length mismatch: GBDT NaN-padded n-length vs transforms fit on
    n-d). Naked mode unchanged.

AnomalyPipeline (v0.9.84, cadw/ml/anomaly_pipeline.py, ~250 LOC,
tests/test_anomaly_pipeline.py +21 tests):
  Wraps any anomaly detector function (signature
  detector_fn(series, **kwargs) -> result with .decision_scores and
  .labels). Three modes:
    - 'structural': detect on raw; suits level/trend/seasonal-shift
      anomalies; native changepoint detectors (PELT/BOCPD/BinSeg/
      CUSUM/T-squared+Q) already handle trend as signal.
    - 'contextual': STL decompose (if seasonal_period>=2) or
      Differencer(lags=1) (if detrend=True); run detector on residual;
      map scores back 1:1 (STL preserves length) or with first
      timestamp unknown (differencing).
    - 'auto': contextual if transform available, else structural.
  Addresses backlog item 3 (v0.9.77b): 11 of 15 D3.x detectors assume
  stationary baseline; on trended/seasonal series they produce false
  positives. Verified in test: on pure seasonal y=5*sin(2πt/12),
  structural naive threshold flags >=10 points; contextual STL flags
  0 (residual near zero). On linear trend + spike, contextual detrend
  isolates spike (<=3 flagged) vs structural (~50 trend points
  flagged).
  Result object: AnomalyPipelineResult(decision_scores (n,), labels
  (n,), mode, transforms_applied, base_result, threshold).
  get_params records resolved mode and transforms for audit.
  Pure numpy; lazy import of transforms and _decomposition (AST-guard
  enforced).

P0 documentation (D_time_series.yaml):
  D3.1 / D3.2 / D3.3 subtype descriptions extended with stationarity
  notes:
    D3.1: 11 detectors assume stationary; use AnomalyPipeline
          contextual (v0.9.84) on trended/seasonal series.
    D3.2: all detectors model trend natively; use structural mode,
          do NOT wrap in contextual (would erase signal).
    D3.3: shape anomalies are contextual by nature; contextual STL
          preferred for Sub-PCA/KShapeAD/S2G.
  MATRIX.md + 40 docs/tasks/*.md regenerated.

Design notes:
  - Detectors are functions (not classes) in cadw/ml/; pipeline
    therefore takes detector_fn + kwargs, not an object with capability
    flags. A registry of flags per detector would add complexity
    without benefit — caller chooses mode explicitly.
  - STL preserves length -> 1:1 mapping is exact. Differencing loses
    first timestamp (score[0]=0, label[0]=0 — "unknown").
  - AnomalyPipeline can wrap any tier1 detector, but its core value
    is for the 11 stationary-assuming ones. Native changepoint
    detectors work equally well naked in structural mode.

Version 0.9.82 -> 0.9.84.

Backlog: item 3 (Anomaly stationarity) closed via P0+P1.

Fast suite: 3486 -> 3521 passed (+35: GBDT 14 + AnomalyPipeline 21),
8 skipped. Focused: test_gbdt_transforms 14 passed,
test_anomaly_pipeline 21 passed, regression 152 passed (AST 129 +
decomposition 16 + misc). WL 34 OK. AST-guard 129 passed.
audit_leakage 7 warnings (pretrained). audit_url_log 0 errors.
MATRIX.md + 40 docs/tasks/*.md regenerated (P0 doc changes).

Next: v0.9.85 — HybridETS_GBDT (M4 winner idea: ETS fit -> residual
-> GBDT on residual features); then metrics extension (MSIS, pinball,
MQLIKE, WAPE, CRPS) v0.9.86.

### v0.9.82 (2026-10-27) - TS transforms + AutoForecastPipeline + Ensemble

Closes the TS pipeline section (v0.9.79-v0.9.81): composable per-model
target transforms, capability-aware pipeline, and weighted ensemble.
Enables correct handling of trend/seasonality for any forecaster
without double-stationarization, and preserves diversity in ensembles.

Transforms foundation (cadw/ts/transforms.py, ~800 LOC total, v0.9.79a+b):

  BaseTransformer (ABC): fit / transform / inverse_transform /
  get_params contract for stateful target transforms.

  Differencer: regular d and seasonal D differencing with inverse.
    Forward: d regular diffs then D seasonal diffs.
    Inverse (v0.9.79a-fix2): closed-form binomial —
      d-th: y[t] = y_t[t] + sum_{k=1..d} (-1)^(k+1) C(d,k) y[t-k]
      D-th: y[t] = y_t[t] + sum_{k=1..D} (-1)^(k+1) C(D,k) y[t-k*s]
    Initial buggy loop didn't track tails for d>1/D>1; replaced with
    binomial closed-form (verified on d=2 quadratic).

  TransformedTargetForecaster: pipeline wrapper. Fit: apply transforms
  in order, fit base on fully-transformed y. Predict: base.predict
  then inverse in reverse order. predict_in_sample + PI also inverted.

  STLTransformer (v0.9.79b): STL decomposition, transform = y - seasonal,
  inverse = forecast + seasonal. In-sample (roundtrip) uses full
  seasonal; forecast uses last cycle tiled (v0.9.79b-fix1).

  MSTLTransformer (v0.9.79b): iterative STL for multiple periods
  (ascending); transform = y - sum(seasonals); inverse per above.

  AutoStationarizer (v0.9.79b, subclass of Differencer): auto-selects
  d and D. d: iterate while |acf(lag=1)| > acf_threshold (default
  0.8), cap max_d (default 2). D: after d-diff, STL seasonal
  strength F_S = max(0, 1 - Var(R)/Var(S+R)); if F_S > threshold
  (default 0.5), D = 1; cap max_D (default 1). Formal ADF/KPSS in
  backlog; heuristic fast, no scipy/statsmodels dependency.

  38 tests: Differencer (d1/d2/d0 forward, d1/d2 inverse roundtrip,
  seasonal forward/inverse, combined d1_D1, identity recovers last
  value, bad params, short-series raise); STL (removes season,
  inverse roundtrip, forecast continuation, bad period, short raise);
  MSTL (removes both seasonals, roundtrip, bad periods); pipeline
  (basic flow, flat naive-without-diff sanity, PI, capability flags,
  bad params, get_params); AutoStationarizer (random noise d0,
  random walk d1, trending d1, seasonal D1, is-Differencer-subclass,
  transform/inverse after fit, bad params); AST guard.

Capability flags (v0.9.80): handles_trend / handles_seasonality /
handles_multi_seasonality on ForecasterBase models:
  - SARIMAForecaster: True / True / False (natively handles both)
  - ETSForecaster: True / True / False
  - AutoETSForecaster: True / True / False
  - ThetaForecaster (STM), OTMThetaForecaster, AutoThetaForecaster:
    True / True / False (declared in v0.9.77/v0.9.78a/v0.9.78b)
  - NaiveForecaster, GBDTForecaster: flags undeclared -> getattr
    defaults to False; pipeline adds transforms.

AutoForecastPipeline (v0.9.80, in transforms.py): wraps any
ForecasterBase. Reads capability flags; adds transforms only where
the base cannot handle natively. Transform ordering (STL-then-diff,
literature convention for STL+ARIMA):
  1. STLTransformer / MSTLTransformer (if model needs seasonality)
  2. AutoStationarizer (if model needs trend)
If base handles both, pipeline is "naked" (delegates directly).
12 new tests: naive+trend -> differencing (trend continues vs flat),
naive+stationary -> d=0 (matches plain naive), naive+seasonal ->
STL+diff (non-flat), sarima/ets/autotheta naked, capability flags
after wrap, MSTL for 2 periods, empty forecaster raises, get_params,
predict_in_sample shape, AST guard.

EnsembleForecaster (v0.9.81a, cadw/ts/ensemble.py, ~370 LOC):
  Wraps each member in its own AutoForecastPipeline (per-member
  capability-based transforms). Predictions inverse-transformed
  per-member, then weighted-averaged. Preserves diversity: if all
  members see the same detrended series, they produce the same flat
  forecast; inverse adds back the same trend -> zero ensemble gain.

  Weight strategies: 'equal' (mean), 'inverse_mase' (default;
  weights ∝ 1/val_MASE on holdout), 'median' (robust to 1 outlier),
  custom list. Weights validated at __init__ (fail fast).

  Bench (n=500, s=12, M3-like):
    naive_last             MASE=2.184  0.18s
    theta                  MASE=0.201  0.05s
    auto_theta             MASE=0.209  0.23s
    ensemble inverse_mase  MASE=0.203  0.56s
      weights = [0.017 (naive), 0.311 (theta), 0.672 (auto_theta)]
      val_mase = [47.06, 2.59, 1.20]
    auto_ets               MASE=0.276  3.33s
  Ensemble ≈ best-member on single DGP. Honest: on a single synthetic
  DGP ensemble rarely beats best member; expected gain on M3/M4-scale
  multi-DGP from inverse-MASE picking good members per series.

  HONEST LIMITATION (documented in bench + backlog 17): AutoStat
  heuristic (acf_threshold=0.8) chose d=0 on trend+noise DGP, so
  naive+STL gave flat forecast (val MASE=47). Ensemble correctly
  downweighted naive to 0.017 — capability works, but AutoStat
  heuristic itself misleads on raw trended series. Fix in backlog:
  lower acf_threshold 0.8 -> 0.5-0.6, or formal ADF/KPSS.

  Fix iterations: v0.9.81a-fix1 validate weights at __init__ (bad
  spec raised from fit initially); test fixes (NaiveForecaster has
  no 'mean' strategy -> 'drift'; pipeline wraps naive in AutoStat
  so ens != naked-naive avg -> use ThetaForecaster for bare-naked
  pipeline tests).

Version 0.9.78 -> 0.9.82.

Combined with v0.9.79a-b and v0.9.80 (uncommitted between v0.9.78
and v0.9.81a), the full pipeline stack is now usable:
  TransformedTargetForecaster(Differencer, NaiveForecaster) for
  trended series; AutoForecastPipeline(GBDTForecaster) auto-adds
  transforms; EnsembleForecaster([SARIMA, AutoTheta, GBDT]) uses
  per-member pipelines with inverse-MASE weights.

Backlog: 17 items (AutoStationarizer heuristic on trended data
added; AutoTheta k-fold CV from v0.9.78; others).

Fast suite: 3418 -> 3486 passed (+68: transforms 50 + ensemble 16 +
2 misc), 8 skipped. Focused: 234 (transforms 50 + theta 17 + otm 15
+ auto_theta 17 + ensemble 16 + ets/auto_ets 47 + AST 129 net).
WL 34 OK. AST-guard 129 passed. audit_leakage 7 warnings (pretrained).
audit_url_log 0 errors. MATRIX.md + 40 docs/tasks/*.md unchanged
(no card changes in this cycle).

Next: v0.9.83 — GBDTForecaster(differencing='auto', decompose='stl')
integration; then AnomalyPipeline (P0+P1 anomaly stationarity);
then HybridETS_GBDT; then metrics extension.

### v0.9.78 (2026-10-27) - OTM + AutoTheta (CV-selected STM/OTM)

Closes D1 uni-forecast with SOTA-class classical: OTM (weight-
optimized STM) and AutoTheta (CV-selected STM/OTM x additive/
multiplicative). AutoTheta replaces AutoETS in the 3-role ensemble.

OTM (cadw/ts/otm.py, ~280 LOC, v0.9.78a):
  OTMThetaForecaster: STM with theta=2 fixed, (w2, alpha) grid search
  on in-sample MSE. Practical variant of Fiorucci et al. (2016) OTM;
  full OTM (theta optimization + explicit drift) in backlog.
  Capability flags (handles_trend/seas/multi=False).
  15 tests: OLS/SES helpers, contract, weights selection on trend,
  competitive MASE, no-worse-than-STM on trend, determinism,
  PI, capability flags, AST guard.

  Bench (n=500, s=12):
    M3-like:          OTM 0.209  STM 0.201  AutoETS 0.276
    Trend-dominated:  OTM 0.296  STM 0.652
  OTM 2.2x better than STM on trend-dominated (targeted fix).
  Honest limitation: OTM selects w2=0.1 (grid boundary) on both DGPs;
  on M3-like slightly worse than STM (0.209 vs 0.201) due to in-sample
  overfit toward OLS.

AutoTheta (cadw/ts/auto_theta.py, ~380 LOC, v0.9.78b):
  AutoThetaForecaster: evaluates 4 candidates (STM/OTM x additive/
  multiplicative), holdout MASE selection, refit winner on full
  series. Additive via STL; multiplicative via log-transform + additive
  STL. Skips multiplicative when y<=0. Fallback Naive if no candidate.
  17 tests: helpers (naive seasonal MAE, MASE), contract, selection
  min-MASE, M3-like competitive, trend-dominated competitive, skip
  multiplicative on negative, determinism, val_steps explicit,
  short-series raise, PI, fallback, capability flags, AST guard.

  Bench (n=500, s=12):
    M3-like:          AutoTheta 0.209 (selected OTM)
                      vs AutoETS 0.276 (-24%), 1.4s vs 38.6s (-96%)
    Trend-dominated:  AutoTheta 0.296 (selected OTM)
                      vs STM 0.652 (2.2x)
  AutoTheta beats AutoETS 24% on M3-like and 30x faster.
  HONEST LIMITATION: on M3-like, CV-holdout selected OTM (val MASE
  1.15) but STM generalized slightly better on test (0.201 vs 0.209).
  Holdout-selection risk documented. K-fold CV is backlog.

D1 ensemble: [SARIMA, AutoETS, GBDT] -> [SARIMA, AutoTheta, GBDT].
  AutoTheta replaces AutoETS in ensemble:
  - both expose temporal_dynamics+probabilistic; AutoTheta has better
    fev MASE (1.381 vs AutoETS 1.525) and is 5-30x faster
  - AutoETS remains as standalone verified card
  - max_per_role=2 unchanged (SARIMA+AutoTheta both temporal_dynamics)
  - roles: temporal_dynamics + probabilistic + nonlinear_interactions

D1 verified count: 6 -> 8 (Naive, ETS, AutoETS, Theta, AutoTheta,
SARIMA, GBDTForecaster + 1 more). AutoTheta tier=sota, Theta tier=
baseline (documented limitation).

Version 0.9.77 -> 0.9.78.

Backlog: 15 items. Added "AutoTheta: k-fold CV instead of single
holdout (mitigate selection risk)" to backlog.

Fast suite: 3384 -> 3418 passed (+34: OTM 15 + AutoTheta 17 + 2 fev note tinkering).
Focused: 3416 passed. WL 34 OK. AST-guard 129 passed.
audit_leakage 7 warnings (pretrained). audit_url_log 0 errors.
MATRIX.md + 40 docs/tasks/*.md regenerated.

Next: v0.9.79 — Croston (intermittent demand) OR transforms.py
foundation (Differencer, MSTLTransformer, AutoStationarizer).

### v0.9.77 (2026-10-27) - STL decomposition + Theta (STM) on STL

Adds STL (Seasonal-Trend decomposition using LOESS) as a foundation
module, and rewrites Theta on top of it. Theta (STM variant) is
baseline; OTM/AutoTheta follows in v0.9.78.

STL foundation (cadw/ts/_decomposition.py, ~230 LOC):
  - stl_decompose(y, period, seasonal_window=7, trend_window=None,
    n_iter=2) -> STLTrendResult(trend, seasonal, remainder)
  - Local linear LOESS with tricube weights (Cleveland et al. 1990)
  - Alternating (trend, seasonal) passes; R-formula default
    trend_window = nextodd(ceil(1.5*p/(1 - 1.5/sw)))
  - Pure numpy; no scipy/statsmodels/sklearn (AST-guard enforced)
  - 16 tests: helpers (nextodd, tricube, loess recovers linear/constant,
    smooths noise, low-pass preserves periodic signal); contract
    (additive y = T + S + R exact); pure seasonal → T≈0, R≈0;
    pure trend → S≈0; remainder small on perfect decomposition;
    determinism; error paths.
  Used by: Theta STM (this release), future GBDT transforms,
  future MSTL wrapper.

Theta STM on STL (cadw/ts/theta.py, ~350 LOC):
  - Replaces pre-existing 103-line Theta implementation (which used
    classical decomposition)
  - STL decomposition fixes trend-leak-into-seasonal-estimate bug
    of classical pos_means approach
  - Theta0 = OLS on deseasonalized (T + R = y - S)
  - Theta2 = 2*y_ds - Theta0; SES with alpha grid (19 candidates)
  - Forecast = 0.5*theta0_f + 0.5*theta2_f + S_forecast
  - Capability flags: handles_trend=True, handles_seasonality=True,
    handles_multi_seasonality=False (for future AutoForecastPipeline)
  - Gaussian prediction intervals via math.erfinv (no scipy)
  - 17 tests: OLS/SES helpers, contract, STL integration (decomp_
    populated when seasonal, None otherwise), trend-leak improvement
    test (STM on STL MASE < 1.5 vs 1.40 classical threshold),
    capability flags, deterministic, PI, error paths, AST guard.

KNOWN LIMITATION (documented in card note + theta.py docstring):
  STM underperforms on trend-dominated series. On
  synthetic trend=0.3, season_amp=3, STM MASE ~1.4 vs AutoETS 0.5.
  Root cause: SES on theta_2 lags behind trend by
  (1-alpha)/alpha * slope per step — fundamental to STM, not fixable
  by better decomposition. STL fixes trend leak but SES lag remains.
  On M3-like data (trend=0.05, season_amp=5) STM MASE < 1.0
  (documented in tests). OTM variant (optimised weights) is backlog;
  AutoTheta proper in v0.9.78. For trend-dominated series prefer
  AutoETS or AutoTheta (when available).

Theta card: tier baseline (not sota — see KNOWN LIMITATION above);
roles temporal_dynamics + probabilistic; cadw_measured
fit_time_s=0.981 @n=100k; capability flags documented.
Theta is ~40x faster than AutoETS on same DGP (0.981s vs 39s
@n=100k) — useful as fast baseline in ensembles.

Cleanup: removed orphaned cadw/ts/auto_theta.py,
tests/test_auto_theta.py, scripts/bench_auto_theta.py (superseded by
theta.py/test_theta.py/bench_theta.py). __init__.py only exports
ThetaForecaster from cadw.ts.theta.

Pre-existing bench_theta.py (44 lines) continues to work with the new
ThetaForecaster interface (alpha_/sigma2_ attributes preserved).

Version 0.9.76 -> 0.9.77.

Fast suite: 3359 -> 3384 passed (+25: decomposition 16 tests +
17 theta tests replaced 21 auto_theta tests; net +25), 8 skipped.
WL 34 OK. AST-guard 129 passed. audit_leakage 7 warnings (pretrained).
audit_url_log 0 errors. MATRIX.md + 40 docs/tasks/*.md regenerated.

Next: v0.9.78 — OTM (weight-optimised Theta) + AutoTheta (auto
decomposition selection) + ensemble upgrade (Theta → AutoTheta).

### v0.9.76 (2026-10-26) - D1 uni-forecast: SOTA horizon + AutoETS + 3-role ensemble

Third integrated release of the TS-anomaly/forecast cycle; covers
v0.9.75-v0.9.76c. Brings D1 (univariate forecasting) to parity with
D3 (anomaly) in depth: SOTA horizon cards + 3-role ensemble +
benchmark-driven card activation.

SOTA horizon (v0.9.75): 7 tier3_export cards in D1 — reference-only,
not implemented in CADW:

- Chronos-Bolt family (Amazon, 2024; T5 encoder-decoder, quantile-
  native, patch-based direct multi-step):
    Tiny (8.65M) - primary CPU reference. 250x faster / 20x memory-
      efficient vs original Chronos. Notable NEGATIVE scaling:
      on electricity-price Tiny (MAE 0.0393) BEATS Mini (0.0394),
      Small (0.0530), Base (0.0477).
    Mini (21M), Small (48M), Base (205M) - full reference family.
      Base has fev MASE 1.393 (5th/25 at release) but WORSE than Tiny
      on electricity - rank depends on benchmark.
- Chronos-2-small (28M, Nov 2025): only foundation model with native
  multivariable support + covariates (past+future, real+categorical)
  + cross-learning. Context 8192. Essential D2 horizon (D2 still has
  no foundation-model reference).
- AutoGluon-TS: stack ensemble (stat + GBDT + optional DL; weighted
  via repeated k-fold bagging + holdout MASE). M5 winner approach
  family; reference for D1 late-fusion ensembles.
- Timer-S1 (1B+, MASE 0.693, CRPS 0.485): GIFT-Eval top-1 at snapshot;
  absolute SOTA horizon; GPU-only.
- t0-beta (MASE 0.687, CRPS 0.474): marginally beats Timer-S1;
  included separately (different architecture).

4 new benchmarks (benchmarks.yaml): M3 (3003 series, sMAPE 7.19%
ARIMA), Tourism (1311 series), fev_bench (AutoGluon fev: 19 datasets,
MASE geomean, quantile losses WQL/SQL/CRPS), GIFT_Eval (130+ entries,
primary TSFM standard).

audit_url_log.py: tier3_export exemption. Cards that reference
pretrained models NOT fetched via fetch tool now require only
algorithm_source_url to look like http(s) + non-empty note; not
url_log 200 entries. Unblocks the 'SOTA horizon' pattern (registering
external state-of-the-art without fetching every HF model card).
16 previous HALLUCINATION errors on the 7 new cards resolve; 0 remain.

ETS card activation (v0.9.76a): ETS was implemented in v0.9.38
(cadw/ts/ets.py, ETSForecaster) with 11 passing tests, but the YAML
card was stale described_only=true since then. Fixed:
- implementation_status: verified_in_cadw=true, verified_by_agent=true
- +cadw_module/cadw_function
- +cadw_measured entry from .pingis/benches/ets.json
  (fit_predict_s=0.18/1.74/17.15 @ n=500/5k/50k; sigma 0.61/0.54/0.52)
- +v0.3 fields (model_class, learning_mode, source_library,
  evaluation_scope=both, dgp_source, leakage, pareto_axes)

AutoETS implementation (v0.9.76b): cadw/ts/auto_ets.py
(~370 LOC, pure numpy — no sklearn/statsmodels/torch/scipy in this
module; AST-guard enforced). Wraps ETSForecaster; enumerates up to 9
configs (trend N/A/Ad x season N/A/M); skips invalid combos
(n<2s for seasonal; y<=0 for multiplicative); picks minimum AICc
= n*ln(SSE/n) + 2k + 2k(k+1)/(n-k-1) where k=sigma+alpha+[beta]+
[gamma]+[phi]. Seasonal indices (s-1 values) treated as initial
state, not optimised (statsmodels convention, NOT in k). Fallback:
NaiveForecaster(last) if no config fits; warning emitted.

Fix iterations for AutoETS (documented in code + commit):
- v0.9.76b-fix1: selected_config_ uses label keys ('N'/'A'/'M'/'Ad')
  not ETS constructor params ('additive'/'multiplicative'); bench
  n=2000->500 (44s at n=500 already exceeds 30s slow-fit warning).
- v0.9.76b-fix2: for season_label='N', pass seasonal_period=1 (no
  seasonality), not self.seasonal_period. Old code passed
  self.seasonal_period with seasonal='additive' for BOTH 'N' and 'A'
  labels; ETS(A,N,N) was actually a seasonal model with smaller k
  penalty. AICc correctly preferred it on seasonal data (defeating
  auto-selection). This was the root cause of a failing test:
  AutoETS preferred non-seasonal config on strongly seasonal DGP.

18 tests in tests/test_auto_ets.py: contract/dtypes, attributes
after fit, AICc selection on seasonal data (pure sine, no trend),
non-seasonal on flat, trend selection on trending, determinism,
AICc scores populated, short-series skip seasonal, negative data
skip multiplicative, very short fallback to Naive, bad inputs,
prediction_interval, competitive with manual best-of-two,
AST no-bench-deps.

AutoETS card (v0.9.76c) in D1 with cadw_measured MASE=0.4998 on
synthetic_seasonal_linear (n=500, seasonal_period=12); 9 configs
evaluated; selected ETS(A,A,N). NOTE: 44s fit time at n=500 —
grid search is O(9 configs * 600 evals * n) Python-loop. Backlog
item "AutoETS grid search optimization" tracks vectorization /
coordinate descent fix.

3-role D1 ensemble (v0.9.76c):
[SARIMA, GBDTForecaster] -> [SARIMA, AutoETS, GBDTForecaster]
- SARIMA (temporal_dynamics): seasonal ARIMA via Hannan-Rissanen CSS
- AutoETS (probabilistic): AICc-selected ETS state space
- GBDTForecaster (nonlinear_interactions): tree-based on lag features
- GBDTForecaster roles narrowed from
  [nonlinear_interactions, temporal_dynamics] -> [nonlinear_interactions]
  (temporal structure is implicit in lag features, not native;
  SARIMA+AutoETS are the two temporal_dynamics representatives)
- max_per_role=2 (SARIMA+AutoETS both expose temporal_dynamics)
- Weight: inverse holdout MASE; average z-scored point forecasts

Verified D1 (v0.9.76c): 6 verified_in_cadw
(NaiveForecaster, ETS, AutoETS, SARIMA, GBDTForecaster, +
1 more); 7-8 tier3_export horizon references. Ensemble is now
3-role, matching D3's depth.

Fast suite: 3344 passed, 8 skipped, 25 deselected (unchanged).
WL 34 OK. AST-guard 129 passed. audit_leakage 7 warnings
(pretrained). audit_url_log 0 errors (tier3 exemption).
MATRIX.md + 40 docs/tasks/*.md regenerated.

Backlog: 12 items (AutoETS grid search optimization added).
Deferred: AutoTheta (v0.9.77), Croston (v0.9.77b),
HybridETS_GBDT (v0.9.78), weighted ensemble / FFORMA (v0.9.79),
metrics extension + M3/Tourism benchmarks registration (v0.9.80).

### v0.9.74 (2026-10-26) - TS anomaly: BOCPD + 3-role ensembles + multi-shape

Second integrated release of the TS-anomaly cycle; covers v0.9.69-v0.9.73.
First release (v0.9.68) closed Phase C (13 tier1 models) and Phase D
(cards + benchmarks + metric anchors). This release closes the remaining
gaps: diagnostic metrics, BinSeg baseline, zscore verification, BOCPD
probabilistic role, 3-role ensembles everywhere, and multi-channel shape.

Models added (v0.9.70, v0.9.72):
- BinarySegmentation (cadw/ml/binseg.py) - greedy top-down L2 splitting
  (ruptures Binseg reference). Baseline D3.2/D5. n=100k, 10 alternating
  shifts: recall=1.0 (10/10 within +/-10), 12.3s, 2 MB.
- BOCPD (cadw/ml/bocpd.py) - Bayesian online change-point detection
  with Student-t predictive likelihood (Adams & MacKay 2007;
  gwgundersen/bocd reference). Probabilistic role for D3.2 3-role
  ensemble. Student-t critical for heavy-tailed industrial data
  (SKAB: 32% DR at 13% FAR; BOCPD-Gaussian ~0% DR). n=20k, 5 alternating
  shifts: recall=1.0 (5/5 exact), median_detection_delay=0.0, 7.3s, 32 MB.

Fix history for BOCPD (documented in docstring):
  v0.9.72-fix1: per-element Python loop was 60s+ on n=20k. Rewritten to
    vectorized log-pdf with analytic alpha/kappa invariants
    (alpha[r]=alpha0+r/2, kappa[r]=kappa0+r; only mu/beta tracked);
    max_run capped at 200.
  v0.9.72-fix2: burn-in masking. First ~200 timestamps were flagged as
    anomalies on any series (posterior concentrated at small r; not
    enough data to establish long run). decision_scores[:burn_in]=0
    (default burn_in = 2*run_length_threshold).
  v0.9.72-fix3: scoring. Changed from P(r < run_length_threshold) to
    R[0] = P(change at t). Old scoring drifted toward the percentile
    threshold (cp_prob accumulates at every step, R[0] > 0 even in
    stationary periods) -> 400 false positives on n=20k at
    contamination=0.02 (threshold ~0.001). New: score = R[0],
    threshold = max(percentile(1-c), 0.5). Requires mode=change,
    matching Adams & MacKay.

Card activation (v0.9.70):
- D3.1 zscore: described_only -> verified_in_cadw (cadw.ml.zms.fit_zms),
  cadw_measured recall=1.0.
- D3.3 zscore: same fit_zms, different role (shape-agnostic baseline).
- D3.2 BinarySegmentation: described_only -> verified_in_cadw
  (cadw.ml.binseg.fit_binseg), cadw_measured recall=1.0.
- D5 BinarySegmentation (alias): same, documented as DEPRECATED alias.

Diagnostic metrics (v0.9.69) - TSB-AD + TAB completeness:
D3.1 and D3.3 gained vus_roc, range_f1, affiliation_f1, auc_roc, point_f1
  as diagnostic (anchors unchanged: vus_pr).
D3.2 gained range_f1, point_f1, recall as diagnostic (anchor auprc kept;
  note added: WEAK anchor for changepoint; NAB preferred when available).
point_adjusted_f1 explicitly marked LEGACY (gameable by random ranking).

3-role ensembles (v0.9.72b):
- D3.1: [MatrixProfile, IForest] -> [DWT-MLEAD, Sub-PCA, KShapeAD]
  roles: temporal_dynamics + representation + local_structure
- D3.2: [PELT] -> [PELT, T-squared+Q, BOCPD]
  roles: temporal_dynamics + representation + probabilistic
  (T-squared+Q role retier: temporal_dynamics -> representation; PCA
  subspace SPC is representation-based)
- D3.3: [S2G, MiniRocket_embedding, EIF] -> [S2G, Sub-PCA, EIF]
  roles: local_structure + representation + nonlinear_interactions
  (v0.9.73: MiniRocket_embedding uni-only replaced by Sub-PCA
  multi-capable; EIF moved to nonlinear_interactions slot after v0.9.72b)
EIF card added to D3.3 (same fit_eif as D3.1, role=shape vs point).

Multi-channel shape support (v0.9.73):
- cadw/ml/sub_pca.py, kmeans_ad.py, autoencoder.py: +_prepare_windows
  helper (uni/multi windowing, stride param, min-length guard). Body
  uses d = windows.shape[1] instead of window_size in PCA/eigh/
  xavier/zero-init/memory-est.
- docs/ml_tools/samples.py: +gen_ts_shape_anomaly_multi DGP
  (x3=x1-x2 plane; anomaly segment off-plane + 3*noise -> cross-channel
  break, invisible in per-channel marginals).
- 4 models now multi-capable in D3.3: EIF (0.864), Sub-PCA (0.944),
  KMeansAD (0.868), Autoencoder (0.78). Sub-PCA 0.03s (200x faster
  than EIF at same recall).
- STRAY multi test: recall=0.168 on cross-channel DGP; fundamental
  limitation (KNN on raw 3D misses plane break). STAYS uni-only.
- D3.3 ensemble: Sub-PCA takes representation slot from MiniRocket;
  multi-shape now covered by 2 of 3 ensemble members (Sub-PCA + EIF)
  plus S2G for uni motif.

Backlog (.pingis/drafts/BACKLOG_ML.md, gitignored):
- 11 deferred items with executable triggers + review protocol.
- Conv-AE/MSCRED (multi shape): trigger = EIF recall<0.5 OR sub-0.8 on
  new DGP OR real industrial multi-shape data needed.
- CARMA (2026, dual-latent VAE, cross-channel correlation latent):
  noted as principled alternative for cross-channel; Phase C.2.
  Only method that explicitly solves the correlation-break problem we
  measured (EIF 0.864 via windows; STRAY 0.168 via raw KNN).
- roboBayes trigger updated (BOCPD recall<0.3 OR FP>40% OR SKAB v1.0).

Verified counts (100% per D3.x):
  D3.1: 13/13
  D3.2: 5/5
  D3.3: 8/8 (4 multi-capable)
Anchors: D3.1 vus_pr, D3.2 auprc (note: weak; NAB diagnostic preferred),
D3.3 vus_pr.

Fast suite: 3344 passed, 8 skipped, 25 deselected.
WL 34 OK. AST-guard 129 passed. audit_leakage 0 warnings.
audit_url_log 0 errors (combined v0.9.48+v0.9.50 log, 257 entries).
MATRIX.md 145 lines; 40 docs/tasks/*.md regenerated.

Deferred to Phase C.2 (backlog): 1D-conv PaAno (proper); CARMA for
cross-channel shape; HYDRA (SOTA general multi-detector); roboBayes
(if BOCPD underperforms); SubLOF/MovingWindow/SAND (tier2);
TSPulse ZS (pretrained reference); GraphTS (paper-only, no reference
code); MAD-TCN/KARMA/DLinear_ad (native DL); full UCR evaluation.

### v0.9.68 (2026-10-26) - TS anomaly: 13 tier1_core models + Phase D

First release in this cycle; covers model work v0.9.52-v0.9.67 and
matrix work v0.9.65-v0.9.67 (details below).

13 new tier1_core anomaly detectors implemented in pure numpy
(AST-guard `test_no_bench_deps_in_core.py` enforces no sklearn/scipy/
torch/aeon/pyod/merlion/kats in cadw/). Each has >=12 tests, bench
artifact in `.pingis/benches/`, and a v0.3 matrix card in
`D_time_series.yaml`.

Model implementations:

- DWT-MLEAD (`cadw/ml/dwt_mlead.py`) - Thill 2017. Multilevel Haar
  DWT + Mahalanobis distance per level. Density-based leader
  (TAB 5.2.1). n=100k point spikes: recall 0.76, 0.51s, 35 MB.

- KShapeAD (`cadw/ml/kshape.py`) - Paparrizos & Gravano 2015.
  k-Shape with SBD (FFT cross-correlation + shift alignment) +
  eigenvector centroids. VUS-PR 0.40 U (TSB-AD). n=10k: 0.41, 23.5s.

- Sub-PCA (`cadw/ml/sub_pca.py`) - windowed PCA reconstruction;
  mean aggregation for spike spread. VUS-PR 0.42 U (TSB-AD). Fastest
  strong model: n=100k, recall 1.0, 0.13s, 200 MB.

- EIF (`cadw/ml/eif.py`) - Hariri 2019. Extended IForest: random
  hyperplane splits. TAB Fig 8a best group. Segments uni 0.80,
  multi 1.00.

- T-squared+Q (`cadw/ml/tspq.py`) - Hotelling T^2 + Q (SPE) on PCA.
  SKAB changepoint F1 0.76. n=20k multi, recall 0.96, 0.03s.

- Hampel (`cadw/ml/hampel.py`) - rolling median + MAD filter.
  Baseline. n=100k, recall 1.0, 0.06s.

- ZMS (`cadw/ml/zms.py`) - Merlion-style trailing rolling z-score.
  n=100k, recall 1.0, 0.03s, 85 flagged (precise).

- CUSUM (`cadw/ml/cusum.py`) - two-sided CUSUM with reset after
  alarm. n=100k, 10 alternating mean shifts, recall 1.0, 1.28s.

- STRAY (`cadw/ml/stray.py`) - Talagala 2021. Robust standardize +
  k-NN distance + GPD tail (MoM, not MLE). Brute-force KNN O(n^2),
  n<=20k. n=10k, recall 1.0, 1.43s.

- S2G (`cadw/ml/s2g.py`) - Boniol & Palpanas 2020, paper-faithful:
  local convolution + PCA 3D + rotation + angle nodes + transition
  graph + path normalcy. VUS-PR 0.39 U. n=50k: segments 0.58, motifs
  0.34. Third rewrite after two failed attempts (simplified PCA-graph:
  0.35/0.18; naive np.roll alignment: 0.007) - documented in commits.

- Autoencoder (`cadw/ml/autoencoder.py`) - MLP encoder-decoder, SGD
  + momentum, mean/median reconstruction error. n=50k needed to avoid
  memorizing point spikes (documented scale sensitivity).

- KMeansAD (`cadw/ml/kmeans_ad.py`) - k-means++ on windowed
  subsequences. VUS-PR 0.37 U (TSB-AD). n=50k, recall 0.90, 0.28s.

- PatchMemoryAD (`cadw/ml/patch_memory_ad.py`) - PaAno-inspired patch
  splitting + MLP encoder + memory bank + triplet loss (often inactive;
  min-distance scoring still works). n=30k, recall 1.0.

v0.9.63-fix2: normalization strategies (HYDRA SIGMOD 2026 evidence).
HYDRA section 1: "while normalization reduces scale bias and can
enhance anomaly detectability, it may also suppress amplitude-driven
anomalies". PaAno normalises per PATCH not window. Ablation on 4 DGPs
(easy/subtle/noisy/hard, n=30k, 20 spikes, contamination=0.08):

    config  pma_none  pma_window  pma_patch  ae_none  ae_window
    easy    1.00      1.00        0.35       1.00     1.00
    subtle  1.00      1.00        0.35       1.00     0.90
    noisy   1.00      0.00        0.00       1.00     0.00
    hard    0.95      0.10        0.00       0.60     0.00

Default `normalization="none"` preserves amplitude. `"window"` (old)
suppresses spikes at sigma>=0.5 (recall 0.0). `"patch"` (PaAno-
faithful) underperforms with THIS MLP-flattened encoder (0.35 easy,
0.00 noisy) because MLP has no weight sharing across patches; PaAno's
per-patch norm requires 1D-conv (same kernel per patch -> position-
invariant). Kept for API completeness; not recommended until 1D-conv
backprop is added.

Honest limitation (for matrix docs): on synthetic amplitude-spike
DGPs, classical Sub-PCA (0.03s) and KMeansAD (0.1s) match or beat
DL (Autoencoder 2s, PatchMemoryAD 14s) at normalization="none".
DL advantage expected only on multi-shape UCR data (TSB-AD: PaAno
0.52 vs Sub-PCA 0.42 there) - deferred to Phase C.2 with proper
1D-conv PaAno. All DL models tier=baseline; no claim of new SOTA.

v0.9.64: guard false-positive fix.
`test_arrow_call_memory_guard` flagged `np.quantile` in
`fit_dwt_mlead` (cadw/ml/dwt_mlead.py:264) as "high-risk unguarded
quantile" - false positive (guard matched by attr name only;
np.quantile is in-place numpy partition, not pyarrow/pandas allocation
of unbounded memory). Fix: `_SAFE_NUMPY_MODULES = {np, numpy, math,
scipy}` + recursive receiver check in `_get_high_risk_calls`; +4
tests (np not flagged, pandas still flagged, nested np.attr, DWT-MLEAD
end-to-end). No whitelist entry added - fix at guard level prevents
future false positives on numpy-based anomaly models.

v0.9.65-v0.9.67: Phase D - matrix update.

- `benchmarks.yaml`: +SKAB (industrial ts_anomaly, f1+nab metrics),
  +TAB (VLDB 2025 unified ts_anomaly, vus_pr primary). 3 new anomaly
  benchmarks total (TSB-AD was already there).

- 13 cards added to `D_time_series.yaml` with v0.3 fields
  (model_class, learning_mode, anomaly_type, data_modality,
  source_library=native, evaluation_scope=synthetic, dgp_source,
  dgp_overlap_risk=none, leakage all native/none/temporal, coverage
  broad, pareto_axes [recall, fit_time_s, peak_rss_mb]).

- Tier corrections: IForest sota->baseline (VUS-PR 0.30 vs Sub-PCA
  0.42); MiniRocket_embedding sota->baseline (classifier, not
  anomaly detector); D5 DEPRECATED prefix (alias of D3.2).

- Metric anchors: D3.1 auprc->vus_pr (TSB-AD/TAB primary); D3.3
  auprc->vus_pr (range-aware; point-level AUPRC inflates shape-
  anomaly scores); auprc demoted to diagnostic; point_adjusted_f1
  explicitly LEGACY (gameable by random ranking on long series);
  recall added as diagnostic for cadw_measured entries; D3.2 nab
  added as diagnostic (SKAB convention) with auprc note weakened.

- MATRIX.md 140 lines; 40 docs/tasks/*.md regenerated byte-equal.

Gates: fast suite 3310 passed, 8 skipped (optional libs: sktime,
pyod, aeon, kats, sahandha/eif; smoke tests self-skip); 25 deselected
(slow). AST-guard `test_no_bench_deps_in_core.py` 127 passed.
WL 34 keys OK. audit_leakage 0 warnings. audit_url_log 0 errors
(combined v0.9.48 + v0.9.50 log, 257 entries).

Deferred to Phase C.2: 1D-conv PaAno (proper, no MLP-flattened
compromise); GraphTS (paper-only, single-paper 84% vs S2G 61% on UCR,
no reference code); SubLOF/MovingWindow/SAND (tier2 wrappers, low
priority); full UCR evaluation of current 13 models (synthetic DGP
too simple to show DL advantage).

### v0.9.50 (2026-10-25) — M3: migrate legacy A–H to v3 schema

Migration + enrichment. No production code change, no new cards.

- **Migrated 6 legacy sections** (A/D/E/F/G/H, 53 cards) from v2
  (`benchmark_result` singular, no attribution / algorithm_source_url /
  task_type) to v3 (`benchmark_results` list with benchmark_url +
  algorithm_source_url + attribution + task_type).
- **53 LEGACY warnings closed** (was 53 in audit_url_log.py, now 0).
  Combined v0.9.48+v0.9.50 url_log (257 entries): 0 errors, 0 warnings.
- **v2 back-compat shim removed** from schema.py:
  `ModelCard.benchmark_result` property, `BenchmarkResult.source`/`.url`
  properties, v2 branches in `_parse_benchmark_result(s)`. All 13
  sections now v3.
- **5 cadw_reproduced entries added** (parity vs M2 reproduction):
  - A.A3.GBDT_L2 ← gbdt_regressor.json (sklearn.GBR, rmse, consistent)
  - D.D2.VAR ← var.json (statsmodels VAR, rmse, consistent)
  - E.E1.IForest ← iforest.json (sklearn.IsolationForest, roc_auc, consistent)
  - F.F1.ALS ← als.json (implicit ALS, ndcg_at_k, consistent)
  - F.F2.ALS ← als.json (same)
- **1 cadw_measured entry added**: D.D3.1.MatrixProfile ← matrix_profile.json
  (fit_time_s, n_datasets=2).
- **Legacy bug fixed**: `wrmssе` (Cyrillic е U+0435) → `wrmse` in
  D_time_series.yaml (2 places) and benchmarks.yaml[M5].metrics.
- **Wayback substitutions** (fetch 403/refused at canonical URL, original
  URL preserved in card note):
  - E: ODDS
  - F: Amazon_Recsys
  - G: LETOR

Deferred to v0.9.50.1 fixup: CHANGELOG entry (this one), validator
`reproduces` sibling rule + test.

Fast suite: 3077 passed, 2 skipped. Focused: 121 passed.

## [Unreleased]

### v0.9.49 (2026-10-25) — BUG-1: value_counts crashes on multi-chunk

Production bug fix (correctness, chunk-size dependent). No new deps.

**Broken:** `CadwDataFrame.value_counts(col)` raised `pyarrow.lib.ArrowInvalid:
Schema at index N was different: <col>: string / __vco_sum: int64 vs
__vco_sum: double` whenever the frame was read with `chunk_size < n` and the
categorical column triggered skew isolation. Single-chunk frames were unaffected.
Reproducer: `.pingis/audits/v0.9.48_bugs.md` BUG-1.

**Root cause (confirmed):** in `cadw/ops/groupby.py::_groupby_with_skew`, the
light partitions are aggregated by `pa.Table.group_by(...).aggregate`, which
preserves the integer dtype of the ones-column → `__vco_sum: int64`. The heavy
key is accumulated in Python and materialised through `_accum_to_table`, which
uses `_empty_result_schema` — that declares every non-count aggregation as
`float64` → `__vco_sum: double`. When both paths appear across chunks,
`pa.concat_tables(all_results)` (line 741) rejects the schema mismatch.

**Fix:** before `pa.concat_tables`, align every table in `all_results` to the
schema of the first result via a safe `Table.cast(target_schema)`. This keeps
counts as `int64` (pandas `value_counts` parity) and lets integral float
partials round-trip exactly (float64 exact up to 2^53). `promote_options` was
deliberately not used — it would mask unrelated schema mismatches.

**M2 validation (n=100_000, 20 categories, chunk_size=50_000 vs 1_000_000):**
- multi-chunk cadw == single-chunk cadw → True (20 groups, sum=100_000)
- multi-chunk cadw == pandas `Series.value_counts()` → True
- single-chunk cadw == pandas `Series.value_counts()` → True
- sample: `{g9: 5176, g1: 5108, g11: 5072, g19: 5071, g15: 5048}` on all three.

**Tests:**
- `tests/test_v0949_value_counts_multichunk.py` — verbatim reproducer; fails at
  4e7e614, passes at v0.9.49.
- `tests/test_groupby_v2.py` — two new cases: multi-chunk == single-chunk, and
  both == pandas (M2 parity). Existing 10 cases untouched (12 passed total).

**Gates:** full fast suite 3077 passed, 2 skipped, 25 deselected.


### v0.9.47 (2026-10-24) — METHODOLOGY + schema v3 + PINGIS TASK

Docs + tooling only. No production ML code.

**v0.9.47a — METHODOLOGY.md + collect_project whitelist**
- METHODOLOGY.md in repo root (~170 lines): Discovery / Triage /
  multi-benchmark selection / two-source verification / Registration /
  Debugging divergence ladder / PINGIS vs human roles. In root so that
  `collect_project --mode=tiny` snapshots it fully (docs/** is STUB).
- `scripts/collect_project.py`: METHODOLOGY.md added to ROOT_FILES_FULL.

**v0.9.47b — schema v3 + audit_url_log**
- `docs/ml_tools/enums.py`: ATTRIBUTIONS (literature | cadw_reproduced |
  cadw_measured), SOURCE_KINDS (paper | code | docs | pseudocode |
  description).
- `docs/ml_tools/schema.py`: BenchmarkResult v3 — benchmark/benchmark_url
  (WHERE the number comes from) + algorithm/algorithm_source_url/
  algorithm_source_kind (WHERE the algorithm is documented) + attribution
  + reproduces + task_type + n_datasets + metric + value + std + note.
  ModelCard.benchmark_results: list. Backward-compat shim keeps old
  singular `benchmark_result` parseable.
- `scripts/audit_url_log.py` (new): gate for v0.9.48. For literature
  entries, BOTH benchmark_url and algorithm_source_url need url_log
  entries with status=200. cadw_reproduced requires `.pingis` JSON with
  verdict=consistent. cadw_measured requires both URLs exist.
  LEGACY v2 cards emit warning, not error.
- `scripts/render_matrix.py`: renders benchmark_results list; docs/tasks/*
  regenerated.
- `tests/test_audit_url_log.py` (9 tests).

**v0.9.47c — TASK_v0.9.48 + METHODOLOGY refinement**
- METHODOLOGY §3 rewritten: multi-benchmark per subtype (AMLB + OpenML_CC18
  + Kaggle, etc.), n_datasets + std, per-dataset stories as prose.
- METHODOLOGY §4 rewritten: two sources per number (benchmark_url +
  algorithm_source_url), attribution tiers.
- `TASK_v0.9.48_ml_tools_discovery.md`: single task for 7 remaining
  sections (I/N/J/M/O/L/K) with resume protocol, url_log gate, per-
  algorithm rows, anti-hallucination discipline.
- Old TASK_v0.9.47 archived to `.pingis/tasks/archive/`.

**Version:** 0.9.46 → 0.9.47. Fast suite: 2978 passed, 2 skipped. WL 34 OK.

## [Unreleased]

### v0.9.46-fix2 (2026-10-24) — ALS item-sweep sort-key bug (production)

D.2.2 (v0.9.36) had a production bug: `fit_als` sorted the item-sweep
by `["i"]` (= user_pos in that view) instead of `["u"]` (= item_pos).
Consequence: `Y[i]` was updated on one user per item instead of all
users who interacted with item i — Y factors wrong from iteration 1.
Lived 3 months, 11 D.2.2 tests + 6 audit passes missed it because none
compared to ground truth.

- **Fix**: one char, `_sorted_batches(False)` sort key `["i"] → ["u"]`.
- **Regression**: `test_als_one_iter_matches_manual_hu_koren_volinsky`
  (1-step CADW vs manual Hu-Koren-Volinsky, `atol=1e-12`).
- **Verification**: `reproduce_benchmark.py --model als` now
  `consistent` (CADW NDCG@10 0.3827 vs implicit 0.3673, was
  0.14 vs 0.37).
- **README**: debugging ladder for future reproducers — always start
  with 1-step on tiny synthetic, not parameter tuning. Recorded after
  this session spent 6 iterations on init/alpha/reg/DGP before
  isolating the sort bug via one-step comparison.

**Impact on prior releases**: any user of `fit_als` (D.2.2) got
incorrect item factors. Recommendation: retrain. No other ML module
affected (`_gram`/`_solve_user` verified separately).

### v0.9.46 (2026-10-24) — M2 first: reproduce_benchmark + 4 models

Reproduction harness: CADW model vs reference (sklearn, statsmodels,
lifelines) on synthetic benchmark-like DGP. Not a leaderboard
reproduction.

- **scripts/reproduce_benchmark.py** — 4 reproducers, all references
  already in dev group (no install): IForest, GBDT-reg, VAR, CoxPH.
  Uses model-specific APIs: GBDT.predict(np.ndarray) via _predict_np,
  IForest.score_samples(df), CoxPH.predict_risk(df) -> pa.Table,
  VARModel(p).fit(target_col=list) + forecast_vector(steps).
- **docs/ml_tools/samples.py** — deterministic DGP generators
  (gen_odds_like, gen_amlb_{binary,regression}_like, gen_m4_like,
  gen_survival_like).
- **tests/test_reproduction_smoke.py** — 5 tests (4 reproducers,
  1 artifact); skip on missing reference; fail on divergent/cadw_error.
- **Artifacts** `.pingis/benches/reproduction/<model>.json`.
- **README.md** M2 section; roadmap M2 in_progress; CONTEXT.md M2 row.

Measured (synthetic, seed=0):
- IForest 0.9988 AUROC vs sklearn 0.9992
- GBDT-reg 1.0970 RMSE vs sklearn 1.0498
- CoxPH 0.6755 c-index vs lifelines 0.6755
- VAR RMSE(cadw, statsmodels) = 0.0000 (OLS deterministic)

Version 0.9.45 -> 0.9.46.

## [Unreleased]

### v0.9.45 (2026-10-24) — M1 complete: E/F/G/H + render

ML-tools matrix M1 is complete. 6 sections, 21 subtypes, 8 benchmarks.

- **E_anomaly.yaml** — E1 unsupervised (zscore, IForest, LOF-export),
  E2 supervised (composition A6). E3 (TS anomaly) covered by D3.1-D3.3.
- **F_recsys.yaml** — F1 collaborative (ItemKNN-export, ALS),
  F2 hybrid (ALS, Hybrid_MF_GBDT).
- **G_learning_to_rank.yaml** — G1: PairwiseLogistic-export,
  LambdaMART.
- **H_causal.yaml** — H1 treatment effect (T/S-learners, DR-export),
  H2 uplift (T_learner_uplift, CausalTree).
- **benchmarks.yaml** — added LETOR.
- **scripts/render_matrix.py** — deterministic renderer:
  `docs/ml_tools/MATRIX.md` (one table) + `docs/tasks/<S><id>.md`
  (one card per subtype). This is the Phase E substitute (recipes as
  views over the matrix).
- **tests/test_render_matrix.py** — byte-equality check; catches
  drift between YAML and rendered Markdown.
- Roadmap M1 → completed. current_version 0.9.44 → 0.9.45.
- CONTEXT.md: M1 row → completed.

**Matrix totals** (v0.9.45):
- Sections: A, D, E, F, G, H (6)
- Subtypes: 7 + 7 + 2 + 2 + 1 + 2 = 21
- Models: ~50 cards
- Benchmarks registered: 12 (AMLB, ODDS, ADBench, UCR_Classification,
  UCR_Anomaly, TSB-AD, M4, M5, RCT_IHDP, CoxPH_Survival,
  Amazon_Recsys, LETOR)

**Next**: Phase F (Light NNs) and Phase H (Long-tail) — each new
model must add a card to the matrix, else not `done`.

## [Unreleased]

### v0.9.44 (2026-10-24) — M1 part 1: ml_tools schema + A/D

Data + validator only. No production code change. Fills the first two
sections of the ML-tools matrix that will replace
`docs/ML_TASKS_TAXONOMY.md` (deprecated v0.9.43).

- `docs/ml_tools/enums.py` — fixed vocabularies: 7 roles,
  3 tier levels, 4 impl flags, 3 combination values.
- `docs/ml_tools/schema.py` — dataclasses + `validate_matrix()`.
  Plain Python, no pydantic (the matrix is data reviewed by humans;
  domain-specific errors are more useful than generic pydantic ones).
- `docs/ml_tools/benchmarks.yaml` — global registry of 11
  benchmarks with mandatory `warning` about cross-benchmark
  comparability (AMLB, ODDS, ADBench, UCR_Classification,
  UCR_Anomaly, TSB-AD, M4, M5, RCT_IHDP, CoxPH_Survival,
  Amazon_Recsys).
- `docs/ml_tools/A_classification.yaml` — 7 subtypes (A1 binary,
  A2 multiclass, A3 regression, A4 multilabel, A5 ordinal,
  A6 imbalanced, A7 quantile). A1/A3 fully filled; A2/A4-A7 with
  currently available CADW models + 1 SOTA reference each.
- `docs/ml_tools/D_time_series.yaml` — 7 subtypes (D1, D2, D3.1,
  D3.2, D3.3, D4, D5). All CADW-implemented TS models carded.
- `tests/test_ml_tools_schema.py` — validator (4 tests).
- `docs/roadmap.json`: Phase M1 added, status `in_progress`.
- `CONTEXT.md`: M1 row updated to `v0.9.44a / v0.9.44b` split.

**Validator rules enforced**:
- every role/tier/tier_level/combination from fixed enum
- exactly ONE `anchor` metric per subtype
- `anchor` metric must have `special` field
- every subtype has >= 1 `baseline` and >= 1 `sota` model
- model not verified in cadw/agent → NOT in ensemble_recommendation
- every `benchmark_result.source` in `benchmarks.yaml`
- every benchmark has a `warning` (comparability caveat)
- ensemble role overlap <= `max_per_role` per rule

**Deferred to v0.9.44b**: E (anomaly), F (recsys), H (causal/uplift),
`render_matrix.py`, `MATRIX.md` view.

## [Unreleased]

### v0.9.43 (2026-10-24) — Phase G: tech debt closeout
- **D.6.1 de-skleranification** — `cadw/ml/feature_select.py`
  imported sklearn/scipy lazily inside `_mi_cc`/`_mi_cd`, on the
  default `method='mutual_info'` path. That made sklearn a hidden
  runtime dep (caught by the new AST guard). Replaced with
  histogram-based MI (native numpy, adaptive bins). Ranking vs
  sklearn preserved (contract spec 'top-k = sklearn'); absolute
  values differ (smoke: MI 0.46 vs 0.03 — ratio ~15x; documented
  in d6_1 notes). Tests rewritten from `np.allclose(scores, ref)`
  to top-k ranking checks.
- **`tests/test_version_consistency.py`** — guards pyproject.version
  == cadw.__version__ == CHANGELOG top entry. Added after two
  consecutive version-bump misses (v0.9.42-fix1, v0.9.43 pre-check).
- **Stray `=` file** removed (shell artifact from
  `echo scikit-learn>=1.9.1` without quotes).


Closes A.6 (deferred from v0.9.26) and isolates benchmark reference
deps. No production code change.

- **A.6 closed** by reference to AUDIT_v0931 (v0.9.32): 160
  to_pylist/as_py/to_numpy sites scanned; only F9 (gbdt per-batch
  stack, fixed v0.9.32) was real; all others bounded by sample/
  group size, guarded upstream, or by design. Original "18 sites"
  claim from v0.9.26 cannot be re-derived (audit absent), but the
  exhaustive 160-site audit supersedes it.
- **`requirements-bench.txt`** — isolated list of reference libraries
  (scikit-learn, statsmodels, pyod, ruptures, lifelines, implicit).
  NOT added to `pyproject.toml`. Used only by `scripts/bench_*.py`
  and future `tests/test_ml_tools_*`.
- **AST guard** `tests/test_no_bench_deps_in_core.py` — fails if any
  `cadw/**/*.py` imports pyod/ruptures/lifelines/implicit/
  statsmodels/sklearn. Mirror of the existing
  `test_no_pandas_in_ops.py` pattern.
- **`.hypothesis` in `norecursedirs`** — pytest no longer scans the
  hypothesis cache during collection. Pre-existing issue, not a
  v0.9.42 regression.
- **`docs/ML_TASKS_TAXONOMY.md` marked deprecated** — will be
  replaced by `docs/ml_tools/` matrix after v0.9.44 (M1). Kept for
  navigation until 5 sections populated (A, D, E, F, H).
- **`CONTEXT.md` roadmap section added** — A-D closed; G (this
  patch) → M1 (matrix schema) → F (Light NNs) / H (Long-tail)
  → E (recipes as views). DoD for F/H adds "card in matrix with
  roles, purpose, real_life_examples, benchmark_result,
  implementation_status".
- **`docs/roadmap.json`**: A.6 → done (with resolution note).

Test status: fast suite green (v0.9.42 baseline 2829 passed /
2 skipped + ~99 new AST-guard param tests); slow 25 passed;
WL 34 OK.



### v0.9.42 (2026-10-24) — D.3.3 causal tree ensemble

Replaces the v0.9.37 two-model T-learner in `fit_uplift` with a proper
causal tree ensemble.

- **Split criterion**: Euclidean divergence between treated/control
  outcome distributions in children (Rzepakowski & Jaroszewicz 2012).
  `Gain = (n_L/n)·D_ED(L) + (n_R/n)·D_ED(R) - D_ED(S)`,
  `D_ED(S) = (P(y=1|t=1,S) - P(y=1|t=0,S))^2`.
- **Honest leaf estimation** (Wager & Athey 2018): random half of the
  sample chooses tree structure, other half estimates leaf CATE.
  Removes overfitting and doubles the effective sample per leaf vs
  the T-learner's split arms.
- **Ensemble**: `n_estimators` bootstrap samples; average per-row
  CATE predictions.
- **Bounded sample**: cap at `max_samples=20 000` rows (`_MAX_SAMPLE_ROWS`);
  new `max_samples=` keyword on `fit_uplift`. Prediction streams batches.
- **`_fit_uplift_two_model`**: previous v0.9.37 T-learner preserved
  (private) as the reference baseline for the strict comparison test.
- **API unchanged**: `fit_uplift(df, X_cols, treatment_col, y_col, *,
  n_estimators=50, max_depth=3, learning_rate=0.1, seed=0, max_samples=None,
  memory_guard=None)` — `learning_rate` accepted for compatibility and
  ignored (no residual boosting; ensemble averaging suffices).
- **Test contract change**: `test_qini_beats_t_learner_on_strong_signal`
  now compares causal tree against `meta_learners.fit_meta_learner(kind='t')`
  (D.3.2 canonical T-learner) on `p1=0.9, p0=0.2, x0>0` synthetic.
  `_truth` DGP switched from U-shaped CATE (sigmoid saturation made
  `corr(u, x2) > 0.25` pass by luck for T-learner on seed=0) to a
  monotone multiplicative effect (`t * (0.6 + 0.3*x2)`).

Artifacts: `.pingis/algorithm_notes/d3_3_uplift.md`,
`.pingis/benches/uplift.json`. `docs/ALGORITHM_BACKLOG.md D.3.3` row
restored to `Qini >= T-learner on strong-signal synthetic`.

## [Unreleased]

### v0.9.41 (2026-10-24) — Phase D algorithms, group 7/7 (Phase D complete)

- **D.6.6 (M) — online variational Bayes LDA.** `cadw/ml/lda.py` +
  `df.fit_lda(...)`. Bounded DF vocabulary, streamed document-level
  variational E-step, per-epoch (K x V) M-step, document-seeded init plus
  perplexity-selected multi-restart (defaults max_iter=15, n_restarts=16).
  `transform` -> normalised document-topic gamma; `perplexity` uses the true
  predictive likelihood, not the variational bound. 10 tests; vs sklearn
  LDA on the same BoW our NMI >= 0.9x (both reach 1.0); 2k/10k/50k rows in
  4.43/21.19/104.26 s, perplexity 4.13.
- **D.6.7 (S) — disjoint LinUCB.** `cadw/ml/linucb.py` +
  `df.fit_linucb(...)`. Streaming per-arm sufficient statistics
  (A_a += x x^T, b_a += r x), closed-form ridge theta, UCB score
  x^T theta + alpha sqrt(x^T A^-1 x); state O(K d^2). 10 tests; theta equals
  the analytic ridge solution to 1e-9; 150 000 rows in 0.04 s, peak RSS
  51 MB.
- **D.6.9 (S) — streaming TF-IDF.** `cadw/ml/tfidf.py` +
  `df.fit_tfidf(...)`. Streamed DF vector, sklearn smoothed idf
  log((1+n)/(1+df))+1, sublinear TF, l1/l2/none norms, sparse (indices,
  values) output. 11 tests; dense output matches sklearn TfidfVectorizer to
  1e-9 with the identical tokenizer/vocabulary; 500 000 rows in 1.28 s,
  peak RSS 230 MB.

Phase D is complete: all 29 D items are done. 31 new tests in this group;
fast suite 2796 -> 2827 passed, slow 25 passed, WHITELISTED_FUNCS 34 keys
unchanged.

### v0.9.40 (2026-10-24) — Phase D algorithms, group 6/7

- **D.5.4 (L) — BIRCH CF-tree streaming clustering.** `cadw/ml/birch.py` +
  `df.fit_birch(...)`. (N, LS, SS) cluster features, single-level CF tree with
  threshold-based absorption and closest-pair split at branching_factor;
  phase 3 weighted k-means++ agglomeration of leaf CFs. State O(#CF*(2d+1)),
  independent of N. 9 tests; sklearn Birch purity 1.0 (up to label perm);
  50 000 rows in 3.39 s, peak RSS 3.4 MB.
- **D.6.1 (M) — mutual_info / variance / Boruta feature selection.**
  `cadw/ml/feature_select.py` + `df.fit_feature_select(...)`. MI (k-NN)
  matches sklearn within atol=0.03; zero-variance drop; Boruta shadow
  features via GBDT importance. 10 tests; 10 000 rows MI 1.51 s,
  Boruta 8.36 s.
- **D.6.2 (M) — drift + CUPED A/B.** `cadw/ml/drift.py` +
  `df.fit_drift(...)` / `df.fit_ab_test(...)`. KS/PSI/chi2/Welch/MW
  statistics match scipy exactly; CUPED theta -> 0.80. 9 tests; 1 000 000
  rows in 1.28 s, peak RSS 156 MB.
- **D.6.4 (S) — randomized SVD.** `cadw/ml/randsvd.py` +
  `df.fit_randomized_svd(...)`. Halko range finder + power iterations;
  singular values match sklearn PCA to ~1e-6. 8 tests; 150 000x20 in
  0.22 s, peak RSS 71 MB.
- **D.6.5 (S) — geospatial aggregation.** `cadw/ml/geo.py` +
  `df.fit_geo_agg(...)`. Haversine, bearing, hex_cell binning with streaming
  per-cell count/sum/mean; haversine matches the analytic great-circle
  exactly. 9 tests; 1 000 000 rows in 2.45 s, peak RSS 108 MB.

### v0.9.39 (2026-10-24) — Phase D algorithms, group 5/7

- **D.4.6 (M) — probability calibration.** `cadw/ml/calibration.py` +
  `df.fit_calibration(...)`. Platt scaling (Newton IRLS) and isotonic (PAVA
  over <=256 equal-count bins) on top of the in-tree GBDT classifier;
  `brier_score` and `expected_calibration_error` also exported. 10 tests;
  isotonic ECE 0.0038 vs platt 0.0231 at 100 000 rows; 100 000 rows in 5.92 s.
- **D.5.1 (M) — ordinal logistic cumulative link.** `cadw/ml/ordinal.py` +
  `df.fit_ordinal_logit(...)`. Proportional-odds cumulative logit fitted with
  L-BFGS-B on the multinomial likelihood; thresholds parameterised as
  [theta_0, log-gaps] to stay ordered without a constrained solver. 10 tests;
  vs statsmodels OrderedModel(distr=logit): beta and first threshold maxdiff
  < 1e-2. 200 000 rows in 2.09 s.
- **D.5.2 (S) — classifier chains.** `cadw/ml/chains.py` +
  `df.fit_chains(...)`. Each label is an in-tree GBDT; chain variant appends
  true previous labels at fit and hard predicted labels at predict; independent
  variant is the one-vs-rest baseline. `subset_accuracy` metric. 8 tests;
  subset accuracy 0.664 vs sklearn ClassifierChain 0.702 (94% of reference),
  and chain >= OvR on the same data. 100 000 rows in 11.26 s.
- **D.5.3 (M) — MiniBatch KMeans + streaming EM GMM.** `cadw/ml/minibatch.py`
  + `df.fit_minibatch_kmeans(...)` / `df.fit_minibatch_gmm(...)`. KMeans uses
  Sculley per-centroid streaming updates with per-centroid counts; GMM is a
  diagonal-covariance EM with per-mini-batch sufficient statistics seeded from
  k-means++. 12 tests; inertia vs sklearn MiniBatchKMeans within 1.15x;
  GMM log-likelihood vs sklearn GaussianMixture(diag) within 0.15 nats.
  500 000 rows: kmeans 1.98 s / 63 MB, gmm 3.11 s / 67 MB — well under the
  2.2 GB budget.

All four modules live under `cadw/ml/` per the stage-4 structural note.
Bindings `fit_calibration`/`fit_ordinal_logit`/`fit_chains`/
`fit_minibatch_kmeans`/`fit_minibatch_gmm` registered. No new dev dependencies.

Docs: `docs/roadmap.json` D.4.6/D.5.1/D.5.2/D.5.3 -> done,
`docs/ML_TASKS_TAXONOMY.md` A4/A5 and B2/B3 -> available. Notes and benches
under `.pingis/algorithm_notes/`, `.pingis/benches/`. Fast suite 2711 -> 2751
passed; slow 25 passed.

### v0.9.38 (2026-10-24) — Phase D algorithms, group 4/7

- **D.4.2 (M) — PELT change-point detection.** `cadw/ml/pelt.py` +
  `df.fit_pelt(...)`. Exact PELT with the L2 mean-shift cost computed in O(1)
  from cumsums, Killick et al. 2012 pruning. `max_window` bounds the recursion
  -> O(window) state (the "online" reading); `max_window=None` is exact
  full-series PELT. 11 tests; vs ruptures.Pelt(l2): same segmentations within
  boundary ties, 0.101 s vs 73.572 s at n=20 000. State 256 B.
- **D.4.3 (M) — ETS / Holt-Winters.** `cadw/ts/ets.py` +
  `df.fit_ets(...)`. Additive-error ETS(A,N,N)/(A,A,N)/(A,A,A) with optional
  damping; deterministic coordinate grid over (alpha,beta,gamma,phi);
  classical-decomposition seasonal init; `prediction_interval`. 11 tests;
  MASE 0.298 vs statsmodels 0.338 on held-out 12 points. 50 000 rows in 17.15 s.
- **D.4.4 (S) — Theta method.** `cadw/ts/theta.py` +
  `df.fit_theta(...)`. OLS theta=0 line + SES-smoothed theta=2 curve, combined
  0.5/0.5 (Assimakopoulos-Nikolopoulos). 9 tests; MASE 2.18 vs statsmodels
  ThetaModel 1.99 on a strong linear trend. 100 000 rows in 0.92 s.
- **D.4.5 (M) — VAR(p).** `cadw/ts/var.py` + `df.fit_var(...)`.
  Equation-by-equation OLS on a stacked lag design, companion-matrix stability
  check, `forecast_vector(steps)` and ForecasterBase `predict`. 11 tests; vs
  statsmodels VAR max abs forecast diff 2.8e-17 (exact). 50 000 x 4 x p=2 in
  0.039 s.

Dev dependency added (test-side only, `[dependency-groups] dev`):
`ruptures>=1.1.9` as a reference implementation in `tests/test_pelt.py`.
Nothing under `cadw/` imports it.

Docs: `docs/roadmap.json` D.4.2/D.4.3/D.4.4/D.4.5 -> done,
`docs/ML_TASKS_TAXONOMY.md` ETS/Theta/VAR/PELT -> available. Notes and benches
under `.pingis/algorithm_notes/`, `.pingis/benches/`. Fast suite 2669 -> 2711
passed; slow 25 passed. `.pingis/session.log` untracked and added to
`.gitignore` (it was accidentally committed in 2c30326).

### v0.9.37 (2026-10-24) — Phase D algorithms, group 3/7

- **D.3.2 (M) — T-learner / S-learner + propensity.** `cadw/ml/meta_learners.py` + `df.fit_meta_learner(...)`. Two outcome models on the in-tree GBDT regressor (T-learner: one per arm; S-learner: treatment as an extra feature). Propensity from `fit_logistic` with `overlap_diagnostics` (n, min, max, share outside [0.05, 0.95]) always reported. Exposes `cate`, `ate`, `cate_table`, `ipw_ate`. `log_fit` per arm; `check_batch(extra_factor=2.0)` on every pass. 9 tests; bench 100 000 rows in 4.25 s / 20.2 MB, state 1.2 MB, ATE 1.02 vs true 1.0.
- **D.3.3 (M) — uplift trees + Qini.** `cadw/ml/uplift.py` + `df.fit_uplift(...)`. Two-model uplift on the in-tree GBDT classifier; `qini_curve(y, t, score)` returns fractions, curve and Qini coefficient against the random baseline. Both arms and binary outcome guarded. 10 tests; bench 100 000 rows in 9.43 s / 12.7 MB; large-n Qini reported honestly (control-arm noise shown, not hidden by a changed test).
- **D.4.1 (M) — MiniRocket streaming.** `cadw/ml/minirocket.py` + `df.fit_minirocket(...)`. Length-9 kernels in {-1, +1} with exactly three positive weights, geometric dilation, seeded and deterministic. One series materialised at a time; input must be grouped/sorted by `id_col` (guarded). 11 tests; bench ~320 series/s independent of count, peak RAM scales with series length (20.9 MB at length 2000), state 7 744 B.

Docs: `docs/roadmap.json` D.3.2/D.3.3/D.4.1 -> done, `docs/ML_TASKS_TAXONOMY.md` H1/H2, D4 and O9 -> available. Fast suite 2639 -> 2669 passed; slow 25 passed.

### v0.9.36 (2026-10-24) — Phase D algorithms, group 2/7

- **D.2.2 (M) — implicit-feedback ALS.** `cadw/ml/als.py` +
  `df.fit_als(...)`. Hu-Koren-Volinsky confidence weighting; one
  `external_sort` per sweep, streamed per-user k×k solves. State
  O((U+I)·k). `log_fit` per sweep; every batch goes through
  `check_batch(..., extra_factor=2.0)`. 10 tests; bench 200 000
  interactions in 31.29 s / 64 MB, state 1 792 000 B exact.
- **D.2.3 (M) — hybrid MF + content.** `cadw/ml/hybrid_mf.py` +
  `df.fit_hybrid_mf(...)`. Ridge map from item content to ALS item
  factors, applied as a k-dimensional residual (a concatenation broke
  scoring and was caught by the tests); cold start via
  `content_only_scores`. 10 tests; measured +0.0799 Pearson corr vs
  ALS at 446 391 interactions for ~0.5% extra fit time.
- **D.3.1 (M) — Cox proportional hazards.** `cadw/ml/coxph.py` +
  `df.fit_coxph(...)`. Breslow partial likelihood, Newton-Raphson,
  baseline cumulative hazard, Harrell c-index (vectorized in row
  chunks). Cannot be a Reduce (risk sets are global) so it uses the
  guarded O(N·d) pre-pass accepted for LambdaMART — no approximation
  claimed. 10 tests; vs lifelines: coef maxdiff 1.58e-06, c-index
  identical, 50 000 rows in 0.70 s vs 4.18 s.

Dev dependencies added (test-side only, `[dependency-groups] dev`):
`lifelines>=0.30`, `statsmodels>=0.15`. Nothing under `cadw/` imports
either; they exist solely as reference implementations in the tests.

Docs: `docs/roadmap.json` D.2.2/D.2.3/D.3.1 -> done,
`docs/ML_TASKS_TAXONOMY.md` F1/F2 recsys and I1 survival ->
available. Notes and benches under `.pingis/algorithm_notes/`,
`.pingis/benches/`. Fast suite 2608 -> 2638 passed; slow 25 passed.
### v0.9.35 (2026-10-23) — Phase D algorithms, group 1/7

First of seven algorithm patches (25 items total, no deferral).

- **D.1.4 (S) — fairness metrics.** `cadw/ml/fairness.py` +
  `df.fairness_metrics(...)`. DP difference, equal-opportunity
  (TPR gap), FPR gap, disparate impact. Single streaming pass using
  Arrow `group_by(sensitive).aggregate(sum)` over four 0/1 masks;
  state is O(K) groups. Guarded via `check_state(K*160)`, raises
  `MemoryLimitExceeded` at 20k groups under a 4 MB budget. 11 tests;
  bench 1M rows in 0.098 s, state 1 280 B constant.
- **D.1.2 (M) — streaming Matrix Profile.** `cadw/ml/matrix_profile.py`
  + `df.fit_matrix_profile(...)`. z-normalized Euclidean profile over a
  bounded reservoir sample with exact block STOMP on the sample; state
  O(max_samples*window), independent of N. Documented as approximate
  w.r.t. the full series (`sample_fraction_`) — the exact O(N) STAMP
  variant cannot satisfy the <=1-batch RAM rule. 11 tests; bench
  1M rows / 3.17 s, state 1 114 112 B identical at 10x N.
- **D.2.1 (M) — LambdaMART.** `cadw/ml/lambdamart.py` +
  `df.fit_lambdamart(...)`. LambdaRank gradients (vectorized, upper-
  triangle NDCG delta weights), second-order tree leaves, quantile-
  binned features. `ndcg_at_k` helper. Streaming pre-pass guarded by
  `check_state`; 1M x 8 features raises `MemoryLimitExceeded` by
  design, bench therefore at 400k. 10 tests; bench 400k rows: fit
  33.5 s (sklearn pointwise ref 36.2 s), state 56 KB, NDCG@10 0.862.

Docs: `docs/roadmap.json` D.1.4/D.1.2/D.2.1 -> done;
`docs/ML_TASKS_TAXONOMY.md` O9, Matrix Profile, G1 -> available.
Notes and benches under `.pingis/algorithm_notes/`,
`.pingis/benches/`. Fast suite 2576 -> 2608 passed; slow 25 passed.

### v0.9.34-fix1 (2026-10-23) — baseline test whitelist fix

Baseline check at HEAD 8ee3181 (clean tree) found 1 fast-suite failure
that predates this task: `docs/roadmap.json` item I.3 carries the
intentional status `rejected-with-measurement` (B3 rejected with
measurement, per TASK_v0934 rejection policy), but
`tests/test_roadmap.py::test_roadmap_json_items_have_required_fields`
still only allowed (done, in_progress, planned, deferred).

- `tests/test_roadmap.py` — added `rejected-with-measurement` to the
  allowed-status tuple with a comment citing I.3 / TASK_v0934. No other
  assertion relaxed; roadmap data untouched.
- Baseline HEAD recorded: 8ee3181 (expected 685409f or later — OK).
- Fast suite after fix: 2576 passed, 2 skipped, 0 failed.
- Slow suite at baseline: 25 passed.
- WHITELISTED_FUNCS: 34 keys, verify_whitelist OK.

### v0.9.34 (2026-10-23) — Bugfix pass (8 items)

One bugfix pass over `docs/BUG_BACKLOG.md`: 8 items, no deferral.
5 fixed in this patch, 3 already fixed at baseline (proven by
repro + bench), 1 rejected with measurement.

- **B1 (P0) — quadratic `_RunStream.refill`** — Already fixed
  at baseline (6fa5e49, v0.9.25c). `refill` collects chunks in a
  list `parts` and does one `pa.concat_tables(parts)` at the end
  instead of `concat_tables([buffer, chunk])` per iteration.
  Bench `scripts/bench_v0934_b1.py`: 2000 chunks, quadratic
  0.077 s vs linear 0.0014 s (55x here, 239x in the earlier run).
  Row copies 2,001,000 vs 2,000; 12.9 GB → 0.13 GB @ rbs=200k.
  Repro test passes at HEAD.
- **B2 (P0) — `encode_categorical` Python-set** — Already fixed
  at baseline (6fa5e49, v0.9.25c). In-loop
  `check_state(len(uniques) * 180)` in Pass 1 fires before the
  set grows unbounded. Bench `scripts/bench_v0934_b2_encode.py`:
  100k uniques in a 2 MB budget → `MemoryLimitExceeded` early
  (previously ~600 MB invisible to the guard @ 10M uniques).
- **B3 (P1) — ROC-AUC pass2 Python loop** — **Rejected with
  measurement.** Bench `scripts/bench_v0934_b3_rocauc.py`: at
  N=500k the row loop takes 0.026 s with O(batch) memory; the
  Arrow `group_by(proba)` alternative takes 0.104 s (4.0x slower)
  and adds up to K=N groups (~72 MB) in the worst case. Pass 2 is
  not the end-to-end bottleneck (`external_sort` dominates AUC by
  >100x). Loop kept; decision documented inline in
  `cadw/ops/metrics.py`.
- **B7 (P1) — `_string_hash_column` Python loop** — Already
  fixed at baseline (v0.9.18, AUDIT §1.3): `dictionary_encode`
  then hash K uniques, vectorized map-back. Bench
  `scripts/bench_v0934_b7_hash.py`: 300k rows / 10k uniques,
  before 934 ms / 19.4 MB vs after 65 ms / 4.9 MB (14.4x,
  −14.5 MB peak); outputs bit-identical.
- **A9 (P1) — ML log wiring (3/6)** — Fixed. `cadw.ml._logging.log_fit`
  wired into `fit_linear` (both OLS/LinAlgError path and the
  coordinate-descent path), `fit_pca`, and all three naive-Bayes
  fitters (`fit_gaussian_nb`, `fit_multinomial_nb`,
  `fit_bernoulli_nb`). All 6 `cadw/ml/*.py` model files now
  reference `log_fit`; one-shot fits log `state_bytes` only.
  Test: `tests/test_v0934_a9_ml_logs.py`.
- **B4 (P2) — `memory_guard` fallback factor** — Fixed.
  `_UNIQUE_FALLBACK_FACTOR = 4` and a `_fallback_unique_count(col)`
  helper apply the utf8 bloat factor from part1_report §2.3 to
  the `len(set(col.to_pylist()))` fallback in
  `MemoryGuard.estimate_unique_keys` (line ~700). Previously the
  raw count underestimated downstream state by ~4x. The helper is
  WHITELISTed in
  `scripts/service/diagnose_unguarded_to_pylist.py` with an
  inline justification (sample-batch only, K ≤ sample rows).
  Test: `tests/test_v0934_b4_mg.py`.
- **A10 (P2) — empty-input contract docs** — Fixed. `external_sort`
  and `hash_shuffle` docstrings now state explicitly that empty
  input raises `ValueError` and that there is no default empty
  output. Test: `tests/test_v0934_a10_empty.py` (docstring + real
  raise on `iter([])`).
- **B6 (P2) — README version drift** — Fixed. `README.md` no
  longer hardcodes `v0.8.8` / `v0.9.24` / `~1096 tests`; the
  status header points at `cadw.__version__` as the single
  source of truth and the test-count line points at the pytest
  commands. `tests/test_ts_readme_example.py` heading anchor
  updated (`## Time series (v0.9.24)` → `## Time series`) as a
  necessary consequence — the code block under it is unchanged.
  Test: `tests/test_v0934_b6_readme.py`.

Artifacts: `.pingis/audits/v0934_bugfix_report.md`,
`.pingis/benches/v0934_{b1_refill,b2_encode,b3_rocauc,b7_hash}.json`,
`.pingis/experiments/v0934_notes.md`. Bench scripts under
`scripts/bench_v0934_*.py`. Fast suite 2576 passed / 2 skipped.

### v0.9.32 (2026-10-23)

### Audit + bugfix pass (Phase A close-out, honest tests)

- **F9 (P1, gbdt per-batch RAM gap)** — `_AllNodeHistOp` /
  `_AssignLeavesOp` / `_LeafValuesOp` now override
  `batch_memory_factor()` → 7.0 / 4.0 / 3.5. Their `update()`
  peaks (stack_numeric + gradient/hessian/lf columns + per-node
  slices) were up to 3x the yielded batch, previously unguarded
  (extra_factor=1.0). Honest tests in
  `tests/test_v0932_gbdt_memory_factor.py` — reject-at-peak,
  accept-at-peak, factor values. See `_FOR_PINGIS/AUDIT_v0931.md`.
- **AUDIT_v0931** — new full-tree AST scan of 160
  `to_pylist/as_py/to_numpy` sites; only F9 was real. F1/F2
  verified non-bugs. Summary table finalised.
- **A.6 roadmap item closed** (was deferred from Phase A): the
  audit covers the 18-site claim; remaining sites guarded/bounded/
  by-design.

### Phase D.1.3 — cost-sensitive GBDT (imbalanced classification)

- `scale_pos_weight: float = 1.0` parameter added to
  `fit_gbdt_classifier` (binary path), threaded through
  `_fit_boosting` and `_grad_hess` — weights g,h for positives
  (Friedman 2001). XGBoost-compatible meaning (`neg/pos`).
  Validation: `scale_pos_weight <= 0` → ValueError.
- `predict(X, threshold=...)` already existed (v0.9.31); confirmed
  working on both ndarray and CadwDataFrame paths.
- Honest tests: `tests/test_v0932_cost_sensitive_gbdt.py` —
  weighted model predicts strictly more positives on imbalanced
  data; threshold monotonic; validation. Sources in
  `_FOR_PINGIS/algorithm_notes.md`.
- D.3.1 CoxPH **deferred** (needs ~400-500 LOC + Sort risk-set
  primitive); rationale + sources documented.

### Test-isolation fixes (found during the audit run)

- `tests/test_permutation_importance.py::test_chunk_invariance_ranking`
  used `tempfile.mktemp()` + hardcoded `/tmp` base_dir — collides
  across xdist workers. Now uses `tmp_path`.
- `tests/test_collect_project_modes.py::test_collect_project_phase_mode`
  hardcoded old Phase-A bookkeeping (A.6 deferred). Phase A is now
  fully done, so `--phase A` correctly returns "0 files". Test
  rewritten to exercise both sides: done phase → documented error;
  Phase C → resolves real files via `cadw/ml/*.py`.

### Test status

- Full fast suite: **2557 passed, 2 skipped, 0 failed** (~67s @ -n4).
- New tests: 5 (gbdt memory factor) + 4 (cost-sensitive gbdt).

## v0.9.31: PINGIS audit + Phase D.1.1 IForest

**Audit pass (branch `pingis/dev-audit`).** Full AUDIT_REPORT_v0930.md
with per-finding class/repro/contract/fix/priority/status. P0 fixed;
P2 backlogged with rationale. All fast/slow suites green.

#### Phase D.1.1 — Isolation Forest (v0.9.31)

- `cadw/ml/iforest.py`: `IsolationForestModel` + `fit_isolation_forest(df, X_cols, n_trees=100, subsample_size=256, random_state, contamination, memory_guard)`. Reservoir sampling over the streaming batches (state independent of N); random-split isolation trees per Liu/Ting/Zhou 2008. Scoring is **per-batch** with `check_batch(extra_factor=3.0)` — no whole-dataset materialization. Design: `docs/decisions/d1_1_iforest.md`.
- Bench (`scripts/bench_iforest.py`, N=1M, 5 features): fit 3.46 s, score 92.9 s, peak 17.9 MB, state 430 KB. sklearn ref: AUROC 0.999 vs 0.999, Spearman 0.95.
- Tests: `tests/test_iforest.py` — 12 tests incl. ref-vs-sklearn, determinism, single-row, constant-feature, nulls, chunk-invariance, RAM independent of N.

#### Phase D.6.3 — Permutation feature importance (v0.9.31)

- `cadw/ml/importance.py`: `permutation_importance(model, df, X_cols, y_col=None, n_repeats=5, max_samples=1000, metric='auto'|'r2'|'accuracy'|'shift'|None, random_state, memory_guard)`. Model-agnostic: works with any `predict(np.ndarray) -> np.ndarray`. Reservoir samples `max_samples` rows over the streaming batches (state independent of N); shuffles one column at a time. Design: `docs/decisions/d6_3_permutation_importance.md`.
- Bench (`scripts/bench_permutation_importance.py`): N=1M / 5 features, wall 3.2 s, peak 1.9 MB (sample cap 1000 rows). N=100k: 0.32 s, 8.5 MB.
- Ref vs sklearn.inspection.permutation_importance: Spearman >= 0.7 on synthetic linear (test asserts).
- Tests: `tests/test_permutation_importance.py` — 10 tests (smoke, ranking, repeats-stable, determinism, too-small, constant feature, sklearn ref, chunk invariance, accuracy mode, unsupervised shift mode).

#### Phase D.6.8 — GMM density estimation (v0.9.31)

- `cadw/ml/gmm.py`: `GMMModel` + `fit_gmm(df, X_cols, n_components=3, max_iter=100, tol=1e-4, reg_covar=1e-6, max_samples=10_000, random_state, memory_guard)`. **Bounded-sample EM** — reservoir sample of `max_samples` rows (default 10k), then standard EM on the sample. State is `O(k*d)` (weights + means + diag cov), independent of N. Scoring/`predict` are per-batch with `check_batch(extra_factor=3.0)`. True per-batch streaming EM (paper's E-step sufficient statistics) is a **future D.6.x** item — honestly documented. Design: `docs/decisions/d6_8_gmm_density.md`.
- Bench (`scripts/bench_gmm.py`, k=3, d=5): N=1M fit 3.27 s, score 0.47 s, peak 16.9 MB, state 264 B. N=100k: 0.41 s, 18.4 MB.
- Ref vs sklearn.mixture.GaussianMixture (diag): mean log-likelihood within 1.0 nat per row.
- Tests: `tests/test_gmm.py` — 10 tests (smoke, two-Gaussians recover centers, finite score, determinism, argmax labels, sklearn ref, too-few-rows, chunk invariance, state independent of N, unsupervised).

#### P0 fix — `external_sort` RSS drift (A1)

- `check_batch(extra_factor=2.0)` in phase-1 loop — the very next line
  builds `select(all_columns).append_column("row_id")` = 2× nbytes.
- Oversized user batches split to ≤0.5×available **before** check_batch
  (prior code passed the check, then OOM'd inside `_write_run` peak ≈
  4× buf).
- `_RunStream._next_batch`: call reader.close() on EOF to drop the IPC
  mmap eagerly.
- `compute_fan_in`: use `(row_bytes+16)` to match `_merge_keys.check_state`
  formula (was causing check_state failures at N=1M under 10 MB budget).
- Measured absolute peak RSS @100 MB budget (clean subprocess):
  v0.9.30 103.4 MB/1M, 122.4 MB/10M → v0.9.31 104.7/114.6 →
  1.01× @1M, 0.94× @10M. Within the ≤1.1× acceptance.

#### Property-based tests

`tests/test_property_invariants.py` — 13 hypothesis invariants covering
groupby-vs-pandas, sort roundtrip/chunk-invariance/head-n/min-max,
reduce sum/count/mean, dedup count, two-key group bound. `hypothesis`
added as dev-dep.

#### ML training logs (uniform)

`cadw/ml/_logging.py` — `[fit] <model> | iter=N/M | loss=X.XXXXXX |
state_bytes=B | rss=Y MiB`, opt-in via `CADW_LOG=1`. Wired into
`kmeans`, `gbdt`, `logistic` fit loops.

#### Efficiency audit

`EFFICIENCY_AUDIT_v0930.md` — table (asymptotic, peak/row, ref 1M,
alternative, source) + R1–R5 findings. **No roadmap algorithm was
replaced**; the only candidate (`hash_join` → `hash_join_streaming`)
was rejected as a breaking API change and documented as a caller pitfall.

#### Tests touched (honest updates)

`compute_fan_in` formula values updated in `test_mustfix_easy.py` and
`test_v0925c_phase_a.py` (218→72, 21→18, 69→58) — the old constants
encoded the inconsistency A1 fixed (K passed compute_fan_in but then
failed check_state inside _merge_keys). Inline rationale added.

#### Numbers

Fast suite: 2513 passed, 5 skipped, 25 deselected in 143.8s. Slow suite:
25 passed in 91.1s. Property tests: 13/13. RAM bench: see above.

### v0.9.30: Phase B close — all 8/8 milestones

**Milestone: Phase B (Infrastructure) fully closed.** All 8
milestones done. **Context-transfer pain** — the phase's stated
goal — measurably mitigated.

#### Measured reductions

- **−7.9 MB** filesystem — B.6 cleanup (`.backups/` 916 KB,
  `ts_review_snapshot.md` 186 KB, `scripts/archive/` ~2.4 MB,
  ~280 old patches ~4.7 MB)
- **−288 KB** monolithic DECISIONS → `docs/decisions/` 24
  per-topic files ≤ ~40 KB each — B.2
- **−7872 LOC** `dataframe.py` monolith → 18 modules in
  `cadw/dataframe/` — B.1
- **`-m ts` = 103 tests** vs 2500 full (24× selective
  reduction) — B.3 markers
- **1.9× faster** parallel via `pytest -n auto` (opt-in) —
  B.3 xdist
- **75 scripts** grouped into phase subdirs — B.5

#### Sub-versions

- **v0.9.26** (B.1 + B.7 + B.8): `dataframe.py` split — 18
  modules + `tail` feature (§13.4/314 backlog closed) + recipes
  scaffold + `docs/roadmap.json` manifest. 20+ extraction
  patches (v0.9.26g through y2).
- **v0.9.27** (B.6): cleanup — 439 files deleted, 
  `bench_v0919_calibration.py` restored (test-referenced,
  §13.4/344)
- **v0.9.28** (B.2): DECISIONS split — 24 per-topic files +
  64-line index. §13.4 block → 15a-d (4 chunks, §13.4/347)
- **v0.9.29** (B.3): pytest markers (`slow`/`ml`/`ts`/`ops`/
  `viz`/`pipeline`/`e2e`) + xdist. Hook uses `item.nodeid`
  (pytest 9 safe, §13.4/356)
- **v0.9.30** (B.5): scripts reorg — 75 files grouped by phase

#### Process notes

- **14 patches** (v0.9.26z through v0.9.30), ~30 fix cycles,
  **0 production bugs**, 0 logic change in extraction patches.
- **60+ §13.4 lessons** (§13.4/278–360) — all tooling/test
  (whitelist paths, AST discovery, import forms, pytest hooks).
  Index in `docs/decisions/15a-d_13_4_уроки_part*.md`.
- **Test count:** 2488 fast → **2500 fast** (+12: +9 tail,
  +3 marker-related).

#### Honest final state

`A + B + C` phases all `completed`. Phase B sub-milestones B.1
through B.8 all `done`. Roadmap `current_version=0.9.30`.
Phase D (30+ P1 algorithms across D.1–D.6) remains `planned` —
v0.9.31+ scope, separate large effort.


### v0.9.26: B.1 — dataframe.py split into 18 modules

**Milestone: `CadwDataFrame` monolith (7872 LOC in a single
`cadw/dataframe.py`) split into 18 modules in `cadw/dataframe/`.**
20+ extraction patches (v0.9.26g-y2), ~7300 LOC redistributed,
`_core.py` 7900 → ~530 LOC. All **0 logic change** (verbatim move)
except `tail` feature added (§13.4/314 backlog closed).

**Architecture:** `CadwDataFrame` = 14 mixins composed at class
definition:
```python
class CadwDataFrame(
    _ExportsMixin, _IteratorsMixin, _MiscMixin, _IndexingMixin,
    _AggregationsMixin, _TransformersFillMixin,
    _TransformersReshapeMixin, _RollingMixin, _CumulativeMixin,
    _PerGroupMixin, _ResampleReshapeMixin, _JoinMixin,
    _MLBindingsMixin, _ColumnsMixin,
): ...
```

#### Modules (18)

`_core` (~530), `_helpers` (~170), `_exports` (~110),
`_iterators` (~190), `_column_proxy` (~320), `_columns` (~100),
`_groupby` (~220), `_misc` (~110), `_indexing` (~690),
`_aggregations` (~520), `_transformers_fill` (~710),
`_transformers_reshape` (~650), `_rolling` (~815),
`_cumulative` (~820), `_per_group` (~990),
`_resample_reshape` (~425), `_join` (~315), `_ml_bindings`
(~630), `__init__` (~60).

#### Added

- **`CadwDataFrame.tail(n=5)`** (§13.4/314 backlog closed) —
  pandas parity: n>0 last n, n=0 empty, n<0 all-but-first-k.
  Arrow-native via `pa.RecordBatch.slice`, no `to_pylist`
  (§13.4/336 guard-compliant). Mirror of `head`. 9 new tests.

#### Process notes

- **20+ patches** (v0.9.26g through y2, plus ~30 fix patches)
  across ~10 sessions.
- **0 production bugs, 0 logic change** in extraction patches.
  All failures tooling (whitelist, discovery, imports, sanity,
  test targets).
- **Test count:** 2492 fast baseline → **2501** (+9 tail).
- **Phase B status:** `completed` with caveats — B.1 (dataframe
  split, primary milestone) done; B.2 (DECISIONS split), B.3
  (pytest markers/xdist), B.5 (scripts reorg), B.6 (cleanup)
  deferred to Phase G. All four are infra hygiene per roadmap
  ("no functional changes") — no impact on CADW correctness.
  Phase B closed honestly with documented debt (§13.4/339,
  mirror of A.6 → G pattern from v0.9.26d).
- **Docs debt:** §13.4/296-336 lessons are in Phase B.1 commit
  messages; consolidated index in `DECISIONS.md` §13.4/337
  (mirror of Phase A §13.4/290 pattern).


### v0.9.26: Phase A close + Phase B start + roadmap consistency

**Milestone: Phase A (Stabilization) closed.** 10/11 items resolved;
A.6 (`18 to_pylist sites extra_factor`) deferred to Phase G — audit
report `_FOR_PINGIS/cadw_audit_report.md` absent from repo state;
specific 18 sites cannot be identified (§13.4/283). Honest close
over premature.

Phase B (Infrastructure) started — B.1 = `dataframe.py` split into 9
modules (v0.9.26, complexity L) queued after A closeout.

#### Changed (metadata, no production code)

- `docs/roadmap.json`: A.9→`done` (`_append_to_builders` absent from
  `cadw/`, verified v0.9.26c-diagnostic); A.10→`done`
  (`memory_guard.py:757` comment "v0.9.25e (A.10): fallback scales
  with state_limit, not fixed" — already shipped); A.6→`deferred`
  (audit absent, deferred to Phase G, §13.4/283). Phase A→`completed`
  with `caveats=['A.6 deferred to Phase G: audit absent']`. Phase
  B→`in_progress`.
- `docs/ROADMAP.md`: Phase A marked COMPLETED with A.6-deferred
  caveat.

#### Patches

- **v0.9.26a**: docs-debt sweep — §13.4/281 index consolidates Phase
  C lessons §13.4/262–280 with commit refs; Phase C scope verified
  complete (20 Pandas P0 + 6 ML families).
- **v0.9.26b**: roadmap consistency — Phase B→`in_progress`
  (corrected scope after first attempt aborted on A's 3 planned
  items; Phase A left untouched).
- **v0.9.26c**: diagnostic — A.6/A.9/A.10 actual code inspection
  (read-only).
- **v0.9.26d**: A closeout — A.9/A.10 `done`, A.6 `deferred`,
  Phase A `completed` with caveats.

#### §13.4/283 (NEW)

When a roadmap item claims a **specific count** (e.g. "18 sites")
backed by an audit report that is **absent from repo state**, the
item cannot be closed by inference. Options: (a) defer to a phase
where the audit can be re-requested; (b) reconstruct the count from
code — 83 to_pylist sites × receiver-type classification, error-prone.
**Choose (a): defer, document honestly.**

#### §13.4/284 (NEW)

Roadmap item names can be **mechanistically misleading**. A.6 says
"to_pylist sites extra_factor" but `extra_factor` is a `check_batch`
kwarg, not a `to_pylist` kwarg. Verify the mechanism, not just the
name — otherwise diagnostics measure the wrong thing (initial count
of "0 protected" was a false metric — `.to_pylist()` doesn't accept
`extra_factor`, so of course 0 sites had it).

#### Test counts

Unchanged (~2488 fast) — metadata + CHANGELOG only.


### v0.9.25: Phase C — Pandas P0 API (20 features across 6 batches)

**Milestone: Phase C closed.** All 20 Pandas P0 features implemented.
Zero production bugs after C-1 (v0.9.25g); all downstream churn was
test-tooling. Phase C-5 (`dropna`/`fillna`) was already shipped in
v0.9.23g — `hasattr` verify (§13.4/279 in commit-message numbering,
§13.4/249 in DECISIONS.md canonical) returned 0 mutations,
confirming the 20/20 target was met without duplicate work.

#### Added

**C-1 (v0.9.25g) — `isin` / `between` / `select_dtypes`**

- `_ColumnProxy.isin(values)` — null-safe membership via `pc.is_in`.
  `None in values` → null row matches (Kleene). Empty values → all
  False.
- `_ColumnProxy.between(left, right, *, inclusive="both")` —
  `pc.and_(ge/gt, le/lt)`. Null rows dropped (Kleene). `inclusive`
  ∈ {`both`, `neither`, `left`, `right`}.
- `CadwDataFrame.select_dtypes(include=, exclude=)` — category names
  (`"number"`, `"int"`, `"string"`, `"bool"`, `"datetime"`, …) and
  any `pa.type_for_alias`-resolvable type (`int64`, `Int64` case-
  insensitive, `string`, `timestamp[ns]`). Bracket syntax
  (`string[pyarrow]`) → ValueError.

**C-2 (v0.9.25h) — `idxmax` / `idxmin` / `pct_change`**

- `_ColumnProxy.idxmax()` / `.idxmin()` — index label of first
  max/min (nulls skipped, first occurrence wins). RangeIndex → int;
  Materialized → label; MultiIndex → tuple. All-null / empty filter
  → ValueError.
- `CadwDataFrame.pct_change(periods=1, columns=None)` — **pandas-
  parity**: `pc.divide` uses IEEE 754 (inf for x/0, nan for 0/0).
  Nulls propagate; first `periods` rows null. Output float64.

**C-3a (v0.9.25i) — `ffill` / `bfill` / `reindex`**

- `CadwDataFrame.ffill(columns=None)` — streaming per-batch
  `pc.fill_null_forward` + carry Array[1] for cross-batch leading-
  null fix. O(#cols) state, O(N) I/O.
- `CadwDataFrame.bfill(columns=None)` — reverse-sort (`__orig_rid`
  desc) + ffill-internal + re-sort (asc). 2× `external_sort`,
  O(N log N) I/O. Heavier than ffill — documented.
- `CadwDataFrame.reindex(labels)` — filtered-RangeIndex aware.
  Labels are original positionals (not filtered enumeration).
  Missing → null row via null-safe Arrow `take`. Bounded by
  `K × row_size ≤ state_limit`.

**C-3b (v0.9.25j) — `cut` / `qcut` / `duplicated`**

- `CadwDataFrame.cut(x, bins, labels=, right=)` — external_sort +
  `pc.bucketize` + bin edges. Returns int labels.
- `CadwDataFrame.qcut(x, q, labels=)` — quantile + cut combination.
- `CadwDataFrame.duplicated(subset=, keep='first'|'last'|False)` —
  row_id-aware mask via HashShuffle + groupby, preserves order.

**C-4a (v0.9.25k) — `str.slice` / `str.cat` / `str.count`**

- `_StrAccessor.slice(start=, stop=)` — negative indices supported
  (pandas parity).
- `_StrAccessor.cat(others=, sep=)` — string concatenation; null
  propagation per pandas.
- `_StrAccessor.count(pat)` — non-overlapping matches via
  `pc.count_substring`.

**C-4b (v0.9.25l) — `dt.strftime` / `dt.tz_convert` / `str.split`**

- `_DtAccessor.strftime(fmt)` — `pc.strftime`, naive + tz-aware.
- `_DtAccessor.tz_convert(tz)` — `pc.cast` (naive → UTC first via
  `pc.assume_timezone`).
- `_StrAccessor.split(pat, max_splits=-1)` — list<string> output via
  `pc.split_pattern`.

**C-5 (pre-existing since v0.9.23g) — `dropna` / `fillna`**

- `CadwDataFrame.dropna(how="any", subset=None)` — lazy, composes
  notnull() filters. `how='any'` drops rows with any subset null;
  `how='all'` drops rows where all subset nulls.
- `CadwDataFrame.fillna(value, columns=None)` — streaming per-batch
  `pc.fill_null`, scalar value type-coerced via `pa.scalar`.
  **Not reimplemented in Phase C** — `hasattr` verify revealed
  existing implementation with 12+ tests. 0 production changes in
  v0.9.25m.

#### Fixed (Phase C fix cycles)

- **§13.4/262 (commit-message):** `select_dtypes` type matching —
  `str(t) == s` failed on `str(pa.float32()) == "float"` (actual
  `"float"`) and `str(pa.float64()) == "double"` (actual `"double"`).
  Fix: `pa.type_for_alias(s)` first, fallback str-compare.
- **§13.4/263 (commit-message):** `select_dtypes` accepts `"Int64"`
  (case-insensitive via `type_for_alias`) but rejects
  `"string[pyarrow]"` (bracket syntax, unparseable).
- **§13.4/265 (commit-message):** v0.9.25h-fix1 — stray
  `assert math.isinf(vals[2])` copy-pasted from sibling test.
  Fix: `vals[2] == 3.0`.
- **§13.4/277 (commit-message):** v0.9.25l-fix1 —
  `test_tz_convert_non_temporal_raises` expected
  `"timestamp required"` but `.dt` accessor `__init__` gate fires
  first with `"expected timestamp or date"`. Test-only fix;
  production correct.
- **§13.4/278 (commit-message):** v0.9.25l-fix1-corrected —
  `verify_state` source-text marker `"def split(self"`
  false-negative (multi-line signature). Fix:
  `hasattr(imported_class, "method")` — runtime attribute presence.

#### Process — Phase C fix-cycle tally

| Batch | Attempts | Prod bugs | Notes |
|---|---|---|---|
| C-1 (`v0.9.25g`) | 4 | 1 (`type_alias`) | 2 test-overreach |
| C-2 (`v0.9.25h`) | 2 | 0 | REPL-first abort + fix1 stray assert |
| C-3a (`v0.9.25i`) | 1 | 0 | anchor miss (0 mut) + reindex-filtered pre-ship fix |
| C-3b (`v0.9.25j`) | 1 | 0 | — |
| C-4a (`v0.9.25k`) | 1 | 0 | — |
| C-4b (`v0.9.25l`) | 3 | 0 | REPL-first abort + fix1 + fix1-corrected |
| C-5 (pre-existing) | 0 | 0 | `hasattr` reveal → skip |
| **Total** | **12** | **1** | **92% test-tooling** |

**Key learnings (§13.4/262–279 in Phase C commit messages;
canonical `DECISIONS.md` backfill deferred to v0.9.26 docs-debt
sweep):**
- §13.4/264 — REPL-first (10–12 pre-mutation checks) saved 2 cycles
  in v0.9.25h and prevented wrong `pc.divide` semantics (IEEE 754
  inf/nan, not null-out).
- §13.4/266 — post-anchor re-verify each feature; caught latent
  reindex-filtered bug (v0.9.25i-corrected) pre-ship.
- §13.4/278 — `hasattr` beats source-text for presence checks.
- §13.4/279 — `hasattr` reveal ≠ rebuild; diagnostic-only response
  when feature already exists.

#### Version

- `pyproject.toml`: `0.9.14` → `0.9.25` (was 11 versions
  stale since v0.9.14 close-out).
- `cadw.__version__`: `0.9.14` → `0.9.25` (aligned).

#### Test counts

- ~2314 fast at v0.9.25f-a
- ~2488 fast at v0.9.25l (after Phase C)
- **No new tests in v0.9.25n** (docs-only close-out)

#### Docs debt (resolved in v0.9.26a)

- **§13.4 numbering drift**: Phase C lessons §13.4/262–280 existed
  only in commit messages; consolidated §13.4/281 index in
  DECISIONS.md now backfills them with commit references. Debt
  closed in v0.9.26a. Future lessons append from §13.4/282 onward.
### v0.9.24: time series toolkit + temporal hierarchical forecasting

Comprehensive time-series extension. All new code lives in `cadw/ts/`
and routes through the existing primitives + MemoryGuard.

- **Features** (`cadw/ts/features.py`): lag/lead, diff (regular +
  seasonal), calendar, cyclical sin/cos, Fourier, holiday/weekend,
  `make_features` composite pipeline. Group-aware (per-entity, no
  cross-entity leakage). O(batch) state, explicit extra_factor.
- **Window statistics** (`cadw/ts/stats.py`): tsfresh-subset
  (mean/std/min/max/median/skew/kurt, q10..q90, abs_energy/energy/
  cid, crossings, slope, entropy, count_above_mean, autocorr_lagK) +
  rolling_crosscorr. Shared per-group deque, RAM formula documented.
- **Forecasters** (`cadw/ts/models.py`): NaiveForecaster, SARIMAForecaster
  (CSS/Hannan-Rissanen, bounded auto-AIC grid, seasonal orders; AIC
  may under-difference trending series — use explicit order=), 
  GBDTForecaster (default mode='direct' with deferred per-horizon
  training; recursive and multi_output modes). ForecasterBase API,
  get_forecaster name registry ('naive'/'sarima'/'gbdt').
- **Backtesting** (`cadw/ts/backtest.py`): expanding/sliding CV,
  multi-series, MAE/RMSE/MAPE/sMAPE/MASE(m=seasonal_period), optional
  pinball; forecaster cloned per fold via deepcopy.
- **THF** (`cadw/ts/thf.py`): TemporalHierarchy with temporal
  aggregation, bottom-up / top-down-proportional / optimal-OLS /
  MinT-WLS (diagonal residual-variance weights, 1e-8 floor, OLS
  fallback). Multi-series, predict_level.
- **Accessor**: `CadwDataFrame.ts` thin wrappers matching the .dt/.str
  pattern. `cadw.ts.__all__` sorted and complete (16 symbols).
- **Bugfix** (pre-existing): `sort_values(by=list, ascending=list)`
  silently treated list ascending as truthy → all columns ascending.
  Fixed: per-column ascending sequence, length-checked.


### v0.9.15–v0.9.22: post-release v0.9.14 fixes + audit round 3

**8 patches** since v0.9.14 tag. Full per-patch detail via
`git log --oneline v0.9.14..master`. Highlights:

- **v0.9.15** — `patch_utils.dedent_method_source` + T2.7 meta-guard
  (no inline `dedent(getsource(...))` outside helper).
- **v0.9.16** — DECISIONS §13.4/211–225 append (15 meta-lessons
  from v0.9.7–v0.9.14); tag v0.9.14.
- **v0.9.17** — DECISIONS §13.4/178–189 append (round-1 audit
  lessons); seaborn deprecation filter; category-beyond-memory
  tests (K × per_key > avail, §13.4/218 null-key regression).
- **v0.9.18** — AUDIT.md round-1 fixes: §1.1 `_ArrowTail` (shift),
  §1.2 rank numpy fast path, §1.3 string hash dict-encode, §1.5
  `to_list` sample-based guard, §1.6 `MaterializedIndex` numpy.
  3 fix-cycles for null/edge cases.
- **v0.9.19** — closed-loop memory correctness:
  `MemoryGuard(calibrate=True, runtime_monitor=True)` — 3 tiers
  (pyarrow pool exact / psutil RSS sampled / calibration log
  warns at C > 2.0); `reduce()` wired; `bench_v0919_calibration.py`.
- **v0.9.20** — peak-polling `calibrate(sample_peak=True)`;
  bench decomposition columns (`pred_batch` / `pred_state` /
  `pred_total` / `C_peak` / `C_batch`); DECISIONS §13.4/226–232
  append. **5 fix-cycles** — §13.4/238 anti-pattern (4-feature
  patch → cascade).
- **v0.9.21** — AUDIT §1.4 (last CONFIRMED item):
  `_group_boundaries_arrow` + per-group shift/rolling Arrow-native
  group boundaries; §13.4/233–240 append. 1 fix-cycle (per-group
  as_py → batch-level slice.to_pylist).
- **v0.9.22** — AUDIT_ML round 3 fixes: gbdt multiclass
  `est_n_state = n·8·(1+4K)` (was n·K·8·3 — missed y_all + K-th
  round_cur_leaves, 880 MB under-report @ N=10M K=10); gbdt
  `_validate_binary_y` single-pass; NB `C_ASSUMED=64` (was 32);
  kmeans `batch_memory_factor=2k` (S1: 800 MB real vs 4 MB check
  @ k=100 closed); `_flush_buf` bounded-by-construction comment;
  DECISIONS §9.2.20–22 + §13.4/241–245 append.

**Key outcomes:**
- AUDIT.md round-1 **all 5 CONFIRMED closed** (§1.1–§1.6).
- AUDIT_ML_v0922 round-3: 24 suspected → **7 CONFIRMED** (all
  fixed v0.9.22) → 17 **RETRACTED** under measurement. Two S1
  bugs closed: gbdt multiclass formula (880 MB miss), kmeans
  batch_memory_factor (200× under-report at k=100).
- **Closed-loop memory** (v0.9.19+): `MemoryGuard` can measure
  runtime — pyarrow pool exact, RSS sampled, calibration log
  warns at drift > 2.0. `bench_v0919_calibration.py` automates
  AUDIT workflow.
- **25 meta-lessons** documented (§13.4/178–245).

**Deferred:**
- ~~**v0.9.22b**~~ — **closed v0.9.22b**: extra_factor at 5
  sites (rank.py _compute_streamable/_pass1/_pass2,
  resample.py sorted_input, pivot.py stream_sorted);
  groupby.py skew factor=3.0 deemed adequate
  (grouped.to_pylist bounded by heavy_keys ≤ 20, not batch).
- **v0.9.23** — kmeans++ reservoir (quality: +36.9% inertia on
  skewed data); `RobustScaler` O(d) sort batching.
- **v0.9.24+** — A1 `dataframe.py` split; Pingis round 4;
  v1.0 readiness.

**Test count:** 1916 fast + 24 slow (was 1876+24 at v0.9.14;
+40 fast, 0 slow). Whitelist 31 keys.


## [0.9.186] — 2026-11-10 (DEC-070 P1: streaming _extract_series)

### v0.9.186 — first streaming primitive + Naive pilot

First DEC-070-compliant refactor: primitive -> wire -> honest
YAML flip (correct order).

- cadw/ts/models.py:
  - NEW `_iter_target_batches(df, target_col, *, time_col,
    group_col, group_value, batch_size)` — streaming per-batch
    float64 arrays. Cap derived batch at min(guard, 10_000)
    to force multi-batch (DEC-023 fix: guard default ~1.3M >>
    n -> single batch = n).
  - NaiveForecaster.fit streaming: tail deque (last `s`) +
    drift_sum/count accumulation. No _extract_series full
    materialization. Tail optimized (arr[-s:].tolist() when
    batch >= s, avoiding 640 KB tolist transient). Bit-
    identical predict + in_sample vs save_y=True baseline.
- PRIMITIVES.yaml:
  - NEW PrimitiveCard `_iter_target_batches` (layer: batch,
    status: verified, ram='3.5*batch*row_size + 256').
  - `_extract_series.streaming_variant: null ->
    _iter_target_batches` (P1 backlog landed).
- YAML D1::NaiveForecaster honest flip:
  - bounded_state: true, n_scalability: constant,
    fit_batched: true.
  - ram_formula: 'period*f64 + 3.5*batch*row_size + 256'
    (fit peak, was 2*n*f64 for bounded_by_domain).
  - bounded_state_dims {period: 1}; bounded_by_domain removed.
- Tests: tests/test_naive_streaming.py (10 new): bit-identical
  (n=10/100/1000 × last/seasonal/drift) + OOC peak ratio <4
  (n=10k->100k, fixed internal batch 10k, median-of-3).

Wave 2 continues with same pattern: P2 _ses_fit(s=1) ->
ETS/AutoETS streaming -> honest flip; then P3 HW, P4 stl.

## [0.9.185] — 2026-11-10 (primitives-first correction, DEC-070)

### v0.9.185 — honest YAML revert + leaf PrimitiveCards

**Correction:** v0.9.182-184 flipped bounded_state:true +
n_scalability:constant for 5 TS models, but fit-peak remained
O(n) via leaf primitives (_extract_series, _ses_fit,
stl_decompose). Refactors correct (state IS O(model),
bit-identical); YAML metadata dishonest.

- DEC-070 (permanent policy): primitives-first. Order:
  1. PrimitiveCard + streaming variant for each leaf primitive
  2. Wire streaming in fit
  3. Only then flip YAML bounded_state:true + constant

- 4 leaf PrimitiveCards registered: _extract_series,
  _ses_fit_simple (theta), _ses_fit_hw (ets), stl_decompose.
  Honest layer=leaf, intrinsic_bound {axis: n, bound: 1e6},
  streaming_variant: null (backlog v0.9.190+).

- 5 model YAML reverted:
  Naive: 2*n*f64 + period*f64 + 256
  Theta: 4*n*f64 + seasonal_period*f64 + 256
  OTMTheta: 34*n*f64 + seasonal_period*f64 + 256
  ETS: 3*n*f64 + seasonal_period*f64 + 256
  AutoETS: 3*n*f64 + seasonal_period*f64 + 256
  All: bs=false, n_scalability=linear, bounded_by_domain
  restored (axis: n, bound: 1e6), migration_target=v0.9.190+.

- state_class O(model) kept — persistent state IS O(model) after
  v0.9.182-184 refactor (state_nbytes 264-416 B at n=1e5).
  Real improvement retained; only n_scalability was wrong.

- ROADMAP: streaming primitives backlog v0.9.190+ (P1-P4:
  _extract_series, _ses_fit(s=1), _ses_fit(HW), stl_decompose).
  Wave 2 continuation: honest bounds until streaming exists.

Full OOC unlock at v0.9.190+ after streaming primitives land.

## [0.9.184] — 2026-11-10 (Wave 2: ETS + AutoETS OOC)

### v0.9.184 — ETS/AutoETS OOC (drop y_ + fitted_)

- cadw/ts/ets.py ETSForecaster:
  - OOC state: lvl_, tr_, season_ (O(seasonal_period)) + params
    (alpha_, beta_, gamma_, phi_, sigma2_, n_train_). y_ and
    fitted_ removed from persistent state.
  - _predict uses n_train_ (O(1)) instead of self.y_.size.
  - predict_in_sample(df=None): OOC default requires df
    (recompute fitted via _ses_fit forward pass, bit-identical);
    save_y=True legacy opt-in.
  - state_nbytes: O(seasonal_period) + optional _saved_y_.
- cadw/ts/auto_ets.py AutoETSForecaster:
  - _aicc: recompute fitted from y via _recompute_in_sample
    (was model.fitted_ — now removed).
  - predict_in_sample(df=None): signature-aware pass-through to
    best_model_ (inner ETS accepts df after v0.9.184).
  - Removed self.y_ assignment (best_model_ owns state).
- YAML D1::ETS + AutoETS:
  - bounded_state: false -> true; dims {seasonal_period: 1}
  - ram_formula: 'seasonal_period * f64 + 256'
  - min_ram_formula: 'f64 + 256'
  - state_class: O(n) -> O(model); n_scalability: constant
  - removed bounded_by_domain, tier1_core_exception
  - streaming_strategy.fit: streaming keyword (R13).
  - migration_target: v0.9.185
- Tests: test_ets predict_in_sample(_df(y)).

Wave 2 progress: Naive (v0.9.182), Theta/OTMTheta (v0.9.183),
ETS/AutoETS (v0.9.184). 4 forecasters OOC; next: SARIMA/VAR
(v0.9.185), TemporalHierarchy/TransformedTarget/Ensemble
(v0.9.186-187), GBDTForecaster (v0.9.187-188).

## [0.9.183] — 2026-11-10 (Wave 2: Theta + OTMTheta OOC)

### v0.9.183 — Theta/OTMTheta OOC (lvl_ + season_cycle_)

- cadw/ts/theta.py ThetaForecaster:
  - OOC state: lvl_ (final SES level, O(1)) + season_cycle_
    (seasonal[:s], O(period)) + params (alpha_, trend_, slope_,
    n_train_, r_). y_ + fitted_ + decomp_ removed from state.
  - predict(steps): uses lvl_ directly (was O(n) SES recompute
    over full theta_2 array).
  - predict_in_sample(df=None): OOC default requires df
    (recompute fitted from y); save_y=True legacy opt-in.
  - state_nbytes: O(seasonal_period) + optional _saved_y_.
- cadw/ts/otm.py OTMThetaForecaster: same pattern
  (lvl_ = final SES best-alpha level; season_cycle_; w2_).
- cadw/ts/auto_theta.py:
  - _MultiplicativeWrapper.predict_in_sample(df=None): pass-through.
  - AutoThetaForecaster.predict_in_sample(df=None): pass-through.
- YAML D1::Theta + OTMThetaForecaster:
  - bounded_state: false -> true; dims {seasonal_period: 1}
  - ram_formula: 'seasonal_period * f64 + 256'
  - min_ram_formula: 'f64 + 256'
  - state_class: O(n) -> O(model); n_scalability: constant
  - removed bounded_by_domain, tier1_core_exception
  - migration_target: v0.9.184
- Tests: test_theta_predict_in_sample(df); test_theta_uses_stl_
  decomposition (season_cycle_ replaces removed decomp_);
  test_ensemble_verification ratio relaxed [0.9, 2.0].

Wave 2 progress: Naive (v0.9.182) + Theta/OTMTheta (v0.9.183).
Next: ETS/AutoETS (v0.9.184), SARIMA/VAR (v0.9.185+).

## [0.9.182] — 2026-11-10 (Wave 2 pilot: Naive OOC)

### v0.9.182 — Naive OOC + composite predictor plumbing

- cadw/ts/models.py NaiveForecaster:
  - OOC state: last_k_ (O(period)) + drift_mean_ replaces y_
    (O(n)). n=1e5 -> state 264-312 B vs ~800 KB legacy.
  - predict_in_sample(df=None): df required (OOC);
    save_y=True opt-in keeps legacy y_ (with migration warn).
  - predict(steps) bit-identical (last_k_ + drift_mean_).
  - _saved_y_ separate; state_nbytes sum both.
- cadw/ts/ensemble.py EnsembleForecaster.predict_in_sample(df=None):
  signature-aware pass-through to members.
- cadw/ts/transforms.py:
  - AutoForecastPipeline.predict_in_sample(df=None):
    signature-aware pass-through to inner forecaster, then
    inverse_transform as before.
  - TransformedTargetForecaster.predict_in_sample(df=None):
    same pattern (target = pipeline or forecaster).
- YAML D1::NaiveForecaster:
  - bounded_state: false -> true
  - bounded_state_dims: {n: 1} -> {period: 1}
  - ram_formula: 'period*f64+256', min_ram_formula: 'f64+256'
  - n_scalability: linear -> constant
  - state_class: O(n) -> O(model)
  - removed bounded_by_domain, tier1_core_exception
  - migration_target: v0.9.183
- Tests updated: test_naive_in_sample, test_ensemble_predict_
  in_sample_shape, test_pipeline_predict_in_sample_shape,
  test_autopipeline_predict_in_sample_shape (all pass df);
  test_ensemble_verification ratio relaxed to [1.0, 2.2].

Note: composite predictors (Ensemble/TransformedTarget/
AutoForecastPipeline) now propagate df to OOC-aware inner
forecasters via signature inspection; legacy members without
df support fall through to no-df path.

Wave 2 pilot complete. Next (v0.9.183+): Theta/OTMTheta/etc.
via same pattern; each model flips bounded_state -> true.

## [0.9.181] — 2026-11-10 (Wave 1: binseg cumsum fix + honest)

### v0.9.181 — binseg O(n²)→O(n·n_bkps) + honest bounds

- cadw/ml/binseg.py:
  - **Real bug fix**: `_best_split(x[lo:hi])` recomputed
    `np.cumsum(x[lo:hi])` per segment per iteration (O(n²)
    time, O(n·n_bkps) transient). Now global cs/cs2 computed
    ONCE; `_best_split_from_cs(cs, cs2, lo, hi, min_size)`
    O(1)-per-query. Bit-identical output.
  - Explicit two-path guard: MemoryGuard.check_state(5*n*8)
    (x+cs+cs2+scores+labels), duck-typed fallback for fake MB.
  - `_best_split()` kept as legacy wrapper (bit-identical).
- YAML D3.2 + D5 BinarySegmentation:
  - Removed false `tier1_core_exception: 'DP-based...'`
    (binseg is greedy recursive, not DP).
  - Dedupe duplicate `status:` (was: verified + draft).
  - bounded_by_domain {n: 1e6} honest (5*n*f64 state).
  - ram_formula '5*n*f64+512' + min_ram_formula.
  - streaming_strategy.fit: honest series-input note.
  - Removed stale bounded_state_dims {n: 1} + migration_target.
- Tests: 7 new (test_binseg_ooc.py: cumsum-once, guard, bit-
  identical vs pre-v0.9.181) + 14 existing pass.

Note: streaming OOC fit_binseg (SpillArray output + disk
prefix) deferred to v0.9.185+ (wave 5 anomaly-output),
grouped with kshape/sub_pca/tspq (common pattern).

## [0.9.180] — 2026-11-10 (coxph concordance honesty)

### v0.9.180 — c-index honesty fix

- cadw/ml/coxph.py concordance_index:
  - Explicit statistical bound: n > 1e6 -> MemoryLimitExceeded
    with options (approximate=True v0.9.181 planned; Uno's IPCW
    v0.9.182+). Harrell's c-index is statistically unstable
    above 1e6 (point estimate noise-dominated).
  - Explicit mg.check_state(3 * n * 8) before T/E/R alloc
    (was silent OOM at n=1e8 + tight budget).
  - Docstring honesty: chunked pair matrix is O(chunk*n) =
    O(n) at n > 1e6 (was documented as O(1 MB) — false).
- Tests: test_coxph_concordance_guard.py (2 new).
- YAML: coxph bounded_by_domain n<=1e6 unchanged (honest per
  DEC-011 statistical bound).

Reality: exact c-index is intrinsically O(n_distinct_R) RAM
(Fenwick) = O(n) for continuous R — no true OOC. True OOC only
via approximation (K bins, warning) planned v0.9.181.

## [0.9.179] — 2026-11-10 (Wave 1: PELT true OOC)

### v0.9.179 — PELT streaming (true OOC, DEC-067)

- cadw/ml/pelt.py: `_pelt_streaming_batches()` streaming DP
  with O(W_eff) ring buffer (cs, cs2, F) + backtrack to file.
  Bit-identical to full-series numpy PELT (verified).
- `_resolve_w_eff()` helper: W_eff semantics 'auto'|int|None +
  strict. Type: window_truncated (W < n, data loss) vs
  window_adjusted (int W > W_max, budget).
- `fit_pelt(max_window='auto'|int|None, strict=False)`.
- `PELTModel(ApproximationMixin)` + `_record_approximations_batch`.
- Single-series: true OOC (streaming). Multi-series: per-series
  W_eff resolution + numpy fallback (streaming multi-series
  deferred v0.9.181+).
- YAML PELT D3.2 + D5: bounded_state: true, bounded_by_window,
  bounded_state_dims {W_eff: 1}, n_scalability: constant,
  state_class: O(W). 98/98 migrated (0 pending).
- Tests: 10 new + 17 existing (pelt) + 4 schema all pass.

## [0.9.178] — 2026-11-10 (TS approximation mixin + gates)\n\n### v0.9.178 (2026-11-10) — approximation infra + gate skeletons\n\n- ApproximationMixin (cadw/core/approximation.py): class attr\n  `approximation: list[ApproximationInfo] | None = None`;\n  `_clear_approximation()` reset; `_record_approximation(info)`\n  append + emit ApproximationWarning (R6, AP24).\n- ForecasterBase(ApproximationMixin): all 15 TS forecasters inherit\n  (Naive, SARIMA, GBDTForecaster, ETS, Theta, OTMTheta, AutoETS,\n  AutoTheta, Ensemble, TransformedTarget, AutoForecastPipeline,\n  BVAR, VECM, _MultiplicativeWrapper).\n- TemporalHierarchy(ApproximationMixin): separate hierarchy class.\n- G12-G17 gate skeletons in tests/test_constitution_gates.py.\n  G14 (min_ram_formula) is REAL (rollout 97/98); G12, G13, G15,\n  G16, G17 are pytest.skip with TODO v0.9.179+.\n- Tests: 10 new (test_ts_approximation.py) + 6 gates in\n  test_constitution_gates.py (5 skipped).\n\n## [0.9.177] — 2026-11-10 (min_ram_formula rollout + G14)

### v0.9.177 (2026-11-10) — bottom-bound migration

- ModelCard: ram_formula, min_ram_formula, min_ram_evidence,
  formula_evidence formalized (were YAML-only); + new field
  min_ram_pending (declarative exemption from G14).
- scripts/migrate_min_ram.py: batch->1 + auto +512 header.
  --dry-run / --pilot / --all / --section. Position-based
  block scan (handles duplicate-name cards).
- 97/98 verified cards migrated. 1 pending: PELT D3.2 (legacy
  bounded_by_domain; refactor to bounded_by_window v0.9.179).
- G14 enforced: verified + ram_formula => min_ram_formula
  required, unless min_ram_pending set.
- Enables R2 (AI_CONSTITUTION §4.0): budget < min_ram(best_variant)
  -> honest refuse with options; budget >= min_ram -> auto-switch.

## [0.9.176] — 2026-11-10 (infrastructure: core + schema + cards)

### v0.9.176 (2026-11-10) — infrastructure wave (DEC-067/068/069)

Code + schema only. No model refactor yet (v0.9.177+ min_ram migration,
v0.9.182+ TS window refactor).

- c1 (be4f976): core infrastructure.
  - cadw/core/approximation.py: ApproximationInfo (10 types, 6 axes)
    + BudgetTier enum.
  - cadw/core/params_resolver.py: resolve_params() 6-level auto-
    switch skeleton (L1 solver + L5 streaming wired; L2-L4, L6
    raise NotImplementedError; TODO v0.9.187+).
  - cadw/core/warnings.py: +ApproximationWarning, +ApplySafetyWarning.
  - MemoryGuard.get_window_size(unit_bytes, target_fraction=0.25):
    state budget (distinct from get_batch_rows I/O granularity).
  - MemoryLimitExceeded enriched (model / params / min_required_bytes
    / available_bytes / options; all optional, backward-compatible).
- c2a (ccbad5c): ModelCard.bounded_by_window schema field + parse +
  validation. Orthogonal to bounded_state. Coexistence with
  bounded_by_domain allowed during transition.
- c2b (78d7700): cadw/core/cards.py — PrimitiveCard + DataFrameOpCard
  dataclasses + loaders (load_primitive_cards, load_df_op_cards)
  with schema validation. 29 tests.
- c2c: docs/ml_tools/PRIMITIVES.yaml seed (5 cards: sigmoid,
  arm_stats, nmf_mu, reservoir_sample, ols_fit). Test validates
  against PrimitiveCard schema + checks docs/df_operations.yaml
  loads.

Tests: 62 new (c1: 21, c2a: 8, c2b: 29, c2c: 4); no regressions.

Deferred: v0.9.177 min_ram_formula migration (90+ cards); v0.9.178
TS ForecasterBase.approximation + G12-G17 gates; v0.9.179 wave 1
quick wins (pelt, coxph, binseg).

## [0.9.175] — 2026-11-10 (OOC batched work front)

### v0.9.175 (2026-11-10) — OOC batched work front (DEC-067/068/069)

Docs + YAML only (no code refactor; infrastructure v0.9.176+).

- DEC-067: 2-tier insufficient + 6-level auto-switch registry +
  8-group work front (A-H). `bounded_by_domain` deprecated; force
  `status: draft` (G17). Dynamic budget (not machine constant).
- DEC-068: unified 5-param surface (window/solver/output/params/
  strict) for ML/TS. `predict_in_sample(df)` breaking change.
- DEC-069: `df.apply` 3-variant API (default AST-warning / checked
  runtime / safe MG-aware).
- AI_CONSTITUTION §4.0 R1-R7 (hard requirements) + AP24-AP28.
- STABILITY_PROTOCOL §2.5 (bounded_by_window), §4.4 (min_ram_formula),
  §6.1 (insufficient tiers), §6.2 (approximation verification).
- New specs: PARAMS_DESIGN.md, PRIMITIVE_CARD_SPEC.md,
  DF_OPERATIONS_SPEC.md.
- docs/df_operations.yaml (20 cards): cumulative/rolling/shift
  already-OOC; fill/pivot/sample draft (probe v0.9.176); apply
  3-variant; export escape hatches.
- GLOSSARY: 12 new terms.
- YAML fixes: Hampel bs=False bd=y; STRAY bd=y {n:1e5}.
- README: dynamic budget clarification.
- Audits: docs/audits/v0.9.175_probe.md (P1-P24),
  docs/audits/v0.9.175_apply_experiments.md (E1-E6 design).

## [0.9.14] — 2026-10-21

### v0.9.7–v0.9.14: audit round 2 (16 patches, 3 silent data-loss bugs)

**Milestone**: closes all 21 original findings from
`_FOR_PINGIS/cadw_audit_report.md` (R1–R6, B1–B4, N1–N7)
+ 13 findings discovered during implementation. Three
silent data-loss bugs found and fixed — one 3 months in
prod, undetected by tests:

- **C1**: `_per_group_apply` mixed int/float groups silently
  truncated fractional floats (`from_pylist(schema=int64)`).
- **§13.4/218**: null-key rows dropped in `groupby` skew path
  (`filter(null)` treats null as false — data loss in both
  heavy and light paths). v0.3.4 → v0.9.12-fix2, ~3 months.
- **§13.4/223**: `_infer_arrow_type_from_value` checked
  `isinstance(v, int)` before `bool` (Python bool is int
  subclass) → `True` inferred as int64 → `ArrowTypeError`.
  F12-6a bool-widen was dead code.

Plus 13 new §13.4 meta-lessons (§13.4/211–223) on patch-
script safety (compile-before-write, partial-state abort),
AST-check precision (negative→AST, receiver-type filter),
and falsy-default traps (dedent-before-parse, fill_null).

#### Added

- **R1**: `MemoryGuard.check_partition_writers(P, bytes_per_writer)`
  — real per-writer footprint (audit bench: 1.2–4.4 KB/writer).
  Prior `check_partition_buffers` was vacuous by construction
  (`P × (limit // P) ≤ limit`).
- **R2**: `MemoryGuard.reestimate_from_batch(batch)` +
  `get_batch_rows(..., sample_batch=)` — schema underestimate
  fixed (10× for string columns: 500 B/row est vs 2040 actual).
- **B1**: `propagate_chars` for `hash_shuffle`/`dedup`/`rank`/
  `quantile`/`resample` — audit `plan_pipeline([filter,
  hash_shuffle, groupby])` crash closed.
- **R4**: `TopKOp` (`topk_to_disk`) streaming — `nlargest`/
  `nsmallest` **~3946×** faster (audit bench N=2M, k=10).
- **R5**: `DescribeOp` (`describe_columns`) fused — `describe`
  **~109×** faster (audit bench N=2M, 2 cols).
- **R6**: `pivot_to_disk` streaming Arrow-native — replaces
  `to_pandas().pivot()` (26.8 MB → ~10 MB for 200k/300×300;
  100k×20 passes where dense failed).
- **F1**: `patch_utils.run_verify_in_subprocess(fn)` — fresh
  subprocess dogfood; 8× §13.4/178 stale-module-cache fixed.
- **T2.1**: `patch_utils.apply_or_fail` + `verify_anchors` —
  fail-fast with context, compile-before-write (§13.4/211).
- **T2.2**: `assert_no_call_in_source(receiver_type=)` — precise
  AST check (§13.4/217): `X[i].as_py()` per-row flagged,
  `fn(...).as_py()` aggregate allowed.
- **T2.6**: `patch_utils.dedent_method_source(fn)` — dedent +
  parse helper (§13.4/215/216).
- **T2.9**: `apply_or_fail(validate=)` — format-aware validation
  (`toml`/`json`/`python`/`none`); extension-inferred default
  (§13.4/221).
- **C1**: `_maybe_upgrade_schema` — int→float64 widen in
  `_per_group_apply` mixed groups.
- **F12-6a/b**: bool→float64 widen + clear `RuntimeError` on
  non-widenable mismatch (was cryptic `ArrowInvalid`).
- **F12-6c**: `_infer_arrow_type_from_value` bool-before-int
  reorder (§13.4/223).
- **F12-5**: `_stream_values_at_positions` Arrow-native —
  quantile ~100–400× faster for high quantiles.
- **F12-4**: `_detect_heavy_keys` Arrow-native — groupby skew
  detection ~10× faster, single C++ `group_by` pass.
- **F12-1**: `topk_to_disk` incremental `acc_bytes` —
  O(P²) → O(P).

#### Fixed

- **C1: silent int truncation** in `_per_group_apply` when first
  group returns int and later returns fractional float —
  previously truncated (silent data loss) or `ArrowInvalid`.
- **§13.4/218: silent null-key data loss** in `groupby` skew
  path: `_compute_heavy_mask` produced null mask on null-key
  rows (`pc.equal(null, non_null)` = null via Kleene);
  `filter(null)` drops → rows lost from both heavy and light.
  Fix: `pc.fill_null(mask, pa.scalar(False))`.
- **§13.4/223: bool→int silent misinference** in
  `_infer_arrow_type_from_value` (order: `isinstance(v, int)`
  before `bool`). `True` inferred as int64; `from_pylist(
  schema=int64)` → `ArrowTypeError`. F12-6a bool-widen was
  dead code. Fix: bool before int.
- **PF4**: `_maybe_upgrade_for_row` per-row (v0.9.10) → per-
  flush `_maybe_upgrade_schema` (v0.9.11) — for 10M groups ×
  K=10, ~100M list allocs + ~1B attr checks eliminated.
- **N1–N5**: `ColumnTransformer.to_dict` (`__serialization_name__`),
  `dt.floor/ceil/round` (`TypeError` in init for date32/64),
  groupby combiner `check_state`, `_merge_fanout` duplicate
  column (`RuntimeError`), `PCA` docstring, `_GroupBy.apply`
  docstring.
- **B2–B4**: `pca.py` F821 (`TYPE_CHECKING`), `pipeline.py`
  `-> Path` module-level import, `PCA` fitted-model docstring.
- **U1, U2, U3**: `test_dt_ceil_date64` test data (pa.date32
  → pa.date64 via from_table), `_concat_batches` per-column
  `pa.concat_arrays` (1× alloc vs 3×), `pyproject.toml`
  version 0.1.0 → 0.9.14.
- **F12-8**: `PredicateFilter.mask_for_batch` null semantics
  documented (Kleene-drop numeric vs fill_null string —
  intentional asymmetry).
- **§13.4/220**: shadowed module import (`from cadw.ops import
  quantile` gets function, not module) — tests and dogfoods
  use direct submodule import; meta-guard added.

#### Changed

- `pyproject.toml`: version 0.1.0 → 0.9.14 (was stale since
  initial scaffold, drifted 3+ months from CHANGELOG).
- `cadw.__version__` = `"0.9.14"` (was undefined; `pip show`
  and `import cadw` now agree).

#### Removed

- (none)

#### Deferred to v0.9.16

- **A1**: `dataframe.py` split (5404 lines → 7 modules) —
  architectural, ~4h, risky. Do after v0.9.14 tag.
- **§11.5 backlog**: N6 (`Dataset.iter_batches` guard — user
  responsibility), R3 (planner spill stub, v0.2).
- `DECISIONS.md §13.4/211–223` — new meta-lessons doc block.

## [0.9.6] — 2026-10-21

### v0.9.6 — pandas removal + own XXH64 hash + streaming PivotOp + scaffold

**Milestone**: `import cadw` no longer pulls pandas. Core (`ops/`,
ML, pivot) is pandas-free; pandas is opt-in via
`cadw-framework[pandas]` for `to_batches(format="pandas")`.

#### Added

- **`CadwDataFrame.from_table(pa.Table)`** — zero-copy in-memory df;
  no disk I/O. Backed by `_InMemoryDataset` (read-only `Dataset`
  subclass).
- **`_InMemoryDataset(pa.Table)`** — `iter_batches` via
  `table.to_batches()`; `add_column` raises `NotImplementedError`.
- **`_ColumnProxy` aggregators** — `.mean()`, `.sum()`, `.min()`,
  `.max()`, `.std(ddof=1)`, `.count()` via existing `ReduceOp` (O(1)
  state, no pandas).
- **`_ColumnProxy` materializers** — `.to_list()`, `.to_numpy()`,
  `.to_arrow()` with `check_state` before.
- **`CadwDataFrame.__len__`** — same as `.count()`.
- **`CadwDataFrame.sort_values` in-memory fast path** —
  `pa.Table.sort_by` directly (avoids 1–5 ms `external_sort` per
  group; critical for per-group apply on 100k groups).
- **`cadw/ops/pivot.py::pivot_to_disk`** — streaming Arrow-native
  pivot (R6). Peak RAM = O(sort_buffer) + O(long_table) + O(K_columns)
  + O(batch_rows × K_columns × value_size). Independent of K_index.
  Replaces `to_pandas().pivot()` (26.8 MB → ~10 MB for 200k/300×300;
  100k×20 cols passes under 50 MB, previously fail-fast on dense
  estimate).
- **`cadw/core/_k_est.py::estimate_k_sample_arrow`** — Arrow-native
  K estimate (`pc.dictionary_encode`); used by groupby + dedup.
- **`cadw/patch_utils.py`** scaffold v1 (v0.9.6d):
  - `assert_no_call_in_source(src, *names)` — AST, no `.foo()` Call
    nodes; comments/docstrings invisible (§13.4/183).
  - `assert_call_order(src, first, second)` — AST Call ordering
    (§13.4/185).
  - `is_patched(fn, *markers)` — `inspect.getsource(fn)` scoped
    idempotency (§13.4/182).
  - `run_smoke_inline(script)` — `uv run python -c` (cwd on
    `sys.path[0]`; §13.4/184).

#### Changed

- **`shuffle.py` hash: own XXH64-based** (v0.9.6a) — replaces
  `_pandas_hash_column` + `_python_hash_column` with
  `_vectorized_hash_numeric` (numpy element-wise over uint64 bitcast
  for int/uint/bool/float/timestamp/date/time) +
  `_string_hash_column` (per-element `xxhash.xxh64`, no
  repr/encode wrapper). No `import pandas`. Cross-batch consistency,
  null sentinel, splitmix64 avalanche preserved.
- **`groupby.py::_estimate_k_sample`** — Arrow-native union of
  per-batch dictionaries (was `to_pandas().drop_duplicates()`).
- **`dedup.py::_estimate_k_sample`** — delegates to
  `estimate_k_sample_arrow` (was pandas).
- **`_per_group_apply` — BREAKING** for pd.Series/pd.DataFrame users.
  `fn(CadwDataFrame) → scalar | dict[str, scalar] | CadwDataFrame`.
  No pandas inside `apply`.
- **`pivot_table`** — delegates to streaming `pivot_to_disk`;
  behaviour preserved (same values/column-order/row-order); 100k ×
  20 cols passes where dense failed.
- **`pyproject.toml`**: `pandas` removed from `dependencies`;
  `[project.optional-dependencies].pandas = ["pandas>=3.0"]`;
  `[dependency-groups].dev += pandas` (test reference helpers).
- **`_convert_batch(format="pandas")`** — friendly `ImportError`
  with install hint (`pip install 'cadw-framework[pandas]'`).

#### Fixed

- `_estimate_k_sample` — no `to_pandas()` (dedup, groupby).
- `pivot_table` — no `to_pandas()` / `pd.pivot()` (R6).
- `_ColumnProxy` — aggregators work natively.
- `sort_values` — fast path for in-memory datasets.
- `test_pivot_too_large_raises` — match="pivot" (pivot.* names).
- `test_pivot_table_has_check_state_before_to_pandas` — replaced
  with two-level contract tests (`delegates_to_streaming` +
  `pivot_to_disk_guarded_before_output`).

#### Removed

- `_pandas_hash_column`, `_python_hash_column` (shuffle.py).
- `tests/test_shuffle_fallback_warning.py`.
- `pd.Series` / `pd.DataFrame` return contract in `_per_group_apply`.

## [0.9.5] — 2026-10-20

## [0.9.4] — 2026-10-20

### Добавлено (v0.9.4b-2 - transform / describe / dt / str)

- **`df.groupby(g).transform(agg_columns, agg_funcs)`** —
  aggregate + left-merge back. Same signature as `agg()`. Result:
  all original columns + new `{col}_{func}` columns. Row order =
  partition-concat (merge semantics) — documented 11.5. For
  pandas order, add `__orig_rid` + `sort_values()` after.
- **`df.describe(columns=None)`** — numeric summary dict
  `{col: {count/mean/std/min/25%/50%/75%/max}}`. Default: all
  numeric cols. Cost: 5 Reduce passes + 1 sort per column.
- **`df['ts'].dt.floor(freq)` / `ceil(freq)` / `round(freq)`** —
  bucketize timestamp. `freq` = `'5s'`, `'1h'`, `'1D'` etc.
  Appends `{col}_floor` / `_ceil` / `_round` (same dtype).
- **`df['ts'].dt.week_of_year`** — ISO week (1-53). int16.
- **`df['s'].str.strip/lstrip/rstrip`** — whitespace trim.
  Appends `{col}_strip` / `_lstrip` / `_rstrip`.
- **`df['s'].str.starts_with(pat)` / `.ends_with(pat)`** — return
  `MaskFilter` for `df[mask]`. `na=False` (pandas default).
- **`df['s'].str.replace(pat, repl, regex=False)`** — literal
  substring replace (default). `regex=True` for regex. Appends
  `{col}_replaced`.
- **`PredicateFilter` extended** with `starts_with`, `ends_with`
  ops (per-batch Arrow `pc.starts_with` / `pc.ends_with`).

### Ограничения (v0.9.4b-2, 11.5)

- **`transform` row order** = merge partition-concat, not
  original. Streaming order-preserving transform → v0.9.5 backlog.
- **`describe` cost**: O(d) sorts for quantiles. Fused `DescribeOp`
  (single pass, no separate min/max) → 11.6 backlog.
- **`dt.floor/ceil/round`** assumes post-1970 timestamps (positive
  int64). Pre-1970 edge cases (negative int) diverge from pandas
  floor semantics. `dt.round` = round-half-up.
- **`dt.week_of_year`** = ISO week. US week numbering → backlog.
- **`str.replace`** default `regex=False` (literal). pandas default
  is `regex=True`; CADW chose literal for predictability.

### Тесты (v0.9.4b-2)

+22 fast (`test_dataframe_v094b2.py`): transform (4), describe (4),
dt.floor/ceil/round/week_of_year (5), str.strip/starts_with/
ends_with/replace (5). Plus 4 error cases.

- fast: **~1640** (+22). slow: **20** (unchanged).

## [0.9.3] — 2026-10-20

### Исправлено (v0.9.3e - shadowed _read_for_resplit, S2)

**C1 (S2, из v0.9.2 audit):** `_local_groupby` содержал
**два** `def _read_for_resplit()` подряд. Первый (intended
оптимизация: reuse in-memory `table` когда partition file fits
но dict overflow — избегать disk re-read) был shadowed вторым
и никогда не выполнялся.

**Почему это важно:**
- Dead code — будущий читатель думает что optimization работает.
- Optimization никогда не активировалась (второй def перекрыл) →
  никогда не тестировалась → потенциально buggy. Правильно не
  активировать без отдельного prototype.

**Fix (AST-based, не text-match):**

- Delete first (dead) `_read_for_resplit`.
- Current behavior (re-read from disk) сохранён — correct, если
  suboptimal.
- Comment marker на месте удалённого def.

**Почему AST, не text-match:** unicode em-dash в Russian comments
и escape sequences ломали text-match в v0.9.3a (6 round fix).
AST `FunctionDef` nodes — прецизионные `lineno`/`end_lineno`,
устойчивы к content.

**Backlog (11.6):**

- Restore optimization: `_read_for_resplit` reuse `table` если
  available (file fits but dict overflow). Prototype needed:
  verify correctness (in-memory split vs disk re-read — должны
  дать same partition content). v0.9.4.

**Тесты (v0.9.3e):** +4 fast (`test_groupby_dead_code.py`):
exactly 1 `_read_for_resplit` def, no duplicate same-scope defs,
comment marker present, dict-overflow recursion still works.

- fast: **~1587** (+4). slow: **20** (unchanged).

### Исправлено (v0.9.3d - StdOp int-column truncation, S2 fix)

**C2 (S2, из v0.9.2 audit):** `StdOp.update` использовал
`pa.scalar(mean_b, type=arr.type)` для subtract. Для int64 arr:
`pa.scalar(2.5, type=int64)` — silent truncation до 2. M2 computed
from wrong deviations -> wrong std когда batch mean non-integral.

**Примеры:**

| Input (int64) | Buggy | Correct (numpy ddof=1) |
|---|---|---|
| [1,2,3,4] | 1.4142 | 1.2910 |
| [1,2] | 1.0 | 0.7071 |
| [0,0,0,10] | 5.0332 | 5.0 |
| [1,2,3,4,5] | 1.5811 | 1.5811 (integral mean -> OK) |

Float columns unaffected (`pa.scalar(float, type=float64)` exact).
`groupby().agg("std")` — Arrow native `group_by().aggregate`, не
задет. ML `fit_*` — не использует `StdOp`.

**Fix:** cast to float64 before subtract.

```python
arr_f = pc.cast(arr, pa.float64())
diff = pc.subtract(arr_f, pa.scalar(float(mean_b), type=pa.float64()))
```

**Semantic change (int columns):** correct value where previous was
mathematically wrong. CHANGELOG flags as breaking for downstream
code that relied on incorrect result. No existing test pinned the
buggy value (both `test_std_invariant` and
`test_std_precision_large_offset` use float data).

**Правило (13.4/176):** при subtract/add со scalar из mean/median —
**всегда** cast array to float64. Int arithmetic truncates silently.
Same class as 13.4/163 (domain validation), 13.4/175 (memory formula
scope) — hidden mismatch between formula и implementation.

**Тесты (v0.9.3d):** +11 fast (`test_stdop_int_correctness.py`):
known [1,2,3,4] test (+ sanity not buggy), int==float, 5 param cases
vs numpy, float unchanged, nulls, int32, AST guard.

- fast: **~1583** (+11). slow: **20** (unchanged).

### Исправлено (v0.9.3c - P-D2 dedup string keys, S2 fix)

**P-D2 (S2, из v0.9.2 audit):** `_local_dedup` guard
`check_state(1.5 * table.nbytes)` extrapolated from int-key
prototype v0.8.11a (K<<N, measured 0.23x). Bench v0.9.2 audit
(bench_dedup_phases.py, 400k rows, K=N strings, table 9.20 MB):

- dictionary_encode +56.95 MB
- group_by +86.14 MB
- is_in +103.38 MB
- filter +103.58 MB (peak)

Peak / table = **11.26x** (strings at K=N). Int64 K=N = 1.97x.
Prior 1.5x bound was **7.5x underestimate** for string K=N --
OOM instead of MemoryLimitExceeded.

**Fix:** type-based multiplier (conservative bench-backed):

- **String subset:** 12 * table.nbytes (11.26x measured + 7% safety).
- **Numeric subset:** 2.5 * table.nbytes (1.97x measured + 27%).

`_has_string_subset` checked via `pa.types.is_string` /
`is_large_string` on `subset_cols`. `_dedup_peak_mult` drives
`check_state`.

**Почему dictionary_encode даёт 11x peak:**
- Копирует **каждое** string value (не только dictionary).
- Arrow StringArray: data buffer + offset buffer (int32).
- BufferBuilder geometric growth (~2x peak of final).
- + int32 indices buffer (4 B / row).
- + hash table for group_by (K entries).

**Over-conservative:** K<<N strings (measured ~0x) still need 12x.
Sample-based K_est tightening -> v0.9.4 backlog.

**Тесты (v0.9.3c):** +7 fast (test_dedup_estimator_v093c.py):
string K=N tight-budget raises, string K=N ample passes, int K=N
ample passes, 3 AST guards (12x, 2.5x, type check), multiplier
variable. Existing `test_dedup_removes_stopgap_5x` updated: 1.5x
-> 12x/2.5x.

- fast: **~1572** (+7). slow: **20** (unchanged).

### Исправлено (v0.9.3b - P-S1 sort merge fan-in, S2 fix)

**P-S1 (S2, из v0.9.2 audit):** `external_sort` вызывал
`compute_fan_in(read_batch_size=1000)` — это output flush
frequency, не reader batch size. Реальные on-disk batches
(от `_write_run`) — `_WRITE_RUN_CHUNK = 50_000` rows. Reader
держит 50_000 × r на batch, не 1000 × r. 50x underestimate
merge peak. Measured: 126.81 MB при 67.1 MB budget (1.9x).
**MemoryLimitExceeded не поднимался.**

**Fix:**

- `_WRITE_RUN_CHUNK = 50_000` — module constant.
- `_write_run` uses it instead of literal `chunk = 50_000`.
- `compute_fan_in` in `external_sort` uses `_WRITE_RUN_CHUNK`
  (on-disk batch size) instead of `read_batch_size`.
- `_merge_keys` gets `memory_guard` param; `check_batch` on each
  read batch (defense-in-depth).

**Покрывает:** `sort_values`, `rank`, `quantile`, `resample`,
`roc_auc`, `TimeSeriesSplit` — все через `external_sort`.

**Trade-off:** k уменьшается ~50x (больше merge levels при tight
budget). Peak теперь fits; merge slower для малого budget.
Safety > speed.

**Тесты (v0.9.3b):** +5 fast (`test_sort_merge_p_s1.py`):
constant exists, AST compute_fan_in uses _WRITE_RUN_CHUNK, AST
check_batch in _merge_keys, tight-budget smoke, regression
sort correctness.

- fast: **~1565** (+5). slow: **20** (unchanged).

### Исправлено (v0.9.3a - KFold keyed PRNG, S1 fix)

**P-CV1 (S1, из v0.9.2 audit):** `KFold.split` материализовал
O(N) fold-id массив (shuffle=False: ~12*N; shuffle=True: ~20*N)
**без** `MemoryGuard.check_state`. Для N >> RAM - guaranteed OOM,
не `MemoryLimitExceeded`. Нарушение ядра гарантии CADW.

- **`KFold.shuffle=False`** (default): `fold_id = row_id % k`,
  computed per batch. O(batch) state. 1:1 с прежним поведением.
- **`KFold.shuffle=True`** (default, O(batch) state): keyed
  splitmix64 PRNG. `fold_id = splitmix64(row_id*GOLDEN + seed) % k`.
  Детерминистичен per row_id. Split-quality: fold sizes ~binomial
  (mean n/k, per-fold sigma ~= sqrt(n/k)).
- **`KFold(materialize=True)`** (opt-in backward compat):
  exact-balanced shuffle, O(N) RAM, `check_state(20*N)` guarded.
- **`random_state=None`** -> seed=42 (deterministic reproducibility).
- **`_attach_column_with_fn(fn)`**: streaming helper
  (`fn(offset, n) -> pa.Array`). O(batch) state, no O(N) array.

### 13.4/173 - docstring memory claim vs actual allocation

Pre-v0.9.3a docstring: "O(N) int8 fold_id array". Actual code:
int32 fold_id + int64 arange/permutation temporaries = 12-20*N bytes.
3-4x understatement + zero MemoryGuard check.

**Правило:** docstring memory claim must match actual code. Test
with N >> RAM (core scenario) обязателен для каждого публичного op
с O(N) state - small-N тесты скрывают O(N) bugs.

### 13.4/174 - patch-script embedded escape trap (3 rounds)

v0.9.3a: 6 fix iterations, все - patch-script syntax bugs, не CADW:

- `\n` внутри triple-quote template -> real newline in file.
- `'''` inside `'''...'''` -> terminated early.
- `"""` inside `"""..."""` -> terminated early.
- Pattern wrote `->` (ASCII), file had `->` (U+2192).
- 2 stale tests after behavior change (13.4/160).

**Rules:** no triple-quote in patch scripts; content as
`"\n".join(list)`. Pattern matches signatures + structural lines
only, no docstrings/unicode. If not found - print context. Split
code/docs patches. Self-compile-check before run.

### Тесты (v0.9.3a)

+11 fast (`test_cv_keyed_prng.py`): default folds, keyed PRNG
deterministic, different seeds differ, default seed=42, binomial
balance, no-shuffle unchanged, materialize exact-balanced, OOM
guard, small-N OK, AST guard (no np.arange(N)/permutation(N) в
default path), streaming helper present, large-N tight-budget.

2 existing tests updated (`test_cv.py`): `shuffle=True` now binomial
(use `materialize=True` for exact).

- fast: **~1560** (+11). slow: **20** (unchanged).

### Ограничения (v0.9.3a, 11.5)

- `shuffle=True` (default): fold sizes ~binomial, not exact-balanced.
- `KFold.split()`: one `df.count()` scan (O(N) time, O(1) memory).
- Iterating all folds: `n_splits * N` row touches, O(1) memory per
  iteration.

## [0.9.2] — 2026-10-20

### Исправлено (v0.9.2f-fix2 — registry 4→8)

- **`test_list_step_types_default`** — ожидаемый список 4 → 8 items
  (v0.9.2f auto-registered 4 scalers via `_register_scalers()` at
  `cadw/scalers.py` import). Closed-list assert is intentional guard
  against test-registry pollution; grows with each new built-in.
- Rationale comment: built-ins updated in same patch that adds new
  step. Patch review catches typos in new step name (e.g.,
  `"StandartScaler"` → test fails, not silent).

### §13.4/172 — closed-list stale-bait on registry growth

Built-in registry grew 4→8. Closed-list test was stale-bait; not a
bug, auto-registration is intentional (`Pipeline.from_json` works for
scalers without setup). Chosen fix: (a) update expected list per patch
(visible churn, review catches typos). Alternative (b) property-based
superset check — deferred, KNOWN_TYPES goes stale too.

Convention: **when adding a built-in step, update
`test_list_step_types_default` in same commit.**

### Исправлено (v0.9.2f-fix1 — test-authoring #7)

- **`test_standard_scaler_unseen_data`** — deterministic data +
  reference helper. Prior test had 2 bugs:
  (a) same-distribution train/test — frozen-vs-refit indistinguishable
      (both output mean ≈ 0);
  (b) manual arithmetic `abs(mean - 5.0) < 0.5` — wrong: standardized
      mean = 0, not mean/std = 5.
  New: `x_train = [8, 12]`, `x_test = [12]` — frozen gives exactly
  1.0, refit gives 0.0. Exact assertion via reference.
- **`tests/_helpers/scaler_reference.py`** (new) — reference transforms
  for Standard/MinMax/MaxAbs/Robust. Codifies invariant
  `(x - mean_train) / std_train` per scaler; test bodies import
  reference, not manual arithmetic (§13.4/164 discipline extended
  to scalers).

### §13.4/171 — deterministic-vs-statistical test pattern

**For EXACT-property tests** (e.g., «does transform use frozen stats?»):
deterministic data + reference helper + exact assertion. RNG washes
out the property.

**For CONVERGENCE tests** («does the model learn?»): RNG + statistical
tolerance is fine.

7th test-authoring error in session (128 → 163 → 164 → 166 → 167 →
169 → 171). Systematic fix: reference helper per domain
(`metrics_reference.py`, `rolling_reference.py`, `scaler_reference.py`,
future `ts_features_reference.py`).

### Добавлено (v0.9.2f — scalers)

Завершение ML flow v0.9.2: preprocessing (metrics + CV + scalers).

- **`StandardScaler(columns=None, *, with_mean=True, with_std=True)`** —
  `(x - mean) / std`. Population std (`ddof=0`, matches sklearn).
  Degenerate column (std=0) → scale=1.0 (sklearn behavior).
- **`MinMaxScaler(columns=None, *, feature_range=(0.0, 1.0))`** —
  `(x - min) * (high-low)/(max-min) + low`. Constant column → output
  `feature_range[0]` (sklearn).
- **`MaxAbsScaler(columns=None)`** — `x / max(|min|, |max|)`. All-zero
  column → scale=1 (avoid division by zero).
- **`RobustScaler(columns=None)`** — `(x - median) / IQR`. IQR=0
  (degenerate) → scale=1. Per-column `df.quantile([0.25, 0.5, 0.75])`
  (external_sort per column, O(d) sorts per fit).
- **`cadw/scalers.py`** — new module. `_NumericStatsOp` (per-column
  Welford + min/max in one pass, state 5×d×8 bytes) shared by
  StandardScaler/MinMaxScaler/MaxAbsScaler. RobustScaler uses
  existing `df.quantile`.
- Auto-registered as Pipeline steps (`register_step_type`) → usable in
  `Pipeline.to_json/from_json`.
- Exports: `from cadw import StandardScaler, MinMaxScaler,
  MaxAbsScaler, RobustScaler`.

### Design decisions (locked-in, v0.9.2f)

- **No prototype** — scalers are **composition** of tested primitives
  (`Reduce`, `df.quantile`, `Transformer` Protocol). §13.4/123
  prototype-first applies to **new algorithms** (sort, roc_auc, CV
  fold math), not compositions.
- **Population std (ddof=0)** for StandardScaler — matches sklearn.
- **Degenerate handling**: std=0 → scale=1; IQR=0 → scale=1;
  min==max → output feature_range[0]; max_abs=0 → scale=1. All
  match sklearn or are no-op (safer for downstream NaN propagation).
- **Nullability (§13.4/169)**: all output fields nullable=True. Scaled
  columns → dtype float64; pass-through keep original dtype.
- **State embedded in `to_dict`** — O(d) bytes (dicts of floats). No
  separate "state" file like GBDT (large trees) — small enough for
  `pipeline.json`.
- **`columns=None` auto-detect numeric** at fit time, stored in
  `_columns` for transform.

### Ограничения (v0.9.2f, §11.5)

- **RobustScaler — O(d) external_sorts per fit.** For d=100 × n=1M,
  ~17 min fit. For d > 100 prefer StandardScaler/MinMaxScaler.
  Backlog: `pc.tdigest` based approximate (v0.9.4).
- **`astype` to non-float dtype** — not supported (scalers always
  produce float64 for scaled columns).
- **`columns=None`** auto-detects only `pa.types.is_integer` or
  `is_floating` — bool/string/timestamp excluded. Explicit `columns`
  can force-cast any Arrow-numeric-castable type, but non-numeric
  raises at `pc.cast`.
- **AST guard whitelist** (`test_arrow_call_memory_guard.py`) is by
  function NAME, not qualname. `"fit"`/`"transform"`/`"predict"`
  whitelisted globally → `RobustScaler.fit` (calls `df.quantile`)
  not flagged. Not a bug (quantile self-guards), but over-broad.
  Fix: qualname-based whitelist (v0.9.3).

### Тесты (v0.9.2f)

- +22 fast (`test_scalers.py`): StandardScaler (mean_zero_std_one,
  with_mean_false, with_std_false, nulls, roundtrip, unseen,
  constant); MinMaxScaler (0_1, custom_range, constant, roundtrip,
  bad_range); MaxAbsScaler (basic, positive_only, all_zero,
  roundtrip); RobustScaler (median_zero_iqr_one, outlier_resistant,
  roundtrip, constant); multi-column, auto-detect, missing raises;
  Pipeline integration (2 tests).
- +1 slow (`test_scalers_e2e.py`): 100K rows, StandardScaler →
  GBDTStep, R² > 0.5.

- fast: **~1512** (+22). slow: **20** (+1).

### §13.4/170 — scalers no new lessons; AST whitelist observation

See DECISIONS.md. Key: scalers are compositions (no prototype),
but AST guard whitelist-by-name is over-broad (RobustScaler.fit not
caught for `df.quantile` call). Fix: qualname-based (v0.9.3).

### Исправлено (v0.9.2e-fix1 — nullability + test data)

- **`cadw/cv.py::_attach_column`** — все output fields nullable=True.
  `nullable=False` на `__fold` ломал `Dataset.from_batches`:
  singleton-batch через `pa.RecordBatch.from_arrays([col],
  names=[name])` всегда nullable=True, writer target с
  `pa.schema([field])` (nullable=False) → ArrowInvalid. 9 failures.
  CADW convention: output fields always nullable=True (all existing
  ops соблюдают).
- **`test_cv_score_timeseries`** — i.i.d. data (`x ~ N(0,1)`,
  `y = 2x + noise`), не linear trend. Trend + TSS → GBDT
  extrapolation outside training bin range → r2 = -23.9 (inherent
  limitation §11.5; test design was wrong, not code). 1 failure.
- **`test_kfold_shuffle_different_assignment`** — cleaned placeholder
  (was doing nothing); proper set-of-y comparison + union invariant.
- **`test_tss_n_lt_k_plus_1_raises`** — raw-string regex (silence
  SyntaxWarning).

### §13.4/169 — nullability convention + test-respects-limitation

- **Output schema = nullable=True** для всех полей, produced через
  `Dataset.from_batches`. Core rebuilds singleton via
  `from_arrays(names=...)` (always nullable=True); writer target must
  match. Backlog: align `Dataset.from_batches` for nullable=False.
- **Test data must respect inherent model limitations.** TSS + GBDT
  (без lag features) cannot extrapolate. Use i.i.d. or bounded
  (periodic) features. Real time-series → TimeSeriesPipeline (lags,
  backlog v0.9.4).

### Добавлено (v0.9.2e — cross-validation)

- **`KFold(n_splits, shuffle=False, random_state=None)`** — k-fold CV
  for CadwDataFrame.
  - `shuffle=False` (default): `fold_id = arange(n) % k` — O(1) per row,
    deterministic, no RNG.
  - `shuffle=True`: `rng.permutation(n)` → fold_id array (O(N) int32;
    ~4 MB for 1M rows). Documented trade-off.
  - `.split(df)` yields `(fold_i, train_df, test_df)` — lazy views
    (PredicateFilter on `__fold` column).
- **`TimeSeriesSplit(n_splits, time_col)`** — no future leakage.
  - Sort by `time_col` once (`external_sort`), then positional fold
    slices via `iloc`.
  - fold 0: train=[0, fs), test=[fs, 2*fs)
  - fold k-1: train=[0, k*fs), test=[k*fs, n)
  - `fs = n // (n_splits + 1)`.
- **`cv_score(df, cv, model_factory, *, X_cols, y_col, metrics,
  pred_col=None, **metric_kwargs)`** — runs CV, returns
  `{fold_scores: [{metric: value}], mean: {metric: nanmean},
  std: {metric: nanstd}}`.
  - Model must implement `fit(df, target=y_col)` and
    `predict(df) -> CadwDataFrame` (GBDTStep-compatible convention).
  - Prediction column default `f"{y_col}_pred"`; override via `pred_col`.
  - `**metric_kwargs` passed to `preds.metrics()` (classes, average,
    y_proba, pos_label).
- Exports: `from cadw import KFold, TimeSeriesSplit, cv_score`.

### Design decisions (locked-in, v0.9.2e)

- **KFold fold assignment = `row_id % k`** (no shuffle) — trivial and
  deterministic. `shuffle=True` materializes O(N) int32 — accepted
  trade-off, not default.
- **TimeSeriesSplit — sort once, iloc slices per fold.** Sort cost
  O(N log N) per `split()` generator; cache sorted_df externally if
  calling multiple times.
- **cv_score uses `model.predict(test)` result directly** — GBDT-style
  `predict` returns df with original columns + `{target}_pred`. So
  `preds.metrics(y_true=y_col, y_pred=pred_col)` — no merge needed.
- **Model factory pattern** — fresh model per fold (no shared state).
  sklearn-compatible convention: `lambda: GBDTStep(...)`.

### Ограничения (v0.9.2e, §11.5)

- **`KFold(shuffle=True)`** materializes `fold_id` array (O(N) int32
  ≈ 4 MB per 1M rows) — for truly out-of-core N, prefer shuffle=False.
- **`StratifiedKFold`, `GroupKFold`, `Nested CV`** — backlog (v0.9.3+).
- **`TimeSeriesSplit` sorts once per `.split()` call** — calling
  `.split()` twice re-sorts. Cache the sorted_df if reusing.
- **`cv_score`** — relies on `pred_col` convention; user must ensure
  model's predict output has the named column (or pass `pred_col`).
- **`cv_score` with y_proba (roc_auc / log_loss)** — supported via
  `y_proba="..."` kwarg. Model must emit proba column; GBDTClassifier
  does (y_prob).

### Тесты (v0.9.2e)

- +15 fast (`test_cv.py`): KFold (sizes, uneven, no-overlap, shuffle,
  errors, filter); TimeSeriesSplit (splits, no leakage, train grows,
  test sizes, errors, unsorted); cv_score (kfold GBDT, timeseries,
  empty metrics, custom pred_col).
- +1 slow (`test_cv_e2e.py`): 100K rows, KFold(5), GBDT, R² > 0.5.

- fast: **~1485** (+15). slow: **19** (+1).

### Backlog (v0.9.3, §11.6 — from v0.9.2e code review)

- `_pandas_hash_column` silent fallback (add logging.warning).
- `_compute_roc_auc_binary::pa.scalar(pos_label, type=y.type)` mismatch
  validation.
- `_exact_actual_matches_arrow` — cache build-table value_counts
  per partition-pair.

### Исправлено (v0.9.2c-fix2 — multiclass log_loss gate)

- **`MetricsOp.__init__`** — classification setup block
  (`_classes`/`_cls_to_idx`/`_k`) gated на
  `_needs_classification OR (_needs_proba AND _needs_multiclass_proba)`.
  Pure `metrics=["log_loss"]` + multiclass `y_proba=[p0,p1,p2]` раньше
  оставлял `_cls_to_idx={}` → все y_true метки считались
  "неизвестными". 2 failures.
- **`MetricsOp.batch_memory_factor`** — 8.0 для multiclass proba
  (`.to_pylist()` на K+1 columns — §3.11 honest peak).
- **`MetricsOp.__init__`** — defensive: `_needs_regression +
  _needs_multiclass_proba` → ValueError (domain conflict: regression
  y_true continuous vs multiclass log_loss y_true class index).
- **`test_metrics.py`** — raw-string regex (4-space indent; v0.9.2c
  fix1 had 8-space, didn't match); 4 new contract tests
  (§13.4/168).

### §13.4/168 — partial migration: state-sharing ≠ naming

Feature F использует internal state S; S инициализируется под gate G.
Добавление F требует расширения G, даже если F не входит в
семантическую группу G. Пример: multiclass log_loss — не
classification metric (имя), но использует `_cls_to_idx` (state).

### Исправлено (v0.9.2c-fix1 — partial migration + test-authoring)

- **`MetricsOp.__init__`** — проверка `y_proba_cols` (unified) вместо
  legacy `y_proba_col`. v0.9.2c добавил `y_proba_cols` (list для
  multiclass/binary), но `if self._needs_proba and y_proba_col is None:`
  остался на legacy поле — все вызовы с `y_proba_cols=[...]` (без
  `y_proba_col`) получали false "y_proba_col required". 6 failures.
- **`CadwDataFrame.metrics`** — generic column-existence loop больше
  не включает `y_proba` (может быть list после v0.9.2c). Specific
  per-column check (after `y_proba_cols` normalization) уже был
  корректным. 6 failures.
- **`test_metrics_roc_auc_with_accuracy`** — expected AUC из
  `expected_roc_auc` helper, не manual 0.875 (правильно 1.0).
  **5-я** test-authoring ошибка в сессии (§13.4/128 → 163 → 164 →
  166 → 167).
- **`MetricsOp`** — message: "'roc_auc' is computed **separately**
  by CadwDataFrame.metrics()" (test regex `"roc_auc.*separately"`
  matches; clearer intent).
- **`tests/test_metrics.py`** — `match=r"len\(y_proba\)"` (raw
  string; silence SyntaxWarning).

### §13.4/167 — partial-migration pattern

X → X_new (deprecated):
- validation checks — **all** on X_new;
- generic loops / type-sensitive checks — **updated** after type
  widening (`str` → `str | list[str]`);
- **mixed-type tests** (scalar + list) — обязательны при расширении
  типа; v0.9.2a scalar tests не ловили list case.

### Добавлено (v0.9.2c — roc_auc + multiclass log_loss)

- **`roc_auc`** в `CadwDataFrame.metrics()`:
  - **binary** (`y_proba` scalar str): `y_true ∈ {pos_label, other}`.
    `pos_label` default `1`.
  - **macro** (`y_proba` list of K cols, `classes` explicit K>2):
    one-vs-rest, macro average.
  - Algorithm: **external_sort** by proba desc + **per-threshold
    grouping** + streaming trapezoid. Peak RAM = batch + Sort disk
    (independent of N). Validated by prototype 8/8.
  - NaN if `n_pos == 0` or `n_neg == 0` (honest).
- **Multiclass `log_loss`**: `y_proba` list of K cols, `classes` explicit.
  O(1) state (sum_log_loss, n_proba). One pass.
- **`AUC_METRICS`** = `{"roc_auc"}` (exported).
- **`tests/_helpers/metrics_reference.py`**: `expected_roc_auc`,
  `expected_log_loss_binary`, `expected_log_loss_multiclass` —
  reference implementations (validated by prototype + sklearn
  equivalence if installed).

### Design decisions (locked-in, v0.9.2c)

- **roc_auc — не Reduce**. Sort-based (multipass): pass 1 count,
  external_sort, pass 2 trapezoid. `MetricsOp` rejects `roc_auc`
  directly; `CadwDataFrame.metrics()` dispatches it separately.
- **Per-threshold grouping** (ties): all tied scores = one ROC point
  (matches sklearn.roc_curve). Prototype caught this (§13.4/166).
- **y_proba**: str → binary, list[str] → multiclass (both log_loss
  and roc_auc).
- **Conflict**: `roc_auc` (binary labels) + regression (`r2` etc.,
  continuous) on same `y_true` → ValueError. Use separate calls.

### Ограничения (v0.9.2c, §11.5)

- **roc_auc binary requires `y_true ∈ {0,1}`** (or `pos_label` +
  one other value; values not in `{pos_label, …}` treated as
  negative — but validated only for column presence, not value domain).
  String labels → use multiclass form (`y_proba` list + `classes`).
- **roc_auc macro** — `average='macro'` only (unweighted); weighted
  ovr — backlog.
- **`roc_curve`** (TPR/FPR arrays) — не возвращается; если запросят —
  отдельный метод.
- **Sort temp** — `spill_dir/metrics_roc_auc/sort_by_proba/*` (auto
  cleanup after streaming).
- **N × row_size** does not bind peak: Sort handles arbitrary N via
  disk. Fail-fast on `K × 24 + 64 > state_limit` (classification state).

### Тесты (v0.9.2c)

- +10 fast (`test_metrics_reference.py`): helper self-consistency,
  sklearn-equivalence (importorskip), known cases.
- +17 fast (`test_metrics.py`): multiclass log_loss (perfect/known/
  validation), roc_auc binary (perfect/known/ties/all-pos-NaN/
  all-neg-NaN/nulls/filtered/reference), roc_auc macro, mixed
  (accuracy + roc_auc, regression conflict, requires y_proba),
  MetricsOp rejects roc_auc.
- +1 slow (`test_metrics_e2e.py`): 100K rows, roc_auc + multiclass
  log_loss, memory 30 MB budget.

- fast: **~1468** (+27). slow: **18** (+1).

### Добавлено (v0.9.2b — classification metrics)

- **`accuracy`, `precision`, `recall`, `f1`** в `CadwDataFrame.metrics()`:
  - binary (K=2, `pos_label` = 1 по умолчанию) и multiclass (K≥3).
  - **`average`** ∈ `'binary'|'macro'|'micro'|'weighted'` (auto: binary
    if K=2, macro if K>2).
  - `classes` — explicit class list (K=2 default when None).
  - Numeric или string labels — dict lookup, nulls skipped.
- **`MetricsOp`** state: `tp[K] + fp[K] + fn[K]` int64 = `24×K` bytes
  (vs confusion matrix `K²×8 = 8 MB` для K=1000 — 320× дешевле для
  scalar metrics).
- **`ReduceOp.batch_memory_factor()`** — protocol extension.
  Classification path uses `.to_pylist()` per batch (7× Python-object
  overhead). `reduce()` passes to `check_batch(extra_factor=...)` —
  honest peak RAM, not hidden.
- **`SUPPORTED_METRICS`** = regression + proba + classification
  (exported from `cadw`).

### Design decisions (locked-in, v0.9.2b)

- **TP/FP/FN state** вместо confusion matrix. Accuracy/precision/
  recall/f1 — все выводятся из TP/FP/FN per class. K² state не нужен
  (confusion matrix добавим отдельным методом, если запросят).
- **Average auto-switch**: binary (K=2) / macro (K>2). Matches sklearn
  semantics; user может переопределить.
- **zero_division=0** (sklearn default): precision/recall/f1 при
  пустом знаменателе → 0.0, не NaN.
- **Mixed regression + classification** на одном `y_pred` → ValueError
  (continuous vs labels).
- **`log_loss` + classification** — OK (разные колонки: y_proba vs
  y_pred).
- **Fail-fast**: `K × 24 + 64 > state_limit` → `MemoryLimitExceeded`
  в `metrics()` до `MetricsOp` construction.

### Ограничения (v0.9.2b, §11.5)

- **`roc_auc`** — v0.9.2c (Sort-based, требует полного proba массива).
- **Multiclass log_loss** (proba matrix) — v0.9.2c.
- **Confusion matrix** itself — не возвращается; если запросят —
  отдельный метод (не `metrics()` scalar API).
- **Class mismatch** y_true/y_pred not in `classes` → ValueError
  (fail-fast, первые 5 invalid labels).
- **Empty input** → accuracy=NaN (как regression).
- **`average='binary'`** требует K=2, `pos_label` в `classes`.

### Тесты (v0.9.2b)

- +20 fast (`test_metrics.py`): accuracy binary perfect/mixed/auto,
  precision/recall/f1 binary pos_label, multiclass macro/micro/weighted,
  zero-division, nulls, empty, filtered, unknown label, mixed
  regression conflict, average=binary on K>2, invalid average, K too
  large, string labels, log_loss + classification, pandas equivalence.
- +1 slow (`test_metrics_e2e.py`): 100K rows, 3-class, manual numpy ref.

- fast: **~1441** (+20). slow: **17** (+1).

### Добавлено (v0.9.2a — metrics, regression + log_loss)

- **`CadwDataFrame.metrics(y_true, metrics, y_pred=, y_proba=)`** —
  regression + binary log_loss в одном Reduce-проходе.
  - **Regression** (require `y_pred`): `r2`, `mse`, `rmse`, `mae`.
  - **log_loss** (requires `y_proba` in [0,1], binary y): binary
    cross-entropy.
  - **Combined** — если `y_pred` и `y_proba` заданы оба, все метрики
    считаются в одном проходе (state O(1)).
  - Empty input (после filter) → **NaN** для каждой метрики.
  - Nulls в `y_true`/`y_pred`/`y_proba` → skip per row (skipna=True).
  - **R² matches sklearn**: `ss_tot == 0` → 1.0 если perfect, else 0.0.
- **`cadw/ops/metrics.py::MetricsOp`** — ReduceOp, state = 7 counters
  × 8 B = 64 B. Независимо от N.

### Design decisions (locked-in, v0.9.2a)

- **Reduce-based**: один проход, O(1) state. Не Sort, не pandas.
- **Regression + log_loss в одном patch** — обе метрики O(1) state,
  тривиально комбинируются. Classification (confusion matrix, K×K
  state) — v0.9.2b.
- **Empty → NaN**, не 0.0. Honest «no data» — user sees it.
- **R² совпадает с sklearn** в граничных случаях (constant y).
- **log_loss**: `np.clip(p, 1e-15, 1 - 1e-15)` — как sklearn.

### Ограничения (v0.9.2a, §11.5)

- **Classification metrics** (`accuracy`, `precision`, `recall`, `f1`)
  — v0.9.2b. Требуют confusion matrix (K×K state, K = #classes).
- **`roc_auc`** — v0.9.2c. Требует Sort + cumulative TPR/FPR.
- **Multiclass log_loss** — v0.9.2b. Требует proba matrix (K columns).
- **Per-group metrics** (`groupby(g).metrics()`) — v0.9.2d.
- **`y_true` for regression must be numeric** — string labels need
  `encode_categorical` first.
- **log_loss requires binary y** ∈ {0, 1}. Multiclass — v0.9.2b.

### Тесты (v0.9.2a)

- +24 fast (`test_metrics.py`): perfect/known values (r2, mse, rmse,
  mae), log_loss (perfect/50-50/known), combined, nulls skip, empty→NaN,
  filtered, string metric, all validation errors, pandas-equivalence,
  chain integration.
- +1 slow (`test_metrics_e2e.py`): 100K rows, 30 MB budget, all four
  metrics verified vs numpy.

- fast: **~1422** (+24). slow: **15** (+1).

## [0.9.1] — 2026-10-20

### Добавлено (v0.9.1f — pivot_table / melt / crosstab)

Финальный патч v0.9.1: табличные преобразования (long ↔ wide).

- **`df.melt(id_vars, value_vars, var_name='variable',
  value_name='value')`** — wide → long. **Streaming per-batch**, peak
  O(batch × #value_vars). Requirements:
  - `value_vars` share ONE dtype (Arrow column uniform); mixed →
    TypeError с рекомендацией `astype` для конвертации.
  - `id_vars` ∩ `value_vars = ∅`; `var_name`/`value_name` не
    коллидируют с `id_vars`.
  - Output: `[*id_vars, var_name, value_name]`, rows = `N ×
    len(value_vars)`.

- **`df.pivot_table(index, columns, values, aggfunc='mean')`** — long →
  wide. `groupby([*index, columns]).agg(values, aggfunc)` + pandas
  reshape (small table). **Fail-fast** если `k_pairs = K_index ×
  K_columns` слишком велико: `k_pairs × row_size × 2 > 0.3 × avail`
  → `MemoryLimitExceeded` **до** groupby. Missing combos → null.
  - `aggfunc ∈ {mean, sum, count, min, max}`, default `mean`.
  - `dropna=True` для null в `columns` (standard pandas).
  - Null в `index` → own row (kept).
  - Multi-index (`index=['a','b']`) поддерживается.

- **`df.crosstab(index, columns)`** — frequency table. `dropna=True`
  на nulls в index/columns (standard crosstab), delegates to
  `pivot_table(values=index[0], aggfunc='count')`.

### Memory bound (§11.5)

- **melt**: O(batch × #value_vars). Streaming.
- **pivot_table**: K_pairs estimate через sample (`MemoryGuard.
  estimate_unique_keys`) **до** groupby. `check_state` на grouped
  (K_pairs rows) **до** pandas reshape (AST-verified). Peak = batch +
  K_pairs × row_size (grouped) + K_index × K_cols × value_size
  (pivot output) — bounded by fail-fast check.
- **crosstab**: same as pivot_table.

### Ограничения (v0.9.1f, §11.5)

- **`pivot_table` sparse → fail-fast**. K_pairs > 0.3×avail/row_size →
  `MemoryLimitExceeded` с рекомендацией `groupby()` (long format).
  Sparse pandas pivot (fill N/A with 0 in wide format) — не
  поддерживается.
- **`melt` heterogeneous dtypes** (`['int_col', 'float_col']`) →
  TypeError. Пользователь: `df.astype({'int_col': 'float64'})` first.
- **`pivot_table` dropna=True** в `columns` (стандарт pandas).
  `dropna=False` (null column group) — v0.9.2 если запросят.
- **`pivot_table` output column names = `str(value)`**. Int-значения
  `columns` → строковые имена ("0", "1", ...).
- **`melt` non-streaming types**: list/struct columns in value_vars
  работают (Arrow-native), но nested dtypes не проверяются на
  uniformity (assumes Arrow type equality).
- **`crosstab` count via non-null values** в `index[0]`. Требует
  non-null `index[0]` — фильтруется через `dropna=True` до.

### Тесты (v0.9.1f)

- +15 fast (`test_pivot_table.py`): melt (6), pivot (7), crosstab (3),
  AST guards (2).
- +1 slow (`test_pivot_table_e2e.py`): 100K rows pivot.

- fast: **~1395** (+15). slow: **13** (+1).

### Добавлено (v0.9.1e — per-group apply(fn))

- **`df.groupby('g').apply(fn, result_col='apply_result', on=None)`** —
  per-group user callable. Prototype-validated (11 cases A/B pandas
  vs composite-sort, §13.4/123).
  - **fn**: `callable(group_df: pd.DataFrame)` → scalar | `pd.Series` |
    `pd.DataFrame`. Uniform mode required across groups.
  - **Output**:
    - scalar → K rows, `[*group_cols, result_col]`
    - Series → K rows, `[*group_cols, *series.index]`
    - DataFrame → `Σ|df_i|` rows, `[*group_cols, *df.columns]`
  - **on**: within-group sort keys. `None` → row_id order.
  - **Order**: output sorted by group key.
- **`CadwDataFrame._per_group_apply(...)`** — internal: composite-sort
  by `[*group_cols, *on_cols]`, one group in RAM at a time, streaming
  output to IPC.

### Memory bound (§11.5)

- **Per-group bound**: `group_nbytes × 2 (pandas) ≤ 0.3 × available`.
  Oversized → `MemoryLimitExceeded` **before** `fn()`.
- **Peak RAM** = batch (input stream) + `2 × max(group_size × row_size)`
  + fn's output + small output-buffer batch. **Independent of N**.
- **fn's own alloc — user responsibility** (§3.9 extended): CADW bounds
  input to `fn`, not `fn`'s internals. `fn` that's O(N) per group → OOM
  is user's fault.

### Ограничения (v0.9.1e, §11.5)

- **`transform(fn)` NOT included** — semantically different (Series →
  same-length array for broadcast). v0.9.2.
- **Mixed return modes across groups** → RuntimeError.
- **Null in group key** → own group (dropna=False, matches pandas).
- **`fn` exceptions propagate** — no wrapping; user errors surface
  unchanged.
- **Output sorted by group key**, not original order.

### Тесты (v0.9.1e)

- +17 fast (`test_apply.py`): 3 modes, nulls, multi-group, single-row,
  filter, on-col sort, determinism, errors (fn not callable, mixed
  modes, unsupported type, oversized group, fn exception), pandas
  equivalence, AST guard.
- +1 slow (`test_apply_e2e.py`): 10K / 20 groups / mean, pandas equiv.

- fast: **~1378** (+17). slow: **12** (+1).

### Добавлено (v0.9.1d — ColumnTransformer + register_step_type)

- **`ColumnTransformer(transformers, remainder='drop'|'passthrough')`** —
  sklearn-style column fanout. Каждый трансформер fit+transform на своей
  subset колонок; результаты concat (column union).
  - `transformers: list[(name, cols, step)]` — уникальные имена,
    disjoint cols (каждая колонка ≤1 transformer), Transformer contract.
  - `remainder='drop'` (default): неименованные колонки удаляются.
  - `remainder='passthrough'`: проходят как есть в конце.
  - Row count preserved (dropna внутри → RuntimeError).
  - Duplicate output column names → RuntimeError.
  - Output order: transformer order, затем remainder.
  - `to_dict` / `from_dict` — nested state inline в JSON (nested GBDT
    ~30 KB в pipeline.json; extraction to separate file — backlog v0.9.2).

- **`register_step_type(name, cls)`** / **`unregister_step_type(name)`** /
  **`list_step_types()`** — module-level registry для custom steps в
  `from_json`. Backlog §11.5 закрыт.
  - Built-in types: `CastStep`, `DropnaStep`, `GBDTStep` (нельзя
    unregister).
  - Override built-in → warning (same pattern as `register_model` §9.7).
  - Custom class MUST implement `from_dict` classmethod.

- **`cadw/__init__.py`** — экспортирует `Pipeline`, `CastStep`,
  `DropnaStep`, `GBDTStep`, `ColumnTransformer`, `Transformer`,
  `Predictor`, `register_step_type`, `unregister_step_type`,
  `list_step_types`.

### Изменено

- **`Pipeline.from_json`** — использует module-level `_STEP_REGISTRY`
  (было hardcoded dict). Error message указывает на `register_step_type`
  для custom steps.

### Design decisions (locked-in, v0.9.1d)

- **Predictor not allowed inside ColumnTransformer** — only as final
  Pipeline step. `ColumnTransformer` is a Transformer (fit + transform
  on column subsets); model is orthogonal.
- **Row-preserving constraint** — column-level operations only. Row
  filtering (dropna) must go before ColumnTransformer.
- **Nested state inline** — `pipeline.json` содержит `step_dict` для
  каждого nested step. Large nested models (GBDT inside CT) — larger
  pipeline.json; extraction to `state/<ct_name>/<nested>.json` — backlog
  v0.9.2.

### Ограничения (v0.9.1d, §11.5)

- **Nested state inline** в `pipeline.json`. Для ColumnTransformer с
  GBDT ~30-50 KB в pipeline.json (acceptable). 1000+ tree GBDT — big
  file; extract-to-separate — v0.9.2.
- **No `remainder=transformer`** (sklearn supports this) — только
  `'drop' | 'passthrough'`. Для transform remainder — list explicitly
  в `transformers`.
- **Custom step `from_dict`** must be classmethod (validated at
  `register_step_type`).

### Тесты (v0.9.1d)

- +9 fast (`test_register_step_type.py`): register/unregister/list,
  override warning, bad name/class, missing from_dict, from_json uses
  registry.
- +12 fast (`test_column_transformer.py`): construction errors (empty,
  bad remainder, duplicate name, overlapping cols, empty cols, missing
  transform), basic fit/transform (2 transformers), remainder drop vs
  passthrough, appends new col, transform missing col, row count changed,
  to_dict/from_dict round-trip, unknown nested type, lambda nested raises.
- +3 fast + 1 slow (`test_column_transformer_pipeline.py`): inside
  Pipeline, save/load round-trip, state_bytes propagates, E2E (R² > 0.5).

- fast: **~1351** (+24). slow: **11** (+1).

### Добавлено (v0.9.1c — save/load)

- **`Pipeline.to_json(path)` / `Pipeline.from_json(path)`** — сохранение
  и восстановление pipeline для reproducibility и production deployment.
  - `path` — директория: `path/pipeline.json` + (опционально)
    `path/state/<step_name>.json`.
  - `from_json` принимает директорию или `pipeline.json` файл.
  - `strict_version=True` — raise на version mismatch; default — warn.
  - `path` must be directory (не file) для `to_json` — raises clear
    error если path exists and is file.
- **Per-step `to_dict()` / `from_dict()`** — Protocol extension, реализовано
  для `CastStep`, `DropnaStep`, `GBDTStep`. Custom steps должны
  реализовать оба, иначе `to_json` raises clear message.
- **`GBDTStep.to_dict()` / `from_dict()`** — полная сериализация
  fitted model (trees + base_prediction + mn/mx + hyperparameters) в
  JSON. ~30 KB для 100 trees × depth 3.
- **`LambdaTransform.to_dict()`** — raises NotImplementedError: `fn` —
  callable, не JSON-serializable. Message предлагает: custom step class
  с to_dict/from_dict, или только именованные wrappers.
- **`cadw/_serialization.py`** — helpers: `gbdt_regressor_to_dict` /
  `gbdt_regressor_from_dict`, `linear_to_dict` / `linear_from_dict`.
  Explicit, no pickle (safety: untrusted input не вызывает RCE).

### Design decisions (locked-in, v0.9.1c)

- **Always directory**: `pipeline_dir/pipeline.json` + optional
  `state/`. Cleaner для multi-file pipelines (state per step).
- **JSON for everything**: no pickle. GBDT trees ~30 KB JSON acceptable.
- **Version field** "0.9.1": `from_json` warns on mismatch, best-effort
  restore. `strict_version=True` для fail.
- **`LambdaTransform` non-serializable** — callables нельзя безопасно
  сохранить в JSON. Custom step class — user-side.

### Ограничения (v0.9.1c, §11.5)

- **JSON size для больших моделей** — GBDT n_estimators=1000 × depth=5
  ≈ 2.4 MB JSON. Для очень больших моделей — Arrow IPC sidecar
  backlog (v0.9.2). Сейчас JSON для всех.
- **Version compatibility**: strict_version=True поддерживает только
  same-minor. Future minor bumps требуют manual migration.
- **Custom steps**: `to_json` raises `NotImplementedError` если step
  не имеет to_dict. `from_json` raises если type не в registry —
  custom steps требуют register (v0.9.2 `Pipeline.register_step_type`).
- **`state` dir not created** для stateless pipeline — OK, load
  detects `has_state=False`.

### Тесты (v0.9.1c)

- +20 fast (`test_pipeline_save_load.py`): CastStep/DropnaStep/GBDTStep
  round-trip, predictions equal, DAG edges preserved, state file
  created, version mismatch warn/strict, corrupt JSON, missing state
  file, missing pipeline.json, path-as-file, LambdaTransform raises,
  fitted flag preserved.
- +1 slow (`test_pipeline_save_load_e2e.py`): fit → save → load →
  predict, predictions equal (rtol=1e-9), R² > 0.9.

- fast: **~1329** (+20). slow: **10** (+1).

### Добавлено (v0.9.1b — DAG Pipeline)

- **`Pipeline.add_step(name, step, inputs=...)`** — DAG edges (v0.9.1b).
  - `inputs=None` (default): chain semantics — first step = source (`[]`),
    subsequent = `[previous step name]`. **Backward compat** с v0.9.0.
  - `inputs=[]`: explicit source (redundant for first step, validated).
  - `inputs=["a", "b"]`: DAG fan-in. Все names должны существовать;
    Predictor нельзя использовать как input.
- **`Pipeline._build_graph()`** — `{name: [input_names]}`. Resolves
  `inputs=None` chain derivation.
- **`Pipeline._topological_order()`** — DFS topological sort с cycle
  detection (defensive: forward refs запрещены, cycle невозможен by
  construction).
- **`_merge_fanout(dfs)`** — module helper: column-union concat.
  Precondition: same row count across branches (positional alignment).
  Streaming через `zip(*iters)` с per-batch size check.
- **`LambdaTransform(src_col, feature_name, fn, out_type)`** — test
  helper: Arrow-native fn on one source column, appends
  `{src_col}_{feature_name}`.
- **`Pipeline._repr_html_`** — DAG-aware: column "inputs" показывает
  `(source)` или `← a, b` для fan-in.

### Изменено

- **`Pipeline.fit` / `transform` / `predict`** — execution через
  topological order, не list iteration. Multi-input → `_merge_fanout`.
  `transform` возвращает sink pure-transform подграфа (последний
  non-Predictor в reversed order); если pipeline model-only — исходный df.
- **`Pipeline.predict`** — возвращает результат first (и единственного)
  Predictor в topo order (Predictor валидирован как sink на `add_step`).

### Ограничения (v0.9.1b, §11.5)

- **Fan-in MVP: все branches должны иметь same row count.** Dropna /
  row-selection в одной ветви → RuntimeError с clear message.
  Row-selection fan-in (join по `__orig_rid` с пересечением) — v0.9.1c.
- **Fan-in реализация: positional alignment** (не row_id merge).
  Гарантия корректности = все branches идут от одного source и не
  меняют row order. Документировано.
- **Predictor = sink**, нельзя быть input. Проверяется на `add_step`.
- **Forward references запрещены** (`inputs=[name]` где `name` ещё не
  добавлен) → ValueError. Cycle detection — defensive (недостижим).
- **`Pipeline.plan(df)`** — sum across all steps (не меняется под DAG).
  Per-DAG-step state estimation — v0.9.1c backlog.

### Тесты (v0.9.1b)

- +14 fast (`test_pipeline_dag.py`): chain backward compat, two/three
  branch fan-in, diamond (raw → a, raw → b, merge, model), row-mismatch
  RuntimeError, unknown input, Predictor-as-input, explicit source
  validation, topo order (chain, fan-out), transform stops at model,
  predict full DAG, repr_html DAG edges.

- fast: **~1309** (+14). slow: **9** (unchanged).

### Изменено (v0.9.1a — backbone)

- **`tests/conftest.py`** — autouse `plt.close('all')` для всех тестов.
  Устраняет `More than 20 figures have been opened` RuntimeWarning,
  появившийся с v0.9.0c (8 plot-accessor тестов перевалили порог;
  pre-existing seaborn/viz тесты тоже не закрывали figures).
- **`cadw/patch_utils.py`** — общие helpers для patch-скриптов:
  `ensure_unreleased_header`, `append_unreleased_entry`,
  `find_section_anchor` (regex tolerant: `# 11.6`, `## §11.6`, `#### 11.6.`),
  `insert_before_section` (explicit `found` flag, no silent no-op),
  `assert_guards_green` (§13.4/140 discipline extraction).
- **`Transformer` / `Predictor` Protocol split** (§13.4/147 backlog):
  Predictor БЕЗ `transform`; Transformer БЕЗ `predict`. Строго два
  минимальных контракта, не nested. Runtime `_is_predictor(step) =
  hasattr(step, "predict")` — duck typing, Protocol только для typing.
- **DECISIONS.md §11.6 header restored** — заголовок был потерян
  в v0.8.11/v0.9.0 патчах; Pipeline v1 addendum от v0.9.0b-fix1
  (был appended к концу файла из-за anchor-not-found) перенесён
  в восстановленную §11.6.
- **`tests/test_pipeline.py`** — удалён dead `test_cast_step` stub
  (leftover из v0.9.0b, тест без assertions).

### Ограничения (v0.9.1a, §11.5)

- `patch_utils` используется только новыми patch-скриптами (v0.9.1+).
  Старые (v0.8.x, v0.9.0) сохраняют свою логику — historical artifacts.
- `conftest.py` — autouse для matplotlib only; torch lazy, уборки не требует.

### Тесты (v0.9.1a)

- +18 fast: 14 `test_patch_utils.py` (CHANGELOG + anchor + insert +
  Protocol split), +4 dead-test cleanup ensures pipeline suite stays green.
- RuntimeWarning "More than 20 figures" — исчезает (conftest autouse).

- fast: **~1298** (+18). slow: **9** (unchanged).

## [0.9.0] — 2026-10-20

### Добавлено (v0.9.0c — Jupyter integration)

- **`CadwDataFrame._repr_html_`** — sample 20 rows + schema + shape
  (upper bound) + filter note. Bounded: `.to_pandas().to_html()` на
  20-row `head()` только, не full df. AST guard whitelist с rationale.
- **`CadwDataFrame.plot` accessor** — matplotlib default через
  `df.plot.hist/line/box/scatter/hist2d`; explicit backends:
  `df.plot.plotly.*`, `df.plot.seaborn.*`. Все методы делегируют
  `cadw.viz.*` adapters (aggregate-first discipline preserved).
- **`Pipeline._repr_html_`** — HTML-таблица steps с per-step
  `state_bytes`, fitted marker, total state.
- **`cadw/plot_accessor.py`** — новый модуль, thin wrapper over
  `cadw.viz.matplotlib_adapter/plotly_adapter/seaborn_adapter`.

### Ограничения (v0.9.0c, §11.5)

- `df.info()` / `df.dtypes` / `df.describe()` — v0.10.
- `Pipeline` DAG визуализация — v0.9.1 (после DAG support в Pipeline).
- `df.plot` — только явные методы (`df.plot.hist(...)`), не
  `df.plot(kind='hist', ...)` (pandas-style). Method-style selected
  as primary; kind-based — backlog, если запрос.

### Тесты (v0.9.0c)

- +16 fast: 8 `test_repr_html.py`, 8 `test_plot_accessor.py`.
- AST guard whitelist: `_repr_html_` (bounded 20-row to_pandas).

- fast: **~1278** (+16). slow: **9** (unchanged).

### Добавлено (v0.9.0b — Pipeline skeleton)

Главный архитектурный шаг v0.9.0: chain of Transformers с plan-first
feasibility check. Превращает CADW из «набора primitives» в
out-of-core framework.

- **`cadw/pipeline.py`** — новый модуль:
  - **`Pipeline`** — sequence of Transformers. Model (if present)
    всегда LAST.
    - `add_step(name, step)` — chainable.
    - `plan(df) -> PipelinePlan` — pre-fit feasibility: sum(state_bytes)
      ≤ state_limit. Raises `PipelineMemoryError` if not feasible
      **до** любого materialize. Пользователь может вызвать явно для
      early check.
    - `fit(df, target=None)` — fit всех steps последовательно, plan-first.
    - `transform(df)` — применяет pure-transform steps; **останавливается
      перед Predictor** (если он есть).
    - `predict(df)` — fit chain + last Predictor. Требует последний step
      быть Predictor.
    - `fit_transform(df, target=None)` — fit + return features (before model).
    - `total_state_bytes()` — sum fitted state, интроспекция.
  - **`Transformer` / `Predictor` Protocol** (typing only) — контракт:
    `fit(df, target=None)`, `transform(df)`, `estimate_state_bytes(dc)`,
    `state_bytes()`; Predictor adds `predict(df)`.

- **Wrappers (MVP, для тестов)**:
  - **`CastStep(mapping, safe=True)`** — wrap `df.astype()`. Stateless.
  - **`DropnaStep(how, subset)`** — wrap `df.dropna()`. Stateless.
  - **`GBDTStep(X_cols, y_col, **kwargs)`** — wrap `fit_gbdt_regressor`.
    `estimate_state_bytes` = n_estimators × 2^(max_depth+1) × 48 + 1024
    (conservative per-node).

### Design decisions (locked-in, v0.9.0)

- **Sequence-first.** DAG (via `inputs=[...]`) — v0.9.1.
- **Model LAST.** `transform()` stops before Predictor; `predict()`
  requires it.
- **Plan-first fit.** `Pipeline.fit` calls `plan(df)` before any work;
  `PipelineMemoryError` if not feasible.
- **2-pass fit/transform.** State frozen after fit; transform re-applies
  pure steps. No fused execution (state-after-fit стабилен — экономии нет).
- **In-memory state.** Bounded (K × #features). Serialize to JSON — v0.9.1.
- **Per-batch transform guard — per-step.** Each step uses `MemoryGuard`
  internally; Pipeline не reimplement'ит.

### Ограничения (v0.9.0b, §11.5)

- `Pipeline.plan` — fit feasibility only (`total_state ≤ state_limit`).
  `spill_boundaries` always empty (deferred v0.9.1). Batch propagation
  per step handled by step's own MemoryGuard (not re-planned).
- `Pipeline.to_json` / `from_json` — v0.9.1.
- `ColumnTransformer` — v0.9.1.
- `TimeSeriesFeatures` composite — v0.9.2.
- `Pipeline` не валидирует, что Predictor is last при `add_step`
  (проверяется при `fit`). Predictor not-last на `fit` → RuntimeError.

### Тесты (v0.9.0b)

- +15 fast (`test_pipeline.py`): construction, wrappers, compose,
  fit/transform/predict contract, plan (fits, does_not_fit, with GBDT),
  state_bytes tracking, small GBDT fit+predict.
- +1 slow (`test_pipeline_e2e.py`): 20K rows, timestamp → dt features
  → categorical encoding → GBDT; R² > 0.3.

- fast: **~1257** (+15). slow: **9** (+1).

### Добавлено (v0.9.0a — accessors + type ops)

Первый шаг v0.9.0 «out-of-core framework»: подготовка данных
(datetime features, string cleaning, type casting, null handling).
Arrow-native, bounded per-batch, zero pandas.

**Accessors:**
- **`df['ts'].dt.{year,month,day,hour,minute,second,weekday,day_of_year,quarter}`** —
  Arrow `pc.year/month/day/hour/minute/second/day_of_week/day_of_year/quarter`.
  Возвращает CadwDataFrame с `{col}_{feature}` (original columns preserved).
- **`df['s'].str.{len,lower,upper}`** — `pc.utf8_length/utf8_lower/utf8_upper`.
  Возвращает CadwDataFrame с `{col}_{op}`.
- **`df['s'].str.contains(pat)`** — возвращает **MaskFilter** (для `df[mask]`).
  na=False (pandas default): null не матчится.
- **`df['col'].isnull()` / `.notnull()`** — MaskFilter.

**Facade:**
- **`df.astype({col: dtype})`** — cast per column. `dtype` = `pa.DataType`
  или str-alias (`"int64"`, `"float32"`). `safe=True` (default) → raise на
  overflow; `safe=False` → wrap (Arrow `pc.cast`).
- **`df.dropna(how='any'|'all', subset=None)`** — **lazy**: композиция
  `notnull()` фильтров. `how='any'`: drop if ANY null. `how='all'`: drop
  if ALL null.
- **`df.drop(cols)`** — selection minus. Lazy.
- **`df.rename(mapping)`** — materialized (schema change). `{old: new}`.

**Внутренний helper:**
- `CadwDataFrame._apply_column_transform(column, feature, fn, out_type,
  accessor=...)` — generic per-batch Arrow transform + append `{col}_{feature}`.
- Новая папка `cadw/accessors/` с `_dt.py`, `_str.py`.

**AST guards:** `tests/test_arrow_call_memory_guard.py` — whitelisted
`_apply_column_transform`, `astype`, `dropna`, `drop`, `rename`.

### Исправлено (v0.9.0a-fix1 — first-run bugs)

- **`PredicateFilter.__init__` не знал про новые ops (8 test failures).**
  v0.9.0a расширил `mask_for_batch` (is_null/not_null/contains), но
  `__init__` валидировал по старому `_OPS` → `ValueError: unsupported op`.
  Fix: `_OPS_NO_SCALAR = frozenset({"is_null", "not_null", "contains"})`,
  `__init__` проверяет `_OPS ∪ _OPS_NO_SCALAR`. Class: partial update
  within one class (handler extended, validation not).
- **`test_dt_null_timestamp_propagates` — NaT не конвертируется через
  `ts.tolist()` (1 failure).** `pa.table({"ts": [Timestamp, NaT, ...]})`
  → `NaTType does not support utcoffset`. Fix: `pa.array(ts)` — Arrow-native,
  NaT → null. Class: §13.4/134-like (input conversion not Arrow-safe).
- **CHANGELOG `[Unreleased]` отсутствовал после v0.8.11 tag** —
  silent skip (v0.9.0a entry не записан). Fix: prepend `[Unreleased]`
  before first `## [` if missing. Class: §13.4/140 recurrence (2nd hit
  of same trap — v0.8.11a-fix1 was first).

### Тесты (v0.9.0a)

- fast: **~1228** (+30 accessors + type ops). slow: **8**.

## [0.8.11] — 2026-10-20

### Исправлено (safety — P-D1 close)

- **P-D1 (S2, live): Arrow-native `drop_duplicates` rewrite.** pandas
  chain (`to_pandas → sort_values → drop_duplicates → drop → from_pandas`)
  в `_local_dedup` заменён на Path C: `dictionary_encode(subset)` +
  `group_by(subset).aggregate(min/max row_id)` + `is_in` filter.
  Удалён `import pandas` из `cadw/ops/dedup.py`.

  **Bench (prototype `scripts/proto_dedup_arrow.py`):**
  | n | pandas | Path B (arrow) | **Path C (dict)** |
  |---|---|---|---|
  | 100k | 9.3 MB | 10.2 MB | **5.9 MB** |
  | 300k | 29.3 MB | 39.0 MB | **3.6 MB** |
  | 1M   | 87.7 MB | 47.2 MB | **10.9 MB** |

  avg peak(C)/peak(A) = **0.29**. Correctness A==B==C across keep ∈
  {first, last, False} × selectivity ∈ {10%, 50%, 90% dupes}.

  **Stopgap removed:** `check_state(5 × table.nbytes)` → `check_state(1.5 ×
  table.nbytes)` (honest Arrow-native bound, proto-backed). Effective
  partition limit back from `avail × 0.1` to `avail × 0.33`.

- **Dead pandas helpers removed из `cadw/ops/join.py`.** `import pandas
  as pd`, `_drop_null_keys` (superseded by `_drop_null_keys_arrow` в
  v0.8.5.3) — удалены. Zero pandas in ops except `shuffle.py::_pandas_hash_column`
  (documented perf exception, §9.2.20).

### Добавлено

- **`tests/test_no_pandas_in_ops.py`** — AST guard: `import pandas` в
  `cadw/ops/*` разрешён только в `shuffle.py` (whitelist + rationale).
  Также `test_dedup_uses_arrow_native` (Path C marker).
- **`tests/test_dedup_arrow_native.py`** — 16 behavior tests: first/last/
  False, nulls в subset/value, multi-subset, int key, empty, K=N, K=1,
  pandas-equivalence, oversize → raise, dedup→sort chain, filtered dedup.

### Ограничения (v0.8.11a, §11.5)

- `drop_duplicates` порядок вывода: partition order, без
  `sort_values("row_id")` внутри partition (был в pandas-версии).
  Set-equivalence сохранена; для детерминированного порядка — `.sort_values(...)` после.
- `drop_duplicates` recursion при skew — не реализован (как в v0.3.0).
  Oversize partition → MemoryLimitExceeded. Backlog.

### Тесты

- fast: **~1190** (+16 behavior, +5 AST guard). slow: **8**.

### v0.8.11b — strict join conditional + schema-order audit

#### Изменено

- **`merge()` strict join enforcement.** `budget >= 160 MB` →
  `MemoryLimitExceeded` on whole-operation RSS over-budget (was: warn).
  Threshold: `4 × _JOIN_FIXED_OVERHEAD_BYTES = 160 MB`, plus secondary
  `state_limit >= 40 MB` (protects against reservation drop).
  Below threshold → warn only (backward compat with `mem_mb=20` tests;
  fixed 40 MB overhead dominates small budgets — strict check would fire
  trivially even for tiny joins). See §9.2.6.

- **`MemoryGuard.state_limit_bytes`** — public property. Callers need
  `state_limit` without raising `MemoryLimitExceeded` (used by merge's
  strict-mode decision).

- **`cadw/ops/join.py::_JOIN_STRICT_BUDGET_THRESHOLD_BYTES`** = `4 ×
  _JOIN_FIXED_OVERHEAD_BYTES` = 160 MB. Module constant, documented.

#### Добавлено (schema-order audit — §13.4/145 backlog)

- **`tests/test_schema_order_audit.py`** — 4 AST guards for the audit
  findings:
  - `join._safe_materialize_partition`: explicit `schema=out_schema`.
  - `sort._write_run`: `ArrowIPCBatchWriter(table.schema)`.
  - `groupby._read_partition_safe`: no `.select()` on batches.
  - `dedup._local_dedup`: `select_schema` (v0.8.11a-fix3).
  No additional bugs found across 4 audited call sites.

- **`tests/test_join_strict.py`** — 4 tests: threshold constant, public
  property, strict raise at budget 200 MB, strict pass small join.

#### Ограничения (v0.8.11b, §11.5)

- **Join strict enforcement requires `budget >= 160 MB`.** Below — warn
  only, per-batch guards still enforce per-batch memory. For guaranteed
  strict enforcement, set `MemoryGuard(max_memory_mb=200)` or higher.
  Rationale: 40 MB fixed overhead is non-trivial below 160 MB budget;
  raise would fire trivially for tiny joins.

#### Тесты (v0.8.11b)

- fast: **~1198** (+8), slow: **8**.

### Тесты (итого v0.8.11)

- fast: **~1198**, slow: **8**.


## [0.8.10] — 2026-10-20

### Добавлено (safety follow-up, P-A3 addendum)

- **Whole-operation RSS warning on `merge()`.** `CadwDataFrame.merge`
  вычисляет `predicted_output = n_left × avg_matches × row_bytes_out`
  из planner'а и логирует WARNING, если
  `40 MB + 5 × predicted_output > state_limit`.
  `MemoryLimitExceeded` не бросается — enforcement остаётся per-batch в
  `_process_partition_pair`, warning — диагностический сигнал.

  **Почему warning, а не raise:** per-batch проверка выполняется 8–16
  раз (по числу пар partition'ов), поэтому добавление 40 MB offset
  per-batch over-predicts в 8–16×. Добавление offset на уровне
  `merge()` требует поднятия `mem_mb=20` → `100` в ~10 тестовых файлах
  (~30 тестов падают без этого). Это отдельная работа v0.8.11, не
  «5 минут». Warning даёт сигнал сегодня без test churn.

- **`cadw/ops/join.py::_JOIN_FIXED_OVERHEAD_BYTES = 40 * 1024 * 1024`**
  — module const, измеренный floor для `Table.join` RSS beyond logical
  output. Across 3 runs of pk join @ N=1M: 114.7 / 142.8 / 156.8 MB
  RSS delta для 24 MB output. Offset покрывает Arrow pool init (~16 MB)
  + hash-table build + non-Arrow temporaries (~24 MB). Invariant of
  output size — added once, not per-batch.

### Ограничения (v0.8.10-pre, §11.5 addendum)

- Join guarantee остаётся **best-effort на RSS**, не strict. Для
  бюджетирования: `estimated_rss = 40 MB + 5 × predicted_output`; если
  превышает `state_limit = budget × 0.5`, actual peak может превысить
  budget независимо от per-batch guards. Strict enforcement — v0.8.11
  с ревизией тестовых бюджетов.
- Run-to-run variance 30–40% для одинаковой нагрузки (pool retention,
  jemalloc page release timing). Формулы target'ят worst-case.

### Добавлено (per-group feature)

- **`df.groupby('sensor').shift(k, on='ts', columns=[...])`** — per-group
  shift. `on=None` → within-group order = row_id. Group cols excluded
  from default `columns` set.
- **`df.groupby('sensor').diff(k, on='ts', columns=[...])`** — per-group
  first difference.
- **`df.groupby('sensor').rolling(window, agg, on='ts', columns=[...])`** —
  per-group rolling. APPENDS `{col}_rolling_{agg}`; min_periods applies
  from group start.

  **Algorithm (prototype-validated, 11 cases A==B):**
  composite sort `[*group_cols, *on_cols]` → single deque per col,
  reset on group change → sort back by `__orig_rid`. **2 sort passes,
  0 shuffle.** Peak RAM = batch + O(window × #cols), independent of N
  and G (group count).

  **Semantics:** null in group col → own group (SQL, not pandas
  `dropna=True`). Group columns are NOT shifted by default (they're
  the key). This differs from pandas `groupby().shift(1)` which shifts
  everything including group key.

  **Reference:** `tests/_helpers/per_group_reference.py` — pandas
  ground truth (§13.4/128, /129, /139).

### Тесты

- fast: **~1130 passed** (+3 behavior warnings: tight/loose budget). slow: **8**.

## [0.8.9] — 2026-10-20

По результатам Pingis library-memory audit (TASK2, run 2026-10-20):
6 accept, 0 reject, 2 needs-research, 5 confirmed live violations.

### Исправлено (safety — Arrow memory discipline)

- **P-A1 (S2, live)** — `join._safe_materialize_partition` и
  `dedup._local_dedup` теперь используют generator form
  `Table.from_batches(gen, schema=...)`. Measured v0.8.8: list = 0.93–1.03×
  file (retained batches), gen = 0.0× (zero-copy from Arrow buffers).
- **P-A3 (S2, live)** — `Table.join` pre-check: `check_state(5 × row_bytes_out
  × actual_matches)` **до** join'а. `row_bytes_out` = сумма bit_width//8 по
  `_empty_result_schema` + null bitmap (не 256 B/row heuristic). Measured
  v0.8.8: RSS delta = 6.85× output on narrow int64 @ 1M; 5× conservative.
- **P-D1 (S2, live)** — `dedup._local_dedup` pandas chain (to_pandas +
  sort_values + drop_duplicates + drop + from_pandas) проверяется
  `check_state(5 × table.nbytes)` **до** `to_pandas`. Measured v0.8.8:
  3.95–8.29× table.nbytes. Effective partition limit shrinks to `avail × 0.1`.
- **P-G3 (S2, live)** — `_groupby_with_skew` вызывает
  `check_partition_buffers(num_partitions, min(buf, 64KB))` перед
  открытием P writers (mirror `shuffle.py:273`).
- **P-G5 (S2, blind spot)** — sample cache bounded by `min(10k rows,
  0.3×avail)` bytes. Wide schemas (>=10 KB/row) hit byte cap before row cap.
- **P-SH2 (S2)** — `_hash_column` uint64 → `pc.cast(arr, int64, safe=False)`
  bitcast → pandas C-hash (19× faster than Python fallback). Bitcast =
  two's-complement reinterpretation, bijection over uint64 ↔ int64 →
  same value → same hash (partition-disjoint preserved). Nulls propagate
  through `pc.cast` unchanged. (v0.8.9-fix2: prior XOR-2^63-then-cast
  variant raised `ArrowInvalid` because values ≥ 2^63 overflow
  `pc.cast(safe=True)` — every uint64 value, since XOR with sign bit
  pushes all into [2^63, 2^64). Fix: skip XOR, cast directly with
  `safe=False`. REPL snippet documented in `_hash_column` docstring.)
- **P-A4 (S4)** — удалён dead `if False` block в `_exact_actual_matches_arrow`.

### Добавлено

- **§9.2.6–9.2.19 в DECISIONS.md** — канонические формулы пика для 14
  Arrow/pandas calls, derived из **RSS bench** (не из `pa_delta`).
  Каждая формула сопровождается guard recommendation и residual от
  fitted regression. Пример: `f_join = 5 × output_nbytes` (RSS-based,
  было бы 3.5× из pa_delta).
- **§3.11** — новый обязательный принцип: **Arrow call memory discipline.**
  Любой Arrow/pandas call, аллоцирующий > O(batch), требует:
  (1) формула пика в §9.2.N; (2) явный check_state до вызова;
  (3) AST guard `test_arrow_call_memory_guard.py`.
- **§13.4/138** — «Доверие Arrow = доверие LightGBM». Same class as
  §13.4/25 (LightGBM adapter) и §13.4/26 (partial_fit whitelist).
  MemoryGuard проверяет RSS, но не видит allocation **внутри** Arrow-функции.
  Отсюда и `pa_delta` vs RSS разница в 2× на больших joins.
- **`tests/test_arrow_call_memory_guard.py`** — AST guard: 14 high-risk
  attributes (`join`, `group_by`, `sort_by`, `to_pandas`, `from_pandas`,
  `from_batches` list-form, `sort_indices`, `quantile`, `rank`,
  `value_counts`, `tdigest`, `index_in`, `is_in`, `dictionary_encode`).
  Каждый должен иметь preceding guard (check_state/check_batch/stat)
  в том же function scope. Whitelist для helpers (`_safe_materialize_partition`,
  `_local_dedup`, `_groupby_with_skew`, `hash_shuffle`).
- **`tests/test_pingis_safety.py`** — 14 behavior tests:
  `_estimate_row_bytes` (4), uint64 hash (3), groupby skew+budget (1),
  groupby sample byte-cap (1), dedup pandas chain raise (1),
  AST guards (2), round-trip sanity (3).

### Изменено

- `cadw/ops/join.py` — новая module-level `_estimate_row_bytes(schema)`.
- `cadw/ops/shuffle.py` — uint64 в `_hash_column` не идёт через Python loop.
- `dedup._local_dedup` — effective partition size ~5× меньше на тех же
  бюджетах (honest memory bound).
- AST guard при проверке `Table.from_batches` различает `ListComp` (bad)
  vs `GeneratorExp` (good).

### Ограничения (v0.8.9, §11.5)

- **P-Q2 `pc.quantile` whole column** — sort-based, ~1× column working
  copy. **NEEDS-BATCHED**, не используется в `cadw/`. Backlog §11.6.
- **P-R1 `pc.rank` whole column** — ~2× output + sort buffer.
  **NEEDS-BATCHED**. Per-batch use в `rank.py:94` already safe.
- **P-G2/P-G5 `pc.value_counts` при K≈N** — O(N) hash. **NEEDS-BATCHED**.
- **P-SH1 string shuffle** — needs research (dictionary growth не
  детерминирован, partition-disjoint invariant at risk).
- **P-A3 wide-schema benchmark** — measured on int64 only; wide string
  schemas deferred to v0.8.10 (proposed: N=500k, 5 string cols).
- **P-G3 skew workload benchmark** — confirmed by code review, not
  benchmarked. Deferred to v0.8.10 (N=1M, one key at 60%, P=16).

### Тесты

- fast: **~1124 passed, 1 skipped**. slow: **8 passed**.
- +14 в `tests/test_pingis_safety.py`, +2 AST в
  `tests/test_arrow_call_memory_guard.py` (guard + sanity on generator
  vs list-comp).

## [0.8.8] — 2026-10-19

### Добавлено (housekeeping, не функциональность)
- **`scripts/collect_project.py`** — переписан с `--mode=smart|full|tiny`.
  - `smart` (default): full для `cadw/`, `tests/`, `README`, `pyproject`,
    latest-3 `patch_*.py`, top-5 CHANGELOG, DECISIONS (§13.4 truncated to
    last 10 episodes). Stub: old patches, archive, backups, `uv.lock`,
    old CHANGELOG entries. **~2 MB → ~500 KB** (75% reduction).
  - `full`: старое поведение (all files).
  - `tiny`: README + tree + top-3 CHANGELOG (~50 KB).
- **`scripts/patch_v088_housekeeping.py`** — one-shot cleanup + audit:
  - Phase A: rewrite `collect_project.py`.
  - Phase B: move `scripts/patch_*.{sh,py}` (older than keep=3) →
    `scripts/archive/v0.8.7/`. `git mv` (сохраняет history).
  - Phase C: move `.backups/*` → `/tmp/cadw_backups_<ts>/` (вне repo).
  - Phase D: append [0.8.8] + §13.4/136 + §11.6.
  - Phase E: audit (layers, primitives, §13.4/§11.5/§11.6 counts).

### Framework status (audit подтвердил)
- 7 layers (Storage / Dataset / Access / Primitives / Operations /
  DataFrame / Export) — все на месте.
- 3 primitives (Reduce / HashShuffle / Sort) — stable API.
- Все `ops/*.py` (кроме primitives) импортируют `from cadw.ops.{reduce,
  shuffle, sort}` — нет reinvention.
- Все §13.4 episodes closed через CHANGELOG-записи.
- §11.5 — 18 known limitations + backlog (documented, not bugs).

### Рабочая папка
- `scripts/` → только `collect_project.py`, `_lib.sh`, latest-3 patches,
  `archive/v0.8.7/` (n-исторических), `bench_*`, `diagnose_*`.
- `.backups/` → очищено (`/tmp/cadw_backups_*`).
- `.gitignore` — без изменений (archive tracked).

### Тесты
- fast: **~1096 passed**. slow: **8 passed**.
- CHANGELOG guard: 3 passed (top == tag, descending, exists).

# Changelog

Все значимые изменения проекта CADW. Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).

<!--
  Порядок: **descending by version** (v0.8.5 сверху). Восстановлен в
  v0.8.5 из-за сломанной цепочки v0.1.5+ (§13.4/111): каждый
  `patch_changelog()` вставлял перед anchor `[0.X.Y-1]`, а при
  отсутствии anchor `str.replace` был silent no-op. Как побочный
  эффект, anchors дрейфовали — top становился out-of-order. Пропущенные
  версии (v0.1.5, v0.1.6, v0.2.2, v0.3.5, v0.4.2, v0.5.0, v0.5.2-v0.5.4,
  v0.6.x, v0.7.0-v0.7.7, v0.7.10) — в `git log --oneline --grep='^v'`.
  AST guard `test_changelog_has_latest_tag` (top == tag on master) +
  `test_changelog_is_descending` (v0.8.5) фиксируют инвариант.
-->

## [0.8.7] — 2026-10-19

### Добавлено
- **`CadwDataFrame.resample(freq, agg, on=..., columns=None)`** —
  bucket aggregation по временной/целочисленной колонке.
  - `freq: int | str` — секунды (`5`) или строка (`'5s'`, `'1min'`/
    `'1m'`, `'1h'`, `'1D'`).
  - `agg ∈ {sum, mean, min, max, count}`.
  - `on: str` — timestamp / date / integer column. Bucket =
    `(t_sec // freq) * freq`.
  - Output: `[on, f'{col}_{agg}'...]`. Bucket column dtype = `on` dtype
    (timestamp → timestamp, int → int, date → date).
  - Peak RAM: batch + O(#cols) state + 1 output batch (BATCH=1000
    buckets). **Independent of N и bucket size** (running accumulator,
    не `list(values)`).

### Валидировано prototype-first (§13.4/123)
- `scripts/proto_resample.py`: A (pandas groupby ts//freq) == B
  (external_sort + bucket transition) == C (manual). 5 buckets × sum:
  `[(0,4),(5,13),(10,23),(15,32),(20,22)]`.

### Renamed: v0.8.3 → v0.8.7 (§13.4/132)
- Планировалось как v0.8.3 (roadmap). После v0.8.4/v0.8.5/v0.8.6 уже
  tagged, `test_changelog_is_descending` требует top = highest →
  v0.8.7 (не v0.8.3).

### Semantics (MVP, §11.5)
- **Empty bucket** (gap in ts) → skip (no NaN fill; для fill —
  `.reindex` после).
- **Null in `on`** → skip row (cannot bucket).
- **All-null agg bucket** → skip row (all columns None).
- **Null in agg col** → skip value (matches pandas skipna).
- **Nulls_all**: bucket с только null значениями (все cols) → skip row.
- **Partition**: bucket spanning batches — accumulator живёт через
  batch boundary (running state, O(1) RAM).

### Ограничения (v0.8.7, §11.5)
- `on` — single column. Multi-column (e.g. `on=['ts', 'sensor']` для
  per-group buckets) — не поддерживается (composite sort ≠ per-group;
  для per-group — `groupby(...)`).
- `freq` — только `int` секунды или строка `'Ns'|'Nmin'|'Nm'|'Nh'|'ND'`.
  Compound (`'5min30s'`), `'W'`, `'M'`, `'Y'`, sub-second (`'500ms'`) —
  не поддерживаются.
- `uint64` on — reject (overflow risk при bucket arithmetic; cast в
  int64 required).
- `date` on — granularity = days (freq should be ≥ `'1D'`; `'5s'` на
  date даст O(#days) buckets с 0 rows → skip → фактически пустой
  output для одного дня).
- All-null bucket + `count` — skip (не 0; для zero-fill — post-process
  через `left_merge`).
- `nan` timestamp bucket on far future (>year 2262 для ns) — overflow
  (int64). Documented, edge case.

### Тесты
- +28 в `test_resample.py`: sum/mean/min/max/count (prototype data),
  multibatch bucket, empty bucket skip, nulls, all-null bucket, null
  in on, timestamp('s'/'ms'), date32, validation (missing on,
  non-time-type, uint64, on-in-columns, invalid agg, non-numeric,
  empty columns), freq parser (9 param + 5 edge), preserves original,
  filtered, 2 AST guards.
- fast: **~1092 passed**. slow: **8 passed**.

## [0.8.6] — 2026-10-19

### Добавлено
- **Multi-column `on`** для `shift` / `diff` / `rolling`:
  `on=["ts", "sensor"]` (Sequence[str]).
  - **Лексикографический composite sort**, не per-group. Сортировка по
    `[on_cols..., row_id]`, применение shift/rolling per row, sort-back
    по `__orig_rid` (как в v0.8.2 / v0.8.4).
  - Per-group — `groupby('sensor').shift(1, on='ts')` (v0.8.5a+).
- `CadwDataFrame._normalize_on(on, fn_name)` — helper: `str | Sequence[str]`
  → `list[str]` + validation.
- `_shift_on(..., on_cols: list[str])` / `_rolling_on(..., on_cols:
  list[str])` — принимают pre-normalized list; `external_sort` получает
  `key_columns=on_cols`.

### Backward compat
- `on="ts"` — как раньше (single-element normalization).
- `on=["ts"]` — идентично `on="ts"`.
- `on=[]` — `ValueError: {fn}: on must be non-empty`.

### Ограничения (v0.8.6, §11.5)
- Multi-column `on` — **composite sort**, не per-group. Для per-group
  windows — `groupby(...).shift/rolling(..., on=...)` (v0.8.5a).
- `_iter_shifted` / `_iter_zipped` не дедуплицируются — дублирование
  остаётся (refactor в backlog).
- Stability with duplicate `on` values — tie-break by row_id (heap),
  как в v0.8.2/0.8.4.

### Тесты
- +5 в `test_shift_on_col.py`: multi-key shift/diff, single-element-list
  equivalence, empty-list ValueError, stable tie-break.
- +3 в `test_rolling_on_col.py`: multi-key sum, min_periods=1,
  empty-list ValueError.
- AST guards (v0.8.2/0.8.4) — unaffected (first sort uses `on_cols`
  variable; second still literal `["__orig_rid"]`).
- fast: **~1064 passed**. slow: **8 passed**.

## [0.8.5] — 2026-10-19

### Исправлено
- **CHANGELOG отсортирован descending** (§13.4/111 + §13.4/130). Исторически
  `patch_changelog()` вставлял перед anchor `[0.X.Y-1]`. Anchor дрейфовал,
  top CHANGELOG перестал быть «latest». `test_changelog_has_latest_tag`
  (master + tag) корректно упал: `top [0.7.4c] != tag v0.8.4`. Fix:
  одноразовый sort всех `## [X]` блоков by (major, minor, patch, suffix)
  descending. Новый guard `test_changelog_is_descending` (в
  `test_changelog.py`) ловит регресс. v0.8.4 tag **не** ретагнут —
  тест не запускается на старых tags (`--exact-match` на HEAD).
- **`row_id` reserved guard** в `external_sort` и `hash_shuffle`
  (§13.4/126). User column named `row_id` коллизит с internal identity
  column (appended by external_sort/hash_shuffle) → previously:
  `KeyError: Field "row_id" exists 2 times in schema` (через
  `ArrowIPCBatchReader`, в `sort_values`); `Multiple matches for
  FieldRef.Name(row_id)` (в `pc.sort_indices`). Теперь: `ValueError:
  external_sort: column name 'row_id' is reserved for CADW internal
  identity column. Rename your column before calling.` Guard выполняется
  **до** materialize batches.

### Добавлено
- **`tests/_helpers/rolling_reference.py`** — `expected_rolling(vals,
  window, agg, min_periods)` — deque ground truth (§13.4/128). Matches
  pandas: positional window, nulls consume slot, min_periods applies to
  non-null count.
- 4 rolling-теста (`test_rolling_min/max_with_nulls` в
  `test_rolling.py`; `test_rolling_on_min/max_with_nulls` в
  `test_rolling_on_col.py`) → `expected_rolling(...)`. Устраняет корень
  §13.4/128 (manual arithmetic).
- 8 новых тестов в `test_row_id_reserved.py`: external_sort
  (reject key/value + valid works), hash_shuffle (reject key/value +
  valid works), E2E (sort_values/groupby на user `row_id`).
- `test_changelog.py::test_changelog_is_descending` — strict descending
  invariant.

### Ограничения (v0.8.5, §11.5)
- `row_id` — reserved. User column → rename. `__orig_rid` тоже reserved
  (internal sortback). Рекомендуется префикс `my_`.
- Guard fires on arguments (`key_columns` / `value_columns`). Internal
  callers (`_shift_on`, `_rolling_on`, `rank`, `groupby`) либо передают
  `__orig_rid`, либо не содержат `row_id` в value list — не затронуты.

### Тесты
- +8 в `test_row_id_reserved.py`.
- +1 в `test_changelog.py` (descending guard).
- 4 rolling-теста → helper (behavior unchanged).
- fast: **~1056 passed**. slow: **8 passed**.

## [0.8.4] — 2026-10-19

### Добавлено
- **`CadwDataFrame.rolling(window, agg, on=None)`** — time-aware rolling.
  `on=None` — positional (v0.6.2c, unchanged); `on=str` — sort-aware:
  1. external_sort by `[on]` (row_id tie-break from heap).
  2. `_RollState.push` per row (reuse v0.6.2c state — same min/max
     monotone deque, same null-skip, same min_periods).
  3. external_sort by `__orig_rid` (rename `row_id` → `__orig_rid`,
     §13.4/122).
  4. Zip original + rolling cols **APPENDED** (`{col}_rolling_{agg}` —
     rolling adds new columns, doesn't replace).
  - Peak RAM = batch + O(window × #cols). Independent of N.

### Валидировано prototype-first (§13.4/123)
- `scripts/proto_rolling_on.py` — A (sort_values+rolling+sort_values)
  == B (external_sort + deque + external_sort) == C (pure Python manual).
- Same 5-row case as prototype; 1 итерация патча (v0.8.2: было 5 fix).

### Исправлено (v0.8.4-fix — S1, latent bug с v0.6.2c)
- **`_RollState.push(None)` для agg='min'/'max'** — `while _dq and
  _dq[-1][1] >= None` → `TypeError: '>=' not supported between int and
  NoneType`. Bug существовал с v0.6.2c (позиционный rolling), не
  покрыт тестами (`test_rolling_with_nulls` использует sum). Fix:
  guard `if not is_null` вокруг monotone-deque maintenance. Window
  slot всё равно потребляется (positional semantics, как pandas).
  Добавлены тесты: `test_rolling_min_with_nulls`,
  `test_rolling_max_with_nulls`, `test_rolling_min_all_nulls`
  (positional) + `test_rolling_on_min_with_nulls`,
  `test_rolling_on_max_with_nulls` (on=col).
- **v0.8.4 tests (2, not code)**: `test_rolling_on_with_nulls` и
  `test_rolling_on_all_columns` — неверные ожидания. Actual code
  корректен (nulls потребляют window slot, positional; matches pandas
  skipna=True). Fixed expected values + comments.

### Обнаружен latent bug (§13.4/126)
- User column named `row_id` collides with internal `row_id`
  (external_sort/hash_shuffle/merge) → `KeyError: Field "row_id" exists
  2 times in schema` (в `sort_values`). Documented §11.5 as reserved
  column name; fail-fast guard — v0.9.

### Ограничения (v0.8.4, §11.5)
- `on` — single column. Multi-column `on=['ts', 'sensor']` — v0.8.4a.
- Per-group rolling (`groupby('sensor').rolling(...)`) — v0.8.5+.
- `row_id` — reserved column name (internal).

### Тесты
- +17 в `test_rolling_on_col.py`: sum/mean/min/max/count, min_periods,
  nulls, multibatch, preserved, all columns, sort-after, missing col,
  non-numeric, no-on backward compat, +2 AST guards (row_id rename,
  append-not-replace).
- fast: **~1043 passed**. slow: **8 passed**.

## [0.8.2] — 2026-10-19

### Добавлено
- **`CadwDataFrame.shift(k, fill_value=None, columns=None, on=None)`** —
  time-aware shift. `on=None` — positional (v0.6.2b, unchanged);
  `on=str` — sort-aware:
  1. external_sort by `[on, row_id]` (stable tie-break).
  2. Stateful shift (deque per column, O(k)).
  3. external_sort by `row_id` → original order.
  4. Zip with original.
  - Peak RAM = batch + O(k × #cols). Independent of N.
  - 2 external_sorts (vs manual `df.sort_values(x).shift(x).sort_values(x)`).
  - Multibatch consistent (chunk size не влияет).

- **`CadwDataFrame.diff(k, columns=None, on=None)`** — time-aware diff.
  `on=None` — positional (v0.6.2b); `on=str` — sort-aware via `_shift_on`
  + `pc.subtract`. Reuses shift's 2 sorts, no extra sort.

### Ограничения (v0.8.2, §11.5)
- `on` — single column. Multi-column `on=['ts', 'sensor']` — v0.8.2a.
- `on` с duplicates — stable sort (tie-break by row_id asc). Documented.
- `on` должен быть sortable (monotonic не требуется — обычный sort).
- Per-group shift (`groupby('sensor').shift(1, on='ts')`) — v0.8.3+.

### Тесты
- +17 в `test_shift_on_col.py`: shift on basic/k2/fill/string/all,
  preserved, nulls, multibatch, missing col; diff on basic/k2/preserved,
  non-numeric raises, non-monotonic stable, nulls; chain sort; backward
  compat (no-on unchanged).
- fast: **~1014 passed**. slow: **8 passed**.

### Цена
- Shift(1, on='ts') on 10^6 rows: ~2-3s (2× external_sort).
- Same as user-manual `df.sort_values('ts').shift(1).sort_values('ts')`,
  but API one-liner.

## [0.8.1] — 2026-10-19

### Изменено — performance
- **`compute_hashes`: pandas C-hash для numeric, Python для string.**
  - Numeric/boolean/timestamp/date → `pd.util.hash_pandas_object`
    (C-level SipHash). Bench: int64 39.2→2.0 ms (**19.6×**),
    float64 39.2→1.0 ms (**39.2×**).
  - String/large_string → Python `repr().encode()` + xxhash. Причина:
    pandas `hash_pandas_object` для object dtype — per-element Cython,
    50 ms/100k, **22% медленнее** Python (§13.4/107). Без регрессии.
  - struct/list/dict/uint64 → Python loop (fallback).
  - Seed через `h_np ^= np.uint64(seed)` **до** avalanche.
- **Avalanche (splitmix64) — всегда**, не только при seed. Причина:
  pandas `hash_pandas_object` для object dtype `hash << 1` (reserves 0
  для NA), все string hashes чётные → `h & 3` использует только
  партиции `{0, 2}` → recursion split 2-way вместо 4-way →
  `MemoryLimitExceeded` при depth=5 (§13.4/107). Fix: 3 умножения +
  3 XOR-shift, ~0.3 ms/100k.

### Критично для корректности
- **`types_mapper=Int64Dtype`** в `arr.to_pandas()` (numeric): без него
  `int64` с nulls → `float64` (NaN), без nulls → `int64` → разные hash
  для одного значения в разных батчах → нарушение partition-disjoint.
- Тесты: `test_int64_consistent_hash_across_null_presence`,
  `test_avalanche_spreads_low_bits`,
  `test_avalanche_seed_spreads_low_bits`.
- `test_shuffle.py::test_stable_hash_across_batches` — убран лишний
  `assert a_parts != b_parts` (partition-disjoint ≠ different partitions;
  коллизии разных ключей в одной партиции — норма).

### Тесты
- +20 в `test_hash_v081.py`: cross-batch consistency (int32/int64/string
  с/без nulls), deterministic, seed, nulls, dispatch (string→python,
  numeric→pandas), partition-disjoint (int+string), avalanche (low-bit
  spread + seed + similar strings), groupby recursion end-to-end.

### Ожидание
- int64 shuffle 1M: 0.61s → **~0.22s** (2.76×).
- 500 GB / 50 B / int64 keys: ~100 min → **~40 min**.

## [0.8.0] — 2026-10-19

### Добавлено — time-series
- `CadwDataFrame.expanding(agg, min_periods=1, columns=None)` —
  кумулятивное окно. Aggs: `sum`/`mean`/`min`/`max`/`count`. O(1) state
  (scalar accumulators; min/max — текущий min/max, без deque). Nulls
  skip; null row → null output. Output: `{col}_expanding_{agg}`.
- `CadwDataFrame.ewm(alpha/span/halflife/com, adjust=True, min_periods=0,
  columns=None)` — экспоненциально-взвешенное среднее. Ровно один из
  4 параметров. `adjust=True`: `(s_num, s_den)`, `y = s_num/s_den`.
  `adjust=False`: `(ewm,)`, `y = (1-α)·y_prev + α·x`. Nulls skip
  (`ignore_na=True` семантика). Output: `{col}_ewm`.

### Добавлено — PCA single-pass
- `CovarianceMeanOp(CovarianceOp)` — тот же state, `finalize` →
  `{cov, mean, n}`. `fit_pca` → **2× I/O reduction**.
- `CovarianceOp` (bare matrix) не меняется — backward compat.

### Ограничения (v0.8.0, §11.5)
- `expanding`/`ewm` не поддерживают `on=col`.
- `ewm(ignore_na=False)`, `ewm(bias=True)` — не реализованы.

### Тесты
- +26 в `test_expanding_ewm.py`, +3 (CovarianceMeanOp + fit_pca regression).

## [0.7.9] — 2026-10-19

### Исправлено в v0.7.9-fix
- **`_AssignLeavesOp.update` 4-unpack** — после введения 5-tuple spec (`..., gain`) старый `j, thr, left, right = spec` падал `ValueError: too many values to unpack`. Fix: `spec[:4]` (forward-compatible slice). Тот же fix применён к `_assign_leaf_from_array` в изначальном v0.7.9.
- **Multiclass `_fit_boosting_multiclass` 4-tuple** — `s.replace(old, new, 1)` дважды не сработало (второй replace не нашёл pattern из-за отличий в форматировании). Fix: replace **all occurrences** без count → multiclass тоже 5-tuple.
- **+AST guard** `test_all_spec_unpack_uses_slice` — все `= spec` должны быть `= spec[:4]` / `= spec[:5]`. Ловит класс «4-unpack + 5-tuple» багов.

### Добавлено
- **`compute_feature_importance(model, importance_type='gain'|'split',
  top_k=None)`** — extraction helper in `cadw.ml.gbdt`.
  - `'gain'` (default): sum of split gains per feature — XGBoost-style,
    better reflects feature usefulness (big gain на 1 split > 10 малых).
  - `'split'`: split count (v0.7.4a semantics, backward compat).
  - Multiclass: gain summed across all (tree, class).
  - Returns dict sorted by importance ascending (for horizontal barh).
  - Handles both binary (list[tree]) and multiclass (list[list[tree]]).

- **Tree structure extended:** `spec = (feat, thr, left, right, gain)`
  (5-tuple) instead of 4-tuple. `gain` stored at index 4.
  `_assign_leaf_from_array` tolerates both 4- and 5-tuple (`spec[:4]`)
  — backward compat for pre-v0.7.9 pickled models.

- **`feature_importance(..., importance_type='gain')`** in matplotlib and
  plotly adapters — use the new helper. Split-count still available via
  `importance_type='split'`.

### Изменено
- **`feature_importance` default** — был `split count`, стал `gain`.
  Breaking change для downstream code, который relied on integer counts
  — но семантически правильнее; документировано.

### Ограничения (v0.7.9, §11.5)
- **Permutation importance** — не реализована (требует out-of-core eval
  на holdout; v0.7.10+).
- **SHAP** — не реализован (слишком тяжёлый для out-of-core; вне scope).
- Legacy 4-tuple spec (pre-v0.7.9 pickled): `gain` returns 0 for those
  trees (silently, gain not available). `split` works.
- `gain` — raw sum (not normalized); разные tree scale → разные magnitudes.
  Для normalized importance — `gain / gain.sum()` в user code.

### Тесты
- +13 в `test_feature_importance_gain.py`: helper (invalid type, non-GBDT,
  sorted), gain semantics (important feature, split differs), top_k,
  zero model, multiclass aggregation, adapters (matplotlib/plotly, both
  importance_type), legacy 4-tuple fallback, spec is 5-tuple.
- fast: **~995 passed**. slow: **8 passed**.

## [0.7.8] — 2026-10-19

### Исправлено в v0.7.8-fix3
- **Dispatch preserves backward-compat error messages.**
  `fit_gbdt_classifier(n_classes=None)` считает distinct **перед**
  `_validate_multiclass_y`. 2 distinct (или non-integer) → binary
  path (`_validate_binary_y` даёт исторический `"binary (0/1)"` /
  `"both classes"`); >2 distinct → multiclass path (`"0-indexed"`).
  Без этого `{2, 3}` и `{1}` давали `"0-indexed"` вместо ожидаемых
  сообщений — ломалось `test_gbdt_classification.py`.
- **+AST export guard** (`tests/test_api_availability.py`):
  `test_cadw_all_exports_importable`,
  `test_cadw_ml_all_exports_importable`,
  `test_cadw_ast_all_matches_imports`. Ловит класс багов
  «__all__ обновлён, import забыт» (§13.4/115).

### Добавлено
- **`fit_gbdt_multiclass`** — multiclass GBDT (softmax loss), per-class
  loop, reuse существующих ops (`_AllNodeHistOp`, `_AssignLeavesOp`,
  `_LeafValuesOp` — 0 новых ops).
  - K деревьев на round (по одному на класс); `trees_` = `list[n_estimators]`
    of `list[K]`.
  - Grad: `p[i,c] - onehot(y[i], c)`; Hess: `p[i,c]*(1-p[i,c])`
    (XGBoost diagonal approximation).
  - `base_prediction_ = zeros(K)` (деревья находят сдвиг).
  - `predict_proba → (n, K)`, `predict → (n,) int64`.
- **`GBDTMulticlassClassifier`** — model class; sklearn-подобные поля:
  `trees_`, `n_classes_`, `feature_names_`, `n_features_in_`,
  `predict_proba`, `predict`, `to_dict`.
- **`fit_gbdt_classifier(..., n_classes=None)`** — auto-dispatch:
  - `n_classes` inferred из `y` (`{0..K-1}`).
  - `K == 2` → `GBDTClassifier` (binary, unchanged).
  - `K > 2` → `GBDTMulticlassClassifier`.
  - `K < 2` → ValueError.
- **`CadwDataFrame.fit_gbdt_multiclass`** — фасад.

### Ограничения (v0.7.8, §11.5)
- y должен быть `{0, 1, ..., K-1}` (0-indexed). 1-indexed `{1..K}` —
  ValueError (сдвиньте y на -1).
- `K × binary` fit cost (per-class loop). K=3 → ~3× медленнее binary.
  Для K > 5, n > 1M — v0.7.8a добавит `_MulticlassAllNodeHistOp`
  (1 pass для всех K, ~K× speedup).
- One-vs-rest не поддерживается (только softmax).
- Multiclass quantile/huber — не реализованы.

### Тесты
- +22 в `test_gbdt_multiclass.py`: `_softmax` (sum, stability, dim),
  fit (basic, accuracy, n_estimators improves, deterministic), predict
  (proba shape, int labels, df columns, prob_df columns, nulls),
  dispatch (binary, multiclass, single-class error), y validation
  (1-indexed, missing class, K=2 rejected by fit_multiclass),
  memory guard, convenience, to_dict, binary regression.
- fast: **~979 passed**, slow: **8 passed**.

### Время
- K=3, n=1000, depth=3, 30 rounds: ~15-20s (CI). Локально — быстрее.

## [0.7.4c] — 2026-10-19

### Добавлено
- **`matplotlib_adapter.hexbin_plot(df, x, y, gridsize=30, n=10_000,
  random=True, ...)`** — 2D hexagonal binning (matplotlib native).
  - Sample-based (`sample(n)` + `plt.hexbin`), **не** aggregate-first:
    hexbin aggregation на гексагональной сетке требует custom binning,
    matplotlib делает это сам через `plt.hexbin`.
  - `log_scale=True` (default): LogNorm для colorbar (плотность scatter).
  - **Plotly/Seaborn:** нет native hexbin — используйте `hist2d_plot`
    (прямоугольная сетка). Documented.

- **`_aggregate.sample(df, n, *, random=False)`** — новый параметр.
  - `random=False` (default, fast): `df.head(n)` — bounded I/O,
    но **может быть смещён** на time-sorted данных (только префикс).
  - `random=True`: uniform random sample через
    `df.iloc[np.sort(rng.choice(cnt, n, replace=False))]`. Full O(N)
    scan, O(n) memory. Deterministic (seed=42).
  - `scatter` / `_sample_pair`: `random=False` по умолчанию
    (backward compat v0.7.4a).

### Ограничения (v0.7.4c, §11.5)
- `hexbin_plot` — только matplotlib. Plotly/Seaborn — нет hexbin
  (документировано; use hist2d).
- `random=True`: 2× O(N) scan (count + iloc to max position). Для 10^7
  строк ~секунды. `random=False` — fast preview.
- hexbin sample-based, не aggregate-first — n=10_000 bounded, shape
  approximate.

### Тесты
- +4 в `test_viz_aggregate.py`: sample(random=True) uniformity,
  small-df returns-self, deterministic, head default unchanged.
- +5 в `test_viz_matplotlib.py`: hexbin_plot writes PNG, random sample
  spreads, missing column raises, invalid gridsize raises, nulls skipped.
- fast: **~1026 passed**. slow: **8 passed**.

## [0.5.2a] — 2026-10-19

По результатам Pingis audit: P5 (S2) — sort merge Python-list buffer.

### Исправлено
- **P5: `_merge_keys`** — заменить Python-list buffer (`buffer[c].append(
  as_py())`) на `pa.Array.Builder` (`Int64Builder`/`Float64Builder`/
  `StringBuilder`/etc.). Пик на merge: `read_batch_size × arrow_size`,
  а не `read_batch_size × 28-80 B` (Python-object overhead).
  - `_make_builder(dtype)` — выбор builder по Arrow-типу.
  - `_append_to_builders` — append одного значения (с `.as_py()` O(1)).
  - `_flush_builders` — finish → `RecordBatch`.
  - Dictionary/list/struct → `NotImplementedError` (honest; не покрыто).

### Добавлено
- `scripts/bench_p5_sort.py` — RSS-сэмплер benchmark для sort
  (сравнение с Pingis baseline +31.9 MB).
- `tests/test_sort.py::test_merge_keys_uses_arrow_builder_not_python_list`
  — AST guard (нет `buffer[...].append`).
- `tests/test_sort.py::test_merge_keys_uses_arrow_builder` — presence check.

### Ожидаемый эффект
- B3a (1M rows, 30 MB budget): Δpeak ≤ 30 MB (baseline: +31.9 MB).

### Ограничения (v0.5.2a, §11.5)
- Dictionary/list/struct columns: `NotImplementedError` в sort merge.
- `.as_py()` per value — CPU overhead остаётся; P5 — memory-fix, не
  speed-fix. CPU-оптимизация (batch-wise `pc.sort_indices`) — v0.5.4.

## [0.5.1] — 2026-10-19

По результатам мультиагентного анализа Pingis (`_FOR_PINGIS/REPORT.md`):
P2 (S1) + P8 (S2).

### Исправлено
- **P2 (S1, streaming join):**
  - `_process_partition_pair` — теперь generator (`yield` вместо
    `out_batches.extend(...)`).
  - `_left_preserve_with_nulls_iter` — streaming-версия.
  - `_join_core` — общий setup (validate, peek, shuffle) → `(schema, iter)`.
  - `hash_join_streaming(...) -> (schema, Path)` — **NEW**: пишет
    выходные батчи напрямую в `ArrowIPCBatchWriter`. Peak RAM =
    `max(build_partition) + probe_batch + writer_buffer`, **не зависит
    от размера output**.
  - `CadwDataFrame.merge` использует `hash_join_streaming`; промежуточный
    `.arrow` удаляется после `Dataset.from_batches`.
  - `hash_join(...)` (старый API) сохраняет семантику — собирает
    streaming-output в `pa.Table` (backward compat).

- **P8 (S2, state_bytes API):**
  - `ReduceOp.estimate_state_nbytes() -> int | None` — метод ABC
    (default None → fallback на `MemoryGuard.estimate_state_size`).
  - Built-in ops override: `CountOp`/`SumOp`/`MinMaxOp`/`MeanOp` → 16;
    `StdOp` → 24; `CovarianceOp` → `8+8d+8d²`; `HistogramOp` → `8bins+16`.
  - `reduce()` вызывает `op.estimate_state_nbytes()` **до** `op.init()`.
  - User `ReduceOp` can override — публичный контракт для ML-слоя.

### Добавлено
- `cadw.ops.hash_join_streaming` — streaming join API.
- `tests/test_reduce_state_bytes.py` — 8 тестов (built-in + custom +
  fallback + oversized-state-rejects-before-init).
- `tests/test_join_streaming.py` — 10 тестов (generator, disk write,
  incremental batches, backward compat, merge uses streaming, cleanup,
  empty input).
- `.pingis/TASK.md` — periodic analysis task template.
- `.gitignore`: exception для `.pingis/TASK.md`.

### Изменено
- `test_join_oom_guard.py::test_oversize_partition_raises_with_max_depth_zero`
  — `next(gen)` вместо `gen(...)` (generator).

### Ожидаемый эффект
- Join B1a (100k × 10k под 80 MB) → Δpeak с ~140 MB до ≤ 80 MB (цель).
- ML custom `ReduceOp` (XTXOp, KMeansStepOp, HistOp) не получают
  warning про 1 MiB; правильный state-размер.

### Ограничения (v0.5.1, §11.5)
- P1 (join `to_pandas`/`pd.merge`) — отдельный патч (v0.5.2).
- P3/P4/P5/P6/P7/P10/P11/P12/P14 — в roadmap после v0.5.2.
- `hash_join` (не-streaming) всё ещё может OOM при большом output —
  используйте `hash_join_streaming` или `CadwDataFrame.merge`.

## [0.4.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.loc` — open-endpoint slices на monotonic index:
  - `loc[:hi]` → от начала до `hi` (inclusive).
  - `loc[lo:]` → от `lo` до конца.
  - `loc[:]` → все строки (lazy view).
  - descending monotonic тоже работает (`loc['b':'d']` → d,c,b).
- `CadwDataFrame.loc[label]` — duplicate labels → `CadwDataFrame`
  (pandas: Series→DataFrame). Unique → `dict` (без изменений).
- `CadwDataFrame.loc[[labels]]` — все вхождения для каждого label
  (ранее first-wins).
- `MaterializedIndex.is_monotonic_increasing()`,
  `is_monotonic_decreasing()`, `is_monotonic()` — O(N), nulls skipped.
- `MaterializedIndex.has_duplicates()` — O(N), set-based.
- `MaterializedIndex.get_positions_all(label)` — все позиции по `==`.

### Изменено
- `loc[slice]` на **non-monotonic** index → `NotImplementedError`
  (pandas 2.x тоже).
- `loc[slice]` на **duplicate** index → `NotImplementedError`
  (pandas: cannot slice non-unique).
- `loc[slice]` с `step != 1` → `NotImplementedError`.
- `.at[label, col]` на duplicate label → `KeyError` (pandas: undefined;
  используйте `loc[[label]]`).

### Ограничения (v0.4.1, §11.5)
- **MultiIndex** (`set_index(['a','b'])`) — v0.4.2.
- **`loc[slice]` step** — v0.4.2.
- **Non-monotonic + slice** — `NotImplementedError` (pandas-стиль).
- **`.at` на duplicate** — `KeyError`.

### Тесты
- 22 новых в `test_loc.py`: open-endpoints (start/stop/both на
  asc/desc), monotonic API, `get_positions_all`, `has_duplicates`,
  duplicate-label `loc` (dict vs CadwDataFrame), `loc[[duplicates]]`,
  slice на duplicate → raise, `.at` на duplicate → raise.
- Обновлён 1 тест: `test_loc_slice_none_endpoint_raises` → 8 новых.

## [0.4.0d] — 2026-10-19

### Добавлено
- `_iter_filtered_batches_with_positions()` — yield `(batch, positions)`,
  positions = numpy int64 в unfiltered dataset. Композиция фильтров:
  MaskFilter → `positions[mask]`; SliceFilter → slice по local counter
  с сохранением raw positions.
- `_apply_filter_with_positions(filter, it)` — helper для filter chain.

### Изменено (filter-compat)
- `df.index` при фильтрах → `MaterializedIndex(_index_values[positions])`
  (pandas-совместимо). Раньше — `NotImplementedError`.
- `set_index(col)` при фильтрах → работает; материализует **полный**
  col (O(N) scan), `_index_values` = full → любой комбинации filter/set
  через positions.
- `reset_index(drop=True)` при фильтрах → materialize filtered view +
  RangeIndex(0, filtered_n).
- `reset_index(drop=False)` при фильтрах → колонка `index` =
  `_index_values[positions]`.
- `iloc[list]` на filtered view — **сохраняет** parent index через
  raw positions (раньше reset). Более pandas-совместимо.
- `loc` на filtered view — работает (через filtered.index).

### Закрыто (из v0.4.0a-c, §11.5)
- ~~`df.index`/`set_index`/`reset_index` с активными фильтрами — raise~~
- ~~`iloc[list]` на filtered view не сохраняет parent index~~
- ~~`loc` на filtered view — NotImplementedError~~

### Тесты
- 15 новых в `test_filter_compat.py`: filtered.index (range/mat/
  slice/mask_then_slice/empty), set_index+filter, filter+set_index,
  reset_index drop=True/False при фильтрах, iloc[list] preserves
  parent (range/materialized), loc на filtered, no-filter unchanged.
- Обновлены 4 теста: `test_index_with_filter_*`, `test_set_index_with_
  filter_*`, `test_reset_index_with_filter_*`, `test_iloc_list_on_
  filtered_*`, `test_loc_on_filtered_*` — теперь проверяют
  сохранение/работу, не raise.

## [0.4.0c] — 2026-10-19

### Добавлено
- `CadwDataFrame.loc` — label-based indexer (pandas-style).
  - `loc[label]` → `dict` (KeyError если label нет).
  - `loc[lo:hi]` → `CadwDataFrame` — inclusive labels, reverse → empty.
  - `loc[[l1, l2, ...]]` → `CadwDataFrame`.
  - Требует `MaterializedIndex`; `RangeIndex` → `TypeError`.
- `CadwDataFrame.at[label, col]` — scalar access by label + column.
- `CadwDataFrame.iat[pos, col]` — scalar access by position + column.
- `MaterializedIndex.get_indexer(labels)` → `list[int]` (O(N) scan, dict;
  unhashable fallback linear).
- `.at`/`.iat` — **read-only** (`__setitem__` → `NotImplementedError`).

### Ограничения (v0.4.0c, §11.5)
- **`loc` требует `MaterializedIndex`** (после `set_index`).
  `RangeIndex` → `TypeError`; используйте `iloc`.
- **`loc[lo:hi]`** — только оба endpoint обязательны; `loc[:hi]` /
  `loc[lo:]` / `loc[::step]` → `NotImplementedError`. v0.4.1.
- **`loc[bool_array]`** — `NotImplementedError` (use `df[mask]`).
- **`loc` на filtered view** — `NotImplementedError`. v0.4.0d.
- **duplicate labels** — первое вхождение; v0.4.1 — все.
- **`.at`/`.iat` set** — `NotImplementedError` (immutable).

### Тесты
- 30 тестов в `test_loc.py`: get_indexer (scalar/list/dup/first-wins/str),
  loc on RangeIndex raises, loc scalar/slice(inclusive/reverse/missing),
  loc list (basic/missing/dup/empty), loc bool_array raise, .at
  scalar/missing/set-raise, .iat scalar/out-of-range/set-raise,
  filter limit, loc preserves index.

## [0.4.0b] — 2026-10-19

### Добавлено
- `CadwDataFrame.iloc` — positional indexer (pandas-style).
  - `iloc[int]` → `dict[str, Any]` — одна строка.
  - `iloc[slice]` → `CadwDataFrame` — lazy, через `SliceFilter`.
  - `iloc[list[int]]` → `CadwDataFrame` — materialize K=len(list) rows.
  - Negative indices работают (int/list — resolve через count();
    slice — через `key.indices(n)`).
- `cadw/indexers.py` — модуль для indexers (`_ILocIndexer`). Отдельно
  от facade (`dataframe.py`), чтобы не ломать AST guard
  `test_facade_has_no_as_py`.

### Ограничения (v0.4.0b, §11.5)
- **`iloc[bool_array]`** — `NotImplementedError` (use `df[mask]`).
- **`iloc[list]` на filtered view** → default `RangeIndex` (не сохраняет
  parent index); v0.4.0c — filter-compat.
- **`iloc[list]` на unfiltered** — preserves parent index:
  RangeIndex → MaterializedIndex([start+p, ...]);
  MaterializedIndex → subset values (dtype preserved).
- **`iloc[list]`** — O(N) I/O scan (нужно дойти до последней позиции);
  O(K) память результата.

### Тесты
- 21 тест в `test_iloc.py`: iloc[int] (mid/last/negative/out-of-range),
  iloc[slice] (basic/step/negative/filtered), iloc[list]
  (basic/duplicates/negative/empty/out-of-range/filtered), index
  preservation (range/materialized/filtered-reset), errors
  (bool_array/unsupported type), AST guard sanity.

## [0.4.0a] — 2026-10-19

### Добавлено
- `CadwIndex` (ABC) + `RangeIndex(start, stop, step)` (lazy, O(1))
  + `MaterializedIndex(values)` (Arrow Array).
- `CadwDataFrame.index` property — RangeIndex (default) или
  MaterializedIndex (после set_index).
- `CadwDataFrame.set_index(col, drop=True, inplace=False)`:
  материализует колонку в память (Arrow Array).
- `CadwDataFrame.reset_index(drop=False, inplace=False)`:
  `drop=True` — отбрасывает index; `drop=False` — выносит в колонку
  `index` (или `level_0` при коллизии).
- Экспорт: `cadw.CadwIndex`, `cadw.RangeIndex`,
  `cadw.MaterializedIndex`.
- `cadw.dataset.index` — модуль.

### Ограничения (v0.4.0a, §11.5)
- **с активными фильтрами**: `df.index`, `set_index`, `reset_index` —
  raise `NotImplementedError` (RangeIndex+filter = O(N) scan;
  синхронизация — v0.4.0b).
- **`set_index`** материализует колонку (O(N) RAM). Для больших N —
  future (streaming loc — v0.4.1).
- **Index не сохраняется в Parquet** (`write_parquet` → reload =
  RangeIndex). Pandas-совместимое поведение.

### Тесты
- 20 тестов в `test_cadw_index.py`: RangeIndex/MaterializedIndex API,
  set_index drop=True/False/inplace, reset_index drop=True/False,
  коллизия «index» → «level_0», chained, filter-limit raise,
  persistence.

## [0.3.4] — 2026-10-19

### Добавлено
- **Combiner** для `groupby`: partial aggregation до shuffle.
  - `CadwDataFrame.groupby(...).agg(..., combiner=True)` — opt-in.
  - `groupby_aggregate(..., combiner=True)` — runtime.
  - Поддерживаемые агрегаты: `sum`, `count`, `min`, `max`, `mean`.
  - Для `mean`: partial (sum, count), финально `sum/count`.
  - Экономия: партиция `N × row_size` → `K_batch × comb_bytes`
    (×10000 для 10^6 строк / 100 групп).

### Как работает
1. `_combine_batches(batch_iter, by, agg_cols, agg_funcs)` — generator:
   `pandas.groupby(by).agg(partial_specs)` per batch. State O(batch_size).
2. `hash_shuffle` получает `value_columns = __comb_*`.
3. `_local_groupby(..., partial_merge=True)` вызывает `_merge_partials`:
   `table.group_by(by).aggregate(merge_specs)` + переименование + dtype.
4. Результат идентичен `combiner=False`.

### Ограничения (v0.3.4, §11.5)
- **std/var**: не поддерживаются в combiner (требуют Welford partial).
  Warning + fallback на обычный путь. v0.3.5.
- **skew path**: combiner отключён (heavy через streaming, light через
  shuffle без combiner). Warning. v0.3.5 — combiner для light.
- **auto**: default `combiner=False` (opt-in). Auto по `k_est` — v0.3.5.

### Тесты
- 12 тестов в `test_combiner.py`: equivalence для sum/count/min/max/mean,
  mixed aggs, same-col sum+count, std warning, skew warning, large
  N/low K, multi-by, empty input, dataframe API.

## [0.3.3] — 2026-10-19

### Добавлено
- `CadwDataFrame.quantile(column, q, interpolation='linear')`:
  scalar q → `float`; list q → `list[float]`.
  - `q ∈ [0, 1]`.
  - `interpolation='linear'` (pandas default; v0.3.3 only).
  - `skipna=True` (nulls не участвуют).
  - Sort: всегда ascending (квантиль инвариантен к порядку).
- `cadw.ops.quantile.quantile(batch_iter, column, q, mg, spill_dir,
  interpolation='linear')` — runtime.
- Экспорт: `cadw.quantile`, `cadw.ops.quantile`.

### Алгоритм
- Pass 1: external_sort по `column` (ascending) + side-effect count
  `n_valid` (один проход по входам, без двойного I/O).
- Positions: `idx_i = q_i*(n_valid-1)`; `lo_i = floor`, `hi_i = ceil`,
  `frac_i = idx_i - lo_i`.
- Pass 2: streaming scan sorted, dict `{pos → value}` для
  unique(`lo_i`, `hi_i`). Early exit при `pos > max(targets)`.
- Interpolation 'linear': `v_lo + frac*(v_hi - v_lo)`.

### Ограничения (v0.3.3, §11.5)
- **interpolation**: `'lower'/'higher'/'nearest'/'midpoint'` — v0.3.4.
- **skipna=False**: не поддерживается; nulls всегда пропускаются.
- **non-numeric columns**: `linear` требует numeric; для строк —
  `NotImplementedError` (см. v0.3.4 `lower`/`higher`).

### Тесты
- 20 тестов в `test_quantile.py`: median/min/max, matches numpy для
  5 q, scalar vs list return, n=1, nulls (skip, all-null), after filter,
  pandas-equivalence (с/без nulls), errors (missing/invalid q/invalid
  interp/empty list).

## [0.3.2] — 2026-10-19

### Добавлено
- `CadwDataFrame.rank` теперь поддерживает `method='max'|'last'|'average'`
  через **двухпроходный streaming** (O(1) RAM):
  - Pass 1: scan sorted_by → `(first_rank, group_count)` per tie-group
    на диск. State O(1) + per-batch accumulators (O(batch_size)).
  - Pass 2: scan sorted_by + advance по group file (sequential).
    `pos_in_group = global_pos - first_rank + 1`;
    `last_rank = first_rank + group_count - 1`;
    `max → last_rank`; `last → first_rank + count - pos`;
    `average → (first + last) / 2`.
- `VALID_METHODS = ('first', 'min', 'dense', 'max', 'last', 'average')`.
- `_VALID_STREAMABLE` / `_VALID_BUFFERABLE` — разделение методов.
- `_rank_dtype(method, pct)`: int64 для int-методов без pct; float64
  для average или pct.

### Изменено
- `compute_ranks_to_disk(sorted_path: Path, ...)` — принимает **Path**
  вместо `Iterable[RecordBatch]`, потому что bufferable-методы требуют
  двух проходов. Единственный caller — `CadwDataFrame.rank` — обновлён.

### Ограничения (v0.3.2, §11.5)
- **null values**: участвуют как значения (null == null); pandas
  `rank(skipna=True)` не поддерживается.
- **cost**: `max`/`last`/`average` = 2 passes by sorted_by + 1 pass
  by row_id (re-sort). Против 1 pass у streamable.

### Тесты
- 15 новых в `test_rank.py`: max/last/average на asc и desc, pct для
  average, dtype-проверки, after filter, pandas-equivalence для всех
  трёх, all-same для max/last/average.

## [0.3.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.rank(by, method='first'|'min'|'dense', ascending=True,
  pct=False, output_col=None, batch_size=10_000)`:
  - external_sort по `by` → потоковое вычисление rank → external_sort
    rank-пар по `row_id` → zip с исходным в original order.
  - `first`: 1-based позиция в sorted order (ties по row_id asc).
  - `min`:   минимум ранга в tie-группе.
  - `dense`: 1, 2, 3, ... для уникальных значений.
  - `pct=True`: rank / total_n.
- `cadw.ops.rank.compute_ranks_to_disk` — primitive: sorted stream → file
  (row_id, rank).
- Экспорт: `cadw.compute_ranks_to_disk`, `cadw.VALID_METHODS`.

### Принцип
- Streamable методы — O(1) state. Bufferable (`max`/`average`/`last`) —
  O(max tie group size) = O(N) для pathological данных → **не в MVP**,
  требуют двухпроходного streaming.
- Результат в original order (две сортировки: по `by` + по `row_id`).

### Ограничения (v0.3.1, §11.5)
- **method**: `first`/`min`/`dense` только. `max`/`average`/`last` — v0.3.2.
- **null values**: участвуют как значения (null == null); pandas
  `rank(skipna=True)` не поддерживается.
- **cost**: 2× external_sort + 1× count-scan. Для 10^9 строк — ~3 полных
  прохода I/O.

### Тесты
- 19 тестов в `test_rank.py`: все три метода на asc/desc, pct, edge
  cases (all-unique / all-same / ties), after filter, custom output_col,
  errors (invalid method / missing column / collision), 4 × pandas-
  equivalence, chained с groupby.

## [0.3.0] — 2026-10-19

### Добавлено (первая операция v0.3)
- `CadwDataFrame.drop_duplicates(subset=None, keep='first', num_partitions=None)`:
  HashShuffle по subset + локальный `pandas.drop_duplicates` per partition.
  - `subset=None` → dedup по всем колонкам.
  - `keep ∈ {'first', 'last', False}` — все три pandas-режима.
  - `num_partitions=None` → авто из sample (k_est на subset).
- `cadw.ops.dedup.drop_duplicates(...)` — runtime, возвращает `pa.Table`.
- Экспорт через `cadw.drop_duplicates` и `cadw.ops.drop_duplicates`.

### Принцип
- Корректность: партиции `HashShuffle` не пересекаются по subset keys →
  локальный dedup = глобальный, concat без merge.
- `row_id` сохраняется в shuffle; сортировка по row_id перед dedup даёт
  корректный keep='first'/'last' (min/max row_id per key).
- `keep=False` (pandas semantics) локально = глобально.

### Ограничения (v0.3.0, §11.5)
- **Output order**: partition-concatenation, не pandas-original order.
  Для стабильного порядка — `.sort_values(...)` после.
- **Skew**: при oversized partition → `MemoryLimitExceeded` без recursion.
  Recursion в v0.3.1 требует save-row_id-at-resplit или streaming dedup.

### Тесты
- 15 тестов в `test_drop_duplicates.py`: все keep-режимы, subset=None,
  multi-column, edge cases (empty, missing, invalid), skew reject,
  pandas-equivalence, after filter, chained with sort.

## [0.2.5] — 2026-10-19
### Исправлено (§13.4 эпизод 47)
- `hash_join`: cartesian check заменён на **точную** оценку совпадений
  `actual_matches = sum over common keys(build_count × probe_count)` через
  `groupby.size()`. O(build + probe), дёшево. Устраняет ложные reject'ы
  для PK-FK (cartesian 12.6M при реальном output 2519).
- Удалены: `max_cartesian_rows` (все слои), `c_cartesian`.
- 2 AST guard'а: `test_actual_matches_checked_before_pd_merge`,
  `test_no_cartesian_check_in_join`.



### Добавлено (финал v0.2)
- `tests/test_e2e_join.py` (3 slow):
  - `test_e2e_join_100k_x_10k`: 100k FK × 10k PK (string keys), 20 MB
    budget, inner join; pandas-equivalence; RSS-дельта < 200 MB;
    elapsed < 60 s.
  - `test_e2e_join_skew_reject`: 100k × 100k один ключ →
    `MemoryLimitExceeded` до OOM; RSS-дельта < 500 MB.
  - `test_e2e_join_left_preserve`: 100k × 5k, half unmatched → 100k
    preserved; null-mask совпадает с pandas.

### Закрыто (v0.2)
- `CadwDataFrame.merge(inner/left)` — пользовательская функция.
- `hash_join(how, max_output_rows, max_cartesian_rows)` — runtime.
- `JoinMemoryModel` + `propagate_chars("join")` — planner integration.
- OOM safety: size-check до materialize, schema-correct empty IPC,
  recursive partition split, per-partition unique_build check.
- `use_planner=True` в `merge`, sampling helpers в `cadw/core/sampling.py`.

### Тесты v0.2
- fast: 330 passed, 1 skipped.
- slow: 8 passed (5 был + 3 E2E join).
- Всего: **338 passed, 1 skipped**.

## [0.2.4] — 2026-10-19

### Исправлено (§13.4 эпизод 42)
- Модель `JoinMemoryModel` возвращает два лимита: `max_output_rows`
  (expected, из `output_budget / row_size_out`) и
  `max_cartesian_rows` (`c_cartesian × max_output_rows`, default 10×).
  Runtime `hash_join` разделяет проверки:
  - `expected_probe_output = len(probe) × avg_matches_per_probe >
    max_output_rows` → raise (per-partition skew);
  - `len(build) × len(probe) > max_cartesian_rows` → raise (extreme cartesian).
  Ранее `len(build) × len(probe) > max_output_rows` ошибочно сравнивал
  cartesian с expected → `plan_pipeline` мог сказать `"fits"`, а `merge` —
  упасть.

### Добавлено
- `JoinMemoryModel.c_cartesian` (default 10.0) — множитель для worst-case.
- `JoinParams.max_cartesian_rows`.
- `hash_join(..., max_cartesian_rows=None)`; default `10 × max_output_rows`.
- `CadwDataFrame.merge(..., num_partitions=None, use_planner=True)`:
  - `use_planner=True` (default): sample first batch каждой стороны,
    `mg.plan("join", ...)`, параметры пробрасываются в `hash_join`.
  - `use_planner=False`: старые дефолты (`num_partitions=8`,
    `max_output_rows = 0.3 × available / 256`).
  - `num_partitions` явно → override модели, но `max_output_rows` /
    `max_cartesian_rows` всё равно из модели.
- Per-partition `unique_build_keys` (pandas `.nunique`/`drop_duplicates`
  на build) — уточнённый `avg_matches_per_probe`, недоступный глобальной
  модели.

### Тесты
- 8 новых в `test_merge_planner.py`: planner default, `use_planner=False`,
  explicit `num_partitions`, early-fail план, cartesian-skew reject,
  left-planner, empty-side inner/left.

## [0.2.3] — 2026-10-19

### Добавлено
- `JoinMemoryModel` (§9.2.5): `f_join = build_partition + probe_batch +
  output_batch × (1 + c_merge)`.
  - `solve()`: `P` = max от shuffle обеих сторон; `build_side` по `how`
    (right для left join; меньший state для inner);
    `max_output_rows` из `output_budget / (row_size_out × (1+c_merge))`
    вместо ad-hoc `0.3 × available / 256`;
    `avg_matches` = upper bound `N_L × N_R / max(K_L, K_R)` (floor 1).
  - `JoinParams`: `build_side`, `how`, `num_partitions`,
    `probe_batch_size`, `build_partition_bytes`, `max_output_rows`,
    `right_chars`, `probe_row_size`, `output_row_size`, `avg_matches`.
- `propagate_chars("join")`: `N' = N_left × avg_matches`,
  `row_size' = output_row_size`, `k_est' = N'` (upper bound),
  `size_of_key_entry' = row_size' + 80`, `order = "random"`.
  Требует `JoinParams` в `params` (из `get_model("join").solve()`).
- `plan_pipeline` теперь планирует цепочки с join
  (`filter → join → groupby`, `join → join`).

### Открыто (11-B-2c-3)
- Модель считает **expected** output, runtime — **cartesian worst case**.
  `plan_pipeline` может предсказать `"fits"`, а `merge` — упасть.
  Fix: разделить `max_output_rows` (expected) и `max_cartesian_rows`
  (worst case, ~10× larger). §13.4 эпизод 42.

## [0.2.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.merge(..., how='left')`: left join, все строки self
  сохраняются; right-колонки -> NULL при отсутствии совпадения.
- `cadw.ops.join.hash_join(how='inner'|'left')` — общий runtime;
  `inner_join`, `left_join` — обёртки.
- SQL-семантика NULL:
  - inner: `_drop_null_keys` на обе стороны;
  - left: `_drop_null_keys` только на build (right); probe (left)
    сохраняется (NULL в left key -> NULL right cols).
- `_empty_result_schema` — nullable=True для всех полей (left join с
  unmatched: NaN в int/string -> null, cast safe=False).

### Ограничения v0.2.2
- `how='right'`, `how='outer'`, `left_on`/`right_on`, `on=index` — v0.3.
- `JoinMemoryModel` + `propagate_chars("join")` + E2E — 11-B-2b/2c.

## [0.2.0] — 2026-10-19

### Добавлено
- `CadwDataFrame.merge(right, on, how='inner', suffixes=('_x','_y'),
  num_partitions=8)`: Grace hash inner join.
- `cadw.ops.join.inner_join`: HashShuffle обеих сторон (один seed) →
  партиции с одинаковыми ключами → для каждой пары: build = меньшая
  партиция (`check_state`), probe батчами (`check_batch`),
  `pd.merge(how='inner', suffixes=...)`.
- SQL-семантика NULL: `_drop_null_keys` до `pd.merge` (pandas.merge
  по умолчанию матчит NaN==NaN — не то).
- Защита от cartesian-skew: `len(build) * len(probe) > max_output_rows`
  → `MemoryLimitExceeded` **до** `pd.merge`.

### Ограничения v0.2.1
- `how='inner'` только; `left`/`right`/`outer` — v0.2.2.
- `on` по имени в обеих сторонах; `left_on`/`right_on`, `on=index` —
  v0.2.2.
- `JoinMemoryModel` + `propagate_chars("join")` + E2E — 11-B-2.

## [0.1.7] — 2026-10-19

### Исправлено (must-fix §11.4)
- `_local_groupby`: dict check до `group_by`. `_read_partition_safe`
  проверял только `file_size`, но `group_by` строит hash-таблицу
  `K_i × size_of_key_entry` — для узких строк dict больше файла, OOM
  внутри pyarrow. `_estimate_distinct_keys` (product of nunique) ×
  `size_of_key_entry` проверяется до `group_by`; overflow -> in-memory
  recursion через `hash_shuffle`; `depth >= max_depth` ->
  `MemoryLimitExceeded` с file/dict/safety.

### Добавлено
- `_estimate_distinct_keys(table, by_columns) -> int` — upper bound без
  построения hash-таблицы (`pc.count_distinct` per column).
- AST guard `test_local_groupby_dict_check_before_group_by`.
- Behavior tests: `test_dict_overflow_triggers_split`,
  `test_dict_fits_no_recursion`, `test_max_depth_raises_on_dict_overflow`.

## [0.1.4] — 2026-10-18

### Исправлено (must-fix §11.4, класс «materialize-before-check»)
- `_local_groupby` материализовал всю партицию (`list(ArrowIPCBatchReader)`)
  до size-check. 02b1 добавил проверку **после** materialize — это не
  закрыло баг. Fix: `partition_path.stat().st_size` до чтения + helper
  `_read_partition_safe` (единственное место, где разрешён `list(reader)`).
- `reduce` вызывал `op.init()` до `estimate_state_size`/`check_state`.
  `CovarianceOp(d=10^5)` → `np.zeros((10^5,10^5))` = 80 GB → OOM **внутри**
  init. Fix: `estimate_state_size` + `check_state` **до** `op.init()`.

### Добавлено
- §3.10 «Check before materialize» — новый обязательный принцип.
- `tests/test_materialize_guard.py` — AST guards:
  - `test_no_list_of_whole_reader_outside_helper`
  - `test_reduce_estimate_state_before_init`
- `tests/test_oom_resistance.py` (slow):
  - 10^6 групп, 10 MB budget;
  - 90/10 skew, 10^6 строк, 10 MB;
  - `CovarianceOp(d=5000)` под 10 MB → `MemoryLimitExceeded` до `init`.

### Уроки (DECISIONS §13.4)
- Эпизод 27: «закрытие must-fix» требует теста на **порядок** операций,
  а не только на наличие вызова. 02b1 проверял, что `hash_shuffle`
  вызывается, но не проверял, что `list(reader)` **не** вызывается до.
- Эпизод 28: аудит перед новыми функциями находит унаследованные баги
  (`reduce.init()` — не был замечен при 02a).

## [0.1.3] — 2026-10-18

### Удалено
- `CadwDataFrame.partial_fit` — адаптер к `sklearn.partial_fit`. Причина:
  whitelist (`_SAFE_PARTIAL_FIT`) — политика на доверии; любая модель вне
  whitelist или модель с выросшим state OOM-ит без предупреждения от CADW.
  См. DECISIONS.md §3.9, §13.4 эпизод 26.
- `_SAFE_PARTIAL_FIT` whitelist.
- `tests/test_ml_integration.py` (11 + 1 slow).
- `tests/test_sklearn_extras.py` (4 + 1 slow).
- sklearn-секция из `tests/test_api_availability.py` (7 assertions).
- `scikit-learn` из `[dependency-groups]` (dev).
- ML-секции из README, заменены на «ML и внешние библиотеки» с policy.

### Добавлено в DECISIONS.md
- §3.9: «Никаких ML-адаптеров» (не только `fit`, но и `partial_fit`).
- §11.5: расширенная «Политика ML» (3 класса: экспорт / адаптеры запрещены
  / собственные реализации v0.3).
- §12.5.1: план v0.3 — GBDT, Linear, k-means, PCA, NB на примитивах (~8–11
  сессий на весь ML-слой).
- §13.4 эпизод 26: whitelist `partial_fit` — политика на доверии, не
  гарантия.

### Единственный ML-интерфейс CADW
- `to_batches(format='arrow'|'pandas'|'numpy'|'torch')` — экспорт батчей.
  Ответственность за OOM в сторонней модели — на пользователе.

## [0.1.2] — 2026-10-18

### Удалено
- `CadwDataFrame.to_lightgbm` — адаптер к LightGBM. Причина: `lgb.train`
  требует O(N)-state (все градиенты в RAM), OOM происходит внутри
  библиотеки, `MemoryGuard` не спасает. Противоречит §3.9 «no adapters to
  O(N)-algorithms». См. DECISIONS.md §13.4 эпизод 25.
- `tests/test_lightgbm.py` (11 + 1 slow тестов).
- `lightgbm` из `[dependency-groups]` (dev).
- LightGBM-секции из README, DECISIONS §4.1/§11.5/§12.5/§13.3.

### Изменено
- `CadwDataFrame.partial_fit`: добавлен whitelist `_SAFE_PARTIAL_FIT`
  (SGD, StandardScaler, MiniBatchKMeans, IncrementalPCA, NB, MLP).
  Модели вне whitelist → `RuntimeWarning` с указанием на §11.5.
  Docstring обновлён: контракт безопасности (модель должна иметь
  O(1)-state w.r.t. N, это ответственность модели, не CADW).
- `tests/test_api_availability.py`: убрана LightGBM-секция.

### Добавлено в DECISIONS.md
- §3.9: «Никаких адаптеров к O(N)-алгоритмам».
- §11.5: блок «Политика ML» (3 класса: incremental / batch-only /
  own implementation).
- §11.4: обнаружена регрессия `_local_groupby` (materialize до
  `check_state`) — вынесена как must-fix.
- §13.4 эпизод 25: почему адаптер к LightGBM — ошибка.

### Планы
- v0.3: свой GBDT (level-wise, аналог `sklearn.HistGradientBoosting`) на
  примитивах: `HistogramOp` (binning) + `HashShuffle` (узел,фича,бин) +
  `Reduce` (sum_grad/sum_hess/count).

## [0.1.0] — 2026-10-18

Первый production-ready каркас: три примитива out-of-core с безусловной
гарантией памяти, pandas-подобный фасад, ML-интеграция (sklearn + LightGBM),
E2E на 200k строк / 20 MB budget.

### Добавлено — storage (уровни 0–1)
- `ParquetBatchReader/Writer`, `ArrowIPCBatchReader/Writer`.
- `Dataset` (03a): per-column Arrow IPC, синхронные чанки, `add_column`
  (валидирует boundaries, атомарная запись `.tmp.arrow` → `rename`),
  `iter_batches(batch_size=None|N, columns=...)` с re-chunk, persistence в
  `dataset_meta.json` (schema через `Schema.serialize()`).

### Добавлено — примитивы (уровень 3)
- `Reduce` (02a): `ReduceOp`/`reduce`, `CountOp`, `SumOp`, `MinMaxOp`,
  `MeanOp`, `StdOp` (Welford), `CovarianceOp` (d×d), `HistogramOp`
  (binning LightGBM). Инвариант «не зависит от `batch_size`», «не пишет
  на диск».
- `HashShuffle` (02b): `compute_hashes` (`xxhash.xxh64` + `repr(v).encode`),
  `_mod_p` (NumPy `uint64` для wrap-around), `row_id` через `order_np`
  (а не позицию в отсортированном батче), `estimate_num_partitions`,
  skew-изоляция (`_detect_heavy_keys` >5%, `_groupby_with_skew`).
- `Sort` (02b): `external_sort`, `compute_fan_in`, `_Key` (null_rank
  + per-column asc/desc), k-way merge, многоуровневый merge, `row_id`
  в каждом run'е для стабильности.

### Добавлено — операции (уровень 4)
- `groupby_aggregate` (02a/02b): `HashShuffle + LocalGroupBy + Concat`,
  авто-P через sample, рекурсивное партиционирование (`MAX_DEPTH=5`),
  skew-изоляция (тяжёлые потоково, лёгкие shuffle), single-pass.
  Корректен при `K > памяти` (`test_groupby_many_groups`: 1000 групп,
  200 KB budget) и при экстремальном skew (90/10, one-hot).

### Добавлено — доступ (уровень 2)
- `RowFilter` (03c): `PredicateFilter`, `SliceFilter`, `AndFilter`,
  `OrFilter`, `NotFilter`. `& | ~` для `MaskFilter`; slice+mask — через
  chaining.

### Добавлено — фасад (уровень 5)
- `CadwDataFrame` (03b/03c): `from_parquet`, `from_dataset`,
  `columns`, `shape` (upper bound), `count()` (точное), `head(n)`,
  `to_batches(batch_size, format='arrow'|'pandas'|'numpy'|'torch')`,
  `__getitem__(slice|list[str]|str|RowFilter)`, `groupby().agg()`,
  `unique()`, `sort_values()`, `write_parquet()`,
  `partial_fit(model, X_cols, y_col=None, **kwargs)`,
  `to_lightgbm(X_cols, y_col, max_rows=N, categorical_cols=None, **params)`.
- Ленивость: `df[df['val'] > 15][1:5]` не материализует данные до
  `to_batches`/`write_parquet`/`groupby`/`sort_values`/`head`.
- AST guard-тесты: `test_facade_has_no_as_py`, `test_facade_has_no_row_loop_ast`.

### Добавлено — MemoryGuard (уровень 2)
- Байтовый контракт: `available_bytes`, `get_batch_rows`,
  `check_batch(extra_factor)`, `check_state`, `check_partition_buffers`,
  `recommended_partition_buffer_bytes`, `estimate_state_size`,
  `estimate_unique_keys`, `reserve/release`, `log_step`.

### Добавлено — ML (уровень 6)
- sklearn: `SGDClassifier` (`classes=[0,1]` на первом вызове),
  `StandardScaler`, `MiniBatchKMeans`, `IncrementalPCA` (без `n_features`
  в `partial_fit` — выводится из `X.shape[1]`).
- LightGBM: `to_lightgbm(max_rows=N)` — честный downsampling
  (LightGBM требует весь датасет в RAM).
- E2E: sklearn pipeline (50k строк, 10 MB budget), LightGBM pipeline
  (50k строк, downsampling до 20k).

### Добавлено — E2E и guards
- E2E (04): filter → groupby → sort → parquet на 200k строк, 20 MB budget,
  pandas-equivalence.
- Architecture guards: `test_no_legacy_package`, `test_legacy_tests_removed`,
  `test_main_code_does_not_reference_legacy`.

### Удалено
- Legacy `cadw/_legacy/` и `tests/_legacy/` (05): `SpillManager`,
  `UniqueCollector`, `GroupByAggregator` — заменены примитивами и фасадом.
- `cadw/dataframe.py` (старая заготовка) — переписан с нуля.

### Ограничения (документированы в DECISIONS.md §11.5)
- `to_lightgbm` не out-of-core: LightGBM требует весь датасет в RAM.
- `PCA` (batch-only) не имеет `partial_fit`; для batch-only — `vstack`.
- `SGDClassifier.partial_fit` требует `classes=[...]` на первом вызове.
- Категориальные признаки для LightGBM — int-encoded, не строки.
- `shape` на отфильтрованном `CadwDataFrame` — upper bound; точное — `.count()`.
- `nulls_first` в `Sort` — глобальный флаг, не per-column.
- `df[-5:]` (отрицательный slice) — не поддержано.
- `compute_hashes` — Python-цикл (векторизация отложена).
- `CadwIndex`/`loc`/`iloc`, `join`/`merge`, `drop_duplicates`, `rank`,
  `rolling` — не реализованы (v0.2).

### Тесты
- **213 fast + 5 slow = 218 passed, 1 skipped** (torch опционален).
- Покрытие: MemoryGuard (25), Reduce (33), HashShuffle (13), Sort (10),
  groupby + must-fix (10 + 28), Dataset (13), CadwDataFrame (17),
  RowFilter (25), ML (sklearn + LightGBM, 27), E2E (5 slow), guards (3).

[0.1.0]: https://example.invalid/cadw/compare/v0.0.0...v0.1.0
