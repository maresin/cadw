# CADW Roadmap

Last updated: 2026-11-08 (after v0.9.141).

## Current: v0.9.141 — all 10 §1 drafts verified + DEC-054/055/056/057

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

### v0.9.142 — `fit_stacking` pilot + predict_batched measurement

**Part A (~1 h): `fit_stacking` infrastructure.**
`cadw/ml/stacking.py::fit_stacking(df, base_models, meta_learner,
*, cv=5, output="auto")`. k-fold OOC + meta on OOF predictions.
Reuse `_iter_filtered_batches` for fold iteration; base models are
existing `fit_*` (fit on train fold); meta-learner fit on OOF.
Returns `StackingModel` with `.predict_proba(df)` (SpillArray auto).
Not a card — infrastructure (like `fit_ab_test`).

**Part B (~30 min): predict_batched measurement.**
For 8 `fit_batched: false` cards (OTM/THF/TTF, PELT, STRAY,
BinarySegmentation): add `test_predict_batched_matches` — verify
`predict` path does not do unexpected full read. For 5-10
representative `fit_batched: true` cards: measure peak(n=1e4 vs 5e4)
on predict with output="auto" (SpillArray).

**Part C (~30 min): EnsembleForecaster backlog.**
YAML: draft_reason = "verify blocked on fit_stacking (DEC-050 level 3);
per-member OOC unverified". After fit_stacking lands, rewrite
EnsembleForecaster as thin wrapper over fit_stacking.

### v0.9.143 — P1 group A: distances + kernels
L1 `_knn_query` (chunked sq_dists + argpartition).
L2 `_core_distances` (k-th neighbour distance; builds on L1).
L3 `_rbf_kernel` + `poly_kernel` + `laplacian_kernel` (pairwise kernel
general helper).
L4 `_chol_solve` (np.linalg.cholesky + solve; jitter=1e-8).
Home: `cadw/ops/linalg.py`. Unblocks: LOF, DBSCAN, HDBSCAN, SMOTE,
GPR, GP forecaster, KernelPCA, GPLVM, GP-EI, BayesianOpt, KNNImputer.

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
| verified | **32** | 48 | framework + 6 recent (v0.9.114-119) |
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
