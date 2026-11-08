# CADW Roadmap

Last updated: 2026-11-08 (after v0.9.136).

## Current: v0.9.136 — DEC-054 dedup + DEC-055 depends_on + DEC-056 PRIMITIVES_BACKLOG

**5 releases since v0.9.128.** v0.9.129 = verify 3 ts drafts;
v0.9.130-133 = DEC-054 dedup pass (7 clusters). 30 files touched,
~600 lines removed via AST-based extraction.

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

### v0.9.137 — DEC-057 impl: fit_batched/predict_batched + test fix

**Part A (~30 min):** `test_fit_ooc_v106.py`: `["py_bytes"]` ->
`["rss_bytes"]`. Extend 6 -> 12 representative models.

**Part B (~30 min):** schema `ModelCard.fit_batched` +
`predict_batched` + `_parse_model`.

**Part C (~30 min):** R13 test (bounded_state=true + fit_batched=false
requires streaming keyword).

**Part D (~30 min):** migration 66 cards (2-pass AST with delegate
list + cadw/ts scan). Expect ~54 true / ~12 false with bounded_by_domain
or refuse_at_guard.

### v0.9.138 — P1 group A: distances + kernels
L1 `_knn_query`, L2 `_core_distances`, L3 `_rbf_kernel`, L4 `_chol_solve`.
Unblocks: LOF, DBSCAN, HDBSCAN, SMOTE, GPR, GP forecaster, spectral
(pending L6), GPLVM, KernelPCA, GP-EI, BayesianOpt, KNNImputer.

### v0.9.139 — P1 group B: graph
L5 `_mst`, L6 `_graph_laplacian` + `_eigsh_topk`, L19 `_eigen_decomp_topk`.
Unblocks: HDBSCAN dendrogram, Agglomerative single, Spectral x3, Consensus x3.

### v0.9.140 — P1 group C: factorization + itemset
L7 `_nmf_mu`, L8 `_itemset_mining` (Apriori/FP-tree/Eclat).

### v0.9.141 — P1 group D: losses + stats
L9-L20: pinball, survival logrank, DR score, LTR gradients, grid_search,
arm_stats, adaboost, ks_stat, cumulative_logit.

#### v0.9.134 — home-correction + DEC-054 tail (DONE, see CHANGELOG)

**Part A (~1 h): home fixes.**
- `sigmoid` -> `cadw/ops/numeric.py` (from `optim.py`: SGD there, sigmoid
  is elementwise numeric, not optimization).
- `znormalize_rows` -> `cadw/ops/stats.py` (new; mean/std = statistics).
- `ols_fit` -> `cadw/ops/linalg.py` (new; or keep in numeric with note).
- `_xavier_init` -> `cadw/ml/_nn_init.py` (new; NN-specific).
- Update all importers (`gbdt`, `calibration`, `ordinal`, `autoencoder`,
  `patch_memory_ad`).

**Part B (~1 h): forward-looking extraction (AP10 exception for >=2
prospective consumers).**
- `_sq_dists` -> `cadw/ops/linalg.py` (prospective: kmeans_ad, kshape,
  tspq, iforest, eif, HDBSCAN, DBSCAN, minibatch_kmeans — 8+).
- `_acf` -> `cadw/ops/stats.py` (prospective: binseg, pelt, bocpd,
  matrix_profile, stray — 5+).

**Part C (~30 min): update DEC-054 with scope actually applied vs
deferred.**

#### v0.9.135 — schema `depends_on` + card migration (DONE)

**Part A (~1.5 h): schema extension.**
- `ModelCard.depends_on: list[str] | None` in `schema.py`.
- `test_R11_depends_on_importable`: every dotted path in `depends_on`
  must resolve via `importlib`.
- `test_R12_depends_on_matches_code`: AST scan of `cadw_module`,
  every top-level `from ... import ...` of a helper in `cadw.ops.*` /
  `cadw.ts._*` / `cadw.ml._*` must appear in `depends_on` (or whitelisted
  as stdlib/np/pa).

**Part B (~1 h): migration.**
- AST-extract imports for 51 verified + 14 draft cards.
- Add `depends_on` field with explicit list.
- described_only cards: `depends_on: []` (nothing to import yet).

#### v0.9.136 — PRIMITIVES_BACKLOG.md (forward scan) (DONE)

**Scan all 302 cards** (48 verified + 14 draft + 61 described_only +
6 inference-only + 5 last-priority + 42 new candidates + 42 not-impl).

For each described_only card: read `purpose` + `algorithm_source_url`,
infer which primitives would be needed. Cluster by primitive:

- `_sq_dists` (Euclidean pairwise) — needs: kmeans, kmeans_ad, kshape,
  tspq, iforest, eif, DBSCAN, HDBSCAN, LOF, bisecting, minibatch,
  spectral — **12+**.
- `_rbf_kernel` — spectral, GP, kernel PCA, kernel k-means, MMD — **5+**.
- `_cholesky_solve` / `_gp_predict` — GP regression, GP forecaster,
  GP-EI, BayesianOpt — **4+**.
- `_mutual_reachability` + `_mst` — HDBSCAN, OPTICS, single-linkage — **3+**.
- `_itemset_count` + `_fp_tree` — Apriori, FP-growth, Eclat — **3+**.
- `_leapfrog` + `_hmc_accept` — NUTS, Metropolis, HMC — **3+**.
- `_sparse_coding` / `_nmf_mu` — NMF, SparsePCA, LDA — **3+**.
- `_softmax_ce_grad` — ListNet, LambdaMART, GBDT-multiclass — **3+**.
- `_fft` (numpy alias) — kshape, miniRocket, ROCKET, matrix_profile — **4+**.
- ~20-30 more clusters.

**Deliverable:** `docs/ml_tools/PRIMITIVES_BACKLOG.md` with:
- cluster table (primitive, consumers, tier priority: 3+ = P1, 2 = P2,
  1 = P3 with prospective notes).
- estimated effort per primitive.
- home module per primitive.

**End of "Next 3 sessions" — v0.9.137+ = implement primitives by tier,
then verify remaining draft models (BVAR, VECM, PELT, T2Q, STRAY).

---

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

**v0.9.129-133 (this session, 5 releases):**
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
