# CADW Roadmap

Last updated: 2026-11-04 (after v0.9.114).

## Current: v0.9.125

**32 verified fns / 48 verified cards / 10 draft fns / 13 draft cards /
24 pending fns / 30 pending cards / 62 no-fn. +19 described_only in P/Q.**

v0.9.116-v0.9.118 (since v0.9.115):
- v0.9.116: sub_pca verified (x2 alias); df-API migration for bocpd/eif/kmeans_ad
  (input_api series->dual, R8 = 0 warnings); DEC-034 partial (lda_peak
  measure_peak_traced); DEC-035 (BIRCH _insert split row-by-row argmin).
- v0.9.117: LDA + TF-IDF verified (component-wise DEC-033); auto K-spill
  (K_raw_est > state_limit//96//2 -> hash_shuffle, else legacy);
  DEC-041 + DEC-042.
- v0.9.118: ordinal_logit + chains (x2 alias) verified; R9 (cadw_function
  importable schema test); R8 was added v0.9.115.

**v0.9.119 (since v0.9.118):**
- pipeline planner v2 (`plan_pipeline(memory_guard=...)`): segments by
  `OpSpec.spill_after`, carried state between segments, per-segment batch,
  time estimation via `throughput.yaml`. Back-compat: `memory_guard=None` -> v1.
- `CostEstimate` + `MemoryGuard.estimate_cost` + `MemoryModel.estimate_time`
  (AP22 skeleton; calibration `scripts/calibrate_throughput.py` -> v0.9.120).
- `MemoryGuard.solve_batch` (DEC-044): inverse of `ram_formula` via numeric
  sampling + `_is_linear_in_batch` guard; `None` -> fallback expansion_factor.
- `fit_ab_test` x3 verified (ProportionsZTest/MannWhitneyU/CUPED): OOC via
  `Reduce` (t/cuped) and `external_sort` + streaming rank (mw);
  component-wise (DEC-033). `DriftABModel.result_` populated.
- P1-P8 synthetic pipeline tests (`tests/test_pipeline_planner_examples.py`).
- DEC-043 (pipeline planner v2), DEC-044 (inverse via solve_batch).

**v0.9.120 (since v0.9.119):**
- P6'-P9' real pipelines (`tests/test_pipeline_planner_real.py`, 9 tests):
  `model_spec={"cadw_function":..., "ram_params":{...}}`; `solve_batch`
  inverse resolves real YAML formulas for verified models.
- Bug fix (planner): model_spec output_chars used `state_bytes` (=0 when
  omitted) instead of computed `estimate` (solve_batch + evaluate_formula).
- `scripts/calibrate_throughput.py` + 5 tests: dry-run/write for
  reduce/sort/hash_shuffle; PyYAML 1.1 string-float quirk fixed.
- DEC-045 (P6'-P9' validation + YAML coverage gap: fit_kmeans/fit_pca),
  DEC-046 (calibrate_throughput design; per-model time -> v0.9.121+).

**v0.9.125 (since v0.9.124):**
- R_association.yaml (5 cards), S_generative.yaml (5), T_optimization.yaml
  (5), U_automl.yaml (3), V_uncertainty.yaml (5), W_probabilistic.yaml (4)
  -- 27 described_only cards, 6 new sections.
- R: Apriori, Eclat, FP-growth (SOTA), GSP, PrefixSpan (SOTA).
- S: TVAE, CTGAN, CTAB-GAN+, TabDDPM, TabSyn (all tier3_export, torch).
- T: GP-EI, TPE (SOTA), SMAC, Nelder-Mead, CMA-ES (SOTA).
- U: Auto-sklearn, H2O AutoML, AutoGluon 1.2 (SOTA per AMLB 2025).
- V: Platt, Temperature, Isotonic (SOTA), SplitConformal, CV+ (SOTA).
- W: Mean-field VI, Black-box VI (SOTA), Metropolis-Hastings, NUTS (SOTA).
- benchmarks.yaml: +HPOBench (tabular_hpo; canonical HPO LB).
- Cumulative described_only: 46 (19 P/Q + 27 here).

**v0.9.124 (since v0.9.123):**
- .pingis/impl/ sync into cards: 10 described_only cards get `purpose`
  reference to working sandbox impl + REPORT.md §6 smoke evidence
  (I_*.KaplanMeier/WeibullAFT; J_*.ComplementNB/NMF;
  M_*.EpsilonGreedy/UCB1/ThompsonSampling; O_*.ReservoirSample/
  StratifiedSplit/OneHotEncoder). No new fields, no benchmark_results.
- 4 deferred O_eda cards (KNNImputer, KBinsDiscretizer, GridSearch,
  RandomizedSearch) get `purpose` reference to REPORT.md §4
  rationale (memory model proved by N/KernelDensity or O/OneHotEncoder).
- L_graph / K_vision unchanged: headers already state 'Out of scope:
  torch required' = REPORT.md §7 (no duplicate).
- Status unchanged (described_only); promotion to verified_in_cadw
  -> v0.9.126+ (sandbox .pingis/impl/ not in cadw/ml/).

**v0.9.123 (since v0.9.122):**
- P_clustering.yaml (new, 13 cards): 2 draft (KMeans, MiniBatchKMeans;
  formula from code) + 11 described_only. Subtypes P1-P6.
- Q_dim_reduction.yaml (new, 9 cards): 1 draft (PCA; formula from
  code) + 8 described_only. Subtypes Q1-Q4.
- Cross-ref policy (DEC-049): canonical home + markdown comment.
- R10 whitelist empty: fit_kmeans/fit_minibatch_kmeans/fit_pca in cards.
- Deferred: R_association, S_generative, T_optimization, U_automl,
  V_uncertainty, W_probabilistic (v0.9.124, all described_only).

**v0.9.122 (since v0.9.121):**
- TASK_TYPES 14 -> 28 (DEC-048). +14 new types (clustering, dim_reduction,
  matrix_factorization, association_rules, generative_modelling,
  structured_prediction, feature_selection, covariance_estimation,
  biclustering, manifold_learning, optimization, automl,
  uncertainty_quantification, probabilistic_inference).
- Discovery: model-level `task_type` = 0/302 cards. Extension safe;
  adoption -> v0.9.123+ (per section, together with candidate selection).
- CADWBoost reference-only card in A_classification (tier: sota,
  described_only). `.pingis/drafts/cadwboost/`: 6/6 better than
  cadw.ml.gbdt; promote target v0.9.124+.
- ML_TASKS_TAXONOMY.md: v0.9.122 section + reference policy.
- DEC-048: taxonomy + CADWBoost + non-goals.

**v0.9.121 (since v0.9.120):**
- R10 (`tests/test_ml_tools_schema.py`): `test_R10_public_fit_covered`
  (every public `fit_*` in `cadw.ml.__all__` has a YAML card or is
  whitelisted with DEC ref) + `test_R10_GBDT_multiclass_function_matches`
  (name -> function for the one case where they diverge).
- GBDT_multiclass card fix: `cadw_function` `fit_gbdt_classifier` ->
  `fit_gbdt_multiclass` (card purpose = softmax, function was binary
  dispatcher; R5 + R9 both passed, invisible until audit).
- DEC-047 (R10 + GBDT bug + coverage gap plan for v0.9.122).

**Invariants:** R8 (input_api df|series|dual) + R9 (cadw_function importable).
CW-5 (test_guard_refuses_at_domain_bound) mandatory for bounded_by_domain fns.
Component-wise (DEC-033) + DEC-034 escalation (tolerance widen on RSS drift).

Next: v0.9.126 expansion_factor migration + YAML coverage sweep +
ML_COVERAGE_MATRIX (baseline/SOTA/ensemble audit).

## Progress (v0.9.105 - v0.9.110g)

- **v0.9.105** — 9 cards formula-calibrated (draft). EXPERIMENT_PROCEDURE.md (274 lines). DEC-018..025.
- **v0.9.106** — framework chunking + state-aware guard (DEC-022/023 closed).
- **v0.9.107** — 9 cards `verified` (tag). 11 verified total (2+9).
- **v0.9.108** — refactor `_stack_numeric` → `_common.py` (23 dedup sites).
- **v0.9.109a** — `fit_cusum` honesty fix (O(1) → O(n); YAML lie caught by probe; DEC-030 draft, SpillArray rewrite pending).
- **v0.9.109b** — `test_R5` aliases consistency (18 fns / 43 cards, 0 inconsistent).
- **v0.9.109c** — `fit_randomized_svd` verified (coef=3.0, OOC ratio 1.00); 12 verified total.
- **v0.9.109d** — snapshot bootstrap includes `EXPERIMENT_PROCEDURE.md` (T2 doc, position 4).
- **v0.9.110a/b** — iforest + linucb OOC improvements (batch_size kwarg, AP4 fix, state_est) → draft status due to measurement tool issue (DEC-031, DEC-032).
- **v0.9.110c** — component-wise verification (DEC-033) + `measure_peak_traced` (VmRSS sampler via /proc/self/statm, 2us). linucb verified.
- **v0.9.110d** — iforest verified × 2 cards (E_anomaly + D_time_series).
- **v0.9.110e** — methodology update: STABILITY §4.1.1 (component-wise), EXPERIMENT_PROCEDURE §3.7, ALGORITHM_TEMPLATE Step 5-alt, EXAMPLE_VERIFIED_CARD Example 2, AI_CONSTITUTION AP21/AP23.
- **v0.9.110f** -- snapshot tooling: TINY_FULL += test_*_verification.py, TINY_SKELETON += experiment scripts, EXAMPLES_INDEX rows.
- **v0.9.110g** -- `fit_gmm` verified (component-wise). batch_size kwarg + state_est = 5*max_samples*d*f64. 15 verified total.
- **v0.9.111** -- Phase 0 baseline (doc-sync regen, iforest flake median-of-3, DEC-034, arrow-gate Thread.join whitelist). Phase 1 fit_birch verified (16 total, DEC-035). Phase 2 R6/R7 schema + 19 no-fn cards patched (described_only / not_implemented_in_cadw).
- **v0.9.112-batch1** -- MiniRocket x2 (embedding+sota) + ALS x2 -> verified (component-wise, DEC-033). ALS `state_nbytes()` extended (was 30% under-report), `batch_size` kwarg + propagation. KShapeAD -> draft (DEC-013). 18 verified fns.
- **v0.9.112-batch2** -- 7 cards -> draft: LDA (AP5 unbounded vocab), CoxPH + T/S_learner (AP2 concatenate/vstack), EIF/KMeansAD/BOCPD (DEC-013 SpillArray). `bounded_by_domain` added to 5 tier1_core cards (R7).
- **v0.9.113** -- STREAMING §1.5 input convention (producer API = fit_*(df, col)). fit_zms ×3 + fit_cusum -> verified. batch_size kwarg for zms/cusum OOC. DEC-036 recorded. 20 verified fns.
- **v0.9.114** -- CoxPH-precedent path (bounded_by_domain n<=1e6) for O(n)-output: fit_coxph (predict_risk pre-alloc + batch_size), fit_kmeans_ad, fit_eif, fit_bocpd. 24 verified fns.


## Status baseline

| Status | Count | Note |
|---|---|---|
| verified | **32 fns / 48 cards** | v0.9.114 (24) + v0.9.115 (+6) + v0.9.116 (+1) + v0.9.117 (+2) + v0.9.118 (+2) + v0.9.119 (+1 ab_test x3 cards) |
| draft | **7 fns / 10 cards** | BVAR/VECM, binseg x2 (permanent refuse_at_guard), pelt x2, t2q, meta_learner (T/S_learner AP2) |
| pending (no status) | **24 fns / 30 cards** | A_classification x8, O_eda x16 |

## Known gaps (remaining)

- **kshape** (DEC-013 residue): `_map_scores_to_timestamps` expand n_w->n; can use CoxPH-precedent OR streaming rolling-max. v0.9.115.
- **K-spill** (AP5): LDA `_build_vocab`, tfidf — v0.9.115.
- **T/S_learner** AP2 `_collect`: intrinsic O(n·d) GBDT fit — draft (verified requires GBDT streaming refactor).
- **DEC-034/035 backlog**: subprocess-isolation for `measure_peak_traced`-based tests; BIRCH `_insert` batching — v0.9.115.
- **`fit_ab_test`** — tiny O(1) state; STABILITY §4 special case.
- **Remaining pending**: hampel, sub_pca, s2g, dwt_mlead, patch_memory_ad, autoencoder — probe fresh in v0.9.115.
## Lessons learned (v0.9.105-109)

- **YAML `ram_formula` / `state_class` often lie.** `fit_cusum` claimed O(1); actual code allocated 6 × 8n. Probe code first, YAML second.
- **Alias cards** — 18 fns / 43 cards. When flipping one, flip all aliases (test_R5 enforces).
- **`bounded_state_dims`** — real axes only (d, C, n_components, n_oversamples). Do NOT list auto-injected `batch`/`row_size` — R2 rejects them in dims; they are implicit names allowed in formula only.
- **`implementation_status` block may be entirely absent** in older cards (e.g. `fit_randomized_svd` had only `verified_in_cadw` originally). `test_R7` (backlog) catches this. For now: when patching YAML, check for block presence, create if missing.
- **YAML write scripts must use `encoding="utf-8"` explicitly** on `read_text` / `write_text`. Locale-agnostic, avoids mojibake on files with non-ASCII characters.
- **Never mix `git checkout <file>` and `python patch_<file>.py` in one block.** Order: checkout first (clean HEAD), then patch. `checkout` after patch silently reverts the patch.
- **RSS-based measurement (v0.9.110):** `measure_peak` (ru_maxrss delta) fails when baseline >> fit peak. 5 tool attempts confirmed fundamental. Solution: component-wise verification (DEC-033) -- state exact via `state_nbytes()`, coef isolated, OOC proven. NOT a relaxation; state is bit-exact.
- **Component-wise pitfalls:** (a) `d=10` noise 1MB > signal 0.3MB -> use d>=200. (b) `intercept` from slope-fit amplifies noise 8x -> only assert `coef` (slope). (c) `state(n1)==state(n2)` for n>=S fails (reservoir order) -> use bounded <=1.5x. See CW-AP1..4 in EXPERIMENT_PROCEDURE 3.7.
- **YAML card-scoped patch:** `str.replace` may hit 2+ cards (`bounded_state_dims` everywhere). Use regex anchors `- name: <Card>` + next card, patch only inside [start, end).

## Lessons learned (v0.9.111)

- **coef честность vs precedent.** BIRCH coef=1.16 (f64 zero-copy `_stack_numeric`); iforest 2.77 (reservoir + per-tree transients). Формула 3.0 — conservative per fit_linear precedent. DEC-035 фиксирует ~5 MB intercept (glibc arena fragmentation от per-row `_insert`) как known allocator-dependent limitation; formula оставляет 3.0 (safe upper), backlog v0.9.112 `_insert` batching.
- **`test_roadmap` regex: субапдейт — часть версии.** `v0.9.110g` не парсился `\d+\.\d+\.\d+$` ни в `## Current:`, ни в `_vkey`. Fixed: `vX.Y.Z<suffix>` + `_vkey` без `$`. Урок: тест, проверяющий версию, должен быть толерантен к проектному формату версий.
- **Doc-sync drift 12 versions.** `render_matrix.py` не запускали с v0.9.99 до v0.9.110g — 15 test_render_matrix fails накопились. Урок: re-run после каждого YAML change — pre-commit hook (`--check` flag) — backlog.

## Lessons learned (v0.9.112)

- **`n_scalability` -> verification test threshold.** `constant` → `ratio < 2.0`; `linear` → rename `test_peak_grows_linearly_with_n` + `ratio < 5.0`. Не расширять threshold ради зелёного — честный `linear` в YAML (ALS external_sort: peak O(K), K=n/run_buffer).
- **`state_nbytes()` extension → grep tests first.** `grep -rn 'state_nbytes' tests/test_<fn>*.py` — все stale assertions в тот же патч (ALS: 2 assertions, обновил только одну в fix1, вторую в fix3).
- **`batch_size` propagation → grep defs.** Module-level helpers — параметр, не closure. `_objective_streaming` в ALS получил NameError, потому что `batch_size` был только в `fit_als` scope.
- **`tier1_core + bounded_state:false` → `bounded_by_domain` сразу.** R7 schema проверяет при каждом YAML change. KShapeAD + 5 draft cards — 2 итерации на забывчивость.
- **Не бороться с `measure_peak`.** Baseline >> peak → component-wise немедленно (DEC-033). A12.
- **Card-scoped YAML patch.** `str.replace` с `bounded_state_dims` может попасть в чужую карточку. Anchor: `- name: <Card>` + next `- name:` (A16).

## Lessons learned (v0.9.113-114)

- **CoxPH-precedent**: `state_nbytes() + bounded_by_domain {n<=1e6}` unlocks verified for O(n)-output algorithms without SpillArray (coxph, kmeans_ad, eif, bocpd). Honest: `state_class: O(n)`, `n_scalability: linear`, `bounded_state: false`.
- **`contamination` default blocks zms-template** (DEC-036): zms/cusum have `None`; eif/km/bocpd had `= 0.1` -> percentile needs full array. CoxPH-precedent avoids the issue.
- **`state_ratio_linear` must use DELTA**: fixed term (e.g., bocpd posterior ~26KB) dominates at small n; ratio 5.29 not 10. Delta `state(n2)-state(n1) == (n2-n1)*per_row` cancels exactly.
- **`fit_peak_unmeasurable` when external_sort run_buffer dominates** (coxph): at mg=50MB, sort buffer ~25MB > baseline ~1MB. CW-1 (state exact) + CW-2 (state bounded) sufficient.
- **`_map_max` stride=1 is 0-copy**: `sliding_window_view + .max(axis=1)` = O(n) output, not O(n*w).
- **Duck-typed vs real MemoryGuard**: real -> chunked OOC (no raise); duck-typed (max_memory_mb only) -> est_bytes check (raise). Tight-guard tests use `_FakeMB`.

## Next 3 sessions

### v0.9.126 — expansion_factor migration + cross_ref schema + promotion

- **Migration of 18 `expansion_factor` call sites** (6 via
  `derive_batch_size` + 12 direct `get_batch_rows`), 14 models (linucb,
  coxph, iforest, zms, randsvd, birch, logistic, als, cusum, drift, gmm,
  naive_bayes) to `solve_batch`. Requires: (a) mapping `ReduceOp`-class ->
  `cadw_function` (derive_batch_size takes op object, not name); (b)
  re-measurement (batch increases -> tighter peak -> risk OOM). Separate
  commit per model group.
- **YAML coverage sweep** (DEC-045): add cards for uncarded base algorithms
  (`fit_kmeans`, `fit_pca`, verify `fit_isolation_forest`/`fit_iforest`).
  `status: draft` + formula; verification separate session.
- **`fit_ab_test` YAML**: state-only formula -> add `+ N*batch*row_size`
  (t=4, cuped=5, mw via SortMemoryModel). Now `solve_batch` returns None.
- **Per-model fit-time calibration** (DEC-046): `estimate_time` for each
  verified model (epochs x n_rows). ~0.5 session.
- **ML_COVERAGE_MATRIX.md**: section x subtype x {baseline, SOTA, ensemble}.

### v0.9.127 — pending fns (A_classification + O_eda)

- A_classification x8 pending: component-wise verify (some have `status: None`
  legacy cards — normalize to `draft`/`verified`).
- O_eda x16 pending: same.
- AP2 sweep (grep `column_stack|vstack|concatenate` in cadw/ml/).

### v0.9.127+ — 30+ verified / 35+ verified tags

Re-check do-not-re-solve per DEC-033/034/035. Tag v0.9.122/123 when
appropriate.

## ## Backlog (not scheduled)

- 55 pending fns: batches ~4/session v0.9.110-v0.9.120
- `solve_batch(op, budget, **params)` inverse solver
- `fit_ab_test` §4 special case (tiny O(1) state)
- AP2 sweep: grep `column_stack|vstack|concatenate` in cadw/ml/
- Refactor `_probe_memory_limit_mb` (15 copies)

## Do not re-solve

Per STREAMING §6.1 — won't-fix with rationale:

- **eif bit-identical chunked**: 40 GB I/O at n_est=100. Zero functional gain.
- **kmeans_ad bit-identical**: 3.2 GB I/O at k=8. Reservoir deterministic.
- **autoencoder beyond n_w = state_limit/16**: 2-3 sessions, no use case.
- **sparse tfidf.transform**: needs CSR SpillBuffer design.
- **binseg / stray**: intrinsic O(n) / O(n²). `refuse_at_guard` is correct.
- **TS single-series fits** (STL, VAR, VECM, ETS, Theta): `tier1_core_exception`, n ≤ 10^6 typical.

## Session discipline

- Before commit: `pytest -q <affected files>` (focused, ~20s).
- Before tag: `pytest -q -m "not slow"` (full fast, ~10min).
- Before release: `pytest -q` (all, includes slow).
- See `AI_CONSTITUTION.md` §4 for any new `fit_*` / `predict_*`.
- See `EXPERIMENT_PROCEDURE.md` §3 for card verification (6 steps).
