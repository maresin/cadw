# CADW Roadmap

Last updated: 2026-11-06 (after v0.9.128).

## Current: v0.9.128 — TAXONOMY FROZEN

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

### v0.9.129 — verify 4 ts drafts + fit_stacking pilot

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

**v0.9.119-128 (this session, 10 releases):**
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
- **`_probe_memory_limit_mb` refactor**: 15 copies → `MemoryGuard` method.
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
