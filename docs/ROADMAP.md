# CADW Roadmap

Last updated: 2026-11-12 (after v0.9.223).

## Current: v0.9.223 — CadwBoost promoted + achieved

**v0.9.223 outcome:**
- CadwBoost promoted from .pingis/drafts/ to cadw/ml/cadwboost/
  (18 modules, 2858 LOC). Card flipped to verified + achieved
  (first new achieved since v0.9.219 BOCPD).
- Docstrings cleanup: 37 .pingis refs removed from cadw/ (35 files).

**Counts after v0.9.223:**
- achieved: 80 (+CADWBoost, was (none))
- achievable: 43 (unchanged)
- debt: 29 (unchanged)
- none: 76 (-1, CADWBoost left none)

**.pingis status:**
- `.pingis/benches/` (69 files) is TRACKED in git — valid for both
  relative path and GitHub URL; card `benchmark_url` values remain
  correct. NO mass YAML cleanup needed.
- `.pingis/impl/` (46 files, tracked) — 9/9 models already promoted to
  cadw/ml/ v0.9.143 (DEC-058). Stale; cleanup pending (low priority).
- `.pingis/audits/`, `drafts/`, `tasks/` — gitignored (historical).
- `.pingis/plan_v0.9.87_refactor.md` — `_gate_whitelist.load_whitelist`
  already returns `_EMPTY` if absent (safe if .pingis removed).

**Next (v0.9.224+):**
- v0.9.224: CadwBoost tests batch 2 (~21: EFB, sampling, missing,
  serialize, shap, df, ooc extended).
- v0.9.225: TS streaming (P3/P4 primitives); anomaly SpillArray batch.

## Next 3 sessions (v0.9.219-221)

### v0.9.219 — Sub-PCA + BOCPD (API change for OOC)
Sub-PCA: `contamination: float|None` + `output='auto'|'disk'` +
SpillArray (decision_scores/labels/window_scores) + 2-pass streaming.
BOCPD: already OOC-by-state (posterior_keep·max_run bounded); add
`output='auto'` param + SpillArray for `anomaly_score`/`argmax_run`/
`changepoints`. Effort 1-1.5 sess. Target: +2 achieved.

### v0.9.220 — KShapeAD + S2G (API change, precedent Sub-PCA)
Same pattern: output='auto' default + SpillArray + 2-pass streaming.
Training fit transient: verify chunked (P35 showed spill=True already).
Effort 1 sess. Target: +2 achieved.

### v0.9.221 — GBDT round-by-round SpillArray (5 cards)
GBDT, GBDT_multiclass, GBDT_L2, GBDT_scale_pos_weight x2. Precedent
LambdaMART v0.9.87-P2b-full-2. Effort 2-3 sess. Target: +5 achieved.

**v0.9.222+ (deferred):** TS streaming (P3/P4 primitives per ROADMAP §B);
AutoTheta, Ensemble, VAR.

## OOC by primitive — decision matrix (MANDATORY, v0.9.204+)

**Rule:** before writing any new function / primitive / model, state the
OOC axis and the pattern explicitly. Three valid OOC patterns; anything
else must have honest bound. **Update `Last updated:` above when this
matrix changes.**

### Three OOC patterns

| # | Pattern | How OOC | Primitive examples | Model examples |
|---|---|---|---|---|
| **1** | **Batch-layer** | Input = Iterator / memmap / SpillArray; state O(batch) or O(model) | `reduce`, `knn_query_stream`, `_collect_last_w`, `_iter_target_batches`, `_iter_filtered_batches`, `SpillArray`, `SpillBuffer` | SARIMA(manual), LOF, KNNImputer, KernelPCA(transform), SpectralClustering(fit kNN affinity) |
| **2** | **Window-truncation** | Fixed W (min(user, guard, n)); state O(W); W independent of n | `_collect_last_w` (building block); `_ses_fit_simple` (via streaming); `_ses_fit_hw` / `stl_decompose` / `_ses` (via caller truncation; cards have `intrinsic_bound` with reason) | Theta(s>1), OTM(s>1), ETS(manual), AutoETS(manual), pelt(max_window) |
| **3** | **Intrinsic bound** | Physically O(n²) / O(n³); guard refuses above bound | `chol_solve`, `mst`, `eclat`, `eigh_topk`(dense), `nan_sq_dists` (dense broadcast), `pairwise_kernel` (dense output) | KernelPCA(fit, n≤1e4), SpectralClustering(fit, n≤1e6) |

**Anti-patterns (each has a known failure):**
- Silent O(n) allocation without bound → OOM, not clean refuse.
- Leaf primitive called with full-n ndarray inside `bounded_state: true`
  model, when axis not in bound → fake OOC.
- "Window" without ApproximationInfo on truncation → AP24.
- `streaming_variant: null` AND `intrinsic_bound: null` AND model
  claims `bounded_state: true` for that axis → **AP33 case A** (bug).

### Pre-implementation checklist (MANDATORY, write in session brief)

For each new **fit_* / predict_* / transform_* / primitive**:

1. **Classify:** batch-layer / window / intrinsic — mark **which axis**
   (n / d / K / n_groups).
2. **Pattern:** if batch-layer → existing primitive from matrix? If
   window → caller uses `_collect_last_w`? If intrinsic → declare
   `bounded_by_domain {axis, bound}` in card.
3. **Callsite audit:** for each leaf primitive used, is caller batching
   (batch-layer) OR model-level windowing (window) OR bound (intrinsic)?
4. **Card:** `layer`, `bounded_state`, `streaming_variant`,
   `intrinsic_bound` all **consistent** with 1-3.
5. **Test:** bit-identical non-OOC vs OOC branch (single-batch or
   n≤W); OOC ratio < 2.0 across 10× axis.

### Consuming patterns (before writing model code)

| Consumer needs OOC in | Use primitive | Do NOT use |
|---|---|---|
| n (rows, aggregation) | Iterator-based (`_iter_*_batches`) + state O(model) | full-n np.ndarray |
| n (self-query, kNN on same X) | intrinsic bound n≤1e4/1e5 honest (P6 example: nan_sq_dists called per-chunk by caller) | leaf called with full X unguarded |
| n_y (pairwise, X small/fixed) | `knn_query_stream` (X Iterator) | `knn_query(X_full, Q)` |
| n_train (transform, Q small) | X_fit in RAM = O(model·d) if bounded; Nystroem alternative above bound | `pairwise_kernel(Q, X_train_full)` без bound |
| K (cardinality) | HashShuffle + K_spill (TFIDF/LDA/OHE with `spill_dir=`) | `dict[word] += 1` unguarded |
| d (features) | SGD (state O(d·K)); closed-form only if `d <= bound` | closed-form XTX при d>1e4 |
| n_groups (TS per-entity) | HashShuffle per group + per-group window | full extract across groups |

### Session discipline (added v0.9.204)

- **Before any new primitive/model in a session brief:** cite this matrix
  pattern # + axis. Session that adds `fit_X` without this line = refuse
  to proceed, ask.
- **After any card flip:** run G22
  (`test_G22_leaf_deps_consistent`) + `test_primitives_audit` +
  `test_constitution_gates`.
- **Post-session:** if new primitive added, update this matrix "Primitive
  examples" column + PRIMITIVES_BACKLOG Appendix B.

## Historical: v0.9.175 — OOC batched work front (DEC-067)

**Big picture:** ~40 cards are not OOC batched (`bounded_by_domain` or `draft`).
User requirement (2026-11-10): OOC for **all** models/primitives/functions with
current budget via auto-switch; refuse only if no variant fits (physical
impossibility or absurd params).

### v0.9.175 (docs + YAML, no code refactor)

- DEC-067: 2-tier insufficient + 6-level auto-switch registry + 8-group inventory.
- DEC-068: unified params (window / solver / output / params / strict).
- DEC-069: df.apply 3-variant API.
- AI_CONSTITUTION: §4.0 R1-R7 (hard requirements), AP24-AP28 (new violations).
- STABILITY_PROTOCOL: §2.5 (bounded_by_window), §4.4 (min_ram_formula),
  §6.1 (insufficient tiers), §6.2 (approximation verification).
- PARAMS_DESIGN.md, PRIMITIVE_CARD_SPEC.md, DF_OPERATIONS_SPEC.md (new specs).
- DF_operations.yaml: 20 df method cards (cumulative/rolling/shift already OOC;
  fill/pivot/sample pending probes; apply 3-variant).
- GLOSSARY: 12 new terms (budget dynamic, min_ram, BudgetTier, auto-switch,
  ApproximationInfo, window, get_window_size, bounded_by_window, strict,
  PrimitiveCard, DataFrameOpCard, insufficient_no_alt, escape_hatch).
- YAML fixes: Hampel `bs=False bd=y`, STRAY `bd=y {axis: n, bound: 1e5}`.
- Probe P1-P24 documented (`docs/audits/v0.9.175_probe.md`).
- 6 apply experiments designed (`docs/audits/v0.9.175_apply_experiments.md`).

### Waves v0.9.176-220 (roadmap summary)

| Wave | Versions | Scope | Sessions |
|---|---|---|---|
| 0 (docs + infra) | v0.9.175-178 | docs, `get_window_size`, `ApproximationInfo`, `resolve_params`, `min_ram_formula` migration (90+ cards) | 3-4 |
| 1 (quick wins) | v0.9.179-181 | pelt `max_window="auto"`, coxph concordance, binseg external cumsum | 3-4 |
| 2 (TS refactor) | v0.9.182-186 | drop y_/fitted_, `_extract_series_tail`, `predict_in_sample(df)` | 5-6 |
| 3 (GBDT bundle) | v0.9.187-190 | GBDT+lambda round-by-round SpillArray, GBDTForecaster lag SpillDir | 3-4 |
| 4 (dataframe) | v0.9.191 | df YAML + apply 3-variant impl + probe P25-P27 | 0.5-1 |
| 5 (anomaly) | v0.9.192-208 | DEC-036 reservoir + SpillArray ×10 | 14-17 |
| 6 (primitives) | v0.9.209-218 | streaming A + sparse B + intrinsic bounds C | 8-10 |
| 7 (L12) | v0.9.219 | consensus_matrix | 1 |

**Total ~38-45 sessions** to full honest OOC.

---

## Historical: Next 3 sessions (v0.9.176-178 — infrastructure wave)

### v0.9.176 — Infrastructure (1.5 sessions)
MemoryGuard.get_window_size(), ApproximationInfo, ApproximationWarning, BudgetTier enum, resolve_params() 6-level auto-switch, ModelCard.bounded_by_window field, PrimitiveCard + DataFrameOpCard dataclasses, result.approximation field in TS bases. G12-G17 gate skeletons.

### v0.9.177 — min_ram_formula migration part 1 (1 session)
AST script + 45+ verified cards: analytical min_ram_formula + min_ram_evidence. G14 test (verified => min_ram_formula present).

### v0.9.178 — min_ram_formula migration part 2 + wave 1 start (1 session)
Remaining 45+ cards + pelt max_window='auto' + MemoryGuard.get_window_size wiring. Wave 1 (quick wins) starts.

---

## Historical: v0.9.142 — depends_on 100% coverage + .pingis cleanup + §1 done

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

## Historical: Next 3 sessions

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

**v0.9.174 (S2b-2b: BorutaSelect subsample cap):**
- `fit_feature_select` boruta path: add `max_samples=20000` default
  + subsample warning (precedent S2b-2a chains/meta/uplift).
- `_collect` in feature_select: add max_rows cap (None-safe).
- YAML: BorutaSelect bounded_state false -> true, n_scalability
  linear -> constant, bounded_by_domain/tier1_core_exception
  removed, ram_formula simplified to `2*d*f64 + 512 +
  2.0*batch*row_size`, subsample_note added.
- Tests: 10 existing + 12 verification + 112 schema, all green.
- **AP2_COLLECT 10/10 complete** (v0.9.170-174): feature_select
  (variance, mutual_info), chains, ordinal, meta_learners, uplift ×2,
  boruta. All original AP2 violations resolved.
- Remaining DEC-064 classes:
  - GBDT_FIT_TRANSIENT (5 cards): partial-OOC via memmap (DEC-065),
    true OOC refactor v0.9.180+ (posix_fadvise / direct file I/O).
  - OUTPUT_N (10 anomaly cards): bounded_by_domain, refactor v0.9.180+
    (DEC-066, needs DEC-036 reservoir sampling first).

**v0.9.173 (S2f-1: CUSUM flip + S2f scope-honesty, DEC-066):**\n- CUSUM YAML: bounded_by_domain removed, bounded_state=true\n  (already `output='auto'` default, SpillArray implemented).\n- 10 anomaly models (kshape, sub_pca, autoencoder,\n  patch_memory_ad, tspq, s2g, dwt_mlead, eif, kmeans_ad,\n  bocpd): honest bounded_by_domain KEPT. Root cause: SpillArray\n  not implemented (NotImplementedError stubs v0.9.116),\n  percentile over n requires reservoir (DEC-036), window-to-\n  timestamp map needs 2nd pass.\n- Refactor deferred v0.9.180+ (DEC-066): DEC-036 reservoir\n  (~2-3 sess) + per-model SpillArray (~1-2 sess × 10).\n- Estimated S2f sweep: 10-15 sessions, not 2-3.\n- Milestone: CUSUM is the only truly OOC anomaly model today.\n\n**v0.9.172 (S2c-1: GBDT fit memmap refactor — partial, DEC-065):**\n- `_fit_boosting` (binary + regression): pred/cur_leaf/nan_mask\n  on SpillDir memmap. `_binning_pass`: nan_mask -> bool memmap.\n- Extract `_assign_leaf_batch` helper (bit-identical to\n  _AssignLeavesOpCodes.update). Shared between ReduceOp and direct\n  memmap-write loop.\n- `batch_size` param on public fit_gbdt_classifier/regressor\n  (DEC-023 class gap; auto-derived from guard if None). Propagated\n  through _binning_pass + _fit_boosting + _iter_x_with_codes.\n- `_madvise_dontneed` helper (Linux MADV_DONTNEED; insufficient at\n  scale — memmap dirty pages not evicted).\n- **Honest outcome (DEC-065)**: ~1.6x RSS improvement (32 B/row ->\n  ~20 B/row) but NOT true OOC. CW-5 40x n ratio = 24-36 (linear).\n- Tests: CW-2 state_nbytes_independent_of_n + batch_size param\n  accepted + memmap_writes_do_not_leak_into_state. CW-5 skipped.\n- YAML: 5 GBDT cards keep bounded_by_domain (refined reason +\n  fit_transient_note; v0.9.172 improvement documented).\n- Refactor to true OOC deferred v0.9.173+ (posix_fadvise or direct\n  file I/O).\n\n**v0.9.171 (S2b-2a: chains/meta/uplift/ordinal subsample cap):**
- Code: `max_samples=20000` param + warning on subsample for:
  fit_chains, fit_meta_learner, _fit_uplift_two_model,
  fit_ordinal_logit. fit_uplift: warning added (had max_samples).
- _collect in chains/meta_learners/ordinal/uplift: max_rows cap
  (None-safe; v1 had float('inf') -> int() OverflowError bug).
- Tests: 71 passed + test_chains_verification subsample_at_domain_bound
  (replaces guard_refuses_at_domain_bound — behavior change honest:
  subsample+warn instead of MemoryLimitExceeded).
- YAML flip 7 cards: bounded_state=false -> true, n_scalability
  linear -> constant, ram_formula '2*d*f64 + 512 + 2.0*batch*row_size',
  bounded_by_domain / tier1_core_exception removed, subsample_note
  added: ClassifierChains, OneVsRest_Logistic, OrdinalLogit,
  T_learner, S_learner, T_learner_uplift, CausalTree.
- R13 fix: T_learner + S_learner fit_batched false -> true
  (subsample via _iter_filtered_batches is truly batched).
- BorutaSelect remains bounded_by_domain (needs S2c GBDT SpillDir).
- AP2_COLLECT: 9 of 10 done (only Boruta deferred to S2c dependency).

**v0.9.170 (S2b-1: feature_select OOC refactor):**
- `cadw/ml/feature_select.py` refactor (variance + mutual_info):
  - _collect AP2 vstack -> SpillDir memmap (X and y on disk)
  - chunked 2-pass variance (O(chunk*d) peak)
  - chunked 2-pass histogram MI per column (x_edges from sample,
    joint per chunk; peak O(chunk*16 + n_bins^2))
  - Boruta stays AP2 (needs hstack(X|shadows); deferred v0.9.172+)
- Tests: 10 existing + 12 verification all pass. Smoke: variance /
  mutual_info produce correct scores, state=544.
- YAML flip: MutualInfoSelect + VarianceThreshold
  bounded_state=false -> true, n_scalability linear->constant,
  bounded_by_domain/tier1_core_exception removed,
  ram_formula d*f64 + 512 + 2.0*batch*row_size.
- BorutaSelect remains bounded_by_domain n<=1e5 (S2b-2 deferred).
- DEC-064 Class 1 AP2_COLLECT: 2 of 10 flipped; 8 remaining in S2b-2.

**v0.9.169 (S2a audit: 40 bounded_by_domain classified, DEC-064):**
- S2a completed: 43 cards classified into 8 classes.
- 4 classes = technical debt (S2b/S2c/S2f refactor):
  AP2_COLLECT=10 (fit vstack); GBDT_FIT_TRANSIENT=5 (grad/hess);
  OUTPUT_N=9 (n-sized persistent output); PARAM_FLIP=1 (DWT).
- 4 classes = permanent domain bound (DEC-011): SINGLE_SERIES=12,
  N_EVENTS=1 (CoxPH), INTRINSIC_DP=1 (PELT).
- CUSUM: already OOC (output='auto' default); YAML flip only.
- Refactor roadmap ~5.5 sessions (S2b 2 + S2c 2 + S2f 1.5).
- Audit methodology formalized: AST probe + runtime probe
  (walked vs state_nbytes + fit_peak vs state).

**v0.9.168 (OOC audit: 3 bugs + leaf-layer docs, DEC-063):**
- Bug #1 fix: GBDTForecaster.state_nbytes() — includes _y/_X/_idx/
  _tail_features (was ~3x under-report at n=1k, ~2500x at n=1e6).
  YAML: bounded_state true -> false, bounded_by_domain n<=1e5.
- Bug #2 (deferred refactor): GBDT family fit transient O(n·90B)
  (pred/grad/hess), not O(batch). YAML honest fix: bounded_by_domain
  n<=1e5 + measurement.fit_transient_note. SpillDir refactor
  planned v0.9.169+ (precedent LambdaMART v0.9.87-P2b-full-2).
- Bug #3: AST probe false positives confirmed via runtime probe;
  session discipline: runtime `walked vs state_nbytes` before verified.
- PRIMITIVES_BACKLOG Appendix A: batch-layer (OOC) vs leaf-layer
  (pure-array, caller batches) explicit taxonomy of ~20 primitives.
- 10 cards updated (GBDT ×5 + T_learner_uplift + CausalTree +
  FeatureSelect ×3): all now carry bounded_by_domain + fit_transient_note.
- Refactor roadmap v0.9.169+: GBDT SpillDir, FeatureSelect/T_learner AP2,
  Ensemble/GBDTForecaster deferred horizon SpillDir. Single-series TS
  remains bounded_by_domain permanently (DEC-011 domain bound).

**v0.9.167 (cross_ref schema field):**
- New `ModelCard.cross_ref: str | None = None` (DEC-062,
  formalizes DEC-049 policy). Format `<section>.<card>`.
- R15 test: format, resolution (section + card exist), no self-ref.
- Migrated 5 dual-classified cards: HDBSCAN, BIRCH -> N_density;
  NMF, randomized_svd -> J_nlp; ALS -> F_recsys.
- File-level markdown comments removed from B_clustering.yaml +
  C_dim_reduction.yaml (replaced by per-card field).
- Tests: 112 passed (added R15).
- Deferred v0.9.168+: R16 bidirectional (`canonical_for`).

**v0.9.166 (fit_stacking tabular infrastructure):**
- New `cadw/ml/stacking.py::fit_stacking` (DEC-061): K-fold OOC
  tabular stacker (not a card; like fit_ab_test).
- Reuses `cadw.cv.KFold(stratify_by=y)` for OOC folds; pass 1 = OOF
  per base model, pass 2 = meta on OOF, pass 3 = refit bases on
  full df.
- state = sum(base.state_nbytes) + meta.state_nbytes + 256.
- Tests: 5 passed (runs+predicts; state>=header; guard-refuse at
  0.001 MB; invalid args; no-regression vs best base on A1).
- ROADMAP §A.2 false coupling (EnsembleForecaster <-> fit_stacking)
  corrected; ROADMAP §2 priority #1 (P1 group A) and #2 (impl/
  promotion) already done in v0.9.144-164; §A.2 closed.

**v0.9.166 (fit_stacking tabular infrastructure):**
- New `cadw/ml/stacking.py::fit_stacking` (DEC-061): K-fold OOC
  tabular stacker (not a card; like fit_ab_test).
- Reuses `cadw.cv.KFold(stratify_by=y)` for OOC folds; pass 1 = OOF
  per base model, pass 2 = meta on OOF, pass 3 = refit bases on
  full df.
- state = sum(base.state_nbytes) + meta.state_nbytes + 256.
- Tests: 5 passed (runs+predicts; state>=header; guard-refuse at
  0.001 MB; invalid args; no-regression vs best base on A1).
- ROADMAP §A.2 false coupling (EnsembleForecaster <-> fit_stacking)
  corrected; ROADMAP §2 priority #1 (P1 group A) and #2 (impl/
  promotion) already done in v0.9.144-164; §A.2 closed.

**v0.9.165 (EnsembleForecaster verified):**
- Flipped EnsembleForecaster (D_time_series) draft -> verified
  via bounded_by_domain (DEC-011): state O(n) per member pipeline
  (base forecaster + AutoStationarizer + fitted).
- ram_formula (upper bound): 5 * n_members * n * f64 + 256.
- bounded_by_domain {axis: n, bound: 1e6}; fit_batched: true,
  predict_batched: false (per-member pipeline predict).
- **False coupling in ROADMAP §A.2 corrected**: EnsembleForecaster
  is a weighted-average (inverse_mase / median / custom), not a
  stacker; independent from `fit_stacking` (tabular). Verified
  independently.
- Tests: 5 passed (state = sum over pipelines; grows linearly
  n=500 -> 2000; guard refuses at n=500k + 2MB; per-member bound
  ≤4n+8192; 2->3 members ratio ~1.5x).
- **All original drafts closed**: D_time_series 9/9 verified;
  total draft fns 3 -> 2 (both TS edge cases deferred).

**v0.9.164 (final legacy batch: 3 complex ML):**
- Flipped Hybrid_MF_GBDT (F_recsys), T_learner_uplift + CausalTree
  (H_causal) from status:None to verified.
- ram_formulas (from state_nbytes bodies):
  - Hybrid_MF_GBDT: ALS base (U,I,K) + ridge B (F,K) + C (I,F).
  - T_learner_uplift: 2× GBDT (treatment/control arms).
  - CausalTree: n_estimators × 2^(max_depth+1)-1 nodes + d + 1024.
- Tests: 9 passed (state bounded >0; state independent of n;
  guard-refuse at 0.001 MB per card).
- **31 legacy status:None -> 0 truly-pending**: 21 verified +
  9 not-a-model + 0 pending. All original legacy cards resolved.

**v0.9.163 (LambdaMART verify):**
- Flipped LambdaMART (G_learning_to_rank): status:None -> verified.
- Uses existing state_nbytes (n_nodes * 96 + n_features * 4096);
  fit OOC via SpillDir memmap (v0.9.87-P2b-full-2).
- ram_formula (upper bound): n_estimators × (2**(max_depth+1)-1) × 96
  + d*f64*512 + 256.
- Tests: 4 passed (CW-1 [0.20, 1.0]; CW-2 ratio ≤3.0;
  CW-6 guard refuses at 0.001 MB; predict smoke).
- Remaining complex ML: Hybrid_MF_GBDT, T_learner_uplift, CausalTree
  (v0.9.164+).

**v0.9.162 (not_a_model reframe: 9 cards):**
- New `ModelCard.not_a_model: bool = False` (DEC-060).
- R14 test: not_a_model=true => status=None (consistency).
- Reframed 9 cards with `not_a_model: true` + `# NOT-A-MODEL:`
  YAML comment:
  - O_eda: Describe, Hist, ValueCounts, SimpleImputer,
    EncodeCategorical, PermutationImportance, FairnessMetrics
    (dataframe methods / analysis functions).
  - D_time_series: backtest_forecaster (utility),
    IntervalFeatures_GBDT (rolling_stats method).
- **31 legacy status:None -> 0 truly-pending**: 18 verified +
  9 not-a-model + 4 complex ML (LambdaMART, Hybrid_MF,
  T_learner_uplift, CausalTree — v0.9.163+).

**v0.9.161 (legacy O_eda batch 1: 4 fit_* cards):**
- Flipped 4 cards from status:None to verified:
  - DriftStats (fit_drift): O(1) lazy container (256 B),
    bounded_state=true, state_class=O(1).
  - MutualInfoSelect, VarianceThreshold, BorutaSelect
    (all fit_feature_select method=): model O(d), fit transient
    O(n*d) via _collect (AP2 with guard). bounded_by_domain n<=1e6.
- Tests: 12 passed (drift container; feature_select state formula
  exact d*8+512 per method; n-independent; guard-refuse).
- Remaining O_eda legacy (7): Describe, Hist, ValueCounts,
  SimpleImputer, EncodeCategorical, PermutationImportance,
  FairnessMetrics — these are dataframe METHODS/analysis
  functions, not fit_* models (reframe in v0.9.162).

**v0.9.160 (GBDT family batch 2: 2 TS cards):**
- Fixed `_NumpyReg.state_nbytes()` (was missing -> inherited 0
  -> GBDTForecaster total was just 256 header). Delegates to
  `.model.state_nbytes()` (GBDTRegressor).
- Flipped 2 cards: GBDTForecaster, GBDT_multiseries
  (D_time_series; both same class, multiseries adds group_col).
- ram_formula (upper bound, iforest precedent):
  n_horizons * n_estimators * (2**(max_depth+1)-1) * 500
  + 2*d*f64 + 256
- Tests: 6 passed (CW-1 [0.20, 1.0]; CW-2 ratio <=3.0;
  lazy fit(horizons=None)->state=0; eager fit(horizons=N)->state>0;
  guard-refuse; recursive eager).
- **D_time_series section complete**: 9/9 TS cards verified.

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

| Metric | Count | Notes |
|---|---|---|
| Total cards | 228 | across [A-Z]_*.yaml |
| Verified (status) | 116 | includes some achievable/debt |
| status=None | 112 | described_only / draft / planned |
| ooc_status=achieved | 77 | OOC confirmed |
| ooc_status=achievable | 64 | honest debt with plan (v0.9.209) |
| ooc_status=debt | 10 | honest debt, no plan (Spectral x3, DBSCAN, Agglom, PlaidModel, Bimax, STRAY, MH, NUTS) |
| ooc_status=None | 77 | pre-v0.9.208 cards, no classification |

**Consistency note:** `verified` (116) and `ooc_status` (74) are not
exclusive. Many verified cards are pre-v0.9.208 (no ooc_status);
some achievable/debt are verified-by-source but not OOC-by-practice.

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

## Permanent discipline (apply EVERY session, v0.9.189+)

### §A. Order: primitives → wire → honest YAML flip

For any model refactor (see AI_CONSTITUTION §4.0 R0):

1. **PrimitiveCard + streaming_variant** (or window-truncation) for
   EACH transitive leaf primitive. Register BEFORE touching model code.
2. **Wire streaming/window in fit path.**
3. **Bit-identical test** (n≤W_eff, 3 strategies × 3 n) + **OOC peak test**
   (n=10k→100k, ratio<4).
4. **Only then** flip model YAML: `bounded_state: true` +
   `n_scalability: bounded_by_window|constant`. `bounded_state_dims` =
   state axes only.

Violations caught by **G18** (`tests/test_constitution_gates.py`).

### §B. Pending streaming primitives (DEC-070 backlog)

| Prim | Status | Target |
|---|---|---|
| `_ses_fit_hw` (HW, s>1) | P3: window-truncation with `initial_periods=K` (default K=10); bit-identical n≤K*s, disclosed truncation otherwise | v0.9.189 |
| `stl_decompose` | P4: window-truncation `max_window="auto"` | v0.9.190 |
| GBDTForecaster `_build_lag_matrix` | SpillDir memmap | v0.9.191 |
| SARIMA `_ar_fit`, `_arma_forecast`, etc. | refactor (honest bounds until) | v0.9.192+ |

**P3/P4 are window-truncatable, NOT intrinsic** (DEC-072): seasonal/STL
pattern from last W_eff is a valid semantic variant (pelt precedent,
DEC-067 §3/§6). Do NOT declare intrinsic without checking window first.

### §D. Testing discipline — targeted only

Per-session rule (v0.9.217 update, user feedback — **hard rule**):

| Change | Run |
|---|---|
| Model-specific code | that model's tests + `test_ml_tools_schema.py` |
| Shared primitive | its tests + 1-2 representative consumers |
| YAML flip | `test_ml_tools_schema.py` + `test_constitution_gates.py` + affected model tests |
| Version bump / release tag | **same targeted tests** — no full suite |

**Full suite is EXPLICITLY FORBIDDEN unless user asks.** Reasons:
- Tagging cadence: ~30 min per release in active session. Full suite
  (30+ min, slow E2E dominates) cannot fit every tag.
- Models are isolated (distinct code paths, distinct tests). Targeted
  tests catch regressions in changed code. Untouched code has its own
  baseline (already tested in its release).
- Pre-existing failures (spectral_biclustering, spectral_clustering
  bench deps, test_als_verification flake) make full suite noisy and
  fail-tag-correlation meaningless.

**When full suite IS appropriate:**
- User explicitly says "run full".
- Post-campaign checkpoint (e.g. after 10+ releases, end of day).
- Before major version (v1.0).

**Anti-pattern (confirmed twice, v0.9.188-191 + v0.9.217):** running
full fast "before tag, just in case". Wastes 30 min per release with
zero actionable signal.

### §E. G18 is formal, not semantic

G18 checks:
- streaming_variant != null
- OR window_truncatable != null
- OR intrinsic_bound != null
- AND reason >= 20 chars

It does NOT check **whether intrinsic_bound is semantically justified**
(e.g. `_ses` (otm) had intrinsic_bound in v0.9.189-c1 — formally valid
for G18, semantically wrong per DEC-072 §3; caught only by manual review
in v0.9.191).

**Rule:** before using `intrinsic_bound`, apply self-check from §C
(window? disk 2-pass? then intrinsic). Manual review required for any
new intrinsic_bound. G18 is a sanity net, not a semantic gate.

---

### §C. Intrinsic primitives (closed, no streaming possible)

Only those where full-n is mathematically required with no valid
window-truncation:
- `chol_solve` — n² factor
- `eclat` — vertical tid sets (uses sum|tx|)
- `ranknet_grad` / `listnet_grad` / `lambdarank_grad` — pairwise per
  query (O(n_q²))
- `mst` — full edge list required

**Self-check before declaring intrinsic:**
1. Can it use `max_window` (pelt-style truncation to last W)? → use
   `bounded_by_window` on model.
2. Can it be 2-pass with SpillDir (disk, bit-identical)? → use
   `bounded_by_disk` (new category if needed).
3. If neither → `intrinsic_bound` legit.

`intrinsic_bound` MUST have non-empty `reason` (G18 checks).

---

## Session discipline

- Before commit: `pytest -q <affected files>` (focused, ~20s).
- Before tag: `pytest -q -m "not slow"` (full fast, ~10min).
- Before release: `pytest -q` (all, includes slow).
- See `AI_CONSTITUTION.md` §4 for any new `fit_*` / `predict_*`.
- See `EXPERIMENT_PROCEDURE.md` §3 for card verification (6 steps).
- See `SESSION_BRIEF_v0.9.129.md` for next-session bootstrap.
