# CADW Roadmap

Last updated: 2026-11-03 (after v0.9.111).

## Current: v0.9.111

Baseline stabilized (Phase 0). fit_birch verified (Phase 1, 16 total).
Schema R6/R7 + 19 no-fn cards patched (Phase 2). Next: Group C
(windowed anomaly + P/k/w knobs) via EXPERIMENT_PROCEDURE.

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


## Status baseline

| Status | Count | Note |
|---|---|---|
| verified | **16** | 2 MatrixProfile + 9 v0.9.107 + 1 randsvd + linucb + iforest x2 + gmm + **BIRCH (v0.9.111)** |
| draft | 14 | 21 legacy - 9 flipped - linucb/iforest/gmm/birch promoted to verified |
| pending (no status) | 56 fns (125 entries) | 88 with fn + 19 no-fn described_only/niic (v0.9.111) |

## Known gaps (remaining)

- **DEC-013**: `fit_zms` x3 — O(n) output needs `SpillArray`. Deferred to v0.9.111.
- **DEC-014**: `fit_pelt` x2 — `_collect_series` container. Deferred.
- **DEC-030** (v0.9.109a): `fit_cusum` — O(n) state + O(n) output, SpillArray rewrite pending. Draft.
- **DEC-034 backlog**: subprocess-isolation for `measure_peak_traced`-based tests (v0.9.112). Median-of-3 is honest interim; restores [0.7, 1.4] once isolated.
- **DEC-035 backlog**: BIRCH `_insert` batching (removes ~5 MB glibc arena churn, reduces coef to true 1.0) — v0.9.112.
- **`fit_ab_test`** — tiny O(1) state; STABILITY §4 special case.
- **`test_R7` status backfill**: 56 has-fn cards with `status=None` (pending); R7 checks block existence only (v0.9.111). Flipping to draft/verified — separate sessions per group.

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

## Next 3 sessions

### v0.9.112 — Batch 3: group C (needs P/k/w second knob)

**Backlog closed from v0.9.111:**
- DEC-034 subprocess-isolation for `measure_peak_traced`-based tests (restores [0.7, 1.4])
- DEC-035 BIRCH `_insert` batching (removes glibc arena churn, coef → true 1.0)

**Group C pending** (from probe-109):
- `fit_zms` ×3, `fit_hampel`, `fit_sub_pca`, `fit_kshape`, `fit_eif`, `fit_autoencoder`, `fit_kmeans_ad`, `fit_patch_memory_ad`, `fit_s2g`, `fit_minirocket`, `fit_dwt_mlead`, `fit_bocpd` — O(batch) anomaly detectors
- `fit_pca` (CovarianceOp), `fit_kmeans` (k knob), `fit_lda` (vocab), `fit_coxph` (rank-based, n·d matrix)
- Per card: probe → derive formula → matrix → verified or draft-with-reason

**SpillArray for O(n) drafts** (DEC-013 + DEC-030):
- `fit_zms` ×3, `fit_cusum` — need SpillArray write before verified
- After SpillArray: cusum draft → verified

### v0.9.113 — SpillArray (DEC-013/030): zms x3 + cusum

`fit_zms` x3 + `fit_cusum` need SpillArray (O(n) output) before verified.
After SpillArray: cusum draft → verified; zms draft → verified.

### v0.9.114 — Group C continued + tag

Tag `v0.9.114` when 20+ verified total.

## Backlog (not scheduled)

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
