# CADW Roadmap

Last updated: 2026-11-02 (after v0.9.109d).

## Current: v0.9.109d

Framework + refactor milestones done (v0.9.105-108). Batch 1 progress
in v0.9.109 (a-d). Next: fill pending cards via EXPERIMENT_PROCEDURE.

## Progress (v0.9.105 - v0.9.109d)

- **v0.9.105** — 9 cards formula-calibrated (draft). EXPERIMENT_PROCEDURE.md (274 lines). DEC-018..025.
- **v0.9.106** — framework chunking + state-aware guard (DEC-022/023 closed).
- **v0.9.107** — 9 cards `verified` (tag). 11 verified total (2+9).
- **v0.9.108** — refactor `_stack_numeric` → `_common.py` (23 dedup sites).
- **v0.9.109a** — `fit_cusum` honesty fix (O(1) → O(n); YAML lie caught by probe; DEC-030 draft, SpillArray rewrite pending).
- **v0.9.109b** — `test_R5` aliases consistency (18 fns / 43 cards, 0 inconsistent).
- **v0.9.109c** — `fit_randomized_svd` verified (coef=3.0, OOC ratio 1.00); 12 verified total.
- **v0.9.109d** — snapshot bootstrap includes `EXPERIMENT_PROCEDURE.md` (T2 doc, position 4).

## Status baseline

| Status | Count | Note |
|---|---|---|
| verified | **12** | 2 MatrixProfile + 9 v0.9.107 + 1 randsvd (v0.9.109c) |
| draft | 31 | 21 legacy + 9 v0.9.105 + 1 cusum (v0.9.109a) |
| pending (no status) | 55 fns (124 entries) | 88 with fn + 62 no-fn (described_only candidates) |

## Known gaps (remaining)

- **DEC-013**: `fit_zms` x3 — O(n) output needs `SpillArray`. Deferred to v0.9.111.
- **DEC-014**: `fit_pelt` x2 — `_collect_series` container. Deferred.
- **DEC-030** (v0.9.109a): `fit_cusum` — O(n) state + O(n) output, SpillArray rewrite pending. Draft.
- **R7 backlog**: `test_R7` — all non-described_only cards must have `implementation_status` block with both fields.
- **62 no-fn cards**: `described_only: true` schema + `test_R6` — v0.9.110 Part 2.
- `fit_ab_test` — tiny O(1) state; STABILITY §4 special case.

## Lessons learned (v0.9.105-109)

- **YAML `ram_formula` / `state_class` often lie.** `fit_cusum` claimed O(1); actual code allocated 6 × 8n. Probe code first, YAML second.
- **Alias cards** — 18 fns / 43 cards. When flipping one, flip all aliases (test_R5 enforces).
- **`bounded_state_dims`** — real axes only (d, C, n_components, n_oversamples). Do NOT list auto-injected `batch`/`row_size` — R2 rejects them in dims; they are implicit names allowed in formula only.
- **`implementation_status` block may be entirely absent** in older cards (e.g. `fit_randomized_svd` had only `verified_in_cadw` originally). `test_R7` (backlog) catches this. For now: when patching YAML, check for block presence, create if missing.
- **YAML write scripts must use `encoding="utf-8"` explicitly** on `read_text` / `write_text`. Locale-agnostic, avoids mojibake on files with non-ASCII characters.
- **Never mix `git checkout <file>` and `python patch_<file>.py` in one block.** Order: checkout first (clean HEAD), then patch. `checkout` after patch silently reverts the patch.

## Next 3 sessions

### v0.9.110 — Batch 2: 3-4 verified cards + described_only schema

**Part 1 — verified cards** (per EXPERIMENT_PROCEDURE):
- `fit_isolation_forest` (O(model), n_estimators × ψ)
- `fit_linucb` (O(model), K × d²)
- `fit_gmm` (sample-based EM, ~10k sample)
- `fit_birch` (CF-tree, bounded n_entries)

**Part 2 — `described_only` for 62 no-fn cards:**
- Cards: TabPFN, Chronos-*, BERT, ResNet50, GCN, etc. — reference-only, not implementable
- Schema: add `described_only: true` + `test_R6` (no-fn must be described_only)
- `test_R7` (all non-described_only have implementation_status block)

**Part 3 — Aliases consistency safeguard:**
- `test_R5` already enforces consistency for existing aliases
- Add helper `scripts/bulk_flip_status.py --fn NAME --from draft --to verified` for batched flips

### v0.9.111 — Batch 3: group C (needs P/k/w second knob)

**Group C pending** (from probe-109):
- `fit_zms` ×3, `fit_hampel`, `fit_sub_pca`, `fit_kshape`, `fit_eif`, `fit_autoencoder`, `fit_kmeans_ad`, `fit_patch_memory_ad`, `fit_s2g`, `fit_minirocket`, `fit_dwt_mlead`, `fit_bocpd` — O(batch) anomaly detectors
- `fit_pca` (CovarianceOp), `fit_kmeans` (k knob), `fit_lda` (vocab), `fit_coxph` (rank-based, n·d matrix)
- Per card: probe → derive formula → matrix → verified or draft-with-reason

**SpillArray for O(n) drafts** (DEC-013 + DEC-030):
- `fit_zms` ×3, `fit_cusum` — need SpillArray write before verified
- After SpillArray: cusum draft → verified

### v0.9.112+ — Group C continued + tag

Tag `v0.9.112` when 20+ verified total.

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
