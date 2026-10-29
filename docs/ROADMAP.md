# CADW Roadmap

Last updated: 2026-10-29 (after v0.9.92).

## Current: v0.9.92

**Closed:** streaming-output migration (S0–S6, default flip).
- All ops + ML predict return `CadwDataFrame` via facade
  (or `AutoResult` / `AutoArrayResult` direct, `output="auto"`).
- No hidden OOM in migrated functions.
- Test count: ~4128 passed, 8 skipped (fast; includes 139 gates).

**Known non-OOC** (per STREAMING §6.1) — documented, not regressions:
- `binseg`, `stray`: intrinsic O(n)/O(n²), refuse-at-guard.
- `linear`/`logistic`/`coxph`/`pca`: O(d²) state, wide-d not handled.
- `encode_categorical`, `metrics`-classification: K guard, no spill.
- TS single-series fits: `tier1_core_exception`.
- sparse `tfidf.transform`: needs CSR SpillBuffer.

## Next 3 sessions

### v0.9.93 — wide-feature OOC (d-axis)

**Goal:** handle `d > 10^4` for closed-form models without OOM.

**Approach:** `cadw/ops/optim.py::sgd_streaming` (ready since v0.9.89).
Migrate:
- `cadw/ml/linear.py::fit_linear` — add `solver="auto"|"closed"|"sgd"`.
  auto = closed if d ≤ 10^4, else sgd.
- `cadw/ml/logistic.py::fit_logistic` — same.
- `cadw/ml/coxph.py::fit_coxph` — SGD on Breslow likelihood.
- `cadw/ml/pca.py::fit_pca` — optional randomized SVD at large d
  (`randsvd` already exists).

**Tests:**
- `d ≤ 1000`: closed-form vs SGD bit-identical (atol=1e-6).
- `d > 10^4`: SGD converges (relative coefficient error < 5%,
  reference closed-form via `rankdef=...` on smaller subset).
- `d = 10^5`: fit succeeds without OOM (only possible via SGD).

**Breaking:** none if default auto selects closed for d ≤ 10^4.

### v0.9.94 — unified `CadwDataFrame` output

**Goal:** collapse 7 return types to 1.

Current: `CadwDataFrame`, `AutoResult`, `AutoArrayResult`,
`pa.Table`, `np.ndarray`, `Dataset`, `(schema, Path)`.

Target: **`CadwDataFrame` on every public facade method; inside —
any wrapper**. Direct callers (`groupby_aggregate(...)` etc.) also
return `CadwDataFrame` — same type as facade.

**Breaking (2nd):** direct callers of `groupby_aggregate` /
`drop_duplicates` / `iforest.score_samples` / etc. get
`CadwDataFrame` instead of `AutoResult` / `np.ndarray`.
`output="memory"` deprecation path (warn, remove v1.0).

**Why:** `Pipeline` currently accepts only `transform(df) -> CadwDataFrame`.
Half of ML models don't participate. Fix unlocks
`df.groupby().agg().fit_predict()` chains.

**Tests:** new `test_pipeline_all_models.py` — chain `iforest` /
`chains` / `ordinal` / `gmm` into `Pipeline`.

### v0.9.95 — S5b-full + K-spill

**S5b-full** — TS group path guards:
- `ts/features.py::_shift_stream` group path.
- `ts/stats.py::rolling_stats`, `rolling_crosscorr` group path.
- `ts/thf.py::_fit_one` multi-series.

**K-spill** — HashShuffle-based:
- `encode_categorical` (vocab > state_limit).
- `metrics` classification (K > state_limit).
- `lda::_build_vocab`, `tfidf::fit_tfidf` (vocab via HashShuffle).

**Tests:** `K = 10^7` unique categories without OOM; classify with
`K = 10^6` classes.

## Do not re-solve

Per STREAMING §6.1 — won't-fix with rationale:

- **eif bit-identical chunked** (S6b-full): 40 GB I/O at n_est=100,
  n_w=10^6, d=50. Zero functional gain (tests pass at recall 0.5).
- **kmeans_ad bit-identical** (S6c-full): 3.2 GB I/O at k=8,
  n_w=10^6. Reservoir is deterministic; tests pass.
- **autoencoder beyond n_w = state_limit/16** (S6d-full):
  2-3 sessions, no urgent use case.
- **sparse tfidf.transform**: variable-width rows, needs CSR
  SpillBuffer design (separate).
- **binseg / stray**: intrinsic O(n) / O(n²). `refuse_at_guard`
  is correct behaviour.
- **TS single-series fits** (STL, VAR, VECM, ETS, Theta):
  `tier1_core_exception`, n ≤ 10^6 typical.

If a future session considers one of these, **read STREAMING §6.1
first** — cost calculations and rationale are there.

## Backlog (not scheduled)

- `df.estimate_memory(op, params)` — pre-flight RAM estimate
  (per-axis). Would let users see `fits: false, reason, alternatives`
  before calling.
- YAML `bounded_state_dims` for all 84 verified cards.
- Audit pass 3 (after v0.9.95).
- `sgd_streaming` mini-batch optimization (currently single-sample).

## Session discipline

- Before commit: `pytest -q <affected files>` (focused, ~20s).
- Before tag: `pytest -q -m "not slow"` (full fast, ~10min).
- Before release: `pytest -q` (all, includes slow).
- See `AI_CONSTITUTION.md` §4 for any new `fit_*` / `predict_*`.
