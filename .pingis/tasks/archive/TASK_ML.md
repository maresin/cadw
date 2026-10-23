# TASK_ML.md — CADW ML / pipeline / cv / scalers audit (round 3)

**Type:** audit + benchmarks + source review
**Branch:** `pingis/ml-audit-v0.9.21`
**Base:** `master @ 598f782` (v0.9.21)
**Output dir:** `_FOR_PINGIS/` (gitignored)
**Prior audits:** `AUDIT.md` (v0.9.14 round 1, ops/+dataframe.py);
`TASK2.md` (library memory audit)

---

## Context — why this task exists

`AUDIT.md` (v0.9.14) audited `cadw/ops/` + `cadw/dataframe.py` (~10k
lines). It found **5 CONFIRMED RAM violations** + 4 SUSPECTED + 3
silent data-loss bugs. All CONFIRMED findings closed in
v0.9.18–v0.9.21 (atomic patches + 1 fix-cycle per patch, except
v0.9.20 with 5 cycles — §13.4/238 lesson).

**~6000 lines of code have never been audited:**

| Module | Lines | Audited | Last touched |
|---|---|---|---|
| `cadw/ml/gbdt.py` | 1492 | **NO** | v0.7.9 |
| `cadw/ml/naive_bayes.py` | 687 | **NO** | v0.7.6 |
| `cadw/ml/linear.py` | 463 | **NO** | v0.7.7 |
| `cadw/ml/logistic.py` | 404 | **NO** | v0.7.0 |
| `cadw/ml/kmeans.py` | 358 | **NO** | v0.7.1 |
| `cadw/ml/pca.py` | 332 | **NO** | v0.9.9 |
| `cadw/pipeline.py` | 1325 | **NO** | v0.9.1e |
| `cadw/cv.py` | 374 | **NO** | v0.9.3a |
| `cadw/scalers.py` | 600 | **NO** | v0.9.2f |
| `cadw/ops/metrics.py` | 711 | **NO** | v0.9.2c |
| **Total** | **6746** | — | — |

CADW's memory guarantee (§3.11): "every high-risk call has a guard
before it" — enforced by AST guard on **`cadw/ops/` + `dataframe.py` +
few others** (see `tests/test_arrow_call_memory_guard.py::AUDITED_FILES`).
ML layer, pipeline, cv, scalers are **excluded from the AST guard** —
either audit lag or because they use a different memory profile
(§9.2.N formulas). Any RAM bug there ships silently.

Three prior silent data-loss bugs (§13.4/184, /218, /223) took months
to find. They were in `dataframe.py` / `groupby.py` / `_per_group_apply`.
**Same class** — null handling, type coercion, state validation — is
possible in ML gradient accumulation, per-class loops, kmeans init,
pipeline fanout. This audit targets that.

---

## Constraints

- **Do NOT modify `cadw/`.** Audit only.
- Branch: `pingis/ml-audit-v0.9.21` from `master`.
- Output only to `_FOR_PINGIS/` (gitignored) and `.pingis/`.
- No new dependencies. Existing: `pyarrow`, `numpy`, `psutil`, `pandas`
  (dev-only), `xxhash`.
- All measurement via `tracemalloc` + `psutil.Process().memory_info().rss`
  + optional `RSSSampler` thread (see `scripts/_measure_stage4.py` pattern).
- **No dogfood** (§13.4/228): audit scripts are measurement harnesses,
  not tests. Each `_measure_*.py` asserts **output correctness** vs
  reference impl + measures peak. No pytest integration, no dogfood.
- Do NOT modify `master`, no tags, no commits to `master`.
- All commits on `pingis/ml-audit-v0.9.21` branch.

---

## Priorities (order — you may reorder based on findings)

### K1. ML layer audit — HIGHEST (3736 lines, 6 modules)

#### K1a. `cadw/ml/gbdt.py` (1492 lines)

**Suspected findings:**

1. **S1: `_AllNodeHistOp` state = S × F × B × 16 B (g + h, double-buffer).**
   For F=1000, B=255, S=63 (level d=5): 63 × 1000 × 255 × 32 ≈ **514 MB**.
   Does `estimate_state_nbytes` return this? Is it checked before `op.init()`?
   `S` = nodes at current level, not known before init — formula may
   underestimate.

2. **S2: `_fit_boosting` internal state O(n × K).** `pred` (n × 8 B) +
   `cur_leaf` (n × 8 B) + (multiclass) `y_all` (n × K × 8) + `grad_all`
   (n × K × 8) + `hess_all` (n × K × 8). For N=10M, K=10: ~2.5 GB.
   `check_state(est_n_state)` at line ~731 — `est_n_state = n_total * 32`
   for binary / `n_total * K * 8 * 3` for multiclass. **Verify formula
   covers `y_all` in multiclass path.**

3. **S3: `_validate_binary_y` redundant passes.** Line ~625 — reads
   `col.unique()` + `col.to_pylist()` + `_column_to_f64(col)` for the
   same batch. One `_column_to_f64` suffices.

4. **S3: predict_df per-tree loops.** Line ~430 — per-batch loop, not
   per-tree. Verify no per-tree allocation.

**Bench matrix:**
- Shape: N ∈ {10k, 100k, 1M}, d ∈ {2, 10, 100}, B ∈ {16, 32, 255},
  depth ∈ {2, 3, 5}.
- Measure: `_fit_boosting` peak RSS delta vs `n_total × 32` (binary)
  / `n_total × K × 8 × 3` (multiclass). Report ratio C = measured / formula.
- Test: `_AllNodeHistOp.estimate_state_nbytes()` vs
  `state_nbytes(init())` — must be `>=`. Test S ∈ {3, 7, 31}, F ∈ {2,
  100, 1000}, B ∈ {16, 255}.
- Error case: tight MemoryGuard (5 MB) on N=100k, F=1000 → raise BEFORE
  OOM?

#### K1b. `cadw/ml/naive_bayes.py` (687 lines)

**Suspected:**

1. **S1: `_GaussianStatsOp` dynamic per-class state.** Estimate
   `32 × (8 + 16×d)` assumes 32 classes. If K > 32, actual > estimate.
   Race window: state grows per-update, check_state fires **after**
   growth (per-update), not before. For K=100, actual = 3.1× estimate.
   Does dict growth OOM between check_states?

2. **S3: per-class Python loop.** Line ~85: `for cls in np.unique(y)` —
   K classes per batch. Unavoidable for Reduce but document.

3. **S3: `_check_X_non_negative` two-pass.** Separate scan from fit.
   2× I/O. Can fold into fit first Reduce.

#### K1c. linear.py / logistic.py / kmeans.py / pca.py (1557 lines combined)

**Suspected:**

1. **S3 (retracted by AUDIT §2): `_coordinate_descent` O(d²)/iter.**
   Real win is warm-start over alpha-path — feature, not optimization.
   Do NOT re-report without new evidence.

2. **S1 (kmeans): `_stack_numeric` per-batch.** Line ~88 —
   `parts = [_column_to_f64(...)]`. Zero-copy if no nulls, else materialize.
   For d=100, batch=10k: 8 MB/batch. Verify `to_numpy(zero_copy_only=False)`
   path — does it copy for null-free int64?

3. **S1 (kmeans): `_kmeans_plus_plus_init` first-batch only.** §2.5
   SUSPECTED — **quality** bug. Verify with skewed data: first-batch
   k-means++ vs reservoir-sampled. If quality drop > 5% on skewed data
   → CONFIRM + backlog for reservoir fix. Prototype acceptable.

4. **S1 (pca): CovarianceOp O(d²).** `np.zeros((d, d))`. For d=1000:
   8 MB, d=10000: 80 MB. Checked via `estimate_state_nbytes`? §9.2.1
   formula `8 + 8d + 8d²`. Verify.

5. **S3 (logistic): Newton IRLS per-iteration full Reduce.** `max_iter=100`
   default → 100 passes × ~1 sec = 100 sec. Perf, not RAM. Document.

### K2. `cadw/pipeline.py` audit (1325 lines)

**Suspected:**

1. **S1: `_merge_fanout` row-alignment.** `n_first = dfs[0].count()`
   then per-branch `count()` — O(branches) full scans. Then `zip(*iters)`.
   For 3 branches × 100k rows: 3 counts + 3 open readers. Peak = 3 ×
   (one batch) + writer buffer. Verify bounded.

2. **S1: `ColumnTransformer.transform` fanout.** `step.transform(subset)`
   per transformer — all CadwDataFrames (disk-backed). Then
   `_merge_fanout(results)`. Same bounded analysis. Verify per-step
   `.fit()` state summing in `estimate_state_bytes`.

3. **S2: `Pipeline.plan` = sum(step.state_bytes) only.** Doesn't model
   **peak within step**. If transform needs 200 MB/batch but
   state_bytes = 5 MB, plan says `fits`, runtime OOM. Verify — either
   plan is honest "state only" (document) or missing peak model (bug).

4. **S3: `to_json` state size.** GBDT with 1000 trees → `state/GBDTStep.json`
   — measured size? If huge, document serialization cost.

### K3. `cadw/cv.py` audit (374 lines)

**Suspected:**

1. **S2: `KFold(materialize=True)` vs default.** Default uses
   `_attach_column_with_fn` (streaming, O(batch)). `materialize=True`
   calls `_attach_column` (full array) with `check_state(20*n)` — formula
   = int32 fold_id (4n) + int64 arange (8n) + int64 perm (8n) = 20n.
   Verify.

2. **S3: `KFold.split` count per fold?** Line ~200 — `n = df.count()`
   once before loop. Verify no re-count.

3. **S3: `TimeSeriesSplit` re-sort per split().** `sorted_df =
   df.sort_values(...)` — one sort per split() generator. Multiple calls
   → multiple sorts. Document.

### K4. `cadw/scalers.py` audit (600 lines)

**Suspected:**

1. **S2: `RobustScaler.fit` O(d) external_sorts.** Line ~525 — `df.quantile`
   per column. For d=100, N=1M: 100 × ~5 sec = 500 sec. §11.6 backlog.

2. **S2: `_NumericStatsOp` per-batch cast.** `pc.cast(..., float64)` +
   `_column_to_f64`. Zero-copy for float64 null-free. Verify.

3. **S3: `transform` per-batch kernels.** Bounded. No accumulation.

### K5. `cadw/ops/metrics.py` audit (711 lines)

**Suspected:**

1. **S2: classification per-batch to_pylist.** `batch_memory_factor() = 8.0`.
   Verify 8.0 sufficient for K=100 multiclass log_loss: per batch = (K+1) ×
   batch_rows × 28 B pylist vs (K+1) × batch_rows × 8 B Arrow. Ratio 3.5×.
   Factor 8.0 over-conservative but safe. OK — mark verified.

2. **S3: `_compute_roc_auc_binary` two-pass.** Pass 1 count, pass 2 stream.
   Sort-based AUC. Verify no O(N) intermediate.

### K6. Verify remaining AUDIT.md §2.x SUSPECTED (5 items)

- **§2.1** `check_batch(extra_factor=1.0)` default — grep call sites with
  `to_pylist` downstream, no explicit `extra_factor`. Confirm or retract.
- **§2.3** `Table.join` build-side re-check missing — bench 100k×100k
  join, compare `check_state(5 × row_bytes_out × matches)` to actual
  RSS peak. §9.2.6 formula.
- **§2.4** `_per_group_apply` `_emit_group` per-row to_pylist.
  Verify `_flush_buf` has `check_batch`.
- **§2.5** kmeans++ — covered by K1c.3.

### K7. Report consolidation

One file: `_FOR_PINGIS/AUDIT_ML_v0922.md`.

Structure (mirror AUDIT.md v0.9.14):
- **§0 Executive summary** — N confirmed, M suspected, K retracted. Table.
- **§1 CONFIRMED** — file:line, snippet, benchmark, measured effect,
  proposed fix (with reasoning), output-identity assertion.
- **§2 SUSPECTED** — rationale, why unmeasurable (or deferred).
- **§3 RETRACTED** — reason.
- **§4 Priority matrix** — S1/S2/S3/S4, suggested v0.9.22 scope.
- **§5 Evidence** — bench scripts, raw output, test matrix.

---

## Deliverables in `_FOR_PINGIS/`

### `AUDIT_ML_v0922.md` (~150–300 lines)

Sections as §K7. Measured numbers, not opinions.

### `_measure_ml_*.py` (bench scripts)

One per module: `_measure_gbdt.py`, `_measure_nb.py`,
`_measure_kmeans.py`, `_measure_pipeline.py`, `_measure_cv.py`,
`_measure_scalers.py`, `_measure_metrics.py` (7+ files).

Each:
- Diverse shapes (N, d, K, nulls, depth, bins).
- **Restart process per case** (or `gc.collect() + del`) — else RSSSampler
  reads stale baseline.
- Assert output correctness vs reference impl (numpy formula, sklearn if
  available via `importorskip`, or existing behavior).
- Emit JSON: `_out_ml_<module>.json` — `[{shape, metric, measured,
  formula, ratio, rss_peak, rss_baseline, verdict}]`.

### `bench_ml/RESULTS.md`

Consolidated table: `# | Module | Shape | Metric | Measured | Formula | Ratio | Verdict`.

### `_FOR_PINGIS/proto/` (optional)

For kmeans reservoir prototype if worth testing (K1c.3).

---

## Process

1. **Setup:**
   ```bash
   git checkout -b pingis/ml-audit-v0.9.21 master
   uv run pytest -q -m "not slow"   # 1912 pass baseline
   ```

2. **K1a gbdt.py** — grep, read, bench. Write `_measure_gbdt.py`.
   Report findings incrementally (chat 20–30 lines per finding).

3. **K1b/K1c** — same for naive_bayes / linear / logistic / kmeans / pca.

4. **K2–K5** — same for pipeline / cv / scalers / metrics.

5. **K6** — verify remaining AUDIT §2.x, mark CONFIRMED / RETRACTED.

6. **Write `AUDIT_ML_v0922.md`** — consolidate.

7. **Commit** to `pingis/ml-audit-v0.9.21`. No push to remote.

**Checkpoint every 30–45 min:** short summary in chat. Don't wait for
full report.

---

## Acceptance criteria

- [ ] Branch `pingis/ml-audit-v0.9.21` exists, all commits on it.
- [ ] Fast+slow tests green on branch tip (baseline sanity; no code changes
      → must be green).
- [ ] `_FOR_PINGIS/AUDIT_ML_v0922.md` — full report with measured numbers.
- [ ] Every K1–K5 SUSPECTED (15+ items) CONFIRMED or RETRACTED with reasoning.
- [ ] K6 §2.x (5 items) CONFIRMED or RETRACTED.
- [ ] ≥7 `_measure_ml_*.py` — reproducible, asserts output correctness.
- [ ] `bench_ml/RESULTS.md` — consolidated table.
- [ ] Priority matrix S1/S2/S3/S4 with **suggested v0.9.22 scope**.
- [ ] `git diff cadw/` empty.
- [ ] `_FOR_PINGIS/` gitignored — `git status` clean after commit.

---

## Re-run criteria

- New ML module or significant change to existing (`v0.9.22+`).
- sklearn/numpy/pyarrow major version bump.
- ~2–3 months since last run.

---

## Appendix A — reference bench patterns

`scripts/_measure_stage4.py`, `_measure_stage4b.py`, `_measure_index.py` —
proven from AUDIT.md v0.9.14. Study before writing new scripts. Patterns:
- `RSSSampler` thread (20 ms interval).
- Restart per case (or `gc.collect()` between).
- Assert output correctness vs reference.
- Emit JSON.

## Appendix B — finding format (from AUDIT.md §1.1)

```
### 1.1 `CadwDataFrame.shift()` — batch-boundary buffer materialized to Python

**File:** `cadw/dataframe.py:2017-2051`

**Problem.** (2-3 sentences: what, why bad.)

**Measured** (synthetic 300k rows, batch=50k, k=2, int64):

| | peak (tracemalloc) | time |
|---|---|---|
| current | 2.05 MB | 16.6–17.9 ms |
| fixed | ~0 MB | 0.4–0.5 ms |
| delta | -2.05 MB | **40.6×** |

**Fix** (verified — same output asserted): (code snippet + reasoning).
```

Every CONFIRMED: file:line, snippet, measured peak (tracemalloc) + time,
proposed fix + reasoning, test verifying output identity.

## Appendix C — CADW memory model context

Read `DECISIONS.md` §9.2.1–§9.2.19 for reference formulas. Each op
should have a formula bounding peak. Audit question: does actual peak
≤ formula × safety? For ML:

- **GBDT:** no formula in §9.2.N currently. State formula in
  `estimate_state_nbytes`. Missing or wrong.
- **NB:** same.
- **PCA:** `CovarianceOp` — §9.2.1. Verify.
- **KMeans:** `KMeansStepOp` state O(k×d) — formula in comment, not §9.2.N.
- **Pipeline:** `plan()` sums step_state_bytes — no per-step peak model.

**Finding class:** formula exists and actual > formula×2 → CONFIRMED
drift. No formula and actual peak large → CONFIRMED missing formula
(guard-by-omission). Both actionable.

## Appendix D — read first

1. `DECISIONS.md` §9.2.1–§9.2.19, §11.4, §13.4/178–240 (new meta-lessons
   from v0.9.17–v0.9.21).
2. `_FOR_PINGIS/AUDIT.md` (v0.9.14 round 1) — pattern reference.
3. `tests/test_arrow_call_memory_guard.py` — `AUDITED_FILES` list.
4. `cadw/ml/gbdt.py` — biggest, most complex state, highest priority.
5. `cadw/core/models_reduce.py`, `models_groupby.py`, `models_join.py` —
   reference memory model patterns.

## Appendix E — non-goals

- Not **fixing** findings (v0.9.22 patches based on this report).
- Not auditing `cadw/ops/` (AUDIT.md v0.9.14, all CONFIRMED closed by
  v0.9.21).
- Not auditing `dataframe.py` fully (mostly covered; K6 verifies §2.4).
- Not testing correctness beyond what's needed to prove benchmark isn't
  measuring the wrong thing.
- Not modifying tests to skip/relax.

---

**Branch:** `pingis/ml-audit-v0.9.21`
**Base commit:** `master @ 598f782` (v0.9.21)
**Output dir:** `_FOR_PINGIS/` only

---

**Suggested starting point:** K1a (`cadw/ml/gbdt.py`) — biggest module,
most complex state, highest priority. Check `estimate_state_nbytes`
vs `state_nbytes(init())` first — quick win if formula wrong.