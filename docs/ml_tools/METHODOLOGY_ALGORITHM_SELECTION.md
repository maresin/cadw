# Algorithm Selection, Verification, and Framework Design

**Version:** 1.0 (2026-10-27, v0.9.87)
**Status:** normative; any new ML model MUST pass §8 gates before
`verified_in_cadw: true`.
**Scope:** all of `cadw/ml/`, `cadw/ts/`, `cadw/ops/`, `cadw/scalers.py`,
`cadw/pipeline.py`, `cadw/cv.py`. Plus any future ML section (DU-1,
D2, D4).

Complements `docs/ml_tools/README.md` (which explains how to add a
card). This document explains what a card must claim and prove.

---

## §1. Purpose

CADW's promise: "no more than one batch of data in RAM; the rest on
disk." Until 2026-10-27 this promise was honoured only where
developers happened to follow it. The audit (`.pingis/audits/
FINAL_REPORT.md`, d07072d) showed:
- 9 windowed TS anomaly modules allocated O(n·w) dense arrays before
  the guard even ran (HIGH-1).
- BOCPD allocated O(n·max_run) posterior history — 163 MB at n=100k,
  1.6 GB at n=1M (HIGH).
- GBDT multiclass allocates O(n·K) state for pred, grad, hess —
  240 MB at n=1M, K=10 (MEDIUM); guard correctly refuses at
  state_limit, but refuses ≠ works.
- Lambdamart vstack O(N·d) — same pattern.

Root cause: no written rule distinguishing "out-of-core by design"
from "in-core for small n." This document is that rule.

## §1.1 Basic cases, not exceptions

The following are *basic* cases, not exceptional. They must work at
any n (given disk). If any of them fails, this is a bug in CADW, not
in user data.

- A list of unique categories (dict / set / `np.unique` / `pd.unique`)
  whose cardinality exceeds `state_limit`.
- A whole column (n values of one feature) that does not fit in
  `state_limit`.
- Intermediate state of an iterative algorithm (pred in boosting, X
  stack in rank, residuals in OLS with diagnostics) that does not fit.
- One feature matrix for model training that does not fit.
- Sorting, grouping, joining, ranking, deduplicating a dataset that
  does not fit.

**Consequence.** No escape hatches in public API: no `read_all()`, no
`collect()`, no `to_numpy()` for n-sized results. If a model needs all
data in RAM, it is not out-of-core: it is either refactored to
O(batch) state, or marked `described_only: true` (not implemented in
CADW). The framework exists precisely to make these cases work at
arbitrary scale — anything less is a bug.

## §2. Five-step process

Any algorithm added to CADW MUST progress through these five steps.
Each step has a deliverable; skipping a step is a bug.

1. **Select** — pick from the literature with a reference implementation
   (paper + code, or widely-used library). Criterion: verified SOTA
   OR baseline-must-beat OR reference-family. No "sounds useful."
2. **Verify original** — is CADW's implementation paper-faithful? A
   documented approximation? A toy? Attribution must be honest:
   `literature` | `cadw_reproduced` (numeric parity vs reference) |
   `cadw_measured` (CADW-native synthetic).
3. **Design for framework** — fill in `state_class`, `state_type`,
   `bounded_state`, `streaming_strategy`, `primitive_reuse` (§4-§6).
   If bounded_state would be false and no fix exists — do not proceed
   to step 4. Refuse, or mark `described_only: true`.
4. **Implement** — primitives first (Reduce / HashShuffle / Sort /
   SpillArray). Element-wise / batch-local numpy is fine. n-sized
   numpy without one of the above is a code smell and must have a
   whitelist entry + backlog item.
5. **Register** — YAML card with all mandatory fields (§7) + gate
   tests pass (§8). Whitelist entries are visible debt, not
   "justification."

## §3. State taxonomy

Every fit / predict / transform must declare three fields:

| Field | Values | Meaning |
|-------|--------|---------|
| `state_class` | `O(1)`, `O(d)`, `O(d²)`, `O(batch)`, `O(model)`, `O(n)`, `O(n·w)`, `O(n·K)`, `O(N·d)` | Asymptotic RAM state |
| `state_type` | `aggregation`, `iterative`, `partitioned`, `cross_row`, `windowed`, `elementwise`, `mixed` | *Why* it grows (selects primitive) |
| `bounded_state` | `true` / `false` | RAM does NOT grow with n *provided disk is available* |

**Crucial:** `bounded_state` is the out-of-core criterion, not
`state_class` alone. `O(n)` with SpillArray → `bounded_state: true`
(RAM is O(batch), disk is O(n)). `O(n)` without SpillArray →
`bounded_state: false`.

**`n_scalability`** — second complementary field:

| Value | Meaning | Example |
|-------|---------|---------|
| `constant` | Works at any n given disk | Reduce-based (Logistic, PCA) |
| `linear` | RAM = O(n) but does not grow super-linearly | BOCPD with rolling (fix1c) |
| `quadratic` | RAM = O(n·something), needs guard + threshold | GBDT multiclass pre-refactor |
| `refuse_at_guard` | Correctly raises when n exceeds guard; not out-of-core | Lambdamart pre-refactor |

## §4. State types → primitives

| `state_type` | Primitive | Meaning |
|--------------|-----------|---------|
| `aggregation` | **Reduce** | Accumulate summary across batches (XTX, histograms, sums) |
| `partitioned` | **HashShuffle** | Split-by-key to disk, bounded RAM per partition |
| `sort_based` | **Sort** | Global median / top-k / k-fold / rank |
| `iterative` | **SpillArray** (new, §5) | n-sized array accessed across iterations |
| `windowed` | **strided_view + chunked** | Sliding windows without `.copy()` |
| `elementwise` | *(none needed)* | fit/predict streaming per batch |
| `cross_row` | **Sort** or **HashShuffle** | Stratified split, LOO encoding |

If state_type is not on this list — new primitive required; open a
design discussion, do not implement as numpy.

## §5. SpillArray (planned primitive)

Arrow IPC-backed n×d array. RAM always O(batch); disk is n·d·8 bytes
plus IPC overhead.

```python
with SpillArray(shape=(n, K), dtype=np.float64,
                path=tmp, guard=mg) as arr:
    for batch_idx, batch in df.batches():
        arr.write(batch_idx, compute(batch))
    for round_i in range(n_estimators):
        for batch_idx, batch in df.batches():
            prev = arr.read(batch_idx)      # O(batch·K) RAM
            new = update(prev, batch)
            arr.write(batch_idx, new)
    final = arr.read_all()                  # opt-in; warns if > guard
```

Rules:
- Guard calls `check_state(O(batch))`, never `check_state(O(n))`.
- `read_all()` is guarded by `check_state` and emits a warning.
- Lifecycle managed by `with`; file deleted on `__exit__`.

Until SpillArray is implemented (Phase 0b), models with
`state_type: iterative` and `bounded_state: false` must either:
(a) run only when n·d·8 ≤ state_limit (current GBDT multiclass), or
(b) refuse with an actionable error.

## §6. Primitive reuse check

Every `fit_*`, `predict_*`, `transform_*` in `cadw/ml/`, `cadw/ts/`,
`cadw/ops/` must satisfy one of:

1. No n-sized allocation anywhere in the function (streaming).
2. Any n-sized allocation is wrapped in a SpillArray (or equivalent
   primitive) with explicit lifecycle.
3. A `# noqa: gate_G1 (refactor pending: <backlog item>)` comment
   and a whitelist entry in `.pingis/plan_v0.9.87_refactor.md`.

`test_primitive_reuse.py` enforces via AST scan (§8).

## §7. Mandatory YAML card fields

Any model with `verified_in_cadw: true` MUST include:

```yaml
- name: <Model>
  ...
  state_class: O(<...>)
  state_type: aggregation | iterative | partitioned | cross_row |
              windowed | elementwise | mixed
  bounded_state: true | false
  n_scalability: constant | linear | quadratic | refuse_at_guard
  streaming_strategy: |
    fit: <1 sentence on how fit achieves stated state_class>
    predict: <1 sentence>
  primitive_reuse: [reduce, spill_array, ...]
```

Missing any of these fields → schema validation fails
(`test_ml_tools_schema.py`).

`tier1_core` + `bounded_state: false` → schema validation fails
(unless explicit `# tier1_core_exception: <reason>` comment + entry
in `.pingis/plan_v0.9.87_refactor.md`; exceptions are debt, not
policy).

## §8. Enforcement gates (planned)

Four gate tests run in the fast suite. Failures block merge.

**G1 — primitive reuse** (`tests/test_primitive_reuse.py`)
AST-scan: for each `fit_*` with n-sized `np.zeros / empty / full`
not wrapped in `SpillArray` → fail, unless whitelisted.

**G2 — guard before alloc** (`tests/test_guard_before_alloc.py`)
AST-scan: for each n-sized allocation, is there a `check_state` or
`check_batch` call earlier in the same function? If not → fail.

**G3 — no dense windows** (`tests/test_no_dense_windows.py`)
AST-scan: `sliding_window_view(...).copy()` → fail, unless
whitelisted with `# gate_G3 (refactor: chunked consumption)`.

**G4 — YAML completeness** (`tests/test_ml_tools_schema.py`
extended)
Every `verified_in_cadw: true` card has §7 fields. `tier1_core` +
`bounded_state: false` without exception → fail.

Whitelist lives in `.pingis/plan_v0.9.87_refactor.md`. As refactoring
proceeds, entries are removed. **When the whitelist is empty, the
refactoring is complete.** Gate tests become pure regression guards.

## §9. Anti-patterns (blocked by gates)

| Anti-pattern | Gate | Fix |
|---|---|---|
| `np.zeros((n, K))` for iterative state | G1 | SpillArray |
| `sliding_window_view(...).copy()` full-n | G3 | strided + chunked |
| Guard check after allocation | G2 | move guard up |
| `state_class` missing in card | G4 | add field |
| `tier1_core` + `bounded_state: false` | G4 | demote tier or refactor |
| `np.vstack(list_of_batches)` with len=n | G1 | SpillArray or pre-allocated Reduce |

## §10. Tier rules

| bounded_state | n_scalability | tier1_core | tier2_wrapper | tier3_export |
|---|---|---|---|---|
| true | constant | ✅ | ✅ | ✅ |
| true | linear | ✅ | ✅ | ✅ |
| false | refuse_at_guard | ⚠️ exception | ⚠️ exception | ✅ |
| false | quadratic | ❌ | ❌ | ✅ |

A `tier3_export` model is one CADW declares "not implementable in
CADW's constraint" — reference-only. This is honest.

## §11. Legacy migration

Existing models that fail gates are whitelisted with a backlog item:

```
# .pingis/plan_v0.9.87_refactor.md
- [ ] P1: GBDT multiclass → SpillArray for pred/grad/hess
- [ ] P2: Lambdamart → SpillArray for X stack
- [ ] P3: 9 windowed modules → strided view (sub_pca done; ...)
- [ ] P4: kshape → 2-pass chunked
- [ ] P5: Pass 2 audit (47 ML + 15 TS manual classification)
- [ ] P6: YAML registration with §7 fields for all
- [ ] P7: Remove all whitelist entries
```

Each refactor commit removes its corresponding whitelist entries.
Gates are added once (Phase 0c) with the initial whitelist populated.
`test_primitive_reuse` fails if a new violation is added without
whitelisting.

## §12. Checklist for a new model

Use this verbatim when adding any model. If any box can't be
checked, either fix or `described_only: true`.

1. [ ] Reference: paper or library code cited
2. [ ] Original vs CADW parity recorded (`cadw_reproduced` or
   `cadw_measured` with numeric evidence)
3. [ ] `state_class` filled and matches source (verified by grep)
4. [ ] `state_type` filled
5. [ ] `bounded_state` filled; if `false`, has refactor plan and is
   not `tier1_core` without exception
6. [ ] `streaming_strategy` filled
7. [ ] `primitive_reuse` lists actual primitives used
8. [ ] Guard before every n-sized alloc (G2)
9. [ ] No dense windows unless chunked (G3)
10. [ ] YAML card validated by `test_ml_tools_schema.py` (G4)
11. [ ] Gate tests pass or model whitelisted with backlog item

## §13. Reference: current state (2026-10-27)

| Model | state_class | state_type | bounded_state | status |
|---|---|---|---|---|
| Logistic | O(d²) | aggregation | true | ✅ Reduce |
| Ridge / Lasso | O(d²) | aggregation | true | ✅ Reduce |
| PCA | O(d²) | aggregation | true | ✅ Reduce |
| kmeans | O(k·d) | aggregation+iterative | true | ✅ (k bounded) |
| Naive Bayes | O(C·d) | aggregation | true | ✅ Reduce |
| CoxPH | O(d²) | aggregation | true | ✅ Reduce |
| IForest/EIF | O(n_trees·subsample) | aggregation | true | ✅ (subsample bounded) |
| ALS | O(m·k+n·k) | partitioned | true | ✅ HashShuffle |
| BOCPD | O(keep·max_run) | iterative | true | ✅ fix1c rolling |
| sub_pca | O(batch·w) | windowed | true | ✅ fix1a |
| s2g | O(batch·w) | windowed | true | ✅ fix1b est fix |
| kshape | O(n·w) | windowed+iterative | **false** | ⚠️ P4 2-pass needed |
| dwt_mlead | O(batch·w) | windowed | true | ✅ fix1b |
| eif | O(batch·w·c) | windowed | true | ✅ fix1b |
| tspq | O(batch·w) | windowed | true | ✅ fix1b2 |
| patch_memory_ad | O(batch·w) | windowed | true | ✅ audit-corrected |
| autoencoder | O(batch·w) | windowed | true | ✅ fix1b2 |
| kmeans_ad | O(batch·w) | windowed | true | ✅ fix1b2 |
| **GBDT multiclass** | **O(n·K)** | **iterative** | **false** | ⚠️ **P1** |
| **Lambdamart** | **O(N·d)** | **iterative** | **false** | ⚠️ **P2** |

(Remaining ~30 models classified in P5.)

## §14. Version history

- 1.0 (2026-10-27, v0.9.87): initial version. Follows audit d07072d,
  fix1a-c (sub_pca/eif/kshape/s2g/dwt_mlead/tspq/autoencoder/kmeans_ad/
  BOCPD). Defines SpillArray as the missing primitive for iterative
  state. Establishes 4 gates + whitelist pattern for legacy migration.
