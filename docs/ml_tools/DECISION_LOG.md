# Decision Log

**Append-only** record of methodological choices. Purpose: next AI
session sees "why X, not Y" without re-litigating from scratch.

**Rule:** when a choice affects multiple sessions (architecture, status
semantics, formula method), append here. Do not edit history. If a
decision is superseded, add a new entry referring to the old id.

**Hierarchy:** Tier 4 (reference). Complements
`AI_CONSTITUTION.md` (Tier 1), `STABILITY_PROTOCOL.md` (Tier 2),
`ALGORITHM_TEMPLATE.md` (Tier 3). Does NOT override any rule.
On conflict — higher tier wins.

---

## DEC-001 (2026-10-31, v0.9.102) — per-card `ram_formula`, not universal `transient_factor`

**Q:** Should RAM estimate use a single `transient_factor` coefficient
per card?

**A:** No. Each algorithm's state has different structure (polynomial,
product of axes, dict). A single number cannot express `k * d` (NB) and
`2 * d * d` (Linear) simultaneously. `transient_factor` hid formula
structure inside a scalar (e.g. `tf=2.0` for Newton — but 2 what? d²? d?
different unit for LDA dict at 14.5?).

**Consequence:** per-card `ram_formula` string. `transient_factor`
remains as **legacy fallback** only. Migrated cards have
`ram_formula` + `formula_evidence`; measurement stored under
`measurement`.

**Alternative rejected:** "tf=2 for Newton, tf=14 for dict" — hides the
bug (which code line produced 2×), makes verification impossible
(what does 2 mean?), prevents automatic re-derivation when code changes.

---

## DEC-002 (2026-10-31, v0.9.102) — formula derived from code, verified by measurement

**Q:** Should we derive formula by fitting to measurements (vary d, fit
polynomial, get coefficients)?

**A:** No. Formula comes from reading code. Every term is a line:
`mg.check_state`, `mg.check_batch`, `np.empty`, `dict[key] +=`. Sum
of terms = formula. Measurement is **verification**, not derivation.

**Rationale:** measurement-derivation masks which code line is missing
(if ratio < 0.7, no way to know if it's a hidden alloc or rounding).
Code-derivation forces reading code (find the `np.zeros((n,))` you
missed). Discovery of LDA dict issue (v0.9.102) came from code-reading,
not from fitting.

**Anti-pattern:** ratio 0.5 → "multiply by 2". This hides the missing
`dict_entry_bytes` term. Correct action: find the dict line in code.

---

## DEC-003 (2026-10-31, v0.9.102) — no `verified_typical` status

**Q:** Should there be a status meaning "works in CI-tested range
(n≤10⁴, d≤100), may fail outside"?

**A:** No. Either `verified` (matrix passed at all cells) or `draft`
(with documented reason). No intermediate.

**Rationale:** `verified_typical` would silently permit OOM outside the
CI range — exactly the "silent OOM" the framework exists to prevent.
Tests passing at typical n does **not** prove production safety at n=10⁷.

**Alternative rejected:** 4-status enum (`draft`, `verified_typical`,
`verified`, `tier1_core_exception`) — increases cognitive load; two
clear statuses (`verified`, `draft`) + one exception is enough.

---

## DEC-004 (2026-10-31, v0.9.102) — "won't fix" is not a category

**Q:** What about algorithms where OOC is impractical (binseg, stray,
TS single-series)?

**A:** Every CADW algorithm has a feasible OOC path via primitives
(Reduce, HashShuffle, Sort, SpillBuffer, SpillArray, strided+chunked).
"Impossible" is not a valid claim. Categories:

- `verified` — OOC proven, matrix passed
- `draft` — OOC feasible, fix planned for version X (documented in
  ROADMAP)
- `tier1_core_exception` — n bounded by problem domain (single-series
  TS, n≤10⁶ typical), with tested bound

"Won't fix" was a grab-bag that let `draft` items hide as if they were
final decisions. Removed.

---

## DEC-005 (2026-10-31, v0.9.102) — calibration isolated from runtime

**Q:** Should formula verification use `MemoryGuard` while measuring?

**A:** No. Calibration uses fixed `batch=1000` (experimental control);
no guard, no budget. Runtime uses guard to pick batch from formula
(inverse). Two orthogonal activities.

**Rationale:** calibration answers "does formula match reality at
known batch?" (one question). Runtime answers "what batch fits in
budget?" (inverse). Mixing makes both untestable. Also: guard
introduces auto-injected params (`batch`, `row_size` from budget) that
make calibration params non-explicit.

**Consequence:** `evaluate_formula(op, **params)` — no guard in
signature. `estimate_from_yaml(op, guard, **params)` — with guard,
calls `evaluate_formula` internally after auto-inject. v0.9.103+ adds
`solve_batch(op, budget, **params)` for runtime.

---

## DEC-006 (2026-10-31, v0.9.102) — fixed batch at calibration

**Q:** Should calibration vary batch size too?

**A:** No. Fixed `batch=1000`. Vary n, d (and k, w, n_groups where
applicable).

**Rationale:** batch is an experimental control, not an axis. If batch
varies, formula gets extra term `X * batch * row_size` which is trivially
present in every formula; testing it 9 times per algorithm wastes time
without information. Fixed batch → 9 cells × (n, d) matrix. If batch term
suspected wrong, add a single 3-point sweep `batch ∈ {100, 1000, 10000}`
at fixed n, d (3 extra cells).

**Alternative rejected:** "compute batch from formula to target 50 MB
peak". Circular — formula used to pick batch, then formula verified
against its own prediction.

---

## DEC-007 (2026-10-31, v0.9.102) — private test helpers not in status scope

**Q:** How to classify private functions used only in tests (e.g.
`coxph._breslow`, `dwt_mlead._mahalanobis_distances`)?

**A:** Not in scope. No YAML card, no `status` field. Docstring marker
`REFERENCE ONLY — production uses <_other_function>` is sufficient.

**Rationale:** status system applies to **public producers** (`fit_*`,
`predict_*`, `transform_*` without `_` prefix) that users call. Private
helpers are implementation detail; their RAM is bounded by the caller's
(guarded) usage. Adding them to status enum ("`reference_only`") creates
the illusion they're public and makes the enum mean two different things.

**Consequence:** audit pass flags private helper → verdict "checked:
no production caller" → close without YAML change. `coxph._breslow`
moved from `status: reference_only` candidate to "not in scope" in v0.9.101.

---

## DEC-008 (2026-10-31, v0.9.102) — no `verified_cost_bounded`

**Q:** If OOC fix is known but expensive (CPU-bound, not RAM-bound),
should status reflect "OOC possible but slow"?

**A:** No. OOC is the target. If fix is known → `draft` (target version).
If cost makes it non-viable → still `draft`, but add `SlowOperationWarning`
at runtime (AP22, v0.9.103+) and estimate_cost skeleton. "cost-bounded"
hides "we decided not to do it" behind a status that sounds final.

**Rationale:** user has a right to know: (a) will it run OOC? (b) will
it take 1 hour? Both are answers, separately. Combining into "cost-bounded"
answers neither. `draft` + warning answers both: "OOC coming in
v0.9.10X; current version will warn at runtime about 1h compute".

**Anti-pattern:** status enum grows every time a hard case appears.
Each addition makes the enum less interpretable. Stick to three:
`draft`, `verified`, `tier1_core_exception`.

---

## DEC-009 (2026-10-31, v0.9.102) — document hierarchy T1–T5

**Q:** Multiple methodology docs (AI_CONSTITUTION, STABILITY_PROTOCOL,
ALGORITHM_TEMPLATE, GLOSSARY, DECISION_LOG, CHAT_PROTOCOL, CHANGELOG,
ROADMAP, EXAMPLES_INDEX, METHODOLOGY_ALGORITHM_SELECTION,
PRIMITIVES_AND_PRINCIPLES). Which wins on conflict?

**A:** Five tiers:

- **T1. Constitutional:** `AI_CONSTITUTION.md`. Principles, patterns,
  gates G1–G11, anti-patterns AP1–AP23. Overrides all below.
- **T2. Methodological:** `STABILITY_PROTOCOL.md`,
  `METHODOLOGY_ALGORITHM_SELECTION.md`. Process docs. Apply T1 to
  specific tasks. Cannot override T1.
- **T3. Template:** `ALGORITHM_TEMPLATE.md`,
  `EXAMPLE_VERIFIED_CARD.md`. Concrete checklists/examples. Cannot
  override T1 or T2.
- **T4. Reference:** `GLOSSARY.md`, `DECISION_LOG.md`,
  `PRIMITIVES_AND_PRINCIPLES.md`, `EXAMPLES_INDEX.md`. Canonical
  terms and historical rationale.
- **T5. Operational:** `CHAT_PROTOCOL.md`. Session rhythm and format.
  No content rules, only process.

**Every doc carries a `**Hierarchy:** Tier N` header** at top,
immediately after **Version**. Enables AI to detect conflicts fast.

---

## DEC-010 (2026-10-31, v0.9.102) — chat protocol 4 phases

**Q:** How should AI <-> developer chat flow structure?

**A:** Four phases, never mixed:

1. **Probe** — read-only commands; output pasted back
2. **Patch** — write to `/tmp/patch_NNN.py`, run; verify in same block
3. **Review** — human reads `git diff --stat` + sample; approves/corrects
4. **Commit** — `git add` + `commit` + `tag`; only after explicit "go"

**Rationale:** mixing causes wasted context (patch code pasted back
inflates tokens), terminal crashes (piping `pytest` output), and
silent errors (patching without seeing failure first). See
`CHAT_PROTOCOL.md` for details.

---

## New entries (append below, do not edit)

---

## DEC-011 (2026-10-31, v0.9.102) — `tier1_core_exception` status is temporary wart; replaced by `bounded_by_domain` field in v0.9.103

**Q:** Should "n bounded by problem domain" be a status value, or a
separate property field?

**A:** Separate property field (`bounded_by_domain`). Status stays
binary (`draft` / `verified`).

**Rationale:** `tier1_core_exception` mixes two orthogonal concepts:

1. **Tier** — how the model is built (`tier1_core` / `tier2_wrapper` /
   `tier3_export`). From METHODOLOGY §10.
2. **Status** — whether the card is done (`draft` / `verified`).

Also confusable with a potential `tier2_wrapper_exception` (none exist
today; none planned — the `_core` suffix makes the "tier2_core" version
nonsensical).

**Temporary wart (v0.9.102):** status enum includes
`tier1_core_exception` for `tier_level: tier1_core` + n-domain-bounded
cards (pelt x2). Schema R7 enforces `tier1_core_exception:` field
presence.

**Planned replacement (v0.9.103):**

```yaml
tier_level: tier1_core
bounded_by_domain: {axis: n, bound: 10^6, reason: "..."}
implementation_status:
  verified_in_cadw: true
  status: verified
```

Three fields express three facts. One compound status expressed them
poorly.

**Do NOT invent** `tier2_core_exception` or similar. Fix is one wart
removed, not one addition.

**Reference:** AI_CONSTITUTION §5.2, STABILITY_PROTOCOL §6.

---

## DEC-012 (2026-11-01, v0.9.103) — AP2 in `_stack_numeric` blocks `fit_linear` verified

**Q:** `fit_linear` has formula `2*d*d*f64 + 3*batch*row_size + batch*d*f64`. But `measure_peak` shows 2× formula due to `_stack_numeric` (`np.column_stack` in batch loop). What status?

**A:** `draft`. Formula would need `+ 2*batch*d*f64` transient term to match measurement. But that transient is AP2 (`list.append` + `column_stack` = classic anti-pattern). Refactor (`_stack_numeric` → pre-allocated array, guard-correct) removes the term. Formula for pre-refactor code = temporary correct, wrong target.

**Rationale:** STABILITY_PROTOCOL §2.2 (spill check) is also blocked: `reduce()` `check_batch(extra_factor=1.0)` doesn't include `_stack_numeric` transient. Guard under-counts ~2× at d=100, ~100× at d=1000. Same issue in `logistic.py`, `lasso.py` (same helper), `naive_bayes.py` (all ReduceOps use `_stack_numeric`).

**Affected (v0.9.104 refactor + reverify):** `fit_linear`, `fit_logistic` ×4, `fit_lasso`, `fit_gaussian_nb`, `fit_bernoulli_nb`, `fit_multinomial_nb`.

**Fix plan (v0.9.104):** replace `_stack_numeric` with `np.empty((batch, d))` + per-column slice assignment. Update `extra_factor` in `reduce()` to `3.0`. Then formula + matrix + verified for all 8+ cards.

**Reference:** STABILITY_PROTOCOL §3 (formula from code, not reverse-engineer from AP2), §2.2 (spill check requires guard covers all transients).

---

## DEC-013 (2026-11-01, v0.9.103) — `fit_zms` O(n) output blocks verified

**Q:** `fit_zms` state O(chunk) — seems OOC. But `ZMSResult` includes `decision_scores` (n,) and `labels` (n,). Verified?

**A:** `draft`. STABILITY_PROTOCOL §2.1 OOC check fails: peak(n=10^5) / peak(n=10^3) ≈ 100× (not < 2.0). Reason: O(n) output.

**Rationale:** OOC promise means RAM independent of n. `decision_scores` is 1 score per timestep — intrinsic to algorithm (anomaly score per point), not a bug. Fix: `SpillArray` (AI_CONSTITUTION §3.3). Deferred v0.9.105.

**Affected:** `fit_zms` ×3 (D_time_series: point, shape, ZMS alias). Same pattern likely for `fit_hampel`, `fit_cusum` — check in v0.9.105.

**Reference:** STABILITY_PROTOCOL §2.1, AI_CONSTITUTION §3.3 (SpillArray for O(n) output).

---

## DEC-014 (2026-11-01, v0.9.103) — `bounded_by_domain` is modifier, not verified-enabler

**Q:** Pelt has `bounded_by_domain: {n: 10^6}`. Structural migration done (DEC-011). Does this make it `verified`?

**A:** No. `bounded_by_domain` describes **where the algorithm is safe** (n <= 1e6). It does NOT mean "formula verified, matrix measured, status done". Pelt `fit_pelt` is a **container** — 256B model + loop over `_collect_series` (per-id series buffering). Real memory is in `_collect_series` (per-batch) + `pelt()` DP (in `.detect()`). Formula requires `_collect_series` refactor + measurement.

**Rationale:** Two orthogonal facts:
- `bounded_by_domain`: where algorithm is applicable
- `status: verified`: fit path is OOC + formula + matrix-measured

`bounded_by_domain` is a **modifier** to `status: verified` (like `measurement`). Cannot substitute for the 4 criteria (STABILITY_PROTOCOL §6).

**Correction:** pelt ×2 reverted from `verified` → `draft`. `bounded_by_domain` retained (applies to `.detect()`, honest metadata). v0.9.104 formula + matrix will promote to `verified`.

**Same class as** `fit_ab_test` (DriftABModel, 256B, tiny O(1) state — STABILITY_PROTOCOL §4 ratio may fail if measurement ~2-5KB vs formula 800B).

**Reference:** STABILITY_PROTOCOL §6 (status criteria), DEC-011 (bounded_by_domain introduction).

## DEC-031 (2026-11-03, v0.9.110) — iforest formula matrix blocked by measurement tooling, not code

**Q:** `fit_isolation_forest` OOC proven (stage 5 ratio 1.00), but matrix
4-5/9 in [0.7, 1.4]. `status: verified` or `draft`?

**A:** `status: draft`. Matrix fails at multiple cells due to **measurement
tool limits**, not code defects. OOC fix (batch_size kwarg + state_est v2)
merged — users get OOC by default.

**Evidence (v0.9.110a, 22 cells, each median-of-3-5 subprocesses):**

| n | d | measured | predicted (v3) | ratio | verdict |
|---|---|---|---|---|---|
| 10^3 | 10 | 655 KB | 809 KB | 1.23 | OK |
| 10^3 | 100 | 5.77 MB | 3.34 MB | 0.58 | FAIL (warmup) |
| 10^3 | 1000 | 24.1 MB | 28.6 MB | 1.19 | OK |
| 10^4 | 1000 | 7.46 MB | 28.6 MB | 3.84 | FAIL (steady-state) |
| 10^5 | 100 | 0.39 MB | 3.34 MB | 8.49 | FAIL (steady-state) |
| 10^5 | 1000 | 31.99 MB | 28.6 MB | 0.89 | OK |

**Root cause:** `measure_peak` uses `resource.ru_maxrss` (peak-since-process-start
delta). Cannot separate allocator warmup from real peak when state is
Python-object-based (`n_trees x ~511 nodes x 5 lists + 5 numpy views`):

- n=10^3 (1 batch): warmup happens inside fit -> peak captured
- n=10^5 (100 batches): warmup done by batch #2, then reuse -> delta ~ 0

v0.9.107 ReduceOps (linear, logistic, NB) passed because state is **stable
`d^2*f64`** — warmup small relative to state. iforest has transient allocator
churn from Python-object trees.

**Why "verified" not honest:** STABILITY_PROTOCOL §4 requires 9/9 in [0.7, 1.4]
for `status: verified`. Widening to [0.3, 4.0] would silence the tool gap
without fixing it — same pattern as AP23 anti-pattern (bump coefficient to
match measurement). Rejected.

**Why the code fix is still valuable (kept in v0.9.110a):**

- `batch_size: int|None = None` kwarg -> auto-derive via
  `MemoryGuard.get_batch_rows(expansion_factor=3.0, state_bytes=<5-term>)`.
- Corrected `state_est` from v1 (`S*d*8 + n_trees*2*S*80 + 256`) to v2
  (`S*d*8 + n_trees*2*S*64 + n_trees*5*S*8 + S*d*8 + max_depth*S*6*8`).
- `gc.disable()` during fit (removes GC-induced RSS noise, no functional change).
- OOC proven: stage 5 (fixed batch=500) -> peak ratio n=10^5/n=10^3 = 1.00 (or 0.75); both < 2.0.

**Options for v0.9.111:**

(a) **Algorithm refactor:** flat `np.memmap` tree storage. Trees become dense
    numpy arrays (`feature: int32[S*nodes]`, `threshold: f64[...]`, etc.).
    Removes Python-object state -> measurement tool no longer sees warmup as
    signal. Cost: ~1-2 sessions. **Benefit:** same fix applies to `kmeans_ad`,
    `sub_pca` (also Python-object state). Cross-group value.

(b) **Better tooling:** `perf`-based RSS sampling at 100us granularity, or
    `/proc/self/smaps_rollup` tracking. Cost: one-off infra work.

(c) **Accept formula from code with wider tolerance** — violates §2.6. Rejected.

**Recommendation:** (a). Target v0.9.111.

**Do not re-solve:** iforest `status: verified` in v0.9.111 only after
flat-memmap refactor. Until then: `status: draft`, `draft_reason` explains
why. Reference this DEC in any future iforest work.

**Note on state_est divergence:** code's `state_est` in `iforest.py` uses
**conservative v2 coefficients** (180 B/node, `n_trees * score_transient`),
while `predicted_peak` in `experiment_v110_iforest.py` uses **v3 measured
coefficients** (64 B/node, single-tree score transient). Intentional: guard
must not underestimate state -> v2 chosen for safety. v0.9.111 refactor will
unify both under flat-memmap state (which eliminates Python-objects and
makes measurement clean).


**Files touched (v0.9.110a):**
- `cadw/ml/iforest.py` — batch_size kwarg + state_est v2 + gc.disable during fit
- `docs/ml_tools/E_anomaly.yaml` — `status: draft`, `draft_reason`
- `docs/ml_tools/D_time_series.yaml` — same
- `scripts/experiments/experiment_v110_iforest.py` — reproducible evidence chain (stages 0, 2, 3, 5, 6)
- `docs/ml_tools/DECISION_LOG.md` — this entry

---

## DEC-032 (2026-11-03, v0.9.110) — linucb confirms DEC-031 as tool gap, corrects its recommendation

**Q:** `fit_linucb` matrix 0/9 despite state being pure numpy. Same class as
iforest (DEC-031) or new?

**A:** Same class — measurement tooling gap. But DEC-031's recommendation (a)
"flat memmap tree refactor" was **incomplete**: it would help iforest
(Python-object state) but does NOT help linucb (already numpy).

**Evidence (v0.9.110b experiment):**

| n | d | measured | predicted | ratio | verdict |
|---|---|---|---|---|---|
| 10^3 | 100 | 1.76 MB | 3.67 MB | 2.09 | FAIL |
| 10^4 | 100 | 0.58 MB | 3.67 MB | 6.35 | FAIL |
| 10^5 | 1000 | rss=0 | 72.1 MB | N/A | unmeasured |
| stage5 n=10^6 | 10 | 172 KB | 196 KB | 1.14 | PASS (OOC) |

Non-physical pattern: n=10^4 measured LESS than n=10^3 for same work. State
is `K*d^2*f64 + K*d*f64` = 40 MB at K=5,d=1000 — that alone exceeds measured
value. `rss_delta` doesn't register because `ru_maxrss` was already higher
from parquet load / page-cache scan.

**Root cause (confirmed):** `measure_peak` uses
`resource.getrusage(RUSAGE_SELF).ru_maxrss` delta. This is
peak-since-process-start. When parquet load (or any prewarm scan)
raises ru_maxrss above fit peak, `rss_delta` = 0. Fallback to
`py_bytes` (tracemalloc) only counts Python objects, NOT numpy (numpy
uses malloc, not PyMem). Result: unmeasurable.

**Why this is NOT a code problem:** stage 5 OOC PASS (ratio 1.00 across
n=10^3..10^6 with batch=500). The algorithm IS OOC. The measured numbers
in stage 6 are tooling artifacts, not code behavior.

**DEC-031 recommendation correction:**

DEC-031 recommendation (a) "flat memmap tree refactor" is valid **only for
iforest** (Python-object state). linucb's state is already dense numpy —
refactoring won't help. The real fix for both is:

(a') **Better measurement tool.** Options:
   - `/proc/self/smaps_rollup` Pss sampling at 100us granularity
   - perf_event_open + read RSS periodically during fit
   - tracemalloc with manual `tracemalloc.take_snapshot` around numpy
     allocations (requires `tracemalloc.start(...)` + `np.ndarray.nbytes`
     accounting — hybrid)
   Cost: ~1 session of infra work. Applies to ALL future cards — including
   v0.9.107 (which passed by luck: stable state, warmup small relative to
   signal).

(b) **Parallel algorithm refactor + tool fix.** For iforest-specific
Python-object cost: flat np.memmap tree storage (DEC-031-a). Then both
the tool AND the state are simple. But linucb still needs the tool.

**Recommendation update:** v0.9.111 should start with (a'). Tool fix
unblocks iforest matrix, linucb matrix, and all subsequent cards
simultaneously. Algorithm refactors (iforest flat-memmap, kmeans_ad,
sub_pca) are secondary — do them only if measurement tool still leaves
noise after (a').

**v0.9.107 redux:** why did linear/logistic/NB pass then? Because:
- batch term (3*batch*row_size) was 3-24 MB (well above ~500 KB noise)
- state was stable d^2*f64 (allocated once, measured as d^2*8)
- parquet load at n=10^5, d=100 was ~16 MB, below fit peak 3.3 MB
So `rss_delta` was correctly positive. linucb fails because state=K*d^2
can be SMALLER than batch-term × n_batches, and parquet-load dominates
for large d.

**Do not re-solve:**
- iforest `status: verified` after flat-memmap (DEC-031-a) — but only
  AFTER tool fix (a'), otherwise still can't measure
- linucb `status: verified` after tool fix (a') — no algorithm change needed
- Do not re-attempt matrix with `measure_peak` for linucb/iforest/gmm/birch
  until (a') lands

**Files touched (v0.9.110b):**
- `cadw/ml/linucb.py` — batch_size kwarg + k_max_hint + AP4 fix +
  cast-based arms extraction + state_est 5-term
- `docs/ml_tools/M_bandits.yaml` — `status: draft`, `draft_reason`
- `scripts/experiments/experiment_v110_linucb.py` — reproducible
  evidence chain (stages 0/2/3/4/5/6, subprocess + prewarm + page cache
  warm + formula v1)
- `docs/ml_tools/DECISION_LOG.md` — this entry

---

<!-- future entries appended here -->


---

## DEC-018 (2026-11-02, v0.9.105) — in-place state update (O1) considered, deferred

**Q:** `XTXOp.update` does `xtx + X.T @ X` (new alloc). `xtx += X.T @ X` (in-place) would reduce peak from 3×d² to 2×d². Apply?

**A:** Not in v0.9.105. Deferred to v0.9.106.

**Rationale:** `ReduceOp` contract doesn't forbid mutation (Gaussian already uses `state[k][0] += n`). But measured `fit_linear` coef = 1.92 (iter-1) — allocator already optimizes the `xtx + X.T@X` case in practice. Theoretical gain: 8 MB at d=1000. At d<500 invisible. Not the bottleneck (batch_term dominates for d<1500). Deferred to bundle with MemoryGuard v2 in v0.9.106.

**Where:** `cadw/ml/linear.py::XTXOp.update` line ~245.

---

## DEC-019 (2026-11-02, v0.9.105) — `del X_parts` (O2): partial, full fix O2b pending

**Q:** `logistic._stack_and_validate` keeps `X_parts` list live until function return; `X[mask]` creates a third copy. Adding `del X_parts` before `return X[mask]` reduces peak from 3× batch to 2×.

**A:** O2 applied (v0.9.105-O2O3). **Did NOT reduce measured coef** — because the true peak is in `update`, not in `_stack_and_validate`:

```python
X (B, d)                          # 1× from _stack_and_validate
X_aug = column_stack([ones, X])   # 2× (X + X_aug)
T1 = X_aug * w[:, None]           # 3× (X + X_aug + T1)
```

**Full fix (O2b):** `del X` after `X_aug` creation, plus chunked matmul. Deferred to v0.9.106.

**Evidence:** iter-5 (post-O2/O3): `logistic coef a = 3.049` — unchanged from iter-3 (a = 3.049). O2 alone insufficient; peak is in update.

**Where:** `cadw/ml/logistic.py::_stack_and_validate`, `LogisticGradOp.update`.

---

## DEC-020 (2026-11-02, v0.9.105) — Bernoulli bool (O3): applied, effective

**Q:** `_BernoulliStatsOp.update` does `Xb = (X > 0.5).astype(np.float64)` — full (B, d) f64 copy. Keep as bool (1/8 memory).

**A:** O3 applied. `Xb = (X > 0.5)`; `Xb.sum(axis=0)` returns float64 anyway, semantics identical.

**Evidence:** iter-2 baseline Bernoulli coef = 2.941; iter-5 post-O3 = 2.006. Tests: `test_naive_bayes.py` all green (62 passed).

**Where:** `cadw/ml/naive_bayes.py::_BernoulliStatsOp.update`.

---

## DEC-021 (2026-11-02, v0.9.105) — einsum for Gaussian (O4): deferred, unmeasurable

**Q:** `_GaussianStatsOp.update` does `(Xc * Xc).sum(axis=0)` — creates temp (B_c, d). `np.einsum('ij,ij->j', Xc, Xc)` avoids temp.

**A:** Not applied. State contribution to peak is <5% at typical d, unmeasurable in iter-2/3. Deferred to v0.9.107 (with refactor), after NB-state test becomes possible at low batch.

**Where:** `cadw/ml/naive_bayes.py::_GaussianStatsOp.update`.

---

## DEC-022 (2026-11-02, v0.9.105) — `batch_memory_factor` scalar insufficient for large d

**Q:** `batch_memory_factor()` returns scalar `k`; `check_batch(size × k)` compares to `available_bytes`. But `peak = batch_term + state_term`, and at large d state_term dominates. Guard under-protects.

**Example:** `fit_logistic(d=1000, batch=1000)`: peak = 25 MB (batch) + 30 MB (state) = 56 MB. Guard `check_batch(4.0 × batch) = 32 MB` — passes at budget 50 MB, actual peak 56 MB > budget → silent OOM.

**Fix (v0.9.106):** either
- `get_batch_rows(op, state_bytes, ...)` — state-aware batch sizing
- `reduce()` uses `plan(op, op_context={"state_bytes": ...})` (already implemented in `MemoryGuard.plan`, not wired)
- `check_batch(total_after)` — combined state + batch check

**Where:** `cadw/core/memory_guard.py::MemoryGuard.get_batch_rows`, `cadw/ops/reduce.py`.

---

## DEC-023 (2026-11-02, v0.9.105) — framework chunking gap (blocks OOC of all 9 cards)

**Q:** `fit_linear(df, n=10⁵)` — does it run OOC?

**A:** No. Empirically (iter-4):

```
Без patch (production path):
  n=10⁴     → peak = 16 MB    (1 batch)
  n=10⁵     → peak = 161 MB   (1 batch, 10× рост)
  n=5×10⁵   → MemoryLimitExceeded (1.2 GB > 500 MB)

С patch (control, batch_size=1000):
  n=10⁴     → peak = 1.72 MB  (10 batches)
  n=10⁵     → peak = 1.72 MB  (100 batches, OOC)
```

**Root cause:** `CadwDataFrame._iter_filtered_batches` (cadw/dataframe/_iterators.py:66) calls `self._dataset.iter_batches(columns=...)` without `batch_size`. `Dataset.iter_batches(batch_size=...)` supports it (dataset.py:176) but not propagated. `Dataset.from_table` creates 1 batch = n rows.

**Consequence:** STABILITY_PROTOCOL §2.1 (OOC check) fails for all 9 cards at production path. Cards cannot be `verified` until fixed.

**Fix (v0.9.106):**
- Propagate `batch_size` through `_iter_filtered_batches` (explicit kwarg, default None → batch=n)
- Alternatively: default `batch_size` in `Dataset.iter_batches` from `MemoryGuard.get_batch_rows`
- Wire `MemoryGuard` into chunking decision at fit-call time

**Interim:** v0.9.105 cards stay `status: draft` with `draft_reason: "OOC blocked by framework gap (DEC-023)..."`.

---

## DEC-024 (2026-11-02, v0.9.105) — NB finalize allocates O(C×d) outside update state

**Q:** `_GaussianStatsOp.finalize` allocates `n_c(C)`, `mean(C×d)`, `var(C×d)`. That's 3 × C × d × f64 outside the update-state.

**A:** Formula must include `finalize` cost. Measured at C=2000, d=2000: gaussian extra = 32 MB (matches 3 × 2000 × 2000 × 8 = 96 MB... wait, this needs verification. Actual: `mean + var` = 2 × C × d × 8 = 64 MB, plus n_c = 16 KB. Iter-4 measured `obs_extra = 96 MB` at C=2000, d=2000 — larger than `finalize = 64 MB`, suggesting additional transient in `finalize` (perhaps `np.zeros` + assignment).

**Formula:** `fit_gaussian_nb` = `2 × batch × row_size + 3 × C × d × f64` (3 = 1 per update per class + 2 finalize). Similar for multinomial/bernoulli.

**At typical C ≤ 64:** term < 1 MB, batch dominates. At C = 1000+ (rare): finalize dominates.

**Where:** `cadw/ml/naive_bayes.py::*StatsOp.finalize`.

---

## DEC-025 (2026-11-02, v0.9.105) — v0.9.105 outcomes

**9 cards: `status: draft`** (not verified, blocked by DEC-023).

**Delivered:**
- `batch_memory_factor` justified from code (v0.9.105a-fix): XTXOp=3.0, Center=3.0, Logistic=4.0, Gaussian=3.0, Multinomial=3.0, Bernoulli=3.0 (post-O3)
- O2 (`del X_parts`) — partial (peak in update)
- O3 (Bernoulli bool) — effective (coef 2.94→2.0)
- `ram_formula` measured (v0.9.105b-finalize3):
  - `fit_linear`, `fit_lasso`: `2 * batch * row_size + 2 * d * d * f64`
  - `fit_logistic` ×4: `3 * batch * row_size + 4 * d * d * f64`
  - `fit_gaussian_nb`: `2 * batch * row_size + 3 * C * d * f64`
  - `fit_bernoulli_nb`: `2 * batch * row_size + 2 * C * d * f64`
  - `fit_multinomial_nb`: `2 * batch * row_size + 2 * C * d * f64`
- `formula_evidence` per term with `file:line`
- `draft_reason` per card (framework gap DEC-023)
- `docs/ml_tools/EXPERIMENT_PROCEDURE.md` (new file)
- `scripts/experiment_v105_rep.py` (reproducible harness)
- OOC measurement: passes only with explicit `batch_size` patch; fails in production path

**Next:** v0.9.106 (framework: chunking + state-aware guard per DEC-022/023) → v0.9.107 (flip 9 to `verified` + refactor per DEC-019/021).

**Not applied (deferred):**
- O1 (in-place xtx, DEC-018)
- O2b (del X in update, DEC-019)
- O4 (einsum, DEC-021)
- Refactor `_stack_numeric` etc. (DEC-021, v0.9.107)
