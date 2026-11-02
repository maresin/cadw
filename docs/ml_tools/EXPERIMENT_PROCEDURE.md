# Experiment Procedure — formula calibration for CADW ops

**Version:** 1.0 (2026-11-02, v0.9.105)
**Purpose:** reproducible method to derive and verify `ram_formula` for
any `fit_*` / `predict_*` / `transform_*` before `status: verified`.
**Hierarchy:** Tier 2 (methodological). Complements STABILITY_PROTOCOL.md.

---

## §1. Scope

Applies to all CADW public producers in `cadw/ml/`, `cadw/ts/`,
`cadw/ops/` that will get a YAML card with `status: verified`.

**Prerequisites:**
- Read AI_CONSTITUTION.md §3 (patterns), §5 (AP23)
- Read STABILITY_PROTOCOL.md §1-§4 (formula from code, not from measurement)
- Card exists with `verified_in_cadw: true` (fidelity to source)

**Not in scope:** algorithmic correctness (that's §2 of STABILITY_PROTOCOL).

---

## §2. Choosing representatives

For a **group** of similar ops (e.g. "6 ReduceOps with state O(d²) or O(C·d)"),
pick 3-4 representatives covering all structural variants:

| Variant | Representative | Why |
|---|---|---|
| single-pass, state O(d²) | `fit_linear` | base case |
| single-pass, parametrized state | `fit_gaussian_nb` | C-term in formula |
| multi-pass, state O(d²) | `fit_logistic` | iteration independence |
| already verified | `fit_matrix_profile` | **control**: reproduce known formula |

If control fails — procedure itself is broken. Fix procedure first.

---

## §3. Six steps

### Step 0 — Instrument calibration

Run the smallest possible op 3 times. Median. Assert `spread < 20%`.
If larger — `measure_peak` noisy at this scale; increase magnitudes.

Purpose: baseline / sanity. Cheap (seconds). Prevents wrong conclusions
from noisy data.

### Step 1 — Code-reading → hypothesis

Read `op.update()` (or manual loop) fully. **Write down every allocation**
with `file:line`, size, and lifetime. Identify **concurrent-live** arrays
at each program point. Peak = max sum of live sizes.

**Output:** hypothesis formula:
```
peak = <coef_batch> × batch × row_size + <coef_state> × <state_expr>(d, C)
```

Each term must point to a specific line. **No fudge coefficients**
(AP23 — no "ratio is 0.5, make it 2.0").

### Step 2 — Exp 1: batch-sweep → `coef_batch`

Fix `d` small (e.g. 10 or 20) so state contributes <5%.
Vary `batch ∈ {1000, 2000, 4000, 8000}` (each = one batch, n=batch).
Budget huge (1000 MB); no memory limit in play.

Fit `peak(batch)` linear. `coef_batch = slope / row_size`.
Assert intercept ≈ state(d).

**Pitfall:** if state at small `d` is too large to ignore, coef biased
low. Check intercept consistency.

### Step 3 — Exp 2: d-sweep → `coef_state`

Fix `batch=1000`, `n=batch` (single batch).
Vary `d ∈ {100, 500, 1000}` (enough points for 2×2 solve or polyfit).
Budget huge.

**Two methods:**
- **(a) Direct state:** after `fit`, run `op.update` in a loop on prepared
  state; sum `state_nbytes` manually. Compare with `finalize` allocation
  separately (DEC-024 pattern). Gives exact state as ground truth.
- **(b) Decomposition:** if formula shape is `a×batch_term + b×state_term(d)`,
  solve 2×2 from smallest and largest d. Assert on intermediate d.

**Prefer (a)** — no fit assumptions. Use (b) to confirm a/b ratios.

### Step 4 — Exp 3: C-sweep (only if parametrized state)

For ops with `C` (classes) in state: `state(C) ∝ C × d`.
Fix `batch` small (so state is visible: `C × d × 8 > 2 × batch × row`).
Vary `C ∈ {2, 10, 30}`. Assert per-class slope matches code.

**Visibility threshold:** `C > 4 × batch` for the state to dominate.
Failing this → state unmeasurable, defer to a direct-state check.

### Step 5 — OOC check (final)

Fixed **budget** (not batch). `n ∈ {10³, 10⁴, 10⁵}` at the same budget.
Assert `peak(n=10⁵) / peak(n=10³) < 1.2`.

**Fails if framework does not chunk** (DEC-023 class): peak grows with n.
That is not a formula problem — it's a framework gap; see §6.

### Step 6 — Output

- `ram_formula` string (measured coefficients, `formula_evidence` per term)
- `guard_factor` (upper bound from code, may exceed measured formula)
- `measurement: {matrix, cells_passed, date, env}` in YAML
- All issues either fixed or DEC-scheduled

---

## §4. Rules

### R1. Fix-or-schedule (no indefinite backlog)

Every issue found during experiment is either:
- **Fixed** in the same session (≤30 min fix, tested)
- **DEC with explicit target version** (`v0.9.10X`)

**Forbidden:** adding to `.pingis/drafts/BACKLOG_ML.md` without target
version. Silent "we'll get to it" is a corruption of the framework
(AP21: no verified with open backlog; the corollary: no experiment
closes with open silent backlog).

### R2. Duplicate functions → DEC, not immediate refactor

If experiment reveals duplicated code (`_stack_numeric` ×5,
`_probe_memory_limit_mb` ×15), record in DEC with target version
(v0.9.107 currently). Do **not** refactor during experiment — mixes
concerns, breaks validation.

Exception: if the duplicate causes **semantic inconsistency** between
groups (e.g. `_stack_numeric` in logistic vs linear gives 3× vs 2×
batch peak), fix **both/all** copies synchronously in the same session.
O2 applied to `_stack_and_validate` only was fine because
`_stack_numeric` doesn't have the same transient.

### R3. Comments: in YAML, not in code (long term)

`formula_evidence` in YAML is **canonical**. Code gets short comment:
`# see DEC-XXX` or `# see formula_evidence in YAML card`.

Rationale: comment in code drifts when formula changes; YAML card is
under schema test.

In experiment scripts, **verbose** comments are correct — they're
explanations of the derivation, not permanent artifacts.

### R4. When formula ≠ measurement — find missing term, not bump coefficient

Per AP23. If ratio < 0.7 (formula under-estimates), **read code again**.
The missing allocation is a line. Add term. Re-measure.

If ratio > 1.4 (formula over-estimates by 40%), that's usually acceptable
(conservative formula is OK). Document `max_ratio_deviation` in YAML.

**Never:** "ratio is 0.5, let's make tf=2.0". That hides a bug.

---

## §5. Exit criterion (definition of done)

An experiment closes for a given algorithm when:

1. **Formula derived** from code (§3 step 1) and confirmed by measurement
   (§3 steps 2-4)
2. **OOC tested** (§3 step 5) — either passes, or explicitly blocked by
   a documented framework gap with target version
3. **Every issue found** is either `fixed` or `DEC with target version`
4. **No "I don't understand"** — every anomaly is either explained
   or explicitly deferred (with reason in DEC)
5. **Card written** with `ram_formula`, `formula_evidence`, `measurement`
6. **Status decided**: `verified` (all above pass) or `draft` with
   `draft_reason:` field

Without exit criterion, "one more iteration" is infinite. Force closure.

---

## §6. Blockers (framework gaps)

If experiment reveals a framework bug (e.g. DEC-023 chunking gap,
DEC-022 scalar guard factor), the experiment is **not abandoned**. Instead:

1. Record as DEC with target version (v0.9.106 typically)
2. Card stays `status: draft` with `draft_reason: "<description> (<DEC ref>)"`
3. Experiment **closes** — next iteration is a *different* session
   that addresses the framework gap
4. Do **not** "wait for framework" — record the finding, move to next
   algorithm in the group. The whole group may be blocked by the same gap.

**Rule:** if one card in a group is blocked by framework gap,
check if all others in the group are blocked by the same gap.
If yes — record one DEC for the whole group, close all as `draft`
with the same `draft_reason`.

---

## §7. Reference: v0.9.105 example

**Group:** 6 ReduceOps (linear, lasso, logistic ×4, NB ×3).

**Representatives chosen:**
- P1 `fit_linear` (batch + d² state)
- P2 `fit_gaussian_nb` (batch + C·d state)
- P3 `fit_logistic` (batch + d² + multi-pass)
- P4 `fit_matrix_profile` (**control** — already verified)

**Findings:**
- Linear coef=1.92, Logistic (a,b)=(3.05, 3.97), NB coef=2.0
  → formula: `2 × batch × row + 2 × d² × f64` for linear
- Control P4 reproduced within ±21% (3/3 in [0.7, 1.4])
- OOC: **fails without framework patch** (DEC-023)
- Result: 9 cards → `status: draft`, formulas final, guard factors justified
- v0.9.106 fixes framework, v0.9.107 flips to `verified`

Full evidence: `.pingis/audits/` + DEC-018..025.

---

## §8. Anti-patterns (from v0.9.105 real mistakes)

### A1. Fitting coefficient to measurement (my iter-1 mistake)

Proposed `2.1 / 3.5 / 2.2` from iter-1 numbers without code derivation.
**Rejected** by developer. Correct: derive from code (iter-5 confirmed
`del X_parts` alone insufficient; true peak in update — code reading).

### A2. Fixed batch instead of budget (my iter-1 mistake)

`measure_peak` at fixed `batch=1000` — but production chooses batch
from budget. **Correction:** fix budget, let guard/Dataset choose batch
(DEC-023 landed only because we ran this test).

### A3. 2-point fit for `(a, b)` without intermediate points

Iter-2 used `d=50, d=1000` to solve `(a,b)`, got `a=3.2, b=3.8`.
Iter-3 with `d=100, 500, 1000` got `a=3.05, b=3.97` — closer to theory.
**Lesson:** always verify on intermediate points.

### A4. Sanity test without baseline

Iter-1 measured `peak=872 B` on `materialize(100,10)` — good (baseline).
Without it, `measure_peak` noise (up to 50% at small scale) would
have been mistaken for signal. Step 0 is mandatory.

### A5. Comment `# draft_reason:` as YAML comment, not field

First finalize patch put `draft_reason` as YAML comment (indent 4). Schema
didn't detect, verify showed stale reason. **Correct:** `implementation_status.draft_reason:`
field at indent 6, detectable by `_load_cards()`.

---

## §9. Link to STABILITY_PROTOCOL

- **STABILITY §1**: two activities (calibration vs runtime)
- **STABILITY §2**: stability check (OOC, spill, backlog)
- **STABILITY §3**: formula from code (evidence per term)
- **STABILITY §4**: matrix measurement
- **STABILITY §5**: YAML card structure
- **STABILITY §6**: status decision

This PROCEDURE adds:
- **§3** steps: detailed DOE with numbers
- **§4** rules: fix-or-schedule, dedupe DEC, comments in YAML
- **§5** exit criterion
- **§6** framework-gap handling
- **§8** anti-patterns (concrete, from real mistakes)
