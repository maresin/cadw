# Stability Protocol and Formula Verification

**Version:** 1.1 (2026-11-03, v0.9.110)
**Status:** normative. Every `fit_*` / `predict_*` / `transform_*` in
`cadw/ml/`, `cadw/ts/` MUST follow this protocol before
`status: verified`.
**Companions:** `AI_CONSTITUTION.md` §5 (AP21, AP22, AP23),
`GLOSSARY.md`, `ALGORITHM_TEMPLATE.md`, `EXAMPLE_VERIFIED_CARD.md`,
`EXPERIMENT_PROCEDURE.md` (detailed DOE, rules R1-R4, exit criterion,
framework-gap handling — see §8 checklist).

**Hierarchy:** Tier 2 (methodological). Complements
`AI_CONSTITUTION.md` (Tier 1). Does NOT override §1 principles,
§3 patterns, §5 anti-patterns. On conflict — `AI_CONSTITUTION.md`
wins. If you believe a rule here contradicts the constitution,
stop and ask: which should change?

---

## §1. Two orthogonal activities: calibration vs runtime

| | **Calibration** (this doc) | **Runtime** (uses formulas) |
|---|---|---|
| Purpose | Prove formula matches measurement | Pick batch size given budget |
| Input | Fixed batch=1000, explicit (n, d, k) | MemoryGuard budget + params |
| Guard | **Not used** | Primary consumer |
| Formula role | Hypothesis (from code) | Same formula, inverted |
| Measurement | `measure_peak()` in test | Not done at runtime |
| Module | `evaluate_formula(op, **params)` | `solve_batch(op, budget, **params)` (v0.9.103+) |
| Output | Predicted bytes | Chosen batch size |

**Rule:** calibration is isolated. No `MemoryGuard`, no budget, no
auto-inject. If `evaluate_formula` sees a guard, it's a defect.

**Rule:** runtime is inverse. `MemoryGuard` uses formula to pick
batch: `batch = solve(formula, budget)`. Not `batch = expansion_factor
heuristic` (deprecated v0.9.103+).

---

## §2. Step 1 — Stability check (MANDATORY before formula)

An algorithm is **stable** if and only if all three pass.

### §2.1. OOC check

```python
peak_n3 = measure_peak(fit_X, df_n=10**3, d=fixed)["total_bytes"]
peak_n5 = measure_peak(fit_X, df_n=10**5, d=fixed)["total_bytes"]
ratio = peak_n5 / max(1, peak_n3)
assert ratio < 2.0, f"peak grew {ratio:.1f}x over 100x n — NOT OOC"
```

Rationale: OOC means RAM independent of n. Ratio >= 2 means n-scaling
state (n-sized array, unguarded list, dict). Fix algorithm first.

### §2.2. Spill check

```python
with pytest.raises(MemoryLimitExceeded):
    fit_X(df_n=10**5, memory_guard=MemoryGuard(max_memory_mb=1))
```

Rationale: OOC promise means "refuse before OOM". Silent OOM (process
killed) instead of clean raise = not stable.

### §2.3. No open backlog

```bash
grep -rn '# BACKLOG:' cadw/ml/X.py            # must be empty
grep 'fit_X' .pingis/plan_v0.9.87_refactor.md # must not be `[ ]`
grep 'fit_X' docs/ROADMAP.md | grep -i deferred # must be empty
```

Rationale: algorithm will change -> formula will change -> do not write
formula for unstable algorithm.

### §2.4. If any fails

`status: draft`, with comment on which failed:

```yaml
implementation_status:
  verified_in_cadw: true
  status: draft
  # draft_reason: "not OOC (n-scaling dict at _build_vocab); v0.9.103 K-spill"
```

**Do not** write a formula for `status: draft`. Formula follows
stability, not precedes it.

---

### §2.5. Window bound check (v0.9.175, DEC-067 / DEC-068)

For models declared `n_scalability: bounded_by_window` (pelt, rolling, expanding, ewm,
ETS, Theta, SARIMA, VAR, ARIMA, GBDTForecaster), verify:

**(a) OOC in W**: at fixed `W`, varying `n` changes peak <2.0×. Peak depends on W, not n.

```python
peak_n3 = measure_peak(fit_X, df_n=10**3, window=W_fixed)["total_bytes"]
peak_n5 = measure_peak(fit_X, df_n=10**5, window=W_fixed)["total_bytes"]
assert peak_n5 / max(peak_n3, 1) < 2.0
```

**(b) W adjustment**: `window=None` at tight budget → W_eff auto-adjusts down,
emits `ApproximationInfo(type="window_adjusted")`. `peak(W_eff)` fits budget.

```python
mg = MemoryGuard(max_memory_mb=2)  # small
result = fit_X(df, window=None, memory_guard=mg)
assert result.approximation is not None
assert result.approximation[0].type == "window_adjusted"
assert result.approximation[0].effective <= result.approximation[0].requested or \
       result.approximation[0].requested is None
```

**(c) Approximation disclosed**: `predict_in_sample(df)` returns NaN for lost range
`[0, n-W_eff)`.

```python
pred = result.predict_in_sample(df)
assert pred[:n - W_eff].isna().all()  # or all-NaN
assert pred[n - W_eff:].notna().all()
```

**(d) Insufficient**: if `W_max < min_history(model)` (e.g., `2s+1` for ETS) →
`MemoryLimitExceeded` (insufficient_no_alt tier).

---

## §3. Step 2 — Formula from code

**Order is mandatory. Do not derive formula from measurement.**

Read `cadw/ml/X.py` line by line. Every memory-relevant line becomes
a term:

| Code construct | Formula term |
|---|---|
| `mg.check_batch(b, extra_factor=Y)` | `Y * batch * row_size` |
| `mg.check_state(a, name=...)` | `a` (must be in terms of params) |
| `np.empty((batch, d), f64)` | `batch * d * f64` |
| `np.zeros((d, d), f64)` | `d * d * f64` |
| `np.empty((k, nodes), f64)` | `k * nodes * f64` |
| `dict[key] += 1` (K entries) | `K * dict_entry_bytes` |
| `list.append` of py objects | `N * py_object_base_bytes` |
| `b.column(c).to_numpy()` | `batch * <dtype>_bytes` |
| `b.column(c).to_pylist()` | `batch * (py_object_base_bytes + avg_elem_bytes)` |

Sum the terms. **That is the formula.** Nothing more, nothing less.

**Do NOT add invented components:**

- Do NOT add "transient_peak" — if the alloc isn't in code, it doesn't exist
- Do NOT add "overhead_factor" — use typed constants
- Do NOT add "fudge_coefficient" — if formula and measurement disagree, find
  the missing code line

**Write evidence per term:**

```yaml
ram_formula: "2 * d * d * f64 + 3 * batch * row_size + batch * d * f64"
formula_evidence:
  d_squared: "linear.py:142 -- mg.check_state(2*d*d, name='linear.xtx')"
  batch_input: "linear.py:88 -- mg.check_batch(b, extra_factor=3.0)"
  batch_transient: "linear.py:145 -- np.empty((b.num_rows, d), f64)"
```

Next session can verify by `sed -n '142p' cadw/ml/linear.py`.

---

## §4. Step 3 — Matrix measurement

**Fixed batch. Vary other axes.** Batch is experimental control — its
value chosen so that peak is approximately 50 MB.

### §4.1. Matrix

For algorithms with single primary axis:

```
n in {10^3, 10^4, 10^5}    (3 values)
d in {10, 100, 1000}       (3 values, or skip if d not in formula)
= 9 cells
```

For algorithms with secondary axis (k, w, n_groups):

```
Add that axis with 2 values (small, large) -> 18 cells
```

### §4.1.1. Component-wise mode (DEC-033, v0.9.110)

When **combined RSS matrix** (§4.1) is unmeasurable because
`baseline >> fit peak` — Python objects + Arrow buffers + parquet mmap
already exceed the fit's peak before fit starts — §4.1 is **not
achievable**. This is a physical limit of RSS-delta measurement, not a
formula bug. See DEC-033 for the 16-iteration investigation across
`measure_peak`, `measure_peak_pss`, `measure_peak_traced`, malloc_trim,
fadvise.

**Detection:** `measure_peak(fit)["rss_bytes"]` returns 0 or <5% of
formula prediction at d>=500 while formula > baseline noise.

**Component-wise verification** replaces combined matrix with four
orthogonal checks:

1. **State exact** — `model.state_nbytes()` must equal analytic formula
   (bit-level exact, no tolerance). Parametrize over K, d.
2. **State OOC** — `state_nbytes()` at n1, n2 (both >= subsample/estimators
   params) differ by <= 1.5x (bounded, not n-growing). For n < S, monotone.
3. **coef_batch isolated** — d<=200 (small baseline), sweep
   `batch in {1000, 2000, 4000}`, fit `peak = state + coef*batch*row_size`.
   `coef` must be physical (2-4 copies of row_size, per code inspection).
   RSS noise <= 15% (signal >= 5 MB > noise <= 1 MB).
4. **Formula sanity** — d<=200, batch=1000, peak 7-10 MB.
   `predicted / measured in [0.7, 1.4]`, median-of-3.

**OOC proof is unchanged:** fixed batch, vary n, peak ratio < 2.0.

**YAML fields:**

```yaml
measurement:
  verification_mode: "component-wise (DEC-033)"
  state_check: "test_state_nbytes_exact: model.state_nbytes() == <formula>"
  state_ooc: "test_state_bounded_for_n_ge_s: max/min <= 1.5x"
  coef_batch: "test_batch_coef_isolated: slope/row_size in [lo, hi]"
  formula_sanity: "test_formula_small_scale: predicted/measured in [0.7, 1.4]"
  ooc: "test_ooc_batch_fixed: peak ratio < 2.0"
  cells_passed: "8/8"
  runtime_sec: "~3"
  note: "Combined RSS matrix unmeasurable at large d. Component-wise per DEC-033."
```

`status: verified` is valid for component-wise **if all four checks pass**.
This is **not a relaxation** of §4: state is measured exactly (better than
±30% estimate), coef is decomposed by physical source, OOC is proven
independently.

**Applies to:** models where `state_nbytes()` is implementable and
state < batch*row_size*(few) — linucb, iforest, gmm, birch, kmeans_ad,
sub_pca, eif. Does NOT apply to: ReduceOps already verified via combined
matrix (linear, logistic, NB, randsvd) — those stay on §4.1.

**Runtime:** component-wise 2-3 seconds vs 15 min per attempt for RSS
matrix (which fails anyway).

### §4.2. Per-cell assertion

```python
for n, d in matrix:
    predicted = evaluate_formula(op, batch=1000, row_size=16, d=d)["bytes"]
    measured = measure_peak(fit_X, df_n=n, d=d)["total_bytes"]
    ratio = predicted / max(measured, 1.0)
    assert 0.7 <= ratio <= 1.4, (
        f"{op} n={n} d={d}: formula says {predicted/1e6:.2f}MB, "
        f"measured {measured/1e6:.2f}MB, ratio {ratio:.2f}"
    )
```

### §4.3. Failure handling

| ratio | Meaning | Action |
|---|---|---|
| < 0.7 | Formula under-estimates | **Find missing alloc in code.** Do not bump coefficient. |
| 0.7 -- 1.4 | Match | Accept cell |
| > 1.4 | Over-estimates | Usually acceptable (conservative). Document in `measurement.max_ratio_deviation`. |
| > 3.0 | Way over | Likely formula wrong. Re-check. |

**Failure example (real, LDA):**
Formula from code said `d * 8`. Measurement showed `d * 116`.
Ratio = 0.07. Missing alloc: `dict[str, int]`. Fix formula:
`d * dict_entry_bytes + d * py_str_overhead_bytes`. Re-measure: 1.03.

**Anti-pattern:** ratio 0.5 -> "let's make tf=2.0". Hides the bug.
Find missing line in code.

---

### §4.4. min_ram_formula derivation (v0.9.175, DEC-067)

`min_ram_formula` is the **bottom-bound**: minimum RAM for the algorithm to work at all.
Complements `ram_formula` (peak) with "must-have" (floor).

**Decomposition:**

```
min_ram = state_min + minimal_working_batch + fixed_overhead
```

| Term | Source | Example |
|---|---|---|
| `state_min` | Required state for any variant | `fit_linear`: `d*d*8` (closed) or `d*8` (sgd); `fit_ets`: `(2*s + 1)*8` (seasonal init) |
| `minimal_working_batch` | Minimum rows for one update | Reduce: 1; Sort: 2 (merge); ETS: `2*s + 1` observations |
| `fixed_overhead` | header + guard | 256–1024 B (model-specific) |

**Derivation rules (AP23 analog for bottom-bound):**

1. **Read code** — find every n-sized **unavoidable** allocation. "Unavoidable" = no
   SpillArray alternative in this variant.
2. **Each term points to `file:line`** in `min_ram_evidence`.
3. **Minimum over variants**: if solver closed min_ram > sgd min_ram, the effective
   model min_ram = sgd min_ram (auto-switch can reach sgd).
4. **No invented components**: if you can't find the allocation in code, it doesn't
   exist. Under-estimate is OK for bottom-bound (R2 refuses earlier), over-estimate
   would falsely refuse.

**Verification:**

```python
# Assertion: fit with tight budget = min_ram * 1.05 succeeds
mg = MemoryGuard(max_memory_mb=(min_ram * 1.05) / 1e6)
fit_X(df_small, memory_guard=mg)  # must succeed

# Assertion: fit with tight budget = min_ram * 0.95 fails with actionable
mg = MemoryGuard(max_memory_mb=(min_ram * 0.95) / 1e6)
with pytest.raises(MemoryLimitExceeded) as exc_info:
    fit_X(df_small, memory_guard=mg)
assert "requires" in str(exc_info.value)
assert "options" in str(exc_info.value).lower()
```

**YAML:**

```yaml
min_ram_formula: "d * d * f64 + 3 * row_size + 512"
min_ram_evidence:
  closed_state: "linear.py:142 -- XTX d²·8, unavoidable for closed solver"
  min_batch: "linear.py:88 -- at least 1 row per update"
  overhead: "linear.py:90 -- 512 B header"
```

**Anti-pattern:** `min_ram_formula = ram_formula * 0.1` (fitted). Violates AP23
analog. Must read code.

**Migration:** v0.9.177-178 for 90+ cards (script + AST). Mandatory for `verified`
(R5 + G14).

---

## §5. Step 4 — YAML card

```yaml
- name: Linear/Ridge (Newton)
  cadw_function: fit_linear
  cadw_module: cadw.ml.linear
  implementation_status:
    verified_in_cadw: true
    status: verified
  n_scalability: constant
  bounded_state: true
  ram_formula: "2 * d * d * f64 + 3 * batch * row_size + batch * d * f64"
  formula_evidence:
    d_squared: "linear.py:142 -- check_state(2*d*d)"
    batch_input: "linear.py:88 -- check_batch(extra_factor=3.0)"
    batch_transient: "linear.py:145 -- np.empty((batch, d), f64)"
  measurement:
    matrix: {n: [10^3, 10^4, 10^5], d: [10, 100, 1000], batch: 1000}
    cells_passed: "9/9"
    max_ratio_deviation: 0.28
    date: "2026-10-31"
    env: "py3.12/np2.4/pa26"
  transient_factor: 1.0    # legacy fallback, deprecated
```

`transient_factor` remains only for backward compat. When
`ram_formula` is present — it is ignored.

---

## §6. Step 5 — Status decision

| Condition | Status |
|---|---|
| Stability §2 pass + formula §3 + matrix §4 (all cells) | `verified` |
| Stability pass, formula exists, matrix 1-2 cells outside but not critical | `verified` with `measurement.max_ratio_deviation` |
| Stability pass, formula exists, matrix shows big mismatch | `draft` + `# formula: missing <alloc>` |
| Stability fails | `draft` + `# draft_reason: ...` |
| Bounded by domain (single-series TS, n<=10^6) | `tier1_core_exception` + tested bound |

**No other statuses.** No `verified_typical`, no `verified_with_backlog`,
no `reference_only`, no `verified_cost_bounded`. If known gap -> `draft`.

**`tier1_core_exception` (temporary wart, v0.9.102):** for
`tier_level: tier1_core` cards where n is bounded by problem domain
(single-series TS, n<=10^6). Replaced in v0.9.103 by
`bounded_by_domain` field + `status: verified` (see AI_CONSTITUTION §5.2,
DEC-011). Do NOT invent other `tier*_exception` statuses.

---

### §6.1. Insufficient tier semantics (v0.9.175, DEC-067)

2-tier model replaces old single "draft". Additional YAML field: `insufficient_tier`.

**`operational`** (default for verified): `budget >= min_ram(best_variant)` at any
reasonable budget. Model works via auto-switch if needed.

**`insufficient_no_alt`**: `min_ram(best_variant)` can exceed typical budget. Verified
meaning "auto-switch works when budget >= X". Below X: `MemoryLimitExceeded` with
actionable options. Documented `min_ram_formula` provides X.

Example: `fit_gbdt(n_estimators=100, max_depth=3)` — `min_ram ≈ 17 KB`. Always
operational on any sane budget. `fit_gbdt(n_estimators=1000, max_depth=10)` — `min_ram
≈ 200 GB`. `insufficient_no_alt` tier: refuse with "reduce n_estimators/depth".

Both tiers can be `status: verified` — the difference is `min_ram_formula` magnitude
relative to typical deployment budget. Documentation must be honest about which tier.

### §6.2. Approximation verification (v0.9.175, R6)

Any approximation (auto-switch, window truncation, subsample, output spill) MUST:

1. Attach `ApproximationInfo` to result (list, 0/1/2 entries).
2. Emit `ApproximationWarning` (unless explicitly suppressed via
   `suppress_approximation_warnings=True`).
3. `ApproximationInfo.accuracy_impact` field documents expected loss ("none",
   "negligible", "estimated: ~3%", "documented DEC-033 §4.2").

**Test pattern:**

```python
# Trigger auto-switch
result = fit_X(df, solver="closed", memory_guard=tight_mg)
assert result.approximation is not None
assert any(a.type == "solver_switch" for a in result.approximation)
assert result.approximation[0].accuracy_impact != "none"

# Exact (no approximation)
result = fit_X(df, memory_guard=large_mg)
assert result.approximation is None  # or [] depending on convention
```

Silent approximation = AP24 violation.

---

## §7. Re-verification triggers

A `status: verified` card must be re-checked when:

1. Algorithm code changes (`git log -1 --format=%h cadw/ml/X.py` differs
   from `formula_evidence.date`)
2. Python/numpy/arrow version changes (`memory_constants.yaml` `calibrated:`)
3. `measure_peak` output drifts >20% from `measurement.max_ratio_deviation`

If re-check fails -> `status: draft` until formula updated. Opening a
`verified` card is a signal: earlier closure was dishonest.

---

## §8. Checklist (for AI session start)

Before writing any formula:

- [ ] Have you read `AI_CONSTITUTION` §5? (AP21-AP23)
- [ ] Have you read `GLOSSARY.md`?
- [ ] Which algorithm are you working on?
- [ ] Current status? (`grep 'name: <X>' docs/ml_tools/*.yaml`)
- [ ] If `status: draft`: is blocking reason resolved? If no -> do NOT
      proceed. If yes -> §2 stability first.
- [ ] If `status: verified`: sure you need to touch it? Re-opening is a
      red flag — document why (DECISION_LOG entry).

Before marking `status: verified`:

- [ ] §2 stability all pass
- [ ] §3 formula has evidence per term
- [ ] §4 matrix 9/9 (or 18/18)
- [ ] `test_X_formula_accuracy` in fast suite
- [ ] `AI_CONSTITUTION` gates G1-G11 green
- [ ] No `# BACKLOG:` mentioning this function
- [ ] `DECISION_LOG` updated
- [ ] Followed `EXPERIMENT_PROCEDURE.md` §3 (six steps) and §4 (R1-R4)
- [ ] Exit criterion (PROCEDURE §5) satisfied for this algorithm
