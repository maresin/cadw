# Adding a New Model to CADW — Template

**Version:** 1.1 (2026-11-03, v0.9.110)
**Purpose:** one-page checklist. Follow order exactly.
**Companions:** `AI_CONSTITUTION.md`, `STABILITY_PROTOCOL.md`,
`GLOSSARY.md`, `DECISION_LOG.md`, `CHAT_PROTOCOL.md`.

**Hierarchy:** Tier 2 (methodological). Complements
`AI_CONSTITUTION.md` (Tier 1). Does NOT override §1 principles,
§3 patterns, §5 anti-patterns. On conflict — `AI_CONSTITUTION.md`
wins. If you believe a rule here contradicts the constitution,
stop and ask: which should change?

---

## Prerequisites

- [ ] Read `AI_CONSTITUTION.md` (rules)
- [ ] Read `STABILITY_PROTOCOL.md` (process)
- [ ] Read `GLOSSARY.md` (terms)
- [ ] Read `DECISION_LOG.md` (why previous choices)

---

## Step 1. Choose pattern

From `AI_CONSTITUTION` §3:

| Pattern | Reference test |
|---|---|
| Reduce (aggregation) | `test_reduce.py` |
| HashShuffle (partition) | `test_shuffle.py` |
| Sort (global order) | `test_sort.py` |
| SpillBuffer / SpillArray | `test_spill_buffer.py` |
| Chunked strided (windowed) | `test_hampel.py` |
| Refuse-at-guard (documented) | `test_binseg.py` |

If none fits — open design discussion. Do not invent.

---

## Step 2. Implement (with guard BEFORE alloc)

Every n-sized or O(n*k) allocation preceded by `mg.check_state` or
`mg.check_batch`. Gate G2 enforces via AST scan.

```python
def fit_X(df, *, ..., memory_guard=None):
    mg = memory_guard or df._memory_guard
    mg.check_state(<formula for state>, name="X.state")
    state = np.zeros(<...>)
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="X.batch", extra_factor=Y)
        X = b.to_numpy(...)
    return XModel(state, ...)
```

---

## Step 3. Stability check (do not skip)

```bash
pytest -q tests/test_X_stability.py
```

Checklist:

- [ ] `measure_peak` ratio n=10^3 vs n=10^5 < 2.0 (OOC)
- [ ] tight-guard raises `MemoryLimitExceeded` (spill)
- [ ] no `# BACKLOG:`, not in plan_v0.9.87, not deferred in ROADMAP

**If fails** -> `status: draft`, fix algorithm, **do not write formula**.

---

## Step 4. Formula from code

Not from measurement. Read `cadw/ml/X.py`:

```
Line 142: mg.check_state(2*d*d)             -> 2 * d * d * f64
Line 145: mg.check_batch(b, extra_factor=3) -> 3 * batch * row_size
Line 148: np.empty((batch, d), f64)         -> batch * d * f64
```

Write formula as sum. Add `formula_evidence` per term in YAML.

---

## Step 5. Matrix measurement

```python
# tests/test_X_formula_accuracy.py
@pytest.mark.parametrize("n", [10**3, 10**4, 10**5])
@pytest.mark.parametrize("d", [10, 100, 1000])
def test_X_formula_matrix(n, d):
    predicted = evaluate_formula("fit_X", batch=1000, row_size=16, d=d)["bytes"]
    measured = measure_peak(fit_X, make_df(n, d))["total_bytes"]
    assert 0.7 <= predicted / measured <= 1.4
```

All cells must pass. If any fails -> back to Step 4 (find missing alloc).

---

## Step 5-alt. Component-wise verification (DEC-033)

**When to use:** Step 5 (combined matrix) is unmeasurable because
`baseline >> fit peak`. Detection: `measure_peak(fit)["rss_bytes"] <
5% of formula prediction` while formula > baseline noise.

**Do not fight the tool.** 5 attempts (fadvise, malloc_trim,
smaps_rollup, phase-trace, fork) all confirmed baseline > peak is
fundamental. Switch immediately (see EXPERIMENT_PROCEDURE §3.7).

**Five checks (2-3 seconds total):**

```python
# 1. state exact (parametrize K, d)
def test_state_nbytes_exact(K, d):
    m = fit_X(df, ...)
    assert m.state_nbytes() == <analytic>    # bit-exact, no tolerance

# 2. state bounded for n >= S, monotone for n < S
def test_state_bounded_for_n_ge_s():
    states = [fit_X(df_n).state_nbytes() for n in [500, 2000, 5000]]
    assert max(states) / min(states) <= 1.5

def test_state_monotone_for_n_lt_s():
    assert state(100) < state(500)

# 3. coef isolated (d=200, batch 1000-4000)
def test_batch_coef_isolated():
    peaks = [measure_peak_traced(fit_X, df_batch=b)["peak_bytes"]
             for b in [1000, 2000, 4000]]
    coef = np.polyfit([1000, 2000, 4000], peaks, 1)[0] / row_size
    assert 2.5 <= coef <= 4.5    # 3 physical copies of row_size

# 4. formula small-scale (median-of-3, d=200, batch=1000)
def test_formula_small_scale():
    measured = statistics.median([...])
    assert 0.7 <= predicted / measured <= 1.4

# 5. OOC (same as Step 5)
def test_ooc_batch_fixed():
    assert peak(n=5000) / peak(n=1000) < 2.0
```

**Prerequisites:**
- `model.state_nbytes()` must exist (5-20 lines: `sum(arr.nbytes for arr
  in persistent_state) + header`). Add it if missing — it's 5-20 lines.
- `measure_peak_traced` from `cadw/core/memory_guard.py` (v0.9.110c+).

**Runtime:** 2-3 seconds vs 15 min per matrix attempt (which fails anyway).

**YAML marker:**

```yaml
measurement:
  verification_mode: "component-wise (DEC-033)"
  cells_passed: "8/8"
  runtime_sec: "~3"
```

**Anti-patterns (CW-specific):**

| Anti-pattern | Why wrong |
|---|---|
| Retry matrix at larger d | baseline grows faster than fit peak; unmeasurable |
| `state(n1) == state(n2)` for n>=S | reservoir order-dependent; use bounded <=1.5x |
| `coef` at d=10 (signal 0.3 MB) | RSS noise ~1 MB swallows signal |
| `intercept` from slope-fit | extrapolation to batch=0 amplifies noise 8x |
| Relax matrix tolerance to [0.1, 10] | silences bug, hides physios |

**Valid `status: verified` if all 5 checks pass.** This is not a
relaxation: state is exact (better than ±30%), coef physical (3 code
allocations), OOC independent.

## Step 6. YAML card

```yaml
- name: X
  cadw_function: fit_X
  cadw_module: cadw.ml.X
  implementation_status:
    verified_in_cadw: true
    status: verified
  n_scalability: constant
  bounded_state: true
  bounded_state_dims: {d: 1}
  ram_formula: "<formula from step 4>"
  formula_evidence:
    <term_name>: "<file.py:line -- description>"
  measurement:
    matrix: {n: [10^3,10^4,10^5], d: [10,100,1000], batch: 1000}
    cells_passed: "9/9"
    max_ratio_deviation: <max abs(1 - predicted/measured)>
    date: "<YYYY-MM-DD>"
    env: "<py/np/pa versions>"
```

---

## Step 7. Gate run

```bash
pytest -q \
  tests/test_primitive_reuse.py \
  tests/test_guard_before_alloc.py \
  tests/test_no_dense_windows.py \
  tests/test_ml_tools_schema.py \
  tests/test_constitution_gates.py \
  tests/test_X.py \
  tests/test_X_formula_accuracy.py
```

All green before commit.

---

## Step 8. Commit

```bash
git add cadw/ml/X.py cadw/ml/__init__.py \
        tests/test_X.py tests/test_X_formula_accuracy.py \
        docs/ml_tools/<section>.yaml \
        CHANGELOG.md docs/ROADMAP.md

git commit -F - <<'COMMITMSG'
vX.Y.Z: fit_X (pattern: <reduce|shuffle|sort|...>)

- state: O(...) per formula
- stability: n=10^3..10^5 ratio < 2.0; spill OK
- formula: <formula>
- matrix: 9/9 cells within 30% (max dev X.XX)
- status: verified
COMMITMSG
```

---

## Anti-patterns (each one cost us a session)

| Anti-pattern | Correct |
|---|---|
| Derive formula from measurements | Formula from code; measurement verifies |
| "tf=2.0 looks right" | explicit `2 * d * d * f64` with evidence |
| `verified_typical` (works in range X) | `verified` (all cells) or `draft` |
| "won't fix" for OOC-impossible | `draft` (target version) or `tier1_core_exception` |
| Batch from budget heuristic at calibration | Batch = fixed=1000 |
| Skip stability -> write formula | Stability first; formula after |
| `# BACKLOG:` in verified card | Forbidden; if open -> status=draft |

---

## When in doubt

1. Read `DECISION_LOG.md` — has this come up before?
2. If not, **ask**. Do not invent. A wrong formula persisted = future
   OOM for user.
3. If you must decide: update `DECISION_LOG` with choice + rationale.
