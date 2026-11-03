# Example: Verified Card (fit_linear)

**Purpose:** one full example showing what a `status: verified` card
looks like with formula derived from code, evidence per term, and
matrix measurement. New card authors copy this structure.

**Hierarchy:** Tier 4 (reference). Complements
`AI_CONSTITUTION.md` (Tier 1), `STABILITY_PROTOCOL.md` (Tier 2),
`ALGORITHM_TEMPLATE.md` (Tier 3). Does NOT override any rule.
On conflict — higher tier wins.

---

## Algorithm: `fit_linear` (Linear/Ridge, Newton closed-form)

**Source:** `cadw/ml/linear.py`
**Pattern:** Reduce (XTX accumulation across batches).

---

## Step 1. Stability check

**OOC check** (`tests/test_linear_stability.py`):

```python
def test_linear_ooc():
    peak_n3 = measure_peak(fit_linear, make_df(n=10**3, d=100))["total_bytes"]
    peak_n5 = measure_peak(fit_linear, make_df(n=10**5, d=100))["total_bytes"]
    assert peak_n5 / max(peak_n3, 1) < 2.0
```

**Result:** ratio ≈ 1.0 (state is d²-based, independent of n). PASS.

**Spill check:**

```python
def test_linear_spill():
    with pytest.raises(MemoryLimitExceeded):
        fit_linear(df_n=10**7, d=1000, memory_guard=MemoryGuard(max_memory_mb=1))
```

**Result:** raises `MemoryLimitExceeded` (state 2·d²·8 = 16 MB > 1 MB budget). PASS.

**No open backlog:** `grep '# BACKLOG:' cadw/ml/linear.py` — empty.
`.pingis/plan_v0.9.87_refactor.md` — no `fit_linear` entry.
`ROADMAP.md` — no "fit_linear deferred". PASS.

---

## Step 2. Formula from code

Read `cadw/ml/linear.py`. Relevant lines:

```python
# line 88: mg.check_batch(b, name="linear.batch", extra_factor=3.0)
# line 142: mg.check_state(2 * d * d, name="linear.xtx")
# line 145: X = np.empty((b.num_rows, d), dtype=np.float64)
```

**Terms:**

| Code | Formula term |
|---|---|
| `mg.check_batch(..., extra_factor=3.0)` | `3 * batch * row_size` |
| `mg.check_state(2 * d * d)` (bytes as elements × f64) | `2 * d * d * f64` |
| `np.empty((batch, d), f64)` | `batch * d * f64` |

**Formula:**

```
2 * d * d * f64 + 3 * batch * row_size + batch * d * f64
```

**Evidence per term:**

```yaml
formula_evidence:
  d_squared: "linear.py:142 — mg.check_state(2*d*d, name='linear.xtx')"
  batch_input: "linear.py:88 — mg.check_batch(b, extra_factor=3.0)"
  batch_transient: "linear.py:145 — np.empty((b.num_rows, d), f64)"
```

Verification: `sed -n '142p' cadw/ml/linear.py` must show
`mg.check_state(2 * d * d, name="linear.xtx")`.

---

## Step 3. Matrix measurement

**Fixed batch=1000.** Matrix: `n ∈ {10³, 10⁴, 10⁵}` × `d ∈ {10, 100, 1000}`.

Test:

```python
@pytest.mark.parametrize("n", [10**3, 10**4, 10**5])
@pytest.mark.parametrize("d", [10, 100, 1000])
def test_linear_formula_matrix(n, d):
    predicted = evaluate_formula(
        "fit_linear", batch=1000, row_size=16, d=d
    )["bytes"]
    measured = measure_peak(
        fit_linear, make_df(n=n, d=d), memory_guard=MemoryGuard(max_memory_mb=500)
    )["total_bytes"]
    ratio = predicted / max(measured, 1.0)
    assert 0.7 <= ratio <= 1.4, (
        f"n={n} d={d}: predicted {predicted/1e6:.2f}MB, "
        f"measured {measured/1e6:.2f}MB, ratio {ratio:.2f}"
    )
```

**Result matrix (representative):**

| d \ n | 10³ | 10⁴ | 10⁵ |
|---|---|---|---|
| 10 | 1.03 | 1.02 | 1.01 |
| 100 | 1.05 | 1.04 | 1.03 |
| 1000 | 1.08 | 1.05 | 1.06 |

Max deviation: 1.08 → `max_ratio_deviation = 0.08`. All 9/9 in
[0.7, 1.4]. PASS.

**Why d=1000 row is slightly higher:** at d=1000, `2 · d² · 8 = 16 MB`
state dominates; small state-fragmentation (numpy alignment, allocator
overhead) pushes ratio up. Still well within ±30%. Documented as
`max_ratio_deviation = 0.08`.

---

## Step 4. YAML card

File: `docs/ml_tools/A_classification.yaml`

```yaml
- name: Linear/Ridge (Newton)
  cadw_function: fit_linear
  cadw_module: cadw.ml.linear
  implementation_status:
    verified_in_cadw: true
    status: verified
  tier_level: tier1_core
  n_scalability: constant
  bounded_state: true
  bounded_state_dims: {d: 2}
  ram_formula: "2 * d * d * f64 + 3 * batch * row_size + batch * d * f64"
  formula_evidence:
    d_squared: "linear.py:142 — mg.check_state(2*d*d, name='linear.xtx')"
    batch_input: "linear.py:88 — mg.check_batch(b, extra_factor=3.0)"
    batch_transient: "linear.py:145 — np.empty((b.num_rows, d), f64)"
  measurement:
    matrix: {n: [10^3, 10^4, 10^5], d: [10, 100, 1000], batch: 1000}
    cells_passed: "9/9"
    max_ratio_deviation: 0.08
    date: "2026-10-31"
    env: "py3.12/np2.4/pa26"
  transient_factor: 1.0    # legacy fallback, ignored when ram_formula present
  verified_by_agent: true
  roles:
    - linear_signal
```

---

## Step 5. Tests in fast suite

- `tests/test_linear.py` — algorithm correctness (existing)
- `tests/test_linear_stability.py` — OOC + spill (new)
- `tests/test_linear_formula_accuracy.py` — matrix (new)

---

## Step 6. Verification flow

```bash
# 1. Formula parse + evidence sanity:
uv run pytest -q tests/test_linear_formula_accuracy.py > /tmp/v.log 2>&1
echo "exit: $?"
tail -6 /tmp/v.log

# 2. Stability:
uv run pytest -q tests/test_linear_stability.py > /tmp/v.log 2>&1
echo "exit: $?"
tail -6 /tmp/v.log

# 3. Gates:
uv run pytest -q   tests/test_primitive_reuse.py   tests/test_guard_before_alloc.py   tests/test_ml_tools_schema.py   tests/test_constitution_gates.py   > /tmp/v.log 2>&1
echo "exit: $?"
tail -6 /tmp/v.log
```

All green → `status: verified`.

---

## What makes this card "verified" (checklist)

- [x] Stability §2: OOC + spill + no backlog
- [x] Formula §3: each term has evidence pointer to code line
- [x] Matrix §4: 9/9 cells in [0.7, 1.4], max deviation 0.08
- [x] Tests: correctness + stability + accuracy in fast suite
- [x] Gates G1–G11 green
- [x] No `# BACKLOG:` anywhere mentioning `fit_linear`
- [x] No `DECISION_LOG` entry needed (no novel choice)

---

## What would make this card `draft` (counter-example)

**Case: no formula yet.** Card without `ram_formula`, only legacy
`bounded_state_dims: {d: 2}` + `transient_factor: 2.0`. Status: `draft`.
Reason: `# formula: not derived; pending v0.9.10X`.

**Case: formula fails matrix.** If d=1000 cell gave ratio 0.4 (predicted
16 MB, measured 40 MB), status: `draft`. Reason: `# formula: missing
alloc — investigate linear.py:145 (X temp maybe copied in solver)`.
Correct action: read code, find the copy, add term. NOT bump tf.

**Case: not OOC.** If peak scaled linearly with n (ratio n=10⁵/n=10³ > 2),
status: `draft`. Reason: `# draft_reason: not OOC (n-sized state at
<line>); target v0.9.10X`. Do NOT write formula — formula for unstable
algorithm will be wrong by next session.

---

---

## Example 2: Verified Card (fit_linucb, component-wise, DEC-033)

**Purpose:** second reference pattern. When combined RSS matrix is
unmeasurable (baseline >> peak), use component-wise verification.
See STABILITY_PROTOCOL §4.1.1.

**Source:** `cadw/ml/linucb.py` + `tests/test_linucb_verification.py`

### Why component-wise

`fit_linucb` state is `A: K*d^2*f64 + b: K*d*f64` (~40 MB at K=5,d=1000),
fit peak ~80-100 MB. Baseline (Python + Arrow + parquet mmap) ~400-450 MB
already exceeds fit peak before fit starts. `measure_peak` (ru_maxrss
delta) returns 0-1.4 MB. Physically unmeasurable — not a formula or code
bug. See DEC-033 for 16-iteration investigation.

### Five checks (all in `test_linucb_verification.py`, 2.5s)

**CW-1: state exact** (parametrized K, d):

```python
@pytest.mark.parametrize("K,d", [(2,10), (5,50), (5,100), (10,200)])
def test_state_nbytes_exact(K, d):
    m = fit_linucb(_make_df(100, d), cols, ..., k_max_hint=K)
    assert m.state_nbytes() == K*d*d*8 + K*d*8 + 256
```

**CW-2: state bounded for n >= S** (reservoir sample varies tree shape,
bounded not identical):

```python
def test_state_bounded_for_n_ge_s():
    states = [fit_linucb(_make_df(n, d)).state_nbytes() for n in [500, 2000, 5000]]
    assert max(states) / min(states) <= 1.5
```

**CW-3: coef isolated** (d=200, batch 1000-4000, signal 1.6-6.4 MB):

```python
def test_batch_coef_isolated():
    peaks = [measure_peak_traced(fit_linucb, df_batch=b)["peak_bytes"]
             for b in [1000, 2000, 4000]]
    coef = np.polyfit([1000, 2000, 4000], peaks, 1)[0] / row_size
    assert 2.5 <= coef <= 4.5   # measured 3.08-3.64; 3 physical copies
```

**CW-4: formula small-scale** (median-of-3, SNR 13%):

```python
def test_formula_small_scale():
    peaks = [measure_peak_traced(fit_linucb, df, batch=1000)["peak_bytes"]
             for _ in range(3)]
    measured = statistics.median(peaks)
    predicted = state_nbytes + 1*d^2*8 + 3.5*batch*row_size
    assert 0.7 <= predicted / measured <= 1.4
```

**CW-5: OOC** (unchanged):

```python
def test_ooc_batch_fixed():
    peaks = [measure_peak_traced(fit_linucb, df_n, batch=500)["peak_bytes"]
             for n in [1000, 5000]]
    assert peaks[1] / peaks[0] < 2.0
```

### YAML card

```yaml
- name: LinUCB
  cadw_function: fit_linucb
  implementation_status:
    verified_in_cadw: true
    status: verified
  ram_formula: "k*d*d*f64 + k*d*f64 + d*d*f64 + 256 + 3.5*batch*row_size"
  formula_evidence:
    state_A: "linucb.py -- A = np.zeros((k, d, d)); A.nbytes = k*d*d*8"
    state_b: "linucb.py -- b = np.zeros((k, d)); b.nbytes = k*d*8"
    state_header: "linucb.py -- LinUCBModel.state_nbytes() adds +256 header"
    temp_one: "linucb.py -- Xa.T @ Xa inside per-arm loop; ONE temp d*d*8"
    coef_batch: "measured 3.08-3.64 (isolated sweep d=200); 3 copies: X + Xa + Arrow batch"
  measurement:
    verification_mode: "component-wise (DEC-033)"
    state_check: "test_state_nbytes_exact: bit-exact, 4 (K,d) parametrized"
    state_ooc: "test_state_bounded_for_n_ge_s: max/min <= 1.5x"
    coef_batch: "test_batch_coef_isolated: 2.5-4.5"
    formula_sanity: "test_formula_small_scale: 0.7-1.4"
    ooc: "test_ooc_batch_fixed: ratio < 2.0"
    cells_passed: "8/8"
    runtime_sec: "~3"
  bounded_state_dims: {k: 1, d: 1}
```

### What "verified" means here

- State formula exact to the byte (better than ±30% RSS estimate)
- OOC proven (peak does not grow with n at fixed batch)
- coef physical (3 code allocations, not fitted coefficient)
- formula small-scale validated (d=200, peak 7-10 MB, measurable)

**Not claimed:** combined matrix 9/9 at large d — unmeasurable, open
issue v0.9.111 (fork-based instrument). Documented in DEC-033.

## Anti-patterns (from this card's history)

**AP-x1: "tf=2.0 for Newton, tf=14 for LDA"** — hiding formula in
scalar. Replaced by explicit `2 * d * d * f64` (Newton) vs
`d * dict_entry_bytes + d * py_str_overhead_bytes` (LDA). Two
different units — cannot share tf.

**AP-x2: matrix measured but no formula evidence** — if ratio is
within ±30% but formula has no evidence pointer, card is `draft`.
Formula that "happens to match" without code derivation is fragile:
next refactor breaks it silently.
