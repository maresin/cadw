# AI Constitution — read first, always

**Audience:** AI assistant in a new session.
**Order (T1 first, see §4.5):** this file → ROADMAP.md →
STABILITY_PROTOCOL.md → ALGORITHM_TEMPLATE.md → GLOSSARY.md →
DECISION_LOG.md → CHAT_PROTOCOL.md → EXAMPLES_INDEX.md →
PRIMITIVES_AND_PRINCIPLES.md → code.
**Companion:** METHODOLOGY §1.1/§3/§5/§7/§8, STREAMING §1/§2/§5/§6.1,
GLOSSARY (canonical terms).

---

## §1. What OOC means — 4 axes, not one

OOC = out-of-core = RAM does not grow with the axis. RAM = `O(batch) +
O(model) + O(d)` regardless of the axis.

| Axis | OOC possible? | Typical state |
|---|---|---|
| **n** (rows) | yes, for ops + ML predict + most fits | O(batch) |
| **d** (features) | partial: SGD yes, closed-form no | O(d²) or O(d·K) |
| **K** (cardinality) | yes via HashShuffle (not built yet) | O(K) or O(partition) |
| **n_groups** (TS) | yes via HashShuffle per group | O(group_partition) |

**Never say "all OOC".** Say which axis.

Per-axis truth table (current v0.9.92):

| Class | n | d | K | n_groups |
|---|---|---|---|---|
| ops (groupby, dedup, sort, quantile, topk) | ✅ | ✅ | ✅ (K→partitions) | — |
| linear / logistic / coxph / pca (closed-form) | ✅ | ❌ O(d²) | — | — |
| kmeans, naive_bayes, gmm (diag), iforest, eif | ✅ | ✅ | — | — |
| gbdt, lambdamart | ✅ | ✅ (bins fixed) | ✅ (classes) | — |
| chunked anomaly (hampel, zms, sub_pca, s2g, kshape, tspq, dwt_mlead, autoencoder, kmeans_ad, patch_memory_ad) | ✅ | ✅ | — | — |
| binseg, stray | ❌ refuse | — | — | — |
| TS fits (stl, VAR, VECM, BVAR, ETS, Theta) | ❌ tier1_core_exception | — | — | ✅ |
| encode_categorical, metrics-classification | ✅ | — | ⚠️ guard | — |

---

## §2. 6 postulates (compressed)

1. **O(batch) RAM**, not O(n). One batch resident at a time.
2. **Base cases, not exceptions** (METHODOLOGY §1.1): n > RAM, K > RAM, d > RAM — all must work with disk. If not, it's a CADW bug.
3. **No escape hatches** (§1.3): no `read_all()`, no `collect()`, no `to_numpy()` on O(n) results. `OnDiskResult.to_table()` raises `MemoryLimitExceeded`.
4. **Three primitives**: `Reduce`, `HashShuffle`, `Sort` + `SpillDir`/`SpillArray`.
5. **Guard before alloc**: `check_state` / `check_batch` BEFORE `np.zeros`/`np.empty`/`np.vstack`/`.append` in batch loop.
6. **Refuse-at-guard is OK** if documented (YAML: `bounded_state: false`, `n_scalability: refuse_at_guard`, `tier1_core_exception`). Silent OOM is not OK.

---

## §3. Five patterns (copy-paste, do not invent)

### §3.1. Reduce — for aggregation state

```python
from cadw.ops.reduce import ReduceOp, reduce

class MyOp(ReduceOp):
    def init(self):             return (0, np.zeros(self.d))
    def update(self, state, b): return (state[0] + b.num_rows, state[1] + ...)
    def merge(self, a, b):      return (a[0] + b[0], a[1] + b[1])
    def finalize(self, s):      return s
    def state_nbytes(self, s):  return int(s[1].nbytes + 8)
    def estimate_state_nbytes(self): return 8 * self.d + 8  # for pre-check

result = reduce(df._iter_filtered_batches(), MyOp(), mg, log=False)
```

State: `O(d)`. Fine for sum/mean/cov (d×d), XTX. **Not** for iterative per-row.

### §3.2. HashShuffle — for partitioned (groupby/join/dedup)

```python
from cadw.ops.shuffle import hash_shuffle

paths = hash_shuffle(
    df._iter_filtered_batches(), key_columns=["k"], value_columns=["v"],
    num_partitions=64, memory_guard=mg, spill_dir=spill / "shuffle")
for p in paths:
    partition = read_partition(p)      # bounded by partition size
    process(partition)
```

State: `O(n / P)`. Use for groupby, join build side, dedup, K-spill.

### §3.3. SpillBuffer / SpillArray — for O(n) output

```python
from cadw.core.spill import SpillArray, AutoArrayResult

def predict_proba(self, df, *, output="auto") -> AutoArrayResult:
    buf = SpillArray(guard=df._memory_guard, width=K, dtype=np.float64,
                     name="myalgo.proba",
                     spill_ratio=0.0 if output == "disk" else 0.5)
    for b in df._iter_filtered_batches():
        df._memory_guard.check_batch(b, name="myalgo.pred", extra_factor=3.0)
        buf.write_array(self._predict_batch(b))   # (batch, K)
    return buf.finalize()
```

**Invariants:**
- `spill_ratio=0.5` for `auto`, `0.0` for `disk` (forces immediate spill).
- `width=1` → output is `(n,)`, not `(n, 1)` (memory-branch compat).
- `to_numpy()` / `to_table()` on spilled raises `MemoryLimitExceeded`.
- `for arr in result` — re-iterable, batches of `(b, K)`.

For `pa.Table` results — `SpillBuffer(guard=mg, schema=..., name=...)`, `write_batch`, `finalize() -> AutoResult`.

### §3.4. Chunked strided — for windowed ops (sliding median, PCA, etc.)

```python
view = np.lib.stride_tricks.sliding_window_view(padded, window_size)
n_w = view.shape[0]
chunk = _resolve_chunk_nw(window_size, guard)   # ~8 MB target
out = np.empty(n_w, dtype=np.float64)
for i in range(0, n_w, chunk):
    end = min(i + chunk, n_w)
    if has_check_state:
        guard.check_state((end-i) * window_size * 8 * 2,
                          name="myalgo.chunk")
    blk = np.ascontiguousarray(view[i:end])      # ONLY here contiguous
    out[i:end] = np.median(blk, axis=1)
```

**Why:** `sliding_window_view` is a 0-copy view, but `np.median(view)`
materialises `(n_w, w)` inside numpy. Peak becomes `n_w·w·8`. Chunk
cuts it to `chunk·w·8`.

### §3.5. Two-path — duck-typed guard back-compat

```python
has_check_state = (guard is not None
                   and hasattr(guard, "check_state")
                   and callable(guard.check_state))

if not has_check_state:
    # Duck-typed guard (max_memory_mb only) or guard=None:
    # old path (bit-identical to pre-migration). Tight → raise.
    est = <analytic bound>
    limit_mb = _probe_memory_limit_mb(guard)
    if limit_mb and est > limit_mb * 1024 * 1024:
        raise MemoryLimitExceeded(...)
    return old_impl(...)
else:
    # Real MemoryGuard: chunked / SpillArray / SpillBuffer path.
    guard.check_state(est_chunked, name="myalgo.chunked")
    return chunked_impl(...)
```

**Why:** legacy tests use `class _FakeMB: max_memory_mb = 0.001` — they
must keep raising. Real users get OOC.

### §3.6. Facade unwrap — AutoResult → CadwDataFrame

If your function is called from a `df.something()` method, the facade
must return `CadwDataFrame`, not `AutoResult`:

```python
# In cadw/dataframe/_core.py::_wrap_result_table (already exists):
if isinstance(result, AutoResult):
    return CadwDataFrame(Dataset.from_batches(
        result.schema, iter(result), self._new_dataset_dir()), ...)
```

`Dataset.from_batches` is eager — fully consumes `iter(result)` before
returning, so OnDiskResult tmpdir cannot be GC'd mid-consumption.

---

## §4. Checklist for any new `fit_*` / `predict_*` / `transform_*`

Before writing:

- [ ] What axis does this state grow with? (n / d / K / n_groups)
- [ ] Which pattern from §3 fits? (Reduce / HashShuffle / SpillArray /
      chunked / two-path / refuse)
- [ ] YAML card with: `state_class`, `state_type`, `bounded_state`,
      `bounded_state_dims`, `n_scalability`, `streaming_strategy`,
      `primitive_reuse`. If not OOC → `tier1_core_exception: <reason>`.

Writing:

- [ ] `check_state` or `check_batch` BEFORE first `np.zeros/empty/vstack`
      or `list.append` in batch loop.
- [ ] Return type: `CadwDataFrame` (if facade) or `AutoResult` /
      `AutoArrayResult` (if direct). **Never** `pa.Table` or `np.ndarray`
      for O(n) or O(K) results.
- [ ] `output="auto"` default for direct producers.
- [ ] Test both branches: `output="memory"` (bit-identical to old) and
      `output="auto"` (AutoResult, iterable).

After:

- [ ] G1–G11 gates pass (`pytest -q tests/test_primitive_reuse.py
      tests/test_guard_before_alloc.py tests/test_no_dense_windows.py
      tests/test_constitution_gates.py tests/test_ml_tools_schema.py`).
      G11 (transitive-migration) runs on every `cadw/**/*.py` file
      and catches callers using `pa.Table` / `np.ndarray` API on the
      return of a producer that was migrated to `CadwDataFrame` — see
      §5 AP7.

---

## §4.0. Hard requirements (R1-R7) — normative, v0.9.175

Every `fit_*` / `predict_*` / `transform_*` / df-method / primitive MUST satisfy
R1-R7. Verified status requires all pass. Normative, not advisory.

### R0. Primitives before models (v0.9.189, DEC-070/072) — MANDATORY

Every model refactor MUST follow this order. G18 enforces.

1. **Enumerate leaf primitives** in fit/predict/predict_in_sample path:
   imports (`from cadw.(ts|ops).X import _y`) + local `^def _y(` used
   transitively.
2. **For each leaf primitive, ensure coverage:**
   - a) `streaming_variant` implemented (bit-identical full-n), OR
   - b) window-truncation (`max_window="auto"`, bit-identical n≤W_eff,
        disclosed truncation n>W_eff via `ApproximationInfo`), OR
   - c) `intrinsic_bound.reason` in PrimitiveCard AND model **remains**
        `bounded_state: false` (honest — never flip).
3. **Only then** flip model YAML:
   - `bounded_state: true`
   - `n_scalability: bounded_by_window` (fit transient O(W_eff)) OR
     `constant` (fully streaming)
   - `bounded_state_dims` = **state axes only** (never `n`)

**Forbidden (G18 catches):**
- `bounded_state: true` while ANY transitive leaf has `streaming_variant:
  null` AND no `bounded_by_window` on the model (v0.9.182-184 class).
- Declaring "intrinsic" without checking window-truncation
  (DEC-067 §3/§6: bounded_by_domain is NOT final; refuse = debt with
  target version; window/streaming must be tried first).
- YAML flip without both: bit-identical test (n≤W_eff, 3 strategies ×
  3 n) + OOC peak test (n=10k→100k, ratio <4).

---

### R1. RAM = O(batch + state)

`peak(n=10^6) / peak(n=10^3) < 2.0` at fixed batch (STABILITY_PROTOCOL §2.1).
State does not grow with n (or declared bound axis). If n-sized state is
unavoidable, use SpillArray/SpillDir/SpillBuffer (RAM O(chunk), disk O(n)) — this
**counts as R1-pass** (RAM=O(chunk), disk-managed).

### R2. Always process (unless physically impossible)

If `budget >= min_ram(best_variant)`: **must work**. Auto-switch through candidates
in registry (DEC-067 6 levels: solver/iteration/blocked/sparse/streaming/recursive),
pick first fit. Emit `ApproximationInfo` if the switched variant is not bit-identical.

If `budget < min_ram(best_variant)`: raise `MemoryLimitExceeded` with actionable
`options`. This is **legitimate refuse**, not debt. Message MUST include:
- `model.name`, `params`
- `min_required_bytes`, `available_bytes` (**current** budget, not machine constant)
- `options`: list of concrete suggestions (increase budget / reduce specific params /
  alternative algorithm / sanity check on params)

`budget = MemoryGuard.available_bytes()` — **dynamic**, not machine constant.

### R3. Refuse = clean exception, not OOM kill

`MemoryLimitExceeded` is a Python exception, caught by caller, not process-killed by
kernel OOM. `check_state` / `check_batch` MUST precede all n-sized allocations
(G2 gate enforces via AST). Enforced by `test_guard_before_alloc.py`.

### R4. Formula from code, verified by measurement

`ram_formula` + `min_ram_formula` derived from reading code (AP23), each term
points to `file:line`. Measurement matrix ±30% (STABILITY_PROTOCOL §4), or
component-wise per DEC-033 if combined RSS unmeasurable.

### R5. Verified = R1-R4 pass + no open backlog

`verified_in_cadw: true` + `status: verified` requires R1-R4 pass + measurement
block + no `# BACKLOG:` in source + no `bounded_by_domain` (G17 forbids). Missing any
→ `status: draft` with `draft_reason`.

### R6. Approximation always disclosed

Any approximation (window truncation, solver switch, streaming, subsample, output
spill) MUST attach:
- `result.approximation: list[ApproximationInfo]` (0, 1, or 2 entries)
- `ApproximationWarning` (unless user suppressed via `strict=False` + explicit
  `suppress_approximation_warnings=True`)

Silent approximation = AP24 violation.

### R7. Primitives, not ad-hoc

Every OOC achievement uses a primitive from §3 (Reduce / HashShuffle / Sort /
SpillArray / SpillBuffer / chunked-strided / window). New primitive requires design
discussion (PRIMITIVES_BACKLOG §Maintenance). Auto-switch resolves via primitives,
not via new code paths.

### Orthogonality

R1-R7 orthogonal to §2 postulates (which are descriptive) — R1-R7 are **enforcement**
of postulates. On conflict — R1-R7 win (§4.0 is Tier 1).

## §4.1. Self-verification protocol (MANDATORY — enforced by G8–G10)

Human review is not available. Every new function that allocates
O(n)/O(K)/O(d²) or has `output=`/`solver=` param MUST self-verify:

- [ ] **Shape match**: memory-branch shape == auto-branch shape
      == documented shape. (Test both branches.)
- [ ] **Bit-identical** (fit with quadratic loss) or **documented
      convergence** (logistic/coxph): closed vs sgd. Enforced by G10.
- [ ] **Spill semantics**: guard budget = small, n = large →
      `MemoryLimitExceeded` (not OOM), actionable message.
- [ ] **Determinism**: same seed → same result across 2 runs.
- [ ] **Parameter bounds**: `spill_ratio ∈ {0.0, 0.5}` (G8).
      `chunk < n_w` when n_w > target. Threshold derived from
      `guard.available_bytes()`, NOT a literal int.
- [ ] **Both branches tested**: `output="memory"` AND `output="auto"`
      (or default). G9 enforces for all `output=` producers.

**Green test suite is NOT sufficient.** Gates G8–G10 are the
contract. New producer without self-test = merge fails. If you
believe a whitelist entry is justified, document the rationale
with a roadmap reference — never ship silent O(n) RAM.

**Not enforced automatically** (acknowledged limits):
- "Right pattern, wrong algorithm" (e.g., SGD for Laplace, not MLE)
- "Right pattern, wrong hyperparameter scale" (learning rate)
- "Right test, wrong assertion" (e.g., `assert result is not None`)

These require careful reasoning; when uncertain, prefer refuse-
at-guard over silent approximation (§2 p.6).

## §4.2. G10 dual-branch test criteria by loss type (v0.9.93)

G10 requires closed vs sgd comparison for any `solver=` param.
The comparison metric differs by loss type and conditioning:

| Case | Compare | Threshold |
|------|---------|-----------|
| Quadratic loss (linear/ridge) | coef + intercept | bit-identical, atol=1e-6 |
| Non-quadratic at real reg (logistic α~n·lam) | hard predict + soft RMSE + LL + coef-norm ratio | ≥98%, <0.10, <2%, [0.5,2] |
| Non-quadratic at strong reg (logistic α≫n) | RMSE | <0.10 |
| **Unregularized ill-posed (α=0, separable)** | **do not test** — both MLEs at/near infinity, closed lstsq escapes, SGD stops at finite W | — |

**Anti-pattern #1: comparing coefficients for non-quadratic loss.**
MLE unique but manifold near-degenerate at weak signal + reg.
Coefficients can differ 2-3× while predictions match to RMSE < 0.01.

**Anti-pattern #2: α ≪ n in logistic SGD test.**
`lam = alpha/n` is the correct SGD-to-closed scaling. At n=1000,
alpha=1 → lam=0.001 (illusory regularization). Test appears to
exercise reg but doesn't. Use alpha ≈ n × desired_lam
(e.g., alpha=500 → lam=0.5, alpha=1000 → lam=1.0).

**Anti-pattern #3: RMSprop at strong reg.**
When penalty dominates gradient (lam·|W| ≳ ||data_grad||),
RMSprop step → lr·sign(W), oscillates ±lr. Use plain GD
(`use_rmsprop=False`) when lam*||W|| is not much smaller than
||data_grad||. Plain GD contracts geometrically at 1 - lr*lam.

**Diagnostic signature of implementation bug (vs test artifact):**
- If test fails on predictions but closed/sgd coefs differ by >2×
  AND ||closed|| > ||sgd|| at high alpha → **closed path likely
  doesn't minimize penalized objective** (fixed-point bug). Check
  gradient: is penalty term in the Newton step? This pattern found
  the v0.7.0b closed-path logistic bug (fixed v0.9.93): closed
  ignored penalty, fix point was unregularized MLE.
- Print the penalized objective `LL - alpha/2·||β||²` for both
  branches. Whichever is LOWER (more negative) is closer to the
  true optimum. In v0.9.93: closed obj=502, sgd obj=44 →
  sgd was correct, closed was buggy.
- Only after confirming both branches minimize the SAME objective
  should you tighten test thresholds. Do not relax thresholds to
  pass a buggy-branch test.

**Signature of ill-posedness (test is invalid):**
- Hard predict agreement < 0.90 at α=0 on near-separable data
  → switch to α=500..1000 (real reg) or document α=0 as known
  limitation. MLE at infinity cannot be tested.

This rule came from 5 test iterations in v0.9.93-logistic. See
CHANGELOG v0.9.93-logistic for full attempt log. Future AI
sessions: before writing G10 test, check (a) loss type,
(b) α vs n (need lam≈desired), (c) lam*||W|| vs data_grad
(choose use_rmsprop accordingly), (d) if failing, is it the
test or the implementation (compare objectives, not coefs).

### §4.5. Document hierarchy (T1-T5)

Five tiers. On conflict, higher tier wins. Every doc carries a
`**Hierarchy:** Tier N` header after `**Version:**`.

| Tier | Documents | Overrides |
|---|---|---|
| **T1** Constitutional | `AI_CONSTITUTION.md` | all |
| **T2** Methodological | `STABILITY_PROTOCOL.md`, `METHODOLOGY_ALGORITHM_SELECTION.md`, `STREAMING_OUTPUT_ARCHITECTURE.md` | T3-T5 |
| **T3** Template | `ALGORITHM_TEMPLATE.md`, `EXAMPLE_VERIFIED_CARD.md` | T4-T5 |
| **T4** Reference | `GLOSSARY.md`, `DECISION_LOG.md`, `PRIMITIVES_AND_PRINCIPLES.md`, `EXAMPLES_INDEX.md` | T5 |
| **T5** Operational | `CHAT_PROTOCOL.md` | (none) |

**Conflict resolution:**

- T2 vs T1: T1 wins. Update T2 or escalate to user.
- Session brief vs T1: T1 wins. Ask user (CHAT_PROTOCOL §7).
- T4 vs T2: T2 wins (T2 is process; T4 is reference).

**Rule:** if session brief contradicts T1, do NOT proceed silently. State
the conflict, ask user to either update T1 or issue explicit override.

---

## §5. Anti-patterns (running log; AP1-AP14, AP21-AP23)

> AP numbers are stable IDs, not sequential. AP11 and AP15-AP20 are
> reserved for historical reasons; new APs continue from AP21.
> Rationale: numbers persist across CHANGELOG versions and other docs
> reference APs by ID (e.g. "see AP21"). Renumbering would break
> those references.

### AP1. `np.zeros((n_total, ...))` or `np.zeros((hyperparam, ...))`

```python
# WRONG (G1/G5 fail):
pred = np.zeros((n_total, K))             # n-sized
codes = np.zeros((vocab_size, d), ...)    # hyperparam-sized

# RIGHT:
with SpillDir(guard=mg, name="pred") as sdir:
    pred = np.memmap(sdir.path("pred.bin"), dtype=np.float64,
                     mode="w+", shape=(n_total, K))
```

### AP2. `list.append` in batch loop, then `np.concatenate` / `np.vstack`

```python
# WRONG (G6 fail):
parts = []
for b in df._iter_filtered_batches():
    parts.append(transform(b))
return np.concatenate(parts)     # O(n) return RAM

# RIGHT:
buf = SpillArray(guard=mg, width=K, dtype=np.float64, name="transform")
for b in df._iter_filtered_batches():
    buf.write_array(transform(b))
return buf.finalize()
```

### AP3. `sliding_window_view(...).copy()` or `np.median(view)` full-array

```python
# WRONG (G3 fail):
windows = sliding_window_view(x, w).copy()      # O(n*w) dense
med = np.median(windows, axis=1)                # O(n*w) temp

# RIGHT:
view = sliding_window_view(x, w)                # 0-copy
for i in range(0, n_w, chunk):
    blk = np.ascontiguousarray(view[i:i+chunk])
    med[i:i+chunk] = np.median(blk, axis=1)
```

### AP4. `check_state` AFTER `append` in batch loop

```python
# WRONG (G2 fail):
for b in df._iter_filtered_batches():
    Xs.append(process(b))
    mg.check_state(n * d * 8, name="state")   # OOM window

# RIGHT:
for b in df._iter_filtered_batches():
    n += b.num_rows
    mg.check_state(n * d * 8, name="state")   # BEFORE append
    Xs.append(process(b))
```

### AP5. `len(set(col.to_pylist()))` or `dict[word] += 1` unguarded

```python
# WRONG:
k = len(set(col.to_pylist()))      # O(K) Python objects, no guard
for tok in tokens:
    dfc[tok] = dfc.get(tok, 0) + 1  # dict grows unbounded

# RIGHT:
mg.check_state((len(dfc) + b.num_rows) * 96, name="vocab")
for tok in tokens:
    dfc[tok] = dfc.get(tok, 0) + 1
# Or for K-spill: HashShuffle + per-partition count (backlog v0.9.95)
```

### AP6. Changing public API default → breaks internal self-calls

```python
# WRONG (real bug v0.9.92):
def predict(self, df, *, output="auto"):
    if output == "memory":
        p = self.predict_proba(df, mg)   # inherits "auto" → wrapper
        return classes[np.argmax(p, ...)]  # np.argmax fails

# RIGHT:
def predict(self, df, *, output="auto"):
    if output == "memory":
        p = self.predict_proba(df, mg, output="memory")  # pin old default
        return classes[np.argmax(p, ...)]
```

**Rule:** after flipping a public default, grep all `self.<same_method>(...)`
calls in the same class and pin the old default explicitly.

---

### AP7. Producer has multiple return points — patch ALL of them

A single `def produce(...)` may have:
- memory branch: `return pa.Table(...)` (early, BEFORE tail)
- auto/disk branch: `return buf.finalize()` (tail)
- skew branch: `return _groupby_with_skew(...)` (early, before main path)
- combiner / partial_merge / dict_overflows branches: separate returns
- internal callers: `self.method(...)` might expect old contract

If you patch only the tail, `output="memory"` and `output="skew"`
still return pa.Table while `output="auto"` returns CadwDataFrame —
silent inconsistency. Discovered in v0.9.95 (5 waves, 13 producers).

Checklist before "producer migrated":
- `grep -n 'return' <file>` — count ALL returns in function
- `grep -n 'return _groupby\|return _local\|return groupby_' ops/*.py`
- `grep 'if output == "memory"' <file>` — ensure that branch wraps
- `grep 'self.<method>(' <file>` — internal callers break too

Wrong:
```python
if output == "memory":
    return np.concatenate(parts)   # ← bypasses tail
buf = SpillBuffer(...)
for b in ...: buf.write_batch(b)
return buf.finalize()              # ← tail patched only here
```

Right:
```python
if output == "memory":
    raw = np.concatenate(parts)    # → raw
else:
    buf = SpillBuffer(...)
    for b in ...: buf.write_batch(b)
    raw = buf.finalize()
return CadwDataFrame.from_producer_result(raw, ...)  # one return
```

### AP8. `str.replace` without `else: sys.exit` — silent anchor-miss

Patch scripts must fail loudly when `OLD not in src`:

```python
# WRONG:
src = src.replace(OLD, NEW, 1)  # if OLD absent → silent no-op

# RIGHT:
if OLD not in src:
    print(f"ERR: anchor not found: {OLD[:80]!r}")
    sys.exit(1)
src = src.replace(OLD, NEW, 1)
```

In v0.9.95 two patches silently missed because my OLD-block
assumed state after a prior failed attempt. Test suite caught it
later (full fast), but cost 2 extra fix iterations.

Always:
- print `OK: <what>` after successful apply
- `else: print("ERR <anchor>") + sys.exit`
- verify with `grep <marker> <file>` after patch

### AP9. Arrow API mismatch — pa.concat_tables vs Table.from_batches vs concat_arrays

Three Arrow APIs, three different list-type contracts:

| API | Input | Return |
|---|---|---|
| `pa.concat_tables([...])` | `list[pa.Table]` | Table |
| `pa.Table.from_batches([...])` | `list[pa.RecordBatch]` | Table |
| `pa.concat_arrays([...])` | `list[pa.Array]` | **Array** (not ChunkedArray) |

Common errors:
- `pa.concat_tables(list_batches)` → `TypeError: Cannot convert
  pyarrow.lib.RecordBatch to pyarrow.lib.Table`
- `pa.concat_arrays(list_arrays).combine_chunks()` →
  `AttributeError: 'pyarrow.lib.Array' object has no attribute
  'combine_chunks'` (that's a ChunkedArray method)

Both hit in v0.9.95 wave 5 (value_counts, duplicated).
Fix: use `from_batches` for RecordBatches; drop `.combine_chunks()`
after `concat_arrays`.

### AP10. Patch вслепую — «add X» без git history check

В v0.9.96 дважды написан патч (`to_pylist`, rewrites
`test_groupby_v2`/`test_streaming_output_ops`) для кода, **который
уже был** в `HEAD` от v0.9.95-wave5. `git add` дал 0 diff.

**Правило перед «add X to file Y»:**

```bash
git log --oneline --all -- FILE | head -3        # когда трогали
git show HEAD:FILE | grep -n 'def X\|class X'    # уже есть?
```

**Признак проблемы:** `git add N files` → `commit M (M<N)`. Не
баг — проверять `git diff --stat <parent> HEAD -- <remaining>`:
пусто = ок, непусто = коммит неполный.

### AP12. `\b`-pattern с `.match()` — silent no-op

`.match()` анкорит на **позицию 0**. `\b` требует word-символ
там; пробел в начале строки (`        verified_in_cadw: true`) —
non-word → ни одна строка не матчится.

```python
# WRONG (silent no-op):
RE = re.compile(r"\bverified_in_cadw:\s*true\b")
RE.match(line)              # никогда не True

# RIGHT:
RE = re.compile(r"\bverified_in_cadw:\s*true(?=[,\s}]|$)")
RE.search(line)
```

**Правило:** паттерн без `^` → всегда `.search()`, никогда
`.match()`. End-token guard `(?=[,\s}]|$)` после `true` защищает
от `true_xyz` в notes.

### AP13. `set -eo pipefail; cmd | tail` — visual «crash» в VSCodium

`set -eo pipefail; uv run pytest ... | tail` при падении теста →
pipeline non-zero → `-e` закрывает shell → VSCodium визуально
«терминал упал». Это **не crash**, а закрытие (pty-pipe-баг).

```bash
# WRONG:
set -eo pipefail
uv run pytest ... | tail -20     # крашит терминал при fail

# RIGHT:
uv run pytest ... > /tmp/log 2>&1
echo "exit: $?"
tail -20 /tmp/log
```

**Правило:** `set -eo pipefail` — для скриптов. Для интерактивной
диагностики — `> /tmp/log 2>&1; echo $?; tail`.

### AP14. YAML anchor «next N chars» — ломается о вложенные коллекции

В YAML между `cadw_function: fit_binseg` и `state_class: O(n)`
стоит `benchmark_results:` (сотни строк). Regex «next 500 chars
after anchor» не нашёл `bounded_state_dims`.

**Правило:** YAML-anchor — только по структурному маркеру
(`- name:`, `state_class:`) + **block-scoped search** (до
следующего `- name:` того же indent).

```python
# WRONG:
pos = src.find("cadw_function: fit_binseg")
tail = src[pos:pos+500]

# RIGHT:
m = re.search(r"^(\s*)- name: BinarySegmentation\s*$", src, re.M)
indent = m.group(1)
next_m = re.search(rf"^{re.escape(indent)}- name:", src[m.end():], re.M)
block = src[m.start():m.end() + next_m.start()]
```

### AP21. Atomic closure — no "verified with backlog"

A `status: verified` card means ALL of:

- §2 stability checks pass (STABILITY_PROTOCOL): OOC via n-sweep,
  spill raises `MemoryLimitExceeded`, no `# BACKLOG:` marker anywhere
- Formula derived from code with evidence per term (§3)
- Matrix measurement 9/9 cells within ±30% (§4), **or**
  component-wise verification per DEC-033 (state exact + coef
  isolated + OOC proven) when combined RSS is unmeasurable
  (baseline >> fit peak). See STABILITY_PROTOCOL §4.1.1.
- Last git commit touching function body was the closure commit

If any fail -> `status: draft`, with `# draft_reason:` OR entry in
`ROADMAP` with target version. No intermediate status.

**Symptom:** `fit_lda` was touched 5 times (v0.9.42 -> v0.9.87 ->
v0.9.90 -> v0.9.101 -> v0.9.102) while its YAML card read
`verified_in_cadw: true` from v0.9.42 onward. The field conflated
algorithmic fidelity (verified) with completion (verified). Every
session re-read docstring + CHANGELOG + plan + git log to reconstruct
true state — 20-40 min lost per session.

**Root cause:** `verified_in_cadw: true` was treated as "done". It only
means "algorithm faithful to source". Formula could be missing, stability
could be broken, backlog could be open.

**Correct pattern:**

- Two orthogonal fields:
  - `verified_in_cadw: true|false` — fidelity to source/paper
  - `status: draft|verified|tier1_core_exception` — completion
- `# BACKLOG: <func>` in docstring OR in `plan_v0.9.87` -> status=draft
- Schema test R3-R5 enforces: `verified + # BACKLOG:` -> FAIL

**Reference:** STABILITY_PROTOCOL §2.3, DECISION_LOG DEC-001, DEC-002.

---

### AP22. Slow operations warn, they do not OOM silently (CPU instead of RAM)

OOC fix can trade RAM for CPU. `fit_stray` at n=10^6 runs batched
distances in O(n^2) compute — peak RAM OK (O(n*k) top-k merge), but
runtime ~1 hour. Silent 1-hour compute = same user pain as OOM.

**Correct pattern:**

- Every `fit_*` supports `estimate_cost(op, **params)`:
  - `fits_ram: bool`, `est_ram_mb`, `est_ops`, `est_time_seconds`
  - `time_class: fast | medium | slow | very_slow`
- Time = `ops_per_batch * n_batches / rate` (from `benchmarks.yaml`)
- At runtime: `est_time > 60s` -> `warnings.warn(SlowOperationWarning)`
- Caller may pass `confirm_slow=True` (suppress) or `strict_slow=True`
  (raise `SlowOperationRequiresConfirm`)

**Anti-pattern:** silent slow compute. User runs `fit_stray(n=10^6)`,
expects seconds, gets hour. Same UX failure as silent OOM.

**Reference:** STABILITY_PROTOCOL §4 (ops_per_batch), CHAT_PROTOCOL §4.

---

### AP23. Formula from code, verified by measurement

Formula is a hypothesis derived from reading code. Measurement verifies
it. NOT the other way around.

**Order (mandatory):**

1. Read `cadw/ml/X.py`, extract terms from `mg.check_state`,
   `mg.check_batch`, `np.empty`, `dict[key] += 1`
2. Formula = sum of terms. Each term has evidence pointer (file:line).
3. Fixed `batch=1000`. Vary n, d, k. Matrix 9/9 cells. (Or
   component-wise per DEC-033 if combined RSS unmeasurable.)
4. Assert `0.7 <= predicted/measured <= 1.4` for each cell.
5. If any cell fails: **find missing alloc in code.** Do NOT bump
   coefficients.

**Anti-pattern #1:** ratio 0.5 -> "let's make tf=2.0". This hides the
bug. Real bug: missing `dict_entry_bytes` term. Find the dict line in
code.

**Anti-pattern #2:** derive formula from measurements ("vary d, fit
polynomial, get coefficients"). This masks the same bug: formula
"happens to match" without code derivation is fragile — next refactor
breaks it silently.

**Root symptom:** `transient_factor=12.0` (LDA) was a "measured
coefficient" that turned out 20% wrong because Python dict overhead
changed with Python version. Explicit formula `d * dict_entry_bytes
+ d * py_str_overhead_bytes` + `_load_constants()` reads measured
constants at runtime, adapts to Python version automatically.

**Reference:** STABILITY_PROTOCOL §3-4, EXAMPLE_VERIFIED_CARD.md.

---

### AP24. Silent approximation

Approximation without `ApproximationInfo` in result + `ApproximationWarning` in
warnings = R6 violation.

Wrong: `fit_theta(...)` internally truncates `y_` to last 1e6 obs without warning.
Right: `result.approximation = [ApproximationInfo(type="window_truncated", ...)]`
+ `warnings.warn(ApproximationWarning(...))`.

Reference: DEC-067, DEC-068.

### AP25. Refuse without alternatives when alternative exists

`MemoryLimitExceeded` when `budget >= min_ram(some_variant)` = R2 violation. Must
try all `auto_switch` candidates first (DEC-067 6 levels).

Wrong: `fit_linear(solver="closed")` raises because `d²·8 > budget` — without trying
`solver="sgd"` (state = d·8).
Right: auto-switch to `sgd`, emit `ApproximationInfo(type="solver_switch", ...)`.

### AP26. Budget as machine constant, not dynamic

Hardcoded `8 * 1024**3` or `"weak machine"` checks instead of `mg.available_bytes()`.
Budget comes from user `MemoryGuard(max_memory_mb=...)` — dynamic per call.

Wrong: `if n * 8 > 8 * 1024**3: raise ...`
Right: `if n * 8 > mg.available_bytes(): auto_switch_or_raise(...)`

### AP27. Verified card without min_ram_formula

`status: verified` requires both `ram_formula` AND `min_ram_formula` (R2 depends on
the latter). Schema test G14 enforces.

### AP28. n-sized alloc without check_state (G2)

Same as AP4 restated for §4.0 R3. AST gate G2 (test_guard_before_alloc.py) enforces.

### AP29. intrinsic_bound without checking fit-time vs predict-time

`intrinsic_bound {axis: n, bound: ...}` claims state grows with n. Only
valid for **fit-time** primitives that materialize a training-series-sized
array (`np.empty((n, ...))`, `np.full(n, ...)`) — and even then, caller
may truncate via `_collect_last_w` (fit_leaf) or accept as intrinsic
(chol_solve, eclat).

**Invalid for predict-time helpers** (`_integrate_back`,
`_arma_forecast`): their inputs are `steps` (forecast horizon) or
coefficients, not `n`. Copying `{n: 1e6}` from neighbouring cards is
dishonest (wrong axis, misleads `min_ram_formula` derivations).

**Rule (v0.9.195, DEC-075):**

- PrimitiveCard declares `layer: fit_leaf | intrinsic | predict_leaf`.
- `predict_leaf` -> `intrinsic_bound: null` unless the function genuinely
  calls a fit-time helper (then `intrinsic_bound` required, and `reason`
  must mention the internal call — e.g. `_arma_forecast` calls `_ar_fit`).
- `fit_leaf`/`intrinsic` -> `intrinsic_bound` required (reason >= 20 chars).

**Self-check before adding a primitive card:**

1. Does the function body allocate `np.empty((n, ...))` or `np.full(n, ...)` where n is the **training** size? If no -> `predict_leaf`, `intrinsic_bound: null`.
2. If n-sized alloc is present, is it a training-series array? If it's `steps` (caller-controlled), it's still `predict_leaf`.
3. Does the function call another primitive that IS fit-time? Document in `reason` and keep `intrinsic_bound`.

**Reference:** DEC-075; regression test `test_integrate_back_predict_leaf`.


### AP31. Dishonest tag — files missing from tagged release

`git tag vX.Y.Z` on HEAD implies "the tag reflects release vX.Y.Z". But:
- Tests run against **working tree**, not against the tagged commit.
- `git commit` may have missed files (`git add <specific files>` without `-A`).
- Patch scripts that touch multiple files + `git add` with explicit list can silently skip.

**Symptom:** tag says "Part B merged" but `git show vX.Y.Z:cadw/ts/theta.py | grep _fit_auto` returns 0. Discovery usually weeks later, when someone tries `pip install` from tag.

**Real case:** v0.9.194 tag `11af2aa` contained only version bumps + `_theta_common.py` (create mode in commit `b689bbf` labeled "fix"); core Part B files (`theta.py` dispatchers, `auto_theta.py` shim, `D_time_series.yaml`, `test_auto_theta.py`) were **uncommitted**. Fixed in `9e1241b v0.9.194-d`.

**Rule (mandatory before every `git tag`):**

1. `git status --short` → **empty**. If not, `git add <missing>` + `git commit` (or `git stash` + investigate).
2. `git diff <prev-tag>..HEAD --name-only` → contains all expected files (from CHANGELOG features).
3. For each CHANGELOG feature, `git show HEAD:<file> | grep <marker>` — e.g. new class name, new method, new function.
4. Then `git tag`.

Enforced in CHAT_PROTOCOL §6.5 (pre-tag checklist). Manual check (no automated gate yet — v0.9.196 backlog: `scripts/verify_tag.py`).

**Anti-pattern:** `git tag` immediately after `git commit` without `git status --short`. Tests green ≠ tag honest.

#### AP31.1 Probe pitfalls (v0.9.194-196)

AP31 probe itself can be broken. Before trusting a FAIL, check:

| # | Pitfall | Correct |
|---|---|---|
| P1 | `cat file` (working tree) instead of `git show HEAD:file` | Always `git show HEAD:<file>` after `git status --short` empty check |
| P2 | Substring collision: `_y_centered_recent_` matches `_y_centered_` | Regex `(?<!\w)X(?!\w)`, or whitespace/punct split set-membership |
| P3 | `all_ok = all_ok and not stale` inside `else:` | Put boolean composition **outside** if/else — test with deliberately bad input |
| P4 | awk `/name: X/,/^  - name:/` — empty if start=end line | Python YAML parse: `yaml.safe_load(git show ...)` |
| P5 | f-string `{axis: n}` -> NameError in patch scripts | Escape `{{axis: n}}` or `.format(i=indent)` with `{{...}}` |

**Rule:** every AP31 FAIL must be triaged against P1-P5 before "fix the artifact". Three false positives in v0.9.196 (P1, P2, P3) + one real bug (P4/P5 crash in P54) wasted 3 iterations. Log each false positive class; accumulate into probe test suite (v0.9.197+ `scripts/verify_tag.py`).

### AP32. Verified card with unregistered primitive dependency

Card declares `status: verified` and `depends_on: [cadw.ops.X.Y]`, but
`cadw.ops.X.Y` has no `PRIMITIVES.yaml` card. Hides primitive-audit
gap: model claims OOC, primitive state/streaming semantics undocumented.

**Real case (v0.9.197):** LOF depended on `cadw.ops.linalg.knn_query`
(v0.9.144) — implemented, no card. G18 checks `_name` functions only;
G21 checks `intrinsic_bound` semantics for existing cards. Neither
catches "depends_on X, no card for X". Discovered via manual probe
(P4, v0.9.197); 5 gaps across verified cards (knn_query, core_distances,
sq_dists, sgd_streaming, znormalize_rows). Fixed v0.9.198.

**Rule (v0.9.198, `test_all_verified_depends_on_resolve`):**

For every `verified` card, every `depends_on` entry starting with
`cadw.ops.` MUST resolve to a `PRIMITIVES.yaml` card (by
`cadw_function`), UNLESS in `OPS_LAYER_EXEMPT`:

  reduce, shuffle, sort, join, groupby, pivot, quantile, topk, rank,
  resample, dedup, describe, histogram, metrics — pipeline operations,
  not model primitives (documented in PRIMITIVES_BACKLOG Appendix B).

`cadw.ts.*` deps are covered by G18 separately (leaf streaming_variant
for verified+bounded_state models).

**Anti-pattern:** adding `depends_on: [cadw.ops.new_thing]` to a card
without registering `new_thing`. Fix: register in PRIMITIVES.yaml
(layer/bounded_state/formula_evidence/streaming_variant/intrinsic_bound
per existing card convention).

**Complements:** G18 (leaf streaming_variant audit), AP29
(intrinsic_bound semantic self-check), DEC-071 (^def _name( audit —
narrower scope).


### AP33. Leaf primitive with inconsistent card (bounded_state model)

A `verified` + `bounded_state: true` model depends on a leaf primitive
whose card has `streaming_variant: null` AND `intrinsic_bound: null` AND
model card does NOT declare `bounded_by_window` on that axis. Three cases:

- **Case A (real bug):** model claims OOC in an axis that leaf primitive
  does NOT provide, and no bound declared → fake OOC, silent OOM risk.
  Fix: refactor to batch-layer (add `_chunks` variant) OR add
  `bounded_by_domain` to model card.
- **Case B (card inconsistency, not dishonesty):** model honestly uses
  window-truncation via `_collect_last_w` (bounded_by_window), but leaf
  card omits BOTH `window_truncatable` AND `intrinsic_bound`. Fix: add
  `window_truncatable {axis: n, caller_prim: _collect_last_w,
  min_window: "<expr>", reason: "bit-identical n<=W_eff"}` to leaf card
  (DEC-072: window check before intrinsic; `intrinsic_bound` and
  `window_truncatable` mutually exclusive). Only if window-truncation is
  semantically invalid, add `intrinsic_bound` instead.
- **Case C (predict-leaf, AP29):** predict-time primitive O(steps),
  no fit-time n-scaling. Fix: `intrinsic_bound: null` valid, card note
  `"predict-time O(steps); fit-time no n-scaling (AP29 predict_leaf)"`.

**Real case (v0.9.204):** `_ses_fit_hw`, `stl_decompose`, `_ses` cards
had `streaming_variant: null` but DID have `window_truncatable`
(`caller_prim: _collect_last_w`). Consumers (Theta(s>1), OTM(s>1), ETS,
AutoETS) are `bounded_by_window` with `applies_to: params=manual` —
truncation via `_collect_last_w`. **First draft of this AP33 added
`intrinsic_bound` to these 3 cards — schema rejected (mutually exclusive
with `window_truncatable`, DEC-072).** Correct: `window_truncatable` IS
the bound; gate G22 must accept it as one of three valid (streaming_variant
/ window_truncatable / intrinsic_bound).

**Gate G22** (`test_G22_leaf_deps_consistent`): for each `verified +
bounded_state: true` model, AST-scan the `cadw_module` file's top-level
`from cadw.X.Y import _y` imports. Each leaf `_y` must have either
`streaming_variant != null` OR `intrinsic_bound != null` OR model has
`bounded_by_window` on corresponding axis. Failure = case A.

**Complements:** AP29 (predict_leaf semantic), AP32 (verified depends_on
resolution), G18 (leaf streaming_variant audit).

**Anti-pattern:** add `from cadw.ts.X import _new_primitive` to a
`bounded_state: true` model's module without checking the primitive's
card has bound/streaming. Fix: update card first, then wire.

### §5.1. Single source of truth (one fact, one place)

Every concept has exactly one canonical location:

| Concept | Only in | Others reference by |
|---|---|---|
| Status enum | §4.5 | "see §4.5" |
| Anti-patterns (AP1-AP23) | §5 | "see AP21" |
| Formula grammar | STABILITY_PROTOCOL §3 | code refs docstring |
| Typed constants | memory_constants.yaml | formula names |
| Stability criteria | STABILITY_PROTOCOL §2 | "see STABILITY §2" |
| Decision rationale | DECISION_LOG | "see DEC-XXX" |
| Document hierarchy | §4.5 | "see §4.5" |

**Rule:** if you find yourself duplicating a definition — stop.
Reference by ID instead. If the fact needs update, there is one file to
change.

**Anti-pattern:** GLOSSARY says "verified means matrix passed",
STABILITY_PROTOCOL §6 lists "verified means (criteria)", §4.5 says
"verified means ...". Three slightly different definitions -> drift.
Correct: one canonical (§4.5 status enum), others say "see §4.5".

---

### §5.2. Tier vs status are orthogonal (temporary wart in `tier1_core_exception`)

**Tier** (from `METHODOLOGY_ALGORITHM_SELECTION` §10) describes *how* the model
is built:
- `tier1_core` — CADW-native on primitives (Reduce / HashShuffle / Sort / Spill)
- `tier2_wrapper` — wrapper over external library (sklearn, statsmodels, ...)
- `tier3_export` — export only, not implementable in CADW constraints

**Status** describes *whether the card is done*: `draft` or `verified`.

**The two are independent.** A `tier1_core` model can be `draft` (unstable,
no formula) or `verified`. A `tier2_wrapper` model can be `verified`.

**Temporary wart (v0.9.102):** the status value `tier1_core_exception`
mixes tier and status. It exists for cards where:
- `tier_level: tier1_core`
- `bounded_state: false` (e.g. PELT DP array O(n), single-series TS)
- n is bounded by problem domain (n<=10^6 typical), tested

Meaning: "OOC-ready at all declared axes except n; n-domain bounded,
documented, tested."

**Planned replacement (v0.9.103, DEC-011):**

```yaml
tier_level: tier1_core
bounded_by_domain:
  axis: n
  bound: 10^6
  reason: "single-series DP; n<=10^6 typical"
implementation_status:
  verified_in_cadw: true
  status: verified
```

Rationale: `status` stays binary (`draft` / `verified`); the "exception" is
a **property** of the algorithm (`bounded_by_domain`), not a status.
Three fields (tier_level, bounded_by_domain, status) express three
independent facts. One compound status expressed them poorly.

**Until v0.9.103:** `tier1_core_exception` is the current status value.
Schema test R7 enforces `tier1_core_exception:` field presence when
`status: tier1_core_exception`. Do not add new cards with this status
unless domain bound is real and tested.

**Do NOT invent `tier2_core_exception`** or any other `tier*_exception`
status. Only one temporary wart exists, fixed in v0.9.103.

---

### §5.3. Taxonomy freeze (v0.9.128+)

**Frozen at v0.9.128.** 58 subtypes, каждый имеет ≥3 разнотипных
кандидатов (или explicit `# NOT-ENSEMBLE: <reason>` в YAML).

**Rule:** after v0.9.128, **no new cards** without DEC entry + explicit
rationale (entirely new model category, not "another candidate in
existing subtype"). Post-freeze allowed:

- `status: draft → verified` (via `ram_formula` from code + measurement,
  AP23 + DEC-033).
- `ram_formula` / `formula_evidence` completion for existing `draft` cards.
- `cross_ref` field formalization (markdown -> schema field).
- `ensemble_recommendation` additions/refinements.
- `tier: ensemble` card creation **only if** explicit DEC (e.g.,
  CADW-native ensembling framework, not per-subtype duplicated recipes).

**Why freeze:** taxonomy filling became the dominant activity
(v0.9.123-128: 5 of 10 releases). Each session found gaps; filling them
increased surface without changing what CADW actually *does*. After
v0.9.128, all subtypes have ≥3 diverse candidates or explicit NOT-ENSEMBLE.
Further additions are diminishing returns — implementation, verification,
migration dominate future sessions.

**Anti-pattern:** adding "one more candidate" to a subtype without DEC
rationale. This is how v0.9.119-127 accumulated 8 releases of card-only
work. Post-freeze, card additions are exceptional, not routine.

**Enforcement:** `test_R10_public_fit_covered` + `test_R10_cadw_ts_covered`
ensure new public functions get cards. There is no automated "no new
candidate cards" check (would need heuristic on card names/types). The
freeze is a **declared convention** (this §), enforced by review.

---

## §6. Roadmap summary (3 sessions)

**Next sessions** (see docs/ROADMAP.md for details):

- **v0.9.97**: YAML `bounded_state_dims` (84 cards), CoxPH
  concordance streaming, `df.estimate_memory`.
- **v0.9.98**: sparse `tfidf.transform` (CSR SpillBuffer), K-spill
  (`encode_categorical`, `metrics`, `lda`, `tfidf`).
- **v0.9.99**: D4 ROCKET variants, TS forecast extension
  (AutoARIMA, Croston).

**Do not re-solve** (per STREAMING §6.1):

- `eif` bit-identical: 40 GB I/O for 0 functional gain (won't-fix).
- `kmeans_ad` bit-identical: 3.2 GB I/O (won't-fix).
- `autoencoder` OOC beyond `n_w=state_limit/16`: 2-3 sessions, no use case.
- sparse `tfidf.transform`: needs CSR SpillBuffer, separate design.
- `binseg` / `stray`: intrinsic O(n)/O(n²), refuse-at-guard.
- TS single-series fits: `tier1_core_exception`, n ≤ 10^6 typical.

---

## §7. Where to look

| Question | Read |
|---|---|
| What is OOC exactly? | METHODOLOGY §1.1, §3 |
| What patterns exist? | PRIMITIVES §2, §5 |
| What's the migration status? | STREAMING §3, §6 |
| What's not OOC and why? | STREAMING §6.1 |
| Which code to copy? | EXAMPLES_INDEX.md |
| What's the current task? | ROADMAP.md |
| Why was X migrated? | CHANGELOG v0.9.90-v0.9.92 |
