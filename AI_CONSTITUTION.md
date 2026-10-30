# AI Constitution — read first, always

**Audience:** AI assistant in a new session.
**Order:** this file → ROADMAP.md → EXAMPLES_INDEX.md → PRIMITIVES → code.
**Companion:** METHODOLOGY §1.1/§3/§5/§7/§8, STREAMING §1/§2/§5/§6.1.

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

- [ ] G1–G7 gates pass (`pytest -q tests/test_primitive_reuse.py
      tests/test_guard_before_alloc.py tests/test_no_dense_windows.py
      tests/test_constitution_gates.py tests/test_ml_tools_schema.py`).

---

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

## §5. Anti-patterns (5 most common)

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

## §6. Roadmap summary (3 sessions) + won't-fix

**Next sessions** (see docs/ROADMAP.md for details):

- **v0.9.93**: wide-feature OOC — migrate `linear`/`logistic`/`coxph`/`pca`
  to `cadw/ops/optim.py::sgd_streaming` when `d > 10^4`.
- **v0.9.94**: unified `CadwDataFrame` output on facade (7 return types → 1).
- **v0.9.95**: S5b-full (TS group path), K-spill (`encode_categorical`,
  `metrics` classification).

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
