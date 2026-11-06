# Card: GBDT_multiclass

**YAML:** `docs/ml_tools/A_classification.yaml`
**Function:** `fit_gbdt_classifier` (`cadw.ml.gbdt`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** high
- **Confidence:** medium
- **Blocking items:** 7 (see §Blocking)
- **Next action:** gbdt.py:997/1598: replace n_total*32 state guard with batch-bounded SpillDir estimate and add batch_size kwarg to fit_gbdt_classifier; then derive+measure formula.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(batch)` |
| `state_type` | `iterative` |
| `bounded_state` | `true` |
| `n_scalability` | `constant` |
| `bounded_state_dims` | `{"batch": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["reduce", "spill_dir", "memmap", "codes_bin"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `absent` |

## Code inventory

**File:** `cadw/ml/gbdt.py` (exists: yes; 1899 lines)
**Function:** `fit_gbdt_classifier` at fit_gbdt_classifier at 1197, multiclass branch at 1508-1598
**Recent git history (3 commits):** `3c920d6 v0.9.108: refactor _stack_numeric + _column_to_f64 -> _common.py` / `388213b v0.9.87-P1a-4: multiclass y_all -> memmap (last n-sized RAM, true OOC)` / `3dd67b1 v0.9.87-P1a-3b: multiclass pred -> memmap (true OOC for pred)`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - same module |
| `batch_size kwarg` | no - fit_gbdt_classifier:1197 |
| `check_state calls` | 1598 (multiclass), 997 |
| `check_batch calls` | 0 direct |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no |
| `np.concatenate/vstack/column_stack` | 461/528 ReduceOp |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_gbdt_multiclass.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | `O(batch)` | n_total*32 state allowed by guard at 997 -> O(n) worst case | NO |
| `state_type` | `iterative` | multi-pass boosting | YES |
| `bounded_state` | `true` | guard permits O(n) internal state | NO |
| `n_scalability` | `constant` | no batch_size kwarg; guard O(n) | NO |
| `bounded_state_dims` | `{batch:1}` | needs n for internal state | NO |

## YAML lies detected

- ['**bounded_state / n_scalability** claims true/constant, but multiclass fit allocates `np.full(n,-1)` and `np.full((n,K),nan)` at `cadw/ml/gbdt.py:1512-1513` and guards `n_total*8*2` at `1598`.']

## APs detected

- **AP1** at `cadw/ml/gbdt.py:1512-1513` - `np.full(n, -1)` and `np.full((n, K), np.nan)` outside the guard at 1598 (allocated in caller before guard)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "2*d*f64 + S*(f*bins*8*2 + 8*3) + 7.0*batch*row_size + n_total*8*2"
formula_evidence_hypothesis:
  2*d*f64: "gbdt.py:208/212 - MinMax ReduceOp estimate_state_nbytes 8*d*2"
  S*(f*bins*8*2 + 8*3): "gbdt.py:289/380 - histogram ReduceOp per-slot state"
  7.0*batch*row_size: "gbdt.py:295/456 - batch_memory_factor 7.0 / 4.0"
  n_total*32: "gbdt.py:997 - est_n_state = n_total*32"
  n_total*8*2: "gbdt.py:1598 - est_n_state = n_total*8*2"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | NO | gbdt.py:997 - O(n) internal state; no batch_size kwarg |
| d | PARTIAL | gbdt.py:289 - O(f*bins) histogram, d unbounded |
| K | PARTIAL | gbdt.py:1598 - multiclass state O(n*8*2) per class |
| n_groups | N/A | no group axis |

## Blocking items for `status: verified`

- [ ] `implementation_status.status` missing - YAML field
- [ ] `ram_formula` missing - YAML field
- [ ] `formula_evidence` missing - YAML field
- [ ] `measurement` missing - YAML field
- [ ] No `batch_size` kwarg on fit - gbdt.py:1197
- [ ] O(n) internal-state guard contradicts `n_scalability: constant` - gbdt.py:997
- [ ] No verification test `tests/test_gbdt_verification.py`

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** GBDT is the most-used tier1_core card (4 cards share this code path) but the guard still admits O(n) internal state and the fit entry point has no batch_size kwarg, so the declared constant-n promise is not met. Formula is derivable from ReduceOp state estimates and batch_memory_factor.
- **Priority:** high
- **Confidence:** medium
- **Next action:** `gbdt.py:997/1598: replace n_total*32 state guard with batch-bounded SpillDir estimate and add batch_size kwarg to fit_gbdt_classifier; then derive+measure formula.`

## References

- tests/test_gbdt_classification.py
- tests/test_gbdt_multiclass.py
- tests/test_iforest_verification.py (template)