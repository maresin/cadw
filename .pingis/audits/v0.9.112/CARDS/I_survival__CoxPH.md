# Card: CoxPH

**YAML:** `docs/ml_tools/I_survival.yaml`
**Function:** `fit_coxph` (`cadw.ml.coxph`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** high
- **Confidence:** medium
- **Blocking items:** 7 (see §Blocking)
- **Next action:** after formula: add `bounded_by_domain: {axis: n, bound: 1e7, reason: ...}`, set `state_class: O(d^2 + pending)`, flip status. Evidence path: cadw/ml/coxph.py:455-500; binding not needed (module-level fit_coxph).

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(n)` |
| `state_type` | `iterative` |
| `bounded_state` | `false` |
| `n_scalability` | `linear` |
| `bounded_state_dims` | `{"n": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["external_sort", "reduce", "memmap", "spill_dir"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `present` |

## Code inventory

**File:** `cadw/ml/coxph.py` (exists: yes; 597 lines)
**Function:** `fit_coxph` at (class method)
**Recent git history (3 commits):** `f16f48f v0.9.101: coxph reference-only + LDA dims fix` / `73ad562 v0.9.97: YAML bounded_state_dims + df.estimate_memory` / `c82cdc9 v0.9.95-wave2: unified CadwDataFrame (coxph.predict_risk + gmm)`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - CoxPHModel.state_nbytes:112 |
| `batch_size kwarg` | no - fit_coxph:431 |
| `check_state calls` | 411, 461, 535, 545, 582, 593 |
| `check_batch calls` | 475 (extra_factor=2.0) |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no |
| `np.concatenate/vstack/column_stack` | 191/264/344 bounded pending ties |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_coxph.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(d^2 + pending) | see lies |
| `state_type` | see inventory | multi-pass Newton | see lies |
| `bounded_state` | see inventory | streaming + external_sort | see lies |
| `n_scalability` | see inventory | streamed | see lies |
| `bounded_state_dims` | see inventory | {n} declared, real is d^2 | see lies |

## YAML lies detected

- **state_class / bounded_state / n_scalability** claim `O(n)` / false / linear, but code streams every n-sized pass through `external_sort` + `_iter_filtered_batches` (`cadw/ml/coxph.py:474-500`) and bounds memory by `pending_budget` (`coxph.py:460`). Real state is `O(d^2 + pending)`; n is OOC.

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "3*d*d*f64 + 2.0*batch*row_size + 5*pending_budget*f64 + 2*n_events*f64"
formula_evidence_hypothesis:
  3*d*d*f64: "coxph.py:461 - mg.check_state(3*d*d*8 + 5*pending_budget)"
  2.0*batch*row_size: "coxph.py:475 - mg.check_batch(extra_factor=2.0)"
  5*pending_budget*f64: "coxph.py:460 - pending_budget = state_limit_bytes//8"
  2*n_events*f64: "coxph.py:535 - mg.check_state(2*n_events*8+256)"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | YES | see formula/evidence above; assessed by code inspection |
| d | PARTIAL | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] `implementation_status.status` missing
- [ ] `ram_formula` missing
- [ ] `formula_evidence` missing
- [ ] `measurement` missing
- [ ] `tier1_core_exception` wart (DEC-011) - use `bounded_by_domain` instead
- [ ] `state_class`/`bounded_state`/`n_scalability` mismatch - coxph.py:474-500
- [ ] No verification test `tests/test_coxph_verification.py`

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** CoxPH is an n-OOC algorithm (external_sort + streaming passes) mislabelled as O(n)/linear in the YAML. The remaining work is a formula + component-wise measurement plus removing the DEC-011 tier1_core_exception wart.
- **Priority:** high
- **Confidence:** medium
- **Next action:** `after formula: add `bounded_by_domain: {axis: n, bound: 1e7, reason: ...}`, set `state_class: O(d^2 + pending)`, flip status. Evidence path: cadw/ml/coxph.py:455-500; binding not needed (module-level fit_coxph).`

## References

- tests/test_coxph.py
- DEC-011
- DEC-014