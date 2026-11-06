# Card: MiniRocket

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_minirocket` (`cadw.ml.minirocket`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** low
- **Confidence:** high
- **Blocking items:** 4 (see §Blocking)
- **Next action:** cheap flip: add formula num_kernels*(8+8)+5*8, fix bounded_state_dims to {num_kernels}, run tests/test_minirocket.py + add verification test.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(batch)` |
| `state_type` | `windowed` |
| `bounded_state` | `true` |
| `n_scalability` | `constant` |
| `bounded_state_dims` | `{"batch": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["strided_view", "reduce"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `absent` |

## Code inventory

**File:** `cadw/ml/minirocket.py` (exists: yes; 129 lines)
**Function:** `fit_minirocket` at (class method)
**Recent git history (3 commits):** `9365f2a v0.9.90-S3: 11 G2 (guard-after-alloc) fixes in ML _collect family` / `fab517f v0.9.89-D4-followup: _series.py + optim.py (missing deps of 0dfdc34)` / `2c30326 v0.9.37: Phase D group 3/7 - T/S-learner (D.3.2), uplift+Qini (D.3.3), MiniRocket (D.4.1)`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - MiniRocketModel.state_nbytes:84 |
| `batch_size kwarg` | no - fit_minirocket:108 |
| `check_state calls` | 128 (minirocket.model) |
| `check_batch calls` | 0 |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | yes - 102 rows.append in transform() (AP2) |
| `np.concatenate/vstack/column_stack` | 105 np.vstack(rows) in transform |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_minirocket.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(num_kernels) | see lies |
| `state_type` | see inventory | single-pass | see lies |
| `bounded_state` | see inventory | fixed kernel list | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {batch} wrong | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "num_kernels*(8 + 8) + 5*8"
formula_evidence_hypothesis:
  num_kernels*16: "minirocket.py:31/52 - make_kernels list of (w,d) tuples + dilations"
  model state: "minirocket.py:84 - MiniRocketModel.state_nbytes"
  guard: "minirocket.py:128 - mg.check_state(model.state_nbytes())"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | N/A | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] `bounded_state_dims {batch:1}` wrong - state is O(num_kernels), not batch
- [ ] transform() accumulates `rows` list -> vstack at 102-105 (AP2, but transform not fit)

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** fit_minirocket only constructs a fixed-size kernel set; state is O(num_kernels) and n-independent. Formula is trivial and the guard already exists at line 128.
- **Priority:** low
- **Confidence:** high
- **Next action:** `cheap flip: add formula num_kernels*(8+8)+5*8, fix bounded_state_dims to {num_kernels}, run tests/test_minirocket.py + add verification test.`

## References

- tests/test_minirocket.py