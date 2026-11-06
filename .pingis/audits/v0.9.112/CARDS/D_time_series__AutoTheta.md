# Card: AutoTheta

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `AutoThetaForecaster` (`cadw.ts.auto_theta`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `tier1_core_exception`
- **Priority:** medium
- **Confidence:** high
- **Blocking items:** 6 (see §Blocking)
- **Next action:** docs-only: bounded_by_domain n<=1e6, formula from cadw/ts/auto_theta.py:284 and guard at :94.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(n)` |
| `state_type` | `iterative` |
| `bounded_state` | `false` |
| `n_scalability` | `linear` |
| `bounded_state_dims` | `{"n": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `[]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `present` |

## Code inventory

**File:** `cadw/ts/auto_theta.py` (exists: yes; 287 lines)
**Function:** `AutoThetaForecaster` at (class method)
**Recent git history (3 commits):** `22d37f3 v0.9.90-S5: TS guards — _extract_series + 3 unguarded O(n) allocations` / `703a280 v0.9.78: AutoTheta (CV-selected STM/OTM) + external-only benchmark audit`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - AutoThetaForecaster.state_nbytes:284-285 |
| `batch_size kwarg` | n/a |
| `check_state calls` | 94 (3*len(y)*8) |
| `check_batch calls` | none |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | cands.append 168/175, cv_scores.append 239 (bounded) |
| `np.concatenate/vstack/column_stack` | no |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_auto_theta.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n) inner Theta | see lies |
| `state_type` | see inventory | iterative CV | see lies |
| `bounded_state` | see inventory | CV on full series | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {n} ok | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "3*n*f64 + 256"
formula_evidence_hypothesis:
  inner state: "auto_theta.py:284-285 - state_nbytes = best_model_.state_nbytes()"
  CV candidate buffers: "auto_theta.py:94 - mg.check_state(3*len(y)*8)"
  candidates: "auto_theta.py:168/175 - cands list (bounded)"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | NO | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] DEC-011: `tier1_core_exception` is a wart - replace with `bounded_by_domain` (I_survival/D_time_series YAML)
- [ ] n_scalability: linear + bounded_state false is honest, but no `bounded_by_domain` block present
- [ ] AutoTheta card `cadw_function: AutoThetaForecaster` is a class, not fit_* - no DataFrame binding
- [ ] No verification test

## Recommendation

- **Bucket:** `tier1_core_exception`
- **Rationale:** AutoTheta runs CV over Theta configs on the full series; state is the best model so O(n). It already has an explicit guard at auto_theta.py:94.
- **Priority:** medium
- **Confidence:** high
- **Next action:** `docs-only: bounded_by_domain n<=1e6, formula from cadw/ts/auto_theta.py:284 and guard at :94.`

## References

- DEC-011
- DEC-014
- tests/test_auto_theta.py