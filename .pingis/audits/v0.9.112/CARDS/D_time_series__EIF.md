# Card: EIF

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_eif` (`cadw.ml.eif`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** medium
- **Confidence:** medium
- **Blocking items:** 4 (see §Blocking)
- **Next action:** needs-patch: add DEC-013 note for O(n_w) output and write formula from eif.py:367-372; if output is accepted as bounded-by-domain, flip after component-wise test.

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

**File:** `cadw/ml/eif.py` (exists: yes; 486 lines)
**Function:** `fit_eif` at (class method)
**Recent git history (3 commits):** `f79c323 v0.9.91-S6b-eif: chunked fit (OOC; not bit-identical to pre-refactor)` / `e10cf73 v0.9.87-fix1b: guard fixes for 4 windowed TS anomaly modules (audit HIGH-1)` / `4136a36 v0.9.55: EIF implementation (tier1_core, Phase C.4)`

| Check | Result |
|---|---|
| `state_nbytes()` | no |
| `batch_size kwarg` | no - fit_eif:285 |
| `check_state calls` | 373 (eif.windows) |
| `check_batch calls` | 0 |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | tree_indices.append at 412 (hyperparameter-bound) |
| `np.concatenate/vstack/column_stack` | no |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_eif.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n_est*psi*d + n_w) | see lies |
| `state_type` | see inventory | iterative | see lies |
| `bounded_state` | see inventory | chunked gather | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {batch} ok | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "n_estimators*psi*d*f64 + n_w*f64 + chunk_nw*d*f64*2"
formula_evidence_hypothesis:
  n_est*psi*d*8: "eif.py:367-372 - est_bytes = n_estimators*psi_est*d_pre*8 + n_w_pre*8 + chunk*d*8*2"
  n_w*f64: "eif.py:413/454 - window_scores np.empty(n_w) (output)"
  chunk_nw*d*8*2: "eif.py:372 - chunk buffer term"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | YES | see formula/evidence above; assessed by code inspection |
| d | PARTIAL | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] O(n_w) output `window_scores` - eif.py:413/454 (DEC-013 class; may block true OOC)
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] No verification test

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** EIF has a full in-code estimator at eif.py:367-372 and gathers only hyperparameter-bound unique windows. However it still returns an O(n_w) score vector.
- **Priority:** medium
- **Confidence:** medium
- **Next action:** `needs-patch: add DEC-013 note for O(n_w) output and write formula from eif.py:367-372; if output is accepted as bounded-by-domain, flip after component-wise test.`

## References

- tests/test_eif.py
- DEC-013
- tests/test_iforest_verification.py (template)