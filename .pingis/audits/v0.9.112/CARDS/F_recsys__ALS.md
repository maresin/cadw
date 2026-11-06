# Card: ALS

**YAML:** `docs/ml_tools/F_recsys.yaml`
**Function:** `fit_als` (`cadw.ml.als`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** medium
- **Confidence:** medium
- **Blocking items:** 4 (see §Blocking)
- **Next action:** cheap flip: formula (U+I)*n_factors*f64*2 + 8.0*batch*row_size, fix bounded_state_dims to {U, I, n_factors}, add component-wise test.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(N·d)` |
| `state_type` | `partitioned` |
| `bounded_state` | `true` |
| `n_scalability` | `linear` |
| `bounded_state_dims` | `{"N": 1, "d": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["hash_shuffle", "external_sort", "reduce"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `absent` |

## Code inventory

**File:** `cadw/ml/als.py` (exists: yes; 235 lines)
**Function:** `fit_als` at (class method)
**Recent git history (3 commits):** `2a12956 v0.9.87-P5a-fix-als: streaming dict build + streaming objective` / `1bf5ebb v0.9.46-fix2: ALS item-sweep sort-key bug (found via M2)` / `a9e559e v0.9.36: Phase D group 2/7 - ALS (D.2.2), hybrid MF (D.2.3), CoxPH (D.3.1)`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - ALSModel.state_nbytes:56 |
| `batch_size kwarg` | no - fit_als:151 |
| `check_state calls` | 183 (als.state) |
| `check_batch calls` | 120 (als.objective 8.0), 174 (als.dict 8.0) |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no (streaming sets since P5a-fix-als) |
| `np.concatenate/vstack/column_stack` | 166 comment: pre-fix O(n) concat removed |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_als.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O((U+I)*k) | see lies |
| `state_type` | see inventory | partitioned | see lies |
| `bounded_state` | see inventory | streaming sets | see lies |
| `n_scalability` | see inventory | grouped | see lies |
| `bounded_state_dims` | see inventory | {N,d} wrong names | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "(U+I)*n_factors*f64*2 + 8.0*batch*row_size + dict(U+I entries)"
formula_evidence_hypothesis:
  (U+I)*k*8: "als.py:183 - mg.check_state((U+I)*n_factors*8)"
  X,Y factors: "als.py:194-195 - rng.uniform size=(U,k) and (I,k)"
  8.0*batch*row_size: "als.py:174 - mg.check_batch(extra_factor=8.0)"
  sets/dicts: "als.py:170-189 - u_set/i_set/u_pos/i_pos O(U+I)"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | YES | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | PARTIAL | see formula/evidence above; assessed by code inspection |
| n_groups | YES | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] `bounded_state_dims {N,d}` uses undefined names (formula names U,I,n_factors)
- [ ] No verification test

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** ALS was already made n-OOC in v0.9.87-P5a-fix-als (streaming dict + streaming objective at als.py:165-174); remaining state is O((U+I)*k) factors. Formula is easy to write from the existing guard.
- **Priority:** medium
- **Confidence:** medium
- **Next action:** `cheap flip: formula (U+I)*n_factors*f64*2 + 8.0*batch*row_size, fix bounded_state_dims to {U, I, n_factors}, add component-wise test.`

## References

- tests/test_als.py
- tests/test_iforest_verification.py (template)