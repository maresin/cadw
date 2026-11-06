# Card: Sub-PCA

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_sub_pca` (`cadw.ml.sub_pca`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `draft`
- **Priority:** medium
- **Confidence:** medium
- **Blocking items:** 4 (see §Blocking)
- **Next action:** draft_reason: O(n) output via AP2 list concat (DEC-013 class); refactor `_stream_residual_scores` to spill scores, then formula+measure. Code: cadw/ml/sub_pca.py:255-272.

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

**File:** `cadw/ml/sub_pca.py` (exists: yes; 418 lines)
**Function:** `fit_sub_pca` at (class method)
**Recent git history (3 commits):** `4083339 v0.9.87-P4a: sub_pca streaming (audit HIGH-1 full fix)` / `9fc9f8f v0.9.87-fix1a: sub_pca guard order fix (audit HIGH-1)` / `8e1565c v0.9.73: multi-channel shape support (Level 1)`

| Check | Result |
|---|---|
| `state_nbytes()` | no |
| `batch_size kwarg` | no - fit_sub_pca:307 |
| `check_state calls` | 366 (sub_pca.chunk) |
| `check_batch calls` | 0 |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | yes - 269 scores_chunks.append (AP2) |
| `np.concatenate/vstack/column_stack` | 272 np.concatenate(scores_chunks) |
| `sliding_window_view` | 134 (view only) |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_sub_pca.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n_w) scores via AP2 | see lies |
| `state_type` | see inventory | windowed | see lies |
| `bounded_state` | see inventory | list append + concat | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {batch} misleading | see lies |

## YAML lies detected

(none)

## APs detected

- **AP2** at `cadw/ml/sub_pca.py:269` - `scores_chunks.append(...)` in chunk loop, then `np.concatenate` at 272 (bounded by n_w, not chunk); output is O(n)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "w*d*f64 + chunk_nw*d*f64*3 + n_w*f64 (scores)"
formula_evidence_hypothesis:
  w*d*f64: "sub_pca.py:107-120 - _fit_pca_eigh on window matrix (w,d)"
  chunk_nw*d*f64*3: "sub_pca.py:365-366 - guard.check_state(est_bytes) chunked residual temps"
  n_w*f64: "sub_pca.py:269/272 - scores_chunks.append + np.concatenate -> O(n_w) output"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | NO | see formula/evidence above; assessed by code inspection |
| d | PARTIAL | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] O(n) output `scores_chunks` accumulated then concatenated - sub_pca.py:269-272 (SpillArray needed)
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] No verification test

## Recommendation

- **Bucket:** `draft`
- **Rationale:** Sub-PCA streams the residual computation but still accumulates per-window scores in a list and concatenates them, so the output path is O(n).
- **Priority:** medium
- **Confidence:** medium
- **Next action:** `draft_reason: O(n) output via AP2 list concat (DEC-013 class); refactor `_stream_residual_scores` to spill scores, then formula+measure. Code: cadw/ml/sub_pca.py:255-272.`

## References

- DEC-013
- tests/test_sub_pca.py