# Card: DWT-MLEAD

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_dwt_mlead` (`cadw.ml.dwt_mlead`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `draft`
- **Priority:** low
- **Confidence:** medium
- **Blocking items:** 3 (see §Blocking)
- **Next action:** draft_reason: O(n) output (DEC-013 class); revisit after SpillArray. Code: cadw/ml/dwt_mlead.py:326-342.

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

**File:** `cadw/ml/dwt_mlead.py` (exists: yes; 391 lines)
**Function:** `fit_dwt_mlead` at (class method)
**Recent git history (3 commits):** `9365f2a v0.9.90-S3: 11 G2 (guard-after-alloc) fixes in ML _collect family` / `a12b6bb v0.9.87-P4b: dwt_mlead streaming mahalanobis (audit HIGH-1)` / `e10cf73 v0.9.87-fix1b: guard fixes for 4 windowed TS anomaly modules (audit HIGH-1)`

| Check | Result |
|---|---|
| `state_nbytes()` | no |
| `batch_size kwarg` | no - fit_dwt_mlead:261 |
| `check_state calls` | 308 (dwt_mlead.state) |
| `check_batch calls` | 0 |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | yes - 82 details.append, 162 scores_chunks.append |
| `np.concatenate/vstack/column_stack` | 165 np.concatenate(scores_chunks) |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_dwt_mlead.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n) outputs | see lies |
| `state_type` | see inventory | single-pass DWT | see lies |
| `bounded_state` | see inventory | np.zeros(n) x3 | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {batch} misleading | see lies |

## YAML lies detected

(none)

## APs detected

- **AP1** at `cadw/ml/dwt_mlead.py:326-328` - `np.zeros(n)` arrays for anomaly_scores/is_anomaly/counts

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "UNDERIVABLE - reason: est_bytes_pre at dwt_mlead.py:302-308 is not visible as a single expression in the read window; output is O(n) (anomaly_scores + is_anomaly + counts at 326-342)."
formula_evidence_hypothesis:
  n output: "dwt_mlead.py:326-342 - anomaly_scores np.zeros(n), is_anomaly np.zeros(n), anomaly_counts np.zeros(n)"
  guard: "dwt_mlead.py:307-308 - guard.check_state(est_bytes_pre)"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | NO | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] O(n) output arrays - dwt_mlead.py:326-342 (DEC-013 class)
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing

## Recommendation

- **Bucket:** `draft`
- **Rationale:** DWT-MLEAD returns per-point scores and labels plus per-level counts, so the output path is O(n).
- **Priority:** low
- **Confidence:** medium
- **Next action:** `draft_reason: O(n) output (DEC-013 class); revisit after SpillArray. Code: cadw/ml/dwt_mlead.py:326-342.`

## References

- tests/test_dwt_mlead.py
- DEC-013