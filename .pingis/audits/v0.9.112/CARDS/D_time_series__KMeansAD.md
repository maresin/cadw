# Card: KMeansAD

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_kmeans_ad` (`cadw.ml.kmeans_ad`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** medium
- **Confidence:** medium
- **Blocking items:** 4 (see §Blocking)
- **Next action:** needs-patch: formula from kmeans_ad.py:256-306; declare bounded_by_domain on output or spill it; then component-wise test.

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

**File:** `cadw/ml/kmeans_ad.py` (exists: yes; 341 lines)
**Function:** `fit_kmeans_ad` at (class method)
**Recent git history (3 commits):** `f2d5434 v0.9.91-S6b-tspq + S6c-kmeans_ad: chunked OOC` / `763ee5b v0.9.87-fix1b(2/2): guard fixes for 3 more windowed TS modules` / `8e1565c v0.9.73: multi-channel shape support (Level 1)`

| Check | Result |
|---|---|
| `state_nbytes()` | no |
| `batch_size kwarg` | no - fit_kmeans_ad:175 |
| `check_state calls` | none (relies on chunk buffer sizes) |
| `check_batch calls` | 0 |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no |
| `np.concatenate/vstack/column_stack` | no |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_kmeans_ad.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(k*d + S*d + n_w) | see lies |
| `state_type` | see inventory | iterative Lloyd | see lies |
| `bounded_state` | see inventory | chunked | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {batch} ok | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "S*d*f64 + k*d*f64 + n_w*f64 + n_w*i64 + chunk_nw*d*f64*2"
formula_evidence_hypothesis:
  S*d*f64: "kmeans_ad.py:256 - sample = np.empty((S,d)) reservoir, S = max(k*10, min(10000,n_w))"
  k*d*f64: "kmeans_ad.py:275-276 - sums np.zeros((k,d)) + counts np.zeros(k)"
  n_w*f64+i64: "kmeans_ad.py:305-306 - window_scores + labels_per_window np.empty(n_w) (output)"
  chunk*d*2: "kmeans_ad.py:249 - chunk_nw buffers"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | YES | see formula/evidence above; assessed by code inspection |
| d | PARTIAL | see formula/evidence above; assessed by code inspection |
| K | YES | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] O(n_w) output `window_scores`/`labels_per_window` - kmeans_ad.py:305-306 (DEC-013 class)
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] No verification test

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** Chunked Lloyd with bounded reservoir init and k*d centroid state; n is OOC except for the O(n_w) output vector.
- **Priority:** medium
- **Confidence:** medium
- **Next action:** `needs-patch: formula from kmeans_ad.py:256-306; declare bounded_by_domain on output or spill it; then component-wise test.`

## References

- tests/test_kmeans_ad.py
- DEC-013
- tests/test_iforest_verification.py (template)