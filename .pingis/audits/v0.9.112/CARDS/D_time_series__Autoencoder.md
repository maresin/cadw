# Card: Autoencoder

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_autoencoder` (`cadw.ml.autoencoder`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** medium
- **Confidence:** medium
- **Blocking items:** 4 (see §Blocking)
- **Next action:** needs-patch: write formula from autoencoder.py:322-327, record O(n_w) output as bounded_by_domain modifier, run component-wise test.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(batch)` |
| `state_type` | `windowed` |
| `bounded_state` | `true` |
| `n_scalability` | `constant` |
| `bounded_state_dims` | `{"batch": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["strided_view"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `absent` |

## Code inventory

**File:** `cadw/ml/autoencoder.py` (exists: yes; 410 lines)
**Function:** `fit_autoencoder` at (class method)
**Recent git history (3 commits):** `63c5bb7 v0.9.91-S6d: autoencoder chunked SGD (bit-identical)` / `763ee5b v0.9.87-fix1b(2/2): guard fixes for 3 more windowed TS modules` / `8e1565c v0.9.73: multi-channel shape support (Level 1)`

| Check | Result |
|---|---|
| `state_nbytes()` | no |
| `batch_size kwarg` | yes arg - fit_autoencoder(..., batch_size=64) at 183 |
| `check_state calls` | 329 (autoencoder.chunked) |
| `check_batch calls` | 0 |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | training_losses.append 291/365 (epoch-count) |
| `np.concatenate/vstack/column_stack` | no |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_autoencoder.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(batch*d*6 + n_w) | see lies |
| `state_type` | see inventory | iterative SGD | see lies |
| `bounded_state` | see inventory | chunked | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {batch} ok | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "16*n_w + batch_size*d*f64*6 + 2*f64*(d*h + h + h*lat + lat + lat*h + h + h*d + d)"
formula_evidence_hypothesis:
  16*n_w: "autoencoder.py:322-324 - permutation (8n_w) + scores (8n_w)"
  batch*d*8*6: "autoencoder.py:324 - batch buffer forward+backward temps"
  params+velocities: "autoencoder.py:324-327 - (d*h+h+h*lat+...) * 8 * 2"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | YES | see formula/evidence above; assessed by code inspection |
| d | PARTIAL | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] O(n_w) output `window_scores` + permutation - autoencoder.py:322-323, 368 (DEC-013 class)
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] No verification test

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** Chunked SGD with an explicit in-code estimator at autoencoder.py:322-327; n is OOC via chunked windows, but permutation+scores are O(n_w).
- **Priority:** medium
- **Confidence:** medium
- **Next action:** `needs-patch: write formula from autoencoder.py:322-327, record O(n_w) output as bounded_by_domain modifier, run component-wise test.`

## References

- tests/test_autoencoder.py
- DEC-013
- tests/test_iforest_verification.py (template)