# Card: KShapeAD

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_kshape` (`cadw.ml.kshape`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** medium
- **Confidence:** medium
- **Blocking items:** 3 (see §Blocking)
- **Next action:** flip after formula+evidence+component-wise measurement; formula already written at kshape.py:443-449.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(batch)` |
| `state_type` | `windowed` |
| `bounded_state` | `true` |
| `n_scalability` | `constant` |
| `bounded_state_dims` | `{"batch": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["memmap", "spill_dir", "strided_view"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `absent` |

## Code inventory

**File:** `cadw/ml/kshape.py` (exists: yes; 519 lines)
**Function:** `fit_kshape` at (class method)
**Recent git history (3 commits):** `f97c22f v0.9.87-P3: kshape streaming (audit HIGH-1 full fix; G3 whitelist now empty)` / `e10cf73 v0.9.87-fix1b: guard fixes for 4 windowed TS anomaly modules (audit HIGH-1)` / `a797207 v0.9.53: KShapeAD implementation (tier1_core, Phase C.2)`

| Check | Result |
|---|---|
| `state_nbytes()` | no |
| `batch_size kwarg` | no - fit_kshape:391 |
| `check_state calls` | 450 (kshape.windows+znorm) |
| `check_batch calls` | 0 |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no |
| `np.concatenate/vstack/column_stack` | 128/153/309 FFT cc concat (bounded) |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_kshape.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(chunk + k) + SpillDir | see lies |
| `state_type` | see inventory | iterative k-means | see lies |
| `bounded_state` | see inventory | chunked + memmap | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {batch} ok | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "chunk_nw*(m*8 + L*16 + k*L*16 + k*(2m-1)*8 + k*8) + n_w*24 + m*m*f64 + k*m*f64 + 16"
formula_evidence_hypothesis:
  per_chunk: "kshape.py:443-446 - _per_chunk formula copied in code"
  n_w*24: "kshape.py:447 - _output = (n-m+1)*24"
  model: "kshape.py:448 - _model = m*m*8 + k*m*8 + 16"
  windows/ffts memmap: "kshape.py:481-492 - SpillDir memmap-backed windows + ffts (OOC)"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | YES | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | PARTIAL | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] No verification test

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** KShape already uses SpillDir memmaps for windows and FFTs and a chunked assignment loop; the per-chunk estimator at 443-446 is a ready-made formula. Only docs/measurement are missing.
- **Priority:** medium
- **Confidence:** medium
- **Next action:** `flip after formula+evidence+component-wise measurement; formula already written at kshape.py:443-449.`

## References

- tests/test_kshape.py
- tests/test_iforest_verification.py (template)