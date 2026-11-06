# Card: PatchMemoryAD

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_patch_memory_ad` (`cadw.ml.patch_memory_ad`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** medium
- **Confidence:** medium
- **Blocking items:** 4 (see §Blocking)
- **Next action:** needs-patch: formula from patch_memory_ad.py:255-259; spill output or bounded_by_domain; component-wise test.

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

**File:** `cadw/ml/patch_memory_ad.py` (exists: yes; 415 lines)
**Function:** `fit_patch_memory_ad` at (class method)
**Recent git history (3 commits):** `7215019 v0.9.87-P4d: patch_memory_ad via SpillDir+memmap (audit HIGH-1)` / `24db796 v0.9.63-fix2: normalization strategies + honest limitation` / `251faaf v0.9.63: PatchMemoryAD (PaAno-inspired, tier1_core, Phase C.11)`

| Check | Result |
|---|---|
| `state_nbytes()` | no |
| `batch_size kwarg` | no - fit_patch_memory_ad:207 |
| `check_state calls` | 263 (patch_memory_ad.state) |
| `check_batch calls` | 0 |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | training_losses.append 372 (epoch-count) |
| `np.concatenate/vstack/column_stack` | 331 np.vstack (mem warmup, bounded) |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_patch_memory_ad.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(chunk*d + n) | see lies |
| `state_type` | see inventory | iterative | see lies |
| `bounded_state` | see inventory | SpillDir memmap | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {batch} ok | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "max(chunk_nw*d_patches*f64*6, (n_w+n)*16) + 2*f64*(d_p*h + h + h*lat + lat) + 2*f64*n_memory*lat"
formula_evidence_hypothesis:
  per_chunk: "patch_memory_ad.py:255 - _chunk_nw*_d_patches*8*6"
  output_phase: "patch_memory_ad.py:256 - (n_w + n)*8*2"
  model_params: "patch_memory_ad.py:257-258"
  mem_params: "patch_memory_ad.py:259 - n_memory*latent*8*2"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | YES | see formula/evidence above; assessed by code inspection |
| d | PARTIAL | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] O(n) output `(n_w + n)*16` - patch_memory_ad.py:256 (DEC-013 class; SpillDir already used for X_norm)
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] No verification test

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** Uses SpillDir+memmap for normalized windows and a ready-made estimator at 255-259; output phase is O(n).
- **Priority:** medium
- **Confidence:** medium
- **Next action:** `needs-patch: formula from patch_memory_ad.py:255-259; spill output or bounded_by_domain; component-wise test.`

## References

- tests/test_patch_memory_ad.py
- DEC-013
- tests/test_iforest_verification.py (template)