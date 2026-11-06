# Card: ZMS

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_zms` (`cadw.ml.zms`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `draft`
- **Priority:** medium
- **Confidence:** high
- **Blocking items:** 4 (see §Blocking)
- **Next action:** draft_reason: O(n) output (DEC-013); revisit after SpillArray. Code evidence: cadw/ml/zms.py:132-156.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(d)` |
| `state_type` | `windowed` |
| `bounded_state` | `true` |
| `n_scalability` | `constant` |
| `bounded_state_dims` | `{"d": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["strided_view"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `absent` |

## Code inventory

**File:** `cadw/ml/zms.py` (exists: yes; 164 lines)
**Function:** `fit_zms` at (class method)
**Recent git history (3 commits):** `032af3d v0.9.91-S6a: hampel + zms strided+chunked (O(n*w) -> O(chunk*w))` / `09acd3f v0.9.57: Hampel + ZMS baseline implementations (tier1_core, Phase C.6a)`

| Check | Result |
|---|---|
| `state_nbytes()` | no (ZMSResult has no state_nbytes) |
| `batch_size kwarg` | no - fit_zms:77 |
| `check_state calls` | 138 (zms.chunk per chunk) |
| `check_batch calls` | 0 (uses check_state only) |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no |
| `np.concatenate/vstack/column_stack` | 127 np.concatenate (pad) |
| `sliding_window_view` | 128 (view only, no .copy) |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_zms.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n) output arrays | see lies |
| `state_type` | see inventory | windowed | see lies |
| `bounded_state` | see inventory | O(n) scores/labels | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {d} wrong | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "n*f64 + chunk*window_size*f64*2 + n*f64 (labels as int)"
formula_evidence_hypothesis:
  n*f64: "zms.py:132-133/149-150 - mu,sigma np.empty(n_w) and scores = abs(x-mu)/sigma (n,)"
  chunk*window_size*f64*2: "zms.py:138-141 - guard.check_state(nb*window_size*8*2)"
  n*f64 labels: "zms.py:156 - labels = (scores>thr).astype(int)"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | NO | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] DEC-013: `ZMSResult.decision_scores` + `labels` are O(n) output - blocked until SpillArray
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] `bounded_state` true is wrong - output is O(n)

## Recommendation

- **Bucket:** `draft`
- **Rationale:** DEC-013 already ruled this: ZMS returns one score and one label per timestep, so peak/n grows ~linearly. It cannot be verified until SpillArray exists for the output.
- **Priority:** medium
- **Confidence:** high
- **Next action:** `draft_reason: O(n) output (DEC-013); revisit after SpillArray. Code evidence: cadw/ml/zms.py:132-156.`

## References

- DEC-013
- tests/test_zms.py