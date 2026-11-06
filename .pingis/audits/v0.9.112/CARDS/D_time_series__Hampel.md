# Card: Hampel

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_hampel` (`cadw.ml.hampel`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `draft`
- **Priority:** medium
- **Confidence:** high
- **Blocking items:** 4 (see §Blocking)
- **Next action:** draft_reason: O(n) output (DEC-013 same class); revisit after SpillArray. Code: cadw/ml/hampel.py:140-178.

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

**File:** `cadw/ml/hampel.py` (exists: yes; 180 lines)
**Function:** `fit_hampel` at (class method)
**Recent git history (3 commits):** `032af3d v0.9.91-S6a: hampel + zms strided+chunked (O(n*w) -> O(chunk*w))` / `09acd3f v0.9.57: Hampel + ZMS baseline implementations (tier1_core, Phase C.6a)`

| Check | Result |
|---|---|
| `state_nbytes()` | no |
| `batch_size kwarg` | no - fit_hampel:81 |
| `check_state calls` | 150 (hampel.chunk) |
| `check_batch calls` | 0 |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no |
| `np.concatenate/vstack/column_stack` | none |
| `sliding_window_view` | 140 (view only) |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_hampel.py |
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
ram_formula_hypothesis: "n*f64*2 + chunk*window_size*f64*2 + n*f64 (labels)"
formula_evidence_hypothesis:
  n*8*2: "hampel.py:144-145/165-166 - med,mad np.empty(n_w) and scores = abs(x-med)/sigma"
  chunk*window_size*16: "hampel.py:149-151 - guard.check_state(nb*window_size*8*2)"
  n labels: "hampel.py:171 - labels = (scores>threshold).astype(int)"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | NO | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] DEC-013 predicted this: O(n) output (decision_scores, labels) - blocked until SpillArray
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] `bounded_state` true wrong (O(n) output)

## Recommendation

- **Bucket:** `draft`
- **Rationale:** Same class as ZMS: Hampel produces a score and label per timestep. Chunked median is O(chunk*w) but the returned arrays are O(n).
- **Priority:** medium
- **Confidence:** high
- **Next action:** `draft_reason: O(n) output (DEC-013 same class); revisit after SpillArray. Code: cadw/ml/hampel.py:140-178.`

## References

- DEC-013
- tests/test_hampel.py