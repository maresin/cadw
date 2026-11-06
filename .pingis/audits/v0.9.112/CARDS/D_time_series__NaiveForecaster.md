# Card: NaiveForecaster

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `NaiveForecaster` (`cadw.ts.models`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** high
- **Confidence:** high
- **Blocking items:** 4 (see §Blocking)
- **Next action:** needs-patch: change state_class to `O(n)`, bounded_state to false, add bounded_by_domain n<=1e6, formula n*f64+256 from models.py:220; then flip.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(1)` |
| `state_type` | `elementwise` |
| `bounded_state` | `true` |
| `n_scalability` | `constant` |
| `bounded_state_dims` | `{}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["reduce"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `absent` |

## Code inventory

**File:** `cadw/ts/models.py` (exists: yes; 1081 lines)
**Function:** `NaiveForecaster` at (class method)
**Recent git history (3 commits):** `22d37f3 v0.9.90-S5: TS guards — _extract_series + 3 unguarded O(n) allocations` / `36ff03d v0.9.84: GBDT transforms + AnomalyPipeline + anomaly stationarity docs` / `8115ce0 v0.9.83: GBDTForecaster capability-based transforms integration`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - state_nbytes:1008 (inherited) |
| `batch_size kwarg` | yes - _BATCH at 80/104 in _extract_series |
| `check_state calls` | none |
| `check_batch calls` | none |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no |
| `np.concatenate/vstack/column_stack` | 237 np.concatenate (predict only) |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_models.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n) series | see lies |
| `state_type` | see inventory | elementwise | see lies |
| `bounded_state` | see inventory | self.y_ = y | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {} wrong | see lies |

## YAML lies detected

- **state_class / bounded_state_dims / bounded_state** claim `O(1)` / `{}` / true, but `fit` stores the whole series at `cadw/ts/models.py:220` and the class docstring (models.py:187) says "RAM: O(N)".

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "n*f64 + 256"
formula_evidence_hypothesis:
  y_: "models.py:220 - self.y_ = y (full series retained)"
  docstring: "models.py:187-190 - docstring itself says "RAM: O(N) for storing the training series""
  predict O(1): "models.py:227 - np.full(steps, base) at predict"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | NO | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] YAML says O(1) but code is O(n) - models.py:220
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] No verification test

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** NaiveForecaster is a 3-line baseline that retains the training series for in-sample and drift mean. The YAML O(1) claim is false; O(n) is intrinsic and cheap.
- **Priority:** high
- **Confidence:** high
- **Next action:** `needs-patch: change state_class to `O(n)`, bounded_state to false, add bounded_by_domain n<=1e6, formula n*f64+256 from models.py:220; then flip.`

## References

- DEC-011
- DEC-014
- tests/test_models.py