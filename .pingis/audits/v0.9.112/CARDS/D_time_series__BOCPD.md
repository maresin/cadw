# Card: BOCPD

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_bocpd` (`cadw.ml.bocpd`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** medium
- **Confidence:** medium
- **Blocking items:** 5 (see §Blocking)
- **Next action:** needs-patch: add formula from bocpd.py:259-268, declare O(n) output + bounded_by_domain or spill, then component-wise test.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(model)` |
| `state_type` | `iterative` |
| `bounded_state` | `true` |
| `n_scalability` | `constant` |
| `bounded_state_dims` | `{"k": 1, "d": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["reduce"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `absent` |

## Code inventory

**File:** `cadw/ml/bocpd.py` (exists: yes; 380 lines)
**Function:** `fit_bocpd` at (class method)
**Recent git history (3 commits):** `6ca54bd v0.9.87-fix1c: BOCPD rolling posterior history (audit HIGH-1)` / `079ba41 v0.9.72: BOCPD-StudentT implementation (tier1_core, probabilistic)`

| Check | Result |
|---|---|
| `state_nbytes()` | no |
| `batch_size kwarg` | no - fit_bocpd:122 |
| `check_state calls` | 224 (bocpd.history) |
| `check_batch calls` | 0 |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no |
| `np.concatenate/vstack/column_stack` | 339 np.vstack (rolling history) |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_bocpd.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(max_run) + O(n) outputs | see lies |
| `state_type` | see inventory | iterative | see lies |
| `bounded_state` | see inventory | bounded run-length | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {k,d} missing max_run | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "n*8 (score) + n*8 (argmax) + _keep*(max_run+1)*f64 + 3*(max_run+1)*f64 + chunk"
formula_evidence_hypothesis:
  n*8 score: "bocpd.py:264 - anomaly_score = np.zeros(n)"
  n*8 argmax: "bocpd.py:265 - argmax_run = np.zeros(n, dtype=int64)"
  R_hist: "bocpd.py:268 - R_hist = np.zeros((_keep, max_run+1))"
  R/mu/beta: "bocpd.py:259-262"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | NO | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | PARTIAL | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] O(n) output `anomaly_score` + `argmax_run` - bocpd.py:264-265 (DEC-013 class)
- [ ] R_hist is O(n*(max_run+1)) when `full_posterior` - bocpd.py:268
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] `bounded_state_dims {k,d}` wrong - run-length dim `max_run` not declared

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** BOCPD state is bounded by max_run, but the returned per-timestep score/argmax arrays are O(n), and full_posterior mode stores an O(n*max_run) history.
- **Priority:** medium
- **Confidence:** medium
- **Next action:** `needs-patch: add formula from bocpd.py:259-268, declare O(n) output + bounded_by_domain or spill, then component-wise test.`

## References

- tests/test_bocpd.py
- DEC-013
- tests/test_iforest_verification.py (template)