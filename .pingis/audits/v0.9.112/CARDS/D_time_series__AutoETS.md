# Card: AutoETS

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `AutoETSForecaster` (`cadw.ts.auto_ets`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `tier1_core_exception`
- **Priority:** medium
- **Confidence:** high
- **Blocking items:** 6 (see §Blocking)
- **Next action:** docs-only: bounded_by_domain n<=1e6, formula = inner ETS state from cadw/ts/auto_ets.py:283-286.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(n)` |
| `state_type` | `iterative` |
| `bounded_state` | `false` |
| `n_scalability` | `linear` |
| `bounded_state_dims` | `{"n": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `[]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `present` |

## Code inventory

**File:** `cadw/ts/auto_ets.py` (exists: yes; 287 lines)
**Function:** `AutoETSForecaster` at (class method)
**Recent git history (3 commits):** `ae85c5f v0.9.80: AutoForecastPipeline + capability flags on native handlers` / `5f0bdb2 v0.9.76: D1 uni-forecast — AutoETS + 3-role ensemble + SOTA horizon`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - AutoETSForecaster.state_nbytes:283-286 |
| `batch_size kwarg` | n/a (class .fit) |
| `check_state calls` | none (auto_theta sibling has 94) |
| `check_batch calls` | none |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | configs.append 126, aicc_scores.append 220 (bounded) |
| `np.concatenate/vstack/column_stack` | no |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_auto_ets.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n) inner ETS | see lies |
| `state_type` | see inventory | iterative configs | see lies |
| `bounded_state` | see inventory | best ETS model | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {n} ok | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "3*n*f64 + 256 + configs*8"
formula_evidence_hypothesis:
  inner model state: "auto_ets.py:283-286 - state_nbytes returns best_model_.state_nbytes()"
  config enumeration: "auto_ets.py:126/220 - configs + aicc_scores lists (bounded by config count)"
  binding: "no DataFrame binding found for AutoETS in _ml_bindings.py (uses class directly)"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | NO | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] DEC-011: `tier1_core_exception` is a wart - replace with `bounded_by_domain` (I_survival/D_time_series YAML)
- [ ] n_scalability: linear + bounded_state false is honest, but no `bounded_by_domain` block present
- [ ] AutoETS card `cadw_function: AutoETSForecaster` is a class, not a fit_* function - no DataFrame binding
- [ ] No verification test

## Recommendation

- **Bucket:** `tier1_core_exception`
- **Rationale:** AutoETS wraps ETS over a small fixed config grid; state is the best ETS model so O(n) comes from the retained series. Same bounded_by_domain story as ETS.
- **Priority:** medium
- **Confidence:** high
- **Next action:** `docs-only: bounded_by_domain n<=1e6, formula = inner ETS state from cadw/ts/auto_ets.py:283-286.`

## References

- DEC-011
- DEC-014
- tests/test_auto_ets.py