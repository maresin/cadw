# Card: Theta

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `ThetaForecaster` (`cadw.ts.theta`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `tier1_core_exception`
- **Priority:** medium
- **Confidence:** high
- **Blocking items:** 5 (see §Blocking)
- **Next action:** docs-only: bounded_by_domain n<=1e6, formula from cadw/ts/theta.py:245; binding evidence cadw/dataframe/_ml_bindings.py:867.

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

**File:** `cadw/ts/theta.py` (exists: yes; 249 lines)
**Function:** `ThetaForecaster` at (class method)
**Recent git history (3 commits):** `54e9e21 v0.9.77: STL decomposition + Theta (STM) on STL` / `4b243c4 v0.9.38: Phase D group 4/7 - PELT (D.4.2), ETS (D.4.3), Theta (D.4.4), VAR (D.4.5)`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - ThetaForecaster.state_nbytes:245 |
| `batch_size kwarg` | n/a |
| `check_state calls` | none |
| `check_batch calls` | none |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no |
| `np.concatenate/vstack/column_stack` | no |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_theta.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n) series+STL | see lies |
| `state_type` | see inventory | iterative | see lies |
| `bounded_state` | see inventory | STL on full series | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {n} ok | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "n*f64*2 + 256"
formula_evidence_hypothesis:
  state: "theta.py:245-247 - state_nbytes (series + decomposition)"
  STL decomp: "theta.py:170-175 - stl_decompose(y) on full series"
  binding: "cadw/dataframe/_ml_bindings.py:867-882 - fit_theta -> ThetaForecaster(...).fit(...)"
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
- [ ] No verification test `tests/test_theta_verification.py`

## Recommendation

- **Bucket:** `tier1_core_exception`
- **Rationale:** Theta (optionally STL) keeps the full series; O(n) is intrinsic to single-series forecasting.
- **Priority:** medium
- **Confidence:** high
- **Next action:** `docs-only: bounded_by_domain n<=1e6, formula from cadw/ts/theta.py:245; binding evidence cadw/dataframe/_ml_bindings.py:867.`

## References

- DEC-011
- DEC-014
- tests/test_theta.py