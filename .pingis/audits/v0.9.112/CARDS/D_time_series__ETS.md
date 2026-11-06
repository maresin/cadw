# Card: ETS

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `ETSForecaster` (`cadw.ts.ets`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `tier1_core_exception`
- **Priority:** medium
- **Confidence:** high
- **Blocking items:** 5 (see §Blocking)
- **Next action:** docs-only: replace `tier1_core_exception` with `bounded_by_domain: {axis: n, bound: 1000000, reason: single-series forecasting}` and add formula `3*n*f64 + 256` from cadw/ts/ets.py:169-172. Binding evidence: cadw/dataframe/_ml_bindings.py:850.

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

**File:** `cadw/ts/ets.py` (exists: yes; 200 lines)
**Function:** `ETSForecaster` at (class method)
**Recent git history (3 commits):** `ae85c5f v0.9.80: AutoForecastPipeline + capability flags on native handlers` / `4b243c4 v0.9.38: Phase D group 4/7 - PELT (D.4.2), ETS (D.4.3), Theta (D.4.4), VAR (D.4.5)`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - ETSForecaster.state_nbytes:169-172 |
| `batch_size kwarg` | n/a (class .fit) |
| `check_state calls` | none |
| `check_batch calls` | none |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no |
| `np.concatenate/vstack/column_stack` | no |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_ets.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n) series | see lies |
| `state_type` | see inventory | iterative grid search | see lies |
| `bounded_state` | see inventory | full series + fitted | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {n} ok | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "3*n*f64 + 256"
formula_evidence_hypothesis:
  3*n*f64: "ets.py:169-172 - state_nbytes = y_.nbytes + fitted_.nbytes + season_.nbytes + 256"
  grid search cost: "ets.py:86 - _grid_search over (alpha,beta,gamma,phi), O(n*grid) time not RAM"
  binding: "cadw/dataframe/_ml_bindings.py:850-865 - fit_ets -> ETSForecaster(...).fit(...)"
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
- [ ] No verification test `tests/test_ets_verification.py`

## Recommendation

- **Bucket:** `tier1_core_exception`
- **Rationale:** Holt-Winters needs the full series and returns it as fitted_/season_ state, so O(n) is intrinsic; the card already carries the right tier1_core_exception text. DEC-011 says the correct field is `bounded_by_domain`.
- **Priority:** medium
- **Confidence:** high
- **Next action:** `docs-only: replace `tier1_core_exception` with `bounded_by_domain: {axis: n, bound: 1000000, reason: single-series forecasting}` and add formula `3*n*f64 + 256` from cadw/ts/ets.py:169-172. Binding evidence: cadw/dataframe/_ml_bindings.py:850.`

## References

- DEC-011
- DEC-014
- tests/test_ets.py