# Card: SARIMA

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `SARIMAForecaster` (`cadw.ts.models`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `tier1_core_exception`
- **Priority:** low
- **Confidence:** medium
- **Blocking items:** 5 (see §Blocking)
- **Next action:** docs-only: bounded_by_domain n<=1e6 + formula from models.py:334/494; add tests/test_sarima.py.

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

**File:** `cadw/ts/models.py` (exists: yes; 1081 lines)
**Function:** `SARIMAForecaster` at (class method)
**Recent git history (3 commits):** `22d37f3 v0.9.90-S5: TS guards — _extract_series + 3 unguarded O(n) allocations` / `36ff03d v0.9.84: GBDT transforms + AnomalyPipeline + anomaly stationarity docs` / `8115ce0 v0.9.83: GBDTForecaster capability-based transforms integration`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - state_nbytes:1008 (delegates to inner pipeline) |
| `batch_size kwarg` | yes - _BATCH const used in _iter_arrow_batches at 80/104 |
| `check_state calls` | none (models.py:698 gbdt lag matrix only) |
| `check_batch calls` | none |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | yes - y_ext.append 429, levels.append 618, all_lags.append 687 |
| `np.concatenate/vstack/column_stack` | 237 np.concatenate (predict full) |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_models.py, test_ts_models.py (no test_sarima.py) |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n + n*p) | see lies |
| `state_type` | see inventory | iterative HR-OLS | see lies |
| `bounded_state` | see inventory | np.empty((n-p,p)) | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {n} ok | see lies |

## YAML lies detected

(none)

## APs detected

- **AP1** at `cadw/ts/models.py:334/366` - `np.empty((n-p, p))` full design matrix for Hannan-Rissanen OLS

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "n*f64*2 + (n-p)*p*f64 + 256"
formula_evidence_hypothesis:
  y_raw_: "models.py:494 - self.y_raw_ = y.copy() (O(n))"
  centered: "models.py:502 - self._y_centered_ = y - mean (O(n))"
  design matrix: "models.py:334/366 - X = np.empty((n-p, p)) Hannan-Rissanen OLS"
  extract: "models.py:47-110 - _extract_series with batch_size=_BATCH (streamed)"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | PARTIAL | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] O(n) retained series + O(n*p) design matrix - models.py:334/366/494-502
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] DEC-011: `tier1_core_exception` -> `bounded_by_domain`
- [ ] No dedicated `tests/test_sarima.py` (only test_models*.py)

## Recommendation

- **Bucket:** `tier1_core_exception`
- **Rationale:** SARIMA is a single-series forecaster: extraction streams, but the model keeps the series and builds an O(n*p) OLS design matrix. O(n) is intrinsic; the honest fix is bounded_by_domain.
- **Priority:** low
- **Confidence:** medium
- **Next action:** `docs-only: bounded_by_domain n<=1e6 + formula from models.py:334/494; add tests/test_sarima.py.`

## References

- DEC-011
- DEC-014
- tests/test_models.py