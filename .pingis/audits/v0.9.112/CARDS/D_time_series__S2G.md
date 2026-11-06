# Card: S2G

**YAML:** `docs/ml_tools/D_time_series.yaml`
**Function:** `fit_s2g` (`cadw.ml.s2g`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `draft`
- **Priority:** low
- **Confidence:** medium
- **Blocking items:** 4 (see §Blocking)
- **Next action:** draft_reason: n*8 input + n_nodes^2 graph + O(n) output; needs input streaming + SpillArray. Code: cadw/ml/s2g.py:446-456.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(batch)` |
| `state_type` | `windowed` |
| `bounded_state` | `true` |
| `n_scalability` | `constant` |
| `bounded_state_dims` | `{"batch": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["strided_view", "reduce"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `absent` |

## Code inventory

**File:** `cadw/ml/s2g.py` (exists: yes; 502 lines)
**Function:** `fit_s2g` at (class method)
**Recent git history (3 commits):** `f464656 v0.9.87-P4c: s2g streaming + dedup (audit HIGH-1)` / `e10cf73 v0.9.87-fix1b: guard fixes for 4 windowed TS anomaly modules (audit HIGH-1)` / `c25865b v0.9.60: S2G (Series2Graph) — paper-faithful implementation (Phase C.8)`

| Check | Result |
|---|---|
| `state_nbytes()` | no |
| `batch_size kwarg` | no - fit_s2g:398 |
| `check_state calls` | 456 (s2g.state) |
| `check_batch calls` | 0 |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | yes - 239/240 points/nodes chunks |
| `np.concatenate/vstack/column_stack` | 242/243 np.concatenate -> O(n_w) |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_s2g.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n + n_nodes^2) | see lies |
| `state_type` | see inventory | single-pass | see lies |
| `bounded_state` | see inventory | keeps full n | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {batch} misleading | see lies |

## YAML lies detected

(none)

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "UNDERIVABLE - reason: est_bytes at s2g.py:452-454 includes n*8 (full input series) and n_nodes*n_nodes*8; graph size is data-dependent (n_nodes bounded by quantile binning but not formula-expressed)."
formula_evidence_hypothesis:
  n*8: "s2g.py:452 - est_bytes = int(n*8 + per_chunk + output + W_bounded)"
  W_bounded: "s2g.py:453 - n_nodes*n_nodes*8"
  output: "s2g.py:453 - _n_w*24 (O(n) output)"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | NO | see formula/evidence above; assessed by code inspection |
| d | N/A | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] `est_bytes` keeps full input `n*8` and O(n) output - s2g.py:452-453
- [ ] O(n_nodes^2) graph state - s2g.py:453
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing

## Recommendation

- **Bucket:** `draft`
- **Rationale:** S2G keeps the full input series plus an n_nodes^2 adjacency and an O(n) output; the formula is data-dependent and the n path is not OOC.
- **Priority:** low
- **Confidence:** medium
- **Next action:** `draft_reason: n*8 input + n_nodes^2 graph + O(n) output; needs input streaming + SpillArray. Code: cadw/ml/s2g.py:446-456.`

## References

- tests/test_s2g.py
- DEC-013