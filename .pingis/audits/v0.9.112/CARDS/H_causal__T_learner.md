# Card: T_learner

**YAML:** `docs/ml_tools/H_causal.yaml`
**Function:** `fit_meta_learner` (`cadw.ml.meta_learners`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `draft`
- **Priority:** medium
- **Confidence:** high
- **Blocking items:** 4 (see §Blocking)
- **Next action:** draft_reason: AP2 O(n*d) collect at meta_learners.py:137-144; needs streaming two-pass refactor + SpillDir before formula. Code: cadw/ml/meta_learners.py:125-150.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(batch)` |
| `state_type` | `iterative` |
| `bounded_state` | `true` |
| `n_scalability` | `constant` |
| `bounded_state_dims` | `{"batch": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["reduce", "spill_dir", "codes_bin"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `absent` |

## Code inventory

**File:** `cadw/ml/meta_learners.py` (exists: yes; 221 lines)
**Function:** `fit_meta_learner` at (class method)
**Recent git history (3 commits):** `3c920d6 v0.9.108: refactor _stack_numeric + _column_to_f64 -> _common.py` / `9365f2a v0.9.90-S3: 11 G2 (guard-after-alloc) fixes in ML _collect family` / `2c30326 v0.9.37: Phase D group 3/7 - T/S-learner (D.3.2), uplift+Qini (D.3.3), MiniRocket (D.4.1)`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - MetaLearnerModel.state_nbytes:80 |
| `batch_size kwarg` | no - fit_meta_learner:154 |
| `check_state calls` | 136 (meta_learner.state), 220 |
| `check_batch calls` | 38/47/118/133 (extra_factor=2.0) |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | yes - 137/138 Xs/ts append (AP2) |
| `np.concatenate/vstack/column_stack` | 144 np.vstack/np.concatenate -> O(n*d) |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_meta_learners.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(n*d) collect | see lies |
| `state_type` | see inventory | iterative | see lies |
| `bounded_state` | see inventory | np.vstack(Xs) | see lies |
| `n_scalability` | see inventory | n/a | see lies |
| `bounded_state_dims` | see inventory | {batch} wrong | see lies |

## YAML lies detected

- **bounded_state / n_scalability** claim true/constant, but `_collect` materializes all rows at `cadw/ml/meta_learners.py:137-144`.

## APs detected

- **AP2** at `cadw/ml/meta_learners.py:137-144` - list.append in batch loop + np.vstack/np.concatenate -> O(n*d)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "UNDERIVABLE - reason: `_collect` at meta_learners.py:131-144 appends every batch to Xs/ts/ys and then np.vstack/np.concatenate -> O(n*d) materialized matrix; peak depends on total n, not batch."
formula_evidence_hypothesis:
  _collect append: "meta_learners.py:137-138 - Xs.append(_stack_numeric(b, cols)); ts.append(...)"
  vstack: "meta_learners.py:144 - return np.vstack(Xs), np.concatenate(ts), np.concatenate(ys)"
  guard: "meta_learners.py:136 - mg.check_state(n*(len(cols)*8+16))"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | NO | see formula/evidence above; assessed by code inspection |
| d | NO | see formula/evidence above; assessed by code inspection |
| K | N/A | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] AP2 O(n*d) materialization in `_collect` - meta_learners.py:137-144
- [ ] `implementation_status.status` missing
- [ ] `ram_formula`/`formula_evidence`/`measurement` missing
- [ ] No verification test

## Recommendation

- **Bucket:** `draft`
- **Rationale:** T-learner delegates to GBDT but pre-collects the entire design matrix in `_collect`, so the constant-n claim is false. This is a real refactor (two-pass or per-arm streaming), not a docs fix.
- **Priority:** medium
- **Confidence:** high
- **Next action:** `draft_reason: AP2 O(n*d) collect at meta_learners.py:137-144; needs streaming two-pass refactor + SpillDir before formula. Code: cadw/ml/meta_learners.py:125-150.`

## References

- tests/test_meta_learners.py
- DEC-012 (same AP2 class)