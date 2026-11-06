# Card: LDA

**YAML:** `docs/ml_tools/J_nlp.yaml`
**Function:** `fit_lda` (`cadw.ml.lda`)
**Tier:** `tier1_core`
**Current status:** `(pending)`

## TL;DR
- **Recommendation:** `needs-patch`
- **Priority:** high
- **Confidence:** medium
- **Blocking items:** 6 (see §Blocking)
- **Next action:** flip after adding formula + measurement + fixing bounded_state_dims to {n_topics,vocab_size} in J_nlp.yaml LDA card.

## YAML inventory (from card)

| Field | Value |
|---|---|
| `state_class` | `O(model)` |
| `state_type` | `iterative` |
| `bounded_state` | `true` |
| `n_scalability` | `constant` |
| `bounded_state_dims` | `{"d": 1}` |
| `streaming_strategy` | `present` |
| `primitive_reuse` | `["reduce", "spill_dir"]` |
| `ram_formula` | `MISSING` |
| `formula_evidence` | `absent` |
| `measurement` | `absent` |
| `bounded_by_domain` | `absent` |
| `tier1_core_exception` | `absent` |

## Code inventory

**File:** `cadw/ml/lda.py` (exists: yes; 380 lines)
**Function:** `fit_lda` at (class method)
**Recent git history (3 commits):** `9365f2a v0.9.90-S3: 11 G2 (guard-after-alloc) fixes in ML _collect family` / `ae9f4a0 v0.9.87-P5a-fix-lda-a: sparse E-step (large-vocab OOC)` / `fc10079 v0.9.42: D.3.3 causal tree ensemble + D.6.6 LDA bounded memory`

| Check | Result |
|---|---|
| `state_nbytes()` | yes - LdaModel.state_nbytes:87-90 |
| `batch_size kwarg` | yes arg - fit_lda(..., batch_size=None) at 298 (not propagated to _iter_filtered_batches) |
| `check_state calls` | 345, 360, 368, 379 |
| `check_batch calls` | 342 (extra_factor=3.0) |
| `check_partition_buffers` | none observed |
| `list.append in batch loop` | no (doc_sample.append bounded) |
| `np.concatenate/vstack/column_stack` | 259/260 in transform helper; 362 np.vstack(doc_sample) bounded |
| `sliding_window_view` | no |
| `# BACKLOG: in body` | no |
| `Test file` | yes - tests/test_lda.py, test_lda_bounded_memory.py |
| `Verification test` | no |

## YAML vs code cross-check

| Field | YAML | Code | Match? |
|---|---|---|---|
| `state_class` | see inventory | O(K*V) model | see lies |
| `state_type` | see inventory | multi-pass (max_iter) | see lies |
| `bounded_state` | see inventory | sparse batch E-step | see lies |
| `n_scalability` | see inventory | n/a (single stream) | see lies |
| `bounded_state_dims` | see inventory | {d} wrong, needs {K,V} | see lies |

## YAML lies detected

- **bounded_state_dims** claims `{d: 1}`, but the persistent model is `K*V*f64` (`cadw/ml/lda.py:368`) - dims must be `{n_topics, vocab_size}`.

## APs detected

(none)

## Formula hypothesis (from code, per AP23)

```yaml
ram_formula_hypothesis: "K*V*f64 + max_sample_rows*V*f64 + 3.0*batch*row_size + 6*max(nnz_batch)*K*f64"
formula_evidence_hypothesis:
  K*V*f64: "lda.py:368/87-90 - mg.check_state(K*V*8+512) and LdaModel.state_nbytes"
  max_sample_rows*V*f64: "lda.py:360/362 - doc_sample np.vstack(n_ds*V*8)"
  3.0*batch*row_size: "lda.py:342 - mg.check_batch(b, extra_factor=3.0)"
  6*nnz*K*f64: "lda.py:345 - C.nnz*K*8*6 sparse E-step temp"
```

## OOC verdict

| Axis | Verdict | Evidence |
|---|---|---|
| n | YES | see formula/evidence above; assessed by code inspection |
| d | PARTIAL | see formula/evidence above; assessed by code inspection |
| K | NO | see formula/evidence above; assessed by code inspection |
| n_groups | N/A | see formula/evidence above; assessed by code inspection |

## Blocking items for `status: verified`

- [ ] `implementation_status.status` missing
- [ ] `ram_formula` missing
- [ ] `formula_evidence` missing
- [ ] `measurement` missing
- [ ] `bounded_state_dims` wrong ({d:1} vs {n_topics,vocab_size}) - lda.py:368
- [ ] No verification test `tests/test_lda_verification.py`

## Recommendation

- **Bucket:** `needs-patch`
- **Rationale:** LDA is already OOC by batch with explicit check_batch(extra_factor=3.0) and a sparse E-step; only the YAML dims declaration and formula/measurement block are missing.
- **Priority:** high
- **Confidence:** medium
- **Next action:** `flip after adding formula + measurement + fixing bounded_state_dims to {n_topics,vocab_size} in J_nlp.yaml LDA card.`

## References

- tests/test_lda.py
- tests/test_lda_bounded_memory.py