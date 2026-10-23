# D.1.4 — Fairness metrics (DP / EO / TPR-gap / FPR-gap / disparate impact)

**Version:** v0.9.35 · **Module:** `cadw/ml/fairness.py` · **Mixin:** `df.fairness_metrics(...)`
**Complexity:** S · **State:** O(K) groups, independent of N

## Paper / sources

- Hardt, Price, Srebro, *Equality of Opportunity in Supervised Learning*,
  NeurIPS 2016, arXiv:1610.02413. Defines equalized odds / equal opportunity.
- fairlearn `MetricFrame` (BSD-3) — reference formulas for
  `demographic_parity_difference`, `equalized_odds_difference`,
  `true_positive_rate_difference`, `false_positive_rate_difference`.
- Barocas, Hardt, Narayanan, *Fairness and Machine Learning* (fairmlbook.org), ch. 2-3.

## Definitions implemented

Per group g, from TP/FP/FN/TN counts:

```
PPR_g = (TP_g + FP_g) / n_g          # positive prediction rate
TPR_g = TP_g / (TP_g + FN_g)         # recall / true positive rate
FPR_g = FP_g / (FP_g + TN_g)         # false positive rate

demographic_parity_difference = max_g PPR_g - min_g PPR_g
equal_opportunity_difference   = max_g TPR_g - min_g TPR_g
tpr_gap                        = equal_opportunity_difference (alias)
fpr_gap                        = max_g FPR_g - min_g FPR_g
disparate_impact               = min_g PPR_g / max_g PPR_g   (4/5 rule: >= 0.8)
```

Degenerate groups (no positives for TPR, no negatives for FPR) give NaN and
are ignored by `nanmax`/`nanmin` — matching fairlearn behaviour.

## Primitive composition

**Reduce-compatible, but implemented as a single pass with Arrow group-by.**
Each batch:

1. `check_batch(batch, extra_factor=2.0)` — the extra factor covers the 4
   int64 mask columns built from the two binary inputs.
2. Four `pc.and_` masks (TP/FP/FN/TN) cast to int64.
3. `table.group_by("g").aggregate(sum)` — Arrow-native, no row loop.
4. K group keys are pulled with `to_pylist()`; this is a **K-sized** list
   (guarded by the `check_batch` above), not an N-sized one.
5. Counts are merged into a Python dict; `check_state(K * 160)` after each
   batch bounds group cardinality (raises `MemoryLimitExceeded`).

HashShuffle is not needed: group cardinality is small by assumption, and
Sort is not needed: the aggregate is order-independent.

## RAM model

| Component | Bytes |
|---|---|
| Per-batch peak | `batch.nbytes * 2.0` + 4 × int64 masks (32 B/row) |
| State | `K * 160` (dict entry + key + 4-int list) |
| Output | O(K) rates + 5 scalars |

No `to_pylist()` over N. Guard tested with 20 000 distinct groups under a
4 MB budget → `MemoryLimitExceeded` (test_guard_high_cardinality_groups).

## Benchmark (measured, not estimated)

`scripts/bench_fairness.py` → `.pingis/benches/fairness.json`, 8 groups,
2 int64 label columns, 50 000-row batches:

| N | time | peak RSS delta | state |
|---|---|---|---|
| 100 000 | 0.029 s | 8.5 MB | 1 280 B |
| 1 000 000 | 0.098 s | 0.6 MB | 1 280 B |

The 1M row peak is lower than the 100k row peak because the input table is
built from batches and the per-batch working set dominates; the number is
reported as measured, not smoothed. State is exactly constant in N.

Reference comparison: no fairlearn install in this environment, so the
reference test is an independent NumPy brute-force recount of the confusion
matrix (`test_reference_vs_numpy_bruteforce`, 2000 rows, 3 groups) plus a
hand-computed fixture with known TP/FP/FN/TN and exact rate values.

## Tests (11)

Synthetic known confusion · NumPy reference · DP-fair constructed case ·
mixin binding · determinism + to_dict · multi-batch chunk invariance ·
state independent of N · high-cardinality guard · non-binary/missing/empty
errors · AST guard (no pandas, no unguarded to_pylist) · source contract.

## Edge cases

- Nulls in the sensitive column → `ValueError` (silently dropping a group
  would understate the gap).
- Labels outside {0,1} → `ValueError` with the observed range.
- Empty input → `ValueError("fairness: no rows")`.
- Boolean label columns are accepted and cast to int64.
- Single group → all gaps are 0.0 by definition.
