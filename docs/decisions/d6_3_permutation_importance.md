# D.6.3 Permutation Feature Importance — design

**Status:** implemented v0.9.31 (Phase D.6.3, S)
**Reference:** L. Breiman, *Random Forests*, Machine Learning 45, 2001 §"Variable importance" (permutation-based). sklearn docs: `inspection.permutation_importance` (BSD).

## Why

Model-agnostic feature attribution. Works on any fitted model with a
`predict(np.ndarray) -> np.ndarray` interface: GBDT, linear, logistic,
kmeans (score shift), iforest (score shift). Uses the CADW out-of-core
paradigm by only materializing a **bounded sample** of the input, so
importances are computed on `max_samples` rows (default 1000).

## Algorithm

1. **Draw a bounded sample.** Reservoir sample `S = min(max_samples, N)`
   rows from `df[X_cols (+ y_col)]`. `O(S)` state.
2. **Baseline score.** Compute the reference metric on the unshuffled
   sample:
   - regression: `R^2` (or provided `metric` callable)
   - classification: accuracy
   - unsupervised (`y_col=None`): mean absolute prediction shift = 0
3. **For each feature `j`:** repeat `n_repeats` times:
   - shuffle column `j` (in place on a **copy** of the sample)
   - recompute the score
   - importance = baseline - shuffled (for R^2 / accuracy: drop)
4. **Report** mean and std over repeats per feature.

## Complexity

| Phase | Time | RAM |
|---|---|---|
| sample | O(N) one pass | O(S) |
| importance | O(d * n_repeats * S * predict_cost) | O(S * d) |

No O(N) materialization anywhere — the row sample cap `S` bounds state.

## API

```python
permutation_importance(model, df, X_cols, *, y_col=None,
                       n_repeats=5, max_samples=1000,
                       metric="r2" | "accuracy" | None,
                       random_state=42,
                       memory_guard=None) -> pa.Table
# columns: feature, importance_mean, importance_std
```

## Edge cases

- `S < 2` -> ValueError (need at least 2 rows to score).
- Constant feature -> zero importance (shuffle does not change the value).
- `y_col=None` + classifier (predict returns labels) -> accuracy is
  undefined; use `metric=None` (prediction-shift mode).
- Deterministic seed -> reproducible rows.

## Test plan (>=8 tests)

1. Smoke: returns Table[feature, importance_mean, importance_std].
2. On a synthetic dataset where feature 0 strongly predicts y and
   feature 1 is pure noise -> imp[0] >> imp[1].
3. `n_repeats=1` vs `5` -> same ranking.
4. Determinism (same seed).
5. `S < 2` -> ValueError.
6. Constant feature -> importance near 0.
7. Ref: sklearn `inspection.permutation_importance` Spearman >= 0.7 on a
   synthetic linear dataset.
8. Multi-chunk input (chunk=100 vs 1000) -> same ranking.
9. `metric="accuracy"` on a classifier.
10. `y_col=None` -> prediction-shift mode.
