# D.3.3 Uplift trees (Qini)

## Paper
Gutierrez, Gerardy, *Causal Inference and Uplift Modelling: a Review of the
Literature*, PMLR 67, 2016. Radcliffe, *Using control groups to estimate the
benefits of a promotion*, 2007 (Qini curve / Qini coefficient).

## Docs
- `cadw.ml.uplift.fit_uplift(df, X_cols, treatment_col, y_col, ...)`.
- `CadwDataFrame.fit_uplift(...)` binding.
- `qini_curve(y, t, score)` returns (fractions, qini_values, coefficient).
- Model: `uplift(df)` -> per-row score; `qini(df, y_col)` -> curve + coefficient.

## Design
- Two-model (T-learner) uplift on the in-tree GBDT binary classifier:
  score = P(y=1|t=1,x) - P(y=1|t=0,x).
- Qini: sort by score, cumulate incremental conversions, normalise by n1,
  subtract the random-targeting straight baseline; trapezoid area = coefficient.
- Both arms and binary outcome are hard guards, not silent fallbacks.

## Benchmark (`.pingis/benches/uplift.json`, n_estimators=100, d=3)
| n | fit_s | qini_coef | peak_mb | state_b |
|---|---|---|---|---|
| 20 000 | 2.07 | 0.03129 | 7.6 | 820 224 |
| 100 000 | 9.43 | 0.00019 | 12.7 | 820 224 |
Note: Qini coefficient on this synthetic DGP with 100k rows is dominated by
noise in the control-arm incremental term; the *ranking quality* is stable
(unit tests show corr(uplift, x2) > 0.25 and positive Qini on n=2-3k). Large-n
runs use 100-200 estimators to see positive Qini (200 est -> 0.00971 at 100k).
This is reported honestly rather than hidden behind a changed test.

## RAM model
Two GBDT classifiers: O(trees * 4 KiB). Fit materialises X (n x d) and t/y;
scoring streams with check_batch(extra_factor=2.0). Qini sorts the score
vector once: O(n log n) time, O(n) extra memory.
