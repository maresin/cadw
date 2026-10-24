# D.3.2 T/S-learner + propensity

## Paper
Kunzel, Sekhon, Bickel, Yu, *Metalearners for estimating heterogeneous
treatment effects using machine learning*, PNAS 116(10), 2019.
Rosenbaum, Rubin, *The central role of the propensity score*, Biometrika
70(1), 1983 (propensity score, positivity/overlap).

## Docs
- `cadw.ml.meta_learners.fit_meta_learner(df, X_cols, treatment_col, y_col, kind=...)`
- `CadwDataFrame.fit_meta_learner(...)` binding in `cadw/dataframe/_ml_bindings.py`.
- Model exposes `cate(df)`, `ate(df)`, `cate_table(df)`, `ipw_ate(df, y_col)`,
  `overlap_diagnostics(e)`, `to_dict()`.

## Design
- T-learner: two GBDT regressors (treated/control arms); CATE = mu1 - mu0.
- S-learner: one GBDT regressor with treatment as an extra feature;
  CATE = mu(x, 1) - mu(x, 0).
- Propensity e(x) via in-tree `fit_logistic` (Newton IRLS); never a silent
  import of statsmodels/sklearn in `cadw/`.
- Overlap diagnostics always returned: n, min, max, share outside [0.05, 0.95].

## Benchmark (`.pingis/benches/meta_learners.json`, n_estimators=50, d=3)
| n | kind | fit_s | ate | peak_mb | state_b |
|---|---|---|---|---|---|
| 20 000 | t | 1.10 | 1.0196 | 8.9 | 570 624 |
| 20 000 | s | 1.08 | 0.9821 | 3.8 | 365 824 |
| 100 000 | t | 4.25 | 1.0204 | 20.2 | 1 210 624 |
| 100 000 | s | 5.53 | 0.9747 | 14.1 | 1 005 824 |
True ATE = 1.0 (tau = 1 + 0.5*x2, E[x2] = 0). Both learners within ~2-3%.

## RAM model
State = trees * ~4 KiB + n * 8 (propensity) + O(1). Fit pass materialises
X (n x d float64) and t/y; prediction streams batches with
check_batch(extra_factor=2.0). No per-batch to_pylist on hot path.
