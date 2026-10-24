# D.5.1 Ordinal logistic cumulative link

## Paper
McCullagh, *Regression Models for Ordinal Data*, JRSS-B 42(2), 1980.
Agresti, *Categorical Data Analysis*, 3rd ed., Wiley 2013, ch. 8. Tutz,
*Regression for Categorical Data*, CUP 2012.

## Docs
- `cadw.ml.ordinal.fit_ordinal_logit(df, X_cols, y_col)`
- `CadwDataFrame.fit_ordinal_logit(...)` binding.
- Model: `predict_proba(df)` (n x K), `predict(df)`, `beta_`, `theta_`.

## Design
- Proportional-odds cumulative logit: P(Y<=k|x) = sigmoid(theta_k - x.beta).
- K-1 ordered thresholds parameterised as [theta_0, log-gaps] so they stay
  strictly increasing without a constrained solver.
- L-BFGS-B minimises the multinomial negative log-likelihood (scipy).
- log_fit with the final nll; state = beta + theta + classes.

## Benchmark (`.pingis/benches/ordinal_logit.json`)
| n | fit_s | iters | acc | peak_mb | state_b |
|---|---|---|---|---|---|
| 5 000 | 0.292 | 12 | 0.764 | 36.4 | 296 |
| 50 000 | 0.582 | 11 | 0.764 | 7.3 | 296 |
| 200 000 | 2.089 | 11 | 0.764 | 30.5 | 296 |
Reference: statsmodels OrderedModel(distr="logit") — beta maxdiff < 1e-2 and
first threshold maxdiff < 1e-2 on synthetic ordinal data (same likelihood).

## RAM model
State = O(d + K). Fit materialises X (n x d) and y (n); one L-BFGS-B run.
Predict streams batches with check_batch(extra_factor=3.0).
