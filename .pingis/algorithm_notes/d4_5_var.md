# D.4.5 VAR(p) vector autoregression

## Paper
Lutkepohl, *New Introduction to Multiple Time Series Analysis*, Springer 2005,
ch. 2-3. Hamilton, *Time Series Analysis*, Princeton 1994, ch. 11.

## Docs
- `cadw.ts.var.VARModel(p=1)`; `fit(df, target_col)` where target_col is a
  string or a list of endogenous column names.
- `CadwDataFrame.fit_var(...)` binding.
- `forecast_vector(steps)` -> (steps x k); `predict(steps)` (first series,
  ForecasterBase contract); `predict_in_sample()`; `stable_` flag.

## Design
- Lag design matrix [1, Y(t-1), ..., Y(t-p)]; equation-by-equation OLS via
  one `np.linalg.lstsq`.
- `coef_[lag][j, m]` = effect of variable j on equation m; forecasting uses
  `coef_[lag].T @ history`.
- Stability via spectral radius of the companion matrix (must be < 1).
- log_fit with the residual sum of squares.

## Benchmark (`.pingis/benches/var.json`)
| n | k | p | fit+forecast s | stable | peak_mb | state_b |
|---|---|---|---|---|---|---|
| 1 000 | 2 | 1 | 0.001 | True | 0.1 | 16 304 |
| 10 000 | 4 | 2 | 0.007 | True | 2.4 | 320 544 |
| 50 000 | 4 | 2 | 0.039 | True | 12.3 | 1 600 544 |
Reference: statsmodels VAR(Y).fit(p).forecast(...); max abs forecast diff
2.8e-17 on a synthetic VAR(1) (exact agreement), VAR(2) also exact.

## RAM model
State = O(n*k + k^2*p^2): training panel, coefficients, intercept, sigma2.
One lstsq over an (n-p) x (1+kp) matrix; forecasting is O(steps * k^2 * p).
