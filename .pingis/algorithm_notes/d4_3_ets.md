# D.4.3 ETS / Holt-Winters exponential smoothing

## Paper
Hyndman, Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed.,
OTexts 2021, ch. 8. Hyndman, Koehler, Snyder, Grose, *A state space framework
for automatic forecasting using exponential smoothing*, IJF 18, 2002.

## Docs
- `cadw.ts.ets.ETSForecaster(seasonal_period, seasonal='additive'|'multiplicative', trend, damped)`
- `CadwDataFrame.fit_ets(...)` binding.
- Standard ForecasterBase contract: `fit(df, target_col)`, `predict(steps)`,
  `predict_in_sample()`, plus `prediction_interval(steps, level=0.95)`.

## Design
- Additive-error ETS: ETS(A,N,N), ETS(A,A,N), ETS(A,A,A); optional damping.
- Parameters (alpha, beta, gamma, phi) chosen by deterministic coordinate
  grid search minimising the one-step SSE (no RNG).
- Classical-decomposition initial seasonal indices; standard Holt-Winters
  recursion (additive and multiplicative forms).
- log_fit with the SSE; state = y_ + fitted_ + season_ + params.

## Benchmark (`.pingis/benches/ets.json`, seasonal_period=12, additive)
| n | fit+predict s | alpha | beta | gamma | sigma | peak_mb | state_b |
|---|---|---|---|---|---|---|---|
| 500 | 0.18 | 0.10 | 0.05 | 0.30 | 0.610 | 4.9 | 8 352 |
| 5 000 | 1.74 | 0.10 | 0.01 | 0.10 | 0.538 | 0.1 | 80 352 |
| 50 000 | 17.15 | 0.10 | 0.01 | 0.05 | 0.523 | 1.4 | 800 352 |
Reference: statsmodels ExponentialSmoothing(initialization_method="estimated");
MASE on a held-out 12 points: ours 0.298 vs statsmodels 0.338 (n=96).

## RAM model
State = O(n + s): training series, fitted values, seasonal indices. No
matrix factorisation; the grid search is 6x5x5x(1|4) evaluations of an O(n)
recursion, all sequential.
