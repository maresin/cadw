# D.4.4 Theta-method forecasting

## Paper
Assimakopoulos, Nikolopoulos, *The theta model: a decomposition approach to
forecasting*, IJF 16(4), 2000. Hyndman, Athanasopoulos, *FPP3*, ch. 12.

## Docs
- `cadw.ts.theta.ThetaForecaster(seasonal=False, seasonal_period=1)`
- `CadwDataFrame.fit_theta(...)` binding.
- ForecasterBase contract: `fit`, `predict`, `predict_in_sample`.

## Design
- OLS trend line (theta=0) and theta=2 curve = 2*y - line.
- Theta=2 curve smoothed by SES; alpha chosen by deterministic grid search
  minimising in-sample one-step SSE.
- Forecast = 0.5 * theta0_fc + 0.5 * theta2_fc (Assimakopoulos-Nikolopoulos
  combination), which is what makes it match statsmodels.
- log_fit with the SSE; state = y_ + fitted_ + params.

## Benchmark (`.pingis/benches/theta.json`)
| n | fit+predict s | alpha | sigma | peak_mb | state_b |
|---|---|---|---|---|---|
| 1 000 | 0.01 | 0.65 | 1.353 | 0.5 | 16 256 |
| 10 000 | 0.09 | 0.65 | 1.349 | 0.4 | 160 256 |
| 100 000 | 0.92 | 0.65 | 1.360 | 6.5 | 1 600 256 |
Reference: statsmodels ThetaModel(period=1); MASE on held-out 8 points:
ours 2.18 vs statsmodels 1.99 on a strong linear trend (both near the same
level; forecast vectors agree within ~0.15).

## RAM model
State = O(n): training series + fitted values. Alpha grid is 19 evaluations
of an O(n) SES recursion, sequential.
