## [Unreleased]

### v0.9.85 (2026-10-27) - Forecast metrics extension (interval + robust + quantile)

Extends cadw/ts/backtest.py with 7 new metric families used by
TSB-AD/TAB/fev-bench evaluations. Complements the existing 5 point
metrics (MAE, RMSE, MAPE, sMAPE, MASE).

New metrics (cadw/ts/backtest.py, ~200 LOC + refactor of _mase):

  Point metrics:
    WAPE    = sum|y-f| / sum|y| * 100; robust to zero actuals (unlike
              MAPE, which NaNs); NaN when sum|y|=0.
    MQLIKE  = mean(y/f - log(y/f) - 1); variance-robust for volatility
              forecasts; requires y>0, f>0; NaN otherwise.

  Interval metric:
    MSIS    = Mean Scaled Interval Score (Gneiting & Raftery 2007):
              mean[(hi-lo) + (2/alpha)*(miss_lo) + (2/alpha)*(miss_hi)]
              / naive_MAE(m).
              Requires forecaster.prediction_interval(steps, level).
              Skipped (warning) if unsupported. Configurable via new
              msis_alpha parameter (default 0.05 => 95% interval).

  Quantile metrics (require `quantiles` parameter + forecaster support):
    pinball  = per-q pinball loss (existing; now complemented by
               aggregations below).
    WQL      = weighted quantile loss: mean over q of w(q)*pinball(q)
               with w(q)=2q for q<=0.5, 2(1-q) for q>0.5. Primary
               probabilistic metric in fev-bench.
    SQL      = WQL / naive_MAE(m); scale-free WQL for cross-series
               comparison.
    CRPS     = 2 * mean pinball over quantile levels; approximate
               CRPS from quantile losses (not exact integral).

  Helper refactor: _mase now delegates its scale computation to new
  _mase_scale helper (shared by MSIS/SQL).
  _predict_interval helper wraps forecaster.prediction_interval.

Integration (backtest_forecaster):
  - `metrics` validation extended from 5 to 11 valid names.
  - `msis_alpha=0.05` new kwarg (must be in (0, 1)).
  - If wql/sql/crps requested without `quantiles`, emit warning and
    drop them from the effective metric set (do not silently compute
    wrong values).
  - If MSIS requested but forecaster lacks prediction_interval: emit
    warning once per fold sequence, drop msis from summary (row absent).
  - Empty per_metric lists (e.g. msis unsupported) are skipped from
    summary (no NaN mean row).

12 new tests (tests/test_ts_backtest.py, total now 22 + 1 skip):
  wape_perfect_forecast_zero (positive value); wape_zero_actuals_nan;
  mqlike_perfect_zero (constant y=2, naive=2 -> 0); mqlike_negative_y_
  nan; msis_skipped_for_naive (warning + row absent); msis_perfect_
  interval_zero (Spy PI on y_train with variation -> 0);
  msis_nonzero_interval (skip if scale=0); wql_requires_quantiles
  (warning + drop); wql_crps_sql_from_quantiles (Spy quantile=2 on
  y=2 -> all ~0); invalid_metric_raises; msis_alpha_validation
  (must be in (0,1)); wape_matches_hand_compute (Spy predicts 0 ->
  WAPE=100 on y=1).

  Spy test doubles (_SpyPI, _SpyQuantile) isolate interval and
  quantile paths from real forecaster specifics.

YAML D1 metrics (+7 diagnostic): mape, rmse, mae, wape, mqlike,
msis, pinball (per-q), wql, sql, crps. Anchor remains MASE; smape
unchanged. Total D1 metrics now: 1 anchor + 12 diagnostic (was 1 + 2).
MATRIX.md + 40 docs/tasks/*.md regenerated.

Fixup (v0.9.85-fix1): initial YAML patch used a multi-line plain
scalar with a colon-space inside (`quantile losses: CRPS ~ ...`) —
YAML scanner rejected it. Collapsed CRPS responsibility/special to
single lines; render_matrix OK.

Version 0.9.84 -> 0.9.85 (skipped HybridETS_GBDT for now; hybrid is
M4-specialized marginal gain over AutoTheta, lower priority than
metrics — needed for ALL future forecast sections).

Fast suite: 3522 -> 3534 passed (+12), 8 skipped. Focused:
test_ts_backtest 22 passed + 1 skipped. WL 34 OK. AST-guard 129
passed. audit_leakage 7 warnings (pretrained). audit_url_log 0
errors.

Backlog: 16 items unchanged (no item closed, no item added).

Next: v0.9.86 — HybridETS_GBDT (M4 winner idea: ETS fit -> residual
-> GBDT on residual features); OR pivot to DU-1 (supervised tabular
section) which needs the pipeline + metrics now in place.

### v0.9.84 (2026-10-27) - AnomalyPipeline + GBDT transforms integration

Closes the anomaly-stationarity gap (backlog item 3) and wires GBDT
into the transforms foundation. P0 documentation and P1 code from
the v0.9.77b backlog decision.

GBDTForecaster transforms integration (v0.9.83, cadw/ts/models.py
+~120 LOC, tests/test_gbdt_transforms.py +14 tests):
  New kwargs: differencing='none'|'auto'|int>0, decompose='none'|
  'stl'|'mstl', decompose_periods=list[int]|None. Defaults preserve
  v0.9.82 API (no-op).
  - differencing='auto': AutoStationarizer wraps y before GBDT fit;
    fixes GBDT's inability to extrapolate trend (Pritzsche: GBDT/
    GBM/gamboost cannot forecast trend without detrending).
  - decompose='stl'|'mstl': STL/MSTL wraps y; seasonality removed
    before fit, added back at predict.
  - _build_inner_pipeline uses lazy import (avoid circular dep —
    transforms.py imports ForecasterBase from models.py).
  - Instance-level capability flags (handles_trend/seas/multi)
    override class defaults; with transforms enabled GBDT reports
    True so outer AutoForecastPipeline does not double-wrap (verified
    via pipeline_idempotent_with_autoforecastpipeline test).
  - predict_in_sample raises NotImplementedError in transform mode
    (length mismatch: GBDT NaN-padded n-length vs transforms fit on
    n-d). Naked mode unchanged.

AnomalyPipeline (v0.9.84, cadw/ml/anomaly_pipeline.py, ~250 LOC,
tests/test_anomaly_pipeline.py +21 tests):
  Wraps any anomaly detector function (signature
  detector_fn(series, **kwargs) -> result with .decision_scores and
  .labels). Three modes:
    - 'structural': detect on raw; suits level/trend/seasonal-shift
      anomalies; native changepoint detectors (PELT/BOCPD/BinSeg/
      CUSUM/T-squared+Q) already handle trend as signal.
    - 'contextual': STL decompose (if seasonal_period>=2) or
      Differencer(lags=1) (if detrend=True); run detector on residual;
      map scores back 1:1 (STL preserves length) or with first
      timestamp unknown (differencing).
    - 'auto': contextual if transform available, else structural.
  Addresses backlog item 3 (v0.9.77b): 11 of 15 D3.x detectors assume
  stationary baseline; on trended/seasonal series they produce false
  positives. Verified in test: on pure seasonal y=5*sin(2πt/12),
  structural naive threshold flags >=10 points; contextual STL flags
  0 (residual near zero). On linear trend + spike, contextual detrend
  isolates spike (<=3 flagged) vs structural (~50 trend points
  flagged).
  Result object: AnomalyPipelineResult(decision_scores (n,), labels
  (n,), mode, transforms_applied, base_result, threshold).
  get_params records resolved mode and transforms for audit.
  Pure numpy; lazy import of transforms and _decomposition (AST-guard
  enforced).

P0 documentation (D_time_series.yaml):
  D3.1 / D3.2 / D3.3 subtype descriptions extended with stationarity
  notes:
    D3.1: 11 detectors assume stationary; use AnomalyPipeline
          contextual (v0.9.84) on trended/seasonal series.
    D3.2: all detectors model trend natively; use structural mode,
          do NOT wrap in contextual (would erase signal).
    D3.3: shape anomalies are contextual by nature; contextual STL
          preferred for Sub-PCA/KShapeAD/S2G.
  MATRIX.md + 40 docs/tasks/*.md regenerated.

Design notes:
  - Detectors are functions (not classes) in cadw/ml/; pipeline
    therefore takes detector_fn + kwargs, not an object with capability
    flags. A registry of flags per detector would add complexity
    without benefit — caller chooses mode explicitly.
  - STL preserves length -> 1:1 mapping is exact. Differencing loses
    first timestamp (score[0]=0, label[0]=0 — "unknown").
  - AnomalyPipeline can wrap any tier1 detector, but its core value
    is for the 11 stationary-assuming ones. Native changepoint
    detectors work equally well naked in structural mode.

Version 0.9.82 -> 0.9.84.

Backlog: item 3 (Anomaly stationarity) closed via P0+P1.

Fast suite: 3486 -> 3521 passed (+35: GBDT 14 + AnomalyPipeline 21),
8 skipped. Focused: test_gbdt_transforms 14 passed,
test_anomaly_pipeline 21 passed, regression 152 passed (AST 129 +
decomposition 16 + misc). WL 34 OK. AST-guard 129 passed.
audit_leakage 7 warnings (pretrained). audit_url_log 0 errors.
MATRIX.md + 40 docs/tasks/*.md regenerated (P0 doc changes).

Next: v0.9.85 — HybridETS_GBDT (M4 winner idea: ETS fit -> residual
-> GBDT on residual features); then metrics extension (MSIS, pinball,
MQLIKE, WAPE, CRPS) v0.9.86.

### v0.9.82 (2026-10-27) - TS transforms + AutoForecastPipeline + Ensemble

Closes the TS pipeline section (v0.9.79-v0.9.81): composable per-model
target transforms, capability-aware pipeline, and weighted ensemble.
Enables correct handling of trend/seasonality for any forecaster
without double-stationarization, and preserves diversity in ensembles.

Transforms foundation (cadw/ts/transforms.py, ~800 LOC total, v0.9.79a+b):

  BaseTransformer (ABC): fit / transform / inverse_transform /
  get_params contract for stateful target transforms.

  Differencer: regular d and seasonal D differencing with inverse.
    Forward: d regular diffs then D seasonal diffs.
    Inverse (v0.9.79a-fix2): closed-form binomial —
      d-th: y[t] = y_t[t] + sum_{k=1..d} (-1)^(k+1) C(d,k) y[t-k]
      D-th: y[t] = y_t[t] + sum_{k=1..D} (-1)^(k+1) C(D,k) y[t-k*s]
    Initial buggy loop didn't track tails for d>1/D>1; replaced with
    binomial closed-form (verified on d=2 quadratic).

  TransformedTargetForecaster: pipeline wrapper. Fit: apply transforms
  in order, fit base on fully-transformed y. Predict: base.predict
  then inverse in reverse order. predict_in_sample + PI also inverted.

  STLTransformer (v0.9.79b): STL decomposition, transform = y - seasonal,
  inverse = forecast + seasonal. In-sample (roundtrip) uses full
  seasonal; forecast uses last cycle tiled (v0.9.79b-fix1).

  MSTLTransformer (v0.9.79b): iterative STL for multiple periods
  (ascending); transform = y - sum(seasonals); inverse per above.

  AutoStationarizer (v0.9.79b, subclass of Differencer): auto-selects
  d and D. d: iterate while |acf(lag=1)| > acf_threshold (default
  0.8), cap max_d (default 2). D: after d-diff, STL seasonal
  strength F_S = max(0, 1 - Var(R)/Var(S+R)); if F_S > threshold
  (default 0.5), D = 1; cap max_D (default 1). Formal ADF/KPSS in
  backlog; heuristic fast, no scipy/statsmodels dependency.

  38 tests: Differencer (d1/d2/d0 forward, d1/d2 inverse roundtrip,
  seasonal forward/inverse, combined d1_D1, identity recovers last
  value, bad params, short-series raise); STL (removes season,
  inverse roundtrip, forecast continuation, bad period, short raise);
  MSTL (removes both seasonals, roundtrip, bad periods); pipeline
  (basic flow, flat naive-without-diff sanity, PI, capability flags,
  bad params, get_params); AutoStationarizer (random noise d0,
  random walk d1, trending d1, seasonal D1, is-Differencer-subclass,
  transform/inverse after fit, bad params); AST guard.

Capability flags (v0.9.80): handles_trend / handles_seasonality /
handles_multi_seasonality on ForecasterBase models:
  - SARIMAForecaster: True / True / False (natively handles both)
  - ETSForecaster: True / True / False
  - AutoETSForecaster: True / True / False
  - ThetaForecaster (STM), OTMThetaForecaster, AutoThetaForecaster:
    True / True / False (declared in v0.9.77/v0.9.78a/v0.9.78b)
  - NaiveForecaster, GBDTForecaster: flags undeclared -> getattr
    defaults to False; pipeline adds transforms.

AutoForecastPipeline (v0.9.80, in transforms.py): wraps any
ForecasterBase. Reads capability flags; adds transforms only where
the base cannot handle natively. Transform ordering (STL-then-diff,
literature convention for STL+ARIMA):
  1. STLTransformer / MSTLTransformer (if model needs seasonality)
  2. AutoStationarizer (if model needs trend)
If base handles both, pipeline is "naked" (delegates directly).
12 new tests: naive+trend -> differencing (trend continues vs flat),
naive+stationary -> d=0 (matches plain naive), naive+seasonal ->
STL+diff (non-flat), sarima/ets/autotheta naked, capability flags
after wrap, MSTL for 2 periods, empty forecaster raises, get_params,
predict_in_sample shape, AST guard.

EnsembleForecaster (v0.9.81a, cadw/ts/ensemble.py, ~370 LOC):
  Wraps each member in its own AutoForecastPipeline (per-member
  capability-based transforms). Predictions inverse-transformed
  per-member, then weighted-averaged. Preserves diversity: if all
  members see the same detrended series, they produce the same flat
  forecast; inverse adds back the same trend -> zero ensemble gain.

  Weight strategies: 'equal' (mean), 'inverse_mase' (default;
  weights ∝ 1/val_MASE on holdout), 'median' (robust to 1 outlier),
  custom list. Weights validated at __init__ (fail fast).

  Bench (n=500, s=12, M3-like):
    naive_last             MASE=2.184  0.18s
    theta                  MASE=0.201  0.05s
    auto_theta             MASE=0.209  0.23s
    ensemble inverse_mase  MASE=0.203  0.56s
      weights = [0.017 (naive), 0.311 (theta), 0.672 (auto_theta)]
      val_mase = [47.06, 2.59, 1.20]
    auto_ets               MASE=0.276  3.33s
  Ensemble ≈ best-member on single DGP. Honest: on a single synthetic
  DGP ensemble rarely beats best member; expected gain on M3/M4-scale
  multi-DGP from inverse-MASE picking good members per series.

  HONEST LIMITATION (documented in bench + backlog 17): AutoStat
  heuristic (acf_threshold=0.8) chose d=0 on trend+noise DGP, so
  naive+STL gave flat forecast (val MASE=47). Ensemble correctly
  downweighted naive to 0.017 — capability works, but AutoStat
  heuristic itself misleads on raw trended series. Fix in backlog:
  lower acf_threshold 0.8 -> 0.5-0.6, or formal ADF/KPSS.

  Fix iterations: v0.9.81a-fix1 validate weights at __init__ (bad
  spec raised from fit initially); test fixes (NaiveForecaster has
  no 'mean' strategy -> 'drift'; pipeline wraps naive in AutoStat
  so ens != naked-naive avg -> use ThetaForecaster for bare-naked
  pipeline tests).

Version 0.9.78 -> 0.9.82.

Combined with v0.9.79a-b and v0.9.80 (uncommitted between v0.9.78
and v0.9.81a), the full pipeline stack is now usable:
  TransformedTargetForecaster(Differencer, NaiveForecaster) for
  trended series; AutoForecastPipeline(GBDTForecaster) auto-adds
  transforms; EnsembleForecaster([SARIMA, AutoTheta, GBDT]) uses
  per-member pipelines with inverse-MASE weights.

Backlog: 17 items (AutoStationarizer heuristic on trended data
added; AutoTheta k-fold CV from v0.9.78; others).

Fast suite: 3418 -> 3486 passed (+68: transforms 50 + ensemble 16 +
2 misc), 8 skipped. Focused: 234 (transforms 50 + theta 17 + otm 15
+ auto_theta 17 + ensemble 16 + ets/auto_ets 47 + AST 129 net).
WL 34 OK. AST-guard 129 passed. audit_leakage 7 warnings (pretrained).
audit_url_log 0 errors. MATRIX.md + 40 docs/tasks/*.md unchanged
(no card changes in this cycle).

Next: v0.9.83 — GBDTForecaster(differencing='auto', decompose='stl')
integration; then AnomalyPipeline (P0+P1 anomaly stationarity);
then HybridETS_GBDT; then metrics extension.

### v0.9.78 (2026-10-27) - OTM + AutoTheta (CV-selected STM/OTM)

Closes D1 uni-forecast with SOTA-class classical: OTM (weight-
optimized STM) and AutoTheta (CV-selected STM/OTM x additive/
multiplicative). AutoTheta replaces AutoETS in the 3-role ensemble.

OTM (cadw/ts/otm.py, ~280 LOC, v0.9.78a):
  OTMThetaForecaster: STM with theta=2 fixed, (w2, alpha) grid search
  on in-sample MSE. Practical variant of Fiorucci et al. (2016) OTM;
  full OTM (theta optimization + explicit drift) in backlog.
  Capability flags (handles_trend/seas/multi=False).
  15 tests: OLS/SES helpers, contract, weights selection on trend,
  competitive MASE, no-worse-than-STM on trend, determinism,
  PI, capability flags, AST guard.

  Bench (n=500, s=12):
    M3-like:          OTM 0.209  STM 0.201  AutoETS 0.276
    Trend-dominated:  OTM 0.296  STM 0.652
  OTM 2.2x better than STM on trend-dominated (targeted fix).
  Honest limitation: OTM selects w2=0.1 (grid boundary) on both DGPs;
  on M3-like slightly worse than STM (0.209 vs 0.201) due to in-sample
  overfit toward OLS.

AutoTheta (cadw/ts/auto_theta.py, ~380 LOC, v0.9.78b):
  AutoThetaForecaster: evaluates 4 candidates (STM/OTM x additive/
  multiplicative), holdout MASE selection, refit winner on full
  series. Additive via STL; multiplicative via log-transform + additive
  STL. Skips multiplicative when y<=0. Fallback Naive if no candidate.
  17 tests: helpers (naive seasonal MAE, MASE), contract, selection
  min-MASE, M3-like competitive, trend-dominated competitive, skip
  multiplicative on negative, determinism, val_steps explicit,
  short-series raise, PI, fallback, capability flags, AST guard.

  Bench (n=500, s=12):
    M3-like:          AutoTheta 0.209 (selected OTM)
                      vs AutoETS 0.276 (-24%), 1.4s vs 38.6s (-96%)
    Trend-dominated:  AutoTheta 0.296 (selected OTM)
                      vs STM 0.652 (2.2x)
  AutoTheta beats AutoETS 24% on M3-like and 30x faster.
  HONEST LIMITATION: on M3-like, CV-holdout selected OTM (val MASE
  1.15) but STM generalized slightly better on test (0.201 vs 0.209).
  Holdout-selection risk documented. K-fold CV is backlog.

D1 ensemble: [SARIMA, AutoETS, GBDT] -> [SARIMA, AutoTheta, GBDT].
  AutoTheta replaces AutoETS in ensemble:
  - both expose temporal_dynamics+probabilistic; AutoTheta has better
    fev MASE (1.381 vs AutoETS 1.525) and is 5-30x faster
  - AutoETS remains as standalone verified card
  - max_per_role=2 unchanged (SARIMA+AutoTheta both temporal_dynamics)
  - roles: temporal_dynamics + probabilistic + nonlinear_interactions

D1 verified count: 6 -> 8 (Naive, ETS, AutoETS, Theta, AutoTheta,
SARIMA, GBDTForecaster + 1 more). AutoTheta tier=sota, Theta tier=
baseline (documented limitation).

Version 0.9.77 -> 0.9.78.

Backlog: 15 items. Added "AutoTheta: k-fold CV instead of single
holdout (mitigate selection risk)" to backlog.

Fast suite: 3384 -> 3418 passed (+34: OTM 15 + AutoTheta 17 + 2 fev note tinkering).
Focused: 3416 passed. WL 34 OK. AST-guard 129 passed.
audit_leakage 7 warnings (pretrained). audit_url_log 0 errors.
MATRIX.md + 40 docs/tasks/*.md regenerated.

Next: v0.9.79 — Croston (intermittent demand) OR transforms.py
foundation (Differencer, MSTLTransformer, AutoStationarizer).

### v0.9.77 (2026-10-27) - STL decomposition + Theta (STM) on STL

Adds STL (Seasonal-Trend decomposition using LOESS) as a foundation
module, and rewrites Theta on top of it. Theta (STM variant) is
baseline; OTM/AutoTheta follows in v0.9.78.

STL foundation (cadw/ts/_decomposition.py, ~230 LOC):
  - stl_decompose(y, period, seasonal_window=7, trend_window=None,
    n_iter=2) -> STLTrendResult(trend, seasonal, remainder)
  - Local linear LOESS with tricube weights (Cleveland et al. 1990)
  - Alternating (trend, seasonal) passes; R-formula default
    trend_window = nextodd(ceil(1.5*p/(1 - 1.5/sw)))
  - Pure numpy; no scipy/statsmodels/sklearn (AST-guard enforced)
  - 16 tests: helpers (nextodd, tricube, loess recovers linear/constant,
    smooths noise, low-pass preserves periodic signal); contract
    (additive y = T + S + R exact); pure seasonal → T≈0, R≈0;
    pure trend → S≈0; remainder small on perfect decomposition;
    determinism; error paths.
  Used by: Theta STM (this release), future GBDT transforms,
  future MSTL wrapper.

Theta STM on STL (cadw/ts/theta.py, ~350 LOC):
  - Replaces pre-existing 103-line Theta implementation (which used
    classical decomposition)
  - STL decomposition fixes trend-leak-into-seasonal-estimate bug
    of classical pos_means approach
  - Theta0 = OLS on deseasonalized (T + R = y - S)
  - Theta2 = 2*y_ds - Theta0; SES with alpha grid (19 candidates)
  - Forecast = 0.5*theta0_f + 0.5*theta2_f + S_forecast
  - Capability flags: handles_trend=True, handles_seasonality=True,
    handles_multi_seasonality=False (for future AutoForecastPipeline)
  - Gaussian prediction intervals via math.erfinv (no scipy)
  - 17 tests: OLS/SES helpers, contract, STL integration (decomp_
    populated when seasonal, None otherwise), trend-leak improvement
    test (STM on STL MASE < 1.5 vs 1.40 classical threshold),
    capability flags, deterministic, PI, error paths, AST guard.

KNOWN LIMITATION (documented in card note + theta.py docstring):
  STM underperforms on trend-dominated series. On
  synthetic trend=0.3, season_amp=3, STM MASE ~1.4 vs AutoETS 0.5.
  Root cause: SES on theta_2 lags behind trend by
  (1-alpha)/alpha * slope per step — fundamental to STM, not fixable
  by better decomposition. STL fixes trend leak but SES lag remains.
  On M3-like data (trend=0.05, season_amp=5) STM MASE < 1.0
  (documented in tests). OTM variant (optimised weights) is backlog;
  AutoTheta proper in v0.9.78. For trend-dominated series prefer
  AutoETS or AutoTheta (when available).

Theta card: tier baseline (not sota — see KNOWN LIMITATION above);
roles temporal_dynamics + probabilistic; cadw_measured
fit_time_s=0.981 @n=100k; capability flags documented.
Theta is ~40x faster than AutoETS on same DGP (0.981s vs 39s
@n=100k) — useful as fast baseline in ensembles.

Cleanup: removed orphaned cadw/ts/auto_theta.py,
tests/test_auto_theta.py, scripts/bench_auto_theta.py (superseded by
theta.py/test_theta.py/bench_theta.py). __init__.py only exports
ThetaForecaster from cadw.ts.theta.

Pre-existing bench_theta.py (44 lines) continues to work with the new
ThetaForecaster interface (alpha_/sigma2_ attributes preserved).

Version 0.9.76 -> 0.9.77.

Fast suite: 3359 -> 3384 passed (+25: decomposition 16 tests +
17 theta tests replaced 21 auto_theta tests; net +25), 8 skipped.
WL 34 OK. AST-guard 129 passed. audit_leakage 7 warnings (pretrained).
audit_url_log 0 errors. MATRIX.md + 40 docs/tasks/*.md regenerated.

Next: v0.9.78 — OTM (weight-optimised Theta) + AutoTheta (auto
decomposition selection) + ensemble upgrade (Theta → AutoTheta).

### v0.9.76 (2026-10-26) - D1 uni-forecast: SOTA horizon + AutoETS + 3-role ensemble

Third integrated release of the TS-anomaly/forecast cycle; covers
v0.9.75-v0.9.76c. Brings D1 (univariate forecasting) to parity with
D3 (anomaly) in depth: SOTA horizon cards + 3-role ensemble +
benchmark-driven card activation.

SOTA horizon (v0.9.75): 7 tier3_export cards in D1 — reference-only,
not implemented in CADW:

- Chronos-Bolt family (Amazon, 2024; T5 encoder-decoder, quantile-
  native, patch-based direct multi-step):
    Tiny (8.65M) - primary CPU reference. 250x faster / 20x memory-
      efficient vs original Chronos. Notable NEGATIVE scaling:
      on electricity-price Tiny (MAE 0.0393) BEATS Mini (0.0394),
      Small (0.0530), Base (0.0477).
    Mini (21M), Small (48M), Base (205M) - full reference family.
      Base has fev MASE 1.393 (5th/25 at release) but WORSE than Tiny
      on electricity - rank depends on benchmark.
- Chronos-2-small (28M, Nov 2025): only foundation model with native
  multivariable support + covariates (past+future, real+categorical)
  + cross-learning. Context 8192. Essential D2 horizon (D2 still has
  no foundation-model reference).
- AutoGluon-TS: stack ensemble (stat + GBDT + optional DL; weighted
  via repeated k-fold bagging + holdout MASE). M5 winner approach
  family; reference for D1 late-fusion ensembles.
- Timer-S1 (1B+, MASE 0.693, CRPS 0.485): GIFT-Eval top-1 at snapshot;
  absolute SOTA horizon; GPU-only.
- t0-beta (MASE 0.687, CRPS 0.474): marginally beats Timer-S1;
  included separately (different architecture).

4 new benchmarks (benchmarks.yaml): M3 (3003 series, sMAPE 7.19%
ARIMA), Tourism (1311 series), fev_bench (AutoGluon fev: 19 datasets,
MASE geomean, quantile losses WQL/SQL/CRPS), GIFT_Eval (130+ entries,
primary TSFM standard).

audit_url_log.py: tier3_export exemption. Cards that reference
pretrained models NOT fetched via fetch tool now require only
algorithm_source_url to look like http(s) + non-empty note; not
url_log 200 entries. Unblocks the 'SOTA horizon' pattern (registering
external state-of-the-art without fetching every HF model card).
16 previous HALLUCINATION errors on the 7 new cards resolve; 0 remain.

ETS card activation (v0.9.76a): ETS was implemented in v0.9.38
(cadw/ts/ets.py, ETSForecaster) with 11 passing tests, but the YAML
card was stale described_only=true since then. Fixed:
- implementation_status: verified_in_cadw=true, verified_by_agent=true
- +cadw_module/cadw_function
- +cadw_measured entry from .pingis/benches/ets.json
  (fit_predict_s=0.18/1.74/17.15 @ n=500/5k/50k; sigma 0.61/0.54/0.52)
- +v0.3 fields (model_class, learning_mode, source_library,
  evaluation_scope=both, dgp_source, leakage, pareto_axes)

AutoETS implementation (v0.9.76b): cadw/ts/auto_ets.py
(~370 LOC, pure numpy — no sklearn/statsmodels/torch/scipy in this
module; AST-guard enforced). Wraps ETSForecaster; enumerates up to 9
configs (trend N/A/Ad x season N/A/M); skips invalid combos
(n<2s for seasonal; y<=0 for multiplicative); picks minimum AICc
= n*ln(SSE/n) + 2k + 2k(k+1)/(n-k-1) where k=sigma+alpha+[beta]+
[gamma]+[phi]. Seasonal indices (s-1 values) treated as initial
state, not optimised (statsmodels convention, NOT in k). Fallback:
NaiveForecaster(last) if no config fits; warning emitted.

Fix iterations for AutoETS (documented in code + commit):
- v0.9.76b-fix1: selected_config_ uses label keys ('N'/'A'/'M'/'Ad')
  not ETS constructor params ('additive'/'multiplicative'); bench
  n=2000->500 (44s at n=500 already exceeds 30s slow-fit warning).
- v0.9.76b-fix2: for season_label='N', pass seasonal_period=1 (no
  seasonality), not self.seasonal_period. Old code passed
  self.seasonal_period with seasonal='additive' for BOTH 'N' and 'A'
  labels; ETS(A,N,N) was actually a seasonal model with smaller k
  penalty. AICc correctly preferred it on seasonal data (defeating
  auto-selection). This was the root cause of a failing test:
  AutoETS preferred non-seasonal config on strongly seasonal DGP.

18 tests in tests/test_auto_ets.py: contract/dtypes, attributes
after fit, AICc selection on seasonal data (pure sine, no trend),
non-seasonal on flat, trend selection on trending, determinism,
AICc scores populated, short-series skip seasonal, negative data
skip multiplicative, very short fallback to Naive, bad inputs,
prediction_interval, competitive with manual best-of-two,
AST no-bench-deps.

AutoETS card (v0.9.76c) in D1 with cadw_measured MASE=0.4998 on
synthetic_seasonal_linear (n=500, seasonal_period=12); 9 configs
evaluated; selected ETS(A,A,N). NOTE: 44s fit time at n=500 —
grid search is O(9 configs * 600 evals * n) Python-loop. Backlog
item "AutoETS grid search optimization" tracks vectorization /
coordinate descent fix.

3-role D1 ensemble (v0.9.76c):
[SARIMA, GBDTForecaster] -> [SARIMA, AutoETS, GBDTForecaster]
- SARIMA (temporal_dynamics): seasonal ARIMA via Hannan-Rissanen CSS
- AutoETS (probabilistic): AICc-selected ETS state space
- GBDTForecaster (nonlinear_interactions): tree-based on lag features
- GBDTForecaster roles narrowed from
  [nonlinear_interactions, temporal_dynamics] -> [nonlinear_interactions]
  (temporal structure is implicit in lag features, not native;
  SARIMA+AutoETS are the two temporal_dynamics representatives)
- max_per_role=2 (SARIMA+AutoETS both expose temporal_dynamics)
- Weight: inverse holdout MASE; average z-scored point forecasts

Verified D1 (v0.9.76c): 6 verified_in_cadw
(NaiveForecaster, ETS, AutoETS, SARIMA, GBDTForecaster, +
1 more); 7-8 tier3_export horizon references. Ensemble is now
3-role, matching D3's depth.

Fast suite: 3344 passed, 8 skipped, 25 deselected (unchanged).
WL 34 OK. AST-guard 129 passed. audit_leakage 7 warnings
(pretrained). audit_url_log 0 errors (tier3 exemption).
MATRIX.md + 40 docs/tasks/*.md regenerated.

Backlog: 12 items (AutoETS grid search optimization added).
Deferred: AutoTheta (v0.9.77), Croston (v0.9.77b),
HybridETS_GBDT (v0.9.78), weighted ensemble / FFORMA (v0.9.79),
metrics extension + M3/Tourism benchmarks registration (v0.9.80).

### v0.9.74 (2026-10-26) - TS anomaly: BOCPD + 3-role ensembles + multi-shape

Second integrated release of the TS-anomaly cycle; covers v0.9.69-v0.9.73.
First release (v0.9.68) closed Phase C (13 tier1 models) and Phase D
(cards + benchmarks + metric anchors). This release closes the remaining
gaps: diagnostic metrics, BinSeg baseline, zscore verification, BOCPD
probabilistic role, 3-role ensembles everywhere, and multi-channel shape.

Models added (v0.9.70, v0.9.72):
- BinarySegmentation (cadw/ml/binseg.py) - greedy top-down L2 splitting
  (ruptures Binseg reference). Baseline D3.2/D5. n=100k, 10 alternating
  shifts: recall=1.0 (10/10 within +/-10), 12.3s, 2 MB.
- BOCPD (cadw/ml/bocpd.py) - Bayesian online change-point detection
  with Student-t predictive likelihood (Adams & MacKay 2007;
  gwgundersen/bocd reference). Probabilistic role for D3.2 3-role
  ensemble. Student-t critical for heavy-tailed industrial data
  (SKAB: 32% DR at 13% FAR; BOCPD-Gaussian ~0% DR). n=20k, 5 alternating
  shifts: recall=1.0 (5/5 exact), median_detection_delay=0.0, 7.3s, 32 MB.

Fix history for BOCPD (documented in docstring):
  v0.9.72-fix1: per-element Python loop was 60s+ on n=20k. Rewritten to
    vectorized log-pdf with analytic alpha/kappa invariants
    (alpha[r]=alpha0+r/2, kappa[r]=kappa0+r; only mu/beta tracked);
    max_run capped at 200.
  v0.9.72-fix2: burn-in masking. First ~200 timestamps were flagged as
    anomalies on any series (posterior concentrated at small r; not
    enough data to establish long run). decision_scores[:burn_in]=0
    (default burn_in = 2*run_length_threshold).
  v0.9.72-fix3: scoring. Changed from P(r < run_length_threshold) to
    R[0] = P(change at t). Old scoring drifted toward the percentile
    threshold (cp_prob accumulates at every step, R[0] > 0 even in
    stationary periods) -> 400 false positives on n=20k at
    contamination=0.02 (threshold ~0.001). New: score = R[0],
    threshold = max(percentile(1-c), 0.5). Requires mode=change,
    matching Adams & MacKay.

Card activation (v0.9.70):
- D3.1 zscore: described_only -> verified_in_cadw (cadw.ml.zms.fit_zms),
  cadw_measured recall=1.0.
- D3.3 zscore: same fit_zms, different role (shape-agnostic baseline).
- D3.2 BinarySegmentation: described_only -> verified_in_cadw
  (cadw.ml.binseg.fit_binseg), cadw_measured recall=1.0.
- D5 BinarySegmentation (alias): same, documented as DEPRECATED alias.

Diagnostic metrics (v0.9.69) - TSB-AD + TAB completeness:
D3.1 and D3.3 gained vus_roc, range_f1, affiliation_f1, auc_roc, point_f1
  as diagnostic (anchors unchanged: vus_pr).
D3.2 gained range_f1, point_f1, recall as diagnostic (anchor auprc kept;
  note added: WEAK anchor for changepoint; NAB preferred when available).
point_adjusted_f1 explicitly marked LEGACY (gameable by random ranking).

3-role ensembles (v0.9.72b):
- D3.1: [MatrixProfile, IForest] -> [DWT-MLEAD, Sub-PCA, KShapeAD]
  roles: temporal_dynamics + representation + local_structure
- D3.2: [PELT] -> [PELT, T-squared+Q, BOCPD]
  roles: temporal_dynamics + representation + probabilistic
  (T-squared+Q role retier: temporal_dynamics -> representation; PCA
  subspace SPC is representation-based)
- D3.3: [S2G, MiniRocket_embedding, EIF] -> [S2G, Sub-PCA, EIF]
  roles: local_structure + representation + nonlinear_interactions
  (v0.9.73: MiniRocket_embedding uni-only replaced by Sub-PCA
  multi-capable; EIF moved to nonlinear_interactions slot after v0.9.72b)
EIF card added to D3.3 (same fit_eif as D3.1, role=shape vs point).

Multi-channel shape support (v0.9.73):
- cadw/ml/sub_pca.py, kmeans_ad.py, autoencoder.py: +_prepare_windows
  helper (uni/multi windowing, stride param, min-length guard). Body
  uses d = windows.shape[1] instead of window_size in PCA/eigh/
  xavier/zero-init/memory-est.
- docs/ml_tools/samples.py: +gen_ts_shape_anomaly_multi DGP
  (x3=x1-x2 plane; anomaly segment off-plane + 3*noise -> cross-channel
  break, invisible in per-channel marginals).
- 4 models now multi-capable in D3.3: EIF (0.864), Sub-PCA (0.944),
  KMeansAD (0.868), Autoencoder (0.78). Sub-PCA 0.03s (200x faster
  than EIF at same recall).
- STRAY multi test: recall=0.168 on cross-channel DGP; fundamental
  limitation (KNN on raw 3D misses plane break). STAYS uni-only.
- D3.3 ensemble: Sub-PCA takes representation slot from MiniRocket;
  multi-shape now covered by 2 of 3 ensemble members (Sub-PCA + EIF)
  plus S2G for uni motif.

Backlog (.pingis/drafts/BACKLOG_ML.md, gitignored):
- 11 deferred items with executable triggers + review protocol.
- Conv-AE/MSCRED (multi shape): trigger = EIF recall<0.5 OR sub-0.8 on
  new DGP OR real industrial multi-shape data needed.
- CARMA (2026, dual-latent VAE, cross-channel correlation latent):
  noted as principled alternative for cross-channel; Phase C.2.
  Only method that explicitly solves the correlation-break problem we
  measured (EIF 0.864 via windows; STRAY 0.168 via raw KNN).
- roboBayes trigger updated (BOCPD recall<0.3 OR FP>40% OR SKAB v1.0).

Verified counts (100% per D3.x):
  D3.1: 13/13
  D3.2: 5/5
  D3.3: 8/8 (4 multi-capable)
Anchors: D3.1 vus_pr, D3.2 auprc (note: weak; NAB diagnostic preferred),
D3.3 vus_pr.

Fast suite: 3344 passed, 8 skipped, 25 deselected.
WL 34 OK. AST-guard 129 passed. audit_leakage 0 warnings.
audit_url_log 0 errors (combined v0.9.48+v0.9.50 log, 257 entries).
MATRIX.md 145 lines; 40 docs/tasks/*.md regenerated.

Deferred to Phase C.2 (backlog): 1D-conv PaAno (proper); CARMA for
cross-channel shape; HYDRA (SOTA general multi-detector); roboBayes
(if BOCPD underperforms); SubLOF/MovingWindow/SAND (tier2);
TSPulse ZS (pretrained reference); GraphTS (paper-only, no reference
code); MAD-TCN/KARMA/DLinear_ad (native DL); full UCR evaluation.

### v0.9.68 (2026-10-26) - TS anomaly: 13 tier1_core models + Phase D

First release in this cycle; covers model work v0.9.52-v0.9.67 and
matrix work v0.9.65-v0.9.67 (details below).

13 new tier1_core anomaly detectors implemented in pure numpy
(AST-guard `test_no_bench_deps_in_core.py` enforces no sklearn/scipy/
torch/aeon/pyod/merlion/kats in cadw/). Each has >=12 tests, bench
artifact in `.pingis/benches/`, and a v0.3 matrix card in
`D_time_series.yaml`.

Model implementations:

- DWT-MLEAD (`cadw/ml/dwt_mlead.py`) - Thill 2017. Multilevel Haar
  DWT + Mahalanobis distance per level. Density-based leader
  (TAB 5.2.1). n=100k point spikes: recall 0.76, 0.51s, 35 MB.

- KShapeAD (`cadw/ml/kshape.py`) - Paparrizos & Gravano 2015.
  k-Shape with SBD (FFT cross-correlation + shift alignment) +
  eigenvector centroids. VUS-PR 0.40 U (TSB-AD). n=10k: 0.41, 23.5s.

- Sub-PCA (`cadw/ml/sub_pca.py`) - windowed PCA reconstruction;
  mean aggregation for spike spread. VUS-PR 0.42 U (TSB-AD). Fastest
  strong model: n=100k, recall 1.0, 0.13s, 200 MB.

- EIF (`cadw/ml/eif.py`) - Hariri 2019. Extended IForest: random
  hyperplane splits. TAB Fig 8a best group. Segments uni 0.80,
  multi 1.00.

- T-squared+Q (`cadw/ml/tspq.py`) - Hotelling T^2 + Q (SPE) on PCA.
  SKAB changepoint F1 0.76. n=20k multi, recall 0.96, 0.03s.

- Hampel (`cadw/ml/hampel.py`) - rolling median + MAD filter.
  Baseline. n=100k, recall 1.0, 0.06s.

- ZMS (`cadw/ml/zms.py`) - Merlion-style trailing rolling z-score.
  n=100k, recall 1.0, 0.03s, 85 flagged (precise).

- CUSUM (`cadw/ml/cusum.py`) - two-sided CUSUM with reset after
  alarm. n=100k, 10 alternating mean shifts, recall 1.0, 1.28s.

- STRAY (`cadw/ml/stray.py`) - Talagala 2021. Robust standardize +
  k-NN distance + GPD tail (MoM, not MLE). Brute-force KNN O(n^2),
  n<=20k. n=10k, recall 1.0, 1.43s.

- S2G (`cadw/ml/s2g.py`) - Boniol & Palpanas 2020, paper-faithful:
  local convolution + PCA 3D + rotation + angle nodes + transition
  graph + path normalcy. VUS-PR 0.39 U. n=50k: segments 0.58, motifs
  0.34. Third rewrite after two failed attempts (simplified PCA-graph:
  0.35/0.18; naive np.roll alignment: 0.007) - documented in commits.

- Autoencoder (`cadw/ml/autoencoder.py`) - MLP encoder-decoder, SGD
  + momentum, mean/median reconstruction error. n=50k needed to avoid
  memorizing point spikes (documented scale sensitivity).

- KMeansAD (`cadw/ml/kmeans_ad.py`) - k-means++ on windowed
  subsequences. VUS-PR 0.37 U (TSB-AD). n=50k, recall 0.90, 0.28s.

- PatchMemoryAD (`cadw/ml/patch_memory_ad.py`) - PaAno-inspired patch
  splitting + MLP encoder + memory bank + triplet loss (often inactive;
  min-distance scoring still works). n=30k, recall 1.0.

v0.9.63-fix2: normalization strategies (HYDRA SIGMOD 2026 evidence).
HYDRA section 1: "while normalization reduces scale bias and can
enhance anomaly detectability, it may also suppress amplitude-driven
anomalies". PaAno normalises per PATCH not window. Ablation on 4 DGPs
(easy/subtle/noisy/hard, n=30k, 20 spikes, contamination=0.08):

    config  pma_none  pma_window  pma_patch  ae_none  ae_window
    easy    1.00      1.00        0.35       1.00     1.00
    subtle  1.00      1.00        0.35       1.00     0.90
    noisy   1.00      0.00        0.00       1.00     0.00
    hard    0.95      0.10        0.00       0.60     0.00

Default `normalization="none"` preserves amplitude. `"window"` (old)
suppresses spikes at sigma>=0.5 (recall 0.0). `"patch"` (PaAno-
faithful) underperforms with THIS MLP-flattened encoder (0.35 easy,
0.00 noisy) because MLP has no weight sharing across patches; PaAno's
per-patch norm requires 1D-conv (same kernel per patch -> position-
invariant). Kept for API completeness; not recommended until 1D-conv
backprop is added.

Honest limitation (for matrix docs): on synthetic amplitude-spike
DGPs, classical Sub-PCA (0.03s) and KMeansAD (0.1s) match or beat
DL (Autoencoder 2s, PatchMemoryAD 14s) at normalization="none".
DL advantage expected only on multi-shape UCR data (TSB-AD: PaAno
0.52 vs Sub-PCA 0.42 there) - deferred to Phase C.2 with proper
1D-conv PaAno. All DL models tier=baseline; no claim of new SOTA.

v0.9.64: guard false-positive fix.
`test_arrow_call_memory_guard` flagged `np.quantile` in
`fit_dwt_mlead` (cadw/ml/dwt_mlead.py:264) as "high-risk unguarded
quantile" - false positive (guard matched by attr name only;
np.quantile is in-place numpy partition, not pyarrow/pandas allocation
of unbounded memory). Fix: `_SAFE_NUMPY_MODULES = {np, numpy, math,
scipy}` + recursive receiver check in `_get_high_risk_calls`; +4
tests (np not flagged, pandas still flagged, nested np.attr, DWT-MLEAD
end-to-end). No whitelist entry added - fix at guard level prevents
future false positives on numpy-based anomaly models.

v0.9.65-v0.9.67: Phase D - matrix update.

- `benchmarks.yaml`: +SKAB (industrial ts_anomaly, f1+nab metrics),
  +TAB (VLDB 2025 unified ts_anomaly, vus_pr primary). 3 new anomaly
  benchmarks total (TSB-AD was already there).

- 13 cards added to `D_time_series.yaml` with v0.3 fields
  (model_class, learning_mode, anomaly_type, data_modality,
  source_library=native, evaluation_scope=synthetic, dgp_source,
  dgp_overlap_risk=none, leakage all native/none/temporal, coverage
  broad, pareto_axes [recall, fit_time_s, peak_rss_mb]).

- Tier corrections: IForest sota->baseline (VUS-PR 0.30 vs Sub-PCA
  0.42); MiniRocket_embedding sota->baseline (classifier, not
  anomaly detector); D5 DEPRECATED prefix (alias of D3.2).

- Metric anchors: D3.1 auprc->vus_pr (TSB-AD/TAB primary); D3.3
  auprc->vus_pr (range-aware; point-level AUPRC inflates shape-
  anomaly scores); auprc demoted to diagnostic; point_adjusted_f1
  explicitly LEGACY (gameable by random ranking on long series);
  recall added as diagnostic for cadw_measured entries; D3.2 nab
  added as diagnostic (SKAB convention) with auprc note weakened.

- MATRIX.md 140 lines; 40 docs/tasks/*.md regenerated byte-equal.

Gates: fast suite 3310 passed, 8 skipped (optional libs: sktime,
pyod, aeon, kats, sahandha/eif; smoke tests self-skip); 25 deselected
(slow). AST-guard `test_no_bench_deps_in_core.py` 127 passed.
WL 34 keys OK. audit_leakage 0 warnings. audit_url_log 0 errors
(combined v0.9.48 + v0.9.50 log, 257 entries).

Deferred to Phase C.2: 1D-conv PaAno (proper, no MLP-flattened
compromise); GraphTS (paper-only, single-paper 84% vs S2G 61% on UCR,
no reference code); SubLOF/MovingWindow/SAND (tier2 wrappers, low
priority); full UCR evaluation of current 13 models (synthetic DGP
too simple to show DL advantage).

### v0.9.50 (2026-10-25) — M3: migrate legacy A–H to v3 schema

Migration + enrichment. No production code change, no new cards.

- **Migrated 6 legacy sections** (A/D/E/F/G/H, 53 cards) from v2
  (`benchmark_result` singular, no attribution / algorithm_source_url /
  task_type) to v3 (`benchmark_results` list with benchmark_url +
  algorithm_source_url + attribution + task_type).
- **53 LEGACY warnings closed** (was 53 in audit_url_log.py, now 0).
  Combined v0.9.48+v0.9.50 url_log (257 entries): 0 errors, 0 warnings.
- **v2 back-compat shim removed** from schema.py:
  `ModelCard.benchmark_result` property, `BenchmarkResult.source`/`.url`
  properties, v2 branches in `_parse_benchmark_result(s)`. All 13
  sections now v3.
- **5 cadw_reproduced entries added** (parity vs M2 reproduction):
  - A.A3.GBDT_L2 ← gbdt_regressor.json (sklearn.GBR, rmse, consistent)
  - D.D2.VAR ← var.json (statsmodels VAR, rmse, consistent)
  - E.E1.IForest ← iforest.json (sklearn.IsolationForest, roc_auc, consistent)
  - F.F1.ALS ← als.json (implicit ALS, ndcg_at_k, consistent)
  - F.F2.ALS ← als.json (same)
- **1 cadw_measured entry added**: D.D3.1.MatrixProfile ← matrix_profile.json
  (fit_time_s, n_datasets=2).
- **Legacy bug fixed**: `wrmssе` (Cyrillic е U+0435) → `wrmse` in
  D_time_series.yaml (2 places) and benchmarks.yaml[M5].metrics.
- **Wayback substitutions** (fetch 403/refused at canonical URL, original
  URL preserved in card note):
  - E: ODDS
  - F: Amazon_Recsys
  - G: LETOR

Deferred to v0.9.50.1 fixup: CHANGELOG entry (this one), validator
`reproduces` sibling rule + test.

Fast suite: 3077 passed, 2 skipped. Focused: 121 passed.

## [Unreleased]

### v0.9.49 (2026-10-25) — BUG-1: value_counts crashes on multi-chunk

Production bug fix (correctness, chunk-size dependent). No new deps.

**Broken:** `CadwDataFrame.value_counts(col)` raised `pyarrow.lib.ArrowInvalid:
Schema at index N was different: <col>: string / __vco_sum: int64 vs
__vco_sum: double` whenever the frame was read with `chunk_size < n` and the
categorical column triggered skew isolation. Single-chunk frames were unaffected.
Reproducer: `.pingis/audits/v0.9.48_bugs.md` BUG-1.

**Root cause (confirmed):** in `cadw/ops/groupby.py::_groupby_with_skew`, the
light partitions are aggregated by `pa.Table.group_by(...).aggregate`, which
preserves the integer dtype of the ones-column → `__vco_sum: int64`. The heavy
key is accumulated in Python and materialised through `_accum_to_table`, which
uses `_empty_result_schema` — that declares every non-count aggregation as
`float64` → `__vco_sum: double`. When both paths appear across chunks,
`pa.concat_tables(all_results)` (line 741) rejects the schema mismatch.

**Fix:** before `pa.concat_tables`, align every table in `all_results` to the
schema of the first result via a safe `Table.cast(target_schema)`. This keeps
counts as `int64` (pandas `value_counts` parity) and lets integral float
partials round-trip exactly (float64 exact up to 2^53). `promote_options` was
deliberately not used — it would mask unrelated schema mismatches.

**M2 validation (n=100_000, 20 categories, chunk_size=50_000 vs 1_000_000):**
- multi-chunk cadw == single-chunk cadw → True (20 groups, sum=100_000)
- multi-chunk cadw == pandas `Series.value_counts()` → True
- single-chunk cadw == pandas `Series.value_counts()` → True
- sample: `{g9: 5176, g1: 5108, g11: 5072, g19: 5071, g15: 5048}` on all three.

**Tests:**
- `tests/test_v0949_value_counts_multichunk.py` — verbatim reproducer; fails at
  4e7e614, passes at v0.9.49.
- `tests/test_groupby_v2.py` — two new cases: multi-chunk == single-chunk, and
  both == pandas (M2 parity). Existing 10 cases untouched (12 passed total).

**Gates:** full fast suite 3077 passed, 2 skipped, 25 deselected.


### v0.9.47 (2026-10-24) — METHODOLOGY + schema v3 + PINGIS TASK

Docs + tooling only. No production ML code.

**v0.9.47a — METHODOLOGY.md + collect_project whitelist**
- METHODOLOGY.md in repo root (~170 lines): Discovery / Triage /
  multi-benchmark selection / two-source verification / Registration /
  Debugging divergence ladder / PINGIS vs human roles. In root so that
  `collect_project --mode=tiny` snapshots it fully (docs/** is STUB).
- `scripts/collect_project.py`: METHODOLOGY.md added to ROOT_FILES_FULL.

**v0.9.47b — schema v3 + audit_url_log**
- `docs/ml_tools/enums.py`: ATTRIBUTIONS (literature | cadw_reproduced |
  cadw_measured), SOURCE_KINDS (paper | code | docs | pseudocode |
  description).
- `docs/ml_tools/schema.py`: BenchmarkResult v3 — benchmark/benchmark_url
  (WHERE the number comes from) + algorithm/algorithm_source_url/
  algorithm_source_kind (WHERE the algorithm is documented) + attribution
  + reproduces + task_type + n_datasets + metric + value + std + note.
  ModelCard.benchmark_results: list. Backward-compat shim keeps old
  singular `benchmark_result` parseable.
- `scripts/audit_url_log.py` (new): gate for v0.9.48. For literature
  entries, BOTH benchmark_url and algorithm_source_url need url_log
  entries with status=200. cadw_reproduced requires `.pingis` JSON with
  verdict=consistent. cadw_measured requires both URLs exist.
  LEGACY v2 cards emit warning, not error.
- `scripts/render_matrix.py`: renders benchmark_results list; docs/tasks/*
  regenerated.
- `tests/test_audit_url_log.py` (9 tests).

**v0.9.47c — TASK_v0.9.48 + METHODOLOGY refinement**
- METHODOLOGY §3 rewritten: multi-benchmark per subtype (AMLB + OpenML_CC18
  + Kaggle, etc.), n_datasets + std, per-dataset stories as prose.
- METHODOLOGY §4 rewritten: two sources per number (benchmark_url +
  algorithm_source_url), attribution tiers.
- `TASK_v0.9.48_ml_tools_discovery.md`: single task for 7 remaining
  sections (I/N/J/M/O/L/K) with resume protocol, url_log gate, per-
  algorithm rows, anti-hallucination discipline.
- Old TASK_v0.9.47 archived to `.pingis/tasks/archive/`.

**Version:** 0.9.46 → 0.9.47. Fast suite: 2978 passed, 2 skipped. WL 34 OK.

## [Unreleased]

### v0.9.46-fix2 (2026-10-24) — ALS item-sweep sort-key bug (production)

D.2.2 (v0.9.36) had a production bug: `fit_als` sorted the item-sweep
by `["i"]` (= user_pos in that view) instead of `["u"]` (= item_pos).
Consequence: `Y[i]` was updated on one user per item instead of all
users who interacted with item i — Y factors wrong from iteration 1.
Lived 3 months, 11 D.2.2 tests + 6 audit passes missed it because none
compared to ground truth.

- **Fix**: one char, `_sorted_batches(False)` sort key `["i"] → ["u"]`.
- **Regression**: `test_als_one_iter_matches_manual_hu_koren_volinsky`
  (1-step CADW vs manual Hu-Koren-Volinsky, `atol=1e-12`).
- **Verification**: `reproduce_benchmark.py --model als` now
  `consistent` (CADW NDCG@10 0.3827 vs implicit 0.3673, was
  0.14 vs 0.37).
- **README**: debugging ladder for future reproducers — always start
  with 1-step on tiny synthetic, not parameter tuning. Recorded after
  this session spent 6 iterations on init/alpha/reg/DGP before
  isolating the sort bug via one-step comparison.

**Impact on prior releases**: any user of `fit_als` (D.2.2) got
incorrect item factors. Recommendation: retrain. No other ML module
affected (`_gram`/`_solve_user` verified separately).

### v0.9.46 (2026-10-24) — M2 first: reproduce_benchmark + 4 models

Reproduction harness: CADW model vs reference (sklearn, statsmodels,
lifelines) on synthetic benchmark-like DGP. Not a leaderboard
reproduction.

- **scripts/reproduce_benchmark.py** — 4 reproducers, all references
  already in dev group (no install): IForest, GBDT-reg, VAR, CoxPH.
  Uses model-specific APIs: GBDT.predict(np.ndarray) via _predict_np,
  IForest.score_samples(df), CoxPH.predict_risk(df) -> pa.Table,
  VARModel(p).fit(target_col=list) + forecast_vector(steps).
- **docs/ml_tools/samples.py** — deterministic DGP generators
  (gen_odds_like, gen_amlb_{binary,regression}_like, gen_m4_like,
  gen_survival_like).
- **tests/test_reproduction_smoke.py** — 5 tests (4 reproducers,
  1 artifact); skip on missing reference; fail on divergent/cadw_error.
- **Artifacts** `.pingis/benches/reproduction/<model>.json`.
- **README.md** M2 section; roadmap M2 in_progress; CONTEXT.md M2 row.

Measured (synthetic, seed=0):
- IForest 0.9988 AUROC vs sklearn 0.9992
- GBDT-reg 1.0970 RMSE vs sklearn 1.0498
- CoxPH 0.6755 c-index vs lifelines 0.6755
- VAR RMSE(cadw, statsmodels) = 0.0000 (OLS deterministic)

Version 0.9.45 -> 0.9.46.

## [Unreleased]

### v0.9.45 (2026-10-24) — M1 complete: E/F/G/H + render

ML-tools matrix M1 is complete. 6 sections, 21 subtypes, 8 benchmarks.

- **E_anomaly.yaml** — E1 unsupervised (zscore, IForest, LOF-export),
  E2 supervised (composition A6). E3 (TS anomaly) covered by D3.1-D3.3.
- **F_recsys.yaml** — F1 collaborative (ItemKNN-export, ALS),
  F2 hybrid (ALS, Hybrid_MF_GBDT).
- **G_learning_to_rank.yaml** — G1: PairwiseLogistic-export,
  LambdaMART.
- **H_causal.yaml** — H1 treatment effect (T/S-learners, DR-export),
  H2 uplift (T_learner_uplift, CausalTree).
- **benchmarks.yaml** — added LETOR.
- **scripts/render_matrix.py** — deterministic renderer:
  `docs/ml_tools/MATRIX.md` (one table) + `docs/tasks/<S><id>.md`
  (one card per subtype). This is the Phase E substitute (recipes as
  views over the matrix).
- **tests/test_render_matrix.py** — byte-equality check; catches
  drift between YAML and rendered Markdown.
- Roadmap M1 → completed. current_version 0.9.44 → 0.9.45.
- CONTEXT.md: M1 row → completed.

**Matrix totals** (v0.9.45):
- Sections: A, D, E, F, G, H (6)
- Subtypes: 7 + 7 + 2 + 2 + 1 + 2 = 21
- Models: ~50 cards
- Benchmarks registered: 12 (AMLB, ODDS, ADBench, UCR_Classification,
  UCR_Anomaly, TSB-AD, M4, M5, RCT_IHDP, CoxPH_Survival,
  Amazon_Recsys, LETOR)

**Next**: Phase F (Light NNs) and Phase H (Long-tail) — each new
model must add a card to the matrix, else not `done`.

## [Unreleased]

### v0.9.44 (2026-10-24) — M1 part 1: ml_tools schema + A/D

Data + validator only. No production code change. Fills the first two
sections of the ML-tools matrix that will replace
`docs/ML_TASKS_TAXONOMY.md` (deprecated v0.9.43).

- `docs/ml_tools/enums.py` — fixed vocabularies: 7 roles,
  3 tier levels, 4 impl flags, 3 combination values.
- `docs/ml_tools/schema.py` — dataclasses + `validate_matrix()`.
  Plain Python, no pydantic (the matrix is data reviewed by humans;
  domain-specific errors are more useful than generic pydantic ones).
- `docs/ml_tools/benchmarks.yaml` — global registry of 11
  benchmarks with mandatory `warning` about cross-benchmark
  comparability (AMLB, ODDS, ADBench, UCR_Classification,
  UCR_Anomaly, TSB-AD, M4, M5, RCT_IHDP, CoxPH_Survival,
  Amazon_Recsys).
- `docs/ml_tools/A_classification.yaml` — 7 subtypes (A1 binary,
  A2 multiclass, A3 regression, A4 multilabel, A5 ordinal,
  A6 imbalanced, A7 quantile). A1/A3 fully filled; A2/A4-A7 with
  currently available CADW models + 1 SOTA reference each.
- `docs/ml_tools/D_time_series.yaml` — 7 subtypes (D1, D2, D3.1,
  D3.2, D3.3, D4, D5). All CADW-implemented TS models carded.
- `tests/test_ml_tools_schema.py` — validator (4 tests).
- `docs/roadmap.json`: Phase M1 added, status `in_progress`.
- `CONTEXT.md`: M1 row updated to `v0.9.44a / v0.9.44b` split.

**Validator rules enforced**:
- every role/tier/tier_level/combination from fixed enum
- exactly ONE `anchor` metric per subtype
- `anchor` metric must have `special` field
- every subtype has >= 1 `baseline` and >= 1 `sota` model
- model not verified in cadw/agent → NOT in ensemble_recommendation
- every `benchmark_result.source` in `benchmarks.yaml`
- every benchmark has a `warning` (comparability caveat)
- ensemble role overlap <= `max_per_role` per rule

**Deferred to v0.9.44b**: E (anomaly), F (recsys), H (causal/uplift),
`render_matrix.py`, `MATRIX.md` view.

## [Unreleased]

### v0.9.43 (2026-10-24) — Phase G: tech debt closeout
- **D.6.1 de-skleranification** — `cadw/ml/feature_select.py`
  imported sklearn/scipy lazily inside `_mi_cc`/`_mi_cd`, on the
  default `method='mutual_info'` path. That made sklearn a hidden
  runtime dep (caught by the new AST guard). Replaced with
  histogram-based MI (native numpy, adaptive bins). Ranking vs
  sklearn preserved (contract spec 'top-k = sklearn'); absolute
  values differ (smoke: MI 0.46 vs 0.03 — ratio ~15x; documented
  in d6_1 notes). Tests rewritten from `np.allclose(scores, ref)`
  to top-k ranking checks.
- **`tests/test_version_consistency.py`** — guards pyproject.version
  == cadw.__version__ == CHANGELOG top entry. Added after two
  consecutive version-bump misses (v0.9.42-fix1, v0.9.43 pre-check).
- **Stray `=` file** removed (shell artifact from
  `echo scikit-learn>=1.9.1` without quotes).


Closes A.6 (deferred from v0.9.26) and isolates benchmark reference
deps. No production code change.

- **A.6 closed** by reference to AUDIT_v0931 (v0.9.32): 160
  to_pylist/as_py/to_numpy sites scanned; only F9 (gbdt per-batch
  stack, fixed v0.9.32) was real; all others bounded by sample/
  group size, guarded upstream, or by design. Original "18 sites"
  claim from v0.9.26 cannot be re-derived (audit absent), but the
  exhaustive 160-site audit supersedes it.
- **`requirements-bench.txt`** — isolated list of reference libraries
  (scikit-learn, statsmodels, pyod, ruptures, lifelines, implicit).
  NOT added to `pyproject.toml`. Used only by `scripts/bench_*.py`
  and future `tests/test_ml_tools_*`.
- **AST guard** `tests/test_no_bench_deps_in_core.py` — fails if any
  `cadw/**/*.py` imports pyod/ruptures/lifelines/implicit/
  statsmodels/sklearn. Mirror of the existing
  `test_no_pandas_in_ops.py` pattern.
- **`.hypothesis` in `norecursedirs`** — pytest no longer scans the
  hypothesis cache during collection. Pre-existing issue, not a
  v0.9.42 regression.
- **`docs/ML_TASKS_TAXONOMY.md` marked deprecated** — will be
  replaced by `docs/ml_tools/` matrix after v0.9.44 (M1). Kept for
  navigation until 5 sections populated (A, D, E, F, H).
- **`CONTEXT.md` roadmap section added** — A-D closed; G (this
  patch) → M1 (matrix schema) → F (Light NNs) / H (Long-tail)
  → E (recipes as views). DoD for F/H adds "card in matrix with
  roles, purpose, real_life_examples, benchmark_result,
  implementation_status".
- **`docs/roadmap.json`**: A.6 → done (with resolution note).

Test status: fast suite green (v0.9.42 baseline 2829 passed /
2 skipped + ~99 new AST-guard param tests); slow 25 passed;
WL 34 OK.



### v0.9.42 (2026-10-24) — D.3.3 causal tree ensemble

Replaces the v0.9.37 two-model T-learner in `fit_uplift` with a proper
causal tree ensemble.

- **Split criterion**: Euclidean divergence between treated/control
  outcome distributions in children (Rzepakowski & Jaroszewicz 2012).
  `Gain = (n_L/n)·D_ED(L) + (n_R/n)·D_ED(R) - D_ED(S)`,
  `D_ED(S) = (P(y=1|t=1,S) - P(y=1|t=0,S))^2`.
- **Honest leaf estimation** (Wager & Athey 2018): random half of the
  sample chooses tree structure, other half estimates leaf CATE.
  Removes overfitting and doubles the effective sample per leaf vs
  the T-learner's split arms.
- **Ensemble**: `n_estimators` bootstrap samples; average per-row
  CATE predictions.
- **Bounded sample**: cap at `max_samples=20 000` rows (`_MAX_SAMPLE_ROWS`);
  new `max_samples=` keyword on `fit_uplift`. Prediction streams batches.
- **`_fit_uplift_two_model`**: previous v0.9.37 T-learner preserved
  (private) as the reference baseline for the strict comparison test.
- **API unchanged**: `fit_uplift(df, X_cols, treatment_col, y_col, *,
  n_estimators=50, max_depth=3, learning_rate=0.1, seed=0, max_samples=None,
  memory_guard=None)` — `learning_rate` accepted for compatibility and
  ignored (no residual boosting; ensemble averaging suffices).
- **Test contract change**: `test_qini_beats_t_learner_on_strong_signal`
  now compares causal tree against `meta_learners.fit_meta_learner(kind='t')`
  (D.3.2 canonical T-learner) on `p1=0.9, p0=0.2, x0>0` synthetic.
  `_truth` DGP switched from U-shaped CATE (sigmoid saturation made
  `corr(u, x2) > 0.25` pass by luck for T-learner on seed=0) to a
  monotone multiplicative effect (`t * (0.6 + 0.3*x2)`).

Artifacts: `.pingis/algorithm_notes/d3_3_uplift.md`,
`.pingis/benches/uplift.json`. `docs/ALGORITHM_BACKLOG.md D.3.3` row
restored to `Qini >= T-learner on strong-signal synthetic`.

## [Unreleased]

### v0.9.41 (2026-10-24) — Phase D algorithms, group 7/7 (Phase D complete)

- **D.6.6 (M) — online variational Bayes LDA.** `cadw/ml/lda.py` +
  `df.fit_lda(...)`. Bounded DF vocabulary, streamed document-level
  variational E-step, per-epoch (K x V) M-step, document-seeded init plus
  perplexity-selected multi-restart (defaults max_iter=15, n_restarts=16).
  `transform` -> normalised document-topic gamma; `perplexity` uses the true
  predictive likelihood, not the variational bound. 10 tests; vs sklearn
  LDA on the same BoW our NMI >= 0.9x (both reach 1.0); 2k/10k/50k rows in
  4.43/21.19/104.26 s, perplexity 4.13.
- **D.6.7 (S) — disjoint LinUCB.** `cadw/ml/linucb.py` +
  `df.fit_linucb(...)`. Streaming per-arm sufficient statistics
  (A_a += x x^T, b_a += r x), closed-form ridge theta, UCB score
  x^T theta + alpha sqrt(x^T A^-1 x); state O(K d^2). 10 tests; theta equals
  the analytic ridge solution to 1e-9; 150 000 rows in 0.04 s, peak RSS
  51 MB.
- **D.6.9 (S) — streaming TF-IDF.** `cadw/ml/tfidf.py` +
  `df.fit_tfidf(...)`. Streamed DF vector, sklearn smoothed idf
  log((1+n)/(1+df))+1, sublinear TF, l1/l2/none norms, sparse (indices,
  values) output. 11 tests; dense output matches sklearn TfidfVectorizer to
  1e-9 with the identical tokenizer/vocabulary; 500 000 rows in 1.28 s,
  peak RSS 230 MB.

Phase D is complete: all 29 D items are done. 31 new tests in this group;
fast suite 2796 -> 2827 passed, slow 25 passed, WHITELISTED_FUNCS 34 keys
unchanged.

### v0.9.40 (2026-10-24) — Phase D algorithms, group 6/7

- **D.5.4 (L) — BIRCH CF-tree streaming clustering.** `cadw/ml/birch.py` +
  `df.fit_birch(...)`. (N, LS, SS) cluster features, single-level CF tree with
  threshold-based absorption and closest-pair split at branching_factor;
  phase 3 weighted k-means++ agglomeration of leaf CFs. State O(#CF*(2d+1)),
  independent of N. 9 tests; sklearn Birch purity 1.0 (up to label perm);
  50 000 rows in 3.39 s, peak RSS 3.4 MB.
- **D.6.1 (M) — mutual_info / variance / Boruta feature selection.**
  `cadw/ml/feature_select.py` + `df.fit_feature_select(...)`. MI (k-NN)
  matches sklearn within atol=0.03; zero-variance drop; Boruta shadow
  features via GBDT importance. 10 tests; 10 000 rows MI 1.51 s,
  Boruta 8.36 s.
- **D.6.2 (M) — drift + CUPED A/B.** `cadw/ml/drift.py` +
  `df.fit_drift(...)` / `df.fit_ab_test(...)`. KS/PSI/chi2/Welch/MW
  statistics match scipy exactly; CUPED theta -> 0.80. 9 tests; 1 000 000
  rows in 1.28 s, peak RSS 156 MB.
- **D.6.4 (S) — randomized SVD.** `cadw/ml/randsvd.py` +
  `df.fit_randomized_svd(...)`. Halko range finder + power iterations;
  singular values match sklearn PCA to ~1e-6. 8 tests; 150 000x20 in
  0.22 s, peak RSS 71 MB.
- **D.6.5 (S) — geospatial aggregation.** `cadw/ml/geo.py` +
  `df.fit_geo_agg(...)`. Haversine, bearing, hex_cell binning with streaming
  per-cell count/sum/mean; haversine matches the analytic great-circle
  exactly. 9 tests; 1 000 000 rows in 2.45 s, peak RSS 108 MB.

### v0.9.39 (2026-10-24) — Phase D algorithms, group 5/7

- **D.4.6 (M) — probability calibration.** `cadw/ml/calibration.py` +
  `df.fit_calibration(...)`. Platt scaling (Newton IRLS) and isotonic (PAVA
  over <=256 equal-count bins) on top of the in-tree GBDT classifier;
  `brier_score` and `expected_calibration_error` also exported. 10 tests;
  isotonic ECE 0.0038 vs platt 0.0231 at 100 000 rows; 100 000 rows in 5.92 s.
- **D.5.1 (M) — ordinal logistic cumulative link.** `cadw/ml/ordinal.py` +
  `df.fit_ordinal_logit(...)`. Proportional-odds cumulative logit fitted with
  L-BFGS-B on the multinomial likelihood; thresholds parameterised as
  [theta_0, log-gaps] to stay ordered without a constrained solver. 10 tests;
  vs statsmodels OrderedModel(distr=logit): beta and first threshold maxdiff
  < 1e-2. 200 000 rows in 2.09 s.
- **D.5.2 (S) — classifier chains.** `cadw/ml/chains.py` +
  `df.fit_chains(...)`. Each label is an in-tree GBDT; chain variant appends
  true previous labels at fit and hard predicted labels at predict; independent
  variant is the one-vs-rest baseline. `subset_accuracy` metric. 8 tests;
  subset accuracy 0.664 vs sklearn ClassifierChain 0.702 (94% of reference),
  and chain >= OvR on the same data. 100 000 rows in 11.26 s.
- **D.5.3 (M) — MiniBatch KMeans + streaming EM GMM.** `cadw/ml/minibatch.py`
  + `df.fit_minibatch_kmeans(...)` / `df.fit_minibatch_gmm(...)`. KMeans uses
  Sculley per-centroid streaming updates with per-centroid counts; GMM is a
  diagonal-covariance EM with per-mini-batch sufficient statistics seeded from
  k-means++. 12 tests; inertia vs sklearn MiniBatchKMeans within 1.15x;
  GMM log-likelihood vs sklearn GaussianMixture(diag) within 0.15 nats.
  500 000 rows: kmeans 1.98 s / 63 MB, gmm 3.11 s / 67 MB — well under the
  2.2 GB budget.

All four modules live under `cadw/ml/` per the stage-4 structural note.
Bindings `fit_calibration`/`fit_ordinal_logit`/`fit_chains`/
`fit_minibatch_kmeans`/`fit_minibatch_gmm` registered. No new dev dependencies.

Docs: `docs/roadmap.json` D.4.6/D.5.1/D.5.2/D.5.3 -> done,
`docs/ML_TASKS_TAXONOMY.md` A4/A5 and B2/B3 -> available. Notes and benches
under `.pingis/algorithm_notes/`, `.pingis/benches/`. Fast suite 2711 -> 2751
passed; slow 25 passed.

### v0.9.38 (2026-10-24) — Phase D algorithms, group 4/7

- **D.4.2 (M) — PELT change-point detection.** `cadw/ml/pelt.py` +
  `df.fit_pelt(...)`. Exact PELT with the L2 mean-shift cost computed in O(1)
  from cumsums, Killick et al. 2012 pruning. `max_window` bounds the recursion
  -> O(window) state (the "online" reading); `max_window=None` is exact
  full-series PELT. 11 tests; vs ruptures.Pelt(l2): same segmentations within
  boundary ties, 0.101 s vs 73.572 s at n=20 000. State 256 B.
- **D.4.3 (M) — ETS / Holt-Winters.** `cadw/ts/ets.py` +
  `df.fit_ets(...)`. Additive-error ETS(A,N,N)/(A,A,N)/(A,A,A) with optional
  damping; deterministic coordinate grid over (alpha,beta,gamma,phi);
  classical-decomposition seasonal init; `prediction_interval`. 11 tests;
  MASE 0.298 vs statsmodels 0.338 on held-out 12 points. 50 000 rows in 17.15 s.
- **D.4.4 (S) — Theta method.** `cadw/ts/theta.py` +
  `df.fit_theta(...)`. OLS theta=0 line + SES-smoothed theta=2 curve, combined
  0.5/0.5 (Assimakopoulos-Nikolopoulos). 9 tests; MASE 2.18 vs statsmodels
  ThetaModel 1.99 on a strong linear trend. 100 000 rows in 0.92 s.
- **D.4.5 (M) — VAR(p).** `cadw/ts/var.py` + `df.fit_var(...)`.
  Equation-by-equation OLS on a stacked lag design, companion-matrix stability
  check, `forecast_vector(steps)` and ForecasterBase `predict`. 11 tests; vs
  statsmodels VAR max abs forecast diff 2.8e-17 (exact). 50 000 x 4 x p=2 in
  0.039 s.

Dev dependency added (test-side only, `[dependency-groups] dev`):
`ruptures>=1.1.9` as a reference implementation in `tests/test_pelt.py`.
Nothing under `cadw/` imports it.

Docs: `docs/roadmap.json` D.4.2/D.4.3/D.4.4/D.4.5 -> done,
`docs/ML_TASKS_TAXONOMY.md` ETS/Theta/VAR/PELT -> available. Notes and benches
under `.pingis/algorithm_notes/`, `.pingis/benches/`. Fast suite 2669 -> 2711
passed; slow 25 passed. `.pingis/session.log` untracked and added to
`.gitignore` (it was accidentally committed in 2c30326).

### v0.9.37 (2026-10-24) — Phase D algorithms, group 3/7

- **D.3.2 (M) — T-learner / S-learner + propensity.** `cadw/ml/meta_learners.py` + `df.fit_meta_learner(...)`. Two outcome models on the in-tree GBDT regressor (T-learner: one per arm; S-learner: treatment as an extra feature). Propensity from `fit_logistic` with `overlap_diagnostics` (n, min, max, share outside [0.05, 0.95]) always reported. Exposes `cate`, `ate`, `cate_table`, `ipw_ate`. `log_fit` per arm; `check_batch(extra_factor=2.0)` on every pass. 9 tests; bench 100 000 rows in 4.25 s / 20.2 MB, state 1.2 MB, ATE 1.02 vs true 1.0.
- **D.3.3 (M) — uplift trees + Qini.** `cadw/ml/uplift.py` + `df.fit_uplift(...)`. Two-model uplift on the in-tree GBDT classifier; `qini_curve(y, t, score)` returns fractions, curve and Qini coefficient against the random baseline. Both arms and binary outcome guarded. 10 tests; bench 100 000 rows in 9.43 s / 12.7 MB; large-n Qini reported honestly (control-arm noise shown, not hidden by a changed test).
- **D.4.1 (M) — MiniRocket streaming.** `cadw/ml/minirocket.py` + `df.fit_minirocket(...)`. Length-9 kernels in {-1, +1} with exactly three positive weights, geometric dilation, seeded and deterministic. One series materialised at a time; input must be grouped/sorted by `id_col` (guarded). 11 tests; bench ~320 series/s independent of count, peak RAM scales with series length (20.9 MB at length 2000), state 7 744 B.

Docs: `docs/roadmap.json` D.3.2/D.3.3/D.4.1 -> done, `docs/ML_TASKS_TAXONOMY.md` H1/H2, D4 and O9 -> available. Fast suite 2639 -> 2669 passed; slow 25 passed.

### v0.9.36 (2026-10-24) — Phase D algorithms, group 2/7

- **D.2.2 (M) — implicit-feedback ALS.** `cadw/ml/als.py` +
  `df.fit_als(...)`. Hu-Koren-Volinsky confidence weighting; one
  `external_sort` per sweep, streamed per-user k×k solves. State
  O((U+I)·k). `log_fit` per sweep; every batch goes through
  `check_batch(..., extra_factor=2.0)`. 10 tests; bench 200 000
  interactions in 31.29 s / 64 MB, state 1 792 000 B exact.
- **D.2.3 (M) — hybrid MF + content.** `cadw/ml/hybrid_mf.py` +
  `df.fit_hybrid_mf(...)`. Ridge map from item content to ALS item
  factors, applied as a k-dimensional residual (a concatenation broke
  scoring and was caught by the tests); cold start via
  `content_only_scores`. 10 tests; measured +0.0799 Pearson corr vs
  ALS at 446 391 interactions for ~0.5% extra fit time.
- **D.3.1 (M) — Cox proportional hazards.** `cadw/ml/coxph.py` +
  `df.fit_coxph(...)`. Breslow partial likelihood, Newton-Raphson,
  baseline cumulative hazard, Harrell c-index (vectorized in row
  chunks). Cannot be a Reduce (risk sets are global) so it uses the
  guarded O(N·d) pre-pass accepted for LambdaMART — no approximation
  claimed. 10 tests; vs lifelines: coef maxdiff 1.58e-06, c-index
  identical, 50 000 rows in 0.70 s vs 4.18 s.

Dev dependencies added (test-side only, `[dependency-groups] dev`):
`lifelines>=0.30`, `statsmodels>=0.15`. Nothing under `cadw/` imports
either; they exist solely as reference implementations in the tests.

Docs: `docs/roadmap.json` D.2.2/D.2.3/D.3.1 -> done,
`docs/ML_TASKS_TAXONOMY.md` F1/F2 recsys and I1 survival ->
available. Notes and benches under `.pingis/algorithm_notes/`,
`.pingis/benches/`. Fast suite 2608 -> 2638 passed; slow 25 passed.
### v0.9.35 (2026-10-23) — Phase D algorithms, group 1/7

First of seven algorithm patches (25 items total, no deferral).

- **D.1.4 (S) — fairness metrics.** `cadw/ml/fairness.py` +
  `df.fairness_metrics(...)`. DP difference, equal-opportunity
  (TPR gap), FPR gap, disparate impact. Single streaming pass using
  Arrow `group_by(sensitive).aggregate(sum)` over four 0/1 masks;
  state is O(K) groups. Guarded via `check_state(K*160)`, raises
  `MemoryLimitExceeded` at 20k groups under a 4 MB budget. 11 tests;
  bench 1M rows in 0.098 s, state 1 280 B constant.
- **D.1.2 (M) — streaming Matrix Profile.** `cadw/ml/matrix_profile.py`
  + `df.fit_matrix_profile(...)`. z-normalized Euclidean profile over a
  bounded reservoir sample with exact block STOMP on the sample; state
  O(max_samples*window), independent of N. Documented as approximate
  w.r.t. the full series (`sample_fraction_`) — the exact O(N) STAMP
  variant cannot satisfy the <=1-batch RAM rule. 11 tests; bench
  1M rows / 3.17 s, state 1 114 112 B identical at 10x N.
- **D.2.1 (M) — LambdaMART.** `cadw/ml/lambdamart.py` +
  `df.fit_lambdamart(...)`. LambdaRank gradients (vectorized, upper-
  triangle NDCG delta weights), second-order tree leaves, quantile-
  binned features. `ndcg_at_k` helper. Streaming pre-pass guarded by
  `check_state`; 1M x 8 features raises `MemoryLimitExceeded` by
  design, bench therefore at 400k. 10 tests; bench 400k rows: fit
  33.5 s (sklearn pointwise ref 36.2 s), state 56 KB, NDCG@10 0.862.

Docs: `docs/roadmap.json` D.1.4/D.1.2/D.2.1 -> done;
`docs/ML_TASKS_TAXONOMY.md` O9, Matrix Profile, G1 -> available.
Notes and benches under `.pingis/algorithm_notes/`,
`.pingis/benches/`. Fast suite 2576 -> 2608 passed; slow 25 passed.

### v0.9.34-fix1 (2026-10-23) — baseline test whitelist fix

Baseline check at HEAD 8ee3181 (clean tree) found 1 fast-suite failure
that predates this task: `docs/roadmap.json` item I.3 carries the
intentional status `rejected-with-measurement` (B3 rejected with
measurement, per TASK_v0934 rejection policy), but
`tests/test_roadmap.py::test_roadmap_json_items_have_required_fields`
still only allowed (done, in_progress, planned, deferred).

- `tests/test_roadmap.py` — added `rejected-with-measurement` to the
  allowed-status tuple with a comment citing I.3 / TASK_v0934. No other
  assertion relaxed; roadmap data untouched.
- Baseline HEAD recorded: 8ee3181 (expected 685409f or later — OK).
- Fast suite after fix: 2576 passed, 2 skipped, 0 failed.
- Slow suite at baseline: 25 passed.
- WHITELISTED_FUNCS: 34 keys, verify_whitelist OK.

### v0.9.34 (2026-10-23) — Bugfix pass (8 items)

One bugfix pass over `docs/BUG_BACKLOG.md`: 8 items, no deferral.
5 fixed in this patch, 3 already fixed at baseline (proven by
repro + bench), 1 rejected with measurement.

- **B1 (P0) — quadratic `_RunStream.refill`** — Already fixed
  at baseline (6fa5e49, v0.9.25c). `refill` collects chunks in a
  list `parts` and does one `pa.concat_tables(parts)` at the end
  instead of `concat_tables([buffer, chunk])` per iteration.
  Bench `scripts/bench_v0934_b1.py`: 2000 chunks, quadratic
  0.077 s vs linear 0.0014 s (55x here, 239x in the earlier run).
  Row copies 2,001,000 vs 2,000; 12.9 GB → 0.13 GB @ rbs=200k.
  Repro test passes at HEAD.
- **B2 (P0) — `encode_categorical` Python-set** — Already fixed
  at baseline (6fa5e49, v0.9.25c). In-loop
  `check_state(len(uniques) * 180)` in Pass 1 fires before the
  set grows unbounded. Bench `scripts/bench_v0934_b2_encode.py`:
  100k uniques in a 2 MB budget → `MemoryLimitExceeded` early
  (previously ~600 MB invisible to the guard @ 10M uniques).
- **B3 (P1) — ROC-AUC pass2 Python loop** — **Rejected with
  measurement.** Bench `scripts/bench_v0934_b3_rocauc.py`: at
  N=500k the row loop takes 0.026 s with O(batch) memory; the
  Arrow `group_by(proba)` alternative takes 0.104 s (4.0x slower)
  and adds up to K=N groups (~72 MB) in the worst case. Pass 2 is
  not the end-to-end bottleneck (`external_sort` dominates AUC by
  >100x). Loop kept; decision documented inline in
  `cadw/ops/metrics.py`.
- **B7 (P1) — `_string_hash_column` Python loop** — Already
  fixed at baseline (v0.9.18, AUDIT §1.3): `dictionary_encode`
  then hash K uniques, vectorized map-back. Bench
  `scripts/bench_v0934_b7_hash.py`: 300k rows / 10k uniques,
  before 934 ms / 19.4 MB vs after 65 ms / 4.9 MB (14.4x,
  −14.5 MB peak); outputs bit-identical.
- **A9 (P1) — ML log wiring (3/6)** — Fixed. `cadw.ml._logging.log_fit`
  wired into `fit_linear` (both OLS/LinAlgError path and the
  coordinate-descent path), `fit_pca`, and all three naive-Bayes
  fitters (`fit_gaussian_nb`, `fit_multinomial_nb`,
  `fit_bernoulli_nb`). All 6 `cadw/ml/*.py` model files now
  reference `log_fit`; one-shot fits log `state_bytes` only.
  Test: `tests/test_v0934_a9_ml_logs.py`.
- **B4 (P2) — `memory_guard` fallback factor** — Fixed.
  `_UNIQUE_FALLBACK_FACTOR = 4` and a `_fallback_unique_count(col)`
  helper apply the utf8 bloat factor from part1_report §2.3 to
  the `len(set(col.to_pylist()))` fallback in
  `MemoryGuard.estimate_unique_keys` (line ~700). Previously the
  raw count underestimated downstream state by ~4x. The helper is
  WHITELISTed in
  `scripts/service/diagnose_unguarded_to_pylist.py` with an
  inline justification (sample-batch only, K ≤ sample rows).
  Test: `tests/test_v0934_b4_mg.py`.
- **A10 (P2) — empty-input contract docs** — Fixed. `external_sort`
  and `hash_shuffle` docstrings now state explicitly that empty
  input raises `ValueError` and that there is no default empty
  output. Test: `tests/test_v0934_a10_empty.py` (docstring + real
  raise on `iter([])`).
- **B6 (P2) — README version drift** — Fixed. `README.md` no
  longer hardcodes `v0.8.8` / `v0.9.24` / `~1096 tests`; the
  status header points at `cadw.__version__` as the single
  source of truth and the test-count line points at the pytest
  commands. `tests/test_ts_readme_example.py` heading anchor
  updated (`## Time series (v0.9.24)` → `## Time series`) as a
  necessary consequence — the code block under it is unchanged.
  Test: `tests/test_v0934_b6_readme.py`.

Artifacts: `.pingis/audits/v0934_bugfix_report.md`,
`.pingis/benches/v0934_{b1_refill,b2_encode,b3_rocauc,b7_hash}.json`,
`.pingis/experiments/v0934_notes.md`. Bench scripts under
`scripts/bench_v0934_*.py`. Fast suite 2576 passed / 2 skipped.

### v0.9.32 (2026-10-23)

### Audit + bugfix pass (Phase A close-out, honest tests)

- **F9 (P1, gbdt per-batch RAM gap)** — `_AllNodeHistOp` /
  `_AssignLeavesOp` / `_LeafValuesOp` now override
  `batch_memory_factor()` → 7.0 / 4.0 / 3.5. Their `update()`
  peaks (stack_numeric + gradient/hessian/lf columns + per-node
  slices) were up to 3x the yielded batch, previously unguarded
  (extra_factor=1.0). Honest tests in
  `tests/test_v0932_gbdt_memory_factor.py` — reject-at-peak,
  accept-at-peak, factor values. See `_FOR_PINGIS/AUDIT_v0931.md`.
- **AUDIT_v0931** — new full-tree AST scan of 160
  `to_pylist/as_py/to_numpy` sites; only F9 was real. F1/F2
  verified non-bugs. Summary table finalised.
- **A.6 roadmap item closed** (was deferred from Phase A): the
  audit covers the 18-site claim; remaining sites guarded/bounded/
  by-design.

### Phase D.1.3 — cost-sensitive GBDT (imbalanced classification)

- `scale_pos_weight: float = 1.0` parameter added to
  `fit_gbdt_classifier` (binary path), threaded through
  `_fit_boosting` and `_grad_hess` — weights g,h for positives
  (Friedman 2001). XGBoost-compatible meaning (`neg/pos`).
  Validation: `scale_pos_weight <= 0` → ValueError.
- `predict(X, threshold=...)` already existed (v0.9.31); confirmed
  working on both ndarray and CadwDataFrame paths.
- Honest tests: `tests/test_v0932_cost_sensitive_gbdt.py` —
  weighted model predicts strictly more positives on imbalanced
  data; threshold monotonic; validation. Sources in
  `_FOR_PINGIS/algorithm_notes.md`.
- D.3.1 CoxPH **deferred** (needs ~400-500 LOC + Sort risk-set
  primitive); rationale + sources documented.

### Test-isolation fixes (found during the audit run)

- `tests/test_permutation_importance.py::test_chunk_invariance_ranking`
  used `tempfile.mktemp()` + hardcoded `/tmp` base_dir — collides
  across xdist workers. Now uses `tmp_path`.
- `tests/test_collect_project_modes.py::test_collect_project_phase_mode`
  hardcoded old Phase-A bookkeeping (A.6 deferred). Phase A is now
  fully done, so `--phase A` correctly returns "0 files". Test
  rewritten to exercise both sides: done phase → documented error;
  Phase C → resolves real files via `cadw/ml/*.py`.

### Test status

- Full fast suite: **2557 passed, 2 skipped, 0 failed** (~67s @ -n4).
- New tests: 5 (gbdt memory factor) + 4 (cost-sensitive gbdt).

## v0.9.31: PINGIS audit + Phase D.1.1 IForest

**Audit pass (branch `pingis/dev-audit`).** Full AUDIT_REPORT_v0930.md
with per-finding class/repro/contract/fix/priority/status. P0 fixed;
P2 backlogged with rationale. All fast/slow suites green.

#### Phase D.1.1 — Isolation Forest (v0.9.31)

- `cadw/ml/iforest.py`: `IsolationForestModel` + `fit_isolation_forest(df, X_cols, n_trees=100, subsample_size=256, random_state, contamination, memory_guard)`. Reservoir sampling over the streaming batches (state independent of N); random-split isolation trees per Liu/Ting/Zhou 2008. Scoring is **per-batch** with `check_batch(extra_factor=3.0)` — no whole-dataset materialization. Design: `docs/decisions/d1_1_iforest.md`.
- Bench (`scripts/bench_iforest.py`, N=1M, 5 features): fit 3.46 s, score 92.9 s, peak 17.9 MB, state 430 KB. sklearn ref: AUROC 0.999 vs 0.999, Spearman 0.95.
- Tests: `tests/test_iforest.py` — 12 tests incl. ref-vs-sklearn, determinism, single-row, constant-feature, nulls, chunk-invariance, RAM independent of N.

#### Phase D.6.3 — Permutation feature importance (v0.9.31)

- `cadw/ml/importance.py`: `permutation_importance(model, df, X_cols, y_col=None, n_repeats=5, max_samples=1000, metric='auto'|'r2'|'accuracy'|'shift'|None, random_state, memory_guard)`. Model-agnostic: works with any `predict(np.ndarray) -> np.ndarray`. Reservoir samples `max_samples` rows over the streaming batches (state independent of N); shuffles one column at a time. Design: `docs/decisions/d6_3_permutation_importance.md`.
- Bench (`scripts/bench_permutation_importance.py`): N=1M / 5 features, wall 3.2 s, peak 1.9 MB (sample cap 1000 rows). N=100k: 0.32 s, 8.5 MB.
- Ref vs sklearn.inspection.permutation_importance: Spearman >= 0.7 on synthetic linear (test asserts).
- Tests: `tests/test_permutation_importance.py` — 10 tests (smoke, ranking, repeats-stable, determinism, too-small, constant feature, sklearn ref, chunk invariance, accuracy mode, unsupervised shift mode).

#### Phase D.6.8 — GMM density estimation (v0.9.31)

- `cadw/ml/gmm.py`: `GMMModel` + `fit_gmm(df, X_cols, n_components=3, max_iter=100, tol=1e-4, reg_covar=1e-6, max_samples=10_000, random_state, memory_guard)`. **Bounded-sample EM** — reservoir sample of `max_samples` rows (default 10k), then standard EM on the sample. State is `O(k*d)` (weights + means + diag cov), independent of N. Scoring/`predict` are per-batch with `check_batch(extra_factor=3.0)`. True per-batch streaming EM (paper's E-step sufficient statistics) is a **future D.6.x** item — honestly documented. Design: `docs/decisions/d6_8_gmm_density.md`.
- Bench (`scripts/bench_gmm.py`, k=3, d=5): N=1M fit 3.27 s, score 0.47 s, peak 16.9 MB, state 264 B. N=100k: 0.41 s, 18.4 MB.
- Ref vs sklearn.mixture.GaussianMixture (diag): mean log-likelihood within 1.0 nat per row.
- Tests: `tests/test_gmm.py` — 10 tests (smoke, two-Gaussians recover centers, finite score, determinism, argmax labels, sklearn ref, too-few-rows, chunk invariance, state independent of N, unsupervised).

#### P0 fix — `external_sort` RSS drift (A1)

- `check_batch(extra_factor=2.0)` in phase-1 loop — the very next line
  builds `select(all_columns).append_column("row_id")` = 2× nbytes.
- Oversized user batches split to ≤0.5×available **before** check_batch
  (prior code passed the check, then OOM'd inside `_write_run` peak ≈
  4× buf).
- `_RunStream._next_batch`: call reader.close() on EOF to drop the IPC
  mmap eagerly.
- `compute_fan_in`: use `(row_bytes+16)` to match `_merge_keys.check_state`
  formula (was causing check_state failures at N=1M under 10 MB budget).
- Measured absolute peak RSS @100 MB budget (clean subprocess):
  v0.9.30 103.4 MB/1M, 122.4 MB/10M → v0.9.31 104.7/114.6 →
  1.01× @1M, 0.94× @10M. Within the ≤1.1× acceptance.

#### Property-based tests

`tests/test_property_invariants.py` — 13 hypothesis invariants covering
groupby-vs-pandas, sort roundtrip/chunk-invariance/head-n/min-max,
reduce sum/count/mean, dedup count, two-key group bound. `hypothesis`
added as dev-dep.

#### ML training logs (uniform)

`cadw/ml/_logging.py` — `[fit] <model> | iter=N/M | loss=X.XXXXXX |
state_bytes=B | rss=Y MiB`, opt-in via `CADW_LOG=1`. Wired into
`kmeans`, `gbdt`, `logistic` fit loops.

#### Efficiency audit

`EFFICIENCY_AUDIT_v0930.md` — table (asymptotic, peak/row, ref 1M,
alternative, source) + R1–R5 findings. **No roadmap algorithm was
replaced**; the only candidate (`hash_join` → `hash_join_streaming`)
was rejected as a breaking API change and documented as a caller pitfall.

#### Tests touched (honest updates)

`compute_fan_in` formula values updated in `test_mustfix_easy.py` and
`test_v0925c_phase_a.py` (218→72, 21→18, 69→58) — the old constants
encoded the inconsistency A1 fixed (K passed compute_fan_in but then
failed check_state inside _merge_keys). Inline rationale added.

#### Numbers

Fast suite: 2513 passed, 5 skipped, 25 deselected in 143.8s. Slow suite:
25 passed in 91.1s. Property tests: 13/13. RAM bench: see above.

### v0.9.30: Phase B close — all 8/8 milestones

**Milestone: Phase B (Infrastructure) fully closed.** All 8
milestones done. **Context-transfer pain** — the phase's stated
goal — measurably mitigated.

#### Measured reductions

- **−7.9 MB** filesystem — B.6 cleanup (`.backups/` 916 KB,
  `ts_review_snapshot.md` 186 KB, `scripts/archive/` ~2.4 MB,
  ~280 old patches ~4.7 MB)
- **−288 KB** monolithic DECISIONS → `docs/decisions/` 24
  per-topic files ≤ ~40 KB each — B.2
- **−7872 LOC** `dataframe.py` monolith → 18 modules in
  `cadw/dataframe/` — B.1
- **`-m ts` = 103 tests** vs 2500 full (24× selective
  reduction) — B.3 markers
- **1.9× faster** parallel via `pytest -n auto` (opt-in) —
  B.3 xdist
- **75 scripts** grouped into phase subdirs — B.5

#### Sub-versions

- **v0.9.26** (B.1 + B.7 + B.8): `dataframe.py` split — 18
  modules + `tail` feature (§13.4/314 backlog closed) + recipes
  scaffold + `docs/roadmap.json` manifest. 20+ extraction
  patches (v0.9.26g through y2).
- **v0.9.27** (B.6): cleanup — 439 files deleted, 
  `bench_v0919_calibration.py` restored (test-referenced,
  §13.4/344)
- **v0.9.28** (B.2): DECISIONS split — 24 per-topic files +
  64-line index. §13.4 block → 15a-d (4 chunks, §13.4/347)
- **v0.9.29** (B.3): pytest markers (`slow`/`ml`/`ts`/`ops`/
  `viz`/`pipeline`/`e2e`) + xdist. Hook uses `item.nodeid`
  (pytest 9 safe, §13.4/356)
- **v0.9.30** (B.5): scripts reorg — 75 files grouped by phase

#### Process notes

- **14 patches** (v0.9.26z through v0.9.30), ~30 fix cycles,
  **0 production bugs**, 0 logic change in extraction patches.
- **60+ §13.4 lessons** (§13.4/278–360) — all tooling/test
  (whitelist paths, AST discovery, import forms, pytest hooks).
  Index in `docs/decisions/15a-d_13_4_уроки_part*.md`.
- **Test count:** 2488 fast → **2500 fast** (+12: +9 tail,
  +3 marker-related).

#### Honest final state

`A + B + C` phases all `completed`. Phase B sub-milestones B.1
through B.8 all `done`. Roadmap `current_version=0.9.30`.
Phase D (30+ P1 algorithms across D.1–D.6) remains `planned` —
v0.9.31+ scope, separate large effort.


### v0.9.26: B.1 — dataframe.py split into 18 modules

**Milestone: `CadwDataFrame` monolith (7872 LOC in a single
`cadw/dataframe.py`) split into 18 modules in `cadw/dataframe/`.**
20+ extraction patches (v0.9.26g-y2), ~7300 LOC redistributed,
`_core.py` 7900 → ~530 LOC. All **0 logic change** (verbatim move)
except `tail` feature added (§13.4/314 backlog closed).

**Architecture:** `CadwDataFrame` = 14 mixins composed at class
definition:
```python
class CadwDataFrame(
    _ExportsMixin, _IteratorsMixin, _MiscMixin, _IndexingMixin,
    _AggregationsMixin, _TransformersFillMixin,
    _TransformersReshapeMixin, _RollingMixin, _CumulativeMixin,
    _PerGroupMixin, _ResampleReshapeMixin, _JoinMixin,
    _MLBindingsMixin, _ColumnsMixin,
): ...
```

#### Modules (18)

`_core` (~530), `_helpers` (~170), `_exports` (~110),
`_iterators` (~190), `_column_proxy` (~320), `_columns` (~100),
`_groupby` (~220), `_misc` (~110), `_indexing` (~690),
`_aggregations` (~520), `_transformers_fill` (~710),
`_transformers_reshape` (~650), `_rolling` (~815),
`_cumulative` (~820), `_per_group` (~990),
`_resample_reshape` (~425), `_join` (~315), `_ml_bindings`
(~630), `__init__` (~60).

#### Added

- **`CadwDataFrame.tail(n=5)`** (§13.4/314 backlog closed) —
  pandas parity: n>0 last n, n=0 empty, n<0 all-but-first-k.
  Arrow-native via `pa.RecordBatch.slice`, no `to_pylist`
  (§13.4/336 guard-compliant). Mirror of `head`. 9 new tests.

#### Process notes

- **20+ patches** (v0.9.26g through y2, plus ~30 fix patches)
  across ~10 sessions.
- **0 production bugs, 0 logic change** in extraction patches.
  All failures tooling (whitelist, discovery, imports, sanity,
  test targets).
- **Test count:** 2492 fast baseline → **2501** (+9 tail).
- **Phase B status:** `completed` with caveats — B.1 (dataframe
  split, primary milestone) done; B.2 (DECISIONS split), B.3
  (pytest markers/xdist), B.5 (scripts reorg), B.6 (cleanup)
  deferred to Phase G. All four are infra hygiene per roadmap
  ("no functional changes") — no impact on CADW correctness.
  Phase B closed honestly with documented debt (§13.4/339,
  mirror of A.6 → G pattern from v0.9.26d).
- **Docs debt:** §13.4/296-336 lessons are in Phase B.1 commit
  messages; consolidated index in `DECISIONS.md` §13.4/337
  (mirror of Phase A §13.4/290 pattern).


### v0.9.26: Phase A close + Phase B start + roadmap consistency

**Milestone: Phase A (Stabilization) closed.** 10/11 items resolved;
A.6 (`18 to_pylist sites extra_factor`) deferred to Phase G — audit
report `_FOR_PINGIS/cadw_audit_report.md` absent from repo state;
specific 18 sites cannot be identified (§13.4/283). Honest close
over premature.

Phase B (Infrastructure) started — B.1 = `dataframe.py` split into 9
modules (v0.9.26, complexity L) queued after A closeout.

#### Changed (metadata, no production code)

- `docs/roadmap.json`: A.9→`done` (`_append_to_builders` absent from
  `cadw/`, verified v0.9.26c-diagnostic); A.10→`done`
  (`memory_guard.py:757` comment "v0.9.25e (A.10): fallback scales
  with state_limit, not fixed" — already shipped); A.6→`deferred`
  (audit absent, deferred to Phase G, §13.4/283). Phase A→`completed`
  with `caveats=['A.6 deferred to Phase G: audit absent']`. Phase
  B→`in_progress`.
- `docs/ROADMAP.md`: Phase A marked COMPLETED with A.6-deferred
  caveat.

#### Patches

- **v0.9.26a**: docs-debt sweep — §13.4/281 index consolidates Phase
  C lessons §13.4/262–280 with commit refs; Phase C scope verified
  complete (20 Pandas P0 + 6 ML families).
- **v0.9.26b**: roadmap consistency — Phase B→`in_progress`
  (corrected scope after first attempt aborted on A's 3 planned
  items; Phase A left untouched).
- **v0.9.26c**: diagnostic — A.6/A.9/A.10 actual code inspection
  (read-only).
- **v0.9.26d**: A closeout — A.9/A.10 `done`, A.6 `deferred`,
  Phase A `completed` with caveats.

#### §13.4/283 (NEW)

When a roadmap item claims a **specific count** (e.g. "18 sites")
backed by an audit report that is **absent from repo state**, the
item cannot be closed by inference. Options: (a) defer to a phase
where the audit can be re-requested; (b) reconstruct the count from
code — 83 to_pylist sites × receiver-type classification, error-prone.
**Choose (a): defer, document honestly.**

#### §13.4/284 (NEW)

Roadmap item names can be **mechanistically misleading**. A.6 says
"to_pylist sites extra_factor" but `extra_factor` is a `check_batch`
kwarg, not a `to_pylist` kwarg. Verify the mechanism, not just the
name — otherwise diagnostics measure the wrong thing (initial count
of "0 protected" was a false metric — `.to_pylist()` doesn't accept
`extra_factor`, so of course 0 sites had it).

#### Test counts

Unchanged (~2488 fast) — metadata + CHANGELOG only.


### v0.9.25: Phase C — Pandas P0 API (20 features across 6 batches)

**Milestone: Phase C closed.** All 20 Pandas P0 features implemented.
Zero production bugs after C-1 (v0.9.25g); all downstream churn was
test-tooling. Phase C-5 (`dropna`/`fillna`) was already shipped in
v0.9.23g — `hasattr` verify (§13.4/279 in commit-message numbering,
§13.4/249 in DECISIONS.md canonical) returned 0 mutations,
confirming the 20/20 target was met without duplicate work.

#### Added

**C-1 (v0.9.25g) — `isin` / `between` / `select_dtypes`**

- `_ColumnProxy.isin(values)` — null-safe membership via `pc.is_in`.
  `None in values` → null row matches (Kleene). Empty values → all
  False.
- `_ColumnProxy.between(left, right, *, inclusive="both")` —
  `pc.and_(ge/gt, le/lt)`. Null rows dropped (Kleene). `inclusive`
  ∈ {`both`, `neither`, `left`, `right`}.
- `CadwDataFrame.select_dtypes(include=, exclude=)` — category names
  (`"number"`, `"int"`, `"string"`, `"bool"`, `"datetime"`, …) and
  any `pa.type_for_alias`-resolvable type (`int64`, `Int64` case-
  insensitive, `string`, `timestamp[ns]`). Bracket syntax
  (`string[pyarrow]`) → ValueError.

**C-2 (v0.9.25h) — `idxmax` / `idxmin` / `pct_change`**

- `_ColumnProxy.idxmax()` / `.idxmin()` — index label of first
  max/min (nulls skipped, first occurrence wins). RangeIndex → int;
  Materialized → label; MultiIndex → tuple. All-null / empty filter
  → ValueError.
- `CadwDataFrame.pct_change(periods=1, columns=None)` — **pandas-
  parity**: `pc.divide` uses IEEE 754 (inf for x/0, nan for 0/0).
  Nulls propagate; first `periods` rows null. Output float64.

**C-3a (v0.9.25i) — `ffill` / `bfill` / `reindex`**

- `CadwDataFrame.ffill(columns=None)` — streaming per-batch
  `pc.fill_null_forward` + carry Array[1] for cross-batch leading-
  null fix. O(#cols) state, O(N) I/O.
- `CadwDataFrame.bfill(columns=None)` — reverse-sort (`__orig_rid`
  desc) + ffill-internal + re-sort (asc). 2× `external_sort`,
  O(N log N) I/O. Heavier than ffill — documented.
- `CadwDataFrame.reindex(labels)` — filtered-RangeIndex aware.
  Labels are original positionals (not filtered enumeration).
  Missing → null row via null-safe Arrow `take`. Bounded by
  `K × row_size ≤ state_limit`.

**C-3b (v0.9.25j) — `cut` / `qcut` / `duplicated`**

- `CadwDataFrame.cut(x, bins, labels=, right=)` — external_sort +
  `pc.bucketize` + bin edges. Returns int labels.
- `CadwDataFrame.qcut(x, q, labels=)` — quantile + cut combination.
- `CadwDataFrame.duplicated(subset=, keep='first'|'last'|False)` —
  row_id-aware mask via HashShuffle + groupby, preserves order.

**C-4a (v0.9.25k) — `str.slice` / `str.cat` / `str.count`**

- `_StrAccessor.slice(start=, stop=)` — negative indices supported
  (pandas parity).
- `_StrAccessor.cat(others=, sep=)` — string concatenation; null
  propagation per pandas.
- `_StrAccessor.count(pat)` — non-overlapping matches via
  `pc.count_substring`.

**C-4b (v0.9.25l) — `dt.strftime` / `dt.tz_convert` / `str.split`**

- `_DtAccessor.strftime(fmt)` — `pc.strftime`, naive + tz-aware.
- `_DtAccessor.tz_convert(tz)` — `pc.cast` (naive → UTC first via
  `pc.assume_timezone`).
- `_StrAccessor.split(pat, max_splits=-1)` — list<string> output via
  `pc.split_pattern`.

**C-5 (pre-existing since v0.9.23g) — `dropna` / `fillna`**

- `CadwDataFrame.dropna(how="any", subset=None)` — lazy, composes
  notnull() filters. `how='any'` drops rows with any subset null;
  `how='all'` drops rows where all subset nulls.
- `CadwDataFrame.fillna(value, columns=None)` — streaming per-batch
  `pc.fill_null`, scalar value type-coerced via `pa.scalar`.
  **Not reimplemented in Phase C** — `hasattr` verify revealed
  existing implementation with 12+ tests. 0 production changes in
  v0.9.25m.

#### Fixed (Phase C fix cycles)

- **§13.4/262 (commit-message):** `select_dtypes` type matching —
  `str(t) == s` failed on `str(pa.float32()) == "float"` (actual
  `"float"`) and `str(pa.float64()) == "double"` (actual `"double"`).
  Fix: `pa.type_for_alias(s)` first, fallback str-compare.
- **§13.4/263 (commit-message):** `select_dtypes` accepts `"Int64"`
  (case-insensitive via `type_for_alias`) but rejects
  `"string[pyarrow]"` (bracket syntax, unparseable).
- **§13.4/265 (commit-message):** v0.9.25h-fix1 — stray
  `assert math.isinf(vals[2])` copy-pasted from sibling test.
  Fix: `vals[2] == 3.0`.
- **§13.4/277 (commit-message):** v0.9.25l-fix1 —
  `test_tz_convert_non_temporal_raises` expected
  `"timestamp required"` but `.dt` accessor `__init__` gate fires
  first with `"expected timestamp or date"`. Test-only fix;
  production correct.
- **§13.4/278 (commit-message):** v0.9.25l-fix1-corrected —
  `verify_state` source-text marker `"def split(self"`
  false-negative (multi-line signature). Fix:
  `hasattr(imported_class, "method")` — runtime attribute presence.

#### Process — Phase C fix-cycle tally

| Batch | Attempts | Prod bugs | Notes |
|---|---|---|---|
| C-1 (`v0.9.25g`) | 4 | 1 (`type_alias`) | 2 test-overreach |
| C-2 (`v0.9.25h`) | 2 | 0 | REPL-first abort + fix1 stray assert |
| C-3a (`v0.9.25i`) | 1 | 0 | anchor miss (0 mut) + reindex-filtered pre-ship fix |
| C-3b (`v0.9.25j`) | 1 | 0 | — |
| C-4a (`v0.9.25k`) | 1 | 0 | — |
| C-4b (`v0.9.25l`) | 3 | 0 | REPL-first abort + fix1 + fix1-corrected |
| C-5 (pre-existing) | 0 | 0 | `hasattr` reveal → skip |
| **Total** | **12** | **1** | **92% test-tooling** |

**Key learnings (§13.4/262–279 in Phase C commit messages;
canonical `DECISIONS.md` backfill deferred to v0.9.26 docs-debt
sweep):**
- §13.4/264 — REPL-first (10–12 pre-mutation checks) saved 2 cycles
  in v0.9.25h and prevented wrong `pc.divide` semantics (IEEE 754
  inf/nan, not null-out).
- §13.4/266 — post-anchor re-verify each feature; caught latent
  reindex-filtered bug (v0.9.25i-corrected) pre-ship.
- §13.4/278 — `hasattr` beats source-text for presence checks.
- §13.4/279 — `hasattr` reveal ≠ rebuild; diagnostic-only response
  when feature already exists.

#### Version

- `pyproject.toml`: `0.9.14` → `0.9.25` (was 11 versions
  stale since v0.9.14 close-out).
- `cadw.__version__`: `0.9.14` → `0.9.25` (aligned).

#### Test counts

- ~2314 fast at v0.9.25f-a
- ~2488 fast at v0.9.25l (after Phase C)
- **No new tests in v0.9.25n** (docs-only close-out)

#### Docs debt (resolved in v0.9.26a)

- **§13.4 numbering drift**: Phase C lessons §13.4/262–280 existed
  only in commit messages; consolidated §13.4/281 index in
  DECISIONS.md now backfills them with commit references. Debt
  closed in v0.9.26a. Future lessons append from §13.4/282 onward.
### v0.9.24: time series toolkit + temporal hierarchical forecasting

Comprehensive time-series extension. All new code lives in `cadw/ts/`
and routes through the existing primitives + MemoryGuard.

- **Features** (`cadw/ts/features.py`): lag/lead, diff (regular +
  seasonal), calendar, cyclical sin/cos, Fourier, holiday/weekend,
  `make_features` composite pipeline. Group-aware (per-entity, no
  cross-entity leakage). O(batch) state, explicit extra_factor.
- **Window statistics** (`cadw/ts/stats.py`): tsfresh-subset
  (mean/std/min/max/median/skew/kurt, q10..q90, abs_energy/energy/
  cid, crossings, slope, entropy, count_above_mean, autocorr_lagK) +
  rolling_crosscorr. Shared per-group deque, RAM formula documented.
- **Forecasters** (`cadw/ts/models.py`): NaiveForecaster, SARIMAForecaster
  (CSS/Hannan-Rissanen, bounded auto-AIC grid, seasonal orders; AIC
  may under-difference trending series — use explicit order=), 
  GBDTForecaster (default mode='direct' with deferred per-horizon
  training; recursive and multi_output modes). ForecasterBase API,
  get_forecaster name registry ('naive'/'sarima'/'gbdt').
- **Backtesting** (`cadw/ts/backtest.py`): expanding/sliding CV,
  multi-series, MAE/RMSE/MAPE/sMAPE/MASE(m=seasonal_period), optional
  pinball; forecaster cloned per fold via deepcopy.
- **THF** (`cadw/ts/thf.py`): TemporalHierarchy with temporal
  aggregation, bottom-up / top-down-proportional / optimal-OLS /
  MinT-WLS (diagonal residual-variance weights, 1e-8 floor, OLS
  fallback). Multi-series, predict_level.
- **Accessor**: `CadwDataFrame.ts` thin wrappers matching the .dt/.str
  pattern. `cadw.ts.__all__` sorted and complete (16 symbols).
- **Bugfix** (pre-existing): `sort_values(by=list, ascending=list)`
  silently treated list ascending as truthy → all columns ascending.
  Fixed: per-column ascending sequence, length-checked.


### v0.9.15–v0.9.22: post-release v0.9.14 fixes + audit round 3

**8 patches** since v0.9.14 tag. Full per-patch detail via
`git log --oneline v0.9.14..master`. Highlights:

- **v0.9.15** — `patch_utils.dedent_method_source` + T2.7 meta-guard
  (no inline `dedent(getsource(...))` outside helper).
- **v0.9.16** — DECISIONS §13.4/211–225 append (15 meta-lessons
  from v0.9.7–v0.9.14); tag v0.9.14.
- **v0.9.17** — DECISIONS §13.4/178–189 append (round-1 audit
  lessons); seaborn deprecation filter; category-beyond-memory
  tests (K × per_key > avail, §13.4/218 null-key regression).
- **v0.9.18** — AUDIT.md round-1 fixes: §1.1 `_ArrowTail` (shift),
  §1.2 rank numpy fast path, §1.3 string hash dict-encode, §1.5
  `to_list` sample-based guard, §1.6 `MaterializedIndex` numpy.
  3 fix-cycles for null/edge cases.
- **v0.9.19** — closed-loop memory correctness:
  `MemoryGuard(calibrate=True, runtime_monitor=True)` — 3 tiers
  (pyarrow pool exact / psutil RSS sampled / calibration log
  warns at C > 2.0); `reduce()` wired; `bench_v0919_calibration.py`.
- **v0.9.20** — peak-polling `calibrate(sample_peak=True)`;
  bench decomposition columns (`pred_batch` / `pred_state` /
  `pred_total` / `C_peak` / `C_batch`); DECISIONS §13.4/226–232
  append. **5 fix-cycles** — §13.4/238 anti-pattern (4-feature
  patch → cascade).
- **v0.9.21** — AUDIT §1.4 (last CONFIRMED item):
  `_group_boundaries_arrow` + per-group shift/rolling Arrow-native
  group boundaries; §13.4/233–240 append. 1 fix-cycle (per-group
  as_py → batch-level slice.to_pylist).
- **v0.9.22** — AUDIT_ML round 3 fixes: gbdt multiclass
  `est_n_state = n·8·(1+4K)` (was n·K·8·3 — missed y_all + K-th
  round_cur_leaves, 880 MB under-report @ N=10M K=10); gbdt
  `_validate_binary_y` single-pass; NB `C_ASSUMED=64` (was 32);
  kmeans `batch_memory_factor=2k` (S1: 800 MB real vs 4 MB check
  @ k=100 closed); `_flush_buf` bounded-by-construction comment;
  DECISIONS §9.2.20–22 + §13.4/241–245 append.

**Key outcomes:**
- AUDIT.md round-1 **all 5 CONFIRMED closed** (§1.1–§1.6).
- AUDIT_ML_v0922 round-3: 24 suspected → **7 CONFIRMED** (all
  fixed v0.9.22) → 17 **RETRACTED** under measurement. Two S1
  bugs closed: gbdt multiclass formula (880 MB miss), kmeans
  batch_memory_factor (200× under-report at k=100).
- **Closed-loop memory** (v0.9.19+): `MemoryGuard` can measure
  runtime — pyarrow pool exact, RSS sampled, calibration log
  warns at drift > 2.0. `bench_v0919_calibration.py` automates
  AUDIT workflow.
- **25 meta-lessons** documented (§13.4/178–245).

**Deferred:**
- ~~**v0.9.22b**~~ — **closed v0.9.22b**: extra_factor at 5
  sites (rank.py _compute_streamable/_pass1/_pass2,
  resample.py sorted_input, pivot.py stream_sorted);
  groupby.py skew factor=3.0 deemed adequate
  (grouped.to_pylist bounded by heavy_keys ≤ 20, not batch).
- **v0.9.23** — kmeans++ reservoir (quality: +36.9% inertia on
  skewed data); `RobustScaler` O(d) sort batching.
- **v0.9.24+** — A1 `dataframe.py` split; Pingis round 4;
  v1.0 readiness.

**Test count:** 1916 fast + 24 slow (was 1876+24 at v0.9.14;
+40 fast, 0 slow). Whitelist 31 keys.


## [0.9.14] — 2026-10-21

### v0.9.7–v0.9.14: audit round 2 (16 patches, 3 silent data-loss bugs)

**Milestone**: closes all 21 original findings from
`_FOR_PINGIS/cadw_audit_report.md` (R1–R6, B1–B4, N1–N7)
+ 13 findings discovered during implementation. Three
silent data-loss bugs found and fixed — one 3 months in
prod, undetected by tests:

- **C1**: `_per_group_apply` mixed int/float groups silently
  truncated fractional floats (`from_pylist(schema=int64)`).
- **§13.4/218**: null-key rows dropped in `groupby` skew path
  (`filter(null)` treats null as false — data loss in both
  heavy and light paths). v0.3.4 → v0.9.12-fix2, ~3 months.
- **§13.4/223**: `_infer_arrow_type_from_value` checked
  `isinstance(v, int)` before `bool` (Python bool is int
  subclass) → `True` inferred as int64 → `ArrowTypeError`.
  F12-6a bool-widen was dead code.

Plus 13 new §13.4 meta-lessons (§13.4/211–223) on patch-
script safety (compile-before-write, partial-state abort),
AST-check precision (negative→AST, receiver-type filter),
and falsy-default traps (dedent-before-parse, fill_null).

#### Added

- **R1**: `MemoryGuard.check_partition_writers(P, bytes_per_writer)`
  — real per-writer footprint (audit bench: 1.2–4.4 KB/writer).
  Prior `check_partition_buffers` was vacuous by construction
  (`P × (limit // P) ≤ limit`).
- **R2**: `MemoryGuard.reestimate_from_batch(batch)` +
  `get_batch_rows(..., sample_batch=)` — schema underestimate
  fixed (10× for string columns: 500 B/row est vs 2040 actual).
- **B1**: `propagate_chars` for `hash_shuffle`/`dedup`/`rank`/
  `quantile`/`resample` — audit `plan_pipeline([filter,
  hash_shuffle, groupby])` crash closed.
- **R4**: `TopKOp` (`topk_to_disk`) streaming — `nlargest`/
  `nsmallest` **~3946×** faster (audit bench N=2M, k=10).
- **R5**: `DescribeOp` (`describe_columns`) fused — `describe`
  **~109×** faster (audit bench N=2M, 2 cols).
- **R6**: `pivot_to_disk` streaming Arrow-native — replaces
  `to_pandas().pivot()` (26.8 MB → ~10 MB for 200k/300×300;
  100k×20 passes where dense failed).
- **F1**: `patch_utils.run_verify_in_subprocess(fn)` — fresh
  subprocess dogfood; 8× §13.4/178 stale-module-cache fixed.
- **T2.1**: `patch_utils.apply_or_fail` + `verify_anchors` —
  fail-fast with context, compile-before-write (§13.4/211).
- **T2.2**: `assert_no_call_in_source(receiver_type=)` — precise
  AST check (§13.4/217): `X[i].as_py()` per-row flagged,
  `fn(...).as_py()` aggregate allowed.
- **T2.6**: `patch_utils.dedent_method_source(fn)` — dedent +
  parse helper (§13.4/215/216).
- **T2.9**: `apply_or_fail(validate=)` — format-aware validation
  (`toml`/`json`/`python`/`none`); extension-inferred default
  (§13.4/221).
- **C1**: `_maybe_upgrade_schema` — int→float64 widen in
  `_per_group_apply` mixed groups.
- **F12-6a/b**: bool→float64 widen + clear `RuntimeError` on
  non-widenable mismatch (was cryptic `ArrowInvalid`).
- **F12-6c**: `_infer_arrow_type_from_value` bool-before-int
  reorder (§13.4/223).
- **F12-5**: `_stream_values_at_positions` Arrow-native —
  quantile ~100–400× faster for high quantiles.
- **F12-4**: `_detect_heavy_keys` Arrow-native — groupby skew
  detection ~10× faster, single C++ `group_by` pass.
- **F12-1**: `topk_to_disk` incremental `acc_bytes` —
  O(P²) → O(P).

#### Fixed

- **C1: silent int truncation** in `_per_group_apply` when first
  group returns int and later returns fractional float —
  previously truncated (silent data loss) or `ArrowInvalid`.
- **§13.4/218: silent null-key data loss** in `groupby` skew
  path: `_compute_heavy_mask` produced null mask on null-key
  rows (`pc.equal(null, non_null)` = null via Kleene);
  `filter(null)` drops → rows lost from both heavy and light.
  Fix: `pc.fill_null(mask, pa.scalar(False))`.
- **§13.4/223: bool→int silent misinference** in
  `_infer_arrow_type_from_value` (order: `isinstance(v, int)`
  before `bool`). `True` inferred as int64; `from_pylist(
  schema=int64)` → `ArrowTypeError`. F12-6a bool-widen was
  dead code. Fix: bool before int.
- **PF4**: `_maybe_upgrade_for_row` per-row (v0.9.10) → per-
  flush `_maybe_upgrade_schema` (v0.9.11) — for 10M groups ×
  K=10, ~100M list allocs + ~1B attr checks eliminated.
- **N1–N5**: `ColumnTransformer.to_dict` (`__serialization_name__`),
  `dt.floor/ceil/round` (`TypeError` in init for date32/64),
  groupby combiner `check_state`, `_merge_fanout` duplicate
  column (`RuntimeError`), `PCA` docstring, `_GroupBy.apply`
  docstring.
- **B2–B4**: `pca.py` F821 (`TYPE_CHECKING`), `pipeline.py`
  `-> Path` module-level import, `PCA` fitted-model docstring.
- **U1, U2, U3**: `test_dt_ceil_date64` test data (pa.date32
  → pa.date64 via from_table), `_concat_batches` per-column
  `pa.concat_arrays` (1× alloc vs 3×), `pyproject.toml`
  version 0.1.0 → 0.9.14.
- **F12-8**: `PredicateFilter.mask_for_batch` null semantics
  documented (Kleene-drop numeric vs fill_null string —
  intentional asymmetry).
- **§13.4/220**: shadowed module import (`from cadw.ops import
  quantile` gets function, not module) — tests and dogfoods
  use direct submodule import; meta-guard added.

#### Changed

- `pyproject.toml`: version 0.1.0 → 0.9.14 (was stale since
  initial scaffold, drifted 3+ months from CHANGELOG).
- `cadw.__version__` = `"0.9.14"` (was undefined; `pip show`
  and `import cadw` now agree).

#### Removed

- (none)

#### Deferred to v0.9.16

- **A1**: `dataframe.py` split (5404 lines → 7 modules) —
  architectural, ~4h, risky. Do after v0.9.14 tag.
- **§11.5 backlog**: N6 (`Dataset.iter_batches` guard — user
  responsibility), R3 (planner spill stub, v0.2).
- `DECISIONS.md §13.4/211–223` — new meta-lessons doc block.

## [0.9.6] — 2026-10-21

### v0.9.6 — pandas removal + own XXH64 hash + streaming PivotOp + scaffold

**Milestone**: `import cadw` no longer pulls pandas. Core (`ops/`,
ML, pivot) is pandas-free; pandas is opt-in via
`cadw-framework[pandas]` for `to_batches(format="pandas")`.

#### Added

- **`CadwDataFrame.from_table(pa.Table)`** — zero-copy in-memory df;
  no disk I/O. Backed by `_InMemoryDataset` (read-only `Dataset`
  subclass).
- **`_InMemoryDataset(pa.Table)`** — `iter_batches` via
  `table.to_batches()`; `add_column` raises `NotImplementedError`.
- **`_ColumnProxy` aggregators** — `.mean()`, `.sum()`, `.min()`,
  `.max()`, `.std(ddof=1)`, `.count()` via existing `ReduceOp` (O(1)
  state, no pandas).
- **`_ColumnProxy` materializers** — `.to_list()`, `.to_numpy()`,
  `.to_arrow()` with `check_state` before.
- **`CadwDataFrame.__len__`** — same as `.count()`.
- **`CadwDataFrame.sort_values` in-memory fast path** —
  `pa.Table.sort_by` directly (avoids 1–5 ms `external_sort` per
  group; critical for per-group apply on 100k groups).
- **`cadw/ops/pivot.py::pivot_to_disk`** — streaming Arrow-native
  pivot (R6). Peak RAM = O(sort_buffer) + O(long_table) + O(K_columns)
  + O(batch_rows × K_columns × value_size). Independent of K_index.
  Replaces `to_pandas().pivot()` (26.8 MB → ~10 MB for 200k/300×300;
  100k×20 cols passes under 50 MB, previously fail-fast on dense
  estimate).
- **`cadw/core/_k_est.py::estimate_k_sample_arrow`** — Arrow-native
  K estimate (`pc.dictionary_encode`); used by groupby + dedup.
- **`cadw/patch_utils.py`** scaffold v1 (v0.9.6d):
  - `assert_no_call_in_source(src, *names)` — AST, no `.foo()` Call
    nodes; comments/docstrings invisible (§13.4/183).
  - `assert_call_order(src, first, second)` — AST Call ordering
    (§13.4/185).
  - `is_patched(fn, *markers)` — `inspect.getsource(fn)` scoped
    idempotency (§13.4/182).
  - `run_smoke_inline(script)` — `uv run python -c` (cwd on
    `sys.path[0]`; §13.4/184).

#### Changed

- **`shuffle.py` hash: own XXH64-based** (v0.9.6a) — replaces
  `_pandas_hash_column` + `_python_hash_column` with
  `_vectorized_hash_numeric` (numpy element-wise over uint64 bitcast
  for int/uint/bool/float/timestamp/date/time) +
  `_string_hash_column` (per-element `xxhash.xxh64`, no
  repr/encode wrapper). No `import pandas`. Cross-batch consistency,
  null sentinel, splitmix64 avalanche preserved.
- **`groupby.py::_estimate_k_sample`** — Arrow-native union of
  per-batch dictionaries (was `to_pandas().drop_duplicates()`).
- **`dedup.py::_estimate_k_sample`** — delegates to
  `estimate_k_sample_arrow` (was pandas).
- **`_per_group_apply` — BREAKING** for pd.Series/pd.DataFrame users.
  `fn(CadwDataFrame) → scalar | dict[str, scalar] | CadwDataFrame`.
  No pandas inside `apply`.
- **`pivot_table`** — delegates to streaming `pivot_to_disk`;
  behaviour preserved (same values/column-order/row-order); 100k ×
  20 cols passes where dense failed.
- **`pyproject.toml`**: `pandas` removed from `dependencies`;
  `[project.optional-dependencies].pandas = ["pandas>=3.0"]`;
  `[dependency-groups].dev += pandas` (test reference helpers).
- **`_convert_batch(format="pandas")`** — friendly `ImportError`
  with install hint (`pip install 'cadw-framework[pandas]'`).

#### Fixed

- `_estimate_k_sample` — no `to_pandas()` (dedup, groupby).
- `pivot_table` — no `to_pandas()` / `pd.pivot()` (R6).
- `_ColumnProxy` — aggregators work natively.
- `sort_values` — fast path for in-memory datasets.
- `test_pivot_too_large_raises` — match="pivot" (pivot.* names).
- `test_pivot_table_has_check_state_before_to_pandas` — replaced
  with two-level contract tests (`delegates_to_streaming` +
  `pivot_to_disk_guarded_before_output`).

#### Removed

- `_pandas_hash_column`, `_python_hash_column` (shuffle.py).
- `tests/test_shuffle_fallback_warning.py`.
- `pd.Series` / `pd.DataFrame` return contract in `_per_group_apply`.

## [0.9.5] — 2026-10-20

## [0.9.4] — 2026-10-20

### Добавлено (v0.9.4b-2 - transform / describe / dt / str)

- **`df.groupby(g).transform(agg_columns, agg_funcs)`** —
  aggregate + left-merge back. Same signature as `agg()`. Result:
  all original columns + new `{col}_{func}` columns. Row order =
  partition-concat (merge semantics) — documented 11.5. For
  pandas order, add `__orig_rid` + `sort_values()` after.
- **`df.describe(columns=None)`** — numeric summary dict
  `{col: {count/mean/std/min/25%/50%/75%/max}}`. Default: all
  numeric cols. Cost: 5 Reduce passes + 1 sort per column.
- **`df['ts'].dt.floor(freq)` / `ceil(freq)` / `round(freq)`** —
  bucketize timestamp. `freq` = `'5s'`, `'1h'`, `'1D'` etc.
  Appends `{col}_floor` / `_ceil` / `_round` (same dtype).
- **`df['ts'].dt.week_of_year`** — ISO week (1-53). int16.
- **`df['s'].str.strip/lstrip/rstrip`** — whitespace trim.
  Appends `{col}_strip` / `_lstrip` / `_rstrip`.
- **`df['s'].str.starts_with(pat)` / `.ends_with(pat)`** — return
  `MaskFilter` for `df[mask]`. `na=False` (pandas default).
- **`df['s'].str.replace(pat, repl, regex=False)`** — literal
  substring replace (default). `regex=True` for regex. Appends
  `{col}_replaced`.
- **`PredicateFilter` extended** with `starts_with`, `ends_with`
  ops (per-batch Arrow `pc.starts_with` / `pc.ends_with`).

### Ограничения (v0.9.4b-2, 11.5)

- **`transform` row order** = merge partition-concat, not
  original. Streaming order-preserving transform → v0.9.5 backlog.
- **`describe` cost**: O(d) sorts for quantiles. Fused `DescribeOp`
  (single pass, no separate min/max) → 11.6 backlog.
- **`dt.floor/ceil/round`** assumes post-1970 timestamps (positive
  int64). Pre-1970 edge cases (negative int) diverge from pandas
  floor semantics. `dt.round` = round-half-up.
- **`dt.week_of_year`** = ISO week. US week numbering → backlog.
- **`str.replace`** default `regex=False` (literal). pandas default
  is `regex=True`; CADW chose literal for predictability.

### Тесты (v0.9.4b-2)

+22 fast (`test_dataframe_v094b2.py`): transform (4), describe (4),
dt.floor/ceil/round/week_of_year (5), str.strip/starts_with/
ends_with/replace (5). Plus 4 error cases.

- fast: **~1640** (+22). slow: **20** (unchanged).

## [0.9.3] — 2026-10-20

### Исправлено (v0.9.3e - shadowed _read_for_resplit, S2)

**C1 (S2, из v0.9.2 audit):** `_local_groupby` содержал
**два** `def _read_for_resplit()` подряд. Первый (intended
оптимизация: reuse in-memory `table` когда partition file fits
но dict overflow — избегать disk re-read) был shadowed вторым
и никогда не выполнялся.

**Почему это важно:**
- Dead code — будущий читатель думает что optimization работает.
- Optimization никогда не активировалась (второй def перекрыл) →
  никогда не тестировалась → потенциально buggy. Правильно не
  активировать без отдельного prototype.

**Fix (AST-based, не text-match):**

- Delete first (dead) `_read_for_resplit`.
- Current behavior (re-read from disk) сохранён — correct, если
  suboptimal.
- Comment marker на месте удалённого def.

**Почему AST, не text-match:** unicode em-dash в Russian comments
и escape sequences ломали text-match в v0.9.3a (6 round fix).
AST `FunctionDef` nodes — прецизионные `lineno`/`end_lineno`,
устойчивы к content.

**Backlog (11.6):**

- Restore optimization: `_read_for_resplit` reuse `table` если
  available (file fits but dict overflow). Prototype needed:
  verify correctness (in-memory split vs disk re-read — должны
  дать same partition content). v0.9.4.

**Тесты (v0.9.3e):** +4 fast (`test_groupby_dead_code.py`):
exactly 1 `_read_for_resplit` def, no duplicate same-scope defs,
comment marker present, dict-overflow recursion still works.

- fast: **~1587** (+4). slow: **20** (unchanged).

### Исправлено (v0.9.3d - StdOp int-column truncation, S2 fix)

**C2 (S2, из v0.9.2 audit):** `StdOp.update` использовал
`pa.scalar(mean_b, type=arr.type)` для subtract. Для int64 arr:
`pa.scalar(2.5, type=int64)` — silent truncation до 2. M2 computed
from wrong deviations -> wrong std когда batch mean non-integral.

**Примеры:**

| Input (int64) | Buggy | Correct (numpy ddof=1) |
|---|---|---|
| [1,2,3,4] | 1.4142 | 1.2910 |
| [1,2] | 1.0 | 0.7071 |
| [0,0,0,10] | 5.0332 | 5.0 |
| [1,2,3,4,5] | 1.5811 | 1.5811 (integral mean -> OK) |

Float columns unaffected (`pa.scalar(float, type=float64)` exact).
`groupby().agg("std")` — Arrow native `group_by().aggregate`, не
задет. ML `fit_*` — не использует `StdOp`.

**Fix:** cast to float64 before subtract.

```python
arr_f = pc.cast(arr, pa.float64())
diff = pc.subtract(arr_f, pa.scalar(float(mean_b), type=pa.float64()))
```

**Semantic change (int columns):** correct value where previous was
mathematically wrong. CHANGELOG flags as breaking for downstream
code that relied on incorrect result. No existing test pinned the
buggy value (both `test_std_invariant` and
`test_std_precision_large_offset` use float data).

**Правило (13.4/176):** при subtract/add со scalar из mean/median —
**всегда** cast array to float64. Int arithmetic truncates silently.
Same class as 13.4/163 (domain validation), 13.4/175 (memory formula
scope) — hidden mismatch between formula и implementation.

**Тесты (v0.9.3d):** +11 fast (`test_stdop_int_correctness.py`):
known [1,2,3,4] test (+ sanity not buggy), int==float, 5 param cases
vs numpy, float unchanged, nulls, int32, AST guard.

- fast: **~1583** (+11). slow: **20** (unchanged).

### Исправлено (v0.9.3c - P-D2 dedup string keys, S2 fix)

**P-D2 (S2, из v0.9.2 audit):** `_local_dedup` guard
`check_state(1.5 * table.nbytes)` extrapolated from int-key
prototype v0.8.11a (K<<N, measured 0.23x). Bench v0.9.2 audit
(bench_dedup_phases.py, 400k rows, K=N strings, table 9.20 MB):

- dictionary_encode +56.95 MB
- group_by +86.14 MB
- is_in +103.38 MB
- filter +103.58 MB (peak)

Peak / table = **11.26x** (strings at K=N). Int64 K=N = 1.97x.
Prior 1.5x bound was **7.5x underestimate** for string K=N --
OOM instead of MemoryLimitExceeded.

**Fix:** type-based multiplier (conservative bench-backed):

- **String subset:** 12 * table.nbytes (11.26x measured + 7% safety).
- **Numeric subset:** 2.5 * table.nbytes (1.97x measured + 27%).

`_has_string_subset` checked via `pa.types.is_string` /
`is_large_string` on `subset_cols`. `_dedup_peak_mult` drives
`check_state`.

**Почему dictionary_encode даёт 11x peak:**
- Копирует **каждое** string value (не только dictionary).
- Arrow StringArray: data buffer + offset buffer (int32).
- BufferBuilder geometric growth (~2x peak of final).
- + int32 indices buffer (4 B / row).
- + hash table for group_by (K entries).

**Over-conservative:** K<<N strings (measured ~0x) still need 12x.
Sample-based K_est tightening -> v0.9.4 backlog.

**Тесты (v0.9.3c):** +7 fast (test_dedup_estimator_v093c.py):
string K=N tight-budget raises, string K=N ample passes, int K=N
ample passes, 3 AST guards (12x, 2.5x, type check), multiplier
variable. Existing `test_dedup_removes_stopgap_5x` updated: 1.5x
-> 12x/2.5x.

- fast: **~1572** (+7). slow: **20** (unchanged).

### Исправлено (v0.9.3b - P-S1 sort merge fan-in, S2 fix)

**P-S1 (S2, из v0.9.2 audit):** `external_sort` вызывал
`compute_fan_in(read_batch_size=1000)` — это output flush
frequency, не reader batch size. Реальные on-disk batches
(от `_write_run`) — `_WRITE_RUN_CHUNK = 50_000` rows. Reader
держит 50_000 × r на batch, не 1000 × r. 50x underestimate
merge peak. Measured: 126.81 MB при 67.1 MB budget (1.9x).
**MemoryLimitExceeded не поднимался.**

**Fix:**

- `_WRITE_RUN_CHUNK = 50_000` — module constant.
- `_write_run` uses it instead of literal `chunk = 50_000`.
- `compute_fan_in` in `external_sort` uses `_WRITE_RUN_CHUNK`
  (on-disk batch size) instead of `read_batch_size`.
- `_merge_keys` gets `memory_guard` param; `check_batch` on each
  read batch (defense-in-depth).

**Покрывает:** `sort_values`, `rank`, `quantile`, `resample`,
`roc_auc`, `TimeSeriesSplit` — все через `external_sort`.

**Trade-off:** k уменьшается ~50x (больше merge levels при tight
budget). Peak теперь fits; merge slower для малого budget.
Safety > speed.

**Тесты (v0.9.3b):** +5 fast (`test_sort_merge_p_s1.py`):
constant exists, AST compute_fan_in uses _WRITE_RUN_CHUNK, AST
check_batch in _merge_keys, tight-budget smoke, regression
sort correctness.

- fast: **~1565** (+5). slow: **20** (unchanged).

### Исправлено (v0.9.3a - KFold keyed PRNG, S1 fix)

**P-CV1 (S1, из v0.9.2 audit):** `KFold.split` материализовал
O(N) fold-id массив (shuffle=False: ~12*N; shuffle=True: ~20*N)
**без** `MemoryGuard.check_state`. Для N >> RAM - guaranteed OOM,
не `MemoryLimitExceeded`. Нарушение ядра гарантии CADW.

- **`KFold.shuffle=False`** (default): `fold_id = row_id % k`,
  computed per batch. O(batch) state. 1:1 с прежним поведением.
- **`KFold.shuffle=True`** (default, O(batch) state): keyed
  splitmix64 PRNG. `fold_id = splitmix64(row_id*GOLDEN + seed) % k`.
  Детерминистичен per row_id. Split-quality: fold sizes ~binomial
  (mean n/k, per-fold sigma ~= sqrt(n/k)).
- **`KFold(materialize=True)`** (opt-in backward compat):
  exact-balanced shuffle, O(N) RAM, `check_state(20*N)` guarded.
- **`random_state=None`** -> seed=42 (deterministic reproducibility).
- **`_attach_column_with_fn(fn)`**: streaming helper
  (`fn(offset, n) -> pa.Array`). O(batch) state, no O(N) array.

### 13.4/173 - docstring memory claim vs actual allocation

Pre-v0.9.3a docstring: "O(N) int8 fold_id array". Actual code:
int32 fold_id + int64 arange/permutation temporaries = 12-20*N bytes.
3-4x understatement + zero MemoryGuard check.

**Правило:** docstring memory claim must match actual code. Test
with N >> RAM (core scenario) обязателен для каждого публичного op
с O(N) state - small-N тесты скрывают O(N) bugs.

### 13.4/174 - patch-script embedded escape trap (3 rounds)

v0.9.3a: 6 fix iterations, все - patch-script syntax bugs, не CADW:

- `\n` внутри triple-quote template -> real newline in file.
- `'''` inside `'''...'''` -> terminated early.
- `"""` inside `"""..."""` -> terminated early.
- Pattern wrote `->` (ASCII), file had `->` (U+2192).
- 2 stale tests after behavior change (13.4/160).

**Rules:** no triple-quote in patch scripts; content as
`"\n".join(list)`. Pattern matches signatures + structural lines
only, no docstrings/unicode. If not found - print context. Split
code/docs patches. Self-compile-check before run.

### Тесты (v0.9.3a)

+11 fast (`test_cv_keyed_prng.py`): default folds, keyed PRNG
deterministic, different seeds differ, default seed=42, binomial
balance, no-shuffle unchanged, materialize exact-balanced, OOM
guard, small-N OK, AST guard (no np.arange(N)/permutation(N) в
default path), streaming helper present, large-N tight-budget.

2 existing tests updated (`test_cv.py`): `shuffle=True` now binomial
(use `materialize=True` for exact).

- fast: **~1560** (+11). slow: **20** (unchanged).

### Ограничения (v0.9.3a, 11.5)

- `shuffle=True` (default): fold sizes ~binomial, not exact-balanced.
- `KFold.split()`: one `df.count()` scan (O(N) time, O(1) memory).
- Iterating all folds: `n_splits * N` row touches, O(1) memory per
  iteration.

## [0.9.2] — 2026-10-20

### Исправлено (v0.9.2f-fix2 — registry 4→8)

- **`test_list_step_types_default`** — ожидаемый список 4 → 8 items
  (v0.9.2f auto-registered 4 scalers via `_register_scalers()` at
  `cadw/scalers.py` import). Closed-list assert is intentional guard
  against test-registry pollution; grows with each new built-in.
- Rationale comment: built-ins updated in same patch that adds new
  step. Patch review catches typos in new step name (e.g.,
  `"StandartScaler"` → test fails, not silent).

### §13.4/172 — closed-list stale-bait on registry growth

Built-in registry grew 4→8. Closed-list test was stale-bait; not a
bug, auto-registration is intentional (`Pipeline.from_json` works for
scalers without setup). Chosen fix: (a) update expected list per patch
(visible churn, review catches typos). Alternative (b) property-based
superset check — deferred, KNOWN_TYPES goes stale too.

Convention: **when adding a built-in step, update
`test_list_step_types_default` in same commit.**

### Исправлено (v0.9.2f-fix1 — test-authoring #7)

- **`test_standard_scaler_unseen_data`** — deterministic data +
  reference helper. Prior test had 2 bugs:
  (a) same-distribution train/test — frozen-vs-refit indistinguishable
      (both output mean ≈ 0);
  (b) manual arithmetic `abs(mean - 5.0) < 0.5` — wrong: standardized
      mean = 0, not mean/std = 5.
  New: `x_train = [8, 12]`, `x_test = [12]` — frozen gives exactly
  1.0, refit gives 0.0. Exact assertion via reference.
- **`tests/_helpers/scaler_reference.py`** (new) — reference transforms
  for Standard/MinMax/MaxAbs/Robust. Codifies invariant
  `(x - mean_train) / std_train` per scaler; test bodies import
  reference, not manual arithmetic (§13.4/164 discipline extended
  to scalers).

### §13.4/171 — deterministic-vs-statistical test pattern

**For EXACT-property tests** (e.g., «does transform use frozen stats?»):
deterministic data + reference helper + exact assertion. RNG washes
out the property.

**For CONVERGENCE tests** («does the model learn?»): RNG + statistical
tolerance is fine.

7th test-authoring error in session (128 → 163 → 164 → 166 → 167 →
169 → 171). Systematic fix: reference helper per domain
(`metrics_reference.py`, `rolling_reference.py`, `scaler_reference.py`,
future `ts_features_reference.py`).

### Добавлено (v0.9.2f — scalers)

Завершение ML flow v0.9.2: preprocessing (metrics + CV + scalers).

- **`StandardScaler(columns=None, *, with_mean=True, with_std=True)`** —
  `(x - mean) / std`. Population std (`ddof=0`, matches sklearn).
  Degenerate column (std=0) → scale=1.0 (sklearn behavior).
- **`MinMaxScaler(columns=None, *, feature_range=(0.0, 1.0))`** —
  `(x - min) * (high-low)/(max-min) + low`. Constant column → output
  `feature_range[0]` (sklearn).
- **`MaxAbsScaler(columns=None)`** — `x / max(|min|, |max|)`. All-zero
  column → scale=1 (avoid division by zero).
- **`RobustScaler(columns=None)`** — `(x - median) / IQR`. IQR=0
  (degenerate) → scale=1. Per-column `df.quantile([0.25, 0.5, 0.75])`
  (external_sort per column, O(d) sorts per fit).
- **`cadw/scalers.py`** — new module. `_NumericStatsOp` (per-column
  Welford + min/max in one pass, state 5×d×8 bytes) shared by
  StandardScaler/MinMaxScaler/MaxAbsScaler. RobustScaler uses
  existing `df.quantile`.
- Auto-registered as Pipeline steps (`register_step_type`) → usable in
  `Pipeline.to_json/from_json`.
- Exports: `from cadw import StandardScaler, MinMaxScaler,
  MaxAbsScaler, RobustScaler`.

### Design decisions (locked-in, v0.9.2f)

- **No prototype** — scalers are **composition** of tested primitives
  (`Reduce`, `df.quantile`, `Transformer` Protocol). §13.4/123
  prototype-first applies to **new algorithms** (sort, roc_auc, CV
  fold math), not compositions.
- **Population std (ddof=0)** for StandardScaler — matches sklearn.
- **Degenerate handling**: std=0 → scale=1; IQR=0 → scale=1;
  min==max → output feature_range[0]; max_abs=0 → scale=1. All
  match sklearn or are no-op (safer for downstream NaN propagation).
- **Nullability (§13.4/169)**: all output fields nullable=True. Scaled
  columns → dtype float64; pass-through keep original dtype.
- **State embedded in `to_dict`** — O(d) bytes (dicts of floats). No
  separate "state" file like GBDT (large trees) — small enough for
  `pipeline.json`.
- **`columns=None` auto-detect numeric** at fit time, stored in
  `_columns` for transform.

### Ограничения (v0.9.2f, §11.5)

- **RobustScaler — O(d) external_sorts per fit.** For d=100 × n=1M,
  ~17 min fit. For d > 100 prefer StandardScaler/MinMaxScaler.
  Backlog: `pc.tdigest` based approximate (v0.9.4).
- **`astype` to non-float dtype** — not supported (scalers always
  produce float64 for scaled columns).
- **`columns=None`** auto-detects only `pa.types.is_integer` or
  `is_floating` — bool/string/timestamp excluded. Explicit `columns`
  can force-cast any Arrow-numeric-castable type, but non-numeric
  raises at `pc.cast`.
- **AST guard whitelist** (`test_arrow_call_memory_guard.py`) is by
  function NAME, not qualname. `"fit"`/`"transform"`/`"predict"`
  whitelisted globally → `RobustScaler.fit` (calls `df.quantile`)
  not flagged. Not a bug (quantile self-guards), but over-broad.
  Fix: qualname-based whitelist (v0.9.3).

### Тесты (v0.9.2f)

- +22 fast (`test_scalers.py`): StandardScaler (mean_zero_std_one,
  with_mean_false, with_std_false, nulls, roundtrip, unseen,
  constant); MinMaxScaler (0_1, custom_range, constant, roundtrip,
  bad_range); MaxAbsScaler (basic, positive_only, all_zero,
  roundtrip); RobustScaler (median_zero_iqr_one, outlier_resistant,
  roundtrip, constant); multi-column, auto-detect, missing raises;
  Pipeline integration (2 tests).
- +1 slow (`test_scalers_e2e.py`): 100K rows, StandardScaler →
  GBDTStep, R² > 0.5.

- fast: **~1512** (+22). slow: **20** (+1).

### §13.4/170 — scalers no new lessons; AST whitelist observation

See DECISIONS.md. Key: scalers are compositions (no prototype),
but AST guard whitelist-by-name is over-broad (RobustScaler.fit not
caught for `df.quantile` call). Fix: qualname-based (v0.9.3).

### Исправлено (v0.9.2e-fix1 — nullability + test data)

- **`cadw/cv.py::_attach_column`** — все output fields nullable=True.
  `nullable=False` на `__fold` ломал `Dataset.from_batches`:
  singleton-batch через `pa.RecordBatch.from_arrays([col],
  names=[name])` всегда nullable=True, writer target с
  `pa.schema([field])` (nullable=False) → ArrowInvalid. 9 failures.
  CADW convention: output fields always nullable=True (all existing
  ops соблюдают).
- **`test_cv_score_timeseries`** — i.i.d. data (`x ~ N(0,1)`,
  `y = 2x + noise`), не linear trend. Trend + TSS → GBDT
  extrapolation outside training bin range → r2 = -23.9 (inherent
  limitation §11.5; test design was wrong, not code). 1 failure.
- **`test_kfold_shuffle_different_assignment`** — cleaned placeholder
  (was doing nothing); proper set-of-y comparison + union invariant.
- **`test_tss_n_lt_k_plus_1_raises`** — raw-string regex (silence
  SyntaxWarning).

### §13.4/169 — nullability convention + test-respects-limitation

- **Output schema = nullable=True** для всех полей, produced через
  `Dataset.from_batches`. Core rebuilds singleton via
  `from_arrays(names=...)` (always nullable=True); writer target must
  match. Backlog: align `Dataset.from_batches` for nullable=False.
- **Test data must respect inherent model limitations.** TSS + GBDT
  (без lag features) cannot extrapolate. Use i.i.d. or bounded
  (periodic) features. Real time-series → TimeSeriesPipeline (lags,
  backlog v0.9.4).

### Добавлено (v0.9.2e — cross-validation)

- **`KFold(n_splits, shuffle=False, random_state=None)`** — k-fold CV
  for CadwDataFrame.
  - `shuffle=False` (default): `fold_id = arange(n) % k` — O(1) per row,
    deterministic, no RNG.
  - `shuffle=True`: `rng.permutation(n)` → fold_id array (O(N) int32;
    ~4 MB for 1M rows). Documented trade-off.
  - `.split(df)` yields `(fold_i, train_df, test_df)` — lazy views
    (PredicateFilter on `__fold` column).
- **`TimeSeriesSplit(n_splits, time_col)`** — no future leakage.
  - Sort by `time_col` once (`external_sort`), then positional fold
    slices via `iloc`.
  - fold 0: train=[0, fs), test=[fs, 2*fs)
  - fold k-1: train=[0, k*fs), test=[k*fs, n)
  - `fs = n // (n_splits + 1)`.
- **`cv_score(df, cv, model_factory, *, X_cols, y_col, metrics,
  pred_col=None, **metric_kwargs)`** — runs CV, returns
  `{fold_scores: [{metric: value}], mean: {metric: nanmean},
  std: {metric: nanstd}}`.
  - Model must implement `fit(df, target=y_col)` and
    `predict(df) -> CadwDataFrame` (GBDTStep-compatible convention).
  - Prediction column default `f"{y_col}_pred"`; override via `pred_col`.
  - `**metric_kwargs` passed to `preds.metrics()` (classes, average,
    y_proba, pos_label).
- Exports: `from cadw import KFold, TimeSeriesSplit, cv_score`.

### Design decisions (locked-in, v0.9.2e)

- **KFold fold assignment = `row_id % k`** (no shuffle) — trivial and
  deterministic. `shuffle=True` materializes O(N) int32 — accepted
  trade-off, not default.
- **TimeSeriesSplit — sort once, iloc slices per fold.** Sort cost
  O(N log N) per `split()` generator; cache sorted_df externally if
  calling multiple times.
- **cv_score uses `model.predict(test)` result directly** — GBDT-style
  `predict` returns df with original columns + `{target}_pred`. So
  `preds.metrics(y_true=y_col, y_pred=pred_col)` — no merge needed.
- **Model factory pattern** — fresh model per fold (no shared state).
  sklearn-compatible convention: `lambda: GBDTStep(...)`.

### Ограничения (v0.9.2e, §11.5)

- **`KFold(shuffle=True)`** materializes `fold_id` array (O(N) int32
  ≈ 4 MB per 1M rows) — for truly out-of-core N, prefer shuffle=False.
- **`StratifiedKFold`, `GroupKFold`, `Nested CV`** — backlog (v0.9.3+).
- **`TimeSeriesSplit` sorts once per `.split()` call** — calling
  `.split()` twice re-sorts. Cache the sorted_df if reusing.
- **`cv_score`** — relies on `pred_col` convention; user must ensure
  model's predict output has the named column (or pass `pred_col`).
- **`cv_score` with y_proba (roc_auc / log_loss)** — supported via
  `y_proba="..."` kwarg. Model must emit proba column; GBDTClassifier
  does (y_prob).

### Тесты (v0.9.2e)

- +15 fast (`test_cv.py`): KFold (sizes, uneven, no-overlap, shuffle,
  errors, filter); TimeSeriesSplit (splits, no leakage, train grows,
  test sizes, errors, unsorted); cv_score (kfold GBDT, timeseries,
  empty metrics, custom pred_col).
- +1 slow (`test_cv_e2e.py`): 100K rows, KFold(5), GBDT, R² > 0.5.

- fast: **~1485** (+15). slow: **19** (+1).

### Backlog (v0.9.3, §11.6 — from v0.9.2e code review)

- `_pandas_hash_column` silent fallback (add logging.warning).
- `_compute_roc_auc_binary::pa.scalar(pos_label, type=y.type)` mismatch
  validation.
- `_exact_actual_matches_arrow` — cache build-table value_counts
  per partition-pair.

### Исправлено (v0.9.2c-fix2 — multiclass log_loss gate)

- **`MetricsOp.__init__`** — classification setup block
  (`_classes`/`_cls_to_idx`/`_k`) gated на
  `_needs_classification OR (_needs_proba AND _needs_multiclass_proba)`.
  Pure `metrics=["log_loss"]` + multiclass `y_proba=[p0,p1,p2]` раньше
  оставлял `_cls_to_idx={}` → все y_true метки считались
  "неизвестными". 2 failures.
- **`MetricsOp.batch_memory_factor`** — 8.0 для multiclass proba
  (`.to_pylist()` на K+1 columns — §3.11 honest peak).
- **`MetricsOp.__init__`** — defensive: `_needs_regression +
  _needs_multiclass_proba` → ValueError (domain conflict: regression
  y_true continuous vs multiclass log_loss y_true class index).
- **`test_metrics.py`** — raw-string regex (4-space indent; v0.9.2c
  fix1 had 8-space, didn't match); 4 new contract tests
  (§13.4/168).

### §13.4/168 — partial migration: state-sharing ≠ naming

Feature F использует internal state S; S инициализируется под gate G.
Добавление F требует расширения G, даже если F не входит в
семантическую группу G. Пример: multiclass log_loss — не
classification metric (имя), но использует `_cls_to_idx` (state).

### Исправлено (v0.9.2c-fix1 — partial migration + test-authoring)

- **`MetricsOp.__init__`** — проверка `y_proba_cols` (unified) вместо
  legacy `y_proba_col`. v0.9.2c добавил `y_proba_cols` (list для
  multiclass/binary), но `if self._needs_proba and y_proba_col is None:`
  остался на legacy поле — все вызовы с `y_proba_cols=[...]` (без
  `y_proba_col`) получали false "y_proba_col required". 6 failures.
- **`CadwDataFrame.metrics`** — generic column-existence loop больше
  не включает `y_proba` (может быть list после v0.9.2c). Specific
  per-column check (after `y_proba_cols` normalization) уже был
  корректным. 6 failures.
- **`test_metrics_roc_auc_with_accuracy`** — expected AUC из
  `expected_roc_auc` helper, не manual 0.875 (правильно 1.0).
  **5-я** test-authoring ошибка в сессии (§13.4/128 → 163 → 164 →
  166 → 167).
- **`MetricsOp`** — message: "'roc_auc' is computed **separately**
  by CadwDataFrame.metrics()" (test regex `"roc_auc.*separately"`
  matches; clearer intent).
- **`tests/test_metrics.py`** — `match=r"len\(y_proba\)"` (raw
  string; silence SyntaxWarning).

### §13.4/167 — partial-migration pattern

X → X_new (deprecated):
- validation checks — **all** on X_new;
- generic loops / type-sensitive checks — **updated** after type
  widening (`str` → `str | list[str]`);
- **mixed-type tests** (scalar + list) — обязательны при расширении
  типа; v0.9.2a scalar tests не ловили list case.

### Добавлено (v0.9.2c — roc_auc + multiclass log_loss)

- **`roc_auc`** в `CadwDataFrame.metrics()`:
  - **binary** (`y_proba` scalar str): `y_true ∈ {pos_label, other}`.
    `pos_label` default `1`.
  - **macro** (`y_proba` list of K cols, `classes` explicit K>2):
    one-vs-rest, macro average.
  - Algorithm: **external_sort** by proba desc + **per-threshold
    grouping** + streaming trapezoid. Peak RAM = batch + Sort disk
    (independent of N). Validated by prototype 8/8.
  - NaN if `n_pos == 0` or `n_neg == 0` (honest).
- **Multiclass `log_loss`**: `y_proba` list of K cols, `classes` explicit.
  O(1) state (sum_log_loss, n_proba). One pass.
- **`AUC_METRICS`** = `{"roc_auc"}` (exported).
- **`tests/_helpers/metrics_reference.py`**: `expected_roc_auc`,
  `expected_log_loss_binary`, `expected_log_loss_multiclass` —
  reference implementations (validated by prototype + sklearn
  equivalence if installed).

### Design decisions (locked-in, v0.9.2c)

- **roc_auc — не Reduce**. Sort-based (multipass): pass 1 count,
  external_sort, pass 2 trapezoid. `MetricsOp` rejects `roc_auc`
  directly; `CadwDataFrame.metrics()` dispatches it separately.
- **Per-threshold grouping** (ties): all tied scores = one ROC point
  (matches sklearn.roc_curve). Prototype caught this (§13.4/166).
- **y_proba**: str → binary, list[str] → multiclass (both log_loss
  and roc_auc).
- **Conflict**: `roc_auc` (binary labels) + regression (`r2` etc.,
  continuous) on same `y_true` → ValueError. Use separate calls.

### Ограничения (v0.9.2c, §11.5)

- **roc_auc binary requires `y_true ∈ {0,1}`** (or `pos_label` +
  one other value; values not in `{pos_label, …}` treated as
  negative — but validated only for column presence, not value domain).
  String labels → use multiclass form (`y_proba` list + `classes`).
- **roc_auc macro** — `average='macro'` only (unweighted); weighted
  ovr — backlog.
- **`roc_curve`** (TPR/FPR arrays) — не возвращается; если запросят —
  отдельный метод.
- **Sort temp** — `spill_dir/metrics_roc_auc/sort_by_proba/*` (auto
  cleanup after streaming).
- **N × row_size** does not bind peak: Sort handles arbitrary N via
  disk. Fail-fast on `K × 24 + 64 > state_limit` (classification state).

### Тесты (v0.9.2c)

- +10 fast (`test_metrics_reference.py`): helper self-consistency,
  sklearn-equivalence (importorskip), known cases.
- +17 fast (`test_metrics.py`): multiclass log_loss (perfect/known/
  validation), roc_auc binary (perfect/known/ties/all-pos-NaN/
  all-neg-NaN/nulls/filtered/reference), roc_auc macro, mixed
  (accuracy + roc_auc, regression conflict, requires y_proba),
  MetricsOp rejects roc_auc.
- +1 slow (`test_metrics_e2e.py`): 100K rows, roc_auc + multiclass
  log_loss, memory 30 MB budget.

- fast: **~1468** (+27). slow: **18** (+1).

### Добавлено (v0.9.2b — classification metrics)

- **`accuracy`, `precision`, `recall`, `f1`** в `CadwDataFrame.metrics()`:
  - binary (K=2, `pos_label` = 1 по умолчанию) и multiclass (K≥3).
  - **`average`** ∈ `'binary'|'macro'|'micro'|'weighted'` (auto: binary
    if K=2, macro if K>2).
  - `classes` — explicit class list (K=2 default when None).
  - Numeric или string labels — dict lookup, nulls skipped.
- **`MetricsOp`** state: `tp[K] + fp[K] + fn[K]` int64 = `24×K` bytes
  (vs confusion matrix `K²×8 = 8 MB` для K=1000 — 320× дешевле для
  scalar metrics).
- **`ReduceOp.batch_memory_factor()`** — protocol extension.
  Classification path uses `.to_pylist()` per batch (7× Python-object
  overhead). `reduce()` passes to `check_batch(extra_factor=...)` —
  honest peak RAM, not hidden.
- **`SUPPORTED_METRICS`** = regression + proba + classification
  (exported from `cadw`).

### Design decisions (locked-in, v0.9.2b)

- **TP/FP/FN state** вместо confusion matrix. Accuracy/precision/
  recall/f1 — все выводятся из TP/FP/FN per class. K² state не нужен
  (confusion matrix добавим отдельным методом, если запросят).
- **Average auto-switch**: binary (K=2) / macro (K>2). Matches sklearn
  semantics; user может переопределить.
- **zero_division=0** (sklearn default): precision/recall/f1 при
  пустом знаменателе → 0.0, не NaN.
- **Mixed regression + classification** на одном `y_pred` → ValueError
  (continuous vs labels).
- **`log_loss` + classification** — OK (разные колонки: y_proba vs
  y_pred).
- **Fail-fast**: `K × 24 + 64 > state_limit` → `MemoryLimitExceeded`
  в `metrics()` до `MetricsOp` construction.

### Ограничения (v0.9.2b, §11.5)

- **`roc_auc`** — v0.9.2c (Sort-based, требует полного proba массива).
- **Multiclass log_loss** (proba matrix) — v0.9.2c.
- **Confusion matrix** itself — не возвращается; если запросят —
  отдельный метод (не `metrics()` scalar API).
- **Class mismatch** y_true/y_pred not in `classes` → ValueError
  (fail-fast, первые 5 invalid labels).
- **Empty input** → accuracy=NaN (как regression).
- **`average='binary'`** требует K=2, `pos_label` в `classes`.

### Тесты (v0.9.2b)

- +20 fast (`test_metrics.py`): accuracy binary perfect/mixed/auto,
  precision/recall/f1 binary pos_label, multiclass macro/micro/weighted,
  zero-division, nulls, empty, filtered, unknown label, mixed
  regression conflict, average=binary on K>2, invalid average, K too
  large, string labels, log_loss + classification, pandas equivalence.
- +1 slow (`test_metrics_e2e.py`): 100K rows, 3-class, manual numpy ref.

- fast: **~1441** (+20). slow: **17** (+1).

### Добавлено (v0.9.2a — metrics, regression + log_loss)

- **`CadwDataFrame.metrics(y_true, metrics, y_pred=, y_proba=)`** —
  regression + binary log_loss в одном Reduce-проходе.
  - **Regression** (require `y_pred`): `r2`, `mse`, `rmse`, `mae`.
  - **log_loss** (requires `y_proba` in [0,1], binary y): binary
    cross-entropy.
  - **Combined** — если `y_pred` и `y_proba` заданы оба, все метрики
    считаются в одном проходе (state O(1)).
  - Empty input (после filter) → **NaN** для каждой метрики.
  - Nulls в `y_true`/`y_pred`/`y_proba` → skip per row (skipna=True).
  - **R² matches sklearn**: `ss_tot == 0` → 1.0 если perfect, else 0.0.
- **`cadw/ops/metrics.py::MetricsOp`** — ReduceOp, state = 7 counters
  × 8 B = 64 B. Независимо от N.

### Design decisions (locked-in, v0.9.2a)

- **Reduce-based**: один проход, O(1) state. Не Sort, не pandas.
- **Regression + log_loss в одном patch** — обе метрики O(1) state,
  тривиально комбинируются. Classification (confusion matrix, K×K
  state) — v0.9.2b.
- **Empty → NaN**, не 0.0. Honest «no data» — user sees it.
- **R² совпадает с sklearn** в граничных случаях (constant y).
- **log_loss**: `np.clip(p, 1e-15, 1 - 1e-15)` — как sklearn.

### Ограничения (v0.9.2a, §11.5)

- **Classification metrics** (`accuracy`, `precision`, `recall`, `f1`)
  — v0.9.2b. Требуют confusion matrix (K×K state, K = #classes).
- **`roc_auc`** — v0.9.2c. Требует Sort + cumulative TPR/FPR.
- **Multiclass log_loss** — v0.9.2b. Требует proba matrix (K columns).
- **Per-group metrics** (`groupby(g).metrics()`) — v0.9.2d.
- **`y_true` for regression must be numeric** — string labels need
  `encode_categorical` first.
- **log_loss requires binary y** ∈ {0, 1}. Multiclass — v0.9.2b.

### Тесты (v0.9.2a)

- +24 fast (`test_metrics.py`): perfect/known values (r2, mse, rmse,
  mae), log_loss (perfect/50-50/known), combined, nulls skip, empty→NaN,
  filtered, string metric, all validation errors, pandas-equivalence,
  chain integration.
- +1 slow (`test_metrics_e2e.py`): 100K rows, 30 MB budget, all four
  metrics verified vs numpy.

- fast: **~1422** (+24). slow: **15** (+1).

## [0.9.1] — 2026-10-20

### Добавлено (v0.9.1f — pivot_table / melt / crosstab)

Финальный патч v0.9.1: табличные преобразования (long ↔ wide).

- **`df.melt(id_vars, value_vars, var_name='variable',
  value_name='value')`** — wide → long. **Streaming per-batch**, peak
  O(batch × #value_vars). Requirements:
  - `value_vars` share ONE dtype (Arrow column uniform); mixed →
    TypeError с рекомендацией `astype` для конвертации.
  - `id_vars` ∩ `value_vars = ∅`; `var_name`/`value_name` не
    коллидируют с `id_vars`.
  - Output: `[*id_vars, var_name, value_name]`, rows = `N ×
    len(value_vars)`.

- **`df.pivot_table(index, columns, values, aggfunc='mean')`** — long →
  wide. `groupby([*index, columns]).agg(values, aggfunc)` + pandas
  reshape (small table). **Fail-fast** если `k_pairs = K_index ×
  K_columns` слишком велико: `k_pairs × row_size × 2 > 0.3 × avail`
  → `MemoryLimitExceeded` **до** groupby. Missing combos → null.
  - `aggfunc ∈ {mean, sum, count, min, max}`, default `mean`.
  - `dropna=True` для null в `columns` (standard pandas).
  - Null в `index` → own row (kept).
  - Multi-index (`index=['a','b']`) поддерживается.

- **`df.crosstab(index, columns)`** — frequency table. `dropna=True`
  на nulls в index/columns (standard crosstab), delegates to
  `pivot_table(values=index[0], aggfunc='count')`.

### Memory bound (§11.5)

- **melt**: O(batch × #value_vars). Streaming.
- **pivot_table**: K_pairs estimate через sample (`MemoryGuard.
  estimate_unique_keys`) **до** groupby. `check_state` на grouped
  (K_pairs rows) **до** pandas reshape (AST-verified). Peak = batch +
  K_pairs × row_size (grouped) + K_index × K_cols × value_size
  (pivot output) — bounded by fail-fast check.
- **crosstab**: same as pivot_table.

### Ограничения (v0.9.1f, §11.5)

- **`pivot_table` sparse → fail-fast**. K_pairs > 0.3×avail/row_size →
  `MemoryLimitExceeded` с рекомендацией `groupby()` (long format).
  Sparse pandas pivot (fill N/A with 0 in wide format) — не
  поддерживается.
- **`melt` heterogeneous dtypes** (`['int_col', 'float_col']`) →
  TypeError. Пользователь: `df.astype({'int_col': 'float64'})` first.
- **`pivot_table` dropna=True** в `columns` (стандарт pandas).
  `dropna=False` (null column group) — v0.9.2 если запросят.
- **`pivot_table` output column names = `str(value)`**. Int-значения
  `columns` → строковые имена ("0", "1", ...).
- **`melt` non-streaming types**: list/struct columns in value_vars
  работают (Arrow-native), но nested dtypes не проверяются на
  uniformity (assumes Arrow type equality).
- **`crosstab` count via non-null values** в `index[0]`. Требует
  non-null `index[0]` — фильтруется через `dropna=True` до.

### Тесты (v0.9.1f)

- +15 fast (`test_pivot_table.py`): melt (6), pivot (7), crosstab (3),
  AST guards (2).
- +1 slow (`test_pivot_table_e2e.py`): 100K rows pivot.

- fast: **~1395** (+15). slow: **13** (+1).

### Добавлено (v0.9.1e — per-group apply(fn))

- **`df.groupby('g').apply(fn, result_col='apply_result', on=None)`** —
  per-group user callable. Prototype-validated (11 cases A/B pandas
  vs composite-sort, §13.4/123).
  - **fn**: `callable(group_df: pd.DataFrame)` → scalar | `pd.Series` |
    `pd.DataFrame`. Uniform mode required across groups.
  - **Output**:
    - scalar → K rows, `[*group_cols, result_col]`
    - Series → K rows, `[*group_cols, *series.index]`
    - DataFrame → `Σ|df_i|` rows, `[*group_cols, *df.columns]`
  - **on**: within-group sort keys. `None` → row_id order.
  - **Order**: output sorted by group key.
- **`CadwDataFrame._per_group_apply(...)`** — internal: composite-sort
  by `[*group_cols, *on_cols]`, one group in RAM at a time, streaming
  output to IPC.

### Memory bound (§11.5)

- **Per-group bound**: `group_nbytes × 2 (pandas) ≤ 0.3 × available`.
  Oversized → `MemoryLimitExceeded` **before** `fn()`.
- **Peak RAM** = batch (input stream) + `2 × max(group_size × row_size)`
  + fn's output + small output-buffer batch. **Independent of N**.
- **fn's own alloc — user responsibility** (§3.9 extended): CADW bounds
  input to `fn`, not `fn`'s internals. `fn` that's O(N) per group → OOM
  is user's fault.

### Ограничения (v0.9.1e, §11.5)

- **`transform(fn)` NOT included** — semantically different (Series →
  same-length array for broadcast). v0.9.2.
- **Mixed return modes across groups** → RuntimeError.
- **Null in group key** → own group (dropna=False, matches pandas).
- **`fn` exceptions propagate** — no wrapping; user errors surface
  unchanged.
- **Output sorted by group key**, not original order.

### Тесты (v0.9.1e)

- +17 fast (`test_apply.py`): 3 modes, nulls, multi-group, single-row,
  filter, on-col sort, determinism, errors (fn not callable, mixed
  modes, unsupported type, oversized group, fn exception), pandas
  equivalence, AST guard.
- +1 slow (`test_apply_e2e.py`): 10K / 20 groups / mean, pandas equiv.

- fast: **~1378** (+17). slow: **12** (+1).

### Добавлено (v0.9.1d — ColumnTransformer + register_step_type)

- **`ColumnTransformer(transformers, remainder='drop'|'passthrough')`** —
  sklearn-style column fanout. Каждый трансформер fit+transform на своей
  subset колонок; результаты concat (column union).
  - `transformers: list[(name, cols, step)]` — уникальные имена,
    disjoint cols (каждая колонка ≤1 transformer), Transformer contract.
  - `remainder='drop'` (default): неименованные колонки удаляются.
  - `remainder='passthrough'`: проходят как есть в конце.
  - Row count preserved (dropna внутри → RuntimeError).
  - Duplicate output column names → RuntimeError.
  - Output order: transformer order, затем remainder.
  - `to_dict` / `from_dict` — nested state inline в JSON (nested GBDT
    ~30 KB в pipeline.json; extraction to separate file — backlog v0.9.2).

- **`register_step_type(name, cls)`** / **`unregister_step_type(name)`** /
  **`list_step_types()`** — module-level registry для custom steps в
  `from_json`. Backlog §11.5 закрыт.
  - Built-in types: `CastStep`, `DropnaStep`, `GBDTStep` (нельзя
    unregister).
  - Override built-in → warning (same pattern as `register_model` §9.7).
  - Custom class MUST implement `from_dict` classmethod.

- **`cadw/__init__.py`** — экспортирует `Pipeline`, `CastStep`,
  `DropnaStep`, `GBDTStep`, `ColumnTransformer`, `Transformer`,
  `Predictor`, `register_step_type`, `unregister_step_type`,
  `list_step_types`.

### Изменено

- **`Pipeline.from_json`** — использует module-level `_STEP_REGISTRY`
  (было hardcoded dict). Error message указывает на `register_step_type`
  для custom steps.

### Design decisions (locked-in, v0.9.1d)

- **Predictor not allowed inside ColumnTransformer** — only as final
  Pipeline step. `ColumnTransformer` is a Transformer (fit + transform
  on column subsets); model is orthogonal.
- **Row-preserving constraint** — column-level operations only. Row
  filtering (dropna) must go before ColumnTransformer.
- **Nested state inline** — `pipeline.json` содержит `step_dict` для
  каждого nested step. Large nested models (GBDT inside CT) — larger
  pipeline.json; extraction to `state/<ct_name>/<nested>.json` — backlog
  v0.9.2.

### Ограничения (v0.9.1d, §11.5)

- **Nested state inline** в `pipeline.json`. Для ColumnTransformer с
  GBDT ~30-50 KB в pipeline.json (acceptable). 1000+ tree GBDT — big
  file; extract-to-separate — v0.9.2.
- **No `remainder=transformer`** (sklearn supports this) — только
  `'drop' | 'passthrough'`. Для transform remainder — list explicitly
  в `transformers`.
- **Custom step `from_dict`** must be classmethod (validated at
  `register_step_type`).

### Тесты (v0.9.1d)

- +9 fast (`test_register_step_type.py`): register/unregister/list,
  override warning, bad name/class, missing from_dict, from_json uses
  registry.
- +12 fast (`test_column_transformer.py`): construction errors (empty,
  bad remainder, duplicate name, overlapping cols, empty cols, missing
  transform), basic fit/transform (2 transformers), remainder drop vs
  passthrough, appends new col, transform missing col, row count changed,
  to_dict/from_dict round-trip, unknown nested type, lambda nested raises.
- +3 fast + 1 slow (`test_column_transformer_pipeline.py`): inside
  Pipeline, save/load round-trip, state_bytes propagates, E2E (R² > 0.5).

- fast: **~1351** (+24). slow: **11** (+1).

### Добавлено (v0.9.1c — save/load)

- **`Pipeline.to_json(path)` / `Pipeline.from_json(path)`** — сохранение
  и восстановление pipeline для reproducibility и production deployment.
  - `path` — директория: `path/pipeline.json` + (опционально)
    `path/state/<step_name>.json`.
  - `from_json` принимает директорию или `pipeline.json` файл.
  - `strict_version=True` — raise на version mismatch; default — warn.
  - `path` must be directory (не file) для `to_json` — raises clear
    error если path exists and is file.
- **Per-step `to_dict()` / `from_dict()`** — Protocol extension, реализовано
  для `CastStep`, `DropnaStep`, `GBDTStep`. Custom steps должны
  реализовать оба, иначе `to_json` raises clear message.
- **`GBDTStep.to_dict()` / `from_dict()`** — полная сериализация
  fitted model (trees + base_prediction + mn/mx + hyperparameters) в
  JSON. ~30 KB для 100 trees × depth 3.
- **`LambdaTransform.to_dict()`** — raises NotImplementedError: `fn` —
  callable, не JSON-serializable. Message предлагает: custom step class
  с to_dict/from_dict, или только именованные wrappers.
- **`cadw/_serialization.py`** — helpers: `gbdt_regressor_to_dict` /
  `gbdt_regressor_from_dict`, `linear_to_dict` / `linear_from_dict`.
  Explicit, no pickle (safety: untrusted input не вызывает RCE).

### Design decisions (locked-in, v0.9.1c)

- **Always directory**: `pipeline_dir/pipeline.json` + optional
  `state/`. Cleaner для multi-file pipelines (state per step).
- **JSON for everything**: no pickle. GBDT trees ~30 KB JSON acceptable.
- **Version field** "0.9.1": `from_json` warns on mismatch, best-effort
  restore. `strict_version=True` для fail.
- **`LambdaTransform` non-serializable** — callables нельзя безопасно
  сохранить в JSON. Custom step class — user-side.

### Ограничения (v0.9.1c, §11.5)

- **JSON size для больших моделей** — GBDT n_estimators=1000 × depth=5
  ≈ 2.4 MB JSON. Для очень больших моделей — Arrow IPC sidecar
  backlog (v0.9.2). Сейчас JSON для всех.
- **Version compatibility**: strict_version=True поддерживает только
  same-minor. Future minor bumps требуют manual migration.
- **Custom steps**: `to_json` raises `NotImplementedError` если step
  не имеет to_dict. `from_json` raises если type не в registry —
  custom steps требуют register (v0.9.2 `Pipeline.register_step_type`).
- **`state` dir not created** для stateless pipeline — OK, load
  detects `has_state=False`.

### Тесты (v0.9.1c)

- +20 fast (`test_pipeline_save_load.py`): CastStep/DropnaStep/GBDTStep
  round-trip, predictions equal, DAG edges preserved, state file
  created, version mismatch warn/strict, corrupt JSON, missing state
  file, missing pipeline.json, path-as-file, LambdaTransform raises,
  fitted flag preserved.
- +1 slow (`test_pipeline_save_load_e2e.py`): fit → save → load →
  predict, predictions equal (rtol=1e-9), R² > 0.9.

- fast: **~1329** (+20). slow: **10** (+1).

### Добавлено (v0.9.1b — DAG Pipeline)

- **`Pipeline.add_step(name, step, inputs=...)`** — DAG edges (v0.9.1b).
  - `inputs=None` (default): chain semantics — first step = source (`[]`),
    subsequent = `[previous step name]`. **Backward compat** с v0.9.0.
  - `inputs=[]`: explicit source (redundant for first step, validated).
  - `inputs=["a", "b"]`: DAG fan-in. Все names должны существовать;
    Predictor нельзя использовать как input.
- **`Pipeline._build_graph()`** — `{name: [input_names]}`. Resolves
  `inputs=None` chain derivation.
- **`Pipeline._topological_order()`** — DFS topological sort с cycle
  detection (defensive: forward refs запрещены, cycle невозможен by
  construction).
- **`_merge_fanout(dfs)`** — module helper: column-union concat.
  Precondition: same row count across branches (positional alignment).
  Streaming через `zip(*iters)` с per-batch size check.
- **`LambdaTransform(src_col, feature_name, fn, out_type)`** — test
  helper: Arrow-native fn on one source column, appends
  `{src_col}_{feature_name}`.
- **`Pipeline._repr_html_`** — DAG-aware: column "inputs" показывает
  `(source)` или `← a, b` для fan-in.

### Изменено

- **`Pipeline.fit` / `transform` / `predict`** — execution через
  topological order, не list iteration. Multi-input → `_merge_fanout`.
  `transform` возвращает sink pure-transform подграфа (последний
  non-Predictor в reversed order); если pipeline model-only — исходный df.
- **`Pipeline.predict`** — возвращает результат first (и единственного)
  Predictor в topo order (Predictor валидирован как sink на `add_step`).

### Ограничения (v0.9.1b, §11.5)

- **Fan-in MVP: все branches должны иметь same row count.** Dropna /
  row-selection в одной ветви → RuntimeError с clear message.
  Row-selection fan-in (join по `__orig_rid` с пересечением) — v0.9.1c.
- **Fan-in реализация: positional alignment** (не row_id merge).
  Гарантия корректности = все branches идут от одного source и не
  меняют row order. Документировано.
- **Predictor = sink**, нельзя быть input. Проверяется на `add_step`.
- **Forward references запрещены** (`inputs=[name]` где `name` ещё не
  добавлен) → ValueError. Cycle detection — defensive (недостижим).
- **`Pipeline.plan(df)`** — sum across all steps (не меняется под DAG).
  Per-DAG-step state estimation — v0.9.1c backlog.

### Тесты (v0.9.1b)

- +14 fast (`test_pipeline_dag.py`): chain backward compat, two/three
  branch fan-in, diamond (raw → a, raw → b, merge, model), row-mismatch
  RuntimeError, unknown input, Predictor-as-input, explicit source
  validation, topo order (chain, fan-out), transform stops at model,
  predict full DAG, repr_html DAG edges.

- fast: **~1309** (+14). slow: **9** (unchanged).

### Изменено (v0.9.1a — backbone)

- **`tests/conftest.py`** — autouse `plt.close('all')` для всех тестов.
  Устраняет `More than 20 figures have been opened` RuntimeWarning,
  появившийся с v0.9.0c (8 plot-accessor тестов перевалили порог;
  pre-existing seaborn/viz тесты тоже не закрывали figures).
- **`cadw/patch_utils.py`** — общие helpers для patch-скриптов:
  `ensure_unreleased_header`, `append_unreleased_entry`,
  `find_section_anchor` (regex tolerant: `# 11.6`, `## §11.6`, `#### 11.6.`),
  `insert_before_section` (explicit `found` flag, no silent no-op),
  `assert_guards_green` (§13.4/140 discipline extraction).
- **`Transformer` / `Predictor` Protocol split** (§13.4/147 backlog):
  Predictor БЕЗ `transform`; Transformer БЕЗ `predict`. Строго два
  минимальных контракта, не nested. Runtime `_is_predictor(step) =
  hasattr(step, "predict")` — duck typing, Protocol только для typing.
- **DECISIONS.md §11.6 header restored** — заголовок был потерян
  в v0.8.11/v0.9.0 патчах; Pipeline v1 addendum от v0.9.0b-fix1
  (был appended к концу файла из-за anchor-not-found) перенесён
  в восстановленную §11.6.
- **`tests/test_pipeline.py`** — удалён dead `test_cast_step` stub
  (leftover из v0.9.0b, тест без assertions).

### Ограничения (v0.9.1a, §11.5)

- `patch_utils` используется только новыми patch-скриптами (v0.9.1+).
  Старые (v0.8.x, v0.9.0) сохраняют свою логику — historical artifacts.
- `conftest.py` — autouse для matplotlib only; torch lazy, уборки не требует.

### Тесты (v0.9.1a)

- +18 fast: 14 `test_patch_utils.py` (CHANGELOG + anchor + insert +
  Protocol split), +4 dead-test cleanup ensures pipeline suite stays green.
- RuntimeWarning "More than 20 figures" — исчезает (conftest autouse).

- fast: **~1298** (+18). slow: **9** (unchanged).

## [0.9.0] — 2026-10-20

### Добавлено (v0.9.0c — Jupyter integration)

- **`CadwDataFrame._repr_html_`** — sample 20 rows + schema + shape
  (upper bound) + filter note. Bounded: `.to_pandas().to_html()` на
  20-row `head()` только, не full df. AST guard whitelist с rationale.
- **`CadwDataFrame.plot` accessor** — matplotlib default через
  `df.plot.hist/line/box/scatter/hist2d`; explicit backends:
  `df.plot.plotly.*`, `df.plot.seaborn.*`. Все методы делегируют
  `cadw.viz.*` adapters (aggregate-first discipline preserved).
- **`Pipeline._repr_html_`** — HTML-таблица steps с per-step
  `state_bytes`, fitted marker, total state.
- **`cadw/plot_accessor.py`** — новый модуль, thin wrapper over
  `cadw.viz.matplotlib_adapter/plotly_adapter/seaborn_adapter`.

### Ограничения (v0.9.0c, §11.5)

- `df.info()` / `df.dtypes` / `df.describe()` — v0.10.
- `Pipeline` DAG визуализация — v0.9.1 (после DAG support в Pipeline).
- `df.plot` — только явные методы (`df.plot.hist(...)`), не
  `df.plot(kind='hist', ...)` (pandas-style). Method-style selected
  as primary; kind-based — backlog, если запрос.

### Тесты (v0.9.0c)

- +16 fast: 8 `test_repr_html.py`, 8 `test_plot_accessor.py`.
- AST guard whitelist: `_repr_html_` (bounded 20-row to_pandas).

- fast: **~1278** (+16). slow: **9** (unchanged).

### Добавлено (v0.9.0b — Pipeline skeleton)

Главный архитектурный шаг v0.9.0: chain of Transformers с plan-first
feasibility check. Превращает CADW из «набора primitives» в
out-of-core framework.

- **`cadw/pipeline.py`** — новый модуль:
  - **`Pipeline`** — sequence of Transformers. Model (if present)
    всегда LAST.
    - `add_step(name, step)` — chainable.
    - `plan(df) -> PipelinePlan` — pre-fit feasibility: sum(state_bytes)
      ≤ state_limit. Raises `PipelineMemoryError` if not feasible
      **до** любого materialize. Пользователь может вызвать явно для
      early check.
    - `fit(df, target=None)` — fit всех steps последовательно, plan-first.
    - `transform(df)` — применяет pure-transform steps; **останавливается
      перед Predictor** (если он есть).
    - `predict(df)` — fit chain + last Predictor. Требует последний step
      быть Predictor.
    - `fit_transform(df, target=None)` — fit + return features (before model).
    - `total_state_bytes()` — sum fitted state, интроспекция.
  - **`Transformer` / `Predictor` Protocol** (typing only) — контракт:
    `fit(df, target=None)`, `transform(df)`, `estimate_state_bytes(dc)`,
    `state_bytes()`; Predictor adds `predict(df)`.

- **Wrappers (MVP, для тестов)**:
  - **`CastStep(mapping, safe=True)`** — wrap `df.astype()`. Stateless.
  - **`DropnaStep(how, subset)`** — wrap `df.dropna()`. Stateless.
  - **`GBDTStep(X_cols, y_col, **kwargs)`** — wrap `fit_gbdt_regressor`.
    `estimate_state_bytes` = n_estimators × 2^(max_depth+1) × 48 + 1024
    (conservative per-node).

### Design decisions (locked-in, v0.9.0)

- **Sequence-first.** DAG (via `inputs=[...]`) — v0.9.1.
- **Model LAST.** `transform()` stops before Predictor; `predict()`
  requires it.
- **Plan-first fit.** `Pipeline.fit` calls `plan(df)` before any work;
  `PipelineMemoryError` if not feasible.
- **2-pass fit/transform.** State frozen after fit; transform re-applies
  pure steps. No fused execution (state-after-fit стабилен — экономии нет).
- **In-memory state.** Bounded (K × #features). Serialize to JSON — v0.9.1.
- **Per-batch transform guard — per-step.** Each step uses `MemoryGuard`
  internally; Pipeline не reimplement'ит.

### Ограничения (v0.9.0b, §11.5)

- `Pipeline.plan` — fit feasibility only (`total_state ≤ state_limit`).
  `spill_boundaries` always empty (deferred v0.9.1). Batch propagation
  per step handled by step's own MemoryGuard (not re-planned).
- `Pipeline.to_json` / `from_json` — v0.9.1.
- `ColumnTransformer` — v0.9.1.
- `TimeSeriesFeatures` composite — v0.9.2.
- `Pipeline` не валидирует, что Predictor is last при `add_step`
  (проверяется при `fit`). Predictor not-last на `fit` → RuntimeError.

### Тесты (v0.9.0b)

- +15 fast (`test_pipeline.py`): construction, wrappers, compose,
  fit/transform/predict contract, plan (fits, does_not_fit, with GBDT),
  state_bytes tracking, small GBDT fit+predict.
- +1 slow (`test_pipeline_e2e.py`): 20K rows, timestamp → dt features
  → categorical encoding → GBDT; R² > 0.3.

- fast: **~1257** (+15). slow: **9** (+1).

### Добавлено (v0.9.0a — accessors + type ops)

Первый шаг v0.9.0 «out-of-core framework»: подготовка данных
(datetime features, string cleaning, type casting, null handling).
Arrow-native, bounded per-batch, zero pandas.

**Accessors:**
- **`df['ts'].dt.{year,month,day,hour,minute,second,weekday,day_of_year,quarter}`** —
  Arrow `pc.year/month/day/hour/minute/second/day_of_week/day_of_year/quarter`.
  Возвращает CadwDataFrame с `{col}_{feature}` (original columns preserved).
- **`df['s'].str.{len,lower,upper}`** — `pc.utf8_length/utf8_lower/utf8_upper`.
  Возвращает CadwDataFrame с `{col}_{op}`.
- **`df['s'].str.contains(pat)`** — возвращает **MaskFilter** (для `df[mask]`).
  na=False (pandas default): null не матчится.
- **`df['col'].isnull()` / `.notnull()`** — MaskFilter.

**Facade:**
- **`df.astype({col: dtype})`** — cast per column. `dtype` = `pa.DataType`
  или str-alias (`"int64"`, `"float32"`). `safe=True` (default) → raise на
  overflow; `safe=False` → wrap (Arrow `pc.cast`).
- **`df.dropna(how='any'|'all', subset=None)`** — **lazy**: композиция
  `notnull()` фильтров. `how='any'`: drop if ANY null. `how='all'`: drop
  if ALL null.
- **`df.drop(cols)`** — selection minus. Lazy.
- **`df.rename(mapping)`** — materialized (schema change). `{old: new}`.

**Внутренний helper:**
- `CadwDataFrame._apply_column_transform(column, feature, fn, out_type,
  accessor=...)` — generic per-batch Arrow transform + append `{col}_{feature}`.
- Новая папка `cadw/accessors/` с `_dt.py`, `_str.py`.

**AST guards:** `tests/test_arrow_call_memory_guard.py` — whitelisted
`_apply_column_transform`, `astype`, `dropna`, `drop`, `rename`.

### Исправлено (v0.9.0a-fix1 — first-run bugs)

- **`PredicateFilter.__init__` не знал про новые ops (8 test failures).**
  v0.9.0a расширил `mask_for_batch` (is_null/not_null/contains), но
  `__init__` валидировал по старому `_OPS` → `ValueError: unsupported op`.
  Fix: `_OPS_NO_SCALAR = frozenset({"is_null", "not_null", "contains"})`,
  `__init__` проверяет `_OPS ∪ _OPS_NO_SCALAR`. Class: partial update
  within one class (handler extended, validation not).
- **`test_dt_null_timestamp_propagates` — NaT не конвертируется через
  `ts.tolist()` (1 failure).** `pa.table({"ts": [Timestamp, NaT, ...]})`
  → `NaTType does not support utcoffset`. Fix: `pa.array(ts)` — Arrow-native,
  NaT → null. Class: §13.4/134-like (input conversion not Arrow-safe).
- **CHANGELOG `[Unreleased]` отсутствовал после v0.8.11 tag** —
  silent skip (v0.9.0a entry не записан). Fix: prepend `[Unreleased]`
  before first `## [` if missing. Class: §13.4/140 recurrence (2nd hit
  of same trap — v0.8.11a-fix1 was first).

### Тесты (v0.9.0a)

- fast: **~1228** (+30 accessors + type ops). slow: **8**.

## [0.8.11] — 2026-10-20

### Исправлено (safety — P-D1 close)

- **P-D1 (S2, live): Arrow-native `drop_duplicates` rewrite.** pandas
  chain (`to_pandas → sort_values → drop_duplicates → drop → from_pandas`)
  в `_local_dedup` заменён на Path C: `dictionary_encode(subset)` +
  `group_by(subset).aggregate(min/max row_id)` + `is_in` filter.
  Удалён `import pandas` из `cadw/ops/dedup.py`.

  **Bench (prototype `scripts/proto_dedup_arrow.py`):**
  | n | pandas | Path B (arrow) | **Path C (dict)** |
  |---|---|---|---|
  | 100k | 9.3 MB | 10.2 MB | **5.9 MB** |
  | 300k | 29.3 MB | 39.0 MB | **3.6 MB** |
  | 1M   | 87.7 MB | 47.2 MB | **10.9 MB** |

  avg peak(C)/peak(A) = **0.29**. Correctness A==B==C across keep ∈
  {first, last, False} × selectivity ∈ {10%, 50%, 90% dupes}.

  **Stopgap removed:** `check_state(5 × table.nbytes)` → `check_state(1.5 ×
  table.nbytes)` (honest Arrow-native bound, proto-backed). Effective
  partition limit back from `avail × 0.1` to `avail × 0.33`.

- **Dead pandas helpers removed из `cadw/ops/join.py`.** `import pandas
  as pd`, `_drop_null_keys` (superseded by `_drop_null_keys_arrow` в
  v0.8.5.3) — удалены. Zero pandas in ops except `shuffle.py::_pandas_hash_column`
  (documented perf exception, §9.2.20).

### Добавлено

- **`tests/test_no_pandas_in_ops.py`** — AST guard: `import pandas` в
  `cadw/ops/*` разрешён только в `shuffle.py` (whitelist + rationale).
  Также `test_dedup_uses_arrow_native` (Path C marker).
- **`tests/test_dedup_arrow_native.py`** — 16 behavior tests: first/last/
  False, nulls в subset/value, multi-subset, int key, empty, K=N, K=1,
  pandas-equivalence, oversize → raise, dedup→sort chain, filtered dedup.

### Ограничения (v0.8.11a, §11.5)

- `drop_duplicates` порядок вывода: partition order, без
  `sort_values("row_id")` внутри partition (был в pandas-версии).
  Set-equivalence сохранена; для детерминированного порядка — `.sort_values(...)` после.
- `drop_duplicates` recursion при skew — не реализован (как в v0.3.0).
  Oversize partition → MemoryLimitExceeded. Backlog.

### Тесты

- fast: **~1190** (+16 behavior, +5 AST guard). slow: **8**.

### v0.8.11b — strict join conditional + schema-order audit

#### Изменено

- **`merge()` strict join enforcement.** `budget >= 160 MB` →
  `MemoryLimitExceeded` on whole-operation RSS over-budget (was: warn).
  Threshold: `4 × _JOIN_FIXED_OVERHEAD_BYTES = 160 MB`, plus secondary
  `state_limit >= 40 MB` (protects against reservation drop).
  Below threshold → warn only (backward compat with `mem_mb=20` tests;
  fixed 40 MB overhead dominates small budgets — strict check would fire
  trivially even for tiny joins). See §9.2.6.

- **`MemoryGuard.state_limit_bytes`** — public property. Callers need
  `state_limit` without raising `MemoryLimitExceeded` (used by merge's
  strict-mode decision).

- **`cadw/ops/join.py::_JOIN_STRICT_BUDGET_THRESHOLD_BYTES`** = `4 ×
  _JOIN_FIXED_OVERHEAD_BYTES` = 160 MB. Module constant, documented.

#### Добавлено (schema-order audit — §13.4/145 backlog)

- **`tests/test_schema_order_audit.py`** — 4 AST guards for the audit
  findings:
  - `join._safe_materialize_partition`: explicit `schema=out_schema`.
  - `sort._write_run`: `ArrowIPCBatchWriter(table.schema)`.
  - `groupby._read_partition_safe`: no `.select()` on batches.
  - `dedup._local_dedup`: `select_schema` (v0.8.11a-fix3).
  No additional bugs found across 4 audited call sites.

- **`tests/test_join_strict.py`** — 4 tests: threshold constant, public
  property, strict raise at budget 200 MB, strict pass small join.

#### Ограничения (v0.8.11b, §11.5)

- **Join strict enforcement requires `budget >= 160 MB`.** Below — warn
  only, per-batch guards still enforce per-batch memory. For guaranteed
  strict enforcement, set `MemoryGuard(max_memory_mb=200)` or higher.
  Rationale: 40 MB fixed overhead is non-trivial below 160 MB budget;
  raise would fire trivially for tiny joins.

#### Тесты (v0.8.11b)

- fast: **~1198** (+8), slow: **8**.

### Тесты (итого v0.8.11)

- fast: **~1198**, slow: **8**.


## [0.8.10] — 2026-10-20

### Добавлено (safety follow-up, P-A3 addendum)

- **Whole-operation RSS warning on `merge()`.** `CadwDataFrame.merge`
  вычисляет `predicted_output = n_left × avg_matches × row_bytes_out`
  из planner'а и логирует WARNING, если
  `40 MB + 5 × predicted_output > state_limit`.
  `MemoryLimitExceeded` не бросается — enforcement остаётся per-batch в
  `_process_partition_pair`, warning — диагностический сигнал.

  **Почему warning, а не raise:** per-batch проверка выполняется 8–16
  раз (по числу пар partition'ов), поэтому добавление 40 MB offset
  per-batch over-predicts в 8–16×. Добавление offset на уровне
  `merge()` требует поднятия `mem_mb=20` → `100` в ~10 тестовых файлах
  (~30 тестов падают без этого). Это отдельная работа v0.8.11, не
  «5 минут». Warning даёт сигнал сегодня без test churn.

- **`cadw/ops/join.py::_JOIN_FIXED_OVERHEAD_BYTES = 40 * 1024 * 1024`**
  — module const, измеренный floor для `Table.join` RSS beyond logical
  output. Across 3 runs of pk join @ N=1M: 114.7 / 142.8 / 156.8 MB
  RSS delta для 24 MB output. Offset покрывает Arrow pool init (~16 MB)
  + hash-table build + non-Arrow temporaries (~24 MB). Invariant of
  output size — added once, not per-batch.

### Ограничения (v0.8.10-pre, §11.5 addendum)

- Join guarantee остаётся **best-effort на RSS**, не strict. Для
  бюджетирования: `estimated_rss = 40 MB + 5 × predicted_output`; если
  превышает `state_limit = budget × 0.5`, actual peak может превысить
  budget независимо от per-batch guards. Strict enforcement — v0.8.11
  с ревизией тестовых бюджетов.
- Run-to-run variance 30–40% для одинаковой нагрузки (pool retention,
  jemalloc page release timing). Формулы target'ят worst-case.

### Добавлено (per-group feature)

- **`df.groupby('sensor').shift(k, on='ts', columns=[...])`** — per-group
  shift. `on=None` → within-group order = row_id. Group cols excluded
  from default `columns` set.
- **`df.groupby('sensor').diff(k, on='ts', columns=[...])`** — per-group
  first difference.
- **`df.groupby('sensor').rolling(window, agg, on='ts', columns=[...])`** —
  per-group rolling. APPENDS `{col}_rolling_{agg}`; min_periods applies
  from group start.

  **Algorithm (prototype-validated, 11 cases A==B):**
  composite sort `[*group_cols, *on_cols]` → single deque per col,
  reset on group change → sort back by `__orig_rid`. **2 sort passes,
  0 shuffle.** Peak RAM = batch + O(window × #cols), independent of N
  and G (group count).

  **Semantics:** null in group col → own group (SQL, not pandas
  `dropna=True`). Group columns are NOT shifted by default (they're
  the key). This differs from pandas `groupby().shift(1)` which shifts
  everything including group key.

  **Reference:** `tests/_helpers/per_group_reference.py` — pandas
  ground truth (§13.4/128, /129, /139).

### Тесты

- fast: **~1130 passed** (+3 behavior warnings: tight/loose budget). slow: **8**.

## [0.8.9] — 2026-10-20

По результатам Pingis library-memory audit (TASK2, run 2026-10-20):
6 accept, 0 reject, 2 needs-research, 5 confirmed live violations.

### Исправлено (safety — Arrow memory discipline)

- **P-A1 (S2, live)** — `join._safe_materialize_partition` и
  `dedup._local_dedup` теперь используют generator form
  `Table.from_batches(gen, schema=...)`. Measured v0.8.8: list = 0.93–1.03×
  file (retained batches), gen = 0.0× (zero-copy from Arrow buffers).
- **P-A3 (S2, live)** — `Table.join` pre-check: `check_state(5 × row_bytes_out
  × actual_matches)` **до** join'а. `row_bytes_out` = сумма bit_width//8 по
  `_empty_result_schema` + null bitmap (не 256 B/row heuristic). Measured
  v0.8.8: RSS delta = 6.85× output on narrow int64 @ 1M; 5× conservative.
- **P-D1 (S2, live)** — `dedup._local_dedup` pandas chain (to_pandas +
  sort_values + drop_duplicates + drop + from_pandas) проверяется
  `check_state(5 × table.nbytes)` **до** `to_pandas`. Measured v0.8.8:
  3.95–8.29× table.nbytes. Effective partition limit shrinks to `avail × 0.1`.
- **P-G3 (S2, live)** — `_groupby_with_skew` вызывает
  `check_partition_buffers(num_partitions, min(buf, 64KB))` перед
  открытием P writers (mirror `shuffle.py:273`).
- **P-G5 (S2, blind spot)** — sample cache bounded by `min(10k rows,
  0.3×avail)` bytes. Wide schemas (>=10 KB/row) hit byte cap before row cap.
- **P-SH2 (S2)** — `_hash_column` uint64 → `pc.cast(arr, int64, safe=False)`
  bitcast → pandas C-hash (19× faster than Python fallback). Bitcast =
  two's-complement reinterpretation, bijection over uint64 ↔ int64 →
  same value → same hash (partition-disjoint preserved). Nulls propagate
  through `pc.cast` unchanged. (v0.8.9-fix2: prior XOR-2^63-then-cast
  variant raised `ArrowInvalid` because values ≥ 2^63 overflow
  `pc.cast(safe=True)` — every uint64 value, since XOR with sign bit
  pushes all into [2^63, 2^64). Fix: skip XOR, cast directly with
  `safe=False`. REPL snippet documented in `_hash_column` docstring.)
- **P-A4 (S4)** — удалён dead `if False` block в `_exact_actual_matches_arrow`.

### Добавлено

- **§9.2.6–9.2.19 в DECISIONS.md** — канонические формулы пика для 14
  Arrow/pandas calls, derived из **RSS bench** (не из `pa_delta`).
  Каждая формула сопровождается guard recommendation и residual от
  fitted regression. Пример: `f_join = 5 × output_nbytes` (RSS-based,
  было бы 3.5× из pa_delta).
- **§3.11** — новый обязательный принцип: **Arrow call memory discipline.**
  Любой Arrow/pandas call, аллоцирующий > O(batch), требует:
  (1) формула пика в §9.2.N; (2) явный check_state до вызова;
  (3) AST guard `test_arrow_call_memory_guard.py`.
- **§13.4/138** — «Доверие Arrow = доверие LightGBM». Same class as
  §13.4/25 (LightGBM adapter) и §13.4/26 (partial_fit whitelist).
  MemoryGuard проверяет RSS, но не видит allocation **внутри** Arrow-функции.
  Отсюда и `pa_delta` vs RSS разница в 2× на больших joins.
- **`tests/test_arrow_call_memory_guard.py`** — AST guard: 14 high-risk
  attributes (`join`, `group_by`, `sort_by`, `to_pandas`, `from_pandas`,
  `from_batches` list-form, `sort_indices`, `quantile`, `rank`,
  `value_counts`, `tdigest`, `index_in`, `is_in`, `dictionary_encode`).
  Каждый должен иметь preceding guard (check_state/check_batch/stat)
  в том же function scope. Whitelist для helpers (`_safe_materialize_partition`,
  `_local_dedup`, `_groupby_with_skew`, `hash_shuffle`).
- **`tests/test_pingis_safety.py`** — 14 behavior tests:
  `_estimate_row_bytes` (4), uint64 hash (3), groupby skew+budget (1),
  groupby sample byte-cap (1), dedup pandas chain raise (1),
  AST guards (2), round-trip sanity (3).

### Изменено

- `cadw/ops/join.py` — новая module-level `_estimate_row_bytes(schema)`.
- `cadw/ops/shuffle.py` — uint64 в `_hash_column` не идёт через Python loop.
- `dedup._local_dedup` — effective partition size ~5× меньше на тех же
  бюджетах (honest memory bound).
- AST guard при проверке `Table.from_batches` различает `ListComp` (bad)
  vs `GeneratorExp` (good).

### Ограничения (v0.8.9, §11.5)

- **P-Q2 `pc.quantile` whole column** — sort-based, ~1× column working
  copy. **NEEDS-BATCHED**, не используется в `cadw/`. Backlog §11.6.
- **P-R1 `pc.rank` whole column** — ~2× output + sort buffer.
  **NEEDS-BATCHED**. Per-batch use в `rank.py:94` already safe.
- **P-G2/P-G5 `pc.value_counts` при K≈N** — O(N) hash. **NEEDS-BATCHED**.
- **P-SH1 string shuffle** — needs research (dictionary growth не
  детерминирован, partition-disjoint invariant at risk).
- **P-A3 wide-schema benchmark** — measured on int64 only; wide string
  schemas deferred to v0.8.10 (proposed: N=500k, 5 string cols).
- **P-G3 skew workload benchmark** — confirmed by code review, not
  benchmarked. Deferred to v0.8.10 (N=1M, one key at 60%, P=16).

### Тесты

- fast: **~1124 passed, 1 skipped**. slow: **8 passed**.
- +14 в `tests/test_pingis_safety.py`, +2 AST в
  `tests/test_arrow_call_memory_guard.py` (guard + sanity on generator
  vs list-comp).

## [0.8.8] — 2026-10-19

### Добавлено (housekeeping, не функциональность)
- **`scripts/collect_project.py`** — переписан с `--mode=smart|full|tiny`.
  - `smart` (default): full для `cadw/`, `tests/`, `README`, `pyproject`,
    latest-3 `patch_*.py`, top-5 CHANGELOG, DECISIONS (§13.4 truncated to
    last 10 episodes). Stub: old patches, archive, backups, `uv.lock`,
    old CHANGELOG entries. **~2 MB → ~500 KB** (75% reduction).
  - `full`: старое поведение (all files).
  - `tiny`: README + tree + top-3 CHANGELOG (~50 KB).
- **`scripts/patch_v088_housekeeping.py`** — one-shot cleanup + audit:
  - Phase A: rewrite `collect_project.py`.
  - Phase B: move `scripts/patch_*.{sh,py}` (older than keep=3) →
    `scripts/archive/v0.8.7/`. `git mv` (сохраняет history).
  - Phase C: move `.backups/*` → `/tmp/cadw_backups_<ts>/` (вне repo).
  - Phase D: append [0.8.8] + §13.4/136 + §11.6.
  - Phase E: audit (layers, primitives, §13.4/§11.5/§11.6 counts).

### Framework status (audit подтвердил)
- 7 layers (Storage / Dataset / Access / Primitives / Operations /
  DataFrame / Export) — все на месте.
- 3 primitives (Reduce / HashShuffle / Sort) — stable API.
- Все `ops/*.py` (кроме primitives) импортируют `from cadw.ops.{reduce,
  shuffle, sort}` — нет reinvention.
- Все §13.4 episodes closed через CHANGELOG-записи.
- §11.5 — 18 known limitations + backlog (documented, not bugs).

### Рабочая папка
- `scripts/` → только `collect_project.py`, `_lib.sh`, latest-3 patches,
  `archive/v0.8.7/` (n-исторических), `bench_*`, `diagnose_*`.
- `.backups/` → очищено (`/tmp/cadw_backups_*`).
- `.gitignore` — без изменений (archive tracked).

### Тесты
- fast: **~1096 passed**. slow: **8 passed**.
- CHANGELOG guard: 3 passed (top == tag, descending, exists).

# Changelog

Все значимые изменения проекта CADW. Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).

<!--
  Порядок: **descending by version** (v0.8.5 сверху). Восстановлен в
  v0.8.5 из-за сломанной цепочки v0.1.5+ (§13.4/111): каждый
  `patch_changelog()` вставлял перед anchor `[0.X.Y-1]`, а при
  отсутствии anchor `str.replace` был silent no-op. Как побочный
  эффект, anchors дрейфовали — top становился out-of-order. Пропущенные
  версии (v0.1.5, v0.1.6, v0.2.2, v0.3.5, v0.4.2, v0.5.0, v0.5.2-v0.5.4,
  v0.6.x, v0.7.0-v0.7.7, v0.7.10) — в `git log --oneline --grep='^v'`.
  AST guard `test_changelog_has_latest_tag` (top == tag on master) +
  `test_changelog_is_descending` (v0.8.5) фиксируют инвариант.
-->

## [0.8.7] — 2026-10-19

### Добавлено
- **`CadwDataFrame.resample(freq, agg, on=..., columns=None)`** —
  bucket aggregation по временной/целочисленной колонке.
  - `freq: int | str` — секунды (`5`) или строка (`'5s'`, `'1min'`/
    `'1m'`, `'1h'`, `'1D'`).
  - `agg ∈ {sum, mean, min, max, count}`.
  - `on: str` — timestamp / date / integer column. Bucket =
    `(t_sec // freq) * freq`.
  - Output: `[on, f'{col}_{agg}'...]`. Bucket column dtype = `on` dtype
    (timestamp → timestamp, int → int, date → date).
  - Peak RAM: batch + O(#cols) state + 1 output batch (BATCH=1000
    buckets). **Independent of N и bucket size** (running accumulator,
    не `list(values)`).

### Валидировано prototype-first (§13.4/123)
- `scripts/proto_resample.py`: A (pandas groupby ts//freq) == B
  (external_sort + bucket transition) == C (manual). 5 buckets × sum:
  `[(0,4),(5,13),(10,23),(15,32),(20,22)]`.

### Renamed: v0.8.3 → v0.8.7 (§13.4/132)
- Планировалось как v0.8.3 (roadmap). После v0.8.4/v0.8.5/v0.8.6 уже
  tagged, `test_changelog_is_descending` требует top = highest →
  v0.8.7 (не v0.8.3).

### Semantics (MVP, §11.5)
- **Empty bucket** (gap in ts) → skip (no NaN fill; для fill —
  `.reindex` после).
- **Null in `on`** → skip row (cannot bucket).
- **All-null agg bucket** → skip row (all columns None).
- **Null in agg col** → skip value (matches pandas skipna).
- **Nulls_all**: bucket с только null значениями (все cols) → skip row.
- **Partition**: bucket spanning batches — accumulator живёт через
  batch boundary (running state, O(1) RAM).

### Ограничения (v0.8.7, §11.5)
- `on` — single column. Multi-column (e.g. `on=['ts', 'sensor']` для
  per-group buckets) — не поддерживается (composite sort ≠ per-group;
  для per-group — `groupby(...)`).
- `freq` — только `int` секунды или строка `'Ns'|'Nmin'|'Nm'|'Nh'|'ND'`.
  Compound (`'5min30s'`), `'W'`, `'M'`, `'Y'`, sub-second (`'500ms'`) —
  не поддерживаются.
- `uint64` on — reject (overflow risk при bucket arithmetic; cast в
  int64 required).
- `date` on — granularity = days (freq should be ≥ `'1D'`; `'5s'` на
  date даст O(#days) buckets с 0 rows → skip → фактически пустой
  output для одного дня).
- All-null bucket + `count` — skip (не 0; для zero-fill — post-process
  через `left_merge`).
- `nan` timestamp bucket on far future (>year 2262 для ns) — overflow
  (int64). Documented, edge case.

### Тесты
- +28 в `test_resample.py`: sum/mean/min/max/count (prototype data),
  multibatch bucket, empty bucket skip, nulls, all-null bucket, null
  in on, timestamp('s'/'ms'), date32, validation (missing on,
  non-time-type, uint64, on-in-columns, invalid agg, non-numeric,
  empty columns), freq parser (9 param + 5 edge), preserves original,
  filtered, 2 AST guards.
- fast: **~1092 passed**. slow: **8 passed**.

## [0.8.6] — 2026-10-19

### Добавлено
- **Multi-column `on`** для `shift` / `diff` / `rolling`:
  `on=["ts", "sensor"]` (Sequence[str]).
  - **Лексикографический composite sort**, не per-group. Сортировка по
    `[on_cols..., row_id]`, применение shift/rolling per row, sort-back
    по `__orig_rid` (как в v0.8.2 / v0.8.4).
  - Per-group — `groupby('sensor').shift(1, on='ts')` (v0.8.5a+).
- `CadwDataFrame._normalize_on(on, fn_name)` — helper: `str | Sequence[str]`
  → `list[str]` + validation.
- `_shift_on(..., on_cols: list[str])` / `_rolling_on(..., on_cols:
  list[str])` — принимают pre-normalized list; `external_sort` получает
  `key_columns=on_cols`.

### Backward compat
- `on="ts"` — как раньше (single-element normalization).
- `on=["ts"]` — идентично `on="ts"`.
- `on=[]` — `ValueError: {fn}: on must be non-empty`.

### Ограничения (v0.8.6, §11.5)
- Multi-column `on` — **composite sort**, не per-group. Для per-group
  windows — `groupby(...).shift/rolling(..., on=...)` (v0.8.5a).
- `_iter_shifted` / `_iter_zipped` не дедуплицируются — дублирование
  остаётся (refactor в backlog).
- Stability with duplicate `on` values — tie-break by row_id (heap),
  как в v0.8.2/0.8.4.

### Тесты
- +5 в `test_shift_on_col.py`: multi-key shift/diff, single-element-list
  equivalence, empty-list ValueError, stable tie-break.
- +3 в `test_rolling_on_col.py`: multi-key sum, min_periods=1,
  empty-list ValueError.
- AST guards (v0.8.2/0.8.4) — unaffected (first sort uses `on_cols`
  variable; second still literal `["__orig_rid"]`).
- fast: **~1064 passed**. slow: **8 passed**.

## [0.8.5] — 2026-10-19

### Исправлено
- **CHANGELOG отсортирован descending** (§13.4/111 + §13.4/130). Исторически
  `patch_changelog()` вставлял перед anchor `[0.X.Y-1]`. Anchor дрейфовал,
  top CHANGELOG перестал быть «latest». `test_changelog_has_latest_tag`
  (master + tag) корректно упал: `top [0.7.4c] != tag v0.8.4`. Fix:
  одноразовый sort всех `## [X]` блоков by (major, minor, patch, suffix)
  descending. Новый guard `test_changelog_is_descending` (в
  `test_changelog.py`) ловит регресс. v0.8.4 tag **не** ретагнут —
  тест не запускается на старых tags (`--exact-match` на HEAD).
- **`row_id` reserved guard** в `external_sort` и `hash_shuffle`
  (§13.4/126). User column named `row_id` коллизит с internal identity
  column (appended by external_sort/hash_shuffle) → previously:
  `KeyError: Field "row_id" exists 2 times in schema` (через
  `ArrowIPCBatchReader`, в `sort_values`); `Multiple matches for
  FieldRef.Name(row_id)` (в `pc.sort_indices`). Теперь: `ValueError:
  external_sort: column name 'row_id' is reserved for CADW internal
  identity column. Rename your column before calling.` Guard выполняется
  **до** materialize batches.

### Добавлено
- **`tests/_helpers/rolling_reference.py`** — `expected_rolling(vals,
  window, agg, min_periods)` — deque ground truth (§13.4/128). Matches
  pandas: positional window, nulls consume slot, min_periods applies to
  non-null count.
- 4 rolling-теста (`test_rolling_min/max_with_nulls` в
  `test_rolling.py`; `test_rolling_on_min/max_with_nulls` в
  `test_rolling_on_col.py`) → `expected_rolling(...)`. Устраняет корень
  §13.4/128 (manual arithmetic).
- 8 новых тестов в `test_row_id_reserved.py`: external_sort
  (reject key/value + valid works), hash_shuffle (reject key/value +
  valid works), E2E (sort_values/groupby на user `row_id`).
- `test_changelog.py::test_changelog_is_descending` — strict descending
  invariant.

### Ограничения (v0.8.5, §11.5)
- `row_id` — reserved. User column → rename. `__orig_rid` тоже reserved
  (internal sortback). Рекомендуется префикс `my_`.
- Guard fires on arguments (`key_columns` / `value_columns`). Internal
  callers (`_shift_on`, `_rolling_on`, `rank`, `groupby`) либо передают
  `__orig_rid`, либо не содержат `row_id` в value list — не затронуты.

### Тесты
- +8 в `test_row_id_reserved.py`.
- +1 в `test_changelog.py` (descending guard).
- 4 rolling-теста → helper (behavior unchanged).
- fast: **~1056 passed**. slow: **8 passed**.

## [0.8.4] — 2026-10-19

### Добавлено
- **`CadwDataFrame.rolling(window, agg, on=None)`** — time-aware rolling.
  `on=None` — positional (v0.6.2c, unchanged); `on=str` — sort-aware:
  1. external_sort by `[on]` (row_id tie-break from heap).
  2. `_RollState.push` per row (reuse v0.6.2c state — same min/max
     monotone deque, same null-skip, same min_periods).
  3. external_sort by `__orig_rid` (rename `row_id` → `__orig_rid`,
     §13.4/122).
  4. Zip original + rolling cols **APPENDED** (`{col}_rolling_{agg}` —
     rolling adds new columns, doesn't replace).
  - Peak RAM = batch + O(window × #cols). Independent of N.

### Валидировано prototype-first (§13.4/123)
- `scripts/proto_rolling_on.py` — A (sort_values+rolling+sort_values)
  == B (external_sort + deque + external_sort) == C (pure Python manual).
- Same 5-row case as prototype; 1 итерация патча (v0.8.2: было 5 fix).

### Исправлено (v0.8.4-fix — S1, latent bug с v0.6.2c)
- **`_RollState.push(None)` для agg='min'/'max'** — `while _dq and
  _dq[-1][1] >= None` → `TypeError: '>=' not supported between int and
  NoneType`. Bug существовал с v0.6.2c (позиционный rolling), не
  покрыт тестами (`test_rolling_with_nulls` использует sum). Fix:
  guard `if not is_null` вокруг monotone-deque maintenance. Window
  slot всё равно потребляется (positional semantics, как pandas).
  Добавлены тесты: `test_rolling_min_with_nulls`,
  `test_rolling_max_with_nulls`, `test_rolling_min_all_nulls`
  (positional) + `test_rolling_on_min_with_nulls`,
  `test_rolling_on_max_with_nulls` (on=col).
- **v0.8.4 tests (2, not code)**: `test_rolling_on_with_nulls` и
  `test_rolling_on_all_columns` — неверные ожидания. Actual code
  корректен (nulls потребляют window slot, positional; matches pandas
  skipna=True). Fixed expected values + comments.

### Обнаружен latent bug (§13.4/126)
- User column named `row_id` collides with internal `row_id`
  (external_sort/hash_shuffle/merge) → `KeyError: Field "row_id" exists
  2 times in schema` (в `sort_values`). Documented §11.5 as reserved
  column name; fail-fast guard — v0.9.

### Ограничения (v0.8.4, §11.5)
- `on` — single column. Multi-column `on=['ts', 'sensor']` — v0.8.4a.
- Per-group rolling (`groupby('sensor').rolling(...)`) — v0.8.5+.
- `row_id` — reserved column name (internal).

### Тесты
- +17 в `test_rolling_on_col.py`: sum/mean/min/max/count, min_periods,
  nulls, multibatch, preserved, all columns, sort-after, missing col,
  non-numeric, no-on backward compat, +2 AST guards (row_id rename,
  append-not-replace).
- fast: **~1043 passed**. slow: **8 passed**.

## [0.8.2] — 2026-10-19

### Добавлено
- **`CadwDataFrame.shift(k, fill_value=None, columns=None, on=None)`** —
  time-aware shift. `on=None` — positional (v0.6.2b, unchanged);
  `on=str` — sort-aware:
  1. external_sort by `[on, row_id]` (stable tie-break).
  2. Stateful shift (deque per column, O(k)).
  3. external_sort by `row_id` → original order.
  4. Zip with original.
  - Peak RAM = batch + O(k × #cols). Independent of N.
  - 2 external_sorts (vs manual `df.sort_values(x).shift(x).sort_values(x)`).
  - Multibatch consistent (chunk size не влияет).

- **`CadwDataFrame.diff(k, columns=None, on=None)`** — time-aware diff.
  `on=None` — positional (v0.6.2b); `on=str` — sort-aware via `_shift_on`
  + `pc.subtract`. Reuses shift's 2 sorts, no extra sort.

### Ограничения (v0.8.2, §11.5)
- `on` — single column. Multi-column `on=['ts', 'sensor']` — v0.8.2a.
- `on` с duplicates — stable sort (tie-break by row_id asc). Documented.
- `on` должен быть sortable (monotonic не требуется — обычный sort).
- Per-group shift (`groupby('sensor').shift(1, on='ts')`) — v0.8.3+.

### Тесты
- +17 в `test_shift_on_col.py`: shift on basic/k2/fill/string/all,
  preserved, nulls, multibatch, missing col; diff on basic/k2/preserved,
  non-numeric raises, non-monotonic stable, nulls; chain sort; backward
  compat (no-on unchanged).
- fast: **~1014 passed**. slow: **8 passed**.

### Цена
- Shift(1, on='ts') on 10^6 rows: ~2-3s (2× external_sort).
- Same as user-manual `df.sort_values('ts').shift(1).sort_values('ts')`,
  but API one-liner.

## [0.8.1] — 2026-10-19

### Изменено — performance
- **`compute_hashes`: pandas C-hash для numeric, Python для string.**
  - Numeric/boolean/timestamp/date → `pd.util.hash_pandas_object`
    (C-level SipHash). Bench: int64 39.2→2.0 ms (**19.6×**),
    float64 39.2→1.0 ms (**39.2×**).
  - String/large_string → Python `repr().encode()` + xxhash. Причина:
    pandas `hash_pandas_object` для object dtype — per-element Cython,
    50 ms/100k, **22% медленнее** Python (§13.4/107). Без регрессии.
  - struct/list/dict/uint64 → Python loop (fallback).
  - Seed через `h_np ^= np.uint64(seed)` **до** avalanche.
- **Avalanche (splitmix64) — всегда**, не только при seed. Причина:
  pandas `hash_pandas_object` для object dtype `hash << 1` (reserves 0
  для NA), все string hashes чётные → `h & 3` использует только
  партиции `{0, 2}` → recursion split 2-way вместо 4-way →
  `MemoryLimitExceeded` при depth=5 (§13.4/107). Fix: 3 умножения +
  3 XOR-shift, ~0.3 ms/100k.

### Критично для корректности
- **`types_mapper=Int64Dtype`** в `arr.to_pandas()` (numeric): без него
  `int64` с nulls → `float64` (NaN), без nulls → `int64` → разные hash
  для одного значения в разных батчах → нарушение partition-disjoint.
- Тесты: `test_int64_consistent_hash_across_null_presence`,
  `test_avalanche_spreads_low_bits`,
  `test_avalanche_seed_spreads_low_bits`.
- `test_shuffle.py::test_stable_hash_across_batches` — убран лишний
  `assert a_parts != b_parts` (partition-disjoint ≠ different partitions;
  коллизии разных ключей в одной партиции — норма).

### Тесты
- +20 в `test_hash_v081.py`: cross-batch consistency (int32/int64/string
  с/без nulls), deterministic, seed, nulls, dispatch (string→python,
  numeric→pandas), partition-disjoint (int+string), avalanche (low-bit
  spread + seed + similar strings), groupby recursion end-to-end.

### Ожидание
- int64 shuffle 1M: 0.61s → **~0.22s** (2.76×).
- 500 GB / 50 B / int64 keys: ~100 min → **~40 min**.

## [0.8.0] — 2026-10-19

### Добавлено — time-series
- `CadwDataFrame.expanding(agg, min_periods=1, columns=None)` —
  кумулятивное окно. Aggs: `sum`/`mean`/`min`/`max`/`count`. O(1) state
  (scalar accumulators; min/max — текущий min/max, без deque). Nulls
  skip; null row → null output. Output: `{col}_expanding_{agg}`.
- `CadwDataFrame.ewm(alpha/span/halflife/com, adjust=True, min_periods=0,
  columns=None)` — экспоненциально-взвешенное среднее. Ровно один из
  4 параметров. `adjust=True`: `(s_num, s_den)`, `y = s_num/s_den`.
  `adjust=False`: `(ewm,)`, `y = (1-α)·y_prev + α·x`. Nulls skip
  (`ignore_na=True` семантика). Output: `{col}_ewm`.

### Добавлено — PCA single-pass
- `CovarianceMeanOp(CovarianceOp)` — тот же state, `finalize` →
  `{cov, mean, n}`. `fit_pca` → **2× I/O reduction**.
- `CovarianceOp` (bare matrix) не меняется — backward compat.

### Ограничения (v0.8.0, §11.5)
- `expanding`/`ewm` не поддерживают `on=col`.
- `ewm(ignore_na=False)`, `ewm(bias=True)` — не реализованы.

### Тесты
- +26 в `test_expanding_ewm.py`, +3 (CovarianceMeanOp + fit_pca regression).

## [0.7.9] — 2026-10-19

### Исправлено в v0.7.9-fix
- **`_AssignLeavesOp.update` 4-unpack** — после введения 5-tuple spec (`..., gain`) старый `j, thr, left, right = spec` падал `ValueError: too many values to unpack`. Fix: `spec[:4]` (forward-compatible slice). Тот же fix применён к `_assign_leaf_from_array` в изначальном v0.7.9.
- **Multiclass `_fit_boosting_multiclass` 4-tuple** — `s.replace(old, new, 1)` дважды не сработало (второй replace не нашёл pattern из-за отличий в форматировании). Fix: replace **all occurrences** без count → multiclass тоже 5-tuple.
- **+AST guard** `test_all_spec_unpack_uses_slice` — все `= spec` должны быть `= spec[:4]` / `= spec[:5]`. Ловит класс «4-unpack + 5-tuple» багов.

### Добавлено
- **`compute_feature_importance(model, importance_type='gain'|'split',
  top_k=None)`** — extraction helper in `cadw.ml.gbdt`.
  - `'gain'` (default): sum of split gains per feature — XGBoost-style,
    better reflects feature usefulness (big gain на 1 split > 10 малых).
  - `'split'`: split count (v0.7.4a semantics, backward compat).
  - Multiclass: gain summed across all (tree, class).
  - Returns dict sorted by importance ascending (for horizontal barh).
  - Handles both binary (list[tree]) and multiclass (list[list[tree]]).

- **Tree structure extended:** `spec = (feat, thr, left, right, gain)`
  (5-tuple) instead of 4-tuple. `gain` stored at index 4.
  `_assign_leaf_from_array` tolerates both 4- and 5-tuple (`spec[:4]`)
  — backward compat for pre-v0.7.9 pickled models.

- **`feature_importance(..., importance_type='gain')`** in matplotlib and
  plotly adapters — use the new helper. Split-count still available via
  `importance_type='split'`.

### Изменено
- **`feature_importance` default** — был `split count`, стал `gain`.
  Breaking change для downstream code, который relied on integer counts
  — но семантически правильнее; документировано.

### Ограничения (v0.7.9, §11.5)
- **Permutation importance** — не реализована (требует out-of-core eval
  на holdout; v0.7.10+).
- **SHAP** — не реализован (слишком тяжёлый для out-of-core; вне scope).
- Legacy 4-tuple spec (pre-v0.7.9 pickled): `gain` returns 0 for those
  trees (silently, gain not available). `split` works.
- `gain` — raw sum (not normalized); разные tree scale → разные magnitudes.
  Для normalized importance — `gain / gain.sum()` в user code.

### Тесты
- +13 в `test_feature_importance_gain.py`: helper (invalid type, non-GBDT,
  sorted), gain semantics (important feature, split differs), top_k,
  zero model, multiclass aggregation, adapters (matplotlib/plotly, both
  importance_type), legacy 4-tuple fallback, spec is 5-tuple.
- fast: **~995 passed**. slow: **8 passed**.

## [0.7.8] — 2026-10-19

### Исправлено в v0.7.8-fix3
- **Dispatch preserves backward-compat error messages.**
  `fit_gbdt_classifier(n_classes=None)` считает distinct **перед**
  `_validate_multiclass_y`. 2 distinct (или non-integer) → binary
  path (`_validate_binary_y` даёт исторический `"binary (0/1)"` /
  `"both classes"`); >2 distinct → multiclass path (`"0-indexed"`).
  Без этого `{2, 3}` и `{1}` давали `"0-indexed"` вместо ожидаемых
  сообщений — ломалось `test_gbdt_classification.py`.
- **+AST export guard** (`tests/test_api_availability.py`):
  `test_cadw_all_exports_importable`,
  `test_cadw_ml_all_exports_importable`,
  `test_cadw_ast_all_matches_imports`. Ловит класс багов
  «__all__ обновлён, import забыт» (§13.4/115).

### Добавлено
- **`fit_gbdt_multiclass`** — multiclass GBDT (softmax loss), per-class
  loop, reuse существующих ops (`_AllNodeHistOp`, `_AssignLeavesOp`,
  `_LeafValuesOp` — 0 новых ops).
  - K деревьев на round (по одному на класс); `trees_` = `list[n_estimators]`
    of `list[K]`.
  - Grad: `p[i,c] - onehot(y[i], c)`; Hess: `p[i,c]*(1-p[i,c])`
    (XGBoost diagonal approximation).
  - `base_prediction_ = zeros(K)` (деревья находят сдвиг).
  - `predict_proba → (n, K)`, `predict → (n,) int64`.
- **`GBDTMulticlassClassifier`** — model class; sklearn-подобные поля:
  `trees_`, `n_classes_`, `feature_names_`, `n_features_in_`,
  `predict_proba`, `predict`, `to_dict`.
- **`fit_gbdt_classifier(..., n_classes=None)`** — auto-dispatch:
  - `n_classes` inferred из `y` (`{0..K-1}`).
  - `K == 2` → `GBDTClassifier` (binary, unchanged).
  - `K > 2` → `GBDTMulticlassClassifier`.
  - `K < 2` → ValueError.
- **`CadwDataFrame.fit_gbdt_multiclass`** — фасад.

### Ограничения (v0.7.8, §11.5)
- y должен быть `{0, 1, ..., K-1}` (0-indexed). 1-indexed `{1..K}` —
  ValueError (сдвиньте y на -1).
- `K × binary` fit cost (per-class loop). K=3 → ~3× медленнее binary.
  Для K > 5, n > 1M — v0.7.8a добавит `_MulticlassAllNodeHistOp`
  (1 pass для всех K, ~K× speedup).
- One-vs-rest не поддерживается (только softmax).
- Multiclass quantile/huber — не реализованы.

### Тесты
- +22 в `test_gbdt_multiclass.py`: `_softmax` (sum, stability, dim),
  fit (basic, accuracy, n_estimators improves, deterministic), predict
  (proba shape, int labels, df columns, prob_df columns, nulls),
  dispatch (binary, multiclass, single-class error), y validation
  (1-indexed, missing class, K=2 rejected by fit_multiclass),
  memory guard, convenience, to_dict, binary regression.
- fast: **~979 passed**, slow: **8 passed**.

### Время
- K=3, n=1000, depth=3, 30 rounds: ~15-20s (CI). Локально — быстрее.

## [0.7.4c] — 2026-10-19

### Добавлено
- **`matplotlib_adapter.hexbin_plot(df, x, y, gridsize=30, n=10_000,
  random=True, ...)`** — 2D hexagonal binning (matplotlib native).
  - Sample-based (`sample(n)` + `plt.hexbin`), **не** aggregate-first:
    hexbin aggregation на гексагональной сетке требует custom binning,
    matplotlib делает это сам через `plt.hexbin`.
  - `log_scale=True` (default): LogNorm для colorbar (плотность scatter).
  - **Plotly/Seaborn:** нет native hexbin — используйте `hist2d_plot`
    (прямоугольная сетка). Documented.

- **`_aggregate.sample(df, n, *, random=False)`** — новый параметр.
  - `random=False` (default, fast): `df.head(n)` — bounded I/O,
    но **может быть смещён** на time-sorted данных (только префикс).
  - `random=True`: uniform random sample через
    `df.iloc[np.sort(rng.choice(cnt, n, replace=False))]`. Full O(N)
    scan, O(n) memory. Deterministic (seed=42).
  - `scatter` / `_sample_pair`: `random=False` по умолчанию
    (backward compat v0.7.4a).

### Ограничения (v0.7.4c, §11.5)
- `hexbin_plot` — только matplotlib. Plotly/Seaborn — нет hexbin
  (документировано; use hist2d).
- `random=True`: 2× O(N) scan (count + iloc to max position). Для 10^7
  строк ~секунды. `random=False` — fast preview.
- hexbin sample-based, не aggregate-first — n=10_000 bounded, shape
  approximate.

### Тесты
- +4 в `test_viz_aggregate.py`: sample(random=True) uniformity,
  small-df returns-self, deterministic, head default unchanged.
- +5 в `test_viz_matplotlib.py`: hexbin_plot writes PNG, random sample
  spreads, missing column raises, invalid gridsize raises, nulls skipped.
- fast: **~1026 passed**. slow: **8 passed**.

## [0.5.2a] — 2026-10-19

По результатам Pingis audit: P5 (S2) — sort merge Python-list buffer.

### Исправлено
- **P5: `_merge_keys`** — заменить Python-list buffer (`buffer[c].append(
  as_py())`) на `pa.Array.Builder` (`Int64Builder`/`Float64Builder`/
  `StringBuilder`/etc.). Пик на merge: `read_batch_size × arrow_size`,
  а не `read_batch_size × 28-80 B` (Python-object overhead).
  - `_make_builder(dtype)` — выбор builder по Arrow-типу.
  - `_append_to_builders` — append одного значения (с `.as_py()` O(1)).
  - `_flush_builders` — finish → `RecordBatch`.
  - Dictionary/list/struct → `NotImplementedError` (honest; не покрыто).

### Добавлено
- `scripts/bench_p5_sort.py` — RSS-сэмплер benchmark для sort
  (сравнение с Pingis baseline +31.9 MB).
- `tests/test_sort.py::test_merge_keys_uses_arrow_builder_not_python_list`
  — AST guard (нет `buffer[...].append`).
- `tests/test_sort.py::test_merge_keys_uses_arrow_builder` — presence check.

### Ожидаемый эффект
- B3a (1M rows, 30 MB budget): Δpeak ≤ 30 MB (baseline: +31.9 MB).

### Ограничения (v0.5.2a, §11.5)
- Dictionary/list/struct columns: `NotImplementedError` в sort merge.
- `.as_py()` per value — CPU overhead остаётся; P5 — memory-fix, не
  speed-fix. CPU-оптимизация (batch-wise `pc.sort_indices`) — v0.5.4.

## [0.5.1] — 2026-10-19

По результатам мультиагентного анализа Pingis (`_FOR_PINGIS/REPORT.md`):
P2 (S1) + P8 (S2).

### Исправлено
- **P2 (S1, streaming join):**
  - `_process_partition_pair` — теперь generator (`yield` вместо
    `out_batches.extend(...)`).
  - `_left_preserve_with_nulls_iter` — streaming-версия.
  - `_join_core` — общий setup (validate, peek, shuffle) → `(schema, iter)`.
  - `hash_join_streaming(...) -> (schema, Path)` — **NEW**: пишет
    выходные батчи напрямую в `ArrowIPCBatchWriter`. Peak RAM =
    `max(build_partition) + probe_batch + writer_buffer`, **не зависит
    от размера output**.
  - `CadwDataFrame.merge` использует `hash_join_streaming`; промежуточный
    `.arrow` удаляется после `Dataset.from_batches`.
  - `hash_join(...)` (старый API) сохраняет семантику — собирает
    streaming-output в `pa.Table` (backward compat).

- **P8 (S2, state_bytes API):**
  - `ReduceOp.estimate_state_nbytes() -> int | None` — метод ABC
    (default None → fallback на `MemoryGuard.estimate_state_size`).
  - Built-in ops override: `CountOp`/`SumOp`/`MinMaxOp`/`MeanOp` → 16;
    `StdOp` → 24; `CovarianceOp` → `8+8d+8d²`; `HistogramOp` → `8bins+16`.
  - `reduce()` вызывает `op.estimate_state_nbytes()` **до** `op.init()`.
  - User `ReduceOp` can override — публичный контракт для ML-слоя.

### Добавлено
- `cadw.ops.hash_join_streaming` — streaming join API.
- `tests/test_reduce_state_bytes.py` — 8 тестов (built-in + custom +
  fallback + oversized-state-rejects-before-init).
- `tests/test_join_streaming.py` — 10 тестов (generator, disk write,
  incremental batches, backward compat, merge uses streaming, cleanup,
  empty input).
- `.pingis/TASK.md` — periodic analysis task template.
- `.gitignore`: exception для `.pingis/TASK.md`.

### Изменено
- `test_join_oom_guard.py::test_oversize_partition_raises_with_max_depth_zero`
  — `next(gen)` вместо `gen(...)` (generator).

### Ожидаемый эффект
- Join B1a (100k × 10k под 80 MB) → Δpeak с ~140 MB до ≤ 80 MB (цель).
- ML custom `ReduceOp` (XTXOp, KMeansStepOp, HistOp) не получают
  warning про 1 MiB; правильный state-размер.

### Ограничения (v0.5.1, §11.5)
- P1 (join `to_pandas`/`pd.merge`) — отдельный патч (v0.5.2).
- P3/P4/P5/P6/P7/P10/P11/P12/P14 — в roadmap после v0.5.2.
- `hash_join` (не-streaming) всё ещё может OOM при большом output —
  используйте `hash_join_streaming` или `CadwDataFrame.merge`.

## [0.4.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.loc` — open-endpoint slices на monotonic index:
  - `loc[:hi]` → от начала до `hi` (inclusive).
  - `loc[lo:]` → от `lo` до конца.
  - `loc[:]` → все строки (lazy view).
  - descending monotonic тоже работает (`loc['b':'d']` → d,c,b).
- `CadwDataFrame.loc[label]` — duplicate labels → `CadwDataFrame`
  (pandas: Series→DataFrame). Unique → `dict` (без изменений).
- `CadwDataFrame.loc[[labels]]` — все вхождения для каждого label
  (ранее first-wins).
- `MaterializedIndex.is_monotonic_increasing()`,
  `is_monotonic_decreasing()`, `is_monotonic()` — O(N), nulls skipped.
- `MaterializedIndex.has_duplicates()` — O(N), set-based.
- `MaterializedIndex.get_positions_all(label)` — все позиции по `==`.

### Изменено
- `loc[slice]` на **non-monotonic** index → `NotImplementedError`
  (pandas 2.x тоже).
- `loc[slice]` на **duplicate** index → `NotImplementedError`
  (pandas: cannot slice non-unique).
- `loc[slice]` с `step != 1` → `NotImplementedError`.
- `.at[label, col]` на duplicate label → `KeyError` (pandas: undefined;
  используйте `loc[[label]]`).

### Ограничения (v0.4.1, §11.5)
- **MultiIndex** (`set_index(['a','b'])`) — v0.4.2.
- **`loc[slice]` step** — v0.4.2.
- **Non-monotonic + slice** — `NotImplementedError` (pandas-стиль).
- **`.at` на duplicate** — `KeyError`.

### Тесты
- 22 новых в `test_loc.py`: open-endpoints (start/stop/both на
  asc/desc), monotonic API, `get_positions_all`, `has_duplicates`,
  duplicate-label `loc` (dict vs CadwDataFrame), `loc[[duplicates]]`,
  slice на duplicate → raise, `.at` на duplicate → raise.
- Обновлён 1 тест: `test_loc_slice_none_endpoint_raises` → 8 новых.

## [0.4.0d] — 2026-10-19

### Добавлено
- `_iter_filtered_batches_with_positions()` — yield `(batch, positions)`,
  positions = numpy int64 в unfiltered dataset. Композиция фильтров:
  MaskFilter → `positions[mask]`; SliceFilter → slice по local counter
  с сохранением raw positions.
- `_apply_filter_with_positions(filter, it)` — helper для filter chain.

### Изменено (filter-compat)
- `df.index` при фильтрах → `MaterializedIndex(_index_values[positions])`
  (pandas-совместимо). Раньше — `NotImplementedError`.
- `set_index(col)` при фильтрах → работает; материализует **полный**
  col (O(N) scan), `_index_values` = full → любой комбинации filter/set
  через positions.
- `reset_index(drop=True)` при фильтрах → materialize filtered view +
  RangeIndex(0, filtered_n).
- `reset_index(drop=False)` при фильтрах → колонка `index` =
  `_index_values[positions]`.
- `iloc[list]` на filtered view — **сохраняет** parent index через
  raw positions (раньше reset). Более pandas-совместимо.
- `loc` на filtered view — работает (через filtered.index).

### Закрыто (из v0.4.0a-c, §11.5)
- ~~`df.index`/`set_index`/`reset_index` с активными фильтрами — raise~~
- ~~`iloc[list]` на filtered view не сохраняет parent index~~
- ~~`loc` на filtered view — NotImplementedError~~

### Тесты
- 15 новых в `test_filter_compat.py`: filtered.index (range/mat/
  slice/mask_then_slice/empty), set_index+filter, filter+set_index,
  reset_index drop=True/False при фильтрах, iloc[list] preserves
  parent (range/materialized), loc на filtered, no-filter unchanged.
- Обновлены 4 теста: `test_index_with_filter_*`, `test_set_index_with_
  filter_*`, `test_reset_index_with_filter_*`, `test_iloc_list_on_
  filtered_*`, `test_loc_on_filtered_*` — теперь проверяют
  сохранение/работу, не raise.

## [0.4.0c] — 2026-10-19

### Добавлено
- `CadwDataFrame.loc` — label-based indexer (pandas-style).
  - `loc[label]` → `dict` (KeyError если label нет).
  - `loc[lo:hi]` → `CadwDataFrame` — inclusive labels, reverse → empty.
  - `loc[[l1, l2, ...]]` → `CadwDataFrame`.
  - Требует `MaterializedIndex`; `RangeIndex` → `TypeError`.
- `CadwDataFrame.at[label, col]` — scalar access by label + column.
- `CadwDataFrame.iat[pos, col]` — scalar access by position + column.
- `MaterializedIndex.get_indexer(labels)` → `list[int]` (O(N) scan, dict;
  unhashable fallback linear).
- `.at`/`.iat` — **read-only** (`__setitem__` → `NotImplementedError`).

### Ограничения (v0.4.0c, §11.5)
- **`loc` требует `MaterializedIndex`** (после `set_index`).
  `RangeIndex` → `TypeError`; используйте `iloc`.
- **`loc[lo:hi]`** — только оба endpoint обязательны; `loc[:hi]` /
  `loc[lo:]` / `loc[::step]` → `NotImplementedError`. v0.4.1.
- **`loc[bool_array]`** — `NotImplementedError` (use `df[mask]`).
- **`loc` на filtered view** — `NotImplementedError`. v0.4.0d.
- **duplicate labels** — первое вхождение; v0.4.1 — все.
- **`.at`/`.iat` set** — `NotImplementedError` (immutable).

### Тесты
- 30 тестов в `test_loc.py`: get_indexer (scalar/list/dup/first-wins/str),
  loc on RangeIndex raises, loc scalar/slice(inclusive/reverse/missing),
  loc list (basic/missing/dup/empty), loc bool_array raise, .at
  scalar/missing/set-raise, .iat scalar/out-of-range/set-raise,
  filter limit, loc preserves index.

## [0.4.0b] — 2026-10-19

### Добавлено
- `CadwDataFrame.iloc` — positional indexer (pandas-style).
  - `iloc[int]` → `dict[str, Any]` — одна строка.
  - `iloc[slice]` → `CadwDataFrame` — lazy, через `SliceFilter`.
  - `iloc[list[int]]` → `CadwDataFrame` — materialize K=len(list) rows.
  - Negative indices работают (int/list — resolve через count();
    slice — через `key.indices(n)`).
- `cadw/indexers.py` — модуль для indexers (`_ILocIndexer`). Отдельно
  от facade (`dataframe.py`), чтобы не ломать AST guard
  `test_facade_has_no_as_py`.

### Ограничения (v0.4.0b, §11.5)
- **`iloc[bool_array]`** — `NotImplementedError` (use `df[mask]`).
- **`iloc[list]` на filtered view** → default `RangeIndex` (не сохраняет
  parent index); v0.4.0c — filter-compat.
- **`iloc[list]` на unfiltered** — preserves parent index:
  RangeIndex → MaterializedIndex([start+p, ...]);
  MaterializedIndex → subset values (dtype preserved).
- **`iloc[list]`** — O(N) I/O scan (нужно дойти до последней позиции);
  O(K) память результата.

### Тесты
- 21 тест в `test_iloc.py`: iloc[int] (mid/last/negative/out-of-range),
  iloc[slice] (basic/step/negative/filtered), iloc[list]
  (basic/duplicates/negative/empty/out-of-range/filtered), index
  preservation (range/materialized/filtered-reset), errors
  (bool_array/unsupported type), AST guard sanity.

## [0.4.0a] — 2026-10-19

### Добавлено
- `CadwIndex` (ABC) + `RangeIndex(start, stop, step)` (lazy, O(1))
  + `MaterializedIndex(values)` (Arrow Array).
- `CadwDataFrame.index` property — RangeIndex (default) или
  MaterializedIndex (после set_index).
- `CadwDataFrame.set_index(col, drop=True, inplace=False)`:
  материализует колонку в память (Arrow Array).
- `CadwDataFrame.reset_index(drop=False, inplace=False)`:
  `drop=True` — отбрасывает index; `drop=False` — выносит в колонку
  `index` (или `level_0` при коллизии).
- Экспорт: `cadw.CadwIndex`, `cadw.RangeIndex`,
  `cadw.MaterializedIndex`.
- `cadw.dataset.index` — модуль.

### Ограничения (v0.4.0a, §11.5)
- **с активными фильтрами**: `df.index`, `set_index`, `reset_index` —
  raise `NotImplementedError` (RangeIndex+filter = O(N) scan;
  синхронизация — v0.4.0b).
- **`set_index`** материализует колонку (O(N) RAM). Для больших N —
  future (streaming loc — v0.4.1).
- **Index не сохраняется в Parquet** (`write_parquet` → reload =
  RangeIndex). Pandas-совместимое поведение.

### Тесты
- 20 тестов в `test_cadw_index.py`: RangeIndex/MaterializedIndex API,
  set_index drop=True/False/inplace, reset_index drop=True/False,
  коллизия «index» → «level_0», chained, filter-limit raise,
  persistence.

## [0.3.4] — 2026-10-19

### Добавлено
- **Combiner** для `groupby`: partial aggregation до shuffle.
  - `CadwDataFrame.groupby(...).agg(..., combiner=True)` — opt-in.
  - `groupby_aggregate(..., combiner=True)` — runtime.
  - Поддерживаемые агрегаты: `sum`, `count`, `min`, `max`, `mean`.
  - Для `mean`: partial (sum, count), финально `sum/count`.
  - Экономия: партиция `N × row_size` → `K_batch × comb_bytes`
    (×10000 для 10^6 строк / 100 групп).

### Как работает
1. `_combine_batches(batch_iter, by, agg_cols, agg_funcs)` — generator:
   `pandas.groupby(by).agg(partial_specs)` per batch. State O(batch_size).
2. `hash_shuffle` получает `value_columns = __comb_*`.
3. `_local_groupby(..., partial_merge=True)` вызывает `_merge_partials`:
   `table.group_by(by).aggregate(merge_specs)` + переименование + dtype.
4. Результат идентичен `combiner=False`.

### Ограничения (v0.3.4, §11.5)
- **std/var**: не поддерживаются в combiner (требуют Welford partial).
  Warning + fallback на обычный путь. v0.3.5.
- **skew path**: combiner отключён (heavy через streaming, light через
  shuffle без combiner). Warning. v0.3.5 — combiner для light.
- **auto**: default `combiner=False` (opt-in). Auto по `k_est` — v0.3.5.

### Тесты
- 12 тестов в `test_combiner.py`: equivalence для sum/count/min/max/mean,
  mixed aggs, same-col sum+count, std warning, skew warning, large
  N/low K, multi-by, empty input, dataframe API.

## [0.3.3] — 2026-10-19

### Добавлено
- `CadwDataFrame.quantile(column, q, interpolation='linear')`:
  scalar q → `float`; list q → `list[float]`.
  - `q ∈ [0, 1]`.
  - `interpolation='linear'` (pandas default; v0.3.3 only).
  - `skipna=True` (nulls не участвуют).
  - Sort: всегда ascending (квантиль инвариантен к порядку).
- `cadw.ops.quantile.quantile(batch_iter, column, q, mg, spill_dir,
  interpolation='linear')` — runtime.
- Экспорт: `cadw.quantile`, `cadw.ops.quantile`.

### Алгоритм
- Pass 1: external_sort по `column` (ascending) + side-effect count
  `n_valid` (один проход по входам, без двойного I/O).
- Positions: `idx_i = q_i*(n_valid-1)`; `lo_i = floor`, `hi_i = ceil`,
  `frac_i = idx_i - lo_i`.
- Pass 2: streaming scan sorted, dict `{pos → value}` для
  unique(`lo_i`, `hi_i`). Early exit при `pos > max(targets)`.
- Interpolation 'linear': `v_lo + frac*(v_hi - v_lo)`.

### Ограничения (v0.3.3, §11.5)
- **interpolation**: `'lower'/'higher'/'nearest'/'midpoint'` — v0.3.4.
- **skipna=False**: не поддерживается; nulls всегда пропускаются.
- **non-numeric columns**: `linear` требует numeric; для строк —
  `NotImplementedError` (см. v0.3.4 `lower`/`higher`).

### Тесты
- 20 тестов в `test_quantile.py`: median/min/max, matches numpy для
  5 q, scalar vs list return, n=1, nulls (skip, all-null), after filter,
  pandas-equivalence (с/без nulls), errors (missing/invalid q/invalid
  interp/empty list).

## [0.3.2] — 2026-10-19

### Добавлено
- `CadwDataFrame.rank` теперь поддерживает `method='max'|'last'|'average'`
  через **двухпроходный streaming** (O(1) RAM):
  - Pass 1: scan sorted_by → `(first_rank, group_count)` per tie-group
    на диск. State O(1) + per-batch accumulators (O(batch_size)).
  - Pass 2: scan sorted_by + advance по group file (sequential).
    `pos_in_group = global_pos - first_rank + 1`;
    `last_rank = first_rank + group_count - 1`;
    `max → last_rank`; `last → first_rank + count - pos`;
    `average → (first + last) / 2`.
- `VALID_METHODS = ('first', 'min', 'dense', 'max', 'last', 'average')`.
- `_VALID_STREAMABLE` / `_VALID_BUFFERABLE` — разделение методов.
- `_rank_dtype(method, pct)`: int64 для int-методов без pct; float64
  для average или pct.

### Изменено
- `compute_ranks_to_disk(sorted_path: Path, ...)` — принимает **Path**
  вместо `Iterable[RecordBatch]`, потому что bufferable-методы требуют
  двух проходов. Единственный caller — `CadwDataFrame.rank` — обновлён.

### Ограничения (v0.3.2, §11.5)
- **null values**: участвуют как значения (null == null); pandas
  `rank(skipna=True)` не поддерживается.
- **cost**: `max`/`last`/`average` = 2 passes by sorted_by + 1 pass
  by row_id (re-sort). Против 1 pass у streamable.

### Тесты
- 15 новых в `test_rank.py`: max/last/average на asc и desc, pct для
  average, dtype-проверки, after filter, pandas-equivalence для всех
  трёх, all-same для max/last/average.

## [0.3.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.rank(by, method='first'|'min'|'dense', ascending=True,
  pct=False, output_col=None, batch_size=10_000)`:
  - external_sort по `by` → потоковое вычисление rank → external_sort
    rank-пар по `row_id` → zip с исходным в original order.
  - `first`: 1-based позиция в sorted order (ties по row_id asc).
  - `min`:   минимум ранга в tie-группе.
  - `dense`: 1, 2, 3, ... для уникальных значений.
  - `pct=True`: rank / total_n.
- `cadw.ops.rank.compute_ranks_to_disk` — primitive: sorted stream → file
  (row_id, rank).
- Экспорт: `cadw.compute_ranks_to_disk`, `cadw.VALID_METHODS`.

### Принцип
- Streamable методы — O(1) state. Bufferable (`max`/`average`/`last`) —
  O(max tie group size) = O(N) для pathological данных → **не в MVP**,
  требуют двухпроходного streaming.
- Результат в original order (две сортировки: по `by` + по `row_id`).

### Ограничения (v0.3.1, §11.5)
- **method**: `first`/`min`/`dense` только. `max`/`average`/`last` — v0.3.2.
- **null values**: участвуют как значения (null == null); pandas
  `rank(skipna=True)` не поддерживается.
- **cost**: 2× external_sort + 1× count-scan. Для 10^9 строк — ~3 полных
  прохода I/O.

### Тесты
- 19 тестов в `test_rank.py`: все три метода на asc/desc, pct, edge
  cases (all-unique / all-same / ties), after filter, custom output_col,
  errors (invalid method / missing column / collision), 4 × pandas-
  equivalence, chained с groupby.

## [0.3.0] — 2026-10-19

### Добавлено (первая операция v0.3)
- `CadwDataFrame.drop_duplicates(subset=None, keep='first', num_partitions=None)`:
  HashShuffle по subset + локальный `pandas.drop_duplicates` per partition.
  - `subset=None` → dedup по всем колонкам.
  - `keep ∈ {'first', 'last', False}` — все три pandas-режима.
  - `num_partitions=None` → авто из sample (k_est на subset).
- `cadw.ops.dedup.drop_duplicates(...)` — runtime, возвращает `pa.Table`.
- Экспорт через `cadw.drop_duplicates` и `cadw.ops.drop_duplicates`.

### Принцип
- Корректность: партиции `HashShuffle` не пересекаются по subset keys →
  локальный dedup = глобальный, concat без merge.
- `row_id` сохраняется в shuffle; сортировка по row_id перед dedup даёт
  корректный keep='first'/'last' (min/max row_id per key).
- `keep=False` (pandas semantics) локально = глобально.

### Ограничения (v0.3.0, §11.5)
- **Output order**: partition-concatenation, не pandas-original order.
  Для стабильного порядка — `.sort_values(...)` после.
- **Skew**: при oversized partition → `MemoryLimitExceeded` без recursion.
  Recursion в v0.3.1 требует save-row_id-at-resplit или streaming dedup.

### Тесты
- 15 тестов в `test_drop_duplicates.py`: все keep-режимы, subset=None,
  multi-column, edge cases (empty, missing, invalid), skew reject,
  pandas-equivalence, after filter, chained with sort.

## [0.2.5] — 2026-10-19
### Исправлено (§13.4 эпизод 47)
- `hash_join`: cartesian check заменён на **точную** оценку совпадений
  `actual_matches = sum over common keys(build_count × probe_count)` через
  `groupby.size()`. O(build + probe), дёшево. Устраняет ложные reject'ы
  для PK-FK (cartesian 12.6M при реальном output 2519).
- Удалены: `max_cartesian_rows` (все слои), `c_cartesian`.
- 2 AST guard'а: `test_actual_matches_checked_before_pd_merge`,
  `test_no_cartesian_check_in_join`.



### Добавлено (финал v0.2)
- `tests/test_e2e_join.py` (3 slow):
  - `test_e2e_join_100k_x_10k`: 100k FK × 10k PK (string keys), 20 MB
    budget, inner join; pandas-equivalence; RSS-дельта < 200 MB;
    elapsed < 60 s.
  - `test_e2e_join_skew_reject`: 100k × 100k один ключ →
    `MemoryLimitExceeded` до OOM; RSS-дельта < 500 MB.
  - `test_e2e_join_left_preserve`: 100k × 5k, half unmatched → 100k
    preserved; null-mask совпадает с pandas.

### Закрыто (v0.2)
- `CadwDataFrame.merge(inner/left)` — пользовательская функция.
- `hash_join(how, max_output_rows, max_cartesian_rows)` — runtime.
- `JoinMemoryModel` + `propagate_chars("join")` — planner integration.
- OOM safety: size-check до materialize, schema-correct empty IPC,
  recursive partition split, per-partition unique_build check.
- `use_planner=True` в `merge`, sampling helpers в `cadw/core/sampling.py`.

### Тесты v0.2
- fast: 330 passed, 1 skipped.
- slow: 8 passed (5 был + 3 E2E join).
- Всего: **338 passed, 1 skipped**.

## [0.2.4] — 2026-10-19

### Исправлено (§13.4 эпизод 42)
- Модель `JoinMemoryModel` возвращает два лимита: `max_output_rows`
  (expected, из `output_budget / row_size_out`) и
  `max_cartesian_rows` (`c_cartesian × max_output_rows`, default 10×).
  Runtime `hash_join` разделяет проверки:
  - `expected_probe_output = len(probe) × avg_matches_per_probe >
    max_output_rows` → raise (per-partition skew);
  - `len(build) × len(probe) > max_cartesian_rows` → raise (extreme cartesian).
  Ранее `len(build) × len(probe) > max_output_rows` ошибочно сравнивал
  cartesian с expected → `plan_pipeline` мог сказать `"fits"`, а `merge` —
  упасть.

### Добавлено
- `JoinMemoryModel.c_cartesian` (default 10.0) — множитель для worst-case.
- `JoinParams.max_cartesian_rows`.
- `hash_join(..., max_cartesian_rows=None)`; default `10 × max_output_rows`.
- `CadwDataFrame.merge(..., num_partitions=None, use_planner=True)`:
  - `use_planner=True` (default): sample first batch каждой стороны,
    `mg.plan("join", ...)`, параметры пробрасываются в `hash_join`.
  - `use_planner=False`: старые дефолты (`num_partitions=8`,
    `max_output_rows = 0.3 × available / 256`).
  - `num_partitions` явно → override модели, но `max_output_rows` /
    `max_cartesian_rows` всё равно из модели.
- Per-partition `unique_build_keys` (pandas `.nunique`/`drop_duplicates`
  на build) — уточнённый `avg_matches_per_probe`, недоступный глобальной
  модели.

### Тесты
- 8 новых в `test_merge_planner.py`: planner default, `use_planner=False`,
  explicit `num_partitions`, early-fail план, cartesian-skew reject,
  left-planner, empty-side inner/left.

## [0.2.3] — 2026-10-19

### Добавлено
- `JoinMemoryModel` (§9.2.5): `f_join = build_partition + probe_batch +
  output_batch × (1 + c_merge)`.
  - `solve()`: `P` = max от shuffle обеих сторон; `build_side` по `how`
    (right для left join; меньший state для inner);
    `max_output_rows` из `output_budget / (row_size_out × (1+c_merge))`
    вместо ad-hoc `0.3 × available / 256`;
    `avg_matches` = upper bound `N_L × N_R / max(K_L, K_R)` (floor 1).
  - `JoinParams`: `build_side`, `how`, `num_partitions`,
    `probe_batch_size`, `build_partition_bytes`, `max_output_rows`,
    `right_chars`, `probe_row_size`, `output_row_size`, `avg_matches`.
- `propagate_chars("join")`: `N' = N_left × avg_matches`,
  `row_size' = output_row_size`, `k_est' = N'` (upper bound),
  `size_of_key_entry' = row_size' + 80`, `order = "random"`.
  Требует `JoinParams` в `params` (из `get_model("join").solve()`).
- `plan_pipeline` теперь планирует цепочки с join
  (`filter → join → groupby`, `join → join`).

### Открыто (11-B-2c-3)
- Модель считает **expected** output, runtime — **cartesian worst case**.
  `plan_pipeline` может предсказать `"fits"`, а `merge` — упасть.
  Fix: разделить `max_output_rows` (expected) и `max_cartesian_rows`
  (worst case, ~10× larger). §13.4 эпизод 42.

## [0.2.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.merge(..., how='left')`: left join, все строки self
  сохраняются; right-колонки -> NULL при отсутствии совпадения.
- `cadw.ops.join.hash_join(how='inner'|'left')` — общий runtime;
  `inner_join`, `left_join` — обёртки.
- SQL-семантика NULL:
  - inner: `_drop_null_keys` на обе стороны;
  - left: `_drop_null_keys` только на build (right); probe (left)
    сохраняется (NULL в left key -> NULL right cols).
- `_empty_result_schema` — nullable=True для всех полей (left join с
  unmatched: NaN в int/string -> null, cast safe=False).

### Ограничения v0.2.2
- `how='right'`, `how='outer'`, `left_on`/`right_on`, `on=index` — v0.3.
- `JoinMemoryModel` + `propagate_chars("join")` + E2E — 11-B-2b/2c.

## [0.2.0] — 2026-10-19

### Добавлено
- `CadwDataFrame.merge(right, on, how='inner', suffixes=('_x','_y'),
  num_partitions=8)`: Grace hash inner join.
- `cadw.ops.join.inner_join`: HashShuffle обеих сторон (один seed) →
  партиции с одинаковыми ключами → для каждой пары: build = меньшая
  партиция (`check_state`), probe батчами (`check_batch`),
  `pd.merge(how='inner', suffixes=...)`.
- SQL-семантика NULL: `_drop_null_keys` до `pd.merge` (pandas.merge
  по умолчанию матчит NaN==NaN — не то).
- Защита от cartesian-skew: `len(build) * len(probe) > max_output_rows`
  → `MemoryLimitExceeded` **до** `pd.merge`.

### Ограничения v0.2.1
- `how='inner'` только; `left`/`right`/`outer` — v0.2.2.
- `on` по имени в обеих сторонах; `left_on`/`right_on`, `on=index` —
  v0.2.2.
- `JoinMemoryModel` + `propagate_chars("join")` + E2E — 11-B-2.

## [0.1.7] — 2026-10-19

### Исправлено (must-fix §11.4)
- `_local_groupby`: dict check до `group_by`. `_read_partition_safe`
  проверял только `file_size`, но `group_by` строит hash-таблицу
  `K_i × size_of_key_entry` — для узких строк dict больше файла, OOM
  внутри pyarrow. `_estimate_distinct_keys` (product of nunique) ×
  `size_of_key_entry` проверяется до `group_by`; overflow -> in-memory
  recursion через `hash_shuffle`; `depth >= max_depth` ->
  `MemoryLimitExceeded` с file/dict/safety.

### Добавлено
- `_estimate_distinct_keys(table, by_columns) -> int` — upper bound без
  построения hash-таблицы (`pc.count_distinct` per column).
- AST guard `test_local_groupby_dict_check_before_group_by`.
- Behavior tests: `test_dict_overflow_triggers_split`,
  `test_dict_fits_no_recursion`, `test_max_depth_raises_on_dict_overflow`.

## [0.1.4] — 2026-10-18

### Исправлено (must-fix §11.4, класс «materialize-before-check»)
- `_local_groupby` материализовал всю партицию (`list(ArrowIPCBatchReader)`)
  до size-check. 02b1 добавил проверку **после** materialize — это не
  закрыло баг. Fix: `partition_path.stat().st_size` до чтения + helper
  `_read_partition_safe` (единственное место, где разрешён `list(reader)`).
- `reduce` вызывал `op.init()` до `estimate_state_size`/`check_state`.
  `CovarianceOp(d=10^5)` → `np.zeros((10^5,10^5))` = 80 GB → OOM **внутри**
  init. Fix: `estimate_state_size` + `check_state` **до** `op.init()`.

### Добавлено
- §3.10 «Check before materialize» — новый обязательный принцип.
- `tests/test_materialize_guard.py` — AST guards:
  - `test_no_list_of_whole_reader_outside_helper`
  - `test_reduce_estimate_state_before_init`
- `tests/test_oom_resistance.py` (slow):
  - 10^6 групп, 10 MB budget;
  - 90/10 skew, 10^6 строк, 10 MB;
  - `CovarianceOp(d=5000)` под 10 MB → `MemoryLimitExceeded` до `init`.

### Уроки (DECISIONS §13.4)
- Эпизод 27: «закрытие must-fix» требует теста на **порядок** операций,
  а не только на наличие вызова. 02b1 проверял, что `hash_shuffle`
  вызывается, но не проверял, что `list(reader)` **не** вызывается до.
- Эпизод 28: аудит перед новыми функциями находит унаследованные баги
  (`reduce.init()` — не был замечен при 02a).

## [0.1.3] — 2026-10-18

### Удалено
- `CadwDataFrame.partial_fit` — адаптер к `sklearn.partial_fit`. Причина:
  whitelist (`_SAFE_PARTIAL_FIT`) — политика на доверии; любая модель вне
  whitelist или модель с выросшим state OOM-ит без предупреждения от CADW.
  См. DECISIONS.md §3.9, §13.4 эпизод 26.
- `_SAFE_PARTIAL_FIT` whitelist.
- `tests/test_ml_integration.py` (11 + 1 slow).
- `tests/test_sklearn_extras.py` (4 + 1 slow).
- sklearn-секция из `tests/test_api_availability.py` (7 assertions).
- `scikit-learn` из `[dependency-groups]` (dev).
- ML-секции из README, заменены на «ML и внешние библиотеки» с policy.

### Добавлено в DECISIONS.md
- §3.9: «Никаких ML-адаптеров» (не только `fit`, но и `partial_fit`).
- §11.5: расширенная «Политика ML» (3 класса: экспорт / адаптеры запрещены
  / собственные реализации v0.3).
- §12.5.1: план v0.3 — GBDT, Linear, k-means, PCA, NB на примитивах (~8–11
  сессий на весь ML-слой).
- §13.4 эпизод 26: whitelist `partial_fit` — политика на доверии, не
  гарантия.

### Единственный ML-интерфейс CADW
- `to_batches(format='arrow'|'pandas'|'numpy'|'torch')` — экспорт батчей.
  Ответственность за OOM в сторонней модели — на пользователе.

## [0.1.2] — 2026-10-18

### Удалено
- `CadwDataFrame.to_lightgbm` — адаптер к LightGBM. Причина: `lgb.train`
  требует O(N)-state (все градиенты в RAM), OOM происходит внутри
  библиотеки, `MemoryGuard` не спасает. Противоречит §3.9 «no adapters to
  O(N)-algorithms». См. DECISIONS.md §13.4 эпизод 25.
- `tests/test_lightgbm.py` (11 + 1 slow тестов).
- `lightgbm` из `[dependency-groups]` (dev).
- LightGBM-секции из README, DECISIONS §4.1/§11.5/§12.5/§13.3.

### Изменено
- `CadwDataFrame.partial_fit`: добавлен whitelist `_SAFE_PARTIAL_FIT`
  (SGD, StandardScaler, MiniBatchKMeans, IncrementalPCA, NB, MLP).
  Модели вне whitelist → `RuntimeWarning` с указанием на §11.5.
  Docstring обновлён: контракт безопасности (модель должна иметь
  O(1)-state w.r.t. N, это ответственность модели, не CADW).
- `tests/test_api_availability.py`: убрана LightGBM-секция.

### Добавлено в DECISIONS.md
- §3.9: «Никаких адаптеров к O(N)-алгоритмам».
- §11.5: блок «Политика ML» (3 класса: incremental / batch-only /
  own implementation).
- §11.4: обнаружена регрессия `_local_groupby` (materialize до
  `check_state`) — вынесена как must-fix.
- §13.4 эпизод 25: почему адаптер к LightGBM — ошибка.

### Планы
- v0.3: свой GBDT (level-wise, аналог `sklearn.HistGradientBoosting`) на
  примитивах: `HistogramOp` (binning) + `HashShuffle` (узел,фича,бин) +
  `Reduce` (sum_grad/sum_hess/count).

## [0.1.0] — 2026-10-18

Первый production-ready каркас: три примитива out-of-core с безусловной
гарантией памяти, pandas-подобный фасад, ML-интеграция (sklearn + LightGBM),
E2E на 200k строк / 20 MB budget.

### Добавлено — storage (уровни 0–1)
- `ParquetBatchReader/Writer`, `ArrowIPCBatchReader/Writer`.
- `Dataset` (03a): per-column Arrow IPC, синхронные чанки, `add_column`
  (валидирует boundaries, атомарная запись `.tmp.arrow` → `rename`),
  `iter_batches(batch_size=None|N, columns=...)` с re-chunk, persistence в
  `dataset_meta.json` (schema через `Schema.serialize()`).

### Добавлено — примитивы (уровень 3)
- `Reduce` (02a): `ReduceOp`/`reduce`, `CountOp`, `SumOp`, `MinMaxOp`,
  `MeanOp`, `StdOp` (Welford), `CovarianceOp` (d×d), `HistogramOp`
  (binning LightGBM). Инвариант «не зависит от `batch_size`», «не пишет
  на диск».
- `HashShuffle` (02b): `compute_hashes` (`xxhash.xxh64` + `repr(v).encode`),
  `_mod_p` (NumPy `uint64` для wrap-around), `row_id` через `order_np`
  (а не позицию в отсортированном батче), `estimate_num_partitions`,
  skew-изоляция (`_detect_heavy_keys` >5%, `_groupby_with_skew`).
- `Sort` (02b): `external_sort`, `compute_fan_in`, `_Key` (null_rank
  + per-column asc/desc), k-way merge, многоуровневый merge, `row_id`
  в каждом run'е для стабильности.

### Добавлено — операции (уровень 4)
- `groupby_aggregate` (02a/02b): `HashShuffle + LocalGroupBy + Concat`,
  авто-P через sample, рекурсивное партиционирование (`MAX_DEPTH=5`),
  skew-изоляция (тяжёлые потоково, лёгкие shuffle), single-pass.
  Корректен при `K > памяти` (`test_groupby_many_groups`: 1000 групп,
  200 KB budget) и при экстремальном skew (90/10, one-hot).

### Добавлено — доступ (уровень 2)
- `RowFilter` (03c): `PredicateFilter`, `SliceFilter`, `AndFilter`,
  `OrFilter`, `NotFilter`. `& | ~` для `MaskFilter`; slice+mask — через
  chaining.

### Добавлено — фасад (уровень 5)
- `CadwDataFrame` (03b/03c): `from_parquet`, `from_dataset`,
  `columns`, `shape` (upper bound), `count()` (точное), `head(n)`,
  `to_batches(batch_size, format='arrow'|'pandas'|'numpy'|'torch')`,
  `__getitem__(slice|list[str]|str|RowFilter)`, `groupby().agg()`,
  `unique()`, `sort_values()`, `write_parquet()`,
  `partial_fit(model, X_cols, y_col=None, **kwargs)`,
  `to_lightgbm(X_cols, y_col, max_rows=N, categorical_cols=None, **params)`.
- Ленивость: `df[df['val'] > 15][1:5]` не материализует данные до
  `to_batches`/`write_parquet`/`groupby`/`sort_values`/`head`.
- AST guard-тесты: `test_facade_has_no_as_py`, `test_facade_has_no_row_loop_ast`.

### Добавлено — MemoryGuard (уровень 2)
- Байтовый контракт: `available_bytes`, `get_batch_rows`,
  `check_batch(extra_factor)`, `check_state`, `check_partition_buffers`,
  `recommended_partition_buffer_bytes`, `estimate_state_size`,
  `estimate_unique_keys`, `reserve/release`, `log_step`.

### Добавлено — ML (уровень 6)
- sklearn: `SGDClassifier` (`classes=[0,1]` на первом вызове),
  `StandardScaler`, `MiniBatchKMeans`, `IncrementalPCA` (без `n_features`
  в `partial_fit` — выводится из `X.shape[1]`).
- LightGBM: `to_lightgbm(max_rows=N)` — честный downsampling
  (LightGBM требует весь датасет в RAM).
- E2E: sklearn pipeline (50k строк, 10 MB budget), LightGBM pipeline
  (50k строк, downsampling до 20k).

### Добавлено — E2E и guards
- E2E (04): filter → groupby → sort → parquet на 200k строк, 20 MB budget,
  pandas-equivalence.
- Architecture guards: `test_no_legacy_package`, `test_legacy_tests_removed`,
  `test_main_code_does_not_reference_legacy`.

### Удалено
- Legacy `cadw/_legacy/` и `tests/_legacy/` (05): `SpillManager`,
  `UniqueCollector`, `GroupByAggregator` — заменены примитивами и фасадом.
- `cadw/dataframe.py` (старая заготовка) — переписан с нуля.

### Ограничения (документированы в DECISIONS.md §11.5)
- `to_lightgbm` не out-of-core: LightGBM требует весь датасет в RAM.
- `PCA` (batch-only) не имеет `partial_fit`; для batch-only — `vstack`.
- `SGDClassifier.partial_fit` требует `classes=[...]` на первом вызове.
- Категориальные признаки для LightGBM — int-encoded, не строки.
- `shape` на отфильтрованном `CadwDataFrame` — upper bound; точное — `.count()`.
- `nulls_first` в `Sort` — глобальный флаг, не per-column.
- `df[-5:]` (отрицательный slice) — не поддержано.
- `compute_hashes` — Python-цикл (векторизация отложена).
- `CadwIndex`/`loc`/`iloc`, `join`/`merge`, `drop_duplicates`, `rank`,
  `rolling` — не реализованы (v0.2).

### Тесты
- **213 fast + 5 slow = 218 passed, 1 skipped** (torch опционален).
- Покрытие: MemoryGuard (25), Reduce (33), HashShuffle (13), Sort (10),
  groupby + must-fix (10 + 28), Dataset (13), CadwDataFrame (17),
  RowFilter (25), ML (sklearn + LightGBM, 27), E2E (5 slow), guards (3).

[0.1.0]: https://example.invalid/cadw/compare/v0.0.0...v0.1.0
