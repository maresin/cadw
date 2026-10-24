# D.6.2 Drift detection (KS/PSI/chi2/Welch/MW) + CUPED A/B

## Paper
Gama et al., *A Survey on Concept Drift Adaptation*, ACM CSUR 46(4), 2014.
Sculley et al., *Detecting Adversarial and Drift in ML*, KDD 2018 (PSI).
Deng, Xu, Kohavi, Walker, *Improving the Sensitivity of Online Controlled
Experiments with CUPED*, WSDM 2013. Reference in tests: scipy.stats
`ks_2samp`, `chi2_contingency`, `ttest_ind`, `mannwhitneyu`.

## Docs
- `cadw.ml.drift.ks_2samp(a,b)`, `psi(expected,actual,n_bins=10)`,
  `chi2_categorical(a,b)`, `welch_t(a,b)`, `mann_whitney(a,b)`,
  `cuped(y,x,t)`.
- `fit_drift(df, col, reference_col, method="ks", n_bins=10)` -> `DriftABModel`
- `fit_ab_test(df, outcome_col, treatment_col, covariate_col=None)`.
- `CadwDataFrame.fit_drift(...)` / `fit_ab_test(...)`.

## Design
- Pure numpy/scipy statistic kernels; no point retention.
- `DriftABModel` streams both columns' histograms/quantiles and computes the
  chosen statistic; CUPED estimates theta = cov(y,x)/var(x) and reports the
  variance-reduced treatment effect.

## Benchmark (`.pingis/benches/drift.json`)
| n | total_s | ks_p | psi | chi2_p | cuped_theta | peak_mb | state_b |
|---|---|---|---|---|---|---|---|
| 10 000 | 1.244 | 4.4e-82 | 0.108 | 0.390 | 0.798 | 58.5 | 256 |
| 100 000 | 0.088 | 0.0 | 0.089 | 0.639 | 0.800 | 15.3 | 256 |
| 1 000 000 | 1.279 | 0.0 | 0.086 | 0.761 | 0.800 | 156.2 | 256 |
Reference: KS/chi2/Welch/MW match scipy exactly (tolerance 1e-9 in tests);
CUPED theta converges to the true 0.8 correlation. Peak RSS at n=1M is
156 MB, well under budget.

## RAM model
State = O(#bins + 1) floats (256 bytes measured). Streaming histogram and
sufficient statistics only; every batch passes
`check_batch(extra_factor>=2.0)`.
