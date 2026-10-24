# D.4.6 Probability calibration (Platt / isotonic)

## Paper
Platt, *Probabilistic Outputs for SVMs*, Advances in Large Margin Classifiers,
2000 (Platt scaling). Zadrozny, Elkan, *Transforming Classifier Scores into
Accurate Multiclass Probability Estimates*, KDD 2002 (isotonic).
Niculescu-Mizil, Caruana, *Predicting Good Probabilities With Supervised
Learning*, ICML 2005 (Brier / ECE).

## Docs
- `cadw.ml.calibration.fit_calibration(df, X_cols, y_col, method="platt"|"isotonic")`
- `CadwDataFrame.fit_calibration(...)` binding.
- Model: `predict_proba(df)`, `predict(df, threshold)`, `brier(df)`, `ece(df)`.
- Metrics: `brier_score(y, p)`, `expected_calibration_error(y, p, n_bins)`.

## Design
- Wraps the in-tree GBDT binary classifier; calibration fit on the same data.
- Platt: p = sigmoid(A*score + B), Newton IRLS, A,B scalar.
- Isotonic: equal-count binning into <=256 points + PAVA; piecewise-linear
  interpolation at predict time.
- State O(1) for Platt (A,B) / O(bins) for isotonic.

## Benchmark (`.pingis/benches/calibration.json`, n_estimators=50)
| n | method | fit_s | Brier | ECE | peak_mb | state_b |
|---|---|---|---|---|---|---|
| 20 000 | platt | 1.23 | 0.1677 | 0.0227 | 7.6 | 512 |
| 20 000 | isotonic | 1.31 | 0.1669 | 0.0067 | 2.0 | 1 456 |
| 100 000 | platt | 6.31 | 0.1671 | 0.0231 | 14.5 | 512 |
| 100 000 | isotonic | 5.92 | 0.1663 | 0.0038 | 10.7 | 1 824 |
Reference: sklearn CalibratedClassifierCV(sigmoid/isotonic, cv=3) — Brier
within 1.3x of reference on the same data.

## RAM model
O(1) state for Platt; O(n_bins) for isotonic. Fit streams mini-batches through
check_batch(extra_factor=3.0); scores collected once into O(n) numpy vector.
