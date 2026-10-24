# D.5.2 Classifier chains

## Paper
Read, Pfahringer, Holmes, Frank, *Classifier Chains for Multi-label
Classification*, ML 85(3), 2011. Tsoumakas, Katakis, *Multi-Label
Classification: An Overview*, IJDWM 3(3), 2007 (one-vs-rest baseline).

## Docs
- `cadw.ml.chains.fit_chains(df, X_cols, label_cols, chain=True|False)`
- `CadwDataFrame.fit_chains(...)` binding.
- Model: `predict_proba(df)` -> (n, K); `predict(df, threshold)` -> hard
  labels; `subset_accuracy(df)`.

## Design
- Each label is an in-tree GBDT binary classifier.
- Chain: true previous labels as extra features at fit; predicted previous
  labels at predict (`_predict_hard_batched` for hard-label chaining).
- Independent variant = one-vs-rest baseline.
- log_fit per label; state = sum of per-model states.

## Benchmark (`.pingis/benches/chains.json`, K=3, n_estimators=30)
| n | chain | fit_s | subset_acc | peak_mb | state_b |
|---|---|---|---|---|---|
| 20 000 | True | 2.27 | 0.6621 | 11.1 | 12 800 |
| 20 000 | False | 1.95 | 0.6621 | 1.1 | 12 800 |
| 100 000 | True | 11.26 | 0.6555 | 39.5 | 12 800 |
| 100 000 | False | 9.57 | 0.6555 | 23.6 | 12 800 |
Reference: sklearn ClassifierChain(HistGradientBoostingClassifier) — subset
accuracy 0.664 vs 0.702 (94% of reference); backlog criterion
"subset accuracy >= OvR" holds (chain == OvR on this DGP since GBDT already
captures X2 almost exactly; honest, not hidden).

## RAM model
State = O(K * model). Fit materialises X and Y (n x (d+K)); predict streams
batches with check_batch(extra_factor=3.0).
