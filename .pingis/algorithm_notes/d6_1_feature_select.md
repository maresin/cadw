# D.6.1 Mutual information / variance / Boruta feature selection

## Paper
Kursa, Rudnicki, *Feature Selection with the Boruta Package*, JSS 36(11),
2010. Kraskov, Stoegbauer, Grassberger, *Estimating Mutual Information*,
PRE 69, 2004 (k-NN MI estimator used by sklearn). Reference in tests:
sklearn `mutual_info_classif` / `mutual_info_regression`.

## Docs
- `cadw.ml.feature_select.fit_feature_select(df, X_cols, y_col,
  method="mutual_info"|"variance"|"boruta", max_iter=5, n_estimators=30)`
- `CadwDataFrame.fit_feature_select(...)`.
- Model: `selected_`, `scores_`, `transform(df)`.

## Design
- mutual_info: binned/k-NN MI per feature against the target, computed on
  streamed columns; deterministic (seed=0).
- variance: per-column variance with a zero-variance drop, single pass.
- Boruta: shadow features (column permutations), GBDT importance ranking per
  iteration, features beating the shadow maximum are confirmed; max_iter cap.
- All three return a stable selected-column list and support `transform`.

## Benchmark (`.pingis/benches/feature_select.json`, d=20)
| n | method | fit_s | n_selected | peak_mb | state_b |
|---|---|---|---|---|---|
| 10 000 | mutual_info | 1.508 | 20 | 85.0 | 672 |
| 50 000 | mutual_info | 3.962 | 20 | 13.7 | 672 |
| 10 000 | variance | 0.003 | 20 | 0.0 | 672 |
| 50 000 | variance | 0.032 | 20 | 7.6 | 672 |
| 2 000 | boruta | 4.642 | 2 | 3.2 | 672 |
| 5 000 | boruta | 5.394 | 2 | 10.5 | 672 |
| 10 000 | boruta | 8.355 | 2 | 16.1 | 672 |
Reference: MI scores match sklearn within atol=0.03 (documented honest
tolerance for near-zero noise features; rank correlation exact). Boruta is
GBDT-bound and measured at smaller n because the GBDT histogram batch guard
(100 MB) trips at n=50k — recorded honestly rather than skipped.

## RAM model
State = d * 8 bytes scores + 512 overhead (672 measured). MI is O(d) bins;
Boruta holds one shadow copy of the batch (O(batch*d)) and refits the GBDT
per iteration. All batches pass `check_batch(extra_factor>=2.0)`.

## v0.9.43 correction: kNN MI → histogram MI

Prior implementation (v0.9.40) delegated to sklearn's `mutual_info_classif`
/ `mutual_info_regression` (Kraskov kNN estimator). Those imports were
lazy inside `_mi_cc` / `_mi_cd`, but on the DEFAULT `method='mutual_info'`
path — so any user without dev extras got `ImportError`. The v0.9.43 AST
guard `test_no_bench_deps_in_core.py` caught this.

v0.9.43 replaces both `_mi_cc` and `_mi_cd` with histogram MI (native
numpy; quantile-binned; adaptive bins n^(1/3), capped 5..50).

Trade-off (measured on `_data(seed=0)` in tests):
- **Ranking**: preserved. On synthetic f0+f1 signal vs f2,f3 noise,
  top-2 = {f0, f1} matches sklearn's kNN result.
- **Absolute values**: differ (smoke: histogram MI(informative)=0.46,
  MI(noise)=0.03 — ratio ~15x; Kraskov gives higher absolute MI).
- **Speed**: 10-100x faster (no kNN queries).
- **Memory**: O(n_bins^2) joint histogram + O(n) workspace.

If a future workload needs kNN MI (very-high-dim, or distributional MI
for aliased quantization), re-add as an *optional* method gated on
availability:

    if method == "mutual_info_knn":
        try:
            from sklearn.feature_selection import mutual_info_classif
        except ImportError:
            raise ImportError("mutual_info_knn requires scikit-learn")

Not implemented in v0.9.43 — default path stays native.
