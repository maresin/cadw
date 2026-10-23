# D.1.1 Isolation Forest (IForest) — design

**Status:** implemented v0.9.31 (Phase D.1.1, P0/S)
**Reference:** F. T. Liu, K. M. Ting, Z.-H. Zhou. *Isolation Forest*. ICDM 2008. DOI:10.1109/ICDM.2008.17. Docs: sklearn.ensemble.IsolationForest.

## Why

Unsupervised anomaly detection. The IForest paper shows that isolation trees
need only a **small subsample** (default 256) of the data — anomalies are
"few and different" and are isolated in shallow partitions. This is a
perfect fit for the CADW sample+stream paradigm: the heavy state is
`O(n_trees × subsample_size)`, independent of N.

## Algorithm

1. **Subsample.** Stream the full dataset once, draw a uniform reservoir
   sample of size `S = min(subsample_size, N)`. `O(S)` state.
2. **Fit.** For each of `n_trees` trees:
   - Draw a bootstrap subsample of size `min(256, S)` from the reservoir.
   - Recursively split by a random feature (from `X_cols`) and a random
     threshold between min/max of that feature on the current node.
   - Stop when: node has ≤1 point OR depth ≥ max_depth.
   - `max_depth = ceil(log2(psi))` where `psi = len(subsample)` — the
     standard truncated-depth rule from the paper.
3. **Score (streaming).** For each row `x`:
   - For each tree: traverse from root, count path length `h` (add
     `c(size)` if terminating at an internal node with >1 points).
   - `E[h] = mean(path_lengths)`; `s(x) = 2^(-E[h]/c(S))` where
     `c(n) = 2·H(n-1) - 2·(n-1)/n` (harmonic approximation).
   - `s(x) ∈ (0, 1)`; higher = more anomalous.
4. **Predict.** `score(df) -> pa.Table[id, score]` per-batch. `predict(df,
   threshold=0.5)` → binary labels using `sklearn`-equivalent convention
   (sklearn uses `offset_` from `contamination`, we expose `threshold`).

## Complexity

| Phase | Time | RAM (state) |
|---|---|---|
| fit | O(S · d · n_trees · log S) | O(n_trees · subsample_size) nodes |
| score | O(N · n_trees · max_depth) | O(batch) + O(1) |

State size: `n_trees=100, subsample_size=256, max_depth≈8` → ~100·(2·256)
≈ 50k nodes × ~64 B ≈ 3.2 MB. Independent of N, `check_state`-guarded.

## API

```python
model = fit_isolation_forest(df, X_cols,
                             n_trees=100, subsample_size=256,
                             max_samples=256, random_state=42,
                             contamination=0.1)
model.score(df)  # -> pa.Table["score"] with per-row anomaly scores (0..1)
model.predict(df)  # -> pa.Table["is_outlier"] with binary labels
model.score_samples(df)  # -> negative mean path length (sklearn decision_function-ish)
```

## Edge cases

- `N < subsample_size` → use all rows.
- 1 row → score = 0.5 (undefined path length).
- Constant feature → tree depth 0; path length defined as `c(n)`.
- Null values in `X_cols` → row is excluded from fit; scored as NaN.
- Determinism: numpy RandomState seeded by `random_state`.

## Not replaced

No algorithm on the roadmap is replaced by IForest — this is a new addition.

## Reference comparison

Validated against `sklearn.ensemble.IsolationForest` on synthetic data:
- Rank correlation (Spearman) of scores on 10% outliers should be ≥ 0.7.
- AUROC on labelled anomaly data ≥ 0.85.

## Test plan (8+ tests)

1. Synthetic 1 anomaly in 100 normals — highest score on anomaly.
2. Ref vs sklearn: Spearman ≥ 0.7 on a 1000-row set.
3. Chunked (chunk=1) vs chunk=1000 — same result.
4. Empty input → empty model or raises cleanly.
5. Single row → 0.5 score.
6. Constant feature → no crash, score finite.
7. Determinism: same seed → same scores.
8. RAM: peak delta ≤ 2× predicted state.
9. `predict(threshold)` labels count correlates with contamination.
10. Nulls in features → skipped (not crash).
