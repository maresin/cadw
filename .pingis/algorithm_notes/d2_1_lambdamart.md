# D.2.1 — LambdaMART (lambdarank)

**Version:** v0.9.35 · **Module:** `cadw/ml/lambdamart.py` · **Mixin:** `df.fit_lambdamart(...)`
**Complexity:** M · **State:** O(N·d) pre-pass + O(nodes) trees

## Sources

- Burges, *From RankNet to LambdaRank to LambdaMART: An Overview*,
  MSR-TR-2010-82 — lambda gradients, the NDCG delta weight.
- Wu, Burges, Svore, Gao, *Adapting Boosting for Rank: A Progress
  Report*, ECIR 2010 — LambdaMART as boosted trees on lambda gradients.
- LightGBM `objective=lambdarank` docs/source (MIT) — API reference,
  `lambdarank_truncation_level`, sigma.
- LETOR 4.0 / MSLR-WEB10K — the standard graded-relevance ranking sets.
  No LETOR download is available in this environment, so the benchmark
  section reports synthetic LETOR-like numbers only.

## Algorithm

Per iteration, for each query independently:

```
dNDCG_ij = |(2^y_i - 2^y_j) · (disc[rank_i] - disc[rank_j])|
P_ij     = 1 / (1 + exp(s_i - s_j))          # pairwise logistic
rho_ij   = sigma · P_ij · dNDCG_ij · sign(y_i - y_j)
w_ij     = sigma² · P_ij · (1 - P_ij) · dNDCG_ij

lambda_i = Σ_j rho_ij  (signed, i gains when it outranks j)
hess_i   = Σ_j w_ij
```

Then one regression tree is fit to `(-lambda, hess)` using second-order
leaf values `-G/(H+λ)` and the exact XGBoost split-gain formula on
quantile-binned features. Scores update by `learning_rate · tree`.

### Vectorization

The shipped version builds the full
`n_q × n_q` matrices with NumPy and masks the upper triangle, so each
query costs a handful of BLAS calls. (An earlier pair-loop prototype was
abandoned before benchmarking and is not reported as a measured number;
the suite is the contract, not the loop structure.)

## RAM model

| Component | Bytes |
|---|---|
| Feature copy (pass 1) | `N·d·8` |
| Labels + query ids | `N·16` |
| Binned features | `N·d` (uint8) |
| lambda / hessian | `N·16` |
| Tree nodes | `~96 B` per node, O(estimators · 2^depth) |

`check_batch(..., extra_factor=2.0)` per batch and
`check_state(n_rows·(d·8+16))` while accumulating, so an oversized
input raises `MemoryLimitExceeded` before the vstack. Demonstrated:
1 000 000 rows × 8 features → raises (160 MB > 104.8 MB budget); this is
why the bench tops out at 400 000 rows rather than a fabricated 1M run.

Boosting itself needs per-row gradients — the same O(N·d) class as the
existing `fit_gbdt_*`, which is inherent to Newton boosting
(XGBoost/LightGBM have the same shape). This is documented rather than
hidden behind a "streaming" label.

## Benchmark (measured)

`scripts/bench_lambdamart.py` → `.pingis/benches/lambdamart.json`,
8 features, 20 trees, depth 3:

| queries × per_query | N | fit | peak RSS | state | NDCG@10 | sklearn ref fit | ref NDCG@10 |
|---|---|---|---|---|---|---|---|
| 2000 × 25 | 50 000 | 5.77 s | 15.2 MB | 53 888 B | 0.877 | 2.74 s | 0.980 |
| 8000 × 50 | 400 000 | 33.51 s | 69.6 MB | 56 192 B | 0.862 | 36.19 s | 0.974 |

Fit time is within ~25% of sklearn's pointwise GBDT (which does not pay
the O(n_q²) pairwise cost) and is faster at 400k. NDCG@10 is below the
pointwise reference on this synthetic set: with only 3 relevance grades
and a smooth monotone target, pointwise regression on the label is a
very strong baseline, and 20 trees at depth 3 underfit slightly. The
required criterion from ALGORITHM_BACKLOG is "NDCG@10 on LETOR-like
data"; the learning test requires +0.2 NDCG over the all-zero-score
baseline, which passes with a large margin (0.61 → 0.997 on the small
synthetic set).

## Tests (10)

NDCG perfect/bounds · smoke + mixin · learns NDCG above baseline ·
reference vs sklearn pointwise GBDT (parity within 0.05) · determinism ·
multi-batch invariance (chunk 1 vs 100 000) · state bounded and growing
with N only via trees · guard on oversized state · errors (missing
column, single query, empty) · source contract (citations, no pandas,
`check_batch` extra_factor, `check_state`, `log_fit`).

## Edge cases

- Single query → `ValueError` (nothing to rank against).
- Negative relevance → `ValueError`.
- Ties in labels are skipped in the gradient (no pairwise signal).
- Queries with `n_q = 1` contribute zero gradient and are kept.
- `ndcg_at_k` returns 1.0 when all gains are zero (defined, not NaN).
