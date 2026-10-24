# v0.9.35+ — Algorithm progress (25 items)

| ID | Version | Status | Tests | Bench (time / RSS / state) | Reference |
|---|---|---|---|---|---|
| D.1.4 | v0.9.35 | done | 11 | 1M rows: 0.098 s / 0.6 MB / 1 280 B | NumPy brute-force recount + hand fixture (no fairlearn install) |
| D.1.2 | v0.9.35 | done | 11 | 1M rows: 3.174 s / 23.7 MB / 1 114 112 B | NumPy brute-force over stored sample; transform vs closed-form |
| D.2.1 | v0.9.35 | done | 10 | 400k rows: 33.51 s / 69.6 MB / 56 192 B | sklearn GradientBoostingRegressor (36.19 s, NDCG 0.974 vs 0.862) |
| D.2.2 | v0.9.36 | done | 10 | 200k interactions: 31.29 s / 64.0 MB / 1 792 000 B | closed-form per-user solve; implicit ALS formulation (no `implicit` install) |
| D.2.3 | v0.9.36 | done | 10 | 446k interactions: 96.93 s / 79.4 MB / 166 784 B | vs plain ALS: +0.0799 Pearson corr (same data, same sweeps) |
| D.3.1 | v0.9.36 | done | 10 | 50k rows: 0.70 s / 36.7 MB / 464 864 B | lifelines: 4.18 s, coef maxdiff 1.58e-06, c-index 0.7244 both |
| D.3.2 | v0.9.37 | pending | — | — | — |
| D.3.3 | v0.9.37 | pending | — | — | — |
| D.4.1 | v0.9.37 | pending | — | — | — |
| D.4.2 | v0.9.38 | pending | — | — | — |
| D.4.3 | v0.9.38 | pending | — | — | — |
| D.4.4 | v0.9.38 | pending | — | — | — |
| D.4.5 | v0.9.38 | pending | — | — | — |
| D.4.6 | v0.9.39 | pending | — | — | — |
| D.5.1 | v0.9.39 | pending | — | — | — |
| D.5.2 | v0.9.39 | pending | — | — | — |
| D.5.3 | v0.9.39 | pending | — | — | — |
| D.5.4 | v0.9.40 | pending | — | — | — |
| D.6.1 | v0.9.40 | pending | — | — | — |
| D.6.2 | v0.9.40 | pending | — | — | — |
| D.6.4 | v0.9.40 | pending | — | — | — |
| D.6.5 | v0.9.40 | pending | — | — | — |
| D.6.6 | v0.9.41 | pending | — | — | — |
| D.6.7 | v0.9.41 | pending | — | — | — |
| D.6.9 | v0.9.41 | pending | — | — | — |

Done: 6/25. Fast suite 2638 passed, 2 skipped. Slow 25 passed.
