# v0.9.35+ — Algorithm progress (25 items)

| ID | Version | Status | Tests | Bench (time / RSS / state) | Reference |
|---|---|---|---|---|---|
| D.1.4 | v0.9.35 | done | 11 | 1M rows: 0.098 s / 0.6 MB / 1 280 B | NumPy brute-force recount + hand fixture (no fairlearn install) |
| D.1.2 | v0.9.35 | done | 11 | 1M rows: 3.174 s / 23.7 MB / 1 114 112 B | NumPy brute-force over stored sample; transform vs closed-form |
| D.2.1 | v0.9.35 | done | 10 | 400k rows: 33.51 s / 69.6 MB / 56 192 B | sklearn GradientBoostingRegressor (36.19 s, NDCG 0.974 vs 0.862) |
| D.2.2 | v0.9.36 | done | 10 | 200k interactions: 31.29 s / 64.0 MB / 1 792 000 B | closed-form per-user solve; implicit ALS formulation (no `implicit` install) |
| D.2.3 | v0.9.36 | done | 10 | 446k interactions: 96.93 s / 79.4 MB / 166 784 B | vs plain ALS: +0.0799 Pearson corr (same data, same sweeps) |
| D.3.1 | v0.9.36 | done | 10 | 50k rows: 0.70 s / 36.7 MB / 464 864 B | lifelines: 4.18 s, coef maxdiff 1.58e-06, c-index 0.7244 both |
| D.3.2 | v0.9.37 | done | 9 | 100k rows: 4.25 s / 20.2 MB / 1 210 624 B | vs true ATE 1.0: T 1.0204, S 0.9747; overlap diagnostics always returned |
| D.3.3 | v0.9.37 | done | 10 | 100k rows: 9.43 s / 12.7 MB / 820 224 B | Qini vs random baseline; 100k Qini 0.00019, 200k 0.00971 (reported honestly) |
| D.4.1 | v0.9.37 | done | 11 | 1000x500: 3.13 s / 15.3 MB / 7 744 B | PPV vs direct convolution (closed form); ~320 series/s |
| D.4.2 | v0.9.38 | done | 11 | 20k rows: 0.101 s / 1.9 MB / 256 B | ruptures.Pelt(l2): 73.572 s, same cps within ties |
| D.4.3 | v0.9.38 | done | 11 | 50k rows: 17.15 s / 1.4 MB / 800 352 B | statsmodels ETS: MASE 0.298 vs 0.338 |
| D.4.4 | v0.9.38 | done | 9 | 100k rows: 0.92 s / 6.5 MB / 1 600 256 B | statsmodels Theta: MASE 2.18 vs 1.99 |
| D.4.5 | v0.9.38 | done | 11 | 50k x k=4 p=2: 0.039 s / 12.3 MB / 1 600 544 B | statsmodels VAR: max fc diff 2.8e-17 |
| D.4.6 | v0.9.39 | done | 10 | 100k rows: 5.92 s / 10.7 MB / 1 824 B | sklearn CCV (sigmoid/isotonic): Brier within 1.3x, ECE 0.0038 |
| D.5.1 | v0.9.39 | done | 10 | 200k rows: 2.09 s / 30.5 MB / 296 B | statsmodels OrderedModel: beta maxdiff < 1e-2 |
| D.5.2 | v0.9.39 | done | 8 | 100k rows: 11.26 s / 39.5 MB / 12 800 B | sklearn ClassifierChain 0.702 vs ours 0.664 (94%) |
| D.5.3 | v0.9.39 | done | 12 | 500k rows: km 1.98 s / 63 MB, gmm 3.11 s / 67 MB | sklearn MiniBatchKMeans inertia within 1.15x; GMM loglik within 0.15 |
| D.5.4 | v0.9.40 | done | 9 | 50k rows: 3.39 s / 3.4 MB / 4 176 B | sklearn Birch: purity 1.0 (up to label perm) |
| D.6.1 | v0.9.40 | done | 10 | 10k rows: MI 1.51 s, variance 0.003 s, Boruta 8.36 s | sklearn mutual_info: within atol=0.03 |
| D.6.2 | v0.9.40 | done | 9 | 1M rows: 1.28 s / 156.2 MB / 256 B | scipy KS/chi2/Welch/MW exact; CUPED theta 0.80 |
| D.6.4 | v0.9.40 | done | 8 | 150k x 20: 0.22 s / 70.8 MB / 920 B | sklearn PCA: singular values ~1e-6 |
| D.6.5 | v0.9.40 | done | 9 | 1M rows: 2.45 s / 107.6 MB / 256 B | analytic haversine exact (1e-9) |
| D.6.6 | v0.9.41 | done | 10 | 50k rows: 104.26 s / 1319.7 MB / 896 B | sklearn LDA: NMI >= 0.9x (both 1.0); ppl 4.131 |
| D.6.7 | v0.9.41 | done | 10 | 150k rows: 0.04 s / 51.1 MB / 3 136 B | analytic ridge: theta maxdiff < 1e-9 |
| D.6.9 | v0.9.41 | done | 11 | 500k rows: 1.28 s / 229.6 MB / 768 B | sklearn TfidfVectorizer: atol 1e-9 exact |

Done: 25/25. Phase D complete. Fast suite 2751 passed, 2 skipped. Slow 25 passed.
