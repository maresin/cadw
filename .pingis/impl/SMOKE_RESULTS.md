# Reference impl smoke results

Peak RSS measured by `resource.getrusage(RUSAGE_SELF).ru_maxrss` in each impl subprocess (KiB on Linux, converted to MiB). Runtime measured by the runner (wall time of the subprocess).

| Section | Impl | Runtime (s) | Peak RSS (MiB) | Correctness check |
|---|---|---|---|---|
| M | bandits.py | 0.61 | 115.5 | EG/UCB1/TS q_ monotone increasing in true_p; best arm = last |
| N | kernel_density.py | 9.22 | 274.0 | n_centers=2048, n_seen=100000; log_lik finite |
| I | kaplan_meier.py | 0.98 | 135.0 | S(1)~0.93, S(10)~0.52, median ~10.7 |
| I | weibull_aft.py | 0.49 | 116.2 | n_events=75033; beta finite; sigma>0 (plumbing-only) |
| J | complement_nb.py | 0.59 | 115.2 | C=4 V=200; feature_log_prob finite |
| J | nmf.py | 0.93 | 112.6 | k=8 V=128; H>=0; recon_err finite |
| O | reservoir.py | 0.64 | 120.7 | K=5000 d=16; mean~0 std~1 |
| O | stratified_split.py | 0.53 | 120.5 | fold_counts balanced within +-1 |
| O | one_hot_encoder.py | 0.53 | 117.1 | c0=12, c1=8 categories; transform rows=5000 |

Total: 9/9 passed.