# PATCH PROPOSAL 2: cadw/ml/bocpd.py — rolling R_history
# Status: proposed (not applied).
#
# Current: R_history = np.zeros((n, max_run + 1)); R_history[t] = R each step.
# Memory = 8*n*(max_run+1) bytes; n=1M, max_run=200 -> 1.6 GB.
#
# Proposed: keep last `keep` rows only (default 64), full argmax + score arrays.
# Memory = 8*keep*(max_run+1) + 8*n*2 (argmax int64 + score float64).
# For n=1M, keep=64, max_run=200: 8*64*201 + 16*1M = 103 KB + 16 MB = 16 MB.
# -> ~100x reduction.
#
# Consumers of `run_length_posterior` in tests: typically check the last few
# rows or argmax; verify by grep before merging.

# BEFORE (current, near line 236-296):
#     R_history = np.zeros((n, max_run + 1), dtype=np.float64)
#     for t in range(n):
#         ...
#         R_history[t] = R
#     return BOCPDResult(..., run_length_posterior=R_history, ...)
#
# AFTER (proposed):
#     keep = 64  # or parameter `posterior_keep`
#     R_hist = np.zeros((min(n, keep), max_run + 1), dtype=np.float64)
#     argmax_run = np.zeros(n, dtype=np.int64)
#     anomaly_score = np.zeros(n, dtype=np.float64)
#     for t in range(n):
#         ...
#         R_hist[t % keep] = R
#         argmax_run[t] = int(np.argmax(R))
#         anomaly_score[t] = R[0]
#     # reorder the ring buffer into chronological order for the last `keep` steps
#     tail = np.vstack([R_hist[(t0+i) % keep] for i in range(min(n,keep))])
#     return BOCPDResult(..., run_length_posterior=tail,
#                        argmax_run_length=argmax_run,
#                        anomaly_score=anomaly_score, ...)
#
# API change: run_length_posterior shape changes from (n, max_run+1) to
# (min(n,keep), max_run+1). To preserve backward compat, add a keyword
# `full_posterior: bool = False`; when True, keep the current behaviour
# (user opts into the RAM cost).
