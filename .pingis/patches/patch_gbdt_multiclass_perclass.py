# PATCH PROPOSAL 3: cadw/ml/gbdt.py — per-class grad/hess buffers
# Status: proposed (not applied).
#
# Current (v0.9.86.3, lines ~1324-1342):
#   for round_i in range(n_estimators):
#       grad_all = np.zeros((n_total, K))   # 8*n*K
#       hess_all = np.zeros((n_total, K))   # 8*n*K
#       for b in df._iter_filtered_batches():
#           grad_all[offset:offset+n][valid] = g_valid
#           hess_all[offset:offset+n][valid] = h_valid
#       for k in range(K):
#           fit_tree(grad_all[:,k], hess_all[:,k])   # only column k used
#
# Memory = 16*n*K per round (transient); at n=1M, K=10 -> 160 MB.
# Plus pred (n,K) = 80 MB retained across rounds.
#
# Proposed: allocate one (n, K) grad/hess buffer ONCE outside the round loop
# and refill it per round; or (better) process one class at a time with (n,)
# arrays:
#   for round_i in range(n_estimators):
#       for k in range(K):
#           g_k = np.empty(n_total); h_k = np.empty(n_total)
#           for b in df._iter_filtered_batches(): ...
#           fit_tree(g_k, h_k)
#
# Memory = 16*n per round (K-fold reduction). pred still O(n*K) — that is
# inherent to softmax boosting; could also be spilled to Arrow IPC if needed.
#
# This patch is a refactor of the inner loop; numerical result is identical
# because trees only ever read a single class column at a time.
