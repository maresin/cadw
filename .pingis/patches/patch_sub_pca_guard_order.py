# PATCH PROPOSAL 1: cadw/ml/sub_pca.py
# - Move memory guard BEFORE window allocation.
# - Replace eager dense windows with strided view + chunked consumption.
# Status: proposed (not applied). See .pingis/audits/inefficiency.md.

# --- helper to add near the top of sub_pca.py ---
def _estimate_windows_bytes(n: int, w: int, c: int = 1) -> int:
    """Analytic bytes for a (n_w, w*c) float64 window array."""
    n_w = max(0, n - w + 1)
    return 8 * n_w * (w * c)

# --- replacements for the top of fit_sub_pca (around current lines 242-261) ---
# BEFORE (current):
#     windows, n, c = _prepare_windows(series, window_size, stride=1)
#     ...
#     if guard is not None:
#         est_bytes = 8 * n_w * d
#         limit_mb = _probe_memory_limit_mb(guard)
#         if est_bytes > limit_mb * 1024 * 1024:
#             raise MemoryLimitExceeded(...)
#
# AFTER (proposed):
#     X = np.asarray(series, dtype=np.float64)
#     n = len(X) if X.ndim == 1 else X.shape[0]
#     c = 1 if X.ndim == 1 else X.shape[1]
#     if n < window_size:
#         raise ValueError(...)
#     n_w = n - window_size + 1
#     if guard is not None:
#         guard.check_state(_estimate_windows_bytes(n, window_size, c),
#                           name="sub_pca.windows")
#     # then build windows lazily or chunked:
#     sw = np.lib.stride_tricks.sliding_window_view(X, window_size, axis=0)
#     # iterate sw in chunks of `chunk_rows` (e.g. 2000) for PCA fit/score
