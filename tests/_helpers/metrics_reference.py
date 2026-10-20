"""Reference metrics for tests (per-threshold ROC-AUC, log-loss).

Validated by prototype scripts/proto_roc_auc.py (PASS 8/8). Manual
arithmetic in test bodies prohibited (§13.4/166); use these helpers
instead.
"""
from __future__ import annotations

import numpy as np


def expected_roc_auc(y_true, y_score) -> float:
    """Per-threshold grouping ROC-AUC. Matches sklearn.roc_auc_score.

    All tied scores → one threshold. NaN if n_pos==0 or n_neg==0.
    """
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    area = 0.0
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        tp += int((y[i:j] == 1).sum())
        fp += int((y[i:j] == 0).sum())
        tpr = tp / n_pos
        fpr = fp / n_neg
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr = fpr
        prev_tpr = tpr
        i = j
    return float(area)


def expected_log_loss_binary(y_true, y_proba, eps: float = 1e-15) -> float:
    """Binary cross-entropy, matches sklearn.log_loss (skipna=True)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_proba, dtype=np.float64)
    mask = ~(np.isnan(y) | np.isnan(p))
    y = y[mask]
    p = np.clip(p[mask], eps, 1.0 - eps)
    if len(y) == 0:
        return float("nan")
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def expected_log_loss_multiclass(
    y_true_idx, y_proba_matrix, eps: float = 1e-15,
) -> float:
    """Multiclass cross-entropy.

    y_true_idx: (N,) int array in [0, K-1].
    y_proba_matrix: (N, K) float array (rows sum to 1).
    NaN if empty after null filter.
    """
    y = np.asarray(y_true_idx, dtype=np.int64)
    P = np.asarray(y_proba_matrix, dtype=np.float64)
    mask = ~(np.isnan(y)) & ~(np.isnan(P).any(axis=1))
    y = y[mask]
    P = P[mask]
    if len(y) == 0:
        return float("nan")
    P = np.clip(P, eps, 1.0 - eps)
    return float(-np.log(P[np.arange(len(y)), y]).mean())
