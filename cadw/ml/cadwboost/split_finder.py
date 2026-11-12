"""Split finder: best (feature, threshold, default_left) given histogram.

Two modes:
- symmetric: one split per level (aggregated over all leaves)
- leaf_wise: per-node best split

Missing handling: if `missing_bin=True`, NaN values are in bin `n_bins_j - 1`
(last valid bin) and excluded from regular thresholds. Then try
default_left=True (add NaN to left) and default_left=False (add to right).
Otherwise, NaN is in bin 0 (goes left automatically) — no default logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .histogram import Histogram


@dataclass
class SplitResult:
    feature: int       # -1 if no valid split
    threshold: int     # bin threshold: bins < t go left
    default_left: bool
    gain: float


def _score(G: np.ndarray, H: np.ndarray, reg_lambda: float) -> np.ndarray:
    """G²/(H+λ). Both arrays (..., ) — elementwise."""
    eps = 1e-12
    return G * G / (H + reg_lambda + eps)


def best_split_symmetric(
    hist: Histogram,
    reg_lambda: float,
    min_child_weight: float,
    min_gain: float,
    *,
    missing_bin: bool = False,
    feature_mask: np.ndarray | None = None,
) -> SplitResult:
    """Best split aggregated over all leaves (symmetric tree).

    feature_mask: optional bool (d,) — features allowed (colsample).
    """
    nl, d, n_bins_max = hist.G.shape
    eps = 1e-12

    best_gain = -np.inf
    best = SplitResult(-1, -1, True, 0.0)

    for j in range(d):
        if feature_mask is not None and not feature_mask[j]:
            continue
        n_bins_j = int(hist.n_bins_per_feature[j])
        if n_bins_j < 2:
            continue

        G = hist.G[:, j, :n_bins_j]  # (nl, n_bins_j)
        H = hist.H[:, j, :n_bins_j]

        if missing_bin:
            # Last bin is NaN bin; exclude from regular
            G_miss = G[:, -1]
            H_miss = H[:, -1]
            G_reg = G[:, :-1]
            H_reg = H[:, :-1]
        else:
            G_miss = np.zeros(nl)
            H_miss = np.zeros(nl)
            G_reg = G
            H_reg = H

        n_bins_reg = G_reg.shape[1]
        if n_bins_reg < 2:
            continue

        # Cumulative sums
        cumG = np.cumsum(G_reg, axis=1)
        cumH = np.cumsum(H_reg, axis=1)
        G_reg_total = cumG[:, -1]
        H_reg_total = cumH[:, -1]

        # Thresholds t = 1..n_bins_reg-1 (t-1 indices in cumsum)
        G_left = cumG[:, :-1]                              # (nl, t-1)
        H_left = cumH[:, :-1]
        G_right = G_reg_total[:, None] - G_left
        H_right = H_reg_total[:, None] - H_left

        # Default left: add missing to left
        GL_l = G_left + G_miss[:, None]
        HL_l = H_left + H_miss[:, None]
        GR_l = G_right
        HR_l = H_right
        # Default right: add missing to right
        GL_r = G_left
        HL_r = H_left
        GR_r = G_right + G_miss[:, None]
        HR_r = H_right + H_miss[:, None]

        # Score per leaf per threshold
        score_l = _score(GL_l, HL_l, reg_lambda) + \
                  _score(GR_l, HR_l, reg_lambda)
        score_r = _score(GL_r, HL_r, reg_lambda) + \
                  _score(GR_r, HR_r, reg_lambda)

        # min_child_weight mask
        ok_l = (HL_l >= min_child_weight) & (HR_l >= min_child_weight)
        ok_r = (HL_r >= min_child_weight) & (HR_r >= min_child_weight)
        score_l = np.where(ok_l, score_l, -np.inf)
        score_r = np.where(ok_r, score_r, -np.inf)

        # Aggregate over leaves
        agg_l = score_l.sum(axis=0)   # (t-1,)
        agg_r = score_r.sum(axis=0)

        # Parent (same for both, = total²/(total+λ))
        parent = _score(G_reg_total, H_reg_total, reg_lambda).sum()

        i_l = int(np.argmax(agg_l))
        i_r = int(np.argmax(agg_r))
        g_l = float(agg_l[i_l]) - float(parent)
        g_r = float(agg_r[i_r]) - float(parent)

        if not np.isfinite(g_l) and not np.isfinite(g_r):
            continue

        if g_l >= g_r:
            cand_gain, cand_thresh, cand_dl = g_l, i_l + 1, True
        else:
            cand_gain, cand_thresh, cand_dl = g_r, i_r + 1, False

        if cand_gain > best_gain:
            best_gain = cand_gain
            best = SplitResult(
                feature=j, threshold=cand_thresh,
                default_left=cand_dl, gain=cand_gain,
            )

    if best_gain < min_gain:
        return SplitResult(-1, -1, True, 0.0)
    return best


def best_split_per_node(
    G: np.ndarray,          # (d, n_bins) for one node
    H: np.ndarray,
    n_bins_per_feature: np.ndarray,
    reg_lambda: float,
    min_child_weight: float,
    min_gain: float,
    *,
    missing_bin: bool = False,
    feature_mask: np.ndarray | None = None,
) -> SplitResult:
    """Leaf-wise: best split for a single node's histogram (vectorized).

    All features batched: no Python loop. One cumsum, one argmax per
    default direction, one max across features.
    """
    d, n_bins = G.shape
    eps = 1e-12

    # Parent score (single scalar): use feature 0's total (all features
    # have identical total = sum of grad over all rows)
    G_tot = float(G[0].sum())
    H_tot = float(H[0].sum())
    parent = G_tot * G_tot / (H_tot + reg_lambda + eps)

    # Build per-feature valid mask (n_bins limit per feature)
    valid = np.arange(n_bins)[None, :] < n_bins_per_feature[:, None]  # (d, n_bins)
    # Also feature_mask (colsample)
    if feature_mask is not None:
        valid &= feature_mask[:, None]

    # Extract G_miss, G_reg if missing_bin
    if missing_bin:
        # Last valid bin per feature = NaN bin (only if n_bins_per_feature > 1)
        # Simplify: assume all features have same n_bins (typical 255)
        # Robust: use per-feature last valid index
        last_idx = (n_bins_per_feature - 1).astype(np.int64)
        last_idx = np.clip(last_idx, 0, n_bins - 1)
        G_miss = G[np.arange(d), last_idx]              # (d,)
        H_miss = H[np.arange(d), last_idx]
        # Zero out the last bin in regular range
        G_reg = G.copy()
        H_reg = H.copy()
        G_reg[np.arange(d), last_idx] = 0.0
        H_reg[np.arange(d), last_idx] = 0.0
        # Regular valid = all but last
        valid_reg = valid.copy()
        valid_reg[np.arange(d), last_idx] = False
    else:
        G_miss = np.zeros(d)
        H_miss = np.zeros(d)
        G_reg = G
        H_reg = H
        valid_reg = valid

    # Cumsum along bins (feature, bin)
    cumG = np.cumsum(G_reg, axis=1)       # (d, n_bins)
    cumH = np.cumsum(H_reg, axis=1)

    # Per-feature totals
    G_feat_tot = cumG[:, -1]              # (d,)
    H_feat_tot = cumH[:, -1]

    # Left = bins [0..t-1], right = bins [t..n_bins-1]
    # For threshold t ∈ [1, n_bins-1], need cumsum[t-1]
    G_left = cumG[:, :-1]                 # (d, n_bins-1)
    H_left = cumH[:, :-1]
    G_right = G_feat_tot[:, None] - G_left
    H_right = H_feat_tot[:, None] - H_left

    # Default left: add missing to left
    GL_l = G_left + G_miss[:, None]
    HL_l = H_left + H_miss[:, None]
    GR_l = G_right
    HR_l = H_right
    # Default right: add missing to right
    GL_r = G_left
    HL_r = H_left
    GR_r = G_right + G_miss[:, None]
    HR_r = H_right + H_miss[:, None]

    def _score(Gx, Hx):
        return Gx * Gx / (Hx + reg_lambda + eps)

    score_l = _score(GL_l, HL_l) + _score(GR_l, HR_l)
    score_r = _score(GL_r, HL_r) + _score(GR_r, HR_r)

    # min_child_weight
    mcw = min_child_weight
    ok_l = (HL_l >= mcw) & (HR_l >= mcw)
    ok_r = (HL_r >= mcw) & (HR_r >= mcw)
    score_l = np.where(ok_l, score_l, -np.inf)
    score_r = np.where(ok_r, score_r, -np.inf)

    # Apply valid_reg mask for threshold positions (t-1 = 0..n_bins-2)
    # A threshold at position t-1 is valid if both t-1 and t are in valid_reg
    # (left has bins 0..t-1, right has bins t..n_bins-1)
    # Simpler: threshold t-1 valid if cumsum at t-1 makes sense
    # Use valid_reg[:, :n_bins-1] (bin t-1) and valid_reg[:, 1:] (bin t)
    valid_thresh = valid_reg[:, :n_bins - 1] & valid_reg[:, 1:]
    score_l = np.where(valid_thresh, score_l, -np.inf)
    score_r = np.where(valid_thresh, score_r, -np.inf)

    # Best per feature per direction
    best_l_idx = np.argmax(score_l, axis=1)    # (d,)
    best_r_idx = np.argmax(score_r, axis=1)
    best_l_gain = score_l[np.arange(d), best_l_idx] - parent
    best_r_gain = score_r[np.arange(d), best_r_idx] - parent

    # Choose best direction per feature
    use_left = best_l_gain >= best_r_gain
    best_gain_per_feat = np.where(use_left, best_l_gain, best_r_gain)
    best_idx_per_feat = np.where(use_left, best_l_idx, best_r_idx)

    # Best across features
    overall_best_feat = int(np.argmax(best_gain_per_feat))
    overall_gain = float(best_gain_per_feat[overall_best_feat])

    if overall_gain < min_gain or not np.isfinite(overall_gain):
        return SplitResult(-1, -1, True, 0.0)

    return SplitResult(
        feature=overall_best_feat,
        threshold=int(best_idx_per_feat[overall_best_feat]) + 1,
        default_left=bool(use_left[overall_best_feat]),
        gain=overall_gain,
    )

