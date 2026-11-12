"""Missing value handling (XGBoost-style default direction).

For each split, `default_left` decides where NaN goes.
- "left" (default): NaN rows go left
- "right": NaN rows go right
- "learn": at split time, evaluate both directions, pick the one with
  higher gain (XGBoost's "learn" mode). Requires 2× split evaluation.

The tree stores `default_left` per split. Predict path: NaN in feature
→ follow default_left. Non-NaN: follow threshold.

If binning uses `missing_bin=True`, NaN has its own bin and no default
direction needed (bin comparison handles it uniformly).
"""
from __future__ import annotations

import numpy as np


def split_gain_with_default(
    G_left: float, H_left: float,
    G_right: float, H_right: float,
    G_missing: float, H_missing: float,
    reg_lambda: float,
) -> tuple[float, bool]:
    """Evaluate split with both default directions, return (best_gain, default_left).

    G_left/H_left: gradients/hessians in bins < threshold (non-NaN).
    G_right/H_right: bins >= threshold (non-NaN).
    G_missing/H_missing: NaN rows (would go to default side).
    """
    eps = 1e-12
    # Default left: missing added to left
    gl_l = G_left + G_missing
    hl_l = H_left + H_missing
    gr_l = G_right
    hr_l = H_right
    gain_left = (
        gl_l**2 / (hl_l + reg_lambda + eps)
        + gr_l**2 / (hr_l + reg_lambda + eps)
    )
    # Default right: missing added to right
    gl_r = G_left
    hl_r = H_left
    gr_r = G_right + G_missing
    hr_r = H_right + H_missing
    gain_right = (
        gl_r**2 / (hl_r + reg_lambda + eps)
        + gr_r**2 / (hr_r + reg_lambda + eps)
    )
    if gain_left >= gain_right:
        return float(gain_left), True
    return float(gain_right), False
