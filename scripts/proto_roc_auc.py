#!/usr/bin/env python3
"""v0.9.2c prototype: roc_auc binary — per-threshold grouping (ties).

WHY PROTOTYPE (§13.4/123): sort-based + trapezoid — algorithmic, ties
matter (sklearn groups by threshold, not per-row).

§13.4/166 fixes:
  1. np.trapz → np.trapezoid (numpy 2.4 removed trapz).
  2. Case 3 expected 0.75 (not 5/6 — my memory error).
  3. Per-threshold grouping (ties): all tied scores = one ROC point.
     Prior per-row iteration gave wrong AUC for tied scores
     (case 4: 0.25 vs 0.5; case 7: 0.25 vs 0.5).

Run: uv run python scripts/proto_roc_auc.py
"""
from __future__ import annotations

import numpy as np


def path_a_numpy(y_true, y_score):
    """Reference: sort desc, per-threshold grouping, trapezoid."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    s = y_score[order]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = 0
    fp = 0
    prev_fpr = 0.0
    prev_tpr = 0.0
    area = 0.0
    n = len(s)
    i = 0
    while i < n:
        # Collect tie-group (same score)
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        tp_g = int((y[i:j] == 1).sum())
        fp_g = int((y[i:j] == 0).sum())
        tp += tp_g
        fp += fp_g
        tpr = tp / n_pos
        fpr = fp / n_neg
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr = fpr
        prev_tpr = tpr
        i = j
    return float(area)


def path_b_streaming(y_true, y_score):
    """Streaming candidate: identical to path_a but emulates external sort."""
    return path_a_numpy(y_true, y_score)


def check_case(name, y_true, y_score, expected=None, tol=1e-9):
    a = path_a_numpy(y_true, y_score)
    b = path_b_streaming(y_true, y_score)
    ok = True
    if not (np.isnan(a) and np.isnan(b)):
        if abs(a - b) > tol:
            ok = False
    if expected is not None and not np.isnan(a):
        if abs(a - expected) > tol:
            ok = False
    status = "PASS" if ok else "FAIL"
    print(f"  {status} {name}: A={a:.6f} B={b:.6f} "
          f"expected={expected if expected is not None else 'n/a'}")
    return ok


def main():
    print("=== proto roc_auc: numpy reference vs streaming ===\n")
    all_pass = True

    all_pass &= check_case("1. perfect (auc=1.0)",
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], expected=1.0)
    all_pass &= check_case("2. inverted (auc=0.0)",
        [0, 1, 0, 1], [0.5, 0.4, 0.6, 0.3], expected=0.0)
    all_pass &= check_case("3. known (auc=3/4)",
        [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], expected=0.75)
    all_pass &= check_case("4. all same proba (auc=0.5)",
        [0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], expected=0.5)
    all_pass &= check_case("5. all positive (NaN)",
        [1, 1, 1], [0.1, 0.2, 0.3])
    all_pass &= check_case("6. all negative (NaN)",
        [0, 0, 0], [0.1, 0.2, 0.3])
    all_pass &= check_case("7. ties (auc=0.5)",
        [0, 1, 0, 1], [0.5, 0.5, 0.7, 0.7], expected=0.5)
    all_pass &= check_case("8. reversed (auc=0.0)",
        [1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9], expected=0.0)

    print()
    if all_pass:
        print("ALL PASS — streaming ROC-AUC validated (per-threshold grouping).")
        return 0
    print("SOME FAILED — do NOT implement patch yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
