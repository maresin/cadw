"""Probability calibration: Platt scaling and isotonic regression (D.4.6, v0.9.39).

Reference: Platt, *Probabilistic Outputs for Support Vector Machines and
Comparisons to Regularized Likelihood Methods*, Advances in Large Margin
Classifiers, 2000 (Platt scaling); Zadrozny, Elkan, *Transforming Classifier
Scores into Accurate Multiclass Probability Estimates*, KDD 2002 (isotonic
calibration); Niculescu-Mizil, Caruana, *Predicting Good Probabilities With
Supervised Learning*, ICML 2005 (Brier score / ECE).
See .pingis/algorithm_notes/d4_6_calibration.md.

Works on top of any binary classifier exposing `predict_proba(X)` and is
exposed here as a wrapper around the in-tree GBDT binary classifier. Both
calibrators are fit on a held-out slice (or the full set, per the caller).
RAM: O(n) score vector + O(1) isotonic breakpoints.
"""
from __future__ import annotations

from cadw.ml._common import _stack_numeric as _stack_numeric_safe

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit




def _collect(df, cols, y_col, mg, name):
    Xs, ys = [], []
    n = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name=name, extra_factor=3.0)
        # v0.9.90-S3 (G2): pre-check BEFORE append.
        n += b.num_rows
        mg.check_state(n * (len(cols) * 8 + 8), name=name + ".state")
        Xs.append(_stack_numeric_safe(b, cols))
        ys.append(b.column(y_col).cast(pa.float64(), safe=False)
                  .to_numpy(zero_copy_only=False))
    if n == 0:
        raise ValueError("calibration: empty input")
    return np.vstack(Xs), np.concatenate(ys)


# v0.9.131 (DEC-054): numerically stable sigmoid from
# cadw.ops.optim (handles overflow for |z|>>0 without RuntimeWarning).
from cadw.ops.numeric import sigmoid as _sigmoid



def _fit_platt(scores, y, max_iter=100, tol=1e-8):
    """Platt scaling: p = sigmoid(A * score + B) fit by Newton IRLS."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A, B = 1.0, 0.0
    eps = 1e-12
    for _ in range(max_iter):
        p = _sigmoid(A * scores + B)
        w = p * (1.0 - p) + eps
        gA = np.sum((p - y) * scores)
        gB = np.sum(p - y)
        hAA = np.sum(w * scores * scores) + eps
        hAB = np.sum(w * scores)
        hBB = np.sum(w) + eps
        det = hAA * hBB - hAB * hAB
        if det <= 0:
            break
        dA = (hBB * gA - hAB * gB) / det
        dB = (hAA * gB - hAB * gA) / det
        A -= dA; B -= dB
        if abs(dA) < tol and abs(dB) < tol:
            break
    return float(A), float(B)


def _fit_isotonic(scores, y, n_bins=256):
    """Isotonic regression (PAVA) on a binned score grid."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    s, yy = scores[order], y[order]
    # bin by equal-count quantiles
    n = s.size
    bins = max(2, min(n_bins, n))
    edges = np.linspace(0, n, bins + 1, dtype=np.int64)
    bx, by = [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        bx.append(float(s[lo:hi].mean()))
        by.append(float(yy[lo:hi].mean()))
    bx = np.array(bx); by = np.array(by)
    # PAVA
    w = np.diff(edges)[:len(by)].astype(np.float64)
    vals = by.copy(); wts = w.copy()
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1]:
            new_v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / (wts[i] + wts[i + 1])
            vals[i] = new_v; wts[i] = wts[i] + wts[i + 1]
            vals = np.delete(vals, i + 1); wts = np.delete(wts, i + 1)
            bx = np.delete(bx, i + 1)
            i = max(0, i - 1)
        else:
            i += 1
    return bx, vals


def _apply_isotonic(scores, bx, vals):
    return np.interp(np.asarray(scores, dtype=np.float64), bx, vals)


def brier_score(y_true, proba):
    """Mean squared error between probability and outcome."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(proba, dtype=np.float64)
    if y.size == 0 or y.size != p.size:
        raise ValueError("brier_score: y and proba must be equal non-zero length")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true, proba, n_bins=10):
    """Equal-frequency ECE (Niculescu-Mizil & Caruana 2005)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(proba, dtype=np.float64)
    if y.size == 0 or y.size != p.size:
        raise ValueError("expected_calibration_error: y and proba must be equal non-zero length")
    order = np.argsort(p, kind="stable")
    p_s, y_s = p[order], y[order]
    n = p_s.size
    bins = max(2, min(n_bins, n))
    edges = np.linspace(0, n, bins + 1, dtype=np.int64)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        w = (hi - lo) / n
        conf = p_s[lo:hi].mean()
        acc = y_s[lo:hi].mean()
        ece += w * abs(conf - acc)
    return float(ece)
class CalibratedModel:
    """Platt- or isotonic-calibrated wrapper around a fitted binary classifier."""

    def __init__(self, base, method, A, B, iso_x, iso_y, feature_cols, y_col,
                 params):
        self.base_ = base
        self.method_ = method
        self.A_ = A
        self.B_ = B
        self.iso_x_ = iso_x
        self.iso_y_ = iso_y
        self.feature_cols_ = list(feature_cols)
        self.y_col_ = y_col
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        base = self.base_.state_nbytes() if hasattr(self.base_, "state_nbytes") else 0
        iso = 0 if self.iso_x_ is None else int(self.iso_x_.nbytes + self.iso_y_.nbytes)
        return int(base + iso + 512)

    def to_dict(self):
        return {"method": self.method_, "params": self.params_,
                "A": self.A_, "B": self.B_,
                "iso_points": 0 if self.iso_x_ is None else len(self.iso_x_)}

    def _raw_proba(self, df, mg):
        out = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="calibration.predict", extra_factor=3.0)
            out.append(self.base_.predict_proba(_stack_numeric_safe(b, self.feature_cols_)))
        return np.concatenate(out) if out else np.zeros(0)

    def predict_proba(self, df, memory_guard=None):
        mg = memory_guard or df._memory_guard
        raw = self._raw_proba(df, mg)
        if self.method_ == "platt":
            return _sigmoid(self.A_ * raw + self.B_)
        return _apply_isotonic(raw, self.iso_x_, self.iso_y_)

    def predict(self, df, memory_guard=None, threshold=0.5):
        p = self.predict_proba(df, memory_guard)
        return (p >= threshold).astype(np.int64)

    def brier(self, df, memory_guard=None):
        mg = memory_guard or df._memory_guard
        y = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="calibration.brier", extra_factor=3.0)
            y.append(b.column(self.y_col_).cast(pa.float64(), safe=False)
                     .to_numpy(zero_copy_only=False))
        return brier_score(np.concatenate(y), self.predict_proba(df, mg))

    def ece(self, df, memory_guard=None, n_bins=10):
        mg = memory_guard or df._memory_guard
        y = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="calibration.ece", extra_factor=3.0)
            y.append(b.column(self.y_col_).cast(pa.float64(), safe=False)
                     .to_numpy(zero_copy_only=False))
        return expected_calibration_error(np.concatenate(y),
                                          self.predict_proba(df, mg), n_bins)


def fit_calibration(df, X_cols, y_col, *, method="platt", n_estimators=50,
                    max_depth=3, learning_rate=0.1, n_bins=256,
                    memory_guard: Optional[MemoryGuard] = None):
    """Fit a binary GBDT classifier and calibrate its probabilities."""
    if method not in ("platt", "isotonic"):
        raise ValueError(f"fit_calibration: method must be 'platt' or 'isotonic', got {method!r}")
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_calibration: X_cols must be non-empty")
    for c in cols + [y_col]:
        if c not in df._columns:
            raise KeyError(f"fit_calibration: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    from cadw.ml.gbdt import fit_gbdt_classifier
    base = fit_gbdt_classifier(df, cols, y_col, n_estimators=n_estimators,
                               max_depth=max_depth, learning_rate=learning_rate,
                               memory_guard=mg)
    raw, y = [], []
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="calibration.fit_scores", extra_factor=3.0)
        raw.append(base.predict_proba(_stack_numeric_safe(b, cols)))
        y.append(b.column(y_col).cast(pa.float64(), safe=False)
                 .to_numpy(zero_copy_only=False))
    scores = np.concatenate(raw) if raw else np.zeros(0)
    yv = np.concatenate(y) if y else np.zeros(0)
    if scores.size == 0:
        raise ValueError("fit_calibration: empty input")
    uniq = np.unique(yv)
    if not np.array_equal(uniq, np.array([0.0, 1.0])):
        raise ValueError(f"fit_calibration: y must be binary 0/1; got {uniq.tolist()}")
    A = B = None
    iso_x = iso_y = None
    if method == "platt":
        A, B = _fit_platt(scores, yv)
    else:
        iso_x, iso_y = _fit_isotonic(scores, yv, n_bins=n_bins)
    model = CalibratedModel(base, method, A, B, iso_x, iso_y, cols, y_col,
                            {"n_estimators": n_estimators,
                             "max_depth": max_depth,
                             "learning_rate": learning_rate,
                             "n_bins": n_bins})
    log_fit("calibration", state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="calibration.model")
    return model
