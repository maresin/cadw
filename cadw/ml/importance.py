"""Permutation Feature Importance (D.6.3, Phase D, v0.9.31).

Reference: L. Breiman, *Random Forests*, Machine Learning 45, 2001,
    section "Variable importance". See sklearn.inspection.permutation_importance
    (BSD) for the modern reference implementation.
Design: docs/decisions/d6_3_permutation_importance.md.

Model-agnostic feature attribution. Samples `max_samples` rows via
reservoir sampling (state independent of N), shuffles one column at a
time and recomputes the score. Any model with `predict(np.ndarray)` works.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _sample_Xy(df, cols, y_col, S, rng, memory_guard):
    """Reservoir sample S rows of X (and y). Returns (X, y, valid_mask)."""
    mg = memory_guard or MemoryGuard(max_memory_mb=64)
    X = np.empty((S, len(cols)), dtype=np.float64)
    y = None if y_col is None else np.empty(S, dtype=np.float64)
    n_seen = 0; n_filled = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="permimp.sample", extra_factor=3.0)
        arrays = []
        for c in cols:
            arr = b.column(c)
            if not (pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type)):
                raise TypeError(f"permutation_importance: X column {c!r} must be numeric")
            arrays.append(arr.cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False))
        X_b = np.column_stack(arrays)
        mask = ~np.isnan(X_b).any(axis=1)
        y_b = None
        if y_col is not None:
            yarr = b.column(y_col)
            y_np = yarr.cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False)
            y_b = y_np
            mask &= ~np.isnan(y_b)
        for i in range(X_b.shape[0]):
            if not mask[i]:
                continue
            if n_filled < S:
                X[n_filled] = X_b[i]
                if y is not None:
                    y[n_filled] = y_b[i]
                n_filled += 1
            else:
                j = int(rng.randint(0, n_seen + 1))
                if j < S:
                    X[j] = X_b[i]
                    if y is not None:
                        y[j] = y_b[i]
            n_seen += 1
    return X[:n_filled], (y[:n_filled] if y is not None else None)


def permutation_importance(
    model, df, X_cols: Sequence[str], *,
    y_col: Optional[str] = None,
    n_repeats: int = 5,
    max_samples: int = 1000,
    metric: str = "auto",
    random_state: Optional[int] = None,
    memory_guard: Optional[MemoryGuard] = None,
) -> pa.Table:
    """Model-agnostic permutation feature importance.

    Parameters
    ----------
    model          Any object with predict(np.ndarray) -> np.ndarray.
    df             CadwDataFrame (iterated in batches, bounded sample).
    X_cols         Columns fed to model.predict as a matrix.
    y_col          Target column, or None (prediction-shift mode).
    n_repeats      Shuffles per feature.
    max_samples    Sample cap (state is O(max_samples * (d + 1))).
    metric         "auto" | "r2" | "accuracy" | "shift" | None.
    """
    if not X_cols:
        raise ValueError("permutation_importance: X_cols must be non-empty")
    if max_samples < 2:
        raise ValueError("permutation_importance: max_samples must be >= 2")
    if n_repeats < 1:
        raise ValueError("permutation_importance: n_repeats must be >= 1")

    rng = np.random.RandomState(0 if random_state is None else int(random_state))
    X, y = _sample_Xy(df, list(X_cols), y_col, int(max_samples), rng, memory_guard)
    if X.shape[0] < 2:
        raise ValueError("permutation_importance: fewer than 2 valid rows")

    # Metric selection
    if metric in ("auto", None):
        metric = "r2" if y is not None else "shift"
    if metric == "r2" and y is None:
        raise ValueError("permutation_importance: metric='r2' requires y_col")
    if metric == "accuracy" and y is None:
        raise ValueError("permutation_importance: metric='accuracy' requires y_col")

    def score_fn(Xm, y_m):
        pred = np.asarray(model.predict(Xm)).ravel()
        if metric == "r2":
            return _r2(y_m, pred)
        if metric == "accuracy":
            return _accuracy(y_m, pred)
        if metric == "shift":
            # Unsupervised: baseline vs self -> 0 baseline; shift mode
            # compares current preds to unshuffled preds.
            return 0.0
        raise ValueError(f"permutation_importance: unknown metric {metric!r}")

    base_pred = np.asarray(model.predict(X)).ravel()
    if metric in ("r2", "accuracy"):
        base = score_fn(X, y)
    else:
        base = 0.0
        base_pred = np.asarray(model.predict(X)).ravel()

    d = X.shape[1]
    means = np.zeros(d, dtype=np.float64)
    stds = np.zeros(d, dtype=np.float64)

    for j in range(d):
        drops = np.empty(n_repeats, dtype=np.float64)
        for r in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            if metric == "shift":
                shifted_pred = np.asarray(model.predict(Xp)).ravel()
                drops[r] = float(np.mean(np.abs(shifted_pred - base_pred)))
            else:
                drops[r] = base - score_fn(Xp, y)
        means[j] = drops.mean()
        stds[j] = drops.std()

    return pa.table({
        "feature": pa.array(list(X_cols), type=pa.string()),
        "importance_mean": pa.array(means, type=pa.float64()),
        "importance_std": pa.array(stds, type=pa.float64()),
    })
