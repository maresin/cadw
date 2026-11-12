"""CadwDataFrame API: fit_df, predict_df.

Bridges CadwDataFrame (arrow-batched, OOC) to CADWBoost's numpy
in-memory core. Two modes:
- "memory": collect all batches → single numpy array (small n)
- "ooc": stream batches, accumulate binning + codes on disk (large n)

For P0, "memory" mode works. OOC mode delegated to ooc.py.
"""
from __future__ import annotations

import numpy as np


def df_to_numpy(df, X_cols, y_col=None, *, memory_guard=None):
    """Collect CadwDataFrame batches into (X, y) numpy arrays.

    For n up to ~10^6 this is fine (8·n·d bytes). Larger n → use
    `fit_df(..., mode="ooc")` (delegates to ooc.stream_fit).
    """
    mg = memory_guard or df._memory_guard
    X_parts = []
    y_parts = []
    n_total = 0
    d = len(X_cols)

    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="cadwboost.batch", extra_factor=2.0)
        n_total += b.num_rows
        mg.check_state(n_total * d * 8 + n_total * 8,
                        name="cadwboost.accumulate")
        cols = [b.column(c).to_numpy(zero_copy_only=False)
                for c in X_cols]
        X_parts.append(np.column_stack(cols).astype(np.float64))
        if y_col is not None:
            y_parts.append(b.column(y_col).to_numpy(
                zero_copy_only=False).astype(np.float64))

    if not X_parts:
        raise ValueError("cadwboost: empty dataframe")
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts) if y_parts else None
    return X, y


def predict_df_to_numpy(model, df, X_cols, *, memory_guard=None):
    """Streaming predict: per-batch, return np.ndarray predictions."""
    mg = memory_guard or df._memory_guard
    out = []
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="cadwboost.predict_batch", extra_factor=2.0)
        cols = [b.column(c).to_numpy(zero_copy_only=False)
                for c in X_cols]
        X_b = np.column_stack(cols).astype(np.float64)
        out.append(model.predict(X_b))
    return np.concatenate(out) if out else np.array([])


def fit_df(model, df, X_cols, y_col, *, mode: str = "memory",
           memory_guard=None):
    """Fit CADWBoost from CadwDataFrame.

    mode="memory": collect → X, y → fit (default; n ≤ ~10^6).
    mode="ooc": stream binning + codes → memmap → fit (n arbitrary).

    Returns fitted model (self).
    """
    if mode not in ("memory", "ooc"):
        raise ValueError(f"mode must be 'memory' or 'ooc'; got {mode!r}")

    if mode == "memory":
        X, y = df_to_numpy(df, X_cols, y_col, memory_guard=memory_guard)
        model.fit(X, y, feature_names=list(X_cols))
        return model

    # OOC path (P1)
    from .ooc import fit_ooc
    return fit_ooc(model, df, X_cols, y_col, memory_guard=memory_guard)


def predict_df(model, df, X_cols, *, output: str = "numpy",
               memory_guard=None):
    """Predict on CadwDataFrame, stream per batch.

    output="numpy" (default) → np.ndarray
    output="df" → CadwDataFrame with column "prediction"
    """
    preds = predict_df_to_numpy(model, df, X_cols, memory_guard=memory_guard)
    if output == "numpy":
        return preds
    if output == "df":
        import pyarrow as pa
        from cadw.dataframe import CadwDataFrame
        tbl = pa.table({"prediction": pa.array(preds)})
        mg = memory_guard or df._memory_guard
        return CadwDataFrame.from_table(tbl, memory_guard=mg)
    raise ValueError(f"output must be 'numpy' or 'df'; got {output!r}")


def predict_proba_df(model, df, X_cols, *, memory_guard=None):
    """Streaming predict_proba (binary classification only)."""
    mg = memory_guard or df._memory_guard
    out = []
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="cadwboost.proba_batch", extra_factor=2.0)
        cols = [b.column(c).to_numpy(zero_copy_only=False)
                for c in X_cols]
        X_b = np.column_stack(cols).astype(np.float64)
        out.append(model.predict_proba(X_b))
    return np.concatenate(out) if out else np.array([])
