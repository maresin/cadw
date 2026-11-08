"""Common ML utilities — single source of truth.

Canonical implementations. Modules should import from here rather than
duplicate (AI_CONSTITUTION §5.1).

History: prior to v0.9.108, `_stack_numeric` had 5 identical copies
(gbdt, linear, naive_bayes, pca, kmeans); `_column_to_f64` had 5 copies
(same 5 files, plus logistic); `_stack_numeric_safe` was a boilerplate
wrapper in 9 more files. Deduplication is bit-identical (proven by
existing tests).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pyarrow as pa


def _column_to_f64(arr: pa.Array) -> np.ndarray:
    """Arrow Array → np.float64 with NaN for nulls.

    Object dtype path: explicit per-element iteration (NaN for None).
    Numeric dtype path: zero-copy astype(float64, copy=False).

    v0.9.108: canonical implementation. Prior to this, the same body
    was copy-pasted in gbdt/linear/logistic/naive_bayes/pca/kmeans.
    """
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


# Alias — matches linear.py's historical name. Semantics identical.
_numeric_to_f64 = _column_to_f64


def _stack_numeric(
    batch: pa.RecordBatch, cols: Sequence[str],
) -> np.ndarray:
    """Stack numeric Arrow columns → (n_rows, len(cols)) float64.

    np.column_stack on per-column float64 arrays. Peak = 2× batch
    (per-column pieces + result). Coef ~1.92 measured in v0.9.105
    experiment (EXPERIMENT_PROCEDURE.md §7).

    v0.9.108: canonical implementation. Prior copies: gbdt/linear/
    naive_bayes/pca/kmeans (identical bodies). Caller `_stack_numeric_safe`
    wrappers (9 more modules) now import this directly as alias.
    """
    parts = [_column_to_f64(batch.column(c)) for c in cols]
    return np.column_stack(parts)






def _to_df_causal(
    X: np.ndarray, t: np.ndarray, y: np.ndarray, cols: list,
):
    """Build a CadwDataFrame from (X, t, y) arrays for causal T/S-learners.

    Columns: `cols` (X), `__t` (treatment), `__y` (outcome). Prior to
    v0.9.133 had 2 identical copies (meta_learners + uplift).
    CadwDataFrame imported locally to avoid circular import with
    cadw.dataframe._ml_bindings.
    """
    from cadw import CadwDataFrame
    d = {c: pa.array(X[:, j]) for j, c in enumerate(cols)}
    d["__t"] = pa.array(t, type=pa.float64())
    d["__y"] = pa.array(y, type=pa.float64())
    return CadwDataFrame.from_table(pa.table(d))
