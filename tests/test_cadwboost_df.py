"""v0.9.224-a: CadwBoost DataFrame API tests (fit_df / predict_df)."""
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard
from cadw.ml.cadwboost import CADWBoost
from cadw.ml.cadwboost.dataframe import (
    fit_df, predict_df, predict_proba_df, predict_df_to_numpy)


def _mk_df(n=300, d=4, kind="reg", seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    if kind == "reg":
        y = X[:, 0] * 2 + X[:, 1] + 0.1 * rng.normal(size=n)
        y_arr = pa.array(y, type=pa.float64())
    else:
        y = (X[:, 0] > 0).astype(np.float64)
        y_arr = pa.array(y, type=pa.float64())
    cols = {f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)}
    cols["y"] = y_arr
    return CadwDataFrame.from_table(pa.table(cols),
                                     memory_guard=MemoryGuard(max_memory_mb=500))


def test_fit_df_memory():
    df = _mk_df(300, 4, "reg")
    m = CADWBoost(n_estimators=10, max_depth=3, loss="mse", random_state=42)
    fit_df(m, df, [f"x{j}" for j in range(4)], "y",
           mode="memory", memory_guard=MemoryGuard(max_memory_mb=500))
    assert m.fitted_


def test_predict_df_numpy_output():
    df = _mk_df(300, 4, "reg")
    m = CADWBoost(n_estimators=10, max_depth=3, loss="mse", random_state=42)
    fit_df(m, df, [f"x{j}" for j in range(4)], "y",
           mode="memory", memory_guard=MemoryGuard(max_memory_mb=500))
    p = predict_df_to_numpy(m, df, [f"x{j}" for j in range(4)],
                             memory_guard=MemoryGuard(max_memory_mb=500))
    assert p.shape == (300,)


def test_predict_proba_df_binary():
    df = _mk_df(300, 4, "bin")
    m = CADWBoost(n_estimators=10, max_depth=3, loss="logistic", random_state=42)
    fit_df(m, df, [f"x{j}" for j in range(4)], "y",
           mode="memory", memory_guard=MemoryGuard(max_memory_mb=500))
    p = predict_proba_df(m, df, [f"x{j}" for j in range(4)],
                          memory_guard=MemoryGuard(max_memory_mb=500))
    # Binary proba returns 1D
    if hasattr(p, "shape"):
        assert p.shape == (300,) or p.shape == (300, 2)
    else:
        # Could be a df; extract shape
        assert len(p) > 0
