"""v0.9.224-a: CadwBoost OOC extended tests (fit_ooc)."""
import gc
import numpy as np
import pyarrow as pa
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.cadwboost import CADWBoost
from cadw.ml.cadwboost.ooc import fit_ooc, MAX_OOC_BATCH, MAX_OOC_SAMPLE


def _mk_df(n, d=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2 + X[:, 1] + 0.1 * rng.normal(size=n)
    cols = {f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)}
    cols["y"] = pa.array(y, type=pa.float64())
    return CadwDataFrame.from_table(pa.table(cols),
                                     memory_guard=MemoryGuard(max_memory_mb=2000))


def test_ooc_constants():
    assert MAX_OOC_BATCH == 20000
    assert MAX_OOC_SAMPLE == 50000


def test_fit_ooc_matches_memory():
    """fit_ooc vs fit_df(mode='memory') on same data -> same predictions (small n)."""
    from cadw.ml.cadwboost.dataframe import fit_df, predict_df_to_numpy
    df = _mk_df(1000, 4)
    X_cols = [f"x{j}" for j in range(4)]
    mg = MemoryGuard(max_memory_mb=2000)
    m_mem = CADWBoost(n_estimators=5, max_depth=3, loss="mse", random_state=42)
    fit_df(m_mem, df, X_cols, "y", mode="memory", memory_guard=mg)
    m_ooc = CADWBoost(n_estimators=5, max_depth=3, loss="mse", random_state=42)
    fit_ooc(m_ooc, df, X_cols, "y", memory_guard=mg)
    # Same seed, same params — predictions should be close (allow small float drift)
    p_mem = predict_df_to_numpy(m_mem, df, X_cols, memory_guard=mg)
    p_ooc = predict_df_to_numpy(m_ooc, df, X_cols, memory_guard=mg)
    diff = float(np.abs(p_mem - p_ooc).max())
    # OOC vs memory may differ due to binning sample (1000 < MAX_OOC_SAMPLE=50000
    # so reservoir keeps all); tree fits identical if binning identical
    assert diff < 0.01, f"ooc vs memory max diff = {diff:.4f}"


def test_ooc_peak_flat_50k_vs_100k():
    """Peak (OOC) should not grow with n at 2x scale."""
    def _peak(n):
        gc.collect()
        mg = MemoryGuard(max_memory_mb=1000)
        df = _mk_df(n, 4)
        m = CADWBoost(n_estimators=5, max_depth=3, loss="mse", random_state=42)
        r = measure_peak_traced(
            fit_ooc, m, df, [f"x{j}" for j in range(4)], "y",
            memory_guard=mg, mg=mg)
        return int(r["peak_bytes"])
    p50 = _peak(50_000)
    p100 = _peak(100_000)
    ratio = p100 / max(p50, 1)
    assert ratio < 2.0, f"OOC peak ratio 100k/50k = {ratio:.2f} (expected <2)"
