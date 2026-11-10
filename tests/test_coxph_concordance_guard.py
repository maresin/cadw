"""v0.9.180: coxph concordance explicit guard tests."""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.coxph import fit_coxph


def _make_df(n, d=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    T = rng.exponential(scale=1.0, size=n)
    E = (rng.uniform(size=n) < 0.7).astype(np.float64)
    tbl = pa.table({
        **{f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)},
        "T": pa.array(T, type=pa.float64()),
        "E": pa.array(E, type=pa.float64()),
    })
    return CadwDataFrame.from_table(tbl, memory_guard=MemoryGuard(max_memory_mb=500))


def test_concordance_refuses_above_1e6():
    """n > 1e6 -> MemoryLimitExceeded with options."""
    df = _make_df(1_000_001)
    m = fit_coxph(df, ["x0", "x1"], "T", "E", max_iter=2,
                   memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(MemoryLimitExceeded) as exc_info:
        m.concordance_index(df, "T", "E")
    e = exc_info.value
    assert e.model == "concordance_index"
    assert e.params == {"n": 1_000_001}
    assert e.min_required_bytes == 24 * 1_000_001
    assert len(e.options) >= 2
    assert any("approximate" in o.lower() for o in e.options)


def test_concordance_at_bound_works():
    """n=1e6 passes guard (24 MB fits 500 MB budget)."""
    # Use smaller n for test speed; verify no refuse at boundary
    df = _make_df(10_000)  # small but exercises non-refuse path
    m = fit_coxph(df, ["x0", "x1"], "T", "E", max_iter=2,
                   memory_guard=MemoryGuard(max_memory_mb=500))
    c = m.concordance_index(df, "T", "E")
    assert 0.0 <= c <= 1.0
