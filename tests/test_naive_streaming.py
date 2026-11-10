"""v0.9.186: Naive streaming fit (DEC-070 P1)."""
import gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ts.models import NaiveForecaster


def _mk_df(y_list, guard_mb=500):
    arr = np.asarray(y_list, dtype=np.float64)
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(arr, type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=guard_mb))


@pytest.mark.parametrize("n", [10, 100, 1000])
@pytest.mark.parametrize("strategy,period", [
    ("last", None), ("seasonal", 7), ("drift", None),
])
def test_naive_fit_streaming_bit_identical(n, strategy, period):
    """Streaming fit == save_y=True fit for predict + in_sample + state."""
    rng = np.random.RandomState(0)
    y = rng.normal(100, 5, n)
    df = _mk_df(y)
    m_stream = NaiveForecaster(strategy=strategy, period=period).fit(df, "y")
    m_saved = NaiveForecaster(
        strategy=strategy, period=period).fit(df, "y", save_y=True)

    np.testing.assert_array_equal(
        m_stream.predict(5), m_saved.predict(5))
    np.testing.assert_array_equal(
        m_stream.predict_in_sample(df), m_saved.predict_in_sample())
    assert m_stream.state_nbytes() < m_saved.state_nbytes()


def test_naive_fit_streaming_ooc_peak():
    """v0.9.186: fit peak O(batch), not O(n).

    Test at wide n with batch fixed (internal cap 10_000):
    - n=10k -> 1 batch of 10k (peak = per-batch)
    - n=100k -> 10 batches of 10k (peak = per-batch, same)

    Ratio should be < 4 (measurement noise + allocator arena).
    NOT comparing n=1k (1 batch of 1k = smaller batch) vs 100k —
    that measures batch_size growth, not n-growth.
    """
    import statistics
    rng = np.random.RandomState(0)

    def _peak(n):
        y = rng.normal(100, 5, n)
        df = _mk_df(y)
        peaks = []
        for _ in range(3):
            gc.collect()
            m = NaiveForecaster(strategy="drift")
            r = measure_peak_traced(
                m.fit, df, "y", mg=df._memory_guard)
            peaks.append(int(r["peak_bytes"]))
        return int(statistics.median(peaks))

    p_small = _peak(10_000)
    p_large = _peak(100_000)
    ratio = p_large / max(p_small, 1)
    # O(batch): both fit 10k-batches; growth only from allocator churn.
    assert ratio < 4.0, (
        f"fit peak ratio {ratio:.2f} (n=10k -> 100k, batch=10k); "
        f"peak_small={p_small}, peak_large={p_large}")
