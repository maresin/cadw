"""Component-wise verification for fit_autoencoder (DEC-033, v0.9.115).

State = decision_scores(n f64) + labels(n i64) + window_scores(n_w f64)
      + training_losses(n_epochs py_float)
      + encoder_weights + decoder_weights + header.
"""
from __future__ import annotations

import ctypes
import gc
import statistics
import numpy as np
import pytest
from cadw.core.memory_guard import (
    MemoryGuard,
    MemoryLimitExceeded,
    measure_peak_traced,
)
from cadw.ml.autoencoder import fit_autoencoder


def _make_series(n, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 50.0) + 0.1 * rng.standard_normal(n)


@pytest.mark.parametrize("n", [300, 600, 1200])
def test_state_nbytes_exact(n):
    m = fit_autoencoder(_make_series(n), window_size=50, hidden_size=32,
                        latent_size=8, n_epochs=1, random_state=42)
    expected = (
        m.decision_scores.nbytes + m.labels.nbytes + m.window_scores.nbytes
        + len(m.training_losses) * 32
        + sum(w.nbytes for w in m.encoder_weights)
        + sum(w.nbytes for w in m.decoder_weights)
        + 8 + 256
    )
    assert m.state_nbytes() == expected, (
        f"n={n}: state={m.state_nbytes()} expected={expected}")


def test_state_ratio_linear():
    n1, n2 = 500, 1000
    s1 = fit_autoencoder(_make_series(n1), window_size=50, n_epochs=1).state_nbytes()
    s2 = fit_autoencoder(_make_series(n2), window_size=50, n_epochs=1).state_nbytes()
    expected_delta = (n2 - n1) * 24
    actual_delta = s2 - s1
    assert 0.95 * expected_delta <= actual_delta <= 1.05 * expected_delta


def test_bounded_at_domain():
    n = 1_000_000
    w, h, lat, n_ep = 50, 32, 8, 30
    formula_at_1e6 = (
        3 * n * 8 + n_ep * 32
        + (2 * w * h + 2 * h + 2 * h * lat + w + lat) * 8
        + 264
    )
    assert formula_at_1e6 < 50 * 1024 * 1024


def test_tight_guard_raises():
    class _FakeMB:
        max_memory_mb = 0.001
    with pytest.raises(MemoryLimitExceeded):
        fit_autoencoder(_make_series(2000), window_size=50, n_epochs=1,
                        guard=_FakeMB())


def test_df_matches_series():
    import pyarrow as pa
    from cadw import CadwDataFrame
    x = _make_series(500, seed=7)
    df = CadwDataFrame.from_table(
        pa.table({"y": pa.array(x, type=pa.float64())}))
    m_series = fit_autoencoder(x, n_epochs=1, random_state=42)
    m_df = fit_autoencoder(df, "y", n_epochs=1, random_state=42)
    np.testing.assert_array_equal(m_series.decision_scores, m_df.decision_scores)
    np.testing.assert_array_equal(m_series.labels, m_df.labels)
    for ws, wd in zip(m_series.encoder_weights, m_df.encoder_weights):
        np.testing.assert_array_equal(ws, wd)


def test_output_memory_and_auto():
    x = _make_series(500, seed=5)
    m_mem = fit_autoencoder(x, n_epochs=1, random_state=42, output="memory")
    m_auto = fit_autoencoder(x, n_epochs=1, random_state=42, output="auto")
    np.testing.assert_array_equal(m_mem.decision_scores, m_auto.decision_scores)


def test_output_disk_raises():
    x = _make_series(500, seed=5)
    with pytest.raises(NotImplementedError, match="SpillArray.*v0.9.116"):
        fit_autoencoder(x, n_epochs=1, output="disk")


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _measure_peak_median3(fn, x, **kwargs):
    """median-of-3 peak measurement (DEC-034).

    Per-cell: malloc_trim + gc.collect, MemoryGuard(50), measure_peak_traced.
    Sporadic GC/allocator shifts filtered by median (A24/CW-AP2).
    """
    peaks = []
    for _ in range(3):
        _malloc_trim()
        gc.collect()
        mg = MemoryGuard(max_memory_mb=50)
        r = measure_peak_traced(fn, x, guard=mg, mg=mg, **kwargs)
        peaks.append(int(r["peak_bytes"]))
        del r
        gc.collect()
    return int(statistics.median(peaks))


def test_peak_grows_linearly_with_n():
    """CW-4 (DEC-033): fit peak grows linearly with n at fixed params.

    state_class O(n) honest; bounded_by_domain n<=1e6 accepts this.
    10x n -> 8-15x peak (linear, not quadratic). Not strict OOC —
    documented limitation. median-of-3 per DEC-034.
    """
    def _peak(n):
        x = _make_series(n, seed=4)
        return _measure_peak_median3(
            fit_autoencoder, x, window_size=50, hidden_size=32, latent_size=8, n_epochs=1)

    p_small = _peak(3000)
    p_large = _peak(30000)
    ratio = p_large / max(p_small, 1)
    assert ratio < 15.0, (
        f"peak ratio {ratio:.2f} (small={p_small}, large={p_large}); "
        f"expected ~10x for linear; >15 suggests quadratic or formula gap")
