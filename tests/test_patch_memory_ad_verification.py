"""Component-wise verification for fit_patch_memory_ad (DEC-033, v0.9.115).

State = decision_scores(n f64) + labels(n i64) + window_scores(n_w f64)
      + training_losses(n_epochs py_float) + memory_bank(n_mem*latent f64)
      + encoder_weights(W1+b1+W2+b2) + header.

Formula (conservative, R2-resolvable):
  3*n*f64 + n_epochs*32 + n_memory*latent_size*f64
  + (w*hidden + hidden + hidden*latent + latent)*f64 + 264
"""
from __future__ import annotations

import numpy as np
import pytest

from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ml.patch_memory_ad import fit_patch_memory_ad


def _make_series(n, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 50.0) + 0.1 * rng.standard_normal(n)


@pytest.mark.parametrize("n", [300, 600, 1200])
def test_state_nbytes_exact(n):
    """state_nbytes() == actual sum of .nbytes (bit-exact)."""
    m = fit_patch_memory_ad(
        _make_series(n), window_size=50, patch_len=10,
        hidden_size=32, latent_size=16, n_memory=100,
        n_epochs=1, random_state=42)
    expected = (
        m.decision_scores.nbytes
        + m.labels.nbytes
        + m.window_scores.nbytes
        + len(m.training_losses) * 32
        + m.memory_bank.nbytes
        + sum(w.nbytes for w in m.encoder_weights)
        + 8 + 256
    )
    assert m.state_nbytes() == expected, (
        f"n={n}: state={m.state_nbytes()} expected={expected}")


def test_state_ratio_linear():
    """state(n2) - state(n1) ~= (n2-n1)*24 (3 n-sized f64 arrays)."""
    n1, n2 = 500, 1000
    s1 = fit_patch_memory_ad(_make_series(n1), window_size=50,
                              n_epochs=1).state_nbytes()
    s2 = fit_patch_memory_ad(_make_series(n2), window_size=50,
                              n_epochs=1).state_nbytes()
    expected_delta = (n2 - n1) * 24
    actual_delta = s2 - s1
    assert 0.95 * expected_delta <= actual_delta <= 1.05 * expected_delta, (
        f"delta={actual_delta} expected~{expected_delta}")


def test_bounded_at_domain():
    """bounded_by_domain n<=1e6: formula at 1e6 must fit in 50 MB."""
    n = 1_000_000
    w, hidden, latent, n_mem, n_ep = 50, 32, 16, 100, 10
    formula_at_1e6 = (
        3 * n * 8
        + n_ep * 32
        + n_mem * latent * 8
        + (w * hidden + hidden + hidden * latent + latent) * 8
        + 264
    )
    assert formula_at_1e6 < 50 * 1024 * 1024, (
        f"state at n=1e6 = {formula_at_1e6} B > 50 MB")


def test_tight_guard_raises():
    """Duck-typed guard (max_memory_mb) -> MemoryLimitExceeded."""
    class _FakeMB:
        max_memory_mb = 0.001
    with pytest.raises(MemoryLimitExceeded):
        fit_patch_memory_ad(_make_series(2000), window_size=50,
                            n_epochs=1, guard=_FakeMB())


def test_df_matches_series():
    """fit_patch_memory_ad(df, col) == fit_patch_memory_ad(series) bit-identical."""
    import pyarrow as pa
    from cadw import CadwDataFrame

    x = _make_series(500, seed=7)
    df = CadwDataFrame.from_table(
        pa.table({"y": pa.array(x, type=pa.float64())}))
    m_series = fit_patch_memory_ad(x, n_epochs=1, random_state=42)
    m_df = fit_patch_memory_ad(df, "y", n_epochs=1, random_state=42)
    np.testing.assert_array_equal(m_series.decision_scores,
                                  m_df.decision_scores)
    np.testing.assert_array_equal(m_series.labels, m_df.labels)
    np.testing.assert_array_equal(m_series.memory_bank, m_df.memory_bank)


def test_output_memory_and_auto():
    """G9: output="memory" and "auto" produce equivalent results."""
    x = _make_series(500, seed=5)
    m_mem = fit_patch_memory_ad(x, n_epochs=1, random_state=42,
                                 output="memory")
    m_auto = fit_patch_memory_ad(x, n_epochs=1, random_state=42,
                                  output="auto")
    np.testing.assert_array_equal(m_mem.decision_scores,
                                  m_auto.decision_scores)


def test_output_disk_raises():
    """output="disk" -> NotImplementedError (SpillArray is v0.9.116+)."""
    x = _make_series(500, seed=5)
    with pytest.raises(NotImplementedError, match="SpillArray.*v0.9.116"):
        fit_patch_memory_ad(x, n_epochs=1, output="disk")
