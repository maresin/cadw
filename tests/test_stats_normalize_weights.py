"""v0.9.148: tests for cadw.ops.stats.normalize_weights (L17)."""
import numpy as np
import pytest

from cadw.ops.stats import normalize_weights


def test_sums_to_one():
    w = np.array([1.0, 2.0, 3.0, 4.0])
    out = normalize_weights(w)
    assert abs(out.sum() - 1.0) < 1e-12
    np.testing.assert_allclose(out, w / w.sum(), rtol=1e-12)


def test_all_zero_returns_zeros():
    w = np.array([0.0, 0.0, 0.0])
    out = normalize_weights(w)
    np.testing.assert_array_equal(out, np.zeros(3))


def test_empty_returns_empty():
    w = np.array([])
    out = normalize_weights(w)
    assert out.size == 0


def test_negative_raises():
    with pytest.raises(ValueError, match="negative"):
        normalize_weights(np.array([1.0, -0.5, 2.0]))


def test_shape_preserved_2d():
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = normalize_weights(w)
    assert out.shape == (2, 2)
    assert abs(out.sum() - 1.0) < 1e-12


def test_single_element():
    w = np.array([5.0])
    out = normalize_weights(w)
    np.testing.assert_allclose(out, [1.0], rtol=1e-12)


def test_int_dtype_coerced():
    w = np.array([1, 2, 3], dtype=np.int64)
    out = normalize_weights(w)
    assert out.dtype == np.float64
    assert abs(out.sum() - 1.0) < 1e-12


def test_tiny_sum_eps_floor():
    """Sum < eps -> zeros (avoids div-by-tiny leading to huge values)."""
    w = np.array([1e-15, 1e-15])  # sum = 2e-15 < 1e-12
    out = normalize_weights(w)
    np.testing.assert_array_equal(out, np.zeros(2))


def test_small_but_above_eps():
    w = np.array([1e-11, 1e-11])  # sum = 2e-11 > 1e-12
    out = normalize_weights(w)
    assert abs(out.sum() - 1.0) < 1e-10
