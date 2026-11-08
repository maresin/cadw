"""v0.9.154: tests for cadw.ml._ltr_common (P1 L13)."""
import numpy as np
import pytest

from cadw.ml._ltr_common import (
    ranknet_grad, listnet_grad, lambdarank_grad, _ndcg_at_k,
)


# ============================================================================
# ranknet_grad
# ============================================================================

def test_ranknet_perfect_ordering_zero_grad():
    """If scores match labels' order perfectly, gradient should be small."""
    y = np.array([3.0, 2.0, 1.0])
    s = y.copy()
    g = ranknet_grad(s, y, sigma=1.0)
    # sigmoid of positive gap -> 1 - sigmoid -> small
    # Not exactly zero because sigmoid(sigma * gap) < 1 for finite gap
    assert np.all(np.abs(g) < 0.5)


def test_ranknet_sums_to_zero():
    """Sum over items of pairwise gradient should be 0."""
    rng = np.random.RandomState(0)
    s = rng.normal(size=10)
    y = rng.randint(0, 3, size=10).astype(np.float64)
    g = ranknet_grad(s, y)
    assert abs(g.sum()) < 1e-10


def test_ranknet_antisymmetric_under_swap():
    """If y_i > y_j but s_i < s_j (wrong order), grad should push s_i up."""
    y = np.array([1.0, 0.0])
    s = np.array([0.0, 1.0])  # reversed
    g = ranknet_grad(s, y, sigma=1.0)
    # grad on i (y_i > y_j, s_i < s_j) should be positive (increase s_i)
    assert g[0] > 0
    assert g[1] < 0


def test_ranknet_empty():
    assert ranknet_grad(np.empty(0), np.empty(0)).size == 0
    assert ranknet_grad(np.array([1.0]), np.array([1.0])).size == 1


def test_ranknet_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        ranknet_grad(np.zeros(3), np.zeros(4))


def test_ranknet_invalid_sigma():
    with pytest.raises(ValueError, match="sigma"):
        ranknet_grad(np.zeros(3), np.zeros(3), sigma=0)


# ============================================================================
# listnet_grad
# ============================================================================

def test_listnet_grad_sums_to_zero():
    rng = np.random.RandomState(1)
    s = rng.normal(size=20)
    y = rng.normal(size=20)
    g = listnet_grad(s, y)
    assert abs(g.sum()) < 1e-10


def test_listnet_grad_zero_at_perfect_match():
    """If s == y, softmax(s) == softmax(y), grad = 0."""
    s = np.array([1.0, 2.0, 3.0])
    g = listnet_grad(s, s.copy())
    np.testing.assert_allclose(g, np.zeros(3), atol=1e-12)


def test_listnet_grad_direction():
    """If s underweights the top item, grad should push it up."""
    s = np.array([0.0, 0.0, 0.0])  # flat
    y = np.array([0.0, 0.0, 5.0])  # top item is #2
    g = listnet_grad(s, y)
    # softmax(y) concentrates on item 2 -> P_s(2) << P_y(2) -> g[2] < 0
    # Wait: grad = P_s - P_y. So g[2] = P_s[2] - P_y[2] < 0 (need push up).
    # Actually to *increase* P_s[2] we'd want g[2] > 0 if grad is for
    # *ascent*. But listnet_grad returns CE gradient (descent direction
    # for -CE), which is g = P_s - P_y. To reduce CE, s moves + g * lr,
    # which increases P_s where P_s < P_y.
    # Test: g[2] < 0 (P_s < P_y at that position)
    assert g[2] < 0


def test_listnet_grad_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        listnet_grad(np.zeros(3), np.zeros(4))


def test_listnet_grad_empty():
    assert listnet_grad(np.empty(0), np.empty(0)).size == 0


# ============================================================================
# lambdarank_grad
# ============================================================================

def test_lambdarank_sums_to_zero():
    rng = np.random.RandomState(2)
    s = rng.normal(size=10)
    y = rng.randint(0, 4, size=10).astype(np.float64)
    g = lambdarank_grad(s, y, k=10)
    assert abs(g.sum()) < 1e-10


def test_lambdarank_direction():
    """Wrong order in top-k pushes highest-relevance item up."""
    y = np.array([0.0, 5.0, 0.0])
    s = np.array([1.0, 0.0, 0.0])  # wrong: item 0 ranked above item 1
    g = lambdarank_grad(s, y, k=10)
    # item 1 (most relevant) should be pushed up (positive g)
    # item 0 should be pushed down (negative g)
    assert g[1] > 0
    assert g[0] < 0


def test_lambdarank_invalid_args():
    with pytest.raises(ValueError, match="sigma"):
        lambdarank_grad(np.zeros(3), np.zeros(3), sigma=-1)
    with pytest.raises(ValueError, match="k"):
        lambdarank_grad(np.zeros(3), np.zeros(3), k=0)


def test_lambdarank_empty_and_single():
    assert lambdarank_grad(np.empty(0), np.empty(0)).size == 0
    g = lambdarank_grad(np.array([1.0]), np.array([0.0]))
    assert g.size == 1
    assert g[0] == 0.0


# ============================================================================
# _ndcg_at_k (helper)
# ============================================================================

def test_ndcg_perfect():
    """Perfect ordering -> NDCG = 1."""
    y = np.array([3.0, 2.0, 1.0, 0.0])
    s = y.copy()
    assert abs(_ndcg_at_k(y, s, k=4) - 1.0) < 1e-10


def test_ndcg_reversed():
    """Reversed ordering -> NDCG < 1."""
    y = np.array([3.0, 2.0, 1.0, 0.0])
    s = -y  # reversed
    assert _ndcg_at_k(y, s, k=4) < 0.8


def test_ndcg_k_truncation():
    """Smaller k focuses on top; both perfect for s=y."""
    y = np.array([3.0, 2.0, 1.0, 0.0])
    assert abs(_ndcg_at_k(y, y, k=1) - 1.0) < 1e-10
    assert abs(_ndcg_at_k(y, y, k=2) - 1.0) < 1e-10


def test_ndcg_all_zero_labels():
    """All zero labels -> IDCG = 0 -> returns 0."""
    y = np.zeros(5)
    s = np.arange(5.0)
    assert _ndcg_at_k(y, s, k=3) == 0.0
