"""tests/metrics_reference.py: helper self-consistency."""
import math
import numpy as np
import pytest

from tests._helpers.metrics_reference import (
    expected_log_loss_binary,
    expected_log_loss_multiclass,
    expected_roc_auc,
)


def test_roc_auc_perfect():
    assert expected_roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_roc_auc_inverted():
    assert expected_roc_auc([0, 1, 0, 1], [0.5, 0.4, 0.6, 0.3]) == pytest.approx(0.0)


def test_roc_auc_ties():
    """All same score → 0.5."""
    assert expected_roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_roc_auc_partial_ties():
    """y=[0,1,0,1], s=[0.5,0.5,0.7,0.7] → 0.5 (ties grouped)."""
    assert expected_roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.7, 0.7]) == pytest.approx(0.5)


def test_roc_auc_all_positive_nan():
    assert math.isnan(expected_roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_roc_auc_all_negative_nan():
    assert math.isnan(expected_roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_roc_auc_sklearn_equivalence_if_installed():
    pytest.importorskip("sklearn")
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(42)
    for _ in range(5):
        n = 200
        y = rng.integers(0, 2, size=n)
        s = rng.random(size=n)
        assert expected_roc_auc(y, s) == pytest.approx(
            roc_auc_score(y, s), rel=1e-9,
        )


def test_log_loss_binary_matches_sklearn_if_installed():
    pytest.importorskip("sklearn")
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100).astype(float)
    p = rng.random(size=100)
    assert expected_log_loss_binary(y, p) == pytest.approx(
        log_loss(y, p), rel=1e-9,
    )


def test_log_loss_multiclass_known():
    """3 samples, 3 classes: -mean(log(p[i, y_i])) = -mean(log([0.9,0.7,0.6]))."""
    import numpy as np
    y = np.array([0, 1, 2])
    P = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
    ])
    expected = float(-np.log(np.array([0.9, 0.7, 0.6])).mean())
    assert expected_log_loss_multiclass(y, P) == pytest.approx(expected)


def test_log_loss_multiclass_matches_sklearn_if_installed():
    pytest.importorskip("sklearn")
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(1)
    n, K = 200, 3
    y = rng.integers(0, K, size=n)
    P = rng.random(size=(n, K))
    P = P / P.sum(axis=1, keepdims=True)
    assert expected_log_loss_multiclass(y, P) == pytest.approx(
        log_loss(y, P, labels=list(range(K))), rel=1e-9,
    )
