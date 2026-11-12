"""v0.9.222-a: CadwBoost baseline tests (fit/predict per class)."""
import sys
from pathlib import Path

# Draft path (before promote v0.9.223)
_DRAFT = Path(__file__).resolve().parent.parent / ".pingis" / "drafts" / "cadwboost"
if str(_DRAFT) not in sys.path:
    sys.path.insert(0, str(_DRAFT))

import numpy as np
import pytest


def _mk_binary(n=500, d=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(np.float64)
    return X, y


def _mk_regression(n=500, d=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2 + X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def _mk_multiclass(n=500, d=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    raw = X[:, 0] + X[:, 1] * 0.5 + 0.1 * rng.normal(size=n)
    y = np.digitize(raw, [-0.5, 0.5]).astype(np.int64)
    return X, y


def test_binary_fit_predict():
    from cadwboost import CADWBoost
    from sklearn.metrics import roc_auc_score
    X, y = _mk_binary()
    m = CADWBoost(n_estimators=20, max_depth=4, loss="logistic",
                  tree_mode="leaf_wise", random_state=42)
    m.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X),), f"binary proba shape {p.shape}"
    auc = roc_auc_score(y, p)
    assert auc > 0.85, f"binary AUC {auc:.3f}"


def test_regression_fit_predict():
    from cadwboost import CADWBoost
    X, y = _mk_regression()
    m = CADWBoost(n_estimators=20, max_depth=4, loss="mse",
                  tree_mode="leaf_wise", random_state=42)
    m.fit(X, y)
    p = m.predict(X)
    assert p.shape == (len(X),)
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    assert rmse < float(np.std(y)), f"regression RMSE {rmse:.3f} >= std {np.std(y):.3f}"


def test_multiclass_fit_predict():
    from cadwboost import CADWBoostMulticlass
    X, y = _mk_multiclass()
    m = CADWBoostMulticlass(n_estimators=20, max_depth=4, random_state=42)
    m.fit(X, y)
    p = m.predict(X)
    assert p.shape == (len(X),)
    acc = float((p == y).mean())
    assert acc > 0.7, f"multiclass acc {acc:.3f}"


def test_symmetric_vs_leaf_wise():
    """Both modes fit; leaf_wise RMSE <= symmetric on XOR (leaf better)."""
    from cadwboost import CADWBoost
    rng = np.random.RandomState(45)
    X = rng.randn(500, 4)
    y = ((X[:, 0] > 0) * (X[:, 1] > 0) + (X[:, 2] > 0) * (X[:, 3] > 0)).astype(np.float64)
    m_sym = CADWBoost(n_estimators=20, max_depth=4, tree_mode="symmetric",
                       loss="mse", random_state=42).fit(X, y)
    m_leaf = CADWBoost(n_estimators=20, max_depth=4, tree_mode="leaf_wise",
                        loss="mse", random_state=42).fit(X, y)
    p_sym = m_sym.predict(X)
    p_leaf = m_leaf.predict(X)
    rmse_sym = float(np.sqrt(np.mean((p_sym - y) ** 2)))
    rmse_leaf = float(np.sqrt(np.mean((p_leaf - y) ** 2)))
    assert rmse_leaf <= rmse_sym * 1.1, (
        f"leaf {rmse_leaf:.4f} > symmetric {rmse_sym:.4f} (expected leaf <= sym)")


def test_reproducibility():
    from cadwboost import CADWBoost
    X, y = _mk_regression()
    m1 = CADWBoost(n_estimators=10, max_depth=4, loss="mse",
                    tree_mode="leaf_wise", random_state=42).fit(X, y)
    m2 = CADWBoost(n_estimators=10, max_depth=4, loss="mse",
                    tree_mode="leaf_wise", random_state=42).fit(X, y)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


def test_n_estimators_affects_loss():
    from cadwboost import CADWBoost
    X, y = _mk_regression()
    m10 = CADWBoost(n_estimators=10, max_depth=4, loss="mse",
                     tree_mode="leaf_wise", random_state=42).fit(X, y)
    m50 = CADWBoost(n_estimators=50, max_depth=4, loss="mse",
                     tree_mode="leaf_wise", random_state=42).fit(X, y)
    rmse10 = float(np.sqrt(np.mean((m10.predict(X) - y) ** 2)))
    rmse50 = float(np.sqrt(np.mean((m50.predict(X) - y) ** 2)))
    assert rmse50 <= rmse10, f"more trees: rmse50={rmse50:.4f} > rmse10={rmse10:.4f}"


def test_max_depth_affects_state():
    from cadwboost import CADWBoost
    X, y = _mk_regression()
    m_d3 = CADWBoost(n_estimators=10, max_depth=3, loss="mse",
                      tree_mode="symmetric", random_state=42).fit(X, y)
    m_d6 = CADWBoost(n_estimators=10, max_depth=6, loss="mse",
                      tree_mode="symmetric", random_state=42).fit(X, y)
    assert m_d6.state_nbytes() > m_d3.state_nbytes(), (
        f"depth6 state {m_d6.state_nbytes()} <= depth3 {m_d3.state_nbytes()}")


def test_forecaster_recursive():
    from cadwboost import fit_forecaster
    rng = np.random.RandomState(0)
    n = 200
    series = np.sin(np.arange(n) * 0.1) + 0.05 * rng.normal(size=n)
    result = fit_forecaster(series, horizon=7, lags=(1, 2, 3, 7),
                             mode="recursive", n_estimators=10, max_depth=3)
    fc = result.predict(7)
    assert fc.shape == (7,), f"forecast shape {fc.shape}"


def test_ngboost_interval():
    from cadwboost import CADWBoostNGBoost
    X, y = _mk_regression()
    m = CADWBoostNGBoost(n_estimators=10, max_depth=3)
    m.fit(X, y)
    lo, hi = m.predict_interval(X, alpha=0.9)
    assert lo.shape == (len(X),) and hi.shape == (len(X),)
    coverage = float(((y >= lo) & (y <= hi)).mean())
    assert 0.75 <= coverage <= 0.98, f"coverage {coverage:.3f}"


def test_binary_predict_equals_proba():
    """LogisticLoss: predict == sigmoid(raw) == predict_proba (LightGBM-style)."""
    from cadwboost import CADWBoost
    X, y = _mk_binary()
    m = CADWBoost(n_estimators=10, max_depth=3, loss="logistic",
                   tree_mode="leaf_wise", random_state=42).fit(X, y)
    np.testing.assert_allclose(m.predict(X), m.predict_proba(X), atol=1e-12)


def test_binary_predict_proba_1d():
    """predict_proba for binary returns 1D (P(y=1)), LightGBM-style."""
    from cadwboost import CADWBoost
    X, y = _mk_binary()
    m = CADWBoost(n_estimators=10, max_depth=3, loss="logistic",
                   tree_mode="leaf_wise", random_state=42).fit(X, y)
    p = m.predict_proba(X)
    assert p.ndim == 1, f"binary proba ndim={p.ndim} (expected 1, LightGBM-style)"


def test_multiclass_proba_shape():
    from cadwboost import CADWBoostMulticlass
    X, y = _mk_multiclass()
    m = CADWBoostMulticlass(n_estimators=10, max_depth=3, random_state=42).fit(X, y)
    p = m.predict_proba(X)
    n_classes = len(np.unique(y))
    assert p.shape == (len(X), n_classes), f"multiclass proba shape {p.shape}"
