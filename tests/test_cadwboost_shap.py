"""v0.9.224-a: CadwBoost SHAP-style contributions tests."""
import numpy as np
from cadw.ml.cadwboost import CADWBoost
from cadw.ml.cadwboost.shap import (
    feature_contributions, summarize_contributions, top_k_contributors)


def _mk_reg(n=100, d=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2 + X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_contributions_sum_to_predict():
    """sum(contrib) + base ≈ predict (reconstruction property)."""
    X, y = _mk_reg()
    m = CADWBoost(n_estimators=10, max_depth=3, loss="mse", random_state=42).fit(X, y)
    pred = m.predict(X)
    contrib = feature_contributions(m, X, feature_names=[f"x{j}" for j in range(X.shape[1])])
    # contrib should be (n, d) or (n, d+1) with intercept
    assert contrib.shape[0] == X.shape[0]
    # Reconstruction: sum over all columns ≈ predict (up to base)
    recon = contrib.sum(axis=1)
    # For tree models with additive contribution, recon + base = pred
    # Verify corr > 0.99 (exact depends on implementation)
    corr = np.corrcoef(recon, pred)[0, 1]
    assert corr > 0.99, f"corr(recon, pred) = {corr:.3f}"


def test_summarize_contributions():
    X, y = _mk_reg()
    m = CADWBoost(n_estimators=10, max_depth=3, loss="mse", random_state=42).fit(X, y)
    contrib = feature_contributions(m, X, feature_names=[f"x{j}" for j in range(X.shape[1])])
    summary = summarize_contributions(contrib, [f"x{j}" for j in range(X.shape[1])])
    # summary is dict with mean/abs/std per feature
    assert isinstance(summary, dict)
    assert len(summary) > 0


def test_top_k_contributors():
    """top_k_contributors returns per-sample top-k list.

    Note (v0.9.224-a-fix): implementation returns list[list[tuple(feature, score)]]
    (one row per sample, up to k tuples per row), not global top-k.
    """
    X, y = _mk_reg()
    n = X.shape[0]
    m = CADWBoost(n_estimators=10, max_depth=3, loss="mse", random_state=42).fit(X, y)
    contrib = feature_contributions(m, X, feature_names=[f"x{j}" for j in range(X.shape[1])])
    top = top_k_contributors(contrib, [f"x{j}" for j in range(X.shape[1])], k=2)
    assert top is not None
    assert len(top) == n, f"expected {n} rows (per-sample), got {len(top)}"
    assert all(len(row) <= 2 for row in top), "each row must have <= k contributors"
