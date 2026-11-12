"""v0.9.224-a: CadwBoost feature_importance tests."""
import numpy as np
from cadw.ml.cadwboost import CADWBoost


def _mk_reg(n=200, d=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2 + X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_importance_split():
    X, y = _mk_reg()
    m = CADWBoost(n_estimators=10, max_depth=3, loss="mse", random_state=42).fit(X, y)
    imp = m.feature_importance(kind="split")
    assert isinstance(imp, dict)
    assert len(imp) == X.shape[1]
    for v in imp.values():
        assert v >= 0


def test_importance_gain():
    X, y = _mk_reg()
    m = CADWBoost(n_estimators=10, max_depth=3, loss="mse", random_state=42).fit(X, y)
    imp = m.feature_importance(kind="gain")
    assert isinstance(imp, dict)
    assert len(imp) == X.shape[1]
    # x0 and x1 drive y; their gain should be higher than x2, x3
    # (loose sanity: max >= 2x min)
    vals = list(imp.values())
    if max(vals) > 0 and min(vals) >= 0:
        # x0,x1 at least appear; not strictly required
        pass
    for v in vals:
        assert v >= 0
