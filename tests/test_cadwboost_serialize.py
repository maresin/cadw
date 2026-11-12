"""v0.9.224-a: CadwBoost serialize roundtrip tests."""
import numpy as np
from cadw.ml.cadwboost import CADWBoost
from cadw.ml.cadwboost.serialize import (
    save_json, load_json, model_to_dict, dict_to_model)


def _mk_reg(n=200, d=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2 + X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_dict_roundtrip():
    X, y = _mk_reg()
    m = CADWBoost(n_estimators=5, max_depth=3, loss="mse", random_state=42).fit(X, y)
    p1 = m.predict(X)
    d = model_to_dict(m)
    m2 = dict_to_model(d, CADWBoost)
    p2 = m2.predict(X)
    np.testing.assert_allclose(p1, p2, atol=1e-10)


def test_json_roundtrip(tmp_path):
    X, y = _mk_reg()
    m = CADWBoost(n_estimators=5, max_depth=3, loss="mse", random_state=42).fit(X, y)
    p1 = m.predict(X)
    path = tmp_path / "model.json"
    save_json(m, str(path))
    m2 = load_json(str(path), CADWBoost)
    p2 = m2.predict(X)
    np.testing.assert_allclose(p1, p2, atol=1e-10)
