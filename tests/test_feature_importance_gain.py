"""v0.7.9: feature_importance(gain) — total gain, not split count."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import (
    CadwDataFrame, MemoryGuard,
    fit_gbdt_regressor, fit_gbdt_classifier, fit_gbdt_multiclass,
)
from cadw.ml.gbdt import compute_feature_importance


def make_df(tmp_path, data, chunk=50, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _reg_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n) * 0.05
    # Only x0 matters
    y = 3.0 * x0 + 0.1 * rng.normal(size=n)
    return {"x0": x0.tolist(), "x1": x1.tolist(), "y": y.tolist()}


# ---------- helper ----------

def test_compute_feature_importance_invalid_type(tmp_path):
    data = _reg_data()
    df = make_df(tmp_path, data)
    model = fit_gbdt_regressor(df, ["x0", "x1"], "y", n_estimators=5)
    with pytest.raises(ValueError, match="importance_type"):
        compute_feature_importance(model, importance_type="shap")


def test_compute_feature_importance_non_gbdt():
    class FakeModel:
        pass
    with pytest.raises(TypeError, match="GBDT"):
        compute_feature_importance(FakeModel())


# ---------- gain > 0 for useful feature ----------

def test_gain_important_feature_higher(tmp_path):
    """x0 predicts y, x1 is noise → gain[x0] >> gain[x1]."""
    data = _reg_data(n=300, seed=1)
    df = make_df(tmp_path, data)
    model = fit_gbdt_regressor(
        df, ["x0", "x1"], "y",
        n_estimators=20, max_depth=3, learning_rate=0.2, bins=16,
    )
    imp_gain = compute_feature_importance(model, importance_type="gain")
    assert imp_gain["x0"] > imp_gain["x1"] * 2, (
        f"gain[x0]={imp_gain['x0']:.3f}, gain[x1]={imp_gain['x1']:.3f}"
    )


def test_gain_sorted_ascending(tmp_path):
    """Dict sorted by value ascending (for horizontal barh)."""
    data = _reg_data(n=300, seed=2)
    df = make_df(tmp_path, data)
    model = fit_gbdt_regressor(df, ["x0", "x1"], "y", n_estimators=10)
    imp = compute_feature_importance(model, importance_type="gain")
    vals = list(imp.values())
    assert vals == sorted(vals), f"not ascending: {imp}"


def test_split_and_gain_differ(tmp_path):
    """gain vs split count — different values (generally)."""
    data = _reg_data(n=300, seed=3)
    df = make_df(tmp_path, data)
    model = fit_gbdt_regressor(df, ["x0", "x1"], "y", n_estimators=20)
    imp_gain = compute_feature_importance(model, importance_type="gain")
    imp_split = compute_feature_importance(model, importance_type="split")
    # Gain should not be all-integer (split is); if x0 gain == split x0
    # it's a suspicious coincidence but not a failure.
    # Better: verify gain[x0] != split[x0] in general.
    assert imp_gain["x0"] != float(imp_split["x0"]) or imp_gain["x0"] == 0.0


# ---------- top_k ----------

def test_top_k_truncates(tmp_path):
    data = _reg_data(n=300, seed=4)
    df = make_df(tmp_path, data)
    model = fit_gbdt_regressor(df, ["x0", "x1"], "y", n_estimators=10)
    imp = compute_feature_importance(
        model, importance_type="gain", top_k=1,
    )
    assert len(imp) == 1


# ---------- zero model ----------

def test_gain_zero_when_no_splits(tmp_path):
    """Constant y → no splits → gain all 0."""
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x0": rng.normal(size=100).tolist(),
        "x1": rng.normal(size=100).tolist(),
        "y": [5.0] * 100,
    })
    model = fit_gbdt_regressor(
        df, ["x0", "x1"], "y",
        n_estimators=5, max_depth=3, learning_rate=0.2, bins=16,
    )
    imp = compute_feature_importance(model, importance_type="gain")
    assert sum(imp.values()) == 0


# ---------- multiclass ----------

def test_multiclass_gain_aggregates_across_classes(tmp_path):
    """GBDTMulticlassClassifier: gain summed across K classes."""
    rng = np.random.default_rng(0)
    n = 300
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)]) * 4.0
    y = rng.integers(0, 3, size=n)
    X = centers[y] + rng.normal(scale=0.5, size=(n, 2))
    data = {
        "x0": X[:, 0].tolist(),
        "x1": X[:, 1].tolist(),
        "y": y.astype(np.int64).tolist(),
    }
    df = make_df(tmp_path, data, chunk=100)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y",
        n_estimators=10, max_depth=3, learning_rate=0.2, bins=16,
    )
    imp = compute_feature_importance(model, importance_type="gain")
    assert sum(imp.values()) > 0
    assert "x0" in imp and "x1" in imp


# ---------- adapters ----------

def test_matplotlib_feature_importance_gain(tmp_path):
    """matplotlib adapter uses gain by default."""
    pytest.importorskip("matplotlib")
    data = _reg_data(n=300, seed=6)
    df = make_df(tmp_path, data)
    model = fit_gbdt_regressor(df, ["x0", "x1"], "y", n_estimators=10)

    from cadw.viz.matplotlib_adapter import feature_importance
    fig, ax = feature_importance(model)
    # Save to confirm no crash
    out = tmp_path / "fi.png"
    fig.savefig(str(out), dpi=60, bbox_inches="tight")
    assert out.stat().st_size > 500


def test_matplotlib_feature_importance_split_compat(tmp_path):
    """importance_type='split' backward compat."""
    pytest.importorskip("matplotlib")
    data = _reg_data(n=200, seed=7)
    df = make_df(tmp_path, data)
    model = fit_gbdt_regressor(df, ["x0", "x1"], "y", n_estimators=10)
    from cadw.viz.matplotlib_adapter import feature_importance
    fig, ax = feature_importance(model, importance_type="split")
    out = tmp_path / "fi_split.png"
    fig.savefig(str(out), dpi=60, bbox_inches="tight")
    assert out.stat().st_size > 500


def test_plotly_feature_importance_gain(tmp_path):
    pytest.importorskip("plotly")
    data = _reg_data(n=200, seed=8)
    df = make_df(tmp_path, data)
    model = fit_gbdt_regressor(df, ["x0", "x1"], "y", n_estimators=10)
    from cadw.viz.plotly_adapter import feature_importance
    fig = feature_importance(model)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert fig.data[0].orientation == "h"


def test_plotly_feature_importance_split_compat(tmp_path):
    pytest.importorskip("plotly")
    data = _reg_data(n=200, seed=9)
    df = make_df(tmp_path, data)
    model = fit_gbdt_regressor(df, ["x0", "x1"], "y", n_estimators=10)
    from cadw.viz.plotly_adapter import feature_importance
    fig = feature_importance(model, importance_type="split")
    assert len(fig.data) == 1


# ---------- backward compat: pre-v0.7.9 4-tuple spec ----------

def test_gain_fallback_4_tuple_spec(tmp_path):
    """Old 4-tuple spec (pre-v0.7.9 pickled models) — gain falls back
    to 0 for those trees, split still works."""
    data = _reg_data(n=200, seed=10)
    df = make_df(tmp_path, data)
    model = fit_gbdt_regressor(df, ["x0", "x1"], "y", n_estimators=5)

    # Simulate legacy: strip gain (5th element) from each spec
    for tree in model.trees_:
        for nid, spec in tree["structure"].items():
            if spec is None:
                continue
            tree["structure"][nid] = tuple(spec[:4])

    # gain — should be 0 (legacy 4-tuple, gain unavailable)
    imp_gain = compute_feature_importance(model, importance_type="gain")
    assert sum(imp_gain.values()) == 0
    # split — still works
    imp_split = compute_feature_importance(model, importance_type="split")
    assert sum(imp_split.values()) > 0

    # Predict still works (backward compat in _assign_leaf_from_array)
    X = np.column_stack([
        model.predict(np.zeros((1, 2)))  # smoke test
    ])


# ---------- tree spec has gain ----------

def test_fitted_tree_spec_has_gain():
    """After v0.7.9, spec is 5-tuple with gain (float)."""
    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as td:
        td = P(td)
        rng = np.random.default_rng(0)
        x = rng.normal(size=100)
        y = 2.0 * x
        df = make_df(td, {"x0": x.tolist(), "y": y.tolist()})
        model = fit_gbdt_regressor(df, ["x0"], "y", n_estimators=5)
        # Every non-None spec is 5-tuple
        for tree in model.trees_:
            for nid, spec in tree["structure"].items():
                if spec is None:
                    continue
                assert len(spec) == 5, f"spec = {spec}"
                assert isinstance(spec[4], float), f"gain is {type(spec[4])}"

def test_all_spec_unpack_uses_slice():
    """Все `j, thr, left, right = spec` в cadw/ml/gbdt.py должны быть
    `spec[:4]` (или `spec[:5]`).

    Регрессия v0.7.9-fix: 4-tuple unpack падает после перехода на 5-tuple
    (gain appended). `spec[:4]` — forward-compatible.
    """
    import ast as _ast
    import inspect as _inspect
    from cadw.ml import gbdt as gbdt_mod

    src = _inspect.getsource(gbdt_mod)
    tree = _ast.parse(src)

    violations: list[str] = []

    def is_target_tuple(node):
        """Check for tuple name pattern `j, thr, left, right` on left side."""
        if not isinstance(node, _ast.Tuple):
            return False
        elts = node.elts
        if len(elts) != 4:
            return False
        names = [e.id for e in elts if isinstance(e, _ast.Name)]
        return names == ["j", "thr", "left", "right"]

    def is_spec_slice(node):
        """spec[:4] — Subscript(spec, Slice(upper=4))."""
        if not isinstance(node, _ast.Subscript):
            return False
        if not (isinstance(node.value, _ast.Name) and node.value.id == "spec"):
            return False
        sl = node.slice
        if not isinstance(sl, _ast.Slice):
            return False
        if sl.lower is not None or sl.step is not None:
            return False
        if sl.upper is None:
            return True  # spec[:] — allowed but rare
        return isinstance(sl.upper, _ast.Constant) and sl.upper.value in (4, 5)

    for fn_node in _ast.walk(tree):
        if not isinstance(fn_node, _ast.FunctionDef):
            continue
        for node in _ast.walk(fn_node):
            if not isinstance(node, _ast.Assign):
                continue
            if len(node.targets) != 1:
                continue
            if not is_target_tuple(node.targets[0]):
                continue
            if not is_spec_slice(node.value):
                violations.append(
                    f"{fn_node.name}: line {node.lineno}, "
                    f"`j, thr, left, right = {_ast.unparse(node.value)}` "
                    f"(expected spec[:4] or spec[:5])"
                )
    assert not violations, (
        "4-unpack of spec must use spec[:4] (forward-compat with 5-tuple):\n"
        + "\n".join(violations)
    )

