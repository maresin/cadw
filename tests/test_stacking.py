"""v0.9.166: tests for cadw.ml.stacking.fit_stacking."""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.gbdt import fit_gbdt_classifier
from cadw.ml.logistic import fit_logistic
from cadw.ml.stacking import fit_stacking, StackingModel, _STACKING_HEADER


def _a1_data(n=1000, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    tbl = {f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(4)}
    tbl["y"] = pa.array(y, type=pa.int64())
    return CadwDataFrame.from_table(
        pa.table(tbl), memory_guard=MemoryGuard(max_memory_mb=guard_mb))


def test_stacking_runs_and_predicts():
    df = _a1_data(1000)
    mg = MemoryGuard(max_memory_mb=500)
    m = fit_stacking(
        df, ["x0", "x1", "x2", "x3"], "y",
        base_models=[("logreg", fit_logistic), ("gbdt", fit_gbdt_classifier)],
        meta_learner=fit_logistic,
        cv=3, memory_guard=mg,
    )
    assert isinstance(m, StackingModel)
    assert m.state_nbytes() > 0
    proba = m.predict_proba(df)
    assert proba.shape == (1000,)
    assert np.all((proba >= 0) & (proba <= 1))
    pred = m.predict(df)
    assert pred.shape == (1000,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_state_bounded():
    """State >= header (sum of base + meta; legacy models may report 0)."""
    df = _a1_data(500)
    m = fit_stacking(
        df, ["x0", "x1", "x2", "x3"], "y",
        base_models=[("logreg", fit_logistic)],
        meta_learner=fit_logistic,
        cv=3, memory_guard=MemoryGuard(max_memory_mb=500),
    )
    assert m.state_nbytes() >= _STACKING_HEADER


def test_guard_refuses():
    """n=2000, n_base=2 -> OOF 32KB; guard 0.001 MB (1KB) -> refuse."""
    df = _a1_data(2000, guard_mb=0.001)
    with pytest.raises(MemoryLimitExceeded):
        fit_stacking(
            df, ["x0", "x1", "x2", "x3"], "y",
            base_models=[("logreg", fit_logistic),
                         ("gbdt", fit_gbdt_classifier)],
            meta_learner=fit_logistic, cv=3,
            memory_guard=MemoryGuard(max_memory_mb=0.001),
        )


def test_invalid_args():
    df = _a1_data(100)
    with pytest.raises(ValueError, match="cv"):
        fit_stacking(df, ["x0"], "y",
                     base_models=[("logreg", fit_logistic)],
                     meta_learner=fit_logistic, cv=1)
    with pytest.raises(ValueError, match="base_models"):
        fit_stacking(df, ["x0"], "y", base_models=[],
                     meta_learner=fit_logistic, cv=3)
    with pytest.raises(ValueError, match="task"):
        fit_stacking(df, ["x0"], "y",
                     base_models=[("logreg", fit_logistic)],
                     meta_learner=fit_logistic, cv=3,
                     task="hamming")


def test_stacking_improves_on_noisy_data():
    """Meta on OOF should not regress vs. best single base on same df."""
    df = _a1_data(2000, seed=42)
    mg = MemoryGuard(max_memory_mb=500)
    # Fit best single base for comparison
    lr = fit_logistic(df, ["x0", "x1", "x2", "x3"], "y", memory_guard=mg)
    base_pred = _extract(lr, df)
    y = _extract_y(df, "y")
    base_acc = float(np.mean((base_pred >= 0.5).astype(np.int64) == y))

    # Stacked
    stk = fit_stacking(
        df, ["x0", "x1", "x2", "x3"], "y",
        base_models=[("logreg", fit_logistic),
                     ("gbdt", fit_gbdt_classifier)],
        meta_learner=fit_logistic, cv=5, memory_guard=mg,
    )
    stk_pred = stk.predict(df)
    stk_acc = float(np.mean(stk_pred == y))

    # Stacked should be at least as good as best base within 2pp.
    assert stk_acc >= base_acc - 0.02, (
        f"stacked_acc={stk_acc:.3f} vs base_acc={base_acc:.3f}")


def _extract(model, df):
    """Extract (n,) probs from a CadwDataFrame-returning predict_proba."""
    try:
        pred = model.predict_proba(df, X_cols=["x0", "x1", "x2", "x3"])
    except TypeError:
        pred = model.predict_proba(df)
    if isinstance(pred, np.ndarray):
        return np.asarray(pred, dtype=np.float64)
    parts = []
    for b in pred._iter_filtered_batches():
        parts.append(b.column(b.num_columns - 1).to_numpy(zero_copy_only=False))
    return np.concatenate(parts).astype(np.float64)


def _extract_y(df, y_col):
    parts = []
    for b in df._iter_filtered_batches():
        parts.append(b.column(y_col).to_numpy(zero_copy_only=False))
    return np.concatenate(parts).astype(np.int64)
