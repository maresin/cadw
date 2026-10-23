"""D.1.3: cost-sensitive GBDT (scale_pos_weight) + threshold tuning.

Honest tests: (a) scale_pos_weight affects binary fit -
weighted model predicts strictly more positives on imbalanced data;
(b) predict(threshold) shifts positive count monotonically;
(c) scale_pos_weight<=0 raises.
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard, fit_gbdt_classifier


def _make_imbalanced_df(tmp_path, n=600, seed=0, name="imb"):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    z = 2.0 * X[:, 0] - 1.0 * X[:, 1] - 3.0
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p).astype(np.int64)
    assert y.sum() >= 5 and (1 - y).sum() >= 5
    data = {"x0": X[:, 0].tolist(), "x1": X[:, 1].tolist(), "y": y.tolist()}
    t = pa.table(data)
    pq.write_table(t, Path(tmp_path) / f"{name}.parquet")
    return CadwDataFrame.from_parquet(
        Path(tmp_path) / f"{name}.parquet",
        Path(tmp_path) / f"ds_{name}",
        chunk_size=200,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    ), X, y


def test_scale_pos_weight_nonpositive_raises(tmp_path):
    df, _, _ = _make_imbalanced_df(tmp_path)
    with pytest.raises(ValueError, match="scale_pos_weight"):
        fit_gbdt_classifier(df, ["x0", "x1"], "y", scale_pos_weight=0.0)
    with pytest.raises(ValueError, match="scale_pos_weight"):
        fit_gbdt_classifier(df, ["x0", "x1"], "y", scale_pos_weight=-2.0)


def test_scale_pos_weight_increases_predicted_positives(tmp_path):
    df, X, y = _make_imbalanced_df(tmp_path)
    pos = int(y.sum())
    neg = len(y) - pos
    spw = neg / max(pos, 1)
    m_def = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=30, max_depth=3)
    m_w = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=30, max_depth=3, scale_pos_weight=spw)
    nd = int(m_def.predict(X).sum())
    nw = int(m_w.predict(X).sum())
    assert nw > nd, f"spw={spw:.2f} neg={neg} pos={pos}: default={nd} weighted={nw}"


def test_threshold_shifts_positive_count(tmp_path):
    df, X, _ = _make_imbalanced_df(tmp_path, seed=1)
    m = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=20, max_depth=3)
    assert int(m.predict(X, threshold=0.9).sum()) <= int(m.predict(X, threshold=0.1).sum())


def test_threshold_out_of_range_raises(tmp_path):
    df, X, _ = _make_imbalanced_df(tmp_path, seed=2)
    m = fit_gbdt_classifier(df, ["x0", "x1"], "y", n_estimators=10, max_depth=2)
    with pytest.raises(ValueError, match="threshold"):
        m.predict(X, threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        m.predict(X, threshold=-0.1)
