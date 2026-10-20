"""v0.9.2a: metrics E2E (slow)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


@pytest.mark.slow
def test_metrics_100k_regression(tmp_path):
    """100K rows — regression metrics (r2/rmse/mae), continuous y.

    §13.4/163: separate regression semantics (continuous y) from
    log_loss semantics (binary y). Prior test mixed both in one call —
    log_loss formula produced invalid negative output for continuous y.
    """
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    yp = y + rng.normal(scale=0.3, size=n)

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"y": y.tolist(), "yp": yp.tolist()}),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["r2", "rmse", "mae"],
    )

    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ref_r2 = 1.0 - ss_res / ss_tot
    ref_rmse = float(np.sqrt(ss_res / n))
    ref_mae = float(np.abs(y - yp).mean())

    assert res["r2"] == pytest.approx(ref_r2, rel=1e-9)
    assert res["rmse"] == pytest.approx(ref_rmse, rel=1e-9)
    assert res["mae"] == pytest.approx(ref_mae, rel=1e-9)


@pytest.mark.slow
def test_metrics_100k_log_loss_binary(tmp_path):
    """100K rows — log_loss with binary y.

    Binary y_bin = (y > 0); proba p = sigmoid(y) — calibrated-ish.
    """
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    y_bin = (y > 0).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-y))  # sigmoid(y) — well-calibrated

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"yb": y_bin.tolist(), "p": p.tolist()}),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(y_true="yb", y_proba="p", metrics=["log_loss"])

    # Reference (numpy, matches sklearn)
    eps = 1e-15
    p_c = np.clip(p, eps, 1.0 - eps)
    ref_ll = float(-(y_bin * np.log(p_c)
                     + (1 - y_bin) * np.log(1 - p_c)).mean())

    assert res["log_loss"] == pytest.approx(ref_ll, rel=1e-9)
    assert 0 < res["log_loss"] < 1.0  # binary CE ∈ (0, ∞); sigmoid-calibrated ~0.35


@pytest.mark.slow
def test_metrics_100k_classification_multiclass(tmp_path):
    """100K rows, 3-class, all metrics verified vs manual numpy."""
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    yp = y.copy()
    flip_mask = rng.random(n) < 0.2  # 20% errors
    yp[flip_mask] = rng.integers(0, 3, size=int(flip_mask.sum()))

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"y": y.tolist(), "yp": yp.tolist()}),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["accuracy", "precision", "recall", "f1"],
        classes=[0, 1, 2], average="macro",
    )

    # Manual reference
    K = 3
    tp = np.zeros(K, dtype=np.int64)
    fp = np.zeros(K, dtype=np.int64)
    fn = np.zeros(K, dtype=np.int64)
    for k in range(K):
        tp[k] = int(((y == k) & (yp == k)).sum())
        fp[k] = int(((y != k) & (yp == k)).sum())
        fn[k] = int(((y == k) & (yp != k)).sum())
    pp = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    pr = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    denom = pp + pr
    pf1 = np.where(denom > 0, 2 * pp * pr / denom, 0.0)
    acc = float(tp.sum()) / float(tp.sum() + fp.sum())
    assert res["accuracy"] == pytest.approx(acc, rel=1e-9)
    assert res["precision"] == pytest.approx(float(pp.mean()), rel=1e-9)
    assert res["recall"] == pytest.approx(float(pr.mean()), rel=1e-9)
    assert res["f1"] == pytest.approx(float(pf1.mean()), rel=1e-9)


@pytest.mark.slow
def test_metrics_100k_roc_auc_and_multiclass(tmp_path):
    """100K rows: roc_auc + multiclass log_loss, memory independent of N."""
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    # well-separated probas via one-hot + noise
    P = np.full((n, 3), 0.05)
    P[np.arange(n), y] = 0.9
    P += rng.normal(scale=0.02, size=(n, 3))
    P = np.clip(P, 1e-6, None)
    P = P / P.sum(axis=1, keepdims=True)

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "y": y.tolist(),
            "p0": P[:, 0].tolist(),
            "p1": P[:, 1].tolist(),
            "p2": P[:, 2].tolist(),
        }),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(
        y_true="y", y_proba=["p0", "p1", "p2"],
        metrics=["log_loss", "roc_auc"], classes=[0, 1, 2],
    )

    # Reference: multiclass log_loss (manual)
    eps = 1e-15
    P_c = np.clip(P, eps, 1.0 - eps)
    ref_ll = float(-np.log(P_c[np.arange(n), y]).mean())
    assert res["log_loss"] == pytest.approx(ref_ll, rel=1e-9)

    # Reference: macro roc_auc (3 ovr passes)
    from tests._helpers.metrics_reference import expected_roc_auc
    aucs = []
    for cls in (0, 1, 2):
        y_bin = (y == cls).astype(np.int64)
        aucs.append(expected_roc_auc(y_bin, P[:, cls]))
    ref_auc = float(np.mean(aucs))
    assert res["roc_auc"] == pytest.approx(ref_auc, rel=1e-6)
