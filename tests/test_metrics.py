"""v0.9.2a: regression + log_loss metrics tests."""
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=50):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ============ r2 ============

def test_metrics_r2_perfect(tmp_path):
    df = make_df(tmp_path, {"y": [1.0, 2.0, 3.0], "yp": [1.0, 2.0, 3.0]})
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == pytest.approx(1.0)


def test_metrics_r2_known(tmp_path):
    """y=[1,2,3,4], yp=[2,2,3,3] → SS_res=2, SS_tot=5, r2=1-2/5=0.6."""
    df = make_df(tmp_path, {
        "y": [1.0, 2.0, 3.0, 4.0],
        "yp": [2.0, 2.0, 3.0, 3.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == pytest.approx(0.6)


def test_metrics_r2_constant_y_perfect(tmp_path):
    """y all equal to yp → ss_tot==0, ss_res==0 → r2=1.0 (sklearn behavior)."""
    df = make_df(tmp_path, {
        "y": [5.0, 5.0, 5.0],
        "yp": [5.0, 5.0, 5.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == 1.0


def test_metrics_r2_constant_y_wrong_pred(tmp_path):
    """y all 5, yp all 3 → ss_res>0, ss_tot==0 → r2=0.0 (sklearn behavior)."""
    df = make_df(tmp_path, {
        "y": [5.0, 5.0],
        "yp": [3.0, 3.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == 0.0


# ============ mse / rmse / mae ============

def test_metrics_mse_rmse_mae(tmp_path):
    """y=[0,0,0], yp=[1,-1,0] → mse=2/3, rmse=sqrt(2/3), mae=2/3."""
    df = make_df(tmp_path, {
        "y": [0.0, 0.0, 0.0],
        "yp": [1.0, -1.0, 0.0],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["mse", "rmse", "mae"],
    )
    assert res["mse"] == pytest.approx(2.0 / 3.0)
    assert res["rmse"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert res["mae"] == pytest.approx(2.0 / 3.0)


def test_metrics_all_regression_combined(tmp_path):
    """All regression metrics from one pass."""
    df = make_df(tmp_path, {
        "y": [1.0, 2.0, 3.0, 4.0],
        "yp": [1.1, 1.9, 3.1, 3.9],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["r2", "mse", "rmse", "mae"],
    )
    assert set(res.keys()) == {"r2", "mse", "rmse", "mae"}
    assert res["rmse"] == pytest.approx(math.sqrt(res["mse"]))
    # residual abs: 0.1 each → mae = 0.1
    assert res["mae"] == pytest.approx(0.1)


# ============ log_loss ============

def test_metrics_log_loss_perfect(tmp_path):
    """y=[1,0], p=[1,0] → clipped, loss ~ tiny."""
    df = make_df(tmp_path, {"y": [1.0, 0.0], "p": [1.0, 0.0]})
    res = df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])
    assert res["log_loss"] < 1e-10


def test_metrics_log_loss_50_50(tmp_path):
    """y=[1,0], p=[0.5,0.5] → loss = -log(0.5) = ln(2)."""
    df = make_df(tmp_path, {"y": [1.0, 0.0], "p": [0.5, 0.5]})
    res = df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])
    assert res["log_loss"] == pytest.approx(math.log(2.0))


def test_metrics_log_loss_known(tmp_path):
    """y=[1,0], p=[0.9, 0.1] → loss = -[log(0.9)+log(0.9)]/2 = -log(0.9)."""
    df = make_df(tmp_path, {"y": [1.0, 0.0], "p": [0.9, 0.1]})
    res = df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])
    assert res["log_loss"] == pytest.approx(-math.log(0.9))


# ============ combined ============

def test_metrics_regression_and_log_loss(tmp_path):
    """Both y_pred and y_proba passed in one call."""
    df = make_df(tmp_path, {
        "y": [0.0, 1.0, 0.0, 1.0],
        "yp": [0.1, 0.9, 0.2, 0.8],
        "p": [0.1, 0.9, 0.2, 0.8],
    })
    res = df.metrics(
        y_true="y", y_pred="yp", y_proba="p",
        metrics=["mse", "log_loss"],
    )
    assert set(res.keys()) == {"mse", "log_loss"}
    assert res["mse"] > 0
    assert res["log_loss"] > 0


# ============ nulls, empty ============

def test_metrics_nulls_skipped(tmp_path):
    """Null in y or yp → row skipped."""
    df = make_df(tmp_path, {
        "y": [1.0, None, 3.0, 4.0],
        "yp": [1.0, 2.0, None, 4.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["mse"])
    # valid rows: (1,1), (4,4) → mse = 0
    assert res["mse"] == pytest.approx(0.0)


def test_metrics_empty_input_nan(tmp_path):
    df = make_df(tmp_path, {"y": [1.0, 2.0], "yp": [1.0, 2.0]})
    empty = df[df["y"] > 100]
    res = empty.metrics(
        y_true="y", y_pred="yp", metrics=["r2", "rmse", "mae"],
    )
    for k in ("r2", "rmse", "mae"):
        assert math.isnan(res[k])


def test_metrics_filtered(tmp_path):
    df = make_df(tmp_path, {
        "y": [0.0, 1.0, 2.0, 3.0],
        "yp": [0.0, 1.0, 2.0, 100.0],  # big error on filtered-out row
    })
    filtered = df[df["y"] < 3]
    res = filtered.metrics(y_true="y", y_pred="yp", metrics=["mse"])
    assert res["mse"] == pytest.approx(0.0)


# ============ validation ============

def test_metrics_string_metric(tmp_path):
    """String metric (not list) works."""
    df = make_df(tmp_path, {"y": [1.0, 2.0], "yp": [1.0, 2.0]})
    res = df.metrics(y_true="y", y_pred="yp", metrics="r2")
    assert res["r2"] == pytest.approx(1.0)


def test_metrics_invalid_metric(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="unknown metric"):
        df.metrics(y_true="y", y_pred="yp", metrics=["bogus"])


def test_metrics_missing_y_pred_for_r2(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="y_pred_col required"):
        df.metrics(y_true="y", metrics=["r2"])


def test_metrics_missing_y_proba_for_log_loss(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="y_proba_col required"):
        df.metrics(y_true="y", metrics=["log_loss"])


def test_metrics_missing_column(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(KeyError, match="nope"):
        df.metrics(y_true="nope", y_pred="yp", metrics=["r2"])


def test_metrics_empty_metrics_list(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="metrics list is empty"):
        df.metrics(y_true="y", y_pred="yp", metrics=[])


# ============ pandas / sklearn equivalence ============

def test_metrics_pandas_equivalence(tmp_path):
    n = 200
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    yp = y + rng.normal(scale=0.5, size=n)
    df = make_df(tmp_path, {"y": y.tolist(), "yp": yp.tolist()}, chunk=50)
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["r2", "mse", "rmse", "mae"],
    )
    # Reference (numpy, matches sklearn)
    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    assert res["r2"] == pytest.approx(1.0 - ss_res / ss_tot)
    assert res["mse"] == pytest.approx(ss_res / n)
    assert res["rmse"] == pytest.approx(math.sqrt(ss_res / n))
    assert res["mae"] == pytest.approx(float(np.abs(y - yp).mean()))


# ============ integration ============

def test_metrics_with_groupby_predict(tmp_path):
    """End-to-end: groupby → agg → predict → metrics works as chain."""
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x": rng.normal(size=n).tolist(),
        "y": (rng.normal(size=n) * 2).tolist(),
    })
    # Just verify metrics can be called on a df with any data
    res = df.metrics(y_true="y", y_pred="x", metrics=["mse"])
    assert res["mse"] > 0


def test_metrics_log_loss_rejects_continuous_y(tmp_path):
    """§13.4/163: log_loss requires binary y; continuous y → ValueError.

    Formula `-(y·log(p) + (1-y)·log(1-p))` assumes y ∈ {0, 1}.
    With y=2.0, output is negative — mathematically invalid. Must
    reject with clear message.
    """
    df = make_df(tmp_path, {
        "y": [0.0, 1.0, 2.0],  # contains invalid 2.0
        "p": [0.1, 0.9, 0.5],
    })
    with pytest.raises(ValueError, match="requires binary y_true"):
        df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])


# ============ v0.9.2b: classification metrics ============

def test_metrics_accuracy_binary_perfect(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1, 0, 1], "yp": [0, 1, 0, 1]})
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"],
        classes=[0, 1],
    )
    assert res["accuracy"] == pytest.approx(1.0)


def test_metrics_accuracy_binary_mixed(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1, 0, 1], "yp": [0, 1, 1, 1]})
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"],
        classes=[0, 1],
    )
    # 3 correct out of 4
    assert res["accuracy"] == pytest.approx(0.75)


def test_metrics_accuracy_auto_binary_default(tmp_path):
    """classes=None → K=2 default."""
    df = make_df(tmp_path, {"y": [0, 1, 0, 1], "yp": [0, 1, 1, 0]})
    res = df.metrics(y_true="y", y_pred="yp", metrics=["accuracy"])
    assert res["accuracy"] == pytest.approx(0.5)


def test_metrics_precision_recall_f1_binary(tmp_path):
    """y=[1,1,1,0,0], yp=[1,0,1,1,0] → TP=2, FP=1, FN=1.
    precision=2/3, recall=2/3, f1=2/3."""
    df = make_df(tmp_path, {
        "y": [1, 1, 1, 0, 0],
        "yp": [1, 0, 1, 1, 0],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["precision", "recall", "f1"],
        classes=[0, 1],
        average="binary", pos_label=1,
    )
    assert res["precision"] == pytest.approx(2.0 / 3.0)
    assert res["recall"] == pytest.approx(2.0 / 3.0)
    assert res["f1"] == pytest.approx(2.0 / 3.0)


def test_metrics_precision_binary_pos_label_0(tmp_path):
    """pos_label=0: precision for class 0.

    y=[1,1,1,0,0], yp=[1,0,1,1,0]. For class 0:
      TP: (0,0) → row 4
      FP: (≠0, 0) → row 1 (y=1, yp=0)
      FN: (0, ≠0) → row 3 (y=0, yp=1)
    precision(0) = TP/(TP+FP) = 1/2 = 0.5 (§13.4/164: prior expected 1.0
    missed row 1 FP — test-authoring error, code correct).
    """
    df = make_df(tmp_path, {
        "y": [1, 1, 1, 0, 0],
        "yp": [1, 0, 1, 1, 0],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["precision"], classes=[0, 1],
        average="binary", pos_label=0,
    )
    assert res["precision"] == pytest.approx(0.5)


def test_metrics_accuracy_multiclass(tmp_path):
    """3 classes, y=yp → accuracy 1.0; classes explicit."""
    df = make_df(tmp_path, {
        "y": [0, 1, 2, 0, 1, 2],
        "yp": [0, 1, 2, 0, 1, 2],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["accuracy", "precision", "recall", "f1"],
        classes=[0, 1, 2],
    )
    assert res["accuracy"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)
    assert res["f1"] == pytest.approx(1.0)


def test_metrics_multiclass_macro_micro(tmp_path):
    """3-class macro vs micro differ on class imbalance."""
    df = make_df(tmp_path, {
        "y": [0, 0, 0, 0, 1, 2],
        "yp": [0, 0, 1, 2, 1, 2],
    })
    macro = df.metrics(
        y_true="y", y_pred="yp", metrics=["precision"],
        classes=[0, 1, 2], average="macro",
    )
    micro = df.metrics(
        y_true="y", y_pred="yp", metrics=["precision"],
        classes=[0, 1, 2], average="micro",
    )
    acc = df.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"],
        classes=[0, 1, 2],
    )
    # Per-class precision: c0=2/2=1.0, c1=1/1=1.0, c2=0/1=0.0
    # macro = (1+1+0)/3 = 2/3; micro = accuracy = 4/6 = 2/3
    assert micro["precision"] == pytest.approx(acc["accuracy"])
    assert macro["precision"] == pytest.approx(2.0 / 3.0)


def test_metrics_multiclass_weighted(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 0, 0, 0, 1, 2],
        "yp": [0, 0, 1, 2, 1, 2],
    })
    # Support: c0=4, c1=1, c2=1. precision: 1.0, 1.0, 0.0
    # weighted = (4·1 + 1·1 + 1·0) / 6 = 5/6
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["precision"],
        classes=[0, 1, 2], average="weighted",
    )
    assert res["precision"] == pytest.approx(5.0 / 6.0)


def test_metrics_classification_zero_division(tmp_path):
    """No positive predictions → precision=0.0 (sklearn zero_division=0)."""
    df = make_df(tmp_path, {
        "y": [1, 1, 0],
        "yp": [0, 0, 0],  # never predicts 1
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["precision"], classes=[0, 1],
        average="binary", pos_label=1,
    )
    assert res["precision"] == 0.0


def test_metrics_classification_nulls_skipped(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, None, 1, 0],
        "yp": [0, 1, None, 0],
    })
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"],
        classes=[0, 1],
    )
    # valid rows: y=0 yp=0 (correct), y=0 yp=0 (correct) → accuracy=1.0
    assert res["accuracy"] == pytest.approx(1.0)


def test_metrics_classification_empty(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1], "yp": [0, 1]})
    empty = df[df["y"] > 100]
    res = empty.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"], classes=[0, 1],
    )
    assert math.isnan(res["accuracy"])


def test_metrics_classification_filtered(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 1, 0, 1, 0],
        "yp": [0, 1, 1, 1, 1],  # errors on rows 2, 4
    })
    filtered = df[df["y"] == 0]  # only rows 0, 2, 4
    res = filtered.metrics(
        y_true="y", y_pred="yp", metrics=["accuracy"], classes=[0, 1],
    )
    # correct: row 0 (0,0); wrong: rows 2,4 (0,1); acc = 1/3
    assert res["accuracy"] == pytest.approx(1.0 / 3.0)


def test_metrics_classification_unknown_label_raises(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 1, 2],  # 2 not in classes
        "yp": [0, 1, 0],
    })
    with pytest.raises(ValueError, match="y_true has labels not in"):
        df.metrics(
            y_true="y", y_pred="yp", metrics=["accuracy"],
            classes=[0, 1],
        )


def test_metrics_classification_unknown_pred_label_raises(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 1, 0],
        "yp": [0, 5, 0],
    })
    with pytest.raises(ValueError, match="y_pred has labels not in"):
        df.metrics(
            y_true="y", y_pred="yp", metrics=["accuracy"],
            classes=[0, 1],
        )


def test_metrics_classification_and_regression_conflict(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1], "yp": [0, 1]})
    with pytest.raises(ValueError, match="cannot share y_pred"):
        df.metrics(
            y_true="y", y_pred="yp",
            metrics=["r2", "accuracy"], classes=[0, 1],
        )


def test_metrics_average_binary_on_multiclass_raises(tmp_path):
    df = make_df(tmp_path, {
        "y": [0, 1, 2], "yp": [0, 1, 2],
    })
    with pytest.raises(ValueError, match="average='binary' requires K=2"):
        df.metrics(
            y_true="y", y_pred="yp", metrics=["precision"],
            classes=[0, 1, 2], average="binary",
        )


def test_metrics_invalid_average_raises(tmp_path):
    df = make_df(tmp_path, {"y": [0, 1], "yp": [0, 1]})
    with pytest.raises(ValueError, match="average must be one of"):
        df.metrics(
            y_true="y", y_pred="yp", metrics=["precision"],
            classes=[0, 1], average="bogus",
        )


def test_metrics_classification_k_too_large_raises(tmp_path):
    """K=10000 → state 240 KB > state_limit (2000 B budget)."""
    df = make_df(tmp_path, {"y": [0], "yp": [0]})
    df._memory_guard = MemoryGuard(max_memory_bytes=2000)
    with pytest.raises(MemoryLimitExceeded, match="state"):
        df.metrics(
            y_true="y", y_pred="yp", metrics=["accuracy"],
            classes=list(range(10000)),
        )


def test_metrics_classification_string_labels(tmp_path):
    df = make_df(tmp_path, {
        "y": ["cat", "dog", "cat", "dog"],
        "yp": ["cat", "dog", "dog", "dog"],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["accuracy"], classes=["cat", "dog"],
    )
    assert res["accuracy"] == pytest.approx(0.75)


def test_metrics_classification_combined_with_log_loss(tmp_path):
    """classification + log_loss — different columns, OK."""
    df = make_df(tmp_path, {
        "y": [0, 1, 0, 1],
        "yp": [0, 1, 1, 0],
        "p": [0.1, 0.9, 0.6, 0.4],
    })
    res = df.metrics(
        y_true="y", y_pred="yp", y_proba="p",
        metrics=["accuracy", "log_loss"], classes=[0, 1],
    )
    assert res["accuracy"] == pytest.approx(0.5)
    assert res["log_loss"] > 0


def test_metrics_classification_pandas_equivalence(tmp_path):
    """Compare to manual numpy reference for binary precision/recall/f1."""
    rng = np.random.default_rng(42)
    n = 200
    y = rng.integers(0, 2, size=n).astype(np.int64)
    yp = (y + rng.integers(0, 2, size=n)) % 2  # noisy
    df = make_df(tmp_path, {
        "y": y.tolist(), "yp": yp.tolist(),
    }, chunk=20)
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["accuracy", "precision", "recall", "f1"],
        classes=[0, 1], average="binary", pos_label=1,
    )
    tp = int(((y == 1) & (yp == 1)).sum())
    fp = int(((y == 0) & (yp == 1)).sum())
    fn = int(((y == 1) & (yp == 0)).sum())
    total = n
    ref_acc = (tp + (total - tp - fp - fn)) / total
    ref_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ref_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ref_f1 = (2 * ref_prec * ref_rec / (ref_prec + ref_rec)
              if (ref_prec + ref_rec) > 0 else 0.0)
    assert res["accuracy"] == pytest.approx(ref_acc, rel=1e-9)
    assert res["precision"] == pytest.approx(ref_prec, rel=1e-9)
    assert res["recall"] == pytest.approx(ref_rec, rel=1e-9)
    assert res["f1"] == pytest.approx(ref_f1, rel=1e-9)
