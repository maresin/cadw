"""Тесты ML-интеграции (06a): to_batches(format) и partial_fit."""
import inspect
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cadw.dataframe as df_mod
from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path: Path, n: int = 100) -> CadwDataFrame:
    rng = np.random.default_rng(seed=0)
    x1 = rng.normal(0, 1, n).astype(np.float64)
    x2 = rng.normal(0, 1, n).astype(np.float64)
    y = (x1 + x2 > 0).astype(np.int64)
    t = pa.table({"x1": x1, "x2": x2, "y": y})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=20,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )


# ---------- format conversion ----------

def test_to_batches_arrow_default(tmp_path):
    df = make_df(tmp_path)
    for b in df.to_batches():
        assert isinstance(b, pa.RecordBatch)


def test_to_batches_format_numpy(tmp_path):
    df = make_df(tmp_path)
    total = 0
    for d in df.to_batches(format="numpy"):
        assert isinstance(d, dict)
        assert set(d.keys()) == {"x1", "x2", "y"}
        for k, v in d.items():
            assert isinstance(v, np.ndarray)
        n = len(d["x1"])
        assert all(len(v) == n for v in d.values())
        total += n
    assert total == 100


def test_to_batches_format_pandas(tmp_path):
    import pandas as pd
    df = make_df(tmp_path)
    total = 0
    for pdf in df.to_batches(format="pandas"):
        assert isinstance(pdf, pd.DataFrame)
        assert list(pdf.columns) == ["x1", "x2", "y"]
        total += len(pdf)
    assert total == 100


def test_to_batches_format_torch_lazy(tmp_path):
    torch = pytest.importorskip("torch")
    df = make_df(tmp_path)
    for d in df.to_batches(format="torch"):
        assert isinstance(d, dict)
        for v in d.values():
            assert isinstance(v, torch.Tensor)


def test_to_batches_unknown_format_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="unsupported format"):
        list(df.to_batches(format="parquet"))


def test_to_batches_format_with_batch_size(tmp_path):
    df = make_df(tmp_path)
    sizes = [len(d["x1"]) for d in df.to_batches(batch_size=30, format="numpy")]
    assert sizes == [30, 30, 30, 10]


def test_to_batches_format_with_filter(tmp_path):
    df = make_df(tmp_path)
    sub = df[df["x1"] > 0]
    total = sum(len(d["x1"]) for d in sub.to_batches(format="numpy"))
    assert total == int((df.to_batches(format="numpy").__iter__().__next__() is not None)) * 0 + sub.count()


# ---------- partial_fit ----------

def test_partial_fit_sgd(tmp_path):
    from sklearn.linear_model import SGDClassifier
    df = make_df(tmp_path, n=200)
    model = SGDClassifier(loss="log_loss", random_state=0)
    df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=50,
                   classes=[0, 1])
    # проверка: модель обучена, делает предсказания
    batches = list(df.to_batches(format="numpy"))
    X = np.vstack([np.column_stack([b["x1"], b["x2"]]) for b in batches])
    preds = model.predict(X)
    # на тренировочных данных accuracy > 0.9 (данные линейно разделимы)
    acc = float((preds == np.concatenate([b["y"] for b in batches])).mean())
    assert acc > 0.9, f"SGD accuracy {acc:.2f}"


def test_partial_fit_standard_scaler(tmp_path):
    from sklearn.preprocessing import StandardScaler
    df = make_df(tmp_path, n=200)
    scaler = StandardScaler()
    df.partial_fit(scaler, X_cols=["x1", "x2"], y_col=None, batch_size=50)
    # scaler обучен: mean ~0, scale ~1 (данные из N(0,1))
    assert abs(float(scaler.mean_[0])) < 0.3
    assert abs(float(scaler.scale_[0])) < 1.5
    assert abs(float(scaler.scale_[0])) > 0.5


def test_partial_fit_missing_x_col_raises(tmp_path):
    from sklearn.linear_model import SGDClassifier
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="X column"):
        df.partial_fit(SGDClassifier(), X_cols=["nope"], y_col="y")


def test_partial_fit_missing_y_col_raises(tmp_path):
    from sklearn.linear_model import SGDClassifier
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="y column"):
        df.partial_fit(SGDClassifier(), X_cols=["x1"], y_col="nope")


def test_partial_fit_after_filter(tmp_path):
    """partial_fit работает на отфильтрованном df."""
    from sklearn.linear_model import SGDClassifier
    df = make_df(tmp_path, n=200)
    filtered = df[df["x1"] > -2]  # отбрасываем экстремальные
    model = SGDClassifier(loss="log_loss", random_state=0)
    filtered.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=50,
                         classes=[0, 1])
    # модель обучена (проверка: coef_ не пустой)
    assert model.coef_.shape == (1, 2)


# ---------- E2E: full sklearn pipeline ----------

@pytest.mark.slow
def test_e2e_sklearn_pipeline(tmp_path):
    """Стандартный sklearn-пайплайн: scaler.partial_fit → sgd.partial_fit → predict."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    # Больше данных — 50k строк, 5 MB бюджет
    rng = np.random.default_rng(seed=42)
    n = 50_000
    x1 = rng.normal(0, 1, n).astype(np.float32)
    x2 = rng.normal(0, 1, n).astype(np.float32)
    y = ((x1 + x2 > 0)).astype(np.int64)
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table({"x1": x1, "x2": x2, "y": y}), src,
                   row_group_size=10_000)

    guard = MemoryGuard(max_memory_mb=10)
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5_000,
        memory_guard=guard, spill_dir=tmp_path / "spill",
    )

    # 1. scaler.partial_fit (без y)
    scaler = StandardScaler()
    df.partial_fit(scaler, X_cols=["x1", "x2"], y_col=None, batch_size=5_000)

    # 2. sgd.partial_fit (с y, на 3 эпохах для сходимости)
    model = SGDClassifier(loss="log_loss", random_state=0)
    for epoch in range(3):
        df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=5_000,
                       classes=[0, 1])

    # 3. predict на 1000 строк — accuracy
    head = df.head(1000)
    X = np.column_stack([
        head.column("x1").to_numpy(zero_copy_only=False),
        head.column("x2").to_numpy(zero_copy_only=False),
    ])
    y_true = head.column("y").to_numpy(zero_copy_only=False)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    acc = float((preds == y_true).mean())
    assert acc > 0.85, f"E2E sklearn pipeline accuracy {acc:.2f}"
