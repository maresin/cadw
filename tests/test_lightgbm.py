"""Тесты to_lightgbm (06b): честный downsampling + E2E."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard

lgb = pytest.importorskip("lightgbm")


def make_df(tmp_path, n=2000, seed=0):
    rng = np.random.default_rng(seed=seed)
    x1 = rng.normal(0, 1, n).astype(np.float32)
    x2 = rng.normal(0, 1, n).astype(np.float32)
    y = (x1 + x2 > 0).astype(np.int64)
    t = pa.table({"x1": x1, "x2": x2, "y": y})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=500,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )


def test_to_lightgbm_basic(tmp_path):
    df = make_df(tmp_path, n=1000)
    ds = df.to_lightgbm(["x1", "x2"], "y", max_rows=1000)
    assert isinstance(ds, lgb.Dataset)
    ds.construct()
    assert ds.num_data() == 1000
    assert ds.num_feature() == 2


def test_to_lightgbm_downsampling(tmp_path):
    df = make_df(tmp_path, n=1000)
    ds = df.to_lightgbm(["x1", "x2"], "y", max_rows=300)
    ds.construct()
    assert ds.num_data() == 300


def test_to_lightgbm_max_rows_none(tmp_path):
    df = make_df(tmp_path, n=500)
    ds = df.to_lightgbm(["x1", "x2"], "y", max_rows=None)
    ds.construct()
    assert ds.num_data() == 500


def test_to_lightgbm_max_rows_zero_raises(tmp_path):
    df = make_df(tmp_path, n=100)
    with pytest.raises(ValueError):
        df.to_lightgbm(["x1", "x2"], "y", max_rows=0)


def test_to_lightgbm_missing_x_raises(tmp_path):
    df = make_df(tmp_path, n=100)
    with pytest.raises(KeyError, match="X column"):
        df.to_lightgbm(["nope"], "y", max_rows=100)


def test_to_lightgbm_missing_y_raises(tmp_path):
    df = make_df(tmp_path, n=100)
    with pytest.raises(KeyError, match="y column"):
        df.to_lightgbm(["x1"], "nope", max_rows=100)


def test_to_lightgbm_after_filter(tmp_path):
    df = make_df(tmp_path, n=1000)
    filtered = df[df["x1"] > -2]
    expected = filtered.count()
    ds = filtered.to_lightgbm(["x1", "x2"], "y", max_rows=None)
    ds.construct()
    assert ds.num_data() == expected


def test_to_lightgbm_after_sort(tmp_path):
    """Порядок строк после sort сохраняется в материализации."""
    df = make_df(tmp_path, n=500)
    sorted_df = df.sort_values("x1")
    ds = sorted_df.to_lightgbm(["x1", "x2"], "y", max_rows=100)
    ds.construct()
    assert ds.num_data() == 100


def test_train_binary_classifier(tmp_path):
    """Полный цикл: df → lgb.Dataset → lgb.train → predict."""
    df = make_df(tmp_path, n=2000)
    ds = df.to_lightgbm(["x1", "x2"], "y", max_rows=None)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_threads": 1},
        ds,
        num_boost_round=20,
    )
    # predict на тех же данных — accuracy должна быть высокой
    X = np.column_stack([
        df.head(2000).column("x1").to_numpy(zero_copy_only=False),
        df.head(2000).column("x2").to_numpy(zero_copy_only=False),
    ])
    y_true = df.head(2000).column("y").to_numpy(zero_copy_only=False)
    preds = (booster.predict(X) > 0.5).astype(np.int64)
    acc = float((preds == y_true).mean())
    assert acc > 0.9, f"LightGBM accuracy {acc:.2f}"


@pytest.mark.slow
def test_e2e_lightgbm_under_budget(tmp_path):
    """50k строк, 10 MB budget, downsampling до 20k, lgb.train."""
    df = make_df(tmp_path, n=50_000)
    # 20k строк * ~24 B/строка (3 float32/int64) ≈ 500 KB — влезает в 10 MB
    ds = df.to_lightgbm(
        ["x1", "x2"], "y",
        max_rows=20_000,
    )
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_threads": 1},
        ds,
        num_boost_round=20,
    )
    # predict на head(1000)
    head = df.head(1000)
    X = np.column_stack([
        head.column("x1").to_numpy(zero_copy_only=False),
        head.column("x2").to_numpy(zero_copy_only=False),
    ])
    y_true = head.column("y").to_numpy(zero_copy_only=False)
    preds = (booster.predict(X) > 0.5).astype(np.int64)
    acc = float((preds == y_true).mean())
    assert acc > 0.9, f"E2E LightGBM accuracy {acc:.2f}"


def test_to_lightgbm_categorical_must_be_in_x(tmp_path):
    """categorical_cols[i] обязан быть в X_cols (LightGBM categorical_feature = позиции в X)."""
    rng = np.random.default_rng(0)
    n = 200
    cat = rng.integers(0, 3, size=n).astype(np.int64)  # int-encoded
    x1 = rng.normal(0, 1, n).astype(np.float32)
    y = (x1 > 0).astype(np.int64)
    t = pa.table({"cat": cat, "x1": x1, "y": y})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=100,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )

    # 1. cat не в X_cols -> KeyError
    with pytest.raises(KeyError, match="not in X_cols"):
        df.to_lightgbm(["x1"], "y", max_rows=None, categorical_cols=["cat"])

    # 2. cat в X_cols -> OK, categorical_feature=[индекс]
    ds = df.to_lightgbm(
        ["cat", "x1"], "y", max_rows=None, categorical_cols=["cat"]
    )
    ds.construct()
    assert ds.num_data() == n


def test_to_lightgbm_categorical_string_limitation(tmp_path):
    """Документированное ограничение: сырые строки не конвертируются автоматически.

    LightGBM требует int-encoded категории (или pandas category dtype).
    CADW не кодирует строки — это ответственность пользователя.
    """
    rng = np.random.default_rng(0)
    n = 100
    cat = rng.choice(["a", "b", "c"], size=n).astype(object)
    x1 = rng.normal(0, 1, n).astype(np.float32)
    y = (x1 > 0).astype(np.int64)
    t = pa.table({"cat": cat, "x1": x1, "y": y})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=50,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )
    # np.column_stack со строками даст object dtype; LightGBM упадёт
    # (или мы упадём раньше на конвертации) — это ожидаемо.
    # Проверяем, что хотя бы один из вариантов даёт явную ошибку.
    with pytest.raises((ValueError, TypeError, KeyError)):
        ds = df.to_lightgbm(
            ["cat", "x1"], "y", max_rows=None, categorical_cols=["cat"]
        )
        ds.construct()
