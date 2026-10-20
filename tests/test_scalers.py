"""v0.9.2f: scalers tests."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import (
    CadwDataFrame, MemoryGuard,
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
)


def make_df(tmp_path, data, chunk=10, name="L", mem_mb=50):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_table(df, tmp_path, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


def _col(t, name):
    return t.column(name).to_pylist()


# ============ StandardScaler ============

def test_standard_scaler_mean_zero_std_one(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(loc=10.0, scale=5.0, size=1000)
    df = make_df(tmp_path, {"x": x.tolist()}, chunk=100)
    scaler = StandardScaler()
    scaler.fit(df)
    out = scaler.transform(df)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    # mean ≈ 0, std ≈ 1 (population std, matches our fit)
    assert abs(float(xs.mean())) < 1e-9
    assert abs(float(xs.std(ddof=0)) - 1.0) < 1e-9


def test_standard_scaler_with_mean_false(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(loc=10.0, scale=5.0, size=1000)
    df = make_df(tmp_path, {"x": x.tolist()}, chunk=100)
    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler.fit(df)
    out = scaler.transform(df)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    # mean ≈ 10/std_train (mean shifted from original), std ≈ 1
    assert abs(float(xs.std(ddof=0)) - 1.0) < 1e-9


def test_standard_scaler_with_std_false(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(loc=10.0, scale=5.0, size=1000)
    df = make_df(tmp_path, {"x": x.tolist()}, chunk=100)
    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit(df)
    out = scaler.transform(df)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    # mean ≈ 0, std unchanged (~5)
    assert abs(float(xs.mean())) < 1e-9
    assert abs(float(xs.std(ddof=0)) - x.std(ddof=0)) < 1e-6


def test_standard_scaler_nulls_skipped(tmp_path):
    x = [1.0, None, 3.0, None, 5.0]
    df = make_df(tmp_path, {"x": x}, chunk=2)
    scaler = StandardScaler()
    scaler.fit(df)
    out = scaler.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert xs[1] is None and xs[3] is None
    # mean of valid = 3.0; std (ddof=0) = sqrt(8/3) ≈ 1.633
    # (1-3)/1.633 ≈ -1.2247
    assert xs[0] is not None
    assert abs(xs[0] - (-2.0 / np.sqrt(8.0/3.0))) < 1e-9


def test_standard_scaler_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=100)
    df = make_df(tmp_path, {"x": x.tolist()})
    s = StandardScaler()
    s.fit(df)
    d = s.to_dict()
    s2 = StandardScaler.from_dict(d)
    out1 = read_table(s.transform(df), tmp_path, "o1.parquet")
    out2 = read_table(s2.transform(df), tmp_path, "o2.parquet")
    np.testing.assert_allclose(_col(out1, "x"), _col(out2, "x"))


def test_standard_scaler_unseen_data(tmp_path):
    """Transform uses FROZEN train stats, not refit on test (§13.4/171).

    Deterministic data + reference helper. Prior test had 2 bugs:
    (a) same-distribution train/test can't distinguish frozen vs refit
        (both give output mean ≈ 0);
    (b) manual arithmetic `abs(mean - 5.0) < 0.5` was wrong —
        (mean - mean_train)/std = 0, not mean/std = 5.

    Setup: x_train = [8, 12] → mean=10, std(ddof=0)=2 (exact).
           x_test  = [12] → frozen: (12-10)/2 = 1.0
                            refit:  (12-12)/1 = 0.0 (would fail)
    """
    from tests._helpers.scaler_reference import standard_scaler_transform

    x_train = [8.0, 12.0]
    x_test = [12.0]
    df_train = make_df(tmp_path, {"x": x_train}, name="train")
    df_test = make_df(tmp_path, {"x": x_test}, name="test")
    s = StandardScaler()
    s.fit(df_train)
    out = s.transform(df_test)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    expected = standard_scaler_transform(x_train, x_test)
    assert len(xs) == 1
    assert abs(xs[0] - expected[0]) < 1e-9, (
        f"frozen-stats output: got {xs[0]:.6f}, expected "
        f"{expected[0]:.6f}. If ~0.0, transform refit on test (BUG)."
    )
    # Specific: (12 - 10) / 2 = 1.0
    assert abs(xs[0] - 1.0) < 1e-9


def test_standard_scaler_constant_column(tmp_path):
    """Constant column → std=0 → scale to 1.0 (sklearn), output zeros."""
    df = make_df(tmp_path, {"x": [5.0] * 20})
    s = StandardScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert all(abs(v) < 1e-9 for v in xs)


# ============ MinMaxScaler ============

def test_minmax_scaler_0_1(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    s = MinMaxScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert xs == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_minmax_scaler_custom_range(tmp_path):
    df = make_df(tmp_path, {"x": [0.0, 5.0, 10.0]})
    s = MinMaxScaler(feature_range=(-1.0, 1.0))
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    np.testing.assert_allclose(xs, [-1.0, 0.0, 1.0])


def test_minmax_scaler_constant_column(tmp_path):
    """Constant → scale=0 → output = feature_range[0]."""
    df = make_df(tmp_path, {"x": [5.0] * 10})
    s = MinMaxScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert all(abs(v) < 1e-9 for v in xs)


def test_minmax_scaler_roundtrip(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0]})
    s = MinMaxScaler()
    s.fit(df)
    s2 = MinMaxScaler.from_dict(s.to_dict())
    out1 = read_table(s.transform(df), tmp_path, "o1.parquet")
    out2 = read_table(s2.transform(df), tmp_path, "o2.parquet")
    np.testing.assert_allclose(_col(out1, "x"), _col(out2, "x"))


def test_minmax_scaler_bad_range():
    with pytest.raises(ValueError, match="feature_range"):
        MinMaxScaler(feature_range=(1.0, 1.0))


# ============ MaxAbsScaler ============

def test_maxabs_scaler_basic(tmp_path):
    df = make_df(tmp_path, {"x": [-4.0, -2.0, 0.0, 2.0, 4.0]})
    s = MaxAbsScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert xs == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_maxabs_scaler_positive_only(tmp_path):
    df = make_df(tmp_path, {"x": [0.0, 5.0, 10.0]})
    s = MaxAbsScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert xs == [0.0, 0.5, 1.0]


def test_maxabs_scaler_all_zero(tmp_path):
    """All zeros → scale=1 (no-op)."""
    df = make_df(tmp_path, {"x": [0.0] * 10})
    s = MaxAbsScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert all(abs(v) < 1e-9 for v in xs)


def test_maxabs_scaler_roundtrip(tmp_path):
    df = make_df(tmp_path, {"x": [-3.0, 0.0, 3.0]})
    s = MaxAbsScaler()
    s.fit(df)
    s2 = MaxAbsScaler.from_dict(s.to_dict())
    out1 = read_table(s.transform(df), tmp_path, "o1.parquet")
    out2 = read_table(s2.transform(df), tmp_path, "o2.parquet")
    np.testing.assert_allclose(_col(out1, "x"), _col(out2, "x"))


# ============ RobustScaler ============

def test_robust_scaler_median_zero_iqr_one(tmp_path):
    """After scaling, median=0, IQR=1 (for non-degenerate data)."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=500)
    df = make_df(tmp_path, {"x": x.tolist()}, chunk=50)
    s = RobustScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    q25, q50, q75 = np.percentile(xs, [25, 50, 75])
    assert abs(float(q50)) < 1e-9
    assert abs(float(q75 - q25) - 1.0) < 1e-9


def test_robust_scaler_outlier_resistant(tmp_path):
    """Single huge outlier doesn't shift scaling much."""
    x = list(np.random.default_rng(0).normal(size=100)) + [1e6]
    df = make_df(tmp_path, {"x": x}, chunk=20)
    s = RobustScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    # Median of transformed ≈ 0
    assert abs(float(np.median(xs))) < 1e-6


def test_robust_scaler_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=100)
    df = make_df(tmp_path, {"x": x.tolist()})
    s = RobustScaler()
    s.fit(df)
    s2 = RobustScaler.from_dict(s.to_dict())
    out1 = read_table(s.transform(df), tmp_path, "o1.parquet")
    out2 = read_table(s2.transform(df), tmp_path, "o2.parquet")
    np.testing.assert_allclose(_col(out1, "x"), _col(out2, "x"))


def test_robust_scaler_constant_column(tmp_path):
    df = make_df(tmp_path, {"x": [5.0] * 20})
    s = RobustScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert all(abs(v) < 1e-9 for v in xs)


# ============ Multi-column / pass-through ============

def test_scaler_multiple_columns(tmp_path):
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
        "c": [100.0, 200.0, 300.0],
    })
    s = StandardScaler(columns=["a", "b"])
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    # a and b standardized; c unchanged
    assert set(t.schema.names) == {"a", "b", "c"}
    assert _col(t, "c") == [100.0, 200.0, 300.0]


def test_scaler_auto_detect_numeric(tmp_path):
    """columns=None → numeric columns auto-detected at fit."""
    df = make_df(tmp_path, {
        "num1": [1.0, 2.0, 3.0],
        "num2": [4.0, 5.0, 6.0],
        "s": ["x", "y", "z"],
    })
    s = StandardScaler()
    s.fit(df)
    assert set(s._columns) == {"num1", "num2"}


def test_scaler_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0]})
    s = StandardScaler(columns=["nope"])
    with pytest.raises(KeyError, match="nope"):
        s.fit(df)


# ============ Pipeline integration ============

def test_pipeline_with_standard_scaler(tmp_path):
    from cadw import Pipeline
    from cadw.pipeline import GBDTStep

    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(loc=100, scale=10, size=n)
    x2 = rng.normal(loc=0.5, scale=0.1, size=n)
    y = 2.0 * (x1 - 100) / 10 + 3.0 * (x2 - 0.5) / 0.1 + rng.normal(
        scale=0.1, size=n)

    df = make_df(tmp_path, {
        "x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist(),
    }, chunk=50, mem_mb=200)

    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("scale", StandardScaler(columns=["x1", "x2"]))
    pipe.add_step(
        "model", GBDTStep(
            ["x1", "x2"], "y",
            n_estimators=20, max_depth=2, learning_rate=0.2,
        ),
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    assert "y_pred" in preds.columns


def test_pipeline_save_load_with_scaler(tmp_path):
    from cadw import Pipeline
    from cadw.pipeline import GBDTStep

    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(loc=10, scale=2, size=n)
    y = 2.0 * (x - 10) / 2 + rng.normal(scale=0.1, size=n)

    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()},
                 chunk=50, mem_mb=200)

    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("scale", StandardScaler())
    pipe.add_step("model", GBDTStep(["x"], "y", n_estimators=10))

    pipe.fit(df, target="y")

    pipe_dir = tmp_path / "mypipe"
    pipe.to_json(pipe_dir)
    restored = Pipeline.from_json(
        pipe_dir, memory_guard=MemoryGuard(max_memory_mb=200),
    )

    preds_orig = read_table(pipe.predict(df), tmp_path, "p1.parquet")
    preds_restored = read_table(
        restored.predict(df), tmp_path, "p2.parquet",
    )
    np.testing.assert_allclose(
        preds_orig.column("y_pred").to_pylist(),
        preds_restored.column("y_pred").to_pylist(),
        rtol=1e-9, equal_nan=True,
    )
