"""Component-wise verification for fit_knn_imputer (DEC-033, v0.9.199).

Reference: sklearn.impute.KNNImputer (P6-P8b verified).
Formula state O(model):
  fit_nan_counts (d_kept) * i64 + X_cols_kept sum(len+8) +
  X_cols_dropped sum(len+8) + 256
  (X memmap on disk, not counted).

sklearn parity is primary evidence (uniform 0.0, distance 1.4e-13 in
smoke; formal tests use atol=1e-9).
"""
from __future__ import annotations
import warnings

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import fit_knn_imputer, KNNImputerResult


def _mk_df(X):
    d = X.shape[1]
    cols = {}
    for j in range(d):
        cols[f"x{j}"] = pa.array(X[:, j], type=pa.float64())
    return CadwDataFrame.from_table(pa.table(cols),
                                     memory_guard=MemoryGuard(max_memory_mb=500))


def _make_with_nan(n=100, d=4, seed=0, frac=0.15):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    X[rng.rand(*X.shape) < frac] = np.nan
    # Ensure every row has >=1 finite and every col has >=1 finite
    X[0, :] = np.arange(d, dtype=np.float64) + 1.0
    for j in range(d):
        if np.isnan(X[:, j]).all():
            X[0, j] = float(j + 1)
    return X


def _extract_output(out_df):
    parts = []
    d = len(out_df.columns)
    for b in out_df._iter_filtered_batches():
        parts.append(np.column_stack([
            b.column(out_df.columns[j]).to_numpy(zero_copy_only=False)
            for j in range(d)
        ]))
    return np.vstack(parts) if parts else np.empty((0, d))


def _warmup():
    X = _make_with_nan(n=30, d=3, seed=999)
    df = _mk_df(X)
    r = fit_knn_imputer(df, [f"x{j}" for j in range(3)], k=2,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    r.release_train_state()


_warmup()


# ---- CW-1: state_nbytes exact (O(model)) ----

@pytest.mark.parametrize("n", [50, 200, 500])
def test_state_nbytes_exact(n):
    """state = 256 + fit_nan_counts + col names; constant in n (path-based)."""
    X = _make_with_nan(n=n, d=4, seed=1)
    df = _mk_df(X)
    r = fit_knn_imputer(df, [f"x{j}" for j in range(4)], k=3,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    d_kept = len(r.X_cols_kept)
    expected = 256 + r.fit_nan_counts.nbytes
    for c in r.X_cols_kept:
        expected += 8 + len(c)
    for c in r.X_cols_dropped:
        expected += 8 + len(c)
    assert r.state_nbytes() == expected, (
        f"n={n}: state={r.state_nbytes()}, expected {expected}")
    r.release_train_state()


# ---- CW-2: state constant in n ----

def test_state_constant_in_n():
    """Spilled state; only column metadata scales with d, not n."""
    states = []
    for n in (50, 500):
        X = _make_with_nan(n=n, d=4, seed=2)
        df = _mk_df(X)
        r = fit_knn_imputer(df, [f"x{j}" for j in range(4)], k=3,
                            memory_guard=MemoryGuard(max_memory_mb=500))
        states.append(r.state_nbytes())
        r.release_train_state()
    assert states[0] == states[1], f"state not n-invariant: {states}"


# ---- CW-3: sklearn parity uniform ----

@pytest.mark.parametrize("k", [1, 3, 5])
def test_sklearn_parity_uniform(k):
    from sklearn.impute import KNNImputer
    X = _make_with_nan(n=80, d=4, seed=3)
    df = _mk_df(X)
    sk = KNNImputer(n_neighbors=k, weights="uniform")
    X_sk = sk.fit_transform(X)
    r = fit_knn_imputer(df, [f"x{j}" for j in range(4)], k=k,
                        weights="uniform",
                        memory_guard=MemoryGuard(max_memory_mb=500))
    out_df = r.transform(df)
    X_cadw = _extract_output(out_df)
    max_diff = float(np.abs(X_cadw - X_sk).max())
    assert max_diff < 1e-9, f"k={k}: max diff {max_diff:.2e}"
    r.release_train_state()


# ---- CW-4: sklearn parity distance ----

@pytest.mark.parametrize("k", [1, 3, 5])
def test_sklearn_parity_distance(k):
    from sklearn.impute import KNNImputer
    X = _make_with_nan(n=80, d=4, seed=4)
    df = _mk_df(X)
    sk = KNNImputer(n_neighbors=k, weights="distance")
    X_sk = sk.fit_transform(X)
    r = fit_knn_imputer(df, [f"x{j}" for j in range(4)], k=k,
                        weights="distance",
                        memory_guard=MemoryGuard(max_memory_mb=500))
    out_df = r.transform(df)
    X_cadw = _extract_output(out_df)
    max_diff = float(np.abs(X_cadw - X_sk).max())
    assert max_diff < 1e-9, f"k={k}: max diff {max_diff:.2e}"
    r.release_train_state()


# ---- CW-5: all-NaN column dropped (sklearn behavior + warning) ----

def test_all_nan_column_dropped():
    X = np.array([
        [1.0, np.nan, 3.0],
        [2.0, np.nan, np.nan],
        [np.nan, np.nan, 5.0],
        [4.0, np.nan, 7.0],
    ], dtype=np.float64)
    df = _mk_df(X)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = fit_knn_imputer(df, ["x0", "x1", "x2"], k=2,
                            memory_guard=MemoryGuard(max_memory_mb=500))
    assert "x1" in r.X_cols_dropped
    assert set(r.X_cols_kept) == {"x0", "x2"}
    assert any("dropped" in str(x.message) for x in w),         f"expected drop warning; got {[str(x.message) for x in w]}"
    # transform output has 2 cols
    out = r.transform(df)
    assert len(out.columns) == 2
    r.release_train_state()


# ---- CW-6: only-1-donor column ----

def test_only_one_donor_column():
    X = np.array([
        [1.0, np.nan, 3.0],
        [2.0, 5.0, np.nan],
        [np.nan, np.nan, 5.0],
        [4.0, np.nan, 7.0],
    ], dtype=np.float64)
    df = _mk_df(X)
    r = fit_knn_imputer(df, ["x0", "x1", "x2"], k=2,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    out = r.transform(df)
    X_out = _extract_output(out)
    # v0.9.199-b2-fix2: sklearn np.ma.average fallback (P10a).
    # Single donor (value 5.0) + receiver with no common dims (row 2,
    # all-NaN except x2) -> sklearn fills with unweighted mean of
    # non-NaN donors = 5.0. CADW fallback parity.
    from sklearn.impute import KNNImputer
    X_sk = KNNImputer(n_neighbors=2).fit_transform(X)
    assert np.allclose(X_out, X_sk, atol=1e-9), (
        f"sklearn={X_sk}\ncadw={X_out}")
    # Explicit: no NaN in col x1 (sklearn fills all receivers with
    # donor value via np.ma.average fallback)
    j1 = r.X_cols_kept.index("x1")
    assert np.all(np.isfinite(X_out[:, j1])), (
        f"col x1 has NaN: {X_out[:, j1]}")
    r.release_train_state()


# ---- CW-7: distance weights with duplicate rows ----

def test_distance_weights_duplicates():
    """P8a X3 case: duplicates -> exact match; filled=3.5."""
    X = np.array([
        [1.0, 2.0, np.nan],
        [1.0, 2.0, 3.0],
        [5.0, 5.0, 5.0],
        [1.0, 2.0, 4.0],
    ], dtype=np.float64)
    df = _mk_df(X)
    from sklearn.impute import KNNImputer
    sk = KNNImputer(n_neighbors=2, weights="distance")
    X_sk = sk.fit_transform(X)
    r = fit_knn_imputer(df, ["x0", "x1", "x2"], k=2,
                        weights="distance",
                        memory_guard=MemoryGuard(max_memory_mb=500))
    out = r.transform(df)
    X_cadw = _extract_output(out)
    assert np.allclose(X_cadw, X_sk, atol=1e-9),         f"sklearn={X_sk}, cadw={X_cadw}"
    # P8a expected value
    j2 = r.X_cols_kept.index("x2")
    assert abs(X_cadw[0, j2] - 3.5) < 1e-9, (
        f"duplicate case: expected 3.5, got {X_cadw[0, j2]}")
    r.release_train_state()


# ---- CW-8: transform new df (n_new rows) ----

def test_transform_new_data():
    """Train on X, transform Q (different rows)."""
    X_train = _make_with_nan(n=100, d=3, seed=5, frac=0.1)
    df_train = _mk_df(X_train)
    r = fit_knn_imputer(df_train, ["x0", "x1", "x2"], k=3,
                        memory_guard=MemoryGuard(max_memory_mb=500))

    # Q = 5 new rows with some NaN
    rng = np.random.RandomState(6)
    X_new = rng.randn(5, 3)
    X_new[0, 0] = np.nan
    X_new[2, 1] = np.nan
    df_new = _mk_df(X_new)
    out = r.transform(df_new)
    X_out = _extract_output(out)
    assert X_out.shape == (5, 3)
    # Filled values are finite (except possibly if no common dims — not
    # the case here since training has finite everywhere)
    assert np.isfinite(X_out[0, 0]), f"row0: {X_out[0]}"
    assert np.isfinite(X_out[2, 1]), f"row2: {X_out[2]}"
    # Non-missing values unchanged
    for i, j in ((1, 0), (3, 2), (4, 1)):
        assert X_out[i, j] == X_new[i, j],             f"non-missing ({i},{j}) modified"
    r.release_train_state()


# ---- CW-9: transform after release raises ----

def test_transform_requires_train_state():
    X = _make_with_nan(n=50, d=3, seed=7)
    df = _mk_df(X)
    r = fit_knn_imputer(df, ["x0", "x1", "x2"], k=3,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    r.release_train_state()
    assert r.train_spill_path is None
    with pytest.raises(RuntimeError, match="train_spill_path"):
        r.transform(df)


# ---- CW-10: release_train_state idempotent ----

def test_release_idempotent():
    X = _make_with_nan(n=50, d=3, seed=8)
    df = _mk_df(X)
    r = fit_knn_imputer(df, ["x0", "x1", "x2"], k=3,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    path = r.train_spill_path
    assert path is not None and Path(path).exists()
    r.release_train_state()
    assert r.train_spill_path is None
    assert not Path(path).exists()
    # second call no-op
    r.release_train_state()
    assert r.train_spill_path is None


# ---- CW-11: all rows with 1 NaN col -> sklearn parity ----

def test_all_rows_missing_one_col():
    """Column x1 NaN on every row except first (so not dropped)."""
    rng = np.random.RandomState(9)
    X = rng.randn(50, 3)
    X[1:, 1] = np.nan  # 49/50 NaN, 1 donor
    from sklearn.impute import KNNImputer
    sk = KNNImputer(n_neighbors=3)
    X_sk = sk.fit_transform(X)
    df = _mk_df(X)
    r = fit_knn_imputer(df, ["x0", "x1", "x2"], k=3,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    out = r.transform(df)
    X_cadw = _extract_output(out)
    max_diff = float(np.abs(X_cadw - X_sk).max())
    assert max_diff < 1e-9, f"max diff {max_diff:.2e}"
    r.release_train_state()


# ---- CW-12: empty df transform raises ----

def test_transform_empty_raises():
    X = _make_with_nan(n=50, d=3, seed=10)
    df = _mk_df(X)
    r = fit_knn_imputer(df, ["x0", "x1", "x2"], k=3,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    empty = CadwDataFrame.from_table(pa.table({
        "x0": pa.array([], type=pa.float64()),
        "x1": pa.array([], type=pa.float64()),
        "x2": pa.array([], type=pa.float64()),
    }), memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="empty"):
        r.transform(empty)
    r.release_train_state()


# ---- Input validation ----
def test_empty_X_cols_raises():
    X = np.random.RandomState(11).randn(10, 2)
    df = _mk_df(X)
    with pytest.raises(ValueError, match="X_cols is empty"):
        fit_knn_imputer(df, [], k=3,
                        memory_guard=MemoryGuard(max_memory_mb=500))


def test_bad_weights_raises():
    X = _make_with_nan(n=30, d=2, seed=12)
    df = _mk_df(X)
    with pytest.raises(ValueError, match="weights must be"):
        fit_knn_imputer(df, ["x0", "x1"], k=3, weights="median",
                        memory_guard=MemoryGuard(max_memory_mb=500))


def test_bad_k_raises():
    X = _make_with_nan(n=30, d=2, seed=13)
    df = _mk_df(X)
    with pytest.raises(ValueError, match="k must be"):
        fit_knn_imputer(df, ["x0", "x1"], k=0,
                        memory_guard=MemoryGuard(max_memory_mb=500))


from pathlib import Path  # noqa: E402  (used in release test)
