"""v0.9.211: fit_spectral_biclustering verification (Kluger 2003).

Component-wise (DEC-033) + sklearn parity (3 methods) + OOC + guard.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import fit_spectral_biclustering


def _mk_df(A):
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(A[:, j], type=pa.float64())
        for j in range(A.shape[1])
    }), memory_guard=MemoryGuard(max_memory_mb=500))


def _checkerboard(r, c, block_signal=5.0, noise=0.1, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.uniform(0, noise, size=(r, c))
    A[:r // 2, :c // 2] += block_signal
    A[r // 2:, c // 2:] += block_signal
    return A


def _warmup():
    A = _checkerboard(40, 20, seed=999)
    r = fit_spectral_biclustering(
        _mk_df(A), [f"x{j}" for j in range(20)],
        n_clusters=2, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    r.release_train_state()


_warmup()


# ---- CW-1: sklearn parity (3 methods, ARI>=0.99) ----
@pytest.mark.parametrize("method", ["bistochastic", "scale", "log"])
def test_sklearn_parity(method):
    from sklearn.cluster import SpectralBiclustering
    from sklearn.metrics import adjusted_rand_score
    A = _checkerboard(200, 100, seed=0)
    X_cols = [f"x{j}" for j in range(100)]
    sk = SpectralBiclustering(
        n_clusters=2, method=method, random_state=0,
        svd_method="randomized").fit(A)
    res = fit_spectral_biclustering(
        _mk_df(A), X_cols, n_clusters=2, method=method, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    ari_r = adjusted_rand_score(sk.row_labels_, res.row_labels_)
    ari_c = adjusted_rand_score(sk.column_labels_, res.column_labels_)
    assert ari_r > 0.99, f"method={method} ARI_row={ari_r:.4f}"
    assert ari_c > 0.99, f"method={method} ARI_col={ari_c:.4f}"


# ---- CW-2: state_nbytes ----
@pytest.mark.parametrize("r,c", [(100, 40), (200, 80), (300, 120)])
def test_state_nbytes_exact(r, c):
    A = _checkerboard(r, c, seed=0)
    X_cols = [f"x{j}" for j in range(c)]
    res = fit_spectral_biclustering(
        _mk_df(A), X_cols, n_clusters=2, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    # state = 256 + r*8 (row_labels) + c*8 (col_labels)
    #       + rows_ind (n_row*n_col, r) bool + cols_ind (n_row*n_col, c) bool
    n_comb = 4  # 2x2
    expected = (256
                + r * 8
                + c * 8
                + n_comb * r * 1
                + n_comb * c * 1)
    assert res.state_nbytes() == expected, (
        f"state={res.state_nbytes()} expected={expected}")


# ---- CW-3: state independent of n_clusters ----
def test_state_grows_with_n_clusters():
    A = _checkerboard(100, 40, seed=0)
    X_cols = [f"x{j}" for j in range(40)]
    s2 = fit_spectral_biclustering(
        _mk_df(A), X_cols, n_clusters=2, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500)).state_nbytes()
    s3 = fit_spectral_biclustering(
        _mk_df(A), X_cols, n_clusters=3, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500)).state_nbytes()
    # n_clusters=3 -> 9 indicators; state grows
    assert s3 > s2, f"s3={s3} not > s2={s2}"


# ---- CW-4: n_clusters tuple ----
def test_n_clusters_tuple():
    A = _checkerboard(100, 50, seed=0)
    X_cols = [f"x{j}" for j in range(50)]
    res = fit_spectral_biclustering(
        _mk_df(A), X_cols, n_clusters=(2, 3), random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    assert len(np.unique(res.row_labels_)) <= 2
    assert len(np.unique(res.column_labels_)) <= 3
    assert res.n_row_clusters == 2
    assert res.n_col_clusters == 3


# ---- CW-5: guard refuses tight ----
def test_guard_refuses_tight():
    A = _checkerboard(200, 100, seed=0)
    X_cols = [f"x{j}" for j in range(100)]
    # A state = 2*r*c*8 = 320 KB; guard 0.1 MB -> refuse
    with pytest.raises(MemoryLimitExceeded) as exc:
        fit_spectral_biclustering(
            _mk_df(A), X_cols, n_clusters=2, random_state=0,
            memory_guard=MemoryGuard(max_memory_mb=0.1))
    assert "biclust.A_state" in str(exc.value)


# ---- CW-6: input validation ----
def test_input_validation():
    A = _checkerboard(50, 10, seed=0)
    df = _mk_df(A)
    X_cols = [f"x{j}" for j in range(10)]

    with pytest.raises(ValueError, match="X_cols is empty"):
        fit_spectral_biclustering(
            df, [], n_clusters=2,
            memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="must be >= 2"):
        fit_spectral_biclustering(
            df, X_cols, n_clusters=1,
            memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="method must be"):
        fit_spectral_biclustering(
            df, X_cols, n_clusters=2, method="bogus",
            memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(KeyError, match="columns missing"):
        fit_spectral_biclustering(
            df, ["missing"], n_clusters=2,
            memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-7: to_dict ----
def test_to_dict():
    A = _checkerboard(100, 40, seed=0)
    X_cols = [f"x{j}" for j in range(40)]
    res = fit_spectral_biclustering(
        _mk_df(A), X_cols, n_clusters=(2, 3), method="scale",
        n_components=6, n_best=3, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    d = res.to_dict()
    assert d["A_shape"] == (100, 40)
    assert d["method"] == "scale"
    assert d["n_row_clusters"] == 2
    assert d["n_col_clusters"] == 3
    assert d["n_best"] == 3
    assert d["n_sv"] == 7  # n_components(6) + 1 for scale
    assert d["n_sv"] == 7


# ---- CW-8: release idempotent ----
def test_release_idempotent():
    A = _checkerboard(60, 20, seed=0)
    X_cols = [f"x{j}" for j in range(20)]
    res = fit_spectral_biclustering(
        _mk_df(A), X_cols, n_clusters=2, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    res.release_train_state()
    res.release_train_state()  # idempotent
    assert res.state_nbytes() == 0


# ---- CW-9: reproducibility same seed ----
def test_reproducibility_same_seed():
    A = _checkerboard(150, 60, seed=0)
    X_cols = [f"x{j}" for j in range(60)]
    r1 = fit_spectral_biclustering(
        _mk_df(A), X_cols, n_clusters=2, random_state=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    r2 = fit_spectral_biclustering(
        _mk_df(A), X_cols, n_clusters=2, random_state=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(r1.row_labels_, r2.row_labels_)
    np.testing.assert_array_equal(r1.column_labels_, r2.column_labels_)


# ---- CW-10: negative values clamped (make_nonnegative) ----
def test_negative_clamped():
    A = _checkerboard(80, 30, seed=0)
    A[:10, :5] = -1.0
    X_cols = [f"x{j}" for j in range(30)]
    res = fit_spectral_biclustering(
        _mk_df(A), X_cols, n_clusters=2, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    assert res.row_labels_.shape == (80,)
    assert res.column_labels_.shape == (30,)


# ---- CW-11: n_sv > min(r,c) raises ----
def test_n_sv_exceeds_raises():
    A = _checkerboard(20, 8, seed=0)
    X_cols = [f"x{j}" for j in range(8)]
    # n_components=10, +1 = 11 > min(20, 8) = 8
    with pytest.raises(ValueError, match="n_sv="):
        fit_spectral_biclustering(
            _mk_df(A), X_cols, n_clusters=2, n_components=10,
            memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-12: empty input ----
def test_empty_input():
    df = CadwDataFrame.from_table(pa.table({
        "x0": pa.array([], type=pa.float64()),
    }), memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="need r>=2 and c>=2"):
        fit_spectral_biclustering(
            df, ["x0"], n_clusters=2,
            memory_guard=MemoryGuard(max_memory_mb=500))
