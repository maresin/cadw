"""v0.9.207: fit_spectral_coclustering verification (Dhillon 2001).

Component-wise (DEC-033) + sklearn parity + OOC + guard.
"""
from __future__ import annotations

import ctypes
import gc
import statistics

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded, measure_peak_traced
from cadw.ml import fit_spectral_coclustering


def _mk_df(A):
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(A[:, j], type=pa.float64())
        for j in range(A.shape[1])
    }), memory_guard=MemoryGuard(max_memory_mb=500))


def _collect(arr_res):
    if arr_res is None:
        return None
    if hasattr(arr_res, "shape"):
        return np.asarray(arr_res)
    parts = []
    for chunk in arr_res:
        a = np.asarray(chunk)
        if a.ndim == 1:
            a = a[:, None]
        parts.append(a)
    return np.vstack(parts) if parts else np.empty((0,))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _checkerboard(r, c, block_signal=5.0, noise=0.1, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.uniform(0, noise, size=(r, c))
    A[:r // 2, :c // 2] += block_signal
    A[r // 2:, c // 2:] += block_signal
    return A


def _warmup():
    A = _checkerboard(40, 20, seed=999)
    r = fit_spectral_coclustering(_mk_df(A), [f"x{j}" for j in range(20)],
                                  2, random_state=0,
                                  memory_guard=MemoryGuard(max_memory_mb=500))
    _collect(r.row_labels_)


_warmup()


# ---- CW-1: state_nbytes exact ----
@pytest.mark.parametrize("r,c,k", [(100, 30, 2), (200, 50, 3), (300, 80, 4)])
def test_state_nbytes_exact(r, c, k):
    """state = c*f64 (col_labels) + c*f64 (col_diag) + k_used*f64 (sigma)
    + 256. Independent of r."""
    A = _checkerboard(r, c, seed=0)
    X_cols = [f"x{j}" for j in range(c)]
    res = fit_spectral_coclustering(
        _mk_df(A), X_cols, k, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    k_sv = 1 + int(np.ceil(np.log2(k)))
    k_used = k_sv - 1
    expected = (c * 8      # col_labels
                + c * 8    # col_diag
                + k_used * 8  # sigma
                + 256)
    assert res.state_nbytes() == expected, (
        f"r={r} c={c} k={k}: state={res.state_nbytes()} expected={expected}")


# ---- CW-2: state independent of r ----
def test_state_independent_of_r():
    """state = f(c, k) only; no r-dependence."""
    c, k = 40, 3
    X_cols = [f"x{j}" for j in range(c)]
    A_small = _checkerboard(100, c, seed=0)
    A_large = _checkerboard(1000, c, seed=0)
    r1 = fit_spectral_coclustering(_mk_df(A_small), X_cols, k,
                                    random_state=0,
                                    memory_guard=MemoryGuard(max_memory_mb=500))
    r2 = fit_spectral_coclustering(_mk_df(A_large), X_cols, k,
                                    random_state=0,
                                    memory_guard=MemoryGuard(max_memory_mb=500))
    assert r1.state_nbytes() == r2.state_nbytes(), (
        f"state depends on r: {r1.state_nbytes()} vs {r2.state_nbytes()}")


# ---- CW-3: sklearn randomized parity (k=2 high contrast) ----
@pytest.mark.parametrize("r,c", [(100, 50), (200, 100), (300, 120)])
def test_sklearn_parity_k2(r, c):
    """sklearn SpectralCoclustering(svd_method='randomized', k=2) parity."""
    from sklearn.cluster import SpectralCoclustering
    from sklearn.metrics import adjusted_rand_score
    A = _checkerboard(r, c, seed=0)
    X_cols = [f"x{j}" for j in range(c)]
    sk = SpectralCoclustering(n_clusters=2, random_state=0,
                               svd_method="randomized").fit(A)
    res = fit_spectral_coclustering(
        _mk_df(A), X_cols, 2, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    r_lbl = _collect(res.row_labels_).ravel()
    c_lbl = res.column_labels_
    ari_r = adjusted_rand_score(sk.row_labels_, r_lbl)
    ari_c = adjusted_rand_score(sk.column_labels_, c_lbl)
    assert ari_r > 0.999, f"r={r} c={c} ARI_row={ari_r:.4f}"
    assert ari_c > 0.999, f"r={r} c={c} ARI_col={ari_c:.4f}"


# ---- CW-4: sklearn parity at k=3 (our exact vs their approximate) ----
def test_sklearn_parity_k3_high_contrast():
    """k=3 high contrast: our exact vs sklearn randomized >= 0.9 ARI
    (not 1.0 — sklearn randomized approximate; our exact deterministic)."""
    from sklearn.cluster import SpectralCoclustering
    from sklearn.metrics import adjusted_rand_score
    A = _checkerboard(200, 100, block_signal=10.0, noise=0.1, seed=0)
    X_cols = [f"x{j}" for j in range(100)]
    sk = SpectralCoclustering(n_clusters=3, random_state=0,
                               svd_method="randomized").fit(A)
    res = fit_spectral_coclustering(
        _mk_df(A), X_cols, 3, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    r_lbl = _collect(res.row_labels_).ravel()
    c_lbl = res.column_labels_
    ari_r = adjusted_rand_score(sk.row_labels_, r_lbl)
    ari_c = adjusted_rand_score(sk.column_labels_, c_lbl)
    # sklearn randomized is approximate on k>=3; our exact differs but
    # should be highly consistent on high contrast
    assert ari_r >= 0.90, f"ARI_row={ari_r:.4f}"
    assert ari_c >= 0.90, f"ARI_col={ari_c:.4f}"


# ---- CW-5: cluster count ----
@pytest.mark.parametrize("k", [2, 3, 4])
def test_n_clusters_matches(k):
    A = _checkerboard(150, 60, seed=0)
    X_cols = [f"x{j}" for j in range(60)]
    res = fit_spectral_coclustering(
        _mk_df(A), X_cols, k, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    r_lbl = _collect(res.row_labels_).ravel()
    # high-contrast: exactly k distinct labels expected
    assert len(np.unique(r_lbl)) == k
    assert len(np.unique(res.column_labels_)) == k


# ---- CW-6: OOC in r (state growth ratio < 2.0 over 10x) ----
def test_ooc_in_r():
    """state_nbytes independent of r (already verified), but also:
    fit completes at r=1000 and r=10000 with same guard."""
    c, k = 30, 2
    X_cols = [f"x{j}" for j in range(c)]
    A_small = _checkerboard(500, c, seed=0)
    A_large = _checkerboard(5000, c, seed=0)
    res_small = fit_spectral_coclustering(
        _mk_df(A_small), X_cols, k, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    res_large = fit_spectral_coclustering(
        _mk_df(A_large), X_cols, k, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    s_small = res_small.state_nbytes()
    s_large = res_large.state_nbytes()
    assert s_small == s_large, f"state grew with r: {s_small} vs {s_large}"


# ---- CW-7: guard refuses at tight budget (c=1e3 -> Gram 8 MB) ----
def test_guard_refuses_tight():
    """c=1000 -> Gram state = 1000*1000*8 = 8 MB; guard 0.1 MB -> refuse."""
    c = 1000
    A = _checkerboard(10, c, seed=0)  # small r, but c defines Gram
    X_cols = [f"x{j}" for j in range(c)]
    with pytest.raises(MemoryLimitExceeded) as exc_info:
        fit_spectral_coclustering(
            _mk_df(A), X_cols, 2, random_state=0,
            memory_guard=MemoryGuard(max_memory_mb=0.1))
    assert "coclust.state" in str(exc_info.value)


# ---- CW-8: input validation ----
def test_input_validation():
    A = _checkerboard(50, 10, seed=0)
    df = _mk_df(A)
    X_cols = [f"x{j}" for j in range(10)]

    with pytest.raises(ValueError, match="X_cols is empty"):
        fit_spectral_coclustering(df, [], 2,
                                   memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(TypeError, match="n_clusters must be int"):
        fit_spectral_coclustering(df, X_cols, "2",
                                   memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="n_clusters must be >= 1"):
        fit_spectral_coclustering(df, X_cols, 0,
                                   memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(KeyError, match="X_cols missing"):
        fit_spectral_coclustering(df, ["missing_col"], 2,
                                   memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="k_used="):
        # k_sv = 1+ceil(log2(2048)) = 12; k_used = 11 > min(50, 10) = 10
        fit_spectral_coclustering(df, X_cols, 2048,
                                   memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-9: empty matrix ----
def test_empty_input():
    df = CadwDataFrame.from_table(pa.table({
        "x0": pa.array([], type=pa.float64()),
        "x1": pa.array([], type=pa.float64()),
    }), memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="need r>=1 and c>=1"):
        fit_spectral_coclustering(df, ["x0", "x1"], 2,
                                   memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-10: to_dict ----
def test_to_dict():
    A = _checkerboard(100, 40, seed=0)
    X_cols = [f"x{j}" for j in range(40)]
    res = fit_spectral_coclustering(
        _mk_df(A), X_cols, 2, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    d = res.to_dict()
    assert d["n_clusters"] == 2
    assert d["r"] == 100
    assert d["c"] == 40
    assert d["k_sv"] == 2  # 1 + ceil(log2(2)) = 2
    assert isinstance(d["sigma"], list)
    assert len(d["sigma"]) == 1  # k_sv - n_discard = 1


# ---- CW-11: all-zero column doesn't crash ----
def test_zero_column():
    A = _checkerboard(100, 20, seed=0)
    A[:, 5] = 0.0  # zero column: col_sum=0 -> col_diag=0 (handled)
    X_cols = [f"x{j}" for j in range(20)]
    res = fit_spectral_coclustering(
        _mk_df(A), X_cols, 2, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    assert res.column_labels_.shape == (20,)
    assert res.col_diag_[5] == 0.0


# ---- CW-12: negative values clamped to 0 (make_nonnegative) ----
def test_negative_values_clamped():
    A = _checkerboard(100, 20, seed=0)
    A[:10, :5] = -1.0  # negative -> clamped to 0
    X_cols = [f"x{j}" for j in range(20)]
    res = fit_spectral_coclustering(
        _mk_df(A), X_cols, 2, random_state=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    # no crash; col_diag well-defined (non-negative input)
    assert np.all(np.isfinite(res.col_diag_))
