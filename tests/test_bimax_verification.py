"""v0.9.213: fit_bimax verification (Prelic 2006).

Checkerboard exact + make_biclusters recovery + state + guard +
refuse_at_guard + validation.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import fit_bimax


def _mk_df(A):
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(A[:, j].astype(np.float64), type=pa.float64())
        for j in range(A.shape[1])
    }), memory_guard=MemoryGuard(max_memory_mb=500))


def _checkerboard_4x4():
    return np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=np.float64)


def _warmup():
    r = fit_bimax(_mk_df(_checkerboard_4x4()),
                   [f"x{j}" for j in range(4)],
                   memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ---- CW-1: checkerboard 4x4 -> exactly 2 biclusters (2x2 each) ----
def test_checkerboard_exact():
    A = _checkerboard_4x4()
    r = fit_bimax(_mk_df(A), [f"x{j}" for j in range(4)],
                   min_rows=2, min_cols=2,
                   memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_biclusters_ == 2, f"n={r.n_biclusters_}"
    sizes = sorted([(len(R), len(C)) for R, C in r.biclusters_])
    assert sizes == [(2, 2), (2, 2)], f"sizes: {sizes}"
    # Verify each is all-1 in A
    for R, C in r.biclusters_:
        sub = A[np.ix_(R, C)]
        assert np.all(sub == 1), f"not all-1: R={R}, C={C}"


# ---- CW-2: all-zero -> 0 biclusters ----
def test_all_zero():
    r = fit_bimax(_mk_df(np.zeros((10, 10))),
                   [f"x{j}" for j in range(10)],
                   memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_biclusters_ == 0


# ---- CW-3: all-one -> 1 bicluster (full) ----
def test_all_one():
    r = fit_bimax(_mk_df(np.ones((10, 10))),
                   [f"x{j}" for j in range(10)],
                   memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_biclusters_ == 1
    R, C = r.biclusters_[0]
    assert R == tuple(range(10))
    assert C == tuple(range(10))


# ---- CW-4: make_biclusters GT recovery (no noise) ----
def test_make_biclusters_no_noise():
    from sklearn.datasets import make_biclusters
    X, rows_gt, cols_gt = make_biclusters(
        shape=(60, 40), n_clusters=2, noise=0.0, random_state=0)
    Xb = (X > np.median(X)).astype(np.float64)
    r = fit_bimax(_mk_df(Xb), [f"x{j}" for j in range(40)],
                   min_rows=3, min_cols=3, max_iterations=10000,
                   memory_guard=MemoryGuard(max_memory_mb=500))
    # Verify all GT biclusters are subsets of some found bicluster
    for k in range(2):
        gt_R = tuple(i for i in range(60) if rows_gt[k][i])
        gt_C = tuple(j for j in range(40) if cols_gt[k][j])
        gt_Rs = set(gt_R)
        gt_Cs = set(gt_C)
        found = False
        for R, C in r.biclusters_:
            if gt_Rs <= set(R) and gt_Cs <= set(C):
                found = True
                break
        assert found, f"GT bicluster {k} ({len(gt_R)}x{len(gt_C)}) not recovered"


# ---- CW-5: state_nbytes includes biclusters ----
def test_state_nbytes():
    A = _checkerboard_4x4()
    r = fit_bimax(_mk_df(A), [f"x{j}" for j in range(4)],
                   memory_guard=MemoryGuard(max_memory_mb=500))
    # state >= 256 header; each bicluster contributes R*8 + C*8
    # 2 biclusters of (2,2): 2 * (2*8 + 2*8) = 64; total = 320
    expected = 256 + 2 * (2 * 8 + 2 * 8)
    assert r.state_nbytes() == expected, (
        f"state={r.state_nbytes()} expected={expected}")


# ---- CW-6: guard refuses at A materialization ----
def test_guard_refuses_tight():
    A = np.ones((100, 100), dtype=np.float64)
    # A state = 100*100 = 10 KB; guard 0.001 MB = 1 KB -> refuse
    with pytest.raises(MemoryLimitExceeded) as exc:
        fit_bimax(_mk_df(A), [f"x{j}" for j in range(100)],
                   memory_guard=MemoryGuard(max_memory_mb=0.001))
    assert "bimax.A_state" in str(exc.value)


# ---- CW-7: max_iterations cap -> MemoryLimitExceeded ----
def test_max_iterations_exceeded():
    # Random binary matrix forces many splits
    rng = np.random.RandomState(0)
    A = (rng.rand(50, 50) > 0.5).astype(np.float64)
    with pytest.raises(MemoryLimitExceeded, match="max_iterations"):
        fit_bimax(_mk_df(A), [f"x{j}" for j in range(50)],
                   max_iterations=10,  # very tight
                   memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-8: threshold param respected ----
def test_threshold():
    # Values 0.1..0.9, threshold=0.5 -> binary pattern
    A = np.array([[0.6, 0.7, 0.1, 0.2],
                  [0.8, 0.9, 0.3, 0.1],
                  [0.1, 0.2, 0.8, 0.9],
                  [0.2, 0.3, 0.7, 0.6]], dtype=np.float64)
    r = fit_bimax(_mk_df(A), [f"x{j}" for j in range(4)],
                   threshold=0.5, min_rows=2, min_cols=2,
                   memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_biclusters_ == 2
    sizes = sorted([(len(R), len(C)) for R, C in r.biclusters_])
    assert sizes == [(2, 2), (2, 2)]


# ---- CW-9: validation ----
def test_input_validation():
    A = _checkerboard_4x4()
    df = _mk_df(A)
    X_cols = [f"x{j}" for j in range(4)]

    with pytest.raises(ValueError, match="X_cols is empty"):
        fit_bimax(df, [], memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="min_rows"):
        fit_bimax(df, X_cols, min_rows=0,
                   memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="max_iterations"):
        fit_bimax(df, X_cols, max_iterations=0,
                   memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(KeyError, match="columns missing"):
        fit_bimax(df, ["missing"], memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-10: to_dict ----
def test_to_dict():
    A = _checkerboard_4x4()
    r = fit_bimax(_mk_df(A), [f"x{j}" for j in range(4)],
                   memory_guard=MemoryGuard(max_memory_mb=500))
    d = r.to_dict()
    assert d["A_shape"] == (4, 4)
    assert d["n_biclusters"] == 2
    assert d["threshold"] == 0.5
    assert d["min_rows"] == 2
    assert d["min_cols"] == 2
    assert len(d["sizes"]) == 2
    assert all(s == (2, 2) for s in d["sizes"])


# ---- CW-11: nested bicluster filtered (inclusion-maximal) ----
def test_inclusion_maximal():
    # 4x4 matrix with overlapping 1s:
    #   1 1 1 0
    #   1 1 1 0
    #   1 1 1 0
    #   0 0 0 0
    # Expected: 1 bicluster (3x3), not also 2x2 sub-patterns
    A = np.array([
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=np.float64)
    r = fit_bimax(_mk_df(A), [f"x{j}" for j in range(4)],
                   min_rows=2, min_cols=2,
                   memory_guard=MemoryGuard(max_memory_mb=500))
    # Should find the 3x3 maximal, not sub-patterns
    assert r.n_biclusters_ == 1, f"n={r.n_biclusters_}"
    R, C = r.biclusters_[0]
    assert len(R) == 3, f"R size {len(R)} (expected 3)"
    assert len(C) == 3, f"C size {len(C)} (expected 3)"


# ---- CW-12: min_rows/min_cols filter ----
def test_min_size_filter():
    # 4x4 checkerboard, but ask for min_rows=3 -> no bicluster >= 3x3
    A = _checkerboard_4x4()
    r = fit_bimax(_mk_df(A), [f"x{j}" for j in range(4)],
                   min_rows=3, min_cols=3,
                   memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_biclusters_ == 0, (
        f"min_rows=3/min_cols=3 should filter 2x2 blocks; got "
        f"{r.n_biclusters_}")
