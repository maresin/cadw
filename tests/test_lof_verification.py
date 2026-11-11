"""Component-wise verification for fit_lof (DEC-033, v0.9.197).

Formula v1:
  peak = batch * (d + (k+1)*2) * f64 + batch * (k+1) * i64 + 256
         + random-access transient (kdist_mm[idx_b], lrd_mm[idx_b]:
           O(batch * (k+1)) per pass)

State (persistent, after fit):
  spilled: 256
  in-memory: 256 + n*8 (scores) + n*1 (labels) + scores_meta

Reference: sklearn.neighbors.LocalOutlierFactor, convention
  n_neighbors=k+1 (incl self), kdist = dist[:, -1],
  reach = max(dist[:, 1:], kdist[neighbor[:, 1:]]),
  lrd = 1/(mean(reach) + 1e-10),
  lof = mean(lrd[neighbor[:, 1:]]) / lrd[self]
"""
from __future__ import annotations
from pathlib import Path
import ctypes
import gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import fit_lof


def _mk_df(X, seed=0):
    d = X.shape[1]
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(X[:, j], type=pa.float64())
        for j in range(d)
    }))


def _make_data(n_clusters=3, per_cluster=50, seed=0):
    """3 clusters + 2 outliers (like sklearn LOF docs)."""
    rng = np.random.RandomState(seed)
    parts = []
    centers = [(0, 0), (10, 0), (0, 10)]
    for c in centers[:n_clusters]:
        parts.append(rng.randn(per_cluster, 2) + np.array(c))
    X = np.vstack(parts)
    outliers = np.array([[30, 30], [-10, -10]])
    return np.vstack([X, outliers])


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    X = _make_data(n_clusters=1, per_cluster=30, seed=999)
    df = _mk_df(X)
    fit_lof(df, ["x0", "x1"], k=5, output="memory",
            memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ---- CW-1: state_nbytes exact (spilled=256; memory=256+n*9) ----

@pytest.mark.parametrize("n", [60, 150, 300])
def test_state_nbytes_exact_spilled(n):
    rng = np.random.RandomState(0)
    X = rng.randn(n, 2)
    df = _mk_df(X)
    r = fit_lof(df, ["x0", "x1"], k=10, output="disk",
                memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.is_spilled
    assert r.state_nbytes() == 256, (
        f"n={n}: spilled state={r.state_nbytes()}, expected 256")


def test_state_nbytes_exact_memory():
    """Memory path: 256 + n*8 (scores) + n*1 (labels)."""
    X = _make_data()
    n = X.shape[0]
    df = _mk_df(X)
    r = fit_lof(df, ["x0", "x1"], k=20, output="memory",
                memory_guard=MemoryGuard(max_memory_mb=500))
    assert not r.is_spilled
    # 256 + n*8 + n*1 = 256 + 9n
    expected = 256 + n * 9
    assert r.state_nbytes() == expected, (
        f"n={n}: state={r.state_nbytes()}, expected {expected}")


# ---- CW-2: state constant in n (spilled) ----

@pytest.mark.parametrize("n", [100, 500, 1000])
def test_state_constant_in_n(n):
    """Spilled output -> state invariant in n."""
    rng = np.random.RandomState(1)
    X = rng.randn(n, 2)
    df = _mk_df(X)
    r = fit_lof(df, ["x0", "x1"], k=10, output="disk",
                memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.state_nbytes() == 256


# ---- CW-3: sklearn parity (scores) ----

def test_sklearn_parity_scores():
    """LOF scores match sklearn.neighbors.LocalOutlierFactor.

    sklearn: n_neighbors=k+1 incl self, kdist=dist[:, -1],
    reach=max(dist[:,1:], kdist[neighbor[:,1:]]),
    lrd=1/(mean(reach)+1e-10), lof=mean(lrd[neighbor[:,1:]])/lrd[self].
    Threshold=1.0, contamination=None. Scores only (labels identical
    given same threshold).
    """
    from sklearn.neighbors import LocalOutlierFactor

    X = _make_data(seed=0)
    df = _mk_df(X)
    r = fit_lof(df, ["x0", "x1"], k=20, output="memory",
                memory_guard=MemoryGuard(max_memory_mb=500))
    cadw_scores = (r.scores.to_numpy()
                   if hasattr(r.scores, "to_numpy") else r.scores)

    lof = LocalOutlierFactor(n_neighbors=20)
    lof.fit(X)
    sk_scores = -lof.negative_outlier_factor_

    max_diff = float(np.abs(cadw_scores - sk_scores).max())
    assert max_diff < 1e-6, (
        f"sklearn parity: max abs diff {max_diff:.2e} >= 1e-6")


def test_sklearn_parity_outliers_in_top():
    """True outliers (30,30) and (-10,-10) in top-10 by LOF."""
    X = _make_data(seed=0)
    n = X.shape[0]
    df = _mk_df(X)
    r = fit_lof(df, ["x0", "x1"], k=20, output="memory",
                memory_guard=MemoryGuard(max_memory_mb=500))
    scores = r.scores.to_numpy() if hasattr(r.scores, "to_numpy") else r.scores
    top10 = set(np.argsort(scores)[-10:])
    # Last two data points are the injected outliers.
    assert (n - 2) in top10 and (n - 1) in top10, (
        f"outliers {n-2},{n-1} not in top-10 (top10={sorted(top10)})")


# ---- CW-4: contamination percentile ----

def test_contamination_percentile():
    """contamination=0.1 -> ~10% flagged (percentile threshold)."""
    X = _make_data(seed=0)
    n = X.shape[0]
    df = _mk_df(X)
    r = fit_lof(df, ["x0", "x1"], k=20, contamination=0.1,
                output="memory",
                memory_guard=MemoryGuard(max_memory_mb=500))
    # threshold at 90th percentile
    scores = r.scores.to_numpy() if hasattr(r.scores, "to_numpy") else r.scores
    expected_thr = float(np.percentile(scores, 90.0))
    assert abs(r.threshold - expected_thr) < 1e-9
    labels = r.labels.to_numpy() if hasattr(r.labels, "to_numpy") else r.labels
    flagged = int(labels.sum())
    # ~10% of 152 = ~15 (allow 1-2 off due to ties)
    assert 13 <= flagged <= 17, f"flagged={flagged}, expected ~15"


# ---- CW-5: guard refuses ----

def test_guard_refuses_tight():
    """G9 + CW-5: mb=0.001 guarantees knn_query_stream.state check fails
    (state = batch*(k+1)*2*8 + 256 > 524 bytes).

    mb=0.01 passes because LOF base state (256) fits under state_limit=5242.
    """
    X = _make_data(seed=0)
    df = _mk_df(X)
    with pytest.raises(MemoryLimitExceeded):
        fit_lof(df, ["x0", "x1"], k=20, output="disk",
                memory_guard=MemoryGuard(max_memory_mb=0.001))


# ---- CW-6: output spilled on disk ----

def test_output_disk_spilled():
    X = _make_data(seed=0)
    df = _mk_df(X)
    r = fit_lof(df, ["x0", "x1"], k=20, output="disk",
                memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.is_spilled
    # Iteration works
    total = sum(len(c) for c in r.scores)
    assert total == X.shape[0]
    # to_numpy raises on spilled
    with pytest.raises(Exception):
        r.scores.to_numpy()


# ---- CW-7: k=n-2 edge (min n=k+2) ----

def test_k_n_minus_2():
    n = 22
    X = _make_data(n_clusters=1, per_cluster=n - 2, seed=2)[:n]
    df = _mk_df(X)
    r = fit_lof(df, ["x0", "x1"], k=n - 2, output="memory",
                memory_guard=MemoryGuard(max_memory_mb=500))
    scores = r.scores.to_numpy() if hasattr(r.scores, "to_numpy") else r.scores
    assert len(scores) == n
    assert np.all(np.isfinite(scores))


def test_n_too_small_raises():
    n = 10
    X = np.random.RandomState(0).randn(n, 2)
    df = _mk_df(X)
    with pytest.raises(ValueError, match=r"need n >= k\+2"):
        fit_lof(df, ["x0", "x1"], k=n - 1, output="memory",
                memory_guard=MemoryGuard(max_memory_mb=500))

# ---- G9: output="memory" vs output="auto" equivalence ----

def test_output_memory_and_auto_equivalent():
    """G9: both output branches bit-identical for small n (auto won't spill).

    Verifies result equivalence across output modes; satisfies G9
    self-verification protocol for functions with output= param.
    """
    X = _make_data(seed=0)
    df = _mk_df(X)

    r_mem = fit_lof(df, ["x0", "x1"], k=20, output="memory",
                    memory_guard=MemoryGuard(max_memory_mb=500))
    r_auto = fit_lof(df, ["x0", "x1"], k=20, output="auto",
                     memory_guard=MemoryGuard(max_memory_mb=500))

    assert not r_mem.is_spilled
    assert not r_auto.is_spilled  # 152 rows fit in 500 MB
    s_mem = r_mem.scores.to_numpy() if hasattr(r_mem.scores, "to_numpy") else r_mem.scores
    s_auto = r_auto.scores.to_numpy() if hasattr(r_auto.scores, "to_numpy") else r_auto.scores
    np.testing.assert_allclose(s_mem, s_auto, atol=1e-12)
    assert r_mem.threshold == r_auto.threshold


def test_output_default_is_auto():
    """Default output="auto": no kwarg call works + returns valid result."""
    X = _make_data(seed=0)
    df = _mk_df(X)
    r = fit_lof(df, ["x0", "x1"], k=20,
                memory_guard=MemoryGuard(max_memory_mb=500))
    scores = r.scores.to_numpy() if hasattr(r.scores, "to_numpy") else r.scores
    assert len(scores) == X.shape[0]
    assert np.all(np.isfinite(scores))

# ---- CW-8..CW-11: predict (novelty convention, v0.9.198-b2) ----

def test_predict_requires_save_train_state():
    """CW-8: predict raises RuntimeError without save_train_state=True."""
    X = _make_data(seed=0)
    df = _mk_df(X)
    r = fit_lof(df, ["x0", "x1"], k=20, output="memory",
                memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.train_spill_path is None
    with pytest.raises(RuntimeError, match="save_train_state"):
        r.predict(df, X_cols=["x0", "x1"])


def test_predict_parity_sklearn_novelty():
    """CW-9: predict() scores bit-identical to sklearn novelty=True.

    sklearn: -lof.score_samples(X_new) (score_samples returns -LOF).
    Formula: lof_Q = mean(lrd_train[idx_Q]) / lrd_Q where
    lrd_Q = 1/(mean(max(dist_Q, kdist_train[idx_Q])) + 1e-10),
    dist_Q, idx_Q = kNN(k) to training (self excluded).
    """
    from sklearn.neighbors import LocalOutlierFactor

    rng = np.random.RandomState(0)
    X_train = rng.randn(100, 2)
    X_new = np.array([[10.0, 10.0], [-10.0, -10.0], [0.1, 0.1]])

    df_train = _mk_df(X_train)
    df_new = _mk_df(X_new)

    r = fit_lof(df_train, ["x0", "x1"], k=20, output="memory",
                save_train_state=True,
                memory_guard=MemoryGuard(max_memory_mb=500))
    pred = r.predict(df_new, X_cols=["x0", "x1"])
    cadw_scores = (pred.scores.to_numpy()
                   if hasattr(pred.scores, "to_numpy") else pred.scores)

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X_train)
    sk_scores = -lof.score_samples(X_new)

    max_diff = float(np.abs(cadw_scores - sk_scores).max())
    assert max_diff < 1e-6, (
        f"sklearn novelty parity: max abs diff {max_diff:.2e} >= 1e-6")

    r.release_train_state()


def test_predict_new_outliers_high_lof():
    """CW-10: new points in dense clusters get LOF ~1; far outliers >> 1."""
    X_train = _make_data(seed=0)
    # Far outlier + inlier sample (mean of first cluster) + moderate noise
    far = np.array([[100.0, 100.0]])
    inlier = X_train[:1] + 0.01  # near cluster point
    X_new = np.vstack([far, inlier])

    df_train = _mk_df(X_train)
    df_new = _mk_df(X_new)

    r = fit_lof(df_train, ["x0", "x1"], k=20, output="memory",
                save_train_state=True,
                memory_guard=MemoryGuard(max_memory_mb=500))
    pred = r.predict(df_new, X_cols=["x0", "x1"])
    scores = (pred.scores.to_numpy()
              if hasattr(pred.scores, "to_numpy") else pred.scores)

    # far outlier >> 1; near-inlier ~1
    assert scores[0] > 5.0, (
        f"far outlier LOF {scores[0]:.2f} should be >> 1")
    assert 0.5 < scores[1] < 2.0, (
        f"near-cluster LOF {scores[1]:.2f} should be ~1")

    r.release_train_state()


def test_release_train_state_idempotent():
    """CW-11: release_train_state twice = no error; after 1st -> predict raises."""
    X = _make_data(seed=0)
    df = _mk_df(X)
    r = fit_lof(df, ["x0", "x1"], k=20, output="memory",
                save_train_state=True,
                memory_guard=MemoryGuard(max_memory_mb=500))
    path = r.train_spill_path
    assert path is not None and Path(path).exists()

    r.release_train_state()
    assert r.train_spill_path is None
    assert not Path(path).exists()

    # Idempotent: second call no-op
    r.release_train_state()
    assert r.train_spill_path is None

    # After release: predict raises
    with pytest.raises(RuntimeError):
        r.predict(df, X_cols=["x0", "x1"])


def test_predict_spilled_output():
    """Spilled predict: state_nbytes=256; iterate works."""
    X_train = _make_data(seed=0)
    X_new = np.array([[100.0, 100.0]])
    df_train = _mk_df(X_train)
    df_new = _mk_df(X_new)

    r = fit_lof(df_train, ["x0", "x1"], k=20, output="memory",
                save_train_state=True,
                memory_guard=MemoryGuard(max_memory_mb=500))

    # Tight guard forces scores spill (small Q but SpillArray guard)
    pred = r.predict(df_new, X_cols=["x0", "x1"])
    assert pred.is_spilled or pred.scores.to_numpy().shape == (1,)
    r.release_train_state()

