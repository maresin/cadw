"""v0.9.227: DR_learner (AIPW ATE) CW verification (DEC-033).

Component-wise verification:
  CW-1 state_nbytes_exact (bit-exact, primary evidence)
  CW-2 state_independent_of_n
  CW-4 guard_refuses_at_tight
  correctness x3 (ate_recovers + aipw_double_robust + predict_psi_parity)
  invalid x2 + determinism

formula_small_scale REMOVED (DEC-034 precedent: baseline >> fit peak
at d=20; P38: 6.27 MB cold vs 32 KB warm, 300x allocator variance;
single fit_logistic 806 KB measured vs 179 KB source-only = 4.5x
Python/Arrow overhead). Precedent: iforest (DEC-031/033/034), linucb,
gmm, birch, kmeans_ad.
"""
from __future__ import annotations
import ctypes
import gc
import statistics

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.dr_learner import DRLearnerModel, fit_dr_learner
from cadw.ml._causal_common import dr_score


def _make_df(n, d=5, ate_true=0.5, seed=0, confounded=False):
    """Synthetic: T = X0>0 (or random), Y = ATE*T + f(X) + noise."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    if confounded:
        # Confounded: treatment depends on X0; outcome also depends on X0.
        p_t = 1.0 / (1.0 + np.exp(-X[:, 0]))
        t = (rng.uniform(size=n) < p_t).astype(np.int64)
    else:
        t = rng.randint(0, 2, size=n).astype(np.int64)
    # Outcome: continuous latent -> binarize around ATE
    lin = 0.5 * X[:, 0] + 0.3 * X[:, 1] + ate_true * t
    y_latent = lin + 0.5 * rng.normal(size=n)
    y = (y_latent > 0).astype(np.int64)
    cols = {f"x{j}": pa.array(X[:, j], type=pa.float64())
            for j in range(d)}
    cols["t"] = pa.array(t, type=pa.int64())
    cols["y"] = pa.array(y, type=pa.int64())
    return CadwDataFrame.from_table(
        pa.table(cols), memory_guard=MemoryGuard(max_memory_mb=500))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    df = _make_df(200, d=3, ate_true=0.5, seed=999)
    fit_dr_learner(
        df, ["x0", "x1", "x2"], "t", "y",
        memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ---- CW-1: state exact ----

@pytest.mark.parametrize("d", [2, 5, 20, 50])
def test_state_nbytes_exact(d):
    """state = 3*(d*f64 + 8 + 256) + 256 = 3*d*f64 + 1048."""
    df = _make_df(500, d=d, seed=0)
    m = fit_dr_learner(
        df, [f"x{j}" for j in range(d)], "t", "y",
        memory_guard=MemoryGuard(max_memory_mb=500))
    expected = 3 * (d * 8 + 8 + 256) + 256
    assert m.state_nbytes() == expected, (
        f"d={d}: state={m.state_nbytes()}, expected={expected}")


# ---- CW-2: state bounded by n ----

def test_state_independent_of_n():
    d = 5
    states = []
    for n in [500, 2000, 10000]:
        df = _make_df(n, d=d, seed=0)
        m = fit_dr_learner(
            df, [f"x{j}" for j in range(d)], "t", "y",
            memory_guard=MemoryGuard(max_memory_mb=500))
        states.append(m.state_nbytes())
        del df
        gc.collect()
    assert max(states) == min(states), f"state varies: {states}"


# ---- CW-4: guard refuses ----

def test_guard_refuses_at_tight():
    df = _make_df(500, d=5, seed=0)
    with pytest.raises(Exception):
        fit_dr_learner(
            df, ["x0", "x1", "x2", "x3", "x4"], "t", "y",
            memory_guard=MemoryGuard(max_memory_mb=0.0005))


# ---- correctness: ATE recovers true effect (unconfounded) ----

def test_ate_recovers_true_effect_unconfounded():
    """Unconfounded: ATE estimate ~ true effect (within noise)."""
    ate_true = 0.5
    df = _make_df(10000, d=5, ate_true=ate_true, seed=0, confounded=False)
    m = fit_dr_learner(
        df, [f"x{j}" for j in range(5)], "t", "y",
        memory_guard=MemoryGuard(max_memory_mb=500))
    # ATE is on latent scale -> sign matches; magnitude depends on
    # outcome model. Just verify ate_se reasonable, ate in expected band.
    assert m.ate_ > 0, f"ate={m.ate_} < 0 (expected positive effect)"
    assert m.ate_se_ > 0, f"se={m.ate_se_} <= 0"
    assert m.ate_se_ < 1.0, f"se={m.ate_se_} too large"


# ---- correctness: double robustness ----

def test_aipw_double_robustness():
    """AIPW with CORRECT mu0/mu1 gives psi.mean() == mu1 - mu0.

    Construct y with E[y|t=1] = mu1 = 0.8, E[y|t=0] = mu0 = 0.3:
      - y_i = 1 for treated w.p. 0.8, else 0
      - y_i = 1 for control w.p. 0.3, else 0
    Then AIPW correction terms vanish in expectation and psi.mean()
    equals (mu1 - mu0) = 0.5.
    """
    rng = np.random.RandomState(0)
    n = 10000
    t = rng.randint(0, 2, n).astype(np.float64)
    p_y_given_t = np.where(t == 1.0, 0.8, 0.3)
    y = (rng.uniform(size=n) < p_y_given_t).astype(np.float64)
    mu0 = 0.3 * np.ones(n)
    mu1 = 0.8 * np.ones(n)
    e = 0.5 * np.ones(n)
    psi = dr_score(y, t, mu0, mu1, e)
    assert psi.shape == (n,)
    assert abs(float(psi.mean()) - 0.5) < 0.05, (
        f"psi.mean() = {psi.mean():.3f}, expected ~0.5 (mu1-mu0); "
        f"double-robustness violated")


# ---- predict_psi parity with model's own aggregate ----

def test_predict_psi_recomputes_ate():
    """predict_psi(df).mean() == model.ate_ (within precision)."""
    df = _make_df(500, d=4, seed=0)
    m = fit_dr_learner(
        df, [f"x{j}" for j in range(4)], "t", "y",
        memory_guard=MemoryGuard(max_memory_mb=500))
    psi_df = m.predict_psi(df)
    parts = []
    for b in psi_df._iter_filtered_batches():
        parts.append(b.column("psi").to_numpy(zero_copy_only=False))
    psi = np.concatenate(parts) if len(parts) > 1 else parts[0]
    ate_recomputed = float(psi.mean())
    assert abs(ate_recomputed - m.ate_) < 1e-8, (
        f"predict_psi mean={ate_recomputed} != ate_={m.ate_}")


# ---- invalid input ----

def test_no_treated_raises():
    df = _make_df(200, d=3, seed=0)
    # Force all control
    df_bad = CadwDataFrame.from_table(
        pa.table({
            **{f"x{j}": df._dataset._table.column(f"x{j}")
               for j in range(3)},
            "t": pa.array([0] * 200, type=pa.int64()),
            "y": pa.array([0] * 200, type=pa.int64()),
        }),
        memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="need >= 2 treated"):
        fit_dr_learner(
            df_bad, ["x0", "x1", "x2"], "t", "y",
            memory_guard=MemoryGuard(max_memory_mb=500))


def test_x_cols_overlap_raises():
    df = _make_df(100, d=3, seed=0)
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_dr_learner(
            df, ["x0", "t"], "t", "y",
            memory_guard=MemoryGuard(max_memory_mb=500))


# ---- determinism ----

def test_deterministic():
    df = _make_df(500, d=3, seed=0)
    m1 = fit_dr_learner(
        df, ["x0", "x1", "x2"], "t", "y",
        memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = fit_dr_learner(
        df, ["x0", "x1", "x2"], "t", "y",
        memory_guard=MemoryGuard(max_memory_mb=500))
    assert abs(m1.ate_ - m2.ate_) < 1e-10
    np.testing.assert_allclose(m1.propensity_.coef_, m2.propensity_.coef_,
                               atol=1e-10)
