"""v0.9.229-b: SRVR-HMC primitive verification.

Reference implementation in-test (oracle); compare trajectory bit-exact
or mean/std against analytic Gaussian posterior.
"""
from __future__ import annotations
import gc
import statistics

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ops.mcmc import srvr_hmc


def _make_df(n, seed=0):
    rng = np.random.RandomState(seed)
    y = 2.0 + rng.normal(size=n) * 0.1
    return CadwDataFrame.from_table(
        pa.table({'y': pa.array(y, type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=500))


def _gdata(batch, theta):
    by = batch.column('y').to_numpy(zero_copy_only=False).astype(np.float64)
    return np.array([(by - theta[0]).sum()])


def _gprior(theta):
    return -theta


def _malloc_trim():
    import ctypes
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    df = _make_df(500, seed=999)
    srvr_hmc(
        df, _gdata, _gprior, np.array([0.0]),
        n_samples=50, n_burnin=100, L=5, B=50, lr=0.01, gamma=20.0,
        seed=0, memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ============================================================
# Reference implementation (in-test oracle)
# ============================================================
def _reference_srvr_hmc(df, gdata, gprior, theta_init, *,
                        n_steps, L, B, lr, gamma, u, seed):
    rng = np.random.default_rng(seed)
    d = theta_init.size
    x = theta_init.copy()
    v = np.zeros(d)
    x_snapshot = x.copy()
    g_snapshot_data = np.zeros(d)

    e1 = np.exp(-gamma * lr)
    e2 = np.exp(-2.0 * gamma * lr)
    var_x = u / gamma**2 * (2.0 * gamma * lr - 3.0 + 4.0 * e1 - e2)
    var_v = u * (1.0 - e2)
    cov_xv = u / gamma * (1.0 - 2.0 * e1 + e2)
    L00 = np.sqrt(var_x)
    L10 = cov_xv / L00
    L11 = np.sqrt(var_v - L10 * L10)

    coef_v = e1
    coef_g_v = -u / gamma * (1.0 - e1)
    coef_v_x = (1.0 - e1) / gamma
    coef_g_x = -u / gamma**2 * (gamma * lr - 1.0 + e1)

    # Cache batches for cycling (ref uses same iteration order)
    batches = list(df._iter_filtered_batches(batch_size=B))
    batch_idx = 0

    samples = []
    for k in range(n_steps):
        if k % L == 0:
            g_sum_logp = np.zeros(d)
            n_total = 0
            for b in df._iter_filtered_batches(batch_size=B):
                g_sum_logp += gdata(b, x)
                n_total += b.num_rows
            g_snapshot_data = -g_sum_logp  # total, P75 fix
            x_snapshot = x.copy()
            g_data_k = g_snapshot_data
        else:
            b = batches[batch_idx % len(batches)]
            batch_idx += 1
            delta_sum = gdata(b, x) - gdata(b, x_snapshot)
            g_data_k = g_snapshot_data - (float(n_total) / float(b.num_rows)) * delta_sum

        g_prior = -gprior(x)
        g_total = g_data_k + g_prior

        z0 = rng.standard_normal(d)
        z1 = rng.standard_normal(d)
        ex = L00 * z0
        ev = L10 * z0 + L11 * z1

        v_new = coef_v * v + coef_g_v * g_total + ev
        x_new = x + coef_v_x * v + coef_g_x * g_total + ex

        x = x_new
        v = v_new
        samples.append(x.copy())
    return np.array(samples)


# ---- test 1: primitive matches reference exactly ----

def test_primitive_matches_reference():
    """Same seed + params -> identical trajectories (oracle test).

    Localizes any bug immediately: if primitive differs from reference
    on same RNG, it's a porting bug (P63-SGHMC lesson: 8 cycles without
    oracle vs 5 min with).
    """
    df = _make_df(1000, seed=0)
    theta_init = np.array([0.0])
    params = dict(L=5, B=100, lr=0.01, gamma=20.0, u=1.0, seed=42)
    n_steps = 30

    samples_ref = _reference_srvr_hmc(
        df, _gdata, _gprior, theta_init,
        n_steps=n_steps, **params)

    samples_prim = srvr_hmc(
        df, _gdata, _gprior, theta_init,
        n_samples=n_steps, n_burnin=0,
        memory_guard=MemoryGuard(max_memory_mb=500), **params)

    np.testing.assert_allclose(
        samples_prim, samples_ref, atol=1e-10,
        err_msg="srvr_hmc primitive differs from reference; bug in port")


# ---- test 2: Gaussian posterior recovery ----

def test_gaussian_posterior():
    """Target: N(1.998, 0.0316) for prior N(0,1), likelihood N(θ,1)."""
    df = _make_df(1000, seed=0)
    target_mean, target_std = 1.998, 0.0316
    samples = srvr_hmc(
        df, _gdata, _gprior, np.array([0.0]),
        n_samples=1000, n_burnin=5000,
        L=10, B=100, lr=0.01, gamma=20.0, u=1.0, seed=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    m, s = float(samples.mean()), float(samples.std())
    assert abs(m - target_mean) < 0.15, f"mean={m:.4f}, target={target_mean}"
    assert 0.4 < s / target_std < 2.5, f"std={s:.4f}, target~{target_std}"


# ---- test 3: OOC ratio ----

def test_ooc_ratio():
    """Peak shouldn't grow with n (streaming reset + mini-batch only)."""
    peaks = []
    for n in [2000, 20000]:
        df = _make_df(n, seed=0)
        cell_peaks = []
        for _ in range(3):
            _malloc_trim(); gc.collect()
            mg = MemoryGuard(max_memory_mb=500)
            r = measure_peak_traced(
                srvr_hmc, df, _gdata, _gprior, np.array([0.0]),
                n_samples=100, n_burnin=200,
                L=5, B=100, reset_batch_size=100,
                lr=0.01, gamma=20.0, u=1.0, seed=0,
                memory_guard=mg, mg=mg)
            cell_peaks.append(int(r["peak_bytes"]))
        peaks.append(int(statistics.median(cell_peaks)))
        del df
        gc.collect()
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 4.0, f"peak ratio {ratio:.2f} (n=2k -> 20k); peaks={peaks}"


# ---- test 4: guard refuses at tight ----

def test_guard_refuses_at_tight():
    df = _make_df(1000, seed=0)
    with pytest.raises(Exception):
        srvr_hmc(
            df, _gdata, _gprior, np.array([0.0]),
            n_samples=100, n_burnin=100,
            L=5, B=100, lr=0.01, gamma=20.0, u=1.0, seed=0,
            memory_guard=MemoryGuard(max_memory_mb=0.0005))


# ---- test 5: determinism ----

def test_deterministic():
    df = _make_df(1000, seed=0)
    a = srvr_hmc(
        df, _gdata, _gprior, np.array([0.0]),
        n_samples=100, n_burnin=100, L=5, B=100,
        lr=0.01, gamma=20.0, u=1.0, seed=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    b = srvr_hmc(
        df, _gdata, _gprior, np.array([0.0]),
        n_samples=100, n_burnin=100, L=5, B=100,
        lr=0.01, gamma=20.0, u=1.0, seed=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(a, b)


# ---- test 6: invalid input ----

@pytest.mark.parametrize("kwargs,match", [
    (dict(L=0), "L >= 1"),
    (dict(B=0), "B >= 1"),
    (dict(lr=0.0), "lr > 0"),
    (dict(gamma=0.0), "gamma > 0"),
    (dict(u=0.0), "u > 0"),
])
def test_invalid_input(kwargs, match):
    df = _make_df(100, seed=0)
    with pytest.raises(ValueError, match=match):
        srvr_hmc(
            df, _gdata, _gprior, np.array([0.0]),
            n_samples=10, n_burnin=0, memory_guard=MemoryGuard(max_memory_mb=500),
            **kwargs)
