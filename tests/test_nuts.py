"""v0.9.229-c1: NUTS primitive verification (independent reference)."""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ops.mcmc import nuts


def _make_df(n, seed=0):
    rng = np.random.RandomState(seed)
    y = 2.0 + rng.normal(size=n) * 0.1
    return CadwDataFrame.from_table(
        pa.table({'y': pa.array(y, type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=500))


def _log_data(batch, theta):
    by = batch.column('y').to_numpy(zero_copy_only=False).astype(np.float64)
    diff = by - theta[0]
    return -0.5 * float(np.dot(diff, diff))


def _log_prior(theta):
    return -0.5 * float(np.dot(theta, theta))


def _grad_log_data(batch, theta):
    by = batch.column('y').to_numpy(zero_copy_only=False).astype(np.float64)
    return np.array([(by - theta[0]).sum()])


def _grad_log_prior(theta):
    return -theta


def _reference_nuts(df, log_data, log_prior, grad_log_data, grad_log_prior,
                    theta_init, *, n_samples, n_burnin, max_tree_depth,
                    lr, target_accept, delta_max, gamma_da, t0_da,
                    kappa_da, seed):
    """Independent port of Hoffman 2014 Algorithm 6 + Nesterov dual avg.

    Same structure as primitive; localizes any porting bug (P63 lesson).
    """
    theta = np.asarray(theta_init, dtype=np.float64).copy()
    d = theta.size
    rng = np.random.default_rng(seed)

    batches = list(df._iter_filtered_batches(batch_size=10000))

    def _joint(th):
        lp = float(log_prior(th))
        for b in batches:
            lp += float(log_data(b, th))
        return lp

    def _grad(th):
        g = np.asarray(grad_log_prior(th), dtype=np.float64).copy()
        for b in batches:
            g += np.asarray(grad_log_data(b, th), dtype=np.float64)
        return g

    def _leapfrog(th, r, eps):
        g0 = _grad(th)
        r_half = r + 0.5 * eps * g0
        th_new = th + eps * r_half
        g1 = _grad(th_new)
        r_new = r_half + 0.5 * eps * g1
        return th_new, r_new

    def _build_tree(th, r, log_u, v, j, eps, joint0):
        if j == 0:
            th_new, r_new = _leapfrog(th, r, v * eps)
            lp_new = _joint(th_new)
            joint_new = lp_new - 0.5 * float(np.dot(r_new, r_new))
            log_size = joint_new - joint0
            n_valid = 1 if log_u <= log_size else 0
            s = 1 if log_u < log_size + delta_max else 0
            return (th_new, r_new, th_new, r_new, th_new,
                    log_size, n_valid, s)
        (tm, rm, tp, rp, t_prime, ls1, n1, s1) = _build_tree(
            th, r, log_u, v, j - 1, eps, joint0)
        if not s1:
            return (tm, rm, tp, rp, t_prime, ls1, n1, s1)
        if v == -1:
            (tm2, rm2, _, _, t2, ls2, n2, s2) = _build_tree(
                tm, rm, log_u, v, j - 1, eps, joint0)
            tm_f, rm_f = tm2, rm2
            tp_f, rp_f = tp, rp
        else:
            (_, _, tp2, rp2, t2, ls2, n2, s2) = _build_tree(
                tp, rp, log_u, v, j - 1, eps, joint0)
            tm_f, rm_f = tm, rm
            tp_f, rp_f = tp2, rp2
        log_total = float(np.logaddexp(ls1, ls2))
        p1 = float(np.exp(ls1 - log_total)) if np.isfinite(log_total) else 0.5
        t_new = t_prime if rng.uniform() < p1 else t2
        delta = tp_f - tm_f
        uturn = (np.dot(delta, rm_f) < 0 or np.dot(delta, rp_f) < 0)
        s_new = s2 if not uturn else 0
        return (tm_f, rm_f, tp_f, rp_f, t_new, log_total,
                n1 + n2, s_new)

    mu = float(np.log(10.0 * lr))
    log_eps_bar = 0.0
    h_bar = 0.0
    eps = float(lr)

    samples = np.empty((n_samples, d), dtype=np.float64)
    sample_idx = 0
    total_iter = int(n_burnin) + int(n_samples)

    for m in range(1, total_iter + 1):
        r0 = rng.standard_normal(d)
        joint0 = _joint(theta) - 0.5 * float(np.dot(r0, r0))
        log_u = float(np.log(rng.uniform())) + joint0

        theta_m = theta.copy()
        r_m = r0.copy()
        theta_p = theta.copy()
        r_p = r0.copy()
        theta_new = theta.copy()
        n_valid = 1
        s = 1
        j = 0
        log_size_total = 0.0

        while s and j < max_tree_depth:
            v_dir = 1 if rng.uniform() < 0.5 else -1
            if v_dir == -1:
                (theta_m, r_m, _, _, theta_prime, ls_sub,
                 n_sub, s) = _build_tree(
                    theta_m, r_m, log_u, v_dir, j, eps, joint0)
            else:
                (_, _, theta_p, r_p, theta_prime, ls_sub,
                 n_sub, s) = _build_tree(
                    theta_p, r_p, log_u, v_dir, j, eps, joint0)
            if s:
                if j == 0:
                    p_new = 1.0 if ls_sub >= log_size_total else 0.5
                else:
                    log_total = float(np.logaddexp(
                        log_size_total, ls_sub))
                    p_new = float(np.exp(ls_sub - log_total))                         if np.isfinite(log_total) else 0.5
                if rng.uniform() < p_new:
                    theta_new = theta_prime
                n_valid += n_sub
                log_size_total = float(np.logaddexp(
                    log_size_total, ls_sub))
            j += 1

        if sample_idx < n_samples and m > n_burnin:
            samples[sample_idx] = theta_new
            sample_idx += 1

        accept_prob = float(n_valid) / float(2 ** min(j, max_tree_depth))
        accept_prob = min(accept_prob, 1.0 - 1e-12)
        eta = 1.0 / (m + t0_da)
        h_bar = (1.0 - eta) * h_bar + eta * (target_accept - accept_prob)
        log_eps = mu - (np.sqrt(m) / gamma_da) * h_bar
        x_eta = m ** (-kappa_da)
        log_eps_bar = x_eta * log_eps + (1.0 - x_eta) * log_eps_bar
        eps = float(np.exp(log_eps_bar))
        theta = theta_new

    return samples


def _warmup():
    df = _make_df(500, seed=999)
    nuts(
        df, _log_data, _log_prior, _grad_log_data, _grad_log_prior,
        np.array([0.0]),
        n_samples=20, n_burnin=50, max_tree_depth=5, lr=0.1,
        seed=0, memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ---- CW-1: primitive matches reference (same seed) ----

def test_primitive_matches_reference():
    """Same seed + params -> identical samples. Localizes any bug."""
    df = _make_df(1000, seed=0)
    theta0 = np.array([0.0])
    kw = dict(n_samples=30, n_burnin=0, max_tree_depth=5,
              lr=0.1, target_accept=0.8, delta_max=1000.0,
              gamma_da=0.05, t0_da=10.0, kappa_da=0.75, seed=42)

    ref = _reference_nuts(
        df, _log_data, _log_prior, _grad_log_data, _grad_log_prior,
        theta0, **kw)
    prim = nuts(
        df, _log_data, _log_prior, _grad_log_data, _grad_log_prior,
        theta0, memory_guard=MemoryGuard(max_memory_mb=500), **kw)

    np.testing.assert_allclose(
        prim, ref, atol=1e-8,
        err_msg="nuts primitive differs from reference; port bug")


# ---- CW-2: Gaussian posterior recovery ----

def test_gaussian_posterior():
    """Target: N(1.998, 0.0316). HMC path uses EXACT gradient."""
    df = _make_df(1000, seed=0)
    target_mean, target_std = 1.998, 0.0316
    samples = nuts(
        df, _log_data, _log_prior, _grad_log_data, _grad_log_prior,
        np.array([0.0]),
        n_samples=1000, n_burnin=1000, max_tree_depth=8, lr=0.1,
        seed=0, memory_guard=MemoryGuard(max_memory_mb=500))
    m, s = float(samples.mean()), float(samples.std())
    assert abs(m - target_mean) < 0.1, (
        f"mean={m:.4f}, target={target_mean}")
    # NUTS should give std close to analytic (no tempering)
    assert 0.6 < s / target_std < 1.5, (
        f"std={s:.4f}, target={target_std}")


# ---- CW-3: determinism ----

def test_deterministic():
    df = _make_df(500, seed=0)
    kw = dict(n_samples=20, n_burnin=20, max_tree_depth=4, lr=0.1,
              seed=42)
    a = nuts(
        df, _log_data, _log_prior, _grad_log_data, _grad_log_prior,
        np.array([0.0]), memory_guard=MemoryGuard(max_memory_mb=500), **kw)
    b = nuts(
        df, _log_data, _log_prior, _grad_log_data, _grad_log_prior,
        np.array([0.0]), memory_guard=MemoryGuard(max_memory_mb=500), **kw)
    np.testing.assert_array_equal(a, b)


# ---- CW-4: guard refuses ----

def test_guard_refuses_at_tight():
    df = _make_df(1000, seed=0)
    with pytest.raises(Exception):
        nuts(
            df, _log_data, _log_prior, _grad_log_data, _grad_log_prior,
            np.array([0.0]),
            n_samples=100, n_burnin=100, max_tree_depth=4, lr=0.1,
            memory_guard=MemoryGuard(max_memory_mb=0.0005))


# ---- CW-5: invalid input ----

@pytest.mark.parametrize("kw,match", [
    (dict(max_tree_depth=0), "max_tree_depth >= 1"),
    (dict(lr=0.0), "lr > 0"),
    (dict(target_accept=0.0), "target_accept"),
    (dict(target_accept=1.0), "target_accept"),
])
def test_invalid_input(kw, match):
    df = _make_df(100, seed=0)
    with pytest.raises(ValueError, match=match):
        nuts(
            df, _log_data, _log_prior, _grad_log_data, _grad_log_prior,
            np.array([0.0]), n_samples=5, n_burnin=0,
            memory_guard=MemoryGuard(max_memory_mb=500), **kw)
