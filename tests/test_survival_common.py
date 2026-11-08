"""v0.9.155: tests for cadw.ml._survival_common.logrank (P1 L10)."""
import numpy as np
import pytest

from cadw.ml._survival_common import logrank


def test_logrank_identical_groups():
    """Same survival in both groups -> not significantly different.

    Under H0 (identical survival), chi2 ~ chi2(1), p ~ Uniform(0,1).
    A single seed can give small p by chance; use large n + p > 0.001
    (99.9% non-rejection, flake rate 0.1%).
    """
    rng = np.random.RandomState(0)
    n = 5000
    t = rng.exponential(scale=10.0, size=n)
    e = (rng.random(n) < 0.7).astype(np.float64)
    groups = rng.randint(0, 2, size=n)
    chi2, p, dof = logrank(t, e, groups)
    assert dof == 1
    assert np.isfinite(chi2)
    assert chi2 >= 0.0
    assert p > 0.001, f"chi2={chi2:.3f}, p={p:.4f} (unexpected rejection)"


def test_logrank_different_groups():
    """Very different survival -> large chi2, small p."""
    rng = np.random.RandomState(1)
    n_each = 300
    t_ctrl = rng.exponential(scale=20.0, size=n_each)
    t_treat = rng.exponential(scale=2.0, size=n_each)  # much shorter survival
    t = np.concatenate([t_ctrl, t_treat])
    e = np.ones(2 * n_each, dtype=np.float64)  # all events
    groups = np.concatenate([
        np.zeros(n_each, dtype=np.int64),
        np.ones(n_each, dtype=np.int64),
    ])
    chi2, p, dof = logrank(t, e, groups)
    assert chi2 > 100.0
    assert p < 0.001
    assert dof == 1


def test_logrank_three_groups():
    """Three groups, same survival -> not significantly different.

    Same flakiness caveat as test_logrank_identical_groups: chi2 ~
    chi2(2) under H0; use large n + p > 0.001.
    """
    rng = np.random.RandomState(2)
    n = 3000
    t = rng.exponential(scale=5.0, size=n)
    e = (rng.random(n) < 0.7).astype(np.float64)
    groups = rng.randint(0, 3, size=n)
    chi2, p, dof = logrank(t, e, groups)
    assert dof == 2
    assert np.isfinite(chi2)
    assert chi2 >= 0.0
    assert p > 0.001, f"chi2={chi2:.3f}, p={p:.4f} (unexpected rejection)"


def test_logrank_three_groups_one_differs():
    """Group 2 has much shorter survival -> reject."""
    rng = np.random.RandomState(3)
    n_each = 200
    t1 = rng.exponential(scale=10.0, size=n_each)
    t2 = rng.exponential(scale=10.0, size=n_each)
    t3 = rng.exponential(scale=1.0, size=n_each)  # much shorter
    t = np.concatenate([t1, t2, t3])
    e = np.ones(3 * n_each, dtype=np.float64)
    groups = np.repeat([0, 1, 2], n_each).astype(np.int64)
    chi2, p, dof = logrank(t, e, groups)
    assert dof == 2
    assert chi2 > 50.0
    assert p < 0.001


def test_logrank_with_censoring():
    """All-censored (no events) -> zero information, chi2=0."""
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.zeros(4, dtype=np.float64)  # all censored
    groups = np.array([0, 0, 1, 1])
    chi2, p, dof = logrank(t, e, groups)
    # no events -> nothing accumulates; chi2 = 0
    assert chi2 == 0.0
    assert dof == 1


def test_logrank_matches_known_chi2():
    """Small deterministic example (R survival::survdiff reference)."""
    # 2 groups, 2 events per group, identical times
    t = np.array([1.0, 1.0, 2.0, 2.0])
    e = np.array([1.0, 1.0, 1.0, 1.0])
    groups = np.array([0, 1, 0, 1])
    chi2, p, dof = logrank(t, e, groups)
    # Perfectly balanced at each time -> O == E -> chi2 = 0
    assert chi2 < 1e-10


def test_logrank_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        logrank(np.empty(0), np.empty(0), np.empty(0))


def test_logrank_single_group_raises():
    t = np.array([1.0, 2.0, 3.0])
    e = np.ones(3)
    groups = np.zeros(3, dtype=np.int64)
    with pytest.raises(ValueError, match=">= 2 groups"):
        logrank(t, e, groups)


def test_logrank_negative_time_raises():
    t = np.array([-1.0, 2.0, 3.0, 4.0])
    e = np.ones(4)
    groups = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="non-negative"):
        logrank(t, e, groups)


def test_logrank_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        logrank(np.zeros(3), np.zeros(4), np.zeros(3))
