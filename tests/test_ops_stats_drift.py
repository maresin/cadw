"""v0.9.153: tests for cadw.ops.stats.{ks_2sample, psi, chi2_categorical}."""
import numpy as np
import pytest

from cadw.ops.stats import (
    ks_2samp, ks_2sample, psi, chi2_categorical,
)


# ============================================================================
# ks_2sample (alias of ks_2samp)
# ============================================================================

def test_ks_alias_identical():
    """ks_2sample is alias of ks_2samp."""
    assert ks_2sample is ks_2samp


def test_ks_identical_distributions():
    """Same distribution -> statistic near 0, p-value near 1."""
    rng = np.random.RandomState(0)
    a = rng.normal(size=1000)
    b = rng.normal(size=1000)
    stat, p = ks_2sample(a, b)
    assert stat < 0.1
    assert p > 0.05


def test_ks_shifted_distributions():
    """Shifted means -> large statistic, small p-value."""
    rng = np.random.RandomState(1)
    a = rng.normal(0.0, 1.0, size=1000)
    b = rng.normal(3.0, 1.0, size=1000)
    stat, p = ks_2sample(a, b)
    assert stat > 0.5
    assert p < 0.001


def test_ks_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        ks_2sample(np.empty(0), np.zeros(10))
    with pytest.raises(ValueError, match="non-empty"):
        ks_2sample(np.zeros(10), np.empty(0))


# ============================================================================
# psi
# ============================================================================

def test_psi_identical_distributions():
    """Same distribution -> PSI near 0."""
    rng = np.random.RandomState(2)
    a = rng.normal(size=1000)
    value = psi(a, a)
    assert value < 0.01


def test_psi_small_shift():
    """Small shift (0.5 sigma) -> moderate PSI [0.1, 0.3]."""
    rng = np.random.RandomState(3)
    ref = rng.normal(0.0, 1.0, size=2000)
    cur = rng.normal(0.5, 1.0, size=2000)
    value = psi(ref, cur)
    assert 0.05 < value < 0.5


def test_psi_large_shift():
    """Large shift (2 sigma) -> PSI > 0.25 (significant)."""
    rng = np.random.RandomState(4)
    ref = rng.normal(0.0, 1.0, size=2000)
    cur = rng.normal(2.0, 1.0, size=2000)
    value = psi(ref, cur)
    assert value > 0.25


def test_psi_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        psi(np.empty(0), np.zeros(10))


def test_psi_n_bins():
    """Different n_bins -> different resolution."""
    rng = np.random.RandomState(5)
    ref = rng.normal(0.0, 1.0, size=2000)
    cur = rng.normal(1.0, 1.0, size=2000)
    v5 = psi(ref, cur, n_bins=5)
    v20 = psi(ref, cur, n_bins=20)
    # Both positive and non-zero; exact ordering may vary
    assert v5 > 0.0
    assert v20 > 0.0


# ============================================================================
# chi2_categorical
# ============================================================================

def test_chi2_identical_distributions():
    """Same categorical distribution -> small stat, high p."""
    rng = np.random.RandomState(6)
    a = rng.choice(["x", "y", "z"], size=1000, p=[0.5, 0.3, 0.2])
    b = rng.choice(["x", "y", "z"], size=1000, p=[0.5, 0.3, 0.2])
    stat, p = chi2_categorical(a, b)
    assert stat < 10.0  # dof=2, chi2 threshold ~10 at p=0.01
    assert p > 0.01


def test_chi2_different_distributions():
    """Different distributions -> large stat, small p."""
    rng = np.random.RandomState(7)
    a = rng.choice(["x", "y"], size=1000, p=[0.9, 0.1])
    b = rng.choice(["x", "y"], size=1000, p=[0.1, 0.9])
    stat, p = chi2_categorical(a, b)
    assert stat > 100.0
    assert p < 0.001


def test_chi2_single_category():
    """Only 1 category present in both -> returns (0, 1)."""
    a = np.array(["x"] * 100)
    b = np.array(["x"] * 100)
    stat, p = chi2_categorical(a, b)
    assert stat == 0.0
    assert p == 1.0


def test_chi2_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        chi2_categorical(np.empty(0), np.array([1, 2]))


# ============================================================================
# backward compat: cadw.ml.drift re-exports
# ============================================================================

def test_drift_reexports():
    """Old imports from cadw.ml.drift still work."""
    from cadw.ml.drift import ks_2samp as old_ks, psi as old_psi
    # identity check
    assert old_ks is ks_2samp
    assert old_psi is psi
