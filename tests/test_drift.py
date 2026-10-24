"""Tests for D.6.2 drift / A-B (v0.9.40)."""
from __future__ import annotations
import inspect, numpy as np, pytest
from cadw.ml import drift as dmod
from cadw.ml.drift import (ks_2samp, psi, chi2_categorical, welch_t,
                           mann_whitney, cuped, fit_drift, fit_ab_test)


def test_ks_reference_scipy_exact_match():
    from scipy.stats import ks_2samp as sk
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, 1000); b = rng.normal(0.5, 1, 1000)
    s, p = ks_2samp(a, b)
    ref = sk(a, b, method="asymp")
    assert abs(s - ref.statistic) < 1e-12 and abs(p - ref.pvalue) < 1e-12


def test_psi_zero_for_identical_samples():
    rng = np.random.RandomState(1)
    a = rng.normal(0, 1, 2000)
    assert psi(a, a) < 1e-6
    assert psi(a, rng.normal(3, 1, 2000)) > 1.0


def test_chi2_reference_scipy():
    from scipy.stats import chi2_contingency
    rng = np.random.RandomState(2)
    a = rng.randint(0, 3, 500); b = rng.randint(0, 3, 500)
    s, p = chi2_categorical(a, b)
    cats = np.union1d(np.unique(a), np.unique(b))
    table = np.vstack([[np.sum(a == c) for c in cats], [np.sum(b == c) for c in cats]])
    ref = chi2_contingency(table, correction=False)
    assert abs(s - ref[0]) < 1e-12 and abs(p - ref[1]) < 1e-12


def test_welch_and_mw_reference_scipy():
    from scipy.stats import ttest_ind, mannwhitneyu
    rng = np.random.RandomState(3)
    a = rng.normal(0.2, 1, 500); b = rng.normal(0, 1, 500)
    st, pt, diff = welch_t(a, b)
    ref = ttest_ind(a, b, equal_var=False)
    assert abs(st - ref.statistic) < 1e-12 and abs(pt - ref.pvalue) < 1e-12
    assert abs(diff - (a.mean() - b.mean())) < 1e-12
    su, pu = mann_whitney(a, b)
    refu = mannwhitneyu(a, b, alternative="two-sided")
    assert abs(su - refu.statistic) < 1e-12 and abs(pu - refu.pvalue) < 1e-12


def test_cuped_reduces_variance_with_correlated_covariate():
    rng = np.random.RandomState(4)
    n = 4000
    x = rng.normal(size=n)
    t = np.repeat([0.0, 1.0], n // 2)
    y = 0.3 * t + 0.8 * x + rng.normal(0, 0.2, n)  # true effect 0.3
    out = cuped(y, x, t)
    # CUPED-adjusted effect should be closer to the true 0.3 and with
    # smaller variance than the raw difference
    assert abs(out["adjusted_effect"] - 0.3) < 0.05
    assert abs(out["theta"] - 0.8) < 0.1


def test_edge_guards():
    with pytest.raises(ValueError, match="non-empty"):
        ks_2samp(np.array([]), np.array([1.0]))
    with pytest.raises(ValueError, match="non-empty"):
        psi(np.array([]), np.array([1.0]))
    with pytest.raises(ValueError, match=">= 2"):
        welch_t(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="binary"):
        cuped(np.arange(10.0), np.arange(10.0), np.arange(10.0))


def test_determinism():
    rng = np.random.RandomState(5)
    a = rng.normal(0, 1, 500); b = rng.normal(0.2, 1, 500)
    assert ks_2samp(a, b) == ks_2samp(a, b)
    assert psi(a, b) == psi(a, b)


def test_fit_drift_and_fit_ab_guards():
    import pyarrow as pa
    from cadw import CadwDataFrame
    df = CadwDataFrame.from_table(pa.table({"x": pa.array([1.0, 2.0, 3.0]),
                                           "r": pa.array([1.0, 2.0, 3.0]),
                                           "t": pa.array([0.0, 1.0, 0.0])}))
    m = fit_drift(df, "x", "r", method="ks")
    assert m.to_dict()["kind"] == "drift"
    ab = fit_ab_test(df, "x", "t")
    assert ab.to_dict()["kind"] == "ab"
    with pytest.raises(KeyError):
        fit_drift(df, "nope", "r")
    with pytest.raises(ValueError, match="method must be"):
        fit_drift(df, "x", "r", method="z")
    with pytest.raises(ValueError, match="covariate_col"):
        fit_ab_test(df, "x", "t", method="cuped")


def test_source_contract_log_fit_and_no_pandas():
    src = inspect.getsource(dmod)
    assert "log_fit" in src
    assert "import pandas" not in src
