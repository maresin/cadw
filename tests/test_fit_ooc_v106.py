"""v0.9.106 (DEC-022/023): OOC production-path verification.

v0.9.137 (DEC-057 F1): metric py_bytes -> rss_bytes. py_bytes uses
tracemalloc (Python objects only); numpy arrays via np.empty use
malloc, invisible to tracemalloc — an unbatched numpy fit passes
the old test. rss_bytes (ru_maxrss delta) sees malloc peaks.
Audit v0.9.136 found this; 3 models (linear, logistic, gaussian_nb)
verified genuinely batched under rss_bytes.

All 6 fit_* derive batch_size from state-aware MemoryGuard; peak is flat
in n once n >= batch_cap. No monkey-patch (unlike v0.9.105 iter-4).
"""
from __future__ import annotations

import gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import measure_peak, measure_peak_traced


BUDGET_MB = 100
D = 50
COLS = [f"x{j}" for j in range(D)]


def _make_df(n, y_kind, *, C=2):
    rng = np.random.RandomState(42)
    cols = {k: rng.randn(n) for k in COLS}
    if y_kind == "float":
        cols["y"] = rng.randn(n)
    elif y_kind == "binary":
        cols["y"] = rng.randint(0, 2, n).astype(np.int64)
    elif y_kind == "counts_nonneg":
        for k in COLS:
            cols[k] = np.abs(cols[k])
        cols["y"] = rng.randint(0, C, n).astype(np.int64)
    else:
        raise ValueError(y_kind)
    return CadwDataFrame.from_table(
        pa.table(cols), memory_guard=MemoryGuard(max_memory_mb=BUDGET_MB))


def _run(fit_fn, y_kind, kwargs, n):
    df = _make_df(n, y_kind)
    mg = df._memory_guard
    # v0.9.137 (DEC-057): measure_peak_traced (sampler-based VmRSS)
    # instead of measure_peak(rss_bytes). ru_maxrss delta is dominated
    # by cross-test baseline drift in full-file runs (5/9 false failures);
    # sampler-based peak is per-fit and independent.
    return measure_peak_traced(
        fit_fn, df, COLS, "y", memory_guard=mg, mg=mg, **kwargs)["peak_bytes"]


def _assert_ooc(fit_fn, y_kind, kwargs, name):
    """Plateau check: n=10^5 and n=10^6 both >= batch_cap."""
    p1 = _run(fit_fn, y_kind, kwargs, 10**5)
    gc.collect()
    p2 = _run(fit_fn, y_kind, kwargs, 10**6)
    ratio = p2 / max(p1, 1)
    budget_bytes = BUDGET_MB * 1024 * 1024
    max_peak = max(p1, p2)
    assert ratio < 2.0, (
        f"{name}: peak grew {ratio:.2f}x between n=1e5 and n=1e6; "
        f"not OOC (peaks: {p1/1e6:.1f} MB, {p2/1e6:.1f} MB)")
    assert max_peak < budget_bytes, (
        f"{name}: peak {max_peak/1e6:.1f} MB > budget {BUDGET_MB} MB")


def test_ooc_linear():
    from cadw.ml.linear import fit_linear
    _assert_ooc(fit_linear, "float",
                dict(alpha=0.0, solver="closed"), "fit_linear")


def test_ooc_lasso():
    from cadw.ml.linear import fit_lasso
    _assert_ooc(fit_lasso, "float",
                dict(alpha=1.0, max_iter=20), "fit_lasso")


def test_ooc_elasticnet():
    from cadw.ml.linear import fit_elasticnet
    _assert_ooc(fit_elasticnet, "float",
                dict(alpha=1.0, l1_ratio=0.5, max_iter=20),
                "fit_elasticnet")


def test_ooc_logistic():
    from cadw.ml.logistic import fit_logistic
    _assert_ooc(fit_logistic, "binary",
                dict(alpha=1.0, max_iter=5), "fit_logistic")


def test_ooc_gaussian_nb():
    from cadw.ml.naive_bayes import fit_gaussian_nb
    _assert_ooc(fit_gaussian_nb, "binary", {}, "fit_gaussian_nb")


def test_ooc_multinomial_nb():
    from cadw.ml.naive_bayes import fit_multinomial_nb
    _assert_ooc(fit_multinomial_nb, "counts_nonneg",
                dict(alpha=1.0), "fit_multinomial_nb")


def test_ooc_bernoulli_nb():
    from cadw.ml.naive_bayes import fit_bernoulli_nb
    _assert_ooc(fit_bernoulli_nb, "counts_nonneg",
                dict(alpha=1.0), "fit_bernoulli_nb")


def test_batch_size_override():
    """Explicit batch_size overrides auto-derive; peak still < budget."""
    from cadw.ml.linear import fit_linear
    n = 10**5
    df = _make_df(n, "float")
    mg = df._memory_guard
    p = measure_peak_traced(
        fit_linear, df, COLS, "y",
        alpha=0.0, solver="closed", batch_size=2000,
        memory_guard=mg, mg=mg)["peak_bytes"]
    assert p < BUDGET_MB * 1024 * 1024


def test_state_exceeds_budget_raises():
    """At d huge, state alone > usable -> MemoryLimitExceeded (honest)."""
    from cadw.core.memory_guard import MemoryLimitExceeded
    mg = MemoryGuard(max_memory_mb=1)
    # d=10^4, linear state = 2*d^2*8 = 1.6 GB >> 0.7 MB usable
    with pytest.raises(MemoryLimitExceeded, match="state_bytes"):
        mg.get_batch_rows(row_size_bytes=80000, expansion_factor=3.0,
                          state_bytes=1_600_000_000)
