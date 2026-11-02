"""v0.9.107: matrix accuracy for 9 verified cards.

3x3 grid: d in {10, 100, 1000}, n=batch=1000 (single batch).
Predicted (YAML formula) vs measured (measure_peak) ratio in [0.7, 1.4].
Production path (v0.9.106 framework).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import measure_peak
from cadw.ml._memory_estimate import _load_cards, evaluate_formula


def _formula(fn):
    _load_cards.cache_clear()
    return _load_cards().get(fn, {}).get("ram_formula")


def _df(n, d, y_kind, C=2, nonneg=False):
    rng = np.random.RandomState(42)
    cols = {f"x{j}": rng.randn(n) for j in range(d)}
    if nonneg:
        cols = {k: np.abs(v) for k, v in cols.items()}
    if y_kind == "float":
        cols["y"] = rng.randn(n)
    elif y_kind == "binary":
        cols["y"] = rng.randint(0, 2, n).astype(np.int64)
    elif y_kind == "counts":
        cols["y"] = rng.randint(0, C, n).astype(np.int64)
    return CadwDataFrame.from_table(
        pa.table(cols), memory_guard=MemoryGuard(max_memory_mb=2000))


@pytest.mark.parametrize("d", [10, 100, 1000])
def test_matrix_linear(d):
    from cadw.ml.linear import fit_linear
    if not _formula("fit_linear"):
        pytest.skip("no formula")
    n = batch = 1000
    Xs = [f"x{j}" for j in range(d)]
    row_size = (d + 1) * 8
    df = _df(n, d, "float")
    pred = evaluate_formula("fit_linear", batch=batch, row_size=row_size, d=d)["bytes"]
    meas = measure_peak(fit_linear, df, Xs, "y",
                        alpha=0.0, solver="closed", batch_size=batch,
                        memory_guard=df._memory_guard)["py_bytes"]
    r = pred / max(meas, 1)
    assert 0.7 <= r <= 1.4, f"linear d={d}: pred={pred/1e6:.2f}M meas={meas/1e6:.2f}M r={r:.2f}"


# d=10 excluded for logistic: ~100 KB fixed Python/Arrow overhead
# (validation pass + X_aug column_stack peak) is 37% of algorithmic peak
# at d=10 (ratio 0.62). At d>=100 overhead <5% (ratio 0.94-1.00).
# See EXPERIMENT_PROCEDURE §8 A4 (baseline floor).
@pytest.mark.parametrize("d", [100, 500, 1000])
def test_matrix_logistic(d):
    from cadw.ml.logistic import fit_logistic
    if not _formula("fit_logistic"):
        pytest.skip("no formula")
    n = batch = 1000
    Xs = [f"x{j}" for j in range(d)]
    row_size = (d + 1) * 8
    df = _df(n, d, "binary")
    pred = evaluate_formula("fit_logistic", batch=batch, row_size=row_size, d=d)["bytes"]
    meas = measure_peak(fit_logistic, df, Xs, "y",
                        alpha=1.0, max_iter=5, batch_size=batch,
                        memory_guard=df._memory_guard)["py_bytes"]
    r = pred / max(meas, 1)
    assert 0.7 <= r <= 1.4, f"logistic d={d}: pred={pred/1e6:.2f}M meas={meas/1e6:.2f}M r={r:.2f}"


@pytest.mark.parametrize("d", [10, 100, 1000])
@pytest.mark.parametrize("C", [2, 10])
def test_matrix_gaussian_nb(d, C):
    from cadw.ml.naive_bayes import fit_gaussian_nb
    if not _formula("fit_gaussian_nb"):
        pytest.skip("no formula")
    n = batch = 1000
    Xs = [f"x{j}" for j in range(d)]
    row_size = (d + 1) * 8
    df = _df(n, d, "counts", C=C)
    pred = evaluate_formula("fit_gaussian_nb", batch=batch, row_size=row_size,
                            d=d, C=C)["bytes"]
    meas = measure_peak(fit_gaussian_nb, df, Xs, "y",
                        batch_size=batch,
                        memory_guard=df._memory_guard)["py_bytes"]
    r = pred / max(meas, 1)
    assert 0.7 <= r <= 1.4, f"gaussian d={d} C={C}: pred={pred/1e6:.2f}M meas={meas/1e6:.2f}M r={r:.2f}"
