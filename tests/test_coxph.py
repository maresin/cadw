"""Tests for D.3.1 CoxPH (v0.9.36)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import coxph as cmod
from cadw.ml.coxph import fit_coxph, CoxPHModel, _breslow


def _surv(n=400, d=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    beta = np.array([0.7, -0.4, 0.2])[:d]
    T = -np.log(rng.uniform(size=n)) / np.exp(X @ beta) * 2.0
    C = rng.exponential(scale=3.0, size=n)
    E = (T <= C).astype(np.int64)
    Tobs = np.minimum(T, C)
    d_ = {f"x{j}": pa.array(X[:, j]) for j in range(d)}
    d_["T"] = pa.array(Tobs); d_["E"] = pa.array(E, type=pa.int64())
    return CadwDataFrame.from_table(pa.table(d_)), X, Tobs, E


def test_smoke_and_state():
    df, X, T, E = _surv()
    m = fit_coxph(df, ["x0", "x1", "x2"], "T", "E")
    assert isinstance(m, CoxPHModel)
    assert m.coef_.shape == (3,)
    assert m.covariance_.shape == (3, 3)
    assert m.n_events_ == int(E.sum())
    assert m.converged_ is True
    assert m.state_nbytes() < 100_000


def test_reference_vs_lifelines():
    from lifelines import CoxPHFitter
    import pandas as pd
    df, X, T, E = _surv(seed=1)
    m = fit_coxph(df, ["x0", "x1", "x2"], "T", "E", max_iter=100)
    pdf = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    pdf["T"] = T; pdf["E"] = E
    ref = CoxPHFitter().fit(pdf, "T", "E")
    assert np.allclose(m.coef_, ref.params_.values, atol=1e-4), (
        f"cadw {m.coef_} vs lifelines {ref.params_.values}")
    assert np.isclose(m.concordance_index(df, "T", "E"),
                      ref.concordance_index_, atol=1e-6)


def test_recovers_sign_and_magnitude():
    df, X, T, E = _surv(n=2000, seed=2)
    m = fit_coxph(df, ["x0", "x1", "x2"], "T", "E", max_iter=100)
    assert m.coef_[0] > 0.3      # true 0.7
    assert m.coef_[1] < -0.1     # true -0.4


def test_breslow_gradient_matches_finite_difference():
    rng = np.random.RandomState(3)
    n, d = 60, 2
    X = rng.normal(size=(n, d))
    T = rng.uniform(0.5, 5.0, size=n)
    E = rng.randint(0, 2, size=n).astype(float)
    E[0] = 1.0
    b = np.array([0.3, -0.2])
    grad, _, _ = _breslow(X, T, E, b)
    eps = 1e-6
    fd = np.zeros(d)
    for j in range(d):
        bp = b.copy(); bp[j] += eps
        bm = b.copy(); bm[j] -= eps
        fd[j] = (_breslow(X, T, E, bp)[2] - _breslow(X, T, E, bm)[2]) / (2 * eps)
    assert np.allclose(grad, fd, atol=1e-5)


def test_determinism():
    df, _, _, _ = _surv(seed=4)
    a = fit_coxph(df, ["x0", "x1", "x2"], "T", "E")
    b = fit_coxph(df, ["x0", "x1", "x2"], "T", "E")
    assert np.array_equal(a.coef_, b.coef_)


def test_multi_batch_invariance():
    import tempfile, pyarrow.parquet as pq
    df, X, T, E = _surv(seed=5, n=200)
    parts = list(df._iter_filtered_batches())
    path = tempfile.mktemp(suffix=".parquet")
    pq.write_table(pa.Table.from_batches(parts), path, row_group_size=13)
    d1 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1)
    d2 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100000)
    a = fit_coxph(d1, ["x0", "x1", "x2"], "T", "E")
    b = fit_coxph(d2, ["x0", "x1", "x2"], "T", "E")
    assert np.allclose(a.coef_, b.coef_, atol=1e-12)


def test_ram_state_is_model_sized():
    small, _, _, _ = _surv(n=100, seed=6)
    big, _, _, _ = _surv(n=5000, seed=6)
    a = fit_coxph(small, ["x0", "x1", "x2"], "T", "E")
    b = fit_coxph(big, ["x0", "x1", "x2"], "T", "E")
    assert a.state_nbytes() < 100_000
    assert b.state_nbytes() < 1_000_000


def test_guard_large_state():
    df, _, _, _ = _surv(n=20000, seed=7, d=3)
    mg = MemoryGuard(max_memory_mb=1)
    with pytest.raises(MemoryLimitExceeded):
        fit_coxph(df, ["x0", "x1", "x2"], "T", "E", memory_guard=mg)


def test_errors():
    df, X, T, E = _surv(n=50, seed=8)
    with pytest.raises(KeyError):
        fit_coxph(df, ["nope"], "T", "E")
    with pytest.raises(ValueError):
        fit_coxph(df, [], "T", "E")
    allc = CadwDataFrame.from_table(pa.table({
        "x0": pa.array(X[:, 0]), "T": pa.array(T),
        "E": pa.array(np.zeros(50, dtype=np.int64), type=pa.int64())}))
    with pytest.raises(ValueError):
        fit_coxph(allc, ["x0"], "T", "E")
    neg = CadwDataFrame.from_table(pa.table({
        "x0": pa.array(X[:, 0]), "T": pa.array(-T),
        "E": pa.array(E, type=pa.int64())}))
    with pytest.raises(ValueError):
        fit_coxph(neg, ["x0"], "T", "E")


def test_source_contract():
    src = inspect.getsource(cmod)
    assert "Cox" in src and "Breslow" in src
    assert "Andersen" in src
    assert "import pandas" not in src
    fit_src = inspect.getsource(cmod.fit_coxph)
    assert "check_state" in fit_src and "log_fit" in fit_src
    for line in src.splitlines():
        if "check_batch(" in line:
            assert "extra_factor=" in line

def test_breslow_streaming_matches_in_ram(tmp_path):
    """P5a-fix-coxph: streaming reduce bit-identical to in-RAM (tol 1e-10)."""
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.io.batch_reader import ArrowIPCBatchReader
    from cadw.ml.coxph import _breslow, _breslow_streaming

    rng = np.random.RandomState(42)
    n, d = 500, 3
    X = rng.normal(size=(n, d))
    T = rng.exponential(scale=2.0, size=n) + 0.01
    E = rng.randint(0, 2, size=n).astype(float)
    # Sort by T desc — streaming expects pre-sorted input.
    order = np.argsort(-T, kind="stable")
    Xo, To, Eo = X[order], T[order], E[order]
    b = np.array([0.3, -0.2, 0.1])

    ref_grad, ref_info, ref_ll = _breslow(X, T, E, b)

    # Write sorted file via Arrow IPC (mimics external_sort output).
    import pyarrow as pa
    path = str(tmp_path / "sorted.arrow")
    schema = pa.schema(
        [pa.field(f"x{j}", pa.float64()) for j in range(d)] +
        [pa.field("T", pa.float64()), pa.field("E", pa.float64())]
    )
    _w = ArrowIPCBatchWriter(path, schema)
    try:
        step = 100
        for start in range(0, n, step):
            end = min(start + step, n)
            cols_dict = {f"x{j}": pa.array(Xo[start:end, j])
                         for j in range(d)}
            cols_dict["T"] = pa.array(To[start:end])
            cols_dict["E"] = pa.array(Eo[start:end])
            _w.write_batch(pa.record_batch(
                [cols_dict[f.name] for f in schema], schema=schema))
    finally:
        _w.close()

    mg = MemoryGuard(max_memory_mb=100)
    reader = ArrowIPCBatchReader(path, None)
    try:
        grad, info, ll, n_ev = _breslow_streaming(
            reader, [f"x{j}" for j in range(d)], "T", "E", b, mg)
    finally:
        reader.close()

    assert np.allclose(grad, ref_grad, atol=1e-10), (
        f"grad streaming {grad} vs in-ram {ref_grad}")
    assert np.allclose(info, ref_info, atol=1e-10)
    assert abs(ll - ref_ll) < 1e-10
    assert n_ev == int((E > 0).sum())


def test_baseline_hazard_streaming_matches_in_ram(tmp_path):
    """P5a-fix-coxph: streaming baseline hazard bit-identical to in-RAM."""
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.io.batch_reader import ArrowIPCBatchReader
    from cadw.ml.coxph import _baseline_hazard, _baseline_hazard_streaming

    rng = np.random.RandomState(7)
    n, d = 400, 2
    X = rng.normal(size=(n, d))
    T = rng.exponential(scale=1.5, size=n) + 0.01
    E = rng.randint(0, 2, size=n).astype(float)
    order = np.argsort(-T, kind="stable")
    Xo, To, Eo = X[order], T[order], E[order]
    b = np.array([0.2, -0.1])

    ref_t, ref_h = _baseline_hazard(X, T, E, b)

    import pyarrow as pa
    path = str(tmp_path / "sorted.arrow")
    schema = pa.schema(
        [pa.field(f"x{j}", pa.float64()) for j in range(d)] +
        [pa.field("T", pa.float64()), pa.field("E", pa.float64())]
    )
    _w = ArrowIPCBatchWriter(path, schema)
    try:
        for start in range(0, n, 50):
            end = min(start + 50, n)
            cols_dict = {f"x{j}": pa.array(Xo[start:end, j])
                         for j in range(d)}
            cols_dict["T"] = pa.array(To[start:end])
            cols_dict["E"] = pa.array(Eo[start:end])
            _w.write_batch(pa.record_batch(
                [cols_dict[f.name] for f in schema], schema=schema))
    finally:
        _w.close()

    mg = MemoryGuard(max_memory_mb=100)
    reader = ArrowIPCBatchReader(path, None)
    try:
        t_s, h_s = _baseline_hazard_streaming(
            reader, [f"x{j}" for j in range(d)], "T", "E", b, mg)
    finally:
        reader.close()

    assert np.allclose(t_s, ref_t, atol=1e-10), (
        f"times differ: streaming {t_s[:3]} vs ram {ref_t[:3]}")
    assert np.allclose(h_s, ref_h, atol=1e-10)


def test_fit_coxph_OOC_small_guard():
    """P5a-fix-coxph: fit completes at n=20000 with 2 MB guard, small
    batches (batch check < budget; streaming works OOC)."""
    import tempfile, pyarrow.parquet as pq
    df, X, T, E = _surv(n=20000, seed=7, d=3)
    parts = list(df._iter_filtered_batches())
    path = tempfile.mktemp(suffix=".parquet")
    pq.write_table(pa.Table.from_batches(parts), path, row_group_size=500)
    d2 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=500)
    mg = MemoryGuard(max_memory_mb=2)
    m = fit_coxph(d2, ["x0", "x1", "x2"], "T", "E", memory_guard=mg)
    assert m.fitted_
    assert m.n_events_ > 0
    assert m.converged_

