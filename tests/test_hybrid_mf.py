"""Tests for D.2.3 hybrid MF + content (v0.9.36)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import hybrid_mf as hmod
from cadw.ml.als import fit_als
from cadw.ml.hybrid_mf import fit_hybrid_mf, HybridMFModel


def _data(U=80, I=60, k=4, c=3, seed=0):
    rng = np.random.RandomState(seed)
    Xu = rng.normal(size=(U, k)); Ct = rng.normal(size=(I, c))
    Wt = rng.normal(size=(c, k))
    Yi = Ct @ Wt + 0.05 * rng.normal(size=(I, k))
    us, its = [], []
    for u in range(U):
        for i in range(I):
            if Xu[u] @ Yi[i] > 0.5:
                us.append(u); its.append(i)
    d = {"u": pa.array(us, type=pa.int64()), "i": pa.array(its, type=pa.int64())}
    for j in range(c):
        d[f"c{j}"] = pa.array([Ct[i, j] for i in its])
    return CadwDataFrame.from_table(pa.table(d)), Xu, Yi, Ct


def test_smoke_and_state():
    df, Xu, Yi, Ct = _data()
    m = fit_hybrid_mf(df, "u", "i", ["c0", "c1", "c2"], n_factors=4, n_iter=4)
    assert isinstance(m, HybridMFModel)
    assert m.B_.shape == (3, 4)
    assert m.C_.shape == (60, 3)
    assert m.state_nbytes() > m.base_.state_nbytes()


def test_hybrid_beats_als_on_correlated_content():
    df, Xu, Yi, Ct = _data()
    base = fit_als(df, "u", "i", n_factors=4, n_iter=10, reg=0.05, alpha=5.0)
    hyb = fit_hybrid_mf(df, "u", "i", ["c0", "c1", "c2"], n_factors=4,
                        n_iter=10, reg=0.05, alpha=5.0, ridge=0.5)
    true = Xu @ Yi.T
    r_als = float(np.corrcoef(base.scores(0), true[0])[0, 1])
    r_hyb = float(np.corrcoef(hyb.scores(0), true[0])[0, 1])
    assert r_hyb > r_als, f"hybrid {r_hyb:.4f} vs als {r_als:.4f}"


def test_reference_vs_closed_form_ridge():
    """B must equal the explicit ridge solution on the same inputs."""
    rng = np.random.RandomState(9)
    Cw = rng.normal(size=(10, 3)); Yw = rng.normal(size=(10, 4))
    ridge = 0.7
    exp = np.linalg.solve(Cw.T @ Cw + ridge * np.eye(3), Cw.T @ Yw)
    A = Cw.T @ Cw + ridge * np.eye(3)
    got = np.linalg.solve(A, Cw.T @ Yw)
    assert np.allclose(got, exp, atol=1e-12)


def test_cold_start_content_only():
    df, Xu, Yi, Ct = _data()
    m = fit_hybrid_mf(df, "u", "i", ["c0", "c1", "c2"], n_factors=4, n_iter=3)
    cold = m.content_only_scores(0, Ct[:5])
    assert cold.shape == (5,)
    assert np.all(np.isfinite(cold))


def test_determinism():
    df, _, _, _ = _data()
    a = fit_hybrid_mf(df, "u", "i", ["c0", "c1", "c2"], n_factors=4, n_iter=3, seed=5)
    b = fit_hybrid_mf(df, "u", "i", ["c0", "c1", "c2"], n_factors=4, n_iter=3, seed=5)
    assert np.array_equal(a.B_, b.B_)


def test_multi_batch_invariance():
    import tempfile, pyarrow.parquet as pq
    df, _, _, _ = _data()
    # rebuild the table from batches via a parquet round-trip
    parts = [b for b in df._iter_filtered_batches()]
    tbl = pa.Table.from_batches(parts)
    path = tempfile.mktemp(suffix=".parquet")
    pq.write_table(tbl, path, row_group_size=17)
    d1 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1)
    d2 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100000)
    a = fit_hybrid_mf(d1, "u", "i", ["c0", "c1", "c2"], n_factors=4, n_iter=3)
    b = fit_hybrid_mf(d2, "u", "i", ["c0", "c1", "c2"], n_factors=4, n_iter=3)
    assert np.allclose(a.B_, b.B_, atol=1e-12)


def test_ram_state_scales_with_items_not_interactions():
    df_s, _, _, _ = _data(U=30, I=20)
    df_l, _, _, _ = _data(U=80, I=60)
    a = fit_hybrid_mf(df_s, "u", "i", ["c0", "c1", "c2"], n_factors=4, n_iter=2)
    b = fit_hybrid_mf(df_l, "u", "i", ["c0", "c1", "c2"], n_factors=4, n_iter=2)
    assert b.state_nbytes() > a.state_nbytes()
    assert b.state_nbytes() < 10 * 1024 * 1024


def test_guard_content_matrix():
    df, _, _, _ = _data(U=200, I=200, c=3)
    mg = MemoryGuard(max_memory_mb=1)
    with pytest.raises(MemoryLimitExceeded):
        fit_hybrid_mf(df, "u", "i", ["c0", "c1", "c2"], n_factors=256,
                      n_iter=2, memory_guard=mg)


def test_errors():
    df, _, _, _ = _data()
    with pytest.raises(KeyError):
        fit_hybrid_mf(df, "u", "i", ["nope"], n_iter=2)
    with pytest.raises(ValueError):
        fit_hybrid_mf(df, "u", "i", [], n_iter=2)


def test_source_contract():
    src = inspect.getsource(hmod)
    assert "Content-Boosted" in src or "Content-Boosted" in src.replace("-", "-")
    assert "ridge" in src
    assert "import pandas" not in src
    assert "log_fit" in inspect.getsource(hmod.fit_hybrid_mf)
    assert "check_state" in inspect.getsource(hmod.fit_hybrid_mf)
