"""Tests for D.1.2 Matrix Profile (v0.9.35)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import matrix_profile as mp
from cadw.ml.matrix_profile import fit_matrix_profile, MatrixProfileModel, _znorm_rows, _sliding_view

def _df(x):
    return CadwDataFrame.from_table(pa.table({"v": pa.array(np.asarray(x, dtype=float))}))

def _brute(x, w, sample):
    V = _sliding_view(np.asarray(x, float), w)
    ref = _znorm_rows(np.asarray(sample, float))
    Z = _znorm_rows(V)
    d = np.sqrt(np.maximum((Z*Z).sum(1)[:,None] + (ref*ref).sum(1)[None,:] - 2*(Z@ref.T), 0))
    return d.min(axis=1)


def test_smoke_and_shapes():
    x = np.sin(np.arange(400) * 0.2)
    m = fit_matrix_profile(_df(x), "v", window=16, max_samples=128)
    assert isinstance(m, MatrixProfileModel)
    assert m.profile_.shape == (128,)
    assert np.all(np.isfinite(m.profile_))
    assert m.state_nbytes() == 128 * (8 + 8 + 16 * 8)


def test_reference_vs_bruteforce():
    rng = np.random.RandomState(0)
    x = rng.normal(size=300)
    m = fit_matrix_profile(_df(x), "v", window=12, max_samples=64, random_state=0)
    # recompute from the stored sample: profile must equal exact brute force
    ref = _brute_of_ref(m)
    assert np.allclose(m.profile_, ref, atol=1e-10)


def _brute_of_ref(m):
    R = m.ref_zn_
    n = R.shape[0]
    sq = (R * R).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * (R @ R.T)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))


def test_planted_anomaly_is_top_discord():
    rng = np.random.RandomState(1)
    x = np.tile(np.sin(np.arange(64) * 0.3), 20)  # periodic
    x[700:716] += 8.0  # spike
    m = fit_matrix_profile(_df(x), "v", window=16, max_samples=4096, random_state=0)
    scores = m.discord_scores()
    # subsequences overlapping the spike must be in the top discords
    top = [i for i, _ in m.top_discords(30)]
    assert any(690 <= i <= 716 for i in top)
    assert scores.max() > 0.5


def test_discord_score_bounds():
    x = np.sin(np.arange(200) * 0.25)
    m = fit_matrix_profile(_df(x), "v", window=16, max_samples=64)
    s = m.discord_scores()
    assert np.all(s <= 2.0 + 1e-9)
    assert np.all(s >= 0.0 - 1e-9)


def test_determinism_same_seed():
    x = np.random.RandomState(2).normal(size=500)
    a = fit_matrix_profile(_df(x), "v", window=10, max_samples=100, random_state=7)
    b = fit_matrix_profile(_df(x), "v", window=10, max_samples=100, random_state=7)
    assert np.array_equal(a.profile_, b.profile_)
    assert np.array_equal(a.index_, b.index_)


def test_multi_batch_invariance():
    import tempfile, pyarrow.parquet as pq
    x = np.random.RandomState(3).normal(size=400)
    path = tempfile.mktemp(suffix=".parquet")
    pq.write_table(pa.table({"v": pa.array(x)}), path, row_group_size=13)
    d1 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1)
    d2 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=10000)
    a = fit_matrix_profile(d1, "v", window=8, max_samples=50, random_state=1)
    b = fit_matrix_profile(d2, "v", window=8, max_samples=50, random_state=1)
    assert a.n_seen_ == b.n_seen_ == 400 - 8 + 1
    assert np.allclose(a.profile_, b.profile_)


def test_state_independent_of_n_and_ram():
    x1 = np.random.RandomState(4).normal(size=2000)
    x2 = np.random.RandomState(4).normal(size=20000)
    a = fit_matrix_profile(_df(x1), "v", window=16, max_samples=256)
    b = fit_matrix_profile(_df(x2), "v", window=16, max_samples=256)
    assert a.state_nbytes() == b.state_nbytes()
    assert b.state_nbytes() < 8 * 256 * 32


def test_guard_sample_too_large():
    mg = MemoryGuard(max_memory_mb=2)
    x = np.random.RandomState(5).normal(size=5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_matrix_profile(_df(x), "v", window=64, max_samples=40000, memory_guard=mg)


def test_errors_window_and_nulls():
    x = np.random.RandomState(6).normal(size=100)
    with pytest.raises(ValueError):
        fit_matrix_profile(_df(x), "v", window=2)
    with pytest.raises(KeyError):
        fit_matrix_profile(_df(x), "zzz", window=8)
    y = x.copy(); y[10] = np.nan
    with pytest.raises(ValueError):
        fit_matrix_profile(_df(y), "v", window=8, max_samples=32)
    with pytest.raises(ValueError):
        # 20 points, window 18 -> only 3 subsequences, below the minimum of 8
        fit_matrix_profile(_df(np.arange(20.0)), "v", window=18, max_samples=64)


def _brute_exact(x, w, ref_zn):
    V = _sliding_view(np.asarray(x, float), w)
    Z = _znorm_rows(V)
    d2 = (Z*Z).sum(1)[:, None] + (ref_zn*ref_zn).sum(1)[None, :] - 2*(Z @ ref_zn.T)
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))


def test_transform_matches_bruteforce_on_profile_values():
    rng = np.random.RandomState(7)
    x = rng.normal(size=250)
    m = fit_matrix_profile(_df(x), "v", window=10, max_samples=64, random_state=0)
    t = m.transform(_df(x), "v")
    got = t.column("matrix_profile").to_numpy(zero_copy_only=False)
    exp = _brute_exact(x, 10, m.ref_zn_)
    assert got.shape == exp.shape
    assert np.allclose(got, exp, atol=1e-9)


def test_exact_path_matches_bruteforce_stamp():
    """When every subsequence is sampled (frac=1.0) the profile must be
    bit-identical to a brute-force STAMP over the full series."""
    rng = np.random.RandomState(11)
    x = rng.normal(size=200) + np.sin(np.arange(200) * 0.3)
    m = fit_matrix_profile(_df(x), "v", window=12, max_samples=4096, random_state=0)
    assert m.sample_fraction_ == 1.0
    assert np.allclose(m.profile_, _brute_of_ref(m), atol=1e-12)
    # and against a direct full-series computation, not just the sample
    V = _sliding_view(np.asarray(x, float), 12)
    Z = _znorm_rows(V)
    sq = (Z * Z).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * (Z @ Z.T)
    np.fill_diagonal(d2, np.inf)
    assert np.allclose(m.profile_, np.sqrt(np.maximum(d2.min(1), 0.0)), atol=1e-12)


def test_source_contract():
    src = inspect.getsource(mp)
    assert "arXiv:1701.02795" in src
    assert "reservoir" in src
    assert "check_batch(" in src
    for line in src.splitlines():
        if "check_batch(" in line:
            assert "extra_factor=" in line
    assert "import pandas" not in src
