"""Tests for VECM (D2, v0.9.89)."""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.ts.vecm import VECMModel


def _cointegrated_data(n=500, seed=0):
    """Two series sharing a common stochastic trend, β=(1,-1)."""
    rng = np.random.RandomState(seed)
    common = np.cumsum(rng.normal(0, 1, n))
    a = common + 0.3 * rng.normal(0, 1, n)
    b = common + 0.3 * rng.normal(0, 1, n)
    return np.column_stack([a, b])


def _df(Y, names=("a", "b")):
    return CadwDataFrame.from_table(pa.table({
        names[j]: pa.array(Y[:, j]) for j in range(Y.shape[1])}))


def test_vecm_cointegration_detected():
    Y = _cointegrated_data(seed=0)
    m = VECMModel(k_ar_diff=1, coint_rank=1).fit(_df(Y), ["a", "b"])
    assert m.beta_.shape == (2, 1)
    assert m.alpha_.shape == (2, 1)
    # β direction should be close to (1, -1): β_0 + β_1 ≈ 0
    beta = m.beta_.ravel()
    assert abs(beta[0] + beta[1]) < 0.1 * abs(beta[0]), (
        f"β should be (1,-1)-ish; got {beta}")


def test_vecm_independent_series_no_coint():
    """Two independent random walks — β should not be meaningful,
    but model should not crash and forecast should be finite."""
    rng = np.random.RandomState(1)
    a = np.cumsum(rng.normal(0, 1, 300))
    b = np.cumsum(rng.normal(0, 1, 300))
    Y = np.column_stack([a, b])
    m = VECMModel(k_ar_diff=1, coint_rank=1).fit(_df(Y), ["a", "b"])
    fc = m.forecast_vector(5)
    assert np.all(np.isfinite(fc))


def test_vecm_determinism():
    Y = _cointegrated_data(seed=2)
    a = VECMModel(k_ar_diff=1, coint_rank=1).fit(_df(Y), ["a", "b"])
    b = VECMModel(k_ar_diff=1, coint_rank=1).fit(_df(Y), ["a", "b"])
    assert np.array_equal(a.beta_, b.beta_)


def test_vecm_multi_batch_invariance(tmp_path):
    import pyarrow.parquet as pq
    Y = _cointegrated_data(seed=3)
    tbl = pa.table({"a": pa.array(Y[:, 0]), "b": pa.array(Y[:, 1])})
    path = str(tmp_path / "d.parquet")
    pq.write_table(tbl, path, row_group_size=17)
    d1 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1)
    d2 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100000)
    a = VECMModel(k_ar_diff=1, coint_rank=1).fit(d1, ["a", "b"])
    b = VECMModel(k_ar_diff=1, coint_rank=1).fit(d2, ["a", "b"])
    assert np.allclose(a.beta_, b.beta_, atol=1e-6)
    assert np.allclose(a.alpha_, b.alpha_, atol=1e-6)


def test_vecm_k_ar_diff_2():
    """k_ar_diff=2 exercises Γ short-run term."""
    Y = _cointegrated_data(seed=4)
    m = VECMModel(k_ar_diff=2, coint_rank=1).fit(_df(Y), ["a", "b"])
    fc = m.forecast_vector(5)
    assert fc.shape == (5, 2)
    assert np.all(np.isfinite(fc))


def test_vecm_reference_statsmodels_forecast():
    """Forecast parity with statsmodels (allows some numerical
    sensitivity due to Johansen eig normalization)."""
    sm = pytest.importorskip("statsmodels.tsa.vector_ar.vecm")
    Y = _cointegrated_data(seed=5, n=400)
    ours = VECMModel(k_ar_diff=1, coint_rank=1).fit(
        _df(Y), ["a", "b"]).forecast_vector(5)
    # statsmodels reference — fit on same data, forecast 5
    res = sm.VECM(Y, k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
    ref = res.predict(steps=5) if hasattr(res, 'predict') else None
    if ref is None:
        pytest.skip("statsmodels VECM predict API changed")
    # Compare ΔY (not level) — level paths diverge due to β normalization
    ours_dY = np.diff(np.vstack([Y[-1:], ours]), axis=0)
    ref_dY = np.diff(np.vstack([Y[-1:], ref]), axis=0)
    # Loose tolerance; Johansen numerical sensitivities
    assert np.allclose(ours_dY, ref_dY, atol=0.5), (
        f"ΔY forecast diverge: ours {ours_dY[:2]} vs ref {ref_dY[:2]}")


def test_vecm_state_bounded():
    Y_small = _cointegrated_data(n=200, seed=6)
    Y_large = _cointegrated_data(n=5000, seed=6)
    a = VECMModel(k_ar_diff=1, coint_rank=1).fit(_df(Y_small), ["a", "b"])
    b = VECMModel(k_ar_diff=1, coint_rank=1).fit(_df(Y_large), ["a", "b"])
    assert a.state_nbytes() == b.state_nbytes()


def test_vecm_edge_guards():
    Y = _cointegrated_data(seed=7)
    with pytest.raises(ValueError, match="k_ar_diff"):
        VECMModel(k_ar_diff=0)
    with pytest.raises(ValueError, match="coint_rank"):
        VECMModel(k_ar_diff=1, coint_rank=0)
    with pytest.raises(KeyError):
        VECMModel(k_ar_diff=1, coint_rank=1).fit(_df(Y), ["nope"])


def test_vecm_source_contract_no_full_materialization():
    import inspect
    from cadw.ts import vecm as vecm_mod
    src = inspect.getsource(vecm_mod)
    assert "_extract_series" not in src
    assert "_iter_filtered_batches" in src
    assert "check_state" in src and "check_batch" in src
