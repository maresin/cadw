"""Tests for D.4.2 PELT change-point detection (v0.9.38)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import pelt as pelt_mod
from cadw.ml.pelt import fit_pelt, pelt, PELTModel


def _three_regimes(seed=0, n=100):
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.normal(0, 1, n),
                           rng.normal(5, 1, n),
                           rng.normal(0, 1, n)])


def _df_from(y, id_col=None, sid=0):
    if id_col is None:
        return CadwDataFrame.from_table(pa.table({"v": pa.array(y)}))
    ids = [sid] * y.size
    return CadwDataFrame.from_table(pa.table({
        "id": pa.array(ids, type=pa.int64()), "v": pa.array(y)}))


def _multi_series(n_series=3, length=120):
    ids, vals = [], []
    for s in range(n_series):
        rng = np.random.RandomState(s)
        y = np.concatenate([rng.normal(0, 1, length // 2),
                            rng.normal(4 + s, 1, length - length // 2)])
        ids += [s] * length
        vals += list(y)
    return CadwDataFrame.from_table(pa.table({
        "id": pa.array(ids, type=pa.int64()), "v": pa.array(vals)}))


def test_synthetic_detects_three_regimes():
    y = _three_regimes()
    cps = pelt(y, penalty=30.0)
    assert len(cps) == 2
    assert abs(cps[0] - 100) <= 2 and abs(cps[1] - 200) <= 2


def test_reference_ruptures_l2_matches():
    rpt = pytest.importorskip("ruptures")
    rng = np.random.RandomState(1)
    y = np.concatenate([rng.normal(0, 1, 80), rng.normal(8, 1, 80),
                        rng.normal(-6, 1, 80), rng.normal(0, 1, 80)])
    ours = pelt(y, penalty=25.0, min_size=2)
    ref = rpt.Pelt(model="l2", min_size=2).fit(y).predict(pen=25.0)[:-1]
    # exact match is not guaranteed at boundary ties; compare segmentations
    # with a 3-sample tolerance, which is the honest equivalence criterion
    assert len(ours) == len(ref), (ours, ref)
    assert all(abs(a - b) <= 3 for a, b in zip(ours, ref)), (ours, ref)
    # both must find the two dominant regime boundaries
    assert abs(ours[0] - 80) <= 3 and abs(ours[1] - 160) <= 3


def test_no_change_when_penalty_large():
    y = _three_regimes()
    assert pelt(y, penalty=1e9) == []


def test_determinism_repeat_calls():
    y = _three_regimes(seed=3)
    a = pelt(y, penalty=30.0)
    b = pelt(y, penalty=30.0)
    assert a == b


def test_edge_guards_bad_inputs():
    with pytest.raises(ValueError, match="empty"):
        pelt(np.array([]), penalty=1.0)
    with pytest.raises(ValueError, match="non-finite"):
        pelt(np.array([1.0, np.nan]), penalty=1.0)
    with pytest.raises(ValueError, match="penalty"):
        pelt(np.arange(10.0), penalty=0.0)
    with pytest.raises(ValueError, match="min_size"):
        pelt(np.arange(10.0), penalty=1.0, min_size=0)
    with pytest.raises(ValueError, match="max_window"):
        pelt(np.arange(10.0), penalty=1.0, min_size=3, max_window=4)


def test_max_window_bounded_recursion():
    y = _three_regimes()
    cps = pelt(y, penalty=30.0, max_window=40)
    # bounded recursion still detects local regimes, just cannot span > 40
    assert isinstance(cps, list)
    assert all(0 < c < y.size for c in cps)


def test_multi_series_dict_and_flat():
    df = _multi_series(3, 120)
    m = fit_pelt(df, "v", id_col="id", penalty=20.0)
    d = m.detect(df)
    assert set(d) == {0, 1, 2}
    for sid, cps in d.items():
        assert abs(cps[0] - 60) <= 2, (sid, cps)
    flat = m.change_points(df)
    assert len(flat) == 3


def test_guard_unsorted_id_raises():
    ids = [0] * 6 + [1] * 6 + [0] * 6
    vals = list(np.arange(18.0))
    df = CadwDataFrame.from_table(pa.table({
        "id": pa.array(ids, type=pa.int64()), "v": pa.array(vals)}))
    # the ordering guard fires in the fit-time validation pass (stricter: bad
    # input fails at fit rather than at detect)
    with pytest.raises(ValueError, match="grouped/sorted"):
        fit_pelt(df, "v", id_col="id", penalty=1.0)


def test_ram_guard_raises_when_tight():
    df = _df_from(_three_regimes())
    mg = MemoryGuard(max_memory_bytes=256)
    with pytest.raises(MemoryLimitExceeded):
        fit_pelt(df, "v", penalty=30.0, memory_guard=mg)


def test_multi_batch_consistent(tmp_path):
    import pyarrow.parquet as pq
    df = _multi_series(3, 120)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "p.parquet")
    pq.write_table(tbl, path, row_group_size=17)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=50)
    m1 = fit_pelt(df, "v", id_col="id", penalty=20.0)
    m2 = fit_pelt(dmb, "v", id_col="id", penalty=20.0)
    assert m1.detect(df) == m2.detect(dmb)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(pelt_mod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
