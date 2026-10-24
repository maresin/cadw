"""Tests for D.5.4 BIRCH CF-tree streaming (v0.9.40)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import birch as bmod
from cadw.ml.birch import fit_birch, BIRCHModel


def _blobs(n=2000, seed=0, k=4):
    rng = np.random.RandomState(seed)
    centers = np.array([[0,0],[5,5],[-5,5],[5,-5]], dtype=float)[:k]
    Y = rng.randint(0, k, size=n)
    X = centers[Y] + rng.normal(0, 0.4, size=(n, 2))
    return X, Y


def _df(X):
    return CadwDataFrame.from_table(pa.table({
        "a": pa.array(X[:, 0]), "b": pa.array(X[:, 1])}))


def _purity(Y, lab):
    from collections import Counter
    by = {}
    for b, l in zip(Y, lab):
        by.setdefault(int(b), Counter())[int(l)] += 1
    return sum(c.most_common(1)[0][1] for c in by.values()) / len(Y)


def test_synthetic_blobs_purity():
    X, Y = _blobs(seed=0)
    m = fit_birch(_df(X), ["a", "b"], n_clusters=4, threshold=0.5)
    assert _purity(Y, m.predict(_df(X))) > 0.95


def test_reference_sklearn_birch_comparable():
    sk = pytest.importorskip("sklearn.cluster")
    X, Y = _blobs(seed=1)
    ours = _purity(Y, fit_birch(_df(X), ["a", "b"], n_clusters=4,
                                threshold=0.5).predict(_df(X)))
    ref = _purity(Y, sk.Birch(n_clusters=4, threshold=0.5,
                              branching_factor=50).fit_predict(X))
    assert ours > ref * 0.95, (ours, ref)


def test_cf_count_and_state_finite():
    X, _ = _blobs(seed=2)
    m = fit_birch(_df(X), ["a", "b"], n_clusters=4)
    d = m.to_dict()
    assert d["n_cf"] > 0
    assert m.state_nbytes() > 0
    assert m.centers_.shape == (4, 2)


def test_determinism():
    X, _ = _blobs(n=800, seed=3)
    a = fit_birch(_df(X), ["a", "b"], n_clusters=4).predict(_df(X))
    b = fit_birch(_df(X), ["a", "b"], n_clusters=4).predict(_df(X))
    assert np.array_equal(a, b)


def test_edge_guards():
    X, _ = _blobs(n=200, seed=4)
    with pytest.raises(ValueError, match="n_clusters"):
        fit_birch(_df(X), ["a"], n_clusters=0)
    with pytest.raises(ValueError, match="threshold"):
        fit_birch(_df(X), ["a"], threshold=0.0)
    with pytest.raises(ValueError, match="branching_factor"):
        fit_birch(_df(X), ["a"], branching_factor=1)
    with pytest.raises(ValueError, match="non-empty"):
        fit_birch(_df(X), [], n_clusters=2)
    with pytest.raises(KeyError):
        fit_birch(_df(X), ["nope"], n_clusters=2)


def test_guard_nulls():
    X, _ = _blobs(n=200, seed=5)
    X = X.copy(); X[3, 0] = np.nan
    with pytest.raises(ValueError, match="nulls"):
        fit_birch(_df(X), ["a", "b"], n_clusters=2)


def test_ram_guard_raises_when_tight():
    X, _ = _blobs(n=1000, seed=6)
    mg = MemoryGuard(max_memory_bytes=512)
    with pytest.raises(MemoryLimitExceeded):
        fit_birch(_df(X), ["a", "b"], n_clusters=2, memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    X, _ = _blobs(n=1200, seed=7)
    df = _df(X)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "b.parquet")
    pq.write_table(tbl, path, row_group_size=97)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100)
    m1 = fit_birch(df, ["a", "b"], n_clusters=4)
    m2 = fit_birch(dmb, ["a", "b"], n_clusters=4)
    assert np.allclose(m1.centers_, m2.centers_, atol=1e-9)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(bmod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
