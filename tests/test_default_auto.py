"""v0.9.92: default output="auto" (BREAKING) contract tests."""
from __future__ import annotations

import inspect
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.spill import AutoArrayResult, AutoResult


def _batches(table, bs=100):
    for i in range(0, table.num_rows, bs):
        yield table.slice(i, bs).to_batches()[0]


# ─── Signature: 12 functions default output="auto" ──────────────

def test_default_auto_signatures():
    """v0.9.92 BREAKING: 12 producers default to output='auto'."""
    from cadw.ops.groupby import groupby_aggregate
    from cadw.ops.dedup import drop_duplicates
    from cadw.ops.topk import topk_to_disk
    from cadw.ml.iforest import IsolationForestModel
    from cadw.ml.chains import ChainsModel
    from cadw.ml.ordinal import OrdinalLogitModel
    from cadw.ml.tfidf import TfidfModel
    from cadw.ml.randsvd import RandomizedSVDModel
    from cadw.ml.gmm import GMMModel

    checks = [
        ('groupby_aggregate', groupby_aggregate),
        ('drop_duplicates', drop_duplicates),
        ('topk_to_disk', topk_to_disk),
        ('iforest.score_samples', IsolationForestModel.score_samples),
        ('chains.predict_proba', ChainsModel.predict_proba),
        ('chains.predict', ChainsModel.predict),
        ('ordinal.predict_proba', OrdinalLogitModel.predict_proba),
        ('ordinal.predict', OrdinalLogitModel.predict),
        ('tfidf.transform_dense', TfidfModel.transform_dense),
        ('randsvd.transform', RandomizedSVDModel.transform),
        ('gmm.score_samples', GMMModel.score_samples),
        ('gmm.predict', GMMModel.predict),
    ]
    for name, fn in checks:
        sig = inspect.signature(fn)
        assert 'output' in sig.parameters, f'{name}: no output param'
        default = sig.parameters['output'].default
        assert default == 'auto', (
            f'{name}: default={default!r}, expected "auto"')


# ─── Behavior: default returns AutoResult, facade unwraps ─────

def test_groupby_default_returns_autoresult(tmp_path):
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ops.groupby import groupby_aggregate
    table = pa.table({'k': ['a', 'b', 'a', 'b'], 'v': [1, 2, 3, 4]})
    result = groupby_aggregate(
        _batches(table), ['k'], ['v'], ['sum'],
        MemoryGuard(max_memory_mb=64), tmp_path, num_partitions=2, log=False)
    assert isinstance(result, AutoResult)


def test_groupby_facade_returns_cadwdataframe(tmp_path):
    """Facade always returns CadwDataFrame, regardless of default."""
    table = pa.table({'k': ['a', 'b', 'a', 'b'], 'v': [1, 2, 3, 4]})
    df = CadwDataFrame.from_table(table)
    result = df.groupby('k').agg('v', 'sum')
    assert isinstance(result, CadwDataFrame)
    assert result.count() == 2


def test_drop_duplicates_facade_returns_cadwdataframe():
    table = pa.table({
        'k': ['a', 'a', 'b', 'b', 'c'],
        'v': [1, 2, 3, 4, 5],
    })
    df = CadwDataFrame.from_table(table)
    result = df.drop_duplicates(subset=['k'], keep='first', num_partitions=2)
    assert isinstance(result, CadwDataFrame)
    assert result.count() == 3


def test_nlargest_facade_returns_cadwdataframe():
    table = pa.table({'v': [3, 1, 4, 1, 5, 9, 2, 6]})
    df = CadwDataFrame.from_table(table)
    result = df.nlargest(3, 'v')
    assert isinstance(result, CadwDataFrame)
    assert result.count() == 3


# ─── Backward compat: output="memory" returns old type ────────

def test_groupby_memory_still_returns_table(tmp_path):
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ops.groupby import groupby_aggregate
    table = pa.table({'k': ['a', 'b'], 'v': [1, 2]})
    result = groupby_aggregate(
        _batches(table), ['k'], ['v'], ['sum'],
        MemoryGuard(max_memory_mb=64), tmp_path, num_partitions=2,
        log=False, output='memory')
    assert isinstance(result, pa.Table)


def test_iforest_memory_still_returns_ndarray():
    from cadw.ml.iforest import fit_isolation_forest
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 2))
    df = CadwDataFrame.from_table(pa.table({
        'x0': pa.array(X[:, 0]), 'x1': pa.array(X[:, 1])}))
    m = fit_isolation_forest(df, ['x0', 'x1'], n_trees=5,
                              subsample_size=64, random_state=0)
    s = m.score_samples(df, output='memory')
    assert isinstance(s, np.ndarray)
    assert s.shape == (200,)
