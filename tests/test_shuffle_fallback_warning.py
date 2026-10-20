"""K1.2: `_pandas_hash_column` fallback is observable, not silent.

Before: `except Exception: return None` — any pandas API change
(e.g. a future drop of `types_mapper`) would route every batch to the
~30x slower Python hash loop with no signal at all; only a benchmark
could catch it. Now: default path logs a warning with the exception
repr (behaviour otherwise unchanged — still returns None so the
dispatch falls back); `strict=True` re-raises for callers that want a
hard failure.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from cadw.ops import shuffle as shuffle_mod
from cadw.ops.shuffle import (
    _hash_column,
    _pandas_hash_column,
    _python_hash_column,
)


def _int_arr() -> pa.Array:
    return pa.array([1, 2, 3, None, 5], type=pa.int64())


def _break_pandas(monkeypatch):
    def boom(*_a, **_kw):
        raise RuntimeError("simulated pandas API change")

    monkeypatch.setattr(pd.util, "hash_pandas_object", boom)


def test_fast_path_returns_uint64_and_no_warning(caplog):
    caplog.set_level(logging.WARNING)
    out = _pandas_hash_column(_int_arr())
    assert out is not None
    assert out.dtype == np.uint64
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_fallback_warns_and_returns_none(monkeypatch, caplog):
    _break_pandas(monkeypatch)
    caplog.set_level(logging.WARNING)
    out = _pandas_hash_column(_int_arr())
    assert out is None
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING]
    assert any("_pandas_hash_column" in m for m in msgs), msgs
    assert any("simulated pandas API change" in m for m in msgs), msgs


def test_strict_raises_instead_of_warning(monkeypatch):
    _break_pandas(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated pandas API change"):
        _pandas_hash_column(_int_arr(), strict=True)


def test_strict_does_not_affect_successful_path():
    out = _pandas_hash_column(_int_arr(), strict=True)
    assert out is not None and out.dtype == np.uint64


def test_dispatch_falls_back_to_python_and_hashes_identically(monkeypatch):
    """Fallback preserves partition-disjoint: same values -> same hashes
    as the Python reference for the same input."""
    arr = pa.array(["a", "b", "a", None, "c"], type=pa.string())
    _break_pandas(monkeypatch)
    got = _hash_column(arr)  # ints/strings route differently; string ok
    want = _python_hash_column(arr)
    assert np.array_equal(got, want)


def test_uint64_still_routes_via_python_without_warning(caplog):
    """uint64 is an explicit `return None` before try — no fallback log."""
    caplog.set_level(logging.WARNING)
    arr = pa.array([2**64 - 1, 1], type=pa.uint64())
    assert _pandas_hash_column(arr) is None
    assert not [r for r in caplog.records
                if r.levelno >= logging.WARNING], caplog.records
