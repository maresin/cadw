"""v0.9.34 (A10): empty-input contract documented + enforced.

Both `external_sort` and `hash_shuffle` raise ValueError on empty input.
Docstrings must say so explicitly (contract, not comment).
"""
import inspect

import pytest

from cadw import MemoryGuard
from cadw.ops.sort import external_sort
from cadw.ops.shuffle import hash_shuffle


def test_external_sort_docstring_documents_empty():
    doc = inspect.getdoc(external_sort) or ""
    low = doc.lower()
    assert "empty" in low or "пуст" in low
    assert "valueerror" in low or "raises" in low


def test_hash_shuffle_docstring_documents_empty():
    doc = inspect.getdoc(hash_shuffle) or ""
    low = doc.lower()
    assert "empty" in low or "пуст" in low
    assert "valueerror" in low or "raises" in low


def test_external_sort_empty_raises(tmp_path):
    g = MemoryGuard(max_memory_mb=20)
    with pytest.raises(ValueError):
        external_sort(iter([]), ["k"], ["v"], g, tmp_path, log=False)


def test_hash_shuffle_empty_raises(tmp_path):
    g = MemoryGuard(max_memory_mb=20)
    with pytest.raises(ValueError):
        hash_shuffle(iter([]), ["k"], ["v"], 2, g, tmp_path, log=False)
