"""K-spill vocab tests for fit_tfidf (v0.9.116, AP5 fix, same as LDA)."""
from __future__ import annotations
import gc, ctypes

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, _read_vmrss_fast
from cadw.ml.tfidf import fit_tfidf


def _clean():
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass
    gc.collect()


def _make_df(n, n_words=50, seed=0):
    rng = np.random.RandomState(seed)
    words = [f"w{i}" for i in range(n_words)]
    texts = [" ".join(rng.choice(words, size=8)) for _ in range(n)]
    return CadwDataFrame.from_table(pa.table({"text": texts}))


def _make_df_diverse(n, seed=0):
    rng = np.random.RandomState(seed)
    texts = [" ".join(f"t{rng.randint(0, 1_000_000)}" for _ in range(8))
             for _ in range(n)]
    return CadwDataFrame.from_table(pa.table({"text": texts}))


def test_k_spill_vocab_matches_legacy(tmp_path):
    """Small corpus: exact vocab + idf match."""
    df = _make_df(500, n_words=50, seed=0)
    mg = MemoryGuard(max_memory_mb=200)
    m_legacy = fit_tfidf(df, "text", vocab_size=20, memory_guard=mg)
    m_spill = fit_tfidf(df, "text", vocab_size=20, memory_guard=mg,
                        spill_dir=tmp_path, num_partitions=4)
    assert list(m_legacy.vocab_) == list(m_spill.vocab_)
    np.testing.assert_array_equal(m_legacy.idf_, m_spill.idf_)


def test_k_spill_min_df(tmp_path):
    """min_df filter: rare tokens excluded per-partition (exact)."""
    df = _make_df(500, n_words=50, seed=0)
    mg = MemoryGuard(max_memory_mb=200)
    m_legacy = fit_tfidf(df, "text", vocab_size=30, min_df=3,
                         memory_guard=mg)
    m_spill = fit_tfidf(df, "text", vocab_size=30, min_df=3,
                        memory_guard=mg, spill_dir=tmp_path,
                        num_partitions=4)
    assert list(m_legacy.vocab_) == list(m_spill.vocab_)


def test_k_spill_bounded_memory(tmp_path):
    """High-cardinality corpus: spill bounded (sanity, not strict)."""
    df = _make_df_diverse(10_000, seed=0)
    mg = MemoryGuard(max_memory_mb=40)
    _clean()
    rss0 = _read_vmrss_fast()
    m_legacy = fit_tfidf(df, "text", vocab_size=100, memory_guard=mg)
    rss_legacy = (_read_vmrss_fast() - rss0) / 1e6
    _clean()
    rss0 = _read_vmrss_fast()
    m_spill = fit_tfidf(df, "text", vocab_size=100, memory_guard=mg,
                        spill_dir=tmp_path, num_partitions=8)
    rss_spill = (_read_vmrss_fast() - rss0) / 1e6
    print(f"legacy dRSS={rss_legacy:.1f}MB, spill dRSS={rss_spill:.1f}MB")
    assert rss_spill < max(rss_legacy * 2.0, 30.0)
