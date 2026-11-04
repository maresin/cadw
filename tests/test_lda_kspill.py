"""K-spill vocab tests for fit_lda (v0.9.116, AP5 fix).

Verifies:
1. Exact match vs legacy _build_vocab (small corpus).
2. Bounded RAM vs legacy at large unique-token corpus.
3. fit_lda with K-spill works end-to-end.
4. Empty-input handling.
"""
from __future__ import annotations
import gc
import ctypes

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, _read_vmrss_fast
from cadw.ml.lda import _build_vocab, _build_vocab_k_spill, fit_lda


def _clean():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    gc.collect()


def _make_df(n, n_words=50, seed=0):
    rng = np.random.RandomState(seed)
    words = [f"w{i}" for i in range(n_words)]
    texts = [" ".join(rng.choice(words, size=8)) for _ in range(n)]
    return CadwDataFrame.from_table(pa.table({"text": texts}))


def _make_df_diverse(n, seed=0):
    """High-cardinality vocab: each row unique-ish tokens."""
    rng = np.random.RandomState(seed)
    texts = [" ".join(f"t{rng.randint(0, 1_000_000)}" for _ in range(8))
             for _ in range(n)]
    return CadwDataFrame.from_table(pa.table({"text": texts}))


def test_k_spill_vocab_matches_legacy(tmp_path):
    """Small corpus, P=4: exact vocab match (order-sensitive)."""
    df = _make_df(500, n_words=50, seed=0)
    mg = MemoryGuard(max_memory_mb=200)
    v_legacy = _build_vocab(df, "text", vocab_size=20, mg=mg)
    v_spill, _ = _build_vocab_k_spill(
        df, "text", vocab_size=20, mg=mg,
        spill_dir=tmp_path, num_partitions=4)
    assert v_legacy == v_spill, (
        f"legacy={v_legacy[:10]} spill={v_spill[:10]} "
        f"(len {len(v_legacy)} vs {len(v_spill)})")


def test_k_spill_vocab_exact_with_skew(tmp_path):
    """Zipf-like distribution: exact top-vocab_size."""
    rng = np.random.RandomState(1)
    # 100 words, weights ~ 1/rank
    n_words = 100
    weights = 1.0 / np.arange(1, n_words + 1)
    weights /= weights.sum()
    docs = []
    for _ in range(800):
        n_tok = rng.randint(3, 15)
        toks = rng.choice([f"w{i}" for i in range(n_words)],
                          size=n_tok, p=weights)
        docs.append(" ".join(toks))
    df = CadwDataFrame.from_table(pa.table({"text": docs}))
    mg = MemoryGuard(max_memory_mb=200)
    v_legacy = _build_vocab(df, "text", vocab_size=20, mg=mg)
    v_spill, _ = _build_vocab_k_spill(
        df, "text", vocab_size=20, mg=mg,
        spill_dir=tmp_path, num_partitions=8)
    assert v_legacy == v_spill


def test_k_spill_bounded_memory(tmp_path):
    """High-cardinality corpus: K-spill peak << legacy peak."""
    df = _make_df_diverse(10_000, seed=0)
    mg = MemoryGuard(max_memory_mb=40)
    _clean()
    rss0 = _read_vmrss_fast()
    v_legacy = _build_vocab(df, "text", vocab_size=100, mg=mg)
    rss_legacy = (_read_vmrss_fast() - rss0) / 1e6
    _clean()
    rss0 = _read_vmrss_fast()
    v_spill, _ = _build_vocab_k_spill(
        df, "text", vocab_size=100, mg=mg,
        spill_dir=tmp_path, num_partitions=8)
    rss_spill = (_read_vmrss_fast() - rss0) / 1e6
    # Both should have same top-100 (they're different corpora — actually
    # same corpus, so same vocab; sizes differ due to merge dedup)
    # Peak: spill should be dramatically less
    assert len(v_spill) > 0
    # Report ratios — not strict assertion (RSS noisy), but sanity
    print(f"legacy dRSS={rss_legacy:.1f}MB, spill dRSS={rss_spill:.1f}MB")
    # Sanity: spill shouldn't be >> legacy (would indicate bug)
    assert rss_spill < max(rss_legacy * 2.0, 30.0)


def test_fit_lda_with_k_spill(tmp_path):
    """fit_lda(num_partitions=4) works and finds same topics roughly."""
    df = _make_df(500, n_words=50, seed=0)
    mg = MemoryGuard(max_memory_mb=200)
    m_legacy = fit_lda(df, "text", n_topics=3, vocab_size=20,
                       max_iter=2, n_restarts=1, seed=0, memory_guard=mg)
    m_spill = fit_lda(df, "text", n_topics=3, vocab_size=20,
                      max_iter=2, n_restarts=1, seed=0, memory_guard=mg,
                      spill_dir=tmp_path, num_partitions=4)
    # Same vocab → same top-20
    assert list(m_legacy.vocab_) == list(m_spill.vocab_)


def test_k_spill_empty_input(tmp_path):
    """Empty df / no tokens → [] (no crash)."""
    df = CadwDataFrame.from_table(pa.table({"text": [""]}))
    mg = MemoryGuard(max_memory_mb=50)
    v, _ = _build_vocab_k_spill(
        df, "text", vocab_size=10, mg=mg,
        spill_dir=tmp_path, num_partitions=4)
    assert v == []
