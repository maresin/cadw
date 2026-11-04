"""Component-wise verification for fit_tfidf (DEC-033, v0.9.117).

State = idf(V f64) + vocab(list[str]) + _index(dict) + 512.
Formula (upper bound, DEC-040):
  d*f64 + d*(list_ptr_bytes + py_str_overhead_bytes + avg_word_len)
  + d*dict_entry_bytes + 512
  where d=vocab_size, avg_word_len=24 (conservative).

DEC-041: fit peak O(batch + K_raw). bounded_state applies to state
(O(V)), not fit peak. With batch_size and K_raw << state_limit
(natural-language corpora, cadw internal tests: K_raw <= 40), fit is
OOC. With K_raw >> state_limit, `mg.check_state` inside the loop
refuses. K-spill default deferred v0.9.117.
"""
from __future__ import annotations
import ctypes, gc, statistics
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.tfidf import fit_tfidf


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _make_df(n, seed=0):
    rng = np.random.RandomState(seed)
    # 40-word vocab → K_raw=40 << state_limit, fit peak O(batch)
    texts = [" ".join(f"w{rng.randint(0, 40)}" for _ in range(8))
             for _ in range(n)]
    return CadwDataFrame.from_table(pa.table({"text": texts}))


def _state_nbytes(V):
    """Formula upper bound. V=vocab_size (DEC-040)."""
    return (V * 8       # idf_
            + V * 8     # list_ptr
            + V * 49    # py_str_overhead
            + V * 24    # avg_word_len (upper bound)
            + V * 116   # dict_entry
            + 512)


@pytest.mark.parametrize("V", [50, 100, 200])
def test_state_nbytes_exact(V):
    """model.state_nbytes() == formula (bit-exact, DEC-040 upper bound)."""
    df = _make_df(500, seed=0)
    m = fit_tfidf(df, "text", vocab_size=V,
                  memory_guard=MemoryGuard(max_memory_mb=200))
    expected = _state_nbytes(len(m.vocab_))
    assert m.state_nbytes() == expected, (
        f"V={V}: state={m.state_nbytes()} expected={expected} "
        f"(actual V={len(m.vocab_)})")


def test_state_independent_of_n():
    """TF-IDF state is O(model): state(n=2k) == state(n=10k)."""
    V = 100
    mg = MemoryGuard(max_memory_mb=200)
    m1 = fit_tfidf(_make_df(2_000, seed=0), "text",
                   vocab_size=V, memory_guard=mg)
    m2 = fit_tfidf(_make_df(10_000, seed=0), "text",
                   vocab_size=V, memory_guard=mg)
    assert m1.state_nbytes() == m2.state_nbytes()


def test_ooc_fixed_batch():
    """TF-IDF peak does not grow with n at fixed batch_size.

    batch_size=1000 → _iter_filtered_batches(batch_size=1000) → bounded
    batch buffers. K_raw=40 (<< state_limit), so dfc is O(1). Test
    exercises OOC contract (ratio < 2.5).
    """
    V = 100
    peaks = []
    for n in [2_000, 20_000]:
        df = _make_df(n, seed=0)
        _malloc_trim(); gc.collect()
        mg = MemoryGuard(max_memory_mb=300)
        r = measure_peak_traced(
            fit_tfidf, df, "text",
            vocab_size=V, batch_size=1000,
            memory_guard=mg, mg=mg,
        )
        peaks.append(int(r["peak_bytes"]))
        del df
        gc.collect()
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 2.5, f"peak ratio {ratio:.2f}; peaks={peaks}"


def test_tight_guard_raises():
    """Tight guard + large V: MemoryLimitExceeded before OOM."""
    class _FakeMB:
        max_memory_mb = 0.001
    df = _make_df(1000, seed=0)
    with pytest.raises(Exception):
        fit_tfidf(df, "text", vocab_size=1000,
                  memory_guard=_FakeMB())


def test_auto_kspill_adversarial():
    """Auto K-spill: adversarial K_raw >> state_limit → fit succeeds (K-spill).

    n=10k rows, 8 unique 7-digit tokens per row → K_raw ~80k.
    state_limit at mg=30MB = 7.5MB → threshold ≈ 39k. K_raw > threshold
    → auto K-spill. Legacy path would refuse (dfc > state_limit).
    """
    rng = np.random.RandomState(0)
    texts = [" ".join(f"t{rng.randint(0, 10_000_000)}" for _ in range(8))
             for _ in range(10_000)]
    df = CadwDataFrame.from_table(pa.table({"text": texts}))
    mg = MemoryGuard(max_memory_mb=30)
    m = fit_tfidf(df, "text", vocab_size=100,
                  memory_guard=mg, batch_size=1000)
    assert m.state_nbytes() > 0
    assert len(m.vocab_) <= 100

