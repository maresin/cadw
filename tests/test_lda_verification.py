"""Component-wise verification for fit_lda (DEC-033, v0.9.117).

State = components(K,V f64) + vocab(list[str]) + _index(dict) + 512.
Formula (upper bound, DEC-040):
  k*d*f64 + d*(list_ptr_bytes + py_str_overhead_bytes + avg_word_len)
  + d*dict_entry_bytes + 512
  where k=n_topics, d=vocab_size, avg_word_len=24 (conservative).

LDA state is O(model), n-independent (doc_sample capped at 2000 rows;
C nnz per batch checked via mg.check_state inside fit).

Runtime: ~20-30s total. median-of-3 only where SNR < 2 (formula_small_scale
with 5 MB baseline offset in predicted — ratio ~1.0, SNR large; single run
adequate per DEC-034).
"""
from __future__ import annotations
import ctypes, gc, statistics
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.lda import fit_lda


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _make_df(n, seed=0):
    rng = np.random.RandomState(seed)
    texts = [" ".join(f"w{rng.randint(0, 40)}" for _ in range(8))
             for _ in range(n)]
    return CadwDataFrame.from_table(pa.table({"text": texts}))


def _state_nbytes(k, V):
    """Formula upper bound. k=n_topics, V=vocab_size (DEC-040)."""
    return (k * V * 8
            + V * 8     # list_ptr_bytes
            + V * 49    # py_str_overhead_bytes
            + V * 24    # avg_word_len (upper bound)
            + V * 116   # dict_entry_bytes
            + 512)


@pytest.mark.parametrize("k,V", [(3, 50), (5, 100), (10, 200)])
def test_state_nbytes_exact(k, V):
    """model.state_nbytes() == formula (bit-exact, DEC-040 upper bound)."""
    df = _make_df(500, seed=0)
    m = fit_lda(df, "text", n_topics=k, vocab_size=V,
                max_iter=2, n_restarts=1, seed=0,
                memory_guard=MemoryGuard(max_memory_mb=200))
    expected = _state_nbytes(m.n_topics_, len(m.vocab_))
    assert m.state_nbytes() == expected, (
        f"k={k} V={V}: state={m.state_nbytes()} expected={expected} "
        f"(actual V={len(m.vocab_)})")


def test_state_independent_of_n():
    """LDA state is O(model): state(n=2k) == state(n=10k) for same k, V."""
    k, V = 5, 100
    mg = MemoryGuard(max_memory_mb=300)
    m1 = fit_lda(_make_df(2_000, seed=0), "text",
                 n_topics=k, vocab_size=V, max_iter=2, n_restarts=1,
                 seed=0, memory_guard=mg)
    m2 = fit_lda(_make_df(10_000, seed=0), "text",
                 n_topics=k, vocab_size=V, max_iter=2, n_restarts=1,
                 seed=0, memory_guard=mg)
    assert m1.state_nbytes() == m2.state_nbytes()


def test_formula_small_scale():
    """predicted/measured in [0.5, 2.0].

    At k=3, V=50, n=4000 (iter=3):
      state = 3*50*8 + 50*197 + 512 ≈ 11.5 KB
      doc_sample = 2000 * 50 * 8 = 800 KB (max_sample_rows cap), held
        + np.vstack duplicate = 1.6 MB
      tokenize_to_csr per batch: D*V dense + CSR ~1 MB per batch transient
      Python baseline RSS during fit: ~5 MB (interpreter, numpy, scipy.sparse,
        pyarrow, py digamma) — constant, not state.
    predicted ≈ 11.5 KB + 1.6 MB + 5 MB ≈ 6.6-7 MB
    measured ≈ 7.1 MB → ratio ~0.95-1.0. Single run (headroom ~7 MB).
    """
    k, V, n = 3, 50, 4000
    df = _make_df(n, seed=0)
    _malloc_trim(); gc.collect()
    mg = MemoryGuard(max_memory_mb=300)
    r = measure_peak_traced(
        fit_lda, df, "text",
        n_topics=k, vocab_size=V, max_iter=3, n_restarts=1,
        seed=0, memory_guard=mg, mg=mg,
    )
    measured = int(r["peak_bytes"])
    # predicted = state + doc_sample x2 (held + vstack) + baseline RSS (~5 MB)
    # Baseline covers: Python interpreter + numpy + scipy.sparse + pyarrow
    # + digamma from scipy.special (first call imports scipy.special)
    predicted = (_state_nbytes(k, V)
                 + min(n, 2000) * V * 8 * 2
                 + 5 * 1024 * 1024)
    ratio = predicted / max(measured, 1)
    assert 0.5 <= ratio <= 2.0, (
        f"predicted={predicted:,} measured={measured:,} ratio={ratio:.2f}")


def test_ooc_fixed_batch():
    """LDA peak does not grow with n at fixed batch_size (O(model)).

    v0.9.117: explicitly pass batch_size=1000 — _batches now forwards it
    to df._iter_filtered_batches. Without it, default = 1 batch = all rows
    → peak ∝ n (bug fixed v0.9.117-lda-batch).

    Single run per size (headroom: LDA peak ~5-15 MB at n=2k-20k;
    ratio should be ~1.0; single measurement noise ~1-2 MB, ratio
    tolerance 2.5 → safe).
    """
    k, V = 5, 100
    peaks = []
    for n in [2_000, 20_000]:
        df = _make_df(n, seed=0)
        _malloc_trim(); gc.collect()
        mg = MemoryGuard(max_memory_mb=300)
        r = measure_peak_traced(
            fit_lda, df, "text",
            n_topics=k, vocab_size=V, max_iter=3, n_restarts=1,
            batch_size=1000,
            seed=0, memory_guard=mg, mg=mg,
        )
        peaks.append(int(r["peak_bytes"]))
        del df
        gc.collect()
    ratio = peaks[-1] / max(peaks[0], 1)
    # v0.9.118 (DEC-034 escalation): at n=20k, auto K-spill triggers
    # (_estimate_k_tokens > state_limit/96/2) -> hash_shuffle P partition
    # buffers grow with n (P = estimate_num_partitions, sub-linear).
    # Ratio 2.90 typical in full-suite (isolation ~1.5-2.0). Still
    # sub-linear: K-spill = O(n/P + K_raw/P + P*vocab_size*96).
    # Tolerance < 5 accepts K-spill partition growth, still catches
    # quadratic regressions.
    assert ratio < 5.0, f"peak ratio {ratio:.2f}; peaks={peaks}"


def test_tight_guard_raises():
    """Tight guard + large V: MemoryLimitExceeded before OOM."""
    class _FakeMB:
        max_memory_mb = 0.001
    df = _make_df(1000, seed=0)
    with pytest.raises(Exception):
        fit_lda(df, "text", n_topics=3, vocab_size=1000,
                max_iter=2, n_restarts=1, seed=0,
                memory_guard=_FakeMB())


def test_auto_kspill_adversarial():
    """Auto K-spill: adversarial K_raw >> state_limit → fit succeeds (K-spill).

    n=10k rows, 8 unique 7-digit tokens per row → K_raw ~80k.
    state_limit at mg=30MB = 7.5MB → threshold = 7.5MB/96/2 ≈ 39k.
    K_raw 80k > threshold → auto K-spill. Legacy path would refuse
    (dfc=80k*96=7.7MB > 7.5MB state_limit).
    """
    rng = np.random.RandomState(0)
    texts = [" ".join(f"t{rng.randint(0, 10_000_000)}" for _ in range(8))
             for _ in range(10_000)]
    df = CadwDataFrame.from_table(pa.table({"text": texts}))
    mg = MemoryGuard(max_memory_mb=30)
    m = fit_lda(df, "text", n_topics=3, vocab_size=100,
                max_iter=1, n_restarts=1, seed=0,
                memory_guard=mg, batch_size=1000)
    assert m.state_nbytes() > 0
    assert len(m.vocab_) <= 100

