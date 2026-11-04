"""LDA memory bound regression (v0.9.42, v0.9.116 measure_peak_traced).

Before fix: peak 1319 MB @ 50k rows (linear O(N)) — doc_sample appended
in a closure that was never reset across restarts × iters. After fix:
bounded at MAX_SAMPLE_ROWS=2000, peak well under 200 MB budget.

v0.9.116 (DEC-034): psutil rss delta replaced with measure_peak_traced
(faster, more accurate VmRSS sampler). Single measurement is safe here:
pre-fix peak 1319 MB, post-fix <100 MB, budget threshold = 160 MB
(200 MB × 0.8) — 100+ MB headroom, no median-of-3 needed.
"""
import ctypes
import gc

import numpy as np
import pyarrow as pa

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.lda import fit_lda


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _make_df(n):
    rng = np.random.RandomState(0)
    words = [f"w{i}" for i in range(50)]
    texts = [" ".join(rng.choice(words, size=8)) for _ in range(n)]
    return CadwDataFrame.from_table(pa.table({"text": texts}))


def test_lda_peak_under_budget_at_50k():
    """LDA fit peak must stay under 80% of 200 MB budget.

    v0.9.116: measure_peak_traced, single run (headroom 100+ MB).
    """
    budget_mb = 200
    df = _make_df(50_000)
    _malloc_trim()
    gc.collect()
    mg = MemoryGuard(max_memory_mb=budget_mb)
    r = measure_peak_traced(
        fit_lda, df, "text",
        n_topics=3, vocab_size=50,
        max_iter=3, n_restarts=2, seed=0,
        memory_guard=mg, mg=mg,
    )
    peak_mb = r["peak_bytes"] / 1e6
    assert peak_mb < budget_mb * 0.8, (
        f"LDA fit peak {peak_mb:.1f}MB > 80% of {budget_mb}MB budget "
        f"(peak_bytes={r['peak_bytes']}). "
        f"Re-check doc_sample bounding in cadw/ml/lda.py."
    )
