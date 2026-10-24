"""LDA memory bound regression (v0.9.42).

Before fix: peak 1319 MB @ 50k rows (linear O(N)) — doc_sample appended
in a closure that was never reset across restarts × iters. After fix:
bounded at MAX_SAMPLE_ROWS=2000, peak under 80% of 200 MB budget.
"""
import os
import numpy as np
import psutil
import pyarrow as pa

from cadw import CadwDataFrame
from cadw.ml.lda import fit_lda
from cadw.core.memory_guard import MemoryGuard


def _peak_rss_mb(fn):
    proc = psutil.Process(os.getpid())
    base = proc.memory_info().rss
    fn()
    return (proc.memory_info().rss - base) / 1e6


def _make_df(n):
    rng = np.random.RandomState(0)
    words = [f"w{i}" for i in range(50)]
    texts = [" ".join(rng.choice(words, size=8)) for _ in range(n)]
    return CadwDataFrame.from_table(pa.table({"text": texts}))


def test_lda_peak_under_budget_at_50k():
    budget_mb = 200
    mg = MemoryGuard(max_memory_mb=budget_mb)
    df = _make_df(50_000)
    p = _peak_rss_mb(lambda: fit_lda(
        df, "text", n_topics=3, vocab_size=50,
        max_iter=3, n_restarts=2, seed=0, memory_guard=mg,
    ))
    assert p < budget_mb * 0.8, (
        f"LDA fit peak {p:.1f}MB > 80% of {budget_mb}MB budget. "
        f"Re-check doc_sample bounding in cadw/ml/lda.py."
    )
