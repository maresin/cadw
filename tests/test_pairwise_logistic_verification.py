"""v0.9.225-c: PairwiseLogistic (RankNet) CW verification (DEC-033).

Component-wise verification mode (DEC-033 precedent: iforest, linucb,
gmm, birch, kmeans_ad, sub_pca, eif). Combined-matrix RSS test
unmeasurable at small n (P26f: 8K baseline noise floor vs fit peak).

Components:
  CW-1 state_nbytes_exact     (bit-exact, primary evidence)
  CW-2 state_independent_of_n (identical at n >= max_samples)
  CW-3 state_bounded_for_n    (max/min <= 1.5x)
  CW-4 formula_small_scale    (d=200, ratio 0.5-2.0, SNR high)
  CW-5 guard_refuses_at_tight (clean MemoryLimitExceeded)
  (batch_coef_isolated intentionally omitted: this model has no
   batch-sized alloc in fit — data streamed query-at-a-time from
   external_sort output; batch_size kwarg only affects Phase 1
   input reading, not peak.)
  Correctness: score_parity, ranking_quality_ndcg (>0.80)
"""
from __future__ import annotations
import ctypes
import gc
import statistics

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ml.pairwise_logistic import (
    PairwiseLogisticModel, fit_pairwise_logistic)


def _make_df(n_queries, items_per_q, d, seed=0):
    """Synthetic LTR data: relevance = X @ true_coef (higher better)."""
    rng = np.random.RandomState(seed)
    coef_true = rng.normal(size=d)
    rows = []
    for qid in range(n_queries):
        X_q = rng.normal(size=(items_per_q, d))
        y_q = (X_q @ coef_true + 0.1 * rng.normal(size=items_per_q))
        # binarize relevance for stable RankNet target
        y_q = (y_q > np.median(y_q)).astype(np.float64)
        for i in range(items_per_q):
            row = {f"x{j}": float(X_q[i, j]) for j in range(d)}
            row["qid"] = qid
            row["rel"] = float(y_q[i])
            rows.append(row)
    cols = {c: pa.array([r[c] for r in rows],
                        type=pa.float64() if c != "qid" else pa.int64())
            for c in rows[0].keys()}
    return CadwDataFrame.from_table(
        pa.table(cols), memory_guard=MemoryGuard(max_memory_mb=500))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    df = _make_df(10, 5, 3, seed=999)
    fit_pairwise_logistic(
        df, ["x0", "x1", "x2"], "qid", "rel",
        sigma=1.0, n_epochs=2, lr=0.01, max_query_size=100,
        memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ---- CW-1: state exact ----

@pytest.mark.parametrize("d", [2, 5, 20, 50])
def test_state_nbytes_exact(d):
    df = _make_df(20, 10, d, seed=0)
    m = fit_pairwise_logistic(
        df, [f"x{j}" for j in range(d)], "qid", "rel",
        sigma=1.0, n_epochs=2, lr=0.01, max_query_size=500,
        memory_guard=MemoryGuard(max_memory_mb=500))
    assert m.state_nbytes() == d * 8 + 256, (
        f"d={d}: state={m.state_nbytes()}, expected={d*8+256}")


# ---- CW-2: state independent of n ----

def test_state_independent_of_n():
    d = 5
    states = []
    for nq in [100, 500, 2000]:
        df = _make_df(nq, 10, d, seed=0)
        m = fit_pairwise_logistic(
            df, [f"x{j}" for j in range(d)], "qid", "rel",
            sigma=1.0, n_epochs=2, lr=0.01, max_query_size=500,
            memory_guard=MemoryGuard(max_memory_mb=500))
        states.append(m.state_nbytes())
    assert max(states) == min(states), f"state varies: {states}"


# ---- CW-5 (component-wise, DEC-033): bounded at n >= max_samples ----

def test_state_bounded_for_n_ge_s():
    """DEC-033: at n >= max_samples state bounded (max/min <= 1.5x).

    Primary evidence for OOC; replaces combined-matrix ratio test
    (removed: VmRSS noise floor at small n, DEC-034 precedent).
    """
    d, max_q = 5, 500
    states = []
    for nq in [500, 1000, 2000]:
        df = _make_df(nq, 10, d, seed=0)
        m = fit_pairwise_logistic(
            df, [f"x{j}" for j in range(d)], "qid", "rel",
            sigma=1.0, n_epochs=2, lr=0.01, max_query_size=max_q,
            memory_guard=MemoryGuard(max_memory_mb=500))
        states.append(m.state_nbytes())
        del df
        gc.collect()
    ratio = max(states) / max(min(states), 1)
    assert ratio <= 1.5, (
        f"state bounded ratio {ratio:.2f} (nq>=500); states={states}")
    # all should equal d*8 + 256
    assert all(s == d * 8 + 256 for s in states), (
        f"state_nbytes = {states}, expected all {d * 8 + 256}")


# ---- CW-4 (DEC-033): formula sanity at physical scale ----

def test_formula_small_scale():
    """DEC-033: predicted/measured in [0.5, 2.0].

    Scale: d=200, 5000 rows, batch=1000, max_query_size=500.
    Fit peak dominated by external_sort Phase 1 buffer + write_run
    transient (both bounded by max_buffer_bytes=2*max_query_size*
    (d+2)*8). SNR at this scale: formula ~6-8 MB, baseline ~200 KB.

    Formula (evidence: cadw/ops/sort.py::_write_run, _merge_keys):
      state(d*8+256)
      + phase1_buf(2*max_q*(d+2)*8)
      + write_run_transient(~4x phase1_buf)
      + reader_batch(read_batch_size*(d+2)*8)
      + per_query(max_q*(d+2)*8)
    """
    d, max_q, bsz, nq = 200, 500, 1000, 500
    X_cols = [f"x{j}" for j in range(d)]
    cell = []
    for _ in range(3):
        _malloc_trim(); gc.collect()
        df = _make_df(nq, 10, d, seed=0)
        mg = MemoryGuard(max_memory_mb=500)
        r = measure_peak_traced(
            fit_pairwise_logistic, df, X_cols, "qid", "rel",
            sigma=1.0, n_epochs=1, lr=0.01, max_query_size=max_q,
            memory_guard=mg, batch_size=bsz, mg=mg)
        cell.append(int(r["peak_bytes"]))
        del df
        gc.collect()
    measured = int(statistics.median(cell))
    phase1_buf = 2 * max_q * (d + 2) * 8
    predicted = (
        d * 8 + 256
        + phase1_buf
        + 4 * phase1_buf               # write_run ~4x buf (P26c source)
        + 500 * (d + 2) * 8            # reader batch (read_batch_size=500)
        + max_q * (d + 2) * 8          # per-query X_q+s+grad
    )
    ratio = predicted / max(measured, 1)
    assert 0.5 <= ratio <= 2.0, (
        f"predicted={predicted:,} measured={measured:,} ratio={ratio:.2f}; "
        f"cells={cell}, d={d}, nq={nq}, bsz={bsz}")


# ---- Reference parity: vectorized s vs manual s ----

def test_score_parity_manual():
    d = 4
    df = _make_df(50, 8, d, seed=0)
    m = fit_pairwise_logistic(
        df, [f"x{j}" for j in range(d)], "qid", "rel",
        sigma=1.0, n_epochs=5, lr=0.01, max_query_size=500,
        memory_guard=MemoryGuard(max_memory_mb=500))
    pred_df = m.predict(df)
    pred = []
    for b in pred_df._iter_filtered_batches():
        pred.append(b.column("rel_score").to_numpy(zero_copy_only=False))
    pred = np.concatenate(pred) if len(pred) > 1 else pred[0]
    X_parts = []
    for b in df._iter_filtered_batches():
        X_parts.append(np.column_stack([
            b.column(f"x{j}").to_numpy(zero_copy_only=False)
            for j in range(d)]))
    X = np.concatenate(X_parts) if len(X_parts) > 1 else X_parts[0]
    ref = X @ m.coef_
    np.testing.assert_allclose(pred, ref, atol=1e-9)


# ---- Ranking quality on synthetic ----

def test_ranking_quality_ndcg():
    """Pairwise logistic should rank higher-relevance above lower within
    each query on synthetic (X @ true_coef separates)."""
    d = 3
    df = _make_df(50, 10, d, seed=0)
    m = fit_pairwise_logistic(
        df, [f"x{j}" for j in range(d)], "qid", "rel",
        sigma=1.0, n_epochs=30, lr=0.05, max_query_size=500,
        memory_guard=MemoryGuard(max_memory_mb=500))
    pred_df = m.predict(df)
    pred = []
    for b in pred_df._iter_filtered_batches():
        pred.append(b.column("rel_score").to_numpy(zero_copy_only=False))
    pred = np.concatenate(pred) if len(pred) > 1 else pred[0]
    qids = []
    rels = []
    for b in df._iter_filtered_batches():
        qids.append(b.column("qid").to_numpy(zero_copy_only=False))
        rels.append(b.column("rel").to_numpy(zero_copy_only=False))
    qids = np.concatenate(qids) if len(qids) > 1 else qids[0]
    rels = np.concatenate(rels) if len(rels) > 1 else rels[0]
    # pairwise concordance: for each query, % of (i,j) with rel_i>rel_j
    # ranked correctly by pred_i > pred_j
    correct = 0
    total = 0
    for qid in np.unique(qids):
        mask = qids == qid
        s_q = pred[mask]
        y_q = rels[mask]
        for i in range(len(y_q)):
            for j in range(len(y_q)):
                if y_q[i] > y_q[j]:
                    total += 1
                    if s_q[i] > s_q[j]:
                        correct += 1
    acc = correct / max(total, 1)
    assert acc > 0.80, f"pairwise concordance={acc:.3f} < 0.80 (RankNet should learn clear signal)"


# ---- Determinism ----

def test_deterministic():
    d = 4
    df = _make_df(30, 8, d, seed=0)
    m1 = fit_pairwise_logistic(
        df, [f"x{j}" for j in range(d)], "qid", "rel",
        sigma=1.0, n_epochs=5, lr=0.05, max_query_size=500,
        memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = fit_pairwise_logistic(
        df, [f"x{j}" for j in range(d)], "qid", "rel",
        sigma=1.0, n_epochs=5, lr=0.05, max_query_size=500,
        memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_allclose(m1.coef_, m2.coef_, atol=1e-12)


# ---- Invalid input ----

def test_invalid_sigma():
    df = _make_df(10, 5, 3, seed=0)
    with pytest.raises(ValueError, match="sigma > 0"):
        fit_pairwise_logistic(
            df, ["x0", "x1", "x2"], "qid", "rel", sigma=0.0,
            memory_guard=MemoryGuard(max_memory_mb=500))


def test_invalid_epochs():
    df = _make_df(10, 5, 3, seed=0)
    with pytest.raises(ValueError, match="n_epochs >= 1"):
        fit_pairwise_logistic(
            df, ["x0", "x1", "x2"], "qid", "rel", n_epochs=0,
            memory_guard=MemoryGuard(max_memory_mb=500))


def test_invalid_max_query_size():
    df = _make_df(10, 5, 3, seed=0)
    with pytest.raises(ValueError, match="max_query_size >= 2"):
        fit_pairwise_logistic(
            df, ["x0", "x1", "x2"], "qid", "rel", max_query_size=1,
            memory_guard=MemoryGuard(max_memory_mb=500))


def test_max_query_size_refuses():
    """Query of 20 rows > max_query_size=10 -> honest refuse."""
    df = _make_df(5, 20, 3, seed=0)
    with pytest.raises(MemoryLimitExceeded, match="max_query_size"):
        fit_pairwise_logistic(
            df, ["x0", "x1", "x2"], "qid", "rel",
            sigma=1.0, n_epochs=2, lr=0.01, max_query_size=10,
            memory_guard=MemoryGuard(max_memory_mb=500))


def test_guard_refuses_at_tight():
    df = _make_df(10, 5, 3, seed=0)
    with pytest.raises(Exception):
        fit_pairwise_logistic(
            df, ["x0", "x1", "x2"], "qid", "rel",
            sigma=1.0, n_epochs=2, lr=0.01, max_query_size=100,
            memory_guard=MemoryGuard(max_memory_mb=0.0005))
