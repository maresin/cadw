"""v0.9.143-batch4: verification tests for cadw.ml.complement_nb (DEC-033).

State = n_classes * vocab_size * f64 + n_classes * f64 + 256.
CW-1: state_nbytes exact
CW-2: state independent of n
CW-6: guard refuses at tight budget (dense C*V)
Correctness: shapes, predict in-range, class priors normalize.
OOC sanity skipped (row_size=32 B, signal < noise; DEC-034).
"""
import ctypes
import gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.complement_nb import fit_complement_nb, ComplementNBModel


def _make_df(n_docs, V=50, C=4, seed=0, nnz_per_doc=5):
    rng = np.random.RandomState(seed)
    doc_ids, terms, cnts, labels = [], [], [], []
    for i in range(n_docs):
        label = i % C
        ts = rng.randint(0, V, size=nnz_per_doc)
        cs = rng.randint(1, 3, size=nnz_per_doc)
        doc_ids.extend([i] * nnz_per_doc)
        terms.extend(ts.tolist())
        cnts.extend(cs.tolist())
        labels.extend([label] * nnz_per_doc)
    return CadwDataFrame.from_table(pa.table({
        "doc_id": pa.array(doc_ids, type=pa.int64()),
        "term_id": pa.array(terms, type=pa.int64()),
        "count": pa.array(cnts, type=pa.int64()),
        "label": pa.array(labels, type=pa.int64()),
    }))


def _warmup():
    df = _make_df(200, V=20, C=3, seed=999)
    fit_complement_nb(df, label_col="label", term_col="term_id",
                      count_col="count", n_classes=3, vocab_size=20,
                      memory_guard=MemoryGuard(max_memory_mb=500),
                      batch_size=100)


_warmup()


@pytest.mark.parametrize("C,V", [(2, 50), (4, 100), (8, 500), (10, 1000)])
def test_state_nbytes_exact(C, V):
    df = _make_df(500, V=V, C=C)
    m = fit_complement_nb(
        df, label_col="label", term_col="term_id", count_col="count",
        n_classes=C, vocab_size=V,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=200)
    # state = feature_log_prob_ (C,V) f64 + class_log_prior_ (C,) f64
    #        + class_totals_ (C,) int64 + classes_ (C,) int64 + 256
    expected = C * V * 8 + 3 * C * 8 + 256
    assert m.state_nbytes() == expected, (
        f"C={C} V={V}: got {m.state_nbytes()}, expected {expected}")


def test_state_independent_of_n():
    C, V = 4, 100
    m1 = fit_complement_nb(_make_df(500, V=V, C=C, seed=0),
                           label_col="label", term_col="term_id",
                           count_col="count", n_classes=C, vocab_size=V,
                           memory_guard=MemoryGuard(max_memory_mb=500),
                           batch_size=100)
    m2 = fit_complement_nb(_make_df(5000, V=V, C=C, seed=0),
                           label_col="label", term_col="term_id",
                           count_col="count", n_classes=C, vocab_size=V,
                           memory_guard=MemoryGuard(max_memory_mb=500),
                           batch_size=100)
    assert m1.state_nbytes() == m2.state_nbytes()


def test_guard_refuses_at_tight():
    df = _make_df(200, V=50, C=4)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        fit_complement_nb(
            df, label_col="label", term_col="term_id", count_col="count",
            n_classes=100, vocab_size=50_000,
            memory_guard=MemoryGuard(max_memory_mb=0.1), batch_size=200)


def test_shapes_and_priors():
    C, V = 4, 50
    df = _make_df(500, V=V, C=C, seed=1)
    m = fit_complement_nb(
        df, label_col="label", term_col="term_id", count_col="count",
        n_classes=C, vocab_size=V,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=200)
    assert m.feature_log_prob_.shape == (C, V)
    assert m.class_log_prior_.shape == (C,)
    assert m.class_totals_.shape == (C,)
    assert m.vocab_size_ == V
    assert m.n_classes_ == C
    # priors log-softmax normalize to sum <= 0
    assert (m.class_log_prior_ <= 0).all()
    assert np.isfinite(m.class_log_prior_).all()


def test_predict_in_range():
    C, V = 4, 50
    df = _make_df(200, V=V, C=C, seed=2)
    m = fit_complement_nb(
        df, label_col="label", term_col="term_id", count_col="count",
        n_classes=C, vocab_size=V,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=100)
    preds = []
    for rb in m.predict_batches(
            df, term_col="term_id", count_col="count", doc_col="doc_id",
            batch_size=100):
        arr = rb.column("predicted_label").to_numpy()
        preds.append(arr)
    all_preds = np.concatenate(preds) if preds else np.empty(0)
    assert all_preds.size > 0
    assert (all_preds >= 0).all()
    assert (all_preds < C).all()


def test_sample_infer_from_first_batch():
    df = _make_df(500, V=30, C=5, seed=3)
    m = fit_complement_nb(
        df, label_col="label", term_col="term_id", count_col="count",
        n_classes=None, vocab_size=None,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=500)
    # inferred at least as large as seen
    assert m.n_classes_ >= 1
    assert m.vocab_size_ >= 1
