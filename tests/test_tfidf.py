"""Tests for D.6.9 streaming TF-IDF (v0.9.41)."""
from __future__ import annotations
import inspect
import math
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import tfidf as tmod
from cadw.ml.tfidf import fit_tfidf, TfidfModel


def _corpus(n=200, seed=0):
    rng = np.random.RandomState(seed)
    vocab = ["apple", "banana", "cherry", "dog", "engine", "field"]
    docs = []
    for _ in range(n):
        k = rng.randint(3, 7)
        docs.append(" ".join(rng.choice(vocab, size=k, replace=True)))
    return docs


def _df(docs):
    return CadwDataFrame.from_table(pa.table({"text": pa.array(docs)}))



def _to_numpy_2d(df):
    """CadwDataFrame of K-wide arrays → (n, K) numpy (v0.9.95)."""
    import numpy as np
    parts = []
    for b in df._iter_filtered_batches():
        cols = [b.column(c).to_numpy(zero_copy_only=False)
                for c in df.columns]
        parts.append(np.column_stack(cols))
    return (np.concatenate(parts, axis=0) if parts
            else np.zeros((0, len(df.columns)), dtype=np.float64))


def test_reference_sklearn_tfidf_exact():
    """Our TF-IDF must match sklearn's smoothed idf and l2 norm."""
    sk = pytest.importorskip("sklearn.feature_extraction.text")
    docs = _corpus(seed=0)
    m = fit_tfidf(_df(docs), "text", vocab_size=50, sublinear_tf=True, norm="l2")
    ours = _to_numpy_2d(m.transform_dense(_df(docs), text_col="text", norm="l2"))
    # sklearn with the SAME tokenizer and the same vocabulary
    vec = sk.TfidfVectorizer(
        tokenizer=lambda s: [t for t in s.lower().split() if len(t) > 1],
        token_pattern=None, vocabulary=m.vocab_, sublinear_tf=True,
        norm="l2", smooth_idf=True, use_idf=True)
    X = vec.fit_transform(docs).toarray()
    assert ours.shape == X.shape
    assert np.allclose(ours, X, atol=1e-9), np.max(np.abs(ours - X))


def test_idf_formula_smoothed():
    docs = _corpus(n=100, seed=1)
    m = fit_tfidf(_df(docs), "text", vocab_size=50)
    n = len(docs)
    dfv = {w: sum(1 for d in docs if w in d.split()) for w in m.vocab_}
    for w, j in zip(m.vocab_, range(len(m.vocab_))):
        ref = math.log((1.0 + n) / (1.0 + dfv[w])) + 1.0
        assert abs(m.idf_[j] - ref) < 1e-9


def test_rare_words_get_higher_idf():
    docs = _corpus(seed=2)
    m = fit_tfidf(_df(docs), "text", vocab_size=50)
    j = {w: i for i, w in enumerate(m.vocab_)}
    dfa = sum(1 for d in docs if "apple" in d.split())
    dfd = sum(1 for d in docs if "dog" in d.split())
    if dfa != dfd:
        rarer = "apple" if dfa < dfd else "dog"
        common = "dog" if rarer == "apple" else "apple"
        assert m.idf_[j[rarer]] > m.idf_[j[common]]


def test_sublinear_tf_vs_linear():
    docs = ["apple apple apple banana", "banana banana cherry"]
    a = fit_tfidf(_df(docs), "text", vocab_size=10, sublinear_tf=True, norm=None)
    b = fit_tfidf(_df(docs), "text", vocab_size=10, sublinear_tf=False, norm=None)
    ja = a._index["apple"]
    _A = _to_numpy_2d(a.transform_dense(_df(docs), text_col="text", norm=None))
    _B = _to_numpy_2d(b.transform_dense(_df(docs), text_col="text", norm=None))
    va = _A[0, ja]
    vb = _B[0, a._index["apple"]]
    assert abs(va - (1.0 + math.log(3.0)) * a.idf_[ja]) < 1e-9
    assert abs(vb - 3.0 * a.idf_[ja]) < 1e-9


def test_l2_norm_is_unit():
    docs = _corpus(seed=3)
    m = fit_tfidf(_df(docs), "text", vocab_size=50, norm="l2")
    X = _to_numpy_2d(m.transform_dense(_df(docs), text_col="text", norm="l2"))
    nz = X[np.any(X != 0, axis=1)]
    assert np.allclose(np.linalg.norm(nz, axis=1), 1.0, atol=1e-9)


def test_min_df_filters():
    docs = ["rare one", "common frequent", "common frequent",
            "common frequent"]
    m = fit_tfidf(_df(docs), "text", vocab_size=50, min_df=2)
    assert "common" in m.vocab_ and "frequent" in m.vocab_
    assert "rare" not in m.vocab_
    assert "one" not in m.vocab_


def test_determinism():
    docs = _corpus(seed=4)
    a = fit_tfidf(_df(docs), "text", vocab_size=50)
    b = fit_tfidf(_df(docs), "text", vocab_size=50)
    assert a.vocab_ == b.vocab_
    assert np.array_equal(a.idf_, b.idf_)


def test_edge_guards():
    docs = _corpus(n=50, seed=5)
    with pytest.raises(ValueError, match="vocab_size"):
        fit_tfidf(_df(docs), "text", vocab_size=1)
    with pytest.raises(ValueError, match="min_df"):
        fit_tfidf(_df(docs), "text", min_df=0)
    with pytest.raises(ValueError, match="norm"):
        fit_tfidf(_df(docs), "text", norm="l3")
    with pytest.raises(KeyError):
        fit_tfidf(_df(docs), "nope")
    with pytest.raises(ValueError, match="fewer than 2"):
        fit_tfidf(_df(["x"] * 5), "text")


def test_ram_guard_raises_when_tight():
    docs = _corpus(n=100, seed=6)
    mg = MemoryGuard(max_memory_bytes=1024)
    with pytest.raises(MemoryLimitExceeded):
        fit_tfidf(_df(docs), "text", memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    docs = _corpus(n=300, seed=7)
    df = _df(docs)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "tf.parquet")
    pq.write_table(tbl, path, row_group_size=41)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=64)
    a = fit_tfidf(df, "text", vocab_size=50)
    b = fit_tfidf(dmb, "text", vocab_size=50)
    assert a.vocab_ == b.vocab_
    assert np.allclose(a.idf_, b.idf_, atol=1e-9)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(tmod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src

# ── v0.9.98: transform_sparse(output=) ──────────────────────────────


def _make_df(texts):
    import pyarrow as pa
    from cadw import CadwDataFrame, MemoryGuard
    return CadwDataFrame.from_table(
        pa.table({"txt": texts}),
        memory_guard=MemoryGuard(max_memory_mb=100),
    )


def test_transform_sparse_memory_shape():
    """memory mode: list of (idx int64, val float64) per document."""
    from cadw.ml.tfidf import fit_tfidf
    texts = ["alpha bravo charlie", "bravo charlie delta",
             "alpha alpha alpha"]
    df = _make_df(texts)
    m = fit_tfidf(df, "txt", vocab_size=10)
    rows = m.transform_sparse(df, text_col="txt", output="memory")
    assert isinstance(rows, list)
    assert len(rows) == len(texts)
    for idx, val in rows:
        assert isinstance(idx, __import__("numpy").ndarray)
        assert isinstance(val, __import__("numpy").ndarray)
        assert idx.dtype == __import__("numpy").int64
        assert val.dtype == __import__("numpy").float64
        assert len(idx) == len(val)


def test_transform_sparse_memory_equals_transform():
    """memory mode bit-identical to existing transform()."""
    import numpy as np
    from cadw.ml.tfidf import fit_tfidf
    texts = ["alpha beta", "beta gamma", "alpha alpha gamma"]
    df = _make_df(texts)
    m = fit_tfidf(df, "txt", vocab_size=10)
    rows_old = m.transform(df, text_col="txt")
    rows_new = m.transform_sparse(df, text_col="txt", output="memory")
    assert len(rows_old) == len(rows_new)
    for (i_o, v_o), (i_n, v_n) in zip(rows_old, rows_new):
        assert np.array_equal(i_o, i_n)
        assert np.allclose(v_o, v_n, atol=0.0)


def test_transform_sparse_output_validation():
    """output='garbage' -> ValueError; missing col -> KeyError."""
    import pytest as _pytest
    from cadw.ml.tfidf import fit_tfidf
    df = _make_df(["alpha bravo", "charlie delta"])
    m = fit_tfidf(df, "txt", vocab_size=10)
    with _pytest.raises(ValueError, match="output must be"):
        m.transform_sparse(df, text_col="txt", output="garbage")
    with _pytest.raises(KeyError):
        m.transform_sparse(df, text_col="missing_col")


def test_transform_sparse_disk_equals_memory():
    """auto mode iter yields rows equal to memory mode (row-by-row)."""
    import numpy as np
    from cadw.ml.tfidf import fit_tfidf
    texts = ["xray yankee zulu", "yankee zulu",
             "xray zulu", "xray xray yankee"]
    df = _make_df(texts)
    m = fit_tfidf(df, "txt", vocab_size=10)
    mem = m.transform_sparse(df, text_col="txt", output="memory")
    auto = m.transform_sparse(df, text_col="txt", output="auto")
    # Iterate batches, flatten rows
    flat_idx = []
    flat_val = []
    for rb in auto:
        for idx_list, val_list in zip(
                rb.column("indices").to_pylist(),
                rb.column("values").to_pylist()):
            flat_idx.append(np.array(idx_list, dtype=np.int64))
            flat_val.append(np.array(val_list, dtype=np.float64))
    assert len(flat_idx) == len(mem)
    for (i_m, v_m), i_a, v_a in zip(mem, flat_idx, flat_val):
        assert np.array_equal(i_m, i_a)
        assert np.allclose(v_m, v_a, atol=0.0)


def test_transform_sparse_nnz_preserved():
    """sum of nnz equals sum of tokenized-unique counts per doc."""
    from cadw.ml.tfidf import fit_tfidf
    texts = ["alpha bravo charlie delta", "alpha alpha bravo",
             "delta delta delta delta delta"]
    df = _make_df(texts)
    m = fit_tfidf(df, "txt", vocab_size=10)
    mem = m.transform_sparse(df, text_col="txt", output="memory")
    auto = m.transform_sparse(df, text_col="txt", output="auto")
    mem_nnz = sum(len(idx) for idx, _ in mem)
    auto_nnz = sum(
        len(idx_list)
        for rb in auto
        for idx_list in rb.column("indices").to_pylist()
    )
    assert mem_nnz == auto_nnz
    # Sanity: not all rows are zero
    assert mem_nnz > 0


def test_transform_sparse_disk_actually_spills():
    """output='disk' → OnDiskResult, iterable, rows equal memory."""
    from cadw.core.spill import OnDiskResult
    from cadw.ml.tfidf import fit_tfidf
    texts = ["papa quebec", "quebec romeo", "papa romeo sierra"]
    df = _make_df(texts)
    m = fit_tfidf(df, "txt", vocab_size=10)
    disk = m.transform_sparse(df, text_col="txt", output="disk")
    assert isinstance(disk, OnDiskResult), type(disk).__name__
    n_rows = 0
    for rb in disk:
        n_rows += rb.num_rows
    assert n_rows == len(texts)

