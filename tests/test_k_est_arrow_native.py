"""v0.9.6c1: _estimate_k_sample Arrow-native (no pandas)."""
from __future__ import annotations

import ast
import inspect
import textwrap

import pyarrow as pa
import pytest

from cadw.core._k_est import estimate_k_sample_arrow
from cadw.ops import dedup as dd
from cadw.ops import groupby as gb


# ============ estimate_k_sample_arrow: positive cases ============


def test_k_est_single_col_basic():
    b = pa.RecordBatch.from_arrays(
        [pa.array(["a", "b", "a", "c", "b"])], names=["k"],
    )
    assert estimate_k_sample_arrow(b, ["k"]) == 3


def test_k_est_single_col_int():
    b = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 2, 3, 1, 1])], names=["k"],
    )
    assert estimate_k_sample_arrow(b, ["k"]) == 3


def test_k_est_single_col_empty():
    b = pa.RecordBatch.from_arrays(
        [pa.array([], type=pa.string())], names=["k"],
    )
    assert estimate_k_sample_arrow(b, ["k"]) == 0


def test_k_est_multi_col_basic():
    b = pa.RecordBatch.from_arrays(
        [pa.array(["a", "a", "b"]), pa.array([1, 2, 1])],
        names=["k1", "k2"],
    )
    assert estimate_k_sample_arrow(b, ["k1", "k2"]) == 3


def test_k_est_multi_col_mixed_types():
    b = pa.RecordBatch.from_arrays(
        [pa.array(["a", "a", "b"]), pa.array([1, 2, 1])],
        names=["k1", "k2"],
    )
    # (a,1), (a,2), (b,1) → 3 distinct
    assert estimate_k_sample_arrow(b, ["k1", "k2"]) == 3


def test_k_est_empty_cols_raises():
    b = pa.RecordBatch.from_arrays([pa.array([1])], names=["k"])
    with pytest.raises(ValueError, match="subset_cols empty"):
        estimate_k_sample_arrow(b, [])


# ============ groupby._estimate_k_sample Arrow-native ============


def test_groupby_k_est_single_col():
    b = pa.RecordBatch.from_arrays(
        [pa.array(["a", "b", "a", "c"])], names=["k"],
    )
    assert gb._estimate_k_sample([b], ["k"]) == 3


def test_groupby_k_est_multi_col():
    b = pa.RecordBatch.from_arrays(
        [pa.array(["a", "a", "b"]), pa.array([1, 2, 1])],
        names=["k1", "k2"],
    )
    assert gb._estimate_k_sample([b], ["k1", "k2"]) == 3


def test_groupby_k_est_multi_batch_union():
    b1 = pa.RecordBatch.from_arrays(
        [pa.array(["a", "b"])], names=["k"],
    )
    b2 = pa.RecordBatch.from_arrays(
        [pa.array(["a", "c"])], names=["k"],
    )
    # {a,b} ∪ {a,c} = {a,b,c} → 3
    assert gb._estimate_k_sample([b1, b2], ["k"]) == 3


def test_groupby_k_est_empty():
    assert gb._estimate_k_sample([], ["k"]) == 0


# ============ dedup._estimate_k_sample Arrow-native ============


def test_dedup_k_est_basic():
    b = pa.RecordBatch.from_arrays(
        [pa.array(["a", "b", "a", "c", "b"])], names=["k"],
    )
    # 3 unique in 5 rows; n_total=5 → 3
    assert dd._estimate_k_sample(b, ["k"], n_total=5) == 3


def test_dedup_k_est_extrapolate():
    b = pa.RecordBatch.from_arrays(
        [pa.array(["a", "b", "a", "c", "b"])], names=["k"],
    )
    # 3 unique / 5 rows; n_total=500 → 3 * 100 = 300
    assert dd._estimate_k_sample(b, ["k"], n_total=500) == 300


def test_dedup_k_est_empty_batch():
    b = pa.RecordBatch.from_arrays(
        [pa.array([], type=pa.string())], names=["k"],
    )
    assert dd._estimate_k_sample(b, ["k"], n_total=0) == 1


# ============ AST guards: no to_pandas in either call site ============


def test_groupby_k_est_no_to_pandas_ast():
    """v0.9.6c1: no .to_pandas() Call in groupby._estimate_k_sample."""
    from cadw.patch_utils import assert_no_call_in_source
    src = inspect.getsource(gb._estimate_k_sample)
    assert_no_call_in_source(src, "to_pandas", fn_name=
                              "groupby._estimate_k_sample")


def test_dedup_k_est_no_to_pandas_ast():
    """v0.9.6c1: no .to_pandas() Call in dedup._estimate_k_sample."""
    from cadw.patch_utils import assert_no_call_in_source
    src = inspect.getsource(dd._estimate_k_sample)
    assert_no_call_in_source(src, "to_pandas", fn_name=
                              "dedup._estimate_k_sample")


def test_groupby_k_est_uses_arrow_helper():
    src = inspect.getsource(gb._estimate_k_sample)
    # Composition of dictionary_encode or estimate_k_sample_arrow
    assert ("estimate_k_sample_arrow" in src
            or "dictionary_encode" in src)


def test_dedup_k_est_uses_arrow_helper():
    src = inspect.getsource(dd._estimate_k_sample)
    assert "estimate_k_sample_arrow" in src


def test_dedup_no_pandas_import_still_clean():
    """Belt-and-suspenders: dedup.py module has no `import pandas`."""
    tree = ast.parse(inspect.getsource(dd))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("pandas"), a.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("pandas"):
                raise AssertionError(node.module)

