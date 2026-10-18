"""Тесты RowFilter (03c)."""
import pyarrow as pa
import pytest

from cadw.dataset import (
    PredicateFilter, SliceFilter, AndFilter, OrFilter, NotFilter, MaskFilter,
)


def make_batches(n: int = 20, chunk: int = 5):
    t = pa.table({"val": list(range(n))})
    for i in range(0, n, chunk):
        yield t.slice(i, chunk).to_batches()[0]


def collect(filter_, n=20, chunk=5):
    it = filter_.iter_filtered(make_batches(n, chunk), None)
    return [b.column("val").to_pylist() for b in it]


# ---------- PredicateFilter ----------

def test_predicate_gt():
    f = PredicateFilter("val", "gt", 15)
    chunks = collect(f)
    assert chunks == [[16, 17, 18, 19]]


def test_predicate_eq():
    f = PredicateFilter("val", "eq", 7)
    assert collect(f) == [[7]]


def test_predicate_returns_empty_if_none_match():
    f = PredicateFilter("val", "gt", 100)
    assert collect(f) == []


def test_predicate_columns_required():
    assert PredicateFilter("val", "gt", 0).columns_required() == {"val"}


# ---------- SliceFilter ----------

def test_slice_basic():
    f = SliceFilter(2, 8, 1)
    chunks = collect(f)
    flat = [v for c in chunks for v in c]
    assert flat == [2, 3, 4, 5, 6, 7]


def test_slice_with_step():
    f = SliceFilter(0, 20, 3)
    flat = [v for c in collect(f) for v in c]
    assert flat == [0, 3, 6, 9, 12, 15, 18]


def test_slice_across_batches():
    f = SliceFilter(3, 12, 1)
    flat = [v for c in collect(f, n=20, chunk=4) for v in c]
    assert flat == [3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_slice_stop_beyond_end():
    f = SliceFilter(15, 100, 1)
    flat = [v for c in collect(f) for v in c]
    assert flat == [15, 16, 17, 18, 19]


def test_slice_columns_required_empty():
    assert SliceFilter(0, 10, 1).columns_required() == set()


# ---------- And/Or/Not ----------

def test_and_filter():
    f = AndFilter(
        PredicateFilter("val", "ge", 5),
        PredicateFilter("val", "lt", 10),
    )
    flat = [v for c in collect(f) for v in c]
    assert flat == [5, 6, 7, 8, 9]


def test_or_filter():
    f = OrFilter(
        PredicateFilter("val", "lt", 3),
        PredicateFilter("val", "gt", 17),
    )
    flat = [v for c in collect(f) for v in c]
    assert flat == [0, 1, 2, 18, 19]


def test_not_filter():
    f = NotFilter(PredicateFilter("val", "lt", 17))
    flat = [v for c in collect(f) for v in c]
    assert flat == [17, 18, 19]


def test_and_with_slice_raises():
    with pytest.raises(TypeError):
        PredicateFilter("val", "gt", 0) & SliceFilter(0, 10, 1)


def test_columns_required_union():
    f = AndFilter(
        PredicateFilter("a", "gt", 0),
        PredicateFilter("b", "lt", 5),
    )
    assert f.columns_required() == {"a", "b"}
