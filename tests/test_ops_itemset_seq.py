"""v0.9.150: tests for fp_growth, prefix_span, gsp."""
import random

import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.itemset import (
    apriori, fp_growth, prefix_span, gsp,
)


def _market_basket():
    return [
        {"milk", "bread"},
        {"milk", "bread", "eggs"},
        {"milk", "eggs"},
        {"bread", "butter"},
        {"milk", "bread", "butter"},
    ]


def _key(items):
    return tuple(sorted(items))


# ============================================================================
# fp_growth
# ============================================================================

def test_fp_growth_matches_apriori():
    min_sup = 2
    a = apriori(_market_basket, min_support=min_sup)
    f = fp_growth(_market_basket, min_support=min_sup)
    d_a = {_key(s): n for s, n in a}
    d_f = {_key(s): n for s, n in f}
    assert d_a == d_f, f"apriori={d_a}\nfp_growth={d_f}"


def test_fp_growth_random_crosscheck():
    rng = random.Random(7)
    items = list("abcdefgh")
    tx = [set(rng.sample(items, rng.randint(1, 4))) for _ in range(300)]
    min_sup = 30
    a = apriori(lambda: iter(tx), min_support=min_sup)
    f = fp_growth(lambda: iter(tx), min_support=min_sup)
    assert {_key(s): n for s, n in a} == {_key(s): n for s, n in f}


def test_fp_growth_max_len():
    result = fp_growth(_market_basket, min_support=2, max_len=1)
    assert all(len(s) == 1 for s, _ in result)


def test_fp_growth_invalid_support():
    with pytest.raises(ValueError, match="min_support"):
        fp_growth(_market_basket, min_support=0)


def test_fp_growth_guard_refuses():
    tx = [{f"i{j}" for j in range(50)} for _ in range(100)]
    with pytest.raises(MemoryLimitExceeded):
        fp_growth(lambda: iter(tx), min_support=100,
                  memory_guard=MemoryGuard(max_memory_mb=0.001))


# ============================================================================
# prefix_span
# ============================================================================

def test_prefix_span_simple():
    # sequences: ['a','b','c'], ['a','c'], ['b','c']
    seqs = [
        ["a", "b", "c"],
        ["a", "c"],
        ["b", "c"],
    ]
    result = prefix_span(seqs, min_support=2)
    d = {tuple(s): n for s, n in result}
    # singletons
    assert d[("a",)] == 2
    assert d[("b",)] == 2
    assert d[("c",)] == 3
    # 2-items
    assert d[("a", "c")] == 2  # ['a','b','c'] and ['a','c']
    assert d[("b", "c")] == 2  # ['a','b','c'] and ['b','c']
    # 3-items: ('a','b','c') appears once -> below min_support=2 -> absent
    assert ("a", "b", "c") not in d


def test_prefix_span_max_len():
    seqs = [["a", "b", "c"]] * 5
    result = prefix_span(seqs, min_support=3, max_len=1)
    assert all(len(s) == 1 for s, _ in result)


def test_prefix_span_invalid():
    with pytest.raises(ValueError, match="min_support"):
        prefix_span([["a"]], min_support=0)


# ============================================================================
# gsp
# ============================================================================

def test_gsp_matches_prefix_span_simple():
    seqs = [
        ["a", "b", "c"],
        ["a", "c"],
        ["b", "c"],
        ["a", "b"],
    ]
    min_sup = 2
    g = gsp(lambda: iter(seqs), min_support=min_sup)
    p = prefix_span(seqs, min_support=min_sup)
    # Both produce all frequent subsequences; sets must match
    d_g = {tuple(s): n for s, n in g}
    d_p = {tuple(s): n for s, n in p}
    assert d_g == d_p, f"gsp={d_g}\nprefix_span={d_p}"


def test_gsp_random_crosscheck_prefix_span():
    rng = random.Random(11)
    items = list("abcd")
    seqs = [[rng.choice(items) for _ in range(rng.randint(2, 5))]
            for _ in range(200)]
    min_sup = 20
    g = gsp(lambda: iter(seqs), min_support=min_sup)
    p = prefix_span(seqs, min_support=min_sup)
    assert {tuple(s): n for s, n in g} == {tuple(s): n for s, n in p}


def test_gsp_max_len():
    seqs = [["a", "b", "c"]] * 5
    result = gsp(lambda: iter(seqs), min_support=3, max_len=1)
    assert all(len(s) == 1 for s, _ in result)


def test_gsp_invalid():
    with pytest.raises(ValueError, match="min_support"):
        gsp([["a"]], min_support=0)
