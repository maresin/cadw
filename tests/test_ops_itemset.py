"""v0.9.149: tests for cadw.ops.itemset (P1 L8: apriori + eclat)."""
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.itemset import (
    support_count, candidate_generation, apriori, eclat,
)


def _key(items):
    """Canonical tuple key for an itemset (matches tuple(sorted(s)))."""
    return tuple(sorted(items))


def _market_basket():
    """Classic example: 5 transactions."""
    return [
        {"milk", "bread"},
        {"milk", "bread", "eggs"},
        {"milk", "eggs"},
        {"bread", "butter"},
        {"milk", "bread", "butter"},
    ]


# ============================================================================
# support_count
# ============================================================================

def test_support_count_basic():
    tx = _market_basket()
    assert support_count(tx, ["milk"]) == 4
    assert support_count(tx, ["bread"]) == 4
    assert support_count(tx, ["milk", "bread"]) == 3
    assert support_count(tx, ["milk", "eggs"]) == 2
    assert support_count(tx, ["milk", "bread", "eggs"]) == 1


def test_support_count_empty_itemset():
    tx = _market_basket()
    # empty subset is subset of every set
    assert support_count(tx, []) == len(tx)


def test_support_count_replayable_factory():
    """Callable factory gets invoked each time."""
    calls = [0]
    def factory():
        calls[0] += 1
        return iter(_market_basket())
    result = support_count(factory, ["milk"])
    assert result == 4
    assert calls[0] == 1


# ============================================================================
# candidate_generation
# ============================================================================

def test_candidate_generation_classic():
    """Apriori-gen for F_1 = {a, b, c, d, e} -> all 2-subsets (all subsets present)."""
    F_1 = [frozenset([c]) for c in "abcde"]
    C_2 = candidate_generation(F_1)
    assert len(C_2) == 10  # C(5,2)
    assert all(len(c) == 2 for c in C_2)


def test_candidate_generation_prune():
    """If a 2-subset is missing, join result is pruned."""
    # F_2 = {ab, ac, bc, bd, cd} — join abc (from ab+ac), but bcd requires
    # {bc, bd, cd} — all present. abd requires {ab, ad, bd} — ad missing -> prune.
    F_2 = [frozenset("ab"), frozenset("ac"), frozenset("bc"),
           frozenset("bd"), frozenset("cd")]
    C_3 = candidate_generation(F_2)
    sets = {tuple(sorted(s)) for s in C_3}
    assert ("a", "b", "c") in sets  # ab+ac -> abc, pruned? {ab,ac,bc} present
    assert ("b", "c", "d") in sets  # bc+bd -> bcd, {bc,bd,cd} present
    # abd not in F_2, so (a,b,d) should NOT appear
    assert ("a", "b", "d") not in sets


def test_candidate_generation_empty():
    assert candidate_generation([]) == []


def test_candidate_generation_size_mismatch():
    with pytest.raises(ValueError, match="same size"):
        candidate_generation([frozenset("ab"), frozenset("abc")])


# ============================================================================
# apriori
# ============================================================================

def test_apriori_classic():
    """Market basket with min_support=2."""
    result = apriori(_market_basket, min_support=2)
    result_dict = {tuple(sorted(s)): n for s, n in result}
    # 1-itemsets: milk(4), bread(4), eggs(2), butter(2)
    assert result_dict[("milk",)] == 4
    assert result_dict[("bread",)] == 4
    assert result_dict[("eggs",)] == 2
    assert result_dict[("butter",)] == 2
    # 2-itemsets (keys are tuple(sorted(s)))
    assert result_dict[_key(["bread", "milk"])] == 3
    assert result_dict[_key(["milk", "eggs"])] == 2
    assert result_dict[_key(["bread", "butter"])] == 2


def test_apriori_matches_eclat():
    """Apriori and Eclat must produce identical (set, count) results."""
    result_a = apriori(_market_basket, min_support=2)
    result_e = eclat(_market_basket, min_support=2)
    d_a = {tuple(sorted(s)): n for s, n in result_a}
    d_e = {tuple(sorted(s)): n for s, n in result_e}
    assert d_a == d_e, f"apriori={d_a}, eclat={d_e}"


def test_apriori_max_len():
    result = apriori(_market_basket, min_support=2, max_len=1)
    assert all(len(s) == 1 for s, _ in result)


def test_apriori_invalid_support():
    with pytest.raises(ValueError, match="min_support"):
        apriori(_market_basket, min_support=0)


def test_apriori_guard_refuses():
    """Artificial: many itemsets > tiny guard."""
    tx = [{f"i{j}" for j in range(50)} for _ in range(100)]
    with pytest.raises(MemoryLimitExceeded):
        apriori(lambda: iter(tx), min_support=100,
                memory_guard=MemoryGuard(max_memory_mb=0.001))


# ============================================================================
# eclat
# ============================================================================

def test_eclat_classic():
    result = eclat(_market_basket(), min_support=2)
    d = {tuple(sorted(s)): n for s, n in result}
    assert d[("milk",)] == 4
    assert d[("bread",)] == 4
    assert d[_key(["bread", "milk"])] == 3
    assert d[_key(["milk", "eggs"])] == 2


def test_eclat_max_len():
    result = eclat(_market_basket(), min_support=2, max_len=1)
    assert all(len(s) == 1 for s, _ in result)


def test_eclat_invalid_support():
    with pytest.raises(ValueError, match="min_support"):
        eclat(_market_basket(), min_support=0)


# ============================================================================
# larger random cross-check
# ============================================================================

def test_apriori_eclat_random_crosscheck():
    """Random transactions: Apriori and Eclat agree."""
    import random
    rng = random.Random(42)
    items = list("abcdef")
    transactions = []
    for _ in range(200):
        k = rng.randint(1, 4)
        transactions.append(set(rng.sample(items, k)))
    min_support = 20
    a = apriori(lambda: iter(transactions), min_support=min_support)
    e = eclat(transactions, min_support=min_support)
    d_a = {tuple(sorted(s)): n for s, n in a}
    d_e = {tuple(sorted(s)): n for s, n in e}
    assert d_a == d_e
