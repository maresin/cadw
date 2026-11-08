"""Itemset mining primitives (v0.9.149, P1 L8, DEC-056).

- support_count: transactions containing a given itemset
- candidate_generation: Apriori F_k -> C_{k+1} (join + prune)
- apriori: level-wise frequent itemset mining
- eclat: vertical tidset intersection

v0.9.150: added fp_growth, prefix_span, gsp (L8 complete).

Input: `transactions_factory` is either:
  - callable returning Iterator[Iterable[Hashable]] (preferred — replayable
    for Apriori's k passes), or
  - Iterable[Iterable[Hashable]] (single-use; safe for single-pass Eclat).

For df input: build iterable via
  (df._iter_filtered_batches() -> transform into per-row item lists).
See YAML R_association cards for usage patterns.
"""
from __future__ import annotations

from typing import Callable, Hashable, Iterable, Iterator

import numpy as np


_HEADER_BYTES = 256


def _iter_transactions(transactions_factory):
    """Yield a fresh iterator of transactions."""
    if callable(transactions_factory):
        return iter(transactions_factory())
    return iter(transactions_factory)


def _single_pass_counts(transactions_factory) -> dict:
    """Count item -> number of transactions containing item (one pass)."""
    counts: dict = {}
    for tx in _iter_transactions(transactions_factory):
        for item in tx:
            counts[item] = counts.get(item, 0) + 1
    return counts


def support_count(
    transactions_factory,
    itemset,
    *,
    memory_guard=None,
) -> int:
    """Count transactions containing all items in `itemset`.

    Args:
      transactions_factory: iterable or factory (see module docstring).
      itemset: any iterable of hashable items (converted to frozenset).

    Returns:
      int — number of transactions whose itemset contains the target.

    OOC: single pass, O(1) state. Guard state = header only.
    """
    if memory_guard is not None:
        memory_guard.check_state(_HEADER_BYTES, name="itemset.support_count")
    target = frozenset(itemset)
    count = 0
    for tx in _iter_transactions(transactions_factory):
        if target.issubset(tx if isinstance(tx, (set, frozenset)) else frozenset(tx)):
            count += 1
    return count


def candidate_generation(F_k: list) -> list:
    """Apriori join + prune: F_k (frequent k-itemsets) -> C_{k+1}.

    Args:
      F_k: list of frozensets of size k (all frequent k-itemsets).

    Returns:
      list of frozensets of size k+1 — candidates for next pass.
      All (k+1)-subsets check via downward-closure; only kept if every
      size-k subset is in F_k (prune property of Apriori).

    Standard definition: Apriori-gen (Agrawal & Srikant 1994, Section 2.1.1).
    """
    F_k = [frozenset(s) for s in F_k]
    if not F_k:
        return []
    k = len(F_k[0])
    if any(len(s) != k for s in F_k):
        raise ValueError("candidate_generation: all F_k itemsets must have same size")
    F_set = set(F_k)
    # Sort each itemset for deterministic join. Two k-itemsets sharing
    # the first (k-1) items in sorted order join to (k+1)-itemset.
    sorted_F = [tuple(sorted(s)) for s in F_k]
    candidates = set()
    n = len(sorted_F)
    for i in range(n):
        for j in range(i + 1, n):
            a = sorted_F[i]; b = sorted_F[j]
            if a[:-1] == b[:-1]:
                joined = frozenset(a) | frozenset(b)
                if len(joined) == k + 1:
                    candidates.add(joined)
    # prune: every k-subset must be frequent
    def _all_subsets_frequent(s):
        items = list(s)
        for idx in range(len(items)):
            sub = frozenset(items[:idx] + items[idx+1:])
            if sub not in F_set:
                return False
        return True
    return [c for c in candidates if _all_subsets_frequent(c)]


def apriori(
    transactions_factory,
    min_support: int,
    *,
    max_len: int | None = None,
    memory_guard=None,
) -> list[tuple[frozenset, int]]:
    """Level-wise Apriori (Agrawal & Srikant 1994).

    Args:
      transactions_factory: replayable (k passes over transactions).
      min_support: absolute minimum transaction count (>= 1).
      max_len: optional max itemset size.
      memory_guard: optional.

    Returns:
      list of (frozenset, support_count), all itemsets with count >= min_support.
      Order: not guaranteed; caller should sort by (-count, itemset) if needed.

    OOC: O(1) state (candidates + counts); k + 1 passes over transactions.
    Guard state checked = candidate buffer size.
    """
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1; got {min_support}")
    counts_1 = _single_pass_counts(transactions_factory)
    frequent = {frozenset([k]): v for k, v in counts_1.items()
                if v >= min_support}
    result = list(frequent.items())
    k = 1
    while frequent:
        if max_len is not None and k >= max_len:
            break
        k += 1
        C_k = candidate_generation(list(frequent.keys()))
        if not C_k:
            break
        if memory_guard is not None:
            # one dict entry per candidate: frozenset (~300B) + count (8B)
            memory_guard.check_state(
                len(C_k) * 308 + _HEADER_BYTES, name="apriori.candidates")
        new_counts = {c: 0 for c in C_k}
        for tx in _iter_transactions(transactions_factory):
            tx_set = tx if isinstance(tx, (set, frozenset)) else frozenset(tx)
            for c in C_k:
                if c.issubset(tx_set):
                    new_counts[c] += 1
        frequent = {c: n for c, n in new_counts.items() if n >= min_support}
        result.extend(frequent.items())
    return result


def eclat(
    transactions_factory,
    min_support: int,
    *,
    max_len: int | None = None,
    memory_guard=None,
) -> list[tuple[frozenset, int]]:
    """Eclat: vertical tidset intersection (Zaki 2000).

    Args:
      transactions_factory: iterable (single pass to build tidsets).
      min_support: absolute minimum count (>= 1).
      max_len: optional max itemset size.
      memory_guard: optional.

    Returns:
      list of (frozenset, support_count), same set as apriori.

    OOC: not fully — vertical tidsets = sum of |tx| across all transactions
    (in-memory Python sets). Guard checked at build time. For very large
    basket data, prefer apriori (streaming k passes).
    """
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1; got {min_support}")

    # Build vertical tidsets: item -> set of transaction ids
    tidsets: dict = {}
    total_tids = 0
    for tid, tx in enumerate(_iter_transactions(transactions_factory)):
        for item in tx:
            s = tidsets.get(item)
            if s is None:
                s = set()
                tidsets[item] = s
            s.add(tid)
            total_tids += 1
            if memory_guard is not None and tid % 10000 == 0:
                # periodic guard: sum of |tx| * py_int + set overhead
                memory_guard.check_state(
                    total_tids * 80 + len(tidsets) * 200 + _HEADER_BYTES,
                    name="eclat.tidsets")

    if memory_guard is not None:
        memory_guard.check_state(
            total_tids * 80 + len(tidsets) * 200 + _HEADER_BYTES,
            name="eclat.tidsets.final")

    # Frequent singletons
    singletons = {frozenset([k]): len(v) for k, v in tidsets.items()
                  if len(v) >= min_support}
    result = list(singletons.items())

    # DFS: prefix items + current tidset -> extend by next items
    def _dfs(prefix: list, prefix_tidset: set, prefix_itemset: frozenset):
        if max_len is not None and len(prefix) >= max_len:
            return
        # Iterate items in sorted order; only those > last item (avoid
        # duplicate combinations)
        sorted_items = sorted(i for i in tidsets if i not in prefix)
        last = prefix[-1] if prefix else None
        for item in sorted_items:
            if last is not None and item <= last:
                continue
            inter = prefix_tidset & tidsets[item]
            if len(inter) < min_support:
                continue
            new_itemset = prefix_itemset | frozenset([item])
            result.append((new_itemset, len(inter)))
            _dfs(prefix + [item], inter, new_itemset)

    for item in sorted(tidsets):
        ts = tidsets[item]
        if len(ts) >= min_support:
            _dfs([item], ts, frozenset([item]))

    return result


# ============================================================================
# FP-growth, PrefixSpan, GSP (v0.9.150, L8 complete)
# ============================================================================

def _fp_project(itemsets, min_support, prefix, results, max_len, depth, mg):
    if depth > 100:
        raise RecursionError("fp_growth: depth > 100")
    counts = {}
    for tx in itemsets:
        for item in tx:
            if item not in prefix:
                counts[item] = counts.get(item, 0) + 1
    frequent = [i for i, c in counts.items() if c >= min_support]
    frequent.sort(key=lambda x: (counts[x], str(x)))
    if mg is not None:
        mg.check_state(len(frequent) * 200 + _HEADER_BYTES,
                       name="fp_growth.recursion")
    for item in frequent:
        new_prefix = prefix | frozenset([item])
        results.append((new_prefix, counts[item]))
        if max_len is not None and len(new_prefix) >= max_len:
            continue
        projected = []
        for tx in itemsets:
            if item in tx:
                remainder = [i for i in tx if i in counts and i != item]
                if remainder:
                    projected.append(remainder)
        if projected:
            _fp_project(projected, min_support, new_prefix, results,
                        max_len, depth + 1, mg)


def fp_growth(
    transactions_factory,
    min_support: int,
    *,
    max_len: int | None = None,
    memory_guard=None,
):
    """FP-growth via projected-DB pattern growth (Han et al. 2000).

    Same frequent itemsets as apriori; avoids FP-tree dict-of-dicts
    by growing on projected databases.
    OOC: not fully - projected DBs in memory. Guard checked per recursion.
    """
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1; got {min_support}")
    itemsets = [list(tx) for tx in _iter_transactions(transactions_factory)]
    results = []
    _fp_project(itemsets, min_support, frozenset(), results,
                max_len, 0, memory_guard)
    return results


def _is_subsequence(pattern, seq):
    i = 0
    for item in seq:
        if i < len(pattern) and pattern[i] == item:
            i += 1
    return i == len(pattern)


def prefix_span(
    sequences_factory,
    min_support: int,
    *,
    max_len=None,
    memory_guard=None,
):
    """PrefixSpan: prefix-projected DB growth (Pei et al. 2001).

    Sequences: iterable of ordered item-lists. Patterns = subsequences.
    OOC: not fully - projected DBs in memory.
    """
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1; got {min_support}")
    sequences = [list(s) for s in _iter_transactions(sequences_factory)]
    results = []

    def _mine(proj_db, prefix, depth):
        if depth > 100:
            raise RecursionError("prefix_span: depth > 100")
        counts = {}
        for seq in proj_db:
            seen = set()
            for item in seq:
                if item not in seen:
                    counts[item] = counts.get(item, 0) + 1
                    seen.add(item)
        frequent = [(i, c) for i, c in counts.items() if c >= min_support]
        if memory_guard is not None:
            memory_guard.check_state(len(frequent) * 200 + _HEADER_BYTES,
                                     name="prefix_span.recursion")
        for item, cnt in frequent:
            new_prefix = prefix + [item]
            results.append((tuple(new_prefix), cnt))
            if max_len is not None and len(new_prefix) >= max_len:
                continue
            projected = []
            for seq in proj_db:
                try:
                    idx = seq.index(item)
                except ValueError:
                    continue
                suffix = seq[idx + 1:]
                if suffix:
                    projected.append(suffix)
            if projected:
                _mine(projected, new_prefix, depth + 1)

    _mine(sequences, [], 0)
    return results


def gsp(
    sequences_factory,
    min_support: int,
    *,
    max_len=None,
    memory_guard=None,
):
    """GSP: level-wise sequential patterns (Srikant & Agrawal 1996).
    OOC: candidate set in memory.
    """
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1; got {min_support}")
    sequences = [list(s) for s in _iter_transactions(sequences_factory)]

    counts_1 = {}
    for seq in sequences:
        for item in set(seq):
            counts_1[item] = counts_1.get(item, 0) + 1
    F_prev = {tuple([item]): c for item, c in counts_1.items()
              if c >= min_support}
    result = list(F_prev.items())
    k = 1

    while F_prev:
        if max_len is not None and k >= max_len:
            break
        k += 1
        F_prev_set = set(F_prev.keys())
        candidates = set()
        for a in F_prev_set:
            for b in F_prev_set:
                if a[1:] == b[:-1]:
                    new = a + (b[-1],)
                    if len(new) == k:
                        if all(new[:i] + new[i + 1:] in F_prev_set
                               for i in range(k)):
                            candidates.add(new)
        if not candidates:
            break
        if memory_guard is not None:
            memory_guard.check_state(len(candidates) * 200 + _HEADER_BYTES,
                                     name="gsp.candidates")
        new_counts = {c: 0 for c in candidates}
        for seq in sequences:
            for c in candidates:
                if _is_subsequence(list(c), seq):
                    new_counts[c] += 1
        F_prev = {c: n for c, n in new_counts.items() if n >= min_support}
        result.extend(F_prev.items())
    return result
