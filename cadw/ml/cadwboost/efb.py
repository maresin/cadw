"""Exclusive Feature Bundling (EFB) — LightGBM technique.

Bundle rarely-co-occurring features into single feature.

When: sparse data with many mutually exclusive features (e.g., one-hot
encoded categoricals). Effective d drops from original to n_bundles.

Discovery:
1. For each pair of features, count rows where both non-zero.
   Cost O(n × d²) or sample-based O(sample × d²).
2. Greedy graph coloring: sort features by degree, assign to first
   bundle where no conflict (no co-occurrence with existing members).
3. Bundle = single feature where each original is placed in disjoint
   value range. E.g. bundles of {A, B} → codes: A uses [0, n_bins_A),
   B uses [n_bins_A, n_bins_A + n_bins_B).

Costs: O(d²) for discovery, acceptable for d ≤ 500.

Constraint: threshold split logic needs multi-range check. For split
at threshold t: go_left if codes < t. But codes span multiple features'
ranges, so semantic is "if feature A bin < a OR feature B bin < b"...
that breaks monotonicity.

Solution (LightGBM): each bundle produces TWO codes per split point
(one per feature). At split time, if bundle has k members, evaluate k
splits (one per feature's range). Still faster than k separate features
when membership is sparse.

Simplified: for prototype, treat EFB as a *pre-processing* step:
- Detect bundles (co-occurrence graph)
- For each bundle, compute "bundle code" per row:
    bundle_code = feature_a_code if a non-zero else feature_b_code + offset
- Pre-bin to n_bins_total = sum(n_bins)
- Tree sees ONE feature instead of k

For non-exclusive rows (rare overlap), fallback: use feature_a (first).
Precision loss on overlap, but EFB assumes exclusive by design.

Usage:
    model = CADWBoost(..., efb=True)
    model.fit(X, y)  # internally: EFB discovery + bundle codes
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Bundle:
    """Group of features that can share a bundle feature."""
    features: list[int]           # original feature indices
    n_bins: list[int]             # bins per original feature (before merge)
    offsets: list[int]            # start offsets in bundle codes
    total_bins: int               # sum(n_bins)


def _co_occurrence_matrix(
    codes: np.ndarray, sample_rows: int = 5000, rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Count (d, d) matrix of rows where both features non-zero.

    Non-zero means code != 0 (assuming 0 = "missing/zero" after binning).
    Sample-based for large n.
    """
    n, d = codes.shape
    if rng is None:
        rng = np.random.RandomState(0)
    if n > sample_rows:
        idx = rng.choice(n, sample_rows, replace=False)
        codes = codes[idx]
        n = sample_rows

    co = np.zeros((d, d), dtype=np.int64)
    nonzero = codes != 0  # (n, d)
    # For each pair (i, j): sum(nonzero[:,i] & nonzero[:,j])
    # Vectorized: nonzero.T @ nonzero → (d, d)
    co = (nonzero.T.astype(np.int64) @ nonzero.astype(np.int64))
    np.fill_diagonal(co, 0)   # ignore self
    return co


def _greedy_bundle(
    co_occ: np.ndarray, max_conflicts: int = 0,
) -> list[Bundle]:
    """Greedy graph coloring into bundles.

    max_conflicts: allowed co-occurrences (0 = strictly exclusive).
    """
    d = co_occ.shape[0]
    # Sort features by degree (ascending — less-connected first, easier to bundle)
    degree = (co_occ > max_conflicts).sum(axis=1)
    order = np.argsort(degree)

    assigned = [-1] * d
    bundles: list[list[int]] = []

    for f in order:
        placed = False
        for b_idx, members in enumerate(bundles):
            # Can f join bundle b? Check conflicts with all members
            can_join = all(co_occ[f, m] <= max_conflicts for m in members)
            if can_join:
                bundles[b_idx].append(int(f))
                assigned[f] = b_idx
                placed = True
                break
        if not placed:
            bundles.append([int(f)])
            assigned[f] = len(bundles) - 1

    return bundles


def build_bundles(
    codes: np.ndarray,
    n_bins_per_feature: np.ndarray,
    *, sample_rows: int = 5000, max_conflicts: int = 0,
    rng: Optional[np.random.RandomState] = None,
) -> list[Bundle]:
    """Detect bundles from co-occurrence graph."""
    co = _co_occurrence_matrix(codes, sample_rows, rng)
    raw_bundles = _greedy_bundle(co, max_conflicts)

    bundles = []
    for members in raw_bundles:
        n_bins = [int(n_bins_per_feature[f]) for f in members]
        offsets = [0]
        for nb in n_bins[:-1]:
            offsets.append(offsets[-1] + nb)
        total = sum(n_bins)
        bundles.append(Bundle(
            features=members, n_bins=n_bins,
            offsets=offsets, total_bins=total,
        ))
    return bundles


def apply_bundles(
    codes: np.ndarray, bundles: list[Bundle],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge codes per bundle. Returns (new_codes (n, n_bundles),
    new_n_bins_per_feature)."""
    n = codes.shape[0]
    n_bundles = len(bundles)
    # Max total_bins determines uint16 vs uint8
    max_total = max(b.total_bins for b in bundles)
    dtype = np.uint16 if max_total > 256 else np.uint8
    new_codes = np.zeros((n, n_bundles), dtype=dtype)
    new_n_bins = np.zeros(n_bundles, dtype=np.uint16)

    for bi, bundle in enumerate(bundles):
        for f, offset, nb in zip(bundle.features, bundle.offsets,
                                  bundle.n_bins):
            # For rows where feature f is "active" (nonzero in original sense)
            # Bundle code = offset + original_code, else unchanged
            mask = codes[:, f] > 0  # active
            new_codes[mask, bi] = offset + codes[mask, f]
        new_n_bins[bi] = bundle.total_bins

    return new_codes, new_n_bins
