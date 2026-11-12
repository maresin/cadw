"""OOC (out-of-core) fit path for CADWBoost.

True OOC: y, pred, codes all on disk (memmap). Per-batch computation.
Peak RAM = 2×batch×row_size + bounded state (independent of n).

Flow:
1. Reservoir sample X → quantile binning thresholds
2. Stream X → codes memmap (n,d) uint8 + y memmap (n,) f64
3. Fit boosting: per iteration:
   a. Per level: stream codes+pred+y, compute histograms in memory
   b. Fit tree split from histograms
   c. Stream codes+pred: update pred_mm += lr × tree(codes_batch)
4. Predict: stream codes, per-batch tree traversal

Applies to constant leaves (linear leaves need X on disk → P1).
Symmetric trees only (leaf_wise needs per-node hist → P1).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

# Hard caps for OOC — guarantees peak independent of n.
MAX_OOC_BATCH = 20_000
MAX_OOC_SAMPLE = 50_000  # fixed reservoir size for binning

import numpy as np

from .binning import compute_binning, apply_binning
from .histogram import Histogram
from .split_finder import best_split_symmetric


def _memmap(path, shape, dtype):
    return np.memmap(str(path), dtype=dtype, mode="w+", shape=shape)


def _reservoir_sample_X(df, X_cols, target_size, rng):
    """Single-pass reservoir sample of X (n × d) for binning."""
    d = len(X_cols)
    sample = np.empty((target_size, d), dtype=np.float64)
    n_seen = 0
    for b in df._iter_filtered_batches(batch_size=MAX_OOC_BATCH):
        cols = [b.column(c).to_numpy(zero_copy_only=False) for c in X_cols]
        X_b = np.column_stack(cols).astype(np.float64)
        n_b = X_b.shape[0]
        for i in range(n_b):
            n_seen += 1
            if n_seen <= target_size:
                sample[n_seen - 1] = X_b[i]
            else:
                j = int(rng.randint(n_seen))
                if j < target_size:
                    sample[j] = X_b[i]
    n_sample = min(n_seen, target_size)
    return sample[:n_sample]


def _write_codes_and_y(df, X_cols, y_col, thresholds,
                       codes_path, y_path, n_total, mg, missing_bin):
    """Stream X→codes_mm, y→y_mm in one pass. Batch size from guard."""
    d = len(X_cols)
    row_size = (d + 1) * 8
    batch_size = min(
        MAX_OOC_BATCH,
        max(1000, mg.get_batch_rows(
            row_size_bytes=row_size, expansion_factor=2.0)))
    codes_mm = _memmap(codes_path, (n_total, d), np.uint8)
    y_mm = _memmap(y_path, (n_total,), np.float64)
    i = 0
    for b in df._iter_filtered_batches(batch_size=batch_size):
        mg.check_batch(b, name="cadwboost.ooc.batch", extra_factor=2.0)
        cols = [b.column(c).to_numpy(zero_copy_only=False) for c in X_cols]
        X_b = np.column_stack(cols).astype(np.float64)
        codes_b = apply_binning(X_b, thresholds, missing_bin=missing_bin)
        codes_mm[i:i + b.num_rows] = codes_b
        y_mm[i:i + b.num_rows] = b.column(y_col).to_numpy(
            zero_copy_only=False).astype(np.float64)
        i += b.num_rows
    return codes_mm, y_mm, batch_size


def _stream_histogram(codes_mm, pred_mm, y_mm, loss, splits_so_far,
                      n_leaves, depth_level, n_bins, batch_size, mg):
    """Histogram for current level: leaf_idx from splits_so_far (per level)."""
    n, d = codes_mm.shape
    n_bins_max = max(1, int(n_bins.max()))
    G = np.zeros((n_leaves, d, n_bins_max), dtype=np.float64)
    H = np.zeros((n_leaves, d, n_bins_max), dtype=np.float64)
    size = n_leaves * n_bins_max

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        gb = np.zeros(end - start)
        # y/pred from memmap
        pred_b = np.array(pred_mm[start:end])
        y_b = np.array(y_mm[start:end])
        grad_b, hess_b = loss.grad_hess(y_b, pred_b)
        # Compute leaf_idx for this level
        codes_b = np.array(codes_mm[start:end])
        leaf_idx = np.zeros(end - start, dtype=np.int64)
        for lvl in range(depth_level):
            feat = int(splits_so_far[lvl, 0])
            if feat < 0:
                continue
            thresh = int(splits_so_far[lvl, 1])
            go_right = (codes_b[:, feat] >= thresh).astype(np.int64)
            leaf_idx |= (go_right << lvl)
        # Accumulate
        for j in range(d):
            nb = int(n_bins[j])
            if nb == 0:
                continue
            flat = leaf_idx * n_bins_max + codes_b[:, j].astype(np.int64)
            if not (codes_b[:, j] < nb).all():
                flat = leaf_idx * n_bins_max + np.clip(
                    codes_b[:, j], 0, nb - 1).astype(np.int64)
            G[:, j, :nb] += np.bincount(flat, weights=grad_b, minlength=size
                                        ).reshape(n_leaves, n_bins_max)[:, :nb]
            H[:, j, :nb] += np.bincount(flat, weights=hess_b, minlength=size
                                        ).reshape(n_leaves, n_bins_max)[:, :nb]
    return Histogram(G, H, n_bins, n_leaves)


def _compute_pred_update(model, tree, codes_mm, pred_mm, n_total, batch_size):
    """pred_mm += lr × tree(codes_mm) per-batch."""
    from .tree import _constant_leaf_values  # noqa: F401 (unused, structure)
    n, d = codes_mm.shape
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        codes_b = np.array(codes_mm[start:end])
        update_b = tree.predict(codes_b)
        pred_mm[start:end] = np.array(pred_mm[start:end]) + \
                             model.learning_rate * update_b


def _fit_symmetric_ooc(model, codes_mm, y_mm, n_total, n_bins,
                       batch_size, mg):
    """Fit symmetric boosting loop with OOC histograms."""
    from .tree import Tree, _constant_leaf_values

    d = codes_mm.shape[1]
    init = model.loss_.init_pred(np.array(y_mm[:min(n_total, 10_000)]))
    model.init_pred_ = float(init)

    # pred_mm: working prediction on disk
    pred_path = Path(str(y_mm.filename)).parent / "pred.bin"
    pred_mm = _memmap(pred_path, (n_total,), np.float64)
    pred_mm[:] = init

    trees = []
    weights = []
    rng = np.random.RandomState(model.random_state)

    for it in range(model.n_estimators):
        depth = model.max_depth
        splits = np.full((depth, 3), -1, dtype=np.int64)
        splits[:, 2] = 1
        split_gains = np.zeros(depth, dtype=np.float64)

        for lvl in range(depth):
            n_leaves = 2 ** lvl
            hist = _stream_histogram(
                codes_mm, pred_mm, y_mm, model.loss_, splits,
                n_leaves, lvl, n_bins, batch_size, mg)
            sp = best_split_symmetric(
                hist, model.reg_lambda, model.min_child_weight,
                model.min_gain, missing_bin=model.missing_bin)
            if sp.feature < 0:
                break
            splits[lvl, 0] = sp.feature
            splits[lvl, 1] = sp.threshold
            splits[lvl, 2] = 1 if sp.default_left else 0
            split_gains[lvl] = sp.gain

        # Compute leaf values: full pass over codes+pred+y at final depth
        n_leaves_final = 2 ** depth
        G_leaf = np.zeros(n_leaves_final)
        H_leaf = np.zeros(n_leaves_final)
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            codes_b = np.array(codes_mm[start:end])
            leaf_b = np.zeros(end - start, dtype=np.int64)
            for lvl in range(depth):
                feat = int(splits[lvl, 0])
                if feat < 0:
                    continue
                thresh = int(splits[lvl, 1])
                leaf_b |= ((codes_b[:, feat] >= thresh).astype(np.int64) << lvl)
            y_b = np.array(y_mm[start:end])
            pred_b = np.array(pred_mm[start:end])
            grad_b, hess_b = model.loss_.grad_hess(y_b, pred_b)
            np.add.at(G_leaf, leaf_b, grad_b)
            np.add.at(H_leaf, leaf_b, hess_b)

        values = _constant_leaf_values(
            G_leaf, H_leaf, model.reg_lambda, model.min_child_weight,
            model.max_delta_step, model.path_smoothing)
        tree = Tree(mode="symmetric", max_depth=depth, linear=False,
                    splits=splits, values=values, coefs=None,
                    split_gains=split_gains)
        trees.append(tree)
        weights.append(model.learning_rate)
        _compute_pred_update(model, tree, codes_mm, pred_mm, n_total,
                              batch_size)
        pred_mm.flush()

    return trees, weights


def fit_ooc(model, df, X_cols, y_col, memory_guard=None):
    """Full OOC fit. Peak RAM independent of n (y/codes/pred on disk)."""
    from .model import CADWBoost

    mg = memory_guard or df._memory_guard
    n_total = df._dataset.total_rows
    d = len(X_cols)
    if model.leaf_predictor == "linear":
        raise NotImplementedError(
            "OOC linear leaves not supported (needs X on disk; P1)")
    if model.tree_mode != "symmetric":
        raise NotImplementedError(
            "OOC leaf_wise not supported (needs per-node hist; P1)")

    model.loss_ = model._loss_instance()
    tmpdir = Path(tempfile.mkdtemp(prefix="cadwboost_ooc_"))

    # 1. Reservoir sample for binning
    rng = np.random.RandomState(model.random_state)
    sample_size = min(n_total, MAX_OOC_SAMPLE)
    X_sample = _reservoir_sample_X(df, X_cols, sample_size, rng)
    b = compute_binning(X_sample, model.max_bins, model.binning,
                        missing_bin=model.missing_bin)

    # 2. Stream codes + y to memmap
    codes_path = tmpdir / "codes.bin"
    y_path = tmpdir / "y.bin"
    codes_mm, y_mm, batch_size = _write_codes_and_y(
        df, X_cols, y_col, b.thresholds, codes_path, y_path, n_total,
        mg, model.missing_bin)

    # 3. Fit boosting
    trees, weights = _fit_symmetric_ooc(
        model, codes_mm, y_mm, n_total, b.n_bins_per_feature,
        batch_size, mg)

    # 4. Populate model
    model.trees_ = trees
    model.tree_weights_ = np.array(weights)
    model.thresholds_ = b.thresholds
    model.n_bins_per_feature_ = b.n_bins_per_feature
    model.has_missing_ = b.has_missing_per_feature
    model.n_features_in_ = d
    model.feature_names_ = list(X_cols)
    model._compute_importance()
    model.fitted_ = True
    return model
