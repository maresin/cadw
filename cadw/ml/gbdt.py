"""Gradient Boosting Decision Tree — regression (v0.7.3a).

Composition: Reduce (histograms) + Reduce (assign leaves) + Reduce
(leaf values). Level-wise tree growth.

Loss: L2 (regression). For classification — v0.7.3b.

Peak RAM = batch + O(n × 32B) internal state (pred, cur_leaf, grad, hess)
+ O(nodes × feat × bins) histograms. Guarded via MemoryGuard.
The O(n) state is inherent to Newton-based boosting (same as XGBoost).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.reduce import ReduceOp, reduce


# ============ helpers ============

def _stack_numeric(
    batch: pa.RecordBatch, cols: Sequence[str],
) -> np.ndarray:
    parts = [_column_to_f64(batch.column(c)) for c in cols]
    return np.column_stack(parts)


def _column_to_f64(arr: pa.Array) -> np.ndarray:
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


def _iter_with_state(
    df,
    X_cols: Sequence[str],
    y_col: str,
    pred: np.ndarray,
    leaf: np.ndarray,
):
    """Yield (batch_index, offset, X, y, grad, hess, leaf_subset).

    Batches come from df._iter_filtered_batches in row order.
    """
    offset = 0
    for bi, b in enumerate(df._iter_filtered_batches()):
        n = b.num_rows
        X = _stack_numeric(b, X_cols)
        y = _column_to_f64(b.column(y_col))
        p = pred[offset:offset + n]
        lf = leaf[offset:offset + n]
        # mask: skip rows with null X or y (they get NaN in pred)
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        # grad = pred - y (L2 loss); hess = 1
        g = np.where(valid, p - y, 0.0)
        h = np.where(valid, 1.0, 0.0)
        yield bi, offset, X, y, g, h, lf, valid
        offset += n


# ============ ReduceOps ============

class _MinMaxOp(ReduceOp):
    """Global min/max per feature (ignore nulls)."""

    def __init__(self, feature_cols: Sequence[str]) -> None:
        self.feature_cols = list(feature_cols)
        self.d = len(feature_cols)

    def init(self):
        return (np.full(self.d, np.inf), np.full(self.d, -np.inf))

    def update(self, state, batch):
        mn, mx = state
        X = _stack_numeric(batch, self.feature_cols)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        if X.shape[0] == 0:
            return state
        return (np.minimum(mn, X.min(axis=0)),
                np.maximum(mx, X.max(axis=0)))

    def merge(self, a, b):
        return (np.minimum(a[0], b[0]), np.maximum(a[1], b[1]))

    def state_nbytes(self, state):
        return int(state[0].nbytes + state[1].nbytes)

    def estimate_state_nbytes(self):
        return 8 * self.d * 2

    def finalize(self, state):
        return {"min": state[0], "max": state[1]}


class _AllNodeHistOp(ReduceOp):
    """Accumulate (grad, hess) histograms for ALL nodes in a single pass.

    State: g_hist (slots × f × bins), h_hist (same), g_tot, h_tot, n_tot.
    Slots = internal node_ids at current level (leaf-of-tree-so-far).

    Batch must have columns: features + __g + __h + __leaf.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        node_ids: Sequence[int],
        mn: np.ndarray,
        mx: np.ndarray,
        bins: int,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.f = len(feature_cols)
        self.node_ids = list(node_ids)
        self.node_to_slot = {nid: i for i, nid in enumerate(self.node_ids)}
        self.mn = np.asarray(mn, dtype=np.float64)
        self.mx = np.asarray(mx, dtype=np.float64)
        self.range = np.where(self.mx - self.mn > 0,
                              self.mx - self.mn, 1.0)
        self.bins = int(bins)
        self.S = len(self.node_ids)

    def init(self):
        return (
            np.zeros((self.S, self.f, self.bins), dtype=np.float64),
            np.zeros((self.S, self.f, self.bins), dtype=np.float64),
            np.zeros(self.S, dtype=np.float64),
            np.zeros(self.S, dtype=np.float64),
            np.zeros(self.S, dtype=np.int64),
        )

    def update(self, state, batch):
        gh, hh, gt, ht, nt = state
        X = _stack_numeric(batch, self.feature_cols)
        g = _column_to_f64(batch.column("__g"))
        h = _column_to_f64(batch.column("__h"))
        lf = batch.column("__leaf").to_numpy(zero_copy_only=False)
        n = X.shape[0]
        if n == 0:
            return state
        for nid, slot in self.node_to_slot.items():
            m = lf == nid
            if not m.any():
                continue
            Xn = X[m]
            gn = g[m]
            hn = h[m]
            # Skip rows with NaN X (should be rare — valid mask applied upstream)
            valid = ~np.isnan(Xn).any(axis=1)
            if not valid.any():
                continue
            Xn = Xn[valid]
            gn = gn[valid]
            hn = hn[valid]
            gt[slot] += gn.sum()
            ht[slot] += hn.sum()
            nt[slot] += Xn.shape[0]
            for j in range(self.f):
                b_idx = np.clip(
                    ((Xn[:, j] - self.mn[j]) / self.range[j] * self.bins)
                    .astype(np.int64),
                    0, self.bins - 1,
                )
                np.add.at(gh[slot, j], b_idx, gn)
                np.add.at(hh[slot, j], b_idx, hn)
        return (gh, hh, gt, ht, nt)

    def merge(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4])

    def state_nbytes(self, state):
        return int(sum(x.nbytes for x in state))

    def estimate_state_nbytes(self):
        per_slot = self.f * self.bins * 8 * 2 + 8 * 3
        return self.S * per_slot

    def finalize(self, state):
        gh, hh, gt, ht, nt = state
        return {
            "g_hist": gh, "h_hist": hh,
            "g_tot": gt, "h_tot": ht, "n_tot": nt,
        }


class _AssignLeavesOp(ReduceOp):
    """Assign rows to leaf_ids per current tree structure.

    Returns (n,) int64 — leaf_id for each row (0 = internal node's
    fallback for null X). Accumulates list of per-batch arrays,
    concatenated in finalize.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        mn: np.ndarray,
        mx: np.ndarray,
        bins: int,
        tree: dict,
        root_id: int = 0,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.mn = np.asarray(mn, dtype=np.float64)
        self.mx = np.asarray(mx, dtype=np.float64)
        self.range = np.where(self.mx - self.mn > 0,
                              self.mx - self.mn, 1.0)
        self.bins = int(bins)
        self.tree = dict(tree)  # node_id -> (feat_idx, bin_thr, left, right) | None
        self.root_id = int(root_id)

    def init(self):
        return []

    def update(self, state, batch):
        X = _stack_numeric(batch, self.feature_cols)
        n = X.shape[0]
        leaf = np.full(n, self.root_id, dtype=np.int64)
        # v0.7.3-fix: rows with NaN feature stay at root (never descend).
        nan_rows = np.isnan(X).any(axis=1)
        # Traverse depth-by-depth.
        max_iter = 32
        with np.errstate(divide="ignore", invalid="ignore"):
            for _ in range(max_iter):
                any_change = False
                for nid, spec in self.tree.items():
                    if spec is None:
                        continue
                    j, thr, left, right = spec
                    m = (leaf == nid) & ~nan_rows
                    if not m.any():
                        continue
                    b_idx = np.clip(
                        ((X[m, j] - self.mn[j]) / self.range[j] * self.bins)
                        .astype(np.int64),
                        0, self.bins - 1,
                    )
                    new_ids = np.where(b_idx <= thr, left, right)
                    leaf[m] = new_ids
                    any_change = True
                if not any_change:
                    break
        state.append(leaf)
        return state

    def merge(self, a, b):
        return a + b

    def state_nbytes(self, state):
        return int(sum(x.nbytes for x in state) + 8)

    def estimate_state_nbytes(self):
        # State is per-batch leaves; total bounded by n_total × 8 bytes,
        # already covered by the n×32 pre-check in fit_gbdt_regressor.
        return 0

    def finalize(self, state):
        return np.concatenate(state) if state else np.array([], dtype=np.int64)


class _LeafValuesOp(ReduceOp):
    """Per-leaf: sum_grad, sum_hess. Output dict leaf_id -> value.

    Leaf values = -sum_grad / (sum_hess + reg_lambda) (Newton step for L2).
    """

    def __init__(self, reg_lambda: float = 1.0) -> None:
        self.reg_lambda = float(reg_lambda)

    def init(self):
        return {}

    def update(self, state, batch):
        g = _column_to_f64(batch.column("__g"))
        h = _column_to_f64(batch.column("__h"))
        lf = batch.column("__leaf").to_numpy(zero_copy_only=False)
        for nid in np.unique(lf):
            m = lf == nid
            key = int(nid)
            sg, sh = state.get(key, (0.0, 0.0))
            state[key] = (sg + float(g[m].sum()), sh + float(h[m].sum()))
        return state

    def merge(self, a, b):
        result = dict(a)
        for k, (g, h) in b.items():
            og, oh = result.get(k, (0.0, 0.0))
            result[k] = (og + g, oh + h)
        return result

    def state_nbytes(self, state):
        return 16 * max(1, len(state))

    def estimate_state_nbytes(self):
        # O(leaves) bounded by 2^depth; covered by n×32 pre-check.
        return 0

    def finalize(self, state):
        return {
            k: -g / (h + self.reg_lambda)
            for k, (g, h) in state.items()
        }


# ============ model ============

@dataclass
class GBDTRegressor:
    """Fitted GBDT regression model (level-wise, L2 loss)."""

    trees_: list
    base_prediction_: float
    feature_names_: list
    target_col_: str
    mn_: np.ndarray
    mx_: np.ndarray
    bins_: int
    learning_rate_: float
    n_estimators_: int
    max_depth_: int
    n_features_in_: int
    fitted_: bool = True

    def predict(self, X):
        """Predict.

        - np.ndarray (n, d) → np.ndarray (n,)
        - CadwDataFrame → CadwDataFrame with `{target}_pred` column.
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"predict: X has {X.shape[1]} features, "
                    f"expected {self.n_features_in_}"
                )
            pred = np.full(X.shape[0], self.base_prediction_, dtype=np.float64)
            for tree in self.trees_:
                leaf = self._assign(X, tree["structure"])
                pred += self.learning_rate_ * self._leaf_vector(
                    tree["leaf_values"], leaf,
                )
            return pred

        from cadw.dataframe import CadwDataFrame
        from cadw.dataset import Dataset
        if not isinstance(X, CadwDataFrame):
            raise TypeError(
                f"predict: X must be np.ndarray or CadwDataFrame; "
                f"got {type(X).__name__}"
            )
        for c in self.feature_names_:
            if c not in X.columns:
                raise KeyError(f"predict: column {c!r} not in df")

        out_col = f"{self.target_col_}_pred"
        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        out_fields.append(pa.field(out_col, pa.float64(), nullable=True))
        out_schema = pa.schema(out_fields)
        trees = self.trees_
        lr = self.learning_rate_
        base = self.base_prediction_
        mn = self.mn_
        mx = self.mx_
        rng = np.where(mx - mn > 0, mx - mn, 1.0)
        bins = self.bins_

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, self.feature_names_)
                n = Xm.shape[0]
                # v0.7.3-fix2: rows with any NaN feature stay at leaf=0,
                # nulled in output. Avoids NaN.astype(int64) warnings.
                nan_rows = np.isnan(Xm).any(axis=1)
                pred = np.full(n, base, dtype=np.float64)
                for tree in trees:
                    leaf = np.zeros(n, dtype=np.int64)
                    for nid, spec in tree["structure"].items():
                        if spec is None:
                            continue
                        j, thr, left, right = spec
                        m = (leaf == nid) & ~nan_rows
                        if not m.any():
                            continue
                        b_idx = np.clip(
                            ((Xm[m, j] - mn[j]) / rng[j] * bins)
                            .astype(np.int64),
                            0, bins - 1,
                        )
                        leaf[m] = np.where(b_idx <= thr, left, right)
                    lv_map = tree["leaf_values"]
                    # Vector: leaf_values array indexed by leaf_id; default 0
                    max_lid = max(lv_map.keys()) if lv_map else 0
                    lv_arr = np.zeros(max_lid + 1, dtype=np.float64)
                    for lid, val in lv_map.items():
                        lv_arr[lid] = val
                    pred += lr * lv_arr[leaf]
                arrays = [b.column(c) for c in X._columns]
                # NaN X → null predictions
                nan_mask = np.isnan(Xm).any(axis=1)
                pred_with_null = pa.array(
                    pred, type=pa.float64(),
                    mask=nan_mask if nan_mask.any() else None,
                )
                arrays.append(pred_with_null)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )

    def _assign(self, X: np.ndarray, tree_struct: dict) -> np.ndarray:
        n = X.shape[0]
        leaf = np.zeros(n, dtype=np.int64)
        rng = np.where(self.mx_ - self.mn_ > 0, self.mx_ - self.mn_, 1.0)
        for nid, spec in tree_struct.items():
            if spec is None:
                continue
            j, thr, left, right = spec
            m = leaf == nid
            if not m.any():
                continue
            b_idx = np.clip(
                ((X[m, j] - self.mn_[j]) / rng[j] * self.bins_)
                .astype(np.int64), 0, self.bins_ - 1,
            )
            leaf[m] = np.where(b_idx <= thr, left, right)
        return leaf

    def _leaf_vector(self, lv_map: dict, leaf: np.ndarray) -> np.ndarray:
        if not lv_map:
            return np.zeros_like(leaf, dtype=np.float64)
        max_lid = max(lv_map.keys())
        arr = np.zeros(max_lid + 1, dtype=np.float64)
        for lid, v in lv_map.items():
            arr[lid] = v
        return arr[leaf]

    def to_dict(self) -> dict:
        return {
            "base_prediction": float(self.base_prediction_),
            "feature_names": self.feature_names_,
            "target_col": self.target_col_,
            "n_estimators": int(self.n_estimators_),
            "max_depth": int(self.max_depth_),
            "learning_rate": float(self.learning_rate_),
            "bins": int(self.bins_),
            "n_trees": len(self.trees_),
        }


# ============ fit ============

def _find_best_split(
    slot: int,
    gh: np.ndarray, hh: np.ndarray,
    gt: float, ht: float, n_tot: int,
    bins: int,
    min_child_weight: float,
    min_gain: float,
    reg_lambda: float,
) -> Optional[tuple[int, int, float]]:
    """Return (feat_idx, bin_thr, gain) or None."""
    if ht <= 1e-12:
        return None
    parent_denom = ht + reg_lambda
    parent_gain = gt * gt / parent_denom
    best = None
    F = gh.shape[0]
    for j in range(F):
        g_l = np.cumsum(gh[j])[:-1]  # (bins-1,)
        h_l = np.cumsum(hh[j])[:-1]
        g_r = gt - g_l
        h_r = ht - h_l
        # min_child_weight: hess in each child
        valid = (h_l >= min_child_weight) & (h_r >= min_child_weight)
        if not valid.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            g_l_sq = g_l * g_l / (h_l + reg_lambda)
            g_r_sq = g_r * g_r / (h_r + reg_lambda)
        gains = np.where(
            valid,
            g_l_sq + g_r_sq - parent_gain,
            -np.inf,
        )
        k = int(np.argmax(gains))
        if gains[k] > (best[2] if best is not None else min_gain):
            best = (j, k, float(gains[k]))
    return best


def fit_gbdt_regressor(
    df,
    X_cols,
    y_col: str,
    *,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    bins: int = 32,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    min_gain: float = 0.0,
    memory_guard: Optional[MemoryGuard] = None,
) -> GBDTRegressor:
    """Fit GBDT regression via level-wise Newton boosting.

    - X_cols: numeric features.
    - y_col: numeric target.
    - n_estimators, max_depth, learning_rate, bins, min_child_weight,
      reg_lambda, min_gain: hyperparams (XGBoost-style defaults).
    - memory_guard: default df._memory_guard.

    Peak RAM = batch + O(n × 32B) for pred/cur_leaf/grad/hess
    + O(nodes × feat × bins) for histograms. Guarded via check_state.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_gbdt_regressor: X_cols must be non-empty")
    if y_col in X_cols:
        raise ValueError(
            f"fit_gbdt_regressor: target {y_col!r} must not be in X_cols"
        )
    for c in X_cols + [y_col]:
        if c not in df.columns:
            raise KeyError(f"fit_gbdt_regressor: column {c!r} not in df")

    schema = df._dataset.schema
    for c in X_cols + [y_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_gbdt_regressor: column {c!r} has type {t}; "
                f"numeric required (use encode_categorical for strings)"
            )
    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1; got {n_estimators}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1; got {max_depth}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0; got {learning_rate}")
    if bins < 2:
        raise ValueError(f"bins must be >= 2; got {bins}")
    if min_child_weight < 0:
        raise ValueError(f"min_child_weight must be >= 0; got {min_child_weight}")
    if reg_lambda < 0:
        raise ValueError(f"reg_lambda must be >= 0; got {reg_lambda}")

    mg = memory_guard or df._memory_guard

    # v0.7.3-fix: n_total MUST be filtered count, not total_rows.
    # pred/cur_leaf are sized to filtered rows, because _iter_filtered_batches
    # iterates filtered batches only.
    n_total = df.count()
    if n_total == 0:
        raise ValueError(
            "fit_gbdt_regressor: no valid (non-null) rows "
            "(filtered count is 0)"
        )
    est_n_state = n_total * 32
    try:
        mg.check_state(est_n_state, name="gbdt.internal_state")
    except MemoryLimitExceeded:
        raise MemoryLimitExceeded(
            f"fit_gbdt_regressor: internal state O(n) = {est_n_state} B "
            f"({n_total} rows × 32B) > available * state_safety. "
            f"GBDT requires materializing per-row gradients (like XGBoost). "
            f"Increase budget or sample. See §13.4/92."
        )

    # 1. Global min/max
    mm_op = _MinMaxOp(X_cols)
    mm_res = reduce(df._iter_filtered_batches(), mm_op, mg, log=False)
    mn = mm_res["min"]
    mx = mm_res["max"]
    if np.any(np.isinf(mn)) or np.any(np.isinf(mx)):
        raise ValueError("fit_gbdt_regressor: no valid (non-null) rows")

    # 2. Base prediction: mean(y) over valid rows
    y_sum = 0.0
    n_valid = 0
    for bi, off, X, y, g, h, lf, valid in _iter_with_state(
        df, X_cols, y_col, np.zeros(n_total), np.zeros(n_total, dtype=np.int64),
    ):
        y_v = y[valid]
        y_sum += float(y_v.sum())
        n_valid += int(valid.sum())
    if n_valid == 0:
        raise ValueError("fit_gbdt_regressor: no valid rows")
    base_pred = y_sum / n_valid

    # 3. Init pred / leaf
    pred = np.full(n_total, base_pred, dtype=np.float64)
    # For invalid rows, keep pred (won't be updated via leaf)
    trees = []

    # 4. Boosting rounds
    max_nodes_full = (1 << (max_depth + 1)) - 1  # full binary tree nodes
    for round_i in range(n_estimators):
        # Fresh tree per round
        # node_id → (feat, bin_thr, left_id, right_id) or None (leaf)
        tree: dict = {0: None}
        cur_leaf = np.zeros(n_total, dtype=np.int64)
        next_node_id = 1

        for level in range(max_depth):
            # Internal nodes at current level (leaves of tree-so-far)
            internal = [nid for nid, t in tree.items() if t is None]
            if not internal:
                break

            # --- Pass 1: histograms for all internal nodes ---
            def _iter_with_ghl():
                for bi, off, X, y, g, h, lf, valid in _iter_with_state(
                    df, X_cols, y_col, pred, cur_leaf,
                ):
                    n = X.shape[0]
                    # Add __g, __h, __leaf columns
                    b = next(df._iter_filtered_batches()) if False else None
                    # We need the batch itself. Re-iterate.
                    yield (bi, off, X, g, h, lf)

            # Rebuild batches with state columns
            def _iter_batches_with_state():
                offset = 0
                for b in df._iter_filtered_batches():
                    n = b.num_rows
                    X = _stack_numeric(b, X_cols)
                    y = _column_to_f64(b.column(y_col))
                    p = pred[offset:offset + n]
                    lf = cur_leaf[offset:offset + n]
                    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                    g = np.where(valid, p - y, 0.0)
                    h = np.where(valid, 1.0, 0.0)
                    # null-y rows: mask pred update later via valid array
                    # Build enriched RecordBatch
                    cols = [b.column(c) for c in X_cols]
                    cols.append(pa.array(g, type=pa.float64()))
                    cols.append(pa.array(h, type=pa.float64()))
                    cols.append(pa.array(lf, type=pa.int64()))
                    names = list(X_cols) + ["__g", "__h", "__leaf"]
                    yield pa.RecordBatch.from_arrays(cols, names=names)
                    offset += n

            hist_op = _AllNodeHistOp(X_cols, internal, mn, mx, bins)
            hist_res = reduce(_iter_batches_with_state(), hist_op, mg, log=False)
            gh = hist_res["g_hist"]     # (S, F, B)
            hh = hist_res["h_hist"]
            gt = hist_res["g_tot"]      # (S,)
            ht = hist_res["h_tot"]
            nt = hist_res["n_tot"]

            # --- Split decisions per internal node ---
            for slot, nid in enumerate(internal):
                if ht[slot] < min_child_weight * 2:
                    continue  # node stays leaf
                best = _find_best_split(
                    slot, gh[slot], hh[slot],
                    float(gt[slot]), float(ht[slot]), int(nt[slot]),
                    bins, min_child_weight, min_gain, reg_lambda,
                )
                if best is None:
                    continue
                j, thr, gain = best
                left_id = next_node_id
                right_id = next_node_id + 1
                next_node_id += 2
                tree[nid] = (j, thr, left_id, right_id)
                tree[left_id] = None
                tree[right_id] = None

            # --- Pass 2: reassign leaves ---
            assign_op = _AssignLeavesOp(X_cols, mn, mx, bins, tree)
            cur_leaf = reduce(
                df._iter_filtered_batches(), assign_op, mg, log=False,
            )

        # --- Pass 3: leaf values ---
        def _iter_for_leaves():
            offset = 0
            for b in df._iter_filtered_batches():
                n = b.num_rows
                X = _stack_numeric(b, X_cols)
                y = _column_to_f64(b.column(y_col))
                p = pred[offset:offset + n]
                lf = cur_leaf[offset:offset + n]
                valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                g = np.where(valid, p - y, 0.0)
                h = np.where(valid, 1.0, 0.0)
                cols = [
                    pa.array(g, type=pa.float64()),
                    pa.array(h, type=pa.float64()),
                    pa.array(lf, type=pa.int64()),
                ]
                yield pa.RecordBatch.from_arrays(cols, names=["__g", "__h", "__leaf"])
                offset += n

        lv_op = _LeafValuesOp(reg_lambda=reg_lambda)
        leaf_values = reduce(_iter_for_leaves(), lv_op, mg, log=False)

        # Store tree
        trees.append({
            "structure": dict(tree),
            "leaf_values": dict(leaf_values),
        })

        # Update pred: for each row, pred += lr * leaf_values[leaf_id]
        if leaf_values:
            max_lid = max(leaf_values.keys())
            lv_arr = np.zeros(max_lid + 1, dtype=np.float64)
            for lid, v in leaf_values.items():
                lv_arr[lid] = v
            pred += learning_rate * lv_arr[cur_leaf]

    return GBDTRegressor(
        trees_=trees,
        base_prediction_=float(base_pred),
        feature_names_=list(X_cols),
        target_col_=y_col,
        mn_=mn,
        mx_=mx,
        bins_=int(bins),
        learning_rate_=float(learning_rate),
        n_estimators_=int(n_estimators),
        max_depth_=int(max_depth),
        n_features_in_=len(X_cols),
    )
