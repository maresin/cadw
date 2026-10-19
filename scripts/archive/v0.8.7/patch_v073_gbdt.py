#!/usr/bin/env python3
"""v0.7.3: GBDT regression (L2 loss, level-wise).

Design:
    Level-wise gradient boosting. Per boosting round:
        for level in range(max_depth):
            Reduce(HistOp): (grad, hess) per (node, feature, bin)
            Split internal leaves by max gain
            Reduce(AssignLeavesOp): reassign rows to new leaves
        Reduce(LeafValuesOp): leaf values = -sum_grad / (sum_hess + λ)
        pred += lr * leaf_value[leaf_id]

    Peak RAM = batch + O(n × 32B) for pred/cur_leaf/grad/hess
    (guarded via check_state). Tree state O(nodes × feat × bins).

    O(n) internal state is unavoidable for Newton-based boosting
    (XGBoost / LightGBM do the same). Documented §11.5.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


# ============ cadw/ml/gbdt.py ============

GBDT_SRC = '''"""Gradient Boosting Decision Tree — regression (v0.7.3a).

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
        # Traverse depth-by-depth. tree is a flat dict; internal nodes have
        # children with higher node_ids (children = 2*parent+1, 2*parent+2).
        max_iter = 32
        for _ in range(max_iter):
            any_change = False
            for nid, spec in self.tree.items():
                if spec is None:
                    continue
                j, thr, left, right = spec
                m = leaf == nid
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
                pred = np.full(n, base, dtype=np.float64)
                for tree in trees:
                    leaf = np.zeros(n, dtype=np.int64)
                    for nid, spec in tree["structure"].items():
                        if spec is None:
                            continue
                        j, thr, left, right = spec
                        m = leaf == nid
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
        gains = np.where(
            valid,
            g_l * g_l / (h_l + reg_lambda)
            + g_r * g_r / (h_r + reg_lambda)
            - parent_gain,
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

    # Guard: internal O(n) state (pred + cur_leaf + grad + hess ≈ 32B/row)
    n_total = df._dataset.total_rows
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
'''


# ============ patch dataframe ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: fit_gbdt_regressor ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def fit_gbdt_regressor" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    method = '''    # ---------------- ML: fit_gbdt_regressor ----------------

    def fit_gbdt_regressor(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        bins: int = 32,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_gain: float = 0.0,
    ):
        """Fit GBDT regression (L2 loss, level-wise) — v0.7.3a.

        Peak RAM = batch + O(n × 32B) internal state + O(nodes × feat × bins).
        O(n) state is inherent to Newton-based boosting (XGBoost/LightGBM
        same). Guarded via MemoryGuard.

        Returns GBDTRegressor (trees_, predict).
        """
        from cadw.ml.gbdt import fit_gbdt_regressor
        return fit_gbdt_regressor(
            self, X_cols, y_col,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, bins=bins,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            min_gain=min_gain,
            memory_guard=self._memory_guard,
        )

'''
    s = s.replace(anchor, method + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: fit_gbdt_regressor method added")


# ============ exports ============

def patch_exports():
    print("\n=== cadw/ml/__init__.py + cadw/__init__.py ===")
    p = Path("cadw/ml/__init__.py")
    p.write_text('''"""CADW ML layer (v0.7.3)."""
from .linear import XTXOp, LinearModel, fit_linear
from .logistic import LogisticGradOp, LogisticModel, fit_logistic
from .kmeans import KMeansStepOp, KMeansModel, fit_kmeans
from .pca import PCA, fit_pca
from .gbdt import GBDTRegressor, fit_gbdt_regressor

__all__ = [
    "XTXOp", "LinearModel", "fit_linear",
    "LogisticGradOp", "LogisticModel", "fit_logistic",
    "KMeansStepOp", "KMeansModel", "fit_kmeans",
    "PCA", "fit_pca",
    "GBDTRegressor", "fit_gbdt_regressor",
]
''')
    compile_check(p)

    p2 = Path("cadw/__init__.py")
    s = p2.read_text()
    if "GBDTRegressor" in s:
        print("  top-level already updated")
        return
    s = s.replace(
        "    PCA, fit_pca,\n)",
        "    PCA, fit_pca,\n"
        "    GBDTRegressor, fit_gbdt_regressor,\n)",
        1,
    )
    s = s.replace(
        '    "PCA", "fit_pca",\n',
        '    "PCA", "fit_pca",\n'
        '    "GBDTRegressor", "fit_gbdt_regressor",\n',
        1,
    )
    p2.write_text(s)
    compile_check(p2)
    print("  OK: exports updated")


# ============ tests ============

TESTS_SRC = '''"""v0.7.3: GBDT regression (L2 loss, level-wise)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ml.gbdt import GBDTRegressor, fit_gbdt_regressor


def make_df(tmp_path, data, chunk=20, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _r2(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ---------- basic fit ----------

def test_fit_gbdt_linear_perfect(tmp_path):
    """Linear y = 2x: single depth-1 tree + lr=1 should fit exactly."""
    n = 200
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, size=n)
    y = 2.0 * x
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(
        df, ["x"], "y",
        n_estimators=10, max_depth=3, learning_rate=0.3,
        bins=32, min_child_weight=1.0, reg_lambda=0.0,
    )
    pred = model.predict(np.column_stack([x]))
    r2 = _r2(y, pred)
    assert r2 > 0.95, f"R²={r2:.3f}"


def test_fit_gbdt_nonlinear(tmp_path):
    """Nonlinear y = sin(2x) + x². GBDT beats constant baseline."""
    n = 1000
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, size=n)
    y = np.sin(2.0 * x) + x * x + 0.05 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(
        df, ["x"], "y",
        n_estimators=50, max_depth=3, learning_rate=0.1, bins=32,
    )
    pred = model.predict(np.column_stack([x]))
    r2 = _r2(y, pred)
    assert r2 > 0.7, f"R²={r2:.3f} (baseline would be 0)"


def test_fit_gbdt_two_features(tmp_path):
    """y = x1 + 2*x2 (linear) — GBDT recovers via splits."""
    n = 500
    rng = np.random.default_rng(42)
    x1 = rng.uniform(-1, 1, size=n)
    x2 = rng.uniform(-1, 1, size=n)
    y = x1 + 2 * x2
    df = make_df(tmp_path, {
        "x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist(),
    })
    model = fit_gbdt_regressor(
        df, ["x1", "x2"], "y",
        n_estimators=30, max_depth=3, learning_rate=0.2, bins=16,
    )
    pred = model.predict(np.column_stack([x1, x2]))
    r2 = _r2(y, pred)
    assert r2 > 0.9, f"R²={r2:.3f}"


def test_fit_gbdt_n_estimators_improves(tmp_path):
    """More trees → better R² (monotone-ish)."""
    n = 500
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, size=n)
    y = np.sin(2.0 * x) + 0.05 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    m10 = fit_gbdt_regressor(
        df, ["x"], "y", n_estimators=10, max_depth=3, learning_rate=0.1,
    )
    m50 = fit_gbdt_regressor(
        df, ["x"], "y", n_estimators=50, max_depth=3, learning_rate=0.1,
    )
    r2_10 = _r2(y, m10.predict(np.column_stack([x])))
    r2_50 = _r2(y, m50.predict(np.column_stack([x])))
    assert r2_50 > r2_10, f"R²(50)={r2_50:.3f} vs R²(10)={r2_10:.3f}"


def test_fit_gbdt_learning_rate_effect(tmp_path):
    """Higher lr converges faster per round (but may overfit)."""
    n = 500
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, size=n)
    y = np.sin(2.0 * x)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    m_small_lr = fit_gbdt_regressor(
        df, ["x"], "y", n_estimators=5, max_depth=3, learning_rate=0.05,
    )
    m_big_lr = fit_gbdt_regressor(
        df, ["x"], "y", n_estimators=5, max_depth=3, learning_rate=0.5,
    )
    r2_small = _r2(y, m_small_lr.predict(np.column_stack([x])))
    r2_big = _r2(y, m_big_lr.predict(np.column_stack([x])))
    # Higher lr with more steps: better fit in first 5 rounds
    assert r2_big > r2_small, f"R²(0.5)={r2_big:.3f} vs R²(0.05)={r2_small:.3f}"


# ---------- predict ----------

def test_predict_numpy_shape(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)
    pred = model.predict(np.column_stack([x[:10]]))
    assert pred.shape == (10,)


def test_predict_df_adds_column(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_pred" in names


def test_predict_null_rows_yield_null(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    x_list = x.tolist()
    x_list[5] = None
    df = make_df(tmp_path, {"x": x_list, "y": y.tolist()})
    model = fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows[5]["y_pred"] is None


# ---------- nulls in fit ----------

def test_fit_gbdt_with_nulls(tmp_path):
    n = 200
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    x_list = x.tolist()
    y_list = y.tolist()
    for i in range(0, n, 10):
        x_list[i] = None
    df = make_df(tmp_path, {"x": x_list, "y": y_list})
    model = fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)
    # Predict non-null rows — should have reasonable R² on remaining
    X_eval = np.column_stack([x])
    mask = ~np.isnan(X_eval).any(axis=1)
    pred = model.predict(X_eval[mask])
    r2 = _r2(y[mask], pred)
    assert r2 > 0.3  # weak threshold; noise


# ---------- determinism ----------

def test_fit_gbdt_deterministic(tmp_path):
    n = 200
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    m1 = fit_gbdt_regressor(df, ["x"], "y", n_estimators=10)
    m2 = fit_gbdt_regressor(df, ["x"], "y", n_estimators=10)
    X = np.column_stack([x])
    np.testing.assert_allclose(
        m1.predict(X), m2.predict(X), rtol=1e-9,
    )


def test_fit_gbdt_multibatch_consistent(tmp_path):
    n = 300
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    df1 = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()},
                  chunk=10, name="a")
    df2 = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()},
                  chunk=100, name="b")
    m1 = fit_gbdt_regressor(df1, ["x"], "y", n_estimators=5)
    m2 = fit_gbdt_regressor(df2, ["x"], "y", n_estimators=5)
    X = np.column_stack([x])
    np.testing.assert_allclose(
        m1.predict(X), m2.predict(X), rtol=1e-9,
    )


# ---------- memory guard ----------

def test_fit_gbdt_guard_rejects_tight_budget(tmp_path):
    """n=10k × 32B = 320KB; budget 100KB → MemoryLimitExceeded."""
    n = 10_000
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    df._memory_guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded, match="internal state"):
        fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)


# ---------- validation ----------

def test_fit_gbdt_empty_X(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="X_cols"):
        fit_gbdt_regressor(df, [], "y")


def test_fit_gbdt_target_in_X(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_gbdt_regressor(df, ["x", "y"], "y")


def test_fit_gbdt_missing_column(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(KeyError, match="nope"):
        fit_gbdt_regressor(df, ["nope"], "y")


def test_fit_gbdt_non_numeric(tmp_path):
    df = make_df(tmp_path, {"x": ["a", "b"], "y": [1.0, 2.0]})
    with pytest.raises(TypeError, match="numeric"):
        fit_gbdt_regressor(df, ["x"], "y")


def test_fit_gbdt_invalid_hyperparams(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="n_estimators"):
        fit_gbdt_regressor(df, ["x"], "y", n_estimators=0)
    with pytest.raises(ValueError, match="max_depth"):
        fit_gbdt_regressor(df, ["x"], "y", max_depth=0)
    with pytest.raises(ValueError, match="learning_rate"):
        fit_gbdt_regressor(df, ["x"], "y", learning_rate=0.0)
    with pytest.raises(ValueError, match="bins"):
        fit_gbdt_regressor(df, ["x"], "y", bins=1)


# ---------- convenience / to_dict ----------

def test_fit_gbdt_dataframe_method(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = df.fit_gbdt_regressor("x", "y", n_estimators=5)
    assert isinstance(model, GBDTRegressor)
    assert len(model.trees_) == 5


def test_gbdt_to_dict(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_regressor(df, ["x"], "y", n_estimators=5)
    d = model.to_dict()
    assert "base_prediction" in d
    assert "feature_names" in d
    assert d["n_estimators"] == 5
    assert d["n_trees"] == 5


# ---------- filtered ----------

def test_fit_gbdt_filtered(tmp_path):
    n = 500
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=n)
    y = np.sin(x) + 0.05 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    filtered = df[df["x"] > 0]
    model = fit_gbdt_regressor(filtered, ["x"], "y", n_estimators=10)
    # Predict on [0.1, 2.0] — should fit sin(2x) reasonably on x>0
    X_eval = np.linspace(0.1, 2.0, 100).reshape(-1, 1)
    pred = model.predict(X_eval)
    y_eval = np.sin(X_eval.flatten())
    r2 = _r2(y_eval, pred)
    assert r2 > 0.5
'''


def write_tests():
    print("\n=== tests/test_gbdt_regression.py ===")
    p = Path("tests/test_gbdt_regression.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: tests written")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v073-gbdt-regression"

    print("=== v0.7.3: GBDT regression ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v073-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v073-{ts_str}",
        "-m", "before patch v0.7.3 GBDT regression"])

    print("\n=== cadw/ml/gbdt.py ===")
    Path("cadw/ml/gbdt.py").write_text(GBDT_SRC)
    compile_check("cadw/ml/gbdt.py")
    print("  OK: gbdt.py created")

    patch_dataframe()
    patch_exports()
    write_tests()

    print("\n=== pytest tests/test_gbdt_regression.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_gbdt_regression.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.3 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.3: GBDT regression (level-wise, L2 loss; XGBoost-style hyperparams)"])

    print("\n==> v0.7.3 applied on", branch)


if __name__ == "__main__":
    main()