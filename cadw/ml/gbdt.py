
"""Gradient Boosting Decision Trees — regression + classification (v0.7.5).

Composition: Reduce (histograms) + Reduce (assign leaves) + Reduce
(leaf values). Level-wise tree growth.

Losses:
    - 'l2'              : regression (GBDTRegressor)
    - 'logloss_binary'  : binary classification (GBDTClassifier)

Peak RAM = batch + O(n × 32B) internal state (pred, cur_leaf, grad, hess)
+ O(nodes × feat × bins) histograms. Guarded via MemoryGuard.
The O(n) state is inherent to Newton-based boosting (XGBoost/LightGBM).

For multiclass (softmax) — v0.7.6.
"""
from __future__ import annotations

from cadw.ml._common import _column_to_f64, _stack_numeric

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.reduce import ReduceOp, reduce


# ============ helpers: type conversion, gradient ============







# v0.9.87-P1a-1a: one-time binning infrastructure for GBDT.
# LightGBM-style persistent codes (n, d) uint8 replace per-pass
# _stack_numeric + clip. Enables ~5-10x speedup (P1a-1b swaps ops),
# and prerequisite for P1a-2 parallel-per-class.
#
# Bit-identical formula (matches _AllNodeHistOp.update exactly):
#   b_idx = clip(int((X[n,j] - mn[j]) / range[j] * bins), 0, bins-1)
#   range[j] = max(mx[j] - mn[j], 1.0)
#
# nan_row_mask[n] = any NaN in X[n]. Row-level: if any feature NaN,
# entire row excluded from histograms (per existing _AllNodeHistOp).


def _madvise_dontneed(mm_array) -> None:
    """Force kernel to evict pages of a numpy memmap.

    Linux-only; no-op on other platforms or if mmap.madvise absent.
    Prevents memmap page cache from counting toward peak RSS during
    multi-round GBDT fit (v0.9.172 S2c-1c).
    """
    try:
        import mmap
        if hasattr(mmap, "MADV_DONTNEED"):
            mm = getattr(mm_array, "_mmap", None)
            if mm is not None:
                mm.madvise(mmap.MADV_DONTNEED)
    except Exception:
        pass


def _binning_pass(
    df, X_cols, *, bins: int, memory_guard, guard=None, batch_size=None,
):
    """Stream df -> codes (n, d) uint8 memmap + mn/mx/range/nan_mask.

    Returns (mn, mx, range_, codes_mm, nan_mask, spill_handle).
    Caller owns spill_handle lifecycle.
    """
    X_cols = list(X_cols)
    d = len(X_cols)
    mg = memory_guard
    n_total = df.count()
    if n_total == 0:
        raise ValueError("_binning_pass: empty df")

    mm_res = reduce(
        df._iter_filtered_batches(batch_size=batch_size),
        _MinMaxOp(X_cols), mg, log=False,
    )
    mn = np.asarray(mm_res["min"], dtype=np.float64)
    mx = np.asarray(mm_res["max"], dtype=np.float64)
    if np.any(np.isinf(mn)) or np.any(np.isinf(mx)):
        raise ValueError("_binning_pass: no valid rows (all NaN)")
    range_ = np.where(mx - mn > 0, mx - mn, 1.0)

    spill = SpillDir(guard=guard, name="gbdt_binning")
    spill.__enter__()
    codes_mm = np.memmap(
        str(spill.path("codes.bin")), dtype=np.uint8, mode="w+",
        shape=(n_total, d))
    # v0.9.172 (S2c): nan_mask on disk (bool memmap). Readers unchanged.
    nan_mask = np.memmap(
        str(spill.path("nan_mask.bin")), dtype=np.bool_, mode="w+",
        shape=(n_total,))
    off = 0
    bins_i = int(bins)
    for b in df._iter_filtered_batches(batch_size=batch_size):
        X_b = _stack_numeric(b, X_cols)
        nb = b.num_rows
        nan_rows_b = np.isnan(X_b).any(axis=1)
        nan_mask[off:off + nb] = nan_rows_b
        with np.errstate(divide="ignore", invalid="ignore"):
            codes_b = np.clip(
                ((X_b - mn) / range_ * bins_i).astype(np.int64),
                0, bins_i - 1,
            ).astype(np.uint8)
        codes_b[nan_rows_b] = 0
        codes_mm[off:off + nb] = codes_b
        off += nb
    if off != n_total:
        codes_mm.flush()
        del codes_mm
        codes_mm = np.memmap(
            str(spill.path("codes.bin")), dtype=np.uint8, mode="r+",
            shape=(off, d))
        nan_mask = np.memmap(
            str(spill.path("nan_mask.bin")), dtype=np.bool_, mode="r+",
            shape=(off,))
        n_total = off
    codes_mm.flush()
    return mn, mx, range_, codes_mm, nan_mask, spill


def _attach_codes(enriched_iter, codes_mm, nan_mask):
    """Augment enriched batches with __code_0..__code_{d-1} uint8 and
    __nan_mask bool columns.
    """
    off = 0
    d = codes_mm.shape[1]
    for b in enriched_iter:
        n = b.num_rows
        codes_b = np.asarray(codes_mm[off:off + n])
        nm_b = np.asarray(nan_mask[off:off + n])
        base_cols = list(b.columns)
        base_names = list(b.schema.names)
        code_cols = [pa.array(codes_b[:, j], type=pa.uint8())
                     for j in range(d)]
        code_names = [f"__code_{j}" for j in range(d)]
        yield pa.RecordBatch.from_arrays(
            base_cols + code_cols + [pa.array(nm_b)],
            names=base_names + code_names + ["__nan_mask"],
        )
        off += n

# v0.9.131 (DEC-054): numerically stable sigmoid from
# cadw.ops.optim (handles overflow for |z|>>0 without RuntimeWarning).
from cadw.ops.numeric import sigmoid as _sigmoid



_LOSSES = ("l2", "logloss_binary")


def _grad_hess(
    pred: np.ndarray, y: np.ndarray, loss: str,
    pos_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Grad and hess per row for given loss."""
    if loss == "l2":
        return pred - y, np.ones_like(y, dtype=np.float64)
    if loss == "logloss_binary":
        p = _sigmoid(pred)
        g = p - y
        h = np.maximum(p * (1.0 - p), 1e-16)
        if pos_weight != 1.0:
            w = np.where(y > 0.5, pos_weight, 1.0)
            g = g * w
            h = h * w
        return g, h
    raise ValueError(f"_grad_hess: unknown loss {loss!r}")


def _base_prediction(y_sum: float, n: int, loss: str) -> float:
    if loss == "l2":
        return y_sum / n
    if loss == "logloss_binary":
        p = y_sum / n
        eps = 1e-6
        p = min(max(p, eps), 1.0 - eps)
        return float(np.log(p / (1.0 - p)))
    raise ValueError(f"_base_prediction: unknown loss {loss!r}")


def _link(pred: np.ndarray, loss: str) -> np.ndarray:
    if loss == "l2":
        return pred
    if loss == "logloss_binary":
        return _sigmoid(pred)
    raise ValueError(f"_link: unknown loss {loss!r}")


# ============ ReduceOps (loss-agnostic) ============

from cadw.core.spill import SpillDir


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
    """Accumulate (grad, hess) histograms for all nodes in one pass.

    State: g_hist (S, F, B), h_hist (S, F, B), g_tot, h_tot, n_tot per slot.
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

    def batch_memory_factor(self):
        # v0.9.32 AUDIT_v0931 F9: update peaks ~3x yielded batch.nbytes.
        return 7.0

    def finalize(self, state):
        gh, hh, gt, ht, nt = state
        return {
            "g_hist": gh, "h_hist": hh,
            "g_tot": gt, "h_tot": ht, "n_tot": nt,
        }




class _AllNodeHistOpCodes(ReduceOp):
    """Histogram from pre-binned codes (uint8). Same output as _AllNodeHistOp.

    Reads __code_0..__code_{d-1} (uint8) and __nan_mask (bool) columns
    attached by _attach_codes; uses np.bincount instead of np.add.at.
    Bit-identical to _AllNodeHistOp.update within float precision
    (order of summation differs: np.bincount vs np.add.at; ~1e-15).

    v0.9.87-P1a-1a: infrastructure. Not yet wired into fit paths.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        node_ids: Sequence[int],
        bins: int,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.f = len(feature_cols)
        self.node_ids = list(node_ids)
        self.node_to_slot = {nid: i for i, nid in enumerate(self.node_ids)}
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
        n = batch.num_rows
        if n == 0:
            return state
        codes = np.stack(
            [batch.column(f"__code_{j}").to_numpy(zero_copy_only=False)
             for j in range(self.f)], axis=1)
        nan_mask = batch.column("__nan_mask").to_numpy(zero_copy_only=False)
        g = _column_to_f64(batch.column("__g"))
        h = _column_to_f64(batch.column("__h"))
        lf = batch.column("__leaf").to_numpy(zero_copy_only=False)
        for nid, slot in self.node_to_slot.items():
            m = lf == nid
            if not m.any():
                continue
            m_valid = m & ~nan_mask
            if not m_valid.any():
                continue
            gn = g[m_valid]
            hn = h[m_valid]
            gt[slot] += float(gn.sum())
            ht[slot] += float(hn.sum())
            nt[slot] += int(m_valid.sum())
            codes_valid = codes[m_valid]
            for j in range(self.f):
                b_idx = codes_valid[:, j]
                gh[slot, j] += np.bincount(
                    b_idx, weights=gn, minlength=self.bins)
                hh[slot, j] += np.bincount(
                    b_idx, weights=hn, minlength=self.bins)
        return (gh, hh, gt, ht, nt)

    def merge(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2],
                a[3] + b[3], a[4] + b[4])

    def state_nbytes(self, state):
        return int(sum(x.nbytes for x in state))

    def estimate_state_nbytes(self):
        per_slot = self.f * self.bins * 8 * 2 + 8 * 3
        return self.S * per_slot

    def batch_memory_factor(self):
        return 7.0

    def finalize(self, state):
        gh, hh, gt, ht, nt = state
        return {
            "g_hist": gh, "h_hist": hh,
            "g_tot": gt, "h_tot": ht, "n_tot": nt,
        }

class _AssignLeavesOp(ReduceOp):
    """Assign rows to leaf_ids per current tree structure."""

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
        self.tree = dict(tree)
        self.root_id = int(root_id)

    def init(self):
        return []

    def update(self, state, batch):
        X = _stack_numeric(batch, self.feature_cols)
        n = X.shape[0]
        leaf = np.full(n, self.root_id, dtype=np.int64)
        nan_rows = np.isnan(X).any(axis=1)
        max_iter = 32
        with np.errstate(divide="ignore", invalid="ignore"):
            for _ in range(max_iter):
                any_change = False
                for nid, spec in self.tree.items():
                    if spec is None:
                        continue
                    j, thr, left, right = spec[:4]
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
        return 0

    def batch_memory_factor(self):
        # v0.9.32 AUDIT_v0931 F9.
        return 4.0

    def finalize(self, state):
        return np.concatenate(state) if state else np.array([], dtype=np.int64)




def _assign_leaf_batch(
    codes_b: np.ndarray,
    nan_mask_b: np.ndarray,
    tree: dict,
    root_id: int = 0,
) -> np.ndarray:
    """Leaf assignment for one batch (extracted from _AssignLeavesOpCodes.update).

    Bit-identical to _AssignLeavesOpCodes.update within same tree.
    v0.9.172 (S2c): used by _fit_boosting direct loop to write leaf
    assignments into cur_leaf_mm without materializing n-sized list.
    """
    n = codes_b.shape[0]
    leaf = np.full(n, root_id, dtype=np.int64)
    max_iter = 32
    for _ in range(max_iter):
        any_change = False
        for nid, spec in tree.items():
            if spec is None:
                continue
            j, thr, left, right = spec[:4]
            m = (leaf == nid) & ~nan_mask_b
            if not m.any():
                continue
            new_ids = np.where(codes_b[m, j] <= thr, left, right)
            leaf[m] = new_ids
            any_change = True
        if not any_change:
            break
    return leaf


class _AssignLeavesOpCodes(ReduceOp):
    """Leaf assignment from pre-binned codes (uint8).

    Tree thr is a bin index (from _find_best_split against histogram bins),
    so no bin recomputation needed: leaf = right if code > thr else left.
    Bit-identical to _AssignLeavesOp within same tree.

    Reads __code_0..__code_{d-1} + __nan_mask only.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        tree: dict,
        root_id: int = 0,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.f = len(feature_cols)
        self.tree = dict(tree)
        self.root_id = int(root_id)

    def init(self):
        return []

    def update(self, state, batch):
        n = batch.num_rows
        codes = np.stack(
            [batch.column(f"__code_{j}").to_numpy(zero_copy_only=False)
             for j in range(self.f)], axis=1)
        nan_mask_b = batch.column("__nan_mask").to_numpy(zero_copy_only=False)
        leaf = _assign_leaf_batch(codes, nan_mask_b, self.tree, self.root_id)
        state.append(leaf)
        return state

    def merge(self, a, b):
        return a + b

    def state_nbytes(self, state):
        return int(sum(x.nbytes for x in state) + 8)

    def estimate_state_nbytes(self):
        return 0

    def batch_memory_factor(self):
        return 4.0

    def finalize(self, state):
        return np.concatenate(state) if state else np.array([], dtype=np.int64)

class _LeafValuesOp(ReduceOp):
    """Per-leaf: sum_grad, sum_hess → Newton step for any loss."""

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
        return 0

    def batch_memory_factor(self):
        # v0.9.32 AUDIT_v0931 F9.
        return 3.5

    def finalize(self, state):
        return {
            k: -g / (h + self.reg_lambda)
            for k, (g, h) in state.items()
        }


# ============ tree utility ============

def _find_best_split(
    slot: int,
    gh: np.ndarray, hh: np.ndarray,
    gt: float, ht: float, n_tot: int,
    bins: int,
    min_child_weight: float,
    min_gain: float,
    reg_lambda: float,
) -> Optional[tuple[int, int, float]]:
    if ht <= 1e-12:
        return None
    parent_denom = ht + reg_lambda
    parent_gain = gt * gt / parent_denom
    best = None
    F = gh.shape[0]
    for j in range(F):
        g_l = np.cumsum(gh[j])[:-1]
        h_l = np.cumsum(hh[j])[:-1]
        g_r = gt - g_l
        h_r = ht - h_l
        valid = (h_l >= min_child_weight) & (h_r >= min_child_weight)
        if not valid.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            g_l_sq = g_l * g_l / (h_l + reg_lambda)
            g_r_sq = g_r * g_r / (h_r + reg_lambda)
        gains = np.where(
            valid, g_l_sq + g_r_sq - parent_gain, -np.inf,
        )
        k = int(np.argmax(gains))
        if gains[k] > (best[2] if best is not None else min_gain):
            best = (j, k, float(gains[k]))
    return best


def _assign_leaf_from_array(
    X: np.ndarray, tree_struct: dict,
    mn: np.ndarray, mx: np.ndarray, bins: int,
) -> np.ndarray:
    n = X.shape[0]
    leaf = np.zeros(n, dtype=np.int64)
    nan_rows = np.isnan(X).any(axis=1)
    rng = np.where(mx - mn > 0, mx - mn, 1.0)
    for nid, spec in tree_struct.items():
        if spec is None:
            continue
        # v0.7.9: tolerate 4-tuple (pre-v0.7.9 pickled models) or
        # 5-tuple (gain appended at index 4).
        j, thr, left, right = spec[:4]
        m = (leaf == nid) & ~nan_rows
        if not m.any():
            continue
        b_idx = np.clip(
            ((X[m, j] - mn[j]) / rng[j] * bins)
            .astype(np.int64), 0, bins - 1,
        )
        leaf[m] = np.where(b_idx <= thr, left, right)
    return leaf


_STATE_HEADER_BYTES = 256


def _deep_nbytes(obj) -> int:
    """Exact recursive byte count for nested dict/list/tuple/ndarray/scalar.

    Used by GBDTModel.state_nbytes(). Not for hot paths — fit/predict
    call it once at model finalization.
    """
    import sys
    total = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            total += sys.getsizeof(k) + _deep_nbytes(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            total += _deep_nbytes(item)
    elif hasattr(obj, "nbytes"):
        total += int(obj.nbytes)
    return int(total)


def _trees_nbytes(trees) -> int:
    """Sum _deep_nbytes over all tree dicts in `trees`.

    Accepts either list of {"structure":..., "leaf_values":...} or
    list of list (multiclass: trees[round][class] = tree dict).
    """
    total = 0
    for t in trees:
        if isinstance(t, list):
            total += _trees_nbytes(t)
        elif isinstance(t, dict):
            total += _deep_nbytes(t)
        else:
            total += _deep_nbytes(t)
    return int(total)


def _leaf_vector(lv_map: dict, leaf: np.ndarray) -> np.ndarray:
    if not lv_map:
        return np.zeros_like(leaf, dtype=np.float64)
    max_lid = max(lv_map.keys())
    arr = np.zeros(max_lid + 1, dtype=np.float64)
    for lid, v in lv_map.items():
        arr[lid] = v
    return arr[leaf]


# ============ models ============

@dataclass
class GBDTRegressor:
    """GBDT regression (L2 loss)."""

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
        if isinstance(X, np.ndarray):
            return self._predict_np(X)
        return self._predict_df(X)

    def _predict_np(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"predict: X has {X.shape[1]} features, "
                f"expected {self.n_features_in_}"
            )
        pred = np.full(X.shape[0], self.base_prediction_, dtype=np.float64)
        for tree in self.trees_:
            leaf = _assign_leaf_from_array(
                X, tree["structure"], self.mn_, self.mx_, self.bins_,
            )
            pred += self.learning_rate_ * _leaf_vector(
                tree["leaf_values"], leaf,
            )
        return pred

    def _predict_df(self, X):
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
        bins = self.bins_
        feat = self.feature_names_

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, feat)
                n = Xm.shape[0]
                nan_rows = np.isnan(Xm).any(axis=1)
                pred = np.full(n, base, dtype=np.float64)
                for tree in trees:
                    leaf = _assign_leaf_from_array(
                        Xm, tree["structure"], mn, mx, bins,
                    )
                    pred += lr * _leaf_vector(tree["leaf_values"], leaf)
                arrays = [b.column(c) for c in X._columns]
                arrays.append(pa.array(
                    pred, type=pa.float64(),
                    mask=nan_rows if nan_rows.any() else None,
                ))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )

    def state_nbytes(self) -> int:
        """Exact byte count of persistent state: trees + min/max + header.

        Tree representation: dict[int, tuple|None] (structure) +
        dict[int, float] (leaf_values). CPython per-node overhead
        dominates; this is exact via _deep_nbytes.
        """
        total = _STATE_HEADER_BYTES
        total += self.mn_.nbytes + self.mx_.nbytes
        total += _trees_nbytes(self.trees_)
        return int(total)

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
            "loss": "l2",
        }


@dataclass
class GBDTClassifier:
    """GBDT binary classification (logloss).

    predict_proba -> P(y=1|x); predict -> {0,1} by threshold.
    """

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

    def _raw_score_np(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"predict: X has {X.shape[1]} features, "
                f"expected {self.n_features_in_}"
            )
        pred = np.full(X.shape[0], self.base_prediction_, dtype=np.float64)
        for tree in self.trees_:
            leaf = _assign_leaf_from_array(
                X, tree["structure"], self.mn_, self.mx_, self.bins_,
            )
            pred += self.learning_rate_ * _leaf_vector(
                tree["leaf_values"], leaf,
            )
        return pred

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            return _sigmoid(self._raw_score_np(X))
        return self._predict_df(X, proba_only=True, threshold=0.5)

    def predict(self, X, threshold: float = 0.5):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"predict: threshold must be in [0, 1]; got {threshold}"
            )
        if isinstance(X, np.ndarray):
            proba = self.predict_proba(X)
            return (proba >= threshold).astype(np.int64)
        return self._predict_df(X, proba_only=False, threshold=threshold)

    def _predict_df(self, X, *, proba_only: bool, threshold: float):
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

        prob_col = f"{self.target_col_}_prob"
        pred_col = f"{self.target_col_}_pred"
        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        out_fields.append(pa.field(prob_col, pa.float64(), nullable=True))
        if not proba_only:
            out_fields.append(pa.field(pred_col, pa.int64(), nullable=True))
        out_schema = pa.schema(out_fields)
        trees = self.trees_
        lr = self.learning_rate_
        base = self.base_prediction_
        mn = self.mn_
        mx = self.mx_
        bins = self.bins_
        feat = self.feature_names_

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, feat)
                n = Xm.shape[0]
                nan_rows = np.isnan(Xm).any(axis=1)
                pred = np.full(n, base, dtype=np.float64)
                for tree in trees:
                    leaf = _assign_leaf_from_array(
                        Xm, tree["structure"], mn, mx, bins,
                    )
                    pred += lr * _leaf_vector(tree["leaf_values"], leaf)
                proba = _sigmoid(pred)
                arrays = [b.column(c) for c in X._columns]
                arrays.append(pa.array(
                    proba, type=pa.float64(),
                    mask=nan_rows if nan_rows.any() else None,
                ))
                if not proba_only:
                    pred01 = (proba >= threshold).astype(np.int64)
                    arrays.append(pa.array(
                        pred01, type=pa.int64(),
                        mask=nan_rows if nan_rows.any() else None,
                    ))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )

    def state_nbytes(self) -> int:
        """Exact byte count of persistent state: trees + min/max + header.

        Tree representation: dict[int, tuple|None] (structure) +
        dict[int, float] (leaf_values). CPython per-node overhead
        dominates; this is exact via _deep_nbytes.
        """
        total = _STATE_HEADER_BYTES
        total += self.mn_.nbytes + self.mx_.nbytes
        total += _trees_nbytes(self.trees_)
        return int(total)

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
            "loss": "logloss_binary",
        }


# ============ shared fit ============

def _validate_fit_args(
    df, X_cols, y_col, *, n_estimators, max_depth, learning_rate, bins,
    min_child_weight, reg_lambda,
):
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_gbdt: X_cols must be non-empty")
    if y_col in X_cols:
        raise ValueError(f"fit_gbdt: target {y_col!r} must not be in X_cols")
    for c in X_cols + [y_col]:
        if c not in df.columns:
            raise KeyError(f"fit_gbdt: column {c!r} not in df")
    schema = df._dataset.schema
    for c in X_cols + [y_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_gbdt: column {c!r} has type {t}; numeric required"
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
    return X_cols


def _validate_binary_y(df, y_col: str) -> tuple[float, int]:
    """Ensure y has both classes {0, 1}. Returns (y_sum, n_valid).

    v0.7.6.1 (I5): returns y_sum and n_valid so _fit_boosting can skip
    a second full scan. Single pass over df instead of two.
    """
    # v0.9.22 (§13.4/246, AUDIT_ML §1.2): single-pass over y.
    # Prior did col.unique() + uniq.to_pylist() + _column_to_f64(col)
    # = 3 passes per batch. Now: one numpy scan; validate + sum +
    # count in one shot. Output identical (same y_sum, n_valid,
    # same ValueError for non-binary y).
    import numpy as np
    seen: set = set()
    y_sum = 0.0
    n_valid = 0
    for b in df._iter_filtered_batches():
        y_np = _column_to_f64(b.column(y_col))
        mask = ~np.isnan(y_np)
        valid = y_np[mask]
        if valid.size == 0:
            continue
        bad = (valid != 0.0) & (valid != 1.0)
        if bad.any():
            raise ValueError(
                f"fit_gbdt_classifier: y must be binary (0/1); "
                f"got {float(valid[bad][0])!r}"
            )
        if (valid == 0.0).any():
            seen.add(0.0)
        if (valid == 1.0).any():
            seen.add(1.0)
        y_sum += float(valid.sum())
        n_valid += int(valid.size)
    if len(seen) < 2:
        raise ValueError(
            f"fit_gbdt_classifier: y must have both classes 0 and 1; got {seen}"
        )
    return y_sum, n_valid


def _fit_boosting(
    df, X_cols, y_col, *,
    loss: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    bins: int,
    min_child_weight: float,
    reg_lambda: float,
    min_gain: float,
    memory_guard: Optional[MemoryGuard],
    scale_pos_weight: float = 1.0,
    batch_size: Optional[int] = None,
):
    """Common level-wise boosting fit for l2 and logloss_binary.

    v0.9.172 (S2c): truly OOC in n. All n-sized state on SpillDir memmap:
    pred (n×f64), cur_leaf (n×i64), nan_mask (n×bool). Batch-only RAM.
    batch_size: explicit or auto-derived from guard (DEC-023).
    """
    if loss not in _LOSSES:
        raise ValueError(f"_fit_boosting: unknown loss {loss!r}")

    X_cols = _validate_fit_args(
        df, X_cols, y_col,
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, bins=bins,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
    )

    precomputed_y_sum: Optional[float] = None
    precomputed_n_valid: Optional[int] = None
    if loss == "logloss_binary":
        precomputed_y_sum, precomputed_n_valid = _validate_binary_y(df, y_col)

    mg = memory_guard or df._memory_guard
    n_total = df.count()
    if n_total == 0:
        raise ValueError("fit_gbdt: no valid (non-null) rows (filtered count 0)")
    # v0.9.172: conservative upper bound for peak RSS.
    # State itself is memmap (pred/cur_leaf/nan_mask), but page-touch
    # transient + per-round batch × (d+2) + trees accumulation ≈ 32 B/row
    # worst case. Refuses early at tight guard (<32 B/row budget).
    mg.check_state(n_total * 32, name="gbdt.internal_state")

    # v0.9.172: auto-derive batch_size for true OOC (DEC-023 class).
    if batch_size is None:
        row_size = len(X_cols) * 8 + 8  # features f64 + y f64
        try:
            batch_size = mg.get_batch_rows(expansion_factor=3.0)
        except (TypeError, AttributeError):
            batch_size = 50_000

    mn, mx, _range_, codes_mm, nan_mask_mm, binning_spill = _binning_pass(
        df, X_cols, bins=int(bins), memory_guard=mg, guard=mg,
        batch_size=batch_size)

    # Base prediction (loss-specific)
    if precomputed_y_sum is not None and precomputed_n_valid is not None:
        y_sum = precomputed_y_sum
        n_valid = precomputed_n_valid
    else:
        y_sum = 0.0
        n_valid = 0
        for b in df._iter_filtered_batches():
            X = _stack_numeric(b, X_cols)
            y = _column_to_f64(b.column(y_col))
            valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            y_v = y[valid]
            y_sum += float(y_v.sum())
            n_valid += int(valid.sum())
    if n_valid == 0:
        raise ValueError("fit_gbdt: no valid rows")
    base_pred = _base_prediction(y_sum, n_valid, loss)

    # v0.9.172: pred + cur_leaf on disk (memmap)
    pred_mm = np.memmap(
        str(binning_spill.path("pred.bin")), dtype=np.float64,
        mode="w+", shape=(n_total,))
    pred_mm[:] = base_pred
    pred_mm.flush()

    cur_leaf_mm = np.memmap(
        str(binning_spill.path("cur_leaf.bin")), dtype=np.int64,
        mode="w+", shape=(n_total,))

    trees = []
    _d = len(X_cols)
    _CHUNK = 50_000

    def _iter_with_state():
        off = 0
        for b in df._iter_filtered_batches(batch_size=batch_size):
            n = b.num_rows
            y = _column_to_f64(b.column(y_col))
            p = np.asarray(pred_mm[off:off + n])
            lf = np.asarray(cur_leaf_mm[off:off + n])
            nm = np.asarray(nan_mask_mm[off:off + n])
            valid = ~nm & ~np.isnan(y)
            g, h = _grad_hess(p, y, loss, pos_weight=scale_pos_weight)
            g = np.where(valid, g, 0.0)
            h = np.where(valid, h, 0.0)
            codes_b = np.asarray(codes_mm[off:off + n])
            cols = [pa.array(codes_b[:, j], type=pa.uint8())
                    for j in range(_d)]
            cols.append(pa.array(g, type=pa.float64()))
            cols.append(pa.array(h, type=pa.float64()))
            cols.append(pa.array(lf, type=pa.int64()))
            cols.append(pa.array(nm))
            names = ([f"__code_{j}" for j in range(_d)]
                     + ["__g", "__h", "__leaf", "__nan_mask"])
            yield pa.RecordBatch.from_arrays(cols, names=names)
            off += n

    def _iter_for_leaves():
        off = 0
        for b in df._iter_filtered_batches(batch_size=batch_size):
            n = b.num_rows
            y = _column_to_f64(b.column(y_col))
            p = np.asarray(pred_mm[off:off + n])
            lf = np.asarray(cur_leaf_mm[off:off + n])
            nm = np.asarray(nan_mask_mm[off:off + n])
            valid = ~nm & ~np.isnan(y)
            g, h = _grad_hess(p, y, loss, pos_weight=scale_pos_weight)
            g = np.where(valid, g, 0.0)
            h = np.where(valid, h, 0.0)
            yield pa.RecordBatch.from_arrays(
                [pa.array(g, type=pa.float64()),
                 pa.array(h, type=pa.float64()),
                 pa.array(lf, type=pa.int64())],
                names=["__g", "__h", "__leaf"],
            )
            off += n

    for round_i in range(n_estimators):
        tree: dict = {0: None}
        cur_leaf_mm[:] = 0
        cur_leaf_mm.flush()
        _madvise_dontneed(cur_leaf_mm)
        next_node_id = 1

        for level in range(max_depth):
            internal = [nid for nid, t in tree.items() if t is None]
            if not internal:
                break

            hist_op = _AllNodeHistOpCodes(X_cols, internal, bins)
            hist_res = reduce(_iter_with_state(), hist_op, mg, log=False)
            gh = hist_res["g_hist"]
            hh = hist_res["h_hist"]
            gt = hist_res["g_tot"]
            ht = hist_res["h_tot"]
            nt = hist_res["n_tot"]

            for slot, nid in enumerate(internal):
                if ht[slot] < min_child_weight * 2:
                    continue
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
                tree[nid] = (j, thr, left_id, right_id, float(gain))
                tree[left_id] = None
                tree[right_id] = None

            off = 0
            for b in _iter_x_with_codes(df, X_cols, codes_mm, nan_mask_mm,
                                         batch_size=batch_size):
                n_b = b.num_rows
                mg.check_batch(b, name="gbdt.assign_leaf",
                                extra_factor=3.0)
                codes_b = np.stack(
                    [b.column(f"__code_{j}").to_numpy(
                        zero_copy_only=False) for j in range(_d)],
                    axis=1)
                nm_b = b.column("__nan_mask").to_numpy(
                    zero_copy_only=False)
                leaf_b = _assign_leaf_batch(
                    codes_b, nm_b, tree, root_id=0)
                cur_leaf_mm[off:off + n_b] = leaf_b
                off += n_b
                # v0.9.172: evict pages after each batch write
                _madvise_dontneed(cur_leaf_mm)
            cur_leaf_mm.flush()

        lv_op = _LeafValuesOp(reg_lambda=reg_lambda)
        leaf_values = reduce(_iter_for_leaves(), lv_op, mg, log=False)

        trees.append({
            "structure": dict(tree),
            "leaf_values": dict(leaf_values),
        })

        try:
            from cadw.ml._logging import log_fit
            log_fit("gbdt", iter_i=round_i + 1, iter_n=n_estimators,
                    state_bytes=len(trees) * 64 * max_depth)
        except Exception:
            pass

        if leaf_values:
            max_lid = max(leaf_values.keys())
            lv_arr = np.zeros(max_lid + 1, dtype=np.float64)
            for lid, v in leaf_values.items():
                lv_arr[lid] = v
            for chunk_start in range(0, n_total, _CHUNK):
                end = min(chunk_start + _CHUNK, n_total)
                leaf_chunk = np.asarray(cur_leaf_mm[chunk_start:end])
                pred_chunk = np.asarray(
                    pred_mm[chunk_start:end]).copy()
                pred_chunk += learning_rate * lv_arr[leaf_chunk]
                pred_mm[chunk_start:end] = pred_chunk
                # v0.9.172 S2c-1c: evict chunk pages immediately after
                # write to keep RSS bounded by O(chunk), not O(n).
                _madvise_dontneed(pred_mm)
                _madvise_dontneed(cur_leaf_mm)
            pred_mm.flush()
            cur_leaf_mm.flush()

    del codes_mm, cur_leaf_mm, pred_mm, nan_mask_mm
    binning_spill.__exit__(None, None, None)

    return {
        "trees_": trees,
        "base_prediction_": float(base_pred),
        "feature_names_": list(X_cols),
        "target_col_": y_col,
        "mn_": mn,
        "mx_": mx,
        "bins_": int(bins),
        "learning_rate_": float(learning_rate),
        "n_estimators_": int(n_estimators),
        "max_depth_": int(max_depth),
        "n_features_in_": len(X_cols),
    }



def fit_gbdt_regressor(
    df, X_cols, y_col: str,
    *, n_estimators: int = 100, max_depth: int = 3,
    learning_rate: float = 0.1, bins: int = 32,
    min_child_weight: float = 1.0, reg_lambda: float = 1.0,
    min_gain: float = 0.0,
    batch_size: Optional[int] = None,
    memory_guard: Optional[MemoryGuard] = None,
) -> GBDTRegressor:
    """GBDT regression (L2 loss) — v0.7.3 (unchanged API)."""
    out = _fit_boosting(
        df, X_cols, y_col, loss="l2",
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, bins=bins,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
        min_gain=min_gain, memory_guard=memory_guard,
        batch_size=batch_size,
    )
    return GBDTRegressor(**out)


def fit_gbdt_classifier(
    df, X_cols, y_col: str,
    *, n_estimators: int = 100, max_depth: int = 3,
    learning_rate: float = 0.1, bins: int = 32,
    min_child_weight: float = 1.0, reg_lambda: float = 1.0,
    min_gain: float = 0.0,
    n_classes: Optional[int] = None,
    scale_pos_weight: float = 1.0,
    batch_size: Optional[int] = None,
    memory_guard: Optional[MemoryGuard] = None,
):
    """GBDT classification: binary (logloss) or multiclass (softmax) — v0.7.8.

    - y in {0, 1}: binary classifier (GBDTClassifier).
        predict_proba → (n,) P(y=1|x); predict → {0,1}.
    - y in {0, 1, ..., K-1}, K >= 3: multiclass (GBDTMulticlassClassifier).
        predict_proba → (n, K); predict → (n,) int64.
    - 1-indexed y ({1, ..., K}) — ValueError (сдвиньте на -1).

    n_classes: override auto-detection. None → infer from y.
    """
    # Infer n_classes if not given
    if n_classes is None:
        # v0.7.8-fix3: count distinct BEFORE full validation.
        # Backward compat: single-class / 2-non-0-indexed cases route to
        # binary path, which raises the historical "both classes" /
        # "binary (0/1)" error. Multiclass path (with stricter "0-indexed"
        # error) only for intended-K>=3 cases.
        distinct: set[int] = set()
        has_non_int = False
        for b in df._iter_filtered_batches():
            y = _column_to_f64(b.column(y_col))
            for v in y[~np.isnan(y)]:
                fv = float(v)
                if fv != int(fv):
                    has_non_int = True
                    break
                distinct.add(int(fv))
            if has_non_int:
                break
        if has_non_int or len(distinct) <= 2:
            K = 2  # route to binary; _validate_binary_y raises proper error
        else:
            K = len(distinct)  # route to multiclass; _validate_multiclass_y
    else:
        K = int(n_classes)

    if scale_pos_weight <= 0:
        raise ValueError(
            f"scale_pos_weight must be > 0; got {scale_pos_weight}"
        )
    if K < 2:
        raise ValueError(
            f"fit_gbdt_classifier: need >= 2 classes; got {K}"
        )

    if K == 2:
        # Binary path (unchanged)
        out = _fit_boosting(
            df, X_cols, y_col, loss="logloss_binary",
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, bins=bins,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            min_gain=min_gain, memory_guard=memory_guard,
            scale_pos_weight=scale_pos_weight,
        )
        return GBDTClassifier(**out)
    else:
        # Multiclass path
        return fit_gbdt_multiclass(
            df, X_cols, y_col,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, bins=bins,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            min_gain=min_gain, memory_guard=memory_guard,
        )

# ============================================================================
# v0.7.8: Multiclass GBDT (softmax, per-class loop)
# ============================================================================

def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax. z: (n, K) -> (n, K)."""
    if z.ndim != 2:
        raise ValueError(f"_softmax: expected 2D array, got {z.ndim}D")
    m = np.max(z, axis=1, keepdims=True)
    exp_d = np.exp(z - m)
    return exp_d / np.sum(exp_d, axis=1, keepdims=True)


def _grad_hess_multiclass(
    pred: np.ndarray, y_idx: np.ndarray, n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Softmax CE loss: grad (n,K), hess (n,K) diagonal approximation.

    grad[i,c] = p[i,c] - onehot(y[i], c)
    hess[i,c] = p[i,c] * (1 - p[i,c])   # XGBoost diagonal approx
    """
    p = _softmax(pred)
    y_onehot = np.zeros((len(y_idx), n_classes), dtype=np.float64)
    y_onehot[np.arange(len(y_idx)), y_idx] = 1.0
    grad = p - y_onehot
    hess = np.maximum(p * (1.0 - p), 1e-16)
    return grad, hess


def _validate_multiclass_y(df, y_col: str) -> tuple[int, int]:
    """Ensure y in {0, 1, ..., K-1} (0-indexed).

    Returns (n_classes, n_valid_non_null). Raises ValueError otherwise.
    Explicitly rejects 1-indexed {1..K} (сообщение объясняет).
    """
    seen: set[int] = set()
    n_valid = 0
    for b in df._iter_filtered_batches():
        y = _column_to_f64(b.column(y_col))
        for v in y[~np.isnan(y)]:
            fv = float(v)
            if fv != int(fv):
                raise ValueError(
                    f"fit_gbdt_multiclass: y must be integer classes; "
                    f"got non-integer {fv}"
                )
            seen.add(int(fv))
            n_valid += 1
    if not seen:
        raise ValueError("fit_gbdt_multiclass: no valid (non-null) y")
    K = len(seen)
    expected = set(range(K))
    if seen != expected:
        raise ValueError(
            f"fit_gbdt_multiclass: y must be {{0, 1, ..., K-1}} "
            f"(0-indexed); got classes {sorted(seen)}. "
            f"1-indexed (e.g. {{1, 2, 3}}) не поддерживается — "
            f"сдвиньте y на -1 перед вызовом."
        )
    return K, n_valid


def _iter_class_enriched(
    df, X_cols, grad_all: np.ndarray, hess_all: np.ndarray,
    cur_leaf_c: np.ndarray, c: int,
    codes_mm=None, nan_mask=None,
):
    """Per-class batch iterator with __g, __h, __leaf for _AllNodeHistOp.

    v0.9.87-P1a-1b: if codes_mm provided, yields __code_0..__code_{d-1}
    and __nan_mask (for _AllNodeHistOpCodes) instead of raw X columns.
    Faster: reads uint8 codes from memmap, no _stack_numeric.
    """
    off = 0
    d = len(X_cols)
    for b in df._iter_filtered_batches():
        n = b.num_rows
        if codes_mm is not None:
            codes_b = np.asarray(codes_mm[off:off + n])
            nm_b = np.asarray(nan_mask[off:off + n])
            g = np.where(~nm_b, grad_all[off:off + n, c], 0.0)
            h = np.where(~nm_b, hess_all[off:off + n, c], 0.0)
            lf = cur_leaf_c[off:off + n]
            cols = [pa.array(codes_b[:, j], type=pa.uint8())
                    for j in range(d)]
            cols.append(pa.array(g, type=pa.float64()))
            cols.append(pa.array(h, type=pa.float64()))
            cols.append(pa.array(lf, type=pa.int64()))
            cols.append(pa.array(nm_b))
            names = ([f"__code_{j}" for j in range(d)]
                     + ["__g", "__h", "__leaf", "__nan_mask"])
        else:
            X = _stack_numeric(b, X_cols)
            valid = ~np.isnan(X).any(axis=1)
            g = np.where(valid, grad_all[off:off + n, c], 0.0)
            h = np.where(valid, hess_all[off:off + n, c], 0.0)
            lf = cur_leaf_c[off:off + n]
            cols = [b.column(cc) for cc in X_cols]
            cols.append(pa.array(g, type=pa.float64()))
            cols.append(pa.array(h, type=pa.float64()))
            cols.append(pa.array(lf, type=pa.int64()))
            names = list(X_cols) + ["__g", "__h", "__leaf"]
        yield pa.RecordBatch.from_arrays(cols, names=names)
        off += n




def _iter_x_with_codes(df, X_cols, codes_mm, nan_mask, batch_size=None):
    """Yield batches with __code_* + __nan_mask only (no raw X, no __g/__h).

    Used by _AssignLeavesOpCodes path; cheaper than _iter_class_enriched.
    """
    off = 0
    d = len(X_cols)
    for b in df._iter_filtered_batches(batch_size=batch_size):
        n = b.num_rows
        codes_b = np.asarray(codes_mm[off:off + n])
        nm_b = np.asarray(nan_mask[off:off + n])
        code_cols = [pa.array(codes_b[:, j], type=pa.uint8())
                     for j in range(d)]
        code_names = [f"__code_{j}" for j in range(d)]
        yield pa.RecordBatch.from_arrays(
            code_cols + [pa.array(nm_b)],
            names=code_names + ["__nan_mask"],
        )
        off += n

def _iter_class_for_leaves(
    df, X_cols, grad_all: np.ndarray, hess_all: np.ndarray,
    cur_leaf_c: np.ndarray, c: int,
):
    """Per-class: __g, __h, __leaf only (no X) — for _LeafValuesOp."""
    off = 0
    for b in df._iter_filtered_batches():
        n = b.num_rows
        X = _stack_numeric(b, X_cols)
        valid = ~np.isnan(X).any(axis=1)
        g = np.where(valid, grad_all[off:off + n, c], 0.0)
        h = np.where(valid, hess_all[off:off + n, c], 0.0)
        lf = cur_leaf_c[off:off + n]
        yield pa.RecordBatch.from_arrays(
            [pa.array(g, type=pa.float64()),
             pa.array(h, type=pa.float64()),
             pa.array(lf, type=pa.int64())],
            names=["__g", "__h", "__leaf"],
        )
        off += n


@dataclass
class GBDTMulticlassClassifier:
    """GBDT multiclass classifier (softmax loss).

    y must be {0, 1, ..., K-1} (0-indexed, K = n_classes).
    predict_proba → (n, K); predict → (n,) int64 in {0..K-1}.

    Fitted via per-class loop (K trees per round). Binary case
    (K == 2) should use fit_gbdt_classifier (existing) — this class
    requires K >= 3.
    """

    trees_: list  # list[n_estimators] of list[K] tree dicts
    base_prediction_: np.ndarray  # (K,), zeros
    feature_names_: list
    target_col_: str
    mn_: np.ndarray
    mx_: np.ndarray
    bins_: int
    learning_rate_: float
    n_estimators_: int
    max_depth_: int
    n_features_in_: int
    n_classes_: int
    fitted_: bool = True

    def _raw_score_np(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"predict: X has {X.shape[1]} features, "
                f"expected {self.n_features_in_}"
            )
        n = X.shape[0]
        pred = np.tile(self.base_prediction_, (n, 1))
        for trees_round in self.trees_:
            for c in range(self.n_classes_):
                tree = trees_round[c]
                leaf = _assign_leaf_from_array(
                    X, tree["structure"], self.mn_, self.mx_, self.bins_,
                )
                pred[:, c] += self.learning_rate_ * _leaf_vector(
                    tree["leaf_values"], leaf,
                )
        return pred

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            return _softmax(self._raw_score_np(X))
        return self._predict_df(X, proba_only=True)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            return _softmax(
                self._raw_score_np(X)
            ).argmax(axis=1).astype(np.int64)
        return self._predict_df(X, proba_only=False)

    def _predict_df(self, X, *, proba_only: bool):
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

        K = self.n_classes_
        target = self.target_col_
        pred_col = f"{target}_pred"
        prob_cols = [f"{target}_prob_{c}" for c in range(K)]
        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        out_fields.append(pa.field(pred_col, pa.int64(), nullable=True))
        if proba_only:
            for pc in prob_cols:
                out_fields.append(pa.field(pc, pa.float64(), nullable=True))
        out_schema = pa.schema(out_fields)
        model = self

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, model.feature_names_)
                n = Xm.shape[0]
                nan_rows = np.isnan(Xm).any(axis=1)
                labels = np.full(n, -1, dtype=np.int64)
                probs_full = np.full((n, K), np.nan, dtype=np.float64)
                if (~nan_rows).any():
                    raw = model._raw_score_np(Xm[~nan_rows])
                    probs = _softmax(raw)
                    labels[~nan_rows] = probs.argmax(axis=1).astype(np.int64)
                    probs_full[~nan_rows] = probs
                arrays = [b.column(c) for c in X._columns]
                arrays.append(pa.array(
                    labels, type=pa.int64(),
                    mask=nan_rows if nan_rows.any() else None,
                ))
                if proba_only:
                    for c in range(K):
                        col = probs_full[:, c]
                        mask = np.isnan(col)
                        arrays.append(pa.array(
                            col, type=pa.float64(),
                            mask=mask if mask.any() else None,
                        ))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )

    def state_nbytes(self) -> int:
        """Exact byte count of persistent state: trees + min/max + header.

        Tree representation: dict[int, tuple|None] (structure) +
        dict[int, float] (leaf_values). CPython per-node overhead
        dominates; this is exact via _deep_nbytes.
        """
        total = _STATE_HEADER_BYTES
        total += self.mn_.nbytes + self.mx_.nbytes
        total += int(self.base_prediction_.nbytes)
        total += _trees_nbytes(self.trees_)
        return int(total)

    def to_dict(self) -> dict:
        return {
            "loss": "multiclass_softmax",
            "n_classes": int(self.n_classes_),
            "n_estimators": int(self.n_estimators_),
            "n_trees_per_round": self.n_classes_,
            "max_depth": int(self.max_depth_),
            "learning_rate": float(self.learning_rate_),
            "bins": int(self.bins_),
            "feature_names": self.feature_names_,
            "target_col": self.target_col_,
        }


def _fit_boosting_multiclass(
    df, X_cols, y_col, *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    bins: int,
    min_child_weight: float,
    reg_lambda: float,
    min_gain: float,
    memory_guard: Optional[MemoryGuard],
) -> dict:
    """Multiclass softmax boosting, per-class loop. Reuses _AllNodeHistOp,
    _AssignLeavesOp, _LeafValuesOp (no new ops). K× cost vs binary.
    """
    X_cols = _validate_fit_args(
        df, X_cols, y_col,
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, bins=bins,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
    )
    K, n_valid = _validate_multiclass_y(df, y_col)
    if K < 3:
        raise ValueError(
            f"fit_gbdt_multiclass: need >= 3 classes; got {K}. "
            f"Use fit_gbdt_classifier for binary."
        )

    mg = memory_guard or df._memory_guard
    n_total = df.count()
    if n_total == 0:
        raise ValueError("fit_gbdt_multiclass: no rows")

    # v0.9.87-P1a-4: pred, grad, hess, codes, AND y_all all on disk
    # (SpillDir memmaps). Persistent RAM: batch buffers only
    # (~batch_size × (K+2) × 8 for _softmax temp, p[valid],
    # y_idx[valid]). Formula n·8·2 is a conservative upper bound
    # (worst case = batch covers full n). Was (K + 2) (P1a-3a),
    # (1 + 2K) (P1a-2), (1 + 4K) (v0.9.22).
    # True OOC for all n-sized state in multiclass fit.
    est_n_state = n_total * 8 * 2
    try:
        mg.check_state(est_n_state, name="gbdt.multiclass_state")
    except MemoryLimitExceeded:
        raise MemoryLimitExceeded(
            f"fit_gbdt_multiclass: state O(n·K) = {est_n_state} B "
            f"(n={n_total}, K={K}) > state_limit. Multiclass GBDT "
            f"requires pred, grad, hess per row per class. "
            f"See §13.4/92."
        )

    # v0.9.87-P1a-1b: one-time binning -> codes_mm (uint8 memmap);
    # subsequent hist/assign passes use codes, no per-pass re-stack.
    mn, mx, range_, codes_mm, nan_mask, binning_spill = _binning_pass(
        df, X_cols, bins=int(bins), memory_guard=mg, guard=mg)

    try:
        # 2. y as int64 (with -1 for null) — memmap-backed
        # (v0.9.87-P1a-4). Removes last n-sized RAM array in
        # multiclass fit (was y_all = np.full(n_total, -1)).
        # Explicit [:]=-1 init because numpy memmap mode="w+" returns
        # 0.0-default bytes on fresh pages, not our -1 sentinel.
        _y_path = binning_spill.path("y.bin")
        y_all = np.memmap(
            str(_y_path), dtype=np.int64, mode="w+",
            shape=(n_total,))
        y_all[:] = -1
        offset = 0
        for b in df._iter_filtered_batches():
            y = _column_to_f64(b.column(y_col))
            valid = ~np.isnan(y)
            y_all[offset:offset + b.num_rows] = np.where(
                valid, y, -1).astype(np.int64)
            offset += b.num_rows
        y_all.flush()

        # 3. Init pred — zeros, memmap-backed (v0.9.87-P1a-3b).
        # mode="w+" (numpy memmap): seeks to n*K*8-1, writes one
        # zero byte → file at correct size; on POSIX the rest is
        # sparse-zero. Subsequent reads of unwritten regions return
        # 0.0, matching np.zeros semantics. Verified by accuracy +
        # determinism tests (predictions identical to pre-P1a-3b).
        base_pred = np.zeros(K, dtype=np.float64)
        _pred_path = binning_spill.path("pred.bin")
        pred = np.memmap(
            str(_pred_path), dtype=np.float64, mode="w+",
            shape=(n_total, K))

        trees: list = []

        for round_i in range(n_estimators):
            # v0.9.87-P1a-2: grad/hess on disk (memmap); 2x RAM cut.
            # Reused across round only. Cleaned up after this round.
            _g_path = binning_spill.path(f"grad_r{round_i:04d}.bin")
            _h_path = binning_spill.path(f"hess_r{round_i:04d}.bin")
            grad_all = np.memmap(
                str(_g_path), dtype=np.float64, mode="w+",
                shape=(n_total, K))
            hess_all = np.memmap(
                str(_h_path), dtype=np.float64, mode="w+",
                shape=(n_total, K))
            offset = 0
            for b in df._iter_filtered_batches():
                n = b.num_rows
                # v0.9.87-P1a-3b/P1a-4: pred + y_all are memmap;
                # np.asarray materializes per-batch (batch_size ×
                # (K+1) × 8 bytes) so softmax runs on contiguous RAM.
                p = np.asarray(pred[offset:offset + n])
                y_idx = np.asarray(y_all[offset:offset + n])
                valid = (~nan_mask[offset:offset + n]) & (y_idx >= 0)
                if valid.any():
                    g_valid, h_valid = _grad_hess_multiclass(
                        p[valid], y_idx[valid], K,
                    )
                    grad_all[offset:offset + n][valid] = g_valid
                    hess_all[offset:offset + n][valid] = h_valid
                offset += n
            grad_all.flush()
            hess_all.flush()

            # 5. Fit K trees per class — codes-based hist+assign
            # P1a-3a: only round_trees kept (per-class tree dicts,
            # small). round_leaf_maps + round_cur_leaves removed —
            # pred update moved inside loop, cur_leaf_c freed per class.
            round_trees = []
            for c in range(K):
                tree = {0: None}
                cur_leaf_c = np.zeros(n_total, dtype=np.int64)
                next_node_id = 1

                for level in range(max_depth):
                    internal = [nid for nid, t in tree.items()
                                if t is None]
                    if not internal:
                        break

                    hist_op = _AllNodeHistOpCodes(X_cols, internal, bins)
                    hist_res = reduce(
                        _iter_class_enriched(
                            df, X_cols, grad_all, hess_all,
                            cur_leaf_c, c,
                            codes_mm=codes_mm, nan_mask=nan_mask,
                        ),
                        hist_op, mg, log=False,
                    )
                    gh = hist_res["g_hist"]
                    hh = hist_res["h_hist"]
                    gt = hist_res["g_tot"]
                    ht = hist_res["h_tot"]
                    nt = hist_res["n_tot"]

                    for slot, nid in enumerate(internal):
                        if ht[slot] < min_child_weight * 2:
                            continue
                        best = _find_best_split(
                            slot, gh[slot], hh[slot],
                            float(gt[slot]), float(ht[slot]),
                            int(nt[slot]),
                            bins, min_child_weight, min_gain,
                            reg_lambda,
                        )
                        if best is None:
                            continue
                        j, thr, gain = best
                        left_id = next_node_id
                        right_id = next_node_id + 1
                        next_node_id += 2
                        tree[nid] = (j, thr, left_id, right_id,
                                     float(gain))
                        tree[left_id] = None
                        tree[right_id] = None

                    assign_op = _AssignLeavesOpCodes(X_cols, tree)
                    cur_leaf_c = reduce(
                        _iter_x_with_codes(
                            df, X_cols, codes_mm, nan_mask),
                        assign_op, mg, log=False,
                    )

                # 6. Leaf values (unchanged — reads __g/__h/__leaf)
                lv_op = _LeafValuesOp(reg_lambda=reg_lambda)
                leaf_values_c = reduce(
                    _iter_class_for_leaves(
                        df, X_cols, grad_all, hess_all,
                        cur_leaf_c, c,
                    ),
                    lv_op, mg, log=False,
                )

                round_trees.append({
                    "structure": dict(tree),
                    "leaf_values": dict(leaf_values_c),
                })

                # v0.9.87-P1a-3a: update pred FOR THIS CLASS immediately
                # (was: after all K classes). Bit-identical: grad_all/
                # hess_all computed once at round start; hist/assign for
                # class c reads grad_all[:, c] only; pred[:, c] update is
                # column-independent of other classes.
                if leaf_values_c:
                    max_lid = max(leaf_values_c.keys())
                    arr = np.zeros(max_lid + 1, dtype=np.float64)
                    for lid, v in leaf_values_c.items():
                        arr[lid] = v
                    pred[:, c] += learning_rate * arr[cur_leaf_c]

                # Free cur_leaf_c (was: kept in round_cur_leaves list,
                # K copies × n·8 = K·n·8 bytes; now: one at a time).
                del cur_leaf_c

            # v0.9.87-P1a-2: free grad/hess memmap files after round.
            del grad_all, hess_all
            binning_spill.cleanup_file(_g_path)
            binning_spill.cleanup_file(_h_path)

            trees.append(round_trees)

        # v0.9.87-P1a-3b/P1a-4: flush + free pred + y_all memmaps
        # before cleanup. del releases mmap handle so cleanup_file
        # unlinks safely (Linux OK either way; Windows needs
        # explicit release).
        pred.flush()
        del pred
        binning_spill.cleanup_file(_pred_path)
        y_all.flush()
        del y_all
        binning_spill.cleanup_file(_y_path)
    finally:
        try:
            del codes_mm
        except Exception:
            pass
        binning_spill.__exit__(None, None, None)

    return {
        "trees_": trees,
        "base_prediction_": base_pred,
        "feature_names_": list(X_cols),
        "target_col_": y_col,
        "mn_": mn,
        "mx_": mx,
        "bins_": int(bins),
        "learning_rate_": float(learning_rate),
        "n_estimators_": int(n_estimators),
        "max_depth_": int(max_depth),
        "n_features_in_": len(X_cols),
        "n_classes_": int(K),
    }


def fit_gbdt_multiclass(
    df, X_cols, y_col: str,
    *, n_estimators: int = 100, max_depth: int = 3,
    learning_rate: float = 0.1, bins: int = 32,
    min_child_weight: float = 1.0, reg_lambda: float = 1.0,
    min_gain: float = 0.0,
    memory_guard: Optional[MemoryGuard] = None,
) -> GBDTMulticlassClassifier:
    """GBDT multiclass classification (softmax) — v0.7.8.

    y must be {0, 1, ..., K-1}, K >= 3. For binary — fit_gbdt_classifier.
    """
    out = _fit_boosting_multiclass(
        df, X_cols, y_col,
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, bins=bins,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
        min_gain=min_gain, memory_guard=memory_guard,
    )
    return GBDTMulticlassClassifier(**out)

def compute_feature_importance(
    model,
    *,
    importance_type: str = "gain",
    top_k: "int | None" = None,
) -> dict:
    """Compute feature importance from GBDT model (v0.7.9).

    - model: GBDTRegressor | GBDTClassifier | GBDTMulticlassClassifier
    - importance_type:
        'gain'  — sum of split gain per feature (default, XGBoost-style).
                  Larger gain = more useful split.
        'split' — count of splits per feature (v0.7.4a semantics).
                  Simpler, но не учитывает качество split.
    - top_k: keep only top-k features (by importance). None = all.

    Returns dict {feature_name: importance} sorted by importance
    ascending (для horizontal barh — biggest наверху).

    Multiclear GBDT: gain summed across all (tree, class).
    Binary + regression: standard, all trees.
    """
    if importance_type not in ("gain", "split"):
        raise ValueError(
            f"compute_feature_importance: importance_type must be "
            f"'gain' or 'split'; got {importance_type!r}"
        )
    trees = getattr(model, "trees_", None)
    names = getattr(model, "feature_names_", None)
    if trees is None or names is None:
        raise TypeError(
            "compute_feature_importance: model must be a GBDT "
            "(trees_, feature_names_)"
        )

    imp: dict[str, float] = {n: 0.0 for n in names}

    def _iter_trees():
        """Yield tree dicts. Handles binary (list[tree]) and multiclass
        (list[list[tree]])."""
        if not trees:
            return
        first = trees[0]
        if isinstance(first, list):
            # Multiclass: trees_ = list[list[tree]]
            for round_trees in trees:
                for tree in round_trees:
                    yield tree
        else:
            # Binary / regression: trees_ = list[tree]
            for tree in trees:
                yield tree

    for tree in _iter_trees():
        structure = tree["structure"]
        for _nid, spec in structure.items():
            if spec is None:
                continue
            j = spec[0]
            if importance_type == "split":
                imp[names[j]] += 1.0
            else:  # gain
                # 5-tuple: (j, thr, left, right, gain)
                if len(spec) >= 5:
                    imp[names[j]] += float(spec[4])
                # else: 4-tuple (legacy) — gain not available; split only
                # count it as a split, not gain. Better than fail.

    items = sorted(imp.items(), key=lambda kv: kv[1])
    if top_k is not None:
        items = items[-top_k:]
    return dict(items)

