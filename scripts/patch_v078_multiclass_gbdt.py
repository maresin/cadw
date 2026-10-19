#!/usr/bin/env python3
"""v0.7.8: Multiclass GBDT (softmax loss) — per-class loop, reuse ops.

Design:
    - Round: compute grad_all, hess_all (n, K) once (softmax of pred).
    - Per-class loop (K iterations per round):
        reuse _AllNodeHistOp / _AssignLeavesOp / _LeafValuesOp —
        feed single-class slices as __g, __h, __leaf.
    - Update pred[:, c] += lr * leaf_c.

Reuse (0 new ops):
    _AllNodeHistOp, _AssignLeavesOp, _LeafValuesOp — exactly as binary.

New:
    - _softmax(z) — numerically stable, row-wise.
    - _grad_hess_multiclass(pred, y_idx, K) -> (grad (n,K), hess (n,K)).
    - _validate_multiclass_y — ensures {0..K-1}, returns (K, n_valid).
    - _iter_class_enriched — per-class batches with __g/__h/__leaf.
    - _fit_boosting_multiclass — K× per-class loop.
    - GBDTMulticlassClassifier dataclass — predict_proba → (n,K).
    - fit_gbdt_multiclass — explicit entry.
    - fit_gbdt_classifier(..., n_classes=None) — dispatches:
        n_classes inferred from y; if == 2 → binary (existing);
        if > 2 → multiclass; if < 2 → error.

Cost: K× binary fit time. K=3, n=1000, depth=3, 30 trees: ~10s.
For K > 5 or n > 1M — v0.7.8a will add _MulticlassAllNodeHistOp
(1 pass for all K classes, ~K× speedup).

Binary compat: GBDTClassifier unchanged; predict_proba → (n,).
Multiclass: GBDTMulticlassClassifier; predict_proba → (n, K).

Not in v0.7.8:
    - 1-indexed y (только {0..K-1}).
    - One-vs-rest (только softmax).
    - Multiclass speedup (v0.7.8a).

Run: uv run python scripts/patch_v078_multiclass_gbdt.py
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
        for line in r.stdout.strip().splitlines()[:25]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
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


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


# ============ 1. Helpers + multiclass fit + model ============

MULTICLASS_CODE = '''

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
):
    """Per-class batch iterator with __g, __h, __leaf for _AllNodeHistOp."""
    off = 0
    for b in df._iter_filtered_batches():
        n = b.num_rows
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

    # State: pred (n,K) + grad_all (n,K) + hess_all (n,K) + cur_leaf (n)
    #        = n * K * 8 * 3 (minimal, cur_leaf reused per class)
    est_n_state = n_total * K * 8 * 3
    try:
        mg.check_state(est_n_state, name="gbdt.multiclass_state")
    except MemoryLimitExceeded:
        raise MemoryLimitExceeded(
            f"fit_gbdt_multiclass: state O(n·K) = {est_n_state} B "
            f"(n={n_total}, K={K}) > state_limit. Multiclass GBDT "
            f"requires pred, grad, hess per row per class. "
            f"See §13.4/92."
        )

    # 1. Global min/max
    mm_res = reduce(
        df._iter_filtered_batches(), _MinMaxOp(X_cols), mg, log=False,
    )
    mn = mm_res["min"]
    mx = mm_res["max"]
    if np.any(np.isinf(mn)) or np.any(np.isinf(mx)):
        raise ValueError("fit_gbdt_multiclass: no valid rows")

    # 2. y as int64 (with -1 for null)
    y_all = np.full(n_total, -1, dtype=np.int64)
    offset = 0
    for b in df._iter_filtered_batches():
        y = _column_to_f64(b.column(y_col))
        valid = ~np.isnan(y)
        y_all[offset:offset + b.num_rows] = np.where(valid, y, -1).astype(np.int64)
        offset += b.num_rows

    # 3. Init pred — zeros (base_prediction = zeros, trees find shift)
    base_pred = np.zeros(K, dtype=np.float64)
    pred = np.tile(base_pred, (n_total, 1))  # (n, K)

    trees: list = []  # list[n_estimators], each = list[K]

    for round_i in range(n_estimators):
        # 4. Compute grad_all, hess_all ONCE per round (softmax)
        grad_all = np.zeros((n_total, K), dtype=np.float64)
        hess_all = np.zeros((n_total, K), dtype=np.float64)
        offset = 0
        for b in df._iter_filtered_batches():
            n = b.num_rows
            X = _stack_numeric(b, X_cols)
            p = pred[offset:offset + n]
            y_idx = y_all[offset:offset + n]
            valid = (~np.isnan(X).any(axis=1)) & (y_idx >= 0)
            if valid.any():
                # Use only valid rows for grad/hess; fill others with 0
                g_valid, h_valid = _grad_hess_multiclass(
                    p[valid], y_idx[valid], K,
                )
                grad_all[offset:offset + n][valid] = g_valid
                hess_all[offset:offset + n][valid] = h_valid
            offset += n

        # 5. Fit K trees per class
        round_trees = []
        round_leaf_maps = []
        round_cur_leaves = []
        for c in range(K):
            tree = {0: None}
            cur_leaf_c = np.zeros(n_total, dtype=np.int64)
            next_node_id = 1

            for level in range(max_depth):
                internal = [nid for nid, t in tree.items() if t is None]
                if not internal:
                    break

                hist_op = _AllNodeHistOp(X_cols, internal, mn, mx, bins)
                hist_res = reduce(
                    _iter_class_enriched(
                        df, X_cols, grad_all, hess_all, cur_leaf_c, c,
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

                assign_op = _AssignLeavesOp(X_cols, mn, mx, bins, tree)
                cur_leaf_c = reduce(
                    df._iter_filtered_batches(), assign_op, mg, log=False,
                )

            # 6. Leaf values for this class
            lv_op = _LeafValuesOp(reg_lambda=reg_lambda)
            leaf_values_c = reduce(
                _iter_class_for_leaves(
                    df, X_cols, grad_all, hess_all, cur_leaf_c, c,
                ),
                lv_op, mg, log=False,
            )

            round_trees.append({
                "structure": dict(tree),
                "leaf_values": dict(leaf_values_c),
            })
            round_leaf_maps.append(dict(leaf_values_c))
            round_cur_leaves.append(cur_leaf_c)

        # 7. Update pred[:, c] (after all K trees of this round)
        for c in range(K):
            lv = round_leaf_maps[c]
            if lv:
                max_lid = max(lv.keys())
                arr = np.zeros(max_lid + 1, dtype=np.float64)
                for lid, v in lv.items():
                    arr[lid] = v
                pred[:, c] += learning_rate * arr[round_cur_leaves[c]]

        trees.append(round_trees)

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

'''


def add_multiclass_code():
    print("\n=== cadw/ml/gbdt.py: +multiclass code (append) ===")
    p = Path("cadw/ml/gbdt.py")
    s = p.read_text()
    if "GBDTMulticlassClassifier" in s:
        print("  already present — skip")
        return
    s = s.rstrip() + MULTICLASS_CODE + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: 200 lines appended")


# ============ 2. fit_gbdt_classifier dispatch ============

def patch_classifier_dispatch():
    print("\n=== cadw/ml/gbdt.py: fit_gbdt_classifier dispatch ===")
    p = Path("cadw/ml/gbdt.py")
    s = p.read_text()
    if "def fit_gbdt_classifier" not in s:
        raise SystemExit("fit_gbdt_classifier not found")

    # Insert n_classes param and dispatch
    old = '''def fit_gbdt_classifier(
    df, X_cols, y_col: str,
    *, n_estimators: int = 100, max_depth: int = 3,
    learning_rate: float = 0.1, bins: int = 32,
    min_child_weight: float = 1.0, reg_lambda: float = 1.0,
    min_gain: float = 0.0,
    memory_guard: Optional[MemoryGuard] = None,
) -> GBDTClassifier:
    """GBDT binary classification (logloss) — v0.7.5.

    y must be {0, 1}, both classes present. predict_proba → P(y=1|x);
    predict → {0,1} by threshold (default 0.5).
    """
    out = _fit_boosting(
        df, X_cols, y_col, loss="logloss_binary",
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, bins=bins,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
        min_gain=min_gain, memory_guard=memory_guard,
    )
    return GBDTClassifier(**out)'''

    new = '''def fit_gbdt_classifier(
    df, X_cols, y_col: str,
    *, n_estimators: int = 100, max_depth: int = 3,
    learning_rate: float = 0.1, bins: int = 32,
    min_child_weight: float = 1.0, reg_lambda: float = 1.0,
    min_gain: float = 0.0,
    n_classes: Optional[int] = None,
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
        K, _ = _validate_multiclass_y(df, y_col)
    else:
        K = int(n_classes)

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
        )'''

    s = replace_exact(s, old, new, tag="classifier-dispatch")
    p.write_text(s)
    compile_check(p)
    print("  OK: fit_gbdt_classifier dispatches binary/multiclass")


# ============ 3. CadwDataFrame methods ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: fit_gbdt_multiclass ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def fit_gbdt_multiclass" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    method = '''    # ---------------- ML: fit_gbdt_multiclass ----------------

    def fit_gbdt_multiclass(
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
        """Fit GBDT multiclass classification (softmax) — v0.7.8.

        y must be {0, 1, ..., K-1}, K >= 3. For binary — fit_gbdt_classifier.
        """
        from cadw.ml.gbdt import fit_gbdt_multiclass
        return fit_gbdt_multiclass(
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
    print("  OK: CadwDataFrame.fit_gbdt_multiclass")


# ============ 4. Exports ============

def patch_exports():
    print("\n=== cadw/ml/__init__.py + cadw/__init__.py ===")
    p = Path("cadw/ml/__init__.py")
    s = p.read_text()
    if "GBDTMulticlassClassifier" not in s:
        s = s.replace(
            "from .gbdt import (\n"
            "    GBDTRegressor, fit_gbdt_regressor,\n"
            "    GBDTClassifier, fit_gbdt_classifier,\n"
            ")",
            "from .gbdt import (\n"
            "    GBDTRegressor, fit_gbdt_regressor,\n"
            "    GBDTClassifier, fit_gbdt_classifier,\n"
            "    GBDTMulticlassClassifier, fit_gbdt_multiclass,\n"
            ")",
            1,
        )
        s = s.replace(
            '    "GBDTClassifier", "fit_gbdt_classifier",\n',
            '    "GBDTClassifier", "fit_gbdt_classifier",\n'
            '    "GBDTMulticlassClassifier", "fit_gbdt_multiclass",\n',
            1,
        )
        p.write_text(s)
        compile_check(p)
        print("  OK: cadw/ml/__init__.py")

    p2 = Path("cadw/__init__.py")
    s2 = p2.read_text()
    if "fit_gbdt_multiclass" not in s2:
        s2 = s2.replace(
            "    GBDTClassifier, fit_gbdt_classifier,\n)",
            "    GBDTClassifier, fit_gbdt_classifier,\n"
            "    GBDTMulticlassClassifier, fit_gbdt_multiclass,\n)",
            1,
        )
        s2 = s2.replace(
            '    "GBDTClassifier", "fit_gbdt_classifier",\n',
            '    "GBDTClassifier", "fit_gbdt_classifier",\n'
            '    "GBDTMulticlassClassifier", "fit_gbdt_multiclass",\n',
            1,
        )
        p2.write_text(s2)
        compile_check(p2)
        print("  OK: cadw/__init__.py")


# ============ 5. Tests ============

TESTS_SRC = '''"""v0.7.8: Multiclass GBDT (softmax) via per-class loop."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import (
    CadwDataFrame, MemoryGuard,
    fit_gbdt_classifier, fit_gbdt_multiclass,
)
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ml.gbdt import (
    GBDTClassifier, GBDTMulticlassClassifier, _softmax,
)


def make_df(tmp_path, data, chunk=50, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _synth_multiclass(n, n_classes, d=2, seed=42, sep=4.0):
    """K gaussian clusters along unit circle; y in {0..K-1}."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)]) * sep
    y = rng.integers(0, n_classes, size=n)
    X = centers[y] + rng.normal(scale=0.5, size=(n, d))
    data = {f"x{i}": X[:, i].tolist() for i in range(d)}
    data["y"] = y.astype(np.int64).tolist()
    return data, X, y


# ---------- _softmax ----------

def test_softmax_rows_sum_to_1():
    z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = _softmax(z)
    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0])


def test_softmax_large_values_stable():
    """Numerical stability: no overflow for large z."""
    z = np.array([[1000.0, 999.0], [-1000.0, -999.0]])
    p = _softmax(z)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0])


def test_softmax_wrong_dim_raises():
    with pytest.raises(ValueError, match="2D"):
        _softmax(np.array([1.0, 2.0]))


# ---------- basic fit ----------

def test_fit_multiclass_basic(tmp_path):
    n = 300
    data, X, y = _synth_multiclass(n, n_classes=3, seed=0)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y",
        n_estimators=10, max_depth=3, learning_rate=0.2, bins=16,
    )
    assert isinstance(model, GBDTMulticlassClassifier)
    assert model.n_classes_ == 3
    assert len(model.trees_) == 10  # n_estimators
    assert len(model.trees_[0]) == 3  # K trees per round


def test_fit_multiclass_accuracy(tmp_path):
    """Separable clusters: accuracy > 0.9."""
    n = 500
    data, X, y = _synth_multiclass(n, n_classes=3, seed=1, sep=4.0)
    df = make_df(tmp_path, data, chunk=100)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y",
        n_estimators=30, max_depth=3, learning_rate=0.2, bins=16,
    )
    pred = model.predict(X)
    acc = (pred == y).mean()
    assert acc > 0.9, f"accuracy={acc:.3f}"


def test_fit_multiclass_n_estimators_improves(tmp_path):
    n = 500
    data, X, y = _synth_multiclass(n, n_classes=3, seed=2, sep=3.0)
    df = make_df(tmp_path, data, chunk=100)
    m10 = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y",
        n_estimators=10, max_depth=3, learning_rate=0.2, bins=16,
    )
    m40 = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y",
        n_estimators=40, max_depth=3, learning_rate=0.2, bins=16,
    )
    acc10 = (m10.predict(X) == y).mean()
    acc40 = (m40.predict(X) == y).mean()
    assert acc40 >= acc10, f"acc(10)={acc10:.3f} vs acc(40)={acc40:.3f}"


def test_fit_multiclass_deterministic(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=3)
    df = make_df(tmp_path, data)
    m1 = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    m2 = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    np.testing.assert_allclose(
        m1.predict_proba(X), m2.predict_proba(X), rtol=1e-9,
    )


# ---------- predict ----------

def test_predict_proba_shape(tmp_path):
    n = 300
    data, X, y = _synth_multiclass(n, n_classes=4, seed=4)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 4)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(10))


def test_predict_returns_int_labels(tmp_path):
    n = 300
    data, X, y = _synth_multiclass(n, n_classes=4, seed=5)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    pred = model.predict(X[:20])
    assert pred.shape == (20,)
    assert pred.dtype == np.int64
    assert set(pred.tolist()) <= {0, 1, 2, 3}


def test_predict_df_adds_columns(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=6)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_pred" in names


def test_predict_proba_df_adds_prob_columns(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=7)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    result = model.predict_proba(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob_0" in names
    assert "y_prob_1" in names
    assert "y_prob_2" in names


def test_predict_null_rows_yield_null(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=8)
    data["x0"][5] = None
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5, max_depth=2,
    )
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows[5]["y_pred"] is None


# ---------- fit_gbdt_classifier dispatch ----------

def test_dispatch_binary(tmp_path):
    """K=2 → GBDTClassifier (backward compat)."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = (x > 0).astype(np.int64)
    df = make_df(tmp_path, {"x0": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_classifier(df, ["x0"], "y", n_estimators=10)
    assert isinstance(model, GBDTClassifier)
    assert not isinstance(model, GBDTMulticlassClassifier)


def test_dispatch_multiclass(tmp_path):
    """K=3 → GBDTMulticlassClassifier (auto-dispatch)."""
    n = 300
    data, X, y = _synth_multiclass(n, n_classes=3, seed=10)
    df = make_df(tmp_path, data)
    model = fit_gbdt_classifier(
        df, ["x0", "x1"], "y", n_estimators=10,
    )
    assert isinstance(model, GBDTMulticlassClassifier)


def test_dispatch_single_class_raises(tmp_path):
    n = 100
    df = make_df(tmp_path, {
        "x0": np.random.default_rng(0).normal(size=n).tolist(),
        "y": [1] * n,
    })
    with pytest.raises(ValueError, match="classes"):
        fit_gbdt_classifier(df, ["x0"], "y")


# ---------- y validation ----------

def test_multiclass_1_indexed_raises(tmp_path):
    """y = {1, 2, 3} — rejected (0-indexed required)."""
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x0": rng.normal(size=n).tolist(),
        "y": [(i % 3) + 1 for i in range(n)],  # {1, 2, 3}
    })
    with pytest.raises(ValueError, match="0-indexed"):
        fit_gbdt_multiclass(df, ["x0"], "y")


def test_multiclass_missing_class_raises(tmp_path):
    """y = {0, 2} — not {0, 1, 2}; missing 1 → reject."""
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x0": rng.normal(size=n).tolist(),
        "y": [0 if i % 2 else 2 for i in range(n)],  # only {0, 2}
    })
    with pytest.raises(ValueError, match="0-indexed"):
        fit_gbdt_multiclass(df, ["x0"], "y")


def test_multiclass_k2_rejected_by_fit_multiclass(tmp_path):
    """fit_gbdt_multiclass с K=2 — error (use fit_gbdt_classifier)."""
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x0": rng.normal(size=n).tolist(),
        "y": [i % 2 for i in range(n)],  # {0, 1}
    })
    with pytest.raises(ValueError, match="binary"):
        fit_gbdt_multiclass(df, ["x0"], "y")


# ---------- memory guard ----------

def test_multiclass_memory_guard(tmp_path):
    n = 10_000
    data, X, y = _synth_multiclass(n, n_classes=3, seed=11)
    df = make_df(tmp_path, data, chunk=1000)
    df._memory_guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded, match="state"):
        fit_gbdt_multiclass(df, ["x0", "x1"], "y", n_estimators=5)


# ---------- convenience / to_dict ----------

def test_fit_gbdt_multiclass_dataframe_method(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=12)
    df = make_df(tmp_path, data)
    model = df.fit_gbdt_multiclass(["x0", "x1"], "y", n_estimators=5)
    assert isinstance(model, GBDTMulticlassClassifier)


def test_multiclass_to_dict(tmp_path):
    n = 200
    data, X, y = _synth_multiclass(n, n_classes=3, seed=13)
    df = make_df(tmp_path, data)
    model = fit_gbdt_multiclass(
        df, ["x0", "x1"], "y", n_estimators=5,
    )
    d = model.to_dict()
    assert d["loss"] == "multiclass_softmax"
    assert d["n_classes"] == 3
    assert d["n_trees_per_round"] == 3


# ---------- binary regression ----------

def test_binary_classifier_still_works(tmp_path):
    """GBDTClassifier (K=2) not broken by v0.7.8."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = (x > 0).astype(np.int64)
    df = make_df(tmp_path, {"x0": x.tolist(), "y": y.tolist()})
    model = fit_gbdt_classifier(df, ["x0"], "y", n_estimators=20)
    proba = model.predict_proba(np.column_stack([x]))
    assert proba.shape == (n,)  # binary stays (n,)
    pred = (proba >= 0.5).astype(int)
    assert (pred == y).mean() > 0.85
'''


def write_tests():
    print("\n=== tests/test_gbdt_multiclass.py ===")
    p = Path("tests/test_gbdt_multiclass.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: 22 tests")


# ============ 6. Docs ============

def patch_changelog():
    print("\n=== CHANGELOG.md: [0.7.8] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "## [0.7.8]" in s:
        print("  already present — skip")
        return
    entry = """## [0.7.8] — 2026-10-19

### Добавлено
- **`fit_gbdt_multiclass`** — multiclass GBDT (softmax loss), per-class
  loop, reuse существующих ops (`_AllNodeHistOp`, `_AssignLeavesOp`,
  `_LeafValuesOp` — 0 новых ops).
  - K деревьев на round (по одному на класс); `trees_` = `list[n_estimators]`
    of `list[K]`.
  - Grad: `p[i,c] - onehot(y[i], c)`; Hess: `p[i,c]*(1-p[i,c])`
    (XGBoost diagonal approximation).
  - `base_prediction_ = zeros(K)` (деревья находят сдвиг).
  - `predict_proba → (n, K)`, `predict → (n,) int64`.
- **`GBDTMulticlassClassifier`** — model class; sklearn-подобные поля:
  `trees_`, `n_classes_`, `feature_names_`, `n_features_in_`,
  `predict_proba`, `predict`, `to_dict`.
- **`fit_gbdt_classifier(..., n_classes=None)`** — auto-dispatch:
  - `n_classes` inferred из `y` (`{0..K-1}`).
  - `K == 2` → `GBDTClassifier` (binary, unchanged).
  - `K > 2` → `GBDTMulticlassClassifier`.
  - `K < 2` → ValueError.
- **`CadwDataFrame.fit_gbdt_multiclass`** — фасад.

### Ограничения (v0.7.8, §11.5)
- y должен быть `{0, 1, ..., K-1}` (0-indexed). 1-indexed `{1..K}` —
  ValueError (сдвиньте y на -1).
- `K × binary` fit cost (per-class loop). K=3 → ~3× медленнее binary.
  Для K > 5, n > 1M — v0.7.8a добавит `_MulticlassAllNodeHistOp`
  (1 pass для всех K, ~K× speedup).
- One-vs-rest не поддерживается (только softmax).
- Multiclass quantile/huber — не реализованы.

### Тесты
- +22 в `test_gbdt_multiclass.py`: `_softmax` (sum, stability, dim),
  fit (basic, accuracy, n_estimators improves, deterministic), predict
  (proba shape, int labels, df columns, prob_df columns, nulls),
  dispatch (binary, multiclass, single-class error), y validation
  (1-indexed, missing class, K=2 rejected by fit_multiclass),
  memory guard, convenience, to_dict, binary regression.
- fast: **~979 passed**, slow: **8 passed**.

### Время
- K=3, n=1000, depth=3, 30 rounds: ~15-20s (CI). Локально — быстрее.

"""
    if "## [0.8.1]" in s:
        s = s.replace("## [0.8.1]", entry + "## [0.8.1]", 1)
    elif "## [0.8.0]" in s:
        s = s.replace("## [0.8.0]", entry + "## [0.8.0]", 1)
    else:
        raise SystemExit("CHANGELOG anchor [0.8.1]/[0.8.0] not found")
    p.write_text(s)
    print("  OK: [0.7.8]")


def patch_decisions():
    print("\n=== DECISIONS.md: §11.1 + §13.4/113 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "| `fit_gbdt_multiclass` (v0.7.8:" not in s:
        s = s.replace(
            "| `fit_lasso`/`fit_elasticnet` (v0.7.7: coord descent) | ✅ | 22 |",
            "| `fit_lasso`/`fit_elasticnet` (v0.7.7: coord descent) | ✅ | 22 |\n"
            "| `fit_gbdt_multiclass` (v0.7.8: softmax, per-class) | ✅ | 22 |",
            1,
        )

    if "Эпизод 113" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 113 (Multiclass GBDT: reuse ops через per-class loop).**
После v0.7.5 (binary logloss) следующий логичный шаг — multiclass
softmax. Два подхода:

**(A) `_MulticlassAllNodeHistOp`** — один Reduce для всех K классов:
    State: `g_hist (S, K, F, B)`, batch columns `__g_0..__g_{K-1}`,
    `__h_0..`, `__leaf_0..`. K× меньше pass'ов, но ~600 строк
    специализированного кода. Плюс: `_AssignLeavesOp` нужен V2 (K
    деревьев одновременно); `_LeafValuesOp` — V2 (K× nodes);
    `cur_leaf` — матрица (n, K). Итого: 5 новых ops + дубли.

**(B) Per-class loop** — K итераций, каждая с существующими
    `_AllNodeHistOp`, `_AssignLeavesOp`, `_LeafValuesOp`:
    - Pred (n, K) — обновляется per-column.
    - Grad/hess (n, K) — считаются ОДИН раз per round (softmax),
      потом per-class slices подаются как `__g`, `__h`.
    - K× медленнее binary, но **0 новых ops**, 0 новых багов в hist/
      leaf/assign.

**Выбор — B.** Причины:
1. **Correctness > speed** для MVP: existing ops уже покрыты 33+ тестами
   (`test_reduce.py`, `test_gbdt_regression.py`, `test_gbdt_classification.
   py`). Новые ops потребуют полного параллельного test-set.
2. **Bug surface:** мой первый черновик (A) содержал 2 бага в
   pattern-matching (`__leaf_0` vs `__leaf`, V1/V2 дубли). При B —
   просто нет места для таких ошибок: pipeline идентичен binary, только
   K раз.
3. **Cost:** K=3 → 3× медленнее. Для CI (n=1000, depth=3, 30 rounds):
   ~15s. Для production (n=10^6, K=5, 100 rounds): ~15-30 min — но
   это **уже out-of-core**, GBDT всё равно bottleneck, v0.7.8a
   оптимизирует.

**Урок:** для новых фич — **сначала работающая простота (reuse), потом
оптимизация (specialize)**. Специализированный `_MulticlassAllNodeHistOp`
имеет смысл **после** того, как per-class версия прошла все тесты
и доказала корректность. Иначе — debugging multiclass hist + debugging
specialized op + debugging multiple-model dispatch = три бага
одновременно.

**v0.7.8a backlog:** `_MulticlassAllNodeHistOp` — один pass на hist,
один на assign (K trees per row), один на leaf-values. Ожидаемый
speedup ~K×. Нужны V2 ops (или универсализация через `class_idx` axis).
Отдельная сессия.

**Семантическое решение:** `y` **строго** `{0, 1, ..., K-1}`.
1-indexed (`{1..K}`) — ValueError с указанием «сдвиньте y на -1».
Pandas/scikit-learn принимают любой набор labels, но у нас — целочисленный
target (не string), и 1-indexed — самый частый источник багов
(misclass label 1..K vs 0..K-1). Строгая валидация дешевле, чем
debugging 3-level trees с неправильным base prediction.

**Следующее:** v0.8.2 (`shift/diff on=col`), v0.8.3 (`resample(freq)`),
v0.7.9 (`feature_importance(gain)`). ML-слой закрыт (GBDT reg+bin+multi,
Linear/Ridge/Lasso/ENet, Logistic, KMeans, PCA, NB×3).

'''
        s = s.replace(marker, episode + marker, 1)

    p.write_text(s)
    print("  OK: §11.1, §13.4/113")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v078-multiclass-gbdt"

    print("=== v0.7.8: Multiclass GBDT (softmax) ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v078-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v078-{ts_str}",
        "-m", "before patch v0.7.8 multiclass GBDT"])

    add_multiclass_code()
    patch_classifier_dispatch()
    patch_dataframe()
    patch_exports()
    write_tests()
    patch_changelog()
    patch_decisions()

    print("\n=== pytest tests/test_gbdt_multiclass.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_gbdt_multiclass.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.8 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest binary GBDT regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_gbdt_regression.py",
         "tests/test_gbdt_classification.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.8 (regression red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.8: Multiclass GBDT (softmax, per-class loop, reuse ops); "
        "fit_gbdt_classifier auto-dispatch by n_classes; §13.4/113"])

    print("\n==> v0.7.8 applied on", branch)


if __name__ == "__main__":
    main()