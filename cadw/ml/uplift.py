"""Uplift trees with Qini evaluation (D.3.3, v0.9.42).

Reference: Rzepakowski & Jaroszewicz, *Decision trees for uplift modeling
with single and multiple treatments*, Knowledge and Information Systems
32(2), 2012 (divergence-based split criterion); Wager & Athey, *Estimation
and inference of heterogeneous treatment effects using random forests*,
JASA 113(523), 2018 (honest splits); Radcliffe, *Using control groups to
estimate the benefits of a promotion*, 2007 (Qini curve and coefficient).
See DECISION_LOG for algorithm reference.

Implementation: causal tree ensemble. Splits chosen on Euclidean
divergence between treated/control outcome distributions in the children;
leaves estimate CATE on the honest half-sample. The two-model T-learner
(v0.9.37) is preserved as `_fit_uplift_two_model` for reference tests.
"""
from __future__ import annotations

from cadw.ml._common import _stack_numeric as _stack_numeric_safe

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.dataframe import CadwDataFrame
from cadw.ml._logging import log_fit

# v0.9.133 (DEC-054): _to_df -> cadw.ml._common._to_df_causal.
from cadw.ml._common import _to_df_causal as _to_df


_MAX_SAMPLE_ROWS = 20_000




def qini_curve(y, t, score, n_bins=50):
    """Qini curve and coefficient from a continuous uplift score.

    Returns (fractions, qini_values, qini_coefficient). Unchanged from v0.9.37.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if y.size == 0 or y.size != t.size or y.size != score.size:
        raise ValueError("qini_curve: y, t, score must be equal non-zero length")
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    n = y.size
    n1 = (t == 1).sum(); n0 = (t == 0).sum()
    if n1 == 0 or n0 == 0:
        raise ValueError("qini_curve: both treated and control arms required")
    cum1 = np.cumsum(y * (t == 1))
    cum0 = np.cumsum(y * (t == 0))
    frac = np.arange(1, n + 1, dtype=np.float64) / n
    qini = cum1 - cum0 * (n1 / n0)
    qini = qini / n1
    baseline = frac * qini[-1]
    coef = float(np.trapezoid(qini - baseline, frac))
    idx = np.linspace(0, n - 1, min(n_bins, n)).astype(np.int64)
    return frac[idx], qini[idx], coef


def _collect_cls(df, cols, t_col, y_col, mg, max_rows=None):
    """Stream batches into X/t/y arrays, capped at max_rows."""
    cap = max_rows if max_rows is not None else _MAX_SAMPLE_ROWS
    Xs, ts, ys = [], [], []
    n = 0
    for b in df._iter_filtered_batches():
        if n >= cap:
            break
        mg.check_batch(b, name="uplift.pass1", extra_factor=2.0)
        take = min(b.num_rows, cap - n)
        if take < b.num_rows:
            b = b.slice(0, take)
        Xs.append(_stack_numeric_safe(b, cols))
        ts.append(b.column(t_col).cast(pa.float64(), safe=False)
                  .to_numpy(zero_copy_only=False))
        ys.append(b.column(y_col).cast(pa.float64(), safe=False)
                  .to_numpy(zero_copy_only=False))
        n += b.num_rows
        mg.check_state(n * (len(cols) * 8 + 16), name="uplift.state")
    if n == 0:
        raise ValueError("uplift: empty input")
    return np.vstack(Xs), np.concatenate(ts), np.concatenate(ys)




# ============================================================
# Causal tree: ED divergence splits + honest leaf CATE
# ============================================================

def _ed_divergence(y, t):
    """(P(y=1|t=1) - P(y=1|t=0))^2, or 0 if either arm empty."""
    t1 = y[t == 1]
    t0 = y[t == 0]
    if t1.size == 0 or t0.size == 0:
        return 0.0
    return float((t1.mean() - t0.mean()) ** 2)


def _ed_gain(y_L, t_L, y_R, t_R, y_S, t_S):
    n = len(y_S)
    if n == 0:
        return 0.0
    dL = _ed_divergence(y_L, t_L)
    dR = _ed_divergence(y_R, t_R)
    dS = _ed_divergence(y_S, t_S)
    return (len(y_L) / n) * dL + (len(y_R) / n) * dR - dS


class _CausalTree:
    """Single causal tree: ED-divergence splits + honest leaf CATE.

    Honest split (Wager & Athey 2018): random half of sample for structure,
    other half for leaf estimation. Ensemble averaging handles variance.
    """
    def __init__(self, max_depth=3, min_samples_leaf=20, n_quantiles=10, rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_quantiles = n_quantiles
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.root_ = None
        self.n_nodes_ = 0

    def fit(self, X, t, y, mg=None):
        n = len(y)
        idx = self.rng.permutation(n)
        mid = n // 2
        s_idx, l_idx = idx[:mid], idx[mid:]
        self.root_ = self._build(X[s_idx], t[s_idx], y[s_idx], depth=0, mg=mg)
        self._assign_leaves(self.root_, X[l_idx], t[l_idx], y[l_idx])
        self.n_nodes_ = self._count_nodes(self.root_)
        return self

    def _build(self, X, t, y, depth, mg=None):
        if mg is not None:
            mg.check_state(int(X.size * 8 + 1024), name="causal_tree.build")
        node = {"feature": None, "threshold": None, "left": None,
                "right": None, "cate": 0.0}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node
        if (t == 1).sum() == 0 or (t == 0).sum() == 0:
            return node

        best_gain, best_j, best_thr = -np.inf, None, None
        for j in range(X.shape[1]):
            col = X[:, j]
            qs = np.unique(np.quantile(col, np.linspace(0.1, 0.9, self.n_quantiles)))
            for thr in qs:
                mask_L = col <= thr
                mask_R = ~mask_L
                if (mask_L.sum() < self.min_samples_leaf
                        or mask_R.sum() < self.min_samples_leaf):
                    continue
                g = _ed_gain(y[mask_L], t[mask_L], y[mask_R], t[mask_R], y, t)
                if g > best_gain:
                    best_gain, best_j, best_thr = g, j, float(thr)
        if best_j is None:
            return node

        node["feature"] = best_j
        node["threshold"] = best_thr
        mask_L = X[:, best_j] <= best_thr
        node["left"] = self._build(X[mask_L], t[mask_L], y[mask_L], depth + 1, mg)
        node["right"] = self._build(X[~mask_L], t[~mask_L], y[~mask_L], depth + 1, mg)
        return node

    def _assign_leaves(self, node, X, t, y):
        if node["feature"] is None:
            m1 = t == 1
            m0 = t == 0
            if m1.sum() == 0 or m0.sum() == 0:
                node["cate"] = 0.0
            else:
                node["cate"] = float(y[m1].mean() - y[m0].mean())
            return
        mask_L = X[:, node["feature"]] <= node["threshold"]
        self._assign_leaves(node["left"], X[mask_L], t[mask_L], y[mask_L])
        self._assign_leaves(node["right"], X[~mask_L], t[~mask_L], y[~mask_L])

    def _count_nodes(self, node):
        if node["feature"] is None:
            return 1
        return 1 + self._count_nodes(node["left"]) + self._count_nodes(node["right"])

    def predict(self, X):
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            node = self.root_
            x = X[i]
            while node["feature"] is not None:
                if x[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            out[i] = node["cate"]
        return out

    def state_nbytes(self):
        return int(self.n_nodes_ * 128 + 256)


# ============================================================
# UpliftModel: ensemble of causal trees
# ============================================================

class UpliftModel:
    def __init__(self, trees, feature_cols, treatment_col, params):
        self.trees_ = list(trees)
        self.feature_cols_ = list(feature_cols)
        self.treatment_col_ = treatment_col
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return int(sum(tr.state_nbytes() for tr in self.trees_) + 1024)

    def to_dict(self):
        return {"params": self.params_, "feature_cols": self.feature_cols_,
                "treatment_col": self.treatment_col_, "n_trees": len(self.trees_)}

    def uplift(self, df, memory_guard=None) -> np.ndarray:
        mg = memory_guard or df._memory_guard
        out = []
        n_trees = float(len(self.trees_))
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="uplift.predict", extra_factor=2.0)
            X = _stack_numeric_safe(b, self.feature_cols_)
            preds = np.zeros(X.shape[0], dtype=np.float64)
            for tr in self.trees_:
                preds += tr.predict(X)
            out.append(preds / n_trees)
        return np.concatenate(out) if out else np.zeros(0)

    def qini(self, df, y_col, memory_guard=None):
        mg = memory_guard or df._memory_guard
        ys, ts = [], []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="uplift.qini", extra_factor=2.0)
            ys.append(b.column(y_col).cast(pa.float64(), safe=False)
                      .to_numpy(zero_copy_only=False))
            ts.append(b.column(self.treatment_col_).cast(pa.float64(), safe=False)
                      .to_numpy(zero_copy_only=False))
        y = np.concatenate(ys); t = np.concatenate(ts)
        return qini_curve(y, t, self.uplift(df, mg))


def fit_uplift(df, X_cols, treatment_col, y_col, *, n_estimators=50,
               max_depth=3, learning_rate=0.1, seed=0, max_samples=None,
               memory_guard: Optional[MemoryGuard] = None):
    """Fit causal-tree ensemble for uplift estimation (D.3.3, v0.9.42).

    Splits: Euclidean divergence between treated/control outcome distributions
    (Rzepakowski & Jaroszewicz 2012). Leaves: honest half-sample CATE
    (Wager & Athey 2018). `learning_rate` accepted for API compatibility
    and ignored (ensemble averaging replaces boosting).

    RAM: bounded sample of at most `max_samples` rows (default 20 000).
    """
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_uplift: X_cols must be non-empty")
    for c in cols + [treatment_col, y_col]:
        if c not in df._columns:
            raise KeyError(f"fit_uplift: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard

    cap = int(max_samples) if max_samples is not None else _MAX_SAMPLE_ROWS
    n_total = df.count()
    if max_samples is None and n_total > cap:
        import warnings
        warnings.warn(
            f"fit_uplift: subsampling to {cap} rows (n={n_total}). "
            f"Pass max_samples=N to override.",
            stacklevel=2,
        )
    X, t, y = _collect_cls(df, cols, treatment_col, y_col, mg, max_rows=cap)

    if np.isnan(X).any() or np.isnan(y).any() or np.isnan(t).any():
        raise ValueError("fit_uplift: nulls in features, outcome or treatment")
    uniq = np.unique(t)
    if not np.array_equal(uniq, np.array([0.0, 1.0])):
        raise ValueError(f"fit_uplift: treatment must be binary 0/1; got {uniq.tolist()}")
    if not np.array_equal(np.unique(y), np.array([0.0, 1.0])):
        raise ValueError(f"fit_uplift: outcome must be binary 0/1; got {np.unique(y).tolist()}")
    n1 = int((t == 1).sum()); n0 = int((t == 0).sum())
    if n1 < 2 or n0 < 2:
        raise ValueError(f"fit_uplift: need >=2 rows per arm; got {n1}/{n0}")

    n_est = int(n_estimators)
    if n_est < 1:
        raise ValueError("fit_uplift: n_estimators must be >= 1")

    rng = np.random.RandomState(int(seed))
    min_leaf = max(20, len(y) // 200)
    trees = []
    for i in range(n_est):
        boot_idx = rng.randint(0, len(y), size=len(y))
        tr = _CausalTree(
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            n_quantiles=10,
            rng=np.random.RandomState(int(seed) * 1000 + i + 1),
        )
        tr.fit(X[boot_idx], t[boot_idx], y[boot_idx], mg=mg)
        trees.append(tr)

    model = UpliftModel(trees, cols, treatment_col,
                        {"n_estimators": n_est, "max_depth": max_depth,
                         "learning_rate": learning_rate, "seed": int(seed),
                         "max_samples": cap})
    log_fit("uplift_trees", state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="uplift.model")
    return model


def _fit_uplift_two_model(df, X_cols, treatment_col, y_col, *, n_estimators=50,
                          max_depth=3, learning_rate=0.1, seed=0,
                          max_samples: Optional[int] = None,
                          memory_guard: Optional[MemoryGuard] = None):
    """Reference T-learner baseline (v0.9.37). Tests only."""
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_uplift: X_cols must be non-empty")
    for c in cols + [treatment_col, y_col]:
        if c not in df._columns:
            raise KeyError(f"fit_uplift: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    n_total = df.count()
    cap = int(max_samples) if max_samples is not None else _MAX_SAMPLE_ROWS
    if max_samples is None and n_total > cap:
        import warnings
        warnings.warn(
            f"_fit_uplift_two_model: subsampling to {cap} rows "
            f"(n={n_total}). Pass max_samples=N to override.",
            stacklevel=2,
        )
    X, t, y = _collect_cls(df, cols, treatment_col, y_col, mg, max_rows=cap)

    from cadw.ml.gbdt import fit_gbdt_classifier
    m1 = m0 = None
    for arm, slot in ((1.0, "1"), (0.0, "0")):
        mask = t == arm
        sub = _to_df(X[mask], t[mask], y[mask], cols)
        m = fit_gbdt_classifier(sub, cols, "__y", n_estimators=n_estimators,
                                max_depth=max_depth, learning_rate=learning_rate,
                                memory_guard=mg)
        if arm == 1.0:
            m1 = m
        else:
            m0 = m
        log_fit(f"uplift_{slot}_twomodel",
                state_bytes=m.state_nbytes() if hasattr(m, "state_nbytes") else None)

    class _TLearnerShim:
        def __init__(self, m1, m0, cols, tcol, params):
            self.m1_, self.m0_ = m1, m0
            self.feature_cols_ = cols
            self.treatment_col_ = tcol
            self.params_ = params

        def uplift(self, df, memory_guard=None):
            mg = memory_guard or df._memory_guard
            out = []
            for b in df._iter_filtered_batches():
                mg.check_batch(b, name="uplift_t.predict", extra_factor=2.0)
                Xb = _stack_numeric_safe(b, self.feature_cols_)
                p1 = self.m1_.predict_proba(Xb)
                p0 = self.m0_.predict_proba(Xb)
                out.append(p1 - p0)
            return np.concatenate(out) if out else np.zeros(0)

        def qini(self, df, y_col, memory_guard=None):
            mg = memory_guard or df._memory_guard
            ys, ts = [], []
            for b in df._iter_filtered_batches():
                mg.check_batch(b, name="uplift_t.qini", extra_factor=2.0)
                ys.append(b.column(y_col).cast(pa.float64(), safe=False)
                          .to_numpy(zero_copy_only=False))
                ts.append(b.column(self.treatment_col_).cast(pa.float64(), safe=False)
                          .to_numpy(zero_copy_only=False))
            y = np.concatenate(ys); t = np.concatenate(ts)
            return qini_curve(y, t, self.uplift(df, mg))

        def state_nbytes(self):
            s1 = self.m1_.state_nbytes() if hasattr(self.m1_, "state_nbytes") else 0
            s0 = self.m0_.state_nbytes() if hasattr(self.m0_, "state_nbytes") else 0
            return int(s1 + s0 + 1024)

    return _TLearnerShim(m1, m0, cols, treatment_col,
                         {"n_estimators": n_estimators, "max_depth": max_depth,
                          "learning_rate": learning_rate})
