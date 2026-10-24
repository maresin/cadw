"""Feature selection: mutual information, variance, Boruta (D.6.1, v0.9.40).

Reference: Kraskov, Stoegbauer, Grassberger, *Estimating Mutual Information*,
Phys. Rev. E 69, 066138, 2004 (kNN MI estimator, as in sklearn
mutual_info_classif / mutual_info_regression); Kursa, Rudnicki, *Feature
Selection with the Boruta Package*, J. Stat. Softw. 36(11), 2010 (Boruta
shadow features); sklearn docs for variance thresholding.
See .pingis/algorithm_notes/d6_1_feature_select.md.

Three selectors share one interface over CadwDataFrame:

  mutual_info       - Kraskov kNN MI; classification and regression.
  variance_threshold- variance filter (constant / quasi-constant drop).
  boruta            - shadow-feature loop on the in-tree GBDT importance;
                      keeps only features that beat all shadows.

All selectors stream the dataframe once into a bounded numpy matrix
(O(n*d)) and then run entirely on numpy; there is no extra state.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit


def _stack_numeric_safe(b, cols):
    from cadw.ml.gbdt import _stack_numeric
    return _stack_numeric(b, cols)


def _collect(df, cols, y_col, mg, name):
    Xs, ys = [], []
    n = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name=name, extra_factor=3.0)
        Xs.append(_stack_numeric_safe(b, cols))
        ys.append(b.column(y_col).cast(pa.float64(), safe=False)
                  .to_numpy(zero_copy_only=False))
        n += b.num_rows
        mg.check_state(n * (len(cols) * 8 + 8), name=name + ".state")
    if n == 0:
        raise ValueError("feature_select: empty input")
    return np.vstack(Xs), np.concatenate(ys)


def _entropy_from_counts(counts):
    p = counts[counts > 0].astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _mi_cc(x, y, k=3):
    """Kraskov MI between two continuous variables (sklearn _compute_mi_cc)."""
    from scipy.special import digamma
    from sklearn.neighbors import NearestNeighbors, KDTree
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n = x.shape[0]
    if n < 4:
        return 0.0
    kk = min(int(k), n - 1)
    xy = np.hstack([x, y])
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=kk).fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)
    kd = KDTree(x, metric="chebyshev")
    nx = np.array(kd.query_radius(x, radius, count_only=True, return_distance=False)) - 1.0
    kd = KDTree(y, metric="chebyshev")
    ny = np.array(kd.query_radius(y, radius, count_only=True, return_distance=False)) - 1.0
    mi = (digamma(n) + digamma(kk)
          - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)))
    return float(max(mi, 0.0))


def _mi_cd(c, d, k=3):
    """Ross 2014 continuous-vs-discrete MI (sklearn _compute_mi_cd)."""
    from scipy.special import digamma
    from sklearn.neighbors import NearestNeighbors, KDTree
    c = np.asarray(c, dtype=np.float64).reshape(-1, 1)
    d = np.asarray(d)
    n = c.shape[0]
    radius = np.empty(n)
    label_counts = np.empty(n)
    k_all = np.empty(n)
    for label in np.unique(d):
        mask = d == label
        count = int(np.sum(mask))
        if count > 1:
            kk = min(k, count - 1)
            nn = NearestNeighbors(n_neighbors=kk).fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = kk
        label_counts[mask] = count
    mask = label_counts > 1
    n_use = int(np.sum(mask))
    if n_use < 3:
        return 0.0
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c_use = c[mask]
    radius = radius[mask]
    kd = KDTree(c_use)
    m_all = np.array(kd.query_radius(c_use, radius, count_only=True, return_distance=False))
    mi = (digamma(n_use)
          + np.mean(digamma(k_all))
          - np.mean(digamma(label_counts))
          - np.mean(digamma(m_all)))
    return float(max(mi, 0.0))


def _mutual_info_knn(x, y, k=3, task="classification"):
    """MI estimator matching sklearn.mutual_info_classif/regression defaults."""
    if task == "classification":
        return _mi_cd(x, y, k=k)
    return _mi_cc(x, y, k=k)


class FeatureSelectorModel:
    """Fitted feature selector: scores, ranking, kept subset."""

    def __init__(self, method, feature_cols, scores, selected, params):
        self.method_ = method
        self.feature_cols_ = list(feature_cols)
        self.scores_ = np.asarray(scores, dtype=np.float64)
        self.selected_ = list(selected)
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return int(self.scores_.nbytes + 512)

    def to_dict(self):
        return {"method": self.method_, "feature_cols": self.feature_cols_,
                "scores": self.scores_.tolist(),
                "selected": self.selected_, "params": self.params_}

    def transform(self, df, memory_guard=None):
        """Return a CadwDataFrame restricted to the selected columns."""
        mg = memory_guard or df._memory_guard
        parts = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="feature_select.transform", extra_factor=3.0)
            parts.append(b.select(self.selected_))
        if not parts:
            return df
        from cadw import CadwDataFrame
        return CadwDataFrame.from_table(pa.Table.from_batches(parts))


def _variance_selector(X, threshold):
    v = X.var(axis=0)
    return v, [i for i, val in enumerate(v) if val > threshold]


def _boruta_selector(X, y, rng, n_estimators, max_iter):
    """Boruta: compare each feature GBDT split-count importance to shadows."""
    from cadw.ml.gbdt import fit_gbdt_classifier
    from cadw import CadwDataFrame
    n, d = X.shape
    kept = np.ones(d, dtype=bool)
    for _ in range(int(max_iter)):
        shadows = np.column_stack([rng.permutation(X[:, j]) for j in range(d)])
        Xa = np.hstack([X, shadows])
        cols = [f"__f{j}" for j in range(Xa.shape[1])]
        d_ = {c: pa.array(Xa[:, j]) for j, c in enumerate(cols)}
        d_["__y"] = pa.array(y, type=pa.float64())
        sub = CadwDataFrame.from_table(pa.table(d_))
        m = fit_gbdt_classifier(sub, cols, "__y", n_estimators=n_estimators,
                                max_depth=3)
        imp = np.zeros(Xa.shape[1], dtype=np.float64)
        for tree in getattr(m, "trees_", []):
            structure = tree.get("structure") or {}
            for nid, spec in structure.items():
                if spec is None:
                    continue
                j = int(spec[0])
                if 0 <= j < Xa.shape[1]:
                    imp[j] += 1.0
        real = imp[:d]
        shadow = imp[d:]
        best_shadow = float(shadow.max()) if shadow.size else 0.0
        for j in range(d):
            if kept[j] and real[j] <= best_shadow:
                kept[j] = False
    return np.arange(d), kept.astype(bool)


def fit_feature_select(df, X_cols, y_col, *, method="mutual_info",
                       task="classification", variance_threshold=0.0,
                       max_iter=20, n_estimators=50, seed=0,
                       memory_guard: Optional[MemoryGuard] = None):
    """Fit a feature selector: 'mutual_info', 'variance', or 'boruta'."""
    if method not in ("mutual_info", "variance", "boruta"):
        raise ValueError(
            f"fit_feature_select: method must be 'mutual_info', 'variance' or 'boruta'; got {method!r}")
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_feature_select: X_cols must be non-empty")
    for c in cols + [y_col]:
        if c not in df._columns:
            raise KeyError(f"fit_feature_select: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    X, y = _collect(df, cols, y_col, mg, "feature_select.pass1")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("fit_feature_select: nulls in features or target")
    if task not in ("classification", "regression"):
        raise ValueError(f"fit_feature_select: task must be 'classification' or 'regression'; got {task!r}")
    if method == "variance":
        scores, sel = _variance_selector(X, variance_threshold)
    elif method == "mutual_info":
        discrete_y = task == "classification"
        scores = np.array([_mutual_info_knn(X[:, j], y, task=task)
                           for j in range(X.shape[1])])
        sel = list(np.argsort(-scores))
    else:
        rng = np.random.RandomState(seed)
        _, kept = _boruta_selector(X, y, rng, n_estimators, max_iter)
        scores = kept.astype(np.float64)
        sel = [j for j in range(X.shape[1]) if kept[j]]
    selected = [cols[j] for j in sel]
    model = FeatureSelectorModel(method, cols, scores, selected,
                                 {"task": task,
                                  "variance_threshold": variance_threshold,
                                  "max_iter": max_iter,
                                  "n_estimators": n_estimators,
                                  "seed": seed})
    log_fit("feature_select", state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="feature_select.model")
    return model
