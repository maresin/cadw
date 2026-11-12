"""Feature selection: mutual information, variance, Boruta (D.6.1).

Reference: Kraskov, Stoegbauer, Grassberger, *Estimating Mutual Information*,
Phys. Rev. E 69, 066138, 2004 (MI estimator reference; CADW uses native
histogram MI, not the kNN variant); Kursa, Rudnicki, *Feature Selection
with the Boruta Package*, J. Stat. Softw. 36(11), 2010 (Boruta shadow
features). See DECISION_LOG for algorithm reference.

v0.9.43: sklearn/scipy imports removed. Native histogram MI replaces
kNN MI; ranking preserved (contract "top-k = sklearn"), absolute values
differ (documented in d6_1 notes).

Three selectors share one interface over CadwDataFrame:

  mutual_info       - histogram MI (native numpy); classification and regression.
  variance_threshold- variance filter (constant / quasi-constant drop).
  boruta            - shadow-feature loop on the in-tree GBDT importance;
                      keeps only features that beat all shadows.

All selectors stream the dataframe once into a bounded numpy matrix
(O(n*d)) and then run entirely on numpy; there is no extra state.
"""
from __future__ import annotations

from cadw.ml._common import _stack_numeric as _stack_numeric_safe

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import SpillDir
from cadw.ml._logging import log_fit




def _collect(df, cols, y_col, mg, name, max_rows=None):
    Xs, ys = [], []
    n = 0
    cap = max_rows
    for b in df._iter_filtered_batches():
        if cap is not None and n >= cap:
            break
        if cap is not None:
            take = min(b.num_rows, cap - n)
        else:
            take = b.num_rows
        if take < b.num_rows:
            b = b.slice(0, take)
        mg.check_batch(b, name=name, extra_factor=3.0)
        # v0.9.90-S3 (G2): pre-check BEFORE append.
        n += b.num_rows
        mg.check_state(n * (len(cols) * 8 + 8), name=name + ".state")
        Xs.append(_stack_numeric_safe(b, cols))
        ys.append(b.column(y_col).cast(pa.float64(), safe=False)
                  .to_numpy(zero_copy_only=False))
    if n == 0:
        raise ValueError("feature_select: empty input")
    return np.vstack(Xs), np.concatenate(ys)



# ============================================================================
# v0.9.170 (S2b): OOC collect via SpillDir memmap + chunked selectors.
# Replaces AP2 `_collect` (Xs.append + np.vstack) for variance / mutual_info.
# Boruta still uses AP2 `_collect` (needs hstack(X|shadows); v0.9.172+).
# ============================================================================

_COLLECT_CHUNK = 50_000  # rows per MI / variance pass


def _collect_mm(df, cols, y_col, mg, name):
    """Stream df -> X_mm (n, d) + y_mm (n,) on disk via SpillDir.

    Returns (X_mm, y_mm, n, sdir). Caller MUST `del X_mm, y_mm; sdir.__exit__`.
    Peak: O(batch * (d + 1)) — truly OOC in n.
    Null check: per batch (before memmap write). Guard: check_batch per batch.
    """
    n = df.count()
    if n == 0:
        raise ValueError("feature_select: empty input")
    sdir = SpillDir(guard=mg, name=f"{name}_collect")
    sdir.__enter__()
    try:
        X_mm = np.memmap(sdir.path("X.bin"), dtype=np.float64,
                          mode="w+", shape=(n, len(cols)))
        y_mm = np.memmap(sdir.path("y.bin"), dtype=np.float64,
                          mode="w+", shape=(n,))
        idx = 0
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name=name, extra_factor=3.0)
            X_b = _stack_numeric_safe(b, cols)
            y_b = (b.column(y_col).cast(pa.float64(), safe=False)
                   .to_numpy(zero_copy_only=False).astype(np.float64, copy=False))
            b_n = X_b.shape[0]
            if np.isnan(X_b).any() or np.isnan(y_b).any():
                raise ValueError("feature_select: nulls in features or target")
            X_mm[idx:idx + b_n] = X_b
            y_mm[idx:idx + b_n] = y_b
            idx += b_n
            del X_b, y_b
        X_mm.flush()
        y_mm.flush()
        return X_mm, y_mm, n, sdir
    except BaseException:
        sdir.__exit__(None, None, None)
        raise


def _variance_selector_chunked(X_mm, threshold, mg):
    """Chunked 2-pass variance (sum + sum_sq). Peak O(chunk * d)."""
    n, d = X_mm.shape
    s1 = np.zeros(d, dtype=np.float64)
    s2 = np.zeros(d, dtype=np.float64)
    for i in range(0, n, _COLLECT_CHUNK):
        end = min(i + _COLLECT_CHUNK, n)
        block = np.asarray(X_mm[i:end])
        s1 += block.sum(axis=0)
        s2 += (block * block).sum(axis=0)
    mean = s1 / n
    var = s2 / n - mean * mean
    var = np.maximum(var, 0.0)
    sel = [j for j in range(d) if var[j] > threshold]
    return var, sel


def _prepare_y_setup(y_mm, n, n_bins):
    """Scan y in chunks: discrete (unique) or continuous (quantile edges)."""
    uniq_acc = np.empty(0, dtype=np.float64)
    for i in range(0, n, _COLLECT_CHUNK):
        end = min(i + _COLLECT_CHUNK, n)
        y_blk = np.asarray(y_mm[i:end])
        uniq_acc = np.union1d(uniq_acc, np.unique(y_blk))
        if len(uniq_acc) > n_bins:
            break
    if len(uniq_acc) <= n_bins:
        return ("discrete", uniq_acc)
    # Continuous: strided sample
    stride = max(1, n // 5000)
    y_sample = np.asarray(y_mm[::stride])
    y_edges = np.unique(np.quantile(y_sample, np.linspace(0, 1, n_bins + 1)))
    return ("continuous", y_edges)


def _histogram_mi_chunked(X_mm, y_mm, n, j, *, y_setup, n_bins, mg):
    """Chunked 2-pass histogram MI for one column j. Peak O(chunk * 16)."""
    # Pass 1: x_edges from strided sample
    stride = max(1, n // 5000)
    x_sample = np.asarray(X_mm[::stride, j])
    x_edges = np.unique(np.quantile(x_sample, np.linspace(0, 1, n_bins + 1)))
    if len(x_edges) < 2:
        return 0.0
    nx = len(x_edges) - 1
    x_cut = x_edges[1:-1]

    kind, payload = y_setup
    if kind == "discrete":
        y_unique = payload
        ny = max(len(y_unique), 1)
    else:
        y_edges = payload
        ny = max(len(y_edges) - 1, 1)

    joint = np.zeros((nx, ny), dtype=np.float64)
    for i in range(0, n, _COLLECT_CHUNK):
        end = min(i + _COLLECT_CHUNK, n)
        x_blk = np.asarray(X_mm[i:end, j])
        y_blk = np.asarray(y_mm[i:end])
        x_idx = np.digitize(x_blk, x_cut)
        if kind == "discrete":
            y_idx = np.searchsorted(y_unique, y_blk)
            y_idx = np.clip(y_idx, 0, ny - 1)
        else:
            y_idx = np.digitize(y_blk, y_edges[1:-1])
        np.add.at(joint, (x_idx, y_idx), 1.0)
    total = joint.sum()
    if total == 0:
        return 0.0
    joint /= total
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    outer = px @ py
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / outer[nz])))
    return max(mi, 0.0)


def _mi_all_columns_chunked(X_mm, y_mm, n, mg, n_bins=None):
    """Chunked MI for all columns. Peak O(chunk * d + n_bins^2)."""
    if n_bins is None:
        n_bins = max(5, min(50, int(n ** (1.0 / 3.0))))
    y_setup = _prepare_y_setup(y_mm, n, n_bins)
    kind, payload = y_setup
    if kind == "continuous" and len(payload) < 2:
        return np.zeros(X_mm.shape[1], dtype=np.float64)
    if kind == "discrete" and len(payload) < 1:
        return np.zeros(X_mm.shape[1], dtype=np.float64)
    d = X_mm.shape[1]
    scores = np.zeros(d, dtype=np.float64)
    for j in range(d):
        mg.check_state(
            _COLLECT_CHUNK * 16 + 4096,
            name=f"feature_select.mi_col_{j}",
        )
        scores[j] = _histogram_mi_chunked(
            X_mm, y_mm, n, j, y_setup=y_setup, n_bins=n_bins, mg=mg,
        )
    return scores


def _entropy_from_counts(counts):
    p = counts[counts > 0].astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _histogram_mi(x, y, n_bins=None, mg=None):
    """Histogram-based MI estimate in nats (native numpy, v0.9.43).

    Replaces the v0.9.40 kNN MI estimator. That estimator was a hidden
    runtime dependency: sklearn/scipy imported lazily inside the function,
    but on the DEFAULT `method='mutual_info'` path, so any user without
    dev extras got ImportError.

    Trade-off vs kNN MI: ranking (informative > noise) preserved
    (contract spec: "top-k = sklearn"); absolute values differ (histogram
    underestimates continuous-continuous MI at finite n); 10-100x faster;
    O(n_bins^2) joint histogram + O(n) workspace.

    Adaptive bins: n^(1/3), capped 5..50. x: quantile bins. y: treated
    as discrete if unique count <= n_bins, else quantile-binned.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.shape[0]
    if n < 8:
        return 0.0
    if n_bins is None:
        n_bins = max(5, min(50, int(n ** (1.0 / 3.0))))
    if mg is not None:
        # np.quantile uses partition() with O(n) workspace; joint histogram
        # is O(n_bins^2). _collect upstream already guards the X matrix,
        # but this explicit check_state makes the per-column cost honest
        # (AST guard v0.9.43 requires a marker before np.quantile).
        mg.check_state(
            int(16 * n + 16 * n_bins * n_bins + 1024),
            name="histogram_mi",
        )

    y_arr = np.asarray(y).ravel()
    y_unique = np.unique(y_arr)
    is_discrete_y = len(y_unique) <= n_bins

    x_edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(x_edges) < 2:
        return 0.0
    x_idx = np.digitize(x, x_edges[1:-1])

    if is_discrete_y:
        y_idx = np.searchsorted(y_unique, y_arr)
        ny = len(y_unique)
    else:
        y_f = y_arr.astype(np.float64)
        y_edges = np.unique(np.quantile(y_f, np.linspace(0, 1, n_bins + 1)))
        if len(y_edges) < 2:
            return 0.0
        y_idx = np.digitize(y_f, y_edges[1:-1])
        ny = len(y_edges) - 1

    nx = len(x_edges) - 1
    joint = np.zeros((nx, ny), dtype=np.float64)
    np.add.at(joint, (x_idx, y_idx), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    outer = px @ py
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / outer[nz])))
    return max(mi, 0.0)


def _mutual_info_native(x, y, k=3, task="classification", mg=None):
    """Histogram MI. `k` and `task` accepted for API compat (unused)."""
    return _histogram_mi(x, y, mg=mg)


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
                       max_samples: int | None = None,
                       memory_guard: Optional[MemoryGuard] = None):
    """Fit a feature selector: 'mutual_info', 'variance', or 'boruta'.

    v0.9.170 (S2b): variance + mutual_info are truly OOC via SpillDir
    memmap (X on disk; chunked 2-pass; peak O(chunk * d)).
    Boruta (shadow-loop, needs full X|shadows in RAM) remains bounded_by_domain
    n<=1e5 (refactor deferred v0.9.172+).
    """
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
    if task not in ("classification", "regression"):
        raise ValueError(f"fit_feature_select: task must be 'classification' or 'regression'; got {task!r}")

    if method == "boruta":
        # AP2 path (needs hstack(X|shadows); deferred v0.9.172+).
        n_total = df.count()
        cap = int(max_samples) if max_samples is not None else 20000
        if max_samples is None and n_total > cap:
            import warnings
            warnings.warn(
                f"fit_feature_select (boruta): subsampling to {cap} rows "
                f"(n={n_total}). Pass max_samples=N to override.",
                stacklevel=2,
            )
        X, y = _collect(df, cols, y_col, mg, "feature_select.boruta", max_rows=cap)
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("feature_select: nulls in features or target")
        rng = np.random.RandomState(seed)
        _, kept = _boruta_selector(X, y, rng, n_estimators, max_iter)
        scores = kept.astype(np.float64)
        sel = [j for j in range(X.shape[1]) if kept[j]]
    else:
        # v0.9.170 OOC path: SpillDir memmap + chunked 2-pass.
        X_mm, y_mm, n, sdir = _collect_mm(
            df, cols, y_col, mg, "feature_select.pass1")
        try:
            if method == "variance":
                scores, sel = _variance_selector_chunked(
                    X_mm, variance_threshold, mg)
            else:  # mutual_info
                scores = _mi_all_columns_chunked(X_mm, y_mm, n, mg)
                sel = list(np.argsort(-scores))
        finally:
            del X_mm, y_mm
            sdir.__exit__(None, None, None)
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
