"""T-learner / S-learner meta-learners for the ATE (D.3.2, v0.9.37).

Reference: Kunzel, Sekhon, Bickel, Yu, *Metalearners for estimating
heterogeneous treatment effects using machine learning*, PNAS 116(10),
2019 (T-learner / S-learner definitions); Rosenbaum, Rubin, *The central
role of the propensity score*, Biometrika 70(1), 1983 (propensity and
positivity). See .pingis/algorithm_notes/d3_2_meta_learners.md.

Two meta-learners, both built on the in-tree GBDT regressor so no extra
dependency is needed:

  T-learner: fit mu1(x) on treated and mu0(x) on controls; CATE = mu1 - mu0.
  S-learner: fit one mu(x, t) with t as a feature; CATE = mu(x,1) - mu(x,0).

Propensity e(x) comes from `fit_logistic` (Newton IRLS), and the report
always carries the overlap diagnostics the critic asked for: min/max e(x)
and the share of rows outside [0.05, 0.95].
"""
from __future__ import annotations

from cadw.ml._common import _stack_numeric

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.dataset import Dataset
from cadw.dataframe import CadwDataFrame
from cadw.ml._logging import log_fit

# v0.9.133 (DEC-054): _to_df -> cadw.ml._common._to_df_causal.
from cadw.ml._common import _to_df_causal as _to_df


def _predict_np(model, df, cols, mg, name):
    """Per-batch numpy prediction through a fitted GBDT regressor."""
    out = []
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name=name, extra_factor=2.0)
        out.append(model._predict_np(_stack_numeric(b, cols)))
    return np.concatenate(out) if out else np.zeros(0)


def _predict_np_with_t(model, df, cols, tval, mg):
    """Same, but with a constant treatment column appended (S-learner)."""
    out = []
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="meta_learner.s_predict", extra_factor=2.0)
        X = _stack_numeric(b, cols)
        Xa = np.column_stack([X, np.full(X.shape[0], float(tval))])
        out.append(model._predict_np(Xa))
    return np.concatenate(out) if out else np.zeros(0)


def overlap_diagnostics(e):
    """Positivity diagnostics: never silent about a violated assumption."""
    e = np.asarray(e, dtype=np.float64)
    if e.size == 0:
        return {"n": 0, "min": float("nan"), "max": float("nan"),
                "share_outside_005_095": float("nan")}
    outside = np.mean((e < 0.05) | (e > 0.95))
    return {"n": int(e.size), "min": float(e.min()), "max": float(e.max()),
            "share_outside_005_095": float(outside)}


class MetaLearnerModel:
    """Fitted T-learner or S-learner with propensity + overlap report."""

    def __init__(self, kind, mu1, mu0, s_model, propensity, feature_cols,
                 treatment_col, params):
        self.kind_ = kind
        self.mu1_ = mu1
        self.mu0_ = mu0
        self.s_model_ = s_model
        self.propensity_ = propensity
        self.feature_cols_ = list(feature_cols)
        self.treatment_col_ = treatment_col
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        trees = 0
        for m in (self.mu1_, self.mu0_):
            if m is not None and hasattr(m, "trees_"):
                trees += len(m.trees_)
        if self.s_model_ is not None and hasattr(self.s_model_, "trees_"):
            trees += len(self.s_model_.trees_)
        return int(trees * 4096 + len(self.propensity_) * 8 + 1024)

    def to_dict(self):
        return {"kind": self.kind_, "params": self.params_,
                "feature_cols": self.feature_cols_,
                "treatment_col": self.treatment_col_,
                "overlap": overlap_diagnostics(self.propensity_)}

    def cate(self, df, memory_guard=None) -> np.ndarray:
        """Estimated CATE mu1(x) - mu0(x) for every row."""
        mg = memory_guard or df._memory_guard
        cols = self.feature_cols_
        if self.kind_ == "t":
            return (_predict_np(self.mu1_, df, cols, mg, "meta_learner.t_predict")
                    - _predict_np(self.mu0_, df, cols, mg, "meta_learner.t_predict"))
        m1 = _predict_np_with_t(self.s_model_, df, cols, 1.0, mg)
        m0 = _predict_np_with_t(self.s_model_, df, cols, 0.0, mg)
        return m1 - m0

    def ate(self, df, memory_guard=None) -> float:
        return float(np.mean(self.cate(df, memory_guard)))

    def cate_table(self, df, memory_guard=None) -> pa.Table:
        return pa.table({"cate": pa.array(self.cate(df, memory_guard),
                                         type=pa.float64())})

    def ipw_ate(self, df, y_col, memory_guard=None) -> float:
        """Horvitz-Thompson IPW ATE using the fitted propensity."""
        mg = memory_guard or df._memory_guard
        ys, ts = [], []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="meta_learner.ipw", extra_factor=2.0)
            ys.append(b.column(y_col).cast(pa.float64(), safe=False)
                      .to_numpy(zero_copy_only=False))
            ts.append(b.column(self.treatment_col_)
                      .cast(pa.float64(), safe=False)
                      .to_numpy(zero_copy_only=False))
        y = np.concatenate(ys); t = np.concatenate(ts)
        e = np.clip(self.propensity_, 1e-6, 1 - 1e-6)
        return float(np.mean(t * y / e - (1 - t) * y / (1 - e)))


def _collect(df, cols, t_col, y_col, mg, max_rows=None):
    Xs, ts, ys = [], [], []
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
        mg.check_batch(b, name="meta_learner.pass1", extra_factor=2.0)
        # v0.9.90-S3 (G2): pre-check BEFORE append.
        n += b.num_rows
        mg.check_state(n * (len(cols) * 8 + 16), name="meta_learner.state")
        Xs.append(_stack_numeric(b, cols))
        ts.append(b.column(t_col).cast(pa.float64(), safe=False)
                  .to_numpy(zero_copy_only=False))
        ys.append(b.column(y_col).cast(pa.float64(), safe=False)
                  .to_numpy(zero_copy_only=False))
    if n == 0:
        raise ValueError("meta_learner: empty input")
    return np.vstack(Xs), np.concatenate(ts), np.concatenate(ys)




def fit_meta_learner(df, X_cols, treatment_col, y_col, *, kind="t",
                     n_estimators=50, max_depth=3, learning_rate=0.1,
                     propensity_alpha=1.0,
                     max_samples: int | None = None,
                     memory_guard: Optional[MemoryGuard] = None):
    """Fit a T-learner or S-learner, plus a propensity model.

    kind="t": separate outcome models for treated and control.
    kind="s": one outcome model with the treatment as a feature.

    Treatment must be binary 0/1. The propensity e(x) is fitted with the
    in-tree logistic regression and its overlap diagnostics are attached
    to the returned model (min / max / share outside [0.05, 0.95]).
    """
    if kind not in ("t", "s"):
        raise ValueError(f"fit_meta_learner: kind must be 't' or 's', got {kind!r}")
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_meta_learner: X_cols must be non-empty")
    for c in cols + [treatment_col, y_col]:
        if c not in df._columns:
            raise KeyError(f"fit_meta_learner: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    n_total = df.count()
    cap = int(max_samples) if max_samples is not None else 20000
    if max_samples is None and n_total > cap:
        import warnings
        warnings.warn(
            f"fit_meta_learner: subsampling to {cap} rows (n={n_total}). "
            f"Pass max_samples=N to override.",
            stacklevel=2,
        )
    X, t, y = _collect(df, cols, treatment_col, y_col, mg, max_rows=cap)
    if np.isnan(X).any() or np.isnan(y).any() or np.isnan(t).any():
        raise ValueError("fit_meta_learner: nulls in features, outcome or treatment")
    uniq = np.unique(t)
    if not np.array_equal(uniq, np.array([0.0, 1.0])):
        raise ValueError(f"fit_meta_learner: treatment must be binary 0/1; got {uniq.tolist()}")
    n1 = int((t == 1).sum()); n0 = int((t == 0).sum())
    if n1 < 2 or n0 < 2:
        raise ValueError(f"fit_meta_learner: need >=2 rows per arm; got {n1} treated / {n0} control")

    # propensity: logistic regression on the same features
    from cadw.ml.logistic import fit_logistic
    pcols = cols
    tdf = _to_df(X, t, y, cols)
    pm = fit_logistic(tdf, pcols, "__t", alpha=propensity_alpha, memory_guard=mg)
    eta = np.column_stack([np.ones(X.shape[0]), X]) @ np.concatenate([[pm.intercept_], pm.coef_])
    propensity = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))

    from cadw.ml.gbdt import fit_gbdt_regressor
    mu1 = mu0 = s_model = None
    if kind == "t":
        for arm, store in ((1.0, "1"), (0.0, "0")):
            mask = t == arm
            sub = _to_df(X[mask], t[mask], y[mask], cols)
            m = fit_gbdt_regressor(sub, cols, "__y", n_estimators=n_estimators,
                                   max_depth=max_depth, learning_rate=learning_rate,
                                   memory_guard=mg)
            if arm == 1.0:
                mu1 = m
            else:
                mu0 = m
            log_fit(f"meta_learner_{kind}", loss=None, state_bytes=m.state_nbytes() if hasattr(m, "state_nbytes") else None)
    else:
        tdf2 = _to_df(X, t, y, cols)
        s_model = fit_gbdt_regressor(tdf2, cols + ["__t"], "__y",
                                     n_estimators=n_estimators, max_depth=max_depth,
                                     learning_rate=learning_rate, memory_guard=mg)
        log_fit(f"meta_learner_{kind}", state_bytes=s_model.state_nbytes() if hasattr(s_model, "state_nbytes") else None)

    model = MetaLearnerModel(kind, mu1, mu0, s_model, propensity, cols,
                             treatment_col,
                             {"n_estimators": n_estimators, "max_depth": max_depth,
                              "learning_rate": learning_rate,
                              "propensity_alpha": propensity_alpha})
    mg.check_state(model.state_nbytes(), name="meta_learner.model")
    return model
