"""Uplift trees with Qini evaluation (D.3.3, v0.9.37).

Reference: Gutierrez, Gerardy, *Causal Inference and Uplift Modelling: a
Review of the Literature*, PMLR 67, 2016 (uplift trees); Radcliffe, *Using
control groups to estimate the benefits of a promotion*, 2007 (Qini curve
and Qini coefficient). See .pingis/algorithm_notes/d3_3_uplift.md.

Implementation: T-learner style two-model uplift on the in-tree GBDT
classifier. CATE = P(y=1|t=1,x) - P(y=1|t=0,x). The Qini curve sorts by
CATE and accumulates incremental conversions; the Qini coefficient is the
trapezoid area under Qini(u) minus the random-targeting baseline.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.dataframe import CadwDataFrame
from cadw.ml._logging import log_fit


def _stack_numeric_safe(b, cols):
    from cadw.ml.gbdt import _stack_numeric
    return _stack_numeric(b, cols)


def qini_curve(y, t, score, n_bins=50):
    """Qini curve and coefficient from a continuous uplift score.

    Returns (fractions, qini_values, qini_coefficient). The Qini value at
    fraction f is the incremental conversion rate at that top-f fraction.
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
    # random baseline: straight line from 0 to qini[-1]
    baseline = frac * qini[-1]
    coef = float(np.trapezoid(qini - baseline, frac))
    # downsample for output
    idx = np.linspace(0, n - 1, min(n_bins, n)).astype(np.int64)
    return frac[idx], qini[idx], coef


def _collect_cls(df, cols, t_col, y_col, mg):
    Xs, ts, ys = [], [], []
    n = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="uplift.pass1", extra_factor=2.0)
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


def _to_df(X, t, y, cols):
    d = {c: pa.array(X[:, j]) for j, c in enumerate(cols)}
    d["__t"] = pa.array(t, type=pa.float64())
    d["__y"] = pa.array(y, type=pa.float64())
    return CadwDataFrame.from_table(pa.table(d))


class UpliftModel:
    def __init__(self, m1, m0, feature_cols, treatment_col, params):
        self.m1_ = m1
        self.m0_ = m0
        self.feature_cols_ = list(feature_cols)
        self.treatment_col_ = treatment_col
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        trees = 0
        for m in (self.m1_, self.m0_):
            if m is not None and hasattr(m, "trees_"):
                trees += len(m.trees_)
        return int(trees * 4096 + 1024)

    def to_dict(self):
        return {"params": self.params_, "feature_cols": self.feature_cols_,
                "treatment_col": self.treatment_col_}

    def _proba_df(self, df, col, mg):
        out = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="uplift.predict", extra_factor=2.0)
            out.append(col.predict_proba(_stack_numeric_safe(b, self.feature_cols_)))
        return np.concatenate(out) if out else np.zeros(0)

    def uplift(self, df, memory_guard=None) -> np.ndarray:
        mg = memory_guard or df._memory_guard
        return self._proba_df(df, self.m1_, mg) - self._proba_df(df, self.m0_, mg)

    def qini(self, df, y_col, memory_guard=None):
        mg = memory_guard or df._memory_guard
        ys, ts = [], []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="uplift.qini", extra_factor=2.0)
            ys.append(b.column(y_col).cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False))
            ts.append(b.column(self.treatment_col_).cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False))
        y = np.concatenate(ys); t = np.concatenate(ts)
        return qini_curve(y, t, self.uplift(df, mg))

def fit_uplift(df, X_cols, treatment_col, y_col, *, n_estimators=50,
               max_depth=3, learning_rate=0.1,
               memory_guard: Optional[MemoryGuard] = None):
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_uplift: X_cols must be non-empty")
    for c in cols + [treatment_col, y_col]:
        if c not in df._columns:
            raise KeyError(f"fit_uplift: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    X, t, y = _collect_cls(df, cols, treatment_col, y_col, mg)
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
        log_fit(f"uplift_{slot}", state_bytes=m.state_nbytes() if hasattr(m, "state_nbytes") else None)

    model = UpliftModel(m1, m0, cols, treatment_col,
                        {"n_estimators": n_estimators, "max_depth": max_depth,
                         "learning_rate": learning_rate})
    mg.check_state(model.state_nbytes(), name="uplift.model")
    return model
