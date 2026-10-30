"""Ordinal logistic regression, cumulative-link model (D.5.1, v0.9.39).

Reference: McCullagh, *Regression Models for Ordinal Data*, JRSS-B 42(2),
1980; Agresti, *Categorical Data Analysis*, 3rd ed., Wiley 2013, ch. 8
(cumulative logit model); Tutz, *Regression for Categorical Data*, CUP 2012.
See .pingis/algorithm_notes/d5_1_ordinal_logit.md.

Proportional-odds cumulative logit:
  P(Y <= k | x) = sigmoid(theta_k - x @ beta),  k = 1..K-1
with ordered thresholds theta_1 < ... < theta_{K-1}. Fitted by minimising the
negative multinomial log-likelihood with L-BFGS-B over [beta, theta_1,
log-gaps] so the thresholds stay ordered without a constrained solver. For
reference comparison we also expose the same likelihood so tests can pin it
against statsmodels OrderedModel (distr="logit"). RAM: O(n*d + d*K).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import SpillArray, AutoArrayResult
from cadw.ml._logging import log_fit


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _unpack(params, d, K):
    beta = params[:d]
    theta0 = params[d]
    gaps = np.exp(params[d + 1:])  # positive gaps
    theta = theta0 + np.concatenate([[0.0], np.cumsum(gaps)])
    return beta, theta


def _pack(beta, theta):
    gaps = np.diff(theta)
    if np.any(gaps <= 0):
        raise ValueError("ordinal: thresholds must be strictly increasing")
    return np.concatenate([beta, [theta[0]], np.log(gaps)])


def _neg_loglik(params, X, y, d, K):
    beta, theta = _unpack(params, d, K)
    eta = X @ beta
    z = theta[None, :] - eta[:, None]
    P = _sigmoid(z)
    eps = 1e-12
    P = np.clip(P, eps, 1 - eps)
    probs = np.empty((X.shape[0], K))
    probs[:, 0] = P[:, 0]
    for k in range(1, K - 1):
        probs[:, k] = P[:, k] - P[:, k - 1]
    probs[:, K - 1] = 1.0 - P[:, K - 2]
    probs = np.clip(probs, eps, 1.0)
    nll = -float(np.sum(np.log(probs[np.arange(X.shape[0]), y.astype(np.int64)])))
    # add a tiny penalty on the log-gaps so extreme thresholds are avoided
    return nll


def _predict_proba_np(X, beta, theta):
    eta = X @ beta
    z = theta[None, :] - eta[:, None]
    P = _sigmoid(z)
    K = theta.size + 1
    probs = np.empty((X.shape[0], K))
    probs[:, 0] = P[:, 0]
    for k in range(1, K - 1):
        probs[:, k] = P[:, k] - P[:, k - 1]
    probs[:, K - 1] = 1.0 - P[:, K - 2]
    return np.clip(probs, 0.0, 1.0)


def _collect(df, cols, y_col, mg):
    from cadw.ml.gbdt import _stack_numeric
    Xs, ys = [], []
    n = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="ordinal.pass1", extra_factor=3.0)
        # v0.9.90-S3 (G2): pre-check BEFORE append.
        n += b.num_rows
        mg.check_state(n * (len(cols) * 8 + 8), name="ordinal.state")
        Xs.append(_stack_numeric(b, cols))
        ys.append(b.column(y_col).cast(pa.float64(), safe=False)
                  .to_numpy(zero_copy_only=False))
    if n == 0:
        raise ValueError("ordinal: empty input")
    return np.vstack(Xs), np.concatenate(ys)


class OrdinalLogitModel:
    """Proportional-odds cumulative-logit classifier."""

    def __init__(self, beta, theta, classes, feature_cols, params):
        self.beta_ = np.asarray(beta, dtype=np.float64)
        self.theta_ = np.asarray(theta, dtype=np.float64)
        self.classes_ = np.asarray(classes)
        self.feature_cols_ = list(feature_cols)
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return int(self.beta_.nbytes + self.theta_.nbytes + 256)

    def to_dict(self):
        return {"beta": self.beta_.tolist(), "theta": self.theta_.tolist(),
                "classes": [float(c) for c in self.classes_],
                "params": self.params_}

    def predict_proba(self, df, memory_guard=None, *, output: str = "auto"):
        """v0.9.90-S4: output= for spill-on-demand (auto/disk)."""
        if output not in ("memory", "auto", "disk"):
            raise ValueError(
                f"ordinal.predict_proba: output must be 'memory', "
                f"'auto' or 'disk'; got {output!r}")
        mg = memory_guard or df._memory_guard
        from cadw.ml.gbdt import _stack_numeric
        K = int(self.classes_.size)
        if output == "memory":
            chunks = []
        else:
            buf = SpillArray(guard=mg, width=K, dtype=np.float64,
                             name="ordinal.proba",
                             spill_ratio=0.0 if output == "disk" else 0.5)
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="ordinal.predict", extra_factor=3.0)
            p_b = _predict_proba_np(_stack_numeric(b, self.feature_cols_),
                                    self.beta_, self.theta_)
            if output == "memory":
                chunks.append(p_b)
            else:
                buf.write_array(p_b)
        if output == "memory":
            raw = (np.vstack(chunks) if chunks
                   else np.zeros((0, K)))
        else:
            raw = buf.finalize()
        # v0.9.95 (BREAKING): unified CadwDataFrame return.
        from cadw.dataframe import CadwDataFrame
        return CadwDataFrame.from_producer_result(
            raw, col_names=[f"prob_{i}" for i in range(K)],
            memory_guard=mg)

    def predict(self, df, memory_guard=None, *, output: str = "auto"):
        """v0.9.95 (BREAKING): unified CadwDataFrame return. output=
        selects storage strategy only.
        Column: ["pred"] (dtype matches self.classes_.dtype).

        v0.9.95-fix3: use SpillBuffer (pa.RecordBatch) for auto/disk
        path — supports string class labels. Prior SpillArray(width=1,
        dtype=float64) crashed on string classes; pre-existing bug.
        """
        if output not in ("memory", "auto", "disk"):
            raise ValueError(
                f"ordinal.predict: output must be 'memory', 'auto' or "
                f"'disk'; got {output!r}")
        mg = memory_guard or df._memory_guard
        from cadw.ml.gbdt import _stack_numeric
        # Determine pa type once from class labels (works for int, str,
        # any numpy dtype supported by pa.array).
        pred_pa_type = pa.array(self.classes_).type
        schema = pa.schema([("pred", pred_pa_type)])

        if output == "memory":
            chunk_cls = []
            for b in df._iter_filtered_batches():
                mg.check_batch(b, name="ordinal.predict", extra_factor=3.0)
                p_b = _predict_proba_np(_stack_numeric(b, self.feature_cols_),
                                        self.beta_, self.theta_)
                chunk_cls.append(self.classes_[np.argmax(p_b, axis=1)])
            cls_arr = (np.concatenate(chunk_cls) if chunk_cls
                       else np.zeros(0, dtype=self.classes_.dtype))
            raw = pa.table({"pred": pa.array(cls_arr, type=pred_pa_type)})
        else:
            from cadw.core.spill import SpillBuffer
            buf = SpillBuffer(guard=mg, schema=schema,
                              name="ordinal.predict",
                              spill_ratio=0.0 if output == "disk" else 0.5)
            for b in df._iter_filtered_batches():
                mg.check_batch(b, name="ordinal.predict", extra_factor=3.0)
                p_b = _predict_proba_np(_stack_numeric(b, self.feature_cols_),
                                        self.beta_, self.theta_)
                cls_b = self.classes_[np.argmax(p_b, axis=1)]
                rb = pa.RecordBatch.from_arrays(
                    [pa.array(cls_b, type=pred_pa_type)], schema=schema)
                buf.write_batch(rb)
            raw = buf.finalize()

        from cadw.dataframe import CadwDataFrame
        return CadwDataFrame.from_producer_result(
            raw, col_names=["pred"], memory_guard=mg)


def fit_ordinal_logit(df, X_cols, y_col, *, max_iter=200, tol=1e-6,
                      memory_guard: Optional[MemoryGuard] = None):
    """Fit a proportional-odds ordinal logit on integer-coded ordered classes."""
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_ordinal_logit: X_cols must be non-empty")
    for c in cols + [y_col]:
        if c not in df._columns:
            raise KeyError(f"fit_ordinal_logit: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    X, y = _collect(df, cols, y_col, mg)
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("fit_ordinal_logit: nulls in features or outcome")
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(f"fit_ordinal_logit: need >= 2 ordered classes; got {classes.size}")
    K = classes.size
    y_idx = np.searchsorted(classes, y)
    d = X.shape[1]
    # init: zero beta, evenly spaced thresholds
    theta0 = np.linspace(-1.0, 1.0, K - 1) if K > 2 else np.array([0.0])
    params0 = _pack(np.zeros(d), theta0)
    from scipy.optimize import minimize
    res = minimize(_neg_loglik, params0, args=(X, y_idx, d, K),
                   method="L-BFGS-B",
                   options={"maxiter": int(max_iter), "ftol": tol, "gtol": tol})
    beta, theta = _unpack(res.x, d, K)
    model = OrdinalLogitModel(beta, theta, classes, cols,
                              {"max_iter": max_iter, "tol": tol,
                               "n_iter": int(res.nit), "success": bool(res.success)})
    log_fit("ordinal_logit", loss=float(res.fun), state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="ordinal.model")
    return model
