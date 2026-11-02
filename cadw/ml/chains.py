"""Classifier chains for multi-label classification (D.5.2, v0.9.39).

Reference: Read, Pfahringer, Holmes, Frank, *Classifier Chains for
Multi-label Classification*, ML 85(3), 2011; Tsoumakas, Katakis,
*Multi-Label Classification: An Overview*, IJDWM 3(3), 2007 (binary
relevance / one-vs-rest baseline).
See .pingis/algorithm_notes/d5_2_chains.md.

Each label is modelled by an in-tree GBDT binary classifier. In the chain
variant the true previous labels are appended as features at fit time and the
predicted previous labels are appended at predict time; the independent
variant is the one-vs-rest baseline. RAM: O(K * model state); input is
materialised once (n x (d + K-1)).
"""
from __future__ import annotations

from cadw.ml._common import _stack_numeric as _stack_numeric_safe

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import SpillArray, AutoArrayResult
from cadw.ml._logging import log_fit




def _collect(df, cols, label_cols, mg):
    Xs, Ys = [], []
    n = 0
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="chains.pass1", extra_factor=3.0)
        # v0.9.90-S3 (G2): pre-check BEFORE append.
        n += b.num_rows
        mg.check_state(n * (len(cols) + len(label_cols)) * 8,
                       name="chains.state")
        Xs.append(_stack_numeric_safe(b, cols))
        Ys.append(np.column_stack([
            b.column(c).cast(pa.float64(), safe=False)
             .to_numpy(zero_copy_only=False) for c in label_cols]))
    if n == 0:
        raise ValueError("chains: empty input")
    return np.vstack(Xs), np.vstack(Ys)


def _to_df(X, y):
    d = {f"__f{j}": pa.array(X[:, j]) for j in range(X.shape[1])}
    d["__y"] = pa.array(y, type=pa.float64())
    return __import__("cadw", fromlist=["CadwDataFrame"]).CadwDataFrame.from_table(pa.table(d))


class ChainsModel:
    """Fitted classifier chain (or independent one-vs-rest) model."""

    def __init__(self, models, label_cols, feature_cols, chain, params):
        self.models_ = list(models)
        self.label_cols_ = list(label_cols)
        self.feature_cols_ = list(feature_cols)
        self.chain_ = bool(chain)
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return int(sum((m.state_nbytes() if hasattr(m, "state_nbytes") else 4096)
                       for m in self.models_) + 512)

    def to_dict(self):
        return {"label_cols": self.label_cols_, "chain": self.chain_,
                "params": self.params_}

    def predict_proba(self, df, memory_guard=None, *, output: str = "auto"):
        """v0.9.90-S4: output= for spill-on-demand (auto/disk)."""
        if output not in ("memory", "auto", "disk"):
            raise ValueError(
                f"chains.predict_proba: output must be 'memory', "
                f"'auto' or 'disk'; got {output!r}")
        mg = memory_guard or df._memory_guard
        K = len(self.models_)
        if output == "memory":
            outs = []
        else:
            buf = SpillArray(guard=mg, width=K, dtype=np.float64,
                             name="chains.proba",
                             spill_ratio=0.0 if output == "disk" else 0.5)
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="chains.predict", extra_factor=3.0)
            X = _stack_numeric_safe(b, self.feature_cols_)
            preds = np.empty((X.shape[0], K))
            prev = []
            for k, m in enumerate(self.models_):
                if self.chain_ and prev:
                    Xk = np.column_stack([X] + prev)
                else:
                    Xk = X
                p = m.predict_proba(Xk)
                preds[:, k] = p
                prev.append(p)
            if output == "memory":
                outs.append(preds)
            else:
                buf.write_array(preds)
        if output == "memory":
            raw = np.vstack(outs) if outs else np.zeros((0, K))
        else:
            raw = buf.finalize()
        # v0.9.95 (BREAKING): unified CadwDataFrame return.
        from cadw.dataframe import CadwDataFrame
        return CadwDataFrame.from_producer_result(
            raw, col_names=[f"prob_{i}" for i in range(K)],
            memory_guard=mg)

    def _predict_hard_batched(self, X, threshold):
        """Hard-label chaining: previous labels as 0/1 features."""
        n = X.shape[0]
        out = np.zeros((n, len(self.models_)), dtype=np.int64)
        prev = []
        for k, m in enumerate(self.models_):
            if self.chain_ and prev:
                Xk = np.column_stack([X] + prev)
            else:
                Xk = X
            p = m.predict_proba(Xk)
            lab = (p >= threshold).astype(np.float64)
            out[:, k] = (p >= threshold).astype(np.int64)
            prev.append(lab)
        return out

    def predict(self, df, memory_guard=None, threshold=0.5, *,
                output: str = "auto"):
        """v0.9.90-S4: output= for spill-on-demand (auto/disk)."""
        if output not in ("memory", "auto", "disk"):
            raise ValueError(
                f"chains.predict: output must be 'memory', 'auto' or "
                f"'disk'; got {output!r}")
        mg = memory_guard or df._memory_guard
        K = len(self.models_)
        if output == "memory":
            chunks = []
        else:
            buf = SpillArray(guard=mg, width=K, dtype=np.int64,
                             name="chains.predict",
                             spill_ratio=0.0 if output == "disk" else 0.5)
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="chains.predict", extra_factor=3.0)
            X = _stack_numeric_safe(b, self.feature_cols_)
            preds = self._predict_hard_batched(X, threshold)
            if output == "memory":
                chunks.append(preds)
            else:
                buf.write_array(preds)
        if output == "memory":
            raw = (np.vstack(chunks) if chunks
                   else np.zeros((0, K), dtype=np.int64))
        else:
            raw = buf.finalize()
        # v0.9.95 (BREAKING): unified CadwDataFrame return.
        from cadw.dataframe import CadwDataFrame
        return CadwDataFrame.from_producer_result(
            raw, col_names=[f"pred_{i}" for i in range(K)],
            memory_guard=mg)

    def subset_accuracy(self, df, memory_guard=None, threshold=0.5):
        mg = memory_guard or df._memory_guard
        Ys = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="chains.metric", extra_factor=3.0)
            Ys.append(np.column_stack([
                b.column(c).cast(pa.float64(), safe=False)
                 .to_numpy(zero_copy_only=False) for c in self.label_cols_]))
        Y = np.vstack(Ys)
        # v0.9.95: predict returns CadwDataFrame; extract via iter.
        P_df = self.predict(df, mg, threshold)
        P_cols = []
        for b in P_df._iter_filtered_batches():
            P_cols.append(np.column_stack([
                b.column(c).to_numpy(zero_copy_only=False)
                for c in P_df.columns]))
        P = (np.concatenate(P_cols, axis=0) if P_cols
             else np.zeros((0, len(P_df.columns)), dtype=np.int64))
        return float(np.mean(np.all(P == Y.astype(np.int64), axis=1)))


def fit_chains(df, X_cols, label_cols, *, chain=True, n_estimators=30,
               max_depth=3, learning_rate=0.1,
               memory_guard: Optional[MemoryGuard] = None):
    """Fit a classifier chain (chain=True) or one-vs-rest baseline."""
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    labels = [label_cols] if isinstance(label_cols, str) else list(label_cols)
    if not cols:
        raise ValueError("fit_chains: X_cols must be non-empty")
    if not labels:
        raise ValueError("fit_chains: label_cols must be non-empty")
    for c in cols + labels:
        if c not in df._columns:
            raise KeyError(f"fit_chains: column {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    X, Y = _collect(df, cols, labels, mg)
    if np.isnan(X).any() or np.isnan(Y).any():
        raise ValueError("fit_chains: nulls in features or labels")
    for c in np.unique(Y):
        if c not in (0.0, 1.0):
            raise ValueError(f"fit_chains: labels must be binary 0/1; saw {c!r}")
    from cadw.ml.gbdt import fit_gbdt_classifier
    models = []
    for k, lab in enumerate(labels):
        if chain and k > 0:
            Xk = np.column_stack([X, Y[:, :k]])
        else:
            Xk = X
        cols_k = [f"__f{j}" for j in range(Xk.shape[1])]
        sub = _to_df(Xk, Y[:, k])
        if np.unique(Y[:, k]).size < 2:
            raise ValueError(f"fit_chains: label {lab!r} has a single class in data")
        m = fit_gbdt_classifier(sub, cols_k, "__y", n_estimators=n_estimators,
                                max_depth=max_depth, learning_rate=learning_rate,
                                memory_guard=mg)
        models.append(m)
        log_fit(f"chains_{k}", state_bytes=(m.state_nbytes()
                                            if hasattr(m, "state_nbytes") else None))
    model = ChainsModel(models, labels, cols, chain,
                        {"n_estimators": n_estimators, "max_depth": max_depth,
                         "learning_rate": learning_rate})
    mg.check_state(model.state_nbytes(), name="chains.model")
    return model
