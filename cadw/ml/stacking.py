"""Stacking (k-fold OOC ensemble) infrastructure (v0.9.166, DEC-061).

Not a card — infrastructure function like `fit_ab_test`. Reused by
`ensemble_recommendation` recipes; do not duplicate per subtype
(DEC-050 level 3, DEC-052).

Algorithm (K-fold OOC stacking):
  Pass 1: KFold(stratify_by=y).split(df) -> for each fold, fit each
          base model on train, predict probability on test. Collect
          OOF predictions (n, n_base).
  Pass 2: fit meta-learner on OOF matrix + y (n, n_base+1).
  Pass 3: refit base models on full df (for downstream predict()).

OOC: pass 1 streams one fold at a time (O(n) filter per fold,
bounded by KFold). OOF matrix = n × n_base × f64 — bounded by n and
n_base (typical n_base <= 5, n=1e6 -> ~40 MB, fits weak machine).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.cv import KFold


_STACKING_HEADER = 256


@dataclass
class StackingModel:
    """Fitted k-fold stacker: base_models_ + meta_model_."""

    base_models_: list  # list[tuple[str, fitted_model]]
    meta_model_: object
    feature_cols_: list
    meta_cols_: list
    y_col_: str
    task_: str
    n_folds_: int

    def state_nbytes(self) -> int:
        """Sum(base.state_nbytes) + meta.state_nbytes() + 256."""
        total = _STACKING_HEADER
        for _name, m in self.base_models_:
            if hasattr(m, "state_nbytes") and callable(m.state_nbytes):
                try:
                    total += int(m.state_nbytes())
                except Exception:
                    pass
        if hasattr(self.meta_model_, "state_nbytes") and callable(
                self.meta_model_.state_nbytes):
            try:
                total += int(self.meta_model_.state_nbytes())
            except Exception:
                pass
        return int(total)

    def _base_scores(self, df):
        """Score matrix (n, n_base) from base models on df."""
        cols = []
        for _name, m in self.base_models_:
            arr = _extract_scores(m, df, self.feature_cols_, self.task_)
            cols.append(np.asarray(arr, dtype=np.float64))
        if not cols:
            return np.zeros((0, 0))
        return np.column_stack(cols)

    def _meta_df(self, meta_X):
        n = meta_X.shape[0]
        tbl = {self.meta_cols_[i]: pa.array(meta_X[:, i], type=pa.float64())
               for i in range(meta_X.shape[1])}
        from cadw import CadwDataFrame
        return CadwDataFrame.from_table(pa.table(tbl))

    def predict_proba(self, df, memory_guard=None) -> np.ndarray:
        """P(y=1|x) via stacked meta. task must be 'classification'."""
        if self.task_ != "classification":
            raise ValueError(
                "StackingModel.predict_proba requires task='classification'")
        meta_X = self._base_scores(df)
        meta_df = self._meta_df(meta_X)
        return _extract_scores(self.meta_model_, meta_df, self.meta_cols_,
                               self.task_)

    def predict(self, df, memory_guard=None) -> np.ndarray:
        """Stacked prediction (classification: 0/1 at 0.5 threshold;
        regression: point estimate)."""
        if self.task_ == "classification":
            return (self.predict_proba(df, memory_guard) >= 0.5).astype(np.int64)
        meta_X = self._base_scores(df)
        meta_df = self._meta_df(meta_X)
        return _extract_scores(self.meta_model_, meta_df, self.meta_cols_,
                               self.task_)


def _extract_scores(model, df, X_cols, task):
    """Get 1-D scores from model.predict_proba (classification) or
    model.predict (regression). Binary classification: P(y=1)."""
    if task == "classification":
        try:
            pred = model.predict_proba(df, X_cols=X_cols)
        except TypeError:
            pred = model.predict_proba(df)
    else:
        try:
            pred = model.predict(df, X_cols=X_cols)
        except TypeError:
            pred = model.predict(df)
    if isinstance(pred, np.ndarray):
        arr = np.asarray(pred, dtype=np.float64)
        if arr.ndim > 1 and arr.shape[1] > 1:
            arr = arr[:, -1]  # binary: take P(y=1) column (multiclass deferred)
        return arr
    # CadwDataFrame: last column holds score
    parts = []
    for b in pred._iter_filtered_batches():
        arr = b.column(b.num_columns - 1).to_numpy(zero_copy_only=False)
        parts.append(np.asarray(arr, dtype=np.float64))
    return np.concatenate(parts) if parts else np.zeros(0)


def _collect_target(df, y_col):
    parts = []
    for b in df._iter_filtered_batches():
        parts.append(np.asarray(
            b.column(y_col).to_numpy(zero_copy_only=False)))
    return np.concatenate(parts) if parts else np.zeros(0)


def _build_meta_df(oof_X, meta_cols, y, y_col, task):
    tbl = {meta_cols[i]: pa.array(oof_X[:, i], type=pa.float64())
           for i in range(oof_X.shape[1])}
    if task == "classification":
        tbl[y_col] = pa.array(y.astype(np.int64), type=pa.int64())
    else:
        tbl[y_col] = pa.array(y.astype(np.float64), type=pa.float64())
    from cadw import CadwDataFrame
    return CadwDataFrame.from_table(pa.table(tbl))


def fit_stacking(
    df, X_cols, y_col, *,
    base_models,
    meta_learner,
    cv: int = 5,
    task: str = "classification",
    memory_guard=None,
    seed: int = 42,
) -> StackingModel:
    """K-fold out-of-core stacking.

    Args:
      df: CadwDataFrame.
      X_cols: feature columns (str | list[str]).
      y_col: target column.
      base_models: list[tuple[str, fit_fn]] — e.g.
        [("logreg", fit_logistic), ("gbdt", fit_gbdt_classifier)].
        Each fit_fn must accept (df, X_cols, y_col, memory_guard=...).
      meta_learner: fit function for meta layer (e.g. fit_logistic).
      cv: number of folds (>= 2, default 5).
      task: "classification" | "regression".
      memory_guard: optional.
      seed: KFold random_state.

    Returns:
      StackingModel with .base_models_ (fitted on full df),
      .meta_model_ (fit on OOF predictions), and predict/predict_proba.

    OOC: pass 1 streams one fold at a time (O(n) filter per fold by
    KFold, bounded by memory guard). OOF matrix = n × n_base × 8 B.

    Binary classification only for meta (multiclass stacking: deferred
    v0.9.167+).

    Raises:
      ValueError for empty X_cols / base_models, cv < 2, invalid task.
      MemoryLimitExceeded if OOF matrix would exceed guard budget.
    """
    cols = [X_cols] if isinstance(X_cols, str) else list(X_cols)
    if not cols:
        raise ValueError("fit_stacking: X_cols must be non-empty")
    if cv < 2:
        raise ValueError(f"fit_stacking: cv must be >= 2; got {cv}")
    if task not in ("classification", "regression"):
        raise ValueError(
            f"fit_stacking: task must be 'classification'|'regression'; "
            f"got {task!r}")
    if not base_models:
        raise ValueError("fit_stacking: base_models must be non-empty")
    for name, fn in base_models:
        if not callable(fn):
            raise ValueError(
                f"fit_stacking: base_models[{name}] is not callable")

    mg = memory_guard or df._memory_guard
    n = df.count()
    if n == 0:
        raise ValueError("fit_stacking: empty input")
    n_base = len(base_models)

    mg.check_state(n * n_base * 8 + _STACKING_HEADER,
                   name="stacking.oof")

    # KFold: materialize=False (streaming per fold); stratify classification.
    kf = KFold(
        cv,
        stratify_by=(y_col if task == "classification" else None),
        random_state=seed,
        materialize=False,
    )

    # Pass 1: OOF predictions in fold order (permutation-invariant meta fit).
    base_oof_lists = [[] for _ in range(n_base)]
    y_oof_lists = []
    n_seen = 0
    for _fold_id, train_df, test_df in kf.split(df):
        test_n = test_df.count()
        if test_n == 0:
            continue
        y_test = _collect_target(test_df, y_col)
        y_oof_lists.append(y_test)
        for bi, (name, fit_fn) in enumerate(base_models):
            try:
                m = fit_fn(train_df, cols, y_col, memory_guard=mg)
            except TypeError:
                m = fit_fn(train_df, cols, y_col)
            scores = _extract_scores(m, test_df, cols, task)
            if scores.size != test_n:
                raise RuntimeError(
                    f"fit_stacking: base {name!r} returned {scores.size} "
                    f"scores, expected {test_n} (fold test rows)")
            base_oof_lists[bi].append(scores)
        n_seen += test_n
        mg.check_state(n_seen * n_base * 8 + _STACKING_HEADER,
                       name="stacking.pass1_progress")
    if n_seen != n:
        raise RuntimeError(
            f"fit_stacking: fold coverage {n_seen} != n={n}")

    oof_X = np.zeros((n, n_base), dtype=np.float64)
    for bi in range(n_base):
        oof_X[:, bi] = np.concatenate(base_oof_lists[bi])
    y_oof = np.concatenate(y_oof_lists)

    # Pass 2: fit meta on OOF.
    meta_cols = [f"base_{name}" for name, _ in base_models]
    meta_df = _build_meta_df(oof_X, meta_cols, y_oof, y_col, task)
    try:
        meta_model = meta_learner(meta_df, meta_cols, y_col, memory_guard=mg)
    except TypeError:
        meta_model = meta_learner(meta_df, meta_cols, y_col)

    # Pass 3: refit bases on full df (production model).
    fitted_bases = []
    for name, fit_fn in base_models:
        try:
            m = fit_fn(df, cols, y_col, memory_guard=mg)
        except TypeError:
            m = fit_fn(df, cols, y_col)
        fitted_bases.append((name, m))

    return StackingModel(
        base_models_=fitted_bases,
        meta_model_=meta_model,
        feature_cols_=cols,
        meta_cols_=meta_cols,
        y_col_=y_col,
        task_=task,
        n_folds_=cv,
    )
