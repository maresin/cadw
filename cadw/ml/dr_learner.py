"""DR_learner (AIPW doubly-robust meta-learner).

Reference: Robins, Rotnitzky & Zhao (1994). Estimation of regression
coefficients in missing data models. JASA.
Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins
(2018). Double/debiased machine learning. Econometrics Journal.

AIPW (augmented inverse-probability weighting) ATE estimate:
  psi_i = mu1(X_i) - mu0(X_i)
        + (T_i / e(X_i))       * (Y_i - mu1(X_i))
        - ((1-T_i)/(1-e(X_i))) * (Y_i - mu0(X_i))
  ATE   = mean(psi)

Double robustness: consistent if EITHER e(X) correct OR (mu0, mu1)
correct. Cadw implementation via 3x fit_logistic (propensity + two
outcome models) — binary outcome required (T, Y in {0,1}).

Architecture:
  1. Propensity: fit_logistic(df, X_cols, treatment_col) — full df.
  2. mu1: fit_logistic(df[treat==1], X_cols, outcome_col).
  3. mu0: fit_logistic(df[treat==0], X_cols, outcome_col).
  4. Aggregate: iterate df batches, per-batch:
       e  = propensity.predict_proba(X_ndarray)
       m1 = mu1.predict_proba(X_ndarray)
       m0 = mu0.predict_proba(X_ndarray)
       psi = dr_score(y, t, m0, m1, e)
     Sum psi, divide by n.

OOC: state O(d) after fit; fit peak O(d^2 + batch*row_size)
     (XTX for each logistic — sequential, so max not sum).
No bounded_by_domain — achieved via bounded_state + max_query
style peak bound. Honest (matches T_learner_uplift precedent).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._causal_common import dr_score
from cadw.ml._logging import log_fit
from cadw.ml.logistic import LogisticModel, fit_logistic


def _stack_X(batch, X_cols: Sequence[str]) -> np.ndarray:
    d = len(X_cols)
    b_n = batch.num_rows
    X = np.empty((b_n, d), dtype=np.float64)
    for j, c in enumerate(X_cols):
        X[:, j] = batch.column(c).to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False)
    return X


def _one_model_state_nbytes(m: LogisticModel) -> int:
    """Inline LogisticModel state: coef_(d)*f64 + intercept_(8) + 256 header.

    LogisticModel has no state_nbytes() (legacy, dataclass). We compute
    here to keep framework scope minimal (v0.9.225-c precedent).
    """
    return int(m.coef_.nbytes + 8 + 256)


@dataclass
class DRLearnerModel:
    """DR-learner (AIPW ATE). State = 3 logistic models + aggregate."""
    propensity_: LogisticModel
    mu0_: LogisticModel
    mu1_: LogisticModel
    ate_: float
    ate_se_: float             # std(psi) / sqrt(n)
    n_rows_: int
    feature_names_: list
    treatment_col_: str
    outcome_col_: str
    propensity_alpha_: float
    outcome_alpha_: float

    def state_nbytes(self) -> int:
        """O(d): 3 * (coef_(d)*f64 + intercept_ + 256) + 256 header."""
        return int(
            _one_model_state_nbytes(self.propensity_)
            + _one_model_state_nbytes(self.mu0_)
            + _one_model_state_nbytes(self.mu1_)
            + 256
        )

    def predict_psi(
        self, df, *, memory_guard: Optional[MemoryGuard] = None,
        batch_size: Optional[int] = None,
    ):
        """Per-row AIPW pseudo-outcome psi. Returns CadwDataFrame."""
        from cadw import CadwDataFrame
        mg = memory_guard or df._memory_guard
        for c in self.feature_names_ + [self.treatment_col_, self.outcome_col_]:
            if c not in df._columns:
                raise KeyError(f"predict_psi: column {c!r} not in df")
        n_total = int(df.count())
        mg.check_state(n_total * 8 + 256,
                       name="dr_learner.predict_psi_output")
        out = np.empty(n_total, dtype=np.float64)
        off = 0
        _bs = batch_size if batch_size is not None else 1000
        for b in df._iter_filtered_batches(batch_size=_bs):
            mg.check_batch(b, name="dr_learner.predict_psi", extra_factor=3.0)
            X = _stack_X(b, self.feature_names_)
            y = b.column(self.outcome_col_).to_numpy(
                zero_copy_only=False).astype(np.float64, copy=False)
            t = b.column(self.treatment_col_).to_numpy(
                zero_copy_only=False).astype(np.float64, copy=False)
            e = self.propensity_.predict_proba(X)
            m1 = self.mu1_.predict_proba(X)
            m0 = self.mu0_.predict_proba(X)
            out[off:off + b.num_rows] = dr_score(y, t, m0, m1, e)
            off += b.num_rows
        if off != n_total:
            out = out[:off]
        return CadwDataFrame.from_table(
            pa.table({"psi": pa.array(out, type=pa.float64())}),
            memory_guard=mg)


def fit_dr_learner(
    df,
    X_cols: "str | Sequence[str]",
    treatment_col: str,
    outcome_col: str,
    *,
    propensity_alpha: float = 1.0,
    outcome_alpha: float = 1.0,
    memory_guard: Optional[MemoryGuard] = None,
    batch_size: "int | None" = None,
) -> DRLearnerModel:
    """Fit AIPW doubly-robust ATE. Binary treatment and outcome required.

    Args:
      df: CadwDataFrame.
      X_cols: feature columns (numeric).
      treatment_col: binary treatment indicator {0, 1}.
      outcome_col: binary outcome {0, 1} (fit_logistic requirement).
      propensity_alpha: L2 regularization for propensity model.
      outcome_alpha: L2 regularization for both outcome models.
      memory_guard, batch_size: standard.

    Returns:
      DRLearnerModel with ate_, ate_se_, propensity_, mu0_, mu1_.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_dr_learner: X_cols is empty")
    for c in [treatment_col, outcome_col]:
        if c in X_cols:
            raise ValueError(
                f"fit_dr_learner: {c!r} must not be in X_cols")
    if treatment_col == outcome_col:
        raise ValueError(
            "fit_dr_learner: treatment_col == outcome_col")

    for c in X_cols + [treatment_col, outcome_col]:
        if c not in df.columns:
            raise KeyError(f"fit_dr_learner: column {c!r} not in df")

    schema = df._dataset.schema
    for c in [treatment_col, outcome_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_dr_learner: {c!r} has type {t}; numeric required")

    mg = memory_guard or df._memory_guard
    d = len(X_cols)
    n_total = int(df.count())
    if n_total == 0:
        raise ValueError("fit_dr_learner: empty input")

    # ─── 1. Split + validate arm counts (before any fit) ───
    # RowFilter: df[boolean_mask] keeps filter view; iteration
    # honors it. No materialization of subset. Check arm counts
    # BEFORE fit_logistic: propensity fit on y=t fails first with
    # "y must have both classes" when all t=0 (misleading).
    try:
        treated_df = df[df[treatment_col] == 1]
        control_df = df[df[treatment_col] == 0]
    except Exception as e:
        raise RuntimeError(
            f"fit_dr_learner: boolean split failed (df[df[treat]==v]): "
            f"{type(e).__name__}: {e}. RowFilter backend required.") from e

    n_treated = int(treated_df.count())
    n_control = int(control_df.count())
    if n_treated < 2:
        raise ValueError(
            f"fit_dr_learner: need >= 2 treated rows; got {n_treated}")
    if n_control < 2:
        raise ValueError(
            f"fit_dr_learner: need >= 2 control rows; got {n_control}")

    # ─── 2. Propensity model e(X) = P(T=1|X) on full df ───
    propensity = fit_logistic(
        df, X_cols, treatment_col, alpha=float(propensity_alpha),
        memory_guard=mg, batch_size=batch_size)
    log_fit("dr_learner.propensity", state_bytes=_one_model_state_nbytes(propensity))

    # ─── 3. Outcome models ───
    mu1 = fit_logistic(
        treated_df, X_cols, outcome_col, alpha=float(outcome_alpha),
        memory_guard=mg, batch_size=batch_size)
    log_fit("dr_learner.mu1", state_bytes=_one_model_state_nbytes(mu1))

    mu0 = fit_logistic(
        control_df, X_cols, outcome_col, alpha=float(outcome_alpha),
        memory_guard=mg, batch_size=batch_size)
    log_fit("dr_learner.mu0", state_bytes=_one_model_state_nbytes(mu0))

    # ─── 4. Aggregate psi -> ATE via Reduce (O(batch)) ───
    if batch_size is None:
        batch_size = 1000
    batch_size = max(1, int(batch_size))

    psi_sum = 0.0
    psi_sum_sq = 0.0
    n_seen = 0

    for b in df._iter_filtered_batches(batch_size=batch_size):
        mg.check_batch(b, name="dr_learner.aggregate", extra_factor=3.0)
        X = _stack_X(b, X_cols)
        y = b.column(outcome_col).to_numpy(
            zero_copy_only=False).astype(np.float64, copy=False)
        t = b.column(treatment_col).to_numpy(
            zero_copy_only=False).astype(np.float64, copy=False)
        e = propensity.predict_proba(X)
        m1 = mu1.predict_proba(X)
        m0 = mu0.predict_proba(X)
        psi = dr_score(y, t, m0, m1, e)  # (b,) float64
        psi_sum += float(psi.sum())
        psi_sum_sq += float((psi * psi).sum())
        n_seen += b.num_rows

    if n_seen != n_total:
        raise RuntimeError(
            f"fit_dr_learner: row count mismatch ({n_seen} != {n_total})")

    ate = psi_sum / n_seen
    psi_var = max(0.0, psi_sum_sq / n_seen - ate * ate)
    ate_se = float(np.sqrt(psi_var / n_seen))

    model = DRLearnerModel(
        propensity_=propensity,
        mu0_=mu0,
        mu1_=mu1,
        ate_=float(ate),
        ate_se_=float(ate_se),
        n_rows_=int(n_seen),
        feature_names_=list(X_cols),
        treatment_col_=treatment_col,
        outcome_col_=outcome_col,
        propensity_alpha_=float(propensity_alpha),
        outcome_alpha_=float(outcome_alpha),
    )
    log_fit("dr_learner", state_bytes=model.state_nbytes())
    return model
