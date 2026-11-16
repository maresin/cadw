"""LinearRegressor — barrier OLS predictor (reference MVP step).

Independent wrapper over cadw.ml.linear.fit_linear.
"""
from __future__ import annotations
from cadw.steps.base import Model


class LinearRegressor(Model):
    is_barrier = True
    cadw_function = "fit_linear"

    def __init__(self, X_cols, y_col, **kwargs):
        self._X_cols = ([X_cols] if isinstance(X_cols, str)
                        else list(X_cols))
        self._y_col = y_col
        self._kwargs = dict(kwargs)
        self._model = None

    def fit(self, df, target=None):
        from cadw.ml.linear import fit_linear
        self._model = fit_linear(
            df, self._X_cols, self._y_col or target,
            memory_guard=df._memory_guard, **self._kwargs)
        return self

    def predict(self, df):
        return self._model.predict(df)

    def estimate_state_bytes(self, dc):
        d = len(self._X_cols)
        return d * 8 + 8 + 256

    def state_bytes(self):
        if self._model is None:
            return 0
        if hasattr(self._model, "state_nbytes"):
            return self._model.state_nbytes()
        return len(self._model.coef_) * 8 + 8 + 256

    def to_dict(self):
        out = {"X_cols": list(self._X_cols),
               "y_col": self._y_col,
               "kwargs": dict(self._kwargs)}
        if self._model is not None:
            # Serialize fitted LinearModel state (JSON-compatible)
            out["state"] = self._model.to_dict()
        return out

    @classmethod
    def from_dict(cls, d):
        step = cls(d["X_cols"], d["y_col"], **d.get("kwargs", {}))
        st = d.get("state")
        if st is None:
            return step
        # Reconstruct fitted LinearModel from state dict
        import numpy as np
        from cadw.ml.linear import LinearModel
        lm = LinearModel.__new__(LinearModel)
        coeffs = st["coefficients"]
        lm.feature_names_ = list(coeffs.keys())
        lm.coef_ = np.array([coeffs[k] for k in lm.feature_names_],
                            dtype=np.float64)
        lm.intercept_ = float(st["intercept"])
        lm.alpha_ = float(st["alpha"])
        lm.penalty_ = st["penalty"]
        lm.l1_ratio_ = float(st["l1_ratio"])
        lm.n_iter_cd_ = int(st["n_iter_cd"])
        lm.converged_ = bool(st["converged"])
        lm.n_rows_ = int(st["n_rows"])
        lm.solver_ = st["solver"]
        lm.target_col_ = step._y_col
        lm.fitted_ = True
        step._model = lm
        return step
