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
        if st is not None:
            # v0.9.242 (butched-fix): knowledge of LinearModel fields
            # lives in LinearModel.from_dict, not here.
            from cadw.ml.linear import LinearModel
            step._model = LinearModel.from_dict(
                st,
                feature_names=step._X_cols,
                target_col=step._y_col,
            )
        return step
