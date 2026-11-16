"""PCAProjector — barrier PCA transformer (reference MVP step).

Independent wrapper over cadw.ml.pca.fit_pca. Barrier: fit needs full df.
"""
from __future__ import annotations
from cadw.steps.base import Transformer


class PCAProjector(Transformer):
    is_barrier = True
    cadw_function = "fit_pca"

    def __init__(self, X_cols, n_components):
        self._X_cols = ([X_cols] if isinstance(X_cols, str)
                        else list(X_cols))
        self._n_components = int(n_components)
        self._model = None

    def fit(self, df, target=None):
        from cadw.ml.pca import fit_pca
        self._model = fit_pca(
            df, self._X_cols, n_components=self._n_components,
            memory_guard=df._memory_guard)
        return self

    def transform(self, df):
        return self._model.transform(df)

    def estimate_state_bytes(self, dc):
        d = len(self._X_cols)
        k = self._n_components
        return d * k * 8 + d * 8 + 256

    def state_bytes(self):
        if self._model is None:
            return 0
        if hasattr(self._model, "state_nbytes"):
            return self._model.state_nbytes()
        return 0

    def to_dict(self):
        return {"X_cols": list(self._X_cols),
                "n_components": self._n_components}

    @classmethod
    def from_dict(cls, d):
        return cls(d["X_cols"], d["n_components"])
