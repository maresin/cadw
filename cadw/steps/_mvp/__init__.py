"""Reference MVP steps (real copies, not mocks).

Used for pipeline integration testing with real models.
Users can copy these as templates for their own steps.
"""
from .linear_regressor import LinearRegressor
from .standardize import Standardize
from .pca_projector import PCAProjector

from cadw.pipeline import _STEP_REGISTRY

for _cls, _name in (
    (Standardize, "Standardize"),
    (PCAProjector, "PCAProjector"),
    (LinearRegressor, "LinearRegressor"),
):
    _STEP_REGISTRY[_name] = _cls
    _cls.__serialization_name__ = _name

__all__ = ["Standardize", "PCAProjector", "LinearRegressor"]
