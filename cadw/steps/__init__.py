"""Concrete pipeline steps (moved from cadw/pipeline.py v1.0.0)."""
from .composite import CompositeStep
from .column import ColumnTransformer
from cadw.pipeline import _STEP_REGISTRY

_STEP_REGISTRY['CompositeStep'] = CompositeStep
CompositeStep.__serialization_name__ = 'CompositeStep'
_STEP_REGISTRY['ColumnTransformer'] = ColumnTransformer
ColumnTransformer.__serialization_name__ = 'ColumnTransformer'

__all__ = ['CompositeStep', 'ColumnTransformer']
