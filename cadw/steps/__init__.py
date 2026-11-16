"""Pipeline steps: base classes + concrete infra + FrameMethod."""
from .base import Step, Transformer, Model
from .frame_method import FrameMethod
from .composite import CompositeStep
from .column import ColumnTransformer

from cadw.pipeline import _STEP_REGISTRY

_STEP_REGISTRY["CompositeStep"] = CompositeStep
CompositeStep.__serialization_name__ = "CompositeStep"
_STEP_REGISTRY["ColumnTransformer"] = ColumnTransformer
ColumnTransformer.__serialization_name__ = "ColumnTransformer"
_STEP_REGISTRY["FrameMethod"] = FrameMethod
FrameMethod.__serialization_name__ = "FrameMethod"

__all__ = [
    "Step", "Transformer", "Model",
    "FrameMethod", "CompositeStep", "ColumnTransformer",
]
