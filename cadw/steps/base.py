"""Pipeline step base classes (v1.0.0-pipeline).

Three ABCs:
  - Step:              mandatory (fit, estimate_state_bytes, state_bytes,
                       to_dict, from_dict)
  - Transformer(Step): + transform (narrow by default: is_barrier=False)
  - Model(Step):       + predict (always barrier: is_barrier=True)

Duck-typing (Pipeline._is_predictor) remains for backward compat: any
object with fit + predict counts as a Model, regardless of inheritance.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cadw.dataframe import CadwDataFrame


class Step(ABC):
    is_barrier: bool = True
    cadw_function: Optional[str] = None

    @abstractmethod
    def fit(self, df: "CadwDataFrame", target: Optional[str] = None) -> "Step":
        ...

    @abstractmethod
    def estimate_state_bytes(self, dc) -> int: ...

    @abstractmethod
    def state_bytes(self) -> int: ...

    @abstractmethod
    def to_dict(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "Step": ...


class Transformer(Step):
    is_barrier: bool = False

    @abstractmethod
    def transform(self, df: "CadwDataFrame") -> "CadwDataFrame": ...


class Model(Step):
    is_barrier: bool = True

    @abstractmethod
    def predict(self, df: "CadwDataFrame") -> "CadwDataFrame": ...
