"""FrameMethod — wrap CadwDataFrame.<method>(**kwargs) as Transformer."""
from __future__ import annotations
from typing import Optional
from cadw.steps.base import Transformer


_NARROW_METHODS = frozenset({
    "dropna", "astype", "fillna", "clip", "replace",
    "where", "mask", "drop", "select", "rename",
})


class FrameMethod(Transformer):
    def __init__(self, method_name: str, **kwargs) -> None:
        if not isinstance(method_name, str) or not method_name:
            raise ValueError(
                f"FrameMethod: method_name must be non-empty str; "
                f"got {method_name!r}")
        self.method_name = method_name
        self.kwargs = dict(kwargs)
        self.is_barrier = method_name not in _NARROW_METHODS

    def fit(self, df, target=None):
        meth = getattr(type(df), self.method_name, None)
        if meth is None or not callable(meth):
            raise AttributeError(
                f"FrameMethod: CadwDataFrame has no method "
                f"{self.method_name!r}")
        return self

    def transform(self, df):
        return getattr(df, self.method_name)(**self.kwargs)

    def estimate_state_bytes(self, dc): return 0
    def state_bytes(self): return 0

    def to_dict(self):
        return {"method_name": self.method_name, "kwargs": self.kwargs}

    @classmethod
    def from_dict(cls, d):
        return cls(d["method_name"], **d.get("kwargs", {}))

    def __repr__(self):
        kw = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"FrameMethod({self.method_name!r}, {kw})"
