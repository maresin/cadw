"""CompositeStep (moved from cadw/pipeline.py v1.0.0)."""
from __future__ import annotations
from typing import Optional

from cadw.core.models import DataChars


class CompositeStep:
    """v0.9.236: composite of child steps (sequential fit, chained transform).

    Behaves like flat pipeline equivalent: fit all children on input df,
    transform chains through children. Scan/write counts equal flat.

    is_barrier = True if any child barrier; else False (fold candidate,
    v0.9.237 streaming will use transform_batch for actual fold).
    """
    cadw_function = None

    def __init__(self, steps: list) -> None:
        if not steps:
            raise ValueError("CompositeStep: empty steps")
        self._steps = list(steps)
        # any child barrier -> barrier
        def _barrier(s) -> bool:
            v = getattr(s, "is_barrier", True)
            return v() if callable(v) else bool(v)
        self.is_barrier = any(_barrier(s) for s in self._steps)
        # v0.9.237b: streamable if all children streamable
        self._streamable = all(
            hasattr(s, "fit_iter")
            and hasattr(s, "transform_batch")
            and hasattr(s, "output_schema")
            for s in self._steps)

    def fit(self, df, target=None):
        for s in self._steps:
            is_pred = (hasattr(s, "predict")
                       and not hasattr(s, "transform"))
            s.fit(df, target=target if is_pred else None)
        return self

    def transform(self, df):
        current = df
        for s in self._steps:
            if not hasattr(s, "transform"):
                raise RuntimeError(
                    f"CompositeStep.transform: "
                    f"{type(s).__name__} has no transform")
            current = s.transform(current)
        return current

    # v0.9.237b: streaming API (delegates to children; only if all streamable)
    def fit_iter(self, batch_factory, template_df):
        if not self._streamable:
            raise NotImplementedError(
                "CompositeStep.fit_iter: not all children streamable")
        cur = batch_factory
        for s in self._steps:
            s.fit_iter(cur, template_df)
            prev = cur
            fn = s.transform_batch
            cur = (lambda pf=prev, f=fn: (f(b) for b in pf()))
        return self

    def transform_batch(self, batch):
        if not self._streamable:
            raise NotImplementedError(
                "CompositeStep.transform_batch: not all children streamable")
        for s in self._steps:
            batch = s.transform_batch(batch)
        return batch

    def output_schema(self, input_schema):
        if not self._streamable:
            raise NotImplementedError(
                "CompositeStep.output_schema: not all children streamable")
        schema = input_schema
        for s in self._steps:
            schema = s.output_schema(schema)
        return schema

    def estimate_state_bytes(self, dc):
        return sum(s.estimate_state_bytes(dc) for s in self._steps)

    def state_bytes(self):
        return sum(
            s.state_bytes() if hasattr(s, "state_bytes") else 0
            for s in self._steps)

    def to_dict(self):
        return {
            "kind": "CompositeStep",
            "steps": [
                {"type": type(s).__name__,
                 "params": s.to_dict() if hasattr(s, "to_dict") else {}}
                for s in self._steps
            ],
        }
