"""MultiPhaseModel — synthetic 2-phase barrier model (v0.9.240-B2, P-Y).

Proves the multi-phase plan principle
(docs/PIPELINE_MEMORY_ARCHITECTURE.md §4):
  plan_peak = max(phase_1_bytes, phase_2_bytes), NOT sum.

Between phases: real materialization (phase_1 output is a new dataset).
Real GBDT/LDA (internal barriers, per-phase peak from primitives) — P2b.

Why synthetic: real GBDT card is a single flat formula (not composite),
so it would not demonstrate `max(...)` plan. Synthetic makes the principle
explicit and testable; real composite formula is P2b (composite from
PRIMITIVES.yaml).
"""
from __future__ import annotations

import logging

from cadw.steps.base import Model

_log = logging.getLogger("cadw.pipeline")


class MultiPhaseModel(Model):
    """Synthetic 2-phase barrier model.

    fit: phase_1.fit(df) -> intermediate = phase_1.transform(df) [real
    materialization] -> phase_2.fit(intermediate, target=...).
    predict: phase_2.predict(phase_1.transform(df)).

    plan_peak semantics (P-Y): estimate_state_bytes returns
    max(phase_1_bytes, phase_2_bytes) — the whole point: plan must use
    max across internal phases, not sum.

    state_bytes: sum(phase_1.state_bytes(), phase_2.state_bytes()) — both
    states held for predict (independent of plan peak semantics).
    """
    is_barrier = True
    cadw_function = None  # synthetic; not registered in YAML formula registry

    def __init__(self, phase_1, phase_2, *,
                 phase_1_bytes: int = 0, phase_2_bytes: int = 0):
        self._phase_1 = phase_1
        self._phase_2 = phase_2
        self._phase_1_bytes = int(phase_1_bytes)
        self._phase_2_bytes = int(phase_2_bytes)

    def fit(self, df, target=None):
        _log.info(
            "fit multi_phase: phase 1 (%s) peak=%d B",
            type(self._phase_1).__name__, self._phase_1_bytes)
        self._phase_1.fit(df, target=None)
        intermediate = self._phase_1.transform(df)  # real materialization
        _log.info(
            "fit multi_phase: phase 2 (%s) peak=%d B, intermediate rows=%d",
            type(self._phase_2).__name__, self._phase_2_bytes,
            intermediate.count())
        self._phase_2.fit(intermediate, target=target)
        return self

    def predict(self, df):
        out = self._phase_1.transform(df)
        return self._phase_2.predict(out)

    def estimate_state_bytes(self, dc) -> int:
        # v0.9.240-B2 (P-Y): synthetic peak = max(phase_1, phase_2).
        # This is what plan must honour (not sum).
        return max(self._phase_1_bytes, self._phase_2_bytes)

    def state_bytes(self) -> int:
        return (self._phase_1.state_bytes()
                + self._phase_2.state_bytes())

    def to_dict(self):
        return {
            "phase_1": {
                "type": type(self._phase_1).__name__,
                "params": self._phase_1.to_dict(),
            },
            "phase_2": {
                "type": type(self._phase_2).__name__,
                "params": self._phase_2.to_dict(),
            },
            "phase_1_bytes": self._phase_1_bytes,
            "phase_2_bytes": self._phase_2_bytes,
        }

    @classmethod
    def from_dict(cls, d):
        from cadw.steps._mvp.standardize import Standardize
        from cadw.steps._mvp.pca_projector import PCAProjector
        from cadw.steps._mvp.linear_regressor import LinearRegressor
        _registry = {
            "Standardize": Standardize,
            "PCAProjector": PCAProjector,
            "LinearRegressor": LinearRegressor,
        }

        def _restore(sd):
            t = sd["type"]
            if t not in _registry:
                raise ValueError(
                    f"MultiPhaseModel.from_dict: unknown phase type {t!r}")
            return _registry[t].from_dict(sd["params"])

        return cls(
            _restore(d["phase_1"]),
            _restore(d["phase_2"]),
            phase_1_bytes=int(d["phase_1_bytes"]),
            phase_2_bytes=int(d["phase_2_bytes"]),
        )
