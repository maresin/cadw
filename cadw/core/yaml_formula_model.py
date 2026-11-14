"""YamlFormulaModel — OpMemoryModel driven by YAML ram_formula.

v0.9.232 (MEMORY_GUARD_ARCHITECTURE.md §2). Validated in
/tmp/memory_guard_proto_v2.py (11/11 equivalence vs individual classes).

Design:
  - 1 class, N instances. Formula from docs/ml_tools/*.yaml (source of truth).
  - Lazy registration on first get_registered_model() call.
  - Semantics (validated): state>budget -> MemoryLimitExceeded;
    batch<1 -> None; unknown op -> None; invariant asserted.

Anti-scope (per doc §8): no individual memory classes, no composite
transformer estimate/solve, no auto-inference from measurement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.core.models import DataChars, OpMemoryModel, Params

logger = logging.getLogger(__name__)


@dataclass
class BatchParams(Params):
    """Generic batch params for any batch-linear op."""
    batch_size: int
    state_bytes: int = 0


class YamlFormulaModel(OpMemoryModel):
    """Generic OpMemoryModel from YAML ram_formula.

    - estimate: evaluate formula with env = dc + params
    - solve:    batch-linear inverse. Returns None for wide/batch-indep.
    """

    def __init__(self, op_name: str, card: dict) -> None:
        self.op_name = op_name
        self._card = card
        self._formula = card.get("ram_formula")
        if not self._formula:
            raise ValueError(f"YamlFormulaModel({op_name}): no ram_formula")

    def _env(self, dc: DataChars) -> dict:
        env = {
            "n": int(dc.n_rows),
            "row_size": int(dc.row_size_bytes),
            "d": len(dc.schema),
            "batch": 1,
        }
        if dc.k_est is not None:
            kk = int(dc.k_est)
            env["k_est"] = kk
            env["k"] = kk
            env["K"] = kk
        return env

    def _eval(self, dc: DataChars, batch: int, **extra) -> int:
        from cadw.ml import _memory_estimate as me
        env = self._env(dc)
        env["batch"] = int(batch)
        env.update(me._load_constants())
        env.update({"max": max, "min": min, "abs": abs})
        env.update(extra)
        return int(me._safe_eval(self._formula, env))

    def estimate(self, params: Params, dc: DataChars) -> int:
        if isinstance(params, BatchParams):
            return self._eval(dc, batch=params.batch_size)
        raise TypeError(
            f"YamlFormulaModel({self.op_name}): unknown params "
            f"{type(params).__name__}")

    def solve(self, available_bytes: int, dc: DataChars, *,
              safety: float = 0.7, **extra) -> Optional[Params]:
        budget = int(available_bytes * safety)
        try:
            e1 = self._eval(dc, batch=1, **extra)
            e1k = self._eval(dc, batch=1000, **extra)
        except Exception as exc:
            logger.debug(
                "YamlFormulaModel(%s): eval failed (%s)", self.op_name, exc)
            return None
        coef = (e1k - e1) / 999.0
        if coef <= 0:
            return None  # batch-independent formula
        state = e1 - int(round(coef))
        if state >= budget:
            raise MemoryLimitExceeded(
                f"{self.op_name}: state={state} >= budget={budget} "
                f"(reduce params or increase budget)")
        slack = budget - state
        batch = int(slack / coef)
        if batch < 1:
            return None
        est = self._eval(dc, batch=batch, **extra)
        if est > budget:
            raise RuntimeError(
                f"{self.op_name}: invariant violated: "
                f"estimate={est} > budget={budget}")
        return BatchParams(batch_size=batch, state_bytes=state)

    def describe(self) -> str:
        return f"YamlFormulaModel({self.op_name})"


# ============================================================
# Registry (lazy init from _load_cards)
# ============================================================

_REGISTRY: dict[str, YamlFormulaModel] = {}
_INITIALIZED = False


def _init_registry() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True  # set BEFORE loading; prevents recursive init
    try:
        from cadw.ml import _memory_estimate as me
        cards = me._load_cards()
    except Exception as exc:
        logger.warning("YamlFormulaModel: _load_cards failed: %s", exc)
        return
    seen: set[str] = set()
    for op_key, card in cards.items():
        if not isinstance(card, dict):
            continue
        if not card.get("ram_formula"):
            continue
        impl = card.get("implementation_status") or {}
        if impl.get("status") != "verified":
            continue
        fn = card.get("cadw_function") or op_key
        if fn in seen:
            continue
        seen.add(fn)
        try:
            _REGISTRY[fn] = YamlFormulaModel(fn, card)
        except Exception as exc:
            logger.debug("YamlFormulaModel skip %r: %s", fn, exc)


def get_registered_model(op_name: str) -> Optional[YamlFormulaModel]:
    _init_registry()
    return _REGISTRY.get(op_name)


def list_registered() -> list[str]:
    _init_registry()
    return sorted(_REGISTRY.keys())


def _reset_registry() -> None:
    """Reset for tests. Do not call in production."""
    global _INITIALIZED
    _REGISTRY.clear()
    _INITIALIZED = False


# ============================================================
# plan_chain (module-level; MemoryGuard.plan_chain wraps)
# ============================================================

def plan_chain(ops: list[str], dc: DataChars, *,
               budget: int, safety: float = 0.7) -> dict:
    """Plan chain of ops via YamlFormulaModel (conservative sum).

    Returns {fits, peak_bytes, steps, reason}.
    Conservative: peak = sum(all step peaks). Refined composition in P2b.
    """
    steps = []
    peak = 0
    for op in ops:
        m = get_registered_model(op)
        if m is None:
            return {"fits": False, "peak_bytes": peak, "steps": steps,
                    "reason": f"{op}: not registered"}
        try:
            params = m.solve(budget, dc, safety=safety)
        except MemoryLimitExceeded as exc:
            return {"fits": False, "peak_bytes": peak, "steps": steps,
                    "reason": str(exc)}
        if params is None:
            return {"fits": False, "peak_bytes": peak, "steps": steps,
                    "reason": f"{op}: no fit"}
        est = m.estimate(params, dc)
        steps.append({"op_name": op, "params": params, "estimate": est})
        peak += est
    return {"fits": True, "peak_bytes": peak, "steps": steps, "reason": None}
