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

    # Reserved env keys — auto-injected, dc.extras must not override
    _RESERVED = frozenset({
        "n", "d", "row_size", "batch",
        "k", "K", "k_est",
        "f64", "f32", "i8", "i16", "i32", "i64", "bool",
        "dict_entry_bytes", "py_str_overhead_bytes", "list_ptr_bytes",
        "max", "min", "abs",
    })

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
        # v0.9.234 (DEC-082): merge extras (hyperparams, data params).
        # Reserved keys filtered (auto-injected takes precedence).
        if dc.extras:
            for k, v in dc.extras.items():
                if k not in self._RESERVED:
                    env[k] = v
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
        """Inverse: largest batch s.t. estimate <= budget.

        v0.9.234 (DEC-082): linearity check (sample slopes at 1, 100,
        1000, 10000); non-linear formula (e.g. batch*batch) -> None.

        data-dependent/hyperparams: merge dc.extras + **extra.
        Returns None if: eval fails, batch-independent, non-linear.
        Raises MemoryLimitExceeded if state(batch=1) > budget.
        """
        budget = int(available_bytes * safety)
        try:
            e1 = self._eval(dc, batch=1, **extra)
            e100 = self._eval(dc, batch=100, **extra)
            e1k = self._eval(dc, batch=1000, **extra)
            e10k = self._eval(dc, batch=10000, **extra)
        except Exception as exc:
            logger.debug(
                "YamlFormulaModel(%s): eval failed (%s)", self.op_name, exc)
            return None
        s1 = (e100 - e1) / 99.0
        s2 = (e1k - e100) / 900.0
        s3 = (e10k - e1k) / 9000.0
        if s1 == 0 and s2 == 0 and s3 == 0:
            return None  # batch-independent formula (state-only)
        # Linearity: slopes must agree within 1%
        for a, b in ((s1, s2), (s2, s3)):
            denom = max(abs(b), 1.0)
            if abs(a - b) / denom > 0.01:
                logger.debug(
                    "YamlFormulaModel(%s): non-linear in batch "
                    "(s1=%.3g s2=%.3g s3=%.3g)", self.op_name, s1, s2, s3)
                return None
        coef = s2  # mid-range slope (more accurate than endpoints)
        state = e1 - int(round(coef))
        if state >= budget:
            raise MemoryLimitExceeded(
                f"{self.op_name}: state={state} >= budget={budget} "
                f"(reduce params or increase budget)")
        slack = budget - state
        batch = int(slack / coef) if coef > 0 else 0
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
               budget: int, state_limit: Optional[int] = None,
               safety: float = 0.7) -> dict:
    """Plan chain of ops via YamlFormulaModel.

    v0.9.235 (DEC-083): peak = max_k [cumulative_state_prior + peak_k].
    Peaks are sequential; state accumulates (all held for downstream).

    Returns {fits, peak_bytes, state_bytes, steps, reason}.
    """
    if state_limit is None:
        state_limit = budget // 2
    steps = []
    cumulative_state = 0
    total_peak = 0
    for op in ops:
        m = get_registered_model(op)
        if m is None:
            return {"fits": False, "peak_bytes": total_peak,
                    "state_bytes": cumulative_state, "steps": steps,
                    "reason": f"{op}: not registered"}
        try:
            params = m.solve(budget, dc, safety=safety)
        except MemoryLimitExceeded as exc:
            return {"fits": False, "peak_bytes": total_peak,
                    "state_bytes": cumulative_state, "steps": steps,
                    "reason": str(exc)}
        if params is None:
            # state-only formula (no batch term): eval at batch=1
            try:
                state_k = m._eval(dc, batch=1)
            except Exception as exc:
                return {"fits": False, "peak_bytes": total_peak,
                        "state_bytes": cumulative_state, "steps": steps,
                        "reason": f"{op}: eval failed: {exc}"}
            peak_k = state_k
            step_info = {"op_name": op, "params": None,
                         "estimate": peak_k}
        else:
            peak_k = m.estimate(params, dc)
            state_k = params.state_bytes
            step_info = {"op_name": op, "params": params,
                         "estimate": peak_k}
        step_total = cumulative_state + peak_k
        total_peak = max(total_peak, step_total)
        cumulative_state += state_k
        steps.append(step_info)
    peak_ok = total_peak <= budget
    state_ok = cumulative_state <= state_limit
    reason = None
    if not peak_ok:
        reason = f"peak={total_peak} > budget={budget}"
    elif not state_ok:
        reason = f"state={cumulative_state} > state_limit={state_limit}"
    return {"fits": peak_ok and state_ok, "peak_bytes": total_peak,
            "state_bytes": cumulative_state, "steps": steps,
            "reason": reason}
