"""resolve_params — 6-level auto-switch (v0.9.176 skeleton, DEC-067).

Try user_params. If ``budget < min_ram(user_params)``, try auto_switch
candidates in min_ram-ascending order and pick first fit.

Skeleton (v0.9.176): only L1 (solver) and L5 (streaming) wired.
L2 (iteration), L3 (blocked), L4 (sparse), L6 (recursive) raise
NotImplementedError with TODO v0.9.187+.

Full integration with YAML card `auto_switch` registry and analytical
`min_ram_formula` in v0.9.177-178.
"""
from __future__ import annotations

from typing import Callable

from cadw.core.approximation import ApproximationInfo
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded


_IMPLEMENTED_LEVELS = frozenset({"L1", "L5"})  # solver, streaming
_NOT_YET_LEVELS = frozenset({"L2", "L3", "L4", "L6"})  # iter, blocked, sparse, rec


def resolve_params(
    model_name: str,
    mg: MemoryGuard,
    user_params: dict,
    *,
    auto_switch: list[dict] | None = None,
    min_ram_fn: Callable[[dict], int] | None = None,
) -> tuple[dict, list[ApproximationInfo]]:
    """Resolve effective params given budget and auto-switch registry.

    Parameters
    ----------
    model_name : str
        For error messages and options.
    mg : MemoryGuard
        Current guard (budget = mg.available_bytes()).
    user_params : dict
        User-supplied params (immutable; copy is made if switch).
    auto_switch : list[dict] | None
        Registry: each entry {from: {k: v}, to: {k: v}, level: "L1"|"L2"|...,
        approximation_type: "...", axis: "...", accuracy_impact: "..."}.
        If None/empty and user_params doesn't fit → raise.
    min_ram_fn : Callable[[dict], int] | None
        Analytical min_ram(params) -> bytes. If None, no auto-switch
        (return user_params unchanged; caller responsible for
        MemoryLimitExceeded).

    Returns
    -------
    (effective_params, approximation_list)
        effective_params == user_params if no switch needed.
        approximation_list = [] if exact; [ApproximationInfo] if switched.

    Raises
    ------
    MemoryLimitExceeded
        When: no matching auto_switch candidate fits, OR registry has
        no candidates for user_params shape, OR all candidates also
        don't fit. Enriched with model, params, min_required_bytes,
        available_bytes, options (R2-compliant).
    NotImplementedError
        For auto_switch level in {L2, L3, L4, L6}: not yet implemented
        (v0.9.187+).
    """
    budget = mg.available_bytes()

    if min_ram_fn is None:
        # No analytical floor → caller handles budget; return unchanged.
        return dict(user_params), []

    user_min = int(min_ram_fn(user_params))
    if user_min <= budget:
        return dict(user_params), []

    # Auto-switch needed
    if not auto_switch:
        raise MemoryLimitExceeded(
            f"{model_name}({user_params}) requires >= {user_min / 1e6:.2f}MB "
            f"minimum; budget = {budget / 1e6:.2f}MB. "
            f"No auto-switch registry available.",
            model=model_name,
            params=dict(user_params),
            min_required_bytes=user_min,
            available_bytes=budget,
            options=[
                f"increase MemoryGuard budget to >= {user_min} bytes",
                "reduce params (d, K, n_estimators, window, ...)",
            ],
        )

    candidates: list[tuple[dict, dict, dict, int]] = []
    for sw in auto_switch:
        from_params = sw.get("from") or {}
        to_params = sw.get("to") or {}
        level = sw.get("level", "")
        # Match: all from_params keys == user_params values
        if not all(user_params.get(k) == v for k, v in from_params.items()):
            continue
        if level in _NOT_YET_LEVELS:
            raise NotImplementedError(
                f"{model_name}: auto-switch level {level} "
                f"({sw.get('approximation_type', '?')}) not yet implemented "
                f"(skeleton v0.9.176; target v0.9.187+). "
                f"Set min_ram high enough or implement L{level[1]} first."
            )
        if level not in _IMPLEMENTED_LEVELS:
            continue
        effective = dict(user_params)
        effective.update(to_params)
        try:
            cand_min = int(min_ram_fn(effective))
        except Exception:
            continue
        candidates.append((effective, to_params, sw, cand_min))

    if not candidates:
        raise MemoryLimitExceeded(
            f"{model_name}({user_params}) requires >= {user_min / 1e6:.2f}MB "
            f"minimum; budget = {budget / 1e6:.2f}MB. "
            f"No matching auto-switch candidate for these params.",
            model=model_name,
            params=dict(user_params),
            min_required_bytes=user_min,
            available_bytes=budget,
            options=[
                f"increase budget to >= {user_min}",
                "reduce params",
                "check that auto_switch.from matches user_params shape",
            ],
        )

    # Smallest min_ram first
    candidates.sort(key=lambda c: c[3])

    for effective, to_params, sw, cand_min in candidates:
        if cand_min <= budget:
            info = ApproximationInfo(
                type=sw.get("approximation_type", "solver_switch"),
                axis=sw.get("axis", "d"),
                requested=_diff_values(user_params, to_params),
                effective=_diff_values(effective, user_params),
                reason="budget_limit",
                accuracy_impact=sw.get("accuracy_impact", "unknown"),
            )
            return effective, [info]

    # Nothing fits
    best_min = candidates[0][3]
    raise MemoryLimitExceeded(
        f"{model_name}({user_params}): all auto-switch candidates exceed "
        f"budget. Best variant requires >= {best_min / 1e6:.2f}MB; "
        f"budget = {budget / 1e6:.2f}MB.",
        model=model_name,
        params=dict(user_params),
        min_required_bytes=best_min,
        available_bytes=budget,
        options=[
            f"increase budget to >= {best_min}",
            "reduce params further",
        ],
    )


def _diff_values(effective: dict, baseline: dict) -> dict:
    """Return dict of keys where effective differs from baseline."""
    out: dict = {}
    for k, v in effective.items():
        if baseline.get(k) != v:
            out[k] = v
    return out
