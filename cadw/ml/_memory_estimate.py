"""v0.9.97: pre-flight RAM estimate for named ops/models.

Loads YAML cards from ``docs/ml_tools/*.yaml``, reads
``bounded_state_dims`` + ``transient_factor``, evaluates the formula
against ``MemoryGuard.available_bytes()``.

See docs/SESSION_BRIEF_v0.9.97.md §3 and
docs/ml_tools/METHODOLOGY_ALGORITHM_SELECTION.md §5-§6 for context.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


YAML_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "ml_tools"


@lru_cache(maxsize=1)
def _load_cards() -> dict[str, dict]:
    """Load all verified YAML cards, index by 3 keys per card.

    Keys per card: ``cadw_function`` (e.g. "fit_linear"),
    ``"cadw_module::cadw_function"`` (e.g. "cadw.ml.linear::fit_linear"),
    and ``name`` (e.g. "Linear/Ridge").

    Cached at process lifetime (~500 KB YAML, ~10 ms parse).
    Returns empty dict if docs/ml_tools is missing (installed package).
    Do NOT mutate the returned dict.
    """
    index: dict[str, dict] = {}
    if not YAML_DIR.exists():
        return index
    for yp in sorted(YAML_DIR.glob("*.yaml")):
        if yp.name == "benchmarks.yaml":
            continue
        try:
            data = yaml.safe_load(yp.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        for st in data.get("subtypes", []):
            for card in st.get("models", []):
                status = card.get("implementation_status") or {}
                if not status.get("verified_in_cadw"):
                    continue
                fn = card.get("cadw_function")
                mod = card.get("cadw_module")
                if fn:
                    index.setdefault(fn, card)
                if mod and fn:
                    index.setdefault(f"{mod}::{fn}", card)
                name = card.get("name")
                if name:
                    index.setdefault(name, card)
    return index


def _lookup_card(op: str) -> dict | None:
    return _load_cards().get(op)


def estimate_from_yaml(op: str, guard, **params) -> dict:
    """Evaluate RAM formula for ``op`` against ``guard`` budget.

    Parameters
    ----------
    op : str
        cadw_function / "module::function" / YAML card name.
    guard : MemoryGuard
        Provides ``available_bytes()``.
    **params
        Axis values: n, d, k, K, batch, w, N, n_groups.
        Missing axis with power>0 in ``bounded_state_dims`` -> ValueError.

    Returns
    -------
    dict with keys: fits (bool), est_ram_mb (float|None),
    bottleneck (str), available_mb (float), alternatives (list[str]).
    For refuse_at_guard cards, ``est_ram_mb`` is None.
    """
    card = _lookup_card(op)
    if card is None:
        keys = sorted(_load_cards().keys())
        sample = keys[:5]
        raise KeyError(
            f"estimate_memory: {op!r} not in YAML cards. "
            f"Try one of: {sample}{'...' if len(keys) > 5 else ''}"
        )

    avail_bytes = guard.available_bytes()
    avail_mb = avail_bytes / 1e6

    if card.get("n_scalability") == "refuse_at_guard":
        return {
            "fits": False,
            "est_ram_mb": None,
            "bottleneck": (
                "refuse_at_guard (unbounded state; see YAML card)"
            ),
            "available_mb": avail_mb,
            "alternatives": _alternatives(card),
        }

    dims = card.get("bounded_state_dims")
    if dims is None:
        raise ValueError(
            f"{op}: bounded_state_dims missing "
            f"(v0.9.97 registration debt; run "
            f"scripts/patch_v0997_yaml_dims.py --apply)"
        )

    tf = float(card.get("transient_factor", 1.0))

    missing = [a for a in dims if a not in params]
    if missing:
        raise ValueError(
            f"{op}: missing axis params {missing}. "
            f"Required: {list(dims)}"
        )

    units = sum(params[a] ** p for a, p in dims.items())
    est_bytes = float(units) * 8.0 * tf
    bottleneck = " + ".join(
        f"{a}^{p}={params[a] ** p:.2e}" for a, p in dims.items()
    ) + f" × 8 B × {tf}"

    return {
        "fits": est_bytes <= avail_bytes,
        "est_ram_mb": est_bytes / 1e6,
        "bottleneck": bottleneck,
        "available_mb": avail_mb,
        "alternatives": _alternatives(card),
    }


def _alternatives(card: dict) -> list[str]:
    """Curated alternatives by family. Empty if none match."""
    combined = (
        (card.get("name") or "") + " "
        + (card.get("cadw_function") or "")
    ).lower()
    if any(w in combined for w in (
            "linear", "logistic", "ridge", "lasso", "enet", "elastic")):
        return ["solver='sgd' (O(d) state, iterative)"]
    if "binseg" in combined:
        return ["pelt (O(window) state)", "stray (refuse at n > 20k)"]
    if "lambdamart" in combined or "gbdt" in combined:
        return ["reduce n_estimators / max_depth"]
    if "pca" in combined:
        return ["method='randomized' (O(d·k) state)"]
    return []
