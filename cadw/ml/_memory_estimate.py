"""RAM estimate for named ops/models.

v0.9.103: primary path is ``ram_formula`` string in YAML card,
evaluated by ``evaluate_formula`` (calibration, no guard) or
``estimate_from_yaml`` (runtime, auto-inject batch/row_size).

``ram_formula`` grammar: ``+ - * / **`` with names. Names resolved
from ``memory_constants.yaml`` + caller params. No ``Call`` /
``Attribute`` / ``Subscript`` — only arithmetic on identifiers.

Legacy fallback: ``bounded_state_dims`` + ``transient_factor`` (v0.9.97),
deprecated, emits LegacyFormulaWarning.

See docs/ml_tools/STABILITY_PROTOCOL.md (protocol: code -> formula
-> measurement), AI_CONSTITUTION §5 (AP21-AP23).
"""

from __future__ import annotations

import ast
import warnings
from functools import lru_cache
from pathlib import Path

import yaml

from cadw.core.warnings import LegacyFormulaWarning


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


@lru_cache(maxsize=1)
def _load_constants() -> dict:
    """Load named constants from docs/ml_tools/memory_constants.yaml."""
    cp = YAML_DIR / "memory_constants.yaml"
    if not cp.exists():
        return {}
    try:
        data = yaml.safe_load(cp.read_text())
    except yaml.YAMLError:
        return {}
    profiles = (data or {}).get("profiles") or {}
    default = profiles.get("default") or {}
    return dict(default.get("constants") or {})


def _formula_names(formula: str) -> set[str]:
    """Extract Name identifiers from formula string. Empty on syntax error."""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _safe_eval(formula: str, env: dict) -> float:
    """Evaluate ram_formula. Only +/-/*//**/() and identifiers."""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid ram_formula: {e}") from e

    def _eval(n):
        if isinstance(n, ast.Constant):
            if not isinstance(n.value, (int, float)) or isinstance(n.value, bool):
                raise ValueError(
                    f"ram_formula: only numeric literals, got "
                    f"{type(n.value).__name__}")
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise KeyError(f"ram_formula: unknown name {n.id!r}")
            v = env[n.id]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(
                    f"ram_formula: {n.id!r} not numeric "
                    f"({type(v).__name__})")
            return float(v)
        if isinstance(n, ast.BinOp):
            l = _eval(n.left)
            r = _eval(n.right)
            if isinstance(n.op, ast.Add): return l + r
            if isinstance(n.op, ast.Sub): return l - r
            if isinstance(n.op, ast.Mult): return l * r
            if isinstance(n.op, ast.Div):
                if r == 0:
                    raise ZeroDivisionError("ram_formula: div by zero")
                return l / r
            if isinstance(n.op, ast.Pow): return l ** r
            raise ValueError(
                f"ram_formula: unsupported op {type(n.op).__name__}")
        if isinstance(n, ast.UnaryOp):
            v = _eval(n.operand)
            if isinstance(n.op, ast.USub): return -v
            if isinstance(n.op, ast.UAdd): return v
            raise ValueError(
                f"ram_formula: unsupported unary {type(n.op).__name__}")
        raise ValueError(
            f"ram_formula: unsupported node {type(n).__name__}")

    return float(_eval(tree.body))


def evaluate_formula(op: str, **params) -> dict:
    """Calibration path: pure formula evaluation, no guard, no budget."""
    card = _lookup_card(op)
    if card is None:
        raise KeyError(f"evaluate_formula: {op!r} not in YAML cards")
    formula = card.get("ram_formula")
    if not formula:
        raise ValueError(
            f"evaluate_formula: {op!r} has no ram_formula "
            f"(status=draft or legacy dims-only card)")

    constants = _load_constants()
    env = {**constants, **params}
    required = _formula_names(formula)
    unresolved = sorted(required - set(env.keys()))
    if unresolved:
        raise ValueError(
            f"evaluate_formula: {op!r} formula requires unresolved "
            f"names {unresolved}. Pass as **params.")
    try:
        est = _safe_eval(formula, env)
    except (ValueError, KeyError, ZeroDivisionError) as e:
        raise ValueError(
            f"evaluate_formula: {op!r} formula eval failed: {e}") from e
    return {
        "bytes": float(est),
        "formula": formula,
        "unknown_names": [],
    }


def derive_batch_size(op, df, guard, batch_size=None):
    """v0.9.106 (DEC-022/023): derive batch size from state-aware guard.

    If `batch_size` is provided, return as-is (user override).
    Otherwise: query op's estimated state, deduct from guard's usable
    budget, size remaining for batch. Ensures check_batch (in reduce)
    and check_state do not overcommit at large d.

    `op` must be a ReduceOp-like with `estimate_state_nbytes()` and
    `batch_memory_factor()`. `df` must have `._dataset.schema`.
    """
    if batch_size is not None:
        return int(batch_size)
    state_est = op.estimate_state_nbytes() or 0
    schema = df._dataset.schema
    row_size = guard.estimate_row_size_from_schema(schema)
    return int(guard.get_batch_rows(
        row_size_bytes=row_size,
        expansion_factor=op.batch_memory_factor(),
        state_bytes=state_est,
    ))


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

    formula = card.get("ram_formula")
    if formula:
        env = dict(params)
        if "batch" not in env:
            # v0.9.106 (DEC-022 follow-up): pre-flight uses MINIMUM
            # viable batch, not get_batch_rows — the question is
            # "will model fit at all?" not "what is peak at production
            # batch?". Production sizing = get_batch_rows(state_bytes=...)
            # called directly by fit_* (v0.9.106.3).
            # Also fixes latent bug: prior call passed `op` (str) as
            # row_size_bytes; TypeError was swallowed by except: pass.
            try:
                env["batch"] = guard.min_batch_rows()
            except Exception as e:
                warnings.warn(
                    f"estimate_from_yaml({op!r}): min_batch_rows "
                    f"failed ({e}); falling back to batch=1000",
                    stacklevel=2)
                env["batch"] = 1000
        if "row_size" not in env:
            rs = getattr(guard, "row_size_bytes", None)
            if rs:
                env["row_size"] = rs
        constants = _load_constants()
        merged = {**constants, **env}
        required = _formula_names(formula) - set(constants.keys())
        missing_names = sorted(required - set(env.keys()))
        if missing_names:
            raise ValueError(
                f"{op}: missing axis params {missing_names} for "
                f"ram_formula. Required: {sorted(required)}")
        try:
            est_bytes = _safe_eval(formula, merged)
        except (ValueError, KeyError, ZeroDivisionError) as e:
            raise ValueError(
                f"{op}: ram_formula eval failed: {e}") from e
        return {
            "fits": est_bytes <= avail_bytes,
            "est_ram_mb": est_bytes / 1e6,
            "bottleneck": f"ram_formula: {formula.strip()[:80]}",
            "available_mb": avail_mb,
            "alternatives": _alternatives(card),
        }

    dims = card.get("bounded_state_dims")
    if dims is None:
        raise ValueError(
            f"{op}: neither ram_formula nor bounded_state_dims present "
            f"(card incomplete; formula migration pending)"
        )

    warnings.warn(
        f"{op}: no ram_formula; using legacy dims+transient_factor "
        f"(deprecated; migration v0.9.104+)",
        LegacyFormulaWarning,
        stacklevel=2,
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
