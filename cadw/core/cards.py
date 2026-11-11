"""PrimitiveCard + DataFrameOpCard (v0.9.176, DEC-067).

Separate from ModelCard (docs/ml_tools/schema.py):
- ModelCard         = fit_* / predict_* / transform_* models
- PrimitiveCard     = cadw/ops/* functions (batch/leaf/intrinsic layers)
- DataFrameOpCard   = CadwDataFrame methods (cumsum, rolling, fillna, ...)

Specs: PRIMITIVE_CARD_SPEC.md, DF_OPERATIONS_SPEC.md.

Loaders validate structure + cross-field consistency at load time.
Files:
- PrimitiveCard:  docs/ml_tools/PRIMITIVES.yaml
- DataFrameOpCard: docs/df_operations.yaml
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PRIMITIVE_LAYERS = frozenset({"batch", "leaf", "intrinsic"})
PRIMITIVE_STATUSES = frozenset({"verified", "draft", "planned"})
DF_OP_CATEGORIES = frozenset({
    "cumulative", "rolling", "shift", "fill", "reshape", "sample",
    "apply", "export",
})
DF_OP_STATUSES = frozenset({"verified", "draft", "escape_hatch"})


@dataclass
class PrimitiveCard:
    """Schema for cadw/ops/* function cards (PRIMITIVE_CARD_SPEC §2).

    layer:
      - batch     — state O(model) given fixed batch; primitive IS OOC.
      - leaf      — state O(n) output; caller must batch/chunk.
      - intrinsic — O(n^2)/O(n*d) unavoidable; caller controls n <= bound.
    """

    name: str
    cadw_function: str
    module: str
    layer: str                      # "batch" | "leaf" | "intrinsic"
    status: str                     # "verified" | "draft" | "planned"

    # RAM contract
    ram_formula: str | None = None
    min_ram_formula: str | None = None
    formula_evidence: dict[str, str] = field(default_factory=dict)
    bounded_state: bool | None = None

    # Leaf-layer specifics
    streaming_variant: str | None = None
    intrinsic_bound: dict | None = None
    # v0.9.189 (DEC-072): window-truncation. When set, caller bounds n by
    # W_eff (via _collect_last_w or similar) before invoking this leaf;
    # effective RAM = O(W_eff), not O(n). Shape:
    #   {axis: "n", caller_prim: "_collect_last_w",
    #    min_window: "<expr>", reason: "<why window is semantic>"}
    # Mutually exclusive with intrinsic_bound (which claims window
    # truncation rejected).
    window_truncatable: dict | None = None
    refuse_above: int | None = None
    alternative: str | None = None

    # Dependencies / consumers
    consumers: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    primitive_reuse: list[str] = field(default_factory=list)

    # Measurement
    measurement: dict | None = None
    notes: str | None = None

    # v0.9.208: OOC-by-card. For layer=leaf|intrinsic: how to use this
    # primitive in OOC context (narrative). Caller responsibility note:
    # what model must do (chunk, use streaming_variant, use sparse).
    # Empty for layer=batch (already OOC by design).
    ooc_usage: str | None = None


@dataclass
class DataFrameOpCard:
    """Schema for CadwDataFrame method cards (DF_OPERATIONS_SPEC §2).

    category:
      cumulative | rolling | shift | fill | reshape | sample | apply | export
    """

    name: str
    cadw_method: str
    category: str
    status: str                     # "verified" | "draft" | "escape_hatch"

    ram_formula: str | None = None
    min_ram_formula: str | None = None
    bounded_state: bool | None = None
    memory_guard_site: str | None = None
    variants: dict | None = None
    escape_hatch: bool = False
    depends_on: list[str] = field(default_factory=list)
    measurement: dict | None = None
    notes: str | None = None


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def _validate_primitive_card(c: PrimitiveCard) -> list[str]:
    errs: list[str] = []
    if c.layer not in PRIMITIVE_LAYERS:
        errs.append(
            f"layer {c.layer!r} not in {sorted(PRIMITIVE_LAYERS)}"
        )
    if c.status not in PRIMITIVE_STATUSES:
        errs.append(
            f"status {c.status!r} not in {sorted(PRIMITIVE_STATUSES)}"
        )

    if c.status == "verified" and not c.ram_formula:
        errs.append("status=verified requires ram_formula")

    if c.ram_formula and not c.formula_evidence:
        errs.append("ram_formula set but formula_evidence empty")

    if c.layer == "intrinsic":
        if (c.intrinsic_bound is None and c.refuse_above is None
                and c.window_truncatable is None):
            errs.append(
                "layer=intrinsic requires intrinsic_bound or refuse_above "
                "or window_truncatable"
            )

    # v0.9.189 (DEC-072): intrinsic_bound and window_truncatable mutually
    # exclusive; intrinsic_bound must explain WHY window rejected.
    if c.intrinsic_bound is not None and c.window_truncatable is not None:
        errs.append(
            "intrinsic_bound and window_truncatable are mutually exclusive "
            "(DEC-072: window check before intrinsic)"
        )
    if c.intrinsic_bound is not None:
        reason = (c.intrinsic_bound or {}).get("reason", "")
        if not reason or len(str(reason).strip()) < 20:
            errs.append(
                "intrinsic_bound.reason must be >=20 chars explaining why "
                "window-truncation was rejected (DEC-072)"
            )
    if c.window_truncatable is not None:
        reason = (c.window_truncatable or {}).get("reason", "")
        if not reason or len(str(reason).strip()) < 20:
            errs.append(
                "window_truncatable.reason must be >=20 chars"
            )

    if c.layer == "batch" and c.status == "verified" and c.bounded_state is not True:
        errs.append(
            "layer=batch + status=verified requires bounded_state=True"
        )

    # v0.9.208: leaf/intrinsic must declare ooc_usage (soft in v0.9.208;
    # hard in v0.9.209 after auto-fill migration). Emitted as Warning,
    # not error — G25 gate handles enforcement.
    if c.layer in ("leaf", "intrinsic") and not c.ooc_usage:
        import warnings as _w
        _w.warn(
            f"PrimitiveCard {c.name!r}: layer={c.layer} without ooc_usage "
            f"(v0.9.208 soft; hard in v0.9.209, G25)",
            stacklevel=2,
        )

    return errs


def _validate_df_op_card(c: DataFrameOpCard) -> list[str]:
    errs: list[str] = []
    if c.category not in DF_OP_CATEGORIES:
        errs.append(
            f"category {c.category!r} not in {sorted(DF_OP_CATEGORIES)}"
        )
    if c.status not in DF_OP_STATUSES:
        errs.append(
            f"status {c.status!r} not in {sorted(DF_OP_STATUSES)}"
        )

    if c.status == "escape_hatch" and not c.escape_hatch:
        errs.append("status=escape_hatch requires escape_hatch=True")

    if c.status == "verified" and not c.ram_formula:
        errs.append("status=verified requires ram_formula")

    return errs


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def _dataclass_kwargs(raw: dict, cls) -> dict:
    """Extract only dataclass-known keys; coerce lists that came as None."""
    fields = cls.__dataclass_fields__
    kw: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in fields:
            continue
        # list fields default to [] on None from YAML
        f = fields[k]
        if v is None and f.default_factory is not f.default_factory.__class__():
            pass
        kw[k] = v
    # Ensure list/dict fields are non-None (they have default_factory)
    for k in ("formula_evidence", "consumers", "depends_on", "primitive_reuse"):
        if k in fields and k in kw and kw[k] is None:
            kw[k] = []
    return kw


def load_primitive_cards(path: str | Path) -> list[PrimitiveCard]:
    """Load + validate PRIMITIVES.yaml. Raise ValueError on schema violation."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PrimitiveCard file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    raw_cards = data.get("primitives") or []

    cards: list[PrimitiveCard] = []
    all_errors: list[str] = []
    for i, raw in enumerate(raw_cards):
        if not isinstance(raw, dict):
            all_errors.append(f"[{i}] not a dict: {type(raw).__name__}")
            continue
        name = raw.get("name", f"<{i}>")
        try:
            kw = _dataclass_kwargs(raw, PrimitiveCard)
            c = PrimitiveCard(**kw)
        except TypeError as e:
            all_errors.append(f"[{i}] {name}: {e}")
            continue
        errs = _validate_primitive_card(c)
        if errs:
            all_errors.extend(f"[{i}] {name}: {e}" for e in errs)
        cards.append(c)

    if all_errors:
        raise ValueError(
            f"{p}: {len(all_errors)} schema violation(s):\n"
            + "\n".join(all_errors[:20])
        )
    return cards


def load_df_op_cards(path: str | Path) -> list[DataFrameOpCard]:
    """Load + validate df_operations.yaml. Raise ValueError on violation."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DataFrameOpCard file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    raw_cards = data.get("operations") or []

    cards: list[DataFrameOpCard] = []
    all_errors: list[str] = []
    for i, raw in enumerate(raw_cards):
        if not isinstance(raw, dict):
            all_errors.append(f"[{i}] not a dict: {type(raw).__name__}")
            continue
        name = raw.get("name", f"<{i}>")
        try:
            kw = _dataclass_kwargs(raw, DataFrameOpCard)
            c = DataFrameOpCard(**kw)
        except TypeError as e:
            all_errors.append(f"[{i}] {name}: {e}")
            continue
        errs = _validate_df_op_card(c)
        if errs:
            all_errors.extend(f"[{i}] {name}: {e}" for e in errs)
        cards.append(c)

    if all_errors:
        raise ValueError(
            f"{p}: {len(all_errors)} schema violation(s):\n"
            + "\n".join(all_errors[:20])
        )
    return cards


__all__ = [
    "PrimitiveCard",
    "DataFrameOpCard",
    "PRIMITIVE_LAYERS",
    "PRIMITIVE_STATUSES",
    "DF_OP_CATEGORIES",
    "DF_OP_STATUSES",
    "load_primitive_cards",
    "load_df_op_cards",
]
