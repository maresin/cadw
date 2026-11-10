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
    refuse_above: int | None = None
    alternative: str | None = None

    # Dependencies / consumers
    consumers: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    primitive_reuse: list[str] = field(default_factory=list)

    # Measurement
    measurement: dict | None = None
    notes: str | None = None


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
        if c.intrinsic_bound is None and c.refuse_above is None:
            errs.append(
                "layer=intrinsic requires intrinsic_bound or refuse_above"
            )

    if c.layer == "batch" and c.status == "verified" and c.bounded_state is not True:
        errs.append(
            "layer=batch + status=verified requires bounded_state=True"
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
