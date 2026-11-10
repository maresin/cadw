"""v0.9.176: PRIMITIVES.yaml seed loads against PrimitiveCard schema."""
from pathlib import Path

import pytest


def test_primitives_yaml_loads():
    from cadw.core.cards import load_primitive_cards
    repo_root = Path(__file__).resolve().parent.parent
    p = repo_root / "docs" / "ml_tools" / "PRIMITIVES.yaml"
    assert p.exists(), f"{p} missing"
    cards = load_primitive_cards(p)
    assert len(cards) >= 5, f"expected >=5 seed cards, got {len(cards)}"

    names = {c.name for c in cards}
    for expected in ("sigmoid", "arm_stats", "nmf_mu",
                     "reservoir_sample", "ols_fit"):
        assert expected in names, f"missing {expected}"


def test_primitives_batch_layer_bounded_state():
    """All batch-layer + verified primitives have bounded_state=True."""
    from cadw.core.cards import load_primitive_cards
    repo_root = Path(__file__).resolve().parent.parent
    p = repo_root / "docs" / "ml_tools" / "PRIMITIVES.yaml"
    cards = load_primitive_cards(p)
    for c in cards:
        if c.layer == "batch" and c.status == "verified":
            assert c.bounded_state is True, (
                f"{c.name}: batch+verified requires bounded_state=True"
            )


def test_primitives_leaf_layer_not_bounded():
    """Leaf-layer has bounded_state=False (caller must batch)."""
    from cadw.core.cards import load_primitive_cards
    repo_root = Path(__file__).resolve().parent.parent
    p = repo_root / "docs" / "ml_tools" / "PRIMITIVES.yaml"
    cards = load_primitive_cards(p)
    for c in cards:
        if c.layer == "leaf":
            assert c.bounded_state is False, (
                f"{c.name}: leaf-layer must have bounded_state=False"
            )


def test_primitives_formula_evidence_has_line_ref():
    """Every formula_evidence value should be a string (non-empty)."""
    from cadw.core.cards import load_primitive_cards
    repo_root = Path(__file__).resolve().parent.parent
    p = repo_root / "docs" / "ml_tools" / "PRIMITIVES.yaml"
    cards = load_primitive_cards(p)
    for c in cards:
        if c.ram_formula:
            assert c.formula_evidence, f"{c.name}: missing evidence"
            for k, v in c.formula_evidence.items():
                assert isinstance(v, str) and v.strip(), (
                    f"{c.name}: evidence[{k!r}] empty"
                )
