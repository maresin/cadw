"""v0.9.176: PrimitiveCard schema + loader tests."""
from pathlib import Path

import pytest

from cadw.core.cards import (
    PrimitiveCard, load_primitive_cards,
    PRIMITIVE_LAYERS, PRIMITIVE_STATUSES,
)


def _minimal(**over):
    base = dict(
        name="test_prim", cadw_function="cadw.ops.x.test_prim",
        module="cadw.ops.x", layer="batch", status="planned",
    )
    base.update(over)
    return base


def _validate(card):
    from cadw.core.cards import _validate_primitive_card
    return _validate_primitive_card(card)


def test_minimal_planned_ok():
    c = PrimitiveCard(**_minimal())
    assert _validate(c) == []


def test_all_layers_valid():
    assert PRIMITIVE_LAYERS == frozenset({"batch", "leaf", "intrinsic"})


def test_all_statuses_valid():
    assert PRIMITIVE_STATUSES == frozenset({"verified", "draft", "planned"})


def test_invalid_layer():
    c = PrimitiveCard(**_minimal(layer="bogus"))
    errs = _validate(c)
    assert any("layer" in e for e in errs)


def test_invalid_status():
    c = PrimitiveCard(**_minimal(status="bogus"))
    errs = _validate(c)
    assert any("status" in e for e in errs)


def test_batch_verified_requires_bounded_state_true():
    c = PrimitiveCard(**_minimal(status="verified",
                                  ram_formula="batch * row_size + 512",
                                  formula_evidence={"batch": "x.py:1"}))
    errs = _validate(c)
    assert any("bounded_state=True" in e for e in errs)


def test_batch_verified_with_bounded_state_true():
    c = PrimitiveCard(**_minimal(status="verified", bounded_state=True,
                                  ram_formula="batch * row_size + 512",
                                  formula_evidence={"batch": "x.py:1"}))
    assert _validate(c) == []


def test_verified_requires_ram_formula():
    c = PrimitiveCard(**_minimal(status="verified", bounded_state=True))
    errs = _validate(c)
    assert any("ram_formula" in e for e in errs)


def test_ram_formula_requires_evidence():
    c = PrimitiveCard(**_minimal(status="draft",
                                  ram_formula="batch * row_size + 512"))
    errs = _validate(c)
    assert any("formula_evidence" in e for e in errs)


def test_intrinsic_requires_bound_or_refuse():
    c = PrimitiveCard(**_minimal(layer="intrinsic", status="draft"))
    errs = _validate(c)
    assert any("intrinsic_bound or refuse_above" in e for e in errs)


def test_intrinsic_with_intrinsic_bound_ok():
    c = PrimitiveCard(**_minimal(
        layer="intrinsic", status="draft",
        intrinsic_bound={"axis": "n", "bound": 10000,
                         "reason": "O(n^2) factor"},
    ))
    assert _validate(c) == []


def test_intrinsic_with_refuse_above_ok():
    c = PrimitiveCard(**_minimal(
        layer="intrinsic", status="draft", refuse_above=10000,
    ))
    assert _validate(c) == []


def test_leaf_with_streaming_variant():
    c = PrimitiveCard(**_minimal(
        layer="leaf", status="draft",
        streaming_variant="test_prim_chunks",
        bounded_state=False,
    ))
    assert _validate(c) == []


# === loader ===

def _write_yaml(tmp_path, content):
    f = tmp_path / "primitives.yaml"
    f.write_text(content)
    return f


def test_load_valid_yaml(tmp_path):
    f = _write_yaml(tmp_path, """
primitives:
  - name: arm_stats
    cadw_function: cadw.ops.stats.arm_stats
    module: cadw.ops.stats
    layer: batch
    status: verified
    bounded_state: true
    ram_formula: "k_arms * f64 + 3 * batch * row_size + 512"
    formula_evidence:
      state: "stats.py:75"
      batch: "stats.py:90"
""")
    cards = load_primitive_cards(f)
    assert len(cards) == 1
    assert cards[0].name == "arm_stats"
    assert cards[0].bounded_state is True
    assert cards[0].formula_evidence["state"] == "stats.py:75"


def test_load_invalid_raises(tmp_path):
    f = _write_yaml(tmp_path, """
primitives:
  - name: broken
    cadw_function: x
    module: y
    layer: bogus
    status: planned
""")
    with pytest.raises(ValueError, match="schema violation"):
        load_primitive_cards(f)


def test_load_missing_file_raises(tmp_path):
    f = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError):
        load_primitive_cards(f)


def test_load_empty_file_returns_empty(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("primitives: []\n")
    cards = load_primitive_cards(f)
    assert cards == []
