"""v0.9.176: DataFrameOpCard schema + loader tests."""
import pytest

from cadw.core.cards import (
    DataFrameOpCard, load_df_op_cards,
    DF_OP_CATEGORIES, DF_OP_STATUSES,
)


def _minimal(**over):
    base = dict(
        name="test_op", cadw_method="CadwDataFrame.test_op",
        category="cumulative", status="draft",
    )
    base.update(over)
    return base


def _validate(card):
    from cadw.core.cards import _validate_df_op_card
    return _validate_df_op_card(card)


def test_minimal_draft_ok():
    c = DataFrameOpCard(**_minimal())
    assert _validate(c) == []


def test_all_categories():
    expected = {"cumulative", "rolling", "shift", "fill", "reshape",
                "sample", "apply", "export"}
    assert DF_OP_CATEGORIES == expected


def test_all_statuses():
    assert DF_OP_STATUSES == frozenset(
        {"verified", "draft", "escape_hatch"})


def test_invalid_category():
    c = DataFrameOpCard(**_minimal(category="bogus"))
    errs = _validate(c)
    assert any("category" in e for e in errs)


def test_invalid_status():
    c = DataFrameOpCard(**_minimal(status="bogus"))
    errs = _validate(c)
    assert any("status" in e for e in errs)


def test_escape_hatch_status_requires_flag():
    c = DataFrameOpCard(**_minimal(status="escape_hatch", escape_hatch=False))
    errs = _validate(c)
    assert any("escape_hatch=True" in e for e in errs)


def test_escape_hatch_with_flag_ok():
    c = DataFrameOpCard(**_minimal(status="escape_hatch", escape_hatch=True))
    assert _validate(c) == []


def test_verified_requires_ram_formula():
    c = DataFrameOpCard(**_minimal(status="verified"))
    errs = _validate(c)
    assert any("ram_formula" in e for e in errs)


def test_verified_with_formula_ok():
    c = DataFrameOpCard(**_minimal(
        status="verified", bounded_state=True,
        ram_formula="3.5 * batch * row_size + 512",
    ))
    assert _validate(c) == []


# === loader ===

def _write_yaml(tmp_path, content):
    f = tmp_path / "df_operations.yaml"
    f.write_text(content)
    return f


def test_load_valid_yaml(tmp_path):
    f = _write_yaml(tmp_path, """
operations:
  - name: cumsum
    cadw_method: "CadwDataFrame.cumsum"
    category: cumulative
    status: verified
    bounded_state: true
    ram_formula: "3.5 * batch * row_size + 512"
""")
    cards = load_df_op_cards(f)
    assert len(cards) == 1
    assert cards[0].name == "cumsum"
    assert cards[0].bounded_state is True


def test_load_invalid_raises(tmp_path):
    f = _write_yaml(tmp_path, """
operations:
  - name: broken
    cadw_method: "x"
    category: bogus
    status: verified
""")
    with pytest.raises(ValueError, match="schema violation"):
        load_df_op_cards(f)


def test_load_current_df_operations_file():
    """Sanity: docs/df_operations.yaml must validate against schema."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    p = repo_root / "docs" / "df_operations.yaml"
    assert p.exists(), f"{p} missing"
    cards = load_df_op_cards(p)
    assert len(cards) >= 15  # 20 cards in v0.9.175

    # Spot-checks
    by_name = {c.name: c for c in cards}
    assert "cumsum" in by_name
    assert by_name["cumsum"].status == "verified"
    assert by_name["cumsum"].bounded_state is True
    assert "apply" in by_name
    assert by_name["apply"].status == "escape_hatch"
    assert by_name["apply"].escape_hatch is True
