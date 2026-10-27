"""G4: YAML state fields — validate format + track strict requirement.

Two tests:

1. Format validation (runs now): if a card has any of
   state_class / state_type / bounded_state / n_scalability /
   streaming_strategy / primitive_reuse, validate the value.

2. Strict requirement (deferred — controlled by
   REQUIRE_STATE_FIELDS): every card with `verified_in_cadw: true`
   must declare state_class and bounded_state. Today this is
   informational (REQUIRE_STATE_FIELDS = False); flip to True after
   P6 (YAML registration across all cards).

Whitelist entries (plan gate-whitelist G4) are section-level
wildcards like `A.*` = all A subtype cards whitelisted until P6
updates each card.

See METHODOLOGY §7 (mandatory YAML fields), §8 (gate G4).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests._gate_whitelist import is_whitelisted, load_whitelist

ROOT = Path(__file__).parent.parent
YAML_DIR = ROOT / "docs" / "ml_tools"

VALID_STATE_CLASS = {
    "O(1)", "O(d)", "O(d²)", "O(d^2)", "O(batch)", "O(model)",
    "O(n)", "O(n·w)", "O(n*w)", "O(n·K)", "O(n*K)", "O(N·d)", "O(N*d)",
}
VALID_STATE_TYPE = {
    "aggregation", "iterative", "partitioned", "cross_row",
    "windowed", "elementwise", "mixed",
}
VALID_N_SCAL = {"constant", "linear", "quadratic", "refuse_at_guard"}

# Flip to True after P6 registers state fields in all cards.
REQUIRE_STATE_FIELDS = False


def _iter_yaml_cards():
    """Yield (file, subtype_id, card_dict) for every model card."""
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
            sid = st.get("id", "?")
            for card in st.get("models", []):
                yield yp, sid, card


def test_g4_state_field_format():
    """G4-part1: state_class/state_type/bounded_state format valid if present."""
    errors: list[str] = []
    for yp, sid, card in _iter_yaml_cards():
        name = card.get("name", "?")
        label = f"{yp.name}::{sid}.{name}"

        sc = card.get("state_class")
        if sc is not None and sc not in VALID_STATE_CLASS:
            errors.append(
                f"{label}: state_class={sc!r} not in {sorted(VALID_STATE_CLASS)}")

        st = card.get("state_type")
        if st is not None and st not in VALID_STATE_TYPE:
            errors.append(
                f"{label}: state_type={st!r} not in {sorted(VALID_STATE_TYPE)}")

        bs = card.get("bounded_state")
        if bs is not None and not isinstance(bs, bool):
            errors.append(
                f"{label}: bounded_state must be bool, got {type(bs).__name__}")

        ns = card.get("n_scalability")
        if ns is not None and ns not in VALID_N_SCAL:
            errors.append(
                f"{label}: n_scalability={ns!r} not in {sorted(VALID_N_SCAL)}")

        ss = card.get("streaming_strategy")
        if ss is not None and not isinstance(ss, (str, dict)):
            errors.append(
                f"{label}: streaming_strategy must be str or dict, "
                f"got {type(ss).__name__}")

        pr = card.get("primitive_reuse")
        if pr is not None and not isinstance(pr, list):
            errors.append(
                f"{label}: primitive_reuse must be list, got {type(pr).__name__}")

    assert not errors, (
        "G4 field format errors:\n" + "\n".join(errors)
    )


def test_g4_verified_cards_have_state_fields():
    """G4-part2: verified cards declare state_class + bounded_state.

    Strict mode: REQUIRE_STATE_FIELDS must be True AND card not
    whitelisted. Today REQUIRE_STATE_FIELDS = False, so this test
    only reports; flip after P6.
    """
    if not REQUIRE_STATE_FIELDS:
        pytest.skip("REQUIRE_STATE_FIELDS = False (deferred to P6)")

    wl = load_whitelist()["G4"]
    violations: list[str] = []
    for yp, sid, card in _iter_yaml_cards():
        status = card.get("implementation_status") or {}
        if not status.get("verified_in_cadw"):
            continue
        name = card.get("name", "?")
        key = f"{sid}.{name}"
        # also check section wildcard
        if is_whitelisted(wl, key) or is_whitelisted(wl, f"{sid}.*"):
            continue
        missing = []
        if card.get("state_class") is None:
            missing.append("state_class")
        if card.get("bounded_state") is None:
            missing.append("bounded_state")
        if missing:
            violations.append(f"{yp.name}::{key} missing {missing}")

    assert not violations, (
        "G4: verified cards missing state fields (flip REQUIRE_STATE_FIELDS"
        " only after P6 fills them):\n" + "\n".join(violations)
    )
