"""G4: YAML state fields — validate format + track strict requirement.

Two tests:

1. Format validation (runs now): if a card has any of
   state_class / state_type / bounded_state / n_scalability /
   streaming_strategy / primitive_reuse, validate the value.

2. Strict requirement (deferred — controlled by
   REQUIRE_STATE_FIELDS): every card with `verified_in_cadw: true`
   must declare state_class and bounded_state. Today this is
   informational until P6 flips REQUIRE_STATE_FIELDS; after flip it
   is enforced (P6a/b registered D*+A*; P7 pending for E-O).

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
    "O(K)",  # v0.9.99: K-axis only (reverse_maps, tp/fp/fn)
    "O(1)", "O(d)", "O(d²)", "O(d^2)", "O(batch)", "O(model)",
    "O(n)", "O(n·w)", "O(n*w)", "O(n·K)", "O(n*K)", "O(N·d)", "O(N*d)",
}
VALID_STATE_TYPE = {
    "aggregation", "iterative", "partitioned", "cross_row",
    "windowed", "elementwise", "mixed",
}
VALID_N_SCAL = {"constant", "linear", "quadratic", "refuse_at_guard"}

# Flipped to True after P6 registered state fields in D*+A*.
# v0.9.97: bounded_state_dims added to all 84 verified cards
# (E/F/G/H/I/J/K/L/M/N/O included). G4 whitelist may still exclude
# some sections at the section-wildcard level; per-card check runs
# for non-whitelisted verified cards.
REQUIRE_STATE_FIELDS = True


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
    """G4-part2: verified cards declare state taxonomy fields.

    Strict mode: REQUIRE_STATE_FIELDS is True (since v0.9.87-P6)
    AND card not whitelisted. Every verified non-refuse card MUST
    declare:
      - state_class    (O(...) asymptotic class)
      - bounded_state  (bool; RAM bounded given disk)
      - bounded_state_dims (v0.9.97; {axis: power} formula components)
    Cards with n_scalability="refuse_at_guard" skip bounded_state_dims
    (they refuse at guard, no RAM formula).
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
        # v0.9.97: dims required for verified non-refuse cards
        if (card.get("n_scalability") != "refuse_at_guard"
                and card.get("bounded_state_dims") is None):
            missing.append("bounded_state_dims")
        if missing:
            violations.append(f"{yp.name}::{key} missing {missing}")

    assert not violations, (
        "G4: verified cards missing state taxonomy fields "
        "(state_class, bounded_state, bounded_state_dims). "
        "Run scripts/patch_v0997_yaml_dims.py --apply to fill "
        "bounded_state_dims mechanically, or add field manually.\n"
        + "\n".join(violations)
    )

def test_is_whitelisted_section_wildcard():
    """v0.9.87-P6c-β-fix1: `E.*` must match E1.x, E2.x, E10.x (not EE1.x)."""
    from tests._gate_whitelist import is_whitelisted
    wl = {"E.*"}
    assert is_whitelisted(wl, "E1.IForest")
    assert is_whitelisted(wl, "E2.Logistic_balanced")
    assert is_whitelisted(wl, "E10.x")
    assert is_whitelisted(wl, "E1.*")     # subtype-level key still matches
    assert not is_whitelisted(wl, "F1.x")   # different section
    assert not is_whitelisted(wl, "EE1.x")  # not a real section key
    assert not is_whitelisted(wl, "D1.x")   # different section


def test_is_whitelisted_subtype_wildcard():
    """v0.9.87-P6c-β-fix1: `D3.1.*` matches only D3.1.x (dots in subtype id)."""
    from tests._gate_whitelist import is_whitelisted
    wl = {"D3.1.*"}
    assert is_whitelisted(wl, "D3.1.Sub-PCA")
    assert is_whitelisted(wl, "D3.1.*")
    assert not is_whitelisted(wl, "D3.2.BOCPD")
    assert not is_whitelisted(wl, "D3.x")
    assert not is_whitelisted(wl, "D1.NaiveForecaster")


def test_is_whitelisted_exact_and_no_wildcard():
    """Exact entry matches; non-wildcard entry matches nothing else."""
    from tests._gate_whitelist import is_whitelisted
    wl = {"cadw/ml/lambdamart.py::fit_lambdamart", "A1.GBDT"}
    assert is_whitelisted(wl, "cadw/ml/lambdamart.py::fit_lambdamart")
    assert is_whitelisted(wl, "A1.GBDT")
    assert not is_whitelisted(wl, "A1.LogisticRegression")
    assert not is_whitelisted(wl, "cadw/ml/gbdt.py::fit_gbdt_multiclass")

