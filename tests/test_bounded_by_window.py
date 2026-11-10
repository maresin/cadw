"""v0.9.176-c2a: ModelCard.bounded_by_window tests.

Construction path = _parse_model (mirrors production). Field override =
dataclasses.replace (clean, no required-positional-arg issues).
"""
from dataclasses import replace

import pytest

from docs.ml_tools.schema import _parse_model, _validate_model_state


def _minimal_dict(**overrides):
    base = {
        "name": "TestModel",
        "tier": "baseline",
        "tier_level": "tier1_core",
        "implementation_status": {"verified_in_cadw": True},
        "roles": ["baseline"],
        "purpose": "test",
        "bounded_state": True,
    }
    base.update(overrides)
    return base


def _make_card(**overrides):
    """Build via _parse_model, then apply field overrides."""
    base = _minimal_dict()
    card = _parse_model(base)
    if overrides:
        card = replace(card, **overrides)
    return card


def _errors(card):
    errs = []
    _validate_model_state(card, "TestModel", lambda m: errs.append(m))
    return errs


# === parse ===

def test_parse_with_bounded_by_window():
    m = _minimal_dict(bounded_by_window={
        "axis": "n",
        "window_param": "window",
        "auto_from_guard": True,
        "truncation_loss": "in_sample [0, n-W)",
    })
    card = _parse_model(m)
    assert card.bounded_by_window == {
        "axis": "n",
        "window_param": "window",
        "auto_from_guard": True,
        "truncation_loss": "in_sample [0, n-W)",
    }


def test_parse_absent_is_none():
    card = _parse_model(_minimal_dict())
    assert card.bounded_by_window is None


# === validation ===

def test_valid_shape():
    card = _make_card(bounded_by_window={
        "axis": "n",
        "window_param": "max_window",
        "auto_from_guard": True,
        "truncation_loss": "changepoints before W",
    })
    assert not _errors(card)


def test_not_dict():
    card = _make_card(bounded_by_window="not-a-dict")
    errs = _errors(card)
    assert any("must be dict" in e for e in errs), errs


def test_missing_keys():
    card = _make_card(bounded_by_window={"axis": "n"})
    errs = _errors(card)
    for k in ("window_param", "auto_from_guard", "truncation_loss"):
        assert any(f"missing key {k!r}" in e for e in errs), (
            f"missing {k}: {errs}"
        )


def test_wrong_types():
    card = _make_card(bounded_by_window={
        "axis": 123,
        "window_param": ["x"],
        "auto_from_guard": "yes",
        "truncation_loss": 42,
    })
    errs = _errors(card)
    assert any("axis must be str" in e for e in errs), errs
    assert any("window_param must be str" in e for e in errs), errs
    assert any("auto_from_guard must be bool" in e for e in errs), errs
    assert any("truncation_loss must be str" in e for e in errs), errs


def test_coexists_with_bounded_by_domain():
    """Transition v0.9.182-186: both fields may be present."""
    card = _make_card(
        bounded_state=False,
        bounded_by_domain={"axis": "n", "bound": 1_000_000,
                           "reason": "test transition"},
        bounded_by_window={
            "axis": "n",
            "window_param": "window",
            "auto_from_guard": True,
            "truncation_loss": "in_sample [0, n-W)",
        },
    )
    errs = _errors(card)
    coexist = [e for e in errs if "bounded_by_window" in e]
    assert not coexist, coexist


def test_bounded_state_true_with_bounded_by_window():
    """Post-refactor (v0.9.182+): state O(model), fit-transient via window."""
    card = _make_card(
        bounded_state=True,
        bounded_by_window={
            "axis": "n",
            "window_param": "window",
            "auto_from_guard": True,
            "truncation_loss": "in_sample [0, n-W)",
        },
    )
    errs = _errors(card)
    bbw_errs = [e for e in errs if "bounded_by_window" in e]
    assert not bbw_errs, bbw_errs
