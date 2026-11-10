"""v0.9.176: ApproximationInfo + BudgetTier tests."""
import pytest

from cadw.core import ApproximationInfo, BudgetTier


def test_approximation_info_constructs():
    a = ApproximationInfo(
        type="window_adjusted", axis="n",
        requested=None, effective=10000,
        reason="budget_limit",
        accuracy_impact="negligible",
    )
    assert a.type == "window_adjusted"
    assert a.axis == "n"
    assert a.effective == 10000


def test_approximation_info_invalid_type():
    with pytest.raises(ValueError, match="invalid type"):
        ApproximationInfo(type="bogus", axis="n")


def test_approximation_info_invalid_axis():
    with pytest.raises(ValueError, match="invalid axis"):
        ApproximationInfo(type="solver_switch", axis="bogus")


def test_approximation_info_all_types_valid():
    for t in ("window_adjusted", "window_truncated", "solver_switch",
              "streaming_switch", "subsample", "output_spilled",
              "round_by_round", "blocked", "sparse", "recursive"):
        ApproximationInfo(type=t, axis="n")


def test_approximation_info_all_axes_valid():
    for axis in ("n", "d", "K", "V", "model", "other"):
        ApproximationInfo(type="solver_switch", axis=axis)


def test_approximation_info_str():
    a = ApproximationInfo(type="window_adjusted", axis="n", effective=500)
    s = str(a)
    assert "window_adjusted" in s
    assert "n" in s


def test_budget_tier_enum():
    assert BudgetTier.OPERATIONAL.value == "operational"
    assert BudgetTier.INSUFFICIENT_NO_ALT.value == "insufficient_no_alt"
