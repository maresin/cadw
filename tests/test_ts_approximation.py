"""v0.9.178: ApproximationMixin integration in TS forecasters (R6)."""
import warnings

import pytest

from cadw.core.approximation import ApproximationInfo, ApproximationMixin
from cadw.core.warnings import ApproximationWarning
from cadw.ts.models import ForecasterBase, NaiveForecaster
from cadw.ts.ets import ETSForecaster
from cadw.ts.thf import TemporalHierarchy


# === Mixin unit tests ===

def test_mixin_default_none():
    """Class attribute default = None."""
    class Dummy(ApproximationMixin):
        pass
    d = Dummy()
    assert d.approximation is None


def test_mixin_clear():
    class Dummy(ApproximationMixin):
        pass
    d = Dummy()
    d.approximation = [ApproximationInfo(type="solver_switch", axis="d")]
    d._clear_approximation()
    assert d.approximation is None


def test_mixin_record_appends():
    class Dummy(ApproximationMixin):
        pass
    d = Dummy()
    info = ApproximationInfo(
        type="window_adjusted", axis="n",
        requested=None, effective=1000,
        reason="budget_limit",
        accuracy_impact="negligible",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d._record_approximation(info)
    assert d.approximation == [info]
    assert len(w) == 1
    assert issubclass(w[0].category, ApproximationWarning)


def test_mixin_record_cascades():
    class Dummy(ApproximationMixin):
        pass
    d = Dummy()
    i1 = ApproximationInfo(type="window_adjusted", axis="n", effective=500)
    i2 = ApproximationInfo(type="window_truncated", axis="n",
                            lost_range=(0, 100))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        d._record_approximation(i1)
        d._record_approximation(i2)
    assert d.approximation == [i1, i2]


def test_mixin_record_wrong_type_raises():
    class Dummy(ApproximationMixin):
        pass
    d = Dummy()
    with pytest.raises(TypeError, match="expected ApproximationInfo"):
        d._record_approximation("not-an-info")


# === Integration ===

def test_forecaster_base_inherits_mixin():
    assert issubclass(ForecasterBase, ApproximationMixin)


def test_temporal_hierarchy_inherits_mixin():
    assert issubclass(TemporalHierarchy, ApproximationMixin)


@pytest.mark.parametrize("cls", [
    NaiveForecaster, ETSForecaster,
])
def test_forecaster_has_approximation_attr(cls):
    """All ForecasterBase subclasses get .approximation via inheritance."""
    instance = cls()
    assert instance.approximation is None
    assert hasattr(instance, "_record_approximation")
    assert hasattr(instance, "_clear_approximation")


def test_forecaster_record_and_warn():
    """Full cycle: record -> warning -> attribute set."""
    n = NaiveForecaster()
    info = ApproximationInfo(
        type="window_adjusted", axis="n",
        requested=None, effective=1000,
        reason="budget_limit",
        accuracy_impact="negligible",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n._record_approximation(info)
    assert n.approximation == [info]
    assert any("window_adjusted" in str(x.message) for x in w)
