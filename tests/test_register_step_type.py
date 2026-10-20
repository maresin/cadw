"""v0.9.1d: register_step_type tests."""
import warnings

import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.pipeline import (
    CastStep, _STEP_REGISTRY,
    list_step_types, register_step_type, unregister_step_type,
    Pipeline,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Snapshot/restore _STEP_REGISTRY so tests don't leak."""
    original = dict(_STEP_REGISTRY)
    yield
    _STEP_REGISTRY.clear()
    _STEP_REGISTRY.update(original)


class _CustomStep:
    """Minimal custom step for tests — wraps CastStep."""
    def __init__(self, mapping):
        self._inner = CastStep(mapping)

    def fit(self, df, target=None):
        self._inner.fit(df, target=target)
        return self

    def transform(self, df):
        return self._inner.transform(df)

    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0

    def to_dict(self):
        return {"mapping": dict(self._inner._mapping)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mapping"])


def test_list_step_types_default():
    """v0.9.1d-fix2 (§13.4/152): 4 built-in types.

    `ColumnTransformer` registered after class definition (v0.9.1d-fix1).
    Sorted order: alphabetical.
    """
    assert list_step_types() == [
        "CastStep", "ColumnTransformer", "DropnaStep", "GBDTStep",
    ]


def test_register_custom_step():
    register_step_type("MyCustom", _CustomStep)
    assert "MyCustom" in list_step_types()


def test_register_overrides_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        register_step_type("CastStep", _CustomStep)
    assert any("overwriting" in str(w.message) for w in caught)
    assert _STEP_REGISTRY["CastStep"] is _CustomStep


def test_unregister_builtin_raises():
    with pytest.raises(ValueError, match="cannot unregister built-in"):
        unregister_step_type("CastStep")


def test_unregister_custom_removes():
    register_step_type("MyCustom", _CustomStep)
    assert "MyCustom" in _STEP_REGISTRY
    unregister_step_type("MyCustom")
    assert "MyCustom" not in _STEP_REGISTRY


def test_register_bad_name_raises():
    with pytest.raises(ValueError, match="name must be non-empty"):
        register_step_type("", _CustomStep)


def test_register_bad_class_raises():
    with pytest.raises(TypeError, match="cls must be class"):
        register_step_type("X", "not_a_class")


def test_register_missing_from_dict_raises():
    class NoFromDict:
        pass
    with pytest.raises(TypeError, match="must have from_dict"):
        register_step_type("X", NoFromDict)


def test_from_json_uses_registry(tmp_path):
    """Custom step registered → from_json recognizes it."""
    from cadw import CadwDataFrame, MemoryGuard
    import pyarrow.parquet as pq

    # Minimal pipeline with custom step, fit, save, load
    t = pa.table({"x": [1, 2, 3]})
    src = tmp_path / "src.parquet"
    pq.write_table(t, src)
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )

    register_step_type("MyCustom", _CustomStep)
    pipe = Pipeline()
    pipe.add_step("custom", _CustomStep({"x": "float64"}))
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)

    restored = Pipeline.from_json(out_dir)
    assert "custom" in restored._step_names
    assert isinstance(restored._steps[0], _CustomStep)
