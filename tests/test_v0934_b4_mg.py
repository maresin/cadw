"""v0.9.34 B4: memory_guard fallback factor."""
import inspect
import pyarrow as pa
from cadw.core.memory_guard import MemoryGuard, _fallback_unique_count


def test_uses_factor_constant():
    src = inspect.getsource(MemoryGuard.estimate_unique_keys)
    assert "_fallback_unique_count" in src
    assert "len(set(col.to_pylist()))" not in src


def test_factor_value():
    import cadw.core.memory_guard as m
    assert m._UNIQUE_FALLBACK_FACTOR == 4


def test_helper_returns_4x():
    arr = pa.array([f"s{i}" for i in range(100)])
    assert _fallback_unique_count(arr) == 400
