"""v0.9.176: MemoryGuard.get_window_size tests."""
import pytest

from cadw.core.memory_guard import MemoryGuard


def test_get_window_size_basic():
    mg = MemoryGuard(max_memory_mb=100)
    w = mg.get_window_size(unit_bytes=8, target_fraction=0.25)
    assert w > 100000


def test_get_window_size_min_2():
    mg = MemoryGuard(max_memory_mb=0.001)
    w = mg.get_window_size(unit_bytes=8, target_fraction=0.25)
    assert w >= 2


def test_get_window_size_scales_with_budget():
    mg_small = MemoryGuard(max_memory_mb=10)
    mg_large = MemoryGuard(max_memory_mb=1000)
    w_small = mg_small.get_window_size(unit_bytes=8)
    w_large = mg_large.get_window_size(unit_bytes=8)
    assert w_large > w_small * 10, f"{w_large} vs {w_small}"


def test_get_window_size_scales_inverse_unit():
    mg = MemoryGuard(max_memory_mb=100)
    w_f64 = mg.get_window_size(unit_bytes=8)
    w_f32 = mg.get_window_size(unit_bytes=4)
    assert abs(w_f32 / w_f64 - 2.0) < 0.1


def test_get_window_size_fraction():
    mg = MemoryGuard(max_memory_mb=100)
    w_full = mg.get_window_size(unit_bytes=8, target_fraction=0.5)
    w_quarter = mg.get_window_size(unit_bytes=8, target_fraction=0.25)
    assert abs(w_full / w_quarter - 2.0) < 0.1


def test_get_window_size_invalid_unit():
    mg = MemoryGuard(max_memory_mb=100)
    with pytest.raises(ValueError):
        mg.get_window_size(unit_bytes=0)
    with pytest.raises(ValueError):
        mg.get_window_size(unit_bytes=-8)


def test_get_window_size_invalid_fraction():
    mg = MemoryGuard(max_memory_mb=100)
    with pytest.raises(ValueError):
        mg.get_window_size(unit_bytes=8, target_fraction=0)
    with pytest.raises(ValueError):
        mg.get_window_size(unit_bytes=8, target_fraction=1.5)
