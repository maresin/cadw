"""v0.9.236: BufferRegistry unit tests."""
from __future__ import annotations

import pytest

from cadw.core.buffer_registry import BufferRegistry


def test_allocate_and_peaks():
    reg = BufferRegistry()
    reg.allocate("pipeline.s0", "state", 100, kind="state")
    reg.allocate("pipeline.s1", "state", 200, kind="state")
    reg.allocate("pipeline.s1", "transient", 500, kind="transient")
    assert reg.peak_state_bytes() == 300
    assert reg.peak_transient_bytes() == 500


def test_collision_raises():
    reg = BufferRegistry()
    reg.allocate("pipeline.s0", "state", 100, kind="state")
    with pytest.raises(RuntimeError, match="collision"):
        reg.allocate("pipeline.s0", "state", 200, kind="state")


def test_same_name_different_path_ok():
    reg = BufferRegistry()
    reg.allocate("pipeline.s0", "state", 100, kind="state")
    reg.allocate("pipeline.s1", "state", 100, kind="state")  # no collision


def test_reset():
    reg = BufferRegistry()
    reg.allocate("pipeline.s0", "state", 100, kind="state")
    reg.reset()
    assert reg.peak_state_bytes() == 0
