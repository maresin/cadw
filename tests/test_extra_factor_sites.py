"""v0.9.22b (§13.4/247): extra_factor at to_pylist call sites."""
import inspect

import pytest


SITES = [
    ("cadw.ops.rank", "_compute_streamable",
     "rank.streamable.input", 5.0),
    ("cadw.ops.rank", "_pass1_groups_to_disk",
     "rank.pass1.input", 3.0),
    ("cadw.ops.rank", "_pass2_ranks_to_disk",
     "rank.pass2.input", 5.0),
    ("cadw.ops.resample", "resample_to_disk",
     "resample.sorted_input", 5.0),
    ("cadw.ops.pivot", "pivot_to_disk",
     "pivot.stream_sorted", 5.0),
]


@pytest.mark.parametrize("module,fn_name,name,factor", SITES)
def test_check_batch_has_extra_factor(module, fn_name, name, factor):
    """v0.9.22b (§13.4/247): each to_pylist site has explicit
    extra_factor in the check_batch line (same line as name=)."""
    mod = __import__(module, fromlist=[fn_name])
    fn = getattr(mod, fn_name)
    src = inspect.getsource(fn)
    for line in src.split("\n"):
        if name in line:
            assert "extra_factor=" in line, (
                f"{module}.{fn_name}: check_batch(name={name!r}) "
                f"missing extra_factor in line: {line!r}"
            )
            assert f"extra_factor={factor}" in line, (
                f"{module}.{fn_name}: expected "
                f"extra_factor={factor} in line: {line!r}"
            )
            return
    raise AssertionError(
        f"{module}.{fn_name}: check_batch(name={name!r}) not found"
    )
