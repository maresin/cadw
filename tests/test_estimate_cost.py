"""v0.9.119-fw1: CostEstimate + MemoryGuard.estimate_cost (AP22)."""
from __future__ import annotations

import warnings

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.core.models import CostEstimate, DataChars
from cadw.core.warnings import LegacyFormulaWarning


def _dc(row_size: int = 16, n_rows: int = 10_000, k_est: int = 100) -> DataChars:
    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    return DataChars(
        schema=schema, n_rows=n_rows, row_size_bytes=row_size,
        k_est=k_est, avg_key_size_bytes=8, size_of_key_entry_bytes=80,
    )


def test_estimate_cost_reduce_returns_costestimate():
    """Basic: reduce op produces CostEstimate with fits_ram=True."""
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc()
    ce = mg.estimate_cost("reduce", dc, op_context={"state_bytes": 16})
    assert isinstance(ce, CostEstimate)
    assert ce.op_name == "reduce"
    assert ce.fits_ram is True
    assert ce.est_ram_bytes > 0


def test_estimate_cost_reduce_has_time_from_throughput():
    """reduce rate exists in throughput.yaml -> time_class != unknown."""
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc(n_rows=10_000)
    ce = mg.estimate_cost("reduce", dc, op_context={"state_bytes": 16})
    assert ce.time_class in ("fast", "medium", "slow", "very_slow")
    assert ce.est_time_seconds > 0.0
    assert ce.est_ops == 10_000


def test_estimate_cost_sort_has_rate():
    """sort rate exists in throughput.yaml."""
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc(n_rows=10_000)
    ce = mg.estimate_cost("sort", dc)
    assert ce.time_class != "unknown"
    assert ce.est_time_seconds > 0.0


def test_estimate_cost_unknown_op_no_rate_warns():
    """Op with model but no rate -> LegacyFormulaWarning, time_class=unknown."""
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc()
    # 'reduce' has rate; use a fake op that has no rate by monkeypatching
    # _load_throughput_rates to return {}.
    mg._load_throughput_rates = lambda: {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ce = mg.estimate_cost("reduce", dc, op_context={"state_bytes": 16})
    assert ce.time_class == "unknown"
    assert any(issubclass(x.category, LegacyFormulaWarning) for x in w)


def test_estimate_cost_does_not_fit():
    """Tight budget where state doesn't fit -> fits_ram=False."""
    mg = MemoryGuard(max_memory_mb=0.01)  # 10 KB
    dc = _dc()
    ce = mg.estimate_cost("reduce", dc, op_context={"state_bytes": 200_000})
    assert ce.fits_ram is False
    assert ce.time_class == "unknown"
    assert ce.reason == "does_not_fit"


def test_estimate_cost_missing_op_raises():
    """Unknown op -> KeyError (from get_model)."""
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc()
    with pytest.raises(KeyError):
        mg.estimate_cost("nonexistent_op_xyz", dc)
