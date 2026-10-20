"""v0.8.10-pre: whole-operation join RSS warning."""
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name, data, chunk=3, mem_mb=20):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def test_join_fixed_overhead_const_exists():
    from cadw.ops.join import _JOIN_FIXED_OVERHEAD_BYTES
    assert _JOIN_FIXED_OVERHEAD_BYTES == 40 * 1024 * 1024


def test_merge_warns_on_tight_budget(tmp_path, caplog):
    """Whole-operation RSS > state_limit → WARNING, not raise."""
    caplog.set_level(logging.WARNING, logger="cadw.dataframe")
    left = make_df(
        tmp_path, "L",
        {"k": list(range(1000)), "a": list(range(1000))},
        mem_mb=20,
    )
    right = make_df(
        tmp_path, "R",
        {"k": list(range(1000)), "b": list(range(1000))},
        mem_mb=20,
    )
    result = left.merge(right, on="k")
    assert result.shape[0] == 1000  # did NOT raise
    assert any(
        "whole-operation RSS" in rec.message for rec in caplog.records
    ), f"expected warning; got: {[r.message for r in caplog.records]}"


def test_merge_no_warn_loose_budget(tmp_path, caplog):
    """Loose budget → no warning (predicted RSS < state_limit)."""
    caplog.set_level(logging.WARNING, logger="cadw.dataframe")
    left = make_df(
        tmp_path, "L",
        {"k": list(range(1000)), "a": list(range(1000))},
        mem_mb=500,
    )
    right = make_df(
        tmp_path, "R",
        {"k": list(range(1000)), "b": list(range(1000))},
        mem_mb=500,
    )
    result = left.merge(right, on="k")
    assert result.shape[0] == 1000
    assert not any(
        "whole-operation RSS" in rec.message for rec in caplog.records
    )


def test_merge_warning_does_not_raise(tmp_path, caplog):
    """Even with very tight budget, per-batch guards pass → merge succeeds."""
    caplog.set_level(logging.WARNING, logger="cadw.dataframe")
    left = make_df(
        tmp_path, "L",
        {"k": [1, 2, 3], "a": [10, 20, 30]},
        mem_mb=20,
    )
    right = make_df(
        tmp_path, "R",
        {"k": [1, 3], "b": [100, 300]},
        mem_mb=20,
    )
    # Small join fits per-batch → completes; whole-op warning may fire.
    result = left.merge(right, on="k")
    assert result.shape[0] == 2


def test_merge_ast_uses_strict_mode():
    """AST: merge() references strict-mode threshold (v0.8.11b).

    Replaces v0.8.10-pre `test_merge_ast_guard_warning_present`
    (try/except pattern) — v0.8.11b uses explicit if/elif branch on
    `strict_mode` instead of try/except.
    """
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod.CadwDataFrame.merge)
    assert "_JOIN_STRICT_BUDGET_THRESHOLD_BYTES" in src, (
        "v0.8.11b: merge() must reference _JOIN_STRICT_BUDGET_THRESHOLD_BYTES"
    )
    assert "strict_mode" in src, (
        "v0.8.11b: merge() must have strict_mode branch"
    )


def test_merge_ast_check_state_guard_name():
    """AST: check_state uses name='join.whole_operation_rss'.

    Replaces v0.8.10-pre `test_merge_ast_does_not_raise_whole_op` —
    v0.8.11b DOES raise (in strict mode), so the guard name is the
    stable marker across both warn/strict modes.
    """
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod.CadwDataFrame.merge)
    assert 'name="join.whole_operation_rss"' in src, (
        "v0.8.11b: check_state must use guard name 'join.whole_operation_rss'"
    )
