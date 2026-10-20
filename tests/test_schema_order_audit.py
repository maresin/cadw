"""v0.8.11b: schema-order audit — verify no latent reorder bugs.

§13.4/145 (discovery of schema-order brittleness in _local_dedup)
prompted an audit of every `Table.from_batches(gen, schema=...)` in
`cadw/ops/`. This file documents the findings and guards against
regressions.

Findings:
- join._safe_materialize_partition: explicit `schema=out_schema` where
  out_schema order matches `.select(keep_cols)`. OK.
- sort._write_run: `ArrowIPCBatchWriter(table.schema)` — table.schema
  derived from pa.Table.from_batches, no reorder through take. OK.
- sort._merge_keys: explicit `schema=schema` in RecordBatch.from_arrays.
  OK (covered by sort tests, not re-asserted here).
- groupby._read_partition_safe: `_iter_batches` yields reader batches
  as-is (no `.select(...)`). schema=_schema matches reader exactly. OK.
- dedup._local_dedup: fixed in v0.8.11a-fix3 (explicit select_schema).

No additional bugs. Four AST guards below lock the invariants.
"""
import inspect

from cadw.ops import dedup as dedup_mod
from cadw.ops import groupby as gb_mod
from cadw.ops import join as join_mod
from cadw.ops import sort as sort_mod


def test_safe_materialize_passes_explicit_schema():
    """join._safe_materialize_partition: explicit `schema=out_schema`."""
    src = inspect.getsource(join_mod._safe_materialize_partition)
    assert "schema=out_schema" in src, (
        "_safe_materialize_partition must pass schema=out_schema explicitly "
        "(no reliance on first-batch order matching generator order)"
    )


def test_read_partition_safe_no_select_on_batches():
    """groupby._read_partition_safe: no `.select()` on batches."""
    src = inspect.getsource(gb_mod._read_partition_safe)
    assert ".select(" not in src, (
        "_read_partition_safe must not call .select() on batches — "
        "it yields reader batches as-is, so schema=_schema matches"
    )


def test_write_run_uses_table_schema():
    """sort._write_run: writer schema from table.schema (stable through take)."""
    src = inspect.getsource(sort_mod._write_run)
    assert "ArrowIPCBatchWriter(str(path), table.schema)" in src, (
        "_write_run must pass table.schema to ArrowIPCBatchWriter"
    )


def test_dedup_explicit_select_schema():
    """dedup._local_dedup: explicit select_schema (v0.8.11a-fix3)."""
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "select_schema = pa.schema(" in src, (
        "_local_dedup must build explicit select_schema (v0.8.11a-fix3)"
    )
    assert "schema=select_schema" in src, (
        "_local_dedup must pass schema=select_schema"
    )
