# v0.9.3e: C1 dead code cleanup tests.
# Prior _local_groupby had two identical-name _read_for_resplit
# defs (first shadowed by second). Dead code removed via AST;
# current behavior (re-read from disk) preserved.

from pathlib import Path

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.groupby import _local_groupby


def write_partition(path, table, chunk=100):
    w = ArrowIPCBatchWriter(str(path), table.schema)
    try:
        for b in table.to_batches(max_chunksize=chunk):
            w.write_batch(b)
    finally:
        w.close()
    return path


def test_exactly_one_read_for_resplit_def():
    # AST: _local_groupby has EXACTLY ONE _read_for_resplit def.
    import ast
    import textwrap
    import importlib
    import inspect
    gb_mod = importlib.import_module("cadw.ops.groupby")
    src = inspect.getsource(gb_mod._local_groupby)
    tree = ast.parse(textwrap.dedent(src))
    count = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_read_for_resplit"):
            count += 1
    assert count == 1, (
        "expected exactly 1 _read_for_resplit def in _local_groupby, "
        "got " + str(count)
    )


def test_no_shadowed_def_ast():
    # AST: no two FunctionDef nodes with same name in same scope.
    import ast
    import textwrap
    import importlib
    import inspect
    gb_mod = importlib.import_module("cadw.ops.groupby")
    src = inspect.getsource(gb_mod._local_groupby)
    tree = ast.parse(textwrap.dedent(src))
    # Collect names of immediate body FunctionDefs (top-level of method
    # body). Nested defs inside other defs are OK (function-scope).
    method_node = tree.body[0]
    seen = {}
    for node in method_node.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in seen:
                raise AssertionError(
                    "duplicate same-scope def: " + node.name +
                    " (lines " + str(seen[node.name]) + " and " +
                    str(node.lineno) + ")"
                )
            seen[node.name] = node.lineno


def test_comment_present_ast():
    # AST: comment marker for v0.9.3e present near old def location.
    import importlib
    import inspect
    gb_mod = importlib.import_module("cadw.ops.groupby")
    src = inspect.getsource(gb_mod._local_groupby)
    assert "v0.9.3e" in src, (
        "expected v0.9.3e comment marker in _local_groupby"
    )


def test_dict_overflow_recursion_still_works(tmp_path):
    # Behavioral: dict-overflow recursion path still functional.
    # N=1000, K=1000 unique int32 keys. row_size=8. File ~10 KB.
    # Budget 100 KB -> safety 50 KB. File fits, dict overflow (80 KB
    # > 50 KB) -> hash_shuffle split.
    n = 1000
    table = pa.table({
        "cat": pa.array(list(range(n)), type=pa.int32()),
        "val": pa.array(list(range(n)), type=pa.int32()),
    })
    p = write_partition(tmp_path / "p.arrow", table)
    guard = MemoryGuard(max_memory_bytes=100_000)
    result = _local_groupby(
        p, ["cat"], ["val"], ["count"],
        memory_guard=guard,
        spill_dir=tmp_path,
        size_of_key_entry_bytes=80,
    )
    assert result.num_rows == n

