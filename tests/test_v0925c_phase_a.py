"""v0.9.25c: A.4 refill + A.5 encode_cat + A.7 fan_in rbs."""
import ast
import inspect
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cadw
from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


# ============ A.4: _RunStream.refill linear ============


def test_runstream_refill_no_concat_in_loop():
    """AST: no pa.concat_tables inside while loop in refill (B1)."""
    import textwrap
    from cadw.ops import sort as sort_mod
    src = textwrap.dedent(inspect.getsource(sort_mod._RunStream.refill))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "concat_tables"):
                    raise AssertionError(
                        f"concat_tables inside while loop at "
                        f"line {sub.lineno} — quadratic concat (B1)"
                    )


def test_runstream_refill_linear_smoke():
    """1000 tiny chunks refill < 0.5s (generous CI bound; old O(n²)
    took ~0.04-0.08s, new <0.005s; but slower CI allows 0.5s)."""
    import time
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.ops.sort import _RunStream

    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "test.arrow"
        schema = pa.schema([
            ("x", pa.int64()), ("row_id", pa.uint64()),
        ])
        w = ArrowIPCBatchWriter(str(path), schema)
        try:
            for i in range(1000):
                w.write_batch(pa.RecordBatch.from_arrays(
                    [pa.array([i], type=pa.int64()),
                     pa.array([i], type=pa.uint64())],
                    schema=schema,
                ))
        finally:
            w.close()
        stream = _RunStream(path, rbs=1000, name="test")
        t0 = time.perf_counter()
        stream.refill(None)
        elapsed = time.perf_counter() - t0
        assert len(stream.buffer) == 1000
        assert elapsed < 0.5, (
            f"refill took {elapsed:.3f}s — quadratic concat regression (B1)"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============ A.5: encode_categorical guard ============


def test_encode_categorical_high_card_raises():
    """B2: 100k unique strings @ 4 MB budget → MemoryLimitExceeded
    before OOM (was silent 18 MB Python-set on top of ~1 MB Arrow)."""
    n = 100_000
    table = pa.table({
        "cat": pa.array(
            [f"s{i:08d}" for i in range(n)], type=pa.string(),
        ),
        "v": list(range(n)),
    })
    df = CadwDataFrame.from_table(
        table, memory_guard=MemoryGuard(max_memory_mb=4),
    )
    # 100k * 180 B = 18 MB >> 2 MB state_limit (4 * 0.5).
    with pytest.raises(MemoryLimitExceeded, match="encode_categorical"):
        df.encode_categorical("cat")


def test_encode_categorical_low_card_ok():
    """Small K passes guard, behavior unchanged."""
    table = pa.table({
        "cat": pa.array(["a", "b", "a", "c", "b"], type=pa.string()),
        "v": list(range(5)),
    })
    df = CadwDataFrame.from_table(
        table, memory_guard=MemoryGuard(max_memory_mb=4),
    )
    encoded, maps = df.encode_categorical("cat")
    assert maps["cat"] == ["a", "b", "c"]


# ============ A.7: compute_fan_in uses read_batch_size ============


def test_compute_fan_in_uses_read_batch_size():
    """AST: external_sort passes read_batch_size (not _WRITE_RUN_CHUNK)."""
    import textwrap
    from cadw.ops import sort as sort_mod
    src = textwrap.dedent(inspect.getsource(sort_mod.external_sort))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name)
                and node.func.id == "compute_fan_in"):
            continue
        assert len(node.args) >= 2
        arg1 = ast.unparse(node.args[1])
        assert "read_batch_size" in arg1, (
            f"compute_fan_in arg={arg1!r} — must use read_batch_size "
            f"(v0.9.25c; Arrow-native merge internal rbs)"
        )
        assert "_WRITE_RUN_CHUNK" not in arg1, (
            f"compute_fan_in arg={arg1!r} — must NOT use _WRITE_RUN_CHUNK "
            f"(was source of 24s @ 500k/8cols/32MB)"
        )
        return
    raise AssertionError("compute_fan_in not called in external_sort")


def test_compute_fan_in_formula():
    """A.7: state_limit // (3 * rbs * row_bytes); min=2, max=256."""
    from cadw.ops.sort import compute_fan_in
    # 32 MB, rbs=1000, row_bytes=640 (8 cols narrow): 16 MB / 1.92 MB = 8
    assert compute_fan_in(32 * 1024 * 1024, 1000, 640) == 8
    # 32 MB, rbs=1000, row_bytes=80 (1 col): 16 MB / 240 KB = 69
    assert compute_fan_in(32 * 1024 * 1024, 1000, 80) == 58
    # Wide schema — 2 is minimum even when 3*K*rbs*row > state_limit
    assert compute_fan_in(32 * 1024 * 1024, 1000, 5000) == 2
    # Large budget capped at max_k=256
    assert compute_fan_in(2 * 1024 * 1024 * 1024, 1000, 80) == 256


def test_external_sort_at_32mb_8cols_reasonable_time():
    """End-to-end smoke: N=50k / 8 cols / 32 MB, wall < 5s.

    Audit §6.6 measured 24.4s @ N=500k/8cols/32MB pre-fix. Post-fix
    K=8 (not 2), fewer merge passes. Smoke test uses N=50k to stay
    fast; full 500k bench in _measure/sort_scaling.py.
    """
    import time
    rng = np.random.default_rng(0)
    n = 50_000
    cols = {"k": pa.array(rng.integers(0, 1000, n), type=pa.int64())}
    for j in range(7):
        cols[f"v{j}"] = pa.array(rng.normal(size=n), type=pa.float64())
    table = pa.table(cols)
    src = Path(tempfile.mkdtemp()) / "src.parquet"
    try:
        pq.write_table(table, src, row_group_size=5000)
        df = CadwDataFrame.from_parquet(
            src, src.parent / "ds", chunk_size=5000,
            memory_guard=MemoryGuard(max_memory_mb=32),
            spill_dir=src.parent / "spill",
        )
        t0 = time.perf_counter()
        out = df.sort_values("k")
        out_path = src.parent / "out.parquet"
        out.write_parquet(out_path)
        elapsed = time.perf_counter() - t0
        got_k = pq.read_table(out_path).column("k").to_pylist()
        assert got_k == sorted(table.column("k").to_pylist())
        assert elapsed < 5.0, (
            f"sort @ 50k/8cols/32MB took {elapsed:.2f}s — fan_in "
            f"regression (was ~24s @ 500k pre-fix; post-fix should be "
            f"scaled to ~2.5s @ 500k → ~0.25s @ 50k)"
        )
    finally:
        shutil.rmtree(src.parent, ignore_errors=True)


# ============ v0.9.25c-fix5 (AST, rewritten v0.9.25c-fix7) ============


def test_no_stale_old_compute_fan_in_formula():
    """AST: no real test asserts pre-v0.9.25c compute_fan_in output.

    v0.9.25c-fix7: prior fix5 was SUBSTRING check ("compute_fan_in" in
    line and "== 1310" in line) — 4th substring-guard bug in project
    (§13.4/161+183). It false-positived on its own condition and on
    docstrings "(was 1310)". AST-based: only ast.Compare nodes where
    left=Call(Name('compute_fan_in')) and comparator=Constant(1310|131).
    Comments/docstrings invisible.
    """
    import ast as _ast
    from pathlib import Path as _P
    stale_values = {1310, 131}
    violations: list[str] = []
    for py in _P("tests").glob("test_*.py"):
        try:
            tree = _ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Compare):
                continue
            left = node.left
            if not (isinstance(left, _ast.Call)
                    and isinstance(left.func, _ast.Name)
                    and left.func.id == "compute_fan_in"):
                continue
            for comp in node.comparators:
                if (isinstance(comp, _ast.Constant)
                        and isinstance(comp.value, int)
                        and comp.value in stale_values):
                    violations.append(
                        f"{py.name}:{node.lineno}"
                    )
    assert not violations, (
        "Real stale compute_fan_in assertions. New values: 218 for "
        "(1_048_576,100,8); 21 for (1_048_576,100,80). "
        f"Violations: {violations}"
    )
