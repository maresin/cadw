# v0.9.3b: P-S1 (S2) sort merge fan-in tests.
# Prior bug: compute_fan_in read_batch_size=1000, but on-disk
# batches are _WRITE_RUN_CHUNK=50000. 50x peak underestimate.

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.sort import external_sort, _WRITE_RUN_CHUNK


def batches(n: int, bs: int, ncol: int = 10):
    rng = np.random.default_rng(0)
    for start in range(0, n, bs):
        m = min(bs, n - start)
        cols = {"k": pa.array(rng.integers(0, 1000, m),
                              type=pa.int64())}
        for j in range(ncol - 1):
            cols["v" + str(j)] = pa.array(rng.random(m))
        yield pa.record_batch(cols)


def test_write_run_chunk_constant_exists():
    assert _WRITE_RUN_CHUNK > 0
    assert isinstance(_WRITE_RUN_CHUNK, int)


def test_compute_fan_in_uses_write_run_chunk():
    # AST: compute_fan_in second positional arg must be
    # _WRITE_RUN_CHUNK (real on-disk chunk), not read_batch_size.
    import ast
    import inspect
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
        assert len(node.args) >= 2, (
            "compute_fan_in needs 2+ positional args"
        )
        arg1 = ast.unparse(node.args[1])
        assert "_WRITE_RUN_CHUNK" in arg1, (
            "compute_fan_in read_batch_size=" + arg1 +
            " must be _WRITE_RUN_CHUNK (P-S1 fix)"
        )
        return
    raise AssertionError("compute_fan_in call not found")


def test_merge_keys_has_check_batch():
    """v0.9.23c: check_batch moved into _RunStream._next_batch.

    Prior (v0.5.2a): _merge_keys directly called check_batch on each
    read batch. v0.9.23c (Arrow batched merge) delegates read + guard
    to _RunStream.refill(memory_guard) -> _RunStream._next_batch,
    which calls memory_guard.check_batch on each batch.
    """
    import inspect
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod._merge_keys)
    assert "memory_guard" in src, (
        "_merge_keys must accept memory_guard param"
    )
    # Either direct check_batch OR refill(memory_guard) delegating to
    # _RunStream._next_batch (which has check_batch).
    refill_with_guard = "refill(memory_guard)" in src
    check_batch_direct = "check_batch" in src
    assert refill_with_guard or check_batch_direct, (
        "_merge_keys must guard read batches: either direct "
        "check_batch or refill(memory_guard) (v0.9.23c: _RunStream "
        "checks internally)"
    )
    # Also verify _RunStream._next_batch has the actual check_batch call.
    stream_src = inspect.getsource(sort_mod._RunStream)
    assert "check_batch" in stream_src, (
        "_RunStream._next_batch must call check_batch on read batches"
    )
    assert "check_batch" in inspect.getsource(
        sort_mod._RunStream._next_batch
    ), "_RunStream._next_batch must call check_batch directly"


def test_external_sort_tight_budget_smoke(tmp_path):
    # v0.9.4a-triage: N=200k->100k, budget=16MB->8MB (v0.9.3
    # test time 20.5s; now ~10s). P-S1 regression still triggers:
    # k computed from read_batch_size=1000 vs on-disk 50000 ->
    # 2x too high; peak = k * 50000 * row_bytes. AST test covers
    # invariant; this smoke just verifies end-to-end completion.
    out = external_sort(
        batches(100_000, 20_000, ncol=10),
        ["k"], ["k"] + ["v" + str(j) for j in range(9)],
        MemoryGuard(max_memory_mb=8), tmp_path,
        read_batch_size=1000,
    )
    assert out.exists()


def test_external_sort_still_correct(tmp_path):
    rng = np.random.default_rng(1)
    n = 1000
    ks = rng.integers(0, 100, n)
    vs = rng.random(n)
    def iter_b():
        for i in range(0, n, 100):
            yield pa.record_batch({
                "k": pa.array(ks[i:i+100], type=pa.int64()),
                "v": pa.array(vs[i:i+100]),
            })
    out = external_sort(
        iter_b(), ["k"], ["k", "v"],
        MemoryGuard(max_memory_mb=10), tmp_path,
    )
    import pyarrow.ipc as ipc
    r = ipc.open_file(str(out))
    keys = []
    for i in range(r.num_record_batches):
        b = r.get_batch(i)
        keys.extend(b.column("k").to_pylist())
    assert keys == sorted(keys)

