"""K1.3: build-side value_counts computed once per partition-pair.

`_process_partition_pair` used to call `_exact_actual_matches_arrow`
for every probe batch, which recomputed `pc.value_counts(build_table)`
each time, though the build side is fixed for the pair. Precomputed
`build_counts` is now passed in; these tests pin equivalence.
"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from cadw.ops.join import (
    _combine_join_keys,
    _estimate_multi_key_matches,
    _exact_actual_matches_arrow,
)


def _tbl(pairs):
    return pa.table({"k": [k for k, _ in pairs], "v": [v for _, v in pairs]})


def test_exact_single_key_with_precomputed_counts_equal():
    build = _tbl([("a", 1), ("a", 2), ("b", 3)])
    probe = _tbl([("a", 9), ("c", 8), ("a", 7)])
    want = _exact_actual_matches_arrow(build, probe, ["k"])
    got = _exact_actual_matches_arrow(
        build, probe, ["k"],
        build_counts=pc.value_counts(build.column("k")),
    )
    assert got == want == 4


def test_exact_empty_sides_zero():
    empty = pa.table({"k": pa.array([], type=pa.string())})
    other = _tbl([("a", 1)])
    assert _exact_actual_matches_arrow(empty, other, ["k"]) == 0
    assert _exact_actual_matches_arrow(other, empty, ["k"]) == 0


def test_estimate_multi_key_with_precomputed_counts_equal():
    build = pa.table({"a": ["x", "x", "y"], "b": [1, 1, 2]})
    probe = pa.table({"a": ["x", "z"], "b": [1, 3]})
    want = _estimate_multi_key_matches(build, probe, ["a", "b"])
    got = _estimate_multi_key_matches(
        build, probe, ["a", "b"],
        build_counts=pc.value_counts(_combine_join_keys(build, ["a", "b"])),
    )
    assert got == want == 2


def test_build_counts_passed_once_per_pair(monkeypatch):
    """The pair loop computes build value_counts once and passes the
    same object to every probe batch of that pair (K1.3)."""
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ops import join as join_mod
    import tempfile
    from pathlib import Path as _P

    seen = []
    orig = join_mod._exact_actual_matches_arrow

    def spy(build_table, probe_table, on, *, build_counts=None):
        seen.append(build_counts)
        return orig(build_table, probe_table, on, build_counts=build_counts)

    monkeypatch.setattr(join_mod, "_exact_actual_matches_arrow", spy)

    build = pa.table({"k": ["k%d" % (i % 5) for i in range(50)]})
    probe = pa.table({"k": ["k%d" % (i % 7) for i in range(400)]})

    def _batches(t, bs):
        for i in range(0, t.num_rows, bs):
            yield t.slice(i, bs).to_batches()[0]

    mg = MemoryGuard(max_memory_bytes=512 * 1024 * 1024)
    with tempfile.TemporaryDirectory() as td:
        join_mod.hash_join(
            _batches(build, 50), _batches(probe, 8),
            build.schema, probe.schema, ["k"], mg, _P(td),
            how="inner", num_partitions=2,
        )
    assert seen, "join produced no probe batches"
    assert all(bc is not None for bc in seen), (
        "build_counts must be precomputed per pair, not None"
    )
    # Same pair → same cached object; many probe batches per pair.
    assert len({id(bc) for bc in seen}) < len(seen), (
        "build_counts must be reused across probe batches of a pair"
    )
