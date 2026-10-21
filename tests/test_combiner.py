"""
Combiner для groupby (v0.3.4): partial aggregation до shuffle.

Тесты используют n=1000, k=50 (20 rows/group = 2%) — БЕЗ skew-trigger,
чтобы combiner путь реально исполнялся (не fallback). Skew-detection
требует count >= _HEAVY_ABSOLUTE_MIN=100 AND ratio > 0.05 (§13.4/52b).
"""
from pathlib import Path

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.groupby import groupby_aggregate


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


def _run(tmp_path, table, by, agg_cols, funcs, *, combiner,
         num_partitions=4, mem_mb=20, batch_size=10):
    guard = MemoryGuard(max_memory_mb=mem_mb)
    return groupby_aggregate(
        make_iter(table, batch_size), by, agg_cols, funcs,
        guard, tmp_path, num_partitions=num_partitions,
        combiner=combiner, log=False,
    )


# helper: non-skew data
def _flat(n=1000, k=50):
    """n rows, k uniform groups (n/k rows each, no key is heavy)."""
    return pa.table({
        "k": [f"g{i % k:03d}" for i in range(n)],
        "v": list(range(n)),
    })


# ---------- equivalence: combiner vs no-combiner ----------

def test_combiner_count_equivalence(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["count"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["count"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_sum_equivalence(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["sum"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["sum"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_min_equivalence(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["min"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["min"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_max_equivalence(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["max"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["max"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_mean_equivalence(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["mean"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["mean"], combiner=True)
    rows0 = r0.sort_by("k").to_pylist()
    rows1 = r1.sort_by("k").to_pylist()
    # Float comparison for mean.
    assert len(rows0) == len(rows1)
    for a, b in zip(rows0, rows1):
        assert a["k"] == b["k"]
        assert a["v_mean"] == pytest.approx(b["v_mean"], rel=1e-10)


# ---------- mixed aggs ----------

def test_combiner_mixed_aggs(tmp_path):
    table = pa.table({
        "k": [f"g{i % 50:03d}" for i in range(1000)],
        "v": list(range(1000)),
        "w": [i * 1.5 for i in range(1000)],
    })
    r0 = _run(tmp_path / "off", table, ["k"], ["v", "w"], ["count", "mean"],
              combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v", "w"], ["count", "mean"],
              combiner=True)
    rows0 = r0.sort_by("k").to_pylist()
    rows1 = r1.sort_by("k").to_pylist()
    assert len(rows0) == len(rows1)
    for a, b in zip(rows0, rows1):
        assert a["k"] == b["k"]
        assert a["v_count"] == b["v_count"]
        assert a["w_mean"] == pytest.approx(b["w_mean"], rel=1e-10)


def test_combiner_same_col_sum_and_count(tmp_path):
    table = _flat()
    r0 = _run(tmp_path / "off", table, ["k"], ["v", "v"], ["sum", "count"],
              combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v", "v"], ["sum", "count"],
              combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


# ---------- std rejected (not in _AGG_FUNCS) ----------

def test_combiner_std_rejected_by_groupby(tmp_path):
    """std не входит в _AGG_FUNCS — ValueError до combiner-логики."""
    table = _flat(n=10, k=3)
    with pytest.raises(ValueError, match="std"):
        _run(tmp_path, table, ["k"], ["v"], ["std"], combiner=True)


# ---------- skew: warning + fallback ----------

def test_combiner_skew_warns_and_falls_back(tmp_path):
    """90/10 на n=1000 → оба ключа heavy → combiner off, warning."""
    import warnings
    table = pa.table({
        "k": ["A"] * 900 + ["B"] * 100,
        "v": [1.0] * 1000,
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run(
            tmp_path, table, ["k"], ["v"], ["count"],
            combiner=True, num_partitions=4,
        )
    assert any("combiner" in str(w.message) and "skew" in str(w.message)
               for w in caught), (
        f"expected combiner-skew warning; got {[str(w.message) for w in caught]}"
    )
    rows = {r["k"]: r["v_count"] for r in result.to_pylist()}
    assert rows == {"A": 900, "B": 100}


# ---------- large N, low K: combiner actually active ----------

def test_combiner_path_actually_used(tmp_path, monkeypatch):
    """Spy on _combine_batches — proves combiner path executes (not fallback)."""
    from cadw.ops import groupby as gb_mod
    calls: list[bool] = []
    orig = gb_mod._combine_batches

    def spy(*args, **kwargs):
        calls.append(True)
        return orig(*args, **kwargs)

    monkeypatch.setattr(gb_mod, "_combine_batches", spy)

    table = _flat(n=1000, k=50)
    result = _run(tmp_path, table, ["k"], ["v"], ["count"], combiner=True)
    assert calls, (
        "combiner path NOT used — data still triggers skew detection, "
        "or combiner_active is False. Check _HEAVY_ABSOLUTE_MIN + ratio."
    )
    for r in result.to_pylist():
        assert r["v_count"] == 20  # 1000 / 50


def test_combiner_large_n_low_k(tmp_path):
    """1000 rows, 50 groups → each count = 20. Combiner path is non-skew."""
    table = _flat(n=1000, k=50)
    r_off = _run(tmp_path / "off", table, ["k"], ["v"], ["count"], combiner=False)
    r_on = _run(tmp_path / "on", table, ["k"], ["v"], ["count"], combiner=True)
    assert r_off.sort_by("k").to_pylist() == r_on.sort_by("k").to_pylist()
    for r in r_on.to_pylist():
        assert r["v_count"] == 20


# ---------- multi-key combiner ----------

def test_combiner_multi_by(tmp_path):
    n = 1000
    table = pa.table({
        "k1": [f"a{i % 10}" for i in range(n)],
        "k2": [f"b{(i // 100) % 10}" for i in range(n)],
        "v": list(range(n)),
    })
    r0 = _run(tmp_path / "off", table, ["k1", "k2"], ["v"], ["sum"],
              combiner=False)
    r1 = _run(tmp_path / "on", table, ["k1", "k2"], ["v"], ["sum"],
              combiner=True)
    rows0 = sorted(r0.to_pylist(), key=lambda r: (r["k1"], r["k2"]))
    rows1 = sorted(r1.to_pylist(), key=lambda r: (r["k1"], r["k2"]))
    assert rows0 == rows1


# ---------- empty input ----------

def test_combiner_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, pa.table({"k": [], "v": []}), ["k"], ["v"], ["sum"],
             combiner=True)


# ---------- API: через CadwDataFrame ----------

def test_dataframe_agg_with_combiner(tmp_path):
    """df.groupby('k').agg('v', 'sum', combiner=True) — non-skew data."""
    from cadw import CadwDataFrame, MemoryGuard
    import pyarrow.parquet as pq

    t = pa.table({
        "k": [f"g{i % 50:03d}" for i in range(1000)],
        "v": list(range(1000)),
    })
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=100,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )
    result = df.groupby("k").agg("v", "sum", combiner=True)
    out = tmp_path / "out.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 50
    # sum of v per group g{i}: values i, i+50, i+100, ... (20 items)
    # = 20*i + 50*(0+1+...+19) = 20*i + 50*190 = 20*i + 9500
    for r in rows:
        i = int(r["k"][1:])
        assert r["v_sum"] == pytest.approx(20 * i + 9500, rel=1e-10)


def test_combiner_check_state_called(tmp_path, monkeypatch):
    # v0.9.9 (N3): _combine_batches / _merge_partials check_state.
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ops import groupby as gb_mod
    import pyarrow as pa
    guard = MemoryGuard(max_memory_mb=10)
    calls = []
    orig_check_state = MemoryGuard.check_state
    def spy(self, nbytes, *, name='state'):
        calls.append(name)
        return orig_check_state(self, nbytes, name=name)
    monkeypatch.setattr(MemoryGuard, 'check_state', spy)
    n = 1000
    table = pa.table({
        'k': [f'g{i % 50:03d}' for i in range(n)],
        'v': list(range(n)),
    })
    def it():
        for i in range(0, n, 100):
            yield table.slice(i, 100).to_batches()[0]
    gb_mod.groupby_aggregate(
        it(), ['k'], ['v'], ['sum'], guard, tmp_path,
        num_partitions=4, combiner=True, log=False,
    )
    assert any('combiner' in c for c in calls), (
        f"expected combiner check_state; got {calls}"
    )