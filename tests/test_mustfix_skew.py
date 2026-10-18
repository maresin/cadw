"""Тесты must-fix §11.4 п.3 (патч 02b2): skew isolation."""
import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.ops import groupby as gb_mod
from cadw.ops.groupby import groupby_aggregate, _detect_heavy_keys


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


def sample_batches_from(table: pa.Table, n: int = 10_000) -> list:
    out = []
    for i in range(0, min(table.num_rows, n), 1000):
        out.append(table.slice(i, min(1000, table.num_rows - i)).to_batches()[0])
    return out


def test_detect_heavy_none_uniform():
    n = 1000
    table = pa.table({"cat": [f"g{i:04d}" for i in range(n)], "val": list(range(n))})
    sample = sample_batches_from(table)
    heavy = _detect_heavy_keys(sample, ["cat"])
    assert heavy == set(), f"uniform data should have no heavy keys; got {heavy}"


def test_detect_heavy_90_10():
    n = 1000
    table = pa.table({"cat": ["A"] * 900 + ["B"] * 100, "val": list(range(n))})
    sample = sample_batches_from(table)
    heavy = _detect_heavy_keys(sample, ["cat"])
    assert ("A",) in heavy
    assert ("B",) in heavy


def test_detect_heavy_one_outlier():
    n = 1000
    keys = [f"g{i:03d}" for i in range(100)] + ["BIG"] * 900
    table = pa.table({"cat": keys, "val": list(range(n))})
    sample = sample_batches_from(table)
    heavy = _detect_heavy_keys(sample, ["cat"])
    assert ("BIG",) in heavy


def test_groupby_90_10_skew(tmp_path):
    n = 1000
    table = pa.table({
        "cat": ["A"] * 900 + ["B"] * 100,
        "val": [1.0] * 900 + [2.0] * 100,
    })
    guard = MemoryGuard(max_memory_bytes=200_000)
    result = groupby_aggregate(
        make_iter(table, 100), ["cat"], ["val", "val"], ["sum", "count"],
        guard, tmp_path, num_partitions=4, log=False,
    )
    rows = sorted(result.to_pylist(), key=lambda r: r["cat"])
    assert rows[0]["cat"] == "A" and rows[0]["val_sum"] == 900.0 and rows[0]["val_count"] == 900
    assert rows[1]["cat"] == "B" and rows[1]["val_sum"] == 200.0 and rows[1]["val_count"] == 100


def test_groupby_one_hot_skew(tmp_path):
    n = 1000
    table = pa.table({"cat": ["HOT"] * n, "val": list(range(n))})
    guard = MemoryGuard(max_memory_bytes=200_000)
    result = groupby_aggregate(
        make_iter(table, 100), ["cat"], ["val"], ["mean"],
        guard, tmp_path, num_partitions=4, log=False,
    )
    assert result.num_rows == 1
    assert result.to_pylist()[0]["val_mean"] == pytest.approx(499.5)


def test_groupby_mixed_skew_and_light(tmp_path):
    n_light = 1000
    n_heavy = 4000
    keys = [f"L{i:03d}" for i in range(200) for _ in range(5)]
    keys += ["HEAVY"] * n_heavy
    vals = [1.0] * (n_light + n_heavy)
    table = pa.table({"cat": keys, "val": vals})
    guard = MemoryGuard(max_memory_bytes=200_000)
    result = groupby_aggregate(
        make_iter(table, 200), ["cat"], ["val"], ["sum"],
        guard, tmp_path, num_partitions=8, log=False,
    )
    rows = {r["cat"]: r["val_sum"] for r in result.to_pylist()}
    assert len(rows) == 201
    assert rows["HEAVY"] == float(n_heavy)
    assert rows["L000"] == 5.0


def test_groupby_no_heavy_uses_shuffle(monkeypatch, tmp_path):
    called = []
    orig = gb_mod._groupby_with_skew
    def spy(*a, **kw):
        called.append(True)
        return orig(*a, **kw)
    monkeypatch.setattr(gb_mod, "_groupby_with_skew", spy)
    n = 200
    table = pa.table({"cat": [f"g{i:03d}" for i in range(n)], "val": list(range(n))})
    groupby_aggregate(
        make_iter(table, 50), ["cat"], ["val"], ["count"],
        MemoryGuard(max_memory_mb=10), tmp_path, num_partitions=4, log=False,
    )
    assert not called


def test_groupby_heavy_mean_correct(tmp_path):
    n = 1000
    table = pa.table({"cat": ["A"] * 950 + ["B"] * 50, "val": [10.0] * 950 + [20.0] * 50})
    guard = MemoryGuard(max_memory_bytes=100_000)
    result = groupby_aggregate(
        make_iter(table, 100), ["cat"], ["val"], ["mean"],
        guard, tmp_path, num_partitions=4, log=False,
    )
    rows = sorted(result.to_pylist(), key=lambda r: r["cat"])
    assert rows[0]["val_mean"] == 10.0
    assert rows[1]["val_mean"] == 20.0


def test_groupby_multiple_heavy_keys(tmp_path):
    keys = ["A"] * 600 + ["B"] * 300 + [f"L{i:02d}" for i in range(50) for _ in range(2)]
    table = pa.table({"cat": keys, "val": list(range(len(keys)))})
    guard = MemoryGuard(max_memory_bytes=100_000)
    result = groupby_aggregate(
        make_iter(table, 100), ["cat"], ["val"], ["count"],
        guard, tmp_path, num_partitions=4, log=False,
    )
    rows = {r["cat"]: r["val_count"] for r in result.to_pylist()}
    assert rows["A"] == 600
    assert rows["B"] == 300
    for i in range(50):
        assert rows[f"L{i:02d}"] == 2


# ---------- no-duplication (Fix D) ----------

def test_no_sample_duplication_heavy(tmp_path):
    """Все ключи heavy: count не удваивается (sample без tee)."""
    table = pa.table({"cat": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
    guard = MemoryGuard(max_memory_mb=10)
    result = groupby_aggregate(
        make_iter(table, 2), ["cat"], ["val"], ["count"],
        guard, tmp_path, num_partitions=None, log=False,
    )
    rows = {r["cat"]: r["val_count"] for r in result.to_pylist()}
    assert rows == {"A": 2, "B": 2}, f"sample duplicated: {rows}"


def test_no_sample_duplication_light(tmp_path):
    """Все ключи light: count=1, не 2."""
    n = 200
    table = pa.table({"cat": [f"k{i}" for i in range(n)], "val": [1] * n})
    guard = MemoryGuard(max_memory_mb=10)
    result = groupby_aggregate(
        make_iter(table, 20), ["cat"], ["val"], ["count"],
        guard, tmp_path, num_partitions=None, log=False,
    )
    assert result.num_rows == n
    for r in result.to_pylist():
        assert r["val_count"] == 1, f"key {r['cat']} count={r['val_count']}"
