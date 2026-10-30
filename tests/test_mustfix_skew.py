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
    assert result.count() == 1  # v0.9.95: CadwDataFrame
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
    assert result.count() == n  # v0.9.95: CadwDataFrame
    for r in result.to_pylist():
        assert r["val_count"] == 1, f"key {r['cat']} count={r['val_count']}"


# ============ v0.9.12 (F12-4): _detect_heavy_keys Arrow-native ============


def test_detect_heavy_keys_arrow_native_correctness():
    """v0.9.12 (F12-4): Arrow-native output == Python reference.

    900 'A' + 100 'B' (total=1000): threshold_abs = max(100,
    int(1000*0.05)+1) = max(100, 51) = 100. Both pass.
    """
    import pyarrow as pa
    from cadw.ops.groupby import _detect_heavy_keys
    table = pa.table({'k': ['A'] * 900 + ['B'] * 100})
    batches = [
        table.slice(0, 500).to_batches()[0],
        table.slice(500).to_batches()[0],
    ]
    result = _detect_heavy_keys(batches, ['k'])
    assert result == {('A',), ('B',)}


def test_detect_heavy_keys_no_heavy_uniform():
    """100 unique keys, 1 row each — none meet threshold_abs=100."""
    import pyarrow as pa
    from cadw.ops.groupby import _detect_heavy_keys
    table = pa.table({'k': [f'k{i}' for i in range(100)]})
    assert _detect_heavy_keys([table.to_batches()[0]], ['k']) == set()


def test_detect_heavy_keys_null_group_excluded():
    """Nulls are NOT isolated as heavy keys (v0.9.12 F12-4 contract).

    Rationale (§13.4/218): null-key groups can't be isolated via
    pc.equal(col, scalar(None)) — null==anything returns null (Kleene
    logic), which filter treats as false → null-heavy mask drops
    everything. So nulls ALWAYS go through light path (shuffled by
    _NULL_SENTINEL, grouped by pyarrow) — see _compute_heavy_mask
    fill_null fix. Asserting (None,) NOT in heavy enforces this.

    Case: 100 nulls + 100 'A' (total=200). threshold_abs=max(100,
    int(200*0.05)+1)=100. 'A': 100>=100 & 0.5>0.05 → heavy. Nulls:
    same count but excluded by pc.greater_equal returning null.
    """
    import pyarrow as pa
    from cadw.ops.groupby import _detect_heavy_keys
    table = pa.table({'k': pa.array([None] * 100 + ['A'] * 100)})
    result = _detect_heavy_keys([table.to_batches()[0]], ['k'])
    assert ('A',) in result
    assert (None,) not in result, (
        "nulls must NOT be in heavy_keys (F12-4 contract): they go "
        "through light path via _compute_heavy_mask fill_null"
    )


def test_detect_heavy_keys_below_absolute_min():
    """Total=50, 'A' appears 30 times (60% ratio) — fails
    _HEAVY_ABSOLUTE_MIN=100 → not heavy (tiny sample protection)."""
    import pyarrow as pa
    from cadw.ops.groupby import _detect_heavy_keys
    table = pa.table({'k': ['A'] * 30 + ['B'] * 20})
    assert _detect_heavy_keys([table.to_batches()[0]], ['k']) == set()

# ============ v0.9.12-fix2 (§13.4/218): null-key preservation ============


def test_compute_heavy_mask_no_null_values():
    """§13.4/218: _compute_heavy_mask returns bool with NO nulls,
    even when input has null keys (was silent data loss)."""
    import pyarrow as pa
    from cadw.ops.groupby import _compute_heavy_mask
    table = pa.table({'k': pa.array([None, 'A', 'B', None, 'A']),
                      'v': [1.0, 2.0, 3.0, 4.0, 5.0]})
    batch = table.to_batches()[0]
    mask = _compute_heavy_mask(batch, ['k'], {('A',)})
    assert mask.type == pa.bool_()
    assert mask.null_count == 0, (
        f'mask has nulls: {mask.null_count} — §13.4/218 regression'
    )
    assert mask.to_pylist() == [False, True, False, False, True]


def test_groupby_skew_preserves_null_keys(tmp_path):
    """§13.4/218 E2E: null-key rows survive skew path.

    Before v0.9.12-fix2: 500 null keys + 400 'A' + 100 'B' with
    skew triggered → null rows dropped from both heavy_batch and
    light_batch (filter treats null mask as false). Result had
    only {A: 400, B: 100}, nulls lost.

    After: nulls → light path (invert(fill_null(False))=True),
    shuffled by _NULL_SENTINEL, grouped → {None: 500, A: 400,
    B: 100}.
    """
    import pyarrow as pa
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ops.groupby import groupby_aggregate
    n = 1000
    table = pa.table({
        'k': pa.array([None] * 500 + ['A'] * 400 + ['B'] * 100),
        'v': [1.0] * n,
    })
    def it():
        for i in range(0, n, 100):
            yield table.slice(i, 100).to_batches()[0]
    guard = MemoryGuard(max_memory_bytes=200_000)
    result = groupby_aggregate(
        it(), ['k'], ['v'], ['count'],
        guard, tmp_path, num_partitions=4, log=False,
    )
    rows = {r['k']: r['v_count'] for r in result.to_pylist()}
    assert rows.get(None) == 500, (
        f'null-key rows dropped (was silent data loss pre-fix2): '
        f'got {rows}, expected None=500'
    )
    assert rows.get('A') == 400
    assert rows.get('B') == 100
    assert sum(rows.values()) == n