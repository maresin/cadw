#!/usr/bin/env bash
# patch_02a-fix: исправляет 2 упавших теста из patch_02a.
# Модульные правки 02a (extra_factor, inline fan-in, auto-P) не трогает.
set -euo pipefail

git rev-parse --abbrev-ref HEAD | grep -q "^patch/02a" \
  || { echo "ERROR: not on patch/02a branch"; exit 1; }

# ---------- 1. compute_fan_in helper в sort.py ----------
python3 - <<'PY'
p = "cadw/ops/sort.py"
s = open(p).read()

helper = '''
def compute_fan_in(available_bytes: int, read_batch_size: int, row_bytes: int) -> int:
    """k (fan-in) для k-way merge: сколько run'ов держать открытыми одновременно.

    Ограничение: k * read_batch_size * row_bytes <= available_bytes.
    Возвращает >= 2 (иначе merge невозможен).
    """
    denom = max(1, read_batch_size * max(1, row_bytes))
    return max(2, available_bytes // denom)


'''
if "def compute_fan_in(" not in s:
    s = s.replace("def external_sort(", helper + "def external_sort(", 1)

old = '''    # --- Фаза 2: k-way merge (при необходимости — многоуровневый) ---
    # k выводим из реального размера строки: каждый открытый run держит
    # буфер размером ~read_batch_size * row_bytes. Иначе (эвристика *128)
    # на широких строках можно превысить бюджет.
    row_bytes = max(1, memory_guard.row_size_bytes)
    k = merge_fan_in or max(
        2, memory_guard.available_bytes() // max(1, read_batch_size * row_bytes)
    )
    memory_guard.check_partition_buffers(k, max(1, read_batch_size * row_bytes // 8))'''

new = '''    # --- Фаза 2: k-way merge (при необходимости — многоуровневый) ---
    row_bytes = max(1, memory_guard.row_size_bytes)
    k = merge_fan_in or compute_fan_in(
        memory_guard.available_bytes(), read_batch_size, row_bytes
    )'''

if old not in s:
    raise SystemExit("external_sort: inline formula not found (patch_02a applied?)")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched sort.py: compute_fan_in")
PY

# ---------- 2. export в ops/__init__.py ----------
python3 - <<'PY'
p = "cadw/ops/__init__.py"
s = open(p).read()
if "compute_fan_in" not in s:
    s = s.replace(
        "from .sort import external_sort, read_sorted",
        "from .sort import external_sort, read_sorted, compute_fan_in",
        1,
    )
    s = s.replace(
        '"external_sort", "read_sorted",',
        '"external_sort", "read_sorted", "compute_fan_in",',
        1,
    )
    open(p, "w").write(s)
    print("patched ops/__init__.py")
PY

# ---------- 3. переписать tests/test_mustfix_easy.py ----------
cat > tests/test_mustfix_easy.py <<'EOF'
"""Тесты must-fix §11.4 пункты 2, 4, 5 (патч 02a + 02a-fix)."""
import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops import groupby as gb_mod
from cadw.ops import sort as sort_mod
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.shuffle import estimate_num_partitions
from cadw.ops.sort import compute_fan_in, external_sort


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


# =================== п.5: check_batch(extra_factor) ===================

def test_check_batch_extra_factor_ok_and_raises():
    guard = MemoryGuard(max_memory_bytes=1000)
    batch = pa.RecordBatch.from_arrays(
        [pa.array(range(30), type=pa.int64())], names=["x"]  # ~240 B
    )
    guard.check_batch(batch)                          # 240 < 1000
    guard.check_batch(batch, extra_factor=3.0)        # 720 < 1000
    with pytest.raises(MemoryLimitExceeded):
        guard.check_batch(batch, extra_factor=5.0)    # 1200 > 1000


def test_check_batch_extra_factor_lt_1_raises():
    guard = MemoryGuard(max_memory_bytes=1000)
    batch = pa.RecordBatch.from_arrays([pa.array([1], type=pa.int64())], names=["x"])
    with pytest.raises(ValueError):
        guard.check_batch(batch, extra_factor=0.5)


def test_hash_shuffle_uses_extra_factor(monkeypatch, tmp_path):
    """hash_shuffle вызывает check_batch с extra_factor=3.0."""
    from cadw.ops import shuffle as sh
    calls = []
    orig = MemoryGuard.check_batch
    def spy(self, batch, *, name="batch", extra_factor=1.0):
        calls.append((name, extra_factor))
        return orig(self, batch, name=name, extra_factor=extra_factor)
    monkeypatch.setattr(MemoryGuard, "check_batch", spy)

    table = pa.table({"cat": ["A", "B"], "val": [1, 2]})
    sh.hash_shuffle(
        make_iter(table, 1), ["cat"], ["val"], 2,
        MemoryGuard(max_memory_mb=10), tmp_path, log=False,
    )
    assert any(n == "hash_shuffle.batch" and f == 3.0 for n, f in calls)


# =================== п.4: compute_fan_in ===================

def test_compute_fan_in_formula():
    # 1_048_576 / (100*8) = 1310
    assert compute_fan_in(1_048_576, 100, 8) == 1310
    # 1_048_576 / (100*80) = 131
    assert compute_fan_in(1_048_576, 100, 80) == 131
    assert compute_fan_in(1_048_576, 100, 8) > compute_fan_in(1_048_576, 100, 80)


def test_compute_fan_in_minimum_2():
    assert compute_fan_in(100, 1000, 1000) == 2
    assert compute_fan_in(0, 1, 1) == 2


def test_compute_fan_in_uses_row_size():
    k_small_rows = compute_fan_in(1_000_000, 1000, 8)
    k_big_rows = compute_fan_in(1_000_000, 1000, 800)
    assert k_small_rows > k_big_rows


def test_external_sort_uses_compute_fan_in(monkeypatch, tmp_path):
    """external_sort вызывает compute_fan_in (а не hardcoded формулу)."""
    called = []
    orig = sort_mod.compute_fan_in
    def spy(avail, rbs, rb):
        r = orig(avail, rbs, rb)
        called.append((avail, rbs, rb, r))
        return r
    monkeypatch.setattr(sort_mod, "compute_fan_in", spy)

    table = pa.table({"val": list(range(20))})
    external_sort(
        make_iter(table, 1), ["val"], ["val"],
        MemoryGuard(max_memory_mb=1), tmp_path,
        merge_fan_in=None, log=False,
    )
    assert called, "compute_fan_in not called from external_sort"


# =================== п.2: auto num_partitions ===================

def test_groupby_auto_partitions_calls_estimate(monkeypatch, tmp_path):
    """При num_partitions=None вызывается estimate_num_partitions."""
    calls = []
    orig = gb_mod.estimate_num_partitions
    def spy(k_est, available_bytes, **kw):
        p = orig(k_est, available_bytes=available_bytes, **kw)
        calls.append((k_est, available_bytes, p))
        return p
    monkeypatch.setattr(gb_mod, "estimate_num_partitions", spy)

    table = pa.table({"cat": ["A", "B", "A", "B"], "val": [1, 2, 3, 4]})
    groupby_aggregate(
        make_iter(table, 2), ["cat"], ["val"], ["sum"],
        MemoryGuard(max_memory_mb=10), tmp_path, num_partitions=None, log=False,
    )
    assert calls, "estimate_num_partitions not called"
    k_est, _, _ = calls[0]
    assert k_est >= 1


def test_groupby_explicit_partitions_no_estimate(monkeypatch, tmp_path):
    """Если num_partitions задан явно — estimate_num_partitions не вызывается."""
    called = []
    orig = gb_mod.estimate_num_partitions
    def spy(*a, **kw):
        called.append(True)
        return orig(*a, **kw)
    monkeypatch.setattr(gb_mod, "estimate_num_partitions", spy)

    table = pa.table({"cat": ["A", "B"], "val": [1, 2]})
    groupby_aggregate(
        make_iter(table, 1), ["cat"], ["val"], ["sum"],
        MemoryGuard(max_memory_mb=10), tmp_path, num_partitions=2, log=False,
    )
    assert not called, "estimate should not be called when num_partitions is explicit"


def test_estimate_num_partitions_formula():
    """Формула: P = ceil(K * safety / groups_per_partition)."""
    # K=10_000, available=100_000 → groups_per_partition=100_000*0.6/80=750
    # raw = 10_000 * 1.5 / 750 = 20
    p = estimate_num_partitions(10_000, available_bytes=100_000)
    assert p == 20


def test_estimate_num_partitions_monotonic_in_budget():
    p_small = estimate_num_partitions(10_000, available_bytes=100_000)
    p_big = estimate_num_partitions(10_000, available_bytes=10_000_000)
    assert p_small > p_big  # меньше бюджет → больше партиций


def test_groupby_small_k_still_correct(tmp_path):
    """Маленькое K: авто-P не ломает результат (floor 8 не мешает)."""
    table = pa.table({
        "cat": ["A", "B", "A", "B", "A", "B"],
        "val": [1, 2, 3, 4, 5, 6],
    })
    result = groupby_aggregate(
        make_iter(table, 2), ["cat"], ["val"], ["mean"],
        MemoryGuard(max_memory_mb=10), tmp_path, num_partitions=None, log=False,
    )
    rows = sorted(result.to_pylist(), key=lambda r: r["cat"])
    assert rows == [
        {"cat": "A", "val_mean": 3.0},
        {"cat": "B", "val_mean": 4.0},
    ]
EOF

# ---------- 4. tests ----------
uv run pytest -q

# ---------- 5. commit ----------
git add -A
git commit -m "02a-fix: compute_fan_in helper, honest tests (formula-based, calls-estimate-based)"

echo "==> sha256"
sha256sum cadw/ops/sort.py cadw/ops/__init__.py tests/test_mustfix_easy.py
echo
echo "==> 02a-fix applied on $(git rev-parse --abbrev-ref HEAD)"
echo "    Revert (all 02a + fix): git reset --hard 8f27afc"
echo "    Revert (only fix):      git checkout HEAD -- cadw/ops/sort.py cadw/ops/__init__.py tests/test_mustfix_easy.py"