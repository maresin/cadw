#!/usr/bin/env bash
# patch_02a: must-fix §11.4 (пункты 2, 4, 5).
#  - groupby_aggregate: авто-выбор num_partitions через sample + estimate_num_partitions
#  - external_sort: merge_fan_in из row_bytes_estimate
#  - MemoryGuard.check_batch(extra_factor=...): учёт overhead compute_hashes
set -euo pipefail

git rev-parse --is-inside-work-tree >/dev/null
[[ -z "$(git status --porcelain)" ]] || { echo "ERROR: tree not clean"; git status --short; exit 1; }

CUR="$(git rev-parse --abbrev-ref HEAD)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-02a-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_02a"

BR="patch/02a-mustfix-easy"
git rev-parse --verify "${BR}" >/dev/null 2>&1 && { echo "branch ${BR} exists"; exit 1; }
git checkout -b "${BR}"

# ---------- 1. MemoryGuard.check_batch: extra_factor ----------
python3 - <<'PY'
import re
p = "cadw/core/memory_guard.py"
s = open(p).read()

old = '''    def check_batch(self, batch: pa.RecordBatch, *, name: str = "batch") -> None:
        size = batch.nbytes
        limit = self.available_bytes()
        if size > limit:
            logger.error("check_batch FAIL: %s nbytes=%d > available=%d", name, size, limit)
            raise MemoryLimitExceeded(f"{name}: {size} B > available {limit} B")
        logger.debug("check_batch OK: %s nbytes=%d available=%d", name, size, limit)'''

new = '''    def check_batch(
        self,
        batch: pa.RecordBatch,
        *,
        name: str = "batch",
        extra_factor: float = 1.0,
    ) -> None:
        """Проверяет, что батч (с учётом extra_factor) влезает в available_bytes.

        extra_factor > 1.0 используется, когда операция создаёт временный
        overhead поверх batch.nbytes: например, compute_hashes делает
        to_pylist() (Python-int ~28 B на значение), что для int-колонок
        даёт x3-x5. Передавайте консервативный коэффициент (3.0-5.0),
        чтобы check_batch отражал реальный пик RAM.
        """
        if extra_factor < 1.0:
            raise ValueError("extra_factor must be >= 1.0")
        size = int(batch.nbytes * extra_factor)
        limit = self.available_bytes()
        if size > limit:
            logger.error(
                "check_batch FAIL: %s nbytes=%d extra_factor=%.2f effective=%d > available=%d",
                name, batch.nbytes, extra_factor, size, limit,
            )
            raise MemoryLimitExceeded(
                f"{name}: {size} B (nbytes={batch.nbytes} * {extra_factor}) > available {limit} B"
            )
        logger.debug(
            "check_batch OK: %s nbytes=%d extra_factor=%.2f effective=%d available=%d",
            name, batch.nbytes, extra_factor, size, limit,
        )'''

if old not in s:
    raise SystemExit("check_batch: pattern not found; manual fix required")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched memory_guard.check_batch")
PY

# ---------- 2. hash_shuffle: extra_factor=3.0 для compute_hashes ----------
python3 - <<'PY'
p = "cadw/ops/shuffle.py"
s = open(p).read()

old = '''    def _process(batch: pa.RecordBatch) -> None:
        nonlocal row_id_offset
        memory_guard.check_batch(batch, name="hash_shuffle.batch")

        hashes = compute_hashes(batch, key_columns, hash_seed)'''

new = '''    def _process(batch: pa.RecordBatch) -> None:
        nonlocal row_id_offset
        # compute_hashes делает to_pylist() → Python-int (~28 B/значение),
        # для int-колонок это x3-x5 overhead. Учитываем консервативно.
        memory_guard.check_batch(
            batch, name="hash_shuffle.batch", extra_factor=3.0
        )

        hashes = compute_hashes(batch, key_columns, hash_seed)'''

if old not in s:
    raise SystemExit("hash_shuffle._process: pattern not found")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched shuffle.hash_shuffle")
PY

# ---------- 3. external_sort: merge_fan_in из row_bytes_estimate ----------
python3 - <<'PY'
p = "cadw/ops/sort.py"
s = open(p).read()

old = '''    # --- Фаза 2: k-way merge (при необходимости — многоуровневый) ---
    k = merge_fan_in or max(2, memory_guard.available_bytes() // (read_batch_size * 128 + 1))'''

new = '''    # --- Фаза 2: k-way merge (при необходимости — многоуровневый) ---
    # k выводим из реального размера строки: каждый открытый run держит
    # буфер размером ~read_batch_size * row_bytes. Иначе (эвристика *128)
    # на широких строках можно превысить бюджет.
    row_bytes = max(1, memory_guard.row_size_bytes)
    k = merge_fan_in or max(
        2, memory_guard.available_bytes() // max(1, read_batch_size * row_bytes)
    )
    memory_guard.check_partition_buffers(k, max(1, read_batch_size * row_bytes // 8))'''

if old not in s:
    raise SystemExit("external_sort: k formula pattern not found")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched sort.external_sort")
PY

# ---------- 4. groupby_aggregate: авто num_partitions ----------
python3 - <<'PY'
p = "cadw/ops/groupby.py"
s = open(p).read()

# 4a. imports
if "import itertools" not in s:
    s = s.replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\n\nimport itertools\n",
        1,
    )
if "from cadw.ops.shuffle import hash_shuffle" in s and "estimate_num_partitions" not in s:
    s = s.replace(
        "from cadw.ops.shuffle import hash_shuffle",
        "from cadw.ops.shuffle import hash_shuffle, estimate_num_partitions",
        1,
    )

# 4b. helper _estimate_k_sample перед groupby_aggregate
helper = '''
def _estimate_k_sample(
    sample_batches: list[pa.RecordBatch],
    by_columns: Sequence[str],
) -> int:
    """Число уникальных ключей в sample. Без экстраполяции (консервативно)."""
    if not sample_batches:
        return 0
    seen: set[tuple] = set()
    cols = list(by_columns)
    for b in sample_batches:
        for i in range(b.num_rows):
            seen.add(tuple(b.column(c)[i].as_py() for c in cols))
    return len(seen)


'''
anchor = "def groupby_aggregate("
if anchor not in s:
    raise SystemExit("groupby_aggregate: def not found")
s = s.replace(anchor, helper + anchor, 1)

# 4c. заменить блок выбора num_partitions
old_block = '''    if num_partitions is None:
        num_partitions = 8  # консервативно; вызывающий может передать больше
'''
new_block = '''    if num_partitions is None:
        # Sample первых ~10k строк, оценить K, выбрать P.
        # Бюджет на партицию — половина available: вторая половина нужна
        # на батчи чтения и состояние агрегации.
        sample_cap = 10_000
        sample_iter, rest_iter = itertools.tee(batch_iter, 2)
        sample_batches: list[pa.RecordBatch] = []
        n_sample = 0
        for b in sample_iter:
            sample_batches.append(b)
            n_sample += b.num_rows
            if n_sample >= sample_cap:
                break
        batch_iter = itertools.chain(sample_batches, rest_iter)

        k_est = _estimate_k_sample(sample_batches, by_columns)
        p_est = estimate_num_partitions(
            k_est,
            available_bytes=max(1, memory_guard.available_bytes() // 2),
        )
        num_partitions = max(8, p_est)
        if log:
            logger.info(
                "groupby_aggregate: auto num_partitions=%d (k_est=%d, sample=%d rows)",
                num_partitions, k_est, n_sample,
            )
'''
if old_block not in s:
    raise SystemExit("groupby_aggregate: num_partitions block not found")
s = s.replace(old_block, new_block, 1)
open(p, "w").write(s)
print("patched groupby.groupby_aggregate")
PY

# ---------- 5. Тесты ----------
cat > tests/test_mustfix_easy.py <<'EOF'
"""Тесты must-fix §11.4 пункты 2, 4, 5 (патч 02a)."""
import numpy as np
import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops import groupby as gb_mod
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.sort import external_sort


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


# ---------- п.5: check_batch с extra_factor ----------

def test_check_batch_extra_factor():
    guard = MemoryGuard(max_memory_bytes=1000)
    batch = pa.RecordBatch.from_arrays(
        [pa.array(range(30), type=pa.int64())], names=["x"]  # ~240 B
    )
    guard.check_batch(batch)                         # 240 < 1000
    guard.check_batch(batch, extra_factor=3.0)       # 720 < 1000
    with pytest.raises(MemoryLimitExceeded):
        guard.check_batch(batch, extra_factor=5.0)   # 1200 > 1000


def test_check_batch_extra_factor_lt_1_raises():
    guard = MemoryGuard(max_memory_bytes=1000)
    batch = pa.RecordBatch.from_arrays([pa.array([1], type=pa.int64())], names=["x"])
    with pytest.raises(ValueError):
        guard.check_batch(batch, extra_factor=0.5)


def test_hash_shuffle_uses_extra_factor(monkeypatch):
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
        MemoryGuard(max_memory_mb=10), "/tmp/_p02a", log=False,
    )
    # должен быть хотя бы один вызов с name="hash_shuffle.batch" и extra_factor=3.0
    assert any(n == "hash_shuffle.batch" and f == 3.0 for n, f in calls)


# ---------- п.4: merge_fan_in из row_bytes ----------

def test_sort_fan_in_from_row_bytes(monkeypatch):
    """При одинаковом бюджете k меньше для 'широких' строк."""
    captured = {}
    from cadw.ops import sort as s_mod
    orig = s_mod._merge_keys
    def spy(run_paths, *a, **kw):
        captured.setdefault("k_seen", []).append(len(run_paths))
        return orig(run_paths, *a, **kw)
    monkeypatch.setattr(s_mod, "_merge_keys", spy)

    # 40 строк, batch=1 -> 40 run'ов; fan_in ограничивает группы.
    table = pa.table({"val": list(range(40, 0, -1))})
    # маленький бюджет -> маленький k
    guard_small = MemoryGuard(max_memory_bytes=20_000)
    external_sort(
        make_iter(table, 1), ["val"], ["val"], guard_small, "/tmp/_p02a_sort_small",
        merge_fan_in=None, read_batch_size=10, log=False,
    )
    k_small = max(captured["k_seen"])
    captured.clear()

    guard_big = MemoryGuard(max_memory_bytes=2_000_000)
    external_sort(
        make_iter(table, 1), ["val"], ["val"], guard_big, "/tmp/_p02a_sort_big",
        merge_fan_in=None, read_batch_size=10, log=False,
    )
    k_big = max(captured["k_seen"])
    assert k_big >= k_small


# ---------- п.2: авто num_partitions ----------

def test_groupby_auto_partitions(monkeypatch):
    """При большом K и малом бюджете P > 8 (дефолт)."""
    calls = []
    orig = gb_mod.estimate_num_partitions
    def spy(k_est, available_bytes, **kw):
        p = orig(k_est, available_bytes=available_bytes, **kw)
        calls.append((k_est, available_bytes, p))
        return p
    monkeypatch.setattr(gb_mod, "estimate_num_partitions", spy)

    n = 2000
    table = pa.table({
        "cat": [f"g{i:05d}" for i in range(n)],
        "val": list(range(n)),
    })
    guard = MemoryGuard(max_memory_bytes=200_000)  # 200 KB
    result = groupby_aggregate(
        make_iter(table, 200), ["cat"], ["val"], ["mean"],
        guard, "/tmp/_p02a_gb", num_partitions=None, log=False,
    )
    assert result.num_rows == n
    assert calls, "estimate_num_partitions not called"
    _, _, p_est = calls[0]
    assert p_est > 8, f"expected P>8 for K={n}, budget=200KB; got {p_est}"


def test_groupby_explicit_partitions_still_works(tmp_path):
    """Если num_partitions задан явно — sample не делается, используется как есть."""
    table = pa.table({
        "cat": ["A", "B", "A", "B"],
        "val": [1, 2, 3, 4],
    })
    result = groupby_aggregate(
        make_iter(table, 2), ["cat"], ["val"], ["sum"],
        MemoryGuard(max_memory_mb=10), tmp_path, num_partitions=2, log=False,
    )
    rows = sorted(result.to_pylist(), key=lambda r: r["cat"])
    assert rows == [
        {"cat": "A", "val_sum": 4},
        {"cat": "B", "val_sum": 6},
    ]
EOF

# ---------- 6. DECISIONS.md: пометить п.2, п.4, п.5 закрытыми ----------
python3 - <<'PY'
p = "DECISIONS.md"
s = open(p).read()

marker = "### 11.4. Известные нарушения принципа памяти (must-fix)"
if marker not in s:
    raise SystemExit("DECISIONS §11.4 marker not found")

# добавить статус после таблицы §11.4 — перед "**Статус гарантии"
insert_after = "**Статус гарантии до закрытия §11.4:"
status = (
    "**Обновление (патч 02a):** пункты **2** (авто-`num_partitions`), "
    "**4** (`merge_fan_in` из `row_bytes`), **5** (`check_batch(extra_factor)` "
    "для `compute_hashes`) — **закрыты**. Остаются пункты 1 и 3 "
    "(см. патч 02b).\n\n"
)
if "патч 02a" not in s:
    s = s.replace(insert_after, status + insert_after, 1)

open(p, "w").write(s)
print("patched DECISIONS §11.4 status")
PY

# ---------- 7. Sync + tests ----------
uv lock
uv sync
uv run pytest -q

# ---------- 8. Commit ----------
git add -A
git commit -m "02a: must-fix §11.4 п.2,4,5 — auto num_partitions, merge_fan_in from row_bytes, check_batch(extra_factor)"

echo "==> sha256 (контроль)"
sha256sum cadw/core/memory_guard.py cadw/ops/shuffle.py cadw/ops/sort.py cadw/ops/groupby.py tests/test_mustfix_easy.py || true

echo
echo "==> Patch 02a applied on branch ${BR}"
echo "    Base:      ${CUR}"
echo "    Backup:    ${BK}  (.backups/${BK}/repo.tgz)"
echo "    Revert:    git checkout ${CUR} && git branch -D ${BR}"
echo "    Hard:      git reset --hard ${BK}"