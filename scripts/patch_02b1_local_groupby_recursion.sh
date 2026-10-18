#!/usr/bin/env bash
# patch_02b1: must-fix §11.4 п.1 — _local_groupby: check_batch + рекурсия.
set -euo pipefail

git rev-parse --is-inside-work-tree >/dev/null
[[ -z "$(git status --porcelain)" ]] || { echo "tree not clean"; git status --short; exit 1; }

CUR="$(git rev-parse --abbrev-ref HEAD)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-02b1-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_02b1"

BR="patch/02b1-local-groupby-recursion"
git rev-parse --verify "${BR}" >/dev/null 2>&1 && { echo "branch exists"; exit 1; }
git checkout -b "${BR}"

# ---------- 1. groupby.py ----------
python3 - <<'PY'
p = "cadw/ops/groupby.py"
s = open(p).read()

if "from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded" not in s:
    s = s.replace(
        "from cadw.core.memory_guard import MemoryGuard",
        "from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded",
        1,
    )

# 1a. helpers + новая сигнатура _local_groupby
old_sig = '''def _local_groupby(
    partition_path: Path,
    by_columns: Sequence[str],
    agg_columns: Sequence[str],
    agg_funcs: Sequence[str] | str,
) -> pa.Table:'''
new_sig = '''_MAX_DEPTH = 5


def _empty_result_schema(by_columns, agg_columns, agg_funcs) -> pa.Schema:
    """Схема пустого результата агрегации (для пустых партиций)."""
    fields = [pa.field(c, pa.string()) for c in by_columns]
    for col, f in zip(agg_columns, agg_funcs):
        out_type = pa.int64() if f == "count" else pa.float64()
        fields.append(pa.field(f"{col}_{f}", out_type))
    return pa.schema(fields)


def _local_groupby(
    partition_path: Path,
    by_columns: Sequence[str],
    agg_columns: Sequence[str],
    agg_funcs: Sequence[str] | str,
    memory_guard: MemoryGuard | None = None,
    spill_dir: Path | None = None,
    depth: int = 0,
    max_depth: int = _MAX_DEPTH,
) -> pa.Table:'''
if old_sig not in s:
    raise SystemExit("_local_groupby sig: not found")
s = s.replace(old_sig, new_sig, 1)

# 1b. тело функции целиком
old_body = '''    if isinstance(agg_funcs, str):
        agg_funcs = [agg_funcs] * len(agg_columns)
    if len(agg_funcs) != len(agg_columns):
        raise ValueError("agg_funcs и agg_columns должны быть одной длины")
    for f in agg_funcs:
        if f not in _AGG_FUNCS:
            raise ValueError(f"Неподдерживаемая агрегация: {f!r}. Доступно: {_AGG_FUNCS}")

    batches = list(ArrowIPCBatchReader(str(partition_path), None))
    if not batches:
        # Пустая партиция — вернуть пустую таблицу с нужной схемой
        fields = [pa.field(c, pa.string()) for c in by_columns]
        for col, f in zip(agg_columns, agg_funcs):
            out_type = pa.int64() if f == "count" else pa.float64()
            fields.append(pa.field(f"{col}_{f}", out_type))
        return pa.Table.from_pylist([], schema=pa.schema(fields))

    table = pa.Table.from_batches(batches)

    if table.num_rows == 0:
        fields = [table.schema.field(c) for c in by_columns]
        for col, f in zip(agg_columns, agg_funcs):
            out_type = pa.int64() if f == "count" else pa.float64()
            fields.append(pa.field(f"{col}_{f}", out_type))
        return pa.Table.from_pylist([], schema=pa.schema(fields))

    # pa.Table.group_by: [(col, "hash"), ...] для нескольких ключей
    agg_specs = []
    for col, f in zip(agg_columns, agg_funcs):
        if f == "mean":
            agg_specs.append((col, "mean"))
        elif f == "sum":
            agg_specs.append((col, "sum"))
        elif f == "count":
            agg_specs.append((col, "count"))
        elif f == "min":
            agg_specs.append((col, "min"))
        elif f == "max":
            agg_specs.append((col, "max"))

    grouped = table.group_by(list(by_columns)).aggregate(agg_specs)
    # Переименовать {col}_{func} — оставляем как есть, downstream знает имена
    return grouped'''

new_body = '''    if isinstance(agg_funcs, str):
        agg_funcs = [agg_funcs] * len(agg_columns)
    if len(agg_funcs) != len(agg_columns):
        raise ValueError("agg_funcs и agg_columns должны быть одной длины")
    for f in agg_funcs:
        if f not in _AGG_FUNCS:
            raise ValueError(f"Неподдерживаемая агрегация: {f!r}. Доступно: {_AGG_FUNCS}")

    # 0. Пустая партиция
    batches = list(ArrowIPCBatchReader(str(partition_path), None))
    if not batches:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    table = pa.Table.from_batches(batches)
    if table.num_rows == 0:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )

    # 1. Размер партиции = сумма nbytes уже прочитанных батчей
    total_bytes = sum(b.nbytes for b in batches)
    safety_limit = int(memory_guard.available_bytes() * 0.5) if memory_guard else total_bytes

    # 2. Влезает — материализуем и агрегируем
    if total_bytes <= safety_limit:
        agg_specs = [(col, f) for col, f in zip(agg_columns, agg_funcs)]
        return table.group_by(list(by_columns)).aggregate(agg_specs)

    # 3. Не влезает — рекурсивное партиционирование с новым seed
    if memory_guard is None or spill_dir is None:
        raise MemoryLimitExceeded(
            f"_local_groupby: partition too large ({total_bytes} B > {safety_limit} B), "
            f"but memory_guard/spill_dir not provided."
        )
    if depth >= max_depth:
        raise MemoryLimitExceeded(
            f"_local_groupby: depth={depth} >= max_depth={max_depth}; "
            f"partition={partition_path.name} size={total_bytes} B > safety={safety_limit} B. "
            f"Вероятен экстремальный skew (см. DECISIONS §11.4 п.3)."
        )

    def _read_for_resplit():
        for b in ArrowIPCBatchReader(str(partition_path), None):
            cols = [c for c in list(by_columns) + list(agg_columns) if c in b.schema.names]
            yield b.select(cols)

    sub_spill = spill_dir / f"resplit_d{depth}_{partition_path.stem}"
    sub_spill.mkdir(parents=True, exist_ok=True)

    sub_value_cols = [c for c in agg_columns if c not in by_columns]
    sub_paths = hash_shuffle(
        _read_for_resplit(),
        key_columns=list(by_columns),
        value_columns=sub_value_cols,
        num_partitions=4,
        memory_guard=memory_guard,
        spill_dir=sub_spill,
        hash_seed=depth + 1,
        log=False,
    )

    results: list[pa.Table] = []
    for sp in sub_paths:
        r = _local_groupby(
            sp, by_columns, agg_columns, agg_funcs,
            memory_guard=memory_guard, spill_dir=spill_dir,
            depth=depth + 1, max_depth=max_depth,
        )
        if r.num_rows > 0:
            results.append(r)
        try:
            sp.unlink()
        except OSError:
            pass

    if not results:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    return pa.concat_tables(results)'''

if old_body not in s:
    raise SystemExit("_local_groupby body: not found")
s = s.replace(old_body, new_body, 1)

# 1c. передача memory_guard/spill_dir в _local_groupby
old_call = '''    for i, p in enumerate(partitions):
        t = _local_groupby(p, by_columns, agg_columns, agg_funcs)'''
new_call = '''    for i, p in enumerate(partitions):
        t = _local_groupby(
            p, by_columns, agg_columns, agg_funcs,
            memory_guard=memory_guard,
            spill_dir=spill_dir,
        )'''
if old_call not in s:
    raise SystemExit("_local_groupby call: not found")
s = s.replace(old_call, new_call, 1)

# 1d. проверка размера sample-буфера
old_sample = '''        batch_iter = itertools.chain(sample_batches, rest_iter)

        k_est = _estimate_k_sample(sample_batches, by_columns)'''
new_sample = '''        sample_bytes = sum(b.nbytes for b in sample_batches)
        sample_limit = int(memory_guard.available_bytes() * 0.5)
        if sample_bytes > sample_limit:
            raise MemoryLimitExceeded(
                f"groupby_aggregate: sample ({sample_bytes} B) > "
                f"available*0.5 ({sample_limit} B). "
                f"Укажите num_partitions явно, чтобы избежать sample."
            )

        batch_iter = itertools.chain(sample_batches, rest_iter)

        k_est = _estimate_k_sample(sample_batches, by_columns)'''
if old_sample not in s:
    raise SystemExit("sample block: not found")
s = s.replace(old_sample, new_sample, 1)

open(p, "w").write(s)
print("patched groupby.py")
PY

# ---------- 2. tests ----------
cat > tests/test_mustfix_local_groupby.py <<'EOF'
"""Тесты must-fix §11.4 п.1 (патч 02b1): рекурсия _local_groupby."""
from pathlib import Path

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops import groupby as gb_mod
from cadw.ops.groupby import _local_groupby, groupby_aggregate


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


def write_partition(path: Path, table: pa.Table, chunk: int = 100) -> Path:
    writer = ArrowIPCBatchWriter(str(path), table.schema)
    try:
        for b in table.to_batches(max_chunksize=chunk):
            writer.write_batch(b)
    finally:
        writer.close()
    return path


# ---------- 1. маленькая партиция: рекурсия не вызывается ----------

def test_local_groupby_no_recursion_small(monkeypatch, tmp_path):
    called = []
    orig = gb_mod.hash_shuffle
    def spy(*a, **kw):
        called.append(kw.get("hash_seed"))
        return orig(*a, **kw)
    monkeypatch.setattr(gb_mod, "hash_shuffle", spy)

    table = pa.table({"cat": ["A", "B", "A", "B"], "val": [1, 2, 3, 4]})
    p = write_partition(tmp_path / "small.arrow", table)
    guard = MemoryGuard(max_memory_mb=10)

    _local_groupby(p, ["cat"], ["val"], ["mean"], memory_guard=guard, spill_dir=tmp_path)
    assert not called, f"recursion should not be called; called={called}"


# ---------- 2. большая партиция: рекурсия вызывается ----------

def test_local_groupby_recursion_called(monkeypatch, tmp_path):
    called = []
    def spy(*a, **kw):
        called.append(kw.get("hash_seed"))
        return []
    monkeypatch.setattr(gb_mod, "hash_shuffle", spy)

    table = pa.table({"cat": ["A"] * 1000, "val": list(range(1000))})
    p = write_partition(tmp_path / "big.arrow", table, chunk=100)
    guard = MemoryGuard(max_memory_bytes=10_000)

    _local_groupby(p, ["cat"], ["val"], ["mean"], memory_guard=guard, spill_dir=tmp_path)
    assert called, "recursion expected but hash_shuffle not called"
    assert any(s is not None and s >= 1 for s in called), f"expected seed>=1; called={called}"


# ---------- 3. MAX_DEPTH -> MemoryLimitExceeded ----------

def test_local_groupby_max_depth_raises_direct(tmp_path):
    table = pa.table({"cat": ["A"] * 100, "val": list(range(100))})
    p = write_partition(tmp_path / "p.arrow", table, chunk=10)
    guard = MemoryGuard(max_memory_bytes=1)

    with pytest.raises(MemoryLimitExceeded, match="max_depth"):
        _local_groupby(
            p, ["cat"], ["val"], ["mean"],
            memory_guard=guard, spill_dir=tmp_path, depth=0, max_depth=0,
        )


# ---------- 4. end-to-end: K=5000, рекурсия возможна, результат корректен ----------

def test_groupby_aggregate_recursion_end_to_end(tmp_path):
    n = 5000
    table = pa.table({
        "cat": [f"g{i:05d}" for i in range(n)],
        "val": list(range(n)),
    })
    guard = MemoryGuard(max_memory_bytes=50_000)
    result = groupby_aggregate(
        make_iter(table, 500), ["cat"], ["val"], ["mean"],
        guard, tmp_path, num_partitions=4, log=False,
    )
    assert result.num_rows == n
    rows = sorted(result.to_pylist(), key=lambda r: r["cat"])
    for i in [0, 1, n // 2, n - 1]:
        assert rows[i]["val_mean"] == float(i), f"row {i}: {rows[i]}"


# ---------- 5. sample > available*0.5 -> MemoryLimitExceeded ----------

def test_groupby_aggregate_sample_too_big_raises(tmp_path):
    n = 5000
    table = pa.table({
        "cat": [f"g{i:05d}" for i in range(n)],
        "val": list(range(n)),
    })
    guard = MemoryGuard(max_memory_bytes=1_000)
    with pytest.raises(MemoryLimitExceeded, match="sample"):
        groupby_aggregate(
            make_iter(table, 500), ["cat"], ["val"], ["mean"],
            guard, tmp_path, num_partitions=None, log=False,
        )
EOF

# ---------- 3. DECISIONS §11.4 status ----------
python3 - <<'PY'
p = "DECISIONS.md"
s = open(p).read()
marker = "**Обновление (патч 02a):**"
if marker not in s:
    raise SystemExit("marker not found")

addition = (
    "**Обновление (патч 02b1):** пункт **1** (`_local_groupby` без `check_batch` + "
    "нет рекурсии) — **закрыт**: партиция оценивается по `sum(b.nbytes)`, при "
    "переполнении — рекурсивный `hash_shuffle` с новым seed, при `depth >= MAX_DEPTH` — "
    "`MemoryLimitExceeded` с указанием на skew. Дополнительно: `groupby_aggregate` "
    "проверяет размер sample-буфера (`sample_bytes > available*0.5` → исключение). "
    "**Остаётся пункт 3 (skew):** при экстремальном skew (один ключ > 10% N) "
    "рекурсия не помогает, нужен skew-детектор + изоляция тяжёлых ключей "
    "(патч 02b2).\n\n"
)
if "патч 02b1" not in s:
    s = s.replace(marker, addition + marker, 1)
open(p, "w").write(s)
print("patched DECISIONS")
PY

# ---------- 4. Sync + tests ----------
uv lock
uv sync
uv run pytest -q

# ---------- 5. Commit ----------
git add -A
git commit -m "02b1: must-fix §11.4 п.1 — _local_groupby recursion + sample budget check"

echo "==> sha256"
sha256sum cadw/ops/groupby.py tests/test_mustfix_local_groupby.py
echo
echo "==> Patch 02b1 applied on ${BR}"
echo "    Base:   ${CUR}"
echo "    Backup: ${BK}  (.backups/${BK}/repo.tgz)"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"
echo "    Hard:   git reset --hard ${BK}"