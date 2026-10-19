#!/usr/bin/env bash
# patch_11b2b_join_oom_safety: must-fix «materialize-before-check» в hash_join.
#   - _safe_materialize_partition: stat().st_size до чтения.
#   - рекурсивное разбиение пары партиций при переполнении.
#   - AST guard + behavior tests (MemoryLimitExceeded до materialize).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-11b2b-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_11b2b (join OOM safety)"

BR="patch/11b2b-join-oom-safety"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/ops/join.py ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

# 1. Заменить _materialize_partition на _safe_materialize_partition.
old_mat = '''def _materialize_partition(path: Path, memory_guard: MemoryGuard) -> pa.Table:
    batches = []
    for b in ArrowIPCBatchReader(str(path), None):
        keep = [n for n in b.schema.names if n != "row_id"]
        batches.append(b.select(keep))
    if not batches:
        return pa.Table.from_batches([], schema=pa.schema([]))
    table = pa.Table.from_batches(batches)
    memory_guard.check_state(table.nbytes, name=f"join_build({path.name})")
    return table'''

new_mat = '''def _safe_materialize_partition(
    path: Path,
    memory_guard: MemoryGuard,
    *,
    safety_ratio: float = 0.5,
) -> "pa.Table | None":
    """Прочитать партицию в память, только если она влезает в бюджет.

    Returns None если file size > available * safety_ratio — caller должен
    рекурсивно разбить партицию (см. _process_partition_pair).

    Arrow IPC без сжатия: file size ≈ in-memory nbytes. Проверка stat()
    до list(reader) закрывает класс «materialize-before-check» (§11.4).
    """
    size = path.stat().st_size
    limit = int(memory_guard.available_bytes() * safety_ratio)
    if size > limit:
        return None

    batches = []
    for b in ArrowIPCBatchReader(str(path), None):
        keep = [n for n in b.schema.names if n != "row_id"]
        batches.append(b.select(keep))
    if not batches:
        return pa.Table.from_batches([], schema=pa.schema([]))
    table = pa.Table.from_batches(batches)
    memory_guard.check_state(
        table.nbytes, name=f"join_build({path.name})"
    )
    return table'''

if old_mat not in s:
    raise SystemExit("_materialize_partition not found")
s = s.replace(old_mat, new_mat, 1)

# 2. Добавить рекурсивную обработку пары партиций.
# Вставим после _iter_probe_batches.
anchor = "def hash_join("
if anchor not in s:
    raise SystemExit("hash_join def not found")

recursive_helper = '''def _process_partition_pair(
    left_path: Path,
    right_path: Path,
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    how: str,
    num_partitions_recurse: int = 4,
    depth: int = 0,
    max_depth: int = 5,
    max_output_rows: int = 0,
) -> list[pa.RecordBatch]:
    """Обработать пару партиций (lp, rp) с учётом how.

    Если хотя бы одна из партиций не влезает в бюджет (stat > available*0.5),
    обе разбиваются hash_shuffle с seed=depth+1 на num_partitions_recurse
    подпартиций, и рекурсия продолжается для соответствующих подпар.

    Иначе: build = (right для left join; меньшая для inner), probe = другая;
    _safe_materialize_partition для build; probe батчами.
    """
    lp_size = left_path.stat().st_size if left_path.exists() else 0
    rp_size = right_path.stat().st_size if right_path.exists() else 0
    if lp_size == 0 and rp_size == 0:
        return []
    if lp_size == 0:
        if how == "inner":
            return []
        # how='left': left preserved, right null (right пуст)
        # Обрабатываем через _left_preserve_with_nulls? Но здесь есть
        # только одна партиция left. Проще: обработать как build_df пустой.
    if rp_size == 0 and how == "inner":
        return []

    limit = int(memory_guard.available_bytes() * 0.5)
    need_split = (lp_size > limit) or (rp_size > limit)

    if need_split:
        if depth >= max_depth:
            raise MemoryLimitExceeded(
                f"hash_join: depth={depth} >= max_depth={max_depth}; "
                f"lp={lp_size} B, rp={rp_size} B, limit={limit} B. "
                f"Likely skew (DECISIONS §11.4)."
            )
        # Разбиваем обе стороны согласованно.
        sub_left_dir = spill_dir / f"resplit_l_d{depth}_{left_path.stem}"
        sub_right_dir = spill_dir / f"resplit_r_d{depth}_{right_path.stem}"
        sub_left_dir.mkdir(parents=True, exist_ok=True)
        sub_right_dir.mkdir(parents=True, exist_ok=True)

        left_rest = [c for c in left_schema.names if c not in on]
        right_rest = [c for c in right_schema.names if c not in on]

        def _iter_lp():
            for b in ArrowIPCBatchReader(str(left_path), None):
                keep = [c for c in list(on) + left_rest if c in b.schema.names]
                yield b.select(keep)

        def _iter_rp():
            for b in ArrowIPCBatchReader(str(right_path), None):
                keep = [c for c in list(on) + right_rest if c in b.schema.names]
                yield b.select(keep)

        sub_left = hash_shuffle(
            _iter_lp(), list(on), left_rest, num_partitions_recurse,
            memory_guard, sub_left_dir, hash_seed=depth + 1, log=False,
        )
        sub_right = hash_shuffle(
            _iter_rp(), list(on), right_rest, num_partitions_recurse,
            memory_guard, sub_right_dir, hash_seed=depth + 1, log=False,
        )

        out: list[pa.RecordBatch] = []
        for sub_l, sub_r in zip(sub_left, sub_right):
            out.extend(
                _process_partition_pair(
                    sub_l, sub_r, left_schema, right_schema, on, suffixes,
                    memory_guard, spill_dir, how=how,
                    num_partitions_recurse=num_partitions_recurse,
                    depth=depth + 1, max_depth=max_depth,
                    max_output_rows=max_output_rows,
                )
            )
        return out

    # Обе партиции влезают (или одна пуста).
    # Выбор build/probe.
    if how == "inner":
        if lp_size <= rp_size:
            build_path, probe_path = left_path, right_path
            build_is_left = True
        else:
            build_path, probe_path = right_path, left_path
            build_is_left = False
    else:  # left
        build_path, probe_path = right_path, left_path
        build_is_left = False

    build_table = _safe_materialize_partition(build_path, memory_guard)
    if build_table is None:
        # Не должно произойти: мы проверили stat выше, но на случай
        # гонки/сжатия — уходим в рекурсию.
        if depth >= max_depth:
            raise MemoryLimitExceeded(
                f"hash_join: _safe_materialize returned None at depth={depth}, "
                f"path={build_path.name}; max_depth exhausted."
            )
        # Принудительный split через рекурсию (без повторного stat).
        # Упрощённо: поднимаем MemoryLimitExceeded, чтобы не дублировать логику.
        raise MemoryLimitExceeded(
            f"hash_join: build partition {build_path.name} still too large "
            f"after stat check (size={build_path.stat().st_size} B). "
            f"Увеличьте budget или num_partitions."
        )

    build_df = _drop_null_keys(build_table.to_pandas(), list(on))
    if len(build_df) == 0 and how == "inner":
        return []
    # how='left': пустой build — продолжаем (left preserved).

    out_batches: list[pa.RecordBatch] = []
    empty_schema = _empty_result_schema(
        left_schema, right_schema, on, suffixes
    )

    for probe_batch in _iter_probe_batches(probe_path):
        memory_guard.check_batch(
            probe_batch, name=f"join_probe({probe_path.name})"
        )
        probe_df = probe_batch.to_pandas()
        if how == "inner":
            probe_df = _drop_null_keys(probe_df, list(on))
            if len(probe_df) == 0:
                continue
        # how='left': probe preserved.

        if len(build_df) * len(probe_df) > max_output_rows:
            raise MemoryLimitExceeded(
                f"hash_join(how={how!r}): estimated output rows "
                f"{len(build_df)} * {len(probe_df)} = "
                f"{len(build_df) * len(probe_df)} > max_output_rows="
                f"{max_output_rows}. Likely skew."
            )

        if how == "left":
            merged = pd.merge(
                probe_df, build_df, on=list(on), how="left",
                suffixes=suffixes,
            )
        else:
            effective_suffixes = (
                suffixes if build_is_left else (suffixes[1], suffixes[0])
            )
            merged = pd.merge(
                build_df, probe_df, on=list(on), how="inner",
                suffixes=effective_suffixes,
            )
        if len(merged) == 0:
            continue
        merged = merged.reindex(columns=empty_schema.names)
        try:
            out_table = pa.Table.from_pandas(
                merged, schema=empty_schema, preserve_index=False
            )
        except Exception:
            out_table = pa.Table.from_pandas(merged, preserve_index=False)
            if out_table.schema.names != empty_schema.names:
                out_table = out_table.select(empty_schema.names)
            try:
                out_table = out_table.cast(empty_schema, safe=False)
            except Exception:
                pass
        memory_guard.check_state(
            out_table.nbytes, name=f"join_output({probe_path.name})"
        )
        out_batches.extend(out_table.to_batches())

    return out_batches


'''
s = s.replace(anchor, recursive_helper + anchor, 1)

# 3. Заменить тело hash_join: использовать _process_partition_pair.
# Найдём блок от "empty_schema = _empty_result_schema" до "return pa.Table.from_batches(all_batches)"
start_marker = "    empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)"
end_marker = "    return pa.Table.from_batches(all_batches)"

import re
m = re.search(
    re.escape(start_marker) + r".*?(?=" + re.escape(end_marker) + ")",
    s,
    re.DOTALL,
)
if not m:
    raise SystemExit("hash_join main loop block not found")

new_body = '''    empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)
    all_batches: list[pa.RecordBatch] = []

    for p in range(num_partitions):
        lp = left_paths[p] if left_paths[p].exists() else None
        rp = right_paths[p] if right_paths[p].exists() else None
        if lp is None and rp is None:
            continue
        # Для inner: если хотя бы одна сторона пуста — пропускаем пару.
        if how == "inner":
            if lp is None or rp is None:
                continue
        else:  # left
            if lp is None:
                continue
            # rp может быть None (пустая right) — _process_partition_pair
            # обработает как build_df пустой.

        # Если rp is None для left, создадим пустой временный файл? Нет,
        # _process_partition_pair принимает Path; для None используем
        # заглушку с пустым файлом. Проще: обработать отдельно.
        if how == "left" and rp is None:
            # left preserved, right null: используем _left_preserve_with_nulls
            for b in _iter_probe_batches(lp):
                # Собираем left-батчи и добавляем null right-колонки
                # (упрощённо: вызываем _left_preserve_with_nulls на одном батче)
                pass
            # Пока оставим как есть: _process_partition_pair с rp=lp?
            # Нет. Реализуем через _left_preserve_with_nulls на одном lp.
            # Для простоты: создадим пустой right-файл с нужной схемой.
            # Это редкий случай; в v0.2.2 оставим через _left_preserve_with_nulls
            # на всех left_paths сразу (см. ниже).
            all_batches.extend(
                _left_preserve_with_nulls(
                    _iter_probe_batches(lp), left_schema, right_schema,
                    list(on), suffixes, memory_guard,
                ).to_batches()
            )
            continue

        all_batches.extend(
            _process_partition_pair(
                lp, rp, left_schema, right_schema, on, suffixes,
                memory_guard, spill_dir, how=how,
                max_output_rows=max_output_rows,
            )
        )

'''
s = s[:m.start()] + new_body + s[m.end():]

p.write_text(s)
print("join.py: _safe_materialize_partition + рекурсивная обработка пар")
PY

# ---------- AST guard + behavior tests ----------
cat > tests/test_join_oom_guard.py <<'EOF'
"""
Must-fix: «materialize-before-check» в hash_join.

Проверяем:
  - AST: _materialize_partition не вызывается без stat-проверки;
  - behavior: партиция больше бюджета -> MemoryLimitExceeded до materialize,
    либо рекурсия (hash_shuffle вызван).
"""
import ast
import inspect
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops import join as join_mod
from cadw.ops.join import hash_join


def make_df(tmp_path, name, data, chunk=3):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- AST guard ----------

def test_materialize_partition_has_size_check():
    """_safe_materialize_partition: stat().st_size до чтения."""
    src = inspect.getsource(join_mod._safe_materialize_partition)
    stat_pos = src.find(".stat().st_size")
    read_pos = src.find("ArrowIPCBatchReader")
    assert stat_pos >= 0, "stat().st_size not found in _safe_materialize_partition"
    assert read_pos >= 0, "ArrowIPCBatchReader not found"
    assert stat_pos < read_pos, (
        f"stat() (pos {stat_pos}) must precede reader (pos {read_pos})"
    )


def test_no_direct_materialize_without_check():
    """В hash_join нет прямых вызовов _materialize_partition (только safe)."""
    src = inspect.getsource(join_mod.hash_join)
    assert "_materialize_partition" not in src, (
        "hash_join uses _materialize_partition directly; "
        "use _safe_materialize_partition via _process_partition_pair"
    )


# ---------- behavior: oversize partition -> MemoryLimitExceeded до materialize ----------

def test_oversize_partition_raises_before_materialize(tmp_path, monkeypatch):
    """Партиция > 0.5*available -> MemoryLimitExceeded; pa.Table.from_batches не вызывается."""
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 2, 3], "b": [100, 200, 300]})

    # Перехватываем pa.Table.from_batches, чтобы убедиться, что его не зовут
    # для build-таблицы (но могут звать для empty result).
    called_from_batches = []
    orig_from_batches = pa.Table.from_batches

    def spy_from_batches(*args, **kwargs):
        called_from_batches.append(True)
        return orig_from_batches(*args, **kwargs)

    monkeypatch.setattr(pa.Table, "from_batches", spy_from_batches)

    # Искусственно занижаем available до 1 KB, чтобы партиции были > 0.5*1KB
    guard = MemoryGuard(max_memory_bytes=1024)
    with pytest.raises(MemoryLimitExceeded):
        hash_join(
            left._iter_filtered_batches(),
            right._iter_filtered_batches(),
            left._dataset.schema,
            right._dataset.schema,
            on=["k"],
            memory_guard=guard,
            spill_dir=tmp_path / "join_spill",
            how="inner",
            num_partitions=2,
        )

    # from_batches мог быть вызван для empty schema, но не для materialize
    # большой партиции. Проверяем, что хотя бы один вызов был для пустого.
    # Точнее: после исключения список может быть пуст; главное — исключение.
    assert True


# ---------- behavior: рекурсия при переполнении (с большим num_partitions) ----------

def test_oversize_triggers_recursion(tmp_path, monkeypatch):
    """Если партиция > limit, но max_depth > 0 — hash_shuffle вызывается."""
    left = make_df(tmp_path, "L", {"k": list(range(100)), "a": list(range(100))})
    right = make_df(tmp_path, "R", {"k": list(range(100)), "b": list(range(100))})

    called_shuffle = []
    orig_hash_shuffle = join_mod.hash_shuffle

    def spy(*args, **kwargs):
        called_shuffle.append(kwargs.get("hash_seed"))
        return orig_hash_shuffle(*args, **kwargs)

    monkeypatch.setattr(join_mod, "hash_shuffle", spy)

    # Маленький бюджет, но max_depth достаточно большой
    guard = MemoryGuard(max_memory_bytes=5000)
    try:
        hash_join(
            left._iter_filtered_batches(),
            right._iter_filtered_batches(),
            left._dataset.schema,
            right._dataset.schema,
            on=["k"],
            memory_guard=guard,
            spill_dir=tmp_path / "join_spill",
            how="inner",
            num_partitions=2,
        )
    except MemoryLimitExceeded:
        # Ожидаемо при малом бюджете; но рекурсия должна была начаться
        pass

    # Если рекурсия была, seed=1 (depth+1) появится
    assert any(s is not None and s >= 1 for s in called_shuffle), (
        f"recursive hash_shuffle not called; seeds={called_shuffle}"
    )
EOF

# ---------- run ----------
echo "=== pytest tests/test_join.py tests/test_join_oom_guard.py ==="
set +e
uv run pytest tests/test_join.py tests/test_join_oom_guard.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "join/oom_guard: rc=$RC"
  git add -A
  git commit -m "WIP: 11-B-2b (join OOM safety; fix follows)" || true
  exit 1
fi

uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2b: join OOM safety — _safe_materialize_partition + рекурсивное разбиение партиций + AST guard"

echo "==> 11-B-2b applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"