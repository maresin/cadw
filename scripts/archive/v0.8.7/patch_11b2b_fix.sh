#!/usr/bin/env bash
# patch_11b2b_fix: пустые файлы → None в hash_join; пустой build_df для
# left join; тест на OOM без monkeypatch на pa.Table.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

BR="patch/11b2b-join-oom-safety"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2b-fix-${TS}"
git diff > ".backups/backup-pre-11b2b-fix-${TS}/wip.diff"

# ---------- Fix 1: hash_join — пустые файлы как None ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

old = '''    for p in range(num_partitions):
        lp = left_paths[p] if left_paths[p].exists() else None
        rp = right_paths[p] if right_paths[p].exists() else None
        if lp is None and rp is None:
            continue'''

new = '''    for p in range(num_partitions):
        # Пустой файл (0 байт) = отсутствующий: hash_shuffle создаёт файлы
        # для всех P, но без данных они 0 байт. materialize такого файла
        # даст Table без колонок → pd.merge KeyError.
        lp = (
            left_paths[p]
            if left_paths[p].exists() and left_paths[p].stat().st_size > 0
            else None
        )
        rp = (
            right_paths[p]
            if right_paths[p].exists() and right_paths[p].stat().st_size > 0
            else None
        )
        if lp is None and rp is None:
            continue'''

if old not in s:
    raise SystemExit("hash_join main loop pattern not found")
s = s.replace(old, new, 1)
print("join.py: hash_join — пустые файлы → None")
PY

# ---------- Fix 2: _process_partition_pair — empty build_df для left ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

old = '''    build_df = _drop_null_keys(build_table.to_pandas(), list(on))
    if len(build_df) == 0 and how == "inner":
        return []
    # how='left': пустой build — продолжаем (left preserved).
'''

new = '''    build_df = _drop_null_keys(build_table.to_pandas(), list(on))
    if len(build_df) == 0:
        if how == "inner":
            return []
        # how='left': build = right; пустой build (0 совпадений) —
        # pd.merge требует on-колонки, поэтому создаём пустой DataFrame
        # с правильными колонками.
        right_rest = [c for c in right_schema.names if c not in on]
        build_df = pd.DataFrame(columns=list(on) + right_rest)
'''

if old not in s:
    raise SystemExit("build_df empty block not found")
s = s.replace(old, new, 1)
print("join.py: _process_partition_pair — empty build_df с колонками")
PY

# ---------- Fix 3: переписать проблемный тест ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_join_oom_guard.py")
s = p.read_text()

old_test = '''def test_oversize_partition_raises_before_materialize(tmp_path, monkeypatch):
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
    assert True'''

new_test = '''def test_safe_materialize_returns_none_for_oversize(tmp_path):
    """_safe_materialize_partition: stat > 0.5*available -> None (без чтения)."""
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.ops.join import _safe_materialize_partition

    table = pa.table({
        "k": list(range(1000)),
        "v": list(range(1000)),
    })
    p = tmp_path / "big.arrow"
    writer = ArrowIPCBatchWriter(str(p), table.schema)
    try:
        for b in table.to_batches(max_chunksize=100):
            writer.write_batch(b)
    finally:
        writer.close()

    # Бюджет 100 B — файл заведомо > 50 B (0.5 * 100)
    guard = MemoryGuard(max_memory_bytes=100)
    result = _safe_materialize_partition(p, guard)
    assert result is None, (
        f"_safe_materialize_partition must return None for oversize file "
        f"(size={p.stat().st_size}, limit=50)"
    )


def test_oversize_partition_raises_with_max_depth_zero(tmp_path):
    """max_depth=0 + oversize партиция -> MemoryLimitExceeded ДО materialize."""
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.ops.join import _process_partition_pair

    # Создаём пару партиций: left с данными, right с данными.
    left_t = pa.table({"k": list(range(1000)), "a": list(range(1000))})
    right_t = pa.table({"k": list(range(1000)), "b": list(range(1000))})
    lp = tmp_path / "lp.arrow"
    rp = tmp_path / "rp.arrow"
    for path, t in [(lp, left_t), (rp, right_t)]:
        w = ArrowIPCBatchWriter(str(path), t.schema)
        try:
            for b in t.to_batches(max_chunksize=100):
                w.write_batch(b)
        finally:
            w.close()

    guard = MemoryGuard(max_memory_bytes=100)  # 100 B; файлы >> 50 B
    left_schema = pa.schema([("k", pa.int64()), ("a", pa.int64())])
    right_schema = pa.schema([("k", pa.int64()), ("b", pa.int64())])

    with pytest.raises(MemoryLimitExceeded, match="max_depth"):
        _process_partition_pair(
            lp, rp, left_schema, right_schema, ["k"], ("_x", "_y"),
            guard, tmp_path / "spill", how="inner",
            depth=0, max_depth=0, max_output_rows=10**9,
        )'''

if old_test not in s:
    raise SystemExit("old test not found in test_join_oom_guard.py")
s = s.replace(old_test, new_test, 1)
p.write_text(s)
print("test_join_oom_guard.py: монkeypatch pa.Table убран, тесты переписаны")
PY

# ---------- run ----------
echo "=== pytest tests/test_join.py tests/test_join_oom_guard.py ==="
set +e
uv run pytest tests/test_join.py tests/test_join_oom_guard.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "join/oom_guard fix: rc=$RC"
  exit 1
fi

uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2b-fix: пустые файлы → None в hash_join; empty build_df с колонками для left; тесты без monkeypatch на pa.Table"

echo "==> 11-B-2b-fix applied"