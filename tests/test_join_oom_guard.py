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
    """_safe_materialize_partition: stat().st_size до любого чтения IPC.

    Чтение может быть через ArrowIPCBatchReader или ipc.open_file — оба
    читают файл. AST-guard: позиция .stat().st_size < позиция первого
    вызова, читающего файл.
    """
    fn = join_mod._safe_materialize_partition
    src = inspect.getsource(fn)
    stat_pos = src.find(".stat().st_size")
    assert stat_pos >= 0, "stat().st_size not found in _safe_materialize_partition"

    # Ищем в AST все Call-узлы, читающие файл.
    import ast
    tree = ast.parse(inspect.cleandoc(src))
    read_names = {"ArrowIPCBatchReader", "ParquetBatchReader"}
    read_attrs = {"open_file", "open_stream", "memory_map"}

    def is_file_read(call: ast.Call) -> bool:
        f = call.func
        if isinstance(f, ast.Name) and f.id in read_names:
            return True
        if isinstance(f, ast.Attribute) and f.attr in read_attrs:
            return True
        return False

    # Найдём минимальную lineno среди file-read calls.
    read_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and is_file_read(node)
    ]
    assert read_lines, (
        "no file-read call found in _safe_materialize_partition "
        "(expected ipc.open_file or ArrowIPCBatchReader)"
    )
    first_read_line = min(read_lines)

    # stat() должен быть раньше первого чтения. Строки считаем по offset:
    # stat_pos — символьная позиция, read_line — номер строки. Сравниваем
    # через подсчёт строк до stat_pos.
    stat_line = src.count("\n", 0, stat_pos) + 1
    assert stat_line < first_read_line, (
        f"stat() at line {stat_line} must precede first file-read at line "
        f"{first_read_line}. Source:\n{src}"
    )


def test_no_direct_materialize_without_check():
    """В hash_join нет прямых вызовов _materialize_partition (только safe)."""
    src = inspect.getsource(join_mod.hash_join)
    assert "_materialize_partition" not in src, (
        "hash_join uses _materialize_partition directly; "
        "use _safe_materialize_partition via _process_partition_pair"
    )


# ---------- behavior: oversize partition -> MemoryLimitExceeded до materialize ----------

def test_safe_materialize_returns_none_for_oversize(tmp_path):
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

    # v0.5.1: _process_partition_pair is now a generator; trigger body.
    with pytest.raises(MemoryLimitExceeded, match="max_depth"):
        next(_process_partition_pair(
            lp, rp, left_schema, right_schema, ["k"], ("_x", "_y"),
            guard, tmp_path / "spill", how="inner",
            depth=0, max_depth=0, max_output_rows=10**9,
        ))


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

def test_safe_materialize_empty_file_has_columns(tmp_path):
    """Schema-only IPC file → таблица с колонками, 0 строк."""
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.ops.join import _safe_materialize_partition

    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    p = tmp_path / "empty.arrow"
    writer = ArrowIPCBatchWriter(str(p), schema)
    try:
        pass  # 0 batches — но schema-заголовок записан
    finally:
        writer.close()

    assert p.stat().st_size > 0, "Arrow IPC schema header must be > 0 B"

    guard = MemoryGuard(max_memory_bytes=10 * 1024 * 1024)
    table = _safe_materialize_partition(p, guard)
    assert table is not None
    assert set(table.schema.names) == {"k", "v"}, (
        f"empty file must have schema columns, got {table.schema.names}"
    )
    assert table.num_rows == 0

def test_actual_matches_checked_before_join():
    """Exact-match guard вызывается ДО join-операции в _process_partition_pair.

    v0.5.3 (P1): pandas → Arrow Table.join. Guard:
    - `_exact_actual_matches_arrow` must be called;
    - `Table.join` (или `.join(..., join_type=...)`) вызывается после.

    Регрессия 11-B-2d: cartesian check (build × probe > max_cartesian_rows)
    давал ложные reject'ы для PK-FK. Fix: точная оценка через
    `_exact_actual_matches_arrow` (pc.value_counts + join).
    """
    from cadw.ops.join import _process_partition_pair
    import ast

    src = inspect.getsource(_process_partition_pair)
    tree = ast.parse(inspect.cleandoc(src))

    exact_lines: list[int] = []
    join_lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        # Arrow-native exact-match check
        if isinstance(f, ast.Name) and f.id == "_exact_actual_matches_arrow":
            exact_lines.append(node.lineno)
        # Table.join — method call with `join_type=` kwarg (signature
        # `left_t.join(right_t, keys=..., join_type=...)`).
        if isinstance(f, ast.Attribute) and f.attr == "join":
            has_join_type = any(
                kw.arg == "join_type" for kw in node.keywords
            )
            if has_join_type:
                join_lines.append(node.lineno)

    assert exact_lines, (
        "_exact_actual_matches_arrow not called in _process_partition_pair "
        "(P1 v0.5.3 expects Arrow-native exact-match guard)"
    )
    assert join_lines, (
        "Table.join (with join_type= kwarg) not called in "
        "_process_partition_pair — P1 v0.5.3 expects Arrow-native join"
    )
    assert min(exact_lines) < min(join_lines), (
        f"_exact_actual_matches_arrow (line {min(exact_lines)}) must precede "
        f"Table.join (line {min(join_lines)})"
    )


def test_no_cartesian_check_in_join():
    """max_cartesian_rows не упоминается в join.py (cartesian — не OOM-граница)."""
    from cadw.ops import join as join_mod
    src = inspect.getsource(join_mod)
    offending = [
        (i + 1, line) for i, line in enumerate(src.splitlines())
        if "max_cartesian_rows" in line and not line.strip().startswith("#")
    ]
    assert not offending, (
        "max_cartesian_rows must be removed from join.py code: "
        f"{offending}"
    )

