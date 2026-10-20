#!/usr/bin/env python3
"""v0.8.11a-fix3: make _local_dedup schema-order agnostic.

DISCOVERY (not creation — §13.4/145):
  fix2-test wrote partition with row_id FIRST (not last like hash_shuffle
  output). _local_dedup called Table.from_batches(gen.select([..., row_id]),
  schema=reader.schema) — select reorders, reader.schema is file order.
  In production hash_shuffle always puts row_id last → bug invisible.
  Test discovered latent brittleness. NOT round 7 of §13.4/139-144 cluster.

FIX:
  1. _local_dedup: explicit select_schema = out_fields + [row_id].
     Robust in any file column order.
  2. New test_dedup_schema_order_agnostic (row_id first).
  3. §13.4/145 — discovery vs creation, test-discipline works.

Run: uv run python scripts/patch_v0811a_fix3.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


def fix_local_dedup_schema():
    print("\n=== cadw/ops/dedup.py::_local_dedup — explicit select_schema ===")
    p = Path("cadw/ops/dedup.py")
    s = p.read_text()

    old = '''    table = pa.Table.from_batches(
        (reader.get_batch(i).select([f.name for f in out_fields] + ["row_id"])
         for i in range(n_batches)),
        schema=reader.schema,
    )'''

    new = '''    # v0.8.11a-fix3: explicit select_schema. `reader.schema` is file
    # column order; `.select(out_fields + ["row_id"])` reorders. In
    # production hash_shuffle always puts row_id last so they coincide,
    # but relying on that is brittle (any other IPC writer breaks).
    # Explicit schema makes _local_dedup agnostic to file column order.
    select_schema = pa.schema(
        list(out_fields) + [reader.schema.field("row_id")]
    )
    table = pa.Table.from_batches(
        (reader.get_batch(i).select([f.name for f in out_fields] + ["row_id"])
         for i in range(n_batches)),
        schema=select_schema,
    )'''

    s = replace_line_block(s, old, new, tag="local-dedup-select-schema")
    p.write_text(s)
    compile_check(p)
    print("  OK: select_schema = out_fields + [row_id]")


def add_schema_order_test():
    print("\n=== tests/test_dedup_arrow_native.py: "
          "test_dedup_schema_order_agnostic ===")
    p = Path("tests/test_dedup_arrow_native.py")
    s = p.read_text()
    if "test_dedup_schema_order_agnostic" in s:
        print("  already present — skip")
        return

    new_test = '''

# ============ v0.8.11a-fix3: schema-order robustness ============

def test_dedup_schema_order_agnostic(tmp_path):
    """_local_dedup works regardless of row_id position in file schema.

    v0.8.11a-fix3 (§13.4/145): production hash_shuffle always puts
    row_id last, but a partition from any other IPC writer might not.
    _local_dedup builds explicit select_schema (out_fields + [row_id])
    so file column order doesn't matter.

    Test writes partition with row_id FIRST (opposite of hash_shuffle).
    """
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.ops.dedup import _local_dedup

    table = pa.table({
        "row_id": pa.array(range(10), type=pa.uint64()),  # FIRST
        "k": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
        "v": list(range(10)),
    })
    p = tmp_path / "partition_rid_first.arrow"
    w = ArrowIPCBatchWriter(str(p), table.schema)
    try:
        for b in table.to_batches(max_chunksize=5):
            w.write_batch(b)
    finally:
        w.close()

    guard = MemoryGuard(max_memory_mb=10)
    result = _local_dedup(p, ["k"], "first", guard)

    assert result.num_rows == 5
    assert "row_id" not in result.schema.names
    assert set(result.column("k").to_pylist()) == {"a", "b", "c", "d", "e"}
'''

    if not s.endswith("\n"):
        s += "\n"
    s += new_test
    p.write_text(s)
    compile_check(p)
    print("  OK: test_dedup_schema_order_agnostic")


def append_decisions_145():
    print("\n=== DECISIONS.md: §13.4/145 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 145" in s:
        print("  already present — skip")
        return

    marker = "### 13.5. Открытые вопросы"
    if marker not in s:
        print("  WARNING: §13.5 marker not found — skip")
        return

    episode = """**Эпизод 145 (discovery, not creation — test found latent bug).**
Кластер §13.4/139-144 = 6 раундов «fix вносит новый баг». Round 7
(v0.8.11a-fix2) — **другой класс**: fix-тест **обнаружил** старую
latent проблему.

**Что нашлось:** `_local_dedup` (v0.8.11a, и v0.8.9 до неё) делал:
`Table.from_batches((batch.select([out_fields..., row_id]) for ...),
schema=reader.schema)`. `.select()` переупорядочивает в
`[out_fields..., row_id]`. `reader.schema` — file order. В production
`hash_shuffle` всегда кладёт row_id последним → порядки совпадают →
баг невидим. fix2-тест написал partition с `row_id` ПЕРВЫМ →
`ArrowInvalid: Schema at index 0 was different`.

**Различие классов:**

| раунд | что | кто нашёл |
|---|---|---|
| 1-6 (§13.4/139-144) | fix-код внёс баг | reviewer / CI |
| **7 (§13.4/145)** | **pre-existing latent bug** | **fix2-тест** |

**Fix (v0.8.11a-fix3):** explicit `select_schema = pa.schema(out_fields
+ [reader.schema.field("row_id")])`. Robust в любом file column order.
Новый `test_dedup_schema_order_agnostic` пишет row_id first — coverage
robustness, не workaround.

**Урок (positive):** тест-дисциплина (§13.4/140 — guard-tests before
commit) работает как задумано. fix2-тест, написанный со смещённым
допущением, **нашёл реальную хрупкость**, которая в production
никогда не проявлялась. 1 wasted round — приемлемая цена.

**Правило:** production код не должен зависеть от **неявных инвариантов
layout входных данных**. Если инвариант нужен — сделать explicit
(`select_schema`, schema cast) или документировать + assert.

**Backlog §11.6 (v0.9.0+):** audit всех `Table.from_batches(gen, schema=...)`
в ops на аналогичные неявные предположения о column order. Кандидаты:
`join._safe_materialize_partition` (явный `out_schema` — OK),
`sort._write_run` (schema после take — OK), `groupby._read_partition_safe`
— проверить в v0.8.11b.

**Связанные:** §13.4/128, /129, /139-144.

"""
    s = s.replace(marker, episode + marker, 1)
    p.write_text(s)
    print("  OK: §13.4/145")


def main():
    print("=== v0.8.11a-fix3 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_local_dedup_schema()
    add_schema_order_test()
    append_decisions_145()

    print("\n=== pytest: targeted (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pingis_safety.py::test_dedup_arrow_native_precheck_raises",
         "tests/test_dedup_arrow_native.py::test_dedup_schema_order_agnostic",
         "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** TARGETED STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: dedup + pingis + guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pingis_safety.py",
         "tests/test_dedup_arrow_native.py",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_changelog.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_drop_duplicates.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FOCUSED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1200)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "-A"])
    sh(["git", "commit", "--amend", "-m",
        "v0.8.11a: Arrow-native dedup (P-D1 close) - Path C "
        "(dictionary_encode + group_by + is_in), 0.29x peak vs pandas; "
        "pandas removed from ops hot path; AST guard no_pandas_in_ops; "
        "§13.4/143-145 (substring, cross-file rename, schema-order)"])

    print("\n==> v0.8.11a-fix3 applied. Amended WIP commit.")
    print("NEXT: v0.8.11b (strict join conditional).")


if __name__ == "__main__":
    main()
