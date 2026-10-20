#!/usr/bin/env python3
"""v0.8.11a-fix2: fix stale test (v0.8.9 → v0.8.11a guard rename).

BUG (round 6 of the cluster, §13.4/144):
  `tests/test_pingis_safety.py::test_dedup_pandas_chain_precheck_raises`
  was added in v0.8.9 to verify the 5×T `pandas_chain` guard. v0.8.11a
  (P-D1 close) replaced the guard with 1.5×T `arrow_native` — but the
  test in `test_pingis_safety.py` was NOT updated (different file from
  the ones the v0.8.11a patch script edited).

  v0.8.11a-fix1's fast run caught it. One wasted iteration.

  Attempting to patch the test by changing num_partitions=1 + budget
  proved fragile: file size ≈ 35 KB, close to size_limit=40 KB at
  80 KB budget, so size-check may fire first with a different message.

FIX:
  Rewrite as direct test of `_local_dedup` (not `drop_duplicates`).
  Budget = 2.5 × table.nbytes (deterministic):
    - file_size ≈ T ⇒ T < 0.5 × budget = 1.25 × T ✓ (size passes)
    - 1.5 × T > 0.5 × budget = 1.25 × T ✓ (arrow_native fires)
  Zero reliance on key-size estimates or partition count.

META (§13.4/144 — round 6 of the "fix creates new bug" cluster):
  - v0.8.9: AST guard too broad (Table vs Dataset)
  - v0.8.9-fix: pc.cast overflow / pa.field nullable defaults
  - v0.8.9-fix2: stale test (uint64 fallback removed, test not updated)
  - v0.8.10-pre: [0.8.10-pre] broke changelog guards
  - v0.8.11a: test substring (5* in 1.5*) / CHANGELOG skip
  - v0.8.11a-fix1: stale test in a DIFFERENT file (pandas_chain → arrow_native)

  Common class: **assumed compatibility, not verified**. This round is
  specifically "test in another file references the guard we renamed".
  Rule extension (§13.4/143 continues): patch that renames/removes a
  guard name MUST `grep <old_name> tests/` and update every reference.

  Long-term fix (v0.9.0 backlog): guard names → module constants
  (`DEDUP_GUARD_NAME = "dedup.arrow_native"`), tests import constants.
  Renaming = one-line source change; tests follow automatically.

Run: uv run python scripts/patch_v0811a_fix2.py
"""
from __future__ import annotations

import os
import re
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


# ============ Pre-check: grep for stale guard-name references ============

def grep_stale_guard_names():
    print("\n=== pre-check: grep 'pandas_chain' across tests/ ===")
    r = subprocess.run(
        ["grep", "-rn", "pandas_chain", "tests/"],
        capture_output=True, text=True,
    )
    lines = [l for l in r.stdout.strip().splitlines() if l]
    if not lines:
        print("  (no matches — already clean)")
        return
    print(f"  {len(lines)} reference(s) found:")
    for l in lines:
        print(f"    {l}")
    print("\n  Expected: only `test_pingis_safety.py` uses `match=\"pandas_chain\"`; "
          "`test_dedup_arrow_native.py` uses `\"dedup.pandas_chain\" not in src` "
          "(intentional — verifies removal).")


# ============ Fix: rewrite test ============

def fix_pingis_test():
    print("\n=== tests/test_pingis_safety.py: "
          "test_dedup_pandas_chain_precheck_raises → test_dedup_arrow_native_precheck_raises ===")
    p = Path("tests/test_pingis_safety.py")
    s = p.read_text()

    old = '''def test_dedup_pandas_chain_precheck_raises(tmp_path):
    """Oversize partition → MemoryLimitExceeded before pandas materialize."""
    from cadw.ops.dedup import drop_duplicates

    n = 2000
    table = pa.table({
        "k": [f"k{i % 100}" for i in range(n)],
        "v": list(range(n)),
    })

    def it():
        for i in range(0, n, 100):
            yield table.slice(i, 100).to_batches()[0]

    # Budget 80 KB: partition ~19.5 KB, size_limit=40 KB (passes old check),
    # but new 5×T = 5×19.5 = 97.5 KB > 40 KB → MemoryLimitExceeded from
    # pandas_chain check (not the older size check). Tests new semantics.
    guard = MemoryGuard(max_memory_bytes=80000)
    with pytest.raises(MemoryLimitExceeded, match="pandas_chain"):
        drop_duplicates(
            it(), subset=["k"], keep="first",
            memory_guard=guard,
            spill_dir=tmp_path,
            num_partitions=4,
        )'''

    new = '''def test_dedup_arrow_native_precheck_raises(tmp_path):
    """Oversize partition → MemoryLimitExceeded from arrow_native guard.

    v0.8.11a-fix2 (§13.4/144): renamed from
    `test_dedup_pandas_chain_precheck_raises`. v0.8.11a (P-D1 close)
    replaced the v0.8.9 5×T `pandas_chain` stopgap with 1.5×T
    `arrow_native` guard. The old test in this file was stale — checked
    for a guard name that no longer exists.

    Direct test of `_local_dedup` (not the full `drop_duplicates`) so we
    control the budget from `table.nbytes`: budget = 2.5 × T means
      - file_size ≈ T < 0.5 × budget = 1.25 × T  → size check passes
      - 1.5 × T > 0.5 × budget = 1.25 × T        → arrow_native fires
    Deterministic — no reliance on string-key size estimate or P.
    """
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.ops.dedup import _local_dedup

    n = 2000
    table = pa.table({
        "row_id": pa.array(range(n), type=pa.uint64()),
        "k": [f"k{i % 100}" for i in range(n)],
        "v": list(range(n)),
    })
    p = tmp_path / "partition.arrow"
    writer = ArrowIPCBatchWriter(str(p), table.schema)
    try:
        for b in table.to_batches(max_chunksize=500):
            writer.write_batch(b)
    finally:
        writer.close()

    budget = int(2.5 * table.nbytes)
    guard = MemoryGuard(max_memory_bytes=budget)
    with pytest.raises(MemoryLimitExceeded, match="arrow_native"):
        _local_dedup(p, ["k"], "first", guard)'''

    s = replace_line_block(s, old, new, tag="pingis-dedup-test")
    p.write_text(s)
    compile_check(p)
    print("  OK: test rewritten (direct `_local_dedup`, budget=2.5×T)")


# ============ §13.4/144 ============

def append_decisions_144():
    print("\n=== DECISIONS.md: §13.4/144 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 144" in s:
        print("  already present — skip")
        return

    marker = "### 13.5. Открытые вопросы"
    if marker not in s:
        print("  WARNING: §13.5 marker not found — skip")
        return

    episode = '''**Эпизод 144 (cross-file test staleness on guard rename).** Шестой
раунд кластера §13.4/139-143.

**Что случилось:** v0.8.9 добавил
`tests/test_pingis_safety.py::test_dedup_pandas_chain_precheck_raises`,
проверяющий `match="pandas_chain"` (guard 5×T stopgap). v0.8.11a
(P-D1 close) заменил guard на `arrow_native` (1.5×T), но test
в **другом файле** не был обновлён — patch-скрипт v0.8.11a правил
`test_dedup_arrow_native.py` и `test_no_pandas_in_ops.py`, а
`test_pingis_safety.py` (историческая куча) был вне его поля зрения.

**Как поймали:** fast run в `patch_v0811a_fix1.py` (§13.4/140
дисциплина: guard-tests до commit). Один wasted round.

**Fix (v0.8.11a-fix2):** переписать test напрямую на `_local_dedup`
с `budget = 2.5 × table.nbytes` (детерминированно: file ≈ T,
size passes, 1.5×T fires). Не через `drop_duplicates` — там
неконтролируемый hash_shuffle, размер партиции зависит от key-size
estimate.

**Правило (расширение 143):** patch, который renaming/removing guard
name, **обязан** `grep <old_name> tests/` и обновить **все** ссылки —
не только в своих новых файлах. Grep — часть pre-check, не после.

**Long-term (v0.9.0 backlog):** guard names → module constants
(`DEDUP_GUARD_NAME = "dedup.arrow_native"`, `JOIN_RSS_WARNING_NAME =
"join.whole_operation_rss"`), tests import from module. Renaming = одна
строка в source; tests follow automatically. Заодно — единый источник
правды для документации (§9.2.N ссылается на те же имена).

**Кластер (6 раундов):**
| раунд | источник | фикс |
|---|---|---|
| v0.8.9 | AST guard broad (Table vs Dataset) | whitelist + `_is_table_from_batches` |
| v0.8.9-fix | pc.cast overflow / pa.field nullable | `safe=False` + expected +1 byte |
| v0.8.9-fix2 | stale test (uint64) | rename + assertion update |
| v0.8.10-pre | `[0.8.10-pre]` vs `[Unreleased]` | rename to `[Unreleased]` |
| v0.8.11a | test substring (5* in 1.5*) / CHANGELOG skip | specific markers |
| **v0.8.11a-fix1** | **stale test in другой файл** (pandas_chain) | **grep before rename** |

**Связанные:** §13.4/128, /129, /139, /140, /141, /142, /143.

'''
    s = s.replace(marker, episode + marker, 1)
    p.write_text(s)
    print("  OK: §13.4/144")


# ============ main ============

def main():
    print("=== v0.8.11a-fix2 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    grep_stale_guard_names()

    fix_pingis_test()
    append_decisions_144()

    # Verify: grep now shows only intentional references
    print("\n=== post-check: grep 'pandas_chain' ===")
    r = subprocess.run(
        ["grep", "-rn", "pandas_chain", "tests/"],
        capture_output=True, text=True,
    )
    lines = [l for l in r.stdout.strip().splitlines() if l]
    print(f"  {len(lines)} reference(s) remaining (expected: 1 intentional):")
    for l in lines:
        print(f"    {l}")

    print("\n=== pytest: targeted (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pingis_safety.py::test_dedup_arrow_native_precheck_raises",
         "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** TARGETED TEST STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: dedup + pingis + guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pingis_safety.py",
         "tests/test_dedup_arrow_native.py",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_changelog.py",
         "tests/test_arrow_call_memory_guard.py",
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
        "v0.8.11a: Arrow-native dedup (P-D1 close) — Path C "
        "(dictionary_encode + group_by + is_in), 0.29× peak vs pandas; "
        "pandas removed from ops hot path; AST guard no_pandas_in_ops; "
        "§13.4/143-144 (test substring vs token; cross-file guard rename)"])

    print("\n==> v0.8.11a-fix2 applied. Amended WIP commit.")
    print("NEXT: v0.8.11b (strict join conditional).")


if __name__ == "__main__":
    main()