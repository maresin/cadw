#!/usr/bin/env python3
"""v0.8.1-fix3-fix2: убрать устаревший assert в test_stable_hash_across_batches;
переписать [0.8.1] CHANGELOG (полная замена, не pattern match).

Причина: partition-disjoint — «каждый ключ в одной партиции», НЕ
«разные ключи в разных». С новым hash (pandas SipHash + avalanche)
A и B могут коллизировать в h&3 — это корректно.

Также fix3-apply упал до CHANGELOG update — полная перезапись блока
[0.8.1] вместо pattern match (формат изменился).
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
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:25]:
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


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


# ============ 1. Fix test_stable_hash_across_batches ============

def fix_stable_hash_test():
    print("\n=== tests/test_shuffle.py: test_stable_hash_across_batches ===")
    p = Path("tests/test_shuffle.py")
    s = p.read_text()

    old = '''def test_stable_hash_across_batches(tmp_path):
    """Одинаковые ключи из разных батчей → одна партиция."""
    # много A в разных батчах + немного B
    table = pa.table({
        "cat": ["A"] * 10 + ["B"] * 5,
        "val": list(range(15)),
    })
    paths = hash_shuffle(
        make_iter(table, 3), ["cat"], ["val"], 4, make_guard(), tmp_path, log=False,
    )
    partitions = collect_partitions(paths)
    a_parts = [i for i, rows in enumerate(partitions) if any(r["cat"] == "A" for r in rows)]
    b_parts = [i for i, rows in enumerate(partitions) if any(r["cat"] == "B" for r in rows)]
    assert len(a_parts) == 1, f"A split across {a_parts}"
    assert len(b_parts) == 1, f"B split across {b_parts}"
    assert a_parts != b_parts'''

    new = '''def test_stable_hash_across_batches(tmp_path):
    """Одинаковые ключи из разных батчей → одна партиция.

    **Не проверяет**, что разные ключи попадают в разные партиции —
    это не invariant. Partition-disjoint = «все строки одного ключа в
    одной партиции»; коллизии h & 3 между разными ключами — норма
    (downstream различает по значению). Раньше xxhash+repr случайно
    разносил "A" и "B"; pandas SipHash + avalanche может их коллизировать
    в одну партицию — корректно.
    """
    table = pa.table({
        "cat": ["A"] * 10 + ["B"] * 5,
        "val": list(range(15)),
    })
    paths = hash_shuffle(
        make_iter(table, 3), ["cat"], ["val"], 4, make_guard(), tmp_path, log=False,
    )
    partitions = collect_partitions(paths)
    a_parts = [i for i, rows in enumerate(partitions) if any(r["cat"] == "A" for r in rows)]
    b_parts = [i for i, rows in enumerate(partitions) if any(r["cat"] == "B" for r in rows)]
    assert len(a_parts) == 1, f"A split across {a_parts}"
    assert len(b_parts) == 1, f"B split across {b_parts}"
    # NOTE: a_parts may equal b_parts (hash collision on h & 3) — valid.'''

    s = replace_exact(s, old, new, tag="stable-hash-test")
    p.write_text(s)
    compile_check(p)
    print("  OK: убран assert a_parts != b_parts")


# ============ 2. Rewrite [0.8.1] CHANGELOG (full block) ============

NEW_081 = """## [0.8.1] — 2026-10-19

### Изменено — performance
- **`compute_hashes`: pandas C-hash для numeric, Python для string**
  (v0.8.0 diagnostic: 32.5× speedup на int64).
  - **Numeric/boolean/timestamp/date**: `pd.util.hash_pandas_object`
    (C-level SipHash). Bench: int64 39.2→2.0 ms (**19.6×**),
    float64 39.2→1.0 ms (**39.2×**).
  - **String/large_string**: Python `repr().encode()` + xxhash.
    Причина: pandas `hash_pandas_object` для object dtype использует
    per-element Cython — 50 ms/100k, **22% медленнее** Python
    (§13.4/107). Без регрессии.
  - Fallback: Python loop для struct/list/dict/uint64.
  - Seed применяется через `h_np ^= np.uint64(seed)` **до** avalanche.

- **Avalanche (splitmix64) — всегда**, не только при seed. Причина:
  pandas `hash_pandas_object` для object dtype сдвигает hash `<< 1`
  (reserves 0 для NA), все string hashes чётные → `h & 3` использует
  только партиции `{0, 2}` → recursion split 2-way вместо 4-way →
  `MemoryLimitExceeded` при depth=5 (§13.4/107). Fix: 3 умножения +
  3 XOR-shift, ~0.3 ms/100k. Платим ~15% speedup за корректность.

### Критично для корректности
- **`types_mapper=Int64Dtype` в `arr.to_pandas()`** (numeric): без него
  `int64` с nulls → `float64` (NaN), без nulls → `int64` → **разные
  hash для одного значения в разных батчах** → нарушение
  partition-disjoint → молча wrong результаты.
  - Тест `test_int64_consistent_hash_across_null_presence` — critical
    regression guard.
- **Avalanche**: тесты `test_avalanche_spreads_low_bits` (P ∈ {2,4,8,16}
  uniform), `test_avalanche_seed_spreads_low_bits` (any seed).
- **`test_stable_hash_across_batches`** (v0.1.0) — убран
  `assert a_parts != b_parts`: partition-disjoint = «все строки ключа
  в одной партиции», коллизии разных ключей — норма.

### Несовместимость с pre-v0.8.1
- Hash values изменились. Partitions ephemeral — не проблема.
  Partition-disjoint invariant сохранён (same value → same hash, nulls
  → same fixed hash).

### Тесты
- +20 в `test_hash_v081.py`: cross-batch consistency (int32/int64/string
  с/без nulls), deterministic, seed, nulls, `pandas_path_used_for_int64`,
  `uint64_fallback`, partition-disjoint (int+string), avalanche
  (low-bit spread + seed + similar strings), groupby recursion
  end-to-end, dispatch (string→python, numeric→pandas).
- `test_shuffle.py::test_stable_hash_across_batches` — убран лишний
  assert (partition-disjoint ≠ different-partitions).

### Ожидание
- int64/float64 hash: 19-40× speedup.
- int64 shuffle 1M: 0.61s → **~0.22s** (2.76×).
- string: без изменений (pandas не быстрее).
- 500 GB / 50 B / int64 keys: ~100 min → **~40 min**.

"""


def rewrite_changelog_081():
    print("\n=== CHANGELOG.md: полная замена [0.8.1] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()

    # Найти блок [0.8.1] ... до [0.8.0]
    pattern = re.compile(
        r"## \[0\.8\.1\].*?(?=## \[0\.8\.0\])",
        re.DOTALL,
    )
    if pattern.search(s):
        s = pattern.sub(NEW_081, s, count=1)
        p.write_text(s)
        print("  OK: [0.8.1] заменён целиком")
        return

    # Fallback: [0.8.1] нет — вставить перед [0.8.0]
    if "## [0.8.0]" in s:
        s = s.replace("## [0.8.0]", NEW_081 + "## [0.8.0]", 1)
        p.write_text(s)
        print("  OK: [0.8.1] вставлен")
        return

    raise SystemExit("[0.8.0] anchor not found in CHANGELOG")


# ============ main ============

def main():
    print("=== v0.8.1-fix3-fix2 ===")
    fix_stable_hash_test()
    rewrite_changelog_081()

    print("\n=== pytest tests/test_hash_v081.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_hash_v081.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix3-fix2 (test red)"], check=False)
        sys.exit(1)

    print("\n=== regression: shuffle + mustfix_local_groupby ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_shuffle.py",
         "tests/test_mustfix_local_groupby.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix3-fix2 (shuffle regression)"], check=False)
        sys.exit(1)

    print("\n=== bench: v0.8.1-fix3 (int64 + string) ===")
    sh(["uv", "run", "python", "scripts/bench_hash_v081.py"], check=False)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.1-fix3-fix2: test_stable_hash_across_batches (убран лишний "
        "assert); CHANGELOG [0.8.1] полная замена (string→python, avalanche)"])

    print("\n==> v0.8.1-fix3-fix2 complete.")


if __name__ == "__main__":
    main()