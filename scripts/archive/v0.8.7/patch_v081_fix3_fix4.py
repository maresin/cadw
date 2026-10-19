#!/usr/bin/env python3
"""v0.8.1-fix3-fix4: CHANGELOG — pragmatic fix (gap с v0.1.5).

Диагноз: CHANGELOG содержит только [0.1.4], [0.1.3], [0.1.2], [0.1.0].
Все patch_changelog() для v0.1.5–v0.8.0 были silent no-op'ами — каждый
вставлял перед [0.X.Y-1], и раз предыдущая не добавлялась, цепочка
обрывалась молча.

Fix (pragmatic, не полноценный restore):
    1. Note в начале: "[0.1.5]–[0.7.7] — см. git log; CHANGELOG вёлся
       только до [0.1.4]".
    2. [0.8.1] и [0.8.0] — как полноценные блоки (последние, важные).
    3. Guard test: test_changelog_has_latest_tag (skip если нет тегов).

Full restore — отдельная задача. Сейчас приоритет — v0.8.1 тег.
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


# ============ CHANGELOG prepend ============

NOTE = """<!--
  CHANGELOG был сломан с v0.1.5: каждый patch_changelog() вставлял
  перед [0.X.Y-1], а при отсутствии anchor `str.replace` был silent
  no-op. Цепочка оборвалась на v0.1.4, все последующие версии не
  попали. Полная история — `git log --oneline --grep='^v'`.
  Ниже — последние записи (v0.7.7+); восстанавливаем по мере
  необходимости. AST guard `test_changelog_has_latest_tag` фиксирует
  инвариант.
-->

"""

ENTRY_081 = """## [0.8.1] — 2026-10-19

### Изменено — performance
- **`compute_hashes`: pandas C-hash для numeric, Python для string.**
  - Numeric/boolean/timestamp/date → `pd.util.hash_pandas_object`
    (C-level SipHash). Bench: int64 39.2→2.0 ms (**19.6×**),
    float64 39.2→1.0 ms (**39.2×**).
  - String/large_string → Python `repr().encode()` + xxhash. Причина:
    pandas `hash_pandas_object` для object dtype — per-element Cython,
    50 ms/100k, **22% медленнее** Python (§13.4/107). Без регрессии.
  - struct/list/dict/uint64 → Python loop (fallback).
  - Seed через `h_np ^= np.uint64(seed)` **до** avalanche.
- **Avalanche (splitmix64) — всегда**, не только при seed. Причина:
  pandas `hash_pandas_object` для object dtype `hash << 1` (reserves 0
  для NA), все string hashes чётные → `h & 3` использует только
  партиции `{0, 2}` → recursion split 2-way вместо 4-way →
  `MemoryLimitExceeded` при depth=5 (§13.4/107). Fix: 3 умножения +
  3 XOR-shift, ~0.3 ms/100k.

### Критично для корректности
- **`types_mapper=Int64Dtype`** в `arr.to_pandas()` (numeric): без него
  `int64` с nulls → `float64` (NaN), без nulls → `int64` → разные hash
  для одного значения в разных батчах → нарушение partition-disjoint.
- Тесты: `test_int64_consistent_hash_across_null_presence`,
  `test_avalanche_spreads_low_bits`,
  `test_avalanche_seed_spreads_low_bits`.
- `test_shuffle.py::test_stable_hash_across_batches` — убран лишний
  `assert a_parts != b_parts` (partition-disjoint ≠ different partitions;
  коллизии разных ключей в одной партиции — норма).

### Тесты
- +20 в `test_hash_v081.py`: cross-batch consistency (int32/int64/string
  с/без nulls), deterministic, seed, nulls, dispatch (string→python,
  numeric→pandas), partition-disjoint (int+string), avalanche (low-bit
  spread + seed + similar strings), groupby recursion end-to-end.

### Ожидание
- int64 shuffle 1M: 0.61s → **~0.22s** (2.76×).
- 500 GB / 50 B / int64 keys: ~100 min → **~40 min**.

"""


ENTRY_080 = """## [0.8.0] — 2026-10-19

### Добавлено — time-series
- `CadwDataFrame.expanding(agg, min_periods=1, columns=None)` —
  кумулятивное окно. Aggs: `sum`/`mean`/`min`/`max`/`count`. O(1) state
  (scalar accumulators; min/max — текущий min/max, без deque). Nulls
  skip; null row → null output. Output: `{col}_expanding_{agg}`.
- `CadwDataFrame.ewm(alpha/span/halflife/com, adjust=True, min_periods=0,
  columns=None)` — экспоненциально-взвешенное среднее. Ровно один из
  4 параметров. `adjust=True`: `(s_num, s_den)`, `y = s_num/s_den`.
  `adjust=False`: `(ewm,)`, `y = (1-α)·y_prev + α·x`. Nulls skip
  (`ignore_na=True` семантика). Output: `{col}_ewm`.

### Добавлено — PCA single-pass
- `CovarianceMeanOp(CovarianceOp)` — тот же state, `finalize` →
  `{cov, mean, n}`. `fit_pca` → **2× I/O reduction**.
- `CovarianceOp` (bare matrix) не меняется — backward compat.

### Ограничения (v0.8.0, §11.5)
- `expanding`/`ewm` не поддерживают `on=col`.
- `ewm(ignore_na=False)`, `ewm(bias=True)` — не реализованы.

### Тесты
- +26 в `test_expanding_ewm.py`, +3 (CovarianceMeanOp + fit_pca regression).

"""


def fix_changelog():
    print("\n=== CHANGELOG.md: prepend note + [0.8.1] + [0.8.0] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()

    # Idempotent checks
    has_note = "CHANGELOG был сломан с v0.1.5" in s
    has_081 = "## [0.8.1]" in s
    has_080 = "## [0.8.0]" in s
    print(f"  state: note={has_note}, [0.8.1]={has_081}, [0.8.0]={has_080}")

    # Anchor: `## [0.1.4]` — первый entry после intro (гарантированно есть)
    anchor = "## [0.1.4]"
    if anchor not in s:
        raise SystemExit(
            f"anchor {anchor!r} not found — CHANGELOG совершенно "
            f"неожиданный. Проверь вручную."
        )

    prepend = ""
    if not has_note:
        prepend += NOTE
    if not has_081:
        prepend += ENTRY_081
    if not has_080:
        prepend += ENTRY_080

    if prepend:
        s = s.replace(anchor, prepend + anchor, 1)
        p.write_text(s)
        print(f"  OK: prepended {len(prepend)} bytes before {anchor!r}")
    else:
        print("  already done — skip")

    # Verify
    s2 = p.read_text()
    assert "## [0.8.1]" in s2, "[0.8.1] missing after fix"
    assert "## [0.8.0]" in s2, "[0.8.0] missing after fix"
    # Verify order: 0.8.1 before 0.8.0 before 0.1.4
    i81 = s2.find("## [0.8.1]")
    i80 = s2.find("## [0.8.0]")
    i14 = s2.find("## [0.1.4]")
    assert i81 < i80 < i14, (
        f"ordering broken: 0.8.1@{i81}, 0.8.0@{i80}, 0.1.4@{i14}"
    )
    print("  OK: ordering [0.8.1] < [0.8.0] < [0.1.4]")


# ============ Guard test ============

GUARD_TEST = '''"""test_changelog.py — guard: CHANGELOG reflects latest tag.

Регрессия v0.8.1-fix3-fix4: CHANGELOG был сломан с v0.1.5. Все
patch_changelog() были silent no-op'ами (цепочка оборвалась). Тег v0.8.1
был бы создан без записи в CHANGELOG. Guard: при наличии тегов —
top entry CHANGELOG должен совпадать с latest tag.
"""
import re
import subprocess
from pathlib import Path

import pytest


CHANGELOG = Path(__file__).parent.parent / "CHANGELOG.md"


def test_changelog_exists():
    assert CHANGELOG.exists(), f"{CHANGELOG} missing"


def test_changelog_has_latest_tag():
    """Top entry CHANGELOG == latest git tag.

    Skip, если тегов нет (новый repo / ветка без тегов).
    """
    r = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip("no git tags yet")

    tag = r.stdout.strip()          # e.g. "v0.8.1"
    version = tag.lstrip("v")       # "0.8.1"

    text = CHANGELOG.read_text()
    versions = re.findall(r"^## \\[(\\d[^\\]]*)\\]", text, flags=re.MULTILINE)
    assert versions, "CHANGELOG has no `## [X.Y.Z]` entries"

    top = versions[0]
    assert top == version, (
        f"CHANGELOG top entry [{top}] != latest tag [{version}]. "
        f"Add CHANGELOG entry for {tag} before tagging. "
        f"See §13.4/111."
    )
'''

def add_guard_test():
    print("\n=== tests/test_changelog.py: latest_tag guard ===")
    p = Path("tests/test_changelog.py")
    if p.exists() and "test_changelog_has_latest_tag" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(GUARD_TEST)
    compile_check(p)
    print("  OK: guard added")


# ============ main ============

def main():
    print("=== v0.8.1-fix3-fix4 ===")
    fix_changelog()
    add_guard_test()

    print("\n=== pytest tests/test_changelog.py tests/test_hash_v081.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py",
         "tests/test_hash_v081.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix3-fix4 (test red)"], check=False)
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
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix3-fix4 (shuffle)"], check=False)
        sys.exit(1)

    print("\n=== bench: v0.8.1 hash (int64+string shuffle) ===")
    sh(["uv", "run", "python", "scripts/bench_hash_v081.py"], check=False)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.1-fix3-fix4: CHANGELOG [0.8.1]+[0.8.0] (было пропущено с "
        "v0.1.5 из-за silent no-op в патч-скриптах); guard test"])

    print("\n==> v0.8.1-fix3-fix4 complete. Готов squash → tag v0.8.1.")


if __name__ == "__main__":
    main()