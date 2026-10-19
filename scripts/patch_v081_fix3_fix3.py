#!/usr/bin/env python3
"""v0.8.1-fix3-fix3: CHANGELOG — вставить [0.8.0] и [0.8.1] (idempotent).

Причина: patch_v080_timeseries_pca::patch_changelog не вставил [0.8.0]
(возможно, regex/pattern mismatch — прошёл WARNING), и
patch_v081_fix3_apply::patch_changelog упал на anchor `[0.8.0]`.

Стратегия: не полагаться на `[0.8.0]` anchor. Используем `## [0.7.7]`
(последний, точно есть) — insert both blocks BEFORE it, каждый
idempotent (skip if already present).
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


# ============ Entries (same as v0.8.1 fix3_apply) ============

ENTRY_081 = """## [0.8.1] — 2026-10-19

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
  3 XOR-shift, ~0.3 ms/100k.

### Критично для корректности
- **`types_mapper=Int64Dtype` в `arr.to_pandas()`** (numeric): без него
  `int64` с nulls → `float64` (NaN), без nulls → `int64` → разные hash
  для одного значения в разных батчах → нарушение partition-disjoint.
- **Avalanche**: `test_avalanche_spreads_low_bits`,
  `test_avalanche_seed_spreads_low_bits`.

### Несовместимость с pre-v0.8.1
- Hash values изменились. Partitions ephemeral — не проблема.

### Тесты
- +20 в `test_hash_v081.py`.
- `test_shuffle.py::test_stable_hash_across_batches` — убран лишний
  `assert a_parts != b_parts` (partition-disjoint ≠ different partitions).

### Ожидание
- int64/float64 hash: 19-40× speedup.
- int64 shuffle 1M: 0.61s → **~0.22s** (2.76×).
- string: без изменений.
- 500 GB / 50 B / int64 keys: ~100 min → **~40 min**.

"""


ENTRY_080 = """## [0.8.0] — 2026-10-19

### Добавлено — time-series
- `CadwDataFrame.expanding(agg, min_periods=1, columns=None)` — кумулятивное
  окно. Aggs: `sum`/`mean`/`min`/`max`/`count`. O(1) state (scalar
  accumulators; min/max — текущий min/max, без deque — cumulative min
  монотонен). Nulls skip; null row → null output.
  - Output: `{col}_expanding_{agg}`.
- `CadwDataFrame.ewm(alpha/span/halflife/com, adjust=True, min_periods=0,
  columns=None)` — экспоненциально-взвешенное среднее.
  - Ровно один из `alpha`/`span`/`halflife`/`com` (pandas-совместимые
    формулы).
  - `adjust=True` (pandas default): `(s_num, s_den)`, `y = s_num/s_den`.
  - `adjust=False`: `(ewm,)`, `y = (1-α)·y_prev + α·x`.
  - Nulls: skip (state preserved), null row → null output
    (= `ignore_na=True` семантика).
  - Output: `{col}_ewm`.

### Добавлено — PCA single-pass
- `CovarianceMeanOp(CovarianceOp)` — тот же state, `finalize` возвращает
  `{cov, mean, n}`. `fit_pca` → **2× I/O reduction**.
- `CovarianceOp` (bare matrix) не меняется — backward compat.

### Ограничения (v0.8.0, §11.5)
- `expanding`/`ewm` не поддерживают `on=col` (сортировать через
  `.sort_values(col)` перед вызовом).
- `ewm(ignore_na=False)` — не реализовано.
- `ewm(bias=True)` — не реализовано (только `adjust=True/False`).

### Тесты
- +26 в `test_expanding_ewm.py`.
- +3: `CovarianceMeanOp` values/estimate + `fit_pca` regression.

### Ожидаемые результаты
- fast: **~927 passed, 8 deselected**.
- slow: **8 passed**.
- PCA 200k × 10-d: 2× speed (1 pass вместо 2).

"""


def insert_if_missing(s: str, header: str, entry: str, anchor: str) -> str:
    """Insert entry before anchor if header not present."""
    if header in s:
        print(f"  {header}: already present — skip")
        return s
    if anchor not in s:
        raise SystemExit(f"anchor {anchor!r} not found in CHANGELOG")
    s = s.replace(anchor, entry + anchor, 1)
    print(f"  {header}: inserted before {anchor}")
    return s


def patch_changelog():
    print("\n=== CHANGELOG.md: insert [0.8.0] and [0.8.1] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()

    # Diagnostic
    print("\n  --- state before ---")
    for line in s.splitlines()[:80]:
        if line.startswith("#") or line.startswith("## ["):
            print(f"    {line}")

    # Find last existing anchor: prefer [0.7.7], fall back to [0.7.6], [0.7.5]...
    anchors = ["## [0.7.7]", "## [0.7.6]", "## [0.7.5]",
               "## [0.7.4b]", "## [0.7.4a]", "## [0.7.3]",
               "## [0.7.2]", "## [0.7.1]", "## [0.7.0a]"]
    anchor = None
    for a in anchors:
        if a in s:
            anchor = a
            break
    if anchor is None:
        raise SystemExit(
            "no anchor found (checked [0.7.7]...[0.7.0a]). "
            "Manual inspect CHANGELOG.md — возможно, файл повреждён."
        )
    print(f"\n  using anchor: {anchor!r}")

    # Insert 0.8.0 first (before anchor), then 0.8.1 (before 0.8.0)
    s = insert_if_missing(s, "## [0.8.0]", ENTRY_080, anchor)
    s = insert_if_missing(s, "## [0.8.1]", ENTRY_081, "## [0.8.0]")

    p.write_text(s)

    # Verify
    s2 = p.read_text()
    assert "## [0.8.1]" in s2 and "## [0.8.0]" in s2
    print("  OK: both entries present")


def verify():
    print("\n=== Verify CHANGELOG structure ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    versions = re.findall(r"^## \[(\d[^\]]*)\]", s, flags=re.MULTILINE)
    print(f"  versions ({len(versions)}): {versions[:10]}")
    # Check ordering: 0.8.1 before 0.8.0 before 0.7.7
    i81 = s.find("## [0.8.1]")
    i80 = s.find("## [0.8.0]")
    i77 = s.find("## [0.7.7]")
    assert i81 >= 0 and i80 >= 0 and i77 >= 0, "missing version"
    assert i81 < i80 < i77, (
        f"ordering broken: [0.8.1]@{i81}, [0.8.0]@{i80}, [0.7.7]@{i77}"
    )
    print("  OK: ordering [0.8.1] < [0.8.0] < [0.7.7]")


def main():
    print("=== v0.8.1-fix3-fix3 ===")
    patch_changelog()
    verify()

    print("\n=== pytest tests/test_shuffle.py -q (verify test fix from prior) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shuffle.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.1-fix3-fix3 (test red)"], check=False)
        sys.exit(1)

    print("\n=== bench: v0.8.1-fix3 (int64 + string) ===")
    sh(["uv", "run", "python", "scripts/bench_hash_v081.py"], check=False)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.1-fix3-fix3: CHANGELOG [0.8.0] + [0.8.1] (был пропущен при v0.8.0)"])

    print("\n==> v0.8.1-fix3-fix3 complete.")


if __name__ == "__main__":
    main()