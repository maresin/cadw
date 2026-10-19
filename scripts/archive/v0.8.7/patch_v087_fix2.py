#!/usr/bin/env python3
"""v0.8.7-fix2: date64 test → unit-level + parquet-limitation test.

DIAGNOSIS: `test_resample_date64` failed because:
1. `date64` = day-granularity type; `datetime(2020,1,1,0,0,2)` truncates
   to `datetime(2020,1,1,0,0,0)` inside `pa.array(..., type=pa.date64())`.
   Sub-day buckets (`5s`) на date64 — semantics broken (all → 1 day-bucket).
2. **Parquet has no ms-date logical type** (Date = int32 days).
   `pq.write_table(date64)` → `pq.read_table` → `date32[day]`. Output
   showed `ts: date32[day]` — confirms silent downcast in `make_df`.

Code is correct (date64 branch handles via int64/1000). Test can't go
through parquet roundtrip. Replace with unit-level helper tests.

Fix:
- `test_to_unix_seconds_date64_unit` — direct `_to_unix_seconds(date64)`
- `test_bucket_to_arrow_date64_unit` — direct `_bucket_to_arrow(..., date64)`
- `test_to_unix_seconds_date32_unit` — same for date32
- `test_resample_date64_day_freq_through_parquet_limitation` —
  documents that real usage date64→date32 (day-level freq works)
- §11.5: date64 through parquet → date32
- §13.4/135: parquet ms-date limitation

Run: uv run python scripts/patch_v087_fix2.py
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


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if len(starts) == 0:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ Fix 1: replace test_resample_date64 ============

def fix_date64_test():
    print("\n=== tests/test_resample.py: date64 → unit + limitation ===")
    p = Path("tests/test_resample.py")
    s = p.read_text()

    old = '''def test_resample_date64(tmp_path):
    """date64 (int64 ms since epoch) on column."""
    import datetime as dt
    dates = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 1, 0, 0, 2),
        dt.datetime(2020, 1, 1, 0, 0, 7),
    ]
    df = make_df(tmp_path, {
        "ts": pa.array(dates, type=pa.date64()),
        "v": [1.0, 2.0, 3.0],
    })
    res = df.resample("5s", "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    assert t.num_rows == 2
    assert t.schema.field("ts").type == pa.date64()'''

    new = '''def test_to_unix_seconds_date64_unit():
    """Unit test for date64 branch in _to_unix_seconds.

    Date64 = int64 ms-since-epoch, BUT day-granularity semantics. Cannot
    test through parquet roundtrip (Parquet has no ms-date type;
    date64 → date32 on read, §11.5/§13.4/135). Test helper directly.
    """
    import datetime as dt
    from cadw.ops.resample import _to_unix_seconds

    arr = pa.array(
        [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)],
        type=pa.date64(),
    )
    result = _to_unix_seconds(arr)
    # 2020-01-01 00:00:00 UTC = 1577836800
    # 2020-01-02 00:00:00 UTC = 1577923200
    assert result.to_pylist() == [1577836800, 1577923200]


def test_bucket_to_arrow_date64_unit():
    """Unit test for date64 branch in _bucket_to_arrow."""
    import datetime as dt
    from cadw.ops.resample import _bucket_to_arrow

    seconds = [1577836800, 1577923200]
    arr = _bucket_to_arrow(seconds, pa.date64())
    assert arr.type == pa.date64()
    assert arr.to_pylist() == [dt.date(2020, 1, 1), dt.date(2020, 1, 2)]


def test_to_unix_seconds_date32_unit():
    """Unit test for date32 branch (via int32, §13.4/133)."""
    import datetime as dt
    from cadw.ops.resample import _to_unix_seconds

    arr = pa.array(
        [dt.date(2020, 1, 1), dt.date(2020, 1, 2)],
        type=pa.date32(),
    )
    result = _to_unix_seconds(arr)
    assert result.to_pylist() == [1577836800, 1577923200]


def test_resample_date64_day_freq_through_parquet_limitation(tmp_path):
    """Parquet has no ms-date logical type → date64 becomes date32 on read.

    Real-world `resample(on=date_col)` sees date32, not date64 (after
    parquet roundtrip). Same day-granularity semantics — day-level
    `freq` (e.g. '1D') works correctly. See §11.5, §13.4/135.
    """
    import datetime as dt
    dates = [
        dt.date(2020, 1, 1),
        dt.date(2020, 1, 1),
        dt.date(2020, 1, 3),
    ]
    df = make_df(tmp_path, {
        "ts": pa.array(dates, type=pa.date32()),  # date32 in practice
        "v": [1.0, 2.0, 3.0],
    })
    res = df.resample("1D", "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_sum"]) for r in t.to_pylist()]
    assert rows == [
        (dt.date(2020, 1, 1), 3.0),
        (dt.date(2020, 1, 3), 3.0),
    ]'''

    s = replace_line_block(s, old, new, tag="date64-test")
    p.write_text(s)
    compile_check(p)
    print("  OK: 4 tests (2 unit date64, 1 unit date32, 1 parquet-limitation)")


# ============ Fix 2: DECISIONS §11.5 + §13.4/135 ============

def patch_decisions():
    print("\n=== DECISIONS.md: §11.5 + §13.4/135 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "| `resample` date64 through parquet → date32 (v0.8.7)" not in s:
        s = s.replace(
            "| `resample` freq — int seconds / '5s'|'1min'|'1h'|'1D' only (v0.8.7) |",
            "| `resample` date64 through parquet → date32 (v0.8.7) | `resample` | Parquet has no ms-date type; date64 read back as date32. Same day-granularity semantics; use `freq >= '1D'`. §13.4/135 |\n"
            "| `resample` freq — int seconds / '5s'|'1min'|'1h'|'1D' only (v0.8.7) |",
            1,
        )
        print("  OK: §11.5")

    if "Эпизод 135" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 135 (date64 через parquet → date32: silent downcast).**
`test_resample_date64` использовал `pa.array([datetime(2020,1,1,0,0,2),
...], type=pa.date64())` — ожидал 2 buckets по 5s. Failure: 1 bucket.
Output schema — `ts: date32[day]`, не `date64`. Два независимых
проблемы:

**Проблема 1 (test data, semantics).** `date64` — day-granularity тип
семантически. `datetime(2020,1,1,0,0,2)` → pyarrow **truncates** до
midnight (2020-01-01). Date64 = int64 ms, но **всегда** midnight-aligned.
Sub-day bucket (`freq='5s'`) на date64 бессмыслен: 5s-бакет ≈
86400*N-бакет ≈ bucket-per-day.

**Проблема 2 (parquet roundtrip).** `pq.write_table(date64)` →
`pq.read_table` → `date32[day]`. Parquet имеет **только Date logical
type** = int32 days; ms-date не представлен. `make_df` (from_parquet)
видит on_type = date32. Так и произошло.

**Fix (v0.8.7-fix2):**
1. Убрать `test_resample_date64` (parquet roundtrip невозможен).
2. Unit-тесты `_to_unix_seconds(date64)` / `_bucket_to_arrow(...,
   date64)` / `_to_unix_seconds(date32)` — direct on helpers.
3. `test_resample_date64_day_freq_through_parquet_limitation` —
   документирует, что real usage date64→date32 через parquet, работает
   корректно для `freq >= '1D'`.
4. §11.5: "date64 through parquet → date32; use freq >= '1D'".

**Урок (аналогичен §13.4/134 «input type different from intended»):**
`pa.date64` — **legacy/compat type** в pyarrow. Parquet не поддерживает
(нет logical type для ms-date). Arrow IPC сохраняет, но
`write_parquet` → `read_parquet` = date32. Правило: **не тестировать
date64 через parquet**; для date64-specific — unit-level helpers или
in-memory Dataset (без parquet).

**Ещё урок (semantics):** date64 = day, не sub-second. Если нужна
sub-second precision — только timestamp, не date. `resample(on=date_col)`
с `freq='5s'` — баг пользователя (не CADW): все → 1 bucket/день.

**Связанные:** §13.4/133 (pyarrow cast matrix incomplete: no
date32→int64), §13.4/134 (null vs float64 inference). Класс «hidden by
green tests»: (133) date32→int64 nonexistent — caught; (134) null→
float64 mismatch — caught; (135) date64→date32 parquet — caught (и
выявил test-authoring error, не code).

**Следующее:** v0.8.8 per-group (prototype-first).

'''
        s = s.replace(marker, episode + marker, 1)
        print("  OK: §13.4/135")

    p.write_text(s)


# ============ main ============

def main():
    print("=== v0.8.7-fix2: date64 test → unit-level ===")

    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    fix_date64_test()
    patch_decisions()

    print("\n=== pytest tests/test_resample.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_resample.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.7-fix2 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest time-series regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_rolling.py",
         "tests/test_rolling_on_col.py",
         "tests/test_shift_diff.py",
         "tests/test_shift_on_col.py",
         "tests/test_expanding_ewm.py",
         "tests/test_resample.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.7-fix2: date64 test → unit-level (parquet has no ms-date; "
        "date64 → date32 roundtrip). §13.4/135"])

    print("\n==> v0.8.7-fix2 applied. Готов squash → tag v0.8.7.")


if __name__ == "__main__":
    main()