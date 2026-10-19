#!/usr/bin/env python3
"""v0.8.7-fix: (a) date32 input cast; (b) test data null-type.

BUG 1 (real, code): `_to_unix_seconds` для `pa.types.is_date32` делает
`pc.cast(arr, pa.int64())` — pyarrow 26 НЕ поддерживает
(`ArrowNotImplementedError: Unsupported cast from date32[day] to int64`).
Но `pc.cast(date32, int32)` работает (same bit-width). Fix:
date32 → int32 → int64 → *86400 → int64 seconds. Similarly for
`date64` (int64 ms since epoch; cast works direct, /1000 → seconds).

Also proactive for OUTPUT `_bucket_to_arrow`: `int64 → date32` — via
int32 intermediate (int64 → int32 → date32), and `int64 → date64`
direct cast (int64 ms). Both tested.

BUG 2 (test-authoring): `test_resample_all_null_bucket_skipped` uses
`{"v": [None, None, None]}` — pyarrow infers `null` type, NOT float64.
`resample` correctly rejects null type as non-numeric (honest: null
type has no aggregation semantics). Fix test: explicit
`pa.array([None]*3, type=pa.float64())`. Add `test_resample_null_typed_
agg_col_raises` — verifies honest rejection of null-typed columns.

Run: uv run python scripts/patch_v087_fix.py
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


# ============ Fix 1: _to_unix_seconds for date32/date64 ============

def fix_to_unix_seconds():
    print("\n=== cadw/ops/resample.py: _to_unix_seconds ===")
    p = Path("cadw/ops/resample.py")
    s = p.read_text()

    old = '''def _to_unix_seconds(arr: pa.Array) -> pa.Array:
    """Cast timestamp/date/int to int64 unix seconds. Nulls preserved."""
    t = arr.type
    if pa.types.is_timestamp(t):
        factor = _TS_UNIT_FACTOR[t.unit]
        int_arr = pc.cast(arr, pa.int64())
        return pc.cast(
            pc.floor(pc.divide(int_arr, pa.scalar(factor, type=pa.int64()))),
            pa.int64(),
        )
    if pa.types.is_date(t):
        int_arr = pc.cast(arr, pa.int64())
        return pc.multiply(int_arr, pa.scalar(86400, type=pa.int64()))
    if pa.types.is_uint64(t):
        raise TypeError(
            "resample: on column uint64 not supported (overflow risk); "
            "cast to int64 or timestamp"
        )
    if pa.types.is_integer(t):
        return pc.cast(arr, pa.int64())
    raise TypeError(
        f"resample: on column type {t} not supported; expected timestamp, "
        f"date, or integer"
    )'''

    new = '''def _to_unix_seconds(arr: pa.Array) -> pa.Array:
    """Cast timestamp/date/int to int64 unix seconds. Nulls preserved.

    v0.8.7-fix: pyarrow 26 does NOT implement `pc.cast(date32, int64)`
    (`Unsupported cast from date32[day] to int64`). But `cast(date32,
    int32)` works (same bit-width reinterpretation). Route via int32.

    date32 = int32 days since epoch. date64 = int64 ms since epoch.
    """
    t = arr.type
    if pa.types.is_timestamp(t):
        factor = _TS_UNIT_FACTOR[t.unit]
        int_arr = pc.cast(arr, pa.int64())
        return pc.cast(
            pc.floor(pc.divide(int_arr, pa.scalar(factor, type=pa.int64()))),
            pa.int64(),
        )
    if pa.types.is_date32(t):
        # v0.8.7-fix: cast via int32 (pyarrow has no date32→int64 kernel).
        int32_arr = pc.cast(arr, pa.int32())
        int64_arr = pc.cast(int32_arr, pa.int64())
        return pc.multiply(int64_arr, pa.scalar(86400, type=pa.int64()))
    if pa.types.is_date64(t):
        # date64 = int64 ms since epoch; cast to int64 supported.
        int64_ms = pc.cast(arr, pa.int64())
        return pc.cast(
            pc.floor(pc.divide(int64_ms, pa.scalar(1000, type=pa.int64()))),
            pa.int64(),
        )
    if pa.types.is_uint64(t):
        raise TypeError(
            "resample: on column uint64 not supported (overflow risk); "
            "cast to int64 or timestamp"
        )
    if pa.types.is_integer(t):
        return pc.cast(arr, pa.int64())
    raise TypeError(
        f"resample: on column type {t} not supported; expected timestamp, "
        f"date32/date64, or integer"
    )'''

    s = replace_line_block(s, old, new, tag="to-unix-seconds")
    p.write_text(s)
    compile_check(p)
    print("  OK: date32 → int32 → int64 (*86400); date64 → int64/1000")


# ============ Fix 2: _bucket_to_arrow for date32/date64 output ============

def fix_bucket_to_arrow():
    print("\n=== cadw/ops/resample.py: _bucket_to_arrow (output) ===")
    p = Path("cadw/ops/resample.py")
    s = p.read_text()

    old = '''def _bucket_to_arrow(
    bucket_seconds: list[int], on_type: pa.DataType,
) -> pa.Array:
    """Convert list of int64 bucket-seconds → Arrow array of `on_type`."""
    int_arr = pa.array(bucket_seconds, type=pa.int64())
    if pa.types.is_timestamp(on_type):
        factor = _TS_UNIT_FACTOR[on_type.unit]
        scaled = pc.multiply(int_arr, pa.scalar(factor, type=pa.int64()))
        return pc.cast(scaled, on_type)
    if pa.types.is_date(on_type):
        days = pc.cast(
            pc.floor(pc.divide(int_arr, pa.scalar(86400, type=pa.int64()))),
            pa.int64(),
        )
        return pc.cast(days, on_type)
    return pc.cast(int_arr, on_type)'''

    new = '''def _bucket_to_arrow(
    bucket_seconds: list[int], on_type: pa.DataType,
) -> pa.Array:
    """Convert list of int64 bucket-seconds → Arrow array of `on_type`.

    v0.8.7-fix: `date32` output via int32 intermediate (same rationale
    as input — no int64→date32 kernel in pyarrow 26; int64→int32 works).
    `date64` direct (int64 ms).
    """
    int_arr = pa.array(bucket_seconds, type=pa.int64())
    if pa.types.is_timestamp(on_type):
        factor = _TS_UNIT_FACTOR[on_type.unit]
        scaled = pc.multiply(int_arr, pa.scalar(factor, type=pa.int64()))
        return pc.cast(scaled, on_type)
    if pa.types.is_date32(on_type):
        days = pc.floor(pc.divide(int_arr, pa.scalar(86400, type=pa.int64())))
        int32_days = pc.cast(days, pa.int32())
        return pc.cast(int32_days, pa.date32())
    if pa.types.is_date64(on_type):
        ms = pc.multiply(int_arr, pa.scalar(1000, type=pa.int64()))
        return pc.cast(ms, pa.date64())
    return pc.cast(int_arr, on_type)'''

    s = replace_line_block(s, old, new, tag="bucket-to-arrow")
    p.write_text(s)
    compile_check(p)
    print("  OK: date32 output via int32; date64 via int64*1000")


# ============ Fix 3: test data — explicit float64 ============

def fix_test_all_null():
    print("\n=== tests/test_resample.py: all_null_bucket_skipped ===")
    p = Path("tests/test_resample.py")
    s = p.read_text()

    old = '''def test_resample_all_null_bucket_skipped(tmp_path):
    df = make_df(tmp_path, {
        "ts": [0, 5, 6],
        "v": [None, None, None],
    })
    res = df.resample(5, "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    assert t.num_rows == 0'''

    new = '''def test_resample_all_null_bucket_skipped(tmp_path):
    """Bucket with all-null values → skipped (not emitted as None-row).

    Note: `v` must be explicit `pa.float64()`. Without type, pyarrow infers
    `null` type (from [None, None, None]) — resample correctly rejects it
    as non-numeric (see test_resample_null_typed_agg_col_raises).
    """
    df = make_df(tmp_path, {
        "ts": [0, 5, 6],
        "v": pa.array([None, None, None], type=pa.float64()),
    })
    res = df.resample(5, "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    assert t.num_rows == 0'''

    s = replace_line_block(s, old, new, tag="test-all-null")

    # Add null-typed rejection test
    if "def test_resample_null_typed_agg_col_raises" not in s:
        extra = '''

def test_resample_null_typed_agg_col_raises(tmp_path):
    """Column with `null` type (not float64) — TypeError (non-numeric).

    `pa.table({"v": [None, None]})` infers `null` type — has no
    aggregation semantics. For "all-null float64" use `pa.array([...],
    type=pa.float64())`.
    """
    df = make_df(tmp_path, {
        "ts": [0, 5],
        "v": [None, None],  # null type
    })
    with pytest.raises(TypeError, match="numeric"):
        df.resample(5, "sum", on="ts", columns=["v"])


def test_resample_date64(tmp_path):
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
    assert t.schema.field("ts").type == pa.date64()
'''
        s = s.rstrip() + extra + "\n"

    p.write_text(s)
    compile_check(p)
    print("  OK: test data explicit float64 + null-typed test + date64 test")


# ============ Fix 4: DECISIONS §13.4/133,134 ============

def patch_decisions():
    print("\n=== DECISIONS.md: §13.4/133 + 134 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 133" in s:
        print("  already present — skip")
        return

    marker = "### 13.5. Открытые вопросы"
    episode = '''**Эпизод 133 (pyarrow 26: date32 → int64 cast not implemented).**
`_to_unix_seconds` для `pa.types.is_date32` делал `pc.cast(arr,
pa.int64())` — **ArrowNotImplementedError: Unsupported cast from
date32[day] to int64**. `date32 = int32 days-since-epoch` — по
представлению это int32, но pyarrow не реализовал `cast(date32, int64)`
(только `date32 → int32` через same bit-width reinterpretation).

**Fix:** route через int32: `date32 → int32 → int64 → *86400 → int64
seconds`. Similarly для output (`int64 → int32 → date32`). date64
(int64 ms) — direct cast работает.

**Урок (API contract):** cast matrix pyarrow **неполный**. То, что
`date32` физически int32, не гарантирует `cast(date32, int64)`. При
cross-type cast — сначала `cast` в same-bit-width (int32/int64),
потом scale. **Правило:** для `date*` типов — через int32/date32 или
int64/date64 (matching bit-width), не jump через оба уровня.

**Эпизод 134 (`null`-typed column в test data).** `test_resample_all_
null_bucket_skipped` использовал `{"v": [None, None, None]}` — pyarrow
вывел `null` type (не `float64`). `resample` корректно отвергает
`null` как non-numeric (honest: null type has no aggregation semantics).
Test authoring: для "all-null float64" явно `pa.array([None]*3,
type=pa.float64())`.

Добавлен `test_resample_null_typed_agg_col_raises` — верифицирует
честное отклонение `null`-typed columns. Documented §11.5.

**Урок (test-authoring):** при `pa.table({...})` из Python lists —
pyarrow **infers** тип. `[None, None]` → `null`, `[None, 1.0]` →
`float64`, `[1, 2]` → `int64`. Если нужен конкретный тип — explicit
`pa.array(..., type=...)`. Тот же класс, что v0.8.4-fix2 (§13.4/128 —
"expected value derived, not authored"), но в **data setup**, не в
expected.

**Связанные:** §13.4/99 (numpy fallback hidden by native), §13.4/107
(pandas low-bit), §13.4/128 (test expectations). Все — "hidden by
green tests": (99) numpy fallback не exercised; (107) low-bit test
missing; (128) expected values wrong; (134) input type different from
intended.

**Следующее:** v0.8.8 per-group (prototype-first).

'''
    s = s.replace(marker, episode + marker, 1)
    p.write_text(s)
    print("  OK: §13.4/133 + 134")


# ============ main ============

def main():
    print("=== v0.8.7-fix ===")

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

    fix_to_unix_seconds()
    fix_bucket_to_arrow()
    fix_test_all_null()
    patch_decisions()

    print("\n=== pytest tests/test_resample.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_resample.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.7-fix (test red)"], check=False)
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
        "v0.8.7-fix: date32 cast via int32 (pyarrow 26 no date32→int64); "
        "test data explicit float64 + null-typed rejection"])

    print("\n==> v0.8.7-fix applied. Готов squash → tag v0.8.7.")


if __name__ == "__main__":
    main()