#!/usr/bin/env python3
"""v0.9.0a-fix1: 3 first-run bugs (all mine, not production-code).

ROUND 1-3 of the "first-run bugs" cluster for v0.9.0a — different from
§13.4/139-144 ("fix creates new bug"), closer to §13.4/145 (discovery).
New code, first run, three partial-update / assumption bugs:

  1. PredicateFilter._OPS validation (8 test failures).
     v0.9.0a extended mask_for_batch with is_null/not_null/contains,
     but __init__ still validates against old _OPS dict → ValueError.
     Class: partial update within one class (handler updated, validation
     not).

  2. test_dt_null_timestamp_propagates NaT conversion (1 test failure).
     `ts.tolist()` yields [Timestamp, NaT, Timestamp].
     `pa.table({"ts": [..., NaT, ...]})` → "NaTType does not support
     utcoffset". Test-authoring: not using Arrow-native pa.array(ts).
     Class: §13.4/134-like (input conversion not Arrow-safe).

  3. CHANGELOG [Unreleased] missing after v0.8.11 tag (silent skip).
     Same trap as patch_v0811a-fix1. After tag, [Unreleased] doesn't
     exist. My v0.9.0a code silently skipped. Class: §13.4/140
     recurrence — 2nd hit of same trap.

FIXES:
  1. row_filter.py: PredicateFilter._OPS_NO_SCALAR = {is_null, not_null,
     contains}; __init__ validates against _OPS ∪ _OPS_NO_SCALAR.
  2. test_dt_null_timestamp_propagates: `pa.array(ts)` instead of
     `ts.tolist()` (NaT → null, Arrow-native).
  3. CHANGELOG: _ensure_unreleased helper — prepend `## [Unreleased]`
     before first `## [` if missing. Backfill v0.9.0a entry.
  4. §13.4/146 — recurrence of [Unreleased]-after-tag trap (2nd time).
     Long-term backlog: extract to `cadw/patch_utils.py` shared helper.

Run: uv run python scripts/patch_v090a_fix1.py
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


# ============ Fix 1: PredicateFilter._OPS_NO_SCALAR ============

def fix_predicate_filter_validation():
    print("\n=== cadw/dataset/row_filter.py: _OPS_NO_SCALAR ============")
    p = Path("cadw/dataset/row_filter.py")
    s = p.read_text()

    old = '''class PredicateFilter(MaskFilter):
    _OPS = {
        "gt": pc.greater,
        "ge": pc.greater_equal,
        "lt": pc.less,
        "le": pc.less_equal,
        "eq": pc.equal,
        "ne": pc.not_equal,
    }

    def __init__(self, column: str, op: str, value) -> None:
        if op not in self._OPS:
            raise ValueError(f"unsupported op {op!r}; expected one of {list(self._OPS)}")
        self.column = column
        self.op = op
        self.value = value'''

    new = '''class PredicateFilter(MaskFilter):
    _OPS = {
        "gt": pc.greater,
        "ge": pc.greater_equal,
        "lt": pc.less,
        "le": pc.less_equal,
        "eq": pc.equal,
        "ne": pc.not_equal,
    }
    # v0.9.0a-fix1: ops that don't take a scalar argument. Handled
    # specially in mask_for_batch (is_null/not_null/contains). Prior
    # code extended mask_for_batch but not __init__ validation → tests
    # for new ops raised "unsupported op". Both must stay in sync.
    _OPS_NO_SCALAR = frozenset({"is_null", "not_null", "contains"})

    def __init__(self, column: str, op: str, value) -> None:
        if op not in self._OPS and op not in self._OPS_NO_SCALAR:
            raise ValueError(
                f"unsupported op {op!r}; expected one of "
                f"{list(self._OPS) + sorted(self._OPS_NO_SCALAR)}"
            )
        self.column = column
        self.op = op
        self.value = value'''

    s = replace_line_block(s, old, new, tag="predicate-op-validation")
    p.write_text(s)
    compile_check(p)
    print("  OK: _OPS_NO_SCALAR = {is_null, not_null, contains}")


# ============ Fix 2: NaT via pa.array ============

def fix_dt_nat_test():
    print("\n=== tests/test_dt_accessor.py: NaT via pa.array ============")
    p = Path("tests/test_dt_accessor.py")
    s = p.read_text()

    old = '''def test_dt_null_timestamp_propagates(tmp_path):
    ts = pd.to_datetime(["2020-01-15", None, "2020-03-20"])
    df = make_df(tmp_path, {"ts": ts.tolist()})
    result = df["ts"].dt.month
    got = read_col(result, tmp_path, "ts_month")
    assert got == [1, None, 3]'''

    new = '''def test_dt_null_timestamp_propagates(tmp_path):
    """NaT (pandas) → null (Arrow) → null (dt.month output).

    v0.9.0a-fix1: use `pa.array(ts)` not `ts.tolist()`. The latter yields
    [Timestamp, NaT, Timestamp]; `pa.table({"ts": [...]})` calls
    `NaT.utcoffset()` → `NaTType does not support utcoffset`. `pa.array`
    handles NaT → null (Arrow-native conversion).
    """
    ts = pd.to_datetime(["2020-01-15", None, "2020-03-20"])
    df = make_df(tmp_path, {"ts": pa.array(ts)})
    result = df["ts"].dt.month
    got = read_col(result, tmp_path, "ts_month")
    assert got == [1, None, 3]'''

    s = replace_line_block(s, old, new, tag="nat-pa-array")
    p.write_text(s)
    compile_check(p)
    print("  OK: pa.array(ts) instead of ts.tolist()")


# ============ Fix 3: CHANGELOG [Unreleased] + backfill v0.9.0a ============

CHANGELOG_V090A_ENTRY = '''## [Unreleased]

### Добавлено (v0.9.0a — accessors + type ops)

Первый шаг v0.9.0 «out-of-core framework»: подготовка данных
(datetime features, string cleaning, type casting, null handling).
Arrow-native, bounded per-batch, zero pandas.

**Accessors:**
- **`df['ts'].dt.{year,month,day,hour,minute,second,weekday,day_of_year,quarter}`** —
  Arrow `pc.year/month/day/hour/minute/second/day_of_week/day_of_year/quarter`.
  Возвращает CadwDataFrame с `{col}_{feature}` (original columns preserved).
- **`df['s'].str.{len,lower,upper}`** — `pc.utf8_length/utf8_lower/utf8_upper`.
  Возвращает CadwDataFrame с `{col}_{op}`.
- **`df['s'].str.contains(pat)`** — возвращает **MaskFilter** (для `df[mask]`).
  na=False (pandas default): null не матчится.
- **`df['col'].isnull()` / `.notnull()`** — MaskFilter.

**Facade:**
- **`df.astype({col: dtype})`** — cast per column. `dtype` = `pa.DataType`
  или str-alias (`"int64"`, `"float32"`). `safe=True` (default) → raise на
  overflow; `safe=False` → wrap (Arrow `pc.cast`).
- **`df.dropna(how='any'|'all', subset=None)`** — **lazy**: композиция
  `notnull()` фильтров. `how='any'`: drop if ANY null. `how='all'`: drop
  if ALL null.
- **`df.drop(cols)`** — selection minus. Lazy.
- **`df.rename(mapping)`** — materialized (schema change). `{old: new}`.

**Внутренний helper:**
- `CadwDataFrame._apply_column_transform(column, feature, fn, out_type,
  accessor=...)` — generic per-batch Arrow transform + append `{col}_{feature}`.
- Новая папка `cadw/accessors/` с `_dt.py`, `_str.py`.

**AST guards:** `tests/test_arrow_call_memory_guard.py` — whitelisted
`_apply_column_transform`, `astype`, `dropna`, `drop`, `rename`.

### Исправлено (v0.9.0a-fix1 — first-run bugs)

- **`PredicateFilter.__init__` не знал про новые ops (8 test failures).**
  v0.9.0a расширил `mask_for_batch` (is_null/not_null/contains), но
  `__init__` валидировал по старому `_OPS` → `ValueError: unsupported op`.
  Fix: `_OPS_NO_SCALAR = frozenset({"is_null", "not_null", "contains"})`,
  `__init__` проверяет `_OPS ∪ _OPS_NO_SCALAR`. Class: partial update
  within one class (handler extended, validation not).
- **`test_dt_null_timestamp_propagates` — NaT не конвертируется через
  `ts.tolist()` (1 failure).** `pa.table({"ts": [Timestamp, NaT, ...]})`
  → `NaTType does not support utcoffset`. Fix: `pa.array(ts)` — Arrow-native,
  NaT → null. Class: §13.4/134-like (input conversion not Arrow-safe).
- **CHANGELOG `[Unreleased]` отсутствовал после v0.8.11 tag** —
  silent skip (v0.9.0a entry не записан). Fix: prepend `[Unreleased]`
  before first `## [` if missing. Class: §13.4/140 recurrence (2nd hit
  of same trap — v0.8.11a-fix1 was first).

### Тесты (v0.9.0a)

- fast: **~1228** (+30 accessors + type ops). slow: **8**.

'''


def fix_changelog():
    print("\n=== CHANGELOG.md: ensure [Unreleased] + backfill v0.9.0a ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()

    if "v0.9.0a — accessors + type ops" in s:
        print("  v0.9.0a entry already present — skip")
        return

    # Ensure [Unreleased] exists
    if "## [Unreleased]" not in s:
        m = re.search(r'^## \[', s, re.MULTILINE)
        if m:
            s = s[:m.start()] + "## [Unreleased]\n\n" + s[m.start():]
            print("  prepended `## [Unreleased]` before first `## [` header")
        else:
            s = "## [Unreleased]\n\n" + s
            print("  prepended `## [Unreleased]` at file start")

    # Append entry under [Unreleased]
    s = s.replace("## [Unreleased]\n\n", CHANGELOG_V090A_ENTRY, 1)
    p.write_text(s)
    print("  OK: v0.9.0a + v0.9.0a-fix1 entries appended")


# ============ Fix 4: §13.4/146 ============

def append_decisions_146():
    print("\n=== DECISIONS.md: §13.4/146 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 146" in s:
        print("  already present — skip")
        return

    marker = "### 13.5. Открытые вопросы"
    if marker not in s:
        print("  WARNING: §13.5 marker not found — skip")
        return

    episode = '''**Эпизод 146 (v0.9.0a — three first-run bugs, three classes).**
Не кластер §13.4/139-144 (fix→bug) и не §13.4/145 (discovery через
fix-тест). Это **первый прогон нового кода** — три бага трёх разных
классов:

**1. Partial update within one class (8 test failures).**
`PredicateFilter._OPS` validation в `__init__` осталась
`{gt, ge, lt, le, eq, ne}`. `patch_v090a` расширил `mask_for_batch`
(is_null/not_null/contains), но **не добавил их в validation** →
`ValueError: unsupported op 'is_null'`. Восемь тестов упали (isnull,
notnull, dropna_any/all/subset, contains×2, dropna_then_astype).

**2. Input conversion not Arrow-safe (1 test failure).**
`test_dt_null_timestamp_propagates` использовал `ts.tolist()` —
это даёт `[Timestamp, NaT, Timestamp]`. `pa.table({"ts": [...]})`
вызывает `NaT.utcoffset()` → `NaTType does not support utcoffset`.
Fix: `pa.array(ts)` — Arrow-native, NaT → null.
Класс — §13.4/134 («input type different from intended»), не bug кода.

**3. `[Unreleased]`-after-tag trap, 2nd hit (silent skip).**
После v0.8.11 tag секции `[Unreleased]` в CHANGELOG нет (squash
переименовал). `patch_v090a` код:
```python
if "## [Unreleased]\\n" not in s:
    print("WARNING: ... — skip")
    return
```
Silent skip. Это **ровно та же** ловушка, что в `patch_v0811a-fix1`.
Первый раз (v0.8.11a-fix1) — 1 ч работы на fix; второй раз (v0.9.0a) —
опять fix. **Правило §13.4/144** («grep before rename») технически
выполнено, но **assumption "если anchor отсутствует — skip"**
не был заменён на «если anchor отсутствует — создать». Это **error
of omission** — отсутствие действия вместо неверного действия.

**Правило (расширение 140+143+146):** patch-скрипт, appending to
CHANGELOG, **обязан** вызвать `_ensure_unreleased()` — prepend
`## [Unreleased]` перед первым `## [` если его нет. Никаких
skip-ов при отсутствии anchor — fallback create.

**Long-term (§11.6 backlog, v0.9.1+):**
- `cadw/patch_utils.py` — общие утилиты для patch-скриптов:
  - `ensure_unreleased_header(text) -> text`
  - `append_unreleased_entry(text, entry) -> text`
  - `assert_guard_green(cmd) -> None` (§13.4/140)
- `PredicateFilter` — единый `_VALID_OPS = frozenset(_OPS) | _OPS_NO_SCALAR`,
  класс-attribute; `__init__` проверяет только `_VALID_OPS`. Никогда
  не может быть out-of-sync с handler'ом, если handler тоже
  использует `_VALID_OPS` для dispatch.

**Связанные:** §13.4/128, /129, /139-145. Кластер 139-145 — 6 раундов
fix→bug; здесь — 3 first-run bugs (новый код, не fix). Класс
принципиально другой, но **итог тот же**: 9 wasted tests, 1 round fix.

'''
    s = s.replace(marker, episode + marker, 1)
    p.write_text(s)
    print("  OK: §13.4/146")


# ============ main ============

def main():
    print("=== v0.9.0a-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_predicate_filter_validation()
    fix_dt_nat_test()
    fix_changelog()
    append_decisions_146()

    print("\n=== pytest: targeted (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_facade_transformers.py::test_isnull_filter",
         "tests/test_facade_transformers.py::test_notnull_filter",
         "tests/test_facade_transformers.py::test_dropna_any",
         "tests/test_str_accessor.py::test_str_contains_filter",
         "tests/test_str_accessor.py::test_str_contains_null_does_not_match",
         "tests/test_dt_accessor.py::test_dt_null_timestamp_propagates",
         "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** TARGETED STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: all new accessor tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dt_accessor.py",
         "tests/test_str_accessor.py",
         "tests/test_facade_transformers.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** ACCESSOR TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_row_filter.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GUARD FAILURE ***")
        sys.exit(1)

    print("\n=== pytest: dataframe + filter regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py", "tests/test_dataframe_filters.py",
         "tests/test_filter_compat.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
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
        "v0.9.0a: accessors (dt.*, str.*) + type ops (astype, dropna, "
        "drop, rename); _ColumnProxy.isnull/notnull; Arrow-native, "
        "zero pandas; §13.4/146 (3 first-run bugs — PredicateFilter "
        "validation, NaT conversion, [Unreleased]-after-tag)"])

    print("\n==> v0.9.0a-fix1 applied. Amended WIP commit.")
    print("NEXT: v0.9.0b (Pipeline skeleton + plan_pipeline real).")


if __name__ == "__main__":
    main()
