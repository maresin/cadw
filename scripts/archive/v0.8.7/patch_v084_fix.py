#!/usr/bin/env python3
"""v0.8.4-fix: (a) S1 bug in _RollState min/max + nulls; (b) 2 test expectations.

FINDINGS:

1. **CRITICAL latent bug (S1, v0.6.2c→v0.8.4)**: `_RollState.push(None)` с
   agg='min'/'max' и непустым `_dq` → `TypeError: '>=' not supported
   between instances of 'int' and 'NoneType'`. Fixed by guarding the
   monotone-deque pop loop with `if not is_null`. Currently untested —
   `test_rolling_with_nulls` uses agg='sum' only.

2. **test_rolling_on_with_nulls**: expected [30,10,30,50], actual (correct)
   [20,10,30,30]. v0.6.2c rolling treats null as consuming window slot
   (positional window, matches pandas). Test author assumed value-based
   window.

3. **test_rolling_on_all_columns**: expected w [7,9,None,7,3], actual
   (correct) [5,7,None,6,8]. Test comment "Sorted w: [3,4,5,2,1]" is
   wrong — w sorted by ts = [3,5,2,4,1], not by w-value.

Run: uv run python scripts/patch_v084_fix.py
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


# ============ Fix 1: _RollState min/max + nulls ============

def fix_rollstate_minmax_nulls():
    print("\n=== cadw/dataframe.py: _RollState min/max + null guard ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''        if self._agg == "min":
            while self._dq and self._dq[-1][1] >= v:
                self._dq.pop()
            if not is_null:
                self._dq.append((i, v))
        elif self._agg == "max":
            while self._dq and self._dq[-1][1] <= v:
                self._dq.pop()
            if not is_null:
                self._dq.append((i, v))'''

    new = '''        # v0.8.4-fix (S1): null + non-empty _dq → `_dq[-1][1] >= None`
        # raises TypeError. Nulls skip monotone-deque maintenance entirely
        # (they don't contribute to min/max; window slot still consumed).
        if self._agg == "min":
            if not is_null:
                while self._dq and self._dq[-1][1] >= v:
                    self._dq.pop()
                self._dq.append((i, v))
        elif self._agg == "max":
            if not is_null:
                while self._dq and self._dq[-1][1] <= v:
                    self._dq.pop()
                self._dq.append((i, v))'''

    s = replace_line_block(s, old, new, tag="rollstate-minmax-null")
    p.write_text(s)
    compile_check(p)
    print("  OK: min/max + null guard (S1 fix)")


# ============ Fix 2: test expectations ============

def fix_test_nulls():
    print("\n=== tests/test_rolling_on_col.py: "
          "test_rolling_on_with_nulls ===")
    p = Path("tests/test_rolling_on_col.py")
    s = p.read_text()

    old = '''def test_rolling_on_with_nulls(tmp_path):
    df = make_df(tmp_path, data={
        "ts": [3, 1, 2, 4],
        "v": [None, 10.0, 20.0, 30.0],
    })
    res = df.rolling(2, "sum", on="ts", columns=["v"], min_periods=1)
    # Sorted: ts=1(v=10),2(20),3(None),4(30).
    # rolling w/ null-skip: [10,30,30,50]. Back to orig: [30,10,30,50]
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [30.0, 10.0, 30.0, 50.0]'''

    new = '''def test_rolling_on_with_nulls(tmp_path):
    """Nulls consume window slot (positional, matches pandas skipna=True).

    Sorted by ts: [10, 20, None, 30]. window=2, sum, min_periods=1:
        pos0: [10] → 10
        pos1: [10,20] → 30
        pos2: [20,None] → 20 (window evicts 10; null contributes 0 to sum)
        pos3: [None,30] → 30 (window evicts 20)
    Sorted: [10, 30, 20, 30]. Map by row_id [1,2,0,3]:
        orig 0 → pos2 → 20
        orig 1 → pos0 → 10
        orig 2 → pos1 → 30
        orig 3 → pos3 → 30
    Expected: [20, 10, 30, 30] (matches pandas rolling skipna=True).
    """
    df = make_df(tmp_path, data={
        "ts": [3, 1, 2, 4],
        "v": [None, 10.0, 20.0, 30.0],
    })
    res = df.rolling(2, "sum", on="ts", columns=["v"], min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [20.0, 10.0, 30.0, 30.0]'''

    s = replace_line_block(s, old, new, tag="test-nulls")
    p.write_text(s)
    compile_check(p)
    print("  OK: expected [20,10,30,30]")


def fix_test_all_columns():
    print("\n=== tests/test_rolling_on_col.py: "
          "test_rolling_on_all_columns ===")
    p = Path("tests/test_rolling_on_col.py")
    s = p.read_text()

    old = '''def test_rolling_on_all_columns(tmp_path):
    """columns=None: all columns rolled (numeric only for agg=sum)."""
    df = make_df(tmp_path, data={
        "ts": [5, 3, 1, 4, 2],
        "v": [50.0, 30.0, 10.0, 40.0, 20.0],
        "w": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    # Note: 'ts' itself is numeric — rolled too (user asked all numeric).
    res = df.rolling(2, "sum", on="ts", columns=["v", "w"])
    v = read_col(res, tmp_path, "v_rolling_sum")
    w = read_col(res, tmp_path, "w_rolling_sum")
    assert v == [90.0, 50.0, None, 70.0, 30.0]
    # Sorted w: [3,4,5,2,1]; w_sum w2: [None,7,9,7,3]. back: [7,9,None,7,3]
    assert w == [7.0, 9.0, None, 7.0, 3.0]'''

    new = '''def test_rolling_on_all_columns(tmp_path):
    """columns=None: all columns rolled (numeric only for agg=sum).

    Sorted by ts [1,2,3,4,5]:
      v: [10, 20, 30, 40, 50]  → v_sum w2 (min_periods=2): [None, 30, 50, 70, 90]
      w: [3, 5, 2, 4, 1]       → w_sum w2: [None, 8, 7, 6, 5]
    Map back to orig ts order [5, 3, 1, 4, 2]:
      v: orig 0(ts=5)→pos4→90, orig 1(ts=3)→pos2→50, orig 2(ts=1)→pos0→None,
         orig 3(ts=4)→pos3→70, orig 4(ts=2)→pos1→30
      w: orig 0(ts=5)→pos4→5,  orig 1(ts=3)→pos2→7,  orig 2(ts=1)→pos0→None,
         orig 3(ts=4)→pos3→6,  orig 4(ts=2)→pos1→8
    """
    df = make_df(tmp_path, data={
        "ts": [5, 3, 1, 4, 2],
        "v": [50.0, 30.0, 10.0, 40.0, 20.0],
        "w": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    res = df.rolling(2, "sum", on="ts", columns=["v", "w"])
    v = read_col(res, tmp_path, "v_rolling_sum")
    w = read_col(res, tmp_path, "w_rolling_sum")
    assert v == [90.0, 50.0, None, 70.0, 30.0]
    assert w == [5.0, 7.0, None, 6.0, 8.0]'''

    s = replace_line_block(s, old, new, tag="test-all-cols")
    p.write_text(s)
    compile_check(p)
    print("  OK: w expected [5,7,None,6,8]")


# ============ Fix 3: new tests for min/max + nulls ============

def add_minmax_null_tests():
    print("\n=== tests/test_rolling.py: min/max + nulls (regression) ===")
    p = Path("tests/test_rolling.py")
    s = p.read_text()
    if "test_rolling_min_with_nulls" in s:
        print("  already present — skip")
        return
    extra = '''

# ---------- v0.8.4-fix (S1): min/max + nulls (was TypeError crash) ----------

def test_rolling_min_with_nulls(tmp_path):
    """S1 regression: min + null must not crash (`_dq[-1][1] >= None`).

    Before fix: TypeError: '>=' not supported between 'int' and 'NoneType'.
    After fix: nulls skip monotone-deque maintenance; window slot still
    consumed (positional semantics, matches pandas).

    pandas: pd.Series([3,None,1,None,5]).rolling(2,min_periods=1).min()
            → [3, 3, 1, 1, 5]
    """
    df = make_df(tmp_path, data={"v": [3.0, None, 1.0, None, 5.0]})
    res = df.rolling(2, "min", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_min")
    assert vals == [3.0, 3.0, 1.0, 1.0, 5.0]


def test_rolling_max_with_nulls(tmp_path):
    """S1 regression: max + null must not crash."""
    df = make_df(tmp_path, data={"v": [3.0, None, 1.0, None, 5.0]})
    res = df.rolling(2, "max", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals == [3.0, 3.0, 3.0, 1.0, 5.0]


def test_rolling_min_all_nulls(tmp_path):
    """All-null: no crash (`_dq` stays empty throughout)."""
    df = make_df(tmp_path, data={
        "v": pa.array([None, None, None], type=pa.float64()),
    })
    res = df.rolling(2, "min", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_min")
    assert vals == [None, None, None]
'''
    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: +3 min/max + nulls tests")

    # Also add an on=col variant
    print("\n=== tests/test_rolling_on_col.py: min/max + nulls ===")
    p2 = Path("tests/test_rolling_on_col.py")
    s2 = p2.read_text()
    if "test_rolling_on_min_with_nulls" in s2:
        print("  already present — skip")
        return
    extra2 = '''

def test_rolling_on_min_with_nulls(tmp_path):
    """S1 regression (on=col): min + null must not crash."""
    df = make_df(tmp_path, data={
        "ts": [1, 2, 3, 4, 5],
        "v": [3.0, None, 1.0, None, 5.0],
    })
    res = df.rolling(2, "min", on="ts", columns=["v"], min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_min")
    # Sorted by ts is already 1..5; positional nulls consume slot:
    # [3, 3, 1, 1, 5]. Row_id = original position.
    assert vals == [3.0, 3.0, 1.0, 1.0, 5.0]


def test_rolling_on_max_with_nulls(tmp_path):
    df = make_df(tmp_path, data={
        "ts": [1, 2, 3, 4, 5],
        "v": [3.0, None, 1.0, None, 5.0],
    })
    res = df.rolling(2, "max", on="ts", columns=["v"], min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals == [3.0, 3.0, 3.0, 1.0, 5.0]
'''
    s2 = s2.rstrip() + extra2 + "\n"
    p2.write_text(s2)
    compile_check(p2)
    print("  OK: +2 on=col min/max + nulls tests")


# ============ Fix 4: CHANGELOG + DECISIONS ============

def patch_docs():
    print("\n=== CHANGELOG.md: extend [0.8.4] with S1 fix ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()

    old_marker = "### Обнаружен latent bug (§13.4/126)"
    if old_marker not in s:
        print("  WARNING: [0.8.4] block marker not found — skip")
        return

    add = """### Исправлено (v0.8.4-fix — S1, latent bug с v0.6.2c)
- **`_RollState.push(None)` для agg='min'/'max'** — `while _dq and
  _dq[-1][1] >= None` → `TypeError: '>=' not supported between int and
  NoneType`. Bug существовал с v0.6.2c (позиционный rolling), не
  покрыт тестами (`test_rolling_with_nulls` использует sum). Fix:
  guard `if not is_null` вокруг monotone-deque maintenance. Window
  slot всё равно потребляется (positional semantics, как pandas).
  Добавлены тесты: `test_rolling_min_with_nulls`,
  `test_rolling_max_with_nulls`, `test_rolling_min_all_nulls`
  (positional) + `test_rolling_on_min_with_nulls`,
  `test_rolling_on_max_with_nulls` (on=col).
- **v0.8.4 tests (2, not code)**: `test_rolling_on_with_nulls` и
  `test_rolling_on_all_columns` — неверные ожидания. Actual code
  корректен (nulls потребляют window slot, positional; matches pandas
  skipna=True). Fixed expected values + comments.

"""
    s = s.replace(old_marker, add + old_marker, 1)
    p.write_text(s)
    print("  OK: [0.8.4] extended")

    # DECISIONS: §13.4/127
    print("\n=== DECISIONS.md: §13.4/127 ===")
    p2 = Path("DECISIONS.md")
    s2 = p2.read_text()
    if "Эпизод 127" in s2:
        print("  already present — skip")
        return

    marker = "### 13.5. Открытые вопросы"
    episode = '''**Эпизод 127 (S1 latent bug: min/max + null в rolling).** Найдено
при аудите v0.8.4 — `_RollState.push(None)` с `agg='min'/'max'` и
непустым `_dq` (monotone deque) падает `TypeError: '>=' not supported
between 'int' and 'NoneType'`. Bug существует с v0.6.2c (позиционный
rolling), **не покрыт тестами**: `test_rolling_with_nulls` использует
`agg="sum"` только, а `test_rolling_min_w3`/`test_rolling_max_w3` — без
nulls. Значит `df.rolling(3, 'min')` на данных с null падал.

**Fix (v0.8.4-fix):** guard `if not is_null:` вокруг `while _dq and
_dq[-1][1] >= v` (и `<=` для max). Nulls skip monotone-deque
maintenance; window slot **всё равно потребляется** (позиционная
семантика, matches pandas skipna=True: `pd.Series([3,None,1]).
rolling(2,min_periods=1).min()` → `[3,3,1]`).

**Урок (усиливает §13.4/52b, 99, 106, 52b):** "зелёные тесты ≠
упражняется функциональность". `test_rolling_with_nulls` (sum) проходил,
но `min`/`max` ветка `_dq` не тестировалась с null. Класс "fallback
скрывает bug": sum-ветка не касается `_dq`, min/max — касается, но
никто не проверял комбинацию.

**Правило (новый AST/coverage guard для будущего):** для каждой
ветки `if self._agg == X` в stateful op — тест с nulls. В `_RollState`
ветки: sum/mean, count, min, max. Сейчас покрыты: sum+null (есть),
count+null (есть), min+null (добавлено), max+null (добавлено). Все 4 — 
хорошо. Для будущих composite (resample, groupby-shift) — тот же
принцип.

**Backlog:** добавить AST guard `test_rollstate_all_aggs_with_nulls` —
walk по `_RollState.push` и assert каждый `_agg ==` имеет тест с
nulls (не покрытие, а явная проверка). 20 минут, отложить.

**Общий аудит v0.8.4 (см. ниже):**
- 2 test-expectation errors (fixed).
- 1 S1 latent bug (fixed).
- Прочее: multi-key join cartesian fallback (S3), product-of-nunique
  overestimate (S3), `_iter_zipped` duplication (S3), `row_id` reserved
  (S3). Documented; см. §11.5/§11.6.

'''
    s2 = s2.replace(marker, episode + marker, 1)
    p2.write_text(s2)
    print("  OK: §13.4/127")


# ============ main ============

def main():
    print("=== v0.8.4-fix: _RollState S1 bug + test expectations ===")

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

    fix_rollstate_minmax_nulls()
    fix_test_nulls()
    fix_test_all_columns()
    add_minmax_null_tests()
    patch_docs()

    print("\n=== pytest tests/test_rolling_on_col.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_rolling_on_col.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.4-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_rolling.py -v (regression, incl. new min/max+null) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_rolling.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE (rolling) ***")
        sys.exit(1)

    print("\n=== pytest: rolling + shift + expanding/ewm ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_rolling.py",
         "tests/test_shift_diff.py",
         "tests/test_shift_on_col.py",
         "tests/test_rolling_on_col.py",
         "tests/test_expanding_ewm.py",
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
        "v0.8.4-fix: S1 _RollState min/max + nulls (v0.6.2c latent); "
        "correct test expectations (nulls consume window slot, positional)"])

    print("\n==> v0.8.4-fix applied. Готов squash → tag v0.8.4.")


if __name__ == "__main__":
    main()