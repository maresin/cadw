#!/usr/bin/env python3
"""v0.8.4-fix2: правильные ожидания для max+nulls тестов (positional window).

Ошибка: в test_rolling_max_with_nulls (test_rolling.py) и
test_rolling_on_max_with_nulls (test_rolling_on_col.py) я указал
[3.0, 3.0, 3.0, 1.0, 5.0]. Правильно — [3.0, 3.0, 1.0, 1.0, 5.0].

Разбор: positional window=2, pos 2 sees [None, 1.0], НЕ [3.0, None, 1.0].
3.0 был evicted на pos 2. pandas rolling(2, min_periods=1).max() даёт
[3,3,1,1,5] — min и max на этом датасете совпадают (в окне после pos 0
только одно non-null значение).

Урок (§13.4/128): тест-expectation для rolling+nulls — считать построчно
с явным trace окна, не в уме. Или лучше — сверять с pandas на том же
датасете (минимальное отклонение от "prototype-first": prototype +
manual ground truth должны совпадать; моё "ручное" совпало, а
написанное в тест ожидание — нет).

Run: uv run python scripts/patch_v084_fix2.py
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


# ============ Fix 1: test_rolling.py::test_rolling_max_with_nulls ============

def fix_max_test_positional():
    print("\n=== tests/test_rolling.py: test_rolling_max_with_nulls ===")
    p = Path("tests/test_rolling.py")
    s = p.read_text()

    old = '''def test_rolling_max_with_nulls(tmp_path):
    """S1 regression: max + null must not crash."""
    df = make_df(tmp_path, data={"v": [3.0, None, 1.0, None, 5.0]})
    res = df.rolling(2, "max", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals == [3.0, 3.0, 3.0, 1.0, 5.0]'''

    new = '''def test_rolling_max_with_nulls(tmp_path):
    """S1 regression: max + null must not crash.

    Positional window=2, min_periods=1, skipna. Walk:
      pos0 window=[3.0]         → max=3.0
      pos1 window=[3.0, None]   → max=3.0
      pos2 window=[None, 1.0]   → max=1.0  (3.0 evicted)
      pos3 window=[1.0, None]   → max=1.0
      pos4 window=[None, 5.0]   → max=5.0
    pandas: pd.Series([3.0,None,1.0,None,5.0]).rolling(2,min_periods=1).max()
            → [3.0, 3.0, 1.0, 1.0, 5.0]
    (min gives same result here — after pos 0 only one non-null per window.)
    """
    df = make_df(tmp_path, data={"v": [3.0, None, 1.0, None, 5.0]})
    res = df.rolling(2, "max", min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals == [3.0, 3.0, 1.0, 1.0, 5.0]'''

    s = replace_line_block(s, old, new, tag="max-positional")
    p.write_text(s)
    compile_check(p)
    print("  OK: expected [3.0, 3.0, 1.0, 1.0, 5.0]")


# ============ Fix 2: test_rolling_on_col.py::test_rolling_on_max_with_nulls ============

def fix_max_test_on_col():
    print("\n=== tests/test_rolling_on_col.py: test_rolling_on_max_with_nulls ===")
    p = Path("tests/test_rolling_on_col.py")
    s = p.read_text()

    old = '''def test_rolling_on_max_with_nulls(tmp_path):
    df = make_df(tmp_path, data={
        "ts": [1, 2, 3, 4, 5],
        "v": [3.0, None, 1.0, None, 5.0],
    })
    res = df.rolling(2, "max", on="ts", columns=["v"], min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals == [3.0, 3.0, 3.0, 1.0, 5.0]'''

    new = '''def test_rolling_on_max_with_nulls(tmp_path):
    """S1 regression (on=col): max + null must not crash.

    ts already sorted 1..5; same positional walk as positional test:
      pos0 [3.0]        → 3.0
      pos1 [3.0, None]  → 3.0
      pos2 [None, 1.0]  → 1.0  (3.0 evicted)
      pos3 [1.0, None]  → 1.0
      pos4 [None, 5.0]  → 5.0
    """
    df = make_df(tmp_path, data={
        "ts": [1, 2, 3, 4, 5],
        "v": [3.0, None, 1.0, None, 5.0],
    })
    res = df.rolling(2, "max", on="ts", columns=["v"], min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals == [3.0, 3.0, 1.0, 1.0, 5.0]'''

    s = replace_line_block(s, old, new, tag="on-max")
    p.write_text(s)
    compile_check(p)
    print("  OK: expected [3.0, 3.0, 1.0, 1.0, 5.0]")


# ============ Fix 3: verify no other wrong [3,3,3,1,5] in tests ============

def verify_no_remaining():
    print("\n=== grep check: no [3,3,3,1,5] expectations ===")
    r = subprocess.run(
        ["grep", "-rn", "3.0, 3.0, 3.0, 1.0, 5.0", "tests/"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print("  WARNING: found matches:")
        for line in r.stdout.strip().splitlines():
            print(f"    {line}")
        # Not fatal — could be a legitimate [3,3,3,1,5] somewhere else
    else:
        print("  OK: no stale [3,3,3,1,5] expectations")


# ============ Fix 4: DECISIONS §13.4/128 (test-expectation errors) ============

def patch_decisions():
    print("\n=== DECISIONS.md: §13.4/128 (test expectations vs prototype) ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 128" in s:
        print("  already present — skip")
        return

    marker = "### 13.5. Открытые вопросы"
    episode = '''**Эпизод 128 (test expectations ≠ prototype).** v0.8.4-fix добавил
тесты `test_rolling_min/max_with_nulls` (позиционные) и `..._on_...`
(+on=col) — 5 тестов, 4 прошли, 1 упал. Причина: **ошибка в
ожидаемом значении** для max, не в коде.

Дано: `v = [3, None, 1, None, 5]`, window=2, min_periods=1, positional.
Walk:
    pos0: [3]         → max=3
    pos1: [3, None]   → max=3
    pos2: [None, 1]   → max=1   (3 evicted — 3 в окне pos1, не pos2)
    pos3: [1, None]   → max=1
    pos4: [None, 5]   → max=5
Правильно: `[3, 3, 1, 1, 5]`. Я написал `[3, 3, 3, 1, 5]` — мысленно
"оставил" 3 как максимум на pos2.

**Почему prototype это не поймал:** prototype проверил алгоритм
(manual == external_sort+deque == sort_values+rolling), но **не проверял
test expectations**. Test writer сгенерировал ожидания вручную
(мысленная арифметика) — тот же класс ошибки, что "off-by-one" в
ручных тестах.

**Урок:** test expectations для composite ops (rolling, quantile,
rank tie-breaking, shift с nulls) **нельзя** писать мысленной
арифметикой. Три варианта:
1. **pandas-equivalence:** `pd.Series(v).rolling(...).max().tolist()`
   в самом тесте (медленно, но надёжно).
2. **Explicit trace в docstring** — пошагово по позициям, как выше.
   Пишешь → видишь ошибку в reasoning до запуска.
3. **Prototype ground-truth** — `manual_rolling_on` уже содержит
   правильную логику (deque на sorted); использовать **его** output
   для test expectation, не пересчитывать вручную.

**Применено:** docstring в `test_rolling_max_with_nulls` и
`test_rolling_on_max_with_nulls` теперь содержит полный пошаговый trace
(после fix2). `test_rolling_min_with_nulls` случайно прошёл — на этом
датасете min и max совпадают (в окне 2 позиции, после pos 0 только
один non-null → min == max).

**Правило:** для тестов на rolling/shift/quantile+nulls — trace в
docstring обязателен. Если trace длинный (>5 позиций) — используй
`pandas.Series` или manual-функцию из prototype.

**Связанный эпизод (§13.4/52b, 99, 106):** "зелёные тесты ≠
упражняется функциональность". Здесь — "prototype passed ≠ test
expectations correct". Два независимых failure mode в test-слое.
Prototype-first (§13.4/123) фиксирует **алгоритм**, но не **тест**.

**Backlog (v0.9):** `tests/_helpers/rolling_reference.py` — функция
`expected_rolling(vals, window, agg, min_periods)` на deque, которая
используется в тестах как ground truth. Разные тесты — одна реализация
(не 4 копии) + нет мысленной арифметики.

'''
    s = s.replace(marker, episode + marker, 1)
    p.write_text(s)
    print("  OK: §13.4/128")


# ============ main ============

def main():
    print("=== v0.8.4-fix2: correct max+nulls test expectations ===")

    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    # After WIP commit, tree should be clean
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    fix_max_test_positional()
    fix_max_test_on_col()
    verify_no_remaining()
    patch_decisions()

    print("\n=== pytest tests/test_rolling.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_rolling.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.4-fix2 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_rolling_on_col.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_rolling_on_col.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.4-fix2 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest: rolling/shift/expanding regression ===")
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
        "v0.8.4-fix2: correct max+nulls test expectations (positional window, "
        "same result as min on this data) + §13.4/128 trace-in-docstring rule"])

    print("\n==> v0.8.4-fix2 complete. Готов squash → tag v0.8.4.")


if __name__ == "__main__":
    main()