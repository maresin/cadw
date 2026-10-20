#!/usr/bin/env python3
"""v0.9.1f-fix4: stale test regex + stale docstring (round 5, §13.4/160).

BUG (§13.4/144 → /149 → /152 → /156 → /160, round 5):
  fix3 replaced error message formula `K_pairs × row_size` with
  `K_index × K_columns × value_size` (dense). Test
  `test_pivot_100k_rows_fails_fast` (fix2) has `match="K_pairs"` —
  no longer in message. 1 test fails.

  ALSO: `pivot_table` docstring still says "Fail-fast if K_pairs ×
  row_size × 2 > 0.3 × available" — stale after fix3.

  Manual grep for "K_pairs" in tests/ before fix3 — NOT done. Same
  discipline failure as rounds 1-4. `assert_no_stale_behavior` not
  called in fix3.

FIX:
  1. Test regex → `match="K_index"` (or broader `match="pivot_table"`).
  2. Docstring → dense formula.
  3. §13.4/160 — round 5, but now caught by E2E (good) — next time
     use assert_no_stale_behavior in patch script.

GUARDS: pivot E2E + guards + fast + slow.

Run: uv run python scripts/patch_v091f_fix4.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


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


def fix_test_regex():
    print("\n=== tests/test_pivot_table_e2e.py: regex K_pairs → pivot_table ===")
    p = Path("tests/test_pivot_table_e2e.py")
    s = p.read_text()

    old = '''    with pytest.raises(MemoryLimitExceeded, match="K_pairs"):
        df.pivot_table(
            index=["row"], columns="col", values="v", aggfunc="mean",
        )'''

    new = '''    with pytest.raises(MemoryLimitExceeded, match="pivot_table"):
        df.pivot_table(
            index=["row"], columns="col", values="v", aggfunc="mean",
        )'''

    s = replace_line_block(s, old, new, tag="test-regex-pivot")
    p.write_text(s)
    compile_check(p)
    print("  OK")


def fix_docstring():
    print("\n=== cadw/dataframe.py::pivot_table docstring — dense formula ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''        Memory: O(K_pairs × row_size) for grouped + O(K_index × K_cols
        × value_size) for pivot. Fail-fast if K_pairs × row_size × 2 >
        0.3 × available.
        """'''

    new = '''        Memory: intermediate buffers are
          - grouped long format: K_pairs × row_size (bounded by
            groupby_aggregate internally),
          - pandas to_pandas: 3 × grouped_table.nbytes (explicit check),
          - dense pivot: K_index × K_columns × value_size × 2 (pandas).
        Fail-fast BEFORE groupby if dense estimate exceeds
        state_limit_bytes (§13.4/159).
        """'''

    s = replace_line_block(s, old, new, tag="docstring-dense")
    p.write_text(s)
    compile_check(p)
    print("  OK")


def fix_earlier_e2e_docstrings():
    print("\n=== tests/test_pivot_table_e2e.py: docstrings reflect dense formula ===")
    p = Path("tests/test_pivot_table_e2e.py")
    s = p.read_text()

    old_success = '''    """10K unique rows × 20 cols, 200 MB budget → ~43 MB est < 60 MB.

    §13.4/158: E2E success test must respect inherent complexity.
    Pivot = O(K_pairs) materialization in pandas (no out-of-core pivot).
    K_pairs = 10K × 20 = 200K cells × ~108 B × 2 (pandas) ≈ 43 MB.
    Fits 0.3 × 200 MB = 60 MB state_limit.
    """'''

    new_success = '''    """10K unique rows × 20 cols, 200 MB budget.

    §13.4/158-159: E2E success test must respect inherent complexity.
    Pivot = dense `K_index × K_columns × value_size × 2` in pandas
    (no out-of-core pivot). K_index=10K, K_columns_est=100 (real 20),
    value_size=8 → 10K × 100 × 8 × 2 = 16 MB < state_limit (200 MB × 0.5
    = 100 MB). Passes.
    """'''

    if old_success in s:
        s = s.replace(old_success, new_success, 1)

    old_fail = '''    """100K unique × 20 cols, 50 MB budget → MemoryLimitExceeded.

    §13.4/158: Pivot of 2M pairs needs 400+ MB RSS (2M × 108 B × 2
    pandas). Fail-fast is the CORRECT contract — no OOM, no silent
    wrong-size pivot. User should reduce size or use groupby() long
    format.
    """'''

    new_fail = '''    """100K unique × 20 cols, 50 MB budget → MemoryLimitExceeded.

    §13.4/158-159: dense pivot of 100K × 400 (K_columns_est) × 8 × 2
    = 640 MB > state_limit 25 MB. Fail-fast is the CORRECT contract —
    no OOM, no silent wrong-size pivot. User should reduce size or
    use groupby() long format.
    """'''

    if old_fail in s:
        s = s.replace(old_fail, new_fail, 1)

    p.write_text(s)
    compile_check(p)
    print("  OK")


def append_decisions_160():
    print("\n=== DECISIONS.md: §13.4/160 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 160" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/160 (round 5 stale test regex).** fix3 changed "
        s += "formula K_pairs → K_index×K_columns; test match=K_pairs "
        s += "stale. Fix regex. assert_no_stale_behavior NOT called in fix3.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 160 (stale test regex — round 5, §13.4/144).**
fix3 заменил pre-check формулу `K_pairs × row_size` на dense `K_index
× K_columns × value_size`. Тест `test_pivot_100k_rows_fails_fast`
(fix2) имел `match="K_pairs"` — подстрока нового сообщения отсутствует.
1 E2E fails.

**Тот же паттерн, ПЯТЫЙ раз:**
- §13.4/144 — guard rename, other file.
- §13.4/149 — validation removal, same file.
- §13.4/152 — registration add, same file.
- §13.4/156 — blanket AST, other file.
- **§13.4/160** — formula change, same file.

**Причина each time:** patch script changes behavior; test asserts old
behavior; manual grep before commit **не сделан**.

**Tooling §13.4/152** (`assert_no_stale_behavior`) существует и
работает. **Но fix3 его не вызвал** — не применил собственный инструмент.
Это regression of discipline, not tooling failure.

**Fix (v0.9.1f-fix4):**
- Test regex `K_pairs` → `pivot_table` (broader, stable across formulas).
- Docstring `pivot_table` — dense formula (fix3 изменил код, docstring
  остался старым).
- Docstrings E2E tests — dense formula.
- **§13.4/160 — правило усилено:** каждый patch, изменяющий **error
  message** (even formula-only), обязан `grep_and_warn(["<old_substring>"],
  ["tests/"])` **перед** `git add`. Не «forgot», не «test passed
  locally» — **всегда**.

**Правило (усиление 149, 152):**
> Patch scripts **always** invoke `patch_utils.assert_no_stale_behavior`
> for:
> - Any removed/renamed validation error message.
> - Any changed formula in error message.
> - Any removed guard name.
> - Any changed return contract.
>
> Manual discipline failed 5x. Tooling exists. **Use it.**

**Backlog (§11.6):** `scaffold_patch.py` должен **enforce**
`assert_no_stale_behavior` call как mandatory step. Пока scaffold нет,
каждый patch — ручное решение (которое 5 раз подряд не принято).

**Метаправило:** когда **у тебя есть инструмент, который поймал бы
баг** — использовать его. `assert_no_stale_behavior` в patch_utils
ловит stale patterns механически. Пять раз подряд **не использовать
готовый механизм** — это не «забыл», это **систематическая ошибка
процесса**. Fix процесса — не «вспомнить», а **enforce** (scaffold).

**Связанные:** §13.4/144, /149, /152, /156 (stale test repeats),
/157 (hasattr pyarrow), /158 (E2E complexity), /159 (dense pre-check).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/160")


def dogfood_grep():
    """Dogfood: grep for 'K_pairs' in tests/ before commit (§13.4/152)."""
    print("\n=== dogfood: grep 'K_pairs' in tests/ ===")
    from cadw.patch_utils import grep_and_warn
    hits = grep_and_warn(
        ["K_pairs"],
        ["tests/test_pivot_table.py", "tests/test_pivot_table_e2e.py"],
        label="K_pairs-stale",
    )
    if hits:
        print(f"  {len(hits)} references remain (should be 0 in tests/):")
        for h in hits[:5]:
            print(f"    {h}")
    else:
        print("  OK: no K_pairs references in pivot tests")


def main():
    print("=== v0.9.1f-fix4 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_test_regex()
    fix_docstring()
    fix_earlier_e2e_docstrings()
    append_decisions_160()
    dogfood_grep()

    print("\n=== pytest: pivot E2E (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pivot_table_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: pivot fast (regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pivot_table.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIVOT FAST REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards via patch_utils ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

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
    sh(["git", "commit", "-m",
        "v0.9.1f: pivot_table / melt / crosstab — long↔wide transforms; "
        "§13.4/157-160 (pa.types.is_*; E2E respects complexity; dense "
        "pre-check K_index×K_columns×value_size; state_limit_bytes; "
        "to_pandas pre-check ×3; stale test regex round 5)"])

    print("\n==> v0.9.1f-fix4 applied. v0.9.1 complete.")
    print()
    print("SQUASH + TAG v0.9.1:")
    print("  git reset --soft v0.9.0")
    print("  git commit -m 'v0.9.1: pipeline advanced'")
    print("  sed -i 's/^## \\[Unreleased\\]$/## [0.9.1] — 2026-10-20/' CHANGELOG.md")
    print("  git add CHANGELOG.md && git commit --amend --no-edit")
    print("  git tag -a v0.9.1 -m 'v0.9.1: pipeline advanced complete'")
    print("  uv run pytest tests/test_changelog.py -v")


if __name__ == "__main__":
    main()