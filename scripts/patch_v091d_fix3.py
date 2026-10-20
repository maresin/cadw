#!/usr/bin/env python3
"""v0.9.1d-fix3: missing `from pathlib import Path` in patch_utils.

BUG: grep_and_warn uses `Path(d).exists()` but `Path` not imported in
patch_utils.py. NameError caught by dogfood call in fix2's main() BEFORE
commit. This is exactly what mechanical tooling is for — bug found at
patch-time, not at next CI run.

STATE AFTER FAILED fix2:
  ✅ test_list_step_types_default updated (4 items)
  ✅ grep_and_warn / assert_no_stale_behavior added
  ✅ test_patch_utils 6 grep tests added
  ✅ §13.4/152 added
  ❌ dogfood NameError (Path missing)
  ❌ no commit

FIX: add `from pathlib import Path` to patch_utils.py imports.

META (§13.4/153 — recursive self-catch):
  This is round 4+ of the "fix creates new bug" cluster, but with a
  twist: the tool that catches stale behaviors caught a bug IN ITSELF
  before commit. Discipline works — but only because we ran it
  immediately. Don't skip dogfood, don't defer to "next run".

GUARDS: full pytest (patches + new tests + regression) + fast + slow.

Run: uv run python scripts/patch_v091d_fix3.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


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


def fix_path_import():
    print("\n=== cadw/patch_utils.py: import Path ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()

    old = '''import re
import subprocess
import sys
from typing import Optional'''

    new = '''import re
import subprocess
import sys
from pathlib import Path
from typing import Optional'''

    if "from pathlib import Path" in s:
        print("  already present — skip")
        return

    s = replace_line_block(s, old, new, tag="import-path")
    p.write_text(s)

    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{p}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  COMPILE FAIL: {p}")
        print(r.stderr[-800:])
        sys.exit(1)
    print("  compile OK")
    print("  OK")


def append_decisions_153():
    print("\n=== DECISIONS.md: §13.4/153 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 153" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/153 (tooling caught its own bug).** "
        s += "grep_and_warn used Path without import; NameError caught by "
        s += "dogfood call BEFORE commit. Tooling works.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 153 (tooling caught its own bug — recursion in discipline).**
v0.9.1d-fix2 добавил `patch_utils.grep_and_warn` /
`assert_no_stale_behavior` — mechanical tooling для §13.4/152. Первый же
dogfood-вызов упал с `NameError: Path not defined` — `grep_and_warn`
использует `Path(d).exists()`, но `from pathlib import Path` не был
добавлен в импорты `patch_utils.py`.

**Ирония:** инструмент против stale behaviors содержал **того же класса
баг**, что и все предыдущие раунды (§13.4/146-152): partial update —
написал новый метод, забыл про import для его работы. Но **dogfood caught
it BEFORE commit**, не после. Не потому что код был правильнее, а потому
что **проверка была немедленной**, не отложенной.

**Класс ошибки:** тот же (§13.4/146-152), но **механизм обнаружения
другой** — встроенная проверка в том же прогоне, а не CI / следующая
сессия. Это ровно то, что должно быть для всех патчей: guard-runs в
`main()` до commit, а не «потом».

**Fix (v0.9.1d-fix3):** добавить `from pathlib import Path` в
`patch_utils.py`. 4 строки.

**Урок:** mechanical tooling **работает**, если его **вызывать сразу**.
`assert_no_stale_behavior` в `main()` (dogfood) поймал missing-import
до коммита. Без dogfood этот баг ушёл бы в коммит и всплыл бы на
следующем использовании `grep_and_warn` (следующий патч).

**Метаправило:** для любого patch-скрипта с новыми tooling/helpers —
**dogfood в самом скрипте** до `git add`. Не «протестируем отдельно» —
**вызвать в main()** в том же прогоне. Иначе tooling сам становится
источником багов.

**Связанные:** §13.4/144-152 (7 раундов «fix creates bug»), §13.4/146
(partial update), §13.4/152 (tooling required — manual discipline fails).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/153")


def main():
    print("=== v0.9.1d-fix3 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_path_import()
    append_decisions_153()

    # Re-run dogfood — should pass now
    print("\n=== dogfood: assert_no_stale_behavior (retry) ===")
    from cadw.patch_utils import assert_no_stale_behavior
    try:
        assert_no_stale_behavior(
            patterns=[
                'assert list_step_types() == ["CastStep", "DropnaStep", "GBDTStep"]',
                '"CastStep", "DropnaStep", "GBDTStep"]',
            ],
            dirs=["tests/"],
            label="list_step_types-old-count",
        )
        print("  OK: no stale references")
    except SystemExit:
        print("\n*** dogfood still finds hits — inspect ***")
        raise

    print("\n=== pytest: registry + CT + patch_utils (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_register_step_type.py",
         "tests/test_column_transformer.py",
         "tests/test_column_transformer_pipeline.py",
         "tests/test_patch_utils.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_pipeline_e2e.py",
         "tests/test_pipeline_save_load.py",
         "tests/test_pipeline_save_load_e2e.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIPELINE REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards via patch_utils ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
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

    sh_add = subprocess.run(["git", "add", "-A"],
                            capture_output=True, text=True)
    print(f"  $ git add -A")
    sh_commit = subprocess.run(
        ["git", "commit", "-m",
         "v0.9.1d: ColumnTransformer (sklearn-style column fanout, "
         "row-preserving, remainder drop/passthrough) + register_step_type "
         "(module-level _STEP_REGISTRY, custom steps for from_json); "
         "cadw/__init__.py exports Pipeline + wrappers; §13.4/151 "
         "(registry __serialization_name__); §13.4/152 (grep tooling); "
         "§13.4/153 (tooling caught its own missing-import via dogfood)"],
        capture_output=True, text=True,
    )
    print(sh_commit.stdout[-500:] if sh_commit.stdout else "")
    if sh_commit.returncode != 0:
        print(f"  STDERR: {sh_commit.stderr[-800:]}")
        sys.exit(1)

    print("\n==> v0.9.1d-fix3 applied.")
    print("NEXT: v0.9.1e (apply(fn) — prototype + implementation).")


if __name__ == "__main__":
    main()