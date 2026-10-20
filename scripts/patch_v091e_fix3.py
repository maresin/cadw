#!/usr/bin/env python3
"""v0.9.1e-fix3: stale test in test_repr_html.py (round 4, §13.4/156).

BUG (§13.4/144 → /149 → /152 → /156, round 4):
  v0.9.0c added `test_no_pandas_import_in_dataframe` in
  `test_repr_html.py` that walks the ENTIRE AST and forbids any
  `import pandas` anywhere in `cadw/dataframe.py`. v0.9.1e-fix2
  added legitimate lazy `import pandas as pd` in `_emit_group` (per
  user fn output). Test fails.

  Same check exists in `test_no_pandas_in_ops.py::test_no_pandas_in_
  dataframe_facade` — updated in fix2 to module-level only. Two
  near-duplicate tests in different files — one updated, one stale.

ROOT CAUSE (§13.4/155 extension):
  File-wide AST guards ("no X anywhere") are fragile — break on any
  legitimate lazy X. `assert_no_stale_behavior` in fix2 only checked
  the specific `list_step_types` pattern — missed this one.

FIX:
  1. Update `test_no_pandas_import_in_dataframe` in test_repr_html.py
     to module-level only (same as fix2's change in test_no_pandas_in_ops).
  2. Consolidate: instead of duplicating the assertion logic, import
     `_uses_pandas_import_at_module_level` from test_no_pandas_in_ops
     helper (or skip the duplicate — but that's risky for a guard test).
     Minimal: keep both, both use module-level semantics.
  3. Add §13.4/156 documenting round 4 + rule "no blanket-walk guards".

GUARDS: new test + apply tests + guards + fast + slow.

Run: uv run python scripts/patch_v091e_fix3.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


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


def fix_stale_test():
    print("\n=== tests/test_repr_html.py: test_no_pandas_import_in_dataframe ===")
    p = Path("tests/test_repr_html.py")
    s = p.read_text()

    old = '''def test_no_pandas_import_in_dataframe():
    """AST: dataframe.py must not `import pandas` even after _repr_html_."""
    import ast
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pandas"), (
                    "dataframe.py must not import pandas — _repr_html_ "
                    "uses .to_pandas() method call only"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert not node.module.startswith("pandas"), (
                    "dataframe.py must not import pandas"
                )'''

    new = '''def test_no_pandas_import_in_dataframe():
    """AST: dataframe.py must not have MODULE-LEVEL `import pandas`.

    v0.9.1e-fix3 (§13.4/156): updated to module-level only.

    Lazy `import pandas as pd` inside a function (e.g., `_emit_group` in
    `_per_group_apply` — needed to construct pd.Series/pd.DataFrame from
    user fn output) is legitimate: it keeps `import cadw` light, pandas
    only loads when `_per_group_apply` is actually invoked. Only
    module-level import (eager pandas load on `import cadw`) is forbidden.

    Prior version walked the entire AST (`ast.walk`) — brittle for
    legitimate lazy imports. See §13.4/155, /156.
    """
    import ast
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod)
    tree = ast.parse(src)
    # Only check module-level nodes (top-level body), not nested.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pandas"), (
                    "dataframe.py must not MODULE-LEVEL import pandas "
                    "(lazy imports inside functions OK — see §13.4/156)"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert not node.module.startswith("pandas"), (
                    "dataframe.py must not MODULE-LEVEL import pandas"
                )'''

    s = replace_line_block(s, old, new, tag="test-repr-html-module-level")
    p.write_text(s)
    compile_check(p)
    print("  OK")


def append_decisions_156():
    print("\n=== DECISIONS.md: §13.4/156 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 156" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/156 (round 4 — blanket-walk AST guards fragile).** "
        s += "test_no_pandas_import_in_dataframe walked ast.walk; broke "
        s += "on legitimate lazy import. Fix: module-level only.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 156 (round 4 stale test — blanket-walk guard fragility).**
v0.9.1e-fix2 прошёл все guard-тесты, упал один **fast test**:
`test_repr_html.py::test_no_pandas_import_in_dataframe` — walks **весь
AST** и запрещает `import pandas` anywhere в `dataframe.py`. Legitimate
lazy import в `_emit_group` (fix2, needed to construct pd.Series/DataFrame)
— флаг.

**Тот же паттерн, 4-й раз:** §13.4/144 → /149 → /152 → **/156**.
Manual discipline + `assert_no_stale_behavior` (в fix2 искал **только**
`list_step_types` pattern) — не поймали.

**Корень (расширение §13.4/155):** file-wide AST guard'ы («no X
anywhere») — хрупки. Ломаются как только legit lazy X появляется.
`test_no_pandas_import_in_dataframe` и
`test_no_pandas_in_ops.py::test_no_pandas_in_dataframe_facade` — **два
теста в разных файлах** проверяют одно и то же; fix2 обновил один,
забыл другой. Дублирование guard'ов — source of drift.

**Fix (v0.9.1e-fix3):** `ast.Module.body` вместо `ast.walk` в
`test_no_pandas_import_in_dataframe`. Consistent с fix2 в
`test_no_pandas_in_ops.py`.

**Правило (§13.4/156, расширение 154, 155):**
- **AST guard'ы должны быть precise** — `tree.body` (module-level) для
  «import policy», `ast.walk` только для «anywhere at all is forbidden»
  (где это реально запрещено).
- **File-wide blanket guards** («no X in file Y») почти всегда
  хрупкие — уточняйте semantic: module-level / specific function /
  specific call pattern.
- **Не дублировать guard-тесты в разных файлах** — один canonical
  test, `import` helper'ы (например
  `_uses_pandas_import_at_module_level` из test_no_pandas_in_ops.py)
  для consistency.

**Backlog (§11.6):**
- Audit file-wide AST guards (`test_repr_html`, `test_no_pandas_in_ops`,
  `test_arrow_call_memory_guard`, `test_facade_has_no_as_py`) —
  убедиться, что каждый использует **precise** semantic.
- Consolidate duplicate guards (2 test_no_pandas_* — one canonical).
- Extend `assert_no_stale_behavior` patterns в каждом patch script:
  не только specific pattern, но и «все guards, что касаются файла
  X» — broader grep. `grep_and_warn(["import pandas"], ["tests/"])` в
  patch_v091e_fix2 поймал бы оба.

**Метаправило:** при **любом** изменении AST-свойства файла
(`import` policy, `from __future__` presence, module-level constants,
public API names) — `grep_and_warn` **по всем** тестам, не только по
одному известному pattern. «Какие тесты могут сломаться» — не «какой
я знаю», а «какие упоминают X».

**Связанные:** §13.4/144, /149, /152 (stale test repeats), /154
(def-prefix), /155 (module-level vs walk).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/156")


def main():
    print("=== v0.9.1e-fix3 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_stale_test()
    append_decisions_156()

    print("\n=== pytest: repr_html tests (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_repr_html.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REPR_HTML STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: guards regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_apply.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GUARDS REGRESSION FAILED ***")
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
        "v0.9.1e: per-group apply(fn) — composite-sort, one group in "
        "RAM, prototype-validated (11 cases); three modes (scalar / "
        "Series / DataFrame); per-group memory bound 2× ≤ 0.3×avail; "
        "§11.5 constraints; §13.4/154 (def-prefix helper check); "
        "§13.4/155 (pre-check before from_batches; module-level pandas); "
        "§13.4/156 (blanket-walk AST guard → module-level, round 4 of "
        "stale-test pattern)"])

    print("\n==> v0.9.1e-fix3 applied.")
    print("NEXT: v0.9.1f (pivot_table / melt / crosstab). Then tag v0.9.1.")


if __name__ == "__main__":
    main()