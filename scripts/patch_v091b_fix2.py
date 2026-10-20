#!/usr/bin/env python3
"""v0.9.1b-fix2: stale test after fix1 removed `inputs=[]` restriction.

BUG (§13.4/144 REPEAT — round 2):
  `test_add_step_explicit_source_only_first` (written by patch_v091b_dag.py)
  asserted OLD (buggy) behavior: `inputs=[]` after first step → ValueError.
  Fix1 removed that restriction (multiple sources are DAG feature). Test
  not updated → "DID NOT RAISE".

  Same class as §13.4/144 (v0.8.11a-fix2: cross-file stale test after
  guard rename). This time: same-file, same patch introduced both code
  and test from the same (buggy) mental model. Fix1 fixed code, forgot
  to grep for tests encoding the old behavior. §13.4/144 discipline NOT
  applied on second opportunity → §13.4/149 addendum.

FIX:
  Replace test with `test_add_step_multiple_sources_allowed` — locks in
  CORRECT behavior (multiple sources valid on any step). Test derived
  from API contract (docstring "inputs=[]: explicit source, can be on
  ANY step"), not from implementation.

  Also: mechanical grep check in this fix — verify no other test in
  `test_pipeline_dag.py` / `test_pipeline.py` / `test_pipeline_e2e.py`
  asserts removed behavior.

GUARDS (before commit): test_pipeline_dag + backward compat + guards +
fast + slow.

Run: uv run python scripts/patch_v091b_fix2.py
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


# ============ Fix: stale test ============

def fix_stale_test():
    print("\n=== tests/test_pipeline_dag.py: replace stale test ===")
    p = Path("tests/test_pipeline_dag.py")
    s = p.read_text()

    old = '''def test_add_step_explicit_source_only_first():
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    with pytest.raises(ValueError, match="explicit source.*first"):
        pipe.add_step("b", DropnaStep(), inputs=[])'''

    new = '''def test_add_step_multiple_sources_allowed():
    """v0.9.1b-fix2 (§13.4/149): inputs=[] valid on ANY step.

    Multiple sources from raw is the DAG feature (fanout). Prior
    v0.9.1b encoded incorrect validation "explicit source only for
    first step" — removed in v0.9.1b-fix1. This test locks in the
    CORRECT behavior, derived from the API contract (docstring
    `add_step`: "inputs=[]: explicit source ... can be on ANY step"),
    not from implementation.
    """
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    # No raise — inputs=[] on non-first step is valid (source).
    pipe.add_step("b", DropnaStep(), inputs=[])
    graph = pipe._build_graph()
    assert graph["a"] == []
    assert graph["b"] == []  # both source from raw df'''

    s = replace_line_block(s, old, new, tag="stale-test-fix")
    p.write_text(s)
    print("  OK")


# ============ Mechanical grep check ============

def grep_removed_behavior():
    print("\n=== grep for removed behavior in test files ===")
    patterns = [
        "explicit source.*first",   # old error message
        "explicit source only",     # alt wording
    ]
    found = []
    for pat in patterns:
        r = subprocess.run(
            ["grep", "-rn", pat,
             "tests/test_pipeline.py",
             "tests/test_pipeline_dag.py",
             "tests/test_pipeline_e2e.py"],
            capture_output=True, text=True,
        )
        if r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                found.append(line)
    if found:
        print(f"  WARNING: {len(found)} references found:")
        for line in found:
            print(f"    {line}")
    else:
        print("  OK: no other stale assertions")


# ============ §13.4/149 ============

def append_decisions_149():
    print("\n=== DECISIONS.md: §13.4/149 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 149" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append via warning")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/149 (stale test after upstream change, round 2).** "
        s += "Same class as §13.4/144 — fix changed behavior, test still "
        s += "asserted old behavior. Discipline not applied twice.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 149 (§13.4/144 REPEAT — round 2, same-file stale test).**
v0.9.1b-fix1 удалил validation `inputs=[] → error after first step`
(правильно: multiple sources — DAG feature). Но `test_add_step_explicit_
source_only_first` (написан в том же `patch_v091b_dag.py`, том же файле
тестов) продолжал **утверждать** старое поведение → 1 fail.

**Почему round 2:**
- v0.8.11a-fix2 (§13.4/144): cross-file stale test (guard rename).
  Я установил правило: «grep `<old_behavior> tests/` before commit».
- v0.9.1b-fix1: удалил validation, не grep'нул. Второй раз та же
  ошибка.

**Специфика this round:** patch_v091b_dag.py сам создал и код (buggy
validation), и тест (`test_add_step_explicit_source_only_first`), который
эту buggy validation проверял. **Одна mental model** — тест слепо
воспроизводит то, что код говорит, вместо того чтобы проверять
намерение (design intent). Тест «прошёл» в v0.9.1b → баг закоммичен.

**Fix (v0.9.1b-fix2):** заменить тест на `test_add_step_multiple_sources_
allowed` — проверяет правильное поведение. **Derived from docstring
API contract**, не из implementation.

**Механический grep в этом фиксе:** проверить другие тесты на отголоски
старого сообщения об ошибке (`explicit source only`, `explicit source.*first`).

**Правило (усиление §13.4/144):** перед коммитом patch, который
**удаляет или меняет** behavior (validation, error message, guard name,
return contract) — обязательный grep по `tests/`:
```python
# Пример:
r = subprocess.run(["grep", "-rn", old_error_pattern, "tests/"], ...)
if r.stdout.strip(): log_warning()
```
Это **не** автоматический fail — но **явный warning** перед `git add`.
Если warning сработал — разобраться до коммита.

**Дополнительно (§13.4/148 связано):** тесты, написанные **из той же
mental model**, что и код, не ловят ошибок этой mental model. Правильно:
писать тесты **из docstring API contract** (что должно быть), не из
implementation (что есть). Но это требует независимого автора (человек/
LLM после паузы/ревью) — в одной сессии я сам себя не проверю.

**Backlog (§11.6, v0.9.2):** AST guard `test_removed_behavior_stale.py`
— парсит `scripts/patch_*.py` на предмет удалённых error messages /
guard names; grep'ает `tests/` на совпадения; warning (не fail) в
pytest output.

**Связанные:** §13.4/144 (cross-file stale test), §13.4/146 (partial
update — validation not updated with handler), §13.4/148 (design
intent vs validation).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/149")


# ============ main ============

def main():
    print("=== v0.9.1b-fix2 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_stale_test()
    grep_removed_behavior()
    append_decisions_149()

    print("\n=== pytest: DAG tests (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pipeline_dag.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** DAG TESTS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: backward compat + E2E ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py", "tests/test_pipeline_e2e.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** BACKWARD COMPAT FAILED ***")
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
        "v0.9.1b: DAG Pipeline — add_step(inputs=[...]), fan-in via "
        "_merge_fanout (column union, same-row precondition), "
        "topological execution (fit/transform/predict), LambdaTransform "
        "test helper, DAG-aware _repr_html_; §11.5 fan-in MVP note; "
        "§13.4/148 (multiple sources in DAG); §13.4/149 (stale test after "
        "removal — round 2 of §13.4/144)"])

    print("\n==> v0.9.1b-fix2 applied.")
    print("NEXT: v0.9.1c (save/load to_json/from_json).")


if __name__ == "__main__":
    main()