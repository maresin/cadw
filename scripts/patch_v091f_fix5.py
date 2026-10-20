#!/usr/bin/env python3
"""v0.9.1f-fix5: stale regex miss (round 6) + AST substring (round 7).

BUG 1 (§13.4/144→...→/161, round 6):
  fix4 updated regex `K_pairs` → `pivot_table` in
  `test_pivot_table_e2e.py`. Missed `test_pivot_table.py::test_pivot_too_
  large_raises` (same `match="K_pairs"`). 1 test fails.

  Dogfood grep in fix4 CAUGHT this (printed warning), but used
  `grep_and_warn` (non-fatal) not `assert_no_stale_behavior` (fatal).
  Pytest caught it later — but 1 wasted round.

BUG 2 (§13.4/161, AST substring):
  `test_pivot_table_has_check_state_before_to_pandas` uses:
      `src.index("check_state") < src.index("to_pandas")`
  After fix4, docstring mentions "to_pandas" BEFORE the actual code
  check_state call → src.index finds docstring at pos 1309 instead of
  code at pos 6430. AST test fails on correct code.

  Same class as §13.4/143 (substring vs token), /154 (def-prefix),
  /156 (blanket-walk). Test uses substring, not AST node.

FIX:
  1. test_pivot_table.py::test_pivot_too_large_raises:
     regex `K_pairs` → `pivot_table`.
  2. test_pivot_table_has_check_state_before_to_pandas:
     parse AST, find Call nodes check_state and to_pandas, compare
     min lineno. Skip docstring entirely.
  3. §13.4/161: (a) round 6 of stale pattern, (b) "use fatal
     assert_no_stale_behavior, not warning grep_and_warn"; (c) AST
     tests must parse nodes, not strings.

META: dogfood grep in fix4 found the bug — tooling works. But wrong
tool (warning) instead of fatal. Two-layer failure: non-fatal grep +
manual pytest.

GUARDS: pivot fast + E2E + guards + fast + slow. Include
`assert_no_stale_behavior` (fatal, not warning) in this fix.

Run: uv run python scripts/patch_v091f_fix5.py
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


# ============ Fix 1: stale regex in test_pivot_table.py ============

def fix_stale_regex():
    print("\n=== tests/test_pivot_table.py: regex K_pairs → pivot_table ===")
    p = Path("tests/test_pivot_table.py")
    s = p.read_text()

    old = '''    df._memory_guard = MemoryGuard(max_memory_bytes=2000)
    with pytest.raises(MemoryLimitExceeded, match="K_pairs"):
        df.pivot_table(
            index=["row"], columns="col", values="v", aggfunc="sum",
        )'''

    new = '''    df._memory_guard = MemoryGuard(max_memory_bytes=2000)
    with pytest.raises(MemoryLimitExceeded, match="pivot_table"):
        df.pivot_table(
            index=["row"], columns="col", values="v", aggfunc="sum",
        )'''

    s = replace_line_block(s, old, new, tag="test-pivot-stale-regex")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2: AST-based check_state/test_pandas order ============

def fix_ast_check_state_test():
    print("\n=== tests/test_pivot_table.py: AST-based check_state order ===")
    p = Path("tests/test_pivot_table.py")
    s = p.read_text()

    old = '''def test_pivot_table_has_check_state_before_to_pandas():
    """AST: pivot_table has check_state BEFORE to_pandas."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.pivot_table)
    assert "to_pandas" in src
    assert "check_state" in src
    assert src.index("check_state") < src.index("to_pandas"), (
        "pivot_table: check_state must precede to_pandas (§3.11)"
    )'''

    new = '''def test_pivot_table_has_check_state_before_to_pandas():
    """AST: pivot_table has check_state CALL before to_pandas CALL.

    §13.4/161: parse AST nodes (Call func.attr). Docstring mentions
    "to_pandas" earlier than code — substring `.index()` would match
    docstring, not code. AST finds actual Call nodes.
    """
    import ast
    import inspect
    import textwrap
    from cadw.dataframe import CadwDataFrame

    src = textwrap.dedent(inspect.getsource(CadwDataFrame.pivot_table))
    tree = ast.parse(src)

    check_state_linenos: list[int] = []
    to_pandas_linenos: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute):
            if f.attr == "check_state":
                check_state_linenos.append(node.lineno)
            elif f.attr == "to_pandas":
                to_pandas_linenos.append(node.lineno)

    assert check_state_linenos, "pivot_table: no check_state call"
    assert to_pandas_linenos, "pivot_table: no to_pandas call"

    first_check = min(check_state_linenos)
    first_pandas = min(to_pandas_linenos)
    assert first_check < first_pandas, (
        f"pivot_table: first check_state call (line {first_check}) must "
        f"precede first to_pandas call (line {first_pandas}) (§3.11). "
        f"Note: docstrings are skipped — we check AST Call nodes, not "
        f"substrings (§13.4/161)."
    )'''

    s = replace_line_block(s, old, new, tag="ast-check-state-order")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 3: §13.4/161 ============

def append_decisions_161():
    print("\n=== DECISIONS.md: §13.4/161 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 161" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/161 (round 6 stale regex; AST substring).** "
        s += "dogfood grep found it but was non-fatal. AST test uses "
        s += "src.index instead of ast.parse.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 161 (round 6 stale regex + AST substring, layered failure).**
v0.9.1f-fix4 обновил regex `K_pairs` → `pivot_table` в
`test_pivot_table_e2e.py`, но **забыл** `test_pivot_table.py::test_pivot_
too_large_raises` (тот же `match="K_pairs"`). Плюс `test_pivot_table_has_
check_state_before_to_pandas` использует `src.index()` — после fix4
docstring упоминает `to_pandas` **раньше** кода → test на substring, не
на AST node.

**Layered failure (три независимых источника):**

1. **Manual discipline 6-й раз** (§13.4/144 → /149 → /152 → /156 →
   /160 → /161): forgot to grep `test_pivot_table.py`.
2. **Dogfood grep found it — но non-fatal.** fix4 вызвал
   `grep_and_warn` (warning, не exit) вместо `assert_no_stale_behavior`
   (fatal). Tooling **сработал**, но wrong mode → script продолжился.
3. **AST test на substring** вместо AST node — после docstring change
   test false-fails на правильном коде.

**Fix (v0.9.1f-fix5):**
- Stale regex `K_pairs` → `pivot_table` в test_pivot_table.py.
- AST test — `ast.parse(inspect.getsource(...))`, найти `Call(func.attr=
  "check_state")` и `Call(func.attr="to_pandas")`, сравнить `lineno`.
  Docstring не парсится как Call.
- **Dogfood в этом fix — `assert_no_stale_behavior`** (fatal), не
  `grep_and_warn`. Must pass BEFORE commit.

**Правило (§13.4/161, три части):**
1. **When a fix changes behavior used by multiple tests** — `grep -rFn`
   **все** test files, не только те, что менял. Список изменённых файлов
   ≠ список affected files.
2. **Use `assert_no_stale_behavior` (fatal), not `grep_and_warn`
   (warning).** Warning-mode = "showed up, I'll fix later" = 0 iterations
   saved. Fatal-mode = "script stops, must fix now" = discipline enforced.
3. **AST tests must `ast.parse` — never `src.index()`.** `src.index`
   matches docstrings, comments, and string literals before code. Class
   of §13.4/143 (test substring vs token), /154 (def-prefix), /156
   (blanket-walk). Use AST for structural checks.

**Мета (усиление 160):** when tooling catches a bug **but is
non-fatal** — the discipline still fails (script continues, bug
propagates to pytest). **Fatal-by-default**: for any grep of patterns
known to be stale-after-behavior-change, use `assert_no_stale_behavior`.
Reserve `grep_and_warn` for exploratory diagnostics, never for patch
gates.

**Backlog (§11.6):** `scaffold_patch.py` — enforce `assert_no_stale_
behavior` (fatal) as mandatory step. Without scaffold, this decision
repeats 6+ times.

**Связанные:** §13.4/144, /149, /152, /156, /160 (stale repeats),
/143 (substring vs token), /154 (def-prefix), /156 (blanket-walk).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/161")


# ============ Fix 4: dogfood with FATAL assert_no_stale_behavior ============

def dogfood_fatal():
    """Dogfood: assert_no_stale_behavior (FATAL) for K_pairs."""
    print("\n=== dogfood (fatal): assert_no_stale_behavior 'K_pairs' ===")
    from cadw.patch_utils import assert_no_stale_behavior
    assert_no_stale_behavior(
        patterns=["K_pairs"],
        dirs=[
            "tests/test_pivot_table.py",
            "tests/test_pivot_table_e2e.py",
        ],
        label="K_pairs-stale",
    )
    print("  OK: no 'K_pairs' references in pivot tests")


def main():
    print("=== v0.9.1f-fix5 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_stale_regex()
    fix_ast_check_state_test()
    append_decisions_161()

    # Fatal dogfood BEFORE test run
    dogfood_fatal()

    print("\n=== pytest: pivot fast (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pivot_table.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIVOT FAST STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: pivot E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pivot_table_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
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
        "v0.9.1f: pivot_table / melt / crosstab — long↔wide; "
        "§13.4/157-161 (pa.types.is_*; E2E respects complexity; dense "
        "pre-check; state_limit_bytes; to_pandas pre-check ×3; stale "
        "regex round 6; AST-based check_state order; fatal "
        "assert_no_stale_behavior)"])

    print("\n==> v0.9.1f-fix5 applied. v0.9.1 complete.")
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