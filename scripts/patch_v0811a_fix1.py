#!/usr/bin/env python3
"""v0.8.11a-fix1: fix test substring match + CHANGELOG skip false positive.

BUG 1 (§13.4/143 — test authoring, substring vs token):
  test_dedup_removes_stopgap_5x asserted:
      assert "5 * table.nbytes" not in src
  But the CORRECT new code contains:
      check_state(1.5 * table.nbytes, ...)
  And "1.5 * table.nbytes" CONTAINS "5 * table.nbytes" as substring.
  Substring `in` doesn't respect token boundaries.

  Fix: check for the old guard NAME `dedup.pandas_chain` (v0.8.9
  stopgap), which is not a substring of any new code.

BUG 2 (§13.4/140 lesson — guard compatibility assumed):
  CHANGELOG append had:
      if "Arrow-native dedup" in s or "P-D1" in s and "closed" in s: skip
  This matched because v0.8.9 CHANGELOG entry contains both "P-D1" and
  "closed" (in different contexts). Result: v0.8.11a entry was SILENTLY
  SKIPPED.

  Also: after v0.8.10 tag, CHANGELOG has no `## [Unreleased]` header
  (squash renamed it to `[0.8.10]`). Anchor path would fail.

  Fix: (a) check specific marker `### Исправлено (safety — P-D1 close)`,
  (b) if `## [Unreleased]` missing, prepend it before `## [0.8.10]`,
  (c) append entry under it.

META (§13.4/143 — round 5 of the "fix creates new bug" cluster):
  - v0.8.9: AST guard too broad (Table vs Dataset)
  - v0.8.9-fix: pc.cast overflow / pa.field nullable defaults
  - v0.8.9-fix2: stale test (uint64 fallback removed, test not updated)
  - v0.8.10-pre: [0.8.10-pre] broke changelog guards
  - v0.8.11a: test substring (5* vs 1.5*) + CHANGELOG skip false positive

  Rule: any patch that adds `assert "X" in/not in src` MUST:
  1. Run the test BEFORE `git commit`.
  2. Verify X is a unique marker (function name, comment, guard name),
     not a code fragment that can be a substring of the fix.

Run: uv run python scripts/patch_v0811a_fix1.py
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


# ============ Fix 1: test substring ============

def fix_test_substring():
    print("\n=== tests/test_dedup_arrow_native.py::test_dedup_removes_stopgap_5x ===")
    p = Path("tests/test_dedup_arrow_native.py")
    s = p.read_text()

    old = '''def test_dedup_removes_stopgap_5x():
    """AST: no `5 * table.nbytes` stopgap in _local_dedup (now 1.5×)."""
    import ast
    import inspect
    from cadw.ops import dedup as dedup_mod
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "5 * table.nbytes" not in src, (
        "v0.8.11a: 5×T stopgap must be replaced with 1.5×T Arrow-native bound"
    )
    # 1.5× must be present
    assert "1.5 * table.nbytes" in src, (
        "v0.8.11a: expect `1.5 * table.nbytes` Arrow-native bound"
    )'''

    new = '''def test_dedup_removes_stopgap_5x():
    """AST: `_local_dedup` uses Arrow-native guard name (v0.8.11a).

    v0.8.11a-fix1 (§13.4/143): check specific OLD guard NAME
    `dedup.pandas_chain`, NOT the arithmetic pattern `5 * table.nbytes` —
    the latter is a substring of `1.5 * table.nbytes` (the CORRECT new
    code), so `"5 * table.nbytes" not in src` fails even after fix.
    Substring `in` doesn't respect token boundaries.
    """
    import inspect
    from cadw.ops import dedup as dedup_mod
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "dedup.pandas_chain" not in src, (
        "v0.8.11a: old pandas_chain stopgap name must be removed"
    )
    assert "dedup.arrow_native" in src, (
        "v0.8.11a: expect `dedup.arrow_native` guard name"
    )
    assert "1.5 * table.nbytes" in src, (
        "v0.8.11a: expect `1.5 * table.nbytes` Arrow-native bound"
    )'''

    s = replace_line_block(s, old, new, tag="test-stopgap-fix")
    p.write_text(s)
    compile_check(p)
    print("  OK: specific markers (pandas_chain → arrow_native)")


# ============ Fix 2: CHANGELOG append ============

def fix_changelog_append():
    print("\n=== CHANGELOG.md [Unreleased]: append v0.8.11a ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()

    # Specific marker for v0.8.11a presence
    if "### Исправлено (safety — P-D1 close)" in s:
        print("  v0.8.11a entry already present — skip")
        return

    # Ensure [Unreleased] header exists; if not, prepend before [0.8.10]
    if "## [Unreleased]\n" not in s:
        m = re.search(r'^## \[0\.8\.10\]', s, re.MULTILINE)
        if not m:
            print("  ERROR: cannot find [0.8.10] header — skip")
            return
        s = s[:m.start()] + "## [Unreleased]\n\n" + s[m.start():]
        print("  added `## [Unreleased]` header before [0.8.10]")

    entry = '''## [Unreleased]

### Исправлено (safety — P-D1 close)

- **P-D1 (S2, live): Arrow-native `drop_duplicates` rewrite.** pandas
  chain (`to_pandas → sort_values → drop_duplicates → drop → from_pandas`)
  в `_local_dedup` заменён на Path C: `dictionary_encode(subset)` +
  `group_by(subset).aggregate(min/max row_id)` + `is_in` filter.
  Удалён `import pandas` из `cadw/ops/dedup.py`.

  **Bench (prototype `scripts/proto_dedup_arrow.py`):**
  | n | pandas | Path B (arrow) | **Path C (dict)** |
  |---|---|---|---|
  | 100k | 9.3 MB | 10.2 MB | **5.9 MB** |
  | 300k | 29.3 MB | 39.0 MB | **3.6 MB** |
  | 1M   | 87.7 MB | 47.2 MB | **10.9 MB** |

  avg peak(C)/peak(A) = **0.29**. Correctness A==B==C across keep ∈
  {first, last, False} × selectivity ∈ {10%, 50%, 90% dupes}.

  **Stopgap removed:** `check_state(5 × table.nbytes)` → `check_state(1.5 ×
  table.nbytes)` (honest Arrow-native bound, proto-backed). Effective
  partition limit back from `avail × 0.1` to `avail × 0.33`.

- **Dead pandas helpers removed из `cadw/ops/join.py`.** `import pandas
  as pd`, `_drop_null_keys` (superseded by `_drop_null_keys_arrow` в
  v0.8.5.3) — удалены. Zero pandas in ops except `shuffle.py::_pandas_hash_column`
  (documented perf exception, §9.2.20).

### Добавлено

- **`tests/test_no_pandas_in_ops.py`** — AST guard: `import pandas` в
  `cadw/ops/*` разрешён только в `shuffle.py` (whitelist + rationale).
  Также `test_dedup_uses_arrow_native` (Path C marker).
- **`tests/test_dedup_arrow_native.py`** — 16 behavior tests: first/last/
  False, nulls в subset/value, multi-subset, int key, empty, K=N, K=1,
  pandas-equivalence, oversize → raise, dedup→sort chain, filtered dedup.

### Ограничения (v0.8.11a, §11.5)

- `drop_duplicates` порядок вывода: partition order, без
  `sort_values("row_id")` внутри partition (был в pandas-версии).
  Set-equivalence сохранена; для детерминированного порядка — `.sort_values(...)` после.
- `drop_duplicates` recursion при skew — не реализован (как в v0.3.0).
  Oversize partition → MemoryLimitExceeded. Backlog.

### Тесты

- fast: **~1190** (+16 behavior, +5 AST guard). slow: **8**.

'''

    s = s.replace("## [Unreleased]\n\n", entry, 1)
    p.write_text(s)
    print("  OK: v0.8.11a entry appended")


# ============ Fix 3: DECISIONS §13.4/143 ============

def append_decisions_143():
    print("\n=== DECISIONS.md: §13.4/143 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 143" in s:
        print("  already present — skip")
        return

    marker = "### 13.5. Открытые вопросы"
    if marker not in s:
        print("  WARNING: §13.5 marker not found — skip")
        return

    episode = '''**Эпизод 143 (test substring ≠ token; CHANGELOG skip false positive).**
Пятый подряд раунд кластера §13.4/139-142: fix (или его тест) вносит
свежий баг.

**Баг 1 (test):** `test_dedup_removes_stopgap_5x` проверял
`assert "5 * table.nbytes" not in src`. Правильный новый код содержит
`check_state(1.5 * table.nbytes, ...)` — и `"1.5 * table.nbytes"`
содержит `"5 * table.nbytes"` как **подстроку**. Python `in` не уважает
границы токенов, тест падает на правильном коде.

**Баг 2 (CHANGELOG):** skip-условие
`if "Arrow-native dedup" in s or "P-D1" in s and "closed" in s: skip`
матчит v0.8.9 entry (там есть `P-D1` и `closed` в разных контекстах).
Результат: v0.8.11a CHANGELOG entry молча пропущен. Плюс: после v0.8.10
tag `[Unreleased]` header не существует (squash переименовал в
`[0.8.10]`) — anchor-path не сработал бы в любом случае.

**Fix (v0.8.11a-fix1):** (a) test проверяет **уникальный маркер** —
`"dedup.pandas_chain"` (имя старого guard'а) вместо `5 * table.nbytes`;
(b) CHANGELOG — проверить специфичный маркер
`### Исправлено (safety — P-D1 close)`, создать `[Unreleased]` header
если отсутствует, затем append.

**Правило (расширение 139+140):** любой `assert "X" in/not in src`
должен пройти проверку:
1. **X — уникальный маркер** (имя функции/guard'а, тег комментария),
   не code fragment, который может быть подстрокой правильного кода.
2. **Guard-тесты прогоняются в том же patch-скрипте до `git commit`**
   (см. §13.4/140). Не после.

Если X — code fragment (арифметика, вызов), использовать
**word-boundary regex** `r'\\b' + re.escape(X) + r'\\b'` или специфичное
имя. `1.5 * table.nbytes` vs `5 * table.nbytes` — учебный пример: 1
символ (`.`), и подстрока ломает тест.

**Кластер (5 раундов):**
| раунд | источник | фикс |
|---|---|---|
| v0.8.9 | AST guard broad (Table vs Dataset) | whitelist + `_is_table_from_batches` |
| v0.8.9-fix | pc.cast overflow / pa.field nullable | `safe=False` + expected +1 byte |
| v0.8.9-fix2 | stale test (uint64) | rename + assertion update |
| v0.8.10-pre | `[0.8.10-pre]` vs `[Unreleased]` | rename to `[Unreleased]` |
| **v0.8.11a** | **test substring / CHANGELOG skip** | **specific markers** |

**Связанные:** §13.4/128 (expectations authored), /129 (reference helpers),
/139 (fix semantics = REPL), /140 (pre-release), /141 (prototype-first),
/142 (stopgap→rewrite). Класс «assumed compatibility, not verified».

'''
    s = s.replace(marker, episode + marker, 1)
    p.write_text(s)
    print("  OK: §13.4/143")


# ============ main ============

def main():
    print("=== v0.8.11a-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_test_substring()
    fix_changelog_append()
    append_decisions_143()

    # Full guard run BEFORE amend (§13.4/140+143)
    print("\n=== pytest: new tests (must be green) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dedup_arrow_native.py",
         "tests/test_no_pandas_in_ops.py",
         "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py",
         "tests/test_arrow_call_memory_guard.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GUARD FAILURE ***")
        sys.exit(1)

    print("\n=== pytest: regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_drop_duplicates.py",
         "tests/test_join.py", "tests/test_join_streaming.py",
         "tests/test_join_oom_guard.py", "tests/test_e2e_join.py",
         "tests/test_merge_planner.py", "tests/test_join_rss_warning.py",
         "-q"],
        timeout=600,
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

    # Amend the WIP commit — one clean commit for v0.8.11a
    sh(["git", "add", "-A"])
    sh(["git", "commit", "--amend", "-m",
        "v0.8.11a: Arrow-native dedup (P-D1 close) — Path C "
        "(dictionary_encode + group_by + is_in), 0.29× peak vs pandas; "
        "pandas removed from ops hot path; AST guard no_pandas_in_ops; "
        "§13.4/143 (test substring vs token)"])

    print("\n==> v0.8.11a-fix1 applied. Amended WIP commit.")
    print("NEXT: v0.8.11b (strict join conditional).")


if __name__ == "__main__":
    main()