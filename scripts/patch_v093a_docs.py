#!/usr/bin/env python3
# v0.9.3a-docs: DECISIONS 13.4/173 + 13.4/174 + 11.5; CHANGELOG entry.
#
# Rules for THIS file (learned from 6 fix iterations):
#   - NO triple-quote anywhere. Content is list[str] + "\n".join().
#   - Single-line strings only. Cyrillic, backticks, section signs,
#     arrows - all safe in utf-8 source.
#   - Where text contains apostrophes, use double-quote outer for that
#     specific line.
#   - py_compile self before run.
#
# Run: uv run python scripts/patch_v093a_docs.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print("  $ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print("    " + line)
    if check and r.returncode != 0:
        print("    STDERR: " + r.stderr.strip()[-1500:])
        sys.exit(r.returncode)
    return r.returncode


def md(lines):
    return "\n".join(lines)


# ============ DECISIONS 13.4/173 + 13.4/174 ============

EPISODE_173_LINES = [
    "**Эпизод 173 (KFold O(N) fold_id без MemoryGuard; docstring lie).**",
    "v0.9.3a: P-CV1 S1 из v0.9.2 audit.",
    "",
    "**Bug.** `KFold.split` в v0.9.2e материализовал O(N) fold-id массив:",
    "",
    "- `shuffle=False`: `np.arange(n) % k` + `astype(int32)` - 12·N байт",
    "  (int32 fold_id + int64 arange temp).",
    "- `shuffle=True`:  + `rng.permutation(n)` - 20·N байт.",
    "",
    "**Ноль MemoryGuard involvement.** Для N >> RAM - guaranteed OOM,",
    "не `MemoryLimitExceeded`. Нарушение ядра гарантии CADW.",
    "",
    "**Docstring claim.** Pre-v0.9.3a: \"O(N) int8 fold_id array\".",
    "Actual code: int32 + int64 temporaries. 3-4x understatement.",
    "",
    "**Почему не поймали в v0.9.2e.** Тесты KFold: n=100-1000 (12 KB",
    "array - незаметно). Slow E2E: n=100k (1.2 MB на 50 MB budget -",
    "тоже OK). Никто не проверил N=1e8 с budget=16 MB - core user",
    "scenario. **Small-N hides O(N) bugs.**",
    "",
    "**Fix (v0.9.3a):**",
    "",
    "- `shuffle=False` (default): `fold_id = row_id % k` per batch.",
    "  O(batch) state. 1:1 с прежним поведением.",
    "- `shuffle=True` (default new): keyed splitmix64(row_id, seed) % k.",
    "  Deterministic per row_id. O(batch) state. Fold sizes ~binomial",
    "  (mean n/k, per-fold sigma ~= sqrt(n/k)). Split-quality.",
    "- `materialize=True` (opt-in): exact-balanced, O(N) RAM,",
    "  `check_state(20*N)` guarded.",
    "- `_attach_column_with_fn(fn: (offset, n) -> pa.Array)`: streaming",
    "  helper, O(batch) state.",
    "",
    "**Правило (13.4/173, расширение 158, 166, 172):**",
    "",
    "- **Docstring memory claims must match actual code.** Pre-commit",
    "  audit: grep все \"O(N)\"/\"O(K)\" утверждения в docstrings, сверить",
    "  с фактическими alloc.",
    "- **Small-N тесты скрывают O(N) bugs.** Каждый публичный op с O(N)",
    "  state должен иметь:",
    "  (a) AST guard \"no np.arange(N)/permutation(N) в default path\";",
    "  (b) functional test с tight budget + большим N;",
    "  (c) либо задокументированное O(N) state в 11.5 + fail-fast.",
    "",
    "**Связанные:** 13.4/158, 166, 172, **13.4/174**.",
    "",
]


EPISODE_174_LINES = [
    "**Эпизод 174 (patch-script embedded escape trap - 3 rounds).**",
    "v0.9.3a: 6 fix iterations before tests green - all from patch-script",
    "syntax bugs, not CADW code bugs.",
    "",
    "**Round 1 (v0.9.3a original).** CV_KEYED_TESTS triple-quoted",
    "template contained `\"_attach_column(\\n\"` - Python parsed `\\n` as",
    "escape -> 1 newline char -> written file had unterminated string",
    "literal -> SyntaxError on py_compile.",
    "",
    "**Round 2 (v0.9.3a-fix1).** Same class, different char: CHANGELOG",
    "entry had `r'''...'''` inside `'''...'''` outer -> `'''` terminated",
    "outer early -> parser error \"Statements must be separated by",
    "newlines or semicolons\".",
    "",
    "**Round 3 (v0.9.3a-fix2).** Docstring itself:",
    "  \"Fix2 uses `\"\"\"` for ALL outer triple-quoted strings\"",
    "The `\"\"\"` inside `\"\"\"...\"\"\"` terminated the docstring.",
    "",
    "**Round 4 (v0.9.3a-fix4).** Pattern match failure: fix4 looked for",
    "\"ASCII `->`\" but file had unicode `->` (U+2192) in docstring. `old",
    "not in s` -> abort. Classic test-authoring (13.4/166): pattern",
    "written from memory, not from file.",
    "",
    "**Round 5 (v0.9.3a-fix6).** 2 stale tests in test_cv.py still",
    "expected exact sizes [20,20,20,20,20] under shuffle=True -> failed",
    "[21,16,20,19,24] because v0.9.3a made default binomial. **13.4/160",
    "repeat:** behavior change should grep ALL tests before commit.",
    "",
    "**Rules (13.4/174):**",
    "",
    "1. **Patch scripts embedding Python code in template strings:**",
    "   - NEVER triple-quote for content. Build from `list[str]` +",
    "     `\"\\n\".join()`.",
    "   - If any content line has backslashes, keep it single-line",
    "     `'...'` or `\"...\"` (no `\\n`, `\\t`, `\\x..` in template).",
    "   - If content has `'''` or `\"\"\"` - escape or reword.",
    "   - **Best: `pathlib.Path.write_text()` from separate template",
    "     file** (v0.9.4 backlog: `patch_utils.write_content()`).",
    "",
    "2. **Pattern matching in `.replace()`:**",
    "   - NEVER include docstrings, comments, or unicode in the pattern.",
    "   - Match only signatures (def, class) + structural lines.",
    "   - If block contains docstring - line-based: `startswith` +",
    "     `endswith`, edit `lines[i] = new`.",
    "   - If not found: print actual context (nearby `def`s), не голый",
    "     \"not found\".",
    "",
    "3. **Self-compile check before run:**",
    "   - `uv run python -c \"import py_compile; py_compile.compile(",
    "     'scripts/patch_X.py', doraise=True)\"`",
    "   - Tooling fix (13.4/174 backlog): `patch_utils.assert_compiles(",
    "     path)` invoked by scaffold.",
    "",
    "4. **Split code and docs patches:**",
    "   - code+test commit: `v0.9.3a`.",
    "   - docs commit: `v0.9.3a-docs` (this).",
    "   - Reason: abort in docs template does NOT lose code commit.",
    "     Sister to 13.4/140 (CHANGELOG discipline).",
    "",
    "**Мета.** Tooling exists (`patch_utils.assert_no_stale_behavior`),",
    "но я его не вызвал в v0.9.3a. **Use it.**",
    "",
    "**Связанные:** 13.4/140, 144, 149, 152, 156, 160, 161, 166, 173.",
    "",
]


def append_episodes_13_4():
    print("\n=== DECISIONS 13.4/173 + 13.4/174 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    from cadw.patch_utils import find_section_anchor

    pos = find_section_anchor(s, "13.5")
    if pos is None:
        raise SystemExit("DECISIONS: section 13.5 not found")

    insert_parts = []
    added = []

    if "Эпизод 173" not in s:
        insert_parts.append(md(EPISODE_173_LINES))
        added.append("173")
    if "Эпизод 174" not in s:
        insert_parts.append(md(EPISODE_174_LINES))
        added.append("174")

    if not insert_parts:
        print("  both episodes present - skip")
        return

    block = "\n".join(insert_parts)
    s = s[:pos] + block + "\n" + s[pos:]
    p.write_text(s)
    print("  added: " + ", ".join(added))


# ============ DECISIONS 11.5 note ============

NOTE_11_5_LINES = [
    "**v0.9.3a (KFold keyed PRNG):**",
    "- `KFold(shuffle=True)` default = keyed splitmix64 on row_id. Fold",
    "  sizes ~binomial (mean n/k, per-fold sigma ~= sqrt(n/k)); not",
    "  exact-balanced. For exact-balanced use `materialize=True`",
    "  (opt-in, O(N) RAM, guarded via `check_state(20*N)`).",
    "- `random_state=None` -> seed=42 (deterministic reproducibility).",
    "- `KFold.split()` calls `df.count()` - one full scan, O(N) time,",
    "  bounded memory.",
    "- Each `split(df)` yields `n_splits` pairs; iterating all folds",
    "  does `n_splits * N` row touches (once per fold per side). O(1)",
    "  memory per iteration.",
    "",
]


def append_note_11_5():
    print("\n=== DECISIONS 11.5 KFold note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.3a (KFold keyed PRNG)" in s:
        print("  already present - skip")
        return
    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        raise SystemExit("DECISIONS: section 11.5 not found")
    note = md(NOTE_11_5_LINES)
    s = s[:pos] + note + "\n" + s[pos:]
    p.write_text(s)
    print("  OK")


# ============ CHANGELOG entry ============

CHANGELOG_LINES = [
    "### Исправлено (v0.9.3a - KFold keyed PRNG, S1 fix)",
    "",
    "**P-CV1 (S1, из v0.9.2 audit):** `KFold.split` материализовал",
    "O(N) fold-id массив (shuffle=False: ~12*N; shuffle=True: ~20*N)",
    "**без** `MemoryGuard.check_state`. Для N >> RAM - guaranteed OOM,",
    "не `MemoryLimitExceeded`. Нарушение ядра гарантии CADW.",
    "",
    "- **`KFold.shuffle=False`** (default): `fold_id = row_id % k`,",
    "  computed per batch. O(batch) state. 1:1 с прежним поведением.",
    "- **`KFold.shuffle=True`** (default, O(batch) state): keyed",
    "  splitmix64 PRNG. `fold_id = splitmix64(row_id*GOLDEN + seed) % k`.",
    "  Детерминистичен per row_id. Split-quality: fold sizes ~binomial",
    "  (mean n/k, per-fold sigma ~= sqrt(n/k)).",
    "- **`KFold(materialize=True)`** (opt-in backward compat):",
    "  exact-balanced shuffle, O(N) RAM, `check_state(20*N)` guarded.",
    "- **`random_state=None`** -> seed=42 (deterministic reproducibility).",
    "- **`_attach_column_with_fn(fn)`**: streaming helper",
    "  (`fn(offset, n) -> pa.Array`). O(batch) state, no O(N) array.",
    "",
    "### 13.4/173 - docstring memory claim vs actual allocation",
    "",
    "Pre-v0.9.3a docstring: \"O(N) int8 fold_id array\". Actual code:",
    "int32 fold_id + int64 arange/permutation temporaries = 12-20*N bytes.",
    "3-4x understatement + zero MemoryGuard check.",
    "",
    "**Правило:** docstring memory claim must match actual code. Test",
    "with N >> RAM (core scenario) обязателен для каждого публичного op",
    "с O(N) state - small-N тесты скрывают O(N) bugs.",
    "",
    "### 13.4/174 - patch-script embedded escape trap (3 rounds)",
    "",
    "v0.9.3a: 6 fix iterations, все - patch-script syntax bugs, не CADW:",
    "",
    "- `\\n` внутри triple-quote template -> real newline in file.",
    "- `'''` inside `'''...'''` -> terminated early.",
    "- `\"\"\"` inside `\"\"\"...\"\"\"` -> terminated early.",
    "- Pattern wrote `->` (ASCII), file had `->` (U+2192).",
    "- 2 stale tests after behavior change (13.4/160).",
    "",
    "**Rules:** no triple-quote in patch scripts; content as",
    "`\"\\n\".join(list)`. Pattern matches signatures + structural lines",
    "only, no docstrings/unicode. If not found - print context. Split",
    "code/docs patches. Self-compile-check before run.",
    "",
    "### Тесты (v0.9.3a)",
    "",
    "+11 fast (`test_cv_keyed_prng.py`): default folds, keyed PRNG",
    "deterministic, different seeds differ, default seed=42, binomial",
    "balance, no-shuffle unchanged, materialize exact-balanced, OOM",
    "guard, small-N OK, AST guard (no np.arange(N)/permutation(N) в",
    "default path), streaming helper present, large-N tight-budget.",
    "",
    "2 existing tests updated (`test_cv.py`): `shuffle=True` now binomial",
    "(use `materialize=True` for exact).",
    "",
    "- fast: **~1560** (+11). slow: **20** (unchanged).",
    "",
    "### Ограничения (v0.9.3a, 11.5)",
    "",
    "- `shuffle=True` (default): fold sizes ~binomial, not exact-balanced.",
    "- `KFold.split()`: one `df.count()` scan (O(N) time, O(1) memory).",
    "- Iterating all folds: `n_splits * N` row touches, O(1) memory per",
    "  iteration.",
    "",
]


def append_changelog():
    print("\n=== CHANGELOG v0.9.3a ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.3a - KFold keyed PRNG" in s:
        print("  already present - skip")
        return
    from cadw.patch_utils import append_unreleased_entry
    entry = md(CHANGELOG_LINES)
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


def main():
    print("=== v0.9.3a-docs ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    append_episodes_13_4()
    append_note_11_5()
    append_changelog()

    print("\n=== guards ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_patch_utils.py",
    ])

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "CHANGELOG.md", "DECISIONS.md"])
    sh(["git", "commit", "-m",
        "v0.9.3a-docs: DECISIONS 13.4/173 (KFold O(N) fold_id, docstring "
        "lie, small-N hides O(N)) + 13.4/174 (patch escape trap: \\n/'''/\"\"\"/"
        "unicode-pattern; rules for patch scripts) + 11.5 KFold keyed PRNG "
        "note; CHANGELOG v0.9.3a entry"])

    print("\n==> v0.9.3a complete (code + docs).")
    print("NEXT: v0.9.3b (P-S1 sort merge fan-in S2)")


if __name__ == "__main__":
    main()
