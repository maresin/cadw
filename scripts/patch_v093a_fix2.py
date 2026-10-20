#!/usr/bin/env python3
"""v0.9.3a-fix2: repair broken test; add CHANGELOG+DECISIONS; commit.

v0.9.3a-fix1 had a SyntaxError itself: CHANGELOG entry used
`raw-string prefixes (r'''...''')` inside `'''...'''` outer string.
The inner ''' terminated outer early → "Statements must be separated
by newlines or semicolons" from the parser.

Fix2 uses `"""` for ALL outer triple-quoted strings (content has no
`"""` sequence), avoiding the '''' vs ''' trap. Sister lesson to
\\n-in-triple-quote trap (§13.4/174 extended).
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


# ============ Fix broken test ============

def fix_broken_test():
    print("\n=== tests/test_cv_keyed_prng.py: fix embedded newline ===")
    p = Path("tests/test_cv_keyed_prng.py")
    s = p.read_text()

    # The broken block spans two physical lines (embedded \n became a
    # real newline). Match exact bytes.
    old = (
        '    assert "_attach_column_with_fn" in src\n'
        '    assert "_attach_column(df," in src or "_attach_column(\n'
        '" in src\n'
    )

    new = (
        '    assert "_attach_column_with_fn" in src\n'
        '    assert "_attach_column(df," in src\n'
    )

    if new in s and old not in s:
        print("  already fixed — skip")
        return
    if old not in s:
        raise SystemExit(
            "broken block not found; inspect "
            "tests/test_cv_keyed_prng.py around line 224"
        )
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ DECISIONS §13.4/173 + §11.5 ============

def append_decisions_173():
    print("\n=== DECISIONS.md: §13.4/173 + §11.5 KFold note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "Эпизод 173" not in s:
        from cadw.patch_utils import find_section_anchor
        pos = find_section_anchor(s, "13.5")
        if pos is None:
            print("  WARNING: §13.5 not found — append")
            s = s.rstrip() + "\n\n"
            s += "**§13.4/173 (KFold docstring lie; O(N) materialize).**\n"
            p.write_text(s)
        else:
            episode = """**Эпизод 173 (KFold O(N) fold_id без MemoryGuard;
docstring lie).** v0.9.3a: P-CV1 S1 из v0.9.2 audit.

Bug: KFold.split в v0.9.2e материализовал O(N) fold_id массив
БЕЗ MemoryGuard.check_state:
  - shuffle=False: 12·N байт (int32 fold_id + int64 arange temp)
  - shuffle=True:  20·N байт (+ int64 rng.permutation)
Для N ≫ RAM — guaranteed OOM, не MemoryLimitExceeded. Нарушение
ядра гарантии CADW.

Docstring claim: "O(N) int8 for fold_id array". Actual code: int32 +
int64 temporaries. 3-4× understatement.

Почему не поймали: тесты KFold n=100-1000 (12 KB array), slow e2e
n=100k (1.2 MB) на 50 MB budget — незаметно. Никто не проверил
N=10^8 с budget=16 MB (core user scenario). Small-N скрывает O(N)
bugs.

Fix (v0.9.3a):
- shuffle=False (default): fold_id = row_id % k per batch, O(batch).
- shuffle=True (default new): keyed splitmix64(row_id, seed) % k,
  O(batch). Fold sizes ~binomial (mean n/k, per-fold σ ≈ sqrt(n/k)).
- materialize=True (opt-in): old behavior + check_state(20·N).
- _attach_column_with_fn: streaming helper, fn(offset, n) → array.

**§13.4/174 — patch script embedded escape trap.** Original v0.9.3a
patch had `\\n` (escape) inside a quoted-string assertion, in a
triple-quoted template → written file had unterminated string.
v0.9.3a-fix1 had a sister bug: bare ''' inside a triple-single-quoted
outer string → terminated early → "Statements must be separated by
newlines or semicolons".

Правило: patch scripts that embed Python code in strings MUST:
- use `\"\"\"` outer triple-quote (not triple-single) when content
  may contain triple-single,
- OR use raw-string prefix (r...) when content has backslashes,
- OR write content via pathlib.Path.write_text() from separate file
  (no template at all — most robust).

Prevent: before commit, run `ast.parse(patch_file.read_text())` on
each patch script (or `python -m py_compile`) to catch this class
of bug. Tooling fix: `patch_utils.assert_compiles(path)`.

**Связанные:** §13.4/158, 166, 172.

"""
            s = s[:pos] + episode + s[pos:]
            p.write_text(s)
            print("  OK: §13.4/173 + §13.4/174 extended")

    # §11.5 note
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.3a (KFold keyed PRNG)" in s:
        print("  §11.5 already present — skip")
        return
    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        print("  WARNING: §11.5 not found — skip")
        return
    note = """**v0.9.3a (KFold keyed PRNG):**
- KFold(shuffle=True) default = keyed splitmix64 on row_id. Fold
  sizes ~binomial (mean n/k, per-fold σ ≈ sqrt(n/k)); not
  exact-balanced. For exact-balanced use materialize=True (opt-in,
  O(N) RAM, guarded via check_state(20·N)).
- random_state=None → seed=42 (deterministic reproducibility).
- KFold.split() calls df.count() — one full scan, O(N) time bounded
  memory.
- Iterating all folds: n_splits × N row touches (once per fold per
  side). O(1) memory per iteration.

"""
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK: §11.5 KFold")


def append_changelog():
    print("\n=== CHANGELOG.md: v0.9.3a entry ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.3a — KFold keyed PRNG" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = """### Исправлено (v0.9.3a — KFold keyed PRNG, S1 fix)

**P-CV1 (S1, из v0.9.2 audit):** KFold.split материализовал O(N)
fold-id массив (shuffle=False: ~12·N; shuffle=True: ~20·N) БЕЗ
MemoryGuard.check_state. Для N ≫ RAM — guaranteed OOM, не
MemoryLimitExceeded.

- **KFold.shuffle=False** (default): fold_id = row_id % k, computed
  per batch. O(batch) state. 1:1 с прежним поведением.
- **KFold.shuffle=True** (default, O(1) state): keyed splitmix64
  PRNG. fold_id = splitmix64(row_id * GOLDEN + seed) % k.
  Детерминистичен per row_id. Split-quality: fold sizes ~binomial
  (mean n/k, per-fold σ ≈ sqrt(n/k)).
- **KFold(materialize=True)** (opt-in backward compat): exact-
  balanced shuffle, O(N) RAM. Guarded via check_state(20·N).
- **random_state=None** → seed=42 (deterministic reproducibility).
- **_attach_column_with_fn** — streaming helper (fn(offset, n) →
  array). O(batch) state, no O(N) array.

### §13.4/173 — docstring memory claim vs actual allocation

Pre-v0.9.3a docstring: "O(N) int8 for fold_id array". Actual code:
int32 fold_id + int64 arange/permutation temporaries = 12-20·N bytes.
3-4× understatement + zero MemoryGuard check.

Правило: docstring memory claim must match actual code.

### Почему не поймали в v0.9.2e

Тесты KFold: n=100-1000. Slow e2e: n=100k. Small-N скрывает O(N)
bugs. Core user scenario (N ≫ RAM) не был протестирован.

### §13.4/174 — patch script embedded escape trap

v0.9.3a patch: `\\n` (escape) inside a quoted-string assertion in a
triple-quoted template → written file had unterminated string.
v0.9.3a-fix1: sister bug — bare triple-single-quote inside triple-
single-quoted outer string → terminated early → "Statements must be
separated by newlines or semicolons".

Правило: patch scripts that embed Python code in strings MUST use
`\"\"\"` outer triple-quote when content has triple-single, OR raw-
string prefix for backslashes, OR pathlib.write_text() from separate
file (no template). Prevent: py_compile each patch script before
commit.

### Тесты (v0.9.3a)

- +10 fast (test_cv_keyed_prng.py): default folds, keyed
  deterministic, different seeds differ, default seed=42, binomial
  balance, no-shuffle unchanged, materialize exact-balanced,
  materialize OOM guard, materialize small-N, AST guard, streaming
  helper presence, large-N tight-budget functional.

- fast: ~1559 (+10). slow: 20 (unchanged).

### Ограничения (v0.9.3a, §11.5)

- shuffle=True (default): fold sizes ~binomial, not exact-balanced.
  For exact — materialize=True (O(N) RAM, guarded).
- KFold.split() calls df.count() — one full scan.
- Iterating all folds: n_splits × N row touches.

"""
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.3a-fix2 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_broken_test()
    append_decisions_173()
    append_changelog()

    print("\n=== pytest: cv keyed prng ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_cv_keyed_prng.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: existing cv tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_cv.py",
         "tests/test_cv_e2e.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** EXISTING CV FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline + scalers regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_scalers.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
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

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.3a: KFold keyed PRNG — O(batch) state default; "
        "materialize=True opt-in (guarded check_state(20*N)); "
        "random_state=None->42; §13.4/173 (docstring lie, small-N "
        "hides O(N)); §13.4/174 (patch script escape trap)"])

    print("\n==> v0.9.3a applied.")
    print("NEXT: v0.9.3b (P-S1 sort merge fan-in S2)")


if __name__ == "__main__":
    main()
