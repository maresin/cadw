#!/usr/bin/env python3
"""v0.8.10-pre-fix: replace [0.8.10-pre] with [Unreleased] in CHANGELOG.

ROOT CAUSE (this is round 4 of the "fix creates a new bug" cluster —
§13.4/139 addendum):
  `patch_v0810_pre_join_offset.py::insert_changelog()` inserted
  `## [0.8.10-pre] — 2026-10-20` at top of CHANGELOG. That broke two
  guards:
  1. `test_changelog_is_descending::_key` regex is
     `^(\\d+)\\.(\\d+)\\.(\\d+)([a-z]?)$` — `-pre` suffix (dash) doesn't
     match, KeyError.
  2. `test_changelog_has_latest_tag`: top [0.8.10-pre] != HEAD tag
     [v0.8.9] (HEAD still on tag, commit not created because fast failed
     before git commit).

FIX (v0.8.10-pre-fix):
  Pre-release versioning goes under `## [Unreleased]` (Keep a Changelog
  convention), which the `^## \\[(\\d[^\\]]*)\\]` regex skips entirely
  (starts with letter U, not digit). This means:
  - `[Unreleased]` doesn't appear in `versions` list → doesn't break
    descending guard.
  - `versions[0]` stays `[0.8.9]` == tag → has_latest_tag PASSES (on
    the next commit that moves HEAD off the tag, the test skips).
  - At v0.8.10 tag time: rename `[Unreleased]` → `[0.8.10]` in one
    commit, then tag → both guards fire correctly for v0.8.10.

  Also patches `patch_v0810_pre_join_offset.py::insert_changelog()` for
  future runs, so this particular anchor bug never re-emerges.

META (§13.4/140, addendum to 139):
  This is not a fix-bug — it's a bug in the *original patch script*,
  which was never run against the changelog guards before commit. Rule
  extension: any patch that touches CHANGELOG must run `test_changelog_*`
  *before* the commit block in its own main(), not after. Same class as
  §13.4/128 (test expectations authored) — here "guard compatibility
  assumed, not verified".

Run: uv run python scripts/patch_v0810_pre_fix.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
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


# ============ Fix 1: CHANGELOG [0.8.10-pre] → [Unreleased] ============

def fix_changelog_header():
    print("\n=== CHANGELOG.md: [0.8.10-pre] → [Unreleased] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()

    if "## [Unreleased]" in s and "## [0.8.10-pre]" not in s:
        print("  already patched — skip")
        return

    old = "## [0.8.10-pre] — 2026-10-20"
    new = "## [Unreleased]"
    if old not in s:
        print(f"  WARNING: anchor {old!r} not found — skip")
        return
    s = s.replace(old, new, 1)
    p.write_text(s)
    print(f"  OK: {old} → {new}")


# ============ Fix 2: patch the source script for future runs ============

def fix_source_script():
    print("\n=== scripts/patch_v0810_pre_join_offset.py: [Unreleased] "
          "(for future runs) ===")
    p = Path("scripts/patch_v0810_pre_join_offset.py")
    if not p.exists():
        print("  source script not present — skip (likely removed)")
        return
    s = p.read_text()

    if '## [Unreleased]' in s and '## [0.8.10-pre]' not in s:
        print("  already patched — skip")
        return

    # In insert_changelog(), the entry starts with `## [0.8.10-pre] — 2026-10-20`.
    # Also there's the marker check `if "## [0.8.10-pre]" in s: return`.
    old_marker = 'if "## [0.8.10-pre]" in s:'
    new_marker = 'if "## [Unreleased]" in s:'
    if old_marker in s:
        s = s.replace(old_marker, new_marker, 1)

    old_header = '## [0.8.10-pre] — 2026-10-20'
    new_header = '## [Unreleased]'
    if old_header in s:
        s = s.replace(old_header, new_header, 1)

    # Section title in the entry ("Добавлено (safety follow-up, P-A3 addendum)")
    # can stay — no version-reference.
    p.write_text(s)

    # Compile-check the script (it's a plain Python file)
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{p}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  COMPILE FAIL: {p}")
        print(r.stderr[-800:])
        sys.exit(1)
    print("  OK")


# ============ Fix 3: DECISIONS §13.4/140 addendum to 139 ============

def append_decisions_140():
    print("\n=== DECISIONS.md: §13.4/140 (pre-release versioning vs guard) ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "§13.4/140" in s:
        print("  already present — skip")
        return

    # Anchor: end of §13.4/139 addendum (see v0.8.9-fix2). Find it by tail.
    anchor = (
        "Это защищает от ровно того класса, что случился трижды подряд."
    )
    if anchor not in s:
        print("  WARNING: §13.4/139 anchor not found — append at 13.5 boundary")
        anchor = "### 13.5. Открытые вопросы"
        addendum = '''**Эпизод 140 (pre-release versioning vs CHANGELOG guard).** Продолжение
кластера 139: `patch_v0810_pre_join_offset.py` вставил в CHANGELOG
строку `## [0.8.10-pre] — 2026-10-20`. Уронил два guard'а:

1. `test_changelog_is_descending::_key` — regex
   `^(\\d+)\\.(\\d+)\\.(\\d+)([a-z]?)$` не матчит `-pre` (dash-суффикс),
   `assert m` падает.
2. `test_changelog_has_latest_tag` — top `[0.8.10-pre]` != tag `[v0.8.9]`
   (HEAD на теге; коммит не создан, потому что fast упал **до**
   `git commit`).

**Fix (v0.8.10-pre-fix):** `[Unreleased]` (Keep a Changelog convention).
Regex `^## \\[(\\d[^\\]]*)\\]` требует **цифру** сразу после `[` —
`[Unreleased]` не попадает в `versions[]`, descending guard видит
только release-теги (`0.8.9`, `0.8.8`, ...). At tag-time: `[Unreleased]`
→ `[0.8.10]` в одном коммите, оба guard'а снова активны.

**Урок (расширение 139):** pre-release и rolling entries идут **только
под `[Unreleased]`**, никогда под version-like header (`[X.Y.Z-pre]`,
`[X.Y.Z-rc1]`). Правило: **patch-скрипт, трогающий CHANGELOG, должен
в собственном `main()` прогнать `test_changelog_*` до `git commit`**,
не после. Здесь `patch_v0810_pre_join_offset.py` этого не делал —
guard сработал уже на CI-подобном прогоне.

**Связанные:** §13.4/130 (anchor drift — descending guard),
§13.4/111 (CHANGELOG chain), §13.4/139 (fix semantics = REPL
verification). Тот же класс: **«assumed guard compatibility, not
verified»**.

'''
        s = s.replace(anchor, addendum + anchor, 1)
    else:
        addendum = anchor + '''

**§13.4/140 (pre-release versioning vs guard).** Продолжение кластера
139. `patch_v0810_pre_join_offset.py` вставил `## [0.8.10-pre] —
2026-10-20` в CHANGELOG. Уронил `test_changelog_is_descending::_key`
(regex `^(\\d+)\\.(\\d+)\\.(\\d+)([a-z]?)$` не матчит `-pre`) и
`test_changelog_has_latest_tag` (top `[0.8.10-pre]` != tag `v0.8.9`,
потому что fast упал до `git commit` — HEAD остался на теге).

**Fix (v0.8.10-pre-fix):** `[Unreleased]` (Keep a Changelog convention).
`^## \\[(\\d[^\\]]*)\\]` требует цифру → `[Unreleased]` невидим для
обоих guard'ов. At tag-time: `[Unreleased]` → `[0.8.10]` в одном
коммите, оба guard'а снова активны.

**Правило:** pre-release/rolling entries — **только под `[Unreleased]`**,
никогда `[X.Y.Z-pre]` / `[X.Y.Z-rcN]`. Patch-скрипт, трогающий
CHANGELOG, в собственном `main()` **обязан** прогнать `test_changelog_*`
до `git commit`. Класс ошибки — тот же, что §13.4/139
(«assumed semantics, not verified»); здесь — «assumed guard
compatibility».'''
        s = s.replace(anchor, addendum, 1)
    p.write_text(s)
    print("  OK: §13.4/140")


# ============ main ============

def main():
    print("=== v0.8.10-pre-fix: CHANGELOG [Unreleased] ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    print(f"  pre-existing uncommitted files: "
          f"{len([l for l in r.stdout.strip().splitlines() if l])}")

    fix_changelog_header()
    fix_source_script()
    append_decisions_140()

    print("\n=== pytest test_changelog (guard check) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_changelog.py", "-v"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** CHANGELOG GUARD STILL RED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=900)
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
        "v0.8.10-pre: merge() whole-operation RSS warning "
        "(P-A3 audit addendum); _JOIN_FIXED_OVERHEAD_BYTES=40 MB; "
        "§9.2.6 variance note; §11.5 best-effort clarification; "
        "CHANGELOG under [Unreleased] (§13.4/140)"])

    print("\n==> v0.8.10-pre applied. Not tagged (rolling into v0.8.10).")
    print("NEXT: uv run python scripts/proto_per_group.py")


if __name__ == "__main__":
    main()