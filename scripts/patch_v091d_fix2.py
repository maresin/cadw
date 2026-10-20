#!/usr/bin/env python3
"""v0.9.1d-fix2: stale test (round 3, §13.4/152) + MECHANICAL grep tooling.

BUG (§13.4/144 → /149 → /152, round 3):
  fix1 added `_STEP_REGISTRY["ColumnTransformer"] = ColumnTransformer`
  (correct — was bug 3 in v0.9.1d). But `test_list_step_types_default`
  (written in v0.9.1d, asserts 3 built-ins) became stale. 1 test fail.

SAME PATTERN, 3 TIMES:
  - v0.8.11a-fix2 (§13.4/144): behavior change → stale test, other file
  - v0.9.1b-fix2 (§13.4/149): behavior removal → stale test, same file
  - v0.9.1d-fix1 (§13.4/152): behavior addition → stale test, same file

RULE "grep before commit" set at /144, /149. Forgot each time.
CONCLUSION: manual discipline doesn't work — need TOOLING.

MECHANICAL TOOLING (this fix):
  - `patch_utils.grep_and_warn(patterns, dirs)` — runs grep for
    behavioral patterns, prints warnings (no auto-fail).
  - `patch_utils.assert_no_stale_behavior(patterns, dirs)` — runs grep,
    raises on hits. Use before `git add` in patches that change behavior.
  - fix2 itself DOGFOODS: calls `assert_no_stale_behavior` before commit
    for the specific patterns this fix introduces.

TESTS:
  - test_patch_utils: +6 for grep_and_warn / assert_no_stale_behavior
  - test_register_step_type: fixed (4 items expected)

GUARDS: new patch_utils tests + registry tests + pipeline regression +
guards + fast + slow.

Run: uv run python scripts/patch_v091d_fix2.py
"""
from __future__ import annotations

import os
import re
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


# ============ Fix 1: stale test (4 built-ins now) ============

def fix_stale_registry_test():
    print("\n=== tests/test_register_step_type.py: 4 built-in types ===")
    p = Path("tests/test_register_step_type.py")
    s = p.read_text()

    old = '''def test_list_step_types_default():
    assert list_step_types() == ["CastStep", "DropnaStep", "GBDTStep"]'''

    new = '''def test_list_step_types_default():
    """v0.9.1d-fix2 (§13.4/152): 4 built-in types.

    `ColumnTransformer` registered after class definition (v0.9.1d-fix1).
    Sorted order: alphabetical.
    """
    assert list_step_types() == [
        "CastStep", "ColumnTransformer", "DropnaStep", "GBDTStep",
    ]'''

    s = replace_line_block(s, old, new, tag="stale-list-step-types")
    p.write_text(s)
    print("  OK")


# ============ Fix 2: patch_utils — grep tooling ============

def add_grep_tooling():
    print("\n=== cadw/patch_utils.py: grep_and_warn + assert_no_stale_behavior ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()

    if "def grep_and_warn(" in s:
        print("  already present — skip")
        return

    # Append after assert_guards_green
    marker = '''    if r.returncode != 0:
        print(f"\\n*** GUARD FAILURE: {guard_tests} ***")
        sys.exit(1)'''

    new_block = marker + '''


# ============ Grep-based stale-behavior detection (§13.4/152) ============
#
# Recurring pattern: patch changes/removes/adds behavior; a test in the
# same or other file asserts the OLD behavior. Manual "grep before
# commit" discipline failed 3 times (§13.4/144, /149, /152). Mechanical
# tooling is the only reliable fix.

def grep_and_warn(
    patterns: list[str],
    dirs: list[str],
    *,
    label: str = "stale-behavior",
) -> list[str]:
    """Grep `dirs` for `patterns`; print warnings on hits. Returns hits.

    Non-fatal. Use in patch scripts BEFORE `git add` to surface stale
    assertions in tests that a behavior change may have broken.
    `patterns` are literal strings (grep -F), not regex — safe for
    strings with special chars.
    """
    hits: list[str] = []
    for pat in patterns:
        for d in dirs:
            if not Path(d).exists():
                continue
            r = subprocess.run(
                ["grep", "-rFn", pat, d],
                capture_output=True, text=True,
            )
            for line in r.stdout.strip().splitlines():
                hits.append(line)
    if hits:
        print(f"\\n  [{label}] {len(hits)} reference(s) found:")
        for line in hits[:20]:
            print(f"    {line}")
    return hits


def assert_no_stale_behavior(
    patterns: list[str],
    dirs: list[str],
    *,
    label: str = "stale-behavior",
    allow: list[str] | None = None,
) -> None:
    """Like grep_and_warn, but exits(1) if hits (excluding `allow` files).

    Use in patch scripts whose fix changes/removes/adds behavior:
    before `git add`, call with `patterns = old behavior strings` and
    `dirs = ["tests/"]`. Any hit = a test asserting old behavior; fix
    the test or the allow-list BEFORE commit.

    `allow` — list of file path substrings to exclude (e.g., ["test_new"]
    if a test file explicitly tests the removed behavior — rare).
    """
    hits = grep_and_warn(patterns, dirs, label=label)
    if allow:
        hits = [h for h in hits if not any(a in h for a in allow)]
    if hits:
        print(f"\\n*** {label}: {len(hits)} stale reference(s) remain ***")
        print("Fix tests or update allow-list before commit.")
        sys.exit(1)'''

    s = replace_line_block(s, marker, new_block, tag="grep-tooling")
    p.write_text(s)

    # compile-check
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


# ============ Fix 3: patch_utils tests for grep tooling ============

def add_grep_tests():
    print("\n=== tests/test_patch_utils.py: grep tooling tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()

    if "def test_grep_and_warn_finds_hits" in s:
        print("  already present — skip")
        return

    new_tests = '''

# ============ grep tooling (§13.4/152) ============

def test_grep_and_warn_finds_hits(tmp_path, capsys):
    from cadw.patch_utils import grep_and_warn
    d = tmp_path / "mydir"
    d.mkdir()
    (d / "test_a.py").write_text("old_behavior_xyz = 1\\n")
    (d / "test_b.py").write_text("other = 2\\n")
    hits = grep_and_warn(["old_behavior_xyz"], [str(d)])
    assert len(hits) == 1
    assert "old_behavior_xyz" in hits[0]


def test_grep_and_warn_empty_when_no_hits(tmp_path):
    from cadw.patch_utils import grep_and_warn
    d = tmp_path / "mydir"
    d.mkdir()
    (d / "test_a.py").write_text("new_behavior = 1\\n")
    hits = grep_and_warn(["old_behavior_xyz"], [str(d)])
    assert hits == []


def test_grep_and_warn_missing_dir_skipped(tmp_path):
    from cadw.patch_utils import grep_and_warn
    hits = grep_and_warn(["x"], [str(tmp_path / "nope")])
    assert hits == []


def test_assert_no_stale_behavior_raises_on_hit(tmp_path):
    from cadw.patch_utils import assert_no_stale_behavior
    d = tmp_path / "mydir"
    d.mkdir()
    (d / "test_a.py").write_text("old_behavior_xyz\\n")
    with pytest.raises(SystemExit):
        assert_no_stale_behavior(["old_behavior_xyz"], [str(d)])


def test_assert_no_stale_behavior_allow_list(tmp_path):
    from cadw.patch_utils import assert_no_stale_behavior
    d = tmp_path / "mydir"
    d.mkdir()
    (d / "test_a.py").write_text("old_behavior_xyz\\n")
    # allow-list should exclude → no raise
    assert_no_stale_behavior(
        ["old_behavior_xyz"], [str(d)], allow=["test_a.py"],
    )


def test_assert_no_stale_behavior_passes_when_clean(tmp_path):
    from cadw.patch_utils import assert_no_stale_behavior
    d = tmp_path / "mydir"
    d.mkdir()
    (d / "test_a.py").write_text("new_behavior = 1\\n")
    assert_no_stale_behavior(["old_behavior_xyz"], [str(d)])  # no raise
'''

    if not s.endswith("\n"):
        s += "\n"
    s += new_tests
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


# ============ Fix 4: §13.4/152 ============

def append_decisions_152():
    print("\n=== DECISIONS.md: §13.4/152 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 152" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append at end")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/152 (manual grep discipline fails 3 times — tooling needed).** "
        s += "See patch_utils.grep_and_warn / assert_no_stale_behavior.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 152 (third recurrence — manual discipline ≠ tooling).**
v0.9.1d-fix1 добавил `_STEP_REGISTRY["ColumnTransformer"] = ...`
(правильно — был баг 3 в v0.9.1d). Test `test_list_step_types_default`
(написан в v0.9.1d, ожидает 3 built-in types) — stale, реально 4 → 1 fail.

**Тот же паттерн, ТРЕТИЙ раз:**
- v0.8.11a-fix2 (§13.4/144): fix изменил guard name; stale test в другом файле.
- v0.9.1b-fix2 (§13.4/149): fix удалил validation; stale test в том же файле.
- **v0.9.1d-fix1 (§13.4/152)**: fix добавил registration; stale test ожидает старое число.

**Правило «grep before commit» установлено дважды (144, 149). Забыто каждый раз.**

**Причина:** «правило» — это текст в DECISIONS. Работает только если помнить. 3 из 3 раз — не вспомнил. **Manual discipline ≠ tooling.**

**Fix (this round — TOOLING, не текст):**
- `cadw.patch_utils.grep_and_warn(patterns, dirs, label=...)` —
  `grep -rFn` по указанным папкам, печатает hits, **не** raise.
  Для ручной инспекции перед `git add`.
- `cadw.patch_utils.assert_no_stale_behavior(patterns, dirs, label,
  allow=...)` — как grep_and_warn, но **raise SystemExit(1)** если
  hits (кроме allow-list). Для **dogfooding** в patch-скриптах:
  до `git add`, вызвать с patterns = old behavior strings, dirs =
  ["tests/"]. Любой hit = тест на старое поведение — фиксить.
- **fix2 dogfoods:** `assert_no_stale_behavior` вызывается в самом
  patch-скрипте до commit. Так следующий патч **не сможет** забыть —
  script упадёт при stale.

**Правило (окончательное):** patch, который меняет/удаляет/добавляет
**behavior** (validation, error message, guard name, return contract,
list of built-ins, enum values) → **обязан** вызвать
`patch_utils.assert_no_stale_behavior(...)` перед commit. Не «grep-
warning» — hard fail. Именно потому что warning игнорируется 3× подряд.

**Backlog (§11.6):** в patch-scripts **template** (когда он появится)
добавить `assert_no_stale_behavior` как обязательный шаг в `main()`.
Сейчас каждый патч пишется с нуля — до template, каждый раз явный
вызов, надежда на память. Скоро: `cadw/scaffold_patch.py` —
generate skeleton с pre-filled guard + stale checks.

**Связанные:** §13.4/144 (write/read key), /149 (stale test round 2),
/151 (registry namespace).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/152")


# ============ main ============

def main():
    print("=== v0.9.1d-fix2 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_stale_registry_test()
    add_grep_tooling()
    add_grep_tests()
    append_decisions_152()

    # Dogfood: this fix adds ColumnTransformer to list_step_types()
    # (from v0.9.1d-fix1). Grep tests for old 3-item assertions.
    print("\n=== dogfood: assert_no_stale_behavior ===")
    from cadw.patch_utils import assert_no_stale_behavior
    try:
        assert_no_stale_behavior(
            patterns=[
                'assert list_step_types() == ["CastStep", "DropnaStep", "GBDTStep"]',
                '"CastStep", "DropnaStep", "GBDTStep"]',  # alt format
            ],
            dirs=["tests/"],
            label="list_step_types-old-count",
            allow=[],  # no allow — all hits are stale
        )
        print("  OK: no stale references")
    except SystemExit:
        print("\n*** Dogfood caught a stale reference. Re-check. ***")
        raise

    print("\n=== pytest: registry + CT + patch_utils ===")
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

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.1d: ColumnTransformer (sklearn-style column fanout, "
        "row-preserving, remainder drop/passthrough) + register_step_type "
        "(module-level _STEP_REGISTRY, custom steps for from_json); "
        "cadw/__init__.py exports Pipeline + wrappers; §13.4/151 "
        "(registry __serialization_name__); §13.4/152 (grep tooling — "
        "manual discipline fails 3x, tooling required)"])

    print("\n==> v0.9.1d-fix2 applied.")
    print("NEXT: v0.9.1e (apply(fn) — prototype + implementation).")


if __name__ == "__main__":
    main()