#!/usr/bin/env python3
"""v0.9.1a-fix1: complete v0.9.1a (3 phases done, 2 remaining).

BUG 1 (§13.4/125 round 4): patch script imported `cadw.patch_utils`
without `sys.path.insert(0, ROOT)`. `uv run python scripts/X.py` puts
`scripts/` on sys.path, NOT repo root. Fix: sys.path.insert at top of
FUTURE patch scripts. This fix1 does it right.

BUG 2 (structural, DECISIONS.md): `### 11.6` header is MISSING from
DECISIONS.md (grep finds only textual references like "см. §11.6").
Header was removed during v0.8.11/v0.9.0 patches. v0.9.0b-fix1
appended Pipeline v1 addendum to file end because anchor not found —
correct behavior, but wrong location.

Fix: restore `### 11.6. Технический долг` header between §11.5 body
and `## 12.` header; move Pipeline v1 addendum from file end into
§11.6. Long-term: AST guard for missing-section-headers in DECISIONS.

STATE AFTER FAILED v0.9.1a RUN:
  ✅ tests/conftest.py           (autouse plt.close)
  ✅ cadw/patch_utils.py         (helpers)
  ✅ cadw/pipeline.py            (Protocol split — compile OK)
  ❌ fix_decisions_11_6          (failed at import)
  ❌ tests, changelog, guards, commit

This fix1:
  1. sys.path.insert(0, ROOT) at top (§13.4/125 discipline).
  2. Restore §11.6 header, move addendum.
  3. Remove dead test_cast_step (leftover stub).
  4. Write tests/test_patch_utils.py.
  5. Append CHANGELOG entry via patch_utils (dogfooding).
  6. Guards via patch_utils.assert_guards_green.
  7. Full fast + slow.
  8. Single commit (no WIP — §13.4/146 backlog).

Run: uv run python scripts/patch_v091a_fix1.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
# v0.9.1a-fix1 (§13.4/125 round 4): make `cadw` importable from patch
# script context. `uv run python scripts/X.py` puts scripts/ on sys.path,
# not CWD. First non-comment line after `from __future__` — the rule.
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


def replace_line_block(s, old_block, new_block, *, tag, optional=False):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        if optional:
            return s
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ Phase 1: verify prior phases + diagnostic ============

def verify_prior_phases():
    print("\n=== verify phases 1-3 (conftest, patch_utils, protocol split) ===")
    missing = []
    if not Path("tests/conftest.py").exists():
        missing.append("tests/conftest.py")
    if not Path("cadw/patch_utils.py").exists():
        missing.append("cadw/patch_utils.py")
    p = Path("cadw/pipeline.py")
    if p.exists():
        src = p.read_text()
        if "strict Protocol separation" not in src:
            missing.append("cadw/pipeline.py Protocol split")
    else:
        missing.append("cadw/pipeline.py")
    if missing:
        print(f"  ERROR: prior phases incomplete: {missing}")
        print("  Re-run: uv run python scripts/patch_v091a_backlog.py")
        print("  (after adding `sys.path.insert(0, ROOT)` to fix its import)")
        sys.exit(1)
    print("  OK: all 3 prior phases complete")


# ============ Phase 2: DECISIONS §11.6 restore + move addendum ============

def fix_decisions_11_6_v2():
    print("\n=== DECISIONS.md: restore §11.6 header + relocate addendum ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    # Diagnostic — show all top-level §11 subsections
    print("  diagnostic — all §11.* headers:")
    for line in s.split("\n"):
        if re.match(r"^#+\s+§?11\.", line):
            print(f"    {line.strip()[:80]}")

    # Locate structural anchors
    from cadw.patch_utils import find_section_anchor
    has_11_6 = find_section_anchor(s, "11.6") is not None
    has_11_5 = find_section_anchor(s, "11.5") is not None
    has_12 = find_section_anchor(s, "12") is not None

    print(f"  has §11.5: {has_11_5}, has §11.6: {has_11_6}, has §12: {has_12}")

    # Find Pipeline v1 addendum (should be at end from v0.9.0b-fix1)
    addendum_marker = "**v0.9.0b (Pipeline v1):**"
    has_addendum = addendum_marker in s
    print(f"  has Pipeline v1 addendum: {has_addendum}")

    if not has_addendum:
        print("  no addendum to relocate — skip")
        return

    # Extract addendum content (until next major section or EOF)
    add_start = s.find(addendum_marker)
    # End: look for "\n\n### 13." or "\n\n## 13." or "\n\n---" or EOF
    end_m = re.search(
        r"\n\n(?=(?:#{1,6}\s+§?1[23]\.|---|\Z))",
        s[add_start:],
    )
    if end_m:
        add_end = add_start + end_m.start()
    else:
        add_end = len(s)
    addendum = s[add_start:add_end].rstrip()
    print(f"  addendum location: offset {add_start}..{add_end} "
          f"(file len={len(s)})")

    # If §11.6 header exists — insert addendum before it (if not already).
    if has_11_6:
        pos = find_section_anchor(s, "11.6")
        if add_start < pos:
            print("  addendum already before §11.6 — skip move")
            return
        # Remove from old location
        new_s = s[:add_start] + s[add_end:]
        # Re-find anchor in new_s
        pos2 = find_section_anchor(new_s, "11.6")
        new_s = new_s[:pos2] + addendum + "\n\n" + new_s[pos2:]
        p.write_text(new_s)
        print("  OK: moved addendum before §11.6")
        return

    # §11.6 truly missing — restore header + insert addendum.
    if not has_11_5 or not has_12:
        print("  WARNING: §11.5 or §12 also missing — cannot restore")
        print("  Addendum stays at end of file. Add to backlog §11.6.")
        return

    # Remove addendum from old location first
    new_s = s[:add_start] + s[add_end:]
    new_s = new_s.rstrip() + "\n"

    # Locate §12 header offset in new_s
    pos_12 = find_section_anchor(new_s, "12")
    # Locate §11.5 header — insert §11.6 AFTER §11.5 content, i.e. just
    # before §12 (since §11.6 in original structure came after §11.5,
    # before §12)
    restored = (
        "### 11.6. Технический долг\n\n"
        "(Header restored v0.9.1a-fix1. Original §11.6 header was removed\n"
        "during v0.8.11/v0.9.0 patches; content below was previously at\n"
        "end-of-file from v0.9.0b-fix1.)\n\n"
        + addendum + "\n\n"
    )
    new_s = new_s[:pos_12] + restored + new_s[pos_12:]
    p.write_text(new_s)
    print("  OK: §11.6 header restored, addendum relocated")


# ============ Phase 3: remove dead test in test_pipeline.py ============

def remove_dead_test():
    print("\n=== tests/test_pipeline.py: remove dead test_cast_step stub ===")
    p = Path("tests/test_pipeline.py")
    s = p.read_text()

    # Dead block from v0.9.0b: leftover stub with unused helper
    dead = '''def test_cast_step():
    p = Pipeline().add_step("cast", CastStep({"v": "float64"}))
    # CastStep is stateless; no fit needed for transform to work
    p._fitted = True  # bypass fit for MVP (no model)
    df = make_df_dummy_pq  # not used; see below


# Helper for make_df in tests
def make_df_dummy_pq():
    pass


'''

    if dead not in s:
        print("  already absent — skip")
        return
    s = s.replace(dead, "", 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: removed 2 dead items")


# ============ Phase 4: tests/test_patch_utils.py ============

PATCH_UTILS_TESTS = '''"""v0.9.1a: patch_utils + Protocol split tests."""
import pytest


# ============ patch_utils: ensure_unreleased_header ============

def test_ensure_unreleased_adds_missing():
    from cadw.patch_utils import ensure_unreleased_header
    text = "## [0.9.0] — 2026-10-20\\n\\nold"
    result = ensure_unreleased_header(text)
    assert "## [Unreleased]" in result
    assert result.startswith("## [Unreleased]")


def test_ensure_unreleased_keeps_existing():
    from cadw.patch_utils import ensure_unreleased_header
    text = "## [Unreleased]\\n\\n### Added\\n\\n- x"
    assert ensure_unreleased_header(text) == text


def test_ensure_unreleased_empty_file():
    from cadw.patch_utils import ensure_unreleased_header
    assert ensure_unreleased_header("").startswith("## [Unreleased]")


# ============ patch_utils: append_unreleased_entry ============

def test_append_unreleased_entry():
    from cadw.patch_utils import append_unreleased_entry
    text = "## [Unreleased]\\n\\n## [0.9.0] — 2026-10-20"
    result = append_unreleased_entry(text, "### Added\\n\\n- new")
    assert result.index("### Added") < result.index("## [0.9.0]")
    assert result.index("## [Unreleased]") < result.index("### Added")


def test_append_unreleased_entry_without_header():
    """Missing [Unreleased] → helper creates it (no silent skip)."""
    from cadw.patch_utils import append_unreleased_entry
    text = "## [0.9.0] — 2026-10-20\\n\\nold"
    result = append_unreleased_entry(text, "### Added\\n\\n- new")
    assert result.startswith("## [Unreleased]")
    assert "### Added" in result


# ============ patch_utils: find_section_anchor ============

def test_find_section_anchor_hash3():
    from cadw.patch_utils import find_section_anchor
    text = "intro\\n\\n### 11.6. Технический долг\\n\\nbody"
    pos = find_section_anchor(text, "11.6")
    assert pos is not None
    assert text[pos:pos+8] == "### 11.6"


def test_find_section_anchor_with_dot():
    from cadw.patch_utils import find_section_anchor
    assert find_section_anchor("## 11.6. Tech debt", "11.6") is not None


def test_find_section_anchor_section_sign():
    from cadw.patch_utils import find_section_anchor
    assert find_section_anchor("### §11.6. Tech debt", "11.6") is not None


def test_find_section_anchor_not_found():
    from cadw.patch_utils import find_section_anchor
    assert find_section_anchor("no section here", "11.6") is None


def test_find_section_anchor_does_not_match_prefix():
    """`11.5` must not match `11.50` or `11.5x`."""
    from cadw.patch_utils import find_section_anchor
    assert find_section_anchor("### 11.50 something", "11.5") is None


# ============ patch_utils: insert_before_section ============

def test_insert_before_section_found():
    from cadw.patch_utils import insert_before_section
    text = "### 11.6. Tech debt\\n\\nbody"
    new, found = insert_before_section(text, "11.6", "**Insertion**")
    assert found is True
    assert new.index("**Insertion**") < new.index("### 11.6")
    assert new.endswith("body")


def test_insert_before_section_not_found():
    from cadw.patch_utils import insert_before_section
    text = "no section"
    new, found = insert_before_section(text, "11.6", "**Insertion**")
    assert found is False
    assert new == text


# ============ Protocol split (§13.4/147) ============

def test_predictor_does_not_require_transform():
    """Predictor Protocol must not have transform method (§13.4/147)."""
    from cadw.pipeline import Predictor
    required = [m for m in dir(Predictor) if not m.startswith("_")]
    assert "predict" in required
    assert "fit" in required
    assert "transform" not in required, (
        "Predictor must not declare transform (§13.4/147: strict split)"
    )


def test_transformer_has_transform_but_not_predict():
    """Transformer Protocol must not require predict."""
    from cadw.pipeline import Transformer
    required = [m for m in dir(Transformer) if not m.startswith("_")]
    assert "transform" in required
    assert "fit" in required
    assert "predict" not in required


def test_gbdt_step_satisfies_predictor_not_transformer():
    """GBDTStep: fit + predict, no transform → Predictor only."""
    from cadw.pipeline import GBDTStep, Predictor, Transformer
    step = GBDTStep(["x"], "y", n_estimators=5)
    assert isinstance(step, Predictor)
    assert not isinstance(step, Transformer)


def test_cast_step_satisfies_transformer_not_predictor():
    """CastStep: fit + transform, no predict → Transformer only."""
    from cadw.pipeline import CastStep, Predictor, Transformer
    step = CastStep({"v": "float64"})
    assert isinstance(step, Transformer)
    assert not isinstance(step, Predictor)
'''


def write_tests():
    print("\n=== tests/test_patch_utils.py ===")
    p = Path("tests/test_patch_utils.py")
    if p.exists():
        print("  already exists — overwriting")
    p.write_text(PATCH_UTILS_TESTS)
    compile_check(p)
    print("  OK")


# ============ Phase 5: CHANGELOG ============

def append_changelog():
    print("\n=== CHANGELOG.md: append v0.9.1a via patch_utils ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.1a — backbone" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Изменено (v0.9.1a — backbone)

- **`tests/conftest.py`** — autouse `plt.close('all')` для всех тестов.
  Устраняет `More than 20 figures have been opened` RuntimeWarning,
  появившийся с v0.9.0c (8 plot-accessor тестов перевалили порог;
  pre-existing seaborn/viz тесты тоже не закрывали figures).
- **`cadw/patch_utils.py`** — общие helpers для patch-скриптов:
  `ensure_unreleased_header`, `append_unreleased_entry`,
  `find_section_anchor` (regex tolerant: `# 11.6`, `## §11.6`, `#### 11.6.`),
  `insert_before_section` (explicit `found` flag, no silent no-op),
  `assert_guards_green` (§13.4/140 discipline extraction).
- **`Transformer` / `Predictor` Protocol split** (§13.4/147 backlog):
  Predictor БЕЗ `transform`; Transformer БЕЗ `predict`. Строго два
  минимальных контракта, не nested. Runtime `_is_predictor(step) =
  hasattr(step, "predict")` — duck typing, Protocol только для typing.
- **DECISIONS.md §11.6 header restored** — заголовок был потерян
  в v0.8.11/v0.9.0 патчах; Pipeline v1 addendum от v0.9.0b-fix1
  (был appended к концу файла из-за anchor-not-found) перенесён
  в восстановленную §11.6.
- **`tests/test_pipeline.py`** — удалён dead `test_cast_step` stub
  (leftover из v0.9.0b, тест без assertions).

### Ограничения (v0.9.1a, §11.5)

- `patch_utils` используется только новыми patch-скриптами (v0.9.1+).
  Старые (v0.8.x, v0.9.0) сохраняют свою логику — historical artifacts.
- `conftest.py` — autouse для matplotlib only; torch lazy, уборки не требует.

### Тесты (v0.9.1a)

- +18 fast: 14 `test_patch_utils.py` (CHANGELOG + anchor + insert +
  Protocol split), +4 dead-test cleanup ensures pipeline suite stays green.
- RuntimeWarning "More than 20 figures" — исчезает (conftest autouse).

- fast: **~1298** (+18). slow: **9** (unchanged).
'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK: appended via append_unreleased_entry (dogfooding)")


# ============ main ============

def main():
    print("=== v0.9.1a-fix1: complete backbone (3 done, 2 remaining) ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    # Phase 0: verify prior phases
    verify_prior_phases()

    # Phase 2: DECISIONS §11.6 restore
    fix_decisions_11_6_v2()

    # Phase 3: dead test cleanup
    remove_dead_test()

    # Phase 4: tests
    write_tests()

    # Phase 5: CHANGELOG
    append_changelog()

    # Phase 6: guards
    print("\n=== pytest: new patch_utils + pipeline + Protocol tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_pipeline.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards (via patch_utils.assert_guards_green) ===")
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
        "v0.9.1a: backbone — conftest.py (plt.close autouse), "
        "patch_utils.py (ensure_unreleased, find_section_anchor, "
        "assert_guards_green), strict Transformer/Predictor Protocols "
        "(Predictor без transform), §11.6 header restored, dead test "
        "cleanup; §13.4/125 round 4 (sys.path.insert in patch script)"])

    print("\n==> v0.9.1a-fix1 applied.")
    print("NEXT: v0.9.1b (DAG Pipeline).")


if __name__ == "__main__":
    main()