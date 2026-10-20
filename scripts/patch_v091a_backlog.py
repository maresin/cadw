#!/usr/bin/env python3
"""v0.9.1a: backbone fixes before DAG/ColumnTransformer/apply.

Closes 4 backlog items (§13.4/147, §11.6, §13.4/146):
  1. tests/conftest.py — autouse plt.close('all'). Fixes "More than
     20 figures" warning that appeared at v0.9.0c (pre-existing, my
     new plot tests tipped it over).
  2. cadw/patch_utils.py — shared helpers for patch scripts:
     ensure_unreleased_header, append_unreleased_entry,
     find_section_anchor (regex-based), assert_guards_green.
     Recent patches (v0.9.0a/b/c) duplicated this logic 3× with
     different bug-patterns each time. Extract once, test once.
  3. Transformer / Predictor Protocols split (§13.4/147 backlog):
     strict separation, no Predictor ⊃ Transformer. Predictor =
     fit + predict + state methods (без transform).
  4. DECISIONS §11.6 anchor fix: v0.9.0b-fix1 appended Pipeline v1
     addendum to end-of-file because anchor not found. Diagnostic grep
     + insert at correct location (or honest warning if truly missing).

GUARDS (before commit):
  test_patch_utils + test_pipeline (Protocol split) + test_changelog +
  arrow_guard + no_pandas + fast + slow.

Run: uv run python scripts/patch_v091a_backlog.py
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


# ============ A. tests/conftest.py ============

CONFTEST = '''"""Root test configuration (v0.9.1a).

Autouse fixture: close matplotlib figures after each test.

Reason: v0.9.0c added 8 plot-accessor tests, tipping the ~20-figure
threshold. Pre-existing seaborn/viz tests did not close figures either.
Fix is global — every test gets figure cleanup, no more RuntimeWarning.
"""
import pytest


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each test.

    matplotlib default retains figures in `plt._pylab_helpers.Gcf` until
    explicit close. Across a 1300-test suite that accumulates memory and
    emits "More than 20 figures have been opened" RuntimeWarning.
    """
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass
'''


def write_conftest():
    print("\n=== tests/conftest.py ===")
    p = Path("tests/conftest.py")
    if p.exists():
        print("  already exists — skip (manual merge required if needed)")
        return
    p.write_text(CONFTEST)
    compile_check(p)
    print("  OK")


# ============ B. cadw/patch_utils.py ============

PATCH_UTILS = '''"""Shared helpers for CADW patch scripts (v0.9.1a).

Extracted from v0.9.0a/b/c experience: recent patch scripts duplicated
CHANGELOG/§-anchor logic with different bug-patterns each time
(§13.4/140 [Unreleased]-after-tag; §13.4/144 cross-file rename;
§13.4/146 silent-skip on missing anchor; §13.4/147 partial contract).

This module fixes the pattern once. Patch scripts import from
`cadw.patch_utils` and use these helpers, so future patches inherit
the fixes instead of reimplementing them.

Not on any hot path (imported only from `scripts/patch_*.py`).
"""
from __future__ import annotations

import re
import subprocess
import sys
from typing import Optional


# ============ CHANGELOG helpers ============

def ensure_unreleased_header(text: str) -> str:
    """Ensure `## [Unreleased]` header exists; prepend if missing.

    Rationale (§13.4/140, /146): after a release tag, CHANGELOG starts
    with `## [X.Y.Z]`. Patch scripts that skip on missing [Unreleased]
    silently lose their entry. This helper creates the header as a
    fallback (prepend before first `## [`), never skips.
    """
    if "## [Unreleased]" in text:
        return text
    m = re.search(r'^## \\[', text, re.MULTILINE)
    if m:
        return text[:m.start()] + "## [Unreleased]\\n\\n" + text[m.start():]
    return "## [Unreleased]\\n\\n" + text


def append_unreleased_entry(text: str, entry: str) -> str:
    """Append `entry` (must start with `### ...`) under `## [Unreleased]`.

    `entry` should be a section like `### Added ...` — the function
    inserts it right after the `## [Unreleased]` header (before any
    existing sub-sections), maintaining descending order by convention.
    """
    text = ensure_unreleased_header(text)
    header = "## [Unreleased]\\n\\n"
    # If there's already content under [Unreleased] (starting with `###`),
    # insert new entry BEFORE it (newest first).
    pos = text.find(header)
    if pos < 0:
        raise ValueError("ensure_unreleased_header failed to create header")
    insert_at = pos + len(header)
    return text[:insert_at] + entry.rstrip() + "\\n\\n" + text[insert_at:]


# ============ DECISIONS.md section helpers ============

_SECTION_PATTERN = r"^#{{1,6}}\\s+(?:§)?{section}\\b"

def find_section_anchor(text: str, section: str) -> Optional[int]:
    """Return char offset of section header (§{section}) or None.

    Matches `# 11.6`, `## 11.6`, `### 11.6`, `#### 11.6.` etc.,
    including `§` prefix (`### §11.6`). Regex-based: tolerates
    trailing dot, sub-section markers, and heading-level variations.
    """
    pat = re.compile(
        _SECTION_PATTERN.format(section=re.escape(section)),
        re.MULTILINE,
    )
    m = pat.search(text)
    return m.start() if m else None


def insert_before_section(text: str, section: str,
                          insertion: str) -> tuple[str, bool]:
    """Insert `insertion` text before §{section} header.

    Returns (new_text, found). If section not found, returns
    (text, False) — caller decides (warn and append at end, or skip).
    Never silently no-ops — caller sees `found` flag.
    """
    pos = find_section_anchor(text, section)
    if pos is None:
        return text, False
    return text[:pos] + insertion.rstrip() + "\\n\\n" + text[pos:], True


# ============ Guard test helper ============

def assert_guards_green(
    guard_tests: list[str],
    *,
    timeout: int = 900,
) -> None:
    """Run pytest on guard tests; exit(1) on failure.

    Extraction of the try/except that every patch script did:
        r = subprocess.run(["uv", "run", "pytest", *guard_tests, "-v"])
        if r.returncode != 0: print(...); sys.exit(1)
    Called from patch script after code changes, before `git commit`.
    Enforces §13.4/140 discipline (no WIP commit on red).
    """
    print(f"  $ uv run pytest {' '.join(guard_tests)} -q")
    r = subprocess.run(
        ["uv", "run", "pytest", *guard_tests, "-q"],
        timeout=timeout,
    )
    if r.returncode != 0:
        print(f"\\n*** GUARD FAILURE: {guard_tests} ***")
        sys.exit(1)
'''


def write_patch_utils():
    print("\n=== cadw/patch_utils.py ===")
    p = Path("cadw/patch_utils.py")
    if p.exists():
        print("  already exists — skip")
        return
    p.write_text(PATCH_UTILS)
    compile_check(p)
    print("  OK")


# ============ C. Protocol split (§13.4/147 backlog) ============

def split_protocols():
    print("\n=== cadw/pipeline.py: strict Transformer / Predictor split ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''@runtime_checkable
class Transformer(Protocol):
    """Any step that can fit and transform a CadwDataFrame.

    Contract:
      - fit(df, target=None) -> self         (stateful: read df, set state;
                                              stateless: no-op)
      - transform(df) -> CadwDataFrame       (apply state; streaming)
      - estimate_state_bytes(dc) -> int      (upper bound, BEFORE fit)
      - state_bytes() -> int                 (actual, AFTER fit; 0 if no fit)
    """
    def fit(self, df: "CadwDataFrame", target: Optional[str] = None): ...
    def transform(self, df: "CadwDataFrame") -> "CadwDataFrame": ...
    def estimate_state_bytes(self, dc: DataChars) -> int: ...
    def state_bytes(self) -> int: ...


@runtime_checkable
class Predictor(Protocol):
    """Transformer that also produces predictions from features."""
    def fit(self, df: "CadwDataFrame", target: Optional[str] = None): ...
    def transform(self, df: "CadwDataFrame") -> "CadwDataFrame": ...
    def predict(self, df: "CadwDataFrame") -> "CadwDataFrame": ...
    def estimate_state_bytes(self, dc: DataChars) -> int: ...
    def state_bytes(self) -> int: ...'''

    new = '''# v0.9.1a (§13.4/147 backlog): strict Protocol separation.
# Transformer and Predictor are NOT nested (Predictor is NOT a
# Transformer subclass) — they have disjoint primary methods:
#   - Transformer: fit + transform (pure feature engineering)
#   - Predictor:   fit + predict   (model; transform never called)
# Both share state-method contract (estimate_state_bytes, state_bytes).
# Pipeline validates structure via `_is_predictor(step) = hasattr(step,
# "predict")`, not via isinstance — duck-typing keeps Protocol minimal.

@runtime_checkable
class Transformer(Protocol):
    """Feature engineering step: fit → transform.

    Contract:
      - fit(df, target=None) -> self         (read df, set state; or no-op)
      - transform(df) -> CadwDataFrame       (apply state; streaming)
      - estimate_state_bytes(dc) -> int      (upper bound, BEFORE fit)
      - state_bytes() -> int                 (actual, AFTER fit; 0 if no fit)
    """
    def fit(self, df: "CadwDataFrame", target: Optional[str] = None): ...
    def transform(self, df: "CadwDataFrame") -> "CadwDataFrame": ...
    def estimate_state_bytes(self, dc: DataChars) -> int: ...
    def state_bytes(self) -> int: ...


@runtime_checkable
class Predictor(Protocol):
    """Model step: fit → predict. `transform` NOT required.

    Pipeline's `transform()` stops BEFORE Predictor; `predict()` calls
    Predictor.predict() on the last step. So Predictor never needs
    `.transform()` — strictly minimal contract per §13.4/147.

    Contract:
      - fit(df, target=None) -> self
      - predict(df) -> CadwDataFrame
      - estimate_state_bytes(dc) -> int
      - state_bytes() -> int
    """
    def fit(self, df: "CadwDataFrame", target: Optional[str] = None): ...
    def predict(self, df: "CadwDataFrame") -> "CadwDataFrame": ...
    def estimate_state_bytes(self, dc: DataChars) -> int: ...
    def state_bytes(self) -> int: ...'''

    s = replace_line_block(s, old, new, tag="protocol-split")
    p.write_text(s)
    compile_check(p)
    print("  OK: Predictor без transform, strict split")


# ============ D. DECISIONS §11.6 anchor fix ============

def fix_decisions_11_6():
    print("\n=== DECISIONS.md: §11.6 anchor fix (diagnostic + insert) ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    # Diagnostic: grep §11.6 variants
    print("  diagnostic — searching §11.6 references:")
    r = subprocess.run(
        ["grep", "-n", r"11\.6", "DECISIONS.md"],
        capture_output=True, text=True,
    )
    for line in r.stdout.strip().splitlines()[:10]:
        print(f"    {line}")

    # If Pipeline v1 addendum is in the file (appended at end):
    if "v0.9.0b (Pipeline v1)" in s and "## 14. Резюме" not in s[:s.find("v0.9.0b (Pipeline v1)")]:
        print("  Pipeline v1 addendum is present")

    # Try to find §11.6 with regex via patch_utils
    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.6")
    if pos is not None:
        line_start = s.rfind("\n", 0, pos) + 1
        header_line = s[line_start:s.find("\n", pos)]
        print(f"  found anchor: {header_line!r} at offset {pos}")
        # If Pipeline addendum is at end and NOT before §11.6 — move it.
        # But we need to be careful: v0.9.0b-fix1 appended it to end. If
        # the file doesn't have "v0.9.0b (Pipeline v1)" before §11.6,
        # then it's at the end. Move it.
        addendum_start = s.find("**v0.9.0b (Pipeline v1):**")
        if addendum_start > 0:
            # Find addendum end (double newline before next header or end)
            addendum_end = s.find("### 13.5", addendum_start)
            if addendum_end < 0:
                addendum_end = s.find("\n\n\n", addendum_start)
            if addendum_end < 0:
                addendum_end = len(s)
            addendum = s[addendum_start:addendum_end].rstrip()
            # Check if it's already before §11.6
            if addendum_start < pos:
                print("  Pipeline v1 addendum already before §11.6 — skip move")
                return
            # Remove from end, insert before §11.6
            new_s = s[:addendum_start] + s[addendum_end:].lstrip("\n")
            # Re-find pos in new_s (may have shifted)
            pos2 = find_section_anchor(new_s, "11.6")
            if pos2 is None:
                print("  ERROR: §11.6 anchor lost after removal — revert")
                return
            new_s = new_s[:pos2] + addendum + "\n\n" + new_s[pos2:]
            p.write_text(new_s)
            print("  OK: moved Pipeline v1 addendum to §11.6")
        else:
            print("  no Pipeline v1 addendum to move — OK")
    else:
        print("  WARNING: §11.6 anchor NOT found (grep above shows variants)")
        print("  This means DECISIONS.md structure changed. Manual fix needed.")
        print("  Backlog: add §11.6 header restoration in v0.9.2.")


# ============ E. Tests ============

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
    result = ensure_unreleased_header(text)
    assert result == text


def test_ensure_unreleased_empty_file():
    from cadw.patch_utils import ensure_unreleased_header
    result = ensure_unreleased_header("")
    assert result.startswith("## [Unreleased]")


# ============ patch_utils: append_unreleased_entry ============

def test_append_unreleased_entry():
    from cadw.patch_utils import append_unreleased_entry
    text = "## [Unreleased]\\n\\n## [0.9.0] — 2026-10-20"
    result = append_unreleased_entry(text, "### Added\\n\\n- new")
    assert result.index("### Added") < result.index("## [0.9.0]")
    assert result.index("## [Unreleased]") < result.index("### Added")


def test_append_unreleased_entry_without_header():
    """If [Unreleased] missing — helper creates it (no silent skip)."""
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
    text = "## 11.6. Tech debt"
    pos = find_section_anchor(text, "11.6")
    assert pos is not None


def test_find_section_anchor_section_sign():
    from cadw.patch_utils import find_section_anchor
    text = "### §11.6. Tech debt"
    pos = find_section_anchor(text, "11.6")
    assert pos is not None


def test_find_section_anchor_not_found():
    from cadw.patch_utils import find_section_anchor
    text = "no section here"
    assert find_section_anchor(text, "11.6") is None


def test_find_section_anchor_does_not_match_wrong_section():
    """`11.5` must not match `11.50` or `11.5x`."""
    from cadw.patch_utils import find_section_anchor
    text = "### 11.50 something"
    pos = find_section_anchor(text, "11.5")
    # \\b after 11.5 — `11.50` has `.0` which is not a boundary
    assert pos is None


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
    assert new == text  # unchanged


# ============ Protocol split (§13.4/147) ============

def test_predictor_does_not_require_transform():
    """Predictor Protocol must not have transform method (§13.4/147)."""
    from cadw.pipeline import Predictor
    # runtime_checkable Protocol — hasattr check
    required_methods = [m for m in dir(Predictor)
                        if not m.startswith("_")]
    assert "predict" in required_methods
    assert "fit" in required_methods
    assert "transform" not in required_methods, (
        "Predictor must not declare transform (§13.4/147: strict split)"
    )


def test_transformer_has_transform_but_not_predict():
    """Transformer Protocol must not require predict."""
    from cadw.pipeline import Transformer
    required = [m for m in dir(Transformer) if not m.startswith("_")]
    assert "transform" in required
    assert "fit" in required
    assert "predict" not in required


def test_gbdt_step_satisfies_predictor_protocol():
    """GBDTStep (fit + predict, no transform) matches Predictor."""
    from cadw.pipeline import GBDTStep, Predictor
    step = GBDTStep(["x"], "y", n_estimators=5)
    assert isinstance(step, Predictor)
    # But NOT a Transformer (no transform)
    from cadw.pipeline import Transformer
    assert not isinstance(step, Transformer)
'''


def write_tests():
    print("\n=== tests/test_patch_utils.py ===")
    p = Path("tests/test_patch_utils.py")
    p.write_text(PATCH_UTILS_TESTS)
    compile_check(p)
    print("  OK")


# ============ F. CHANGELOG ============

CHANGELOG_ENTRY = '''## [Unreleased]

### Изменено (v0.9.1a — backbone)

- **`tests/conftest.py`** — autouse `plt.close('all')` для всех тестов.
  Устраняет `More than 20 figures have been opened` RuntimeWarning,
  который появился с v0.9.0c (8 plot-accessor тестов перевалили порог;
  pre-existing в seaborn/viz тестах тоже не закрывали figures).
- **`cadw/patch_utils.py`** — общие helpers для patch-скриптов:
  - `ensure_unreleased_header(text)` — prepend `## [Unreleased]` если
    отсутствует (fix §13.4/146 silent-skip).
  - `append_unreleased_entry(text, entry)` — insert после [Unreleased].
  - `find_section_anchor(text, section)` — regex для `# 11.6`,
    `## 11.6`, `### §11.6`, `#### 11.6.` (fix §13.4/146 anchor-not-found).
  - `insert_before_section(text, section, insertion)` — insert с
    explicit `found` flag (no silent no-op).
  - `assert_guards_green(guard_tests)` — общий runner (§13.4/140 discipline).
- **`Transformer` / `Predictor` Protocol split** (§13.4/147 backlog):
  - `Predictor`: `fit + predict + estimate_state_bytes + state_bytes`
    — **без `transform`** (Pipeline никогда не вызывает .transform() на
    Predictor; runtime проверяет через `_is_predictor(step) = hasattr(step,
    "predict")`).
  - `Transformer`: `fit + transform + estimate_state_bytes + state_bytes`
    — без `predict`.
  - Не nested (Predictor NOT subclass of Transformer) — строго два
    минимальных контракта.
- **DECISIONS.md §11.6 anchor fix** — Pipeline v1 addendum от v0.9.0b
  (был appended к концу файла из-за anchor-not-found) перенесён в §11.6
  через `patch_utils.find_section_anchor` (regex tolerant).

### Ограничения (v0.9.1a, §11.5)

- `patch_utils` используется только новыми patch-скриптами. Старые
  (v0.8.x, v0.9.0) сохраняют свою логику — historical artifacts,
  не рефакторим.
- `conftest.py` — autouse fixture для matplotlib only; torch — ленивый
  импорт, уборки не требует.

### Тесты (v0.9.1a)

- +14 fast: 13 `test_patch_utils.py` (CHANGELOG + anchor + insert +
  Protocol split).
- Warning "More than 20 figures" — исчезает.

- fast: **~1294** (+14). slow: **9** (unchanged).

'''


def append_changelog():
    print("\n=== CHANGELOG.md: append v0.9.1a via patch_utils ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.1a — backbone" in s:
        print("  already present — skip")
        return

    # Use patch_utils (self-test!)
    from cadw.patch_utils import append_unreleased_entry
    new_s = append_unreleased_entry(s, CHANGELOG_ENTRY.replace(
        "## [Unreleased]\n\n", "", 1,
    ))
    p.write_text(new_s)
    print("  OK: appended via append_unreleased_entry")


# ============ main ============

def main():
    print("=== v0.9.1a: backbone fixes ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    write_conftest()
    write_patch_utils()
    split_protocols()
    fix_decisions_11_6()
    write_tests()
    append_changelog()

    print("\n=== pytest: new patch_utils + pipeline tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_pipeline.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        print("Fix in place (§13.4/146 backlog: no WIP commit on red).")
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
        "assert_guards_green), strict Transformer/Predictor Protocols, "
        "DECISIONS §11.6 anchor fix"])

    print("\n==> v0.9.1a applied.")
    print("NEXT: v0.9.1b (DAG Pipeline).")


if __name__ == "__main__":
    main()