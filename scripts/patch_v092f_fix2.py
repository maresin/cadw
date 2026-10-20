#!/usr/bin/env python3
"""v0.9.2f-fix2: test_list_step_types_default — accept 8 built-ins.

BUG (§13.4/172, stale list from registry growth): v0.9.2f added 4
scalers (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler)
via _register_scalers() at scalers.py import (auto-called by
cadw/__init__.py `from .scalers import ...`). Registry went from 4
built-ins to 8. Test asserted closed 4-item list → stale.

Not a code bug — auto-registration is intentional (Pipeline.from_json
works for scalers without extra setup). Test intent: "verify no
test-scoped registry pollution" — not "only 4 forever".

FIX: update expected list to 8 items (alphabetical: CastStep,
ColumnTransformer, DropnaStep, GBDTStep, MaxAbsScaler, MinMaxScaler,
RobustScaler, StandardScaler). Add rationale comment.

§13.4/172 lesson: closed lists of "current built-ins" ARE stale-baits
— every new built-in breaks them. Either (a) accept churn (update in
each patch that adds built-in) or (b) assert "is superset of N core
built-ins, no unknown types". Choose (a) — churn is visible in patch,
and the review of expected-vs-actual catches typos in new step name.

GUARDS: register_step_type + scalers + pipeline + full suite + guards.

Run: uv run python scripts/patch_v092f_fix2.py
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


# ============ Fix: test expected list 4 → 8 ============

def fix_test_list_step_types():
    print("\n=== tests/test_register_step_type.py: 8 built-ins ===")
    p = Path("tests/test_register_step_type.py")
    s = p.read_text()

    old = '''def test_list_step_types_default():
    """v0.9.1d-fix2 (§13.4/152): 4 built-in types.

    `ColumnTransformer` registered after class definition (v0.9.1d-fix1).
    Sorted order: alphabetical.
    """
    assert list_step_types() == [
        "CastStep", "ColumnTransformer", "DropnaStep", "GBDTStep",
    ]'''

    new = '''def test_list_step_types_default():
    """Built-in step types: 8, alphabetically sorted.

    v0.9.1d: CastStep, DropnaStep, GBDTStep, ColumnTransformer.
    v0.9.2f: + 4 scalers (MaxAbsScaler, MinMaxScaler, RobustScaler,
    StandardScaler) — auto-registered via _register_scalers() at
    cadw/scalers.py import (called by cadw/__init__.py). Pipeline.from_json
    works for scalers without extra setup.

    §13.4/172: this list grows with each new built-in. Closed-list
    assert is intentional — churn visible in patch, and review of
    expected-vs-actual catches typos in new step names. When adding
    a built-in, update this list in the same patch.
    """
    assert list_step_types() == [
        "CastStep", "ColumnTransformer", "DropnaStep", "GBDTStep",
        "MaxAbsScaler", "MinMaxScaler", "RobustScaler", "StandardScaler",
    ]'''

    s = replace_line_block(s, old, new, tag="list-step-types-8")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ §13.4/172 ============

def append_decisions_172():
    print("\n=== DECISIONS.md: §13.4/172 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 172" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/172 (closed-list stale-bait on registry growth).** "
        s += "test_list_step_types_default asserted 4 items; scalers "
        s += "added 4 more.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 172 (closed-list stale-bait on registry growth).**
v0.9.2f-fix2: 1 failure — `test_list_step_types_default` asserted
closed 4-item list, registry grew to 8 (added 4 scalers via
`_register_scalers()` auto-called at `cadw/scalers.py` import).

**Not a bug** — auto-registration intentional: `cadw/__init__.py`
imports `from .scalers import ...` → `_register_scalers()` runs →
`Pipeline.from_json("pipeline_with_scaler")` works without explicit
`register_step_type` in user code.

**Test intent** was "verify no test-scoped registry pollution" (guard
against leaks from test_registry_* fixtures), not "only 4 forever".

**Two competing fixes considered:**
- (a) Update expected list to 8. Churn per built-in add; but churn is
  VISIBLE — patch review of expected-vs-actual catches typos in step
  name (e.g., `"StandartScaler"` → test fails, not silent).
- (b) Assert superset-of-N + no-unknown-types:
  `assert set(list_step_types()) >= {4 core}` +
  `assert set(list_step_types()) <= KNOWN_TYPES`. Requires
  KNOWN_TYPES set that also goes stale. Not simpler.

**Chosen: (a)** — update expected list in each patch that adds a
built-in. Convention: when adding a built-in step to `_STEP_REGISTRY`,
update `test_list_step_types_default` in same commit.

**Правило (§13.4/172, расширение 152, 161):** closed lists ARE
stale-baits. Two valid patterns:
- **(a) Explicit update per add** (chosen here): visible churn in
  patch, review catches typos in new names.
- **(b) Property-based test** (open-set assert): doesn't work well
  for "exactly these N" because the assertion is the SPECIFICATION.
- **(c) No test** — loses the "no pollution" guard.

For small stable registries (built-in steps: 8), pattern (a) is
simplest. For large expanding registries (ML models: 10+), pattern
(b) would be better ("every registered type has from_dict", "no
Test* leaked"). Different scales, different patterns.

**Backlog (§11.6):** audit closed lists across tests:
- `test_register_step_type.py::test_list_step_types_default` — fixed
  (8 items).
- `test_api_availability.py::test_pyarrow_version_and_imports` —
  asserts `pc.X` exists (open set, safe).
- `test_api_availability.py::test_cadw_all_exports_importable` —
  iterates `__all__` (adaptive, safe).
- `test_no_pandas_in_ops.py::ALLOWED_PANDAS_MODULES` — closed set,
  1 item (shuffle.py). Adding new pandas exception → update needed.
  Same pattern as this episode.

**Связанные:** §13.4/152 (stale behavior), /161 (AST substring),
/167 (partial migration — stale check), /171 (test-authoring).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/172")


def append_changelog_fix2():
    print("\n=== CHANGELOG.md: v0.9.2f-fix2 note ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.2f-fix2" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Исправлено (v0.9.2f-fix2 — registry 4→8)

- **`test_list_step_types_default`** — ожидаемый список 4 → 8 items
  (v0.9.2f auto-registered 4 scalers via `_register_scalers()` at
  `cadw/scalers.py` import). Closed-list assert is intentional guard
  against test-registry pollution; grows with each new built-in.
- Rationale comment: built-ins updated in same patch that adds new
  step. Patch review catches typos in new step name (e.g.,
  `"StandartScaler"` → test fails, not silent).

### §13.4/172 — closed-list stale-bait on registry growth

Built-in registry grew 4→8. Closed-list test was stale-bait; not a
bug, auto-registration is intentional (`Pipeline.from_json` works for
scalers without setup). Chosen fix: (a) update expected list per patch
(visible churn, review catches typos). Alternative (b) property-based
superset check — deferred, KNOWN_TYPES goes stale too.

Convention: **when adding a built-in step, update
`test_list_step_types_default` in same commit.**

'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.2f-fix2 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_test_list_step_types()
    append_decisions_172()
    append_changelog_fix2()

    print("\n=== pytest: register_step_type (was 1 failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_register_step_type.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGISTER STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: scalers fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_scalers.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** SCALERS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: scalers E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_scalers_e2e.py", "-q", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline + cv regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_cv.py",
         "tests/test_pipeline_save_load.py",
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
        "v0.9.2f: scalers — StandardScaler/MinMaxScaler/MaxAbsScaler/"
        "RobustScaler; composed from Reduce+quantile; auto-registered "
        "with Pipeline (4→8 built-ins); tests/_helpers/scaler_reference.py; "
        "§13.4/170 (composition, no prototype; AST whitelist over-broad); "
        "§13.4/171 (test-authoring #7: deterministic-vs-statistical; "
        "reference helper); §13.4/172 (closed-list stale-bait on "
        "registry growth)"])

    print("\n==> v0.9.2f applied (v092f + fix1 + fix2 combined).")
    print("==> v0.9.2 COMPLETE (metrics + CV + scalers).")
    print("NEXT: git tag v0.9.2")


if __name__ == "__main__":
    main()