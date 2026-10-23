#!/usr/bin/env python3
# v0.9.26z2-fix1: insert missing v0.9.26 B.1 CHANGELOG section.
# v0.9.26z guard silently skipped it (matched v0.9.26n section).
# Insert with "completed with caveats" text baked in. Docs-only.

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

ROADMAP = "docs/roadmap.json"
CHANGELOG = "CHANGELOG.md"


CHANGELOG_B1 = """### v0.9.26: B.1 — dataframe.py split into 18 modules

**Milestone: `CadwDataFrame` monolith (7872 LOC in a single
`cadw/dataframe.py`) split into 18 modules in `cadw/dataframe/`.**
20+ extraction patches (v0.9.26g-y2), ~7300 LOC redistributed,
`_core.py` 7900 → ~530 LOC. All **0 logic change** (verbatim move)
except `tail` feature added (§13.4/314 backlog closed).

**Architecture:** `CadwDataFrame` = 14 mixins composed at class
definition:
```python
class CadwDataFrame(
    _ExportsMixin, _IteratorsMixin, _MiscMixin, _IndexingMixin,
    _AggregationsMixin, _TransformersFillMixin,
    _TransformersReshapeMixin, _RollingMixin, _CumulativeMixin,
    _PerGroupMixin, _ResampleReshapeMixin, _JoinMixin,
    _MLBindingsMixin, _ColumnsMixin,
): ...
```

#### Modules (18)

`_core` (~530), `_helpers` (~170), `_exports` (~110),
`_iterators` (~190), `_column_proxy` (~320), `_columns` (~100),
`_groupby` (~220), `_misc` (~110), `_indexing` (~690),
`_aggregations` (~520), `_transformers_fill` (~710),
`_transformers_reshape` (~650), `_rolling` (~815),
`_cumulative` (~820), `_per_group` (~990),
`_resample_reshape` (~425), `_join` (~315), `_ml_bindings`
(~630), `__init__` (~60).

#### Added

- **`CadwDataFrame.tail(n=5)`** (§13.4/314 backlog closed) —
  pandas parity: n>0 last n, n=0 empty, n<0 all-but-first-k.
  Arrow-native via `pa.RecordBatch.slice`, no `to_pylist`
  (§13.4/336 guard-compliant). Mirror of `head`. 9 new tests.

#### Process notes

- **20+ patches** (v0.9.26g through y2, plus ~30 fix patches)
  across ~10 sessions.
- **0 production bugs, 0 logic change** in extraction patches.
  All failures tooling (whitelist, discovery, imports, sanity,
  test targets).
- **Test count:** 2492 fast baseline → **2501** (+9 tail).
- **Phase B status:** `completed` with caveats — B.1 (dataframe
  split, primary milestone) done; B.2 (DECISIONS split), B.3
  (pytest markers/xdist), B.5 (scripts reorg), B.6 (cleanup)
  deferred to Phase G. All four are infra hygiene per roadmap
  ("no functional changes") — no impact on CADW correctness.
  Phase B closed honestly with documented debt (§13.4/339,
  mirror of A.6 → G pattern from v0.9.26d).
- **Docs debt:** §13.4/296-336 lessons are in Phase B.1 commit
  messages; consolidated index in `DECISIONS.md` §13.4/337
  (mirror of Phase A §13.4/290 pattern).


"""


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26z":
        raise SystemExit(f"expected HEAD=v0.9.26z, got {head!r}")
    print("HEAD: " + head)

    # Roadmap z2 changes should be applied (dirty tree)
    data = json.loads(Path(ROADMAP).read_text())
    phb = next(
        (ph for ph in data["phases"]
         if isinstance(ph, dict) and ph.get("id") == "B"),
        None,
    )
    if phb is None:
        raise SystemExit("Phase B not found")
    if phb.get("status") != "completed":
        raise SystemExit(
            f"Phase B status={phb.get('status')!r}, expected "
            f"'completed' — v0.9.26z2 roadmap update didn't apply?"
        )
    print("  roadmap: B.status=completed ✓ (z2 applied)")

    # CHANGELOG should NOT yet have B.1 close-out
    cl = Path(CHANGELOG).read_text()
    if "### v0.9.26: B.1 — dataframe.py split" in cl:
        raise SystemExit(
            "CHANGELOG already has v0.9.26 B.1 section — fix1 "
            "already applied?"
        )
    print("  CHANGELOG: v0.9.26 B.1 section absent (needs insert)")

    # Anchor exists
    if "## [Unreleased]\n\n### v0.9.26: Phase A close" not in cl:
        raise SystemExit(
            "CHANGELOG anchor '## [Unreleased]\\n\\n### v0.9.26: "
            "Phase A close' not found — inspect manually"
        )
    print("  CHANGELOG: anchor OK")


def insert_section():
    print("\n--- insert v0.9.26 B.1 section ---")
    p = Path(CHANGELOG)
    src = p.read_text()
    anchor = "## [Unreleased]\n\n### v0.9.26: Phase A close"
    replacement = (
        "## [Unreleased]\n\n"
        + CHANGELOG_B1
        + "### v0.9.26: Phase A close"
    )
    new = src.replace(anchor, replacement, 1)
    if new == src:
        raise SystemExit("no change — anchor replace failed")
    p.write_text(new)
    print(f"  CHANGELOG: {len(src.splitlines())} → "
          f"{len(new.splitlines())} lines "
          f"(+{len(CHANGELOG_B1.splitlines())})")


def run_tests():
    print("\n--- docs tests ---")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_roadmap.py", "tests/test_changelog.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        raise SystemExit("docs tests FAILED")

    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26z2-fix1: CHANGELOG B.1 insert ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()
    insert_section()
    run_tests()

    print("\n--- verify_whitelist ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.patch_utils import verify_whitelist; "
         "verify_whitelist(raise_on_orphan=True); print('WL OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("whitelist failed")
    print(r.stdout.strip())

    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26z2: Phase B close with caveats + fix1 (CHANGELOG "
        "B.1 insert). "
        "Roadmap (from z2): B.status → completed; "
        "B.completed_at=2026-10-23; B.2/B.3/B.5/B.6 → deferred "
        "to Phase G; B.caveats documents debt (§13.4/339). "
        "CHANGELOG (fix1): v0.9.26 B.1 close-out section — "
        "v0.9.26z's guard `if '### v0.9.26:' in src: return 0` "
        "silently matched v0.9.26n's Phase A close section, so "
        "the intended B.1 milestone section was never inserted. "
        "fix1 inserts it with 'Phase B status: completed with "
        "caveats' text baked in. Docs-only; 0 production. "
        "Tests: 2501 fast."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26z2 COMPLETE (fix1). Phase B closed.")


if __name__ == "__main__":
    main()