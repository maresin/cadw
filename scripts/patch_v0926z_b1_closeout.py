#!/usr/bin/env python3
# v0.9.26z: B.1 close-out (docs + roadmap + version).
# Docs-only; 0 production. Bootstrap-safe. §13.4/337 honest partial.

import ast
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

ROADMAP = "docs/roadmap.json"
DECISIONS = "DECISIONS.md"
CHANGELOG = "CHANGELOG.md"
PYPROJECT = "pyproject.toml"
INIT = "cadw/__init__.py"
TODAY = "2026-10-23"
TARGET_VERSION = "0.9.26"


# ============ verify state ============

def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26y2":
        raise SystemExit(f"expected HEAD=v0.9.26y2, got {head!r}")
    print("HEAD: " + head)

    # Files exist
    for p in (ROADMAP, DECISIONS, CHANGELOG, PYPROJECT, INIT):
        if not Path(p).exists():
            raise SystemExit(f"{p} missing")
    print(f"  all files present")

    # roadmap.json parses
    data = json.loads(Path(ROADMAP).read_text())
    print(f"  roadmap: current_version={data.get('current_version')!r}")

    # Find B.1
    phases = data.get("phases", [])
    phb = next(
        (ph for ph in phases
         if isinstance(ph, dict) and ph.get("id") == "B"),
        None,
    )
    if phb is None:
        raise SystemExit("Phase B not found")
    b1 = next(
        (it for it in phb.get("items", [])
         if isinstance(it, dict) and it.get("id") == "B.1"),
        None,
    )
    if b1 is None:
        raise SystemExit("B.1 not found")
    print(f"  B.1: status={b1.get('status')!r}, name={b1.get('name')!r}")
    if b1.get("status") == "done":
        print("  ⚠ B.1 already done — idempotent mode")

    # DECISIONS last §13.4
    dec = Path(DECISIONS).read_text()
    nums = re.findall(r'§13\.4/(\d+)', dec)
    last = max(int(n) for n in nums) if nums else 0
    print(f"  DECISIONS last §13.4 = {last}")

    # Compute new_num = max(last, 336) + 1
    phase_b_max = 336
    new_num = max(last, phase_b_max) + 1
    print(f"  → new §13.4/{new_num}")

    # CHANGELOG not yet has v0.9.26
    cl = Path(CHANGELOG).read_text()
    if "B.1 close-out" in cl and "dataframe.py split" in cl:
        print("  CHANGELOG: B.1 close-out already present — idempotent")
    elif "## [Unreleased]\n\n### v0.9.26:" in cl:
        print("  CHANGELOG: v0.9.26 exists (from v0.9.26n), "
              "insert B.1 section above")
    elif "## [Unreleased]\n\n### v0.9.25:" in cl:
        print("  CHANGELOG: anchor v0.9.25")
    else:
        raise SystemExit(
            "CHANGELOG: no anchor — inspect manually"
        )

    # pyproject version
    pp = Path(PYPROJECT).read_text()
    ppm = re.search(r'^version\s*=\s*"([^"]+)"', pp, re.M)
    cur_pp = ppm.group(1) if ppm else "MISSING"
    print(f"  pyproject.toml: {cur_pp}")

    # cadw.__version__
    init_text = Path(INIT).read_text()
    ivm = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']',
                    init_text)
    cur_init = ivm.group(1) if ivm else "MISSING"
    print(f"  cadw.__version__: {cur_init}")

    return {
        "data": data,
        "phb": phb,
        "b1": b1,
        "new_num": new_num,
        "pyproject_version": cur_pp,
        "init_version": cur_init,
    }


# ============ roadmap update ============

def update_roadmap(state):
    print("\n--- update roadmap.json ---")
    data = state["data"]
    b1 = state["b1"]

    changes = []
    if b1.get("status") != "done":
        b1["status"] = "done"
        changes.append("B.1.status=done")
    if not b1.get("closed_at"):
        b1["closed_at"] = TODAY
        changes.append(f"B.1.closed_at={TODAY}")
    note = (
        "v0.9.26 B.1 milestone: dataframe.py split into 18 modules "
        "across 20+ patches (v0.9.26g-y2). Original spec was 9; "
        "actual 18 due to subsystem granularity. _core.py 7900 → "
        "~530 LOC; ~7300 LOC redistributed to mixin modules. "
        "40+ §13.4 lessons learned (296-336). 0 production bugs, "
        "0 logic change."
    )
    if b1.get("notes") != note:
        b1["notes"] = note
        changes.append("B.1.notes")

    if data.get("current_version") != TARGET_VERSION:
        data["current_version"] = TARGET_VERSION
        changes.append(f"current_version={TARGET_VERSION}")
    if data.get("updated") != TODAY:
        data["updated"] = TODAY
        changes.append(f"updated={TODAY}")

    if not changes:
        print("  no changes — idempotent")
        return 0

    p = Path(ROADMAP)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    # round-trip verify
    rp = json.loads(p.read_text())
    if rp.get("current_version") != TARGET_VERSION:
        raise SystemExit("post-write: current_version not updated")
    print(f"  applied {len(changes)} changes: {changes}")
    return len(changes)


# ============ CHANGELOG ============

CHANGELOG_NEW = """## [Unreleased]

### v0.9.26: B.1 — dataframe.py split into 18 modules

**Milestone: `CadwDataFrame` monolith (7872 LOC in a single
`cadw/dataframe.py`) split into 18 modules in `cadw/dataframe/`.**
20+ extraction patches (v0.9.26g-y2), ~7300 LOC redistributed,
`_core.py` 7900 → ~530 LOC. All **0 logic change** (verbatim move)
except `tail` feature added (§13.4/314 backlog closed).

**Architecture:** `CadwDataFrame` = 14 mixins composed at class
definition:
```
class CadwDataFrame(
    _ExportsMixin, _IteratorsMixin, _MiscMixin, _IndexingMixin,
    _AggregationsMixin, _TransformersFillMixin,
    _TransformersReshapeMixin, _RollingMixin, _CumulativeMixin,
    _PerGroupMixin, _ResampleReshapeMixin, _JoinMixin,
    _MLBindingsMixin, _ColumnsMixin,
): ...
```

#### Modules (18)

| Module | LOC | Content |
|---|---|---|
| `_core.py` | ~530 | class + `__init__`, `__getitem__`, properties, `info`, `__repr__` |
| `_helpers.py` | ~170 | 5 module-level helpers (build/detect/normalize/infer/filter) |
| `_exports.py` | ~110 | `to_batches`, `write_parquet`, `from_*`, `concat` |
| `_iterators.py` | ~190 | `_iter_filtered_batches*`, `_iter_arrow_batches`, `_with_filter` |
| `_column_proxy.py` | ~320 | `_ColumnProxy` class (29 methods) |
| `_columns.py` | ~100 | `select_dtypes` |
| `_groupby.py` | ~220 | `_GroupBy` class (12 methods) |
| `_misc.py` | ~110 | `unique`, `_normalize_on`, `_groupby_impl`, `ts`, `plot` |
| `_indexing.py` | ~690 | `iloc/loc/at/iat`, `head`, `tail`, `set_index`, `reset_index`, `reindex`, `sort_values` |
| `_aggregations.py` | ~520 | `groupby`, `value_counts`, `corr`, `nunique`, `describe`, `nlargest`, `nsmallest`, `isna`, `isna_sum`, `quantile`, `count` |
| `_transformers_fill.py` | ~710 | `dropna`, `fillna`, `ffill`, `bfill`, `shift`, `diff`, `pct_change` |
| `_transformers_reshape.py` | ~650 | `astype`, `drop`, `rename`, `drop_duplicates`, `duplicated`, `cut`, `qcut` |
| `_rolling.py` | ~815 | `rolling`, `_rolling_on`, `_per_group_rolling`, `expanding`, `ewm` |
| `_cumulative.py` | ~820 | `cumsum`, `cumprod`, `cummax`, `cummin`, `_cumulative_impl`, `_per_group_cumulative`, `rank`, `sample` |
| `_per_group.py` | ~990 | `_shift_on`, `_diff_on`, `_per_group_shift`, `_per_group_diff`, `_per_group_apply` |
| `_resample_reshape.py` | ~425 | `resample`, `pivot_table`, `melt`, `crosstab` |
| `_join.py` | ~315 | `merge` |
| `_ml_bindings.py` | ~630 | `encode_categorical`, `metrics`, `fit_*` (13) |
| `__init__.py` | ~60 | package facade |

#### Added

- **`CadwDataFrame.tail(n=5)`** (§13.4/314 backlog closed) —
  pandas parity: n>0 last n, n=0 empty, n<0 all-but-first-k.
  Arrow-native via `pa.RecordBatch.slice`, no `to_pylist`.
  Mirror of `head`. 9 new tests.

#### Fixed (Phase B.1 lessons — §13.4/296-336)

See `DECISIONS.md` §13.4/337 for consolidated lessons index.
Highlights:
- **§13.4/296-302**: mixin extraction scope, `inspect.getsource`
  on packages, lazy import cycles, metadata composite fields.
- **§13.4/303-308**: `inspect.getsource` indented source;
  silent no-op exceptions; multi-line signatures; bootstrap
  lockout; helper deps; nested defs as locals.
- **§13.4/309-311**: no parent-scope subtraction; `ImportFrom`
  asname preservation; whitelist-grep in extraction.
- **§13.4/312-320**: typo fail-fast; sanity `hasattr`;
  `TARGET_SYMBOLS` from blueprint; comprehensive `method_locals`;
  `python -c` for-loop newlines; extended lazy injection;
  class-qualified whitelist keys; relative import level;
  `ImportFrom` asname emission.
- **§13.4/321-336**: repeated-bug patterns; monkeypatch target
  follows symbol after extraction; nested whitelist regex;
  shared helper re-export; import cleanup last; append pattern
  idempotency; auto-generated `__init__.py` needs full rewrite;
  sanity proxy vs frame distinction; backlog feature in
  dedicated patch; `sys.path.insert` boilerplate; guard-compliant
  by construction.

#### Process notes

- **20+ patches** (g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/x2/y/y2
  plus fixes) across ~10 sessions.
- **0 production bugs, 0 logic change** in extraction patches.
  All failures tooling (whitelist, discovery, imports, sanity,
  test targets).
- **Test count:** 2492 fast (baseline) → **2501** (+9 tail tests).
- **Phase B status:** `in_progress` — B.2 (DECISIONS split),
  B.3 (pytest markers), B.5 (scripts reorg), B.6 (cleanup)
  still queued. B.1 is the primary Phase B milestone; the
  remaining tasks are secondary infra hygiene.
- **Docs debt:** §13.4/296-336 lessons are in Phase B.1 commit
  messages; consolidated index in `DECISIONS.md` §13.4/337
  (mirror of Phase A §13.4/290 pattern).


### v0.9.25: Phase C — Pandas P0 API"""


def update_changelog():
    print("\n--- update CHANGELOG.md ---")
    p = Path(CHANGELOG)
    src = p.read_text()
    if "### v0.9.26:" in src:
        print("  already present — idempotent")
        return 0
    # Try both anchors: v0.9.26 (if exists) or v0.9.25
    anchor_b1 = (
        "## [Unreleased]\n\n"
        "### v0.9.26: Phase A close"
    )
    anchor_c = (
        "## [Unreleased]\n\n"
        "### v0.9.25: Phase C — Pandas P0 API"
    )
    if anchor_b1 in src:
        # Insert B.1 close-out section before existing v0.9.26
        # CHANGELOG_NEW ends with "### v0.9.25:", strip and use as header
        insert = CHANGELOG_NEW.replace(
            "### v0.9.25: Phase C — Pandas P0 API", ""
        ).rstrip() + "\n\n"
        # anchor_b1 starts with "## [Unreleased]\n\n### v0.9.26:..."
        # We want: "## [Unreleased]\n\n" + INSERT + "### v0.9.26:..."
        target = "## [Unreleased]\n\n### v0.9.26: Phase A close"
        replacement = "## [Unreleased]\n\n" + insert + "### v0.9.26: Phase A close"
        new = src.replace(target, replacement, 1)
        print(f"  inserted B.1 close-out before existing v0.9.26")
    elif anchor_c in src:
        new = src.replace(anchor_c, CHANGELOG_NEW, 1)
        print(f"  inserted at v0.9.25 anchor")
    else:
        raise SystemExit(
            "CHANGELOG: neither anchor found — inspect manually"
        )
    p.write_text(new)
    print(f"  CHANGELOG.md: {len(new.splitlines())} lines")
    return 1


# ============ DECISIONS ============

DECISIONS_BLOCK = """

#### §13.4/{NEW_NUM} — Phase B.1 lessons index (docs-debt backfill)

**Context:** B.1 `dataframe.py` split (20+ patches, v0.9.26g-y2)
recorded 40 meta-lessons §13.4/296-336 in **commit messages only**.
Same docs-debt pattern as Phase A closeout (§13.4/290) and Phase C
lessons (§13.4/281). This block consolidates them into one
DECISIONS entry. Individual sub-blocks not separately appended —
this index is the canonical record.

**Note on numbering:** §13.4/291-295 are unused (reserved/skipped).
Phase B lessons continue from §13.4/296 (as referenced in commit
messages). This index sits at §13.4/{NEW_NUM} (past all Phase B
lesson refs), mirroring Phase A closeout pattern (§13.4/290 = past
§13.4/282-289 refs).

| # | Lesson (1-line) | Patch | Status |
|---|---|---|---|
| 296 | Mixin extraction scope; `inspect.getsource(pkg)` reads `__init__.py`, not `_core.py` | v0.9.26g | applied |
| 297 | Metadata composite fields (`"path1, path2"`) require substring, not value replacement | v0.9.26g | applied |
| 298 | Same as 297 (dup tracking) | v0.9.26g | applied |
| 299 | Lazy import for circular `_core<->_exports` (mixin needs helper from parent) | v0.9.26h | applied |
| 300 | Off-by-one indent check (`< len(indent)+1`); fixed to `< len(indent)` | v0.9.26h2 | applied |
| 301 | Guard (`assert X not in src`) vs assertion (`assert X in src`) — opposite semantics under glob | v0.9.26h2 | applied |
| 302 | Already-extracted helpers in `dataframe_helpers.py`; discovery must not fail | v0.9.26i-corrected | applied |
| 303 | `inspect.getsource` returns indented source; `ast.parse` needs `textwrap.dedent` | v0.9.26i-fix1 | applied |
| 304 | Silent no-op exception (`except SyntaxError: return src`) hides failures | v0.9.26i-fix1 | applied |
| 305 | Multi-line signatures: insert point = sig-end, not `def+1` | v0.9.26i-fix2 | applied |
| 306 | Bootstrap lockout: recovery script must not `import cadw` on module level | v0.9.26i-fix3 | applied |
| 307 | Extraction misses module-level helper deps (bare `_helper()` calls) | v0.9.26i-fix4 | applied |
| 308 | `method_locals` needs nested `FunctionDef/ClassDef` names + nested args | v0.9.26i-fix4b | applied |
| 309 | Discovery: no parent-scope subtraction (imports are scope-local) | v0.9.26j-corrected | applied |
| 310 | `ImportFrom.asname` preserved in import map (`import X as Y`) | v0.9.26j-corrected | applied |
| 311 | Extraction scope must include whitelist-grep (all candidates) | v0.9.26j-fix1 | applied |
| 312 | Function-name typo → NameError before mutation; pre-execute grep check | v0.9.26l-fix1 | applied |
| 313 | Sanity: `hasattr` only, no API guesses (`df['x'].unique` wrong) | v0.9.26l-fix2 | applied |
| 314 | `TARGET_SYMBOLS` from diagnostic blueprint, not from Pandas API | v0.9.26m-fix1 | applied |
| 315 | `method_locals` must catch all binding constructs (nested args, `ExceptHandler.name`) | v0.9.26m-fix1 | applied |
| 316 | `python -c` with `for`/`try` needs newlines, not `;` | v0.9.26m-fix1 | applied |
| 317 | Sanity runtime API probe = false abort; use tests as authoritative | v0.9.26m-fix2 | applied |
| 318 | Whitelist keys class-qualified (`{file}:{Class}.{method}`) | v0.9.26m-fix3 | applied |
| 319 | Relative imports: `ast.ImportFrom.level` = dots count; emit `from {'.'*level}{module}` | v0.9.26n-fix1 | applied |
| 320 | `ImportFrom.asname` in emission (`from X import Y as Z`) | v0.9.26n-fix1 | applied |
| 321 | Repeated bug: sanity `for` pattern from broken source (fixed source copied from failed) | v0.9.26n-fix1 | applied |
| 322 | Extended lazy injection: handle all lazy names per method, not just `CadwDataFrame` | v0.9.26r | applied |
| 323 | Monkeypatch target follows symbol after extraction (`_core` → `_join`) | v0.9.26u-fix1 | applied |
| 324 | Nested whitelist keys need regex lookahead, not plain replace | v0.9.26v | applied |
| 325 | Shared helper extraction needs `_core.py` re-export for lazy imports | v0.9.26w | applied |
| 326 | Helpers > 100 LOC → extract; < 100 LOC → leave with re-export | v0.9.26w | applied |
| 327 | Test-fix diagnostic prints actual matched pattern, not guessed | v0.9.26w-fix1 | applied |
| 328 | Import cleanup = last step after all extractions | v0.9.26y | applied |
| 329 | Append-to-existing-mixin: idempotency via class body names + import dedupe | v0.9.26x2 | applied |
| 330 | Re-export reorganization: `__init__` imports from actual source module | v0.9.26y | applied |
| 331 | Auto-generated `__init__.py` needs full rewrite post-split, not edit | v0.9.26y-fix1 | applied |
| 332 | Sanity distinguishes `CadwDataFrame` methods vs `_ColumnProxy` (proxy accessor) methods | v0.9.26y-fix2 | applied |
| 333 | Sanity method list from blueprint, not Pandas API (`tail` never existed) | v0.9.26y-fix3 | applied |
| 334 | Backlog features in dedicated feature patch, not refactor patch | v0.9.26y2 | applied |
| 335 | `sys.path.insert(0, str(ROOT))` after `os.chdir(ROOT)` — mandatory boilerplate | v0.9.26y2-fix1 | applied |
| 336 | New methods in extraction-target modules: guard-compliant by construction (mirror existing pattern) | v0.9.26y2-fix1 | applied |

**Unified pattern** (building on §13.4/278-290): *"assume a summary
representation matches semantics."* Every lesson is a specialization
of "the string/count/scope I assumed is not the actual runtime/scope
semantics." Phase B extraction at scale (20 patches) forced
incremental learning of: AST-vs-string, scope-vs-parent, whitelist
patterns, import forms, monkeypatch targets, `python -c` syntax
boundaries, guard invariants.

**Docs debt status:** §13.4/291-295 unused; §13.4/296-336 referenced
from Phase B commits; this index = §13.4/{NEW_NUM}. Future Phase D+
lessons continue from §13.4/{NEXT_NUM}.
"""


def update_decisions(new_num):
    print(f"\n--- update DECISIONS.md (append §13.4/{new_num}) ---")
    p = Path(DECISIONS)
    src = p.read_text()
    if f"§13.4/{new_num}" in src and "Phase B.1 lessons index" in src:
        print("  already present — idempotent")
        return 0
    block = DECISIONS_BLOCK.replace("{NEW_NUM}", str(new_num))
    block = block.replace("{NEXT_NUM}", str(new_num + 1))
    p.write_text(src.rstrip() + block + "\n")
    # verify round-trip
    reparsed = p.read_text()
    if f"§13.4/{new_num}" not in reparsed:
        raise SystemExit(
            f"post-write: §13.4/{new_num} not found"
        )
    print(f"  appended §13.4/{new_num} "
          f"({len(block.splitlines())} lines)")
    return 1


# ============ version bump ============

def bump_versions(cur_pp, cur_init):
    print(f"\n--- version bump {cur_pp}/{cur_init} → {TARGET_VERSION} ---")
    # pyproject.toml
    pp_path = Path(PYPROJECT)
    pp_src = pp_path.read_text()
    pp_new = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\1"{TARGET_VERSION}"',
        pp_src, count=1, flags=re.M,
    )
    if pp_new != pp_src:
        pp_path.write_text(pp_new)
        print(f"  pyproject.toml: {cur_pp} → {TARGET_VERSION}")
    else:
        print(f"  pyproject.toml: no-op")

    # cadw/__init__.py
    init_path = Path(INIT)
    init_src = init_path.read_text()
    init_new = re.sub(
        r'__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{TARGET_VERSION}"',
        init_src, count=1,
    )
    if init_new != init_src:
        init_path.write_text(init_new)
        print(f"  cadw/__init__.py: {cur_init} → {TARGET_VERSION}")
    else:
        print(f"  cadw/__init__.py: no-op")

    # reload verify
    import cadw as _cadw
    importlib.reload(_cadw)
    if _cadw.__version__ != TARGET_VERSION:
        raise SystemExit(
            f"post-bump: cadw.__version__={_cadw.__version__}, "
            f"expected {TARGET_VERSION}"
        )
    print(f"  ✓ cadw.__version__ == {TARGET_VERSION}")


# ============ tests ============

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


# ============ main ============

def main():
    print("=== v0.9.26z: B.1 close-out ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    state = verify_state()
    n_rm = update_roadmap(state)
    n_cl = update_changelog()
    n_dc = update_decisions(state["new_num"])
    bump_versions(state["pyproject_version"], state["init_version"])
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
        f"v0.9.26z: B.1 close-out (docs + roadmap + version). "
        f"docs-only, 0 production. roadmap: B.1 → done "
        f"(closed_at={TODAY}, notes: 18 modules, ~7300 LOC "
        f"redistributed, 40+ lessons, 0 logic change); "
        f"B.status stays in_progress (B.2/B.3/B.5/B.6 still "
        f"planned); current_version={TARGET_VERSION}. "
        f"CHANGELOG v0.9.26 section: full milestone summary "
        f"(architecture, 18-module table, tail feature §13.4/314 "
        f"closed, Phase B.1 lessons index reference, honest "
        f"process notes). DECISIONS §13.4/{state['new_num']}: "
        f"Phase B.1 lessons consolidated index (§13.4/296-336, "
        f"40 lessons in one table, mirror of §13.4/290/281 "
        f"pattern). pyproject.toml + cadw.__version__: 0.9.25 → "
        f"{TARGET_VERSION}. Test suite: 2501 fast (unchanged)."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print(f"\nv0.9.26z COMPLETE. B.1 milestone closed.")


if __name__ == "__main__":
    main()
