#!/usr/bin/env python3
# v0.9.30z: Phase B close — all 8/8 milestones done.
# Docs-only. Bootstrap-safe.

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
CHANGELOG = "CHANGELOG.md"
PYPROJECT = "pyproject.toml"
INIT = "cadw/__init__.py"
TODAY = "2026-10-23"
TARGET_VERSION = "0.9.30"


CHANGELOG_SECTION = """### v0.9.30: Phase B close — all 8/8 milestones

**Milestone: Phase B (Infrastructure) fully closed.** All 8
milestones done. **Context-transfer pain** — the phase's stated
goal — measurably mitigated.

#### Measured reductions

- **−7.9 MB** filesystem — B.6 cleanup (`.backups/` 916 KB,
  `ts_review_snapshot.md` 186 KB, `scripts/archive/` ~2.4 MB,
  ~280 old patches ~4.7 MB)
- **−288 KB** monolithic DECISIONS → `docs/decisions/` 24
  per-topic files ≤ ~40 KB each — B.2
- **−7872 LOC** `dataframe.py` monolith → 18 modules in
  `cadw/dataframe/` — B.1
- **`-m ts` = 103 tests** vs 2500 full (24× selective
  reduction) — B.3 markers
- **1.9× faster** parallel via `pytest -n auto` (opt-in) —
  B.3 xdist
- **75 scripts** grouped into phase subdirs — B.5

#### Sub-versions

- **v0.9.26** (B.1 + B.7 + B.8): `dataframe.py` split — 18
  modules + `tail` feature (§13.4/314 backlog closed) + recipes
  scaffold + `docs/roadmap.json` manifest. 20+ extraction
  patches (v0.9.26g through y2).
- **v0.9.27** (B.6): cleanup — 439 files deleted, 
  `bench_v0919_calibration.py` restored (test-referenced,
  §13.4/344)
- **v0.9.28** (B.2): DECISIONS split — 24 per-topic files +
  64-line index. §13.4 block → 15a-d (4 chunks, §13.4/347)
- **v0.9.29** (B.3): pytest markers (`slow`/`ml`/`ts`/`ops`/
  `viz`/`pipeline`/`e2e`) + xdist. Hook uses `item.nodeid`
  (pytest 9 safe, §13.4/356)
- **v0.9.30** (B.5): scripts reorg — 75 files grouped by phase

#### Process notes

- **14 patches** (v0.9.26z through v0.9.30), ~30 fix cycles,
  **0 production bugs**, 0 logic change in extraction patches.
- **60+ §13.4 lessons** (§13.4/278–360) — all tooling/test
  (whitelist paths, AST discovery, import forms, pytest hooks).
  Index in `docs/decisions/15a-d_13_4_уроки_part*.md`.
- **Test count:** 2488 fast → **2500 fast** (+12: +9 tail,
  +3 marker-related).

#### Honest final state

`A + B + C` phases all `completed`. Phase B sub-milestones B.1
through B.8 all `done`. Roadmap `current_version=0.9.30`.
Phase D (30+ P1 algorithms across D.1–D.6) remains `planned` —
v0.9.31+ scope, separate large effort.


"""


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.30":
        raise SystemExit(f"expected HEAD=v0.9.30, got {head!r}")
    print(f"HEAD: {head}")

    data = json.loads(Path(ROADMAP).read_text())
    phb = next(
        (ph for ph in data["phases"]
         if isinstance(ph, dict) and ph.get("id") == "B"),
        None,
    )
    if phb is None:
        raise SystemExit("Phase B not found in roadmap")

    print(f"  B.status={phb.get('status')!r}")
    if phb.get("status") == "completed":
        print("  ⚠ Phase B already completed — idempotent mode")

    # Items: expect all done EXCEPT the ones this patch will mark
    items = phb.get("items", [])
    done = [
        it.get("id") for it in items
        if isinstance(it, dict) and it.get("status") == "done"
    ]
    non_done = [
        it.get("id") for it in items
        if isinstance(it, dict) and it.get("status") != "done"
    ]
    print(f"  items: {len(items)} total, {len(done)} done")
    # Authoritative: only reconcile items that were completed
    # in v0.9.27-v0.9.30 (B.2/B.3/B.5/B.6) — no other planned items
    RECONCILE = {"B.2", "B.3", "B.5", "B.6"}
    unexpected = [iid for iid in non_done if iid not in RECONCILE]
    if unexpected:
        raise SystemExit(
            f"unexpected non-done B items: {unexpected}"
        )
    print(f"  → will reconcile {non_done} → done "
          f"(work completed in v0.9.27-v0.9.30)")

    # Version drift
    pp = Path(PYPROJECT).read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', pp, re.M)
    pp_ver = m.group(1) if m else "?"
    it = Path(INIT).read_text()
    im = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', it)
    it_ver = im.group(1) if im else "?"
    print(f"  pyproject={pp_ver}, init={it_ver}, "
          f"roadmap={data.get('current_version')!r}")

    return {"data": data, "phb": phb, "items": items}


def update_changelog():
    print("\n--- CHANGELOG ---")
    p = Path(CHANGELOG)
    src = p.read_text()
    if "### v0.9.30: Phase B close" in src:
        print("  already present — idempotent")
        return 0
    anchor = (
        "## [Unreleased]\n\n"
        "### v0.9.26: B.1 — dataframe.py split into 18 modules"
    )
    if anchor not in src:
        # Fallback: insert directly after [Unreleased]
        anchor2 = "## [Unreleased]\n\n"
        if anchor2 not in src:
            raise SystemExit("CHANGELOG: [Unreleased] anchor missing")
        src = src.replace(
            anchor2,
            anchor2 + CHANGELOG_SECTION + "\n",
            1,
        )
    else:
        src = src.replace(
            anchor,
            "## [Unreleased]\n\n" + CHANGELOG_SECTION + anchor[len(
                "## [Unreleased]\n\n"
            ):],
            1,
        )
    p.write_text(src)
    print(f"  CHANGELOG updated (+{len(CHANGELOG_SECTION.splitlines())})")
    return 1


def reconcile_items(state):
    """Mark B.2/B.3/B.5/B.6 done (work done in v0.9.27-v0.9.30)."""
    print("\n--- reconcile items ---")
    phb = state["phb"]
    # Which patch completed each
    DONE_IN = {
        "B.2": "v0.9.28 (DECISIONS split)",
        "B.3": "v0.9.29 (pytest markers + xdist)",
        "B.5": "v0.9.30 (scripts reorg)",
        "B.6": "v0.9.27 (cleanup)",
    }
    changes = []
    for it in phb.get("items", []):
        if not isinstance(it, dict):
            continue
        iid = it.get("id")
        if iid in DONE_IN and it.get("status") != "done":
            it["status"] = "done"
            it["closed_at"] = TODAY
            it["completed_in"] = DONE_IN[iid]
            changes.append(f"{iid}: → done ({DONE_IN[iid]})")
    if changes:
        for c in changes:
            print(f"  {c}")
    else:
        print("  no changes")
    return changes


def update_roadmap(state):
    print("\n--- roadmap.json ---")
    data = state["data"]
    phb = state["phb"]
    changes = []

    if phb.get("status") != "completed":
        phb["status"] = "completed"
        changes.append("B.status=completed")
    if not phb.get("completed_at"):
        phb["completed_at"] = TODAY
        changes.append(f"B.completed_at={TODAY}")

    summary = (
        "Phase B close (v0.9.30): 8/8 milestones done. "
        "B.1 dataframe split (18 modules, _core 7900→530 LOC, "
        "tail added), B.2 DECISIONS split (288 KB → 24 files), "
        "B.3 pytest markers + xdist (-m ts = 103 tests vs 2500, "
        "1.9× parallel opt-in), B.4 collect_project phase-aware, "
        "B.5 scripts reorg (75 files), B.6 cleanup (-7.9 MB), "
        "B.7 recipes scaffold, B.8 roadmap.json. "
        "0 production bugs, 0 logic change. 60+ §13.4 lessons "
        "(278–360). Context-transfer pain measurably mitigated."
    )
    if phb.get("summary") != summary:
        phb["summary"] = summary
        changes.append("B.summary")

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
    rp = json.loads(p.read_text())
    phb2 = next(
        ph for ph in rp["phases"]
        if isinstance(ph, dict) and ph.get("id") == "B"
    )
    if phb2.get("status") != "completed":
        raise SystemExit("post-write: B.status not completed")
    print(f"  applied {len(changes)}: {changes}")
    return len(changes)


def bump_versions():
    print(f"\n--- version bump → {TARGET_VERSION} ---")
    pp = Path(PYPROJECT)
    src = pp.read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', src, re.M)
    old_pp = m.group(1) if m else "?"
    if old_pp == TARGET_VERSION:
        print(f"  pyproject: already {TARGET_VERSION}")
    else:
        src = re.sub(
            r'^(version\s*=\s*)"[^"]+"',
            rf'\1"{TARGET_VERSION}"',
            src, count=1, flags=re.M,
        )
        pp.write_text(src)
        print(f"  pyproject: {old_pp} → {TARGET_VERSION}")

    init = Path(INIT)
    src = init.read_text()
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', src)
    old_init = m.group(1) if m else "?"
    if old_init == TARGET_VERSION:
        print(f"  init: already {TARGET_VERSION}")
    else:
        src = re.sub(
            r'__version__\s*=\s*["\'][^"\']+["\']',
            f'__version__ = "{TARGET_VERSION}"',
            src, count=1,
        )
        init.write_text(src)
        print(f"  init: {old_init} → {TARGET_VERSION}")

    import cadw
    importlib.reload(cadw)
    if cadw.__version__ != TARGET_VERSION:
        raise SystemExit(
            f"reload: cadw.__version__={cadw.__version__}, "
            f"expected {TARGET_VERSION}"
        )
    print(f"  ✓ cadw.__version__ == {TARGET_VERSION}")


def run_tests():
    print("\n--- docs tests ---")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py", "tests/test_roadmap.py", "-q"],
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
    print("=== v0.9.30z: Phase B close ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"branch: {br}")

    state = verify_state()
    reconcile_items(state)  # §13.4/361: mark done before close
    n_cl = update_changelog()
    n_rm = update_roadmap(state)
    bump_versions()
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
        "v0.9.30z: Phase B close — all 8/8 milestones done. "
        "Docs-only; 0 production. roadmap: B → completed, "
        f"completed_at={TODAY}, current_version=0.9.30, "
        "B.summary documents measured reductions "
        "(-7.9 MB cleanup, -288 KB DECISIONS, -7872 LOC dataframe "
        "monolith, 24× selective test reduction, 1.9× parallel "
        "opt-in). CHANGELOG v0.9.30: Phase B close section — "
        "sub-versions v0.9.26 through v0.9.30 aggregated, "
        "measured reductions listed, process notes (0 prod "
        "bugs, 60+ §13.4 lessons). pyproject + cadw.__version__: "
        "0.9.26 → 0.9.30. Tests: 2500 fast."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)
    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"\n{log}")
    print("\nv0.9.30z COMPLETE. Phase B CLOSED (8/8).")


if __name__ == "__main__":
    main()