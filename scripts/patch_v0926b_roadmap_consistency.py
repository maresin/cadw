#!/usr/bin/env python3
# v0.9.26b — roadmap consistency: A→completed, B→in_progress.
#
# Metadata-only. No code, no tests touched (test_roadmap.py +
# test_changelog.py validate).
#
# Context (diag_v0926b, v0.9.26b-diagnostic):
#   - Phase A status='in_progress' but all 5 items (A.1–A.5,
#     v0.9.25a–e) are status='done'. Stale field.
#   - Phase B status='planned', version_range v0.9.26–v0.9.27.
#     B.1 = dataframe.py split into 9 modules (v0.9.26, L).
#     B.2 = DECISIONS.md split into docs/ tree (v0.9.27).
#     We're starting B now (this patch is B's first step).
#   - Phase D = v0.9.31–v0.9.36, 6 sub-phases, 30+ features
#     (IForest, Matrix Profile, LambdaMART, ALS, CoxPH,
#     MiniRocket, etc.) — NOT next scope.
#
# Corrections:
#   - Phase A: status → completed, completed_at=2026-10-23
#     (audit-confirmation date; last actual patch v0.9.25e).
#   - Phase B: status → in_progress, started_at=2026-10-23,
#     patches=["v0.9.26b"] (this patch is B's first).
#   - top.updated = 2026-10-23.
#   - Phase B items (B.1, B.2) stay 'planned' — no production
#     code yet. Honest.
#
# §13.4/281 applied: HEAD check via git log --format=%s -1 exact
# subject token. §13.4/280 applied: Phase A 'completed' only if
# ALL items status='done'; abort otherwise. §13.4/279 applied:
# idempotent skip if A already completed and B already
# in_progress. §13.4/264 applied: json.loads round-trip pre+post.
#
# Backup: docs/roadmap.json.bak — deleted after pytest passes.
# If test_roadmap.py asserts old statuses → v0.9.26b-fix1 updates
# the test (honest — actual state now matches roadmap intent).
#
# Run: uv run python scripts/patch_v0926b_roadmap_consistency.py

from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


ROADMAP = "docs/roadmap.json"
TODAY = "2026-10-23"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.26a; Phase A items done; "
          "Phase B items readable ===")

    # §13.4/281: exact-token HEAD check
    subj = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head_version = subj.split(":", 1)[0].strip()
    if head_version != "v0.9.26a":
        raise SystemExit(
            f"expected HEAD subject prefix 'v0.9.26a:'; "
            f"got {head_version!r} (subj[:200]={subj[:200]!r})"
        )
    if head_version == "v0.9.26b":
        raise SystemExit("v0.9.26b already committed — misplaced")
    print(f"  HEAD = {head_version} ✓")

    p = Path(ROADMAP)
    if not p.exists():
        raise SystemExit(f"missing {p}")
    data = json.loads(p.read_text())  # §13.4/264: round-trip check
    print(f"  {ROADMAP}: parsed OK, "
          f"current_version={data.get('current_version')!r}")

    phases = data.get("phases")
    if not isinstance(phases, list):
        raise SystemExit(
            f"phases not a list: {type(phases).__name__}"
        )

    pha = next(
        (ph for ph in phases
         if isinstance(ph, dict) and ph.get("id") == "A"),
        None,
    )
    phb = next(
        (ph for ph in phases
         if isinstance(ph, dict) and ph.get("id") == "B"),
        None,
    )
    if pha is None:
        raise SystemExit("Phase A (id='A') not found in phases")
    if phb is None:
        raise SystemExit("Phase B (id='B') not found in phases")

    # Phase A: assert ALL items status='done' (§13.4/280 — actual
    # inspection, no lying about completion)
    items_a = pha.get("items") or []
    print(f"\n  Phase A: status={pha.get('status')!r}, "
          f"name={pha.get('name')!r}, items={len(items_a)}")
    not_done = []
    for it in items_a:
        if not isinstance(it, dict):
            raise SystemExit(
                f"Phase A item not dict: {type(it).__name__}"
            )
        iid = it.get("id", "?")
        ist = it.get("status")
        print(f"    {iid}: status={ist!r} "
              f"name={it.get('name', '?')!r}")
        if ist != "done":
            not_done.append(iid)
    if not_done:
        raise SystemExit(
            f"Phase A has non-done items: {not_done} — cannot "
            f"mark phase completed. Fix items first or abort."
        )
    print(f"  ✓ all {len(items_a)} Phase A items status='done'")

    # Phase B: inspect items
    items_b = phb.get("items") or []
    print(f"\n  Phase B: status={phb.get('status')!r}, "
          f"name={phb.get('name')!r}, items={len(items_b)}")
    for it in items_b:
        if not isinstance(it, dict):
            continue
        iid = it.get("id", "?")
        ist = it.get("status")
        patch = it.get("patch")
        cx = it.get("complexity")
        print(f"    {iid}: status={ist!r} patch={patch!r} "
              f"complexity={cx!r} name={it.get('name', '?')!r}")

    # Idempotency check
    a_st = pha.get("status")
    b_st = phb.get("status")
    if a_st == "completed" and b_st == "in_progress":
        print("\n  ⚠ already applied (A=completed, "
              "B=in_progress) — idempotent mode")
    else:
        print(f"\n  → will apply: A {a_st!r} → 'completed'; "
              f"B {b_st!r} → 'in_progress'")

    return {
        "data": data,
        "phase_a": pha,
        "phase_b": phb,
        "items_a_count": len(items_a),
    }


# ============ 1. apply ============

def apply_changes(state):
    print("\n=== apply roadmap consistency changes ===")
    data = state["data"]
    pha = state["phase_a"]
    phb = state["phase_b"]

    changes = []
    if pha.get("status") != "completed":
        pha["status"] = "completed"
        changes.append("A.status=completed")
    if not pha.get("completed_at"):
        pha["completed_at"] = TODAY
        changes.append(
            f"A.completed_at={TODAY} (audit-confirmation; "
            f"last patch v0.9.25e)"
        )
    if phb.get("status") != "in_progress":
        phb["status"] = "in_progress"
        changes.append("B.status=in_progress")
    if not phb.get("started_at"):
        phb["started_at"] = TODAY
        changes.append(f"B.started_at={TODAY}")
    patches_b = phb.get("patches")
    if not isinstance(patches_b, list):
        patches_b = []
    if "v0.9.26b" not in patches_b:
        patches_b.append("v0.9.26b")
        phb["patches"] = patches_b
        changes.append(f"B.patches={patches_b}")
    if data.get("updated") != TODAY:
        data["updated"] = TODAY
        changes.append(f"top.updated={TODAY}")

    if not changes:
        print("  no changes — idempotent")
        return 0

    p = Path(ROADMAP)
    backup = p.with_suffix(".json.bak")
    shutil.copy2(p, backup)
    print(f"  backup → {backup}")

    p.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )

    # §13.4/264: re-parse post-write
    reparsed = json.loads(p.read_text())
    if reparsed.get("updated") != TODAY:
        raise SystemExit(
            "post-write verification failed: top.updated not set"
        )
    reparsed_a = next(
        ph for ph in reparsed["phases"]
        if isinstance(ph, dict) and ph.get("id") == "A"
    )
    if reparsed_a.get("status") != "completed":
        raise SystemExit(
            "post-write: Phase A status not 'completed'"
        )
    print(f"  applied {len(changes)} changes: {changes}")
    return len(changes)


# ============ 2. tests ============

def run_tests():
    print("\n=== pre-flight: facade AST guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "tests/test_dataframe.py::test_facade_has_no_row_loops",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FACADE AST GUARD FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: roadmap + changelog docs tests ===")
    targets = ["tests/test_changelog.py"]
    if Path("tests/test_roadmap.py").exists():
        targets.append("tests/test_roadmap.py")
    r = subprocess.run(
        ["uv", "run", "pytest", *targets,
         "-q", "-W", "error::SyntaxWarning"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** DOCS TEST FAILURE — "
              "test likely asserts old A/B statuses ***")
        print("  → v0.9.26b-fix1 updates test_roadmap.py to accept "
              "new state (honest — actual now matches roadmap "
              "intent).")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 3. main ============

def main():
    print("=== v0.9.26b: roadmap consistency ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    state = verify_state()
    n_changes = apply_changes(state)
    run_tests()

    # Cleanup .bak after success
    bak = Path(ROADMAP + ".bak")
    if bak.exists():
        bak.unlink()
        print(f"  cleaned up {bak} (git holds original)")

    commit_with_guards(
        "v0.9.26b: roadmap consistency — A→completed, "
        "B→in_progress. "
        f"Metadata-only ({n_changes} field changes in "
        f"docs/roadmap.json). No code, no CHANGELOG, no "
        f"DECISIONS. "
        "Context (diag_v0926b, v0.9.26b-diagnostic): Phase A "
        "status='in_progress' but all 5 items (A.1–A.5, "
        "v0.9.25a–e) status='done' — stale field. Phase B "
        "status='planned' with B.1 = dataframe.py split into 9 "
        "modules (v0.9.26, L) + B.2 = DECISIONS.md split into "
        "docs/ tree (v0.9.27) — we're starting B now (this patch "
        "is B's first step). Phase D = v0.9.31–v0.9.36, 6 "
        "sub-phases, 30+ features (IForest, Matrix Profile, "
        "LambdaMART, ALS, CoxPH, MiniRocket, etc.) — NOT next "
        "scope. "
        "Changes: "
        "(1) Phase A: status='completed', "
        "completed_at=2026-10-23 (audit-confirmation date; last "
        "actual patch v0.9.25e). "
        "(2) Phase B: status='in_progress', started_at=2026-10-23, "
        "patches=['v0.9.26b']. "
        "(3) top.updated=2026-10-23. "
        "(4) B.1 and B.2 items stay 'planned' — no production "
        "code yet (honest). "
        "§13.4/281 applied: HEAD check via git log --format=%s -1 "
        "exact subject token (v0.9.26a). "
        "§13.4/280 applied: Phase A 'completed' only if ALL items "
        "status='done'; abort otherwise (don't lie about "
        "completion). "
        "§13.4/279 applied: idempotent skip if A already "
        "'completed' and B already 'in_progress'. "
        "§13.4/264 applied: json.loads round-trip pre+post write; "
        "reparse verifies top.updated + Phase A status. "
        "Backup: docs/roadmap.json.bak → deleted after pytest "
        "passes. If test_roadmap.py asserts old A/B statuses → "
        "v0.9.26b-fix1 updates test (honest — actual now matches "
        "roadmap intent). "
        "Test counts: unchanged (~2488 fast) — metadata-only. "
        "Next: v0.9.26c-dataframe_blueprint (B.1 blueprint — "
        "analyze dataframe.py structure via AST, propose 9-module "
        "split map, list public API surface + AST-guard test "
        "dependencies; read-only, no code changes), then "
        "v0.9.26d+ incremental extraction (one module per "
        "sub-patch, 2488 tests as safety net).",
        self_path="scripts/patch_v0926b_roadmap_consistency.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.26b COMPLETE. Roadmap A=completed, "
          "B=in_progress. Next: v0.9.26c-dataframe_blueprint "
          "(B.1 planning — read-only).")


if __name__ == "__main__":
    main()