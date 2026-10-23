#!/usr/bin/env python3
# v0.9.26b-corrected — roadmap consistency: Phase B only.
#
# PRIOR v0.9.26b attempt aborted in verify_state: Phase A has 3
# items (A.6/A.9/A.10) still status='planned' — cannot mark A
# completed. 0 mutations. HEAD stayed v0.9.26a.
#
# ROOT: my original verify asserted "all 5 patches v0.9.25a–e done →
# Phase A complete." WRONG. Roadmap accumulates ITEMS, not patches:
# Phase A has 11 items from 5 patches (A.1/A.2/A.3/A.3b/A.4/A.5/
# A.7/A.8 done in v0.9.25a–e; A.6/A.9/A.10 planned, not yet
# patched). Patches vs items is a many-to-one mapping — asserting
# on patch list is not equivalent to asserting on item list.
#
# §13.4/282 (NEW): phase-completion checks iterate ALL items (or
# abort if any planned/in_progress), not the patches list. Items
# may be added without patch, or bumped to planned AFTER patch
# shipped. This patch implements the correct semantic: A stays
# 'in_progress' because 3 items planned.
#
# Corrected scope:
#   - Phase B: status='planned' → 'in_progress', started_at,
#     patches=['v0.9.26b']. This is honest — B.1 (dataframe.py
#     split, v0.9.26) starts now.
#   - Phase A: UNTOUCHED. Stays 'in_progress'. Honest — 3 items
#     planned.
#   - top.updated = 2026-10-23.
#
# Diagnostic info: verify_state prints A.6/A.9/A.10 detail for
# next-step decision (v0.9.26c-diagnostic-A: inspect actual code
# state of these 3 items — hasattr/grep, no mutation).
#
# §13.4/281 applied: HEAD via exact subject token. §13.4/279
# applied: idempotent if B already in_progress. §13.4/264 applied:
# json round-trip pre+post.
#
# Run: uv run python scripts/patch_v0926b_roadmap_consistency_corrected.py

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
    print("\n=== verify: HEAD=v0.9.26a; Phase A items NOT all done; "
          "Phase B is planned; B.1 patch target v0.9.26 ===")

    # §13.4/281: exact-token HEAD
    subj = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head_version = subj.split(":", 1)[0].strip()
    if head_version != "v0.9.26a":
        raise SystemExit(
            f"expected HEAD subject 'v0.9.26a:'; got "
            f"{head_version!r} (subj[:200]={subj[:200]!r})"
        )
    if head_version == "v0.9.26b":
        raise SystemExit("v0.9.26b already committed — misplaced")
    print(f"  HEAD = {head_version} ✓")

    p = Path(ROADMAP)
    data = json.loads(p.read_text())  # §13.4/264 round-trip
    print(f"  {ROADMAP}: parsed OK")

    phases = data.get("phases")
    if not isinstance(phases, list):
        raise SystemExit("phases not a list")

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
        raise SystemExit("Phase A not found")
    if phb is None:
        raise SystemExit("Phase B not found")

    # --- Phase A: diagnostic only (NO assertion on completion) ---
    items_a = pha.get("items") or []
    done_a, planned_a = [], []
    for it in items_a:
        if not isinstance(it, dict):
            continue
        iid = it.get("id", "?")
        ist = it.get("status")
        (done_a if ist == "done" else planned_a).append(iid)
    print(f"\n  Phase A: status={pha.get('status')!r}, "
          f"items done={len(done_a)} planned={len(planned_a)}")
    if planned_a:
        print(f"  ℹ Phase A has {len(planned_a)} planned items: "
              f"{planned_a}")
        print(f"    → A stays 'in_progress' (honest). "
              f"Closeout deferred to v0.9.26c.")
        print(f"    Detailed planned items for v0.9.26c-diagnostic:")
        for it in items_a:
            if (isinstance(it, dict)
                    and it.get("status") == "planned"):
                print(f"      {it.get('id')}: "
                      f"name={it.get('name')!r} "
                      f"complexity={it.get('complexity', '?')!r} "
                      f"patch={it.get('patch', '?')!r}")

    # --- Phase B: scope check — B.1 split, v0.9.26 ---
    items_b = phb.get("items") or []
    print(f"\n  Phase B: status={phb.get('status')!r}, "
          f"items={len(items_b)}")
    for it in items_b:
        if not isinstance(it, dict):
            continue
        iid = it.get("id", "?")
        patch = it.get("patch", "?")
        cx = it.get("complexity", "?")
        ist = it.get("status", "?")
        print(f"    {iid}: status={ist!r} patch={patch!r} "
              f"complexity={cx!r} name={it.get('name', '?')!r}")

    if phb.get("status") == "in_progress":
        print(f"  ⚠ Phase B already 'in_progress' — idempotent mode")

    return {
        "data": data,
        "phase_a": pha,
        "phase_b": phb,
        "planned_a": planned_a,
    }


# ============ 1. apply (Phase B only) ============

def apply_changes(state):
    print("\n=== apply Phase B transition ===")
    data = state["data"]
    phb = state["phase_b"]
    pha = state["phase_a"]
    planned_a = state["planned_a"]

    changes = []
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

    # Phase A stays untouched — print what's happening explicitly
    print(f"  Phase A: status={pha.get('status')!r} — UNTOUCHED "
          f"({len(planned_a)} items still planned: {planned_a})")
    print(f"  Phase B: status='planned' → 'in_progress' — B.1 "
          f"dataframe.py split starts now")

    if not changes:
        print("  no changes — idempotent")
        return 0, planned_a

    p = Path(ROADMAP)
    backup = p.with_suffix(".json.bak")
    shutil.copy2(p, backup)
    print(f"  backup → {backup}")

    p.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )

    # §13.4/264: reparse + verify Phase B mutated, Phase A intact
    reparsed = json.loads(p.read_text())
    reparsed_b = next(
        ph for ph in reparsed["phases"]
        if isinstance(ph, dict) and ph.get("id") == "B"
    )
    reparsed_a = next(
        ph for ph in reparsed["phases"]
        if isinstance(ph, dict) and ph.get("id") == "A"
    )
    if reparsed_b.get("status") != "in_progress":
        raise SystemExit(
            "post-write: Phase B status not 'in_progress'"
        )
    if reparsed_a.get("status") != "in_progress":
        raise SystemExit(
            f"post-write: Phase A status changed from "
            f"'in_progress' to {reparsed_a.get('status')!r} — "
            f"this patch must NOT modify Phase A"
        )
    print(f"  applied {len(changes)} changes: {changes}")
    print(f"  ✓ Phase A status unchanged (still 'in_progress')")
    return len(changes), planned_a


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
        print("\n*** DOCS TEST FAILURE ***")
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
    print("=== v0.9.26b-corrected: Phase B transition only ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    state = verify_state()
    n_changes, planned_a = apply_changes(state)
    run_tests()

    bak = Path(ROADMAP + ".bak")
    if bak.exists():
        bak.unlink()
        print(f"  cleaned up {bak} (git holds original)")

    planned_a_str = ", ".join(planned_a) if planned_a else "none"

    commit_with_guards(
        "v0.9.26b: roadmap consistency — Phase B → in_progress "
        "(Phase A stays in_progress, honest). "
        f"Metadata-only ({n_changes} field changes in "
        f"docs/roadmap.json). No code, no CHANGELOG, no "
        f"DECISIONS. "
        "PRIOR v0.9.26b attempt aborted in verify_state: Phase A "
        f"has {len(planned_a)} items still 'planned' "
        f"({planned_a_str}) — cannot mark A completed. 0 mutations, "
        "HEAD stayed v0.9.26a. §13.4/280 (actual inspection) "
        "prevented a lie. "
        "§13.4/282 (NEW): phase-completion checks iterate ALL "
        "items, not patches list. Phase A = 11 items from 5 "
        "patches (v0.9.25a–e); 8 done, 3 planned (A.6 '18 "
        "to_pylist sites extra_factor', A.9 'B5 dead code "
        "_append_to_builders', A.10 'B4 memory_guard fallback "
        "factor'). Patches-vs-items is many-to-one; asserting on "
        "patch list is not equivalent. "
        "Scope (corrected, honest): "
        "(1) Phase B: status='planned' → 'in_progress', "
        "started_at=2026-10-23, patches=['v0.9.26b']. B.1 = "
        "dataframe.py split into 9 modules (v0.9.26, complexity "
        "L) starts now. B.2 = DECISIONS.md split (v0.9.27) "
        "queued. "
        "(2) Phase A: UNTOUCHED. Stays 'in_progress' — 3 planned "
        "items honest. Closeout deferred to v0.9.26c (inspect "
        "A.6/A.9/A.10 actual code state, then patch). "
        "(3) top.updated=2026-10-23. "
        "(4) B.1, B.2 items stay 'planned' — no production code "
        "yet (honest). "
        "§13.4/281 applied: HEAD via exact subject token. "
        "§13.4/279 applied: idempotent skip if B already "
        "in_progress. "
        "§13.4/264 applied: json round-trip pre+post; explicit "
        "post-write assertion that Phase A status UNCHANGED "
        "(this patch must NOT touch A). "
        "Backup: docs/roadmap.json.bak → deleted after pytest "
        "passes. "
        "Test counts: unchanged (~2488 fast) — metadata-only. "
        "Next: v0.9.26c-diagnostic-A (read-only: hasattr/grep "
        "actual A.6/A.9/A.10 state, cross-ref "
        "_FOR_PINGIS/cadw_audit_report.md; decide patch vs "
        "close-as-done), then A closeout patch if needed, then "
        "v0.9.26d-dataframe_blueprint (B.1 planning).",
        self_path="scripts/patch_v0926b_roadmap_consistency_corrected.py",
        check_whitelist=True,
    )
    print(f"\n==> v0.9.26b COMPLETE. Phase B=in_progress; "
          f"Phase A honest (3 items: {planned_a_str}). "
          f"Next: v0.9.26c-diagnostic-A.")


if __name__ == "__main__":
    main()