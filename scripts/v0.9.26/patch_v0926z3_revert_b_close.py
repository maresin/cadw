#!/usr/bin/env python3
# v0.9.26z3: revert Phase B close (§13.4/341).
# B → in_progress, B.2/B.3/B.5/B.6 → planned, remove caveats.
# Docs-only. Bootstrap-safe.

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

ROADMAP = "docs/roadmap.json"
TODAY = "2026-10-23"
REVERT_IDS = ["B.2", "B.3", "B.5", "B.6"]


def main():
    print("=== v0.9.26z3: revert Phase B close ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26z2":
        raise SystemExit(f"expected HEAD=v0.9.26z2, got {head!r}")
    print("HEAD: " + head)

    p = Path(ROADMAP)
    data = json.loads(p.read_text())
    phb = next(
        ph for ph in data["phases"]
        if isinstance(ph, dict) and ph.get("id") == "B"
    )

    changes = []
    # B.status → in_progress
    if phb.get("status") != "in_progress":
        old = phb.get("status")
        phb["status"] = "in_progress"
        changes.append(f"B.status: {old} → in_progress")
    # Remove completed_at (was premature)
    if "completed_at" in phb:
        del phb["completed_at"]
        changes.append("B.completed_at: removed")
    # Remove caveats (was based on wrong deferral)
    if "caveats" in phb:
        del phb["caveats"]
        changes.append("B.caveats: removed")

    # Revert B.2/B.3/B.5/B.6 → planned
    for it in phb.get("items", []):
        if not isinstance(it, dict):
            continue
        bid = it.get("id")
        if bid in REVERT_IDS and it.get("status") == "deferred":
            it["status"] = "planned"
            if "deferred_to" in it:
                del it["deferred_to"]
            changes.append(f"{bid}: deferred → planned")

    data["updated"] = TODAY

    if not changes:
        print("  no changes — idempotent")
        return

    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    rp = json.loads(p.read_text())
    phb2 = next(
        ph for ph in rp["phases"]
        if isinstance(ph, dict) and ph.get("id") == "B"
    )
    if phb2.get("status") != "in_progress":
        raise SystemExit("post-write: B.status not in_progress")
    print(f"  applied {len(changes)}: {changes}")

    # Fast suite
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

    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26z3: revert Phase B close (§13.4/341). "
        "v0.9.26z2 closed B 'completed with caveats', deferring "
        "B.2/B.3/B.5/B.6 to Phase G. But Phase B's stated goal "
        "is 'remove context-transfer pain' — those 4 items ARE "
        "the goal (B.1 dataframe split alone is technical, not "
        "size-reducing for context transfer). Phase B stays "
        "in_progress. B.2/B.3/B.5/B.6 → planned. B.caveats and "
        "B.completed_at removed. Next: B.6 cleanup (biggest "
        "wins: .backups/ 916 KB, ts_review_snapshot.md 186 KB, "
        "scripts/archive/ 1.5 MB, ~600 old patches 4 MB)."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26z3 COMPLETE. Phase B in_progress again.")


if __name__ == "__main__":
    main()