#!/usr/bin/env python3
# v0.9.26z2: Phase B close with caveats.
# B.2/B.3/B.5/B.6 → deferred to G; B → completed with caveats.
# Docs-only; 0 production. Bootstrap-safe.

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
TODAY = "2026-10-23"

DEFERRED_IDS = ["B.2", "B.3", "B.5", "B.6"]
DEFER_TARGET = "G"


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26z":
        raise SystemExit(f"expected HEAD=v0.9.26z, got {head!r}")
    print("HEAD: " + head)

    data = json.loads(Path(ROADMAP).read_text())
    phases = data.get("phases", [])
    phb = next(
        (ph for ph in phases
         if isinstance(ph, dict) and ph.get("id") == "B"),
        None,
    )
    if phb is None:
        raise SystemExit("Phase B not found")
    print(f"  B.status: {phb.get('status')!r}")
    if phb.get("status") == "completed":
        print("  ⚠ B already completed — idempotent mode")

    b1 = next(
        (it for it in phb.get("items", [])
         if isinstance(it, dict) and it.get("id") == "B.1"),
        None,
    )
    if b1 is None or b1.get("status") != "done":
        raise SystemExit("B.1 not done — v0.9.26z didn't apply?")

    # Verify B.2/B.3/B.5/B.6 exist and status
    for bid in DEFERRED_IDS:
        item = next(
            (it for it in phb.get("items", [])
             if isinstance(it, dict) and it.get("id") == bid),
            None,
        )
        if item is None:
            raise SystemExit(f"{bid} not found in Phase B items")
        print(f"  {bid}: status={item.get('status')!r}")

    return {"data": data, "phb": phb}


def update_roadmap(state):
    print("\n--- update roadmap.json ---")
    data = state["data"]
    phb = state["phb"]
    changes = []

    # Defer B.2/B.3/B.5/B.6 → G
    for it in phb.get("items", []):
        if not isinstance(it, dict):
            continue
        bid = it.get("id")
        if bid in DEFERRED_IDS and it.get("status") != "deferred":
            old = it.get("status")
            it["status"] = "deferred"
            it["deferred_to"] = DEFER_TARGET
            changes.append(f"{bid}: {old} → deferred to {DEFER_TARGET}")

    # Phase B → completed with caveats
    if phb.get("status") != "completed":
        phb["status"] = "completed"
        changes.append("B.status=completed")
    if not phb.get("completed_at"):
        phb["completed_at"] = TODAY
        changes.append(f"B.completed_at={TODAY}")

    caveats = phb.get("caveats")
    if not isinstance(caveats, list):
        caveats = []
    caveat_str = (
        "B.2/B.3/B.5/B.6 deferred to Phase G — DECISIONS split, "
        "pytest markers/xdist, scripts reorg, cleanup. All infra "
        "hygiene per roadmap 'no functional changes'. B.1 "
        "(dataframe split) is the primary B milestone and is done."
    )
    if caveat_str not in caveats:
        caveats.append(caveat_str)
        phb["caveats"] = caveats
        changes.append("B.caveats appended")

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


CHANGELOG_OLD = (
    "- **Phase B status:** `in_progress` — B.2 (DECISIONS split),\n"
    "  B.3 (pytest markers), B.5 (scripts reorg), B.6 (cleanup)\n"
    "  still queued. B.1 is the primary Phase B milestone; the\n"
    "  remaining tasks are secondary infra hygiene."
)

CHANGELOG_NEW = (
    "- **Phase B status:** `completed` with caveats — B.1 "
    "(dataframe split, primary milestone) done; B.2 (DECISIONS "
    "split), B.3 (pytest markers/xdist), B.5 (scripts reorg), "
    "B.6 (cleanup) deferred to Phase G. All four are infra "
    "hygiene per roadmap (\"no functional changes\") — no impact "
    "on CADW correctness. Phase B closed honestly with "
    "documented debt (§13.4/339, mirror of A.6 → G pattern)."
)


def update_changelog():
    print("\n--- update CHANGELOG.md ---")
    p = Path(CHANGELOG)
    src = p.read_text()
    if "B.2/B.3/B.5/B.6 deferred to Phase G" in src:
        print("  already updated — idempotent")
        return 0
    if CHANGELOG_OLD not in src:
        print("  WARN: exact old status text not found")
        print("  trying prefix match 'Phase B status:'")
        # Fallback: find by prefix, replace whole block
        m = re.search(
            r"\*\*Phase B status:\*\*.*?(?=\n\n|\n- \*\*|\Z)",
            src, re.DOTALL,
        )
        if not m:
            raise SystemExit(
                "CHANGELOG: no Phase B status found — inspect manually"
            )
        src = src[:m.start()] + CHANGELOG_NEW.rstrip() + src[m.end():]
        p.write_text(src)
        print("  CHANGELOG updated via prefix match")
        return 1
    src = src.replace(CHANGELOG_OLD, CHANGELOG_NEW, 1)
    p.write_text(src)
    print("  CHANGELOG updated (exact match)")
    return 1


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
    print("=== v0.9.26z2: Phase B close with caveats ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    state = verify_state()
    n_rm = update_roadmap(state)
    n_cl = update_changelog()

    print(f"\n  summary: {n_rm} roadmap changes, "
          f"{n_cl} CHANGELOG change")

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
        "v0.9.26z2: Phase B close with caveats (docs + roadmap). "
        "B.status: in_progress → completed; "
        "B.completed_at=2026-10-23. "
        "B.2/B.3/B.5/B.6: planned → deferred to Phase G "
        "(DECISIONS split, pytest markers/xdist, scripts reorg, "
        "cleanup — all infra hygiene per roadmap 'no functional "
        "changes'). B.caveats documents debt (mirror of A.6 → G "
        "pattern from v0.9.26d, §13.4/339). CHANGELOG v0.9.26 "
        "section: status text updated from in_progress to "
        "completed with caveats. Docs-only; 0 production. "
        "Test suite unchanged (2501 fast)."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26z2 COMPLETE. Phase B formally closed.")


if __name__ == "__main__":
    main()