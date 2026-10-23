#!/usr/bin/env python3
# v0.9.30: B.5 scripts reorg. Group patch_* by phase dir; keep
# test-referenced + active tools at top. Bootstrap-safe.

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

SCRIPTS = Path("scripts")
TESTS = Path("tests")

# Files that MUST stay at scripts/ top level (test-referenced or active)
KEEP_TOP = {
    "__init__.py",
    "_lib.sh",
    "collect_project.py",          # test_collect_project_modes.py
    "snapshot_ts_review.py",       # regenerates ts_review_snapshot.md
    "bench_v0919_calibration.py",  # test_runtime_monitor.py
}

# Files that must stay inside scripts/service/ (test-referenced)
KEEP_SERVICE = {
    "diagnose_phase_b_split.py",
    "diagnose_unguarded_to_pylist.py",  # test_v0925f_a.py
}

# Group rules: prefix -> subdir
PHASE_GROUPS = {
    "patch_v0926": "v0.9.26",
    "patch_v0927": "v0.9.27",
    "patch_v0928": "v0.9.28",
    "patch_v0929": "v0.9.29",
    "diag_": "diag",
}


def discover():
    print("=" * 72)
    print("DISCOVERY: scripts/ inventory")
    print("=" * 72)

    items = list(SCRIPTS.glob("*.py")) + list(SCRIPTS.glob("*.sh"))
    print(f"\nTop-level scripts: {len(items)}")
    for f in sorted(items):
        print(f"  {f.name}")

    service = list((SCRIPTS / "service").glob("*.py")) \
        if (SCRIPTS / "service").exists() else []
    print(f"\nservice/: {len(service)}")
    for f in sorted(service):
        print(f"  service/{f.name}")

    # Test references
    print(f"\nTEST REFERENCES to scripts/ (grep)")
    refs = {}
    for py in TESTS.rglob("*.py"):
        src = py.read_text()
        # Match "scripts/..." or "scripts\..."
        for m in re.finditer(r'"scripts/([\w./-]+)"',
                             src):
            path = m.group(1)
            refs.setdefault(path, []).append(str(py))
        for m in re.finditer(r"'scripts/([\w./-]+)'", src):
            path = m.group(1)
            refs.setdefault(path, []).append(str(py))

    for path in sorted(refs.keys()):
        print(f"  scripts/{path}  ← {refs[path]}")

    return items, service, refs


def build_move_plan(items, refs):
    print("\n" + "=" * 72)
    print("MOVE PLAN")
    print("=" * 72)

    plan = []  # (src_path, dst_path, phase_group)
    skipped = []

    for f in items:
        name = f.name
        if name in KEEP_TOP:
            skipped.append((name, "KEEP_TOP"))
            continue

        # Check test refs
        ref_path = f"scripts/{name}"
        if ref_path in refs:
            skipped.append((name, f"TEST_REF: {refs[ref_path]}"))
            continue

        # Group by prefix
        group = None
        for prefix, d in PHASE_GROUPS.items():
            if name.startswith(prefix):
                group = d
                break
        if not group:
            skipped.append((name, "NO_GROUP_MATCH"))
            continue

        dst = SCRIPTS / group / name
        plan.append((f, dst, group))

    print(f"\nWill move {len(plan)} files, "
          f"skip {len(skipped)}:")
    for name, reason in skipped:
        print(f"  SKIP {name}: {reason}")

    print(f"\nMoves:")
    for f, dst, group in plan:
        print(f"  {f.name} → {group}/{f.name}")

    return plan


def execute_moves(plan):
    print("\n" + "=" * 72)
    print("EXECUTE: git mv")
    print("=" * 72)
    if not plan:
        print("  (no moves)")
        return 0

    # Create phase dirs
    groups = sorted(set(g for _, _, g in plan))
    for g in groups:
        gdir = SCRIPTS / g
        if not gdir.exists():
            gdir.mkdir()
            print(f"  mkdir {gdir}")

    # Add __init__.py to each phase dir (make it package for consistency)
    for g in groups:
        gdir = SCRIPTS / g
        init = gdir / "__init__.py"
        if not init.exists():
            init.write_text("")
            subprocess.run(["git", "add", str(init)], check=True)
            print(f"  {init} (empty __init__)")

    n_moved = 0
    for src, dst, group in plan:
        r = subprocess.run(
            ["git", "mv", str(src), str(dst)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  FAIL {src.name}: {r.stderr[:100]}")
        else:
            n_moved += 1

    print(f"\n  moved {n_moved}/{len(plan)}")
    return n_moved


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit(
            "fast suite FAILED — likely test path reference broke. "
            "Check traceback; use `git reset HEAD` to undo moves."
        )


def main():
    print("=== v0.9.30: B.5 scripts reorg ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"branch: {br}")

    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.29":
        raise SystemExit(f"expected HEAD=v0.9.29, got {head!r}")
    print(f"HEAD: {head}")

    items, service, refs = discover()
    plan = build_move_plan(items, refs)
    n_moved = execute_moves(plan)

    if n_moved == 0:
        print("\nNo moves needed — idempotent (already organized)")
        return

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
        f"v0.9.30: B.5 scripts reorg — group {n_moved} patch_*/diag_* "
        f"files into phase subdirs. "
        f"scripts/v0.9.26/ ← patch_v0926* (B.1 dataframe split, "
        f"~25 files); "
        f"scripts/v0.9.27/ ← patch_v0927* (B.6 cleanup + fix1); "
        f"scripts/v0.9.28/ ← patch_v0928* (B.2 DECISIONS split, "
        f"5 files); "
        f"scripts/v0.9.29/ ← patch_v0929* (B.3 pytest markers, "
        f"3 files); "
        f"scripts/diag/ ← diag_* (blueprint, core_state, "
        f"phase_d_scope). "
        f"KEPT top-level (test-referenced or active): "
        f"__init__.py, _lib.sh, collect_project.py, "
        f"snapshot_ts_review.py, bench_v0919_calibration.py "
        f"(§13.4/359 — test_runtime_monitor.py refs). "
        f"service/ unchanged (test_v0925f_a.py refs). "
        f"`git mv` preserves history (§13.4/360). "
        f"0 production changes; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)
    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"\n{log}")
    print("\nv0.9.30 COMPLETE. B.5 done. Phase B ALL 8/8 CLOSED.")


if __name__ == "__main__":
    main()