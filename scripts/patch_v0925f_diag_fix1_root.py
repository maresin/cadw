#!/usr/bin/env python3
# v0.9.25f-diagnostic-fix1 — root path off-by-one in diagnose_phase_b_split.
#
# BUG: Path(__file__).resolve().parent.parent from
# scripts/service/diagnose_phase_b_split.py = scripts/, not project
# root. Needs THREE parents (file→scripts/service/→scripts/→ROOT).
# Prior scripts/patch_*.py used parent.parent (2 levels, correct for
# scripts/file.py). Diagnostic is one level deeper — pasted 2-parent
# template without recalculating.
#
# FIX: marker-file discovery — walk up until pyproject.toml. Immune to
# script location / depth. No off-by-one ever.
#
# §13.4/256 (NEW): script root path is fragile. Rule:
#   - marker-file discovery (walk up to pyproject.toml / .git / etc.)
#   - NOT parent×N hardcoded to script depth
#   - patch scripts and diagnostic/service scripts diverge in depth
#     over time; marker survives moves.
#
# Read-only diagnostic — not in tests, not in cadw/ production, no
# fix-cycles beyond this one. Re-run after fix; verify header appears.
#
# Run: uv run python scripts/patch_v0925f_diag_fix1_root.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape,
)


DIAG = "scripts/service/diagnose_phase_b_split.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: diagnostic exists with buggy parent.parent ===")
    p = Path(DIAG)
    if not p.exists():
        raise SystemExit(f"{DIAG} not found — was it created?")
    src = p.read_text()
    if "ROOT = Path(__file__).resolve().parent.parent\nCADW = ROOT" not in src:
        if "ROOT = _HERE.parent" in src:
            raise SystemExit(
                "already fixed (marker discovery present) — fix1 misplaced"
            )
        raise SystemExit(
            "expected `ROOT = Path(__file__).resolve().parent.parent` "
            "on its own line — inspect file manually"
        )
    print("  buggy parent.parent present — OK")


# ============ 1. apply fix ============

OLD = '''ROOT = Path(__file__).resolve().parent.parent
CADW = ROOT / "cadw"
DF_PATH = CADW / "dataframe.py"'''

NEW = '''# Robust root discovery — walk up until pyproject.toml.
# Bug (v0.9.25f-diagnostic-fix1): prior `parent.parent` from
# `scripts/service/file.py` = `scripts/` not project root
# (needs 3 parents). §13.4/256: marker-file discovery
# immune to script depth / moves.
_HERE = Path(__file__).resolve()
ROOT = _HERE.parent
for _ in range(8):
    if (ROOT / "pyproject.toml").exists():
        break
    ROOT = ROOT.parent
else:
    print(
        f"ERROR: no pyproject.toml above {_HERE}",
        file=sys.stderr,
    )
    raise SystemExit(2)
CADW = ROOT / "cadw"
DF_PATH = CADW / "dataframe.py"'''


def apply_fix():
    print("\n=== apply marker-based root discovery ===")
    applied = apply_or_fail(DIAG, OLD, NEW, label="root_marker")
    if applied:
        assert_compiles(DIAG)
        check_no_invalid_escape(DIAG)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. dogfood: run diagnostic ============

def dogfood():
    print("\n=== dogfood: run diagnostic (must find dataframe.py) ===")
    r = subprocess.run(
        [sys.executable, DIAG],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr[:800]}")
        raise SystemExit(
            f"diagnostic failed after fix: exit {r.returncode}"
        )
    if "dataframe.py not found" in r.stderr:
        raise SystemExit(
            "diagnostic still not finding dataframe.py — fix didn't work"
        )
    if "=== Phase B: dataframe.py split diagnostic ===" not in r.stdout:
        raise SystemExit(
            "diagnostic output missing header — output format changed"
        )
    print("  diagnostic runs cleanly")

    # Echo first ~80 lines for immediate inspection
    print("\n--- diagnostic stdout (first 80 lines) ---")
    out_lines = r.stdout.split("\n")
    for ln in out_lines[:80]:
        print(f"  {ln}")
    if len(out_lines) > 80:
        print(f"  ... ({len(out_lines) - 80} more lines in stdout)")


# ============ 3. main ============

def main():
    print("=== v0.9.25f-diagnostic-fix1: root path marker-based ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    apply_fix()
    dogfood()

    commit_with_guards(
        "v0.9.25f-diagnostic-fix1: root path off-by-one in "
        "scripts/service/diagnose_phase_b_split.py. "
        "Bug: `Path(__file__).resolve().parent.parent` from "
        "scripts/service/ = scripts/, not project root (needs 3 "
        "parents). Pasted 2-parent template from scripts/patch_*.py "
        "without recalculating depth for scripts/service/. "
        "Fix: marker-file discovery — walk up until pyproject.toml. "
        "Immune to script depth / future moves. "
        "§13.4/256 (NEW): script root path is fragile. Rule: "
        "marker-file discovery (walk up to pyproject.toml / .git / "
        "etc.) — NOT parent×N hardcoded to script depth. Script "
        "location depth diverges (patch/ vs patch/sub/); marker "
        "survives moves. "
        "Read-only diagnostic — not in tests, not in cadw/; "
        "no fix-cycles beyond this one. Diagnostic re-run cleanly, "
        "Phase B risk data available. "
        "Fast suite: ~2306 unchanged (no production/test changes).",
        self_path="scripts/patch_v0925f_diag_fix1_root.py",
        check_whitelist=True,
    )
    print(
        "\n==> v0.9.25f-diagnostic-fix1 applied. "
        "See diagnostic stdout above for Phase B decision data."
    )


if __name__ == "__main__":
    main()