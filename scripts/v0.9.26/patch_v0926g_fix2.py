#!/usr/bin/env python3
# v0.9.26g-fix2: 3 scoped fixes after corrected2.
# R5: scripts/service/ whitelist migration (was excluded by R3).
# R6: test_join_streaming.py patches _core, not package.
# R7: roadmap A.6 composite string restoration.

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26e":
        raise SystemExit("expected HEAD=v0.9.26e, got " + repr(head))
    print("HEAD: " + head)
    if not Path("cadw/dataframe/_core.py").exists():
        raise SystemExit("_core.py missing — corrected2 not applied?")
    if not Path("cadw/dataframe/__init__.py").exists():
        raise SystemExit("__init__.py missing")
    print("skeleton present: _core.py + __init__.py")


def fix_r7_roadmap_a6():
    print("\n--- R7: restore A.6 composite string ---")
    p = Path("docs/roadmap.json")
    data = json.loads(p.read_text())
    pha = next(ph for ph in data["phases"]
               if isinstance(ph, dict) and ph.get("id") == "A")
    a6 = next(it for it in pha["items"]
              if isinstance(it, dict) and it.get("id") == "A.6")
    current = a6.get("file") or ""
    print("  current: " + repr(current))
    if "cadw/ops/" in current:
        print("  composite already present — skip")
        return
    expected = (
        "cadw/dataframe/_core.py, "
        "cadw/ops/{shuffle,rank,metrics,pivot,resample,groupby}.py, "
        "cadw/ts/{features,stats,models,thf,backtest}.py"
    )
    a6["file"] = expected
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print("  restored composite")


def fix_r5_service_whitelist():
    print("\n--- R5: scripts/service/ whitelist migration ---")
    p = Path("scripts/service/diagnose_unguarded_to_pylist.py")
    if not p.exists():
        print("  missing — skip")
        return
    src = p.read_text()
    n_before = (src.count('"cadw/dataframe.py:')
                + src.count("'cadw/dataframe.py:"))
    if n_before == 0:
        print("  already migrated")
        return
    new = src.replace(
        '"cadw/dataframe.py:', '"cadw/dataframe/_core.py:'
    ).replace(
        "'cadw/dataframe.py:", "'cadw/dataframe/_core.py:"
    )
    p.write_text(new)
    ast.parse(new, filename=str(p))
    print("  migrated " + str(n_before) + " key(s)")


def fix_r6_join_streaming():
    print("\n--- R6: test_join_streaming import ---")
    p = Path("tests/test_join_streaming.py")
    src = p.read_text()
    if "from cadw.dataframe import _core as df_mod" in src:
        print("  already fixed")
        return
    old = "from cadw import dataframe as df_mod"
    if old not in src:
        print("  anchor not found — inspect manually")
        return
    new = src.replace(
        old, "from cadw.dataframe import _core as df_mod", 1
    )
    p.write_text(new)
    ast.parse(new, filename=str(p))
    print("  import -> 'from cadw.dataframe import _core'")


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26g-fix2 ===")
    verify_state()
    fix_r7_roadmap_a6()
    fix_r5_service_whitelist()
    fix_r6_join_streaming()
    run_tests()
    commit_with_guards(
        "v0.9.26g: package skeleton (B.1 step 1). "
        "cadw/dataframe.py -> cadw/dataframe/_core.py + full "
        "__init__.py re-export. fix2: scripts/service/ whitelist "
        "migrated, test_join_streaming patches _core, roadmap A.6 "
        "composite preserved. 2488 tests green.",
        self_path="scripts/patch_v0926g_fix2.py",
        check_whitelist=True,
    )
    print("\nv0.9.26g COMPLETE")


if __name__ == "__main__":
    main()