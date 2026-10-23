#!/usr/bin/env python3
# v0.9.26z3-diagnostic: Phase D full scope + dependencies.
# READ-ONLY. No mutation, no commit.

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def dump_phase_d():
    print("=" * 72)
    print("PHASE D: ALGORITHM COVERAGE P1")
    print("=" * 72)
    data = json.loads(Path("docs/roadmap.json").read_text())
    phd = next(
        (ph for ph in data["phases"]
         if isinstance(ph, dict) and ph.get("id") == "D"),
        None,
    )
    if phd is None:
        raise SystemExit("Phase D not found")

    print(f"\ndescription: {phd.get('description')}")
    print(f"version_range: {phd.get('version_range')}")
    print(f"status: {phd.get('status')}")

    items = phd.get("items", [])
    print(f"\nTotal items: {len(items)}")

    # Group by sub-phase (D.X prefix)
    groups = {}
    for it in items:
        iid = it.get("id", "?")
        prefix = iid.split(".")[0] if "." in iid else "?"
        # group by first 2 components (e.g. "D.1")
        parts = iid.split(".")
        key = ".".join(parts[:2]) if len(parts) >= 2 else iid
        groups.setdefault(key, []).append(it)

    for grp in sorted(groups.keys()):
        print(f"\n--- {grp} ({len(groups[grp])} items) ---")
        for it in groups[grp]:
            print(f"  {it.get('id'):8s} "
                  f"{it.get('priority', '?'):3s} "
                  f"{it.get('complexity', '?'):3s} "
                  f"patch={it.get('patch', '?'):10s} "
                  f"{it.get('name', '?')[:60]}")
            if it.get("status") and it["status"] != "planned":
                print(f"           status={it['status']!r}")

    return items, groups


def check_existing_ml():
    print("\n" + "=" * 72)
    print("EXISTING ML IMPLEMENTATIONS (cadw/ml/*)")
    print("=" * 72)
    ml_dir = Path("cadw/ml")
    if not ml_dir.is_dir():
        print("  cadw/ml/ missing")
        return
    for py in sorted(ml_dir.glob("*.py")):
        src = py.read_text()
        try:
            import ast
            tree = ast.parse(src)
        except SyntaxError:
            continue
        classes = [
            n.name for n in tree.body
            if isinstance(n, ast.ClassDef)
        ]
        funcs = [
            n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        print(f"\n  {py.name} ({len(src.splitlines())} lines):")
        if classes:
            print(f"    classes: {classes}")
        if funcs:
            print(f"    funcs: {funcs}")


def check_existing_ops():
    print("\n" + "=" * 72)
    print("EXISTING PRIMITIVES (cadw/ops/*)")
    print("=" * 72)
    ops_dir = Path("cadw/ops")
    for py in sorted(ops_dir.glob("*.py")):
        if py.name == "__init__.py":
            continue
        src = py.read_text()
        try:
            import ast
            tree = ast.parse(src)
        except SyntaxError:
            continue
        public = [
            n.name for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef))
            and not n.name.startswith("_")
        ]
        if public:
            print(f"  {py.name}: {public}")


def check_dependencies():
    print("\n" + "=" * 72)
    print("EXTERNAL DEPENDENCIES (pyproject.toml)")
    print("=" * 72)
    pp = Path("pyproject.toml").read_text()
    m = re.search(
        r'dependencies\s*=\s*\[([^\]]+)\]', pp, re.DOTALL,
    )
    if m:
        for line in m.group(1).split("\n"):
            line = line.strip().strip(",").strip()
            if line and not line.startswith("#"):
                print(f"  {line}")
    opt = re.search(
        r'\[project\.optional-dependencies\]\s*\n([^\[]+)',
        pp, re.DOTALL,
    )
    if opt:
        print(f"\n  optional deps:")
        for line in opt.group(1).split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                print(f"    {line}")


def categorize_by_complexity(groups):
    print("\n" + "=" * 72)
    print("RECOMMENDED BATCH ORDER (P0 first, then S complexity)")
    print("=" * 72)
    all_items = []
    for grp, items in groups.items():
        for it in items:
            all_items.append({
                "id": it.get("id"),
                "name": it.get("name"),
                "priority": it.get("priority", "?"),
                "complexity": it.get("complexity", "?"),
                "group": grp,
                "patch": it.get("patch"),
            })
    # sort: P0 < P1 < P2, then S < M < L
    porder = {"P0": 0, "P1": 1, "P2": 2, "?": 9}
    corder = {"S": 0, "M": 1, "L": 2, "?": 9}
    all_items.sort(
        key=lambda x: (
            porder.get(x["priority"], 9),
            corder.get(x["complexity"], 9),
        )
    )

    print(f"\nRecommended order ({len(all_items)} items):")
    for i, it in enumerate(all_items, 1):
        print(f"  {i:2d}. [{it['priority']}/{it['complexity']}] "
              f"{it['id']:8s} {it['group']:5s} {it['name'][:50]}")

    # Counts
    by_pri = {}
    by_cx = {}
    for it in all_items:
        by_pri[it["priority"]] = by_pri.get(it["priority"], 0) + 1
        by_cx[it["complexity"]] = by_cx.get(it["complexity"], 0) + 1
    print(f"\n  By priority: {by_pri}")
    print(f"  By complexity: {by_cx}")

    # First 5 S-complexity P0
    print("\nFirst-wave candidates (P0/S only — quick wins):")
    first = [it for it in all_items
             if it["priority"] == "P0" and it["complexity"] == "S"]
    if not first:
        # no S P0 — show all P0
        first = [it for it in all_items if it["priority"] == "P0"]
    for it in first[:5]:
        print(f"  {it['id']}: {it['name']}  "
              f"(patch={it['patch']})")


def check_dataframe_api_surface():
    print("\n" + "=" * 72)
    print("CADW DATAFRAME API SURFACE (where new methods go)")
    print("=" * 72)
    print(f"""
After B.1, new fit_* / model methods route to:
  cadw/dataframe/_ml_bindings.py  (mixin — ML fit_*)
  cadw/dataframe/_aggregations.py (mixin — stats/agg)
  cadw/dataframe/_rolling.py      (mixin — TS rolling)
  cadw/dataframe/_cumulative.py   (mixin — cumulative)
  cadw/dataframe/_columns.py      (mixin — select_dtypes)

New primitives (ReduceOp / HashShuffle / Sort composition) go to:
  cadw/ops/<name>.py  (algorithm module)
  cadw/ml/<name>.py   (model, .fit() / .predict())

New method on CadwDataFrame → append to appropriate mixin file
(§13.4/329 append pattern).
""")


def main():
    print("=== Phase D scope diagnostic ===")
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    print(f"HEAD: {head}")
    print(f"current_version: ", end="")
    data = json.loads(Path("docs/roadmap.json").read_text())
    print(f"{data.get('current_version')}")

    items, groups = dump_phase_d()
    check_existing_ml()
    check_existing_ops()
    check_dependencies()
    categorize_by_complexity(groups)
    check_dataframe_api_surface()

    print("\n" + "=" * 72)
    print("NEXT STEPS")
    print("=" * 72)
    print("""
Read output above. Recommendations:

1. Phase D is ENORMOUS (30 items, v0.9.31-v0.9.36).
   Not one patch. Not 2-3 patches. ~30-40 patches.

2. Start with first-wave candidates (P0/S in sorted list).
   These are simplest, deliver highest value, lowest risk.

3. Per item: 1 extraction-style patch
   - Implement in cadw/ml/<name>.py or cadw/ops/<name>.py
   - Add mixin method in cadw/dataframe/_ml_bindings.py etc.
   - Tests in tests/test_<name>.py
   - Fast suite green

4. Do NOT start Phase D.1 blind — decide first:
   - Which D sub-phase (D.1 Anomaly? D.6 features?)
   - Which first item (IForest? perm importance? h3?)
   - Cadence (1 patch/session? batch of 3?)
""")


if __name__ == "__main__":
    main()