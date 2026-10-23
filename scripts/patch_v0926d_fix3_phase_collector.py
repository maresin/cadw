#!/usr/bin/env python3
# v0.9.26d-fix3: stale A.9/A.10 assertions in phase collector test.
# ASCII-only. Discovery-based. Resumes v0.9.26d transaction.

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cadw.patch_utils import commit_with_guards


TESTS = "tests/test_collect_project_modes.py"
BAK = "docs/roadmap.json.bak"


def main():
    # --- 1. HEAD check
    head_subj = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head = head_subj.split(":", 1)[0].strip()
    if head != "v0.9.26b":
        raise SystemExit("expected HEAD=v0.9.26b, got " + repr(head))
    print("HEAD: " + head)

    # --- 2. Verify partial state (recovery2 mutations intact)
    import json
    data = json.loads(Path("docs/roadmap.json").read_text())
    pha = next(
        (ph for ph in data["phases"]
         if isinstance(ph, dict) and ph.get("id") == "A"),
        None,
    )
    if not pha or pha.get("status") != "completed":
        raise SystemExit("roadmap Phase A not 'completed'")
    items = {
        it.get("id"): it.get("status")
        for it in pha.get("items", []) if isinstance(it, dict)
    }
    expected = {"A.6": "deferred", "A.9": "done", "A.10": "done"}
    for k, v in expected.items():
        if items.get(k) != v:
            raise SystemExit(
                "roadmap " + k + " expected " + v +
                ", got " + repr(items.get(k))
            )
    tr = Path("tests/test_roadmap.py").read_text()
    if '"deferred"' not in tr or "deferred_to" not in tr:
        raise SystemExit("test_roadmap.py whitelist not extended")
    cl = Path("CHANGELOG.md").read_text()
    if "### v0.9.26:" not in cl:
        raise SystemExit("CHANGELOG v0.9.26 section missing")
    print("partial state verified (roadmap + test_roadmap + CHANGELOG)")

    # --- 3. Read test_collect_project_modes.py, discovery
    path = Path(TESTS)
    src = path.read_text()
    lines = src.split("\n")

    # Find phase-mode test
    tfn_start = None
    for i, ln in enumerate(lines):
        if "def test_collect_project_phase_mode" in ln:
            tfn_start = i
            break
    if tfn_start is None:
        raise SystemExit("test_collect_project_phase_mode not found in " + TESTS)

    print("\nCurrent test function (first 100 lines from def):")
    for i in range(tfn_start, min(tfn_start + 100, len(lines))):
        print(str(i + 1).rjust(4) + "| " + lines[i])

    # --- 4. Discover FILE assertions referencing done items
    done_files = [
        "cadw/ops/sort.py",           # A.9 done
        "cadw/core/memory_guard.py",  # A.10 done
    ]
    modifications = 0
    for target_file in done_files:
        needle = '"FILE: ' + target_file + '"'
        for i, line in enumerate(lines):
            if needle in line and " in text" in line and " not in text" not in line:
                new_line = line.replace(" in text", " not in text", 1)
                print("\nline " + str(i + 1) + " - flipping assertion:")
                print("  OLD: " + line)
                print("  NEW: " + new_line)
                lines[i] = new_line
                modifications += 1

    if modifications == 0:
        raise SystemExit(
            "no assertions to flip - patterns not found. "
            "Check discovery output above."
        )

    # --- 5. Update docstring references
    src_after = "\n".join(lines)
    docstring_repls = [
        (
            "- A.6, A.9, A.10 planned",
            "- A.6 deferred to Phase G (v0.9.26d); A.9/A.10 done (v0.9.26d)",
        ),
        (
            "Assert all 14 files present.",
            "Assert A.6 files present (deferred); A.9/A.10 files absent (done).",
        ),
        (
            "# A.9 planned.",
            "# A.9 done (v0.9.26d) - sort.py excluded from phase A snapshot.",
        ),
    ]
    for old, new in docstring_repls:
        if old in src_after:
            src_after = src_after.replace(old, new, 1)
            print("docstring updated: " + old[:50])

    # --- 6. Write + compile check
    path.write_text(src_after)
    compile(src_after, TESTS, "exec")
    print("\n" + TESTS + " patched and compiles (" +
          str(modifications) + " assertion flips)")

    # --- 7. Docs tests
    print("\nrunning docs tests...")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_roadmap.py", "tests/test_changelog.py",
         "tests/test_collect_project_modes.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        raise SystemExit("docs tests failed")

    # --- 8. Full fast
    print("\nrunning fast suite...")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast tests failed")

    # --- 9. Cleanup .bak
    bak = Path(BAK)
    if bak.exists():
        bak.unlink()
        print("removed " + str(bak))

    # --- 10. Commit (short ASCII)
    commit_with_guards(
        "v0.9.26d: Phase A closeout. A.9/A.10 done; A.6 deferred "
        "to Phase G. Test whitelist and phase collector extended "
        "for deferred/done states. No production code.",
        self_path="scripts/patch_v0926d_fix3_phase_collector.py",
        check_whitelist=True,
    )
    print("\nv0.9.26d committed")


if __name__ == "__main__":
    main()