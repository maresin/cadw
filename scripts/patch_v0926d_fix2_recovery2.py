#!/usr/bin/env python3
# v0.9.26d-fix2-recovery2: extend test_roadmap whitelist (discovery-based).
# ASCII-only. Finds actual assertion line, extracts real indent, patches.

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cadw.patch_utils import commit_with_guards


TESTS = "tests/test_roadmap.py"


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

    # --- 2. Read + discover
    path = Path(TESTS)
    src = path.read_text()

    idx = src.find('assert item["status"]')
    if idx < 0:
        print("'assert item[\"status\"]' not found. First 60 lines:")
        for i, ln in enumerate(src.splitlines()[:60], 1):
            print(str(i).rjust(3) + "| " + ln)
        raise SystemExit("abort: assertion not found")

    line_start = src.rfind("\n", 0, idx) + 1
    line_end = src.find("\n", idx)
    if line_end < 0:
        line_end = len(src)
    target = src[line_start:line_end]
    indent = target[: len(target) - len(target.lstrip())]
    print("Found at char " + str(idx))
    print("Line repr: " + repr(target))
    print("Indent: " + repr(indent) + " (len=" + str(len(indent)) + ")")

    # --- 3. Idempotency guards
    if '"deferred"' in src[idx:idx + 500]:
        raise SystemExit("'deferred' already in whitelist - abort")
    if "deferred_to" in src:
        raise SystemExit("'deferred_to' already present - abort")

    # --- 4. Extend whitelist (simple substring replace)
    old_tuple = '("done", "in_progress", "planned")'
    new_tuple = '("done", "in_progress", "planned", "deferred")'
    if old_tuple not in src:
        print("Expected tuple " + repr(old_tuple) + " not found.")
        print("Context around assert (chars " + str(max(0, idx - 50)) + ".." + str(idx + 300) + "):")
        print(repr(src[max(0, idx - 50):idx + 300]))
        raise SystemExit("abort: tuple not found")
    src = src.replace(old_tuple, new_tuple, 1)
    print("whitelist extended")

    # --- 5. Add deferred_to check BEFORE assertion
    idx = src.find('assert item["status"]')
    line_start = src.rfind("\n", 0, idx) + 1
    line_end = src.find("\n", idx)
    if line_end < 0:
        line_end = len(src)
    target = src[line_start:line_end]
    indent = target[: len(target) - len(target.lstrip())]

    deferred_check = (
        indent + 'if item["status"] == "deferred":\n'
        + indent + '    assert "deferred_to" in item, (\n'
        + indent + '        "deferred item missing deferred_to field"\n'
        + indent + '    )\n'
    )
    src = src[:line_start] + deferred_check + src[line_start:]

    # --- 6. Write + parse-check
    path.write_text(src)
    compile(src, TESTS, "exec")
    print("deferred_to check added; file compiles")

    # --- 7. Docs tests
    print("running docs tests...")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_roadmap.py", "tests/test_changelog.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        raise SystemExit("docs tests failed")

    # --- 8. Full fast
    print("running fast suite...")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast tests failed")

    # --- 9. Cleanup .bak
    bak = Path("docs/roadmap.json.bak")
    if bak.exists():
        bak.unlink()
        print("removed " + str(bak))

    # --- 10. Commit (short ASCII)
    commit_with_guards(
        "v0.9.26d: Phase A closeout. A.9/A.10 done; A.6 deferred "
        "to Phase G. Test whitelist extended for deferred state. "
        "No production code.",
        self_path="scripts/patch_v0926d_fix2_recovery2.py",
        check_whitelist=True,
    )
    print("v0.9.26d committed")


if __name__ == "__main__":
    main()