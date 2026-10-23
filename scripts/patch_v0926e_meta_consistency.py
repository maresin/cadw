#!/usr/bin/env python3
# v0.9.26e: meta-lesson index (282-289) + B.4 status fix.
# ASCII-only. Discovery-based. Read-then-patch.

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cadw.patch_utils import (
    apply_or_fail, commit_with_guards, check_no_invalid_escape,
)


ROADMAP = "docs/roadmap.json"
DECISIONS = "DECISIONS.md"
CHANGELOG = "CHANGELOG.md"
TESTS_COLLECT = "tests/test_collect_project_modes.py"
TODAY = "2026-10-23"


# ---------------- 1. verify state ----------------

def verify_state():
    head_subj = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head = head_subj.split(":", 1)[0].strip()
    if head != "v0.9.26d":
        raise SystemExit("expected HEAD=v0.9.26d, got " + repr(head))
    print("HEAD: " + head)

    # road map state: A completed, B in_progress, B.4 planned
    data = json.loads(Path(ROADMAP).read_text())
    phb = next(
        (ph for ph in data["phases"]
         if isinstance(ph, dict) and ph.get("id") == "B"),
        None,
    )
    if phb is None or phb.get("status") != "in_progress":
        raise SystemExit("Phase B not in_progress")
    b4 = next(
        (it for it in phb.get("items", [])
         if isinstance(it, dict) and it.get("id") == "B.4"),
        None,
    )
    if b4 is None:
        raise SystemExit("B.4 not found in Phase B items")
    if b4.get("status") != "planned":
        raise SystemExit(
            "B.4 status=" + repr(b4.get("status")) +
            ", expected 'planned'"
        )
    print("B.4 currently 'planned' (needs -> done)")

    # B.4 evidence: test exists and --phase mode works
    tc = Path(TESTS_COLLECT).read_text()
    if "def test_collect_project_phase_mode" not in tc:
        raise SystemExit(
            "test_collect_project_modes.py has no phase mode test - "
            "cannot confirm B.4 done"
        )
    print("evidence: test_collect_project_phase_mode present")

    # DECISIONS current state
    dec = Path(DECISIONS).read_text()
    if "§13.4/290" in dec:
        raise SystemExit("DECISIONS already has §13.4/290 - abort")
    if "§13.4/281" not in dec:
        raise SystemExit("DECISIONS missing §13.4/281 anchor - abort")
    print("DECISIONS has §13.4/281 anchor (index will append after)")


# ---------------- 2. append DECISIONS §13.4/290 ----------------

DECISIONS_BLOCK = """

#### §13.4/290 — Phase A closeout meta-lessons index (docs-debt backfill)

**Context:** Phase A closeout (v0.9.26b-d) surfaced 8 new meta-lessons
§13.4/282-289 in commit messages (some also in CHANGELOG v0.9.26).
This block consolidates them into a single DECISIONS.md entry,
mirroring how §13.4/281 indexed Phase C lessons §13.4/262-280.
Same docs-debt pattern; closes for this batch.

| # | Lesson (1-line) | Commit | Status |
|---|---|---|---|
| 282 | Phase completion checks iterate ALL items, not `patches` list (many-to-one) | v0.9.26b | applied |
| 283 | Roadmap item with specific count + absent audit -> cannot close by inference; defer to later phase | v0.9.26d | applied |
| 284 | Roadmap item names may be mechanistically misleading (`to_pylist` vs `check_batch` kwarg) - verify mechanism, not name | v0.9.26d | applied |
| 285 | Post-mutation validators must match file type: `json.loads` for .json, `ast.parse` only for .py; `check_no_invalid_escape` on Markdown fails on em-dash | v0.9.26d | applied |
| 286 | New state class requires BOTH whitelist extension AND field-consistency assertion (`deferred` -> `deferred_to` present) | v0.9.26d-fix2 | applied |
| 287 | Test the script's own compilation (`ast.parse(open(f).read())`) BEFORE `uv run python` - catches Unicode/quote mangling pre-run | v0.9.26d-fix2-recovery | applied |
| 288 | Discovery before patch: `str.find()` invariant marker, extract actual indent/context, print, then patch. Never hard-code surrounding whitespace | v0.9.26d-fix2-recovery2 | applied |
| 289 | State mutation -> ALL downstream consumers stale, not just the obvious test (test_collect_project_modes vs test_roadmap); invariant lives in consumer | v0.9.26d-fix3 | applied |

**Unified family:** §13.4/278, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289 are one pattern - "assume a specific representation/summary matches semantics." New lessons continue from §13.4/291 onward.

**Docs debt status:** Phase C lessons 262-280 indexed in §13.4/281 (v0.9.26a). Phase A closeout lessons 282-289 indexed here (§13.4/290, v0.9.26e). Baseline is now current through v0.9.26e.
"""


def append_decisions():
    print("\nappending DECISIONS §13.4/290 ...")
    p = Path(DECISIONS)
    src = p.read_text()
    if "§13.4/290" in src:
        print("already present")
        return
    src = src.rstrip() + DECISIONS_BLOCK + "\n"
    p.write_text(src)
    print("appended (" + str(len(DECISIONS_BLOCK.splitlines())) + " lines)")


# ---------------- 3. B.4 status fix ----------------

def fix_b4_status():
    print("\nfixing B.4 status in roadmap.json ...")
    p = Path(ROADMAP)
    data = json.loads(p.read_text())
    phb = next(
        ph for ph in data["phases"]
        if isinstance(ph, dict) and ph.get("id") == "B"
    )
    b4 = next(
        it for it in phb["items"]
        if isinstance(it, dict) and it.get("id") == "B.4"
    )
    if b4.get("status") == "done":
        print("B.4 already done")
        return
    old_status = b4.get("status")
    b4["status"] = "done"
    b4["resolution_note"] = (
        "v0.9.26e: --phase mode shipped in v0.9.25c-fix4; "
        "test_collect_project_modes::test_collect_project_phase_mode "
        "passes as evidence"
    )
    if "patch" in b4 and b4["patch"] == "v0.9.25c":
        b4["patch"] = "v0.9.25c-fix4"
    data["updated"] = TODAY
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    # re-parse verify
    rp = json.loads(p.read_text())
    rpb = next(
        ph for ph in rp["phases"]
        if isinstance(ph, dict) and ph.get("id") == "B"
    )
    rpb4 = next(
        it for it in rpb["items"]
        if isinstance(it, dict) and it.get("id") == "B.4"
    )
    if rpb4.get("status") != "done":
        raise SystemExit("post-write: B.4 not done")
    print("B.4: " + repr(old_status) + " -> done")


# ---------------- 4. CHANGELOG cross-reference note ----------------

CHANGELOG_OLD = (
    "#### \u00a713.4/285 (NEW)\n\n"
    "**`check_no_invalid_escape` is Python-source-only. "
    "Match validator to file type.**"
)

CHANGELOG_NEW_ADD = (
    "\n\n#### Consolidated meta-lessons\n\n"
    "\u00a713.4/282-289 (Phase A closeout lessons) consolidated in "
    "`DECISIONS.md` \u00a713.4/290 - same docs-debt pattern as "
    "\u00a713.4/281 for Phase C."
)


def add_changelog_note():
    print("\nadding CHANGELOG cross-reference ...")
    p = Path(CHANGELOG)
    src = p.read_text()
    if "\u00a713.4/290" in src:
        print("CHANGELOG already has §13.4/290 reference")
        return
    if "\u00a713.4/285" not in src:
        print("WARN: §13.4/285 anchor not found - skipping (non-blocking)")
        return
    # Insert after the §13.4/285 paragraph (find next blank-line run)
    idx = src.find("\u00a713.4/285")
    if idx < 0:
        print("WARN: anchor not found in CHANGELOG")
        return
    # Find end of that subsection - next `####` or end of `###` block
    next_sec = src.find("\n####", idx + 1)
    if next_sec < 0:
        # try next "###" heading
        next_sec = src.find("\n###", idx + 1)
    if next_sec < 0:
        # append at end
        src = src.rstrip() + CHANGELOG_NEW_ADD + "\n"
    else:
        src = src[:next_sec] + CHANGELOG_NEW_ADD + src[next_sec:]
    p.write_text(src)
    print("CHANGELOG updated with §13.4/290 cross-ref")


# ---------------- 5. cosmetic: A.10 planned -> done comment ----------------

def fix_a10_comment():
    print("\ncosmetic: fix stale 'A.10 planned' comment in test ...")
    p = Path(TESTS_COLLECT)
    src = p.read_text()
    old = "    # A.10 planned.\n"
    new = "    # A.10 done (v0.9.26d) - memory_guard.py excluded from phase A snapshot.\n"
    if old not in src:
        print("comment already fixed or absent")
        return
    src = src.replace(old, new, 1)
    p.write_text(src)
    compile(src, TESTS_COLLECT, "exec")
    print("comment updated")


# ---------------- 6. tests ----------------

def run_tests():
    print("\npre-flight: facade AST guards")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "tests/test_dataframe.py::test_facade_has_no_row_loops",
         "-q"],
        timeout=120,
    )
    if r.returncode != 0:
        raise SystemExit("AST guards failed")

    print("\ndocs + collect tests ...")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_roadmap.py", "tests/test_changelog.py",
         "tests/test_collect_project_modes.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        raise SystemExit("docs tests failed")

    print("\nfast suite ...")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite failed")


# ---------------- 7. main ----------------

def main():
    print("=== v0.9.26e: meta index + B.4 status ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()
    append_decisions()
    fix_b4_status()
    add_changelog_note()
    fix_a10_comment()
    run_tests()

    # cleanup any leftover .bak
    bak = Path("docs/roadmap.json.bak")
    if bak.exists():
        bak.unlink()
        print("cleaned " + str(bak))

    commit_with_guards(
        "v0.9.26e: meta consolidation + B.4 status. "
        "DECISIONS §13.4/290 indexes 8 Phase A closeout lessons "
        "(282-289). B.4 (collect_project phase-aware) done - "
        "--phase shipped in v0.9.25c-fix4, test passes. "
        "CHANGELOG cross-refs §13.4/290. Docs/metadata only; "
        "no production code.",
        self_path="scripts/patch_v0926e_meta_consistency.py",
        check_whitelist=True,
    )
    print("\nv0.9.26e complete.")


if __name__ == "__main__":
    main()