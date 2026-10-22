#!/usr/bin/env python3
# v0.9.25b-fix1 — dogfood too strict + current_version not bumped.
#
# Two independent small bugs in v0.9.25b (both in dogfood/report, not
# production code):
#
# Bug 1: dogfood asserted every `done` item must be in phase A. Wrong
#   invariant. Roadmap.json legitimately marks B.7 and B.8 as `done`
#   (closed by this same patch v0.9.25b). Correct invariant: a `done`
#   item may live in any phase; its `patch` must be <= current_version.
#
# Bug 2: roadmap.json `current_version` and ROADMAP.md header still
#   say "0.9.25a" / "v0.9.25a". But v0.9.25b closed B.7/B.8 with
#   patch="v0.9.25b". Bump both to 0.9.25b / v0.9.25b — reflects state
#   after this patch.
#
# v0.9.25b changes (docs/ files, .gitignore, tests/test_roadmap.py)
# were applied but NOT committed (dogfood failed before commit).
# This patch fixes 2 strings, re-runs corrected dogfood, and commits
# v0.9.25b + fix1 together.
#
# Run: uv run python scripts/patch_v0925b_fix1_dogfood_versions.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess,
)


# ============ 0. Verify v0.9.25b state on disk ============

def verify_state():
    print("\n=== verify v0.9.25b changes on disk ===")
    checks = [
        ("docs/ROADMAP.md", "# CADW Roadmap"),
        ("docs/ROADMAP.md", "Current CADW version:** v0.9.25a"),
        ("docs/roadmap.json", '"current_version": "0.9.25a"'),
        ("docs/index.md", "# CADW Documentation"),
        ("tests/test_roadmap.py", "def test_roadmap_json_parses"),
    ]
    for path, anchor in checks:
        if anchor not in Path(path).read_text():
            raise SystemExit(
                f"{path}: missing anchor {anchor!r}. "
                f"Re-run v0.9.25b patch (apply_or_fail may have partially "
                f"failed)."
            )
    print(f"  {len(checks)} anchors present — OK")


# ============ 1. Bump current_version in roadmap.json ============

def bump_json_version():
    print("\n=== bump roadmap.json current_version 0.9.25a -> 0.9.25b ===")
    applied = apply_or_fail(
        "docs/roadmap.json",
        old='"current_version": "0.9.25a"',
        new='"current_version": "0.9.25b"',
        label="json_version",
        validate="json",
    )
    print("  applied" if applied else "  already applied — skip")


# ============ 2. Bump header in ROADMAP.md ============

def bump_md_version():
    print("\n=== bump ROADMAP.md header v0.9.25a -> v0.9.25b ===")
    applied = apply_or_fail(
        "docs/ROADMAP.md",
        old="Current CADW version:** v0.9.25a",
        new="Current CADW version:** v0.9.25b",
        label="md_version",
    )
    print("  applied" if applied else "  already applied — skip")


# ============ 3. Verify docs still parse ============

def verify_docs_parse():
    print("\n=== verify docs parse after edit ===")
    import json as _json
    data = _json.loads(Path("docs/roadmap.json").read_text())
    assert data["current_version"] == "0.9.25b", (
        f"current_version={data['current_version']!r}, expected '0.9.25b'"
    )
    # ROADMAP.md header still has v0.9.25b
    md = Path("docs/ROADMAP.md").read_text()
    assert "Current CADW version:** v0.9.25b" in md
    print("  roadmap.json parses, current_version=0.9.25b; "
          "ROADMAP.md header bumped")


# ============ 4. Corrected dogfood ============

def dogfood():
    print("\n=== dogfood: corrected (done-items may live in any phase) ===")

    def _verify():
        import json as _j
        from pathlib import Path as _P
        data = _j.loads(_P("docs/roadmap.json").read_text())
        assert data["schema_version"] == "1"
        assert data["current_version"] == "0.9.25b", (
            f"current_version={data['current_version']!r}, "
            f"expected '0.9.25b'"
        )

        # Phase coverage: A-H all present.
        phase_ids = {p["id"] for p in data["phases"]}
        assert phase_ids == set("ABCDEFGH"), (
            f"expected A-H, got {sorted(phase_ids)}"
        )

        # Corrected invariant: `done` items may be in ANY phase, but
        # their patch must start with a version (v0.9.25a, v0.9.25b, ...)
        # and that patch's (major, minor, patch-letter) must be <=
        # current_version. We don't parse semver strictly; just check
        # that "done" patches are vX.Y.Z{a..f} with patch-number <=
        # the current "0.9.25b".
        def _vtuple(v: str):
            # "v0.9.25b" -> (0, 9, 25, "b"); "v0.9.25a-fix1" -> (0, 9, 25, "a")
            v = v.lstrip("v")
            base = v.split("-")[0]  # strip -fix1
            # last char may be letter suffix
            tail = ""
            if base and base[-1].isalpha():
                tail = base[-1]
                base = base[:-1]
            parts = base.split(".")
            if len(parts) != 3:
                raise ValueError(f"can't parse {v!r}")
            return (int(parts[0]), int(parts[1]), int(parts[2]), tail)

        current = _vtuple(data["current_version"])
        for phase in data["phases"]:
            for item in phase["items"]:
                if item["status"] == "done":
                    patch = item["patch"]
                    assert patch.startswith("v"), (
                        f"done item {item['id']} patch={patch!r} "
                        f"doesn't start with 'v'"
                    )
                    pt = _vtuple(patch)
                    assert pt <= current, (
                        f"done item {item['id']} patch={patch} > "
                        f"current_version={data['current_version']}. "
                        f"Either bump current_version or fix patch."
                    )

        # Specific expectations after v0.9.25b:
        done_ids = {
            item["id"]
            for phase in data["phases"]
            for item in phase["items"]
            if item["status"] == "done"
        }
        expected_done = {
            "A.1", "A.2", "A.3", "A.3b", "A.8",
            "B.7", "B.8",
        }
        assert done_ids == expected_done, (
            f"done items = {sorted(done_ids)}, expected "
            f"{sorted(expected_done)}"
        )

        # ROADMAP.md has all 8 phase headers.
        md = _P("docs/ROADMAP.md").read_text()
        for name in ("Stabilization", "Infrastructure", "Pandas P0",
                     "Algorithm coverage", "Recipes", "Light NNs",
                     "Tech debt close", "P2"):
            assert name in md, f"ROADMAP.md missing {name!r}"

        # Index links.
        idx = _P("docs/index.md").read_text()
        assert "ROADMAP.md" in idx and "roadmap.json" in idx

        print(
            f"  7 done items across phases A+B, all <= current_version "
            f"0.9.25b; ROADMAP.md/index.md consistent"
        )

    run_verify_in_subprocess(_verify)


# ============ 5. Pytest ============

def run_pytest():
    print("\n=== pre-flight: test_facade_has_no_as_py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py", "-q"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** FACADE AS_PY VIOLATION — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: roadmap tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_roadmap.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** ROADMAP TEST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 6. main ============

def main():
    print("=== v0.9.25b-fix1: dogfood + current_version ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    bump_json_version()
    bump_md_version()
    verify_docs_parse()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25b: docs infrastructure — ROADMAP.md + roadmap.json "
        "+ docs/index.md + .gitignore + tests (Phase B prep). "
        "Docs-only patch; cadw/ untouched. "
        "docs/ROADMAP.md — human-readable, phases A..H with items, "
        "principles, cross-reference task-type -> phase. ~500 lines. "
        "docs/roadmap.json — machine-readable schema_version=1, "
        "8 phases, per-item {id, name, patch, status, complexity}. "
        "Ready for collect_project.py --phase (v0.9.25c). "
        "docs/index.md — new docs entry point; links to roadmap, "
        "recipes/ (scaffold for Phase E), audit docs. Replaces "
        "monolithic DECISIONS.md as primary navigation. "
        "DECISIONS.md untouched — split is Phase B.2 (v0.9.27). "
        ".gitignore: ts_review_snapshot.md, scripts/_tmp/, "
        "_FOR_PINGIS/_scratch/, _verify_*.py. Note: ts_review_snapshot "
        "already committed in v0.9.25a; git rm --cached is Phase B.6. "
        "+tests/test_roadmap.py (6 tests: JSON parses, all phases "
        "present, items have required fields, ROADMAP.md headers, "
        "index links, done items match version). "
        "fix1 (2 bugs in dogfood/report, not production): "
        "(1) dogfood asserted all 'done' items must be in phase A — "
        "wrong; B.7 and B.8 legitimately done by this patch. Correct "
        "invariant: done items may live in any phase, patch must be "
        "<= current_version. "
        "(2) current_version/ROADMAP.md header still said v0.9.25a "
        "while this patch closed v0.9.25b items. Bumped to 0.9.25b "
        "in both docs. "
        "Fast suite: ~2269 -> ~2275.",
        self_path="scripts/patch_v0925b_fix1_dogfood_versions.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25b+fix1 applied. Next: v0.9.25c "
          "(B1 refill + B2 encode_cat + fan-in floor + "
          "collect_project.py phase-aware).")


if __name__ == "__main__":
    main()