#!/usr/bin/env python3
# v0.9.25o — roadmap schema: mark Phase C completed.
#
# docs-only metadata patch. No Python code, no CHANGELOG, no DECISIONS.
# Updates existing keys in docs/roadmap.json (status / version /
# completed_at on Phase C entry; current_version / updated at top level)
# + inserts "COMPLETED" status line under Phase C heading in
# docs/ROADMAP.md. Structure-preserving (only values change, no schema
# changes).
#
# §13.4/280 applied (file-state, not commit-message-inference):
# verify_state walks roadmap.json recursively via
# walk_and_find_phase_c() with 6 heuristics (id/name/phase/phase_id/
# key fields, plus parent-key fallback). Prints full top-level shape +
# phases shape + all Phase C candidates. Aborts on 0 or >1 matches.
# No assumptions about list-vs-dict or field names beyond common
# conventions. If abort → manual inspect + targeted fix1.
#
# §13.4/279 applied (hasattr reveal ≠ rebuild):
# Phase C already status="completed" / version="0.9.25" → idempotent
# skip, no double-write.
#
# Backup: docs/roadmap.json.bak — deleted after successful tests
# (git has the original). On failure, .bak survives for manual
# recovery.
#
# tests/test_roadmap.py + test_changelog.py are the schema validators.
# If v0.9.25o breaks them → the roadmap update touched something the
# test asserts; fix1 updates the test (or reverts the specific change).
#
# §13.4/264 applied: verify_state REPL-checks json.loads round-trip
# on the current file BEFORE mutation.
#
# Run: uv run python scripts/patch_v0925o_roadmap_schema.py

from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


ROADMAP_JSON = "docs/roadmap.json"
ROADMAP_MD = "docs/ROADMAP.md"
TODAY = "2026-10-23"
TARGET_VERSION = "0.9.25"


# Heuristic normalizer for phase-id-style field values
_PHASE_C_ALIASES = {
    "c", "phase_c", "phase c", "c_pandas_p0",
    "c__pandas_p0", "phase_c_pandas_p0", "c_pandas_p0_api",
}


def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def walk_and_find(obj, path="", parent_key=None):
    """Recursively find dicts that look like a Phase C entry.

    Heuristics (any one triggers a match):
      1. One of {id, name, phase, phase_id, key} == phase C aliases.
      2. Parent dict key == phase C aliases (covers
         {"phases": {"C": {...}}} shape).
    """
    matches = []
    if isinstance(obj, dict):
        for key in ("id", "name", "phase", "phase_id", "key"):
            v = obj.get(key)
            if isinstance(v, str) and _norm(v) in _PHASE_C_ALIASES:
                matches.append((path or "<root>", obj,
                                f"field:{key}"))
                break
        else:  # no id-field match — check parent key
            if parent_key and _norm(parent_key) in _PHASE_C_ALIASES:
                matches.append((path or "<root>", obj,
                                f"parent_key:{parent_key}"))
        for k, v in obj.items():
            matches.extend(
                walk_and_find(v, f"{path}.{k}" if path else k, k)
            )
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            p = f"{path}[{i}]" if path else f"[{i}]"
            matches.extend(walk_and_find(item, p, None))
    return matches


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25n; inspect roadmap structure; "
          "find Phase C candidates ===")
    # §13.4/281: commit-subject check = exact version token, NOT
    # substring of log line. CADW commit subjects are single-line and
    # reference future versions (e.g. v0.9.25n's subject mentions
    # "v0.9.25o-roadmap_schema") — substring match is false-positive.
    subj = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head_version = subj.split(":", 1)[0].strip()
    if head_version != "v0.9.25n":
        raise SystemExit(
            f"expected HEAD subject prefix 'v0.9.25n:'; "
            f"got {head_version!r} (subj[:200]={subj[:200]!r})"
        )
    if head_version == "v0.9.25o":
        raise SystemExit("v0.9.25o already committed — misplaced")
    print(f"  HEAD = {head_version} ✓")

    p = Path(ROADMAP_JSON)
    if not p.exists():
        raise SystemExit(f"{ROADMAP_JSON} missing")
    data = json.loads(p.read_text())  # §13.4/264: round-trip check
    print(f"  {ROADMAP_JSON} top keys: {list(data.keys())}")
    print(f"  current_version = {data.get('current_version')!r}")
    print(f"  updated = {data.get('updated')!r}")

    phases = data.get("phases")
    if isinstance(phases, list):
        print(f"  phases = list, len={len(phases)}")
        for i, ph in enumerate(phases):
            if isinstance(ph, dict):
                name = (ph.get("id") or ph.get("name")
                        or ph.get("phase"))
                status = ph.get("status")
                print(f"    [{i}] name={name!r} status={status!r}")
            else:
                print(f"    [{i}] {type(ph).__name__}")
    elif isinstance(phases, dict):
        print(f"  phases = dict, keys={list(phases.keys())}")
        for k, ph in phases.items():
            status = (ph.get("status")
                      if isinstance(ph, dict) else None)
            print(f"    {k!r}: status={status!r}")
    else:
        print(f"  phases = {type(phases).__name__} "
              f"(unexpected)")

    matches = walk_and_find(data)
    print(f"\n  Phase C candidates: {len(matches)}")
    for path, obj, reason in matches:
        # Trim obj to interesting keys for readability
        interesting = {
            k: obj[k] for k in
            ("id", "name", "phase", "status", "version",
             "completed_at", "features")
            if k in obj
        }
        print(f"    path={path!r} via={reason!r}")
        print(f"      → {interesting}")

    if len(matches) == 0:
        raise SystemExit(
            "no Phase C entry found — inspect roadmap manually "
            "(§13.4/280: don't guess)"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"ambiguous: {len(matches)} Phase C candidates. "
            f"Inspect and choose manually."
        )

    path, phase_obj, reason = matches[0]
    print(f"  ✓ unique match: path={path!r} via={reason!r}")

    if phase_obj.get("status") == "completed":
        print("  ⚠ Phase C already status='completed' — "
              "idempotent mode")
    if phase_obj.get("version") == TARGET_VERSION:
        print(f"  ⚠ Phase C already version={TARGET_VERSION} — "
              f"idempotent mode")
    if data.get("current_version") == TARGET_VERSION:
        print(f"  ⚠ top.current_version already {TARGET_VERSION} "
              f"— idempotent mode")

    # ROADMAP.md inspection (defensive — no abort if absent)
    md_p = Path(ROADMAP_MD)
    if md_p.exists():
        md_src = md_p.read_text()
        if re.search(r"^#{1,4}\s+.*Phase\s+C\b",
                     md_src, re.IGNORECASE | re.MULTILINE):
            print(f"  {ROADMAP_MD}: Phase C heading present ✓")
        else:
            print(f"  ⚠ {ROADMAP_MD}: no Phase C heading "
                  f"(prose only or different format)")
    else:
        print(f"  ⚠ {ROADMAP_MD} missing — will skip")

    return {
        "data": data,
        "phase_obj": phase_obj,
        "phase_path": path,
        "match_reason": reason,
    }


# ============ 1. apply to roadmap.json ============

def apply_roadmap_json(state):
    print("\n=== apply roadmap.json Phase C completion ===")
    data = state["data"]
    phase = state["phase_obj"]

    changes = []
    if phase.get("status") != "completed":
        phase["status"] = "completed"
        changes.append("phase.status=completed")
    if phase.get("version") != TARGET_VERSION:
        phase["version"] = TARGET_VERSION
        changes.append(f"phase.version={TARGET_VERSION}")
    if phase.get("completed_at") != TODAY:
        phase["completed_at"] = TODAY
        changes.append(f"phase.completed_at={TODAY}")
    if data.get("current_version") != TARGET_VERSION:
        data["current_version"] = TARGET_VERSION
        changes.append(f"top.current_version={TARGET_VERSION}")
    if data.get("updated") != TODAY:
        data["updated"] = TODAY
        changes.append(f"top.updated={TODAY}")

    if not changes:
        print("  no changes needed — idempotent skip")
        return

    p = Path(ROADMAP_JSON)
    backup = p.with_suffix(".json.bak")
    shutil.copy2(p, backup)
    print(f"  backup → {backup}")

    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    # §13.4/264: verify parse round-trip
    reparsed = json.loads(p.read_text())
    if reparsed.get("current_version") != TARGET_VERSION:
        raise SystemExit(
            "post-write verification failed: "
            "current_version not updated"
        )
    print(f"  applied {len(changes)} changes: {changes}")


# ============ 2. apply to ROADMAP.md ============

def apply_roadmap_md():
    print("\n=== apply ROADMAP.md Phase C completion ===")
    p = Path(ROADMAP_MD)
    if not p.exists():
        print(f"  {ROADMAP_MD} missing — skip")
        return
    src = p.read_text()

    if f"COMPLETED (v{TARGET_VERSION}" in src:
        print("  already marked complete — idempotent skip")
        return

    lines = src.split("\n")
    for i, ln in enumerate(lines):
        m = re.match(
            r"^(#{1,4})\s+.*Phase\s+C\b", ln, re.IGNORECASE,
        )
        if not m:
            continue
        insert_at = i + 1
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
        next_3 = "\n".join(
            lines[insert_at:insert_at + 3]
        ).lower()
        if ("completed" in next_3 or "✅" in next_3
                or "[x]" in next_3):
            print(f"  Phase C at line {i + 1} already marked "
                  f"complete inline — skip")
            return
        new_lines = (
            lines[:insert_at]
            + [""]
            + [f"**Status:** COMPLETED (v{TARGET_VERSION}, "
               f"{TODAY})"]
            + lines[insert_at:]
        )
        p.write_text("\n".join(new_lines))
        print(f"  inserted status after line {i + 1} "
              f"(under heading {ln!r})")
        return

    print(f"  no Phase C heading found — file untouched "
          f"(metadata written by JSON is authoritative)")


# ============ 3. tests ============

def run_tests():
    print("\n=== pre-flight: facade AST guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "tests/test_dataframe.py::test_facade_has_no_row_loops",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FACADE AST GUARD FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: roadmap + changelog docs tests ===")
    targets = []
    if Path("tests/test_roadmap.py").exists():
        targets.append("tests/test_roadmap.py")
    if Path("tests/test_changelog.py").exists():
        targets.append("tests/test_changelog.py")
    if targets:
        r = subprocess.run(
            ["uv", "run", "pytest", *targets,
             "-q", "-W", "error::SyntaxWarning"],
            timeout=300,
        )
        if r.returncode != 0:
            print("\n*** DOCS TEST FAILURE — "
                  "roadmap update broke test assumptions ***")
            print("  If test asserts old current_version "
                  "(0.9.24): v0.9.25o-fix1 updates test to "
                  f"expect {TARGET_VERSION}.")
            print("  If test asserts specific schema shape: "
                  "inspect test + revert if scope creep.")
            sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 4. main ============

def main():
    print("=== v0.9.25o: roadmap Phase C completion ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    state = verify_state()
    apply_roadmap_json(state)
    apply_roadmap_md()
    run_tests()

    # Cleanup: git has the original; .bak is transient
    bak = Path(ROADMAP_JSON + ".bak")
    if bak.exists():
        bak.unlink()
        print(f"  cleaned up {bak} (git holds original)")

    commit_with_guards(
        "v0.9.25o: roadmap schema — mark Phase C completed. "
        f"docs/roadmap.json: Phase C {state['match_reason']} → "
        f"status=completed, version={TARGET_VERSION}, "
        f"completed_at={TODAY}; top-level current_version="
        f"{TARGET_VERSION}, updated={TODAY}. "
        f"docs/ROADMAP.md: Phase C heading gets status line "
        f"(if format matches; prose-only files skipped). "
        "Purely metadata — no code, no schema changes (only "
        "values in existing keys). "
        "§13.4/280 applied (file-state assertions, not "
        "commit-message inference): verify_state recursively "
        "walks roadmap.json via walk_and_find() with 6 heuristics "
        "(id/name/phase/phase_id/key fields + parent-key "
        "fallback for {phases: {C: {...}}} shape). Prints full "
        "shape + all candidates. Aborts on 0 or >1 matches — "
        "no guessing. §13.4/264 applied: json.loads round-trip "
        "pre + post write. §13.4/279 applied: idempotent skip if "
        "Phase C already completed. "
        "Backup strategy: docs/roadmap.json.bak → deleted after "
        "successful pytest (git holds original). If test_roadmap "
        "breaks on version assumption → v0.9.25o-fix1. "
        "Test counts: unchanged (~2488 fast) — metadata-only.",
        self_path="scripts/patch_v0925o_roadmap_schema.py",
        check_whitelist=True,
    )
    print(f"\n==> v0.9.25o COMPLETE. Phase C marked done in roadmap. "
          f"Next: v0.9.26 docs-debt sweep "
          f"(§13.4/249–279 backfill from commit messages).")


if __name__ == "__main__":
    main()