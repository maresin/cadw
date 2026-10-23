#!/usr/bin/env python3
# v0.9.26d — Phase A closeout: A.9 done, A.10 done, A.6 deferred.
#
# Context (diag_v0926c, v0.9.26c-diagnostic):
#   A.6  '18 to_pylist sites extra_factor'  M  v0.9.25d
#       → extra_factor is a check_batch() kwarg, not a to_pylist()
#         kwarg (roadmap name imprecise, §13.4/284). ~30 batches
#         protected across 15 files (documented v0.9.22b: 5 sites;
#         plus v0.9.24 rank/_pass1/_pass2 and ts/features,
#         ts/stats, ts/diff, ts/rolling_stats, ts/rolling_crosscorr,
#         str.cat, dt.tz_convert, etc.). The specific 18 audit-named
#         sites are unidentifiable: _FOR_PINGIS/cadw_audit_report.md
#         is ABSENT from repo state (confirmed by Path check).
#       → §13.4/283: cannot close by inference. Defers to Phase G
#         (Tech debt close), where audit report can be re-requested.
#   A.9  'B5 dead code _append_to_builders'  S  v0.9.25e
#       → NOT FOUND in cadw/. Already removed. DONE.
#   A.10 'B4 memory_guard fallback factor'  S  v0.9.25e
#       → memory_guard.py:757 comment explicitly says
#         'v0.9.25e (A.10): fallback scales with state_limit, not
#         fixed'. Already shipped. DONE.
#
# Corrected close:
#   - A.9 → status='done'
#   - A.10 → status='done'
#   - A.6 → status='deferred', deferred_to='G',
#           resolution_note='audit absent, see §13.4/283'
#   - Phase A → status='completed', completed_at=2026-10-23,
#              caveats=['A.6 deferred to Phase G: audit report
#                       absent from repo state']
#
# CHANGELOG v0.9.26 section added: Phase A close + patches
# v0.9.26a–d + new meta-lessons §13.4/283/284.
#
# §13.4/280 applied: verify_state re-checks A.9 absent + A.10
# comment present via actual file inspection, not roadmap claims.
# §13.4/281 applied: HEAD via exact-token. §13.4/264 applied: json
# round-trip pre+post. §13.4/279 applied: idempotent if A.9/A.10
# already 'done'.
#
# Backup: docs/roadmap.json.bak — deleted after pytest passes.
# If test_roadmap.py asserts A status='in_progress' → v0.9.26d-fix1
# updates test (honest — actual A now completed w/ caveats).
#
# Run: uv run python scripts/patch_v0926d_A_closeout.py

from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, commit_with_guards,
    check_no_invalid_escape,
)


ROADMAP = "docs/roadmap.json"
CHANGELOG = "CHANGELOG.md"
TODAY = "2026-10-23"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.26b; A.9 absent; A.10 comment "
          "present; A.6 audit absent ===")
    subj = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head_version = subj.split(":", 1)[0].strip()
    if head_version != "v0.9.26b":
        raise SystemExit(
            f"expected HEAD subject 'v0.9.26b:'; got "
            f"{head_version!r} (subj[:200]={subj[:200]!r})"
        )
    if head_version == "v0.9.26d":
        raise SystemExit("v0.9.26d already committed — misplaced")
    print(f"  HEAD = {head_version} ✓")

    # --- §13.4/280: A.9 actual code inspection ---
    a9_hits = 0
    for py in Path("cadw").rglob("*.py"):
        if "_append_to_builders" in py.read_text():
            a9_hits += 1
    if a9_hits > 0:
        raise SystemExit(
            f"A.9 NOT done: _append_to_builders found in "
            f"{a9_hits} files — cannot close"
        )
    print(f"  ✓ A.9: _append_to_builders absent from cadw/")

    # --- §13.4/280: A.10 actual code inspection ---
    mg = Path("cadw/core/memory_guard.py")
    if not mg.exists():
        raise SystemExit("cadw/core/memory_guard.py missing")
    mg_src = mg.read_text()
    if "v0.9.25e (A.10)" not in mg_src:
        raise SystemExit(
            "A.10 NOT done: no 'v0.9.25e (A.10)' comment in "
            "memory_guard.py — cannot close"
        )
    # Print the line with the comment for honesty
    for i, ln in enumerate(mg_src.splitlines(), 1):
        if "v0.9.25e (A.10)" in ln:
            print(f"  ✓ A.10: memory_guard.py:{i} {ln.strip()[:110]}")
            break

    # --- §13.4/283: A.6 audit absent ---
    audit_paths = [
        Path("_FOR_PINGIS/cadw_audit_report.md"),
        Path("_FOR_PINGIS/docs/audit_report.md"),
        Path("AUDIT.md"),
    ]
    found_audit = [p for p in audit_paths if p.exists()]
    if found_audit:
        print(f"  ⚠ A.6: audit report present at "
              f"{[str(p) for p in found_audit]} — could attempt "
              f"to close A.6, but this patch defers it to Phase G. "
              f"Inspect manually if you want to close now.")
    else:
        print(f"  ✓ A.6: audit report absent from repo state "
              f"({[str(p) for p in audit_paths]} all missing)")
        print(f"    → §13.4/283: cannot close by inference; "
              f"defers to Phase G.")

    # --- roadmap structure ---
    data = json.loads(Path(ROADMAP).read_text())
    phases = data.get("phases")
    if not isinstance(phases, list):
        raise SystemExit("roadmap phases not list")
    pha = next(
        (ph for ph in phases
         if isinstance(ph, dict) and ph.get("id") == "A"),
        None,
    )
    if pha is None:
        raise SystemExit("Phase A not found")
    items_a = pha.get("items") or []
    statuses = {
        it.get("id"): it.get("status")
        for it in items_a if isinstance(it, dict)
    }
    print(f"\n  Phase A current statuses:")
    for iid in ("A.6", "A.9", "A.10"):
        print(f"    {iid}: {statuses.get(iid)!r}")
    if (statuses.get("A.9") == "done"
            and statuses.get("A.10") == "done"
            and statuses.get("A.6") == "deferred"):
        print("  ⚠ A.9/A.10 done + A.6 deferred — idempotent mode")

    return {"data": data, "phase_a": pha}


# ============ 1. apply to roadmap.json ============

def apply_roadmap(state):
    print("\n=== apply roadmap: A.9 done, A.10 done, A.6 deferred, "
          "Phase A completed ===")
    data = state["data"]
    pha = state["phase_a"]
    items = pha.get("items") or []

    changes = []
    for it in items:
        if not isinstance(it, dict):
            continue
        iid = it.get("id")
        if iid == "A.9" and it.get("status") != "done":
            it["status"] = "done"
            it["resolution_note"] = (
                "v0.9.26c-diagnostic: _append_to_builders absent "
                "from cadw/ — already removed in prior patch"
            )
            changes.append("A.9→done")
        elif iid == "A.10" and it.get("status") != "done":
            it["status"] = "done"
            it["resolution_note"] = (
                "v0.9.26c-diagnostic: memory_guard.py:757 comment "
                "'v0.9.25e (A.10): fallback scales with state_limit, "
                "not fixed' — already shipped in v0.9.25e"
            )
            changes.append("A.10→done")
        elif iid == "A.6" and it.get("status") not in (
            "deferred", "done",
        ):
            it["status"] = "deferred"
            it["deferred_to"] = "G"
            it["resolution_note"] = (
                "§13.4/283: _FOR_PINGIS/cadw_audit_report.md absent "
                "from repo state; specific 18 sites cannot be "
                "identified. extra_factor already applied to ~30 "
                "batches across 15 modules (v0.9.22b documented 5; "
                "v0.9.24/v0.9.25e others). Remaining work deferred "
                "to Phase G (Tech debt close) where audit report "
                "can be re-requested."
            )
            changes.append("A.6→deferred (Phase G)")

    # Phase A → completed w/ caveats
    if pha.get("status") != "completed":
        pha["status"] = "completed"
        changes.append("A.status=completed")
    if not pha.get("completed_at"):
        pha["completed_at"] = TODAY
        changes.append(f"A.completed_at={TODAY}")
    caveats = pha.get("caveats")
    if not isinstance(caveats, list):
        caveats = []
    caveat_str = (
        "A.6 deferred to Phase G: audit report absent from repo "
        "state; specific 18 to_pylist sites cannot be identified "
        "without _FOR_PINGIS/cadw_audit_report.md"
    )
    if caveat_str not in caveats:
        caveats.append(caveat_str)
        pha["caveats"] = caveats
        changes.append("A.caveats appended (A.6 deferred)")

    if data.get("updated") != TODAY:
        data["updated"] = TODAY
        changes.append(f"top.updated={TODAY}")

    if not changes:
        print("  no changes — idempotent")
        return

    p = Path(ROADMAP)
    backup = p.with_suffix(".json.bak")
    shutil.copy2(p, backup)
    print(f"  backup → {backup}")
    p.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )
    # §13.4/264: reparse verify
    reparsed = json.loads(p.read_text())
    reparsed_a = next(
        ph for ph in reparsed["phases"]
        if isinstance(ph, dict) and ph.get("id") == "A"
    )
    if reparsed_a.get("status") != "completed":
        raise SystemExit(
            "post-write: Phase A status not 'completed'"
        )
    print(f"  applied {len(changes)} changes: {changes}")


# ============ 2. apply to ROADMAP.md ============

def apply_roadmap_md():
    print("\n=== apply ROADMAP.md Phase A completion ===")
    p = Path("docs/ROADMAP.md")
    if not p.exists():
        print("  ROADMAP.md missing — skip")
        return
    src = p.read_text()
    if f"COMPLETED (v0.9.26" in src and "Phase A" in src[:5000]:
        # crude: check if A section marked
        if "A.6 deferred to Phase G" in src:
            print("  Phase A already marked with caveat — skip")
            return
    import re
    lines = src.split("\n")
    for i, ln in enumerate(lines):
        if not re.match(r"^#{1,4}\s+.*Phase\s+A\b", ln,
                        re.IGNORECASE):
            continue
        insert_at = i + 1
        while (insert_at < len(lines)
               and not lines[insert_at].strip()):
            insert_at += 1
        new_lines = (
            lines[:insert_at]
            + [""]
            + [f"**Status:** COMPLETED (v0.9.26d, {TODAY}) — "
               f"10/11 items done; A.6 deferred to Phase G "
               f"(audit report absent)."]
            + lines[insert_at:]
        )
        p.write_text("\n".join(new_lines))
        print(f"  inserted status after line {i + 1}")
        return
    print("  no Phase A heading — skip (JSON authoritative)")


# ============ 3. apply to CHANGELOG ============

CHANGELOG_ANCHOR = """## [Unreleased]

### v0.9.25: Phase C — Pandas P0 API"""


CHANGELOG_NEW = """## [Unreleased]

### v0.9.26: Phase A close + Phase B start + roadmap consistency

**Milestone: Phase A (Stabilization) closed.** 10/11 items resolved;
A.6 (`18 to_pylist sites extra_factor`) deferred to Phase G — audit
report `_FOR_PINGIS/cadw_audit_report.md` absent from repo state;
specific 18 sites cannot be identified (§13.4/283). Honest close
over premature.

Phase B (Infrastructure) started — B.1 = `dataframe.py` split into 9
modules (v0.9.26, complexity L) queued after A closeout.

#### Changed (metadata, no production code)

- `docs/roadmap.json`: A.9→`done` (`_append_to_builders` absent from
  `cadw/`, verified v0.9.26c-diagnostic); A.10→`done`
  (`memory_guard.py:757` comment "v0.9.25e (A.10): fallback scales
  with state_limit, not fixed" — already shipped); A.6→`deferred`
  (audit absent, deferred to Phase G, §13.4/283). Phase A→`completed`
  with `caveats=['A.6 deferred to Phase G: audit absent']`. Phase
  B→`in_progress`.
- `docs/ROADMAP.md`: Phase A marked COMPLETED with A.6-deferred
  caveat.

#### Patches

- **v0.9.26a**: docs-debt sweep — §13.4/281 index consolidates Phase
  C lessons §13.4/262–280 with commit refs; Phase C scope verified
  complete (20 Pandas P0 + 6 ML families).
- **v0.9.26b**: roadmap consistency — Phase B→`in_progress`
  (corrected scope after first attempt aborted on A's 3 planned
  items; Phase A left untouched).
- **v0.9.26c**: diagnostic — A.6/A.9/A.10 actual code inspection
  (read-only).
- **v0.9.26d**: A closeout — A.9/A.10 `done`, A.6 `deferred`,
  Phase A `completed` with caveats.

#### §13.4/283 (NEW)

When a roadmap item claims a **specific count** (e.g. "18 sites")
backed by an audit report that is **absent from repo state**, the
item cannot be closed by inference. Options: (a) defer to a phase
where the audit can be re-requested; (b) reconstruct the count from
code — 83 to_pylist sites × receiver-type classification, error-prone.
**Choose (a): defer, document honestly.**

#### §13.4/284 (NEW)

Roadmap item names can be **mechanistically misleading**. A.6 says
"to_pylist sites extra_factor" but `extra_factor` is a `check_batch`
kwarg, not a `to_pylist` kwarg. Verify the mechanism, not just the
name — otherwise diagnostics measure the wrong thing (initial count
of "0 protected" was a false metric — `.to_pylist()` doesn't accept
`extra_factor`, so of course 0 sites had it).

#### Test counts

Unchanged (~2488 fast) — metadata + CHANGELOG only.


### v0.9.25: Phase C — Pandas P0 API"""


def apply_changelog():
    print("\n=== CHANGELOG.md: add v0.9.26 section ===")
    p = Path(CHANGELOG)
    src = p.read_text()
    if "### v0.9.26:" in src:
        print("  already present — idempotent skip")
        return
    applied = apply_or_fail(
        CHANGELOG, CHANGELOG_ANCHOR, CHANGELOG_NEW,
        label="changelog_v0926",
    )
    if applied:
        check_no_invalid_escape(CHANGELOG)
    print("  applied" if applied else "  already applied")


# ============ 4. tests ============

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

    print("\n=== pytest: roadmap + changelog ===")
    targets = ["tests/test_changelog.py"]
    if Path("tests/test_roadmap.py").exists():
        targets.append("tests/test_roadmap.py")
    r = subprocess.run(
        ["uv", "run", "pytest", *targets,
         "-q", "-W", "error::SyntaxWarning"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** DOCS TEST FAILURE — test_roadmap may assert "
              "old A status; v0.9.26d-fix1 will update ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 5. main ============

def main():
    print("=== v0.9.26d: Phase A closeout ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    state = verify_state()
    apply_roadmap(state)
    apply_roadmap_md()
    apply_changelog()
    run_tests()

    bak = Path(ROADMAP + ".bak")
    if bak.exists():
        bak.unlink()
        print(f"  cleaned up {bak}")

    commit_with_guards(
        "v0.9.26d: Phase A closeout — A.9 done, A.10 done, A.6 "
        "deferred to Phase G; Phase A completed with caveats. "
        "Metadata + CHANGELOG only; no production code. "
        "Diagnostic (v0.9.26c) revealed: "
        "(a) A.9 `_append_to_builders` absent from cadw/ — "
        "already removed in prior patch → done. "
        "(b) A.10 already shipped in v0.9.25e (memory_guard.py:757 "
        "comment 'v0.9.25e (A.10): fallback scales with "
        "state_limit, not fixed') → done. "
        "(c) A.6 `18 to_pylist sites extra_factor` — audit "
        "report _FOR_PINGIS/cadw_audit_report.md absent from repo "
        "state; specific 18 sites cannot be identified; extra_factor "
        "already applied to ~30 batches across 15 modules (v0.9.22b "
        "documented 5; v0.9.24/v0.9.25e others). "
        "§13.4/283 (NEW): roadmap item claiming specific count "
        "with absent audit → cannot close by inference; defer to "
        "phase where audit can be re-requested. "
        "§13.4/284 (NEW): roadmap item names mechanistically "
        "misleading — 'to_pylist sites extra_factor' but "
        "extra_factor is check_batch kwarg, not to_pylist kwarg; "
        "verify mechanism not just name. "
        "Changes: "
        "(1) docs/roadmap.json: A.9→done, A.10→done, A.6→deferred "
        "(deferred_to='G'); Phase A→completed, completed_at="
        "2026-10-23, caveats=['A.6 deferred to Phase G: audit "
        "report absent from repo state; specific 18 to_pylist "
        "sites cannot be identified without "
        "_FOR_PINGIS/cadw_audit_report.md']. "
        "(2) docs/ROADMAP.md: Phase A heading marked COMPLETED "
        "with A.6 caveat. "
        "(3) CHANGELOG.md: new v0.9.26 section — Phase A close, "
        "Phase B start, patches v0.9.26a–d, meta-lessons "
        "§13.4/283/284. "
        "§13.4/280 applied: verify_state re-checks A.9 absent + "
        "A.10 comment via actual file inspection; aborts if "
        "unverified. §13.4/281 applied: HEAD exact-token. "
        "§13.4/264 applied: json round-trip pre+post. §13.4/279 "
        "applied: idempotent if A.9/A.10 done + A.6 deferred. "
        "Backup: docs/roadmap.json.bak deleted after pytest. "
        "Test counts: ~2488 fast unchanged. "
        "Next: v0.9.26e-dataframe_blueprint — B.1 planning "
        "(AST analysis, 9-module split map, migration order, "
        "rollback plan; read-only). Then v0.9.26f+ extraction "
        "batches.",
        self_path="scripts/patch_v0926d_A_closeout.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.26d COMPLETE. Phase A=completed "
          "(10/11, A.6→Phase G). Next: v0.9.26e-dataframe_blueprint "
          "(B.1 planning, read-only).")


if __name__ == "__main__":
    main()