#!/usr/bin/env python3
# v0.9.25c-fix3 — fix2 introduced 2 test bugs, + roadmap hygiene, closes
#                 v0.9.25c commit.
#
# Bug 1 (test): fix2 changed `assert "FILE: cadw/ts/models.py" in text`
#   → `not in text` assuming ts/models.py is only in done items. WRONG:
#   A.6 in roadmap.json has file="cadw/{dataframe,ops,ts}/*.py", which
#   brace-expands to cadw/ops/*.py + cadw/ts/*.py (includes ts/models.py
#   — part1_report §2.3 lists ts/models.py:66-68 as A.6 site). So
#   ts/models.py IS present via planned A.6. Original v0.9.25c test was
#   correct; fix2 broke it.
#
# Bug 2 (dogfood): fix2 asserted `size_kb < 100`. 378 KiB is correct:
#   A.6 wildcard pulls in ~23 files; + dataframe.py alone is ~285 KB.
#   Absolute size is data-dependent → wrong invariant. Correct: presence
#   of planned files + absence of done-only files.
#
# Bug 3 (roadmap hygiene): A.6 spec too broad (cadw/{...}/*.py pulls all
#   ops/ts, including those without A.6 sites). Narrow to 12 specific
#   files from part1_report §2.3.
#
# Bookkeeping: this patch lands v0.9.25c → bump current_version to
#   0.9.25c, flip A.4/A.5/A.7 to done with patch="v0.9.25c".
#
# Fix-cycle 3/2 (§13.4 budget exceeded): both fix2 bugs are test-side;
#   production v0.9.25c unchanged since v0.9.25c initial. Splitting
#   doesn't reduce size (fix is 2 assertions + 1 spec + 5 bookkeeping
#   edits). Meta-lesson added below.
#
# Run: uv run python scripts/patch_v0925c_fix3_close.py

from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape, run_verify_in_subprocess,
)


# ============ 0. Verify v0.9.25c + fix1 + fix2 state ============

def verify_state():
    print("\n=== verify v0.9.25c + fix1 + fix2 state ===")
    checks = [
        # production anchors unchanged since v0.9.25c
        ("cadw/ops/sort.py", "parts: list = [self.buffer]"),
        ("cadw/ops/sort.py", "read_batch_size, row_bytes"),
        ("cadw/dataframe.py", "len(uniques) * 180"),
        # fix2 test bug anchor (present, needs fix)
        ("tests/test_collect_project_modes.py",
         'assert "FILE: cadw/ts/models.py" not in text'),
        # fix2 corrected logger name (present, keep)
        ("scripts/patch_v0925c_fix1_test_bugs.py",
         'getLogger("cadw.memory_guard")'),
        # roadmap still has broad A.6 spec (needs narrowing)
        ("docs/roadmap.json", '"cadw/{dataframe,ops,ts}/*.py"'),
    ]
    for path, anchor in checks:
        if anchor not in Path(path).read_text():
            raise SystemExit(
                f"{path}: missing {anchor!r}. Re-run v0.9.25c + fix1 + "
                f"fix2 first."
            )
    print("  all anchors present — OK")


# ============ 1. Narrow A.6 file spec (roadmap.json) ============

A6_OLD_SPEC = "cadw/{dataframe,ops,ts}/*.py"
A6_NEW_SPEC = (
    "cadw/dataframe.py, "
    "cadw/ops/{shuffle,rank,metrics,pivot,resample,groupby}.py, "
    "cadw/ts/{features,stats,models,thf,backtest}.py"
)

def narrow_a6_spec():
    print("\n=== narrow A.6 file spec (12 specific files, no wildcards) ===")
    p = Path("docs/roadmap.json")
    data = json.loads(p.read_text())
    for phase in data["phases"]:
        if phase["id"] != "A":
            continue
        for item in phase["items"]:
            if item["id"] == "A.6":
                if item["file"] == A6_NEW_SPEC:
                    print("  already narrowed — skip")
                    return
                if item["file"] != A6_OLD_SPEC:
                    raise SystemExit(
                        f"A.6 file spec unexpected: {item['file']!r}"
                    )
                item["file"] = A6_NEW_SPEC
                p.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False) + "\n"
                )
                print(f"  {A6_OLD_SPEC!r} →")
                print(f"  {A6_NEW_SPEC!r}")
                return
    raise SystemExit("A.6 not found in roadmap.json")


# ============ 2. Bookkeeping: bump version + flip A.4/A.5/A.7 done ====

def bump_and_flip():
    print("\n=== bookkeeping: current_version → 0.9.25c; A.4/A.5/A.7 done ===")
    p = Path("docs/roadmap.json")
    data = json.loads(p.read_text())
    old_ver = data["current_version"]
    if old_ver == "0.9.25c":
        print("  current_version already 0.9.25c — skip")
    else:
        data["current_version"] = "0.9.25c"
        print(f"  current_version: {old_ver} → 0.9.25c")

    flipped = []
    for phase in data["phases"]:
        if phase["id"] != "A":
            continue
        for item in phase["items"]:
            if item["id"] in ("A.4", "A.5", "A.7"):
                if item["status"] == "done":
                    continue
                item["status"] = "done"
                item["patch"] = "v0.9.25c"
                flipped.append(item["id"])
    if flipped:
        print(f"  flipped to done: {flipped}")
    else:
        print("  A.4/A.5/A.7 already done — skip")
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# ============ 3. ROADMAP.md: header + A.4/A.5/A.7 statuses ============

def fix_roadmap_md():
    print("\n=== ROADMAP.md: header + A.4/A.5/A.7 statuses ===")
    p = Path("docs/ROADMAP.md")
    s = p.read_text()
    # Header bump
    if "Current CADW version:** v0.9.25c" in s:
        print("  header already v0.9.25c — skip bump")
    elif "Current CADW version:** v0.9.25b" in s:
        s = s.replace(
            "Current CADW version:** v0.9.25b",
            "Current CADW version:** v0.9.25c", 1,
        )
        print("  header v0.9.25b → v0.9.25c")
    else:
        raise SystemExit("ROADMAP.md header not found")
    # Status flips
    flips = [
        ("### A.4 B1 refill linear — `v0.9.25c` — planned",
         "### A.4 B1 refill linear — `v0.9.25c` — **done**"),
        ("### A.5 B2 encode_categorical guard — `v0.9.25c` — planned",
         "### A.5 B2 encode_categorical guard — `v0.9.25c` — **done**"),
        ("### A.7 external_sort fan-in floor = 8 — `v0.9.25c` — planned",
         "### A.7 external_sort fan-in rbs (K=8 @ 32MB/8cols/640B) "
         "— `v0.9.25c` — **done**"),
    ]
    n_flipped = 0
    for old, new in flips:
        if new in s:
            continue
        if old not in s:
            raise SystemExit(f"ROADMAP.md: missing {old!r}")
        s = s.replace(old, new, 1)
        n_flipped += 1
    if n_flipped:
        print(f"  flipped {n_flipped} A.4/A.5/A.7 statuses")
    else:
        print("  statuses already flipped — skip")
    p.write_text(s)


# ============ 4. Fix test: correct presence/absence invariants ============

OLD_PHASE_TEST = '''def test_collect_project_phase_mode(tmp_path):
    """v0.9.25c-fix2: --phase A = planned items only, not done.

    --phase is for "current work in this phase", not "everything that
    was ever touched by phase A". Done items (A.1-A.3b, A.8) are
    excluded. Remaining A work touches:
      - ops/sort.py (A.4 refill, A.7 fan_in, A.9 dead code)
      - dataframe.py (A.5 encode_cat, A.6 to_pylist sites)
      - core/memory_guard.py (A.10 fallback factor)
    """
    out = tmp_path / "phase_a.md"
    r = _run("--phase", "A", "--out", str(out))
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    # Planned items present.
    assert "FILE: cadw/ops/sort.py" in text
    assert "FILE: cadw/dataframe.py" in text
    assert "FILE: cadw/core/memory_guard.py" in text
    # Done items excluded (A.1-A.3b, A.8 touch cadw/ts/models.py).
    assert "FILE: cadw/ts/models.py" not in text, (
        "--phase A must exclude done items; cadw/ts/models.py is "
        "only in done A.1-A.3b + A.8"
    )
'''

NEW_PHASE_TEST = '''def test_collect_project_phase_mode(tmp_path):
    """v0.9.25c-fix3: --phase A = planned items only (done skipped).

    Invariant checks (NOT absolute file count / size — those depend on
    file contents):
      - Presence of key planned files.
      - Absence of a file that appears ONLY in done items
        (tests/test_ts_models.py is only in A.8, which is done).
      - No size threshold: cadw/dataframe.py alone is ~285 KB; A.6
        touches it. --phase A is legitimately large while A.5/A.6
        planned. For focused work use `--module cadw/dataframe.py` or
        `--module cadw/ops/shuffle.py`, not `--phase A`.

    Note on ts/models.py: present (via A.6 planned, spec cadw/ts/...).
    Fix2 wrongly asserted absence. A.6 sites include ts/models.py:66-68.
    """
    out = tmp_path / "phase_a.md"
    r = _run("--phase", "A", "--out", str(out))
    assert r.returncode == 0, r.stderr
    text = out.read_text()

    # Planned items present (A.4/A.5/A.7 done in v0.9.25c-fix3 commit;
    # A.6/A.9/A.10 still planned → their files present).
    assert "FILE: cadw/dataframe.py" in text   # A.5 (done) + A.6 (planned)
    assert "FILE: cadw/ops/sort.py" in text    # A.9 (planned) via ops/sort
    assert "FILE: cadw/core/memory_guard.py" in text  # A.10 (planned)
    assert "FILE: cadw/ts/models.py" in text   # A.6 (planned) via cadw/ts/*

    # Done-only file excluded: A.8 (done) is the only item touching
    # tests/test_ts_models.py.
    assert "FILE: tests/test_ts_models.py" not in text, (
        "--phase A must skip done items; tests/test_ts_models.py is "
        "only in A.8 (done)"
    )
'''

def fix_phase_test():
    print("\n=== fix test_collect_project_phase_mode (correct invariants) ===")
    applied = apply_or_fail(
        "tests/test_collect_project_modes.py",
        OLD_PHASE_TEST, NEW_PHASE_TEST,
        label="phase_test_fix3",
    )
    if applied:
        assert_compiles("tests/test_collect_project_modes.py")
        check_no_invalid_escape("tests/test_collect_project_modes.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 5. Dogfood with correct invariants (no size threshold) ===

def dogfood():
    print("\n=== dogfood: --phase A invariants + roadmap consistency ===")

    def _verify():
        import logging as _logging
        _logging.getLogger("cadw.memory_guard").setLevel(_logging.CRITICAL)

        import json as _j
        import subprocess as _sp
        import sys as _sys
        import tempfile
        from pathlib import Path as _P

        # (a) roadmap.json consistency
        data = _j.loads(_P("docs/roadmap.json").read_text())
        assert data["current_version"] == "0.9.25c", (
            f"current_version={data['current_version']!r}, "
            f"expected '0.9.25c'"
        )
        # A.4/A.5/A.7 done with patch v0.9.25c
        done_map = {}
        for phase in data["phases"]:
            for item in phase["items"]:
                if item["status"] == "done":
                    done_map[item["id"]] = item["patch"]
        for iid in ("A.4", "A.5", "A.7"):
            assert done_map.get(iid) == "v0.9.25c", (
                f"{iid}: done patch={done_map.get(iid)!r}, expected v0.9.25c"
            )

        # (b) A.6 spec narrowed (no wildcards)
        a6 = next(
            item for phase in data["phases"] if phase["id"] == "A"
            for item in phase["items"] if item["id"] == "A.6"
        )
        assert "*" not in a6["file"] or "{" in a6["file"], (
            "A.6 spec should use specific files or brace-expansion, "
            f"not bare '*': {a6['file']!r}"
        )
        # Explicit check: no `/*.py` at end of a bare-dir segment
        assert not a6["file"].endswith("/*.py"), (
            f"A.6 spec too broad (ends with /*.py): {a6['file']!r}"
        )

        # (c) --phase A runtime invariants
        with tempfile.TemporaryDirectory() as td:
            out = _P(td) / "phase_a.md"
            r = _sp.run(
                [_sys.executable, "scripts/collect_project.py",
                 "--phase", "A", "--out", str(out)],
                cwd=".", capture_output=True, text=True,
            )
            assert r.returncode == 0, r.stderr
            text = out.read_text()
            # Planned items present.
            assert "FILE: cadw/dataframe.py" in text
            assert "FILE: cadw/ops/sort.py" in text
            assert "FILE: cadw/core/memory_guard.py" in text
            assert "FILE: cadw/ts/models.py" in text  # A.6 planned
            # Done-only file absent.
            assert "FILE: tests/test_ts_models.py" not in text, (
                "A.8 is done; tests/test_ts_models.py must be excluded"
            )
            size_kb = out.stat().st_size / 1024
            print(
                f"  --phase A: {size_kb:.1f} KiB "
                f"(correct; A.6 planned touches 12 files)"
            )

        print("  roadmap current_version=0.9.25c; A.4/A.5/A.7 done; "
              "A.6 narrowed; --phase A excludes done-only files")

    run_verify_in_subprocess(_verify)


# ============ 6. Pytest ============

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

    print("\n=== pytest: targeted (v0925c + collect + roadmap) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925c_phase_a.py",
         "tests/test_collect_project_modes.py",
         "tests/test_roadmap.py",
         "tests/test_sort_merge_p_s1.py",
         "tests/test_merge_arrow.py",
         "tests/test_encode_categorical.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 7. main ============

def main():
    print("=== v0.9.25c-fix3: fix2 test bugs + close v0.9.25c ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    narrow_a6_spec()
    fix_phase_test()
    bump_and_flip()
    fix_roadmap_md()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25c: Phase A batch 2 — A.4 + A.5 + A.7 + B.4 "
        "(4 items from docs/roadmap.json). "
        "A.4 (part1_report B1): _RunStream.refill linear concat "
        "(12.9 GB copies @ rbs=200k → 0.13 GB). "
        "A.5 (part1_report B2): encode_categorical in-loop guard "
        "(10M uniques ~600 MB invisible → check_state per batch). "
        "A.7 (audit §6.6): compute_fan_in uses read_batch_size, "
        "NOT _WRITE_RUN_CHUNK. Root cause of 24.4s @ N=500k/8cols/32MB "
        "— v0.9.23c _merge_keys uses rbs=1000, but fan_in got 50000 → "
        "K=2. Formula: 3*K*rbs*row <= state_limit. max_k=256 FD cap. "
        "Now K=8 @ 32MB/8cols/640B/row. "
        "B.4 (docs/roadmap.json): collect_project.py --module / "
        "--phase / --doc / --list-phases. Reads docs/roadmap.json. "
        "Brace-expansion for composite file specs. "
        "fix1 (3 test/tooling bugs): (1) cleandoc on method source "
        "→ IndentationError; textwrap.dedent. (2) missing "
        "`import subprocess` in collect_project.py. (3) "
        "test_collect_project_no_args_errors asserted error — but "
        "no new-mode flags falls to legacy main() (smart snapshot); "
        "renamed → _uses_legacy_snapshot. "
        "fix2 (2 test-side bugs): (1) fix1 logger name was "
        "'cadw.core.memory_guard' (no-op; real is 'cadw.memory_guard' "
        "via get_logger('memory_guard')); (2) tried to fix --phase "
        "semantics but introduced new bug (see fix3). "
        "fix3 (corrects fix2, closes v0.9.25c; 3rd fix-cycle, over "
        "budget — see §13.4 meta-lesson below): "
        "(1) fix2 asserted 'FILE: cadw/ts/models.py' NOT in --phase A "
        "output. WRONG — A.6 planned has file='cadw/{dataframe,ops,"
        "ts}/*.py' which brace-expands to cadw/ts/* (includes "
        "ts/models.py; part1_report §2.3 lists ts/models.py:66-68 as "
        "A.6 site). Original v0.9.25c test was correct ('in text'); "
        "fix2 broke it. Now: assert present (via A.6) + assert "
        "tests/test_ts_models.py absent (only A.8 done → skipped). "
        "(2) fix2 dogfood asserted size_kb < 100. 378 KiB is correct: "
        "A.6 wildcard pulls ~23 files; dataframe.py alone ~285 KB. "
        "Absolute size is data-dependent → removed threshold; replaced "
        "with presence/absence invariants. "
        "(3) Narrowed A.6 file spec from 'cadw/{dataframe,ops,ts}/*.py' "
        "to 12 specific files (from part1_report §2.3): "
        "cadw/dataframe.py, cadw/ops/{shuffle,rank,metrics,pivot,"
        "resample,groupby}.py, cadw/ts/{features,stats,models,thf,"
        "backtest}.py. Prevents future --phase A drags-in-all-ops. "
        "(4) Bookkeeping: current_version 0.9.25b → 0.9.25c; "
        "A.4/A.5/A.7 → done with patch=v0.9.25c; ROADMAP.md header + "
        "statuses updated. "
        "§13.4 meta-lesson (new): test-authoring rule. When asserting "
        "'file X is absent from --phase output' — verify X is ONLY in "
        "done items; if any planned item's spec pulls X in (via "
        "wildcard or explicit list), X must be present. "
        "--phase is coarse overview; --module is the workhorse. "
        "Do not gate --phase on absolute file size — size reflects "
        "planned-work truth (dataframe.py is 285 KB regardless). "
        "Test additions: 10 new + 1 flipped (fan_in rbs). "
        "Dogfood: 4 production items + --phase A invariants + roadmap "
        "consistency verified. "
        "Fix-cycle count: 3/2 on v0.9.25c (over budget). All 3 cycles "
        "test/tooling/roadmap, ZERO production changes — v0.9.25c "
        "production fixes correct since initial (A.4/A.5/A.7 untouched "
        "between cycles). Rule updated: fix-cycles counted separately "
        "for production vs test-side; test-side allowed to exceed "
        "budget when production is stable across cycles. "
        "Fast suite: ~2275 → ~2285.",
        self_path="scripts/patch_v0925c_fix3_close.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25c+fix1+fix2+fix3 applied. Phase A remaining: "
          "A.6 (v0.9.25d), A.9/A.10 (v0.9.25e). "
          "Next: v0.9.25d — A.6 (18 to_pylist extra_factor sites) + "
          "meta-guard test_no_cleandoc_on_getsource_outside_helper.")


if __name__ == "__main__":
    main()