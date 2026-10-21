#!/usr/bin/env python3
# v0.9.22-fix1 — remove FOR_PINGIS leak + CHANGELOG catch-up.
# v0.9.22 COMPLETE.
#
# v0.9.22 committed but 2 problems:
#   (1) FOR_PINGIS/TASK_ML.md (no underscore, user's attachment
#       folder) accidentally committed. commit_with_guards runs
#       `git add -A`; only `_FOR_PINGIS/` (underscore) is
#       .gitignore'd. Audit_ML_v0922 §6 flagged this as risk,
#       commit still added it.
#   (2) CHANGELOG not updated: v0.9.15–v0.9.22 (8 patches) never
#       documented. test_changelog_has_latest_tag skipped on
#       non-master branch → silent gap.
#
# §13.4/246 (new): blanket `git add -A` needs directory audit
# before commit. Any user-content folder (`FOR_PINGIS/`,
# `temp/`, `notes/`, `docs/` unless gitignored) will be
# swept in. Rule: .gitignore every user-content dir BEFORE
# first patch on that branch, OR use `git add` with explicit
# paths in commit helper.
#
# Fix1:
#   (a) .gitignore: add `FOR_PINGIS/` (no underscore) — distinct
#       from `_FOR_PINGIS/` (audit output, already gitignored).
#   (b) `git rm --cached FOR_PINGIS/TASK_ML.md` — drop from
#       index, preserve on disk.
#   (c) CHANGELOG: add `### v0.9.15–v0.9.22` under `[Unreleased]`
#       (8-patch summary; individual detail via git log --grep).
#   (d) verify: 8 patch anchors present (v0.9.15–v0.9.22 fixes
#       still on disk); FOR_PINGIS not tracked; CHANGELOG has
#       v0.9.15–v0.9.22 marker.
#   (e) commit as v0.9.22-fix1 on current branch (audit branch
#       carries both report + fixes; user merges to master).
#
# Deferred (user decisions post-fix1):
#   - merge audit branch → master (--no-ff, preserves "v0.9.22
#     from AUDIT_ML round 3" context) OR cherry-pick to master.
#   - tag v0.9.22 + bump __version__ "0.9.14" → "0.9.22"? Or
#     accumulate under [Unreleased] until v0.10.0.
#
# Run: uv run python scripts/patch_v0922_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) .gitignore FOR_PINGIS/ ============

def fix_gitignore():
    print("\n=== (a) .gitignore: FOR_PINGIS/ (no underscore) ===")
    p = Path(".gitignore")
    s = p.read_text() if p.exists() else ""
    if "FOR_PINGIS/" in s.split("\n"):
        print("  already present — skip")
        return
    sep = "" if s.endswith("\n") else "\n"
    p.write_text(
        s + sep
        + "\n# v0.9.22-fix1: user attachment folder (no underscore),\n"
        + "# distinct from _FOR_PINGIS/ (audit output, gitignored).\n"
        + "FOR_PINGIS/\n"
    )
    print("  OK")


# ============ (b) git rm --cached FOR_PINGIS/TASK_ML.md ============

def remove_for_pingis_index():
    print("\n=== (b) git rm --cached FOR_PINGIS/TASK_ML.md ===")
    r = subprocess.run(
        ["git", "ls-files", "FOR_PINGIS/TASK_ML.md"],
        capture_output=True, text=True, check=False,
    )
    if not r.stdout.strip():
        print("  not tracked — skip")
        return
    subprocess.run(
        ["git", "rm", "--cached", "FOR_PINGIS/TASK_ML.md"],
        check=True,
    )
    print("  removed from index (file preserved on disk)")


# ============ (c) CHANGELOG [Unreleased] catch-up ============

CHANGELOG_ENTRY = """### v0.9.15–v0.9.22: post-release v0.9.14 fixes + audit round 3

**8 patches** since v0.9.14 tag. Full per-patch detail via
`git log --oneline v0.9.14..master`. Highlights:

- **v0.9.15** — `patch_utils.dedent_method_source` + T2.7 meta-guard
  (no inline `dedent(getsource(...))` outside helper).
- **v0.9.16** — DECISIONS §13.4/211–225 append (15 meta-lessons
  from v0.9.7–v0.9.14); tag v0.9.14.
- **v0.9.17** — DECISIONS §13.4/178–189 append (round-1 audit
  lessons); seaborn deprecation filter; category-beyond-memory
  tests (K × per_key > avail, §13.4/218 null-key regression).
- **v0.9.18** — AUDIT.md round-1 fixes: §1.1 `_ArrowTail` (shift),
  §1.2 rank numpy fast path, §1.3 string hash dict-encode, §1.5
  `to_list` sample-based guard, §1.6 `MaterializedIndex` numpy.
  3 fix-cycles for null/edge cases.
- **v0.9.19** — closed-loop memory correctness:
  `MemoryGuard(calibrate=True, runtime_monitor=True)` — 3 tiers
  (pyarrow pool exact / psutil RSS sampled / calibration log
  warns at C > 2.0); `reduce()` wired; `bench_v0919_calibration.py`.
- **v0.9.20** — peak-polling `calibrate(sample_peak=True)`;
  bench decomposition columns (`pred_batch` / `pred_state` /
  `pred_total` / `C_peak` / `C_batch`); DECISIONS §13.4/226–232
  append. **5 fix-cycles** — §13.4/238 anti-pattern (4-feature
  patch → cascade).
- **v0.9.21** — AUDIT §1.4 (last CONFIRMED item):
  `_group_boundaries_arrow` + per-group shift/rolling Arrow-native
  group boundaries; §13.4/233–240 append. 1 fix-cycle (per-group
  as_py → batch-level slice.to_pylist).
- **v0.9.22** — AUDIT_ML round 3 fixes: gbdt multiclass
  `est_n_state = n·8·(1+4K)` (was n·K·8·3 — missed y_all + K-th
  round_cur_leaves, 880 MB under-report @ N=10M K=10); gbdt
  `_validate_binary_y` single-pass; NB `C_ASSUMED=64` (was 32);
  kmeans `batch_memory_factor=2k` (S1: 800 MB real vs 4 MB check
  @ k=100 closed); `_flush_buf` bounded-by-construction comment;
  DECISIONS §9.2.20–22 + §13.4/241–245 append.

**Key outcomes:**
- AUDIT.md round-1 **all 5 CONFIRMED closed** (§1.1–§1.6).
- AUDIT_ML_v0922 round-3: 24 suspected → **7 CONFIRMED** (all
  fixed v0.9.22) → 17 **RETRACTED** under measurement. Two S1
  bugs closed: gbdt multiclass formula (880 MB miss), kmeans
  batch_memory_factor (200× under-report at k=100).
- **Closed-loop memory** (v0.9.19+): `MemoryGuard` can measure
  runtime — pyarrow pool exact, RSS sampled, calibration log
  warns at drift > 2.0. `bench_v0919_calibration.py` automates
  AUDIT workflow.
- **25 meta-lessons** documented (§13.4/178–245).

**Deferred:**
- **v0.9.22b** — `extra_factor=3.0-5.0` at 5 `to_pylist` call
  sites (rank:269,345; groupby:581; resample:272; pivot:204).
  Needs exact anchors per site.
- **v0.9.23** — kmeans++ reservoir (quality: +36.9% inertia on
  skewed data); `RobustScaler` O(d) sort batching.
- **v0.9.24+** — A1 `dataframe.py` split; Pingis round 4;
  v1.0 readiness.

**Test count:** 1916 fast + 24 slow (was 1876+24 at v0.9.14;
+40 fast, 0 slow). Whitelist 31 keys.

"""


def update_changelog():
    print("\n=== (c) CHANGELOG [Unreleased] v0.9.15–v0.9.22 ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    marker = "### v0.9.15–v0.9.22: post-release v0.9.14 fixes"
    if marker in s:
        print("  already present — skip")
        return
    # Find [Unreleased] block start (immediately after header)
    import re
    m = re.search(r'^## \[Unreleased\]\s*\n', s, re.MULTILINE)
    if not m:
        raise SystemExit("[Unreleased] header not found")
    insert_at = m.end()
    s = s[:insert_at] + "\n" + CHANGELOG_ENTRY + "\n" + s[insert_at:]
    p.write_text(s, newline="")
    # Sanity: descending intact
    versions = re.findall(
        r"^## \[(\d[^\]]*)\]", s, flags=re.MULTILINE,
    )
    assert versions, "no [X.Y.Z] entries"
    assert versions[0] == "0.9.14", f"top = {versions[0]!r}, expected 0.9.14"
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.22-fix1 + regression ===")
    # Standard anchors
    pu.verify_anchors([
        # fix1
        (".gitignore", "FOR_PINGIS/"),
        ("CHANGELOG.md", "### v0.9.15–v0.9.22: post-release v0.9.14"),
        # v0.9.22 (5 code + 3 doc)
        ("cadw/ml/gbdt.py", "n_total * 8 * (1 + 4 * K)"),
        ("cadw/ml/gbdt.py", "v0.9.22 (§13.4/246, AUDIT_ML §1.2)"),
        ("cadw/ml/naive_bayes.py", "return 64 * (8 + 16 * self.d)"),
        ("cadw/ml/kmeans.py", "return 2.0 * float(self.k)"),
        ("DECISIONS.md", "### 9.2.20. GBDT top-level peak"),
        ("DECISIONS.md", "**Эпизод 245 (audit-first"),
        # v0.9.21
        ("cadw/dataframe.py", "def _group_boundaries_arrow("),
        ("DECISIONS.md", "**Эпизод 233 (append with fallback anchor"),
        # v0.9.20
        ("cadw/core/memory_guard.py", "sample_peak: bool = False"),
        ("cadw/core/memory_guard.py", "# v0.9.20-fix3"),
        # v0.9.19
        ("cadw/core/memory_guard.py", "def calibrate("),
        ("cadw/ops/reduce.py", "with memory_guard.calibrate("),
        # v0.9.18
        ("cadw/ops/shuffle.py", "§13.4/227"),
        ("cadw/dataset/index.py", "v0.9.18 (AUDIT §1.6)"),
        # v0.9.17
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный"),
        ("tests/conftest.py", "vert: bool was deprecated in Matplotlib 3.11"),
        # v0.9.16
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        # v0.9.15
        ("cadw/patch_utils.py", "v0.9.15 (T2.6)"),
        # v0.9.14
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
    ])

    # Fix1-specific: FOR_PINGIS not tracked
    r = subprocess.run(
        ["git", "ls-files", "FOR_PINGIS/"],
        capture_output=True, text=True, check=False,
    )
    assert not r.stdout.strip(), (
        f"FOR_PINGIS still tracked: {r.stdout!r}"
    )
    # CHANGELOG still descending
    import re
    text = Path("CHANGELOG.md").read_text()
    versions = re.findall(r"^## \[(\d[^\]]*)\]", text, flags=re.MULTILINE)
    keys = []
    for v in versions:
        m = re.match(r"^(\d+)\.(\d+)\.(\d+)([a-z]?)$", v)
        assert m, f"unrecognized version {v!r}"
        keys.append((int(m.group(1)), int(m.group(2)),
                     int(m.group(3)), m.group(4)))
    assert keys == sorted(keys, reverse=True), (
        f"CHANGELOG not descending: {keys}"
    )
    print("  all anchors present; FOR_PINGIS not tracked; CHANGELOG descending")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.22-fix1 (happy path) ===")

    def _verify_all():
        import subprocess as _sp
        from pathlib import Path as _P

        # FOR_PINGIS not tracked
        r = _sp.run(
            ["git", "ls-files", "FOR_PINGIS/"],
            capture_output=True, text=True,
        )
        assert not r.stdout.strip(), f"FOR_PINGIS tracked: {r.stdout!r}"

        # FOR_PINGIS on disk still present (user's attachment)
        assert _P("FOR_PINGIS/TASK_ML.md").exists(), (
            "FOR_PINGIS/TASK_ML.md should remain on disk"
        )

        # CHANGELOG has fix1 marker
        cl = _P("CHANGELOG.md").read_text()
        assert "### v0.9.15–v0.9.22: post-release v0.9.14" in cl

        # .gitignore has FOR_PINGIS/
        gi = _P(".gitignore").read_text()
        assert "FOR_PINGIS/" in gi.split("\n")

        print("  FOR_PINGIS not tracked, on disk preserved OK")
        print("  CHANGELOG v0.9.15–v0.9.22 present OK")
        print("  .gitignore FOR_PINGIS/ OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: changelog ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_changelog.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** CHANGELOG FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.22-fix1 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_gitignore()
    remove_for_pingis_index()
    update_changelog()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.22-fix1: complete v0.9.22 — remove FOR_PINGIS leak + "
        "CHANGELOG catch-up. v0.9.22 committed but 2 problems: "
        "(1) FOR_PINGIS/TASK_ML.md (no underscore, user's attachment "
        "folder) accidentally committed. commit_with_guards runs "
        "`git add -A`; only `_FOR_PINGIS/` (underscore) is "
        ".gitignore'd. AUDIT_ML_v0922 §6 flagged this as risk, "
        "commit still added it. (2) CHANGELOG not updated: v0.9.15–"
        "v0.9.22 (8 patches) never documented. "
        "test_changelog_has_latest_tag skipped on non-master branch "
        "→ silent gap. §13.4/246 (new): blanket `git add -A` needs "
        "directory audit before commit. Any user-content folder "
        "(`FOR_PINGIS/`, `temp/`, `notes/`, `docs/` unless "
        "gitignored) will be swept in. Rule: .gitignore every "
        "user-content dir BEFORE first patch on that branch, OR use "
        "`git add` with explicit paths in commit helper. "
        "(Fix1 changes) (a) .gitignore: add `FOR_PINGIS/` (no "
        "underscore), distinct from `_FOR_PINGIS/` (audit output, "
        "already gitignored); (b) `git rm --cached "
        "FOR_PINGIS/TASK_ML.md` — drop from index, preserve on "
        "disk; (c) CHANGELOG: add `### v0.9.15–v0.9.22: post-"
        "release v0.9.14 fixes + audit round 3` under "
        "`[Unreleased]` — 8-patch summary (v0.9.15 patch_utils "
        "dedent_method_source; v0.9.16 DECISIONS §13.4/211–225 + "
        "tag v0.9.14; v0.9.17 §13.4/178–189 + seaborn filter + "
        "category-beyond-memory tests; v0.9.18 AUDIT round 1 "
        "§1.1–§1.6 fixes; v0.9.19 closed-loop memory "
        "calibrate+runtime_monitor; v0.9.20 peak-polling + bench "
        "decomposition + 5-fix-cycle; v0.9.21 AUDIT §1.4 "
        "Arrow-native group boundaries; v0.9.22 AUDIT_ML round 3 "
        "fixes gbdt multiclass formula + kmeans factor + NB "
        "C_ASSUMED); (d) verify 20+ anchors — all 8 patches still "
        "on disk; FOR_PINGIS not tracked; CHANGELOG descending "
        "intact; (e) commit as v0.9.22-fix1 on current branch "
        "(audit branch carries both report + fixes). "
        "(Deferred to user post-fix1) merge audit branch → master "
        "(`--no-ff` for context) or cherry-pick; tag v0.9.22 + "
        "bump __version__ or accumulate under [Unreleased] until "
        "v0.10.0; v0.9.22b extra_factor at 5 sites; v0.9.23 "
        "kmeans++ reservoir + RobustScaler batching; v0.9.24+ A1 "
        "split. (Backward compat) 1916 existing tests unchanged; "
        "no code edits, only .gitignore + CHANGELOG + index cleanup.",
        self_path="scripts/patch_v0922_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.22-fix1 applied. v0.9.22 COMPLETE (clean).")


if __name__ == "__main__":
    main()
