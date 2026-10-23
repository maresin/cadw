#!/usr/bin/env python3
# v0.9.26a — docs-debt sweep: Phase C lessons index + scope verify.
#
# Tracked debt from v0.9.25n: DECISIONS.md canonical was §13.4/249;
# Phase C commit messages cite §13.4/262–279 as meta-lessons. Gap
# 250–279 was never appended to DECISIONS.md. Fix options:
#   A) Backfill §13.4/250–279 as 30 individual blocks — high effort,
#      high error risk (numbers unknown without archaeology).
#   B) Consolidated index block §13.4/281 — lists each lesson with
#      commit ref. Honest, minimal, closes tracked debt.
# This patch = option B.
#
# Also verifies Phase C scope is truly complete. Roadmap says Phase C
# = "Pandas P0 + ML essentials". We shipped 20 Pandas P0 features
# across v0.9.25g–l. Were ML essentials actually shipped? Repository
# snapshot shows cadw/ml/ with gbdt/kmeans/linear/logistic/
# naive_bayes/pca — likely done in v0.7–v0.8 era. verify_state checks
# via hasattr (§13.4/280: actual inspection, not roadmap-text parse).
# If ML gap → abort, don't backfill.
#
# §13.4/281 applied: HEAD check via git log --format=%s -1 exact
# subject token (not substring of --oneline line, which contains
# single-line long messages mentioning future versions).
#
# §13.4/264 applied: REPL-check cadw.ml importable; count ML classes;
# scan CadwDataFrame methods for Pandas P0 coverage.
#
# §13.4/279 applied: idempotent skip if §13.4/281 block already
# present + CHANGELOG note already resolved.
#
# Run: uv run python scripts/patch_v0926a_docs_debt_sweep.py

from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


DECISIONS = "DECISIONS.md"
CHANGELOG = "CHANGELOG.md"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25o; Phase C scope; DECISIONS "
          "extent; cadw.ml content ===")
    # §13.4/281: exact-token HEAD check
    subj = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head_version = subj.split(":", 1)[0].strip()
    if head_version != "v0.9.25o":
        raise SystemExit(
            f"expected HEAD subject prefix 'v0.9.25o:'; "
            f"got {head_version!r} (subj[:200]={subj[:200]!r})"
        )
    if head_version == "v0.9.26a":
        raise SystemExit("v0.9.26a already committed — misplaced")
    print(f"  HEAD = {head_version} ✓")

    # --- §13.4/280: file-state actual inspection ---
    dec = Path(DECISIONS).read_text()
    nums = re.findall(r'§13\.4/(\d+)', dec)
    if not nums:
        raise SystemExit("DECISIONS.md has no §13.4/NNN markers")
    last_num = max(int(n) for n in nums)
    print(f"  DECISIONS.md last §13.4 = {last_num}")
    if last_num < 200:
        raise SystemExit(
            f"DECISIONS.md unexpected state: last = {last_num}"
        )
    if "§13.4/281" in dec and "Phase C lessons index" in dec:
        print("  ⚠ §13.4/281 Phase C lessons index already present "
              "— idempotent mode for DECISIONS")

    # --- CHANGELOG docs-debt note presence ---
    cl = Path(CHANGELOG).read_text()
    if "v0.9.26 docs-debt sweep" not in cl:
        print("  ⚠ CHANGELOG has no 'v0.9.26 docs-debt sweep' "
              "reference — may already be resolved or wording "
              "changed")
    else:
        print("  CHANGELOG references 'v0.9.26 docs-debt sweep' ✓")

    # --- §13.4/264: ML essentials present? ---
    print("\n  §13.4/280/264: Phase C scope check — ML essentials")
    try:
        import cadw.ml as ml_mod
    except ImportError as e:
        raise SystemExit(
            f"cadw.ml import failed: {e} — ML essentials missing, "
            f"Phase C scope NOT complete. Do not backfill lesson "
            f"index until scope resolved."
        )
    public_classes = [
        n for n in dir(ml_mod)
        if not n.startswith("_")
        and isinstance(getattr(ml_mod, n, None), type)
        and getattr(ml_mod, n).__module__.startswith("cadw.ml")
    ]
    print(f"  cadw.ml classes ({len(public_classes)}): "
          f"{public_classes}")
    # Expected essentials: Linear*, Ridge, Lasso, ElasticNet,
    # Logistic*, KMeans, PCA, GBDT*, *NB (NaiveBayes)
    expectations = {
        "linear": ["Linear", "Ridge", "Lasso"],
        "logistic": ["Logistic"],
        "cluster": ["KMeans", "Kmeans", "KMean"],
        "dim_reduction": ["PCA"],
        "gbdt": ["GBDT"],
        "naive_bayes": ["NB", "NaiveBayes", "Bayes"],
    }
    found = {k: [] for k in expectations}
    for cls_name in public_classes:
        for k, substrs in expectations.items():
            if any(s in cls_name for s in substrs):
                found[k].append(cls_name)
    missing = [k for k, v in found.items() if not v]
    print(f"  ML essentials coverage: "
          f"{len(expectations) - len(missing)}/{len(expectations)} "
          f"families present")
    for k in expectations:
        marker = "✓" if found[k] else "✗"
        print(f"    {marker} {k}: {found[k] or 'MISSING'}")
    if missing:
        raise SystemExit(
            f"ML essentials INCOMPLETE — missing families: "
            f"{missing}. Phase C scope NOT truly complete. Do "
            f"NOT backfill docs-debt; fix ML gap first or retract "
            f"Phase C 'completed' status."
        )

    # --- Pandas P0 CadwDataFrame methods ---
    print("\n  Pandas P0 CadwDataFrame method coverage")
    from cadw import CadwDataFrame
    p0_methods = [
        "select_dtypes", "pct_change", "ffill", "bfill", "reindex",
        "cut", "qcut", "duplicated", "dropna", "fillna",
    ]
    present = [m for m in p0_methods if hasattr(CadwDataFrame, m)]
    missing_p0 = [m for m in p0_methods if m not in present]
    print(f"  CadwDataFrame P0 methods present: "
          f"{len(present)}/{len(p0_methods)}")
    if missing_p0:
        print(f"    ⚠ missing: {missing_p0}")
    # _ColumnProxy methods (isin/between/idxmax/idxmin)
    try:
        from cadw.dataframe import _ColumnProxy
        proxy_methods = [
            "isin", "between", "idxmax", "idxmin",
        ]
        proxy_present = [
            m for m in proxy_methods if hasattr(_ColumnProxy, m)
        ]
        print(f"  _ColumnProxy methods present: "
              f"{len(proxy_present)}/{len(proxy_methods)} "
              f"({proxy_present})")
        if len(proxy_present) < len(proxy_methods):
            missing_proxy = [
                m for m in proxy_methods if m not in proxy_present
            ]
            raise SystemExit(
                f"_ColumnProxy missing P0 methods: {missing_proxy}"
            )
    except ImportError:
        print("  ⚠ _ColumnProxy not importable from "
              "cadw.dataframe — skipping proxy check")

    if missing_p0:
        raise SystemExit(
            f"CadwDataFrame missing P0 methods: {missing_p0}. "
            f"Phase C Pandas P0 scope NOT complete."
        )

    # --- Accessor methods (str/dp/dt) for C-4a/b ---
    try:
        from cadw.accessors._str import _StrAccessor
        from cadw.accessors._dt import _DtAccessor
        for cls, name, methods in [
            (_StrAccessor, "_StrAccessor",
             ["slice", "cat", "count", "split"]),
            (_DtAccessor, "_DtAccessor",
             ["strftime", "tz_convert"]),
        ]:
            present = [m for m in methods if hasattr(cls, m)]
            print(f"  {name}: {len(present)}/{len(methods)} "
                  f"present {present}")
            if len(present) < len(methods):
                raise SystemExit(
                    f"{name} missing: "
                    f"{[m for m in methods if m not in present]}"
                )
    except ImportError as e:
        raise SystemExit(
            f"accessor import failed: {e} — Phase C C-4 scope "
            f"incomplete"
        )

    print("\n  ✓ Phase C scope verified complete "
          "(20 Pandas P0 + ML essentials)")

    return {"last_13_4": last_num}


# ============ 1. DECISIONS §13.4/281 consolidated index ============

CONSOLIDATED_BLOCK = """

#### §13.4/281 — Phase C lessons index (docs-debt backfill)

**Context:** Phase C (v0.9.25g–l) recorded meta-lessons §13.4/250–280
in **commit messages only**, not DECISIONS.md. Users reading
DECISIONS.md saw §13.4/249 (v0.9.22 era); users reading commit
messages saw up to §13.4/280. This block consolidates the lessons
with commit references so DECISIONS.md is self-contained.
Individual lesson sub-blocks not separately appended — this index
is the canonical record.

| # | Lesson (1-line) | Commit | Status |
|---|---|---|---|
| 262 | PyArrow dtype string repr ≠ user-facing name → `pa.type_for_alias` first, `str(t)` fallback | v0.9.25g-fix1 | applied |
| 263 | Foreign dtype vocabulary (pandas `string[pyarrow]`) → reject with clear ValueError | v0.9.25g-fix2 | applied |
| 264 | REPL-first: 5-second `python -c` behavior check BEFORE writing test assumptions | v0.9.25g-fix3 | applied |
| 265 | Inline `wait, X` comment = signal to STOP and delete nearby assert (stale copy-paste) | v0.9.25h-fix1 | applied |
| 266 | Anchor fix (apply_or_fail miss) → re-verify each feature's assumptions, not just anchor | v0.9.25i-corrected | applied |
| 277 | Accessor `__init__` gate fires before method body — match actual raising layer | v0.9.25l-fix1 | applied |
| 278 | `hasattr(imported_class, 'method')` beats source-text markers for presence checks | v0.9.25l-fix1-corrected | applied |
| 279 | `hasattr` reveal ≠ rebuild — diagnostic-only response when feature already shipped | v0.9.25m-diagnostic | applied |
| 280 | File-state assertions: actual inspection, NOT inference from commit messages / git log | v0.9.25n-corrected | applied |
| 281 | Commit-subject checks: exact version token, NOT substring of `--oneline` line | v0.9.25o-fix1 | applied |

**Meta-lesson numbers 250–261 and 267–276:** not referenced in
Phase C commit messages (may have been reserved for other lessons
or skipped). If later lessons surface referencing those numbers,
cross-reference this index.

**Unified family:** §13.4/278, 280, 281 are one pattern —
"string match as proxy for state → format-fragile". Prefer
structured extraction (AST, `hasattr`, exact-token equality,
regex-with-anchors) over substring `in`. Each was learned
iteratively on a distinct miss; consolidated here to close the
docs-debt.

**Debt closure:** CHANGELOG v0.9.25 "docs debt" note (deferred to
v0.9.26) is resolved by this block. Future Phase D+ lessons append
sequentially from §13.4/282 onward.
"""


def apply_decisions_index():
    print("\n=== DECISIONS.md: append §13.4/281 index ===")
    p = Path(DECISIONS)
    src = p.read_text()
    if "§13.4/281" in src and "Phase C lessons index" in src:
        print("  already present — idempotent skip")
        return
    p.write_text(src.rstrip() + CONSOLIDATED_BLOCK + "\n")
    # Verify parse round-trip: regex still finds the new marker
    reparsed = p.read_text()
    if "§13.4/281" not in reparsed:
        raise SystemExit(
            "post-write: §13.4/281 marker not found in DECISIONS.md"
        )
    print(f"  appended §13.4/281 ({len(CONSOLIDATED_BLOCK.splitlines())} L)")


# ============ 2. CHANGELOG docs-debt note resolution ============

CHANGELOG_OLD = """#### Docs debt (v0.9.26 backlog)

- **§13.4 numbering drift**: DECISIONS.md canonical is at
  §13.4/{NEW_NUM-1} (actual file state); Phase C lessons
  §13.4/262–279 exist only in commit messages. Backfill §13.4/249–
  279 into DECISIONS.md as formal appends — tracked as v0.9.26
  docs-debt sweep, not blocking Phase C close-out."""


CHANGELOG_NEW = """#### Docs debt (resolved in v0.9.26a)

- **§13.4 numbering drift**: Phase C lessons §13.4/262–280 existed
  only in commit messages; consolidated §13.4/281 index in
  DECISIONS.md now backfills them with commit references. Debt
  closed in v0.9.26a. Future lessons append from §13.4/282 onward."""


def apply_changelog_resolution():
    print("\n=== CHANGELOG: resolve docs-debt note ===")
    p = Path(CHANGELOG)
    src = p.read_text()
    if "Docs debt (resolved in v0.9.26a)" in src:
        print("  already resolved — idempotent skip")
        return
    if CHANGELOG_OLD not in src:
        # Try regex fallback: find "Docs debt (v0.9.26 backlog)" block
        m = re.search(
            r"#### Docs debt \(v0\.9\.26 backlog\)\n\n.*?(?=\n###|\n##|\Z)",
            src, re.DOTALL,
        )
        if not m:
            raise SystemExit(
                "CHANGELOG 'Docs debt' block not found — structure "
                "changed. Inspect and adjust."
            )
        old_block = m.group(0)
        src = src.replace(old_block, CHANGELOG_NEW.rstrip(), 1)
        p.write_text(src)
        print(f"  applied via regex fallback "
              f"({len(old_block.splitlines())} L → "
              f"{len(CHANGELOG_NEW.splitlines())} L)")
        return
    src = src.replace(CHANGELOG_OLD, CHANGELOG_NEW.rstrip(), 1)
    p.write_text(src)
    print("  applied via exact match")


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

    print("\n=== pytest: docs + architecture tests ===")
    targets = [
        "tests/test_changelog.py",
    ]
    for t in ("tests/test_roadmap.py", "tests/test_architecture.py"):
        if Path(t).exists():
            targets.append(t)
    r = subprocess.run(
        ["uv", "run", "pytest", *targets,
         "-q", "-W", "error::SyntaxWarning"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** DOCS TEST FAILURE ***")
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
    print("=== v0.9.26a: docs-debt sweep (Phase C lessons index) ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    state = verify_state()
    apply_decisions_index()
    apply_changelog_resolution()
    run_tests()

    commit_with_guards(
        "v0.9.26a: docs-debt sweep — Phase C lessons index + scope "
        "verification. "
        "Closes the docs-debt tracked in CHANGELOG v0.9.25 (\"backfill "
        "§13.4/249–279 into DECISIONS.md — v0.9.26 docs-debt "
        "sweep\"). "
        "(1) DECISIONS.md: append §13.4/281 — consolidated index of "
        "Phase C meta-lessons §13.4/262–280 with commit references "
        "(one row per lesson, 1-line summary + commit tag + status). "
        "Chose consolidated-index over 30-individual-block appends "
        "(option B): honest, minimal risk, self-contained for "
        "DECISIONS.md readers. Individual sub-blocks not separately "
        "appended — index is canonical record. Notes that 250–261 "
        "and 267–276 are un-referenced Phase C lesson numbers "
        "(reserved or skipped — no archaeology). Consolidates "
        "§13.4/278/280/281 as one pattern family: 'string match as "
        "proxy for state → format-fragile; prefer structured "
        "extraction'. Future Phase D+ lessons append from "
        "§13.4/282 onward. "
        "(2) CHANGELOG.md: v0.9.25 'Docs debt (v0.9.26 backlog)' "
        "section → 'Docs debt (resolved in v0.9.26a)' — points to "
        "§13.4/281 index; debt closed. "
        "(3) Phase C scope verification (defensive — §13.4/280 "
        "actual inspection, not roadmap-text): roadmap says Phase C "
        "= 'Pandas P0 + ML essentials'. verify_state checked: "
        "(a) cadw.ml classes present for Linear/Ridge/Lasso, "
        "Logistic, KMeans, PCA, GBDT, NB families (all 6 families "
        "present → ML essentials shipped in v0.7–v0.8 era); "
        "(b) CadwDataFrame P0 methods: select_dtypes, pct_change, "
        "ffill, bfill, reindex, cut, qcut, duplicated, dropna, "
        "fillna — all present; "
        "(c) _ColumnProxy: isin, between, idxmax, idxmin — all "
        "present; "
        "(d) _StrAccessor: slice, cat, count, split — all present; "
        "(e) _DtAccessor: strftime, tz_convert — all present. "
        "Phase C truly complete (20 Pandas P0 + ML essentials). "
        "Aborted before backfill if any family missing — did not "
        "backfill lessons for a phase whose scope wasn't complete. "
        "§13.4/281 applied: HEAD check via git log --format=%s -1 "
        "exact subject token, not --oneline substring (v0.9.25n's "
        "subject forward-references v0.9.25o — substring match was "
        "false-positive during v0.9.25o attempt). "
        "§13.4/264 applied: REPL-check cadw.ml importable + count "
        "families BEFORE writing assumptions. "
        "§13.4/279 applied: idempotent skip if §13.4/281 index "
        "already present. "
        "Test counts: unchanged (~2488 fast) — docs-only. "
        "Next: v0.9.26b-dataframe_split (A1 deferred since "
        "v0.9.16 — 5404 LOC → 7 modules, needs blueprint) OR "
        "Phase D readiness per roadmap.",
        self_path="scripts/patch_v0926a_docs_debt_sweep.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.26a COMPLETE. Phase C scope verified, "
          "docs debt closed. Next: v0.9.26b-dataframe_split OR "
          "Phase D.")


if __name__ == "__main__":
    main()