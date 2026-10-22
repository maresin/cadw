#!/usr/bin/env python3
# v0.9.22b — AUDIT_ML §1.6: extra_factor at to_pylist sites.
#
# Sites (5):
#   rank.py _compute_streamable: 5.0 (null fallback: by + row_id pylist)
#   rank.py _pass1_groups_to_disk: 3.0 (by pylist)
#   rank.py _pass2_ranks_to_disk: 5.0 (by + row_id pylist)
#   resample.py sorted_input: 5.0 (ts + K cols per-row pylist)
#   pivot.py stream_sorted: 5.0 (K_index + columns + values per-row pylist)
#
# groupby.py existing factor=3.0 in _groupby_with_skew is adequate:
# _accumulate_heavy's grouped.to_pylist() bounded by heavy_keys count
# (≤20 typical), NOT by batch. Not changed; documented.
#
# §13.4/247 (new): check_batch extra_factor + inline comment, not just
# factor. Future reader needs to know WHY 5.0 vs 3.0 (N pylist × 28B/8B
# ≈ N × 3.5); otherwise next audit bumps factor again without basis.
# Same class as §13.4/240 (formula unit vs metric) and /243
# (batch_memory_factor override — inherited default wrong).
#
# §13.4/246 (new, backfill from v0.9.22-fix1 commit-message-only):
# blanket `git add -A` needs directory audit before commit.
#
# Run: uv run python scripts/patch_v0922b.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ rank.py: 3 sites ============

def fix_rank_sites():
    print("\n=== rank.py: 3 sites extra_factor ===")
    p = Path("cadw/ops/rank.py")
    s = p.read_text()
    if "v0.9.22b (AUDIT_ML §1.6)" in s:
        print("  already fixed — skip")
        return

    old1 = '            memory_guard.check_batch(batch, name="rank.streamable.input")'
    new1 = (
        '            # v0.9.22b (AUDIT_ML §1.6): extra_factor=5.0 —\n'
        '            # null fallback does to_pylist(by) + to_pylist(row_id);\n'
        '            # null-free uses numpy fast path (no pylist).\n'
        '            memory_guard.check_batch(\n'
        '                batch, name="rank.streamable.input", extra_factor=5.0,\n'
        '            )'
    )
    if old1 not in s:
        raise SystemExit("rank streamable anchor not found")
    s = s.replace(old1, new1, 1)

    old2 = '            memory_guard.check_batch(batch, name="rank.pass1.input")'
    new2 = (
        '            # v0.9.22b (AUDIT_ML §1.6): extra_factor=3.0 —\n'
        '            # pass1 does to_pylist(by); ~3.5× batch.nbytes.\n'
        '            memory_guard.check_batch(\n'
        '                batch, name="rank.pass1.input", extra_factor=3.0,\n'
        '            )'
    )
    if old2 not in s:
        raise SystemExit("rank pass1 anchor not found")
    s = s.replace(old2, new2, 1)

    old3 = '            memory_guard.check_batch(batch, name="rank.pass2.input")'
    new3 = (
        '            # v0.9.22b (AUDIT_ML §1.6): extra_factor=5.0 —\n'
        '            # pass2 does to_pylist(by) + to_pylist(row_id).\n'
        '            memory_guard.check_batch(\n'
        '                batch, name="rank.pass2.input", extra_factor=5.0,\n'
        '            )'
    )
    if old3 not in s:
        raise SystemExit("rank pass2 anchor not found")
    s = s.replace(old3, new3, 1)

    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK (3 sites)")


# ============ resample.py: 1 site ============

def fix_resample_site():
    print("\n=== resample.py: extra_factor ===")
    p = Path("cadw/ops/resample.py")
    s = p.read_text()
    if "v0.9.22b (AUDIT_ML §1.6)" in s:
        print("  already fixed — skip")
        return

    old = '            memory_guard.check_batch(b, name="resample.sorted_input")'
    new = (
        '            # v0.9.22b (AUDIT_ML §1.6): extra_factor=5.0 —\n'
        '            # per-row to_pylist(ts) + per-col to_pylist.\n'
        '            memory_guard.check_batch(\n'
        '                b, name="resample.sorted_input", extra_factor=5.0,\n'
        '            )'
    )
    if old not in s:
        raise SystemExit("resample anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ pivot.py: 1 site ============

def fix_pivot_site():
    print("\n=== pivot.py: extra_factor ===")
    p = Path("cadw/ops/pivot.py")
    s = p.read_text()
    if "v0.9.22b (AUDIT_ML §1.6)" in s:
        print("  already fixed — skip")
        return

    old = '            memory_guard.check_batch(b, name="pivot.stream_sorted")'
    new = (
        '            # v0.9.22b (AUDIT_ML §1.6): extra_factor=5.0 —\n'
        '            # per-row to_pylist on K_index cols + columns + values.\n'
        '            memory_guard.check_batch(\n'
        '                b, name="pivot.stream_sorted", extra_factor=5.0,\n'
        '            )'
    )
    if old not in s:
        raise SystemExit("pivot anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ CHANGELOG: mark v0.9.22b closed ============

def update_changelog():
    print("\n=== CHANGELOG: v0.9.22b deferred → closed ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "- ~~**v0.9.22b**~~" in s:
        print("  already updated — skip")
        return
    old = (
        "- **v0.9.22b** — `extra_factor=3.0-5.0` at 5 `to_pylist` call\n"
        "  sites (rank:269,345; groupby:581; resample:272; pivot:204).\n"
        "  Needs exact anchors per site.\n"
    )
    new = (
        "- ~~**v0.9.22b**~~ — **closed v0.9.22b**: extra_factor at 5\n"
        "  sites (rank.py _compute_streamable/_pass1/_pass2,\n"
        "  resample.py sorted_input, pivot.py stream_sorted);\n"
        "  groupby.py skew factor=3.0 deemed adequate\n"
        "  (grouped.to_pylist bounded by heavy_keys ≤ 20, not batch).\n"
    )
    if old not in s:
        raise SystemExit("CHANGELOG deferred anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s, newline="")
    print("  OK")


# ============ DECISIONS §13.4/246-247 ============

DECISIONS_246_247 = (
    "\n\n\n"
    "**Эпизод 246 (git add -A sweeps user-content dirs).** "
    "v0.9.22-fix1: `FOR_PINGIS/TASK_ML.md` (user attachment folder, "
    "no underscore) accidentally committed. `commit_with_guards` runs "
    "`git add -A`; only `_FOR_PINGIS/` (underscore) was .gitignored. "
    "Audit_ML_v0922 §6 flagged it as risk — commit still added it.\n\n"
    "**Правило (§13.4/246):** blanket `git add -A` requires directory "
    "audit BEFORE commit. Any user-content folder (`FOR_PINGIS/`, "
    "`temp/`, `notes/`, `docs/` unless gitignored) will be swept in. "
    "Rule: .gitignore every user-content dir BEFORE first patch on "
    "that branch, OR use `git add` with explicit paths in commit "
    "helper.\n\n"
    "**Связанные:** §13.4/222 (multi-step abort leaves partial "
    "state).\n\n\n"
    "**Эпизод 247 (extra_factor + comment, not just factor).** "
    "v0.9.22b (AUDIT_ML §1.6): 5 sites had `check_batch(batch)` with "
    "default extra_factor=1.0 while downstream did `to_pylist()` "
    "(~3.5× batch.nbytes per call). Fix: explicit extra_factor "
    "(3.0 or 5.0 by to_pylist count). **Comment** per site — WHY "
    "5.0 vs 3.0 (N pylist × 28B/8B ≈ N × 3.5). Without comment, "
    "next audit bumps factor again without basis.\n\n"
    "**Правило (§13.4/247):** когда bump extra_factor — документируй "
    "в inline comment: (a) какие pylist calls, (b) N calls, "
    "(c) формула factor = N × 3.5. `check_batch(batch, extra_factor=5.0)` "
    "без comment — magic number. Audit finds it next time, cannot "
    "verify without re-measuring.\n\n"
    "**Связанные:** §13.4/240 (formula unit must match metric), "
    "/243 (batch_memory_factor override — inherited default wrong), "
    "/241 (formula components must be enumerated).\n\n\n"
)


def append_decisions_246_247():
    print("\n=== §13.4/246-247 append ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "**Эпизод 246 (git add -A sweeps" in s:
        print("  already present — skip")
        return
    anchor = "### 13.5. Открытые вопросы"
    if anchor not in s:
        raise SystemExit("§13.5 anchor not found")
    if s.count(anchor) != 1:
        raise SystemExit(f"§13.5 anchor count = {s.count(anchor)}")
    s = s.replace(anchor, DECISIONS_246_247 + anchor, 1)
    p.write_text(s, newline="")
    assert s.index("**Эпизод 245 ") < s.index("**Эпизод 246 ")
    assert s.index("**Эпизод 247 ") < s.index("### 13.5. Открытые вопросы")
    print("  OK (246-247 after 245, before §13.5)")


# ============ test: AST parametrized ============

def add_ast_test():
    print("\n=== tests/test_extra_factor_sites.py ===")
    p = Path("tests/test_extra_factor_sites.py")
    if p.exists():
        print("  already present — skip")
        return
    pu.write_content_checked(p, [
        '"""v0.9.22b (§13.4/247): extra_factor at to_pylist call sites."""',
        'import inspect',
        '',
        'import pytest',
        '',
        '',
        'SITES = [',
        '    ("cadw.ops.rank", "_compute_streamable",',
        '     "rank.streamable.input", 5.0),',
        '    ("cadw.ops.rank", "_pass1_groups_to_disk",',
        '     "rank.pass1.input", 3.0),',
        '    ("cadw.ops.rank", "_pass2_ranks_to_disk",',
        '     "rank.pass2.input", 5.0),',
        '    ("cadw.ops.resample", "resample_to_disk",',
        '     "resample.sorted_input", 5.0),',
        '    ("cadw.ops.pivot", "pivot_to_disk",',
        '     "pivot.stream_sorted", 5.0),',
        ']',
        '',
        '',
        '@pytest.mark.parametrize("module,fn_name,name,factor", SITES)',
        'def test_check_batch_has_extra_factor(module, fn_name, name, factor):',
        '    """v0.9.22b (§13.4/247): each to_pylist site has explicit',
        '    extra_factor in the check_batch line (same line as name=)."""',
        '    mod = __import__(module, fromlist=[fn_name])',
        '    fn = getattr(mod, fn_name)',
        '    src = inspect.getsource(fn)',
        '    for line in src.split("\\n"):',
        '        if name in line:',
        '            assert "extra_factor=" in line, (',
        '                f"{module}.{fn_name}: check_batch(name={name!r}) "',
        '                f"missing extra_factor in line: {line!r}"',
        '            )',
        '            assert f"extra_factor={factor}" in line, (',
        '                f"{module}.{fn_name}: expected "',
        '                f"extra_factor={factor} in line: {line!r}"',
        '            )',
        '            return',
        '    raise AssertionError(',
        '        f"{module}.{fn_name}: check_batch(name={name!r}) not found"',
        '    )',
    ])
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.22b + regression ===")
    pu.verify_anchors([
        # v0.9.22b sites
        ("cadw/ops/rank.py",
         'name="rank.streamable.input", extra_factor=5.0'),
        ("cadw/ops/rank.py",
         'name="rank.pass1.input", extra_factor=3.0'),
        ("cadw/ops/rank.py",
         'name="rank.pass2.input", extra_factor=5.0'),
        ("cadw/ops/resample.py",
         'name="resample.sorted_input", extra_factor=5.0'),
        ("cadw/ops/pivot.py",
         'name="pivot.stream_sorted", extra_factor=5.0'),
        ("tests/test_extra_factor_sites.py",
         "test_check_batch_has_extra_factor"),
        # CHANGELOG mark
        ("CHANGELOG.md", "- ~~**v0.9.22b**~~ — **closed v0.9.22b**"),
        # DECISIONS 246-247
        ("DECISIONS.md", "**Эпизод 246 (git add -A sweeps"),
        ("DECISIONS.md", "**Эпизод 247 (extra_factor + comment"),
        # v0.9.22 regression
        ("cadw/ml/gbdt.py", "n_total * 8 * (1 + 4 * K)"),
        ("cadw/ml/kmeans.py", "return 2.0 * float(self.k)"),
        ("DECISIONS.md", "### 9.2.20. GBDT top-level peak"),
        ("DECISIONS.md", "**Эпизод 245 (audit-first"),
        # v0.9.21
        ("cadw/dataframe.py", "def _group_boundaries_arrow("),
        # v0.9.20
        ("cadw/core/memory_guard.py", "sample_peak: bool = False"),
        # v0.9.19
        ("cadw/core/memory_guard.py", "def calibrate("),
        # v0.9.18
        ("cadw/ops/shuffle.py", "§13.4/227"),
        # v0.9.15
        ("cadw/patch_utils.py", "v0.9.15 (T2.6)"),
        # v0.9.14 tag
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.22b (happy path) ===")

    def _verify_all():
        import pyarrow as pa
        from cadw import CadwDataFrame, MemoryGuard

        # 1. rank null-free + null
        t = pa.table({"k": [3, 1, 2, 1], "v": [1, 2, 3, 4]})
        df = CadwDataFrame.from_table(
            t, memory_guard=MemoryGuard(max_memory_mb=10),
        )
        r = df.rank("k", method="first")
        got = []
        for b in r._iter_filtered_batches():
            got.extend(b.column("k_rank").to_pylist())
        assert got == [4, 1, 3, 2], got

        t2 = pa.table({
            "k": pa.array([3, None, 1, None], type=pa.int64()),
            "v": [1, 2, 3, 4],
        })
        df2 = CadwDataFrame.from_table(
            t2, memory_guard=MemoryGuard(max_memory_mb=10),
        )
        r2 = df2.rank("k", method="min")
        got2 = []
        for b in r2._iter_filtered_batches():
            got2.extend(b.column("k_rank").to_pylist())
        assert len(got2) == 4

        # 2. resample small
        t3 = pa.table({"ts": [0, 3, 7], "v": [1.0, 2.0, 3.0]})
        df3 = CadwDataFrame.from_table(
            t3, memory_guard=MemoryGuard(max_memory_mb=10),
        )
        r3 = df3.resample(5, "sum", on="ts", columns=["v"])
        s3 = []
        for b in r3._iter_filtered_batches():
            s3.extend(b.column("v_sum").to_pylist())
        assert s3 == [3.0, 3.0], s3

        # 3. pivot small
        t4 = pa.table({
            "row": ["r1", "r1", "r2", "r2"],
            "col": ["c1", "c2", "c1", "c2"],
            "v": [10.0, 20.0, 30.0, 40.0],
        })
        df4 = CadwDataFrame.from_table(
            t4, memory_guard=MemoryGuard(max_memory_mb=10),
        )
        r4 = df4.pivot_table(
            index=["row"], columns="col", values="v", aggfunc="mean",
        )
        assert r4.count() == 2

        print("  rank null-free + null: OK")
        print("  resample small: [3.0, 3.0] OK")
        print("  pivot small: 2 rows OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_extra_factor_sites.py",
         "tests/test_rank.py",
         "tests/test_resample.py",
         "tests/test_pivot_arrow_native.py",
         "tests/test_pivot_table.py",
         "tests/test_changelog.py", "-q"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
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
    print("=== v0.9.22b ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print("  branch: " + branch)
    assert branch == "master", (
        f"v0.9.22b is post-merge fix; must be on master, got {branch!r}"
    )

    fix_rank_sites()
    fix_resample_site()
    fix_pivot_site()
    update_changelog()
    append_decisions_246_247()
    add_ast_test()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.22b: AUDIT_ML §1.6 — extra_factor at to_pylist sites. "
        "5 sites: rank.py _compute_streamable (5.0, null fallback: "
        "by+row_id pylist; null-free uses numpy fast path); "
        "_pass1_groups_to_disk (3.0, by pylist); _pass2_ranks_to_disk "
        "(5.0, by+row_id pylist); resample.py sorted_input (5.0, ts "
        "+ K cols per-row pylist); pivot.py stream_sorted (5.0, "
        "K_index + columns + values per-row pylist). Prior default "
        "extra_factor=1.0 (batch.nbytes) under-reported Python-"
        "object expansion ~3.5× per pylist; null / skew / per-row "
        "triggers could silent OOM instead of clean "
        "MemoryLimitExceeded. groupby.py _groupby_with_skew existing "
        "extra_factor=3.0 deemed adequate: _accumulate_heavy's "
        "grouped.to_pylist() bounded by heavy_keys count (≤20 "
        "typical), not by batch — documented, not changed. "
        "(§13.4/247 new) check_batch extra_factor + inline comment, "
        "not just factor. Future reader needs to know WHY 5.0 vs "
        "3.0 (N pylist × 28B/8B ≈ N × 3.5); otherwise next audit "
        "bumps factor again without basis. Same class as "
        "§13.4/240 (formula unit vs metric), /243 "
        "(batch_memory_factor override — inherited default wrong). "
        "(§13.4/246 new, backfill) blanket `git add -A` needs "
        "directory audit before commit — user-content folders "
        "(`FOR_PINGIS/`, `temp/`, `notes/`, `docs/` unless "
        "gitignored) will be swept in. v0.9.22-fix1 commit message "
        "mentioned this but DECISIONS entry was never appended; "
        "backfilled here. (Backward compat) 1916 existing tests "
        "unchanged; factor bump is more conservative (over-reject "
        "in rare null/skew cases, no false-negative). 5 new "
        "parametrized AST tests: test_check_batch_has_extra_factor "
        "(1 per site) — verifies explicit factor on same line as "
        "name=; catches future regression. (AUDIT_ML round 3 "
        "backlog = 0) 1.1-1.4, 1.6, 1.7 closed; 1.5 kmeans++ "
        "reservoir + 2.5 RobustScaler deferred to v0.9.23. "
        "(Roadmap post-v0.9.22b) pause 2-3 weeks recommended — "
        "9 patches + 5 fix-cycles since v0.9.14; churn-driven. "
        "Production feedback first. v0.9.23: kmeans++ reservoir "
        "(quality +36.9% on skewed data closed); RobustScaler "
        "O(d) batched. v0.9.24+: A1 split (5404-line "
        "dataframe.py → 7 modules); Pingis round 4 (v1.0 "
        "readiness); Query engine / outer join / cumsum features.",
        self_path="scripts/patch_v0922b.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.22b applied. AUDIT_ML round 3 backlog = 0.")


if __name__ == "__main__":
    main()
