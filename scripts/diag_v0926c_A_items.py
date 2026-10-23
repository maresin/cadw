#!/usr/bin/env python3
# v0.9.26c-diagnostic — inspect A.6/A.9/A.10 actual code state.
#
# READ-ONLY. No mutation, no tests, no commit. Purpose: decide
# patch-vs-close for each Phase A planned item via actual code
# inspection (§13.4/280: actual, not roadmap narrative).
#
# Items inspected (from v0.9.26b-corrected roadmap dump):
#   A.6  '18 to_pylist sites extra_factor'    M  v0.9.25d
#   A.9  'B5 dead code _append_to_builders'   S  v0.9.25e
#   A.10 'B4 memory_guard fallback factor'    S  v0.9.25e
#
# Run: uv run python scripts/diag_v0926c_A_items.py

from __future__ import annotations
import ast
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============ A.6: to_pylist sites ============

def scan_to_pylist():
    """Find every `.to_pylist(...)` call in cadw/ via AST."""
    sites = []
    for py in Path("cadw").rglob("*.py"):
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError as e:
            print(f"  ⚠ parse error in {py}: {e}")
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "to_pylist"):
                continue
            has_ef = any(
                kw.arg == "extra_factor" for kw in node.keywords
            )
            has_any_kw = any(
                kw.arg for kw in node.keywords
            )
            sites.append({
                "file": str(py),
                "line": node.lineno,
                "has_extra_factor": has_ef,
                "has_any_keyword": has_any_kw,
                "receiver": ast.unparse(node.func.value)[:80],
            })
    return sites


def scan_extra_factor_uses():
    """Find all `extra_factor` occurrences (helper defs + call sites)."""
    hits = []
    for py in Path("cadw").rglob("*.py"):
        for i, line in enumerate(
            py.read_text().splitlines(), 1
        ):
            if "extra_factor" in line:
                hits.append({
                    "file": str(py),
                    "line": i,
                    "text": line.strip()[:120],
                })
    return hits


def report_A6():
    print("\n" + "=" * 70)
    print("A.6 — to_pylist sites + extra_factor coverage")
    print("=" * 70)

    sites = scan_to_pylist()
    ef_uses = scan_extra_factor_uses()

    by_file = Counter(s["file"] for s in sites)
    with_ef = [s for s in sites if s["has_extra_factor"]]
    without_ef = [s for s in sites if not s["has_extra_factor"]]

    print(f"\nTotal .to_pylist() calls in cadw/: {len(sites)}")
    print(f"  with extra_factor=: {len(with_ef)}")
    print(f"  without extra_factor=: {len(without_ef)}")
    print(f"\nPer-file count:")
    for f, n in sorted(by_file.items(), key=lambda kv: -kv[1]):
        print(f"  {f}: {n}")

    print(f"\n--- Sites WITH extra_factor= ({len(with_ef)}) ---")
    for s in with_ef:
        print(f"  {s['file']}:{s['line']}  "
              f"receiver={s['receiver']}")

    print(f"\n--- Sites WITHOUT extra_factor= ({len(without_ef)}) ---")
    for s in without_ef:
        print(f"  {s['file']}:{s['line']}  "
              f"receiver={s['receiver']}")

    print(f"\n--- All extra_factor occurrences "
          f"(helper defs + call sites, {len(ef_uses)}) ---")
    for h in ef_uses:
        print(f"  {h['file']}:{h['line']}  {h['text']}")

    print(f"\n[A.6 summary]:")
    print(f"  Roadmap claims '18 to_pylist sites' need extra_factor.")
    print(f"  Actual total to_pylist calls: {len(sites)}")
    print(f"  Actual with extra_factor:     {len(with_ef)}")
    print(f"  Actual without extra_factor:  {len(without_ef)}")
    if len(with_ef) >= 18:
        print(f"  → A.6 likely DONE (≥18 have extra_factor)")
    elif len(with_ef) >= 13:  # 5 from v0.9.22b + 8 more
        print(f"  → A.6 likely PARTIAL: {len(with_ef)}/18 done, "
              f"{18 - len(with_ef)} to go")
    else:
        print(f"  → A.6 likely OPEN: {len(with_ef)} sites have "
              f"extra_factor; {18 - len(with_ef)} may still need it")
    print(f"  NOTE: total without extra_factor = {len(without_ef)}"
          f" — some may be scalar (fine), some may need it.")
    print(f"  Human judgment: inspect receiver expressions above.")


# ============ A.9: _append_to_builders ============

def scan_symbol(symbol):
    """Find define + call sites for a symbol via regex."""
    hits = []
    for py in Path("cadw").rglob("*.py"):
        try:
            src = py.read_text()
        except Exception:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if symbol in line:
                hits.append({
                    "file": str(py),
                    "line": i,
                    "text": line.strip()[:140],
                })
    return hits


def report_A9():
    print("\n" + "=" * 70)
    print("A.9 — _append_to_builders dead-code check")
    print("=" * 70)
    sym = "_append_to_builders"
    hits = scan_symbol(sym)
    if not hits:
        print(f"\n  ✓ '{sym}' NOT FOUND anywhere in cadw/ — "
              f"A.9 likely DONE (already removed)")
        return
    defs = [h for h in hits if f"def {sym}" in h["text"]]
    calls = [h for h in hits if f"def {sym}" not in h["text"]]
    print(f"\n  '{sym}' occurrences: {len(hits)} "
          f"({len(defs)} defs, {len(calls)} non-defs)")
    print(f"\n--- Definitions ---")
    for h in defs:
        print(f"  {h['file']}:{h['line']}  {h['text']}")
    print(f"\n--- Non-def usages ({len(calls)}) ---")
    for h in calls:
        print(f"  {h['file']}:{h['line']}  {h['text']}")
    if defs and not calls:
        print(f"\n  → A.9 OPEN: defined but never called "
              f"(dead code). Remove {len(defs)} def(s).")
    elif calls:
        print(f"\n  → A.9 PARTIAL: still called — inspect call "
              f"reachability (grep containing functions).")
    else:
        print(f"\n  → A.9 unknown — inspect manually.")


# ============ A.10: memory_guard fallback factor ============

def report_A10():
    print("\n" + "=" * 70)
    print("A.10 — memory_guard fallback factor")
    print("=" * 70)
    mg_path = Path("cadw/core/memory_guard.py")
    if not mg_path.exists():
        print(f"\n  ⚠ {mg_path} not found")
        return
    src = mg_path.read_text()
    # Search patterns relevant to fallback / factor / C_ASSUMED
    patterns = {
        "C_ASSUMED": r"C_ASSUMED\s*=",
        "fallback": r"fallback",
        "factor": r"factor",
        "DEFAULT_C": r"DEFAULT_C\s*=",
        "c_assumed": r"c_assumed\s*=",
    }
    print(f"\n--- Pattern matches in {mg_path} ---")
    for label, pat in patterns.items():
        matches = list(re.finditer(pat, src))
        print(f"  [{label}]: {len(matches)} match(es)")
        for m in matches[:5]:  # cap at 5 per pattern
            # print surrounding line
            start = src.rfind("\n", 0, m.start()) + 1
            end = src.find("\n", m.end())
            if end == -1:
                end = len(src)
            line_no = src.count("\n", 0, m.start()) + 1
            print(f"    line {line_no}: "
                  f"{src[start:end].strip()[:120]}")
        if len(matches) > 5:
            print(f"    ... and {len(matches) - 5} more")
    print(f"\n[A.10 summary]:")
    print(f"  Roadmap: 'B4 memory_guard fallback factor'. Look for:")
    print(f"    - Existing C_ASSUMED constant → check if audit-suggested")
    print(f"    - Fallback factor when calibration unavailable")
    print(f"    - Memory guard init default factor")
    print(f"  Cross-check CHANGELOG v0.9.19: 'warn at C > 2.0' "
          f"(calibration drift warning).")
    print(f"  Cross-check CHANGELOG v0.9.22: 'NB C_ASSUMED=64 (was 32)'.")


# ============ audit report cross-ref ============

def report_audit():
    print("\n" + "=" * 70)
    print("Cross-ref: _FOR_PINGIS/cadw_audit_report.md")
    print("=" * 70)
    p = Path("_FOR_PINGIS/cadw_audit_report.md")
    if not p.exists():
        print(f"\n  ⚠ {p} not present in this repo state.")
        print(f"  The snapshot shows it as [STUB] — actual file "
              f"may or may not exist. If absent, work from "
              f"CHANGELOG + roadmap + code only.")
        # Alternative: check other audit files
        for alt in ("_FOR_PINGIS", "AUDIT.md", "AUDIT_ML_v0922.md"):
            alt_p = Path(alt)
            if alt_p.exists():
                print(f"  → alternative present: {alt_p}")
        return
    content = p.read_text()
    print(f"\n  File present, {len(content.splitlines())} lines.")
    # Search for A.6 / A.9 / A.10 / B4 / B5 / to_pylist / fallback
    for term in ("A.6", "A.9", "A.10", "B4", "B5",
                 "to_pylist", "fallback_factor",
                 "_append_to_builders"):
        lines = [
            (i + 1, ln) for i, ln in
            enumerate(content.splitlines())
            if term in ln
        ]
        if lines:
            print(f"\n  [{term}]: {len(lines)} mention(s)")
            for n, ln in lines[:5]:
                print(f"    line {n}: {ln.strip()[:120]}")
            if len(lines) > 5:
                print(f"    ... and {len(lines) - 5} more")


# ============ main ============

def main():
    print("=== v0.9.26c-diagnostic: A.6 / A.9 / A.10 actual state ===")
    report_A6()
    report_A9()
    report_A10()
    report_audit()

    print("\n" + "=" * 70)
    print("NEXT-STEP DECISION TREE (based on output above)")
    print("=" * 70)
    print("""
Read the sections above. For each item, decide:

  A.6 (18 to_pylist sites):
    - with_ef >= 18           → CLOSE (mark 'done' in roadmap)
    - with_ef in [13, 17]     → PARTIAL: write patch adding extra_factor
                                 to remaining N sites (identify via output)
    - with_ef < 13            → OPEN: larger patch (inspect receiver
                                 types to classify string vs primitive)

  A.9 (_append_to_builders):
    - Not found               → CLOSE ('done', already removed)
    - Def only, no calls      → OPEN: remove N def lines (small patch)
    - Def + calls             → PARTIAL: check if calls are reachable;
                                 if unreachable → delete both;
                                 if reachable → investigate reason

  A.10 (fallback factor):
    - C_ASSUMED / fallback    → likely present; check value against
      present + correct          audit recommendation (CHANGELOG v0.9.22
                                 mentions NB C_ASSUMED=64 fix).
                                 If matches → CLOSE.
    - Missing entirely        → OPEN: add fallback factor per audit.
    - Present but wrong value → PARTIAL: correct value.

  Combined closeout:
    - ALL THREE close without code → v0.9.26d-A_closeout_metaonly
      (roadmap status='done' for A.6/A.9/A.10 + phase A → 'completed'
      + CHANGELOG note. ~10 min.)
    - ANY open → v0.9.26d-A_closeout_code (code patches per item +
      roadmap + CHANGELOG. Scale by item count.)

Do NOT proceed to dataframe.py split (B.1) with Phase A planned items
open — Phase A is "Stabilization"; incomplete stabilization before
architectural refactor is exactly the debt roadmap sequencing prevents.
""")


if __name__ == "__main__":
    main()