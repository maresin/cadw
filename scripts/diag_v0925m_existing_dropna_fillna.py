#!/usr/bin/env python3
# v0.9.25m-diagnostic — READ-ONLY inspection of existing dropna/fillna.
#
# Hasattr in v0.9.25m verify_state aborted pre-mutation: dropna exists.
# Do NOT mutate. Print actual signatures + source + tests, then
# manually decide: test-only patch, fix-up patch, or already done.
#
# §13.4/264 applied: inspect actual state, don't assume spec matches.
#
# Run: uv run python scripts/diag_v0925m_existing_dropna_fillna.py

from __future__ import annotations
import inspect
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def head():
    return subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def check_method(name):
    from cadw import CadwDataFrame
    if not hasattr(CadwDataFrame, name):
        print(f"\n[{name}] MISSING")
        return None
    fn = getattr(CadwDataFrame, name)
    print(f"\n[{name}] PRESENT")
    try:
        sig = inspect.signature(fn)
        print(f"  signature: {name}{sig}")
    except (TypeError, ValueError) as e:
        print(f"  signature: <unavailable: {e}>")
    src = inspect.getsource(fn)
    n_lines = len(src.splitlines())
    print(f"  source lines: {n_lines}")
    # Marker checks (semantic spec)
    markers = {
        "streaming_batches": "iter_filtered_batches" in src or "_iter_" in src,
        "is_valid_or_null": "is_valid" in src or "is_null" in src,
        "fill_null": "fill_null" in src,
        "how_param": "how" in src,
        "subset_param": "subset" in src,
        "dict_branch": "isinstance(value, dict)" in src or "value.items()" in src,
        "pa_scalar_typed": "pa.scalar(" in src,
        "raises_TypeError": "raise TypeError" in src,
        "has_docstring": '"""' in src,
    }
    for k, v in markers.items():
        print(f"  {'✓' if v else '✗'} {k}")
    # First 3 lines of docstring if present
    if '"""' in src:
        parts = src.split('"""')
        if len(parts) >= 2:
            doc = parts[1].strip().splitlines()[:5]
            print("  docstring head:")
            for ln in doc:
                print(f"    {ln}")
    return (fn, sig, src)


def check_tests():
    print("\n=== existing tests referencing dropna/fillna ===")
    tests_dir = Path("tests")
    hits = []
    for p in tests_dir.glob("*.py"):
        try:
            content = p.read_text()
        except Exception:
            continue
        has_dropna = "dropna" in content
        has_fillna = "fillna" in content
        if has_dropna or has_fillna:
            hits.append(
                (p.name,
                 sum(1 for ln in content.splitlines()
                     if "def test_" in ln and "dropna" in ln),
                 sum(1 for ln in content.splitlines()
                     if "def test_" in ln and "fillna" in ln))
            )
    if not hits:
        print("  (no existing tests reference dropna/fillna)")
    for name, nd, nf in hits:
        print(f"  {name}: {nd} dropna test defs, {nf} fillna test defs")


def check_datasets_git():
    """Was dropna/fillna added in a specific commit?"""
    print("\n=== git log grep for dropna/fillna ===")
    for term in ("dropna", "fillna"):
        r = subprocess.run(
            ["git", "log", "--oneline", "--all",
             f"--grep={term}", "-n", "5"],
            capture_output=True, text=True,
        )
        out = r.stdout.strip()
        if out:
            print(f"  [{term}]:")
            for ln in out.splitlines():
                print(f"    {ln}")
        else:
            print(f"  [{term}]: (no commits mention {term})")


def check_changelog():
    print("\n=== CHANGELOG mentions of dropna/fillna ===")
    p = Path("CHANGELOG.md")
    if not p.exists():
        print("  CHANGELOG.md missing")
        return
    content = p.read_text()
    for term in ("dropna", "fillna"):
        lines = [
            (i + 1, ln) for i, ln in enumerate(content.splitlines())
            if term in ln.lower()
        ]
        if not lines:
            print(f"  [{term}]: (not mentioned)")
        else:
            print(f"  [{term}] mentioned at lines:")
            for n, ln in lines[:10]:
                print(f"    {n}: {ln.strip()[:120]}")


def main():
    print("=== v0.9.25m-diagnostic: existing dropna/fillna ===")
    print(f"  HEAD: {head()}")
    print(f"  branch: {subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True).stdout.strip()}")

    check_method("dropna")
    check_method("fillna")
    check_tests()
    check_datasets_git()
    check_changelog()

    print("\n=== summary + decision tree ===")
    print("""
Read the sections above. Decision tree:

  A. dropna AND fillna are semantically complete
     (how=any/all, subset, dict-value, pa.scalar-typed coercion,
      streaming, TypeErrors on bad value)
     → WRITE TESTS ONLY. tests/test_v0925m.py. Do not touch production.

  B. dropna OR fillna is missing required semantics
     (e.g. no how='all', no dict-value, no streaming, no type-coercion)
     → FIX-UP PATCH. Diagnose exact gap first (inspect.getsource),
       then patch.

  C. Both exist but throw / return wrong on basic case
     (e.g. crash on dropna(), or fillna(0) on int col corrupts)
     → BUG PATCH. Write minimal repro test, then fix.

In all cases:
  - §13.4/278: verify_state via hasattr, not source text.
  - §13.4/266: assert exact anchor text before apply.
  - §13.4/264: REPL-check pa.scalar/pc.is_valid etc BEFORE
    writing test assumptions.
""")


if __name__ == "__main__":
    main()