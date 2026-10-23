#!/usr/bin/env python3
# v0.9.26b-diagnostic — Phase D scope + roadmap consistency.
#
# READ-ONLY. No mutation, no tests, no commit. Purpose: inspect
# actual roadmap.json state before deciding the next patch.
#
# §13.4/280 applied: actual file inspection (json.loads + key-by-key
# walk), NOT inference from roadmap narrative prose or commit
# messages. Surfaces real state, not assumed state.
#
# Run: uv run python scripts/diag_v0926b_phase_d_scope.py

from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _short(obj, limit=240):
    """Truncate strings for readability; repr large objects."""
    if isinstance(obj, str):
        if len(obj) > limit:
            return obj[:limit] + "…"
        return obj
    r = repr(obj)
    if len(r) > limit:
        return r[:limit] + "…"
    return r


def dump_phase(idx, ph):
    print(f"\n--- phases[{idx}] ---")
    if not isinstance(ph, dict):
        print(f"  (not dict — type={type(ph).__name__}, "
              f"value={_short(ph)})")
        return
    known = {
        "id", "name", "status", "version", "completed_at",
        "started_at", "description", "summary", "milestone",
        "target_version", "depends_on", "dependencies", "features",
    }
    for k in sorted(known & set(ph.keys())):
        if k == "features":
            continue  # handled below
        print(f"  {k}: {_short(ph[k])}")
    feats = ph.get("features")
    if isinstance(feats, list):
        print(f"  features ({len(feats)}):")
        for i, f in enumerate(feats):
            if isinstance(f, dict):
                name = (f.get("id") or f.get("name")
                        or f.get("feature") or f"#{i}")
                st = f.get("status", "?")
                desc = (f.get("description")
                        or f.get("summary") or "")
                print(f"    [{i}] {name} (status={st}): "
                      f"{_short(desc, 100)}")
            else:
                print(f"    [{i}] {_short(f, 120)}")
    elif feats is not None:
        print(f"  features ({type(feats).__name__}): "
              f"{_short(feats)}")
    extra = set(ph.keys()) - known
    if extra:
        print(f"  other keys: {sorted(extra)}")
        for k in sorted(extra):
            print(f"    {k}: {_short(ph[k])}")


def extract_md_section(md_text, phase_letter):
    """Extract §Phase <letter> heading section from markdown."""
    start_pat = re.compile(
        rf"^#+\s+.*Phase\s+{phase_letter}\b.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = start_pat.search(md_text)
    if not m:
        return None
    after = md_text[m.end():]
    next_pat = re.compile(
        r"^#+\s+.*Phase\s+[A-Z]\b.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    n = next_pat.search(after)
    end = m.end() + (n.start() if n else len(after))
    section = md_text[m.start():end]
    # Cap at 120 lines for readability
    lines = section.splitlines()[:120]
    return "\n".join(lines)


def main():
    print("=== v0.9.26b-diagnostic: Phase D scope + consistency ===")

    p = Path("docs/roadmap.json")
    if not p.exists():
        raise SystemExit(f"missing {p}")
    data = json.loads(p.read_text())

    # --- top level ---
    print("\n--- top-level ---")
    for k in ("schema_version", "current_version", "updated"):
        if k in data:
            print(f"  {k}: {data[k]}")
    print(f"  keys: {sorted(data.keys())}")

    phases = data.get("phases", [])
    print(f"\n  phases count: {len(phases)}")

    # --- compact overview ---
    print("\n--- phase overview ---")
    for i, ph in enumerate(phases):
        if isinstance(ph, dict):
            pid = ph.get("id", "?")
            name = ph.get("name", "?")
            st = ph.get("status", "?")
            feats = ph.get("features")
            nf = len(feats) if isinstance(feats, list) else "?"
            print(f"  [{i}] id={pid!r} status={st!r} "
                  f"features={nf} name={name!r}")
        else:
            print(f"  [{i}] (non-dict: {type(ph).__name__})")

    # --- detailed dumps ---
    if len(phases) > 0:
        print("\n\n=== Phase A detail (roadmap says in_progress) ===")
        dump_phase(0, phases[0])
    if len(phases) > 1:
        print("\n\n=== Phase B detail (partial per project history) ===")
        dump_phase(1, phases[1])
    if len(phases) > 3:
        print("\n\n=== Phase D detail (NEXT SCOPE) ===")
        dump_phase(3, phases[3])

    print("\n\n=== Phases E–H (brief) ===")
    for i in range(4, min(8, len(phases))):
        ph = phases[i]
        if isinstance(ph, dict):
            print(f"  [{i}] id={ph.get('id')!r} "
                  f"status={ph.get('status')!r} "
                  f"name={ph.get('name')!r}")

    # --- markdown cross-check ---
    print("\n\n=== docs/ROADMAP.md §Phase D section ===")
    md_p = Path("docs/ROADMAP.md")
    if md_p.exists():
        section = extract_md_section(md_p.read_text(), "D")
        if section:
            print(section)
        else:
            print("  (no Phase D heading found in ROADMAP.md)")
    else:
        print("  (ROADMAP.md missing)")

    # --- consistency warnings ---
    print("\n\n=== consistency warnings ===")
    if len(phases) > 0 and isinstance(phases[0], dict):
        a_st = phases[0].get("status")
        if a_st == "in_progress":
            print(
                "  ⚠ Phase A status='in_progress' — but stabilization\n"
                "    closed at v0.9.14 per CHANGELOG (audit close).\n"
                "    Likely stale roadmap field.\n"
                "    → v0.9.26c-roadmap_consistency could set:\n"
                "      A → completed, populate completed_at."
            )
    if len(phases) > 1 and isinstance(phases[1], dict):
        b_st = phases[1].get("status")
        if b_st == "planned":
            print(
                "\n  ⚠ Phase B status='planned' — but partial split\n"
                "    (A1) shipped v0.9.16; only full dataframe.py\n"
                "    split pending. Consider:\n"
                "    → status='partial' OR split as B sub-feature."
            )

    # --- recommendation ---
    print("\n\n=== recommendation (based on state above) ===")
    print("""
Decide based on Phase D dump:

  (A) Phase D has named features + clear scope (like Phase C did)
      → v0.9.26b = first Phase D batch (parallel to how Phase C
        ran C-1 ... C-5). Split per feature-family, 2-3 features
        per batch, REPL-first + hasattr-verify.

  (B) Phase D is placeholder / narrative-only
      → v0.9.26b-roadmap_consistency first (fix A + B statuses,
        verify C details), THEN re-run diagnostic for D.
      → Optionally: v0.9.26b-dataframe_split in parallel (A1,
        deferred since v0.9.16, 5404 LOC → 7 modules; needs its
        own blueprint — architectural, high-risk).

  (C) Phase D is small & well-scoped (1-2 features)
      → v0.9.26b = those features + minimal tests, same
        discipline as Phase C.

Do NOT start Phase D without reading the actual dump above —
guess-based scope is exactly what §13.4/280 prevents.
""")


if __name__ == "__main__":
    main()