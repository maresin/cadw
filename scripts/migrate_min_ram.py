"""migrate_min_ram — derive min_ram_formula from ram_formula (v0.9.177).

Usage:
    python scripts/migrate_min_ram.py --dry-run          # report only
    python scripts/migrate_min_ram.py --pilot NAME1,NAME2  # apply to listed cards
    python scripts/migrate_min_ram.py --all              # apply to all eligible

Rule (AP23 analog for bottom-bound):
    min_ram_formula = ram_formula with `batch` -> `1`, then ensure `+ 512`
    (fixed_overhead header) if no literal `+ 512`/`+ 1024` present.

Rationale: minimal working batch = 1 for batch-layer primitives (Reduce,
HashShuffle, Sort). If model needs > 1 (e.g., sort merge = 2), override
manually.

Cards without `batch` in ram_formula → min = ram (state dominates; honest
bottom-bound).

Evidence:
    `min_ram_evidence: {"derived_from": "ram_formula (batch=1, auto v0.9.177)"}`

Skips:
    - cards with existing min_ram_formula
    - cards marked `# min_ram_pending: <reason>` in YAML comment
    - non-verified cards
    - cards without ram_formula (legacy: bounded_state_dims)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
ML_TOOLS = ROOT / "docs" / "ml_tools"


def _iter_sections():
    for yp in sorted(ML_TOOLS.glob("[A-Z]_*.yaml")):
        yield yp


def _has_batch_term(formula: str) -> bool:
    return bool(re.search(r"\bbatch\b", formula))


def _derive_min_ram(ram_formula: str) -> str:
    """min = ram_formula with batch->1; ensure + 512 header."""
    formula = ram_formula.strip().strip('"').strip("'")

    # Replace standalone `batch` identifier with `1`
    derived = re.sub(r"\bbatch\b", "1", formula)

    # Ensure + 512 (or larger) header present
    if not re.search(r"\+\s*\d{3,}", derived):
        derived = derived + " + 512"

    return derived


def _iter_cards(yfile: Path):
    """Yield (card_dict, line_start, line_end, card_name, yaml_text) for each model."""
    src = yfile.read_text()
    data = yaml.safe_load(src) or {}

    # We need line positions for each `- name: <NAME>` block.
    # Block boundary: next `- name:` at same indent OR next `subtypes:`/EOF.
    for st in data.get("subtypes") or []:
        st_name = st.get("id") or st.get("name") or "?"
        for m in st.get("models") or []:
            name = m.get("name")
            if not name:
                continue
            # Find `- name: <name>` line
            pattern = rf"^(\s*)- name:\s*{re.escape(name)}\s*$"
            mm = re.search(pattern, src, re.MULTILINE)
            if not mm:
                continue
            indent = mm.group(1)
            start = mm.start()
            next_m = re.search(
                rf"^{re.escape(indent)}- name:", src[mm.end():], re.MULTILINE)
            end = (mm.end() + next_m.start()) if next_m else len(src)
            yield m, start, end, name


def _has_pending_marker(block_text: str) -> bool:
    return bool(re.search(r"#\s*min_ram_pending\s*:", block_text))


def _apply_one(yfile: Path, card_name: str, *, dry_run: bool) -> str:
    """Apply migration to one card in yfile. Return status string."""
    src = yfile.read_text()
    data = yaml.safe_load(src) or {}

    # Locate card
    card = None
    for st in data.get("subtypes") or []:
        for m in st.get("models") or []:
            if m.get("name") == card_name:
                card = m
                break
        if card:
            break

    if not card:
        return f"SKIP: {card_name} (not found)"

    status = (card.get("implementation_status") or {}).get("status")
    if status != "verified":
        return f"SKIP: {card_name} (status={status!r}, not verified)"

    ram = card.get("ram_formula")
    if not ram:
        return f"SKIP: {card_name} (no ram_formula; legacy bounded_state_dims)"

    if card.get("min_ram_formula"):
        return f"SKIP: {card_name} (already has min_ram_formula)"

    # Find block
    block_info = None
    for m_dict, start, end, name in _iter_cards(yfile):
        if name == card_name:
            block_info = (start, end, name)
            break

    if not block_info:
        return f"SKIP: {card_name} (block bounds not found)"

    start, end, _ = block_info
    block = src[start:end]

    if _has_pending_marker(block):
        return f"SKIP: {card_name} (# min_ram_pending marker)"

    derived = _derive_min_ram(ram)

    # Build insertion lines: after ram_formula line (indent = 2 more than `- name`)
    # Find ram_formula line in block
    ram_match = re.search(
        r"^(\s*)ram_formula:\s*[^\n]+\n", block, re.MULTILINE)
    if not ram_match:
        return f"SKIP: {card_name} (ram_formula line not found in block)"

    indent = ram_match.group(1)
    insert_after_pos = ram_match.end()

    # Check if `formula_evidence:` block follows; if yes, place min_ram AFTER
    # formula_evidence so that evidence is adjacent to formula.
    tail = block[insert_after_pos:]
    evidence_match = re.search(
        rf"^{re.escape(indent)}formula_evidence:\s*\n"
        rf"(?:{re.escape(indent)}  [^\n]+\n)*",
        tail, re.MULTILINE)
    if evidence_match:
        insert_at = insert_after_pos + evidence_match.end()
    else:
        insert_at = insert_after_pos

    derived_escaped = derived.replace('"', '\\"')
    NEW_LINES = (
        f'{indent}min_ram_formula: "{derived_escaped}"\n'
        f'{indent}min_ram_evidence:\n'
        f'{indent}  derived_from: "ram_formula (batch=1, auto-migrate v0.9.177)"\n'
    )

    if dry_run:
        return f"DRY: {card_name}: {derived}"

    new_block = block[:insert_at] + NEW_LINES + block[insert_at:]
    new_src = src[:start] + new_block + src[end:]
    yfile.write_text(new_src)
    return f"OK: {card_name}: {derived}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, no writes")
    ap.add_argument("--pilot", type=str, default=None,
                    help="comma-separated card names; apply only to these")
    ap.add_argument("--all", action="store_true",
                    help="apply to all eligible cards")
    ap.add_argument("--section", type=str, default=None,
                    help="limit to one section, e.g. A_classification")
    args = ap.parse_args(argv)

    if not (args.dry_run or args.pilot or args.all):
        print("Specify --dry-run, --pilot NAME1,NAME2, or --all")
        return 1

    pilot_set = set(n.strip() for n in args.pilot.split(",")) if args.pilot else None

    stats = {"migrated": 0, "skipped": 0, "pilot_missed": 0}
    for yf in _iter_sections():
        if args.section and yf.stem != args.section:
            continue
        src = yf.read_text()
        data = yaml.safe_load(src) or {}
        for st in data.get("subtypes") or []:
            for m in st.get("models") or []:
                name = m.get("name")
                if not name:
                    continue
                if pilot_set and name not in pilot_set:
                    continue
                result = _apply_one(yf, name, dry_run=args.dry_run)
                print(result)
                if result.startswith("OK") or result.startswith("DRY"):
                    stats["migrated"] += 1
                else:
                    stats["skipped"] += 1

    if pilot_set:
        # Report pilot misses
        applied = set()
        # (re-check by scanning files would need reload; simpler: trust counter)
        if stats["migrated"] + stats["skipped"] < len(pilot_set):
            pass  # some pilots not found; already printed as SKIP

    print(f"\n=== Stats: migrated={stats['migrated']} skipped={stats['skipped']} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
