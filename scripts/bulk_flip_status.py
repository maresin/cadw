#!/usr/bin/env python3
"""Bulk flip implementation_status.status for all cards of a function.

v0.9.111. Companion to test_R5 (aliases consistency): when flipping
status for a fn with multiple cards, all aliases must flip together.

Usage:
    uv run python scripts/bulk_flip_status.py --fn fit_birch --from pending --to verified
    uv run python scripts/bulk_flip_status.py --fn fit_birch --from draft --to verified --dry-run

Rules: card-scoped (A16), anchor-miss fails loud (AP8), idempotent.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YAML_DIR = ROOT / "docs" / "ml_tools"
SKIP = {"benchmarks.yaml", "memory_constants.yaml"}
ALLOWED = {"draft", "verified", "tier1_core_exception", "pending"}


def find_cards(src: str, fn: str):
    out = []
    headers = list(re.finditer(r'^(\s*)-\s+name:\s*(\S.*?)\s*$', src, re.MULTILINE))
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(src)
        block = src[start:end]
        if re.search(rf'^\s*cadw_function:\s*{re.escape(fn)}\s*$', block, re.MULTILINE):
            name = m.group(2).strip().strip('"\'')
            out.append((start, end, name))
    return out


def flip_in_block(block: str, from_s: str, to_s: str):
    m = re.search(r'^(\s*)status:\s*(\S+)\s*$', block, re.MULTILINE)
    if not m:
        return block, "no_status_field"
    cur = m.group(2).strip().strip('"\'')
    cur_norm = "(pending)" if cur.lower() in ("null", "~", "none") else cur
    want = "(pending)" if from_s == "pending" else from_s
    if cur_norm != want:
        return block, f"noop (cur={cur!r}, want={from_s!r})"
    return block[:m.start()] + f"{m.group(1)}status: {to_s}" + block[m.end():], "flipped"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn", required=True)
    ap.add_argument("--from", dest="from_s", required=True, choices=sorted(ALLOWED))
    ap.add_argument("--to", dest="to_s", required=True, choices=sorted(ALLOWED - {"pending"}))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.from_s == args.to_s:
        print("ERR: --from == --to"); sys.exit(1)
    total = 0
    flipped = 0
    for yp in sorted(YAML_DIR.glob("*.yaml")):
        if yp.name in SKIP:
            continue
        src = yp.read_text(encoding="utf-8")
        cards = find_cards(src, args.fn)
        if not cards:
            continue
        changed = False
        for start, end, name in reversed(cards):
            block = src[start:end]
            new_block, action = flip_in_block(block, args.from_s, args.to_s)
            total += 1
            if action == "flipped":
                if not args.dry_run:
                    src = src[:start] + new_block + src[end:]
                    changed = True
                flipped += 1
                print(f"  {yp.name}:{name} — flipped {args.from_s} -> {args.to_s}")
            else:
                print(f"  {yp.name}:{name} — skip ({action})")
        if changed and not args.dry_run:
            yp.write_text(src, encoding="utf-8")
    if total == 0:
        print(f"ERR: no cards found with cadw_function={args.fn!r}"); sys.exit(1)
    print(f"\nTotal: {total}, flipped: {flipped}" + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
