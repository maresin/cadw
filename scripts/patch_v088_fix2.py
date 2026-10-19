#!/usr/bin/env python3
"""v0.8.8-fix2: fix §11.6 counter (cosmetic bug in audit, not --apply blocker).

BUG: `§11.6 tech debt: 2 rows (4 closed, -2 open)` — nonsense.

TWO INDEPENDENT BUGS in the counter:
1. `re.findall(r'^\\| `', sec)` counts only rows starting with `` | ` ``
   (backtick). Misses `| Combiner ...`, `| Параллелизм ...` (open rows
   without backtick in description). Under-counts open rows.
2. `sec.count("~~") // 2` — counts `~~` PAIRS. Each closed row has
   TWO pairs (4 `~~`), so `count = 8`, `// 2 = 4`, `4 - 2 = -2`.

Fix: count DATA rows explicitly (start `| `, not `|-`, not `| Долг`),
closed = rows containing `~~`.

Expected output after fix:
  §11.6 tech debt: 6 rows (2 closed, 4 open)

Cosmetic — does NOT affect --apply behavior (A/B/C/D phases unchanged).
Just fixes the audit summary display.

Run: uv run python scripts/patch_v088_fix2.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:25]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if len(starts) == 0:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


def fix_audit_116_counter():
    print("\n=== patch_v088_housekeeping.py: §11.6 counter ===")
    p = Path("scripts/patch_v088_housekeeping.py")
    s = p.read_text()

    old = '''    # §11.6 tech debt
    # v0.8.8-fix: find next `##`/`###` header (not `---`, which matches
    # the table separator `|---|---|` inside §11.6 before data rows).
    s116 = dec.find("### 11.6")
    if s116 > 0:
        m = re.search(r'^#{2,3} ', dec[s116 + 5:], re.MULTILINE)
        if m:
            e116 = s116 + 5 + m.start()
        else:
            e116 = len(dec)
        sec = dec[s116:e116]
        debt_rows = len(re.findall(r'^\\| `', sec, re.MULTILINE))
        closed_debt = sec.count("~~") // 2
        print(f"\\n§11.6 tech debt: {debt_rows} rows "
              f"({closed_debt} closed, {debt_rows - closed_debt} open)")'''

    new = '''    # §11.6 tech debt
    # v0.8.8-fix: find next `##`/`###` header (not `---`, which matches
    # the table separator `|---|---|` inside §11.6 before data rows).
    # v0.8.8-fix2: count DATA rows explicitly (start `| `, skip `|-` and
    # `| Долг`); closed = rows containing `~~`. Prior impl used
    # `^\\| ` regex (missed open rows without backtick) + `~~ count // 2`
    # (each closed row has 2 pairs → 4 `~~`, so `count=8, //2=4, 4-2=-2`).
    s116 = dec.find("### 11.6")
    if s116 > 0:
        m = re.search(r'^#{2,3} ', dec[s116 + 5:], re.MULTILINE)
        if m:
            e116 = s116 + 5 + m.start()
        else:
            e116 = len(dec)
        sec = dec[s116:e116]
        lines = sec.split("\\n")
        data_rows = [
            l for l in lines
            if l.startswith("| ")
            and not l.startswith("|-")
            and not l.startswith("| Долг")
        ]
        closed_rows = [l for l in data_rows if "~~" in l]
        debt_rows = len(data_rows)
        closed_debt = len(closed_rows)
        print(f"\\n§11.6 tech debt: {debt_rows} rows "
              f"({closed_debt} closed, {debt_rows - closed_debt} open)")'''

    s = replace_line_block(s, old, new, tag="audit-116-counter")
    p.write_text(s)
    compile_check(p)
    print("  OK: count data rows by prefix; closed by `~~` presence")


def main():
    print("=== v0.8.8-fix2: §11.6 counter ===")

    housekeeping = Path("scripts/patch_v088_housekeeping.py")
    if not housekeeping.exists():
        print(f"ERROR: {housekeeping} not found")
        sys.exit(1)

    fix_audit_116_counter()

    print("\n=== re-run dry-run to verify ===")
    r = subprocess.run(
        ["uv", "run", "python", "scripts/patch_v088_housekeeping.py"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** dry-run FAILED ***")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("==> v0.8.8-fix2 complete.")
    print("=" * 60)
    print()
    print("Expected in dry-run above:")
    print("  §11.6 tech debt: 6 rows (2 closed, 4 open)   ← was 2 / -2")
    print()
    print("NEXT:")
    print("  uv run python scripts/patch_v088_housekeeping.py --apply")
    print("  (then squash → tag v0.8.8)")


if __name__ == "__main__":
    main()