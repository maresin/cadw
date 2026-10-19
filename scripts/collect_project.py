#!/usr/bin/env python3
"""Snapshot project to single file. --mode=smart|full|tiny.

SMART MODE (default, ~500 KB vs 2 MB full):
  Full content: cadw/**/*.py, tests/**/*.py, README.md, pyproject.toml,
    scripts/collect_project.py, .pingis/*.md, latest 3 patch_*.py,
    last-5 CHANGELOG entries, DECISIONS.md (§13.4 truncated to last 10
    episodes).
  Stub: older scripts/patch_*, scripts/archive/*, scripts/*.sh,
    .backups/*, uv.lock, older CHANGELOG entries, older §13.4 episodes.

Usage:
    uv run python scripts/collect_project.py                 # smart
    uv run python scripts/collect_project.py --mode=full     # 2MB
    uv run python scripts/collect_project.py --mode=tiny     # 50KB
    uv run python scripts/collect_project.py --tree          # tree only
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

EXCLUDE_DIRS = {
    ".venv", "venv", "env", ".env",
    "__pycache__", ".git", ".github", ".idea", ".vscode",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".basedpyright",
    "node_modules", "dist", "build", "htmlcov",
    "cadw_cache", "cadw_shuffle", ".ipynb_checkpoints", ".OTHER",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".pyd",
                    ".arrow", ".parquet", ".tar", ".gz"}
ROOT_FILES_FULL = ["README.md", "pyproject.toml"]
SEP = "# " + "=" * 76


def _is_excluded(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    return any(p in EXCLUDE_DIRS for p in rel_parts)


def _classify(p: Path, root: Path, keep_patches: int) -> str:
    rel = p.relative_to(root)
    s = rel.as_posix()
    if s == "CHANGELOG.md":
        return "changelog"
    if s == "DECISIONS.md":
        return "decisions"
    if s == "uv.lock":
        return "lock"
    if s in ROOT_FILES_FULL:
        return "full"
    if s.startswith("cadw/") and p.suffix == ".py":
        return "full"
    if s.startswith("tests/") and p.suffix == ".py":
        return "full"
    if s == "scripts/collect_project.py":
        return "full"
    if s.startswith(".pingis/") and p.suffix == ".md":
        return "full"
    if s.startswith("scripts/patch_") and p.suffix == ".py":
        all_patches = sorted(
            (root / "scripts").glob("patch_*.py"), key=lambda x: x.name,
        )
        latest = set(all_patches[-keep_patches:])
        return "full" if p in latest else "stub"
    return "stub"


def _truncate_changelog(text: str, n_keep: int) -> tuple[str, int]:
    matches = list(re.finditer(r'^## \[', text, re.MULTILINE))
    if len(matches) <= n_keep:
        return text, 0
    cut = matches[n_keep].start()
    header = text[:matches[0].start()]
    kept = text[matches[0].start():cut]
    n_omitted = len(matches) - n_keep
    marker = (
        f"\n<!-- [OMITTED {n_omitted} older CHANGELOG entries. "
        f"Use --mode=full. Full: `git log --oneline --grep='^v'`.] -->\n\n"
    )
    return header + kept + marker, n_omitted


def _truncate_decisions_13_4(text: str, n_keep: int) -> tuple[str, int]:
    start = text.find("### 13.4.")
    end = text.find("### 13.5", start)
    if start < 0 or end < 0:
        return text, 0
    sec = text[start:end]
    eps = list(re.finditer(r'\*\*Эпизод (\d+)', sec))
    if len(eps) <= n_keep:
        return text, 0
    first_kept = eps[-n_keep]
    cut_rel = first_kept.start()
    head_n = eps[0].group(1)
    last_omitted = eps[-n_keep - 1].group(1)
    n_omitted = len(eps) - n_keep
    marker = (
        f"<!-- [§13.4 episodes {head_n}..{last_omitted} OMITTED "
        f"({n_omitted} episodes). Use --mode=full.] -->\n\n"
    )
    new_sec = sec[:eps[0].start()] + marker + sec[cut_rel:]
    return text[:start] + new_sec + text[end:], n_omitted


def _print_tree(root: Path, max_depth: int = 6) -> None:
    print(f"Project root: {root}\n")

    def _walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(
                path.iterdir(), key=lambda p: (p.is_file(), p.name),
            )
        except PermissionError:
            return
        entries = [e for e in entries if not _is_excluded(e, root)]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            print(prefix + ("\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 ")
                  + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir():
                _walk(entry,
                      prefix + ("    " if is_last else "\u2502   "),
                      depth + 1)

    _walk(root, "", 0)


def _walk_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if not _is_excluded(Path(dirpath) / d, root)
        )
        for fname in sorted(filenames):
            p = Path(dirpath) / fname
            if p.suffix in EXCLUDE_SUFFIXES:
                continue
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", default="merged_cadw_project.py")
    ap.add_argument("--mode", choices=("smart", "full", "tiny"),
                    default="smart")
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--no-tree", action="store_true")
    ap.add_argument("--keep-patches", type=int, default=3)
    ap.add_argument("--changelog-entries", type=int, default=5)
    ap.add_argument("--decisions-eps", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: {root} not found", file=sys.stderr)
        return 1

    if args.tree:
        _print_tree(root)
        return 0
    if not args.no_tree:
        _print_tree(root)
        print()

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    classified: list[tuple[Path, str]] = []
    for p in _walk_files(root):
        if p.resolve() == output.resolve():
            continue
        if args.mode == "full":
            mode = "full"
            if p.name == "CHANGELOG.md":
                mode = "changelog"
            elif p.name == "DECISIONS.md":
                mode = "decisions"
        elif args.mode == "tiny":
            if p.suffix == ".md" and p.name in ("README.md", "CHANGELOG.md"):
                mode = "changelog" if p.name == "CHANGELOG.md" else "full"
            else:
                mode = "stub"
        else:
            mode = _classify(p, root, args.keep_patches)
        classified.append((p, mode))

    if not classified:
        print("WARNING: no files collected", file=sys.stderr)
        return 2

    n_changelog_omitted = 0
    n_eps_omitted = 0

    def _read(p: Path, mode: str) -> str:
        nonlocal n_changelog_omitted, n_eps_omitted
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"# [ERROR: {e}]\n"
        if mode == "changelog":
            content, n_om = _truncate_changelog(
                content, args.changelog_entries
            )
            n_changelog_omitted += n_om
        elif mode == "decisions":
            content, n_om = _truncate_decisions_13_4(
                content, args.decisions_eps
            )
            n_eps_omitted += n_om
        elif mode == "lock":
            lines = content.splitlines()
            if len(lines) > 50:
                content = "\n".join(lines[:50]) + (
                    f"\n# ... ({len(lines) - 50} more lines)\n"
                )
        return content

    def _stub_marker(p: Path, size: int) -> str:
        rel = p.relative_to(root).as_posix()
        if rel.startswith("scripts/archive/"):
            return f"# [STUB: archived patch ({size} B)]"
        if rel.startswith("scripts/") and rel.endswith((".sh", ".py")):
            return f"# [STUB: old patch ({size} B); use --mode=full]"
        if rel.startswith(".backups/"):
            return f"# [STUB: backup ({size} B)]"
        return f"# [STUB: {size} B]"

    total_bytes_in = 0
    total_bytes_written = 0
    stubs = 0

    with open(output, "w", encoding="utf-8") as out:
        out.write(SEP + "\n# CADW PROJECT SNAPSHOT\n")
        out.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        out.write(f"# Root: {root}\n")
        out.write(f"# Mode: {args.mode}\n")
        out.write(f"# Files: {len(classified)}\n#\n")
        out.write("# ВНИМАНИЕ: для чтения, НЕ для исполнения.\n"
                  + SEP + "\n\n")

        out.write(SEP + "\n# ОГЛАВЛЕНИЕ\n" + SEP + "\n")
        for i, (p, mode) in enumerate(classified, 1):
            rel = p.relative_to(root).as_posix()
            try:
                n = sum(1 for _ in p.open("r",
                                          encoding="utf-8",
                                          errors="replace"))
            except Exception:
                n = 0
            tag = " [STUB]" if mode == "stub" else ""
            out.write(f"# {i:3d}. {rel}  ({n} lines){tag}\n")
        out.write("\n")

        for p, mode in classified:
            rel = p.relative_to(root).as_posix()
            out.write(SEP + f"\n# FILE: {rel}\n" + SEP + "\n")
            total_bytes_in += p.stat().st_size
            if mode == "stub":
                stub = _stub_marker(p, p.stat().st_size)
                out.write(stub + "\n\n")
                total_bytes_written += len(stub) + 2
                stubs += 1
                continue
            content = _read(p, mode)
            if not content.endswith("\n"):
                content += "\n"
            out.write(content + "\n\n")
            total_bytes_written += len(content.encode("utf-8")) + 2

    size_out = output.stat().st_size
    print(f"\n=== Snapshot summary (mode={args.mode}) ===")
    print(f"  files:              {len(classified)}")
    print(f"  stubs:              {stubs}")
    print(f"  changelog omitted:  {n_changelog_omitted}")
    print(f"  §13.4 omitted:      {n_eps_omitted}")
    print(f"  input (approx):     {total_bytes_in / 1024:.1f} KiB")
    print(f"  output:             {size_out / 1024:.1f} KiB")
    if total_bytes_in > 0:
        print(f"  reduction:          "
              f"{100 * (1 - size_out / total_bytes_in):.0f}%")
    print(f"  written:            {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
