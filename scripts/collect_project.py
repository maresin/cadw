#!/usr/bin/env python3
# scripts/collect_project.py — снапшот проекта CADW в один файл.
# Использование:
#   uv run python scripts/collect_project.py --tree
#   uv run python scripts/collect_project.py --output snapshot_v0.1.0.py
from __future__ import annotations
import argparse, os, sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

EXCLUDE_DIRS = {
    ".venv", "venv", "env", ".env",
    "__pycache__", ".git", ".github", ".idea", ".vscode", ".vscode-oss",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".basedpyright",
    "node_modules", "dist", "build", "htmlcov",
    "cadw_cache", "cadw_shuffle", ".ipynb_checkpoints", ".OTHER", ".backups",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".pyd", ".arrow", ".parquet"}
EXTRA_ROOT_FILES = ["pyproject.toml", "uv.lock", "README.md", "DECISIONS.md", "CHANGELOG.md"]
SEP = "# " + "=" * 76

def is_excluded_dir(name: str, rel: str = "") -> bool:
    if name in EXCLUDE_DIRS: return True
    if name.endswith(".egg-info"): return True
    if rel.startswith("scripts/archive/"): return True
    return False

def walk_project(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = str(Path(dirpath).relative_to(root)) if dirpath != str(root) else ""
        dirnames[:] = sorted(d for d in dirnames if not is_excluded_dir(d, os.path.join(rel_dir, d)))
        for fname in sorted(filenames):
            p = Path(dirpath) / fname
            if p.suffix in EXCLUDE_SUFFIXES: continue
            yield p

def print_tree(root: Path, max_depth: int = 10) -> None:
    print(f"Project root: {root}\n")
    def _walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth: return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return
        entries = [e for e in entries if not (e.is_dir() and is_excluded_dir(e.name, str(e.relative_to(root))))]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            print(prefix + ("└── " if is_last else "├── ") + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir():
                _walk(entry, prefix + ("    " if is_last else "│   "), depth + 1)
    _walk(root, "", 0)

def collect_files(root: Path, output: Path) -> list[Path]:
    files: list[Path] = []
    for p in walk_project(root):
        if p.resolve() == output.resolve(): continue
        if p.suffix == ".py":
            if p.parent.name == "scripts" and p.name != "collect_project.py": continue
            files.append(p)
        elif p.suffix == ".sh" and p.parent.name == "scripts":
            files.append(p)
        elif p.name in EXTRA_ROOT_FILES and p.parent == root:
            files.append(p)
    return files

def build_merged(root: Path, output: Path, files: list[Path]) -> tuple[int, int]:
    total, written = 0, 0
    with open(output, "w", encoding="utf-8") as out:
        out.write(SEP + "\n# CADW PROJECT SNAPSHOT\n")
        out.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        out.write(f"# Root: {root}\n# Files: {len(files)}\n#\n")
        out.write("# ВНИМАНИЕ: для чтения, НЕ для исполнения.\n" + SEP + "\n\n")
        out.write(SEP + "\n# ОГЛАВЛЕНИЕ\n" + SEP + "\n")
        for i, f in enumerate(files, 1):
            rel = f.relative_to(root)
            try: n = sum(1 for _ in open(f, "r", encoding="utf-8", errors="replace"))
            except Exception: n = 0
            out.write(f"# {i:3d}. {rel}  ({n} lines)\n")
        out.write("\n")
        for f in files:
            rel = f.relative_to(root)
            out.write(SEP + f"\n# FILE: {rel}\n" + SEP + "\n")
            try:
                content = open(f, "r", encoding="utf-8", errors="replace").read()
            except Exception as e:
                out.write(f"# [ERROR: {e}]\n\n"); continue
            if f.name == "uv.lock":
                lines = content.splitlines()
                if len(lines) > 100:
                    content = "\n".join(lines[:100]) + f"\n# ... ({len(lines) - 100} more lines truncated)\n"
            if not content.endswith("\n"): content += "\n"
            out.write(content + "\n\n")
            total += content.count("\n"); written += 1
    return written, total

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", default="merged_cadw_project.py")
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--no-tree", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: {root} not found", file=sys.stderr); return 1
    if args.tree: print_tree(root); return 0
    if not args.no_tree: print_tree(root); print()
    output = Path(args.output)
    if not output.is_absolute(): output = root / output
    files = collect_files(root, output)
    if not files:
        print("WARNING: no source files", file=sys.stderr); return 2
    n_files, n_lines = build_merged(root, output, files)
    print(f"Собрано файлов: {n_files}\nВсего строк: {n_lines}\n"
          f"Размер: {output.stat().st_size / 1024:.1f} KiB\nВыход: {output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
