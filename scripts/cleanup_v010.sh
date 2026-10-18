#!/usr/bin/env bash
# scripts/cleanup_v010.sh — очистка после v0.1.0:
#   A. .gitignore  — убрать tracked-файлы из ignore, добавить runtime caches
#   B. pyproject   — dev-инструменты в [dependency-groups]
#   C. scripts/    — архив 25 патчей в scripts/archive/v0.1.0/
#   D. collect_project.py — создать в scripts/
#   E. .backups/   — удалить (уже в git history)
#   F. tags        — на всякий случай удалить backup-pre-* (если появятся)
# Использование:
#   ./scripts/cleanup_v010.sh           # dry-run
#   ./scripts/cleanup_v010.sh --apply
set -euo pipefail

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "=== Pre-flight ==="
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: рабочее дерево грязное. Закоммитьте или stash."
  git status --short
  exit 1
fi
echo "OK: дерево чистое, ветка $(git branch --show-current)"

run() {
  if [[ $APPLY -eq 1 ]]; then echo "  [+] $*"; eval "$@"; else echo "  [dry] $*"; fi
}

# ---------- A. .gitignore ----------
echo
echo "=== A. .gitignore ==="
if [[ $APPLY -eq 1 ]]; then
  cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# Test & lint caches
.pytest_cache/
.ruff_cache/
.mypy_cache/
.basedpyright/
htmlcov/
.coverage

# CADW runtime
cadw_cache/
cadw_shuffle/

# Backups & analysis (local only)
.backups/
merged_*.py
snapshot_*.py

# IDE / OS
.vscode/
.idea/
*.swp
.DS_Store
EOF
  echo "  [+] .gitignore rewritten"
else
  echo "  [dry] rewrite .gitignore (убрать pyproject.toml/uv.lock/.git/.gitignore/.python-version/main.py)"
fi

# ---------- B. pyproject.toml ----------
echo
echo "=== B. pyproject.toml: dev-инструменты в [dependency-groups] ==="
if [[ $APPLY -eq 1 ]]; then
  python3 - <<'PY'
import re
from pathlib import Path

p = Path("pyproject.toml")
s = p.read_text()

# 1. Убрать dev-инструменты из [project].dependencies
dev_tools = [
    "pandas-stubs>=3.0.0.260204",
    "pylint>=4.0.5",
    "pyright>=1.1.409",
    "pytest>=9.0.3",
    "ruff>=0.15.12",
]
for dep in dev_tools:
    # запятая + опциональный пробел + dep + опциональный пробел + \n
    s = re.sub(r'\n\s*"' + re.escape(dep) + r'",', '', s)

# 2. Убедиться, что [dependency-groups] dev содержит все dev-инструменты
dev_deps = [
    "scikit-learn>=1.4",
    "lightgbm>=4.0",
    "pytest>=9.0.3",
    "pylint>=4.0.5",
    "pyright>=1.1.409",
    "ruff>=0.15.12",
    "pandas-stubs>=3.0.0.260204",
]

if "[dependency-groups]" not in s:
    s = s.rstrip() + "\n\n[dependency-groups]\n"

# Найти блок dev и заменить целиком
m = re.search(
    r"(\[dependency-groups\]\s*\ndev\s*=\s*\[)(.*?)(\])",
    s,
    re.DOTALL,
)
if m:
    body = "\n" + "".join(f'    "{d}",\n' for d in dev_deps)
    s = s[:m.start()] + m.group(1) + body + m.group(3) + s[m.end():]
else:
    # Нет dev — создать
    s = s.replace(
        "[dependency-groups]\n",
        "[dependency-groups]\ndev = [\n" + "".join(f'    "{d}",\n' for d in dev_deps) + "]\n",
        1,
    )

p.write_text(s)
print("pyproject.toml: dev-инструменты перенесены в [dependency-groups]")
PY
else
  echo "  [dry] перенести pytest/pylint/pyright/ruff/pandas-stubs в [dependency-groups] dev"
fi

# ---------- E (сначала E, чтобы lock был актуален) ----------
echo
echo "=== E. .backups/ ==="
if [[ -d .backups ]]; then
  run "du -sh .backups"
  run "rm -rf .backups"
else
  echo "  (нет .backups/)"
fi

# ---------- C. archive patch scripts ----------
echo
echo "=== C. scripts/patch_*.sh → scripts/archive/v0.1.0/ ==="
mkdir -p scripts/archive/v0.1.0 2>/dev/null || true

patches=( scripts/patch_*.sh )
if [[ ${#patches[@]} -gt 0 ]] && [[ -e "${patches[0]}" ]]; then
  echo "  найдено: ${#patches[@]} скриптов"
  for f in "${patches[@]}"; do
    [[ -e "$f" ]] || continue
    if [[ $APPLY -eq 1 ]]; then
      git mv "$f" "scripts/archive/v0.1.0/$(basename "$f")"
    else
      echo "  [dry] git mv $f → scripts/archive/v0.1.0/"
    fi
  done
else
  echo "  (нет patch_*.sh в scripts/)"
fi

# README архива
if [[ $APPLY -eq 1 ]] && [[ ! -f scripts/archive/v0.1.0/README.md ]]; then
  cat > scripts/archive/v0.1.0/README.md <<'EOF'
# Патчи v0.1.0 (архив)

Последовательность `patch_*.sh`, применённая для построения v0.1.0
из baseline `b63f042`. Каждый патч атомарный: `require_clean_tree`,
`git archive` бэкап, изменения кода (python heredoc), тесты,
`uv lock && uv sync && pytest`, коммит.

**Для воспроизведения v0.1.0** предпочтительнее:
    git log b63f042..v0.1.0

Скрипты сохранены как документация «как это делалось». Контекст и уроки —
DECISIONS.md §13.4 (24 эпизода отладки, 4 класса: API-контракты,
семантические баги в тестах, реальные баги в коде, архитектурные).

**Не использовать для v0.2+** — они несовместимы с post-v0.1.0.
Для новых патчей — `scripts/_lib.sh` + свежий `patch_NN.sh` в `scripts/`.
EOF
  echo "  [+] scripts/archive/v0.1.0/README.md"
fi

# ---------- F. backup tags (если есть) ----------
echo
echo "=== F. backup-pre-* tags ==="
tags="$(git tag -l 'backup-pre-*' || true)"
if [[ -z "$tags" ]]; then
  echo "  (нет backup-тегов — уже чисто)"
else
  echo "  удалить:"
  echo "$tags" | sed 's/^/    /'
  for t in $tags; do
    run "git tag -d '$t'"
  done
fi

# ---------- D. collect_project.py ----------
echo
echo "=== D. scripts/collect_project.py ==="
if [[ -f scripts/collect_project.py ]]; then
  echo "  (уже есть — пропуск; перезапишите вручную, если нужно)"
else
  if [[ $APPLY -eq 1 ]]; then
    cp /dev/stdin scripts/collect_project.py <<'PYEOF'
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
PYEOF
    chmod +x scripts/collect_project.py
    echo "  [+] scripts/collect_project.py создан"
  else
    echo "  [dry] создать scripts/collect_project.py"
  fi
fi

# ---------- Verify markers ----------
echo
echo "=== Verify [tool.pytest.ini_options] ==="
if grep -q '\[tool.pytest.ini_options\]' pyproject.toml; then
  echo "  OK: [tool.pytest.ini_options] присутствует"
else
  echo "  WARN: [tool.pytest.ini_options] отсутствует. slow-маркер даст warning."
  echo "  (не блокирует; добавим позже, если будет нужно)"
fi

# ---------- uv lock + sync ----------
echo
echo "=== uv lock && uv sync ==="
if [[ $APPLY -eq 1 ]]; then
  uv lock
  uv sync
else
  echo "  [dry] uv lock && uv sync (после pyproject)"
fi

# ---------- Tests ----------
echo
echo "=== pytest ==="
if [[ $APPLY -eq 1 ]]; then
  set +e
  uv run pytest -q -m "not slow"
  RC_FAST=$?
  uv run pytest -q -m slow
  RC_SLOW=$?
  set -e
  if [[ $RC_FAST -ne 0 ]] || [[ $RC_SLOW -ne 0 ]]; then
    echo "ERROR: tests failed (fast=$RC_FAST slow=$RC_SLOW)"; exit 1
  fi
else
  echo "  [dry] uv run pytest -q -m 'not slow' && uv run pytest -q -m slow"
fi

# ---------- Commit ----------
echo
echo "=== Commit ==="
if [[ $APPLY -eq 1 ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    git add -A
    git commit -m "chore: cleanup after v0.1.0 — .gitignore, pyproject dev-group, archive patches, collect_project.py"
    echo "  [+] committed"
  else
    echo "  (nothing to commit)"
  fi
fi

echo
echo "=== Done (mode: $([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)) ==="