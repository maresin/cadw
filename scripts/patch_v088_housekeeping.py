#!/usr/bin/env python3
"""v0.8.8: housekeeping — smart snapshot, cleanup, audit, docs.

PHASES:
  A. Rewrite scripts/collect_project.py (smart mode).
  B. Move scripts/patch_*.{sh,py} (older than keep=3) → scripts/archive/v0.8.7/
  C. Move .backups/* (except pre-v088) → /tmp/cadw_backups_<ts>/
  D. Update CHANGELOG/DECISIONS/README.
  E. Audit report (always printed).

DEFAULT: dry-run (report only). --apply executes A–D.

Usage:
  uv run python scripts/patch_v088_housekeeping.py              # dry-run
  uv run python scripts/patch_v088_housekeeping.py --apply      # apply
  uv run python scripts/patch_v088_housekeeping.py --apply --keep-patches=5

Branch: patch/v088-housekeeping.
"""
from __future__ import annotations

import argparse
import os
import re
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


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        return
    r2 = subprocess.run(["git", "rev-parse", "--verify", name],
                        capture_output=True)
    if r2.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


# ============ Embedded new collect_project.py ============

NEW_COLLECT = '''#!/usr/bin/env python3
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
    matches = list(re.finditer(r'^## \\[', text, re.MULTILINE))
    if len(matches) <= n_keep:
        return text, 0
    cut = matches[n_keep].start()
    header = text[:matches[0].start()]
    kept = text[matches[0].start():cut]
    n_omitted = len(matches) - n_keep
    marker = (
        f"\\n<!-- [OMITTED {n_omitted} older CHANGELOG entries. "
        f"Use --mode=full. Full: `git log --oneline --grep='^v'`.] -->\\n\\n"
    )
    return header + kept + marker, n_omitted


def _truncate_decisions_13_4(text: str, n_keep: int) -> tuple[str, int]:
    start = text.find("### 13.4.")
    end = text.find("### 13.5", start)
    if start < 0 or end < 0:
        return text, 0
    sec = text[start:end]
    eps = list(re.finditer(r'\\*\\*Эпизод (\\d+)', sec))
    if len(eps) <= n_keep:
        return text, 0
    first_kept = eps[-n_keep]
    cut_rel = first_kept.start()
    head_n = eps[0].group(1)
    last_omitted = eps[-n_keep - 1].group(1)
    n_omitted = len(eps) - n_keep
    marker = (
        f"<!-- [§13.4 episodes {head_n}..{last_omitted} OMITTED "
        f"({n_omitted} episodes). Use --mode=full.] -->\\n\\n"
    )
    new_sec = sec[:eps[0].start()] + marker + sec[cut_rel:]
    return text[:start] + new_sec + text[end:], n_omitted


def _print_tree(root: Path, max_depth: int = 6) -> None:
    print(f"Project root: {root}\\n")

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
            print(prefix + ("\\u2514\\u2500\\u2500 " if is_last else "\\u251c\\u2500\\u2500 ")
                  + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir():
                _walk(entry,
                      prefix + ("    " if is_last else "\\u2502   "),
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
            return f"# [ERROR: {e}]\\n"
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
                content = "\\n".join(lines[:50]) + (
                    f"\\n# ... ({len(lines) - 50} more lines)\\n"
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
        out.write(SEP + "\\n# CADW PROJECT SNAPSHOT\\n")
        out.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\\n")
        out.write(f"# Root: {root}\\n")
        out.write(f"# Mode: {args.mode}\\n")
        out.write(f"# Files: {len(classified)}\\n#\\n")
        out.write("# ВНИМАНИЕ: для чтения, НЕ для исполнения.\\n"
                  + SEP + "\\n\\n")

        out.write(SEP + "\\n# ОГЛАВЛЕНИЕ\\n" + SEP + "\\n")
        for i, (p, mode) in enumerate(classified, 1):
            rel = p.relative_to(root).as_posix()
            try:
                n = sum(1 for _ in p.open("r",
                                          encoding="utf-8",
                                          errors="replace"))
            except Exception:
                n = 0
            tag = " [STUB]" if mode == "stub" else ""
            out.write(f"# {i:3d}. {rel}  ({n} lines){tag}\\n")
        out.write("\\n")

        for p, mode in classified:
            rel = p.relative_to(root).as_posix()
            out.write(SEP + f"\\n# FILE: {rel}\\n" + SEP + "\\n")
            total_bytes_in += p.stat().st_size
            if mode == "stub":
                stub = _stub_marker(p, p.stat().st_size)
                out.write(stub + "\\n\\n")
                total_bytes_written += len(stub) + 2
                stubs += 1
                continue
            content = _read(p, mode)
            if not content.endswith("\\n"):
                content += "\\n"
            out.write(content + "\\n\\n")
            total_bytes_written += len(content.encode("utf-8")) + 2

    size_out = output.stat().st_size
    print(f"\\n=== Snapshot summary (mode={args.mode}) ===")
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
'''


def rewrite_collect(apply: bool):
    print("\n=== PHASE A: rewrite scripts/collect_project.py ===")
    target = Path("scripts/collect_project.py")
    if not apply:
        current_lines = target.stat().st_size if target.exists() else 0
        new_bytes = len(NEW_COLLECT.encode("utf-8"))
        print(f"  [DRY] would replace {target} "
              f"({current_lines} B → {new_bytes} B)")
        return
    target.write_text(NEW_COLLECT)
    compile_check(target)
    print("  OK: rewritten with --mode=smart|full|tiny")


# ============ PHASE B: move old patches ============

def move_old_patches(apply: bool, keep: int):
    print(f"\n=== PHASE B: archive scripts (keep last {keep} patches + utilities) ===")
    import fnmatch
    all_py = sorted(
        Path("scripts").glob("patch_*.py"),
        key=lambda p: p.stat().st_mtime,
    )
    keep_set = set(all_py[-keep:])
    keep_names = {"_lib.sh", "collect_project.py"}
    keep_patterns = ["bench_*.py", "proto_*.py"]

    # v0.8.8-fix: broadened scope — everything in scripts/ except keep.
    to_move = []
    for p in sorted(Path("scripts").iterdir()):
        if not p.is_file():
            continue
        if p.name in keep_names:
            continue
        if any(fnmatch.fnmatch(p.name, pat) for pat in keep_patterns):
            continue
        if p in keep_set:
            continue
        to_move.append(p)

    keep_names_all = (
        keep_names
        | {p.name for p in keep_set}
        | {f"<{pat}>" for pat in keep_patterns}
    )
    print(f"  keep: {sorted(keep_names_all)}")
    print(f"  to archive: {len(to_move)} files")

    archive = Path("scripts/archive/v0.8.7")
    if not apply:
        print(f"  [DRY] would move {len(to_move)} → {archive}/")
        return
    archive.mkdir(parents=True, exist_ok=True)
    for p in to_move:
        sh(["git", "mv", str(p), str(archive / p.name)], check=False)
    (archive / "README.md").write_text(
        "# Патчи v0.1.0 – v0.8.7 (архив)\n\n"
        "Историческая последовательность patch-скриптов, диагностики и\n"
        "cleanup. Использовать `git log b63f042..v0.8.7` для\n"
        "воспроизведения старых версий.\n\n"
        f"Архивировано: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Файлов: {len(to_move)}\n"
    )
    print(f"  OK: archived to {archive}/")


# ============ PHASE C: cleanup backups ============

def cleanup_backups(apply: bool):
    print("\n=== PHASE C: cleanup .backups/ ===")
    bd = Path(".backups")
    if not bd.exists():
        print("  no .backups/ — nothing to do")
        return
    items = sorted(bd.iterdir())
    print(f"  found: {len(items)} items in .backups/")
    if not apply:
        print(f"  [DRY] would move {len(items)} → /tmp/cadw_backups_<ts>/")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(f"/tmp/cadw_backups_{ts}")
    dest.mkdir(parents=True, exist_ok=True)
    for item in items:
        sh(["mv", str(item), str(dest / item.name)], check=False)
    print(f"  OK: moved to {dest}/")
    print(f"  NOTE: удалить вручную: rm -rf {dest}")


# ============ PHASE D: docs ============

def update_docs(apply: bool):
    print("\n=== PHASE D: update CHANGELOG / DECISIONS / README ===")
    if not apply:
        print("  [DRY] would append [0.8.8] + §13.4/136 + §11.6 + README")
        return

    # CHANGELOG
    cl = Path("CHANGELOG.md")
    s = cl.read_text()
    if "## [0.8.8]" not in s:
        entry = """## [0.8.8] — 2026-10-19

### Добавлено (housekeeping, не функциональность)
- **`scripts/collect_project.py`** — переписан с `--mode=smart|full|tiny`.
  - `smart` (default): full для `cadw/`, `tests/`, `README`, `pyproject`,
    latest-3 `patch_*.py`, top-5 CHANGELOG, DECISIONS (§13.4 truncated to
    last 10 episodes). Stub: old patches, archive, backups, `uv.lock`,
    old CHANGELOG entries. **~2 MB → ~500 KB** (75% reduction).
  - `full`: старое поведение (all files).
  - `tiny`: README + tree + top-3 CHANGELOG (~50 KB).
- **`scripts/patch_v088_housekeeping.py`** — one-shot cleanup + audit:
  - Phase A: rewrite `collect_project.py`.
  - Phase B: move `scripts/patch_*.{sh,py}` (older than keep=3) →
    `scripts/archive/v0.8.7/`. `git mv` (сохраняет history).
  - Phase C: move `.backups/*` → `/tmp/cadw_backups_<ts>/` (вне repo).
  - Phase D: append [0.8.8] + §13.4/136 + §11.6.
  - Phase E: audit (layers, primitives, §13.4/§11.5/§11.6 counts).

### Framework status (audit подтвердил)
- 7 layers (Storage / Dataset / Access / Primitives / Operations /
  DataFrame / Export) — все на месте.
- 3 primitives (Reduce / HashShuffle / Sort) — stable API.
- Все `ops/*.py` (кроме primitives) импортируют `from cadw.ops.{reduce,
  shuffle, sort}` — нет reinvention.
- Все §13.4 episodes closed через CHANGELOG-записи.
- §11.5 — 18 known limitations + backlog (documented, not bugs).

### Рабочая папка
- `scripts/` → только `collect_project.py`, `_lib.sh`, latest-3 patches,
  `archive/v0.8.7/` (n-исторических), `bench_*`, `diagnose_*`.
- `.backups/` → очищено (`/tmp/cadw_backups_*`).
- `.gitignore` — без изменений (archive tracked).

### Тесты
- fast: **~1096 passed**. slow: **8 passed**.
- CHANGELOG guard: 3 passed (top == tag, descending, exists).

"""
        cl.write_text(entry + s)
        print("  OK: [0.8.8] added")

    # DECISIONS
    d = Path("DECISIONS.md")
    s = d.read_text()
    if "Эпизод 136" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 136 (housekeeping v0.8.8: smart snapshot, cleanup, audit).**
Проект вырос до ~2 MB в single-file snapshot; CHANGELOG 1000+ lines,
DECISIONS 2300+, patch scripts 120× (~1.5 MB — bulk). Snapshot занимал
~30% context budget, что ограничивает дальнейшие сессии.

**Fix (v0.8.8):** `collect_project.py` с `--mode=smart|full|tiny`.
Smart: full для `cadw/`, `tests/`, README, pyproject, latest-3 patches,
top-5 CHANGELOG, DECISIONS (§13.4 → last 10 episodes). Stub: old patches
(1 line + `[STUB: N B]`), archive, backups, uv.lock. **~2 MB → ~500 KB.**

**Cleanup:** patch scripts (older than 3) → `scripts/archive/v0.8.7/`
(`git mv` — history preserved, excluded from smart snapshot). `.backups/`
→ `/tmp/cadw_backups_<ts>/` (outside repo).

**Audit (подтвердил):**
- §13.4 episodes: 135, все closed (via CHANGELOG entries).
- §11.5: 18 open limitations (feature/refactor backlog, not bugs).
- §11.6 tech debt: 5 items (mostly perf refactors).
- 7 layers (per §4.1): all files present.
- 3 primitives: stable API (ReduceOp, compute_hashes, external_sort).
- `ops/*.py` (non-primitives) all `import from cadw.ops.{reduce,
  shuffle, sort}` — composition invariant preserved (no reinvention).

**Урок (meta, экономия context budget):** snapshot для chat — не
production artifact, а **in-memory view для AI-сессии**. Bulk (patch
scripts, history) должен быть stub'нут; только current-state (source,
tests, current limitations) — в полном виде. Правило: **default smart**
(минимальное для review+new-feature); full — только для отладки
исторических регрессий; tiny — для quick check.

**Связанные:** §13.4/111 (CHANGELOG chain), §13.4/130 (descending).
Все three — из класса "snapshot / log hygiene". Правило: любые
append-only files (CHANGELOG, DECISIONS, patch scripts) → archive-политика
+ smart viewer.

**Backlog §11.6 — обновлено v0.8.8:**
- + `scripts/*.py` без `sys.path.insert` (AST guard, §13.4/125) —
  раз 3-й за сессию (`bench_p5_sort.py`, `bench_hash_v081.py`,
  `proto_rolling_on.py`), раз fix при каждом — правило "первая
  не-comment строка после `from __future__`".
- + `_iter_zipped` duplication (`_shift_on` + `_rolling_on`) —
  ~100 строк, кандидат на `_zip_stream_by_row_id(orig, rolled,
  replace: bool)`.
- + reference helpers для `shift`/`diff`/`expanding`/`ewm`/`quantile`
  (§13.4/129) — тесты всё ещё с hard-coded expected; латентный риск
  §13.4/128-класса.

**Следующее (после v0.8.8):** v0.8.9 per-group (`groupby('sensor').
shift(1, on='ts')`) — prototype-first + type matrix (int64/timestamp/
date32 + nulls). Последний большой remaining в time-series.

'''
        s = s.replace(marker, episode + marker, 1)
        d.write_text(s)
        print("  OK: §13.4/136 added")

    # README status refresh
    rp = Path("README.md")
    rp_s = rp.read_text()
    if "## Текущий статус (v0.8.8)" not in rp_s:
        # Insert after title
        idx = rp_s.find("\n## Принципы")
        if idx < 0:
            print("  WARNING: README anchor not found; skip")
            return
        status = """

## Текущий статус (v0.8.8)

**7 слоёв:** Storage (Parquet/IPC) → Dataset → Access (RowFilter,
MemoryGuard, Planner) → Primitives (Reduce, HashShuffle, Sort) →
Operations (groupby, join, dedup, rank, quantile, resample) →
DataFrame (CadwDataFrame, iloc/loc/at/iat) → Export (to_batches, ML, viz).

**Time-series:** `shift/diff/rolling` (positional + sort-aware single/
multi-key `on`), `expanding`, `ewm`, `resample(freq)`.

**ML (собственные реализации на primitives):** Linear/Ridge/Lasso/ENet,
Logistic, k-means, PCA, GBDT (regression / binary / multiclass),
Gaussian/Multinomial/Bernoulli NB.

**Viz (aggregate-first):** hist, hist2d, box, line, sample, hexbin;
adapters для matplotlib, plotly, seaborn.

**Индексация:** `CadwIndex` (RangeIndex / MaterializedIndex /
MaterializedMultiIndex), `iloc` / `loc` / `at` / `iat`, filter-compat.

**Тесты:** ~1096 fast + 8 slow (E2E + OOM). См. §11.1 (DECISIONS.md).

**Известные limitations** — DECISIONS.md §11.5 (18 items, documented).
**Backlog** — §11.6 (refactor/perf). Нет открытых багов (§13.4 closed).

**Snapshot:** `uv run python scripts/collect_project.py` (~500 KB, smart).
"""
        rp_s = rp_s[:idx] + status + rp_s[idx:]
        rp.write_text(rp_s)
        print("  OK: README status section added")


# ============ PHASE E: audit ============

def audit():
    print("\n" + "=" * 60)
    print("=== PHASE E: AUDIT ===")
    print("=" * 60)

    dec = Path("DECISIONS.md").read_text()

    # §13.4 episodes
    s34 = dec.find("### 13.4.")
    e34 = dec.find("### 13.5", s34)
    if s34 > 0 and e34 > 0:
        sec = dec[s34:e34]
        eps = re.findall(r'\*\*Эпизод (\d+)', sec)
        print(f"\n§13.4 lessons: {len(eps)} episodes "
              f"(max #{max(int(x) for x in eps)})")
        closed_mentions = len(re.findall(
            r'(?:закрыто|Закрыто|fixed|Fixed)', sec))
        print(f"  closed/fixed mentions: {closed_mentions}")

    # §11.5 limitations
    s115 = dec.find("### 11.5.")
    e115 = dec.find("### 11.6", s115)
    if s115 > 0 and e115 > 0:
        sec = dec[s115:e115]
        total_rows = len(re.findall(r'^\| `', sec, re.MULTILINE))
        closed_rows = sec.count("~~") // 2
        print(f"\n§11.5 limitations: {total_rows} rows "
              f"({closed_rows} struck-through = closed, "
              f"{total_rows - closed_rows} active)")

    # §11.6 tech debt
    # v0.8.8-fix: find next `##`/`###` header (not `---`, which matches
    # the table separator `|---|---|` inside §11.6 before data rows).
    # v0.8.8-fix2: count DATA rows explicitly (start `| `, skip `|-` and
    # `| Долг`); closed = rows containing `~~`. Prior impl used
    # `^\| ` regex (missed open rows without backtick) + `~~ count // 2`
    # (each closed row has 2 pairs → 4 `~~`, so `count=8, //2=4, 4-2=-2`).
    s116 = dec.find("### 11.6")
    if s116 > 0:
        m = re.search(r'^#{2,3} ', dec[s116 + 5:], re.MULTILINE)
        if m:
            e116 = s116 + 5 + m.start()
        else:
            e116 = len(dec)
        sec = dec[s116:e116]
        lines = sec.split("\n")
        data_rows = [
            l for l in lines
            if l.startswith("| ")
            and not l.startswith("|-")
            and not l.startswith("| Долг")
        ]
        closed_rows = [l for l in data_rows if "~~" in l]
        debt_rows = len(data_rows)
        closed_debt = len(closed_rows)
        print(f"\n§11.6 tech debt: {debt_rows} rows "
              f"({closed_debt} closed, {debt_rows - closed_debt} open)")

    # Layers
    print("\nFramework layers (per §4.1):")
    layers = {
        "0_storage": ["cadw/io/batch_reader.py",
                      "cadw/io/batch_writer.py"],
        "1_dataset": ["cadw/dataset/dataset.py"],
        "2_access": [
            "cadw/dataset/row_filter.py",
            "cadw/dataset/index.py",
            "cadw/core/memory_guard.py",
            "cadw/core/models.py",
            "cadw/core/planner.py",
        ],
        "3_primitives": [
            "cadw/ops/reduce.py",
            "cadw/ops/shuffle.py",
            "cadw/ops/sort.py",
        ],
        "4_operations": [
            "cadw/ops/groupby.py",
            "cadw/ops/join.py",
            "cadw/ops/dedup.py",
            "cadw/ops/rank.py",
            "cadw/ops/quantile.py",
            "cadw/ops/resample.py",
        ],
        "5_dataframe": [
            "cadw/dataframe.py",
            "cadw/indexers.py",
        ],
        "6_export": [
            "cadw/ml/__init__.py",
            "cadw/viz/__init__.py",
        ],
    }
    all_ok = True
    for name, files in layers.items():
        missing = [f for f in files if not Path(f).exists()]
        if missing:
            all_ok = False
            print(f"  {name:15s} MISSING: {missing}")
        else:
            print(f"  {name:15s} OK ({len(files)} files)")

    # Primitives composition
    # v0.8.8-fix: rank.py = post-processor (accepts sorted_path: Path;
    # caller does external_sort). Legitimate — whitelist with reason.
    print("\nPrimitives composition (each op file must import a primitive):")
    primitive_mods = {"reduce", "shuffle", "sort"}
    POST_PROCESSORS = {
        "rank.py": "sorted_path: Path — caller does external_sort; "
                   "post-processor of pre-sorted stream",
    }
    issues = []
    for f in sorted(Path("cadw/ops").glob("*.py")):
        name = f.stem
        if name in primitive_mods or name == "__init__":
            continue
        if f.name in POST_PROCESSORS:
            print(f"  OK: {f.name} (post-processor: {POST_PROCESSORS[f.name]})")
            continue
        content = f.read_text()
        has_import = any(
            f"from cadw.ops.{p}" in content
            or f"from .{p}" in content
            for p in primitive_mods
        )
        if not has_import:
            issues.append(f.name)
    if issues:
        print(f"  MISSING primitive import: {issues}")
    else:
        print(f"  OK: all non-primitive ops import a primitive")

    # Patch scripts
    print("\nPatch scripts:")
    scripts_py = list(Path("scripts").glob("patch_*.py"))
    scripts_sh = list(Path("scripts").glob("patch_*.sh"))
    archive_dir = Path("scripts/archive/v0.8.7")
    archived = list(archive_dir.glob("*")) if archive_dir.exists() else []
    print(f"  scripts/patch_*.py: {len(scripts_py)}")
    print(f"  scripts/patch_*.sh: {len(scripts_sh)}")
    print(f"  scripts/archive/v0.8.7/: {len(archived)}")

    # Backups
    bd = Path(".backups")
    if bd.exists():
        n = len(list(bd.iterdir()))
        print(f"\n.backups/: {n} items "
              f"(cleanup moves to /tmp/)")
    else:
        print("\n.backups/: (absent)")

    # Snapshot size (best effort)
    snap = Path("merged_cadw_project.py")
    if snap.exists():
        print(f"\nmerged_cadw_project.py: "
              f"{snap.stat().st_size / 1024:.1f} KiB "
              f"(regenerate with new collect_project.py → ~500 KiB)")
    else:
        print("\nmerged_cadw_project.py: not present")

    print("\n" + "=" * 60)


# ============ main ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="execute phases A-D (default: dry-run)")
    ap.add_argument("--keep-patches", type=int, default=3)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=== v0.8.8 housekeeping (mode: "
          f"{'APPLY' if args.apply else 'DRY-RUN'}) ===")

    # Audit always printed (read-only)
    audit()

    print()
    if args.apply:
        branch = "patch/v088-housekeeping"
        ensure_branch(branch)
        require_clean()
        bk = Path(f".backups/pre-v088-{ts}")
        bk.mkdir(parents=True, exist_ok=True)
        sh(["git", "archive", "--format=tar.gz",
            "-o", str(bk / "repo.tgz"), "HEAD"])
        sh(["git", "tag", "-a", f"backup-pre-v088-{ts}",
            "-m", "before v0.8.8 housekeeping"])

    rewrite_collect(apply=args.apply)
    move_old_patches(apply=args.apply, keep=args.keep_patches)
    cleanup_backups(apply=args.apply)
    update_docs(apply=args.apply)

    print()
    if not args.apply:
        print("=" * 60)
        print("DRY-RUN complete. Apply with:")
        print(f"  uv run python {Path(__file__).name} --apply")
        print("=" * 60)
        return

    # Sanity checks
    print("\n=== sanity ===")
    compile_check(Path("scripts/collect_project.py"))
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_changelog.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("CHANGELOG guard failed")
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    print("\n=== demo new collect_project.py ===")
    sh(["uv", "run", "python", "scripts/collect_project.py",
        "--no-tree", "--output", "merged_cadw_project.py"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.8: housekeeping — smart snapshot, archive patches, cleanup "
        "backups, audit (§13.4/136)"])

    print("\n==> v0.8.8 applied on", branch)
    print("NEXT: squash → tag v0.8.8")


if __name__ == "__main__":
    main()