#!/usr/bin/env python3
"""v0.8.8-fix: repair housekeeping audit + cleanup scope + rank.py note.

DRY-RUN FINDINGS:
1. §11.6 tech debt = 0 rows (BUG in audit): `dec.find("---")` hits the
   `|---|---|` separator INSIDE §11.6's table header BEFORE data rows.
   Fix: find next `###`/`##` header via regex.

2. `rank.py` MISSING primitive import (EXPECTED exception):
   `compute_ranks_to_disk(sorted_path: Path, ...)` accepts pre-sorted
   path; caller (`CadwDataFrame.rank`) does external_sort. Post-processor
   pattern. Fix: whitelist + docstring note in rank.py.

3. Cleanup missed 4 non-`patch_*` files (diagnose_*.sh ×2,
   cleanup_v010.sh, fix_other_leak.sh). Fix: broadened scope — everything
   in scripts/ except keep-list (_lib.sh, collect_project.py, bench_*.py,
   proto_*.py, latest-N patch_*.py by mtime).

4. `merged_cadw_project.py: not present` — deleted at some point. No
   fix needed (gitignored, regenerated after --apply).

Run: uv run python scripts/patch_v088_fix.py
"""
from __future__ import annotations

import os
import re
import shutil
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


# ============ Fix 1: audit §11.6 regex ============

def fix_audit_116():
    print("\n=== patch_v088_housekeeping.py: audit §11.6 ===")
    p = Path("scripts/patch_v088_housekeeping.py")
    s = p.read_text()

    old = '''    # §11.6 tech debt
    s116 = dec.find("### 11.6.")
    if s116 > 0:
        e116 = dec.find("---", s116)
        if e116 < 0:
            e116 = len(dec)
        sec = dec[s116:e116]
        debt_rows = len(re.findall(r'^\\| `', sec, re.MULTILINE))
        closed_debt = sec.count("~~") // 2
        print(f"\\n§11.6 tech debt: {debt_rows} rows "
              f"({closed_debt} closed, {debt_rows - closed_debt} open)")'''

    new = '''    # §11.6 tech debt
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

    s = replace_line_block(s, old, new, tag="audit-116")
    p.write_text(s)
    print("  OK: §11.6 next-header regex")


# ============ Fix 2: rank.py whitelist in audit ============

def fix_audit_rank_whitelist():
    print("\n=== patch_v088_housekeeping.py: audit rank.py whitelist ===")
    p = Path("scripts/patch_v088_housekeeping.py")
    s = p.read_text()

    old = '''    # Primitives composition
    print("\\nPrimitives composition (each op file must import a primitive):")
    primitive_mods = {"reduce", "shuffle", "sort"}
    issues = []
    for f in sorted(Path("cadw/ops").glob("*.py")):
        name = f.stem
        if name in primitive_mods or name == "__init__":
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
        print(f"  OK: all non-primitive ops import a primitive")'''

    new = '''    # Primitives composition
    # v0.8.8-fix: rank.py = post-processor (accepts sorted_path: Path;
    # caller does external_sort). Legitimate — whitelist with reason.
    print("\\nPrimitives composition (each op file must import a primitive):")
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
        print(f"  OK: all non-primitive ops import a primitive")'''

    s = replace_line_block(s, old, new, tag="audit-rank-whitelist")
    p.write_text(s)
    print("  OK: rank.py whitelist")


# ============ Fix 3: broadened cleanup scope ============

def fix_move_old_patches():
    print("\n=== patch_v088_housekeeping.py: move_old_patches broad scope ===")
    p = Path("scripts/patch_v088_housekeeping.py")
    s = p.read_text()

    old = '''def move_old_patches(apply: bool, keep: int):
    print(f"\\n=== PHASE B: archive old patch scripts (keep last {keep}) ===")
    all_py = sorted(Path("scripts").glob("patch_*.py"), key=lambda p: p.name)
    all_sh = sorted(Path("scripts").glob("patch_*.sh"), key=lambda p: p.name)
    keep_set = set(all_py[-keep:])
    to_move = [p for p in all_py if p not in keep_set] + all_sh
    print(f"  found: {len(all_py)} .py + {len(all_sh)} .sh")
    print(f"  keep in scripts/: {[p.name for p in all_py[-keep:]]}")
    print(f"  to archive: {len(to_move)} files")

    archive = Path("scripts/archive/v0.8.7")
    if not apply:
        print(f"  [DRY] would move {len(to_move)} → {archive}/")
        return
    archive.mkdir(parents=True, exist_ok=True)
    for p in to_move:
        sh(["git", "mv", str(p), str(archive / p.name)], check=False)
    # Write README for archive
    (archive / "README.md").write_text(
        "# Патчи v0.1.0 – v0.8.7 (архив)\\n\\n"
        "Историческая последовательность patch-скриптов. Использовать\\n"
        "для воспроизведения старых версий — `git log b63f042..v0.8.7`\\n"
        "предпочтительнее.\\n\\n"
        f"Архивировано: {datetime.now().isoformat(timespec='seconds')}\\n"
        f"Файлов: {len(to_move)}\\n"
    )
    print(f"  OK: archived to {archive}/")'''

    new = '''def move_old_patches(apply: bool, keep: int):
    print(f"\\n=== PHASE B: archive scripts (keep last {keep} patches + utilities) ===")
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
        "# Патчи v0.1.0 – v0.8.7 (архив)\\n\\n"
        "Историческая последовательность patch-скриптов, диагностики и\\n"
        "cleanup. Использовать `git log b63f042..v0.8.7` для\\n"
        "воспроизведения старых версий.\\n\\n"
        f"Архивировано: {datetime.now().isoformat(timespec='seconds')}\\n"
        f"Файлов: {len(to_move)}\\n"
    )
    print(f"  OK: archived to {archive}/")'''

    s = replace_line_block(s, old, new, tag="move-patches")
    p.write_text(s)
    print("  OK: broadened scope (mtime-sorted patches + non-patch .sh/.py)")


# ============ Fix 4: rank.py docstring note ============

def fix_rank_docstring():
    print("\n=== cadw/ops/rank.py: post-processor note ===")
    p = Path("cadw/ops/rank.py")
    s = p.read_text()

    old = '''"""
Rank: external_sort + потоковое вычисление rank.

Алгоритм:'''

    new = '''"""
Rank: post-processor of pre-sorted stream (v0.8.8 note).

`compute_ranks_to_disk(sorted_path: Path, ...)` accepts an ALREADY-sorted
IPC path; caller (`CadwDataFrame.rank` in dataframe.py) does
`external_sort`. Architectural pattern "post-processor": this module
does not import primitives. See §13.4/137.

Алгоритм (caller-side external_sort показан для контекста):'''

    s = replace_line_block(s, old, new, tag="rank-docstring")
    p.write_text(s)
    compile_check(p)
    print("  OK: docstring note")


# ============ Fix 5: DECISIONS §13.4/137 ============

def add_decisions_137():
    print("\n=== DECISIONS.md: §13.4/137 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 137" in s:
        print("  already present — skip")
        return
    marker = "### 13.5. Открытые вопросы"
    episode = '''**Эпизод 137 (rank.py: post-processor pattern).** Аудит v0.8.8
проверяет, что каждый `cadw/ops/*.py` (кроме primitives) импортирует
primitive (`reduce`/`shuffle`/`sort`) — invariant «high-level ops =
compositions of primitives». `rank.py` — исключение:
`compute_ranks_to_disk(sorted_path: Path, ...)` принимает УЖЕ
отсортированный IPC-path; caller (`CadwDataFrame.rank`) делает
`external_sort` до вызова.

**Pattern:** `rank.py` — **post-processor** of pre-sorted stream, не
composite over unsorted input. Legitimate architectural exception:
- Sort (expensive, 2 passes I/O) — в caller (facade, координатор).
- Ranking (cheap, 1 pass I/O) — в `rank.py` (post-processor primitive).

Split обоснован: `CadwDataFrame.rank` делает `external_sort` по `by`,
затем `external_sort` по `row_id` (sort-back). Между ними —
`compute_ranks_to_disk`. Вынести `external_sort` внутрь `rank.py` =
дублировать sort-back, нарушит "post-processor не сортирует" инвариант.

**Аудит fix:** whitelist `POST_PROCESSORS = {"rank.py": ...}` в
`patch_v088_housekeeping.py::audit` + docstring note в `rank.py`.
Для сравнения `quantile.py` **импортирует** `external_sort` — потому
что делает sort внутри (2-pass: sort + scan). Разница: `rank.py`
вызывается **после** sort, `quantile.py` сортирует сам.

**Урок:** composition invariant «ops = primitives» — не «каждый файл
импортирует primitive», а «op выполняется через primitives, где бы они
ни вызывались (caller или local)». `rank.py` — legitimate exception с
явной архитектурной причиной. Whitelist + docstring лучше, чем
force-import в post-processor (последнее = dead import).

**Связанные:** §13.4/133 (pyarrow cast matrix — code vs tests),
§13.4/134 (input type inference), §13.4/135 (parquet limitation),
§13.4/136 (housekeeping). Класс «hidden by green tests» /
«assumptions not verified»: (133/134/135) test-authoring, (137)
audit-assumption too strict.

'''
    s = s.replace(marker, episode + marker, 1)
    p.write_text(s)
    print("  OK: §13.4/137")


# ============ main ============

def main():
    print("=== v0.8.8-fix: repair housekeeping ===")

    housekeeping = Path("scripts/patch_v088_housekeeping.py")
    if not housekeeping.exists():
        print(f"ERROR: {housekeeping} not found. Did you create it?")
        sys.exit(1)

    # Backup WIP housekeeping script before editing
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = Path(f"/tmp/cadw_wip_v088_{ts}.py")
    shutil.copy(housekeeping, backup)
    print(f"  backup: {backup}")

    fix_audit_116()
    fix_audit_rank_whitelist()
    fix_move_old_patches()
    fix_rank_docstring()
    add_decisions_137()

    compile_check(housekeeping)

    print("\n=== re-run dry-run to verify fixes ===")
    r = subprocess.run(
        ["uv", "run", "python", "scripts/patch_v088_housekeeping.py"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** dry-run FAILED ***")
        print(f"Restore: cp {backup} scripts/patch_v088_housekeeping.py")
        sys.exit(1)

    print("\n=== pytest rank (regression from docstring edit) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_rank.py", "-q"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** rank test regression ***")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("==> v0.8.8-fix complete.")
    print("=" * 60)
    print()
    print("NEXT:")
    print(f"  1. Check dry-run output above (should show §11.6 >0 rows,")
    print(f"     'OK: rank.py (post-processor)', archive ~130 files).")
    print(f"  2. Apply: uv run python scripts/patch_v088_housekeeping.py --apply")
    print(f"  3. Backup (WIP): {backup}")


if __name__ == "__main__":
    main()