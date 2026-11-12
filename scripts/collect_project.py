#!/usr/bin/env python3
"""Snapshot project to single file. --mode=smart|full|tiny.

SMART MODE (default, ~500 KB vs 2 MB full):
  Full content: cadw/**/*.py, tests/**/*.py, README.md, pyproject.toml,
    scripts/collect_project.py, .pingis/*.md, latest 3 patch_*.py,
    last-5 CHANGELOG entries, DECISIONS.md (§13.4 truncated to last 10
    episodes), and ALL 11 AI-first methodology files (T1-T4).

TINY MODE (v0.9.224, recommended for session bootstrap):
  13 AI-first docs in full (AI_CONSTITUTION, ROADMAP, STABILITY_PROTOCOL,
    EXPERIMENT_PROCEDURE, ALGORITHM_TEMPLATE, GLOSSARY, DECISION_LOG,
    CHAT_PROTOCOL, EXAMPLES_INDEX, PRIMITIVES_AND_PRINCIPLES,
    EXAMPLE_VERIFIED_CARD, METHODOLOGY_ALGORITHM_SELECTION,
    PRIMITIVES_BACKLOG) in reading order, then CHANGELOG index, then
    skeletonized cadw/**/*.py + 7 cadwboost key modules + tests/*_verification.py
    + tests/test_cadwboost_*.py (bodies omitted, signatures visible).
  Missing AI-first file -> exit 3 (broken tree, do not silently skip).
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
    # v0.9.126-fix2: .pingis/ is temporary (agent scratch; planned removal).
    # Not in snapshot for new sessions.
    ".pingis",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".pyd",
                    ".arrow", ".parquet", ".tar", ".gz"}
ROOT_FILES_FULL = ["README.md", "pyproject.toml", "METHODOLOGY.md"]

# v0.9.93-prep: AI-first files (always full in snapshot, any mode).
# Order in snapshot: AI_CONSTITUTION.md → docs/ROADMAP.md →
# docs/ml_tools/EXAMPLES_INDEX.md → PRIMITIVES_AND_PRINCIPLES.md.
# Purpose: new AI session sees these before any code stub.
# If file is absent on disk, snapshot silently skips it and prints
# a WARNING to stderr (see main()).
# v0.9.102: 11 AI-first files in T1-T4 reading order (AI_CONSTITUTION §4.5).
# Tuple, not set: order = reading order in snapshot.
AI_FIRST_FILES = (
    "AI_CONSTITUTION.md",                                   # T1
    "docs/ROADMAP.md",
    "docs/ml_tools/STABILITY_PROTOCOL.md",                  # T2
    "docs/ml_tools/EXPERIMENT_PROCEDURE.md",                # T2 (v0.9.109d)
    "docs/ml_tools/ALGORITHM_TEMPLATE.md",                  # T3
    "docs/ml_tools/GLOSSARY.md",                            # T4
    "docs/ml_tools/DECISION_LOG.md",                        # T4
    "docs/ml_tools/CHAT_PROTOCOL.md",                       # T5
    "docs/ml_tools/EXAMPLES_INDEX.md",                      # T4
    "docs/ml_tools/PRIMITIVES_AND_PRINCIPLES.md",           # T4
    "docs/ml_tools/EXAMPLE_VERIFIED_CARD.md",               # T3
    "docs/ml_tools/METHODOLOGY_ALGORITHM_SELECTION.md",     # T2
    # v0.9.143: PRIMITIVES_BACKLOG (plan of work: 20 P1 / 15 P2 /
    # 20 P3 clusters + coverage map). Every card's `depends_on` cross-
    # refs this. AI session must see "what's next" before code.
    "docs/ml_tools/PRIMITIVES_BACKLOG.md",                  # T4 (plan)
)
SEP = "# " + "=" * 76

# v0.9.87-phase0a-fix1: tiny-mode context upgrade.
# Ensures future sessions see primitives + principles even in tiny
# mode. Full contents for small/foundational files; skeletons for
# large files whose signatures carry the essential patterns.
TINY_FULL = {
    "README.md",
    "docs/ml_tools/METHODOLOGY_ALGORITHM_SELECTION.md",
    "docs/ml_tools/PRIMITIVES_AND_PRINCIPLES.md",
    "cadw/io/batch_writer.py",
    "cadw/io/batch_reader.py",
    "cadw/core/models.py",
    "cadw/core/models_reduce.py",
    # v0.9.110f: component-wise verification templates (DEC-033).
    # Canonical code/test examples for next-session copy to gmm/birch/
    # kmeans_ad/sub_pca/eif. ~407 lines, +20 KB. Without these, AI has
    # only YAML Example 2 (EXAMPLE_VERIFIED_CARD) but not the actual
    # test structure (parametrized state check, median-of-3 formula,
    # bounded-not-identical for n>=S).
    "tests/test_linucb_verification.py",
    "tests/test_iforest_verification.py",
    # v0.9.119: canonical verification templates (DEC-033 base + 5-CW final).
    # These are procedural references; update only on template change.
    "tests/test_coxph_verification.py",     # CoxPH-precedent (v0.9.114)
    "tests/test_chains_verification.py",    # 5-CW final template (v0.9.118)
    # v0.9.224: CadwBoost CW template (state/OOC/guard pattern for
    # promoted prototype; canonical copy-paste for future promotes).
    "tests/test_cadwboost_cw.py",
}
TINY_SKELETON = {
    "cadw/core/memory_guard.py",
    "cadw/core/models_sort.py",
    "cadw/core/models_shuffle.py",
    "cadw/ops/shuffle.py",
    "cadw/ops/groupby.py",
    # v0.9.110f: reproducible evidence chains (EXPERIMENT_PROCEDURE §3.7).
    # Skeleton reveals predicted_peak() formula signatures, subprocess
    # architecture, prewarm/malloc_trim/fadvise pattern. Bodies omitted
    # (309 + 351 lines -> ~25 lines skeleton each).
    "scripts/experiments/experiment_v110_linucb.py",
    "scripts/experiments/experiment_v110_iforest.py",
    # v0.9.119: canonical reference patterns (skeletonizes to ~20-40 lines each).
    # Update only on procedural change (new pattern class).
    "cadw/ml/coxph.py",     # CoxPH-precedent: state_nbytes + bounded_by_domain
    "cadw/ml/lda.py",       # auto K-spill + batch_size pass-through (DEC-042)
    "cadw/ml/chains.py",    # 5-CW state sum (v0.9.118)
    # v0.9.143: schema.py — visible fields (ModelCard dataclass,
    # R-rules, VALID_STATE_CLASS). Essential for YAML edits in session.
    "docs/ml_tools/schema.py",
    # v0.9.224: CadwBoost key modules (promoted v0.9.223). Skeleton
    # shows class signatures + state_nbytes + fit_df/predict_df API;
    # bodies omitted. Remaining 11 modules stubbed (less critical).
    "cadw/ml/cadwboost/model.py",       # CADWBoost class + fit_df + state_nbytes
    "cadw/ml/cadwboost/ooc.py",         # fit_ooc TRUE OOC pattern
    "cadw/ml/cadwboost/losses.py",      # 9 losses (quantile bug ref)
    "cadw/ml/cadwboost/dataframe.py",   # fit_df / predict_df / predict_proba_df
    "cadw/ml/cadwboost/tree.py",        # symmetric + leaf_wise builders
    "cadw/ml/cadwboost/efb.py",         # EFB (Exclusive Feature Bundling)
    "cadw/ml/cadwboost/multiclass.py",  # CADWBoostMulticlass
}


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


import re as _re

_SKEL_KEEP_RE = _re.compile(
    r'^(@|async\s+def\s|def\s|class\s)'
)


def _skeletonize(text: str) -> str:
    """Keep imports, class/def/decorator signatures, # v0.9 comments.

    Skips function bodies. Purpose: give context reader enough to see
    what API exists without loading 800 lines of implementation.
    """
    lines = text.splitlines(keepends=True)
    keep: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Version-annotated comments (# v0.9...)
        if stripped.startswith("# v0.9"):
            keep.append((i, line))
            continue
        # Module-level imports only (no leading whitespace)
        if not line[0].isspace():
            if stripped.startswith(("import ", "from ")) and (
                " import" in stripped
            ):
                keep.append((i, line))
                continue
        # Class / function / decorator lines at any indent
        if _SKEL_KEEP_RE.match(stripped):
            keep.append((i, line))
            continue
    if not keep:
        return text
    out: list[str] = []
    prev_i = -2
    for i, line in keep:
        if i > prev_i + 1 and out:
            out.append("    # ... (body omitted)\n")
        out.append(line if line.endswith("\n") else line + "\n")
        prev_i = i
    return "".join(out)


def _generate_changelog_index(
    text: str, max_summary_chars: int = 70,
) -> str:
    """One line per version entry. Replaces full CHANGELOG in tiny mode."""
    out = ["# CHANGELOG index (one line per version, newest first)\n",
           "# Full entries in --mode=full / --mode=smart.\n\n"]
    pattern = _re.compile(
        r"^### (v[\d.]+[^\s]*)\s*\((\d{4}-\d{2}-\d{2})\)\s*-\s*(.+)$",
        _re.MULTILINE,
    )
    n = 0
    for m in pattern.finditer(text):
        ver = m.group(1)
        date = m.group(2)
        summary = m.group(3).strip()
        if len(summary) > max_summary_chars:
            summary = summary[:max_summary_chars - 3] + "..."
        out.append(f"{ver:20s} ({date}) {summary}\n")
        n += 1
    out.append(f"\n# ({n} versions total)\n")
    return "".join(out)


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


def _latest_session_brief(root: Path) -> Path | None:
    """Latest SESSION_BRIEF_v*.md by version sort. None if absent.

    v0.9.119: session briefs are supplied separately (user policy:
    universal script, not edited per version). If present on disk,
    auto-included in snapshot as full content. If absent — silent
    skip (no exit 3; brief is not in AI_FIRST_FILES required-set).
    """
    docs = root / "docs"
    if not docs.exists():
        return None
    candidates = list(docs.glob("SESSION_BRIEF_v*.md"))
    if not candidates:
        return None

    def _vk(p: Path) -> tuple[int, int, int]:
        m = _re.search(r"_v(\d+)\.(\d+)\.(\d+)\.md$", p.name)
        if not m:
            return (0, 0, 0)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    return max(candidates, key=_vk)


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
        rel = p.relative_to(root).as_posix()
        # v0.9.93-prep: AI-first files always full, in all modes.
        if rel in AI_FIRST_FILES:
            mode = "full"
        elif args.mode == "full":
            mode = "full"
            if p.name == "CHANGELOG.md":
                mode = "changelog"
            elif p.name == "DECISIONS.md":
                mode = "decisions"
        elif args.mode == "tiny":
            # v0.9.87-phase0a-fix1: context-rich tiny mode.
            if rel in TINY_FULL:
                mode = "full"
            elif rel in TINY_SKELETON:
                mode = "skeleton"
            elif rel == "CHANGELOG.md":
                mode = "changelog_index"
            # v0.9.119: all verification tests skeletonized in tiny.
            # Shows test names + imports + parametrize; bodies omitted.
            # Universal — no per-file list, works for future verifications.
            elif ((rel.startswith("tests/test_")
                   and rel.endswith("_verification.py"))
                  or rel.startswith("tests/test_cadwboost_")):
                mode = "skeleton"
            else:
                mode = "stub"
        else:
            mode = _classify(p, root, args.keep_patches)
        classified.append((p, mode))

    if not classified:
        print("WARNING: no files collected", file=sys.stderr)
        return 2

    # v0.9.102: AI-first files in TUPLE ORDER (reading order).
    _ai_order = {name: i for i, name in enumerate(AI_FIRST_FILES)}
    _ai_first = []
    _others = []
    for pm in classified:
        rel = pm[0].relative_to(root).as_posix()
        if rel in _ai_order:
            _ai_first.append(pm)
        else:
            _others.append(pm)
    _ai_first.sort(key=lambda pm: _ai_order[pm[0].relative_to(root).as_posix()])
    if _ai_first:
        classified = _ai_first + _others

    # v0.9.119: latest SESSION_BRIEF — optional, auto-included if present.
    # Not in AI_FIRST_FILES (no exit-3 failure if missing; supplied separately).
    _brief = _latest_session_brief(root)
    if _brief is not None:
        _brief_rel = _brief.relative_to(root).as_posix()
        _present_rels = {pm[0].relative_to(root).as_posix() for pm in classified}
        if _brief_rel not in _present_rels:
            classified.append((_brief, "full"))
            print(f"  + SESSION_BRIEF: {_brief_rel} (auto-included)")

    # Warn if expected AI-first files are missing on disk (snapshot
    # would silently omit them — check that they were actually created).
    _present = {
        pm[0].relative_to(root).as_posix() for pm in classified}
    _missing = set(AI_FIRST_FILES) - _present
    if _missing:
        print(
            f"ERROR: {len(_missing)} AI-first files missing on disk:\n"
            f"  {sorted(_missing)}\n"
            f"Snapshot will skip them; AI next session cannot follow "
            f"the methodology. Fix file presence before regenerating.",
            file=sys.stderr)
        # Non-zero exit: broken snapshot must not look successful.
        # Override with env CADW_SNAPSHOT_ALLOW_MISSING=1 if you know
        # what you're doing (e.g. half-built tree during development).
        if os.environ.get("CADW_SNAPSHOT_ALLOW_MISSING") != "1":
            return 3

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
        elif mode == "skeleton":
            content = _skeletonize(content)
        elif mode == "changelog_index":
            content = _generate_changelog_index(content)
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
        out.write("# ВНИМАНИЕ: для чтения, НЕ для исполнения.\n")
        out.write("#\n")
        out.write("# READ IN THIS ORDER (T1 first; see AI_CONSTITUTION §4.5):\n")
        out.write("#  1. AI_CONSTITUTION.md          -- rules (AP1-AP23, gates G1-G11)\n")
        out.write("#  2. docs/ROADMAP.md             -- current state\n")
        out.write("#  3. STABILITY_PROTOCOL.md       -- formula method (code->formula->measure)\n")
        out.write("#  4. EXPERIMENT_PROCEDURE.md     -- 6-step DOE + rules R1-R4 + exit criterion\n")
        out.write("#  5. ALGORITHM_TEMPLATE.md       -- 8-step checklist\n")
        out.write("#  6. GLOSSARY.md                 -- canonical terms\n")
        out.write("#  7. DECISION_LOG.md             -- why X not Y\n")
        out.write("#  8. CHAT_PROTOCOL.md            -- session rhythm\n")
        out.write("#  9. EXAMPLES_INDEX.md           -- pattern -> reference\n")
        out.write("# 10. PRIMITIVES_AND_PRINCIPLES.md -- primitives cheat sheet\n")
        out.write("# 11. EXAMPLE_VERIFIED_CARD.md    -- what 'verified' looks like\n")
        out.write("# 12. METHODOLOGY_ALGORITHM_SELECTION.md -- deep method\n")
        out.write("#\n")
        out.write("# SESSION START CHECKLIST:\n")
        out.write("#  [ ] Read EXPERIMENT_PROCEDURE before any card verification\n")
        out.write("#  [ ] Read in order above (do not skip)\n")
        out.write("#  [ ] If session brief contradicts AI_CONSTITUTION -> ask, do not proceed\n")
        out.write("#  [ ] First action: probe (read-only), not patch\n")
        out.write("#  [ ] Check DECISION_LOG.md for prior decisions on your topic\n")
        out.write("#  [ ] Know current status of your algorithm (grep YAML card)\n")
        out.write(SEP + "\n\n")

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




# ============================================================================
# v0.9.25c: phase-aware / module-aware modes (B.4)
# ============================================================================
#
# Runs AFTER main() above — new entry points that reuse the same file
# collection primitives but with a manifest-driven filter.
#
#   --module cadw/ts/models.py
#       collect module + matching tests. Default excludes core, interfaces.
#       Use --with-interfaces to inline cadw/dataframe.py signatures.
#   --phase A
#       reads docs/roadmap.json, collects all phase-A items' files
#       (best-effort glob expansion; warns on unresolvable composites).
#   --doc docs/ROADMAP.md
#       just that doc.
#   --list-phases / --list-modules
#       debug helpers.
#
# Output: ts_review_<mode>.md by default; override with --out.

import json as _json
import re as _re
import subprocess
import sys as _sys

_DOCS_ROADMAP = Path("docs/roadmap.json")


def _split_top_commas(s: str) -> list[str]:
    """Split on commas at brace-depth 0.

    v0.9.25c-fix4: prior _parse_file_spec did spec.split(",") before
    brace-expansion. For 'cadw/ts/{a,b,c}.py' that produced
    ['cadw/ts/{a', 'b', 'c}.py'] — none resolve → silent drop.
    This helper preserves {...} blocks as single units (up to depth).
    """
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for c in s:
        if c == "{":
            depth += 1
            buf.append(c)
        elif c == "}":
            depth = max(0, depth - 1)
            buf.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    if buf:
        parts.append("".join(buf))
    return parts


def _parse_file_spec(spec: str) -> list[Path]:
    """Parse file paths from a roadmap.json 'file' field.

    Supported:
      - 'cadw/ts/models.py'                 → [Path]
      - 'cadw/{a,b,c}.py'                    → brace-expansion (one level)
      - 'cadw/dataframe.py, cadw/ops/x.py'  → comma-separated top-level
      - 'cadw/*.py'                          → glob
      - 'tests/'                             → directory (rglob *.py)
    Unknown / unresolvable → dropped silently (caller aggregates
    warnings if ALL parts of a spec fail).

    v0.9.25c-fix4: comma-split now respects braces via _split_top_commas.
    Previously commas inside '{a,b,c}' split the brace block.
    """
    if not spec:
        return []
    out: list[Path] = []
    for part in _split_top_commas(spec):
        part = part.strip()
        if not part:
            continue
        if "{" in part and "}" in part:
            prefix, rest = part.split("{", 1)
            options, suffix = rest.split("}", 1)
            for opt in options.split(","):
                for sub in _parse_file_spec(
                    prefix + opt.strip() + suffix
                ):
                    out.append(sub)
        elif "*" in part:
            out.extend(Path(".").glob(part))
        elif part.endswith("/"):
            d = Path(part)
            if d.is_dir():
                out.extend(sorted(d.rglob("*.py")))
        else:
            p = Path(part)
            if p.exists() and p.is_file():
                out.append(p)
    return out


def _find_tests_for_module(mod_path: Path) -> list[Path]:
    """Heuristic: tests/test_<parent>_<stem>*.py + tests/test_<stem>*.py."""
    tests_dir = Path("tests")
    if not tests_dir.exists():
        return []
    stem = mod_path.stem
    parent = mod_path.parent.name
    out: list[Path] = []
    if parent and parent != "tests":
        out.extend(sorted(tests_dir.glob(f"test_{parent}_{stem}*.py")))
    out.extend(sorted(tests_dir.glob(f"test_{stem}*.py")))
    # dedupe
    seen = set()
    unique = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _load_roadmap() -> dict:
    if not _DOCS_ROADMAP.exists():
        print(f"ERROR: {_DOCS_ROADMAP} missing; run v0.9.25b first",
              file=_sys.stderr)
        raise SystemExit(2)
    return _json.loads(_DOCS_ROADMAP.read_text())


def _list_phases() -> None:
    data = _load_roadmap()
    print(f"Roadmap v{data['current_version']}, "
          f"schema={data['schema_version']}")
    for phase in data["phases"]:
        n_done = sum(1 for i in phase["items"]
                     if i["status"] == "done")
        n_total = len(phase["items"])
        print(f"  {phase['id']:2s} {phase['name']:30s} "
              f"{n_done}/{n_total} done  ({phase['status']})")


def _extract_phase_files(phase_id: str) -> tuple[list[Path], list[str]]:
    """Return (files, warnings) for a phase."""
    data = _load_roadmap()
    phase = next(
        (p for p in data["phases"] if p["id"] == phase_id), None,
    )
    if phase is None:
        raise SystemExit(
            f"phase {phase_id!r} not in roadmap; "
            f"available: {[p['id'] for p in data['phases']]}"
        )
    files: set[Path] = set()
    warnings: list[str] = []
    for item in phase["items"]:
        if item["status"] == "done":
            continue  # already shipped, not current work
        spec = item.get("file", "")
        if not spec:
            continue
        parsed = _parse_file_spec(spec)
        if not parsed and spec:
            warnings.append(
                f"item {item['id']}: could not resolve {spec!r} — "
                f"manual include"
            )
            continue
        for p in parsed:
            files.add(p)
    return sorted(files), warnings


def _write_snapshot(
    out_path: Path,
    mode_label: str,
    full_files: list[Path],
    test_files: list[Path],
    notes: list[str],
) -> None:
    SEP = "# " + "=" * 76
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(SEP + "\n")
        f.write(f"# CADW SNAPSHOT ({mode_label})\n")
        from datetime import datetime
        f.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        )
        f.write(f"# HEAD: {r.stdout.strip()}\n")
        f.write(f"# Full files: {len(full_files)}\n")
        f.write(f"# Test files: {len(test_files)}\n")
        for n in notes:
            f.write(f"# NOTE: {n}\n")
        f.write(SEP + "\n\n")
        for p in full_files + test_files:
            f.write(SEP + f"\n# FILE: {p}\n" + SEP + "\n")
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                f.write(f"# ERROR: {e}\n\n")
                continue
            if not content.endswith("\n"):
                content += "\n"
            f.write(content + "\n")
    print(f"  wrote {out_path} "
          f"({out_path.stat().st_size / 1024:.1f} KiB)")


def _run_module_mode(args) -> int:
    mod = Path(args.module)
    if not mod.exists():
        raise SystemExit(f"module {mod} does not exist")
    tests = _find_tests_for_module(mod)
    notes = []
    if not tests:
        notes.append(f"no matching tests for {mod.stem}")
    out = Path(args.out or f"snapshot_module_{mod.stem}.md")
    _write_snapshot(out, f"module={mod}", [mod], tests, notes)
    return 0


def _run_phase_mode(args) -> int:
    files, warnings = _extract_phase_files(args.phase)
    if not files:
        print(f"ERROR: phase {args.phase!r} resolved to 0 files",
              file=_sys.stderr)
        for w in warnings:
            print(f"  warning: {w}", file=_sys.stderr)
        return 1
    tests: list[Path] = []
    for f in files:
        if f.parts[0] == "cadw":
            tests.extend(_find_tests_for_module(f))
    tests = sorted(set(tests))
    out = Path(args.out or f"snapshot_phase_{args.phase}.md")
    _write_snapshot(
        out, f"phase={args.phase}", files, tests, warnings,
    )
    return 0


def _run_doc_mode(args) -> int:
    doc = Path(args.doc)
    if not doc.exists():
        raise SystemExit(f"doc {doc} does not exist")
    out = Path(args.out or f"snapshot_doc_{doc.stem}.md")
    _write_snapshot(out, f"doc={doc}", [doc], [], [])
    return 0


# v0.9.25c entry point — recognized by scripts/patch_v0925c invocation.
def _phase_aware_main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="collect_project.py (v0.9.25c phase-aware)",
    )
    ap.add_argument("--module", help="collect one module + tests")
    ap.add_argument("--phase", help="collect phase files from roadmap.json")
    ap.add_argument("--doc", help="collect one doc")
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.list_phases:
        _list_phases()
        return 0
    if args.module:
        return _run_module_mode(args)
    if args.phase:
        return _run_phase_mode(args)
    if args.doc:
        return _run_doc_mode(args)
    print(
        "ERROR: use --module / --phase / --doc / --list-phases. "
        "Full snapshot mode is --mode=smart (see --help).",
        file=_sys.stderr,
    )
    return 2


if __name__ == "__main__":
    if any(a in _sys.argv[1:] for a in (
        "--module", "--phase", "--doc", "--list-phases",
    )):
        raise SystemExit(_phase_aware_main(_sys.argv[1:]))
    raise SystemExit(main())
