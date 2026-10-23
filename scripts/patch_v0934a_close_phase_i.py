#!/usr/bin/env python3
# v0.9.34-docs: close Phase I + fix algorithm count 24 -> 25.
# Docs/data only; no production code, no version bump.

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROADMAP = ROOT / "docs/roadmap.json"
BUG_BACKLOG = ROOT / "docs/BUG_BACKLOG.md"
ALGO_BACKLOG = ROOT / "docs/ALGORITHM_BACKLOG.md"
TASK_V0935 = ROOT / ".pingis/tasks/TASK_v0935_algorithms.md"
CONTEXT = ROOT / "CONTEXT.md"
TODAY = "2026-10-23"

def close_phase_i():
    data = json.loads(ROADMAP.read_text())
    phase = next(p for p in data["phases"] if p["id"] == "I")
    phase["status"] = "completed"
    phase["completed_at"] = TODAY
    phase["version_range"] = "v0.9.34"

    resolutions = {
        "I.1": ("done", "B1 already-fixed-at-baseline (v0.9.25c); repro PASS + bench"),
        "I.2": ("done", "B2 already-fixed-at-baseline (v0.9.25c); repro PASS + bench"),
        "I.3": ("rejected-with-measurement", "B3: Python pass2 loop faster than Arrow group_by; O(batch) memory"),
        "I.4": ("done", "B7 already-fixed-at-baseline (v0.9.18); repro PASS + bench"),
        "I.5": ("done", "A9: log_fit wired into linear/pca/all 3 NB fitters"),
        "I.6": ("done", "B4: _UNIQUE_FALLBACK_FACTOR=4 + justified WHITELIST entry"),
        "I.7": ("done", "A10: empty-input contract docstrings + tests"),
        "I.8": ("done", "B6: README version drift removed"),
    }
    for item in phase["items"]:
        if item["id"] in resolutions:
            item["status"], item["resolution"] = resolutions[item["id"]]

    data["updated"] = TODAY
    ROADMAP.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print("Phase I -> completed; 8 items resolved")

def append_bug_closure():
    text = BUG_BACKLOG.read_text()
    if "## Closure (v0.9.34)" in text:
        print("BUG_BACKLOG closure already present")
        return
    text = text.rstrip() + """

---

## Closure (v0.9.34)

| ID | Final status | Evidence |
|---|---|---|
| B1 | already-fixed-at-baseline | `6fa5e49` v0.9.25c; repro PASS + bench |
| B2 | already-fixed-at-baseline | `6fa5e49` v0.9.25c; repro PASS + bench |
| B3 | rejected-with-measurement | Python loop 0.026s@500k vs Arrow 0.104s; +K=N memory |
| B7 | already-fixed-at-baseline | v0.9.18; repro PASS + bench |
| A9 | fixed | `log_fit` wired into linear/pca/all 3 NB fitters |
| B4 | fixed | `_UNIQUE_FALLBACK_FACTOR=4` + WHITELIST justification |
| A10 | fixed | empty-input contract docstrings + tests |
| B6 | fixed | README version drift removed |

Artifacts: `.pingis/audits/v0934_bugfix_report.md`,
`.pingis/benches/v0934_*.json`, `.pingis/experiments/v0934_notes.md`.
"""
    BUG_BACKLOG.write_text(text + "\n")
    print("BUG_BACKLOG closure appended")

def fix_count(path, replacements):
    text = path.read_text()
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
    path.write_text(text)
    print(f"count fixed: {path}")

def fix_counts():
    common = [
        ("24 items", "25 items"),
        ("24-item", "25-item"),
        ("24 individual patches", "25 individual patches"),
        ("24 algorithms", "25 algorithms"),
        ("24 files", "25 files"),
        ("24×8", "25×8"),
        ("all 24", "all 25"),
        ("24 pending", "25 pending"),
        ("24 pending Phase D", "25 pending Phase D"),
    ]
    fix_count(ALGO_BACKLOG, common)
    fix_count(TASK_V0935, common)
    fix_count(CONTEXT, [
        ("Algorithm backlog: 24 items", "Algorithm backlog: 25 items"),
        ("24 pending Phase D", "25 pending Phase D"),
    ])

def main():
    print("=== v0.9.34-docs closeout ===")
    head = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True
    ).stdout.strip()
    print("HEAD:", head)

    close_phase_i()
    append_bug_closure()
    fix_counts()

    print("\n--- status ---")
    subprocess.run(["git", "status", "--short"], check=False)

if __name__ == "__main__":
    main()