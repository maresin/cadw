"""v0.9.102: migrate verified YAML cards to have `status:` field.

Rules (priority order):
  1. `tier1_core_exception:` field present        -> tier1_core_exception
  2. `n_scalability: refuse_at_guard`             -> draft
  3. `# BACKLOG:` marker in module.py::function   -> draft
  4. function listed as open backlog in
     .pingis/plan_v0.9.87_refactor.md              -> draft
  5. else                                          -> verified

Insert `status: <value>` line inside implementation_status, on the
line immediately after `verified_in_cadw: true`.

Usage:
  python scripts/migrate_status_v102.py --dry-run
  python scripts/migrate_status_v102.py --apply
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


YAML_DIR = Path("docs/ml_tools")
PLAN_FILE = Path(".pingis/plan_v0.9.87_refactor.md")


def _load_plan_backlog() -> set[str]:
    """Extract functions listed as open `- [ ] ... .py::function` in plan."""
    if not PLAN_FILE.exists():
        return set()
    results = set()
    for line in PLAN_FILE.read_text().split("\n"):
        if not re.match(r"\s*- \[ \]", line):
            continue
        for m in re.finditer(r"\b\w+\.py::(\w+)\b", line):
            results.add(m.group(1))
    return results


def _has_backlog_marker(cadw_module: str, cadw_function: str) -> bool:
    """Check for literal `# BACKLOG:` marker inside function body."""
    if not cadw_module or not cadw_function:
        return False
    parts = cadw_module.split(".")
    if len(parts) < 2 or parts[0] != "cadw":
        return False
    file_path = Path("/".join(parts[:-1])) / f"{parts[-1]}.py"
    if not file_path.exists():
        return False
    src = file_path.read_text()
    m = re.search(rf"^def\s+{re.escape(cadw_function)}\b", src, re.M)
    if not m:
        return False
    tail = src[m.start():]
    next_sym = re.search(r"^(def |class )", tail[1:], re.M)
    body = tail[: next_sym.start() + 1] if next_sym else tail
    return "# BACKLOG:" in body


def _detect(card: dict, plan_backlog: set[str]) -> tuple[str, str]:
    if card.get("tier1_core_exception"):
        return "tier1_core_exception", "tier1_core_exception field"
    if card.get("n_scalability") == "refuse_at_guard":
        return "draft", "n_scalability: refuse_at_guard"
    mod = card.get("cadw_module") or ""
    fn = card.get("cadw_function") or ""
    if _has_backlog_marker(mod, fn):
        return "draft", "# BACKLOG: marker in docstring"
    if fn and fn in plan_backlog:
        return "draft", "backlog in plan_v0.9.87"
    return "verified", "default"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.error("must specify --dry-run or --apply")

    plan_backlog = _load_plan_backlog()
    yaml_files = sorted(
        p for p in YAML_DIR.glob("*.yaml") if p.name != "benchmarks.yaml"
    )

    results: dict[str, list[dict]] = {
        "verified": [], "tier1_core_exception": [], "draft": [],
    }
    total = 0

    for yp in yaml_files:
        data = yaml.safe_load(yp.read_text())
        if not isinstance(data, dict):
            continue
        for st in data.get("subtypes", []):
            for card in st.get("models", []):
                impl = card.get("implementation_status") or {}
                if not impl.get("verified_in_cadw"):
                    continue
                total += 1
                status, reason = _detect(card, plan_backlog)
                results[status].append({
                    "file": yp.name,
                    "name": card.get("name"),
                    "function": card.get("cadw_function"),
                    "module": card.get("cadw_module"),
                    "reason": reason,
                    "transient_factor": card.get("transient_factor", 1.0),
                })

    print(f"\n{'='*78}")
    print(f"Migration summary: {total} verified cards in {len(yaml_files)} files")
    print(f"Plan backlog entries (open): {len(plan_backlog)} -> {sorted(plan_backlog)}")
    print(f"{'='*78}")

    for status in ("verified", "tier1_core_exception", "draft"):
        items = results[status]
        print(f"\n## {status}: {len(items)} cards")
        for it in items:
            extra = ""
            if status == "verified":
                tf = it["transient_factor"]
                if tf > 1.0:
                    extra = f"  [WARN tf={tf} — suspicious, may need measured override]"
            fn = (it["function"] or it["name"] or "<unknown>")
            print(f"  {fn:<42} {it['file']:<25} {it['reason']}{extra}")

    if not args.apply:
        print("\n[DRY RUN] No files modified. Use --apply to write.")
        return

    # Apply — line-based insertion, preserves all comments
    written = 0
    for yp in yaml_files:
        src = yp.read_text()
        data = yaml.safe_load(src)
        if not isinstance(data, dict):
            continue

        status_by_name: dict[str, str] = {}
        for st in data.get("subtypes", []):
            for card in st.get("models", []):
                impl = card.get("implementation_status") or {}
                if not impl.get("verified_in_cadw"):
                    continue
                if "status" in impl:
                    continue
                status, _ = _detect(card, plan_backlog)
                status_by_name[card.get("name", "")] = status

        if not status_by_name:
            continue

        out_lines = []
        current_name = None
        for line in src.split("\n"):
            m = re.match(r"^(\s*)- name:\s*(.+?)\s*$", line)
            if m:
                current_name = m.group(2).strip().strip("'\"")
                out_lines.append(line)
                continue

            m = re.match(r"^(\s*)verified_in_cadw:\s*true\s*$", line)
            if m and current_name in status_by_name:
                out_lines.append(line)
                indent = m.group(1)
                out_lines.append(f"{indent}status: {status_by_name[current_name]}")
                written += 1
                continue

            out_lines.append(line)

        new_src = "\n".join(out_lines)
        if new_src != src:
            yp.write_text(new_src)

    print(f"\n[APPLY] Wrote status field in {written} cards")

    # Verify
    has_status = 0
    missing = []
    for yp in yaml_files:
        data = yaml.safe_load(yp.read_text())
        if not isinstance(data, dict):
            continue
        for st in data.get("subtypes", []):
            for card in st.get("models", []):
                impl = card.get("implementation_status") or {}
                if not impl.get("verified_in_cadw"):
                    continue
                if "status" in impl:
                    has_status += 1
                else:
                    missing.append(f"{yp.name}::{card.get('name')}")

    print(f"[VERIFY] {has_status}/{total} cards now have status")
    if missing:
        print(f"[VERIFY] MISSING ({len(missing)}): {missing[:5]}")


if __name__ == "__main__":
    main()
