#!/usr/bin/env python3
"""Leakage-hygiene gate for ML-tools YAML cards (v0.9.51.1).

Rules (only fire if v0.3 fields are populated):
  R1. model_class in (pretrained_ts, llm_based) without leakage block -> ERROR
  R2. leakage.pretraining_overlap_risk == 'high' -> ERROR
  R3. leakage.pretraining_overlap_risk == 'unknown' -> WARN
  R4. cadw_reproduced entry, evaluation_scope unset -> WARN
  R5. cadw_reproduced entry, evaluation_scope != 'synthetic' -> ERROR
  R6. supervised TS forecast/anomaly, temporal_split unset -> WARN
  R7. supervised TS forecast/anomaly, temporal_split == False -> ERROR

Usage:
  python scripts/audit_leakage.py <yaml> [<yaml> ...]

Exit codes: 0 (OK, warnings allowed) | 1 (errors present)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _is_section(sec) -> bool:
    return isinstance(sec, dict) and "section" in sec and "subtypes" in sec


def audit(yaml_paths: list[str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for ypath in yaml_paths:
        try:
            sec = yaml.safe_load(Path(ypath).read_text())
        except yaml.YAMLError as e:
            errors.append(f"[YAML_PARSE] {ypath}: {e}")
            continue
        if not _is_section(sec):
            continue
        sec_id = sec["section"]
        for st in sec.get("subtypes") or []:
            for m in st.get("models") or []:
                card_id = f"{sec_id}.{st['id']}.{m['name']}"
                _check_model(m, card_id, errors, warnings)
    return errors, warnings


def _check_model(m: dict, card_id: str, errors: list[str],
                 warnings: list[str]) -> None:
    model_class = m.get("model_class")
    learning_mode = m.get("learning_mode")
    task_type = m.get("task_type")
    evaluation_scope = m.get("evaluation_scope")
    leakage = m.get("leakage") or {}
    overlap = leakage.get("pretraining_overlap_risk")
    temporal = leakage.get("temporal_split")

    if model_class in ("pretrained_ts", "llm_based"):
        if not leakage or overlap is None:
            errors.append(
                f"[PRETRAINED_NO_LEAKAGE] {card_id}: "
                f"model_class={model_class!r} but "
                f"leakage.pretraining_overlap_risk is missing"
            )
        elif overlap == "high":
            errors.append(
                f"[LEAKAGE_HIGH] {card_id}: pretraining_overlap_risk='high' "
                f"-- exclude or explicitly document"
            )
        elif overlap == "unknown":
            warnings.append(
                f"[LEAKAGE_UNKNOWN] {card_id}: pretraining_overlap_risk="
                f"'unknown' -- document if resolved"
            )

    has_repro = any(
        br.get("attribution") == "cadw_reproduced"
        for br in (m.get("benchmark_results") or [])
    )
    if has_repro:
        if evaluation_scope is None:
            warnings.append(
                f"[REPRO_NO_SCOPE] {card_id}: cadw_reproduced without "
                f"evaluation_scope (recommend 'synthetic')"
            )
        elif evaluation_scope != "synthetic":
            errors.append(
                f"[REPRO_WRONG_SCOPE] {card_id}: cadw_reproduced requires "
                f"evaluation_scope='synthetic', got {evaluation_scope!r}"
            )

    is_sup_ts = (task_type in ("forecast", "anomaly")
                 and learning_mode == "supervised")
    if is_sup_ts:
        if temporal is False:
            errors.append(
                f"[TEMPORAL_SPLIT_FALSE] {card_id}: "
                f"task_type={task_type!r} + learning_mode='supervised' "
                f"but temporal_split=False"
            )
        elif temporal is None:
            warnings.append(
                f"[TEMPORAL_SPLIT_UNSET] {card_id}: supervised TS without "
                f"temporal_split (recommend True)"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit ML-tool YAML cards for leakage hygiene.",
    )
    ap.add_argument("yaml_files", nargs="+")
    args = ap.parse_args()

    errors, warnings = audit(args.yaml_files)
    for w in warnings:
        print(w)
    for e in errors:
        print(e)
    if errors:
        print(f"{len(errors)} error(s)")
        sys.exit(1)
    print(f"OK ({len(warnings)} warning(s))")


if __name__ == "__main__":
    main()

