#!/usr/bin/env python3
"""Audit url_log.jsonl against YAML cards (v0.9.47b).

Every BenchmarkResult with value != null must have a fetched URL in
url_log.jsonl (status=200, matching card+url+metric). Otherwise it's a
hallucination.

Usage:
  python scripts/audit_url_log.py <url_log.jsonl> <yaml> [<yaml> ...]

Exit codes:
  0 — OK (warnings allowed)
  1 — errors: hallucination / missing artifact / empty note
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# v0.9.78c: benchmarks whose leaderboards are client-side rendered (HF
# Spaces, dynamic SaaS) and CANNOT be fetched as static HTML with
# numbers. Cards using these benchmarks are still marked
# attribution=literature (number comes from the benchmark/paper, not
# invented), but cannot have a url_log 200 entry for the leaderboard
# URL. Requirement for these: benchmark_url and algorithm_source_url
# must look like http(s); note must be non-empty (schema-level check).
EXTERNAL_ONLY_BENCHMARKS = frozenset({
    "fev_bench",   # AutoGluon fev: dynamic dashboard
    "GIFT_Eval",   # Salesforce: HuggingFace Space, client-side
    "MTEB",        # HuggingFace Space
    "STS-B",       # GLUE leaderboard dynamic
})


def load_log(path: Path) -> dict[str, list[dict]]:
    """Return dict[for_card] = [entry, ...] from JSON-lines log."""
    entries: dict[str, list[dict]] = {}
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            card = e.get("for_card")
            if not card:
                continue
            entries.setdefault(card, []).append(e)
    return entries


def _card_id(section: str, subtype_id: str, model_name: str) -> str:
    return f"{section}.{subtype_id}.{model_name}"


def _check_literature(br: dict, card_id: str, log: dict,
                       errors: list[str]) -> None:
    hits = log.get(card_id, [])
    bench_url = br.get("benchmark_url", "")
    algo_url = br.get("algorithm_source_url", "")
    metric = br.get("metric")
    bench_name = br.get("benchmark", "")

    # v0.9.78c: external-only benchmarks (fev_bench, GIFT_Eval, MTEB,
    # STS-B) have client-side rendered leaderboards — cannot be fetched
    # as static HTML with numbers. Skip url_log requirement; require
    # only structurally valid URLs (schema already enforces non-empty
    # note).
    if bench_name in EXTERNAL_ONLY_BENCHMARKS:
        if not (bench_url.startswith("http://")
                or bench_url.startswith("https://")):
            errors.append(
                f"[EXTERNAL_BAD_URL] {card_id}: benchmark_url="
                f"{bench_url!r} not http(s)"
            )
        if not (algo_url.startswith("http://")
                or algo_url.startswith("https://")):
            errors.append(
                f"[EXTERNAL_BAD_URL] {card_id}: algorithm_source_url="
                f"{algo_url!r} not http(s)"
            )
        return

    ok_bench = any(
        e.get("url") == bench_url
        and e.get("status") == 200
        and e.get("metric") == metric
        for e in hits
    )
    if not ok_bench:
        errors.append(
            f"[HALLUCINATION] {card_id}: benchmark_url={bench_url!r} "
            f"metric={metric!r} — no url_log 200 entry"
        )

    ok_algo = any(
        e.get("url") == algo_url and e.get("status") == 200
        for e in hits
    )
    if not ok_algo:
        errors.append(
            f"[HALLUCINATION] {card_id}: algorithm_source_url={algo_url!r} "
            f"— no url_log 200 entry"
        )


def _check_cadw_artifact(
    br: dict, card_id: str, errors: list[str],
    root: Path, need_verdict: bool,
) -> None:
    bench_path = root / br.get("benchmark_url", "")
    algo_path = root / br.get("algorithm_source_url", "")

    if not bench_path.exists():
        errors.append(
            f"[CADW_ARTIFACT_MISSING] {card_id}: "
            f"benchmark_url={br.get('benchmark_url')!r} not found"
        )
    if not algo_path.exists():
        errors.append(
            f"[CADW_ARTIFACT_MISSING] {card_id}: "
            f"algorithm_source_url={br.get('algorithm_source_url')!r} not found"
        )

    if need_verdict and bench_path.exists():
        try:
            data = json.loads(bench_path.read_text())
            if data.get("verdict") != "consistent":
                errors.append(
                    f"[CADW_VERDICT] {card_id}: "
                    f"{br.get('benchmark_url')!r} verdict="
                    f"{data.get('verdict')!r}, need 'consistent'"
                )
        except json.JSONDecodeError:
            errors.append(
                f"[CADW_VERDICT] {card_id}: "
                f"{br.get('benchmark_url')!r} not valid JSON"
            )


def audit(
    yaml_paths: list[str],
    url_log_path: str,
    root: Path = ROOT,
) -> tuple[list[str], list[str]]:
    log = load_log(Path(url_log_path))
    errors: list[str] = []
    warnings: list[str] = []

    for ypath in yaml_paths:
        sec = yaml.safe_load(Path(ypath).read_text())
        if not isinstance(sec, dict) or "section" not in sec:
            # benchmarks.yaml (registry) or similar non-section YAML — skip
            continue
        sec_id = sec["section"]
        for st in sec.get("subtypes", []):
            for m in st.get("models", []):
                card_id = _card_id(sec_id, st["id"], m["name"])
                # v0.9.75: tier3_export horizon cards — not fetched via
                # fetch tool; require only algorithm_source_url to look
                # like http(s). Skip url_log 200 requirement.
                is_tier3 = m.get("tier_level") == "tier3_export"
                brs = m.get("benchmark_results") or []
                if not brs and m.get("benchmark_result"):
                    # v2 schema: legacy card, not audited (no algorithm_source_url,
                    # no attribution, no url_log coverage). Will be obогащён PINGIS
                    # in v0.9.48+.
                    warnings.append(
                        f"[LEGACY] {card_id}: v2 schema "
                        f"(benchmark_result singular); not audited"
                    )
                    continue
                for br in brs:
                    attr = br.get("attribution")
                    val = br.get("value")
                    note = (br.get("note") or "").strip()

                    # value!=null or value==null → note must explain
                    if not note:
                        tag = "NULL_NO_NOTE" if val is None else "VALUE_NO_NOTE"
                        errors.append(
                            f"[{tag}] {card_id}: value={val!r} but note empty"
                        )

                    if attr == "literature":
                        if is_tier3:
                            algo_url = br.get("algorithm_source_url", "")
                            if not (algo_url.startswith("http://")
                                    or algo_url.startswith("https://")):
                                errors.append(
                                    f"[TIER3_BAD_URL] {card_id}: "
                                    f"algorithm_source_url={algo_url!r} "
                                    f"not http(s)")
                        else:
                            _check_literature(br, card_id, log, errors)
                    elif attr == "cadw_reproduced":
                        _check_cadw_artifact(
                            br, card_id, errors, root, need_verdict=True,
                        )
                    elif attr == "cadw_measured":
                        _check_cadw_artifact(
                            br, card_id, errors, root, need_verdict=False,
                        )
                    else:
                        warnings.append(
                            f"[UNKNOWN_ATTR] {card_id}: attribution={attr!r}"
                        )
    return errors, warnings


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit url_log.jsonl against YAML cards."
    )
    ap.add_argument("url_log", help="path to url_log.jsonl")
    ap.add_argument("yaml_files", nargs="+", help="one or more section YAMLs")
    args = ap.parse_args()

    errors, warnings = audit(args.yaml_files, args.url_log)
    for w in warnings:
        print(w)
    for e in errors:
        print(e)
    if errors:
        print(f"\n{len(errors)} error(s)")
        sys.exit(1)
    print(f"OK ({len(warnings)} warning(s))")


if __name__ == "__main__":
    main()
