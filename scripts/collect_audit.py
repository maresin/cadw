#!/usr/bin/env python3
"""Collect audit files for CADW base-cases audit (v0.9.90).

Tiers, ordered by audit priority:
  0_minimal   -- smallest bundle, closes 80% of remaining gaps
  A1_ops      -- ops primitives (reduce, join, groupby, dedup, ...)
  A2_ml       -- ML modules not yet audited
  A3_df       -- dataframe layer (ml_bindings, exports, per_group, rolling)
  A4_cross    -- cv, scalers, pipeline
  B_ts_core   -- core helpers + TS modules (D-time-series block)
  C_tests     -- gate tests + CHANGELOG (active whitelist + version history)

Output:
  <root>/audit_bundles/audit_bundle.txt          if total <= --max-bytes
  <root>/audit_bundles/audit_<tier>.txt          per-tier if split
  <root>/audit_bundles/audit_<tier>_partNN.txt   if a tier itself > max
  <root>/audit_bundles/_index.txt                file-to-bundle map

Usage:
  uv run python scripts/collect_audit.py
  uv run python scripts/collect_audit.py --minimal            # only tier 0
  uv run python scripts/collect_audit.py --tiers A1_ops A2_ml # selected tiers
  uv run python scripts/collect_audit.py --single             # force one bundle
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/home/maresin/Dev/sandbox/temp_cadw/Cadw_framework")

TIERS: dict[str, list[str]] = {
    # Smallest useful set (~500 KB): if only one bundle can be sent.
    "0_minimal": [
        "cadw/ops/reduce.py",
        "cadw/ops/join.py",
        "cadw/dataframe/_ml_bindings.py",
        "cadw/dataframe/_exports.py",
        "cadw/ml/_series.py",
        "cadw/ml/logistic.py",
        "cadw/ml/linear.py",
        "cadw/scalers.py",
        "cadw/cv.py",
        "tests/test_category_beyond_memory.py",
    ],
    # A1: ops layer (Reduce/HashShuffle/Sort callers, join, quantile, ...)
    "A1_ops": [
        "cadw/ops/reduce.py",
        "cadw/ops/groupby.py",
        "cadw/ops/join.py",
        "cadw/ops/dedup.py",
        "cadw/ops/rank.py",
        "cadw/ops/quantile.py",
        "cadw/ops/topk.py",
        "cadw/ops/resample.py",
        "cadw/ops/pivot.py",
        "cadw/ops/describe.py",
        "cadw/ops/histogram.py",
        "cadw/ops/metrics.py",
    ],
    # A2: ML modules not yet audited (tier1_core per METHODOLOGY §13)
    "A2_ml": [
        "cadw/ml/_series.py",
        "cadw/ml/logistic.py",
        "cadw/ml/linear.py",
        "cadw/ml/pca.py",
        "cadw/ml/kmeans.py",
        "cadw/ml/kmeans_ad.py",
        "cadw/ml/iforest.py",
        "cadw/ml/eif.py",
        "cadw/ml/naive_bayes.py",
        "cadw/ml/randsvd.py",
        "cadw/ml/tspq.py",
        "cadw/ml/calibration.py",
        "cadw/ml/fairness.py",
        "cadw/ml/drift.py",
        "cadw/ml/importance.py",
        "cadw/ml/uplift.py",
        "cadw/ml/chains.py",
        "cadw/ml/hybrid_mf.py",
        "cadw/ml/cusum.py",
        "cadw/ml/geo.py",
        "cadw/ml/pelt.py",
        "cadw/ml/birch.py",
        "cadw/ml/minibatch.py",
        "cadw/ml/anomaly_pipeline.py",
    ],
    # A3: dataframe layer (entry points for df.fit_*/transform_*)
    "A3_df": [
        "cadw/dataframe/_ml_bindings.py",
        "cadw/dataframe/_exports.py",
        "cadw/dataframe/_per_group.py",
        "cadw/dataframe/_rolling.py",
        "cadw/dataframe/_cumulative.py",
        "cadw/dataframe/_transformers_fill.py",
    ],
    # A4: cross-cutting (cv, scalers, pipeline)
    "A4_cross": [
        "cadw/cv.py",
        "cadw/scalers.py",
        "cadw/pipeline.py",
    ],
    # B: core helpers + TS (to close tier1_core_exception claims)
    "B_ts_core": [
        "cadw/core/sampling.py",
        "cadw/core/planner.py",
        "cadw/core/models_join.py",
        "cadw/core/models_groupby.py",
        "cadw/core/_k_est.py",
        "cadw/ml/_logging.py",
        "cadw/ts/models.py",
        "cadw/ts/features.py",
        "cadw/ts/stats.py",
        "cadw/ts/backtest.py",
        "cadw/ts/transforms.py",
        "cadw/ts/auto_ets.py",
        "cadw/ts/auto_theta.py",
        "cadw/ts/otm.py",
        "cadw/ts/ensemble.py",
        "cadw/ts/thf.py",
        "cadw/ts/_decomposition.py",
        "cadw/ts/ets.py",
        "cadw/ts/theta.py",
        "cadw/ts/bvar.py",
        "cadw/ts/var.py",
        "cadw/ts/vecm.py",
    ],
    # C: gate tests + version history
    "C_tests": [
        "tests/_gate_whitelist.py",
        "tests/test_primitive_reuse.py",
        "tests/test_guard_before_alloc.py",
        "tests/test_no_dense_windows.py",
        "tests/test_ml_tools_schema.py",
        "tests/test_oom_resistance.py",
        "tests/test_category_beyond_memory.py",
        "tests/conftest.py",
        "CHANGELOG.md",
    ],
}


def _read_file(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"ERR: cannot read {p}: {e}", file=sys.stderr)
        return None


def _make_block(rel: str, content: str) -> str:
    sep = "=" * 78
    return f"\n\n# {sep}\n# FILE: {rel}\n# {sep}\n{content}\n"


def _pack_chunks(items: list[tuple[str, str]],
                 max_bytes: int) -> list[list[tuple[str, str]]]:
    """Pack (rel, block) into chunks <= max_bytes. Oversized item -> own chunk."""
    chunks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_bytes = 0
    for rel, block in items:
        b = len(block.encode("utf-8"))
        if b > max_bytes:
            if current:
                chunks.append(current)
                current, current_bytes = [], 0
            print(f"WARN: {rel} alone = {b:,} B > max {max_bytes:,}; "
                  f"dedicated chunk", file=sys.stderr)
            chunks.append([(rel, block)])
            continue
        if current_bytes + b > max_bytes and current:
            chunks.append(current)
            current, current_bytes = [], 0
        current.append((rel, block))
        current_bytes += b
    if current:
        chunks.append(current)
    return chunks


def _emit(chunk: list[tuple[str, str]], out: Path, meta: str) -> int:
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# CADW audit bundle - {meta}\n")
        f.write(f"# files: {len(chunk)}\n")
        for rel, block in chunk:
            f.write(block)
    return out.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--outdir", type=Path, default=None,
                    help="default: <root>/audit_bundles/")
    ap.add_argument("--max-bytes", type=int, default=2_600_000,
                    help="per-bundle ceiling, default 2.6 MB")
    ap.add_argument("--single", action="store_true",
                    help="force one combined bundle regardless of size")
    ap.add_argument("--minimal", action="store_true",
                    help="only tier 0_minimal")
    ap.add_argument("--tiers", nargs="*", default=None,
                    help="only these tiers (exact names)")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        print(f"ERR: root {root} not found", file=sys.stderr)
        return 2

    selected = list(TIERS.keys())
    if args.minimal:
        selected = ["0_minimal"]
    elif args.tiers:
        unknown = [t for t in args.tiers if t not in TIERS]
        if unknown:
            print(f"ERR: unknown tiers: {unknown}", file=sys.stderr)
            return 2
        selected = list(args.tiers)

    outdir: Path = args.outdir or (root / "audit_bundles")
    outdir.mkdir(parents=True, exist_ok=True)

    # Flatten selected tiers, dedupe by relpath
    seen: set[str] = set()
    all_items: list[tuple[str, str]] = []
    missing: list[str] = []
    tier_of: dict[str, str] = {}
    for tier in selected:
        for rel in TIERS[tier]:
            if rel in seen:
                continue
            seen.add(rel)
            p = root / rel
            if not p.exists():
                missing.append(rel)
                continue
            content = _read_file(p)
            if content is None:
                continue
            all_items.append((rel, _make_block(rel, content)))
            tier_of[rel] = tier

    total_bytes = sum(len(b.encode("utf-8")) for _, b in all_items)
    print(f"Collected {len(all_items)} files, {total_bytes:,} bytes total.")
    if missing:
        print(f"Missing ({len(missing)}):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    index_lines: list[str] = []
    n_outputs = 0

    # Try single bundle
    if args.single or total_bytes <= args.max_bytes:
        out = outdir / "audit_bundle.txt"
        size = _emit(all_items, out, "all selected files")
        print(f"OK: {out} = {size:,} bytes, {len(all_items)} files")
        index_lines.append(f"{out.name} ({size:,} B):")
        for rel, _ in all_items:
            index_lines.append(f"  {rel}")
        n_outputs = 1
    else:
        # Per tier
        for tier in selected:
            tier_items = [(rel, b) for rel, b in all_items
                          if tier_of.get(rel) == tier]
            if not tier_items:
                continue
            chunks = _pack_chunks(tier_items, args.max_bytes)
            for i, chunk in enumerate(chunks, 1):
                if len(chunks) == 1:
                    name = f"audit_{tier}.txt"
                    meta = tier
                else:
                    name = f"audit_{tier}_part{i:02d}.txt"
                    meta = f"{tier} part {i}/{len(chunks)}"
                out = outdir / name
                size = _emit(chunk, out, meta)
                print(f"OK: {out} = {size:,} bytes, {len(chunk)} files")
                index_lines.append(f"{name} ({size:,} B):")
                for rel, _ in chunk:
                    index_lines.append(f"  {rel}")
                index_lines.append("")
                n_outputs += 1

    # Index
    index_path = outdir / "_index.txt"
    index_path.write_text("Bundle -> files\n\n" + "\n".join(index_lines),
                          encoding="utf-8")
    print(f"\nIndex: {index_path}")
    print(f"Done. {n_outputs} bundles in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())