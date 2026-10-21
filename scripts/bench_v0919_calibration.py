#!/usr/bin/env python3
"""v0.9.19: calibration harness — formula vs actual for §9.2.N ops.

Runs each op with MemoryGuard(calibrate=True) on a synthetic shape.
Prints per-op {predicted, arrow, rss, C_arrow, C_rss}.

Purpose: close the AUDIT loop. AUDIT.md did this manually;
this script automates the workflow for regression detection.

Usage:
    uv run python scripts/bench_v0919_calibration.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
os.chdir(Path(__file__).resolve().parent.parent)

import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import (
    reduce, CountOp, MeanOp, StdOp, MinMaxOp, CovarianceOp,
)
from cadw.ops.sort import external_sort
from cadw.ops.shuffle import hash_shuffle
from cadw.ops.groupby import groupby_aggregate


def _make_batches(n: int = 100_000, n_batches: int = 10):
    """Uniform 2-col table, split into n_batches."""
    table = pa.table({
        "x": pa.array(list(range(n)), type=pa.int64()),
        "g": pa.array([f"g{i % 100}" for i in range(n)]),
    })
    bs = n // n_batches
    return [
        table.slice(i, bs).to_batches()[0]
        for i in range(0, n, bs)
    ]


def _peek_batch_bytes(batches) -> int:
    """Rough peak per-batch bytes (formula input)."""
    if not batches:
        return 0
    return int(batches[0].nbytes)


def _print_row(name, predicted, sample):
    arrow = sample.get("arrow", 0)
    rss = sample.get("rss", 0)
    if predicted > 0:
        ca = arrow / predicted
        cr = rss / predicted
        print(
            f"{name:22s} {predicted:>12,} {arrow:>12,} {rss:>12,} "
            f"{ca:>8.2f} {cr:>8.2f}"
        )
    else:
        print(
            f"{name:22s} {'unknown':>12s} {arrow:>12,} {rss:>12,} "
            f"{'-':>8s} {'-':>8s}"
        )


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="cadw_bench_v0919_"))
    batches = _make_batches()
    batch_bytes = _peek_batch_bytes(batches)
    n_total = sum(b.num_rows for b in batches)

    print("v0.9.19 calibration bench")
    print("=" * 80)
    print(f"shape: n={n_total:,}, n_batches={len(batches)}, "
          f"batch_bytes={batch_bytes:,}")
    print()
    print(
        f"{'op':22s} {'predicted':>12s} {'arrow':>12s} "
        f"{'rss':>12s} {'C_arrow':>8s} {'C_rss':>8s}"
    )
    print("-" * 80)

    # --- reduce ops ---
    reduce_ops = [
        ("CountOp", CountOp("x"), 16),
        ("MeanOp", MeanOp("x"), 16),
        ("StdOp", StdOp("x"), 24),
        ("MinMaxOp", MinMaxOp("x"), 16),
        ("CovarianceOp", CovarianceOp(["x"]), None),
    ]
    for name, op, state_bytes in reduce_ops:
        if state_bytes is None:
            state_bytes = op.estimate_state_nbytes()
        # Predicted peak = batch.nbytes + 2 × state (per §9.2.1)
        predicted = batch_bytes + 2 * state_bytes
        guard = MemoryGuard(
            max_memory_mb=100, calibrate=True, runtime_monitor=False,
        )
        with guard.calibrate(name, predicted):
            reduce(iter(batches), op, guard, log=False)
        sample = guard.calibration_samples[name][-1]
        _print_row(name, predicted, sample)

    # --- external_sort ---
    # §9.2.7 (AUDIT): Table.sort_by peak = 1.5 × input + 6 MB.
    # But external_sort uses pc.sort_indices + disk runs. Not
    # directly Table.sort_by. Report actuals; formula TBD.
    total_nbytes = sum(b.nbytes for b in batches)
    predicted = 0  # unknown
    guard = MemoryGuard(max_memory_mb=100, calibrate=True)
    with guard.calibrate("external_sort", predicted):
        external_sort(
            iter(batches), ["x"], ["x", "g"], guard,
            tmp_root / "sort", log=False,
        )
    sample = guard.calibration_samples["external_sort"][-1]
    _print_row("external_sort", predicted, sample)

    # --- hash_shuffle ---
    guard = MemoryGuard(max_memory_mb=100, calibrate=True)
    with guard.calibrate("hash_shuffle", predicted):
        hash_shuffle(
            iter(batches), ["g"], ["x"], 8, guard,
            tmp_root / "shuffle", log=False,
        )
    sample = guard.calibration_samples["hash_shuffle"][-1]
    _print_row("hash_shuffle", predicted, sample)

    # --- groupby_aggregate ---
    guard = MemoryGuard(max_memory_mb=100, calibrate=True)
    with guard.calibrate("groupby_aggregate", predicted):
        groupby_aggregate(
            iter(batches), ["g"], ["x"], ["sum"], guard,
            tmp_root / "gb", log=False,
        )
    sample = guard.calibration_samples["groupby_aggregate"][-1]
    _print_row("groupby_aggregate", predicted, sample)

    print("-" * 80)
    print()
    print("Interpretation:")
    print("  C_arrow = arrow_delta / predicted (formula drift signal)")
    print("  C_rss   = rss_delta / predicted (includes Python overhead)")
    print("  drift threshold (AUDIT §9.5): C > 2.0 → investigate formula")
    print()
    print(f"tmp dir (not cleaned): {tmp_root}")


if __name__ == "__main__":
    main()
