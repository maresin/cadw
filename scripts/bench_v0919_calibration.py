#!/usr/bin/env python3
"""v0.9.20-fix1: calibration bench — decomposed predicted vs actual.

Columns:
  pred_batch: formula batch component (batch_bytes for reduce)
  pred_state: formula state component (2×state for reduce)
  pred_total: total formula estimate (sum, or model.solve()+estimate()
              for sort/shuffle/groupby)
  batch:      reference per-batch bytes (transparency)
  peak_pool:  max pa.pool.bytes_allocated during op (measured)
  peak_rss:   max psutil RSS during op (measured)
  C_peak:     peak_pool / pred_total (formula drift signal)
  C_batch:    peak_pool / batch (N× batch memory — transparency)

Interpretation:
  - C_peak ≈ 1.0 → formula accurate
  - C_peak > 2.0 → formula drift (temp buffers, unmodeled state)
  - C_batch ≫ 1 → op needs multi-batch memory (sort: sort_indices
                  int64[N], k×read_buffer; groupby: hash table, etc.)

Formula components (DECISIONS §9.2.N):
  - reduce §9.2.1:  batch_size×row_size + 2×state_bytes
  - sort   §9.2.3:  max(run_buffer, k×read_buffer + k×heap
                    + read_batch×row_size)
  - shuffle §9.2.2: batch×row_size + P×writer_buffer + c_hash×batch
  - groupby §9.2.4: max(shuffle, local_groupby)

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
from cadw.core.models import get_model, data_chars_from_sample
from cadw.ops.reduce import (
    reduce, CountOp, MeanOp, StdOp, MinMaxOp, CovarianceOp,
)
from cadw.ops.sort import external_sort
from cadw.ops.shuffle import hash_shuffle
from cadw.ops.groupby import groupby_aggregate


def _make_batches(n: int = 100_000, n_batches: int = 10):
    table = pa.table({
        "x": pa.array(list(range(n)), type=pa.int64()),
        "g": pa.array([f"g{i % 100}" for i in range(n)]),
    })
    bs = n // n_batches
    return [table.slice(i, bs).to_batches()[0] for i in range(0, n, bs)]


def _dc_from_batches(batches, on_cols):
    from cadw.core._k_est import estimate_k_sample_arrow
    n_total = sum(b.num_rows for b in batches)
    sample = batches[0]
    if sample.num_rows == 0 or n_total == 0:
        return None
    on_present = [c for c in on_cols if c in sample.schema.names]
    k_sample = 1
    if on_present:
        try:
            k_sample = estimate_k_sample_arrow(sample, on_present)
        except Exception:
            k_sample = 1
    k_est = max(1, int(k_sample * n_total / max(1, sample.num_rows)))
    first = {c: sample.column(c)[0].as_py() for c in on_present}
    avg_key_size = len(repr(tuple(first.values())).encode()) if first else 8
    try:
        return data_chars_from_sample(
            sample, n_total,
            k_est=k_est,
            avg_key_size_bytes=avg_key_size,
            size_of_key_entry_bytes=avg_key_size + 80,
        )
    except Exception:
        return None


def _model_predicted(name, batches, on_cols, available):
    try:
        model = get_model(name)
    except KeyError:
        return None
    dc = _dc_from_batches(batches, on_cols)
    if dc is None:
        return None
    try:
        params = model.solve(available, dc)
    except Exception:
        return None
    if params is None:
        return None
    try:
        return model.estimate(params, dc)
    except Exception:
        return None


def _fmt(x):
    if x is None:
        return "?"
    if x == 0:
        return "0"
    return f"{x:,}"


def _print_row(name, pred_batch, pred_state, pred_total, batch, sample):
    peak_pool = sample.get("peak_pool", 0)
    peak_rss = sample.get("peak_rss", 0)
    c_batch = peak_pool / batch if batch else 0
    if pred_total and pred_total > 0:
        c_peak = peak_pool / pred_total
        print(
            f"{name:18s} {_fmt(pred_batch):>11s} {_fmt(pred_state):>11s} "
            f"{_fmt(pred_total):>11s} {_fmt(batch):>10s} "
            f"{_fmt(peak_pool):>12s} {_fmt(peak_rss):>12s} "
            f"{c_peak:>7.2f} {c_batch:>7.2f}"
        )
    else:
        print(
            f"{name:18s} {'?':>11s} {'?':>11s} {'?':>11s} "
            f"{_fmt(batch):>10s} "
            f"{_fmt(peak_pool):>12s} {_fmt(peak_rss):>12s} "
            f"{'?':>7s} {c_batch:>7.2f}"
        )


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="cadw_bench_v0920_"))
    batches = _make_batches()
    batch_bytes = batches[0].nbytes
    n_total = sum(b.num_rows for b in batches)
    available = 100 * 1024 * 1024

    print("v0.9.20-fix1 calibration bench (decomposed predicted)")
    print("=" * 140)
    print(f"shape: n={n_total:,}, n_batches={len(batches)}, "
          f"batch_bytes={batch_bytes:,}")
    print()
    print(
        f"{'op':18s} {'pred_batch':>11s} {'pred_state':>11s} "
        f"{'pred_total':>11s} {'batch':>10s} "
        f"{'peak_pool':>12s} {'peak_rss':>12s} "
        f"{'C_peak':>7s} {'C_batch':>7s}"
    )
    print("-" * 140)

    reduce_specs = [
        ("CountOp", CountOp("x"), 16),
        ("MeanOp", MeanOp("x"), 16),
        ("StdOp", StdOp("x"), 24),
        ("MinMaxOp", MinMaxOp("x"), 16),
        ("CovarianceOp", CovarianceOp(["x"]), None),
    ]
    for name, op, state_bytes in reduce_specs:
        if state_bytes is None:
            state_bytes = op.estimate_state_nbytes()
        pred_batch = batch_bytes
        pred_state = 2 * state_bytes
        pred_total = pred_batch + pred_state
        guard = MemoryGuard(max_memory_mb=100, calibrate=True)
        with guard.calibrate(name, pred_total, sample_peak=True):
            reduce(iter(batches), op, guard, log=False)
        _print_row(
            name, pred_batch, pred_state, pred_total, batch_bytes,
            guard.calibration_samples[name][-1],
        )

    pred_sort = _model_predicted("sort", batches, [], available)
    guard = MemoryGuard(max_memory_mb=100, calibrate=True)
    with guard.calibrate(
        "external_sort", pred_sort or 0, sample_peak=True,
    ):
        external_sort(
            iter(batches), ["x"], ["x", "g"], guard,
            tmp_root / "sort", log=False,
        )
    _print_row(
        "external_sort", None, None, pred_sort, batch_bytes,
        guard.calibration_samples["external_sort"][-1],
    )

    pred_shuf = _model_predicted("hash_shuffle", batches, ["g"], available)
    guard = MemoryGuard(max_memory_mb=100, calibrate=True)
    with guard.calibrate(
        "hash_shuffle", pred_shuf or 0, sample_peak=True,
    ):
        hash_shuffle(
            iter(batches), ["g"], ["x"], 8, guard,
            tmp_root / "shuffle", log=False,
        )
    _print_row(
        "hash_shuffle", None, None, pred_shuf, batch_bytes,
        guard.calibration_samples["hash_shuffle"][-1],
    )

    pred_gb = _model_predicted("groupby", batches, ["g"], available)
    guard = MemoryGuard(max_memory_mb=100, calibrate=True)
    with guard.calibrate(
        "groupby_aggregate", pred_gb or 0, sample_peak=True,
    ):
        groupby_aggregate(
            iter(batches), ["g"], ["x"], ["sum"], guard,
            tmp_root / "gb", log=False,
        )
    _print_row(
        "groupby_aggregate", None, None, pred_gb, batch_bytes,
        guard.calibration_samples["groupby_aggregate"][-1],
    )

    print("-" * 140)
    print()
    print("Formula components (DECISIONS §9.2.N):")
    print("  reduce §9.2.1:  batch + 2×state (decomposed above)")
    print("  sort   §9.2.3:  max(run_buffer, k×read_buffer + k×heap "
          "+ read_batch×row_size)")
    print("  shuffle §9.2.2: batch×row_size + P×writer_buffer "
          "+ c_hash×batch")
    print("  groupby §9.2.4: max(shuffle, local_groupby)")
    print()
    print("Reading the table:")
    print("  C_peak  = peak_pool / pred_total -> formula accuracy")
    print("  C_batch = peak_pool / batch      -> N× batch memory "
          "(transparency)")
    print("  Note: pred_total = model.solve().estimate() is UPPER BOUND.")
    print("  For inputs < buffer capacity, C_peak << 1 (formula "
          "over-conservative).")
    print("  For inputs filling buffers, C_peak ~= 1. C_peak > 2 -> "
          "formula drift.")
    print()
    print(f"tmp dir (not cleaned): {tmp_root}")


if __name__ == "__main__":
    main()
