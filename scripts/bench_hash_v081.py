"""v0.8.1-fix3 bench: compute_hashes + end-to-end shuffle (int64 + string).

Baselines (v0.8.0 diagnostic, 39.2 ms / 100k compute_hashes; 0.61s shuffle):
    int64      → expect ~2 ms   (pandas C-hash, 19×)
    float64    → expect ~1 ms   (pandas C-hash, 39×)
    string     → expect ~39 ms  (Python path, no change — v0.8.1-fix3)
    shuffle int64 1M → expect ~0.22s (2.76×)
    shuffle string 1M → expect ~0.61s (no change)

Run: uv run python scripts/bench_hash_v081.py
"""
from __future__ import annotations

import gc
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

# `python scripts/foo.py` puts scripts/ in sys.path[0], not project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.shuffle import compute_hashes, hash_shuffle


def bench_compute_hashes(label, arr, n_warmup=2, n_iter=5):
    b = pa.RecordBatch.from_arrays([arr], names=["k"])
    for _ in range(n_warmup):
        compute_hashes(b, ["k"], seed=0)
    t0 = time.time()
    for _ in range(n_iter):
        compute_hashes(b, ["k"], seed=0)
    dt = (time.time() - t0) / n_iter
    print(f"  [{label}] {dt * 1000:7.2f} ms per 100k rows")
    return dt


def run_shuffle(table, label, num_rows, baseline_s):
    def bi():
        for i in range(0, num_rows, 10000):
            yield table.slice(i, 10000).to_batches()[0]
    gc.collect()
    with tempfile.TemporaryDirectory() as td:
        t0 = time.time()
        hash_shuffle(
            bi(), ["k"], ["v"], 8,
            MemoryGuard(max_memory_mb=100), Path(td), log=False,
        )
        dt = time.time() - t0
    print(f"  hash_shuffle 1M {label:7s}: {dt:.2f}s  "
          f"(baseline {baseline_s:.2f}s → {baseline_s / dt:.2f}× speedup)")
    return dt


def main():
    n = 100_000
    rng = np.random.default_rng(0)

    print("=" * 60)
    print("compute_hashes v0.8.1-fix3 (pandas numeric, python string)")
    print("=" * 60)

    int_arr = pa.array(rng.integers(0, 100_000, size=n).astype(np.int64))
    t_int = bench_compute_hashes("int64", int_arr)

    str_arr = pa.array([f"k{i % 50000:08d}" for i in range(n)])
    t_str = bench_compute_hashes("string", str_arr)

    float_arr = pa.array(rng.normal(size=n))
    t_float = bench_compute_hashes("float64", float_arr)

    print()
    print("=" * 60)
    print("Baseline (v0.8.0 diagnostic: 39.2 ms / 100k)")
    print("=" * 60)
    baseline_ms = 39.2
    for name, t in (("int64", t_int), ("string", t_str), ("float64", t_float)):
        speedup = baseline_ms / (t * 1000)
        mark = "OK" if speedup >= 1.5 else ("~same" if speedup >= 0.9 else "REGRESSION")
        print(f"  {name:7s}: baseline {baseline_ms:.1f} ms → "
              f"{t * 1000:.1f} ms  ({speedup:.2f}×) [{mark}]")

    print()
    print("=" * 60)
    print("End-to-end shuffle (1M rows, 8 partitions)")
    print("=" * 60)
    n1m = 1_000_000
    rng1m = np.random.default_rng(7)
    keys = rng1m.integers(0, 100_000, size=n1m).astype(np.int64)
    vals = rng1m.integers(0, 1000, size=n1m).astype(np.int64)

    tbl_int = pa.table({"k": pa.array(keys), "v": pa.array(vals)})
    run_shuffle(tbl_int, "int64", n1m, baseline_s=0.61)

    str_keys = [f"k{i:08d}" for i in rng1m.integers(0, 100_000, size=n1m)]
    tbl_str = pa.table({"k": pa.array(str_keys), "v": pa.array(vals)})
    run_shuffle(tbl_str, "string", n1m, baseline_s=0.61)


if __name__ == "__main__":
    main()
