#!/usr/bin/env python
"""v0.9.120: calibrate throughput.yaml rates (AP22, DEC-046).

Usage:
  uv run python scripts/calibrate_throughput.py            # dry-run
  uv run python scripts/calibrate_throughput.py --write    # update yaml

Measures wall-clock throughput for three primitives:
  - reduce (SumOp over float column)
  - sort (external_sort on one float key)
  - hash_shuffle (8 partitions on int64 key)

Updates docs/ml_tools/throughput.yaml (rates + calibrated; header
comment preserved; env untouched). filter/with_column/groupby/join/
dedup keep static conservative defaults — their throughput is dominated
by the underlying primitive (see DEC-046).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import yaml

ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT / "docs" / "ml_tools" / "throughput.yaml"


def _make_batches(n: int, batch_size: int = 10_000):
    rng = np.random.RandomState(0)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        k = end - start
        yield pa.RecordBatch.from_arrays(
            [pa.array(rng.normal(size=k), type=pa.float64()),
             pa.array(rng.randint(0, 1000, k).astype(np.int64),
                      type=pa.int64())],
            names=["x", "g"],
        )


def _measure(fn, *, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(statistics.median(times))


def _calib_reduce(n: int, repeats: int) -> float:
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ops.reduce import SumOp, reduce as _reduce

    def run():
        mg = MemoryGuard(max_memory_mb=500)
        _reduce(_make_batches(n), SumOp("x"), mg, log=False)

    t = _measure(run, repeats=repeats)
    return float(n) / max(t, 1e-9)


def _calib_sort(n: int, repeats: int) -> float:
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ops.sort import external_sort

    def run():
        mg = MemoryGuard(max_memory_mb=200)
        with tempfile.TemporaryDirectory() as td:
            external_sort(
                _make_batches(n), key_columns=["x"],
                value_columns=["x"], memory_guard=mg,
                spill_dir=Path(td) / "s",
            )

    t = _measure(run, repeats=repeats)
    return float(n) / max(t, 1e-9)


def _calib_shuffle(n: int, repeats: int) -> float:
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ops.shuffle import hash_shuffle

    def run():
        mg = MemoryGuard(max_memory_mb=200)
        with tempfile.TemporaryDirectory() as td:
            hash_shuffle(
                _make_batches(n), key_columns=["g"],
                value_columns=["x"], num_partitions=8,
                memory_guard=mg, spill_dir=Path(td) / "s",
            )

    t = _measure(run, repeats=repeats)
    return float(n) / max(t, 1e-9)


def _write_rates(rates: dict, path: Path, *, calibrated: str | None = None) -> None:
    """Update `rates` in yaml. Preserves header comment up to `profiles:`."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    prof = data.setdefault("profiles", {}).setdefault("default", {})
    if calibrated:
        prof["calibrated"] = calibrated
    rates_cfg = prof.setdefault("rates", {})
    for k, v in rates.items():
        rates_cfg.setdefault(k, {})["ops_unit"] = "elements"
        rates_cfg[k]["rate_per_sec"] = float(v)
    # v0.9.120-fix1: PyYAML 1.1 resolves `5.0e7` (no exponent sign) as
    # STRING. Normalize every rate to float before dump, so `--write`
    # never leaves string values in the file.
    for k, entry in rates_cfg.items():
        if not isinstance(entry, dict):
            continue
        if "rate_per_sec" in entry:
            entry["rate_per_sec"] = float(entry["rate_per_sec"])
    raw = path.read_text(encoding="utf-8")
    idx = raw.find("profiles:")
    header = raw[:idx] if idx >= 0 else ""
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--write", action="store_true",
                    help="write measured rates to throughput.yaml")
    ap.add_argument("--calibrated", type=str, default=None,
                    help="override `calibrated` date")
    args = ap.parse_args(argv)

    print(f"Calibration: n={args.n}, repeats={args.repeats}")
    rates = {}
    rates["reduce"] = _calib_reduce(args.n, args.repeats)
    print(f"  reduce:       {rates['reduce']:.3e} el/sec")
    rates["sort"] = _calib_sort(args.n, args.repeats)
    print(f"  sort:         {rates['sort']:.3e} el/sec")
    rates["hash_shuffle"] = _calib_shuffle(args.n, args.repeats)
    print(f"  hash_shuffle: {rates['hash_shuffle']:.3e} el/sec")

    if not args.write:
        print("\n[dry-run] --write to update throughput.yaml")
        return 0

    _write_rates(rates, YAML_PATH, calibrated=args.calibrated)
    print(f"\nWrote {YAML_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
