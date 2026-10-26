#!/usr/bin/env python3
"""Benchmark DWT-MLEAD on synthetic TS anomaly DGP."""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

# Make `import cadw` work whether script is run as
# `python scripts/bench_dwt_mlead.py` or `python -m scripts.bench_dwt_mlead`
# (§13.4/184: scripts/ on sys.path[0] prevents cadw import otherwise).
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from cadw.ml.dwt_mlead import fit_dwt_mlead


def main() -> None:
    n = 100_000
    n_spikes = 50
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    spike_pos = rng.choice(n, size=n_spikes, replace=False)
    x[spike_pos] += 5.0

    tracemalloc.start()
    t0 = time.perf_counter()
    result = fit_dwt_mlead(x)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    detected = int(result.is_anomaly[spike_pos].sum())
    output = {
        "model": "dwt_mlead",
        "n": n,
        "n_spikes": n_spikes,
        "detected": detected,
        "recall": round(detected / n_spikes, 4),
        "total_flagged": int(result.is_anomaly.sum()),
        "fit_time_s": round(elapsed, 4),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
        "start_level": result.start_level,
        "max_level": result.max_level,
        "quantile_epsilon": result.quantile_epsilon,
        "min_anomaly_consistency": result.min_anomaly_consistency,
    }
    out_path = Path(".pingis/benches/dwt_mlead.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

