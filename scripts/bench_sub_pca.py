#!/usr/bin/env python3
"""Benchmark Sub-PCA on synthetic TS point-anomaly DGP."""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from cadw.ml.sub_pca import fit_sub_pca


def main() -> None:
    n = 100_000
    n_spikes = 50
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += 20.0  # amplitude 20: clear separation from natural ~0.6

    tracemalloc.start()
    t0 = time.perf_counter()
    # k=5: small k for point anomalies (larger k leaks spike into subspace).
    # mean aggregation: point anomalies get ~w*mean spike score, decay
    # with distance — better than max for point targets.
    r = fit_sub_pca(x, window_size=50, n_components=5,
                    contamination=0.01, aggregate="mean")
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1))
    out = {
        "model": "sub_pca",
        "n": n,
        "n_spikes": n_spikes,
        "detected": int(r.labels[pos].sum()),
        "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "window_size": r.window_size,
        "n_components": r.n_components,
        "contamination": r.contamination,
        "aggregate": r.aggregate,
        "threshold": round(float(r.threshold), 4),
        "fit_time_s": round(elapsed, 4),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }
    p = Path(".pingis/benches/sub_pca.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

