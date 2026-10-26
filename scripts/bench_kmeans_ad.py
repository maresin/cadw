#!/usr/bin/env python3
"""Benchmark KMeansAD on synthetic TS anomaly DGP."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from cadw.ml.kmeans_ad import fit_kmeans_ad


def main() -> None:
    n = 50_000
    n_spikes = 30
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += 20.0

    tracemalloc.start()
    t0 = time.perf_counter()
    r = fit_kmeans_ad(x, window_size=50, n_clusters=8,
                      contamination=0.05, aggregate="max",
                      random_state=42)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1))
    out = {
        "model": "kmeans_ad",
        "n": n, "n_spikes": n_spikes,
        "detected": int(r.labels[pos].sum()),
        "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "window_size": r.window_size,
        "n_clusters": r.n_clusters,
        "n_iter_run": r.n_iter_run,
        "inertia": round(float(r.inertia), 2),
        "contamination": r.contamination,
        "aggregate": r.aggregate,
        "threshold": round(float(r.threshold), 4),
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }
    p = Path(".pingis/benches/kmeans_ad.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

