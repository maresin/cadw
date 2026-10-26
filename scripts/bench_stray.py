#!/usr/bin/env python3
"""Benchmark STRAY on synthetic TS anomaly DGP.

NOTE: STRAY uses brute-force KNN (O(n^2) time). Use n<=10_000 for
reasonable runtime; for larger n, prefer EIF or Sub-PCA.
"""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from cadw.ml.stray import fit_stray


def main() -> None:
    n = 10_000
    n_spikes = 30
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += 20.0

    tracemalloc.start()
    t0 = time.perf_counter()
    r = fit_stray(x, k=20, tail_fraction=0.05, alpha=0.01)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1))
    out = {
        "model": "stray", "n": n, "n_spikes": n_spikes,
        "detected": int(r.labels[pos].sum()), "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "k": r.k, "tail_fraction": r.tail_fraction,
        "alpha": r.alpha, "threshold": round(float(r.threshold), 4),
        "xi": round(float(r.xi), 4), "beta": round(float(r.beta), 4),
        "n_tail": r.n_tail,
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }
    p = Path(".pingis/benches/stray.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

