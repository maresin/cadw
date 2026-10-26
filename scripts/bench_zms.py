#!/usr/bin/env python3
"""Benchmark ZMS on synthetic TS anomaly DGP."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from cadw.ml.zms import fit_zms


def main() -> None:
    n = 100_000
    n_spikes = 50
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += 20.0
    tracemalloc.start()
    t0 = time.perf_counter()
    r = fit_zms(x, window_size=50, threshold=3.0)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    recall = float(np.mean(r.labels[pos] == 1))
    out = {
        "model": "zms", "n": n, "n_spikes": n_spikes,
        "detected": int(r.labels[pos].sum()), "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "window_size": r.window_size,
        "threshold": round(float(r.threshold), 4),
        "fit_time_s": round(elapsed, 4),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }
    p = Path(".pingis/benches/zms.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

