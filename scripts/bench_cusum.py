#!/usr/bin/env python3
"""Benchmark CUSUM on synthetic changepoint DGP."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from cadw.ml.cusum import fit_cusum


def main() -> None:
    n = 100_000
    n_changes = 10
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) * 0.1
    # Inject mean shifts at random positions
    cps = np.sort(rng.choice(n - 1000, size=n_changes, replace=False)) + 500
    pos = []
    for i, cp in enumerate(cps):
        end = min(cp + 200, n)
        sign = 1 if i % 2 == 0 else -1
        x[cp:end] += sign * 2.5
        pos.extend(range(cp, end))
    pos = np.array(sorted(pos))

    tracemalloc.start()
    t0 = time.perf_counter()
    # Fixed h (standard CUSUM): post-shift flags at h/(step-k) rate;
    # reset after each alarm bounds the accumulation -> no monotone
    # drift of threshold.
    r = fit_cusum(x, k=0.5, h=5.0)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1))
    out = {
        "model": "cusum", "n": n,
        "n_changes": n_changes, "n_anomaly_points": int(len(pos)),
        "detected": int(r.labels[pos].sum()),
        "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "k": r.k, "h": r.h,
        "threshold": round(float(r.threshold), 4),
        "fit_time_s": round(elapsed, 4),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }
    p = Path(".pingis/benches/cusum.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

