#!/usr/bin/env python3
"""Benchmark BinarySegmentation on synthetic changepoint DGP."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from cadw.ml.binseg import fit_binseg


def main() -> None:
    n = 100_000
    n_changes = 10
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) * 0.1
    cps = np.sort(rng.choice(n - 1000, size=n_changes,
                             replace=False)) + 500
    for i, cp in enumerate(cps):
        sign = 1.0 if i % 2 == 0 else -1.0
        x[cp:] += sign * 2.5

    tracemalloc.start()
    t0 = time.perf_counter()
    r = fit_binseg(x, n_bkps=n_changes, min_size=50)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    detected = r.changepoints
    hits = sum(
        1 for cp in cps
        if len(detected) > 0 and np.min(np.abs(detected - cp)) <= 10)
    recall = hits / n_changes if n_changes > 0 else 0.0

    out = {
        "model": "binseg",
        "n": n,
        "n_changes": n_changes,
        "detected": int(len(detected)),
        "hits": int(hits),
        "recall": round(float(recall), 4),
        "n_bkps_requested": r.n_bkps,
        "min_size": r.min_size,
        "changepoints": [int(cp) for cp in detected],
        "true_changepoints": [int(cp) for cp in cps],
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }
    p = Path(".pingis/benches/binseg.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

