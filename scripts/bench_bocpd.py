#!/usr/bin/env python3
"""Benchmark BOCPD-StudentT on synthetic changepoint DGP."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from cadw.ml.bocpd import fit_bocpd


def main() -> None:
    n = 20_000
    n_changes = 5
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) * 0.2
    cps = np.sort(rng.choice(n - 2000, size=n_changes,
                             replace=False)) + 1000
    for i, cp in enumerate(cps):
        sign = 1.0 if i % 2 == 0 else -1.0
        x[cp:] += sign * 2.5

    tracemalloc.start()
    t0 = time.perf_counter()
    # v0.9.72-fix2: burn_in masks posterior warm-up (no data yet).
    # 2*run_length_threshold default; pass explicitly for clarity.
    r = fit_bocpd(x, hazard_rate=1.0 / 500.0, contamination=0.02,
                  mu0=0.0, kappa0=0.01, alpha0=1.0, beta0=0.04,
                  burn_in=20)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    detected = r.changepoints
    hits = sum(
        1 for cp in cps
        if len(detected) > 0 and np.min(np.abs(detected - cp)) <= 50)
    recall = hits / n_changes if n_changes > 0 else 0.0

    # Detection delay: for detected cps, median distance to nearest true
    delays = []
    for cp in cps:
        if len(detected) > 0:
            i = int(np.argmin(np.abs(detected - cp)))
            if abs(int(detected[i]) - int(cp)) <= 50:
                delays.append(int(detected[i]) - int(cp))
    median_delay = float(np.median(delays)) if delays else -1.0

    out = {
        "model": "bocpd",
        "n": n,
        "n_changes": n_changes,
        "burn_in": 20,
        "detected": int(len(detected)),
        "hits": int(hits),
        "recall": round(float(recall), 4),
        "median_detection_delay": median_delay,
        "hazard_rate": r.hazard_rate,
        "run_length_threshold": r.run_length_threshold,
        "contamination": r.contamination,
        "changepoints": [int(cp) for cp in detected[:20]],
        "true_changepoints": [int(cp) for cp in cps],
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }
    p = Path(".pingis/benches/bocpd.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

