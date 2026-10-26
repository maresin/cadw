#!/usr/bin/env python3
"""Benchmark T-squared+Q on synthetic changepoint DGP."""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from cadw.ml.tspq import fit_t2q


def main() -> None:
    n = 20_000
    c = 5
    n_changes = 5
    seg_len = 200
    rng = np.random.default_rng(0)
    t = np.arange(n)
    X = np.empty((n, c))
    for ch in range(c):
        X[:, ch] = np.sin(2 * np.pi * t / (100.0 + 20.0 * ch)) \
            + 0.1 * rng.standard_normal(n)
    cps = np.sort(rng.choice(n - seg_len - 500, size=n_changes,
                             replace=False)) + 500
    pos = []
    for cp in cps:
        end = min(cp + seg_len, n)
        X[cp:end] += 2.0
        pos.extend(range(cp, end))
    pos = np.array(sorted(pos))

    tracemalloc.start()
    t0 = time.perf_counter()
    r = fit_t2q(X, window_size=1, n_components=None, contamination=0.05,
                guard=None)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1))
    out = {
        "model": "tspq",
        "n": n,
        "n_channels": c,
        "n_changes": n_changes,
        "seg_len": seg_len,
        "n_anomaly_points": int(len(pos)),
        "detected": int(r.labels[pos].sum()),
        "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "n_components": r.n_components,
        "t2_limit": round(float(r.t2_limit), 4),
        "q_limit": round(float(r.q_limit), 4),
        "contamination": r.contamination,
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }
    p = Path(".pingis/benches/tspq.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

