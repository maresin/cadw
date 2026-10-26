#!/usr/bin/env python3
"""Benchmark KShapeAD on synthetic TS shape-anomaly DGP."""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from cadw.ml.kshape import fit_kshape


def main() -> None:
    n = 10_000
    seg_len = 50
    n_segments = 20
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    starts = rng.choice(n - seg_len, size=n_segments, replace=False)
    pos = []
    for s in starts:
        lt = np.arange(seg_len)
        x[s:s + seg_len] = np.sign(np.sin(2 * np.pi * lt / 20.0))
        pos.extend(range(s, s + seg_len))
    pos = np.array(sorted(pos))

    tracemalloc.start()
    t0 = time.perf_counter()
    r = fit_kshape(x, n_clusters=3, window_size=50, max_iter=20,
                   contamination=0.1, random_state=42)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1)) if len(pos) else 0.0
    out = {
        "model": "kshape",
        "n": n,
        "window_size": r.window_size,
        "n_clusters": r.n_clusters,
        "max_iter": r.max_iter,
        "n_iter_run": r.n_iter_run,
        "contamination": r.contamination,
        "n_anomaly_points": int(len(pos)),
        "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "threshold": round(float(r.threshold), 4),
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }
    p = Path(".pingis/benches/kshape.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

