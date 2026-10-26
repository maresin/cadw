#!/usr/bin/env python3
"""Benchmark EIF on synthetic TS anomaly DGP (uni + multi)."""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from cadw.ml.eif import fit_eif


def _uni_bench(n: int = 50_000, n_segments: int = 20,
               seg_len: int = 30) -> dict:
    """Univariate sine + constant-amplitude segments (not point spikes)."""
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    starts = rng.choice(n - seg_len, size=n_segments, replace=False)
    pos = []
    for s in starts:
        x[s:s + seg_len] = 20.0
        pos.extend(range(s, s + seg_len))
    pos = np.array(sorted(pos))

    tracemalloc.start()
    t0 = time.perf_counter()
    r = fit_eif(x, n_estimators=100, window_size=50, stride=5,
                contamination=0.02, aggregate="max", random_state=0)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1))
    return {
        "mode": "univariate_segments",
        "n": n, "n_segments": n_segments, "seg_len": seg_len,
        "n_anomaly_points": int(len(pos)),
        "detected": int(r.labels[pos].sum()),
        "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "n_windows": int(r.window_scores.shape[0]),
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }


def _multi_bench(n: int = 20_000, c: int = 5, n_segments: int = 10,
                 seg_len: int = 20) -> dict:
    """Multivariate sine + synchronous segments (all channels)."""
    rng = np.random.default_rng(1)
    t = np.arange(n)
    X = np.empty((n, c))
    for ch in range(c):
        X[:, ch] = np.sin(2 * np.pi * t / (100.0 + 20.0 * ch)) \
            + 0.1 * rng.standard_normal(n)
    starts = rng.choice(n - seg_len, size=n_segments, replace=False)
    pos = []
    for s in starts:
        X[s:s + seg_len, :] = 5.0
        pos.extend(range(s, s + seg_len))
    pos = np.array(sorted(pos))

    tracemalloc.start()
    t0 = time.perf_counter()
    r = fit_eif(X, n_estimators=100, window_size=30, stride=5,
                contamination=0.02, aggregate="max", random_state=0)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1))
    return {
        "mode": "multivariate_segments", "n": n, "n_channels": c,
        "n_segments": n_segments, "seg_len": seg_len,
        "n_anomaly_points": int(len(pos)),
        "detected": int(r.labels[pos].sum()),
        "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "n_windows": int(r.window_scores.shape[0]),
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }


def main() -> None:
    out = {"model": "eif", "uni": _uni_bench(), "multi": _multi_bench()}
    p = Path(".pingis/benches/eif.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

