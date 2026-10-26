#!/usr/bin/env python3
"""Benchmark S2G (paper-faithful) on synthetic TS anomaly DGP."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from cadw.ml.s2g import fit_s2g


def _bench_segments(n: int = 50_000, n_segments: int = 20,
                    seg_len: int = 30) -> dict:
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    starts = rng.choice(n - seg_len, size=n_segments, replace=False)
    pos = []
    for s in starts:
        x[s:s + seg_len] = 5.0
        pos.extend(range(s, s + seg_len))
    pos = np.array(sorted(pos))

    tracemalloc.start()
    t0 = time.perf_counter()
    r = fit_s2g(x, window_size=50, query_window_size=75,
                n_nodes=50, contamination=0.02)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1))
    return {
        "mode": "segments",
        "n": n, "n_segments": n_segments, "seg_len": seg_len,
        "n_anomaly_points": int(len(pos)),
        "detected": int(r.labels[pos].sum()),
        "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }


def _bench_motifs(n: int = 50_000, n_motifs: int = 20,
                  motif_len: int = 60) -> dict:
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    starts = rng.choice(n - motif_len - 10, size=n_motifs,
                        replace=False) + 5
    pos = []
    for s in starts:
        lt = np.arange(motif_len)
        x[s:s + motif_len] = np.sin(2 * np.pi * lt / 50.0) \
            + 0.1 * rng.standard_normal(motif_len)
        pos.extend(range(s, s + motif_len))
    pos = np.array(sorted(pos))

    tracemalloc.start()
    t0 = time.perf_counter()
    r = fit_s2g(x, window_size=50, query_window_size=75,
                n_nodes=50, contamination=0.02)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1))
    return {
        "mode": "motifs",
        "n": n, "n_motifs": n_motifs, "motif_len": motif_len,
        "n_anomaly_points": int(len(pos)),
        "detected": int(r.labels[pos].sum()),
        "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }


def main() -> None:
    seg = _bench_segments()
    mot = _bench_motifs()
    out = {
        "model": "s2g",
        "params": {
            "window_size": 50,
            "query_window_size": 75,
            "convolution_size": 16,
            "n_nodes": 50,
            "contamination": 0.02,
        },
        "segments": seg,
        "motifs": mot,
        "note": (
            "Paper-faithful S2G: local convolution, 3D PCA + rotation, "
            "edge normalcy = w*(deg-1), query path scoring. One "
            "documented deviation: uniform angle bins instead of KDE "
            "peaks (deterministic node count; comparable on synthetic)."
        ),
    }
    p = Path(".pingis/benches/s2g.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

