#!/usr/bin/env python3
"""Benchmark PatchMemoryAD on synthetic TS anomaly DGP."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from cadw.ml.patch_memory_ad import fit_patch_memory_ad


def main() -> None:
    n = 30_000
    n_spikes = 20
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += 20.0

    tracemalloc.start()
    t0 = time.perf_counter()
    r = fit_patch_memory_ad(x, window_size=50, patch_len=10,
                            hidden_size=32, latent_size=16,
                            n_memory=64, n_epochs=5, lr=0.005,
                            contamination=0.05, aggregate="max",
                            random_state=42)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1))
    out = {
        "model": "patch_memory_ad",
        "n": n, "n_spikes": n_spikes,
        "detected": int(r.labels[pos].sum()),
        "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "window_size": r.window_size,
        "patch_len": r.patch_len,
        "hidden_size": r.hidden_size,
        "latent_size": r.latent_size,
        "n_memory": r.n_memory,
        "n_epochs": r.n_epochs,
        "lr": r.lr,
        "contamination": r.contamination,
        "threshold": round(float(r.threshold), 4),
        "first_loss": round(float(r.training_losses[0]), 6),
        "last_loss": round(float(r.training_losses[-1]), 6),
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
        "specialization": "point anomaly (patches; segments are not "
                          "detected — see module docstring)",
        "training_note": "loss can remain 0 (triplet margin inactive) "
                         "when memory warmup already gives close pos + "
                         "far neg; scoring still works via min-distance "
                         "to memory. Not a bug.",
    }
    p = Path(".pingis/benches/patch_memory_ad.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

