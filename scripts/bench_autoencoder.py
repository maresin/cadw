#!/usr/bin/env python3
"""Benchmark MLP Autoencoder on synthetic TS anomaly DGP."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from cadw.ml.autoencoder import fit_autoencoder


def main() -> None:
    n = 50_000
    n_spikes = 30
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.1 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += 20.0

    tracemalloc.start()
    t0 = time.perf_counter()
    # Contamination 0.08 matches window-spread: 30 spikes × 50 window × 2
    # (spread ±50) / 50000 = 6%, use 8% for margin.
    r = fit_autoencoder(x, window_size=50, hidden_size=32, latent_size=8,
                        n_epochs=30, lr=0.01, batch_size=64,
                        contamination=0.08, aggregate="max",
                        reconstruction_error="median", random_state=42)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = float(np.mean(r.labels[pos] == 1))
    out = {
        "model": "autoencoder",
        "n": n, "n_spikes": n_spikes,
        "detected": int(r.labels[pos].sum()),
        "recall": round(recall, 4),
        "total_flagged": int(r.labels.sum()),
        "window_size": r.window_size,
        "hidden_size": r.hidden_size,
        "latent_size": r.latent_size,
        "n_epochs": r.n_epochs,
        "lr": r.lr,
        "batch_size": r.batch_size,
        "contamination": r.contamination,
        "aggregate": r.aggregate,
        "reconstruction_error": r.reconstruction_error,
        "threshold": round(float(r.threshold), 6),
        "first_loss": round(float(r.training_losses[0]), 6),
        "last_loss": round(float(r.training_losses[-1]), 6),
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }
    p = Path(".pingis/benches/autoencoder.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

