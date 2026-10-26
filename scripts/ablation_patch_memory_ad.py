#!/usr/bin/env python3
"""Ablation: does PatchMemoryAD simplification degrade vs alternatives?

Compares PatchMemoryAD (MLP+memory+triplet) against:
  - Autoencoder (MLP reconstruction, no memory/patch)
  - Sub-PCA (linear reconstruction, no MLP)
  - KMeansAD (clustering)

On 4 DGPs of increasing difficulty:
  easy:    amplitude=20, noise=0.1  (bench)
  subtle:  amplitude=5,  noise=0.1
  noisy:   amplitude=20, noise=0.5
  hard:    amplitude=5,  noise=0.5
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from cadw.ml.patch_memory_ad import fit_patch_memory_ad
from cadw.ml.autoencoder import fit_autoencoder
from cadw.ml.sub_pca import fit_sub_pca
from cadw.ml.kmeans_ad import fit_kmeans_ad


def make_dgp(n, n_spikes, amplitude, noise, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + noise * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += amplitude
    return x, pos


def run_one(x, pos, contamination):
    out = {}
    # PatchMemoryAD
    t0 = time.perf_counter()
    try:
        r = fit_patch_memory_ad(x, contamination=contamination,
                                normalize_windows=False, random_state=0)
        out["patch_memory_ad"] = {
            "recall": round(float(r.labels[pos].mean()), 4),
            "time_s": round(time.perf_counter() - t0, 2),
        }
    except Exception as e:
        out["patch_memory_ad"] = {"error": str(e)[:80]}
    # Autoencoder
    t0 = time.perf_counter()
    try:
        r = fit_autoencoder(x, contamination=contamination,
                            reconstruction_error="median",
                            normalize_windows=False, random_state=0)
        out["autoencoder"] = {
            "recall": round(float(r.labels[pos].mean()), 4),
            "time_s": round(time.perf_counter() - t0, 2),
        }
    except Exception as e:
        out["autoencoder"] = {"error": str(e)[:80]}
    # Sub-PCA (k=5, mean aggregation — best for point)
    t0 = time.perf_counter()
    try:
        r = fit_sub_pca(x, n_components=5, contamination=contamination,
                        aggregate="mean")
        out["sub_pca"] = {
            "recall": round(float(r.labels[pos].mean()), 4),
            "time_s": round(time.perf_counter() - t0, 2),
        }
    except Exception as e:
        out["sub_pca"] = {"error": str(e)[:80]}
    # KMeansAD
    t0 = time.perf_counter()
    try:
        r = fit_kmeans_ad(x, contamination=contamination, random_state=0)
        out["kmeans_ad"] = {
            "recall": round(float(r.labels[pos].mean()), 4),
            "time_s": round(time.perf_counter() - t0, 2),
        }
    except Exception as e:
        out["kmeans_ad"] = {"error": str(e)[:80]}
    return out


def main():
    n = 30_000
    n_spikes = 20
    contamination = 0.08  # affected: 20*100/30000 = 6.7%

    configs = [
        ("easy", 20.0, 0.1),
        ("subtle", 5.0, 0.1),
        ("noisy", 20.0, 0.5),
        ("hard", 5.0, 0.5),
    ]
    results = []
    for name, amplitude, noise in configs:
        x, pos = make_dgp(n, n_spikes, amplitude, noise)
        print(f"Running {name} (amp={amplitude}, noise={noise})...",
              file=sys.stderr)
        r = run_one(x, pos, contamination)
        results.append({
            "config": name,
            "amplitude": amplitude, "noise": noise,
            "n": n, "n_spikes": n_spikes,
            "contamination": contamination,
            **r,
        })
        print(f"  {name}: " + ", ".join(
            f"{k}={v.get('recall', v.get('error','?'))}"
            for k, v in r.items()), file=sys.stderr)

    out = {"n": n, "n_spikes": n_spikes,
           "contamination": contamination, "results": results}
    p = Path(".pingis/benches/ablation_patch_memory_ad.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

