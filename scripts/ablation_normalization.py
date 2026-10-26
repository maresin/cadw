#!/usr/bin/env python3
"""Ablation v2: normalization strategies on 4 DGPs (per HYDRA §1)."""
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


def main():
    n = 30_000
    n_spikes = 20
    contamination = 0.08
    configs = [
        ("easy", 20.0, 0.1),
        ("subtle", 5.0, 0.1),
        ("noisy", 20.0, 0.5),
        ("hard", 5.0, 0.5),
    ]
    results = []
    for name, amplitude, noise in configs:
        x, pos = make_dgp(n, n_spikes, amplitude, noise)
        print(f"Running {name}...", file=sys.stderr)
        row = {"config": name, "amplitude": amplitude, "noise": noise}
        # PMA x 3 normalization modes
        for norm in ("none", "window", "patch"):
            try:
                r = fit_patch_memory_ad(x, contamination=contamination,
                                        normalization=norm, random_state=0)
                row[f"pma_{norm}"] = round(
                    float(r.labels[pos].mean()), 4)
            except Exception as e:
                row[f"pma_{norm}"] = str(e)[:40]
        # AE x 2 modes
        for norm in ("none", "window"):
            try:
                r = fit_autoencoder(x, contamination=contamination,
                                    reconstruction_error="median",
                                    normalization=norm, random_state=0)
                row[f"ae_{norm}"] = round(
                    float(r.labels[pos].mean()), 4)
            except Exception as e:
                row[f"ae_{norm}"] = str(e)[:40]
        # Baselines (no normalization, best configs)
        try:
            r = fit_sub_pca(x, n_components=5, contamination=contamination,
                            aggregate="mean")
            row["sub_pca"] = round(float(r.labels[pos].mean()), 4)
        except Exception as e:
            row["sub_pca"] = str(e)[:40]
        try:
            r = fit_kmeans_ad(x, contamination=contamination, random_state=0)
            row["kmeans_ad"] = round(float(r.labels[pos].mean()), 4)
        except Exception as e:
            row["kmeans_ad"] = str(e)[:40]
        results.append(row)
        print(f"  {row}", file=sys.stderr)

    out = {"n": n, "n_spikes": n_spikes,
           "contamination": contamination, "results": results}
    p = Path(".pingis/benches/ablation_normalization.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

