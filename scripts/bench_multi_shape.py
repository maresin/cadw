#!/usr/bin/env python3
"""Benchmark multi-channel shape: 4 models (EIF + Sub-PCA + KMeansAD + Autoencoder)."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from cadw.ml.eif import fit_eif
from cadw.ml.sub_pca import fit_sub_pca
from cadw.ml.kmeans_ad import fit_kmeans_ad
from cadw.ml.autoencoder import fit_autoencoder
from docs.ml_tools.samples import gen_ts_shape_anomaly_multi


def _measure(fn, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    r = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return r, elapsed, peak


def _pack(r, pos, elapsed, peak):
    return {
        "recall": round(float(np.mean(r.labels[pos] == 1)), 4),
        "detected": int(r.labels[pos].sum()),
        "n_anomaly_points": int(len(pos)),
        "total_flagged": int(r.labels.sum()),
        "fit_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
    }


def main() -> None:
    n, c = 5000, 3
    n_segments, seg_len = 5, 50
    contamination = 0.05
    X, pos = gen_ts_shape_anomaly_multi(
        n=n, n_channels=c, n_segments=n_segments, seg_len=seg_len,
        anomaly_noise=3.0, seed=0)
    print(f"n={n}, c={c}, n_anom={len(pos)}", file=sys.stderr)

    results = {}

    r, t, p = _measure(fit_eif, X, n_estimators=100, window_size=30,
                       stride=1, contamination=contamination,
                       aggregate="max", random_state=0)
    results["eif"] = _pack(r, pos, t, p)
    print(f"eif: {results['eif']}", file=sys.stderr)

    r, t, p = _measure(fit_sub_pca, X, window_size=30, n_components=10,
                       contamination=contamination, aggregate="mean")
    results["sub_pca"] = _pack(r, pos, t, p)
    print(f"sub_pca: {results['sub_pca']}", file=sys.stderr)

    r, t, p = _measure(fit_kmeans_ad, X, window_size=30, n_clusters=8,
                       contamination=contamination, aggregate="max",
                       random_state=0)
    results["kmeans_ad"] = _pack(r, pos, t, p)
    print(f"kmeans_ad: {results['kmeans_ad']}", file=sys.stderr)

    r, t, p = _measure(
        fit_autoencoder, X, window_size=30, hidden_size=32, latent_size=8,
        n_epochs=30, lr=0.01, batch_size=64,
        contamination=contamination, aggregate="max",
        reconstruction_error="median", normalization="none",
        random_state=0)
    results["autoencoder"] = _pack(r, pos, t, p)
    print(f"autoencoder: {results['autoencoder']}", file=sys.stderr)

    out = {
        "dgp": "gen_ts_shape_anomaly_multi",
        "n": n, "n_channels": c,
        "n_segments": n_segments, "seg_len": seg_len,
        "contamination": contamination,
        "n_anomaly_points": int(len(pos)),
        "models": results,
    }
    p = Path(".pingis/benches/multi_shape.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

