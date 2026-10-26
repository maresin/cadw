#!/usr/bin/env python3
"""Benchmark AutoETS on synthetic seasonal DGP."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pyarrow as pa
from cadw import CadwDataFrame
from cadw.ts.auto_ets import AutoETSForecaster


def _seasonal(n, s=12, seed=0):
    t = np.arange(n + s)
    rng = np.random.RandomState(seed)
    return (20 + 0.3 * t + 3 * np.sin(2 * np.pi * t / s)
            + rng.normal(0, 0.5, n + s))


def _df(y):
    return CadwDataFrame.from_table(pa.table({"y": pa.array(y)}))


def main() -> None:
    # v0.9.76b-fix1: n=500 (not 2000) because AutoETS grid search is
    # O(n_configs * grid_size * n) Python-loop. At n=2000 this exceeds
    # the 120s timeout. Production users with n>2000 should expect
    # proportional fit time.
    n = 500
    s = 12
    full = _seasonal(n + s, s=s, seed=0)
    y, truth = full[:-s], full[-s:]

    tracemalloc.start()
    t0 = time.perf_counter()
    m = AutoETSForecaster(seasonal_period=s).fit(_df(y), "y")
    f = m.predict(s)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mase = float(np.mean(np.abs(f - truth)) / np.mean(np.abs(np.diff(y))))

    out = {
        "model": "auto_ets",
        "n": n, "seasonal_period": s,
        "n_configs_evaluated": len(m.aicc_scores_),
        "selected_config": m.selected_config_,
        "best_aicc": (None if not np.isfinite(m.best_aicc_)
                      else round(float(m.best_aicc_), 3)),
        "aicc_scores": [
            {k: (round(v, 3) if isinstance(v, float) else v)
             for k, v in s_.items()}
            for s_ in m.aicc_scores_
        ],
        "mase": round(mase, 4),
        "fit_predict_s": round(elapsed, 3),
        "peak_rss_mb": round(peak / 1024 / 1024, 2),
        "state_b": int(m.state_nbytes()),
    }
    p = Path(".pingis/benches/auto_ets.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

