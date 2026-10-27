#!/usr/bin/env python3
"""Benchmark EnsembleForecaster vs individual members on 2 DGPs."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pyarrow as pa
from cadw import CadwDataFrame
from cadw.ts.ensemble import EnsembleForecaster
from cadw.ts.theta import ThetaForecaster
from cadw.ts.auto_theta import AutoThetaForecaster
from cadw.ts.auto_ets import AutoETSForecaster
from cadw.ts.models import NaiveForecaster


def _seasonal(n, s=12, seed=0, trend=0.05, season_amp=5.0):
    t = np.arange(n + s)
    rng = np.random.RandomState(seed)
    return (20 + trend * t + season_amp * np.sin(2 * np.pi * t / s)
            + rng.normal(0, 0.5, n + s))


def _df(y):
    return CadwDataFrame.from_table(pa.table({"y": pa.array(y)}))


def _mase(y_true, y_pred, y_hist):
    denom = np.mean(np.abs(np.diff(y_hist)))
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def _fit_predict(fitter, y, truth):
    t0 = time.perf_counter()
    m = fitter().fit(_df(y), "y")
    f = m.predict(truth.size)
    elapsed = time.perf_counter() - t0
    return _mase(truth, f, y), round(elapsed, 3)


def main():
    n, s = 500, 12
    full = _seasonal(n + s, s=s, seed=0)
    y, truth = full[:-s], full[-s:]
    print(f"n={n}, s={s}, truth mean={truth.mean():.2f}", file=sys.stderr)

    results = {}

    for name, fitter in (
            ("naive_last", lambda: NaiveForecaster(strategy="last")),
            ("theta", lambda: ThetaForecaster(seasonal_period=s)),
            ("auto_theta", lambda: AutoThetaForecaster(seasonal_period=s)),
    ):
        try:
            mase, t = _fit_predict(fitter, y, truth)
            results[name] = {"mase": round(mase, 4), "fit_predict_s": t}
            print(f"  {name}: MASE={mase:.4f} ({t}s)", file=sys.stderr)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {e}"}

    # Ensemble: naive + theta + auto_theta with inverse_mase
    try:
        t0 = time.perf_counter()
        ens = EnsembleForecaster(
            members=[NaiveForecaster(strategy="last"),
                     ThetaForecaster(seasonal_period=s),
                     AutoThetaForecaster(seasonal_period=s)],
            weights="inverse_mase",
            seasonal_periods=[s],
        ).fit(_df(y), "y")
        f = ens.predict(truth.size)
        t = time.perf_counter() - t0
        mase = _mase(truth, f, y)
        results["ensemble_inverse_mase"] = {
            "mase": round(mase, 4),
            "fit_predict_s": round(t, 3),
            "member_weights": [round(w, 4) for w in
                                ens.member_weights_.tolist()],
            "member_val_mase": [round(v, 4) if np.isfinite(v) else None
                                 for v in ens.member_val_mase_.tolist()],
        }
        print(f"  ensemble: MASE={mase:.4f} ({t:.2f}s), weights="
              f"{results['ensemble_inverse_mase']['member_weights']}",
              file=sys.stderr)
    except Exception as e:
        results["ensemble_inverse_mase"] = {
            "error": f"{type(e).__name__}: {e}"}

    # AutoETS only for comparison (slow)
    try:
        mase, t = _fit_predict(
            lambda: AutoETSForecaster(seasonal_period=s), y, truth)
        results["auto_ets"] = {"mase": round(mase, 4),
                                "fit_predict_s": t}
        print(f"  auto_ets: MASE={mase:.4f} ({t}s)", file=sys.stderr)
    except Exception as e:
        results["auto_ets"] = {"error": f"{type(e).__name__}: {e}"}

    out = {
        "n": n, "seasonal_period": s,
        "dgp": "m3_like (trend=0.05, season_amp=5)",
        "results": results,
    }
    p = Path(".pingis/benches/ensemble.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

