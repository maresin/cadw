#!/usr/bin/env python3
"""Benchmark OTM vs STM vs AutoETS on synthetic seasonal DGP."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pyarrow as pa
from cadw import CadwDataFrame
from cadw.ts.otm import OTMThetaForecaster
from cadw.ts.theta import ThetaForecaster
from cadw.ts.auto_ets import AutoETSForecaster


def _seasonal(n, s=12, seed=0, trend=0.05, season_amp=5.0):
    t = np.arange(n + s)
    rng = np.random.RandomState(seed)
    return (20 + trend * t + season_amp * np.sin(2 * np.pi * t / s)
            + rng.normal(0, 0.5, n + s))


def _df(y):
    return CadwDataFrame.from_table(pa.table({"y": pa.array(y)}))


def _run(fitter, y, truth):
    tracemalloc.start()
    t0 = time.perf_counter()
    m = fitter()
    f = m.predict(truth.size)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mase = float(np.mean(np.abs(f - truth))
                 / np.mean(np.abs(np.diff(y))))
    return m, round(mase, 4), round(elapsed, 3), round(peak / 1024 / 1024, 2)


def main() -> None:
    n, s = 500, 12

    # M3-like DGP (STM-favorable)
    full_m3 = _seasonal(n + s, s=s, seed=0, trend=0.05, season_amp=5.0)
    y_m3, truth_m3 = full_m3[:-s], full_m3[-s:]

    # Trend-dominated DGP (STM-weak)
    full_tr = _seasonal(n + s, s=s, seed=0, trend=0.3, season_amp=3.0)
    y_tr, truth_tr = full_tr[:-s], full_tr[-s:]

    results = {"m3_like": {}, "trend_dominated": {}}

    for dgp_name, (y, truth) in (
            ("m3_like", (y_m3, truth_m3)),
            ("trend_dominated", (y_tr, truth_tr))):
        print(f"=== {dgp_name} ===", file=sys.stderr)

        m_otm, mase_otm, t_otm, pk_otm = _run(
            lambda: OTMThetaForecaster(seasonal_period=s).fit(_df(y), "y"),
            y, truth)
        print(f"  otm: mase={mase_otm}, t={t_otm}s", file=sys.stderr)

        _, mase_stm, t_stm, _ = _run(
            lambda: ThetaForecaster(seasonal_period=s).fit(_df(y), "y"),
            y, truth)
        print(f"  stm: mase={mase_stm}, t={t_stm}s", file=sys.stderr)

        # AutoETS (slow, skip on trend-dominated for speed)
        if dgp_name == "m3_like":
            _, mase_ets, t_ets, _ = _run(
                lambda: AutoETSForecaster(seasonal_period=s).fit(_df(y), "y"),
                y, truth)
            print(f"  auto_ets: mase={mase_ets}, t={t_ets}s", file=sys.stderr)
        else:
            mase_ets, t_ets = None, None

        results[dgp_name] = {
            "otm": {"mase": mase_otm, "fit_predict_s": t_otm,
                    "peak_rss_mb": pk_otm,
                    "w2": m_otm.w2_, "alpha": m_otm.alpha_},
            "stm": {"mase": mase_stm, "fit_predict_s": t_stm},
            "auto_ets": ({"mase": mase_ets, "fit_predict_s": t_ets}
                         if mase_ets is not None else None),
        }

    out = {
        "n": n, "seasonal_period": s,
        "note": (
            "OTM: weight-optimized STM (theta=2 fixed, w2/alpha grid "
            "optimized on in-sample MSE). Expected pattern: OTM matches "
            "STM on M3-like (w2 ~= 0.5), beats STM on trend-dominated "
            "(w2 < 0.5, trusts OLS trend). Full Fiorucci OTM (theta "
            "optimization + drift term) is backlog; this is the practical "
            "weight-optimized variant."
        ),
        "results": results,
    }
    p = Path(".pingis/benches/otm.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

