#!/usr/bin/env python3
"""Benchmark AutoTheta vs STM vs OTM vs AutoETS on 2 DGPs."""
from __future__ import annotations
import json, sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pyarrow as pa
from cadw import CadwDataFrame
from cadw.ts.auto_theta import AutoThetaForecaster
from cadw.ts.theta import ThetaForecaster
from cadw.ts.otm import OTMThetaForecaster
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

    results = {}
    for dgp_name, kwargs in (
            ("m3_like", {"trend": 0.05, "season_amp": 5.0}),
            ("trend_dominated", {"trend": 0.3, "season_amp": 3.0})):
        full = _seasonal(n + s, s=s, seed=0, **kwargs)
        y, truth = full[:-s], full[-s:]
        print(f"=== {dgp_name} ===", file=sys.stderr)

        m_auto, mase_auto, t_auto, pk_auto = _run(
            lambda: AutoThetaForecaster(seasonal_period=s).fit(_df(y), "y"),
            y, truth)
        print(f"  auto_theta: mase={mase_auto}, t={t_auto}s, "
              f"selected={m_auto.selected_config_['label']}",
              file=sys.stderr)

        _, mase_stm, t_stm, _ = _run(
            lambda: ThetaForecaster(seasonal_period=s).fit(_df(y), "y"),
            y, truth)
        print(f"  stm: mase={mase_stm}, t={t_stm}s", file=sys.stderr)

        _, mase_otm, t_otm, _ = _run(
            lambda: OTMThetaForecaster(seasonal_period=s).fit(_df(y), "y"),
            y, truth)
        print(f"  otm: mase={mase_otm}, t={t_otm}s", file=sys.stderr)

        ets = None
        if dgp_name == "m3_like":
            _, mase_ets, t_ets, _ = _run(
                lambda: AutoETSForecaster(seasonal_period=s).fit(_df(y), "y"),
                y, truth)
            print(f"  auto_ets: mase={mase_ets}, t={t_ets}s", file=sys.stderr)
            ets = {"mase": mase_ets, "fit_predict_s": t_ets}

        results[dgp_name] = {
            "auto_theta": {
                "mase": mase_auto,
                "fit_predict_s": t_auto,
                "peak_rss_mb": pk_auto,
                "selected_config": m_auto.selected_config_,
                "cv_scores": m_auto.cv_scores_,
            },
            "stm": {"mase": mase_stm, "fit_predict_s": t_stm},
            "otm": {"mase": mase_otm, "fit_predict_s": t_otm},
            "auto_ets": ets,
        }

    out = {
        "n": n, "seasonal_period": s,
        "note": (
            "AutoTheta evaluates 4 candidates (STM/OTM x additive/"
            "multiplicative) on holdout MASE, refits winner. Should "
            "select STM on M3-like (STM wins slightly) and OTM on "
            "trend-dominated (OTM 2.2x better). Never worse than the "
            "best candidate on val by construction."
        ),
        "results": results,
    }
    p = Path(".pingis/benches/auto_theta.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

