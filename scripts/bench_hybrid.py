"""Bench: Hybrid ETS+GBDT vs ETS vs GBDT on two synthetic DGPs.

  - ets_friendly:       trend + season + iid noise
  - nonlinear_residual: trend + season + AR(1) phi=-0.6 residual

Usage:
  uv run python scripts/bench_hybrid.py
  uv run python scripts/bench_hybrid.py --write
"""
import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pyarrow as pa

from cadw import CadwDataFrame
from cadw.ts import (
    ETSForecaster,
    GBDTForecaster,
    HybridETSGBDTForecaster,
)


def _mk(y):
    return CadwDataFrame.from_table(
        pa.table({'y': pa.array(y.astype(np.float64))}))


def _mase(y_hat, y_true, y_train, m=12):
    scale = float(np.mean(np.abs(y_train[m:] - y_train[:-m])))
    if scale == 0 or not np.isfinite(scale):
        return float("nan")
    return float(np.mean(np.abs(y_hat - y_true)) / scale)


def _dgp_ets(n, s, seed):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    return 20 + 0.3 * t + 3 * np.sin(2 * np.pi * t / s) + rng.normal(0, 0.3, n)


def _dgp_nonlinear_residual(n, s, seed):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = -0.6 * resid[i - 1] + rng.normal(0, 0.5)
    return 20 + 0.3 * t + 3 * np.sin(2 * np.pi * t / s) + resid


def _run_one(factory, y_train, y_test, s=12):
    df_tr = _mk(y_train)
    t0 = time.time()
    fc = factory().fit(df_tr, 'y').predict(len(y_test))
    return _mase(fc, y_test, y_train, m=s), time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    n = 500
    s = 12
    horizon = 12
    results = {}

    for dgp_name, dgp_fn in [
        ('ets_friendly', _dgp_ets),
        ('nonlinear_residual', _dgp_nonlinear_residual),
    ]:
        ets_mases, hyb_mases, gbdt_mases = [], [], []
        ets_times, hyb_times, gbdt_times = [], [], []
        ets_kw = {'seasonal_period': s, 'seasonal': 'additive'}
        for seed in range(args.seeds):
            y_full = dgp_fn(n + horizon, s, seed=seed)
            y_train, y_test = y_full[:-horizon], y_full[-horizon:]
            m_ets, t_ets = _run_one(
                lambda: ETSForecaster(**ets_kw), y_train, y_test, s=s)
            m_hyb, t_hyb = _run_one(
                lambda: HybridETSGBDTForecaster(ets_kwargs=ets_kw),
                y_train, y_test, s=s)
            m_gbdt, t_gbdt = _run_one(
                lambda: GBDTForecaster(
                    mode='recursive',
                    lags=(1, 2, 3, s, 2 * s)),
                y_train, y_test, s=s)
            ets_mases.append(m_ets); hyb_mases.append(m_hyb)
            gbdt_mases.append(m_gbdt)
            ets_times.append(t_ets); hyb_times.append(t_hyb)
            gbdt_times.append(t_gbdt)

        results[dgp_name] = {
            'ets': {'mase_mean': float(np.mean(ets_mases)),
                    'mase_std': float(np.std(ets_mases)),
                    'fit_time_s_mean': float(np.mean(ets_times))},
            'hybrid': {'mase_mean': float(np.mean(hyb_mases)),
                       'mase_std': float(np.std(hyb_mases)),
                       'fit_time_s_mean': float(np.mean(hyb_times))},
            'gbdt': {'mase_mean': float(np.mean(gbdt_mases)),
                     'mase_std': float(np.std(gbdt_mases)),
                     'fit_time_s_mean': float(np.mean(gbdt_times))},
        }

        print()
        print(f"[{dgp_name}] n={n}, s={s}, h={horizon}, seeds={args.seeds}")
        for name, r in results[dgp_name].items():
            print(f"  {name:10s} MASE={r['mase_mean']:.4f} +/- "
                  f"{r['mase_std']:.4f}  "
                  f"fit={r['fit_time_s_mean']:.3f}s")
        ets_m = results[dgp_name]['ets']['mase_mean']
        hyb_m = results[dgp_name]['hybrid']['mase_mean']
        delta = (hyb_m - ets_m) / ets_m * 100.0
        print(f"  hybrid vs ets: {delta:+.1f}%")

    if args.write:
        out = Path('.pingis/benches/hybrid.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print()
        print(f"Wrote {out}")


if __name__ == '__main__':
    main()
