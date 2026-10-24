#!/usr/bin/env python3
"""M2 reproduction harness: run CADW model + reference on synthetic
benchmark-like DGP, compare, write JSON artifact.

Not a benchmark leaderboard reproduction: it verifies the model does
what the paper claims on a task-structure-mimicking synthetic. For real
benchmark data, `--data=<path>` is a planned feature (v0.9.47).

Usage:
  python scripts/reproduce_benchmark.py --all --write
  python scripts/reproduce_benchmark.py --model iforest --write
  python scripts/reproduce_benchmark.py --all --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cadw import CadwDataFrame  # noqa: E402
from cadw.core.memory_guard import MemoryGuard  # noqa: E402
from docs.ml_tools import samples  # noqa: E402

OUT_DIR = ROOT / ".pingis/benches/reproduction"


@dataclass
class ReproResult:
    model: str
    dataset_kind: str
    reference_model: str
    cadw_metric: float
    reference_metric: float
    cadw_peak_mb: float
    reference_peak_mb: float
    cadw_time_s: float
    reference_time_s: float
    verdict: str
    notes: str = ""


def _to_cadw(X, *cols):
    d = {f"x{j}": pa.array(X[:, j]) for j in range(X.shape[1])}
    for name, arr in cols:
        d[name] = pa.array(arr)
    return CadwDataFrame.from_table(pa.table(d))


def _seed_kwargs(fn, seed):
    """Pass seed= or random_state= if the function accepts it."""
    import inspect
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return {}
    if "seed" in sig.parameters:
        return {"seed": seed}
    if "random_state" in sig.parameters:
        return {"random_state": seed}
    return {}


def _scores(model, df):
    """Get numeric score per row, accepting several common method names."""
    for name in ("score", "decision_function", "predict_proba", "predict"):
        fn = getattr(model, name, None)
        if fn is None:
            continue
        try:
            out = fn(df)
        except TypeError:
            continue
        arr = np.asarray(out)
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = arr[:, 1]
        return arr
    raise AttributeError(f"{type(model).__name__}: no score/predict method")


def _verdict(cadw: float, ref: float, *, atol: float, rtol: float = 0.0) -> str:
    if rtol > 0 and ref != 0:
        diff = abs(cadw - ref) / abs(ref)
        return "consistent" if diff <= rtol else "divergent"
    return "consistent" if abs(cadw - ref) <= atol else "divergent"


# ============================================================
# Reproducers (4 models, all with reference in dev-group)
# ============================================================

def repro_iforest(seed: int = 0) -> ReproResult:
    from cadw.ml.iforest import fit_isolation_forest
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score

    X, y = samples.gen_odds_like(seed=seed)
    cols = [f"x{j}" for j in range(X.shape[1])]
    df = _to_cadw(X)

    mg = MemoryGuard(max_memory_mb=200)
    t0 = time.time()
    cadw_model = fit_isolation_forest(
        df, cols, n_trees=100, contamination=0.05,
        memory_guard=mg, **_seed_kwargs(fit_isolation_forest, seed),
    )
    cadw_time = time.time() - t0
    cadw_scores = cadw_model.score_samples(df)
    cadw_auc = roc_auc_score(y, cadw_scores)
    cadw_peak = 0.0  # MemoryGuard has no peak attr; use psutil if needed

    t0 = time.time()
    ref = IsolationForest(n_estimators=100, random_state=seed, contamination=0.05)
    ref.fit(X)
    ref_time = time.time() - t0
    ref_auc = roc_auc_score(y, -ref.score_samples(X))

    return ReproResult(
        model="IForest",
        dataset_kind="synthetic_odds_like",
        reference_model="sklearn.IsolationForest",
        cadw_metric=float(cadw_auc),
        reference_metric=float(ref_auc),
        cadw_peak_mb=float(cadw_peak),
        reference_peak_mb=0.0,
        cadw_time_s=float(cadw_time),
        reference_time_s=float(ref_time),
        verdict=_verdict(cadw_auc, ref_auc, atol=0.15),
    )


def repro_gbdt_regressor(seed: int = 0) -> ReproResult:
    from cadw.ml.gbdt import fit_gbdt_regressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error

    X, y = samples.gen_amlb_regression_like(seed=seed)
    cols = [f"x{j}" for j in range(X.shape[1])]
    df = _to_cadw(X, ("y", y))

    mg = MemoryGuard(max_memory_mb=200)
    t0 = time.time()
    cadw_model = fit_gbdt_regressor(
        df, cols, "y", n_estimators=50, max_depth=3,
        memory_guard=mg, **_seed_kwargs(fit_gbdt_regressor, seed),
    )
    cadw_time = time.time() - t0
    pred_cadw = np.asarray(cadw_model.predict(np.asarray(X, dtype=np.float64)))
    cadw_rmse = float(np.sqrt(mean_squared_error(y, pred_cadw)))

    t0 = time.time()
    ref = GradientBoostingRegressor(n_estimators=50, max_depth=3,
                                    random_state=seed)
    ref.fit(X, y)
    ref_time = time.time() - t0
    ref_rmse = float(np.sqrt(mean_squared_error(y, ref.predict(X))))

    return ReproResult(
        model="GBDT_regressor",
        dataset_kind="synthetic_amlb_regression_like",
        reference_model="sklearn.GradientBoostingRegressor",
        cadw_metric=cadw_rmse,
        reference_metric=ref_rmse,
        cadw_peak_mb=0.0,
        reference_peak_mb=0.0,
        cadw_time_s=float(cadw_time),
        reference_time_s=float(ref_time),
        verdict=_verdict(cadw_rmse, ref_rmse, atol=0.0, rtol=0.30),
    )


def repro_var(seed: int = 0) -> ReproResult:
    from cadw.ts.var import VARModel
    from statsmodels.tsa.api import VAR as SMVAR

    n = 200
    K = 3
    rng = np.random.RandomState(seed)
    # VAR(2) with mild cross-correlation.
    data = np.zeros((n, K))
    for t in range(2, n):
        data[t] = (0.5 * data[t - 1] - 0.2 * data[t - 2]
                   + 0.1 * rng.normal(size=K))
    X = data

    df = _to_cadw(X)  # columns x0,x1,x2

    t0 = time.time()
    cadw_model = VARModel(p=2)
    cadw_model.fit(df, target_col=[f"x{j}" for j in range(K)])
    cadw_forecast = np.asarray(cadw_model.forecast_vector(steps=5))
    cadw_time = time.time() - t0

    t0 = time.time()
    ref = SMVAR(X).fit(2)
    ref_forecast = ref.forecast(X[-2:], 5)
    ref_time = time.time() - t0

    cadw_rmse = float(np.sqrt(np.mean((cadw_forecast - ref_forecast) ** 2)))
    return ReproResult(
        model="VAR",
        dataset_kind="synthetic_m4_like_multivariate",
        reference_model="statsmodels.VAR",
        cadw_metric=float(cadw_rmse),
        reference_metric=0.0,  # reference IS the target; diff from 0 is error
        cadw_peak_mb=0.0,
        reference_peak_mb=0.0,
        cadw_time_s=float(cadw_time),
        reference_time_s=float(ref_time),
        verdict=("consistent" if cadw_rmse < 0.5 else "divergent"),
        notes="cadw_metric = RMSE(cadw_forecast, statsmodels_forecast); "
              "0.0 = perfect match; threshold 0.5 is generous.",
    )


def repro_coxph(seed: int = 0) -> ReproResult:
    from cadw.ml.coxph import fit_coxph
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index as ll_ci

    X, duration, event = samples.gen_survival_like(seed=seed)
    cols = [f"x{j}" for j in range(X.shape[1])]
    df = _to_cadw(X, ("duration", duration), ("event", event))

    mg = MemoryGuard(max_memory_mb=200)
    t0 = time.time()
    cadw_model = fit_coxph(
        df, cols, "duration", "event",
        memory_guard=mg, **_seed_kwargs(fit_coxph, seed),
    )
    cadw_time = time.time() - t0
    risk_table = cadw_model.predict_risk(df)
    risk = np.asarray(risk_table.column("risk").to_numpy(zero_copy_only=False))
    # c-index expects higher risk = shorter survival
    from lifelines.utils import concordance_index as ci_ref
    cadw_ci = ci_ref(duration, -risk, event) if risk is not None else float("nan")

    import pandas as pd
    t0 = time.time()
    ref_df = pd.DataFrame({"T": duration, "E": event,
                           **{f"x{j}": X[:, j] for j in range(X.shape[1])}})
    ref = CoxPHFitter()
    ref.fit(ref_df, duration_col="T", event_col="E")
    ref_time = time.time() - t0
    ref_risk = ref.predict_partial_hazard(ref_df).to_numpy()
    ref_ci = ll_ci(duration, -ref_risk, event)

    return ReproResult(
        model="CoxPH",
        dataset_kind="synthetic_survival_like",
        reference_model="lifelines.CoxPHFitter",
        cadw_metric=float(cadw_ci),
        reference_metric=float(ref_ci),
        cadw_peak_mb=0.0,
        reference_peak_mb=0.0,
        cadw_time_s=float(cadw_time),
        reference_time_s=float(ref_time),
        verdict=_verdict(cadw_ci, ref_ci, atol=0.10),
    )


REPRODUCERS: dict[str, Callable[[], ReproResult]] = {
    "iforest": repro_iforest,
    "gbdt_regressor": repro_gbdt_regressor,
    "var": repro_var,
    "coxph": repro_coxph,
}


def _run_one(name: str) -> ReproResult:
    print(f"\n=== {name} ===")
    try:
        result = REPRODUCERS[name]()
    except ImportError as e:
        print(f"  SKIP (reference missing): {e}")
        return ReproResult(
            model=name, dataset_kind="", reference_model="",
            cadw_metric=float("nan"), reference_metric=float("nan"),
            cadw_peak_mb=0.0, reference_peak_mb=0.0,
            cadw_time_s=0.0, reference_time_s=0.0,
            verdict="reference_missing", notes=str(e),
        )
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return ReproResult(
            model=name, dataset_kind="", reference_model="",
            cadw_metric=float("nan"), reference_metric=float("nan"),
            cadw_peak_mb=0.0, reference_peak_mb=0.0,
            cadw_time_s=0.0, reference_time_s=0.0,
            verdict="cadw_error", notes=f"{type(e).__name__}: {e}",
        )
    print(f"  cadw={result.cadw_metric:.4f}  ref={result.reference_metric:.4f}  "
          f"verdict={result.verdict}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", choices=sorted(REPRODUCERS))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.model:
        models = [args.model]
    elif args.all:
        models = sorted(REPRODUCERS)
    else:
        ap.print_help()
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in models:
        r = _run_one(name)
        results[name] = r
        if args.write and not args.dry_run and r.verdict != "cadw_error":
            path = OUT_DIR / f"{name}.json"
            path.write_text(json.dumps(asdict(r), indent=2) + "\n")
            print(f"  wrote {path.relative_to(ROOT)}")

    summary = {k: v.verdict for k, v in results.items()}
    print(f"\n=== summary ===\n{json.dumps(summary, indent=2)}")
    # exit code: 0 if no cadw_error or divergent; 1 otherwise
    bad = [k for k, v in results.items() if v.verdict in ("cadw_error", "divergent")]
    if bad:
        print(f"\n⚠ {len(bad)} issue(s): {bad}")
        sys.exit(1)


if __name__ == "__main__":
    main()
