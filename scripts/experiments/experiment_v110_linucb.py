"""Calibrate fit_linucb ram_formula (v0.9.110b, EXPERIMENT_PROCEDURE §3).

Hypothesis formula v1 (from code):
  peak = 4*batch*row_size + K*d^2*f64 + K*d*f64
  row_size = d*8 (X) + 8 (arm) + 8 (reward) = 8d+16
  coef_batch = 4.0 = _stack_numeric(2x: parts+result) + held_X(1x) + arrow_batch(1x)

Stages:
  0 baseline noise
  2 batch-sweep -> coef_batch
  3 d-sweep (K fixed) -> state grows with d^2
  4 K-sweep (d fixed) -> state grows with K
  5 OOC (batch fixed) -> peak const in n
  6 matrix 9/9 in [0.7, 1.4]

Same measurement policy as experiment_v110_iforest.py:
  fresh subprocess per cell + prewarm + gc.disable + rss_bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded, measure_peak
from cadw.ml.linucb import fit_linucb


def predicted_peak(batch: int, d: int, K: int = 5) -> int:
    """v1: 4*batch*row_size + K*d^2*8 + K*d*8 (row_size = 8d+16)."""
    row_size = 8 * d + 16
    return 4 * batch * row_size + K * d * d * 8 + K * d * 8


def make_df_parquet(n: int, d: int, K: int = 5, seed: int = 0,
                     row_group: int = 1000):
    tmpdir = tempfile.TemporaryDirectory(prefix="lu_exp_")
    rng = np.random.RandomState(seed)
    path = Path(tmpdir.name) / "data.parquet"
    theta = rng.normal(size=(K, d)) * 0.5
    writer = None
    try:
        for start in range(0, n, row_group):
            end = min(start + row_group, n)
            b_n = end - start
            X = rng.normal(size=(b_n, d)).astype(np.float64)
            arms = rng.randint(0, K, size=b_n).astype(np.int64)
            mean_r = np.einsum("ij,ij->i", X, theta[arms])
            r = (mean_r + 0.05 * rng.normal(size=b_n)).astype(np.float64)
            cols = {f"x{j}": pa.array(X[:, j], type=pa.float64())
                    for j in range(d)}
            cols["arm"] = pa.array(arms, type=pa.int64())
            cols["r"] = pa.array(r, type=pa.float64())
            tbl = pa.table(cols)
            if writer is None:
                writer = pq.ParquetWriter(str(path), tbl.schema)
            writer.write_table(tbl, row_group_size=row_group)
    finally:
        if writer is not None:
            writer.close()
    df = CadwDataFrame.from_parquet(str(path), base_dir=tmpdir.name)
    return df, tmpdir


def _cols(d: int) -> list[str]:
    return [f"x{i}" for i in range(d)]


def _run_cell_isolated(cell: dict, timeout: int = 600) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__),
           "--__cell", json.dumps(cell)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(
            f"cell failed (exit {res.returncode}):\n"
            f"stdout: {res.stdout[-2000:]}\n"
            f"stderr: {res.stderr[-2000:]}")
    for line in reversed(res.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"no JSON in output:\n{res.stdout[-2000:]}")


def _run_cell_child(cell: dict) -> None:
    import gc
    n = int(cell["n"])
    d = int(cell["d"])
    K = int(cell.get("K", 5))
    batch = cell.get("batch")
    if batch is not None:
        batch = int(batch)
    budget_mb = float(cell.get("budget_mb", 2000))
    seed = int(cell.get("seed", 0))

    df, handle = make_df_parquet(n, d, K=K, seed=seed)
    try:
        # Prewarm
        warm_rng = np.random.RandomState(999)
        wX = warm_rng.normal(size=(64, max(d, 2)))
        warm_cols = [f"x{i}" for i in range(wX.shape[1])]
        warm_arm = warm_rng.randint(0, max(K, 2), size=64).astype(np.int64)
        warm_r = warm_rng.normal(size=64)
        warm_tbl = pa.table({**{c: pa.array(wX[:, j], type=pa.float64())
                                for j, c in enumerate(warm_cols)},
                             "arm": pa.array(warm_arm, type=pa.int64()),
                             "r": pa.array(warm_r, type=pa.float64())})
        warm_df = CadwDataFrame.from_table(warm_tbl)
        fit_linucb(warm_df, warm_cols, arm_col="arm", reward_col="r",
                   memory_guard=MemoryGuard(max_memory_mb=500),
                   batch_size=32, k_max_hint=max(K, 2))
        del warm_df, warm_tbl, wX
        gc.collect()
        # Page-cache warm: read all batches once (drop immediately).
        # Without this, first fit pass causes page faults that inflate
        # ru_maxrss before fit starts. Then fit peak < preloaded peak ->
        # rss_delta=0 (silent measurement failure).
        for _b in df._iter_filtered_batches(batch_size=1000):
            pass
        del _b
        gc.collect()
        gc.disable()
        try:
            mg = MemoryGuard(max_memory_mb=budget_mb)
            r = measure_peak(
                fit_linucb, df, _cols(d),
                arm_col="arm", reward_col="r",
                memory_guard=mg, batch_size=batch, k_max_hint=K,
            )
            rss = int(r["rss_bytes"])
            # v0.9.110b: rss_bytes=0 means measure_peak caught nothing
            # (ru_maxrss already peaked pre-fit). Falling back to
            # total_bytes masks this as "small measured" (wrong).
            # Report unmeasured explicitly - caller skips this cell.
            if rss == 0:
                out = {"unmeasured": True,
                       "py_bytes": int(r["py_bytes"]),
                       "total_bytes": int(r["total_bytes"])}
            else:
                out = {"bytes": rss}
        finally:
            gc.enable()
    except MemoryLimitExceeded as e:
        out = {"error": "MemoryLimitExceeded", "msg": str(e)[:300]}
    finally:
        handle.cleanup()
    print(json.dumps(out))


def _measure(cell: dict, *, n_repeat: int = 3) -> dict:
    runs = []
    unmeasured_count = 0
    for _ in range(n_repeat):
        r = _run_cell_isolated(cell)
        if "error" in r:
            raise MemoryLimitExceeded(r.get("msg", "?"))
        if r.get("unmeasured"):
            unmeasured_count += 1
            continue
        runs.append(int(r["bytes"]))
    if not runs:
        return {"unmeasured": True, "runs": [],
                "unmeasured_count": unmeasured_count}
    med = int(statistics.median(runs))
    return {"bytes": med, "runs": runs,
            "spread": (max(runs) - min(runs)) / max(med, 1),
            "unmeasured_count": unmeasured_count}


def stage0_baseline():
    print("=== STAGE 0: baseline noise (n=1000, d=10, K=5) ×5 ===")
    cell = {"n": 1000, "d": 10, "K": 5, "batch": 1000,
            "budget_mb": 2000, "seed": 0}
    r = _measure(cell, n_repeat=5)
    print(f"  runs: {[f'{v:,}' for v in r['runs']]}")
    print(f"  median={r['bytes']:,}  spread={r['spread']*100:.1f}%")


def stage2_batch_sweep():
    print("\n=== STAGE 2: batch-sweep (d=10, K=5, n=batch) ===")
    print(f"{'batch':>8}  {'meas(B)':>12}  {'pred(B)':>12}  {'ratio':>8}")
    rows = []
    for b in [1000, 2000, 4000, 8000]:
        cell = {"n": b, "d": 10, "K": 5, "batch": b,
                "budget_mb": 2000, "seed": 0}
        try:
            r = _measure(cell)
        except MemoryLimitExceeded as e:
            print(f"  batch={b}: {e}")
            continue
        meas = r["bytes"]
        pred = predicted_peak(batch=b, d=10, K=5)
        ratio = pred / max(meas, 1)
        rows.append((b, meas))
        print(f"{b:>8}  {meas:>12,}  {pred:>12,}  {ratio:>8.2f}")
    if len(rows) >= 2:
        bs = np.array([r[0] for r in rows], dtype=float)
        ys = np.array([r[1] for r in rows], dtype=float)
        A = np.vstack([bs, np.ones_like(bs)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, ys, rcond=None)
        coef = slope / (8 * 10 + 16)
        print(f"\n  fitted coef_batch={coef:.3f} (hypothesis 4.0)")
        print(f"  fitted intercept={intercept:,.0f} B "
              f"(hypothesis K*d^2*8 + K*d*8 = {5*10*10*8 + 5*10*8:,})")


def stage3_d_sweep():
    print("\n=== STAGE 3: d-sweep (n=1000, batch=1000, K=5) ===")
    print(f"{'d':>6}  {'meas(B)':>12}  {'pred(B)':>12}  {'ratio':>8}")
    for d in [10, 100, 500]:
        cell = {"n": 1000, "d": d, "K": 5, "batch": 1000,
                "budget_mb": 2000, "seed": 0}
        r = _measure(cell)
        meas = r["bytes"]
        pred = predicted_peak(batch=1000, d=d, K=5)
        print(f"{d:>6}  {meas:>12,}  {pred:>12,}  {pred/max(meas,1):>8.2f}")


def stage4_K_sweep():
    print("\n=== STAGE 4: K-sweep (n=1000, d=100, batch=1000) ===")
    print(f"{'K':>4}  {'meas(B)':>12}  {'pred(B)':>12}  {'ratio':>8}")
    for K in [2, 5, 20]:
        cell = {"n": 1000, "d": 100, "K": K, "batch": 1000,
                "budget_mb": 2000, "seed": 0}
        r = _measure(cell)
        meas = r["bytes"]
        pred = predicted_peak(batch=1000, d=100, K=K)
        print(f"{K:>4}  {meas:>12,}  {pred:>12,}  {pred/max(meas,1):>8.2f}")


def stage5_ooc():
    print("\n=== STAGE 5: OOC (batch=500 fixed, d=10, K=5) ===")
    print(f"{'n':>10}  {'meas(B)':>12}  {'pred(B)':>12}  {'ratio':>8}")
    peaks = []
    for n in [10_000, 100_000, 1_000_000]:
        cell = {"n": n, "d": 10, "K": 5, "batch": 500,
                "budget_mb": 2000, "seed": 0}
        try:
            r = _measure(cell)
        except MemoryLimitExceeded as e:
            print(f"  n={n}: {e}")
            continue
        meas = r["bytes"]
        pred = predicted_peak(batch=500, d=10, K=5)
        peaks.append((n, meas))
        print(f"{n:>10,}  {meas:>12,}  {pred:>12,}  {pred/max(meas,1):>8.2f}")
    if len(peaks) >= 2:
        ratio = peaks[-1][1] / max(peaks[0][1], 1)
        print(f"\n  peak ratio = {ratio:.2f}  "
              f"{'PASS' if ratio < 2.0 else 'FAIL'}")


def stage6_matrix():
    print("\n=== STAGE 6: matrix 9/9 (batch=1000, K=5) ===")
    print(f"{'n':>10}  {'d':>6}  {'meas(B)':>12}  {'pred(B)':>12}  "
          f"{'ratio':>8}  {'in[.7,1.4]':>10}")
    ok, total, maxdev = 0, 0, 0.0
    # d in {100, 500, 1000}: d=10 skipped because batch (384 KB) and state
    # (4 KB) are below RSS resolution of measure_peak (~500 KB noise).
    # STABILITY_PROTOCOL §4.1 allows skipping dimensions below signal.
    for n in [10**3, 10**4, 10**5]:
        for d in [100, 500, 1000]:
            total += 1
            cell = {"n": n, "d": d, "K": 5, "batch": 1000,
                    "budget_mb": 2000, "seed": 0}
            try:
                r = _measure(cell)
            except MemoryLimitExceeded as e:
                print(f"{n:>10}  {d:>6}  {e}")
                continue
            if r.get("unmeasured"):
                print(f"{n:>10}  {d:>6}  {'N/A (rss=0)':>12}")
                continue
            meas = r["bytes"]
            pred = predicted_peak(batch=1000, d=d, K=5)
            ratio = pred / max(meas, 1)
            in_range = 0.7 <= ratio <= 1.4
            if in_range:
                ok += 1
            maxdev = max(maxdev, abs(1.0 - ratio))
            print(f"{n:>10}  {d:>6}  {meas:>12,}  {pred:>12,}  "
                  f"{ratio:>8.2f}  {'OK' if in_range else 'FAIL':>10}")
    print(f"\n  cells passed: {ok}/{total}  max|1-ratio|={maxdev:.3f}")


_STAGES = {
    "0": stage0_baseline,
    "2": stage2_batch_sweep,
    "3": stage3_d_sweep,
    "4": stage4_K_sweep,
    "5": stage5_ooc,
    "6": stage6_matrix,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    if args.stage == "all":
        for k in ["0", "2", "3", "4", "5", "6"]:
            try:
                _STAGES[k]()
            except Exception:
                print(f"\n!! stage {k} FAILED")
                traceback.print_exc()
    elif args.stage in _STAGES:
        _STAGES[args.stage]()
    else:
        print(f"unknown --stage {args.stage}; {list(_STAGES)} or 'all'")
        sys.exit(2)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--__cell":
        _run_cell_child(json.loads(sys.argv[2]))
        sys.exit(0)
    main()
