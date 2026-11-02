"""Calibrate fit_isolation_forest ram_formula (v0.9.110, EXPERIMENT_PROCEDURE §3).

Hypothesis (from code, see cadw/ml/iforest.py):
  peak = 3*batch*row_size + S*d*f64 + 2*n_trees*S*80
  where:
    - 3*batch: fit loop, arrays list + column_stack + X_b[mask] copy
    - S*d*f64: reservoir buffer np.empty((S, d))
    - 2*n_trees*S*80: trees (lists ~40 B/node + cached numpy arrays ~40 B/node)

Measurement policy (see cadw/core/memory_guard.py:measure_peak docstring):
  rss_bytes (resource.getrusage.ru_maxrss delta) is MONOTONE PER PROCESS.
  Multiple sequential measure_peak() calls in one process: run1 valid,
  run2+ ≈ 0. py_bytes (tracemalloc) tracks Python objects, NOT numpy
  (numpy uses malloc, not PyMem). Therefore:
    - Each measurement cell runs in its OWN subprocess (fresh rss0).
    - Authoritative number = rss_bytes from that subprocess.
    - py_bytes is reported for cross-check but not used for ratio.
  OOC test uses FIXED small batch (batches << n), so peak must be constant
  in n (varying n is the whole point of OOC).

Stages:
  0 baseline noise (3 subprocesses on same cell)
  2 batch-sweep -> coef_batch (n == batch: single-batch case)
  3 d-sweep -> coef_state at batch=1000
  5 OOC: fixed batch=500, n=10^3..10^5, expect ratio < 2.0
  6 matrix 9/9 in [0.7, 1.4]
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

# Make project root importable when running from scripts/experiments/
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded, measure_peak
from cadw.ml.iforest import fit_isolation_forest


# ---- prediction ---------------------------------------------------------

def predicted_peak(batch: int, d: int, S: int = 256, n_trees: int = 10) -> int:
    """Formula v3 (v0.9.110a-fix3): 6 terms, corrected from code inspection.

    3*batch*row_size    - arrays list (views) + column_stack + row iteration views
    S*d*f64             - reservoir np.empty((S,d))
    n_trees*2*S*64      - _Tree 5 lists, ~64 B/node (small int cache + mixed floats
                          + list pointers; 511 nodes max, so ~327 KB for 10 trees)
    n_trees*5*S*f64     - cached np arrays in __init__ (5 numpy arrays per tree,
                          views of lists, ~20 KB per tree × n_trees)
    S*d*f64             - X_fit[idx] transient per-tree (one at a time, ~2 MB at d=1000)
    max_depth*S*6*f64   - _score_matrix transient, ONE tree at a time (not ×n_trees)
    """
    row_size = 8 * d
    max_depth = max(1, int(np.ceil(np.log2(max(S, 2)))))
    return (
        3 * batch * row_size
        + S * d * 8
        + n_trees * 2 * S * 64
        + n_trees * 5 * S * 8
        + S * d * 8
        + max_depth * S * 6 * 8
    )


# ---- df factory ---------------------------------------------------------

def make_df_parquet(n: int, d: int, seed: int = 0, row_group: int = 1000):
    """Lazy df via parquet; caller must keep tmpdir handle alive."""
    tmpdir = tempfile.TemporaryDirectory(prefix="if_exp_")
    rng = np.random.RandomState(seed)
    path = Path(tmpdir.name) / "data.parquet"
    writer = None
    try:
        for start in range(0, n, row_group):
            end = min(start + row_group, n)
            b_n = end - start
            X = rng.normal(size=(b_n, d)).astype(np.float64)
            tbl = pa.table({f"x{i}": pa.array(X[:, i], type=pa.float64())
                            for i in range(d)})
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


# ---- subprocess isolated measurement ------------------------------------

def _run_cell_isolated(cell: dict, timeout: int = 300) -> dict:
    """Spawn fresh subprocess, one measure_peak, return JSON dict."""
    cmd = [sys.executable, os.path.abspath(__file__),
           "--__cell", json.dumps(cell)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(
            f"cell failed (exit {res.returncode}):\n"
            f"stdout: {res.stdout[-2000:]}\n"
            f"stderr: {res.stderr[-2000:]}"
        )
    for line in reversed(res.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"no JSON in output:\n{res.stdout[-2000:]}")


def _run_cell_child(cell: dict) -> None:
    """Runs inside isolated subprocess. One measure_peak, one JSON line."""
    import gc
    n = int(cell["n"])
    d = int(cell["d"])
    batch = cell.get("batch")
    if batch is not None:
        batch = int(batch)
    S = int(cell.get("S", 256))
    n_trees = int(cell.get("n_trees", 10))
    budget_mb = float(cell.get("budget_mb", 2000))
    seed = int(cell.get("seed", 0))

    df, handle = make_df_parquet(n, d, seed=seed)
    try:
        # Prewarm: tiny fit forces all lazy allocations BEFORE measurement
        # (numpy pools, pyarrow batch cache, GC thresholds, etc.).
        # Without prewarm, first-fit lazy state adds 1-2 MB noise, which
        # is 3x the signal for small d.
        warm_rng = np.random.RandomState(999)
        warm_X = warm_rng.normal(size=(64, max(d, 2)))
        warm_tbl = pa.table({f"x{i}": pa.array(warm_X[:, i], type=pa.float64())
                             for i in range(warm_X.shape[1])})
        warm_cols = [f"x{i}" for i in range(warm_X.shape[1])]
        warm_df = CadwDataFrame.from_table(warm_tbl)
        fit_isolation_forest(
            warm_df, warm_cols, n_trees=2, subsample_size=32,
            random_state=0, memory_guard=MemoryGuard(max_memory_mb=500),
            batch_size=32,
        )
        del warm_df, warm_tbl, warm_X
        gc.collect()
        gc.disable()   # no GC during measurement

        try:
            mg = MemoryGuard(max_memory_mb=budget_mb)
            r = measure_peak(
                fit_isolation_forest, df, _cols(d),
                n_trees=n_trees, subsample_size=S, random_state=seed,
                memory_guard=mg, batch_size=batch,
            )
            out = {
                "py_bytes": int(r["py_bytes"]),
                "rss_bytes": int(r["rss_bytes"]),
                "bytes": int(r["rss_bytes"]) or int(r["total_bytes"]),
            }
        finally:
            gc.enable()
    except MemoryLimitExceeded as e:
        out = {"error": "MemoryLimitExceeded", "msg": str(e)[:300]}
    finally:
        handle.cleanup()
    print(json.dumps(out))


def _measure(cell: dict, *, n_repeat: int = 3) -> dict:
    """Measure a cell. n_repeat times in isolated subprocesses, return median.

    Why n_repeat > 1: measure_peak uses ru_maxrss (peak-since-process-start).
    During fit, page faults from parquet read can inflate OR suppress the
    delta depending on cache state. Empirically: same cell (n=10^4, d=1000,
    batch=1000) gives 9.35 MB or 32 MB across runs — same work, different
    timing. Median-of-3 filters sporadic outliers, preserving correct value
    when 2/3 runs are stable.
    """
    runs = []
    for _ in range(n_repeat):
        r = _run_cell_isolated(cell)
        if "error" in r:
            raise MemoryLimitExceeded(r.get("msg", "?"))
        if "bytes" not in r:
            raise RuntimeError(f"child returned no 'bytes' key: {r}")
        runs.append(int(r["bytes"]))
    med = int(statistics.median(runs))
    return {"bytes": med, "runs": runs, "spread": (max(runs)-min(runs))/max(med, 1)}


# ---- stages -------------------------------------------------------------

def stage0_baseline():
    print("=== STAGE 0: baseline noise (n=1000, d=3) ×3 subprocesses ===")
    cell = {"n": 1000, "d": 3, "batch": 1000, "S": 256, "n_trees": 10,
            "budget_mb": 2000, "seed": 0}
    r = _measure(cell, n_repeat=5)
    print(f"  runs: {[f'{v:,}' for v in r['runs']]}")
    print(f"  median={r['bytes']:,}  spread={r['spread']*100:.1f}%")
    return r["spread"] < 0.20


def stage2_batch_sweep():
    print("\n=== STAGE 2: batch-sweep (d=10, S=256, n_trees=10, n=batch) ===")
    print(f"{'batch':>8}  {'meas(B)':>12}  {'pred(B)':>12}  {'ratio':>8}")
    batches = [1000, 2000, 4000, 8000]
    rows = []
    for b in batches:
        cell = {"n": b, "d": 10, "batch": b, "S": 256, "n_trees": 10,
                "budget_mb": 2000, "seed": 0}
        try:
            r = _measure(cell)
        except MemoryLimitExceeded as e:
            print(f"  batch={b}: {e}")
            continue
        meas = r["bytes"]
        pred = predicted_peak(batch=b, d=10, S=256, n_trees=10)
        ratio = pred / max(meas, 1)
        rows.append((b, meas, pred, ratio))
        print(f"{b:>8}  {meas:>12,}  {pred:>12,}  {ratio:>8.2f}")

    if len(rows) >= 2:
        bs = np.array([r[0] for r in rows], dtype=float)
        ys = np.array([r[1] for r in rows], dtype=float)
        A = np.vstack([bs, np.ones_like(bs)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, ys, rcond=None)
        row_size = 8 * 10
        coef = slope / row_size
        print(f"\n  fitted: coef_batch={coef:.3f} (hypothesis 3.0), "
              f"intercept={intercept:,.0f} B (state_term)")
        print(f"  predicted state_term = "
              f"{256*10*8 + 2*10*256*80:,} B")


def stage3_d_sweep():
    print("\n=== STAGE 3: d-sweep (n=1000, batch=1000, S=256, n_trees=10) ===")
    print(f"{'d':>6}  {'meas(B)':>12}  {'pred(B)':>12}  {'ratio':>8}")
    for d in [10, 100, 1000]:
        cell = {"n": 1000, "d": d, "batch": 1000, "S": 256, "n_trees": 10,
                "budget_mb": 2000, "seed": 0}
        r = _measure(cell)
        meas = r["bytes"]
        pred = predicted_peak(batch=1000, d=d, S=256, n_trees=10)
        ratio = pred / max(meas, 1)
        print(f"{d:>6}  {meas:>12,}  {pred:>12,}  {ratio:>8.2f}")


def stage5_ooc():
    print("\n=== STAGE 5: OOC (fixed batch=500, d=10, n varies) ===")
    print(f"{'n':>10}  {'meas(B)':>12}  {'pred(B)':>12}  {'ratio':>8}")
    peaks = []
    for n in [10_000, 100_000, 1_000_000]:
        cell = {"n": n, "d": 10, "batch": 500, "S": 256, "n_trees": 10,
                "budget_mb": 2000, "seed": 0}
        try:
            r = _measure(cell)
        except MemoryLimitExceeded as e:
            print(f"  n={n}: {e}")
            continue
        meas = r["bytes"]
        pred = predicted_peak(batch=500, d=10, S=256, n_trees=10)
        peaks.append((n, meas))
        print(f"{n:>10,}  {meas:>12,}  {pred:>12,}  {pred/max(meas,1):>8.2f}")

    if len(peaks) >= 2:
        r = peaks[-1][1] / max(peaks[0][1], 1)
        print(f"\n  peak(n={peaks[-1][0]}) / peak(n={peaks[0][0]}) = {r:.2f}")
        print(f"  OOC: < 2.0 -> {'PASS' if r < 2.0 else 'FAIL'}")


def stage6_matrix():
    print("\n=== STAGE 6: matrix 9/9 (batch=1000, S=256, n_trees=10) ===")
    print(f"{'n':>10}  {'d':>6}  {'meas(B)':>12}  {'pred(B)':>12}  "
          f"{'ratio':>8}  {'in[.7,1.4]':>10}")
    ok, total, maxdev = 0, 0, 0.0
    for n in [10**3, 10**4, 10**5]:
        for d in [10, 100, 1000]:
            total += 1
            cell = {"n": n, "d": d, "batch": 1000, "S": 256, "n_trees": 10,
                    "budget_mb": 2000, "seed": 0}
            try:
                r = _measure(cell)
            except MemoryLimitExceeded as e:
                print(f"{n:>10}  {d:>6}  MemoryLimitExceeded: {e}")
                continue
            meas = r["bytes"]
            pred = predicted_peak(batch=1000, d=d, S=256, n_trees=10)
            ratio = pred / max(meas, 1)
            in_range = 0.7 <= ratio <= 1.4
            if in_range:
                ok += 1
            maxdev = max(maxdev, abs(1.0 - ratio))
            mark = "OK" if in_range else "FAIL"
            print(f"{n:>10}  {d:>6}  {meas:>12,}  {pred:>12,}  "
                  f"{ratio:>8.2f}  {mark:>10}")
    print(f"\n  cells passed: {ok}/{total}  max|1-ratio|={maxdev:.3f}")


# ---- CLI ----------------------------------------------------------------

_STAGES = {
    "0": stage0_baseline,
    "2": stage2_batch_sweep,
    "3": stage3_d_sweep,
    "5": stage5_ooc,
    "6": stage6_matrix,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", help="0|2|3|5|6|all")
    args = ap.parse_args()
    if args.stage == "all":
        for k in ["0", "2", "3", "5", "6"]:
            try:
                _STAGES[k]()
            except Exception:
                print(f"\n!! stage {k} FAILED")
                traceback.print_exc()
    elif args.stage in _STAGES:
        _STAGES[args.stage]()
    else:
        print(f"unknown --stage {args.stage}; pick {list(_STAGES)} or 'all'")
        sys.exit(2)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--__cell":
        _run_cell_child(json.loads(sys.argv[2]))
        sys.exit(0)
    main()
