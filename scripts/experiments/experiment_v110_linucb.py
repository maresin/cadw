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
from cadw.core.memory_guard import (
    MemoryGuard, MemoryLimitExceeded, measure_peak_traced,
)
from cadw.ml.linucb import fit_linucb


def predicted_peak(batch: int, d: int, K: int = 5) -> int:
    """Formula v4 (v0.9.110c, phase-traced calibration + fadvise).

    peak = 1*batch*row_size + 2*K*d^2*f64 + K*d*f64  (row_size = 8d+16)

    Term 1 (batch): _column_to_f64 returns zero-copy view for f64 arrays
      -> _stack_numeric peak = 1x result (parts list holds views, not copies).
    Term 2 (state + temp): A state (K*d^2*8) + per-arm Xa.T@Xa temp
      (d^2*8, allocator lags before free) = 2*K*d^2*8.
    Term 3 (b): K*d*8.

    Calibration (v0.9.110c, /tmp/quick_v4_test.py):
      n=1000, d=1000, K=5: measured 99.99 MB, formula 88.06 MB, ratio 0.88
      n=10^4, d=1000, K=5: measured 88.06 MB, formula 88.06 MB, ratio 1.00

    fadvise(DONTNEED) required at n>=10^4: without it, page cache from
    parquet load acts as allocator free-list -> RSS delta collapses to
    14 MB (unmeasurable). With it, fit allocates fresh pages.

    d <= 100: signal below noise floor (predicted <8 MB vs baseline
    >100 MB) -> matrix skips per STABILITY_PROTOCOL §4.1.
    """
    row_size = 8 * d + 16
    # v6: state A(K*d^2*8) + b(K*d*8) + 256 + ONE temp d^2*8
    #     + coef_batch*batch*row_size (X + Xa + arrow batch, coef=3.5 safe).
    # Bug fix vs v5: "2*K*d^2*8" double-counted temp as if K simultaneous,
    # but for-loop frees temp before next arm -> only one alive at any time.
    return (K * d * d * 8 + K * d * 8 + 256
            + d * d * 8
            + 3.5 * batch * row_size)


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
        # v0.9.110c: release glibc malloc free-list before measure.
        # Without this, fit reuses malloc arena from warmup/df-load (RSS
        # stays flat, delta=61KB - unmeasurable). malloc_trim(0) returns
        # free pages to OS -> fit allocates fresh -> real delta visible.
        # fadvise(DONTNEED) alone was insufficient: parquet mmap pages
        # already dropped by then; the issue was malloc free-list, not
        # page cache. Both together = robust.
        try:
            import ctypes
            _libc = ctypes.CDLL("libc.so.6")
            _libc.malloc_trim(0)
        except Exception:
            pass
        try:
            _pq_path = Path(handle.name) / "data.parquet"
            fd = os.open(str(_pq_path), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except Exception:
            pass
        gc.collect()
        gc.disable()
        try:
            mg = MemoryGuard(max_memory_mb=budget_mb)
            # v0.9.110c: measure_peak_traced (fast VmRSS sampler via
            # /proc/self/statm, 200us interval). Fixes DEC-032 issue:
            # ru_maxrss delta was 0 when preload peak >= fit peak.
            r = measure_peak_traced(
                fit_linucb, df, _cols(d),
                arm_col="arm", reward_col="r",
                memory_guard=mg, batch_size=batch, k_max_hint=K,
                mg=mg,
            )
            peak = int(r["peak_bytes"])
            # 100 KB floor: below /proc/self/statm detection noise (.so
            # page faults on first touch). Above 100 KB, peak is real.
            if peak < 100 * 1024:
                out = {"unmeasured": True, "peak_bytes": peak,
                       "samples": int(r["samples"])}
            else:
                out = {"bytes": peak, "samples": int(r["samples"]),
                       "peak_phase": r["peak_phase"]}
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
    # v0.9.110c: use MAX not median. Rationale: formula is upper bound
    # of peak. When allocator state varies across subprocesses (page
    # cache, memory pressure), some runs give true peak, others reuse
    # free-list (collapse to ~0). Median masks this; max captures peak.
    # Conservative: max may slightly over-estimate if one run had noise,
    # but for safety (guard must not underestimate) over > under.
    m = max(runs)
    return {"bytes": m, "runs": runs,
            "spread": (max(runs) - min(runs)) / max(m, 1),
            "unmeasured_count": unmeasured_count}


def stage0_baseline():
    # n=10^4, d=500: state K*d^2*8=10 MB, well above 100 KB noise floor.
    # Small cells (n=1000, d=10) would be filtered as "unmeasured" by
    # _run_cell_child (<100 KB floor) - not useful for baseline.
    print("=== STAGE 0: baseline noise (n=10^4, d=500, K=5) x5 ===")
    cell = {"n": 10_000, "d": 500, "K": 5, "batch": 1000,
            "budget_mb": 2000, "seed": 0}
    r = _measure(cell, n_repeat=5)
    print(f"  runs: {[f'{v:,}' for v in r['runs']]}")
    if r.get("unmeasured") or "bytes" not in r:
        print(f"  ALL UNMEASURED (peak < 100 KB floor)")
        return
    print(f"  median={r['bytes']:,}  spread={r['spread']*100:.1f}%")


def stage2_batch_sweep():  # n_repeat=1 for speed
    print("\n=== STAGE 2: batch-sweep (d=10, K=5, n=batch) ===")
    print(f"{'batch':>8}  {'meas(B)':>12}  {'pred(B)':>12}  {'ratio':>8}")
    rows = []
    for b in [1000, 2000, 4000, 8000]:
        cell = {"n": b, "d": 10, "K": 5, "batch": b,
                "budget_mb": 2000, "seed": 0}
        try:
            r = _measure(cell, n_repeat=1)
        except MemoryLimitExceeded as e:
            print(f"  batch={b}: {e}")
            continue
        if r.get("unmeasured"):
            print(f"{b:>8}  {'N/A':>12}")
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


def stage3_d_sweep():  # n_repeat=1 for speed
    print("\n=== STAGE 3: d-sweep (n=1000, batch=1000, K=5) ===")
    print(f"{'d':>6}  {'meas(B)':>12}  {'pred(B)':>12}  {'ratio':>8}")
    # d >= 500: state dominates baseline noise (see formula v3 docstring)
    for d in [500, 1000]:
        cell = {"n": 1000, "d": d, "K": 5, "batch": 1000,
                "budget_mb": 2000, "seed": 0}
        r = _measure(cell, n_repeat=1)
        meas = r["bytes"]
        pred = predicted_peak(batch=1000, d=d, K=5)
        print(f"{d:>6}  {meas:>12,}  {pred:>12,}  {pred/max(meas,1):>8.2f}")


def stage4_K_sweep():  # n_repeat=1 for speed
    # d=500: state K*d^2*8 is 2-16 MB range -> visible above noise floor.
    print("\n=== STAGE 4: K-sweep (n=1000, d=500, batch=1000) ===")
    print(f"{'K':>4}  {'meas(B)':>12}  {'pred(B)':>12}  {'ratio':>8}")
    for K in [2, 5, 20]:
        cell = {"n": 1000, "d": 500, "K": K, "batch": 1000,
                "budget_mb": 2000, "seed": 0}
        try:
            r = _measure(cell, n_repeat=1)
        except MemoryLimitExceeded as e:
            print(f"  K={K}: {e}")
            continue
        if r.get("unmeasured"):
            print(f"{K:>4}  {'N/A':>12}")
            continue
        meas = r["bytes"]
        pred = predicted_peak(batch=1000, d=500, K=K)
        print(f"{K:>4}  {meas:>12,}  {pred:>12,}  {pred/max(meas,1):>8.2f}")


def stage5_ooc():  # n_repeat=1 for speed
    # d=500 so peak ~12 MB (above noise); batch=500 fixed; n varies.
    print("\n=== STAGE 5: OOC (batch=500 fixed, d=500, K=5) ===")
    print(f"{'n':>10}  {'meas(B)':>12}  {'pred(B)':>12}  {'ratio':>8}")
    peaks = []
    for n in [10_000, 100_000]:
        cell = {"n": n, "d": 500, "K": 5, "batch": 500,
                "budget_mb": 2000, "seed": 0}
        try:
            r = _measure(cell, n_repeat=1)
        except MemoryLimitExceeded as e:
            print(f"  n={n}: {e}")
            continue
        meas = r["bytes"]
        pred = predicted_peak(batch=500, d=500, K=5)
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
    # d in {500, 1000}: d <= 100 skipped because predicted (0.9-8 MB)
    # below baseline noise floor (~100 MB /proc/statm + parquet load).
    # STABILITY_PROTOCOL §4.1 (skip below signal). d=500/1000 cells:
    # state K*d^2*8 = 10/40 MB, well above noise.
    for n in [10**3, 10**4, 10**5]:
        for d in [500, 1000]:
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
            runs_str = ",".join(f"{v/1e6:.1f}" for v in r["runs"])
            print(f"{n:>10}  {d:>6}  {meas:>12,}  {pred:>12,}  "
                  f"{ratio:>8.2f}  {'OK' if in_range else 'FAIL':>10}  "
                  f"runs=[{runs_str}]MB")
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
