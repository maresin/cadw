"""Reproducible evidence chain for v0.9.105 formula calibration.

Reconstructs the numbers reported in DEC-018..025 and
docs/ml_tools/EXPERIMENT_PROCEDURE.md §7.

Run:
    uv run python scripts/experiments/experiment_v105_rep.py [--stage N]

Stages:
    0  baseline / instrument sanity
    1  batch-sweep -> coef for linear, logistic, gaussian, bernoulli
    2  d-sweep -> (a, b) decomposition for logistic
    3  C-sweep (NB parametrized state)
    4  OOC: with vs without batch_size patch (framework gap evidence)
    5  control: matrix_profile formula reproduction
    6  post-fix verification (O2/O3): coef after fixes
    all (default): stages 0-6

Env note: uses measure_peak (measure_peak) — dev tool, not for production.
Requires Linux/macOS (`resource` module).
"""
from __future__ import annotations
import argparse
import os
import statistics
import sys
from types import MethodType

# Make project root importable when running from scripts/experiments/
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pyarrow as pa

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import measure_peak

N_REPEAT = 3


def make_df(n, d, *, y_kind="float", C=2):
    rng = np.random.RandomState(42)
    cols = {f"x{j}": rng.randn(n) for j in range(d)}
    if y_kind == "float":
        cols["y"] = rng.randn(n)
    elif y_kind == "binary":
        cols["y"] = rng.randint(0, 2, n).astype(np.int64)
    elif y_kind == "multiclass":
        cols["y"] = rng.randint(0, C, n).astype(np.int64)
    return CadwDataFrame.from_table(
        pa.table(cols),
        memory_guard=MemoryGuard(max_memory_mb=2000),
    )


def patch_batch_size(df, bs):
    """Force batch_size on this df instance (iter-4 control path)."""
    def patched(self):
        yield from self._dataset.iter_batches(
            batch_size=bs, columns=self._columns)
    df._iter_filtered_batches = MethodType(patched, df)


def med(fn, *a, **kw):
    return int(statistics.median([
        measure_peak(fn, *a, **kw)["py_bytes"] for _ in range(N_REPEAT)]))


def stage0_baseline():
    print("=== Stage 0: baseline ===")
    df = make_df(100, 10)
    m = med(lambda: [b for b in df._iter_filtered_batches()])
    print(f"  materialize(100,10): peak={m} B  (should be stable across repeats)")


def stage1_batch_sweep():
    print("\n=== Stage 1: batch-sweep -> coef_batch ===")
    from cadw.ml.linear import fit_linear
    from cadw.ml.logistic import fit_logistic
    from cadw.ml.naive_bayes import fit_gaussian_nb, fit_bernoulli_nb

    d = 10
    Xs = [f"x{j}" for j in range(d)]
    row_size = (d + 1) * 8
    results = {}
    for batch in (1000, 2000, 4000, 8000):
        row = {}
        for name, fn, yk, kw in (
            ("linear", fit_linear, "float", dict(alpha=0.0)),
            ("logistic", fit_logistic, "binary", dict(alpha=1.0, max_iter=5)),
            ("gaussian", fit_gaussian_nb, "binary", {}),
            ("bernoulli", fit_bernoulli_nb, "binary", {}),
        ):
            df = make_df(batch, d, y_kind=yk)
            patch_batch_size(df, batch)
            row[name] = med(fn, df, Xs, "y",
                            memory_guard=df._memory_guard, **kw)
        results[batch] = row
        print(f"  batch={batch}: {row}")

    bs = np.array([1000, 2000, 4000, 8000], dtype=float)
    print("\n  Fitted coef (slope / row_size):")
    for name in ("linear", "logistic", "gaussian", "bernoulli"):
        peaks = np.array([results[b][name] for b in bs], dtype=float)
        slope, _ = np.polyfit(bs, peaks, 1)
        coef = slope / row_size
        print(f"    {name:12s}: coef = {coef:.3f}")


def stage2_d_sweep():
    print("\n=== Stage 2: d-sweep -> (a, b) for logistic ===")
    from cadw.ml.logistic import fit_logistic

    batch = 1000
    pts = []
    for d in (100, 500, 1000):
        Xs = [f"x{j}" for j in range(d)]
        row_size = (d + 1) * 8
        df = make_df(batch, d, y_kind="binary")
        m = med(fit_logistic, df, Xs, "y", alpha=1.0, max_iter=5,
                memory_guard=df._memory_guard)
        pts.append((batch * row_size, (d + 1) ** 2 * 8, m))
        print(f"  d={d}: batch_term={batch * row_size}  d^2_8={(d + 1) ** 2 * 8}  peak={m}")

    (bt_lo, d2_lo, p_lo), (bt_hi, d2_hi, p_hi) = pts[0], pts[-1]
    denom = bt_lo * d2_hi - bt_hi * d2_lo
    a = (p_lo * d2_hi - p_hi * d2_lo) / denom
    b = (bt_lo * p_hi - bt_hi * p_lo) / denom
    print(f"\n  solved: a = {a:.3f}  b = {b:.3f}")
    print("  Reference (iter-3/5): a=3.05, b=3.97. After O2 only: a=3.05 (O2 insufficient).")


def stage3_c_sweep():
    print("\n=== Stage 3: C-sweep (NB parametrized state) ===")
    from cadw.ml.naive_bayes import (
        fit_gaussian_nb, _GaussianStatsOp,
    )

    batch, d = 200, 2000
    Xs = [f"x{j}" for j in range(d)]
    row_size = (d + 1) * 8
    for C in (2, 10, 30):
        df = make_df(batch, d, y_kind="multiclass", C=C)
        patch_batch_size(df, batch)
        op = _GaussianStatsOp(Xs, "y")
        st = op.init()
        for b in df._iter_filtered_batches():
            st = op.update(st, b)
        direct = op.state_nbytes(st)
        m = med(fit_gaussian_nb, df, Xs, "y", memory_guard=df._memory_guard)
        print(f"  C={C:3d}: direct_state={direct}  peak={m}")


def stage4_ooc():
    print("\n=== Stage 4: OOC check (with vs without patch) ===")
    from cadw.ml.linear import fit_linear

    d = 100
    Xs = [f"x{j}" for j in range(d)]
    mg = MemoryGuard(max_memory_mb=500)

    print("  WITHOUT patch (production path):")
    for n in (10_000, 100_000):
        df = make_df(n, d, y_kind="float")
        df._memory_guard = mg
        n_b = sum(1 for _ in df._iter_filtered_batches())
        try:
            m = med(fit_linear, df, Xs, "y", alpha=0.0, memory_guard=mg)
            print(f"    n={n:>7}  batches={n_b}  peak={m:>10}")
        except Exception as e:
            print(f"    n={n:>7}  RAISED {type(e).__name__}")

    print("\n  WITH patch (batch_size=1000):")
    for n in (10_000, 100_000):
        df = make_df(n, d, y_kind="float")
        df._memory_guard = mg
        patch_batch_size(df, 1000)
        n_b = sum(1 for _ in df._iter_filtered_batches())
        m = med(fit_linear, df, Xs, "y", alpha=0.0, memory_guard=mg)
        print(f"    n={n:>7}  batches={n_b}  peak={m:>10}")
    print("  Expected: with patch, peak is flat (OOC). Without — grows with n (DEC-023).")


def stage5_control():
    print("\n=== Stage 5: control (matrix_profile) ===")
    from cadw.ml.matrix_profile import fit_matrix_profile
    from cadw.ml._memory_estimate import _load_cards, evaluate_formula

    _load_cards.cache_clear()
    card = _load_cards().get("fit_matrix_profile")
    if not card or not card.get("ram_formula"):
        print("  SKIP: no ram_formula")
        return
    for n, S, w in ((100_000, 1024, 16), (100_000, 2048, 32), (100_000, 4096, 64)):
        rng = np.random.RandomState(42)
        df = CadwDataFrame.from_table(
            pa.table({"value": rng.randn(n)}),
            memory_guard=MemoryGuard(max_memory_mb=1000))
        pred = evaluate_formula("fit_matrix_profile", S=S, w=w)["bytes"]
        m = med(fit_matrix_profile, df, "value", window=w, max_samples=S,
                memory_guard=MemoryGuard(max_memory_mb=500))
        print(f"  S={S} w={w}: pred={pred/1e6:.2f}M  meas={m/1e6:.2f}M  ratio={pred/m:.2f}")


def stage6_post_fix():
    print("\n=== Stage 6: post O2/O3 (verify coef reductions) ===")
    from cadw.ml.naive_bayes import fit_bernoulli_nb

    d = 20
    Xs = [f"x{j}" for j in range(d)]
    row_size = (d + 1) * 8
    bs_list, peaks = [], []
    for bs in (1000, 2000, 4000):
        df = make_df(bs, d, y_kind="binary")
        patch_batch_size(df, bs)
        peaks.append(med(fit_bernoulli_nb, df, Xs, "y",
                         memory_guard=df._memory_guard))
        bs_list.append(bs)
    slope, _ = np.polyfit(np.array(bs_list), np.array(peaks), 1)
    print(f"  bernoulli coef = {slope / row_size:.3f}  (expected 2.0 post-O3; was 2.94 pre-O3)")


STAGES = {
    0: stage0_baseline, 1: stage1_batch_sweep, 2: stage2_d_sweep,
    3: stage3_c_sweep, 4: stage4_ooc, 5: stage5_control,
    6: stage6_post_fix,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=None,
                    help="0-6, or omit for all")
    args = ap.parse_args()

    import cadw
    print(f"cadw version: {cadw.__version__}  N_REPEAT={N_REPEAT}")

    if args.stage is None:
        for s in sorted(STAGES):
            STAGES[s]()
    else:
        if args.stage not in STAGES:
            raise SystemExit(f"invalid stage {args.stage}")
        STAGES[args.stage]()
