#!/usr/bin/env python3
"""v0.8.11 prototype: dedup Arrow-native vs pandas chain.

SELF-CONTAINED — no dependency on _FOR_PINGIS/bench/_lib.py (gitignored).
Includes its own minimal RSSSampler (§13.4/141 lesson: prototype must be
runnable from a clean checkout, self-verifying, committed as artifact).

WHY PROTOTYPE-FIRST (§13.4/123, /141): v0.8.2 shift — 5 fix-iterations
without prototype; v0.8.10 per-group — 0 with prototype. Not repeating
the mistake for dedup (P-D1).

THREE PATHS:
  A. Current pandas chain (_local_dedup v0.8.9):
     to_pandas → sort_values(__rid) → drop_duplicates(subset, keep)
     → drop(__rid) → from_pandas.  Measured 3.9-8.3× table (P-D1).
  B. Arrow-native group_by + is_in:
     agg = table.group_by(subset).aggregate([("row_id", "min"/"max")])
     keep_rids = agg["row_id_{agg}"]
     result = table.filter(pc.is_in(table["row_id"], value_set=keep_rids))
     For keep=False: aggregate [("row_id","count"),("row_id","min")],
     filter count==1, use row_id_min as the sole rid.
  C. Arrow-native with dictionary_encode first:
     Same as B, but subset is pc.dictionary_encode'd before group_by —
     compact int ids → smaller hash-table. Filter original table by
     row_id (no join-back). Best when subset is string-heavy, K << N.

CHECKS:
  Correctness: A == B == C on 1000-row case for all keep variants
  and across selectivity (10%/50%/90% dupes).
  RSS: A / B / C on {100k, 300k, 1M} at 10% dupes (K/N = 0.9).

DECISION:
  avg_peak_arrow / avg_peak_pandas <= 1.5  →  ADOPT Arrow path
  Otherwise                                 →  KEEP pandas (§11.5 stopgap)

Run: uv run python scripts/proto_dedup_arrow.py
"""
from __future__ import annotations

import gc
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc


MB = 1024 * 1024


# ============ minimal self-contained RSS sampler ============

class RSSSampler:
    """Peak-RSS tracker via psutil, 5ms sampling thread. Self-contained."""

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline

        def loop():
            while not self._stop.is_set():
                r = self.proc.memory_info().rss
                if r > self.peak:
                    self.peak = r
                self._stop.wait(self.interval)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        r = self.proc.memory_info().rss
        if r > self.peak:
            self.peak = r

    def sample_now(self):
        r = self.proc.memory_info().rss
        if r > self.peak:
            self.peak = r

    @property
    def peak_delta_mb(self) -> float:
        return (self.peak - self.baseline) / MB


# ============ data builder ============

def make_table(n: int, n_uniq_frac: float = 0.9, seed: int = 0) -> pa.Table:
    rng = np.random.default_rng(seed)
    k_count = max(2, int(n * n_uniq_frac))
    keys = [f"k{int(i):09d}" for i in rng.integers(0, k_count, n)]
    return pa.table({
        "row_id": pa.array(np.arange(n, dtype=np.uint64)),
        "key": pa.array(keys, type=pa.string()),
        "v": pa.array(rng.random(n), type=pa.float64()),
    })


# ============ paths ============

def path_a_pandas(table: pa.Table, subset: list, keep) -> pa.Table:
    """A. Current pandas chain (matches _local_dedup v0.8.9 stopgap)."""
    df = table.to_pandas()
    df = df.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    df = df.drop_duplicates(subset=subset, keep=keep)
    df = df.drop(columns=["row_id"])
    return pa.Table.from_pandas(df, preserve_index=False)


def _arrow_dedup(working: pa.Table, original: pa.Table,
                 subset: list, keep) -> pa.Table:
    """Shared algorithm for paths B and C. working = table or encoded."""
    if keep in ("first", "last"):
        agg_fn = "min" if keep == "first" else "max"
        agg = working.group_by(subset).aggregate([("row_id", agg_fn)])
        keep_rids = agg.column(f"row_id_{agg_fn}")
    elif keep is False:
        # count + min: for count==1, min is the sole rid.
        agg = working.group_by(subset).aggregate([
            ("row_id", "count"), ("row_id", "min"),
        ])
        single = agg.filter(pc.equal(agg.column("row_id_count"), 1))
        keep_rids = single.column("row_id_min")
    else:
        raise ValueError(f"unsupported keep={keep!r}")
    mask = pc.is_in(original.column("row_id"), value_set=keep_rids)
    result = original.filter(mask)
    return result.select(
        [c for c in result.schema.names if c != "row_id"]
    )


def path_b_arrow(table: pa.Table, subset: list, keep) -> pa.Table:
    """B. Arrow-native group_by + is_in on original (unencoded) table."""
    return _arrow_dedup(table, table, subset, keep)


def path_c_arrow_dict(table: pa.Table, subset: list, keep) -> pa.Table:
    """C. Arrow-native with dictionary_encode for subset (compact hash)."""
    encoded_cols = {
        c: pc.dictionary_encode(table.column(c)) for c in subset
    }
    working = pa.table({
        **{c: encoded_cols[c] for c in subset},
        "row_id": table.column("row_id"),
    })
    return _arrow_dedup(working, table, subset, keep)


# ============ correctness ============

def _norm(table: pa.Table, subset: list) -> pd.DataFrame:
    df = table.to_pandas()
    cols = subset + ["v"]
    return df[cols].sort_values(cols).reset_index(drop=True)


def check_correctness(n: int = 1000) -> bool:
    print("=== Correctness: A == B == C on all keep variants ===")
    all_ok = True
    table = make_table(n, n_uniq_frac=0.5, seed=1)
    for keep in ("first", "last", False):
        a = _norm(path_a_pandas(table, ["key"], keep), ["key"])
        b = _norm(path_b_arrow(table, ["key"], keep), ["key"])
        c = _norm(path_c_arrow_dict(table, ["key"], keep), ["key"])
        eq_ab = a.equals(b)
        eq_ac = a.equals(c)
        ok = eq_ab and eq_ac
        status = "PASS" if ok else "FAIL"
        print(f"  {status} keep={keep!r}: "
              f"rows A={len(a)} B={len(b)} C={len(c)}  "
              f"A==B:{eq_ab} A==C:{eq_ac}")
        if not ok:
            print(f"    A head:\n{a.head(5)}")
            print(f"    B head:\n{b.head(5)}")
            print(f"    C head:\n{c.head(5)}")
        all_ok = all_ok and ok
    return all_ok


def check_selectivity() -> bool:
    """Verify correctness across duplicate rates (edge: all unique / all dup)."""
    print("\n=== Correctness across selectivity (10% / 50% / 90% dupes) ===")
    all_ok = True
    for frac in (0.1, 0.5, 0.9):
        t = make_table(2000, n_uniq_frac=frac, seed=10)
        for keep in ("first", "last", False):
            a = _norm(path_a_pandas(t, ["key"], keep), ["key"])
            b = _norm(path_b_arrow(t, ["key"], keep), ["key"])
            c = _norm(path_c_arrow_dict(t, ["key"], keep), ["key"])
            if not (a.equals(b) and a.equals(c)):
                print(f"  FAIL frac={frac} keep={keep}")
                all_ok = False
    if all_ok:
        print("  PASS all combinations")
    return all_ok


# ============ RSS bench ============

def _run_path(fn, table, subset, keep) -> dict:
    gc.collect()
    with RSSSampler() as s:
        t0 = time.perf_counter()
        result = fn(table, subset, keep)
        dt = time.perf_counter() - t0
        s.sample_now()
    return {
        "peak_mb": round(s.peak_delta_mb, 2),
        "wall_s": round(dt, 3),
        "out_rows": result.num_rows,
    }


def bench_sizes(sizes=(100_000, 300_000, 1_000_000)) -> dict:
    print(f"\n=== RSS bench (keep='first', 10% dupes) ===")
    results = {"A": [], "B": [], "C": []}
    for n in sizes:
        table = make_table(n, n_uniq_frac=0.9, seed=2)
        tbl_mb = round(table.nbytes / MB, 2)
        print(f"\n  n={n:,}  table={tbl_mb} MB")
        for name, fn in (
            ("A", path_a_pandas),
            ("B", path_b_arrow),
            ("C", path_c_arrow_dict),
        ):
            r = _run_path(fn, table, ["key"], "first")
            results[name].append(r)
            ratio = r["peak_mb"] / tbl_mb if tbl_mb > 0 else 0
            print(f"    {name}: peak={r['peak_mb']:>6.1f} MB "
                  f"({ratio:.2f}× table) wall={r['wall_s']:.2f}s "
                  f"rows={r['out_rows']}")
        del table
        gc.collect()
    return results


def decide(results: dict) -> int:
    print("\n=== Decision ===")
    sizes = (100_000, 300_000, 1_000_000)
    ratios_b = []
    ratios_c = []
    for i, n in enumerate(sizes):
        if i >= len(results["A"]):
            break
        a_mb = results["A"][i]["peak_mb"]
        b_mb = results["B"][i]["peak_mb"]
        c_mb = results["C"][i]["peak_mb"]
        rb = b_mb / a_mb if a_mb > 0 else 0
        rc = c_mb / a_mb if a_mb > 0 else 0
        ratios_b.append(rb)
        ratios_c.append(rc)
        print(f"  n={n:>9,}: A={a_mb:>6.1f}  B={b_mb:>6.1f} ({rb:.2f}×A)  "
              f"C={c_mb:>6.1f} ({rc:.2f}×A)")

    avg_b = float(np.mean(ratios_b)) if ratios_b else 0.0
    avg_c = float(np.mean(ratios_c)) if ratios_c else 0.0
    print(f"\n  avg peak(B)/peak(A) = {avg_b:.2f}")
    print(f"  avg peak(C)/peak(A) = {avg_c:.2f}")

    best = min(avg_b, avg_c)
    best_name = "B" if avg_b <= avg_c else "C"
    if best <= 1.5:
        print(f"\n  DECISION: ADOPT Path {best_name} (avg {best:.2f}× A)")
        return 0
    print(f"\n  DECISION: KEEP pandas (best Arrow {best:.2f}× A > 1.5× threshold)")
    return 1


def main() -> int:
    print("=== proto_dedup_arrow: A (pandas) vs B (arrow) vs C (dict) ===")
    if not check_correctness():
        print("\n*** CORRECTNESS FAILED — do NOT proceed to bench ***")
        return 2
    if not check_selectivity():
        print("\n*** SELECTIVITY FAILED ***")
        return 2
    results = bench_sizes()
    return decide(results)


if __name__ == "__main__":
    raise SystemExit(main())