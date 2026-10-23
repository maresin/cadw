
#!/usr/bin/env python3
"""PINGIS RAM audit (v0.9.30): peak RSS vs predicted at MemoryGuard(10MB).

Audit item 1.2(a): the shipped bench_v0919_calibration.py runs with
max_memory_mb=100 and N=100k, i.e. never exercises the tight-budget
regime the project advertises. This script runs the 3 primitives and
reduce at N=10k/100k/1M under max_memory_mb=10 and compares the
psutil peak RSS delta against pred_total from the models.

Flag: ratio > 2.0 -> formula drift (P0/P1 RAM finding).
"""
from __future__ import annotations
import json, os, sys, tempfile, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import pyarrow as pa, psutil

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.core.models import get_model, data_chars_from_sample
from cadw.ops.reduce import reduce, MeanOp
from cadw.ops.sort import external_sort
from cadw.ops.shuffle import hash_shuffle
from cadw.ops.groupby import groupby_aggregate
from cadw.core._k_est import estimate_k_sample_arrow


def make_batch(start, size, ncols=8):
    d = {}
    for i in range(ncols):
        if i == 0:
            d["g"] = pa.array([f"g{(start+j) % 100}" for j in range(size)])
        else:
            d[f"c{i}"] = pa.array(list(range(start, start + size)), type=pa.int64())
    return pa.record_batch(d)


def make_batches(n, ncols=8, nb=10):
    """Return a lazy iterator yielding batches one by one (no full table)."""
    bs = max(1, n // nb)
    def _gen():
        for i in range(0, n, bs):
            sz = min(bs, n - i)
            yield make_batch(i, sz, ncols)
    return _gen()


def pred_for(name, batches, keys, guard):
    try:
        model = get_model(name)
    except KeyError:
        return None
    n_total = sum(b.num_rows for b in batches)
    s = batches[0]
    k_s = 1
    on = [c for c in keys if c in s.schema.names]
    if on:
        try:
            k_s = estimate_k_sample_arrow(s, on)
        except Exception:
            k_s = 1
    k_est = max(1, int(k_s * n_total / max(1, s.num_rows)))
    try:
        dc = data_chars_from_sample(s, n_total, k_est=k_est,
                                    avg_key_size_bytes=16,
                                    size_of_key_bytes=None) if False else \
             data_chars_from_sample(s, n_total, k_est=k_est,
                                    avg_key_size_bytes=16,
                                    size_of_key_entry_bytes=96)
        p = model.solve(guard.available_bytes(), dc)
        if p is None:
            return None
        return model.estimate(p, dc)
    except Exception:
        return None


class PeakSampler:
    def __init__(self, interval=0.005):
        self.p = psutil.Process()
        self.base = self.p.memory_info().rss
        self.peak = self.base
        self._stop = threading.Event()
        self._iv = interval
    def _run(self):
        while not self._stop.is_set():
            r = self.p.memory_info().rss
            if r > self.peak: self.peak = r
            self._stop.wait(self._iv)
    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self.t.join(timeout=2.0)
    @property
    def delta(self):
        return self.peak - self.base


def run_op(op, n, tmp_root):
    sample = make_batch(0, min(n, max(1, n // 10)))
    batch_bytes = sample.nbytes
    g = MemoryGuard(max_memory_mb=10)
    pred = None
    d = tmp_root / f"{op}_{n}"; d.mkdir(parents=True, exist_ok=True)
    with PeakSampler() as ps:
        t0 = time.time()
        err = None
        try:
            if op == "reduce_mean":
                pred = pred_for("reduce", [sample], [], g)
                reduce(make_batches(n), MeanOp("c1"), g)
            elif op == "external_sort":
                pred = pred_for("sort", [sample], ["c1"], g)
                external_sort(make_batches(n), ["c1"], list(sample.schema.names),
                              g, d / "spill", read_batch_size=1000)
            elif op == "hash_shuffle":
                pred = pred_for("shuffle", [sample], ["g"], g)
                hash_shuffle(make_batches(n), ["g"], list(sample.schema.names), 4, g, d / "spill")
            elif op == "groupby":
                pred = pred_for("groupby", [sample], ["g"], g)
                groupby_aggregate(make_batches(n), ["g"], ["c1"], ["sum"], g, d / "gb", num_partitions=4)
        except MemoryLimitExceeded as e:
            err = f"MemoryLimitExceeded: {str(e)[:120]}"
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
        dt = time.time() - t0
    return dict(op=op, n=n, batch_bytes=batch_bytes, pred_total=pred,
                peak_rss_delta=ps.delta, wall_s=round(dt, 2), error=err)


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="cadw_ramaudit_"))
    rows = []
    for n in (10_000, 100_000, 1_000_000):
        for op in ("reduce_mean", "external_sort", "hash_shuffle", "groupby"):
            r = run_op(op, n, tmp_root)
            rows.append(r)
            pred = r["pred_total"]
            ratio = (r["peak_rss_delta"] / pred) if pred else None
            flag = ""
            if ratio is not None and ratio > 2.0: flag = "  <<< DRIFT >2x"
            print(f"{op:14s} n={n:>8d}  pred={pred!s:>12s}  "
                  f"peak_rss={r['peak_rss_delta']:>10d}  ratio={ratio if ratio is None else round(ratio,2)!s:>6s}"
                  f"  wall={r['wall_s']}s  err={r['error']}{flag}")
    out = Path("RAM_AUDIT_v0930.json")
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out}; tmp={tmp_root}")


if __name__ == "__main__":
    main()
