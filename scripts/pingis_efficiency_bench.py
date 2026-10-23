
#!/usr/bin/env python3
"""Compact efficiency bench: CADW vs pandas/sklearn at N=100k/1M.
Records wall time and psutil peak RSS delta for each algorithm.
"""
from __future__ import annotations
import gc, os, sys, tempfile, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import pyarrow as pa, psutil

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.sort import external_sort
from cadw.ops.join import hash_join


def _batches(n, ncols=8, nb=None):
    nb = nb or max(1, n // 10000)
    bs = max(1, n // nb)
    def _gen():
        for i in range(0, n, bs):
            sz = min(bs, n - i)
            d = {"g": pa.array([f"g{(i+j)%100}" for j in range(sz)]),
                 **{f"c{k}": pa.array(list(range(i,i+sz)), type=pa.int64()) for k in range(1,ncols)}}
            yield pa.record_batch(d)
    return _gen()


class Sampler:
    def __init__(self, iv=0.005):
        self.p = psutil.Process(); self.b = self.p.memory_info().rss; self.peak=self.b
        self._s = threading.Event(); self._iv = iv
    def _run(self):
        while not self._s.is_set():
            r = self.p.memory_info().rss
            if r > self.peak: self.peak = r
            self._s.wait(self._iv)
    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self
    def __exit__(self, *a): self._s.set(); self.t.join(timeout=2.0)
    @property
    def delta(self): return self.peak - self.b


def bench_groupby(n, mem_mb=200):
    g = MemoryGuard(max_memory_mb=mem_mb)
    td = tempfile.mkdtemp(prefix="eff_gb_")
    with Sampler() as s:
        t0=time.time()
        groupby_aggregate(_batches(n), ["g"], ["c1"], ["sum"], g, td, num_partitions=8)
        dt=time.time()-t0
    import pandas as pd
    df = pd.DataFrame({"g":[f"g{i%100}" for i in range(n)],"c1":range(n)})
    t0=time.time(); df.groupby("g")["c1"].sum(); pd_dt=time.time()-t0
    return dict(op="groupby.sum", n=n, cadw_s=round(dt,2), ref_s=round(pd_dt,2), peak_mb=round(s.delta/1024/1024,1))


def bench_sort(n, mem_mb=200):
    g = MemoryGuard(max_memory_mb=mem_mb)
    td = tempfile.mkdtemp(prefix="eff_sort_")
    with Sampler() as s:
        t0=time.time()
        external_sort(_batches(n), ["c1"], ["g","c1","c2","c3"], g, td, read_batch_size=10000)
        dt=time.time()-t0
    import pandas as pd
    df = pd.DataFrame({"c1":range(n)})
    t0=time.time(); df.sort_values("c1"); pd_dt=time.time()-t0
    return dict(op="external_sort", n=n, cadw_s=round(dt,2), ref_s=round(pd_dt,2), peak_mb=round(s.delta/1024/1024,1))


def bench_join(n, mem_mb=200):
    # 1M = right, 100 = left
    left = pa.record_batch({"k": pa.array([f"g{i}" for i in range(100)]), "v": pa.array(list(range(100)), type=pa.int64())})
    right_batches = _batches(n)
    # rename g -> k
    g = MemoryGuard(max_memory_mb=mem_mb)
    td = tempfile.mkdtemp(prefix="eff_join_")
    def _rb():
        for b in right_batches:
            yield b.rename_columns(["k"] + [f"c{k}" for k in range(1, len(b.schema.names))])
    right = _rb().__next__()
    rs = right.schema
    with Sampler() as s:
        t0=time.time()
        hash_join(iter([left]), _rb(), left.schema, rs, ["k"], g, td)
        dt=time.time()-t0
    import pandas as pd
    L = pd.DataFrame({"k":[f"g{i}" for i in range(100)],"v":range(100)})
    R = pd.DataFrame({"k":[f"g{i%100}" for i in range(n)]})
    t0=time.time(); L.merge(R, on="k"); pd_dt=time.time()-t0
    return dict(op="hash_join", n=n, cadw_s=round(dt,2), ref_s=round(pd_dt,2), peak_mb=round(s.delta/1024/1024,1))


if __name__ == "__main__":
    import json
    out = []
    for n in (100_000, 1_000_000):
        for f in (bench_groupby, bench_sort, bench_join):
            try:
                r = f(n)
            except Exception as e:
                r = {"op":f.__name__, "n":n, "error":f"{type(e).__name__}: {str(e)[:120]}"}
            out.append(r); print(r)
            gc.collect()
    Path("EFFICIENCY_BENCH_v0930.json").write_text(json.dumps(out, indent=2))
    print("done")
