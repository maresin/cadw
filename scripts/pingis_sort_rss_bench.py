
import os, sys, tempfile, threading, time, json
sys.path.insert(0, ".")
import pyarrow as pa, psutil

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.sort import external_sort

class S:
    def __init__(self, iv=0.005):
        self.p=psutil.Process(); self.b=self.p.memory_info().rss; self.peak=self.b
        self._s=threading.Event(); self._iv=iv
    def _run(self):
        while not self._s.is_set():
            r=self.p.memory_info().rss
            if r>self.peak: self.peak=r
            self._s.wait(self._iv)
    def __enter__(self):
        self.t=threading.Thread(target=self._run,daemon=True); self.t.start(); return self
    def __exit__(self,*a): self._s.set(); self.t.join(timeout=2.0)
    @property
    def delta(self): return self.peak-self.b

def run(n, mem_mb, chunk_rows):
    g = MemoryGuard(max_memory_mb=mem_mb)
    def gen():
        for i in range(0, n, chunk_rows):
            sz = min(chunk_rows, n - i)
            yield pa.record_batch({
                "g": pa.array([f"g{(i+j)%100}" for j in range(sz)]),
                "c1": pa.array(list(range(i, i+sz)), type=pa.int64()),
                "c2": pa.array([0]*sz, type=pa.int64()),
                "c3": pa.array([0]*sz, type=pa.int64()),
                "c4": pa.array([0]*sz, type=pa.int64()),
                "c5": pa.array([0]*sz, type=pa.int64()),
                "c6": pa.array([0]*sz, type=pa.int64()),
                "c7": pa.array([0]*sz, type=pa.int64()),
            })
    td = tempfile.mkdtemp()
    with S() as s:
        t0=time.time()
        external_sort(gen(), ["c1"], ["g","c1","c2","c3","c4","c5","c6","c7"], g, td, read_batch_size=1000)
        dt=time.time()-t0
    return dict(n=n, mem_mb=mem_mb, chunk_rows=chunk_rows, peak_mb=round(s.delta/1048576,1), wall_s=round(dt,2))

if __name__ == "__main__":
    res = []
    for n in (1_000_000, 10_000_000):
        r = run(n, 100, 100_000)
        res.append(r); print(r); sys.stdout.flush()
    pathlib = None
    import pathlib as _p
    _p.Path("SORT_RSS_BENCH_v0930.json").write_text(json.dumps(res, indent=2))
