
import gc, time, sys, os, tracemalloc
sys.path.insert(0, os.getcwd())
import pyarrow as pa
import numpy as np

def mem_time(fn, reps=3):
    gc.collect(); tracemalloc.start()
    out = fn(); cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    best=None
    for _ in range(reps):
        gc.collect(); t0=time.perf_counter(); fn(); dt=time.perf_counter()-t0
        best=dt if best is None or dt<best else best
    return peak, best

N=1_000_000
arr = pa.array(np.arange(N, dtype=np.int64))

p1,t1 = mem_time(lambda: arr.to_pylist())
print(f"to_pylist  peak={p1/1e6:.2f}MB time={t1*1000:.1f}ms")
p2,t2 = mem_time(lambda: arr.to_numpy(zero_copy_only=False))
print(f"to_numpy   peak={p2/1e6:.2f}MB time={t2*1000:.1f}ms")

# is_monotonic current vs numpy
def mono_current(values):
    v = values.to_pylist(); prev=None; first=True
    for x in v:
        if x is None: continue
        if first: prev=x; first=False
        elif x < prev: return False
        prev=x
    return True

def mono_numpy(values):
    v = values.to_numpy(zero_copy_only=False)
    # assume no nulls for benchmark (real fix uses pc.is_valid mask)
    return bool(np.all(np.diff(v) >= 0))

p3,t3 = mem_time(lambda: mono_current(arr))
p4,t4 = mem_time(lambda: mono_numpy(arr))
print(f"mono current peak={p3/1e6:.2f}MB time={t3*1000:.1f}ms")
print(f"mono numpy   peak={p4/1e6:.2f}MB time={t4*1000:.1f}ms ({t3/t4:.1f}x)")
assert mono_current(arr) == mono_numpy(arr) == True
