
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
    return peak, best, out

N=500_000
vals=pa.array(np.arange(N,dtype=np.int64)//1000)

def rank_current(vals):
    pv=vals.to_pylist(); ranks=[]; gp=0; dense=1; gs=1; prev=None
    for v in pv:
        gp+=1
        if prev is None: gs=gp; dense=1
        elif v!=prev: gs=gp; dense+=1
        ranks.append(gs); prev=v
    return ranks

def rank_arrow_v1(vals):
    n=len(vals)
    if n==0: return pa.array([],type=pa.int64())
    vnp=vals.to_numpy(zero_copy_only=False)
    newg=np.empty(n,dtype=np.int64); newg[0]=1; newg[1:]=(vnp[1:]!=vnp[:-1])
    grp_id=np.cumsum(newg)-1
    starts=np.flatnonzero(newg)+1
    return pa.array(starts[grp_id].astype(np.int64), type=pa.int64())

def rank_arrow_v2(vals):
    # np.repeat, no cumsum
    n=len(vals)
    if n==0: return pa.array([],type=pa.int64())
    vnp=vals.to_numpy(zero_copy_only=False)
    starts=np.flatnonzero(np.concatenate(([True], vnp[1:]!=vnp[:-1]))) + 1
    runs=np.diff(np.append(starts, n+1))
    return pa.array(np.repeat(starts, runs).astype(np.int64), type=pa.int64())

pk0,dt0,out0 = mem_time(lambda: rank_current(vals))
pk1,dt1,out1 = mem_time(lambda: rank_arrow_v1(vals))
pk2,dt2,out2 = mem_time(lambda: rank_arrow_v2(vals))
ref = out0
assert list(out1.to_numpy()) == ref, "v1 mismatch"
assert list(out2.to_numpy()) == ref, "v2 mismatch"
print(f"RANK N={N} current: {pk0/1e6:.2f}MB {dt0*1000:.1f}ms")
print(f"RANK v1(cumsum): {pk1/1e6:.2f}MB {dt1*1000:.1f}ms ({dt0/dt1:.1f}x, mem {(pk1-pk0)/1e6:+.2f}MB)")
print(f"RANK v2(repeat): {pk2/1e6:.2f}MB {dt2*1000:.1f}ms ({dt0/dt2:.1f}x, mem {(pk2-pk0)/1e6:+.2f}MB)")

# ---- groupby key extraction ----
K=50_000
keys=pa.array([f"g_{i%K}" for i in range(N)])
vals_arr=pa.array(np.random.rand(N))

def groupby_current_pylist(keys, vals_arr):
    key_list = keys.to_pylist()
    val_list = vals_arr.to_pylist()
    sums = {}
    for k,v in zip(key_list, val_list):
        sums[k] = sums.get(k, 0.0) + v
    return sums

def groupby_arrow_dict(keys, vals_arr):
    d = keys.dictionary_encode()
    idx = d.indices.to_numpy(zero_copy_only=False)
    vnp = vals_arr.to_numpy(zero_copy_only=False)
    nuniq = len(d.dictionary)
    sums = np.zeros(nuniq, dtype=np.float64)
    np.add.at(sums, idx, vnp)
    return dict(zip(d.dictionary.to_pylist(), sums.tolist()))

pk0,dt0,o0 = mem_time(lambda: groupby_current_pylist(keys, vals_arr))
pk1,dt1,o1 = mem_time(lambda: groupby_arrow_dict(keys, vals_arr))
# verify
for k in o0:
    assert abs(o0[k]-o1[k]) < 1e-9
print(f"GB key: current {pk0/1e6:.2f}MB {dt0*1000:.1f}ms")
print(f"GB key: dict    {pk1/1e6:.2f}MB {dt1*1000:.1f}ms ({dt0/dt1:.1f}x, mem {(pk1-pk0)/1e6:+.2f}MB)")

# ---- Column.to_list guard realism for long strings ----
# Column.to_list assumes 50 B/value. Test with long strings.
for slen in [10, 100, 1000, 10000]:
    n=10000
    strings=pa.array(["x"*slen for _ in range(n)])
    peak, dt, out = mem_time(lambda: strings.to_pylist(), reps=1)
    raw = n*slen
    per_val = peak/n
    print(f"to_pylist n={n} slen={slen}: peak={peak/1e6:.2f}MB = {per_val:.0f} B/val (guard assumes 50)")
    del out
