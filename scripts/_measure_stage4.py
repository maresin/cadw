
import gc, time, sys, os, tracemalloc
sys.path.insert(0, os.getcwd())
import pyarrow as pa
import numpy as np
import xxhash

def mem_peak(fn):
    gc.collect(); tracemalloc.start()
    out = fn()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop(); return peak, out

def best_time(fn, reps=3):
    best=None
    for _ in range(reps):
        gc.collect(); t0=time.perf_counter(); fn(); dt=time.perf_counter()-t0
        best=dt if best is None or dt<best else best
    return best

N=300_000; BATCH=50_000; K=2
values=pa.array(np.arange(N,dtype=np.int64))
batches=[pa.record_batch([values.slice(i,BATCH)],names=["v"]) for i in range(0,N,BATCH)]

def shift_current(batches):
    from collections import deque
    bufs={"v":deque(maxlen=K)}; outs=[]
    for b in batches:
        col_vals=b.column("v"); n=b.num_rows
        prev=list(bufs["v"]); fill_n=max(0,K-len(prev)); parts=[]
        if fill_n>0: parts.append(pa.nulls(fill_n,type=col_vals.type))
        if prev: parts.append(pa.array(prev,type=col_vals.type))
        if n>K: parts.append(col_vals.slice(0,n-K))
        out=pa.concat_arrays(parts) if parts else pa.array([],type=col_vals.type)
        if len(out)>n: out=out.slice(0,n)
        elif len(out)<n: out=pa.concat_arrays([out,pa.nulls(n-len(out),type=col_vals.type)])
        outs.append(out)
        for v in col_vals.to_pylist(): bufs["v"].append(v)
    return outs

def shift_arrow(batches):
    from collections import deque
    bufs={"v":deque(maxlen=K)}; outs=[]
    for b in batches:
        col_vals=b.column("v"); n=b.num_rows
        prev_arr=pa.concat_arrays(list(bufs["v"])) if bufs["v"] else None
        prev_len=len(prev_arr) if prev_arr is not None else 0
        fill_n=max(0,K-prev_len); parts=[]
        if fill_n>0: parts.append(pa.nulls(fill_n,type=col_vals.type))
        if prev_arr is not None: parts.append(prev_arr)
        if n>K: parts.append(col_vals.slice(0,n-K))
        out=pa.concat_arrays(parts) if parts else pa.array([],type=col_vals.type)
        if len(out)>n: out=out.slice(0,n)
        elif len(out)<n: out=pa.concat_arrays([out,pa.nulls(n-len(out),type=col_vals.type)])
        outs.append(out)
        if n>=K: bufs["v"].clear(); bufs["v"].append(col_vals.slice(n-K,K))
        else: bufs["v"].append(col_vals)
    return outs

pk_a,_=mem_peak(lambda: shift_current(batches))
pk_b,_=mem_peak(lambda: shift_arrow(batches))
dt_a=best_time(lambda: shift_current(batches)); dt_b=best_time(lambda: shift_arrow(batches))
assert pa.concat_arrays(shift_current(batches)).to_pylist()==pa.concat_arrays(shift_arrow(batches)).to_pylist()
print(f"SHIFT: mem {pk_a/1e6:.2f}MB -> {pk_b/1e6:.2f}MB | time {dt_a*1000:.1f}ms -> {dt_b*1000:.1f}ms | {dt_a/dt_b:.1f}x")

vals=pa.array(np.arange(N,dtype=np.int64)//1000)
def rank_current(vals):
    pv=vals.to_pylist(); ranks=[]; gp=0; dense=1; gs=1; prev=None
    for v in pv:
        gp+=1
        if prev is None: gs=gp; dense=1
        elif v!=prev: gs=gp; dense+=1
        ranks.append(gs); prev=v
    return ranks
def rank_arrow(vals):
    # min/dense: detect group boundaries with arrow, use cumulative counts
    import pyarrow.compute as pc
    n=len(vals)
    if n==0: return pa.array([],type=pa.int64())
    # new_group[i] = (i==0) or vals[i]!=vals[i-1]
    vnp = vals.to_numpy(zero_copy_only=False)
    same = (vnp[1:] == vnp[:-1])
    newg = np.empty(n, dtype=np.int64); newg[0]=1; newg[1:] = ~same
    grp_id=np.cumsum(newg)-1
    # min: group start position (1-based) = index of first occurrence +1
    starts=np.flatnonzero(newg)+1
    out=starts[grp_id].astype(np.int64)
    return pa.array(out,type=pa.int64())
pk_c,_=mem_peak(lambda: rank_current(vals))
dt_c=best_time(lambda: rank_current(vals))
pk_c2,_=mem_peak(lambda: rank_arrow(vals))
dt_c2=best_time(lambda: rank_arrow(vals))
assert rank_current(vals)==rank_arrow(vals).to_pylist()
print(f"RANK min: mem {pk_c/1e6:.2f}MB -> {pk_c2/1e6:.2f}MB | time {dt_c*1000:.1f}ms -> {dt_c2*1000:.1f}ms | {dt_c/dt_c2:.1f}x")

strings=pa.array([f"key_{i%5000}" for i in range(N)])
from cadw.ops.shuffle import _string_hash_column
def hash_current(arr): return _string_hash_column(arr)
def hash_dict(arr):
    d=arr.dictionary_encode(); uniq=d.dictionary
    idx=d.indices.to_numpy(zero_copy_only=False)
    uh=np.empty(len(uniq),dtype=np.uint64)
    for i,v in enumerate(uniq.to_pylist()):
        uh[i]=0 if v is None else xxhash.xxh64(v.encode("utf-8"),seed=0).intdigest()
    out=np.empty(len(arr),dtype=np.uint64)
    valid=idx>=0; out[valid]=uh[idx[valid]]; out[~valid]=0
    return out
pk_d,_=mem_peak(lambda: hash_current(strings)); dt_d=best_time(lambda: hash_current(strings))
pk_e,_=mem_peak(lambda: hash_dict(strings)); dt_e=best_time(lambda: hash_dict(strings))
assert np.array_equal(hash_current(strings), hash_dict(strings))
print(f"STRHASH: mem {pk_d/1e6:.2f}MB -> {pk_e/1e6:.2f}MB | time {dt_d*1000:.1f}ms -> {dt_e*1000:.1f}ms | {dt_d/dt_e:.1f}x")
