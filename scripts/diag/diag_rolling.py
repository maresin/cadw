
import gc, time, psutil, tracemalloc
import numpy as np, pyarrow as pa
import cadw
from cadw import CadwDataFrame
from cadw.ts import rolling_stats

rng = np.random.default_rng(0)
n = 2000
cols = {
    "y": pa.array(rng.normal(0,1,n).cumsum()+100.0, type=pa.float64()),
    "ts": pa.array([1700000000000000000+i*3600*10**9 for i in range(n)], type=pa.timestamp("ns")),
    "g": pa.array([f"g{i%4}" for i in range(n)], type=pa.string()),
}
df = CadwDataFrame.from_table(pa.table(cols))
guard = cadw.MemoryGuard(max_memory_mb=32)
df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
gc.collect()
proc = psutil.Process(); base = proc.memory_info().rss
tracemalloc.start()
out = rolling_stats(df2, "y", windows=(7,14,30), stats=("mean","std","q50","skew","slope","autocorr_lag1"), group_col="g", time_col="ts")
mx = 0; nb = 0
for b in out._iter_arrow_batches(batch_size=500):
    nb += b.num_rows
    r = proc.memory_info().rss
    if r > mx: mx = r
cur, peak_tm = tracemalloc.get_traced_memory(); tracemalloc.stop()
print("n_out", nb)
print("rss_delta_mb", round((mx-base)/1024/1024,2))
print("rss_final_delta_mb", round((proc.memory_info().rss-base)/1024/1024,2))
print("tm_peak_mb", round(peak_tm/1024/1024,2))
print("tm_cur_mb", round(cur/1024/1024,2))
