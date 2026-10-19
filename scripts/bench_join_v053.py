"""Bench: join 2M × 400k inner, budget 80 MB. Compare to pre-v0.5.3 (pandas)."""
import gc, sys, tempfile, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from cadw import CadwDataFrame, MemoryGuard

proc = psutil.Process()


class PeakSampler(threading.Thread):
    def __init__(self, interval=0.01):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            r = proc.memory_info().rss
            if r > self.peak:
                self.peak = r
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join()


def rss_mb():
    return proc.memory_info().rss / (1024 * 1024)


def run():
    n_left, n_right = 2_000_000, 400_000
    print(f"[v0.5.3] building input: {n_left}L × {n_right}R")
    left = pa.table({
        "k": pa.array([i % n_right for i in range(n_left)], type=pa.int64()),
        "s": pa.array([f"L{i:09d}" for i in range(n_left)], type=pa.string()),
        "v": pa.array(list(range(n_left)), type=pa.float64()),
    })
    right = pa.table({
        "k": pa.array(list(range(n_right)), type=pa.int64()),
        "w": pa.array(list(range(n_right)), type=pa.float64()),
    })

    with tempfile.TemporaryDirectory(prefix="cadw_v053_") as td:
        td = Path(td)
        left_p = td / "left.parquet"
        right_p = td / "right.parquet"
        pq.write_table(left, left_p, row_group_size=50_000)
        pq.write_table(right, right_p, row_group_size=50_000)
        del left, right
        gc.collect()

        left_df = CadwDataFrame.from_parquet(
            left_p, td / "ds_left", chunk_size=20_000,
            memory_guard=MemoryGuard(max_memory_mb=80),
            spill_dir=td / "spill_left",
        )
        right_df = CadwDataFrame.from_parquet(
            right_p, td / "ds_right", chunk_size=20_000,
            memory_guard=MemoryGuard(max_memory_mb=80),
            spill_dir=td / "spill_right",
        )

        gc.collect()
        r0 = rss_mb()
        sampler = PeakSampler()
        sampler.start()
        t0 = time.time()
        result = left_df.merge(right_df, on="k", num_partitions=16)
        out_p = td / "out.parquet"
        result.write_parquet(out_p)
        elapsed = time.time() - t0
        sampler.stop()
        gc.collect()

        peak = sampler.peak / (1024 * 1024)
        delta = peak - r0
        rows = pq.read_metadata(out_p).num_rows
        print(f"[v0.5.3] Δpeak={delta:+.1f}MB  t={elapsed:.1f}s  out_rows={rows}")
        print(f"  (pre-v0.5.3 baseline: t=14.1s, Δpeak=+2.9MB)")
        print(f"  speedup: {14.1 / max(elapsed, 0.01):.2f}×")


if __name__ == "__main__":
    run()
