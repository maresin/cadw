"""Bench: sort strings + numeric, measure Δpeak RSS vs old (Python-list) buffer.

Compare to Pingis B3a: +31.9 MB on 30 MB budget (old code).
"""
import gc, sys, tempfile, threading, time
from pathlib import Path

import psutil
import pyarrow as pa

# Ensure project root on sys.path when run as `python scripts/bench_p5_sort.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.sort import external_sort

proc = psutil.Process()


class PeakSampler(threading.Thread):
    def __init__(self, interval=0.005):
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


def batch_iter(table, bs):
    for i in range(0, table.num_rows, bs):
        yield table.slice(i, bs).to_batches()[0]


def run(label, n, budget_mb):
    table = pa.table({
        "s": pa.array([f"{(i*7919)%n:07d}" for i in range(n)], type=pa.string()),
        "v": pa.array(list(range(n)), type=pa.float64()),
    })
    print(f"[{label}] input: {table.nbytes/1048576:.1f}MB ({n} rows), "
          f"budget={budget_mb}MB")

    gc.collect()
    r0 = rss_mb()
    sampler = PeakSampler()
    sampler.start()
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as td:
            path = external_sort(
                batch_iter(table, 10000), ["s"], ["s", "v"],
                MemoryGuard(max_memory_mb=budget_mb),
                td, ascending=True, log=False,
            )
            dt = time.time() - t0
    except Exception as e:
        dt = time.time() - t0
        err = f"{type(e).__name__}: {str(e)[:150]}"
    else:
        err = None
    finally:
        sampler.stop()

    gc.collect()
    r1 = rss_mb()
    peak = sampler.peak / (1024 * 1024)
    delta = peak - r0
    print(f"[{label}] rss0={r0:.1f} peak={peak:.1f} Δpeak={delta:+.1f}MB "
          f"Δend={r1-r0:+.1f}MB t={dt:.1f}s err={err}")
    if err:
        print(f"[{label}] RESULT: FAIL")
    elif delta > budget_mb:
        print(f"[{label}] RESULT: OVER (>{budget_mb}MB budget)")
    else:
        print(f"[{label}] RESULT: OK (peak ≤ budget)")
    return delta


if __name__ == "__main__":
    print(f"=== P5 sort bench (baseline Pingis B3a: +31.9MB on 30MB) ===")
    run("B3a-new", n=1_000_000, budget_mb=30)
