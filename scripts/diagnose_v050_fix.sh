#!/usr/bin/env bash
# diagnose_v050_fix: rename _stop → _stop_evt в sampler; rerun B1a.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "=== B1a: join 1:M RSS benchmark (budget 80MB) ==="
uv run python << 'PYEOF'
import gc, threading, time, tempfile
from pathlib import Path

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


def run_b1a(tmpdir: Path, budget_mb: int = 80, label: str = "B1a"):
    n_left = 2_000_000
    n_right = 400_000
    print(f"[{label}] building input (2M left, 400k right)...")
    left = pa.table({
        "k": pa.array([i % n_right for i in range(n_left)], type=pa.int64()),
        "s": pa.array([f"L{i:09d}" for i in range(n_left)], type=pa.string()),
        "v": pa.array(list(range(n_left)), type=pa.float64()),
    })
    right = pa.table({
        "k": pa.array(list(range(n_right)), type=pa.int64()),
        "w": pa.array(list(range(n_right)), type=pa.float64()),
    })
    print(f"[{label}] input: left={left.nbytes/1048576:.1f}MB "
          f"({n_left} rows), right={right.nbytes/1048576:.1f}MB "
          f"({n_right} rows)")

    left_p = tmpdir / "left.parquet"
    right_p = tmpdir / "right.parquet"
    pq.write_table(left, left_p, row_group_size=50_000)
    pq.write_table(right, right_p, row_group_size=50_000)
    del left, right
    gc.collect()

    left_df = CadwDataFrame.from_parquet(
        left_p, tmpdir / "ds_left", chunk_size=20_000,
        memory_guard=MemoryGuard(max_memory_mb=budget_mb),
        spill_dir=tmpdir / "spill_left",
    )
    right_df = CadwDataFrame.from_parquet(
        right_p, tmpdir / "ds_right", chunk_size=20_000,
        memory_guard=MemoryGuard(max_memory_mb=budget_mb),
        spill_dir=tmpdir / "spill_right",
    )

    gc.collect()
    r0 = rss_mb()
    sampler = PeakSampler()
    sampler.start()
    t0 = time.time()
    try:
        result = left_df.merge(
            right_df, on="k", how="inner", num_partitions=16,
        )
        out_p = tmpdir / "out.parquet"
        result.write_parquet(out_p)
        out_n = pq.read_metadata(out_p).num_rows
        elapsed = time.time() - t0
        error = None
    except Exception as e:
        elapsed = time.time() - t0
        out_n = None
        error = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        sampler.stop()

    gc.collect()
    r1 = rss_mb()
    peak = sampler.peak / (1024 * 1024)
    delta = peak - r0
    print(
        f"[{label}] budget={budget_mb}MB rss0={r0:.1f} "
        f"peak={peak:.1f} Δpeak={delta:+.1f}MB "
        f"Δend={r1-r0:+.1f}MB t={elapsed:.1f}s "
        f"out_rows={out_n} err={error}"
    )

    # Verdict
    if error:
        print(f"[{label}] RESULT: FAIL — {error}")
    elif delta > budget_mb * 2:
        print(f"[{label}] RESULT: CRIT (>{2}x budget)")
    elif delta > budget_mb * 1.5:
        print(f"[{label}] RESULT: OOM-like (1.5x-2x budget)")
    elif delta > budget_mb:
        print(f"[{label}] RESULT: OVER (>budget, <1.5x)")
    else:
        print(f"[{label}] RESULT: OK (peak ≤ budget)")
    print(f"[{label}] (Pingis baseline: +140 MB / 2.5x budget on pre-v0.5.1)")


with tempfile.TemporaryDirectory(prefix="cadw_b1a_") as td:
    run_b1a(Path(td), budget_mb=80, label="B1a")
PYEOF

echo ""
echo "=== summary ==="
echo "Compare Δpeak above to Pingis baseline +140 MB (2.5× budget on old code)."
echo "Expected: Δpeak ≤ 80 MB → P2 fix works, P1 (to_pandas) may still inflate."
echo "          Δpeak > 140 MB → regression (check join.py after fix)."