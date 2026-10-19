#!/usr/bin/env bash
# diagnose_v050: проверить Table.join API + измерить RSS B1a на новом коде.
# НЕ модифицирует ничего — только собирает данные для v0.5.2.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
echo "=== HEAD ==="
git log --oneline -1
echo ""

# ============ Part 1: API availability ============
echo "=== API check ==="
uv run python << 'PYEOF'
import pyarrow as pa
import pyarrow.compute as pc
import inspect

print(f"pyarrow: {pa.__version__}")

# Table.join
t = pa.table({"k": [1, 2], "v": [10, 20]})
print(f"\nTable.join: {hasattr(t, 'join')}")
if hasattr(t, 'join'):
    try:
        print(f"  signature: {inspect.signature(t.join)}")
    except Exception as e:
        print(f"  signature unavailable: {e}")

# pc.rank, pc.quantile (для P11)
funcs = pc.list_functions() if hasattr(pc, 'list_functions') else []
for name in ('rank', 'quantile', 'rank_quantile', 'value_counts',
             'dictionary_encode', 'is_in', 'sort_indices', 'bucketize'):
    # dictionary_encode — метод Array, не pc.*
    in_pc = name in funcs
    has_method = hasattr(pa.array([1]), name) if name == 'dictionary_encode' else None
    marker = "✓" if (in_pc or has_method) else "✗"
    print(f"{marker} {name}: pc={'yes' if in_pc else 'no'}"
          + (f", arr={has_method}" if has_method is not None else ""))
PYEOF
echo ""

# ============ Part 2: B1a bench — RSS на старом vs новом коде ============
echo "=== B1a: join 1:M RSS benchmark (budget 80MB) ==="
uv run python << 'PYEOF'
import gc, os, threading, time
from pathlib import Path

import psutil
import pyarrow as pa

from cadw import CadwDataFrame, MemoryGuard

proc = psutil.Process()


class PeakSampler(threading.Thread):
    def __init__(self, interval=0.01):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            r = proc.memory_info().rss
            if r > self.peak:
                self.peak = r
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self.join()


def rss_mb():
    return proc.memory_info().rss / (1024 * 1024)


def run_b1a(tmpdir: Path, budget_mb: int = 80, label: str = "B1a"):
    """1:M join: 2M left × 400k right; output ~2M rows.

    Mirror Pingis B1a. But write output to parquet to force full materialize
    (if merge doesn't do it).
    """
    import pyarrow.parquet as pq

    n_left = 2_000_000
    n_right = 400_000
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
        # Force full materialize + write to confirm no hidden accumulate
        out_p = tmpdir / "out.parquet"
        result.write_parquet(out_p)
        out_n = pq.read_metadata(out_p).num_rows
        elapsed = time.time() - t0
        error = None
    except Exception as e:
        elapsed = time.time() - t0
        out_n = None
        error = f"{type(e).__name__}: {e}"
    finally:
        sampler.stop()

    gc.collect()
    r1 = rss_mb()
    peak = sampler.peak / (1024 * 1024)
    print(
        f"[{label}] budget={budget_mb}MB rss0={r0:.1f} "
        f"peak={peak:.1f} Δpeak={peak-r0:+.1f}MB "
        f"Δend={r1-r0:+.1f}MB t={elapsed:.1f}s "
        f"out_rows={out_n} err={error}"
    )
    if error:
        print(f"[{label}] RESULT: FAIL — {error}")
    elif peak - r0 > budget_mb * 1.5:
        print(f"[{label}] RESULT: OOM-like (>1.5× budget)")
    elif peak - r0 > budget_mb:
        print(f"[{label}] RESULT: OVER (>budget)")
    else:
        print(f"[{label}] RESULT: OK (peak ≤ budget)")
    return peak - r0


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory(prefix="cadw_b1a_") as td:
        run_b1a(Path(td), budget_mb=80, label="B1a")
PYEOF