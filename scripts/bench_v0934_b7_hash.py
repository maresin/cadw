"""B7 bench: _string_hash_column dict-encode vs per-element (already fixed).

Writes .pingis/benches/v0934_b7_hash.json.
Run: uv run python scripts/bench_v0934_b7_hash.py
"""
import json
import pathlib
import time
import tracemalloc
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow as pa
import xxhash

from cadw.ops.shuffle import _NULL_SENTINEL, _string_hash_column

OUT = pathlib.Path(".pingis/benches")
OUT.mkdir(parents=True, exist_ok=True)
ARTIFACT = OUT / "v0934_b7_hash.json"


def write(payload):
    ARTIFACT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {ARTIFACT}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def old_way(arr):
    out = np.empty(len(arr), dtype=np.uint64)
    for i, v in enumerate(arr.to_pylist()):
        if v is None:
            out[i] = _NULL_SENTINEL
        else:
            data = v.encode("utf-8") if isinstance(v, str) else v
            out[i] = xxhash.xxh64(data, seed=0).intdigest()
    return out


def main():
    n = 300_000
    k = 10_000
    arr = pa.array([f"val{i % k}" for i in range(n)])

    tracemalloc.start()
    t0 = time.perf_counter()
    old = old_way(arr)
    old_ms = (time.perf_counter() - t0) * 1000
    _, old_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    new = _string_hash_column(arr)
    new_ms = (time.perf_counter() - t0) * 1000
    _, new_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert np.array_equal(old, new), "B7 correctness regression"

    write({
        "bug": "B7",
        "status": "already-fixed-at-baseline",
        "commit": "v0.9.18 (AUDIT 1.3)",
        "n_rows": n,
        "k_uniques": k,
        "before_ms": round(old_ms, 1),
        "after_ms": round(new_ms, 1),
        "speedup_x": round(old_ms / new_ms, 1) if new_ms else None,
        "before_peak_mb": round(old_peak / 1e6, 1),
        "after_peak_mb": round(new_peak / 1e6, 1),
        "correctness_equal": True,
    })


if __name__ == "__main__":
    main()
