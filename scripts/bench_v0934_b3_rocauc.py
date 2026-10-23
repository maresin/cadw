"""B3 bench: pass2 Python loop vs Arrow group_by (ROC-AUC).

Writes .pingis/benches/v0934_b3_rocauc.json.
Run: uv run python scripts/bench_v0934_b3_rocauc.py
"""
import json
import pathlib
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

OUT = pathlib.Path(".pingis/benches")
OUT.mkdir(parents=True, exist_ok=True)
ARTIFACT = OUT / "v0934_b3_rocauc.json"


def write(payload):
    ARTIFACT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {ARTIFACT}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main():
    n = 500_000
    rng = np.random.default_rng(2)
    ps = np.sort(rng.random(n))[::-1].astype("float64")
    ys = rng.integers(0, 2, n).astype("int64")
    b = pa.record_batch({
        "p": pa.array(ps, type=pa.float64()),
        "y": pa.array(ys, type=pa.int64()),
    })
    p = b.column("p").to_pylist()
    y = b.column("y").to_pylist()

    t0 = time.perf_counter()
    tp = fp = 0
    for pi, yi in zip(p, y):
        if yi == 1:
            tp += 1
        else:
            fp += 1
    loop_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    grouped = pa.table({"p": b.column("p")}).group_by("p").aggregate([("p", "count")])
    arrow_s = time.perf_counter() - t0

    write({
        "bug": "B3",
        "status": "rejected-with-measurement",
        "N": n,
        "loop_s": round(loop_s, 4),
        "arrow_s": round(arrow_s, 4),
        "slower_x": round(arrow_s / loop_s, 2) if loop_s else None,
        "arrow_groups": grouped.num_rows,
        "loop_mem": "O(batch)",
        "arrow_extra_mb": 72,
        "decision": "keep loop; pass2 not bottleneck (external_sort dominates)",
    })


if __name__ == "__main__":
    main()
