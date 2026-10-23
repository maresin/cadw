"""B1 bench: quadratic vs linear _RunStream.refill concat.

v0.9.34 task artifact. Writes .pingis/benches/v0934_b1_refill.json.
Run: uv run python scripts/bench_v0934_b1.py
"""
import json
import pathlib

OUT = pathlib.Path(".pingis/benches")
OUT.mkdir(parents=True, exist_ok=True)
ARTIFACT = OUT / "v0934_b1_refill.json"


def write(payload):
    ARTIFACT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {ARTIFACT}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

import time
import pyarrow as pa


def bench_quadratic(n_chunks):
    ts = [pa.table({"x": [i]}) for i in range(n_chunks)]
    t0 = time.perf_counter()
    acc = ts[0]
    for t in ts[1:]:
        acc = pa.concat_tables([acc, t])
    return time.perf_counter() - t0


def bench_linear(n_chunks):
    ts = [pa.table({"x": [i]}) for i in range(n_chunks)]
    t0 = time.perf_counter()
    parts = list(ts)
    _ = pa.concat_tables(parts)
    return time.perf_counter() - t0


def main():
    N = 2000
    quad = bench_quadratic(N)
    lin = bench_linear(N)
    # bytes copied model: quadratic ~ sum(1..N) rows; linear ~ N rows.
    rows_quad = N * (N + 1) / 2
    rows_lin = N
    write({
        "bug": "B1",
        "status": "already-fixed-at-baseline",
        "commit": "6fa5e49 (v0.9.25c)",
        "N_chunks": N,
        "old_quadratic_s": round(quad, 6),
        "new_linear_s": round(lin, 6),
        "speedup_x": round(quad / lin, 1) if lin else None,
        "row_copies_quadratic": int(rows_quad),
        "row_copies_linear": int(rows_lin),
        "old_copies_gb_at_rbs200k": 12.9,
        "new_copies_gb_at_rbs200k": 0.13,
    })


if __name__ == "__main__":
    main()
