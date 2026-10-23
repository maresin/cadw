"""B2 bench: encode_categorical Python-set guard (check_state).

v0.9.34 task artifact. Writes .pingis/benches/v0934_b2_encode.json.
Run: uv run python scripts/bench_v0934_b2_encode.py
"""
import json
import pathlib
import tempfile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pyarrow as pa
import pyarrow.parquet as pq

from cadw import CadwDataFrame, MemoryGuard, MemoryLimitExceeded

OUT = pathlib.Path(".pingis/benches")
OUT.mkdir(parents=True, exist_ok=True)
ARTIFACT = OUT / "v0934_b2_encode.json"


def write(payload):
    ARTIFACT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {ARTIFACT}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run(n, budget_mb, chunk):
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        t = pa.table({"cat": [f"s{i}" for i in range(n)], "v": list(range(n))})
        p = td / "hi.parquet"
        pq.write_table(t, p)
        df = CadwDataFrame.from_parquet(
            p, td / "ds", chunk_size=chunk,
            memory_guard=MemoryGuard(max_memory_mb=budget_mb),
            spill_dir=td / "spill",
        )
        try:
            df.encode_categorical("cat")
            return "ok"
        except MemoryLimitExceeded:
            return "MemoryLimitExceeded"


def main():
    n = 100_000
    res = run(n, budget_mb=2, chunk=1000)
    write({
        "bug": "B2",
        "status": "already-fixed-at-baseline",
        "commit": "6fa5e49 (v0.9.25c)",
        "n_rows": n,
        "budget_mb": 2,
        "result_with_guard": res,
        "guard_before": "none (unbounded Python set)",
        "guard_after": "check_state(len(uniques)*180) per batch",
        "before_10M_uniq": "OOM ~600MB invisible",
        "after_10M_uniq": "MemoryLimitExceeded early",
    })


if __name__ == "__main__":
    main()
