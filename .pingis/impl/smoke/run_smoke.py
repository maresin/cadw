"""Smoke test runner for reference impls under .pingis/impl/.

Each impl is invoked as a separate `python <impl>` subprocess. Peak RSS
is measured with resource.getrusage(RUSAGE_CHILDREN).ru_maxrss (KiB on
Linux) — this is the max RSS over the process lifetime of each child,
which is what we want for out-of-core validation.

The runner fails (non-zero exit) if any impl returns non-zero.
Outputs a Markdown table to .pingis/impl/SMOKE_RESULTS.md.

Usage: uv run python .pingis/impl/smoke/run_smoke.py
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
IMPLS = [
    ("M", "bandits.py",           "EG/UCB1/TS q_ monotone increasing in true_p; best arm = last"),
    ("N", "kernel_density.py",    "n_centers=2048, n_seen=100000; log_lik finite"),
    ("I", "kaplan_meier.py",     "S(1)~0.93, S(10)~0.52, median ~10.7"),
    ("I", "weibull_aft.py",      "n_events=75033; beta finite; sigma>0 (plumbing-only)"),
    ("J", "complement_nb.py",    "C=4 V=200; feature_log_prob finite"),
    ("J", "nmf.py",              "k=8 V=128; H>=0; recon_err finite"),
    ("O", "reservoir.py",        "K=5000 d=16; mean~0 std~1"),
    ("O", "stratified_split.py", "fold_counts balanced within +-1"),
    ("O", "one_hot_encoder.py",  "c0=12, c1=8 categories; transform rows=5000"),
]


def run_one(sec: str, name: str) -> dict:
    path = ROOT / ".pingis" / "impl" / sec / name
    t0 = time.time()
    pre_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    r = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    dt = time.time() - t0
    post_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    # ru_maxrss is cumulative max over ALL children so far; take delta of
    # max fields is not meaningful. Instead, run each child with a fresh
    # subprocess that itself reports getrusage(RUSAGE_SELF) in its stdout.
    # All impls already print "peak RSS = X MiB" from RUSAGE_SELF; parse it.
    peak_mb = None
    for line in (r.stdout or "").splitlines():
        if "peak RSS =" in line:
            try:
                peak_mb = float(line.split("peak RSS =")[1].split("MiB")[0].strip())
            except Exception:
                pass
    return {
        "section": sec, "name": name,
        "runtime_s": round(dt, 3),
        "peak_rss_mb": peak_mb,
        "exit": r.returncode,
        "stdout_tail": (r.stdout or "").strip().splitlines()[-3:],
        "stderr_tail": (r.stderr or "").strip().splitlines()[-3:] if r.returncode != 0 else [],
    }


def main() -> int:
    results = []
    any_fail = False
    for sec, name, check in IMPLS:
        print(f"[run] {sec}/{name} ...", flush=True)
        res = run_one(sec, name)
        res["check"] = check
        results.append(res)
        status = "OK" if res["exit"] == 0 else "FAIL"
        rss = f"{res['peak_rss_mb']:.1f} MiB" if res["peak_rss_mb"] is not None else "?"
        print(f"  {status}  runtime={res['runtime_s']:.2f}s  peak_rss={rss}")
        if res["exit"] != 0:
            any_fail = True
            for ln in res["stderr_tail"]:
                print("   stderr:", ln)

    # write markdown
    md = ["# Reference impl smoke results", ""]
    md.append("Peak RSS measured by `resource.getrusage(RUSAGE_SELF).ru_maxrss` in each impl subprocess (KiB on Linux, converted to MiB). Runtime measured by the runner (wall time of the subprocess).")
    md.append("")
    md.append("| Section | Impl | Runtime (s) | Peak RSS (MiB) | Correctness check |")
    md.append("|---|---|---|---|---|")
    for r in results:
        rss = f"{r['peak_rss_mb']:.1f}" if r["peak_rss_mb"] is not None else "n/a"
        status = "" if r["exit"] == 0 else " **FAIL**"
        md.append(f"| {r['section']} | {r['name']} | {r['runtime_s']:.2f} | {rss} | {r['check']}{status} |")
    md.append("")
    md.append(f"Total: {sum(1 for r in results if r['exit']==0)}/{len(results)} passed.")
    out = ROOT / ".pingis" / "impl" / "SMOKE_RESULTS.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {out}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
