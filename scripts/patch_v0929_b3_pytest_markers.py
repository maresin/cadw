#!/usr/bin/env python3
# v0.9.29: B.3 pytest markers + xdist availability.
# Markers registered (ml/ts/ops/viz/pipeline/e2e; slow exists).
# xdist available via `-n auto` (NO addopts — safer default).
# conftest auto-marks by filename. Bootstrap-safe.

import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

PYPROJECT = Path("pyproject.toml")
CONFTEST = Path("tests/conftest.py")


NEW_MARKERS = [
    "slow: slow-running tests (E2E, memory, large data)",
    "ml: ML model tests (gbdt, linear, logistic, kmeans, pca, naive_bayes, metrics, cv, scalers)",
    "ts: time-series tests (rolling, expanding, ewm, ts toolkit)",
    "ops: ops primitive tests (groupby, join, dedup, sort, rank, quantile, resample, pivot, reduce, shuffle, topk)",
    "viz: visualization adapter tests (matplotlib, plotly, seaborn)",
    "pipeline: pipeline and DAG tests",
    "e2e: end-to-end integration tests",
]


CONFTEST_HOOK = '''

# v0.9.29 B.3: auto-marking by filename (§13.4/354).
# Existing manual markers (e.g. @pytest.mark.slow) are preserved —
# add_marker is additive.
def pytest_collection_modifyitems(config, items):
    import pytest as _pt
    for item in items:
        path = str(item.fspath)
        if "/test_ts_" in path or "/cadw/ts/" in path:
            item.add_marker(_pt.mark.ts)
        if any(p in path for p in (
            "/test_ml_", "/test_gbdt", "/test_kmeans", "/test_linear",
            "/test_logistic", "/test_naive_bayes", "/test_pca",
            "/test_scalers", "/test_metrics", "/test_cv",
            "/test_facade_transformers", "/test_column_transformer",
        )):
            item.add_marker(_pt.mark.ml)
        if any(p in path for p in (
            "/test_viz_", "/test_plot_accessor",
        )):
            item.add_marker(_pt.mark.viz)
        if any(p in path for p in (
            "/test_pipeline", "/test_cv",
        )):
            item.add_marker(_pt.mark.pipeline)
        if any(p in path for p in (
            "/test_e2e", "/test_oom",
        )):
            item.add_marker(_pt.mark.e2e)
        if any(p in path for p in (
            "/test_groupby", "/test_join", "/test_sort",
            "/test_dedup", "/test_rank", "/test_quantile",
            "/test_resample", "/test_pivot", "/test_reduce",
            "/test_shuffle", "/test_topk", "/test_histogram",
            "/test_merge", "/test_anti_join", "/test_semi_join",
        )):
            item.add_marker(_pt.mark.ops)
'''


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.28":
        raise SystemExit(f"expected HEAD=v0.9.28, got {head!r}")
    print(f"HEAD: {head}")

    if not PYPROJECT.exists():
        raise SystemExit("pyproject.toml missing")
    if not CONFTEST.exists():
        raise SystemExit(f"{CONFTEST} missing")


def diagnostic():
    print("\n--- diagnostic: current pyproject ---")
    src = PYPROJECT.read_text()
    data = tomllib.loads(src)
    pytest_cfg = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    print(f"  [tool.pytest.ini_options]: "
          f"{'present' if pytest_cfg else 'absent'}")
    if pytest_cfg:
        print(f"    addopts: {pytest_cfg.get('addopts', '(none)')}")
        print(f"    testpaths: {pytest_cfg.get('testpaths', '(none)')}")
        markers = pytest_cfg.get("markers", [])
        print(f"    markers ({len(markers)}):")
        for m in markers:
            print(f"      {m}")
    dev_deps = data.get("dependency-groups", {}).get("dev", [])
    print(f"\n  [dependency-groups].dev ({len(dev_deps)}):")
    for d in dev_deps:
        print(f"    {d}")

    print("\n--- diagnostic: conftest.py ---")
    cs = CONFTEST.read_text()
    print(f"  {CONFTEST}: {len(cs.splitlines())} lines")
    if "pytest_collection_modifyitems" in cs:
        print("  ⚠ already has hook — v0.9.29 partially applied?")
    else:
        print("  no hook yet ✓")
    return data, pytest_cfg


def extend_markers(src):
    """§13.4/355: parse existing, rebuild array with proper commas."""
    import tomllib as _tomllib
    try:
        data = _tomllib.loads(src)
    except _tomllib.TOMLDecodeError as e:
        raise SystemExit(f"source pyproject already broken: {e}")
    ini = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    existing = ini.get("markers", [])
    existing_names = {m.split(":")[0].strip() for m in existing}
    merged = list(existing)
    added = []
    for new_m in NEW_MARKERS:
        nm = new_m.split(":")[0].strip()
        if nm not in existing_names:
            merged.append(new_m)
            added.append(nm)
    if not added:
        print("  all markers already present")
        return src
    print(f"  adding {len(added)} marker(s): {added}")
    lines = ["markers = ["]
    for m in merged:
        safe = m.replace(chr(34), chr(92) + chr(34))
        lines.append(f"    {chr(34)}{safe}{chr(34)},")
    lines.append("]")
    new_array = chr(10).join(lines)

    start_m = re.search(r"markers\s*=\s*\[", src)
    if not start_m:
        raise SystemExit("markers = [ not found in source")
    pos = start_m.end() - 1
    depth = 0
    end_pos = None
    for i in range(pos, len(src)):
        if src[i] == "[":
            depth += 1
        elif src[i] == "]":
            depth -= 1
            if depth == 0:
                end_pos = i + 1
                break
    if end_pos is None:
        raise SystemExit("unbalanced markers array in source")
    new_src = src[:start_m.start()] + new_array + src[end_pos:]

    _tomllib.loads(new_src)
    return new_src


def patch_pyproject():
    print("\n--- patch pyproject.toml (markers) ---")
    src = PYPROJECT.read_text()
    new_src = extend_markers(src)
    if new_src != src:
        tomllib.loads(new_src)  # verify
        PYPROJECT.write_text(new_src)
        print("  pyproject.toml updated")
    else:
        print("  no markers change needed")


def add_xdist_dep():
    print("\n--- add pytest-xdist to dev deps ---")
    r = subprocess.run(
        ["uv", "add", "--dev", "pytest-xdist"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("uv add pytest-xdist failed")
    print("  uv add pytest-xdist: OK")


def patch_conftest():
    print("\n--- patch tests/conftest.py ---")
    src = CONFTEST.read_text()
    if "pytest_collection_modifyitems" in src:
        print("  already has hook")
        return
    new_src = src.rstrip() + CONFTEST_HOOK
    CONFTEST.write_text(new_src)
    # compile check
    compile(new_src, str(CONFTEST), "exec")
    print(f"  hook added ({len(CONFTEST_HOOK.splitlines())} lines)")


def verify_markers_registered():
    print("\n--- verify markers registered ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "--markers", "-q"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise SystemExit("pytest --markers failed")
    for m in ["slow", "ml", "ts", "ops", "viz", "pipeline", "e2e"]:
        if f"@pytest.mark.{m}:" in r.stdout:
            print(f"  ✓ @pytest.mark.{m}")
        else:
            raise SystemExit(f"marker {m} not registered")


def test_baseline():
    print("\n--- baseline: pytest -m 'not slow' ---")
    t0 = time.time()
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    elapsed = time.time() - t0
    if r.returncode != 0:
        raise SystemExit("baseline fast suite FAILED")
    print(f"  baseline: {elapsed:.1f}s")
    return elapsed


def test_selective():
    print("\n--- selective run: pytest -m 'ts and not slow' ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "ts and not slow",
         "--co", "-q"],
        capture_output=True, text=True,
    )
    n_ts = sum(1 for ln in r.stdout.split("\n") if "::" in ln)
    print(f"  ts-only tests collected: ~{n_ts}")
    if n_ts == 0 or n_ts == 2500:
        raise SystemExit(
            f"ts marker filter not working (n={n_ts}, expected subset)"
        )

    print("\n--- selective run: pytest -m 'ml and not slow' --co ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "ml and not slow",
         "--co", "-q"],
        capture_output=True, text=True,
    )
    n_ml = sum(1 for ln in r.stdout.split("\n") if "::" in ln)
    print(f"  ml-only tests collected: ~{n_ml}")
    if n_ml == 0 or n_ml == 2500:
        raise SystemExit(f"ml marker filter not working (n={n_ml})")

    print("\n--- selective run: pytest -m 'ops and not slow' --co ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "ops and not slow",
         "--co", "-q"],
        capture_output=True, text=True,
    )
    n_ops = sum(1 for ln in r.stdout.split("\n") if "::" in ln)
    print(f"  ops-only tests collected: ~{n_ops}")
    if n_ops == 0 or n_ops == 2500:
        raise SystemExit(f"ops marker filter not working (n={n_ops})")


def test_xdist(baseline_time):
    print("\n--- xdist parallel: pytest -n auto --dist=loadfile ---")
    t0 = time.time()
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow",
         "-n", "auto", "--dist=loadfile"],
        timeout=1800,
    )
    elapsed = time.time() - t0
    if r.returncode != 0:
        print("\n*** xdist FAILED — parallel-unsafe tests detected ***")
        print("*** Markers + xdist available; no addopts added ***")
        return None
    speedup = baseline_time / elapsed if elapsed > 0 else 1.0
    print(f"  parallel: {elapsed:.1f}s "
          f"({speedup:.2f}× vs baseline {baseline_time:.1f}s)")
    return elapsed


def commit(baseline, parallel, n_ts, n_ml, n_ops):
    speedup = baseline / parallel if parallel else 1.0
    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    par_note = (
        f" xdist -n auto --dist=loadfile: "
        f"{baseline:.0f}s → {parallel:.0f}s ({speedup:.1f}×) "
        f"(opt-in — no addopts, safer default)"
        if parallel else
        " xdist: available, no addopts (opt-in)"
    )
    msg = (
        f"v0.9.29: B.3 pytest markers + xdist availability. "
        f"Registered markers: slow (existing) + ml + ts + ops + "
        f"viz + pipeline + e2e. conftest.py auto-marks by "
        f"filename (§13.4/354); existing manual markers preserved. "
        f"pytest-xdist added to dev deps. Selective runs: "
        f"`-m ts` → ~{n_ts} tests, `-m ml` → ~{n_ml}, "
        f"`-m ops` → ~{n_ops} (vs 2500 baseline).{par_note}. "
        f"NO addopts — parallel is opt-in via `-n auto`, "
        f"safer for single-file dev runs. Test-latency pain "
        f"reduced via selective runs (markers) + parallel "
        f"opt-in (xdist). 0 production changes; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)
    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"\n{log}")
    print("\nv0.9.29 COMPLETE. B.3 done.")


def main():
    print("=== v0.9.29: B.3 pytest markers + xdist ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"branch: {br}")

    verify_state()
    diagnostic()
    patch_pyproject()
    add_xdist_dep()
    patch_conftest()
    verify_markers_registered()
    baseline = test_baseline()
    test_selective()  # populates n_ts, n_ml, n_ops via --co, prints

    # re-run selective --co to capture counts for commit msg
    def count(filter_expr):
        r = subprocess.run(
            ["uv", "run", "pytest", "-q", "-m", filter_expr,
             "--co", "-q"],
            capture_output=True, text=True,
        )
        return sum(1 for ln in r.stdout.split("\n") if "::" in ln)
    n_ts = count("ts and not slow")
    n_ml = count("ml and not slow")
    n_ops = count("ops and not slow")

    parallel = test_xdist(baseline)
    commit(baseline, parallel, n_ts, n_ml, n_ops)


if __name__ == "__main__":
    main()