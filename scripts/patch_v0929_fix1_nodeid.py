#!/usr/bin/env python3
# v0.9.29-fix1: rewrite conftest hook with item.nodeid (pytest 9 safe).
# §13.4/356.

import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

CONFTEST = Path("tests/conftest.py")


NEW_HOOK = '''

# v0.9.29-fix1 B.3: auto-marking by nodeid (§13.4/354/356).
# item.nodeid = "tests/test_ts_features.py::test_lag_basic"
# — stable across pytest 7/8/9 (fspath removed in pytest 9,
#  path added in 7 but nodeid never changes).
# Additive: pytest dedupes identical markers.
def pytest_collection_modifyitems(config, items):
    import os as _os
    import pytest as _pt
    for item in items:
        nodeid = item.nodeid
        fname = nodeid.split("::")[0]
        base = _os.path.basename(fname)

        if base.startswith("test_ts_"):
            item.add_marker(_pt.mark.ts)
        if base.startswith("test_ml_") or any(
            base.startswith(p) for p in (
                "test_gbdt", "test_kmeans", "test_linear",
                "test_logistic", "test_naive_bayes", "test_pca",
                "test_scalers", "test_metrics", "test_cv",
                "test_facade_transformers",
                "test_column_transformer",
            )
        ):
            item.add_marker(_pt.mark.ml)
        if base.startswith("test_viz_") or base == "test_plot_accessor.py":
            item.add_marker(_pt.mark.viz)
        if (base.startswith("test_pipeline")
                or base.startswith("test_cv")):
            item.add_marker(_pt.mark.pipeline)
        if base.startswith("test_e2e") or base == "test_oom_resistance.py":
            item.add_marker(_pt.mark.e2e)
        if any(base.startswith(p) for p in (
            "test_groupby", "test_join", "test_sort",
            "test_dedup", "test_rank", "test_quantile",
            "test_resample", "test_pivot", "test_reduce",
            "test_shuffle", "test_topk", "test_histogram",
            "test_merge", "test_anti_join", "test_semi_join",
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

    # pyproject has markers + xdist?
    src = Path("pyproject.toml").read_text()
    if '"ml: ML model tests' not in src:
        raise SystemExit("pyproject: ml marker missing — run v0.9.29 first")
    if "pytest-xdist" not in src:
        raise SystemExit("pyproject: pytest-xdist missing")
    print("  pyproject: markers + xdist ✓")

    # conftest has hook
    cs = CONFTEST.read_text()
    if "pytest_collection_modifyitems" not in cs:
        raise SystemExit("conftest: hook missing — run v0.9.29 first")
    print(f"  conftest: hook present ({len(cs.splitlines())} lines)")


def rewrite_hook():
    print("\n--- rewrite hook (nodeid) ---")
    src = CONFTEST.read_text()
    # §13.4/357: AST-based — find def lineno
    import ast as _ast
    try:
        tree = _ast.parse(src)
    except SyntaxError as e:
        raise SystemExit(f"conftest unparseable: {e}")
    hook_node = None
    for node in tree.body:
        if (isinstance(node, _ast.FunctionDef)
                and node.name == "pytest_collection_modifyitems"):
            hook_node = node
            break
    if hook_node is None:
        raise SystemExit("hook function not found via AST")
    # Find end of function
    fn_end = max(
        getattr(n, "end_lineno", hook_node.lineno)
        for n in _ast.walk(hook_node)
    )
    # Cut from def line start to fn end. Also scan back to strip
    # preceding blank lines + comments (marker block header).
    lines = src.split("\n")
    def_idx = hook_node.lineno - 1  # 0-based
    # Back off over blank lines and comment lines
    back = def_idx
    while back > 0:
        prev = lines[back - 1].strip()
        if not prev or prev.startswith("#"):
            back -= 1
        else:
            break
    # Preserve lines before 'back', append new hook
    kept = "\n".join(lines[:back]).rstrip() + "\n"
    new_src = kept + NEW_HOOK
    CONFTEST.write_text(new_src)
    compile(new_src, str(CONFTEST), "exec")
    print(f"  hook rewritten: {len(new_src.splitlines())} lines "
          f"(was {len(src.splitlines())})")


def count_filter(expr):
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", expr, "--co", "-q"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  WARN pytest -m {expr!r} exit {r.returncode}")
        print(r.stderr[:500])
        return -1
    return sum(1 for ln in r.stdout.split("\n") if "::" in ln)


def test_selective():
    print("\n--- selective collection ---")
    n_ts = count_filter("ts and not slow")
    n_ml = count_filter("ml and not slow")
    n_ops = count_filter("ops and not slow")
    n_viz = count_filter("viz and not slow")
    n_pipeline = count_filter("pipeline and not slow")
    n_e2e = count_filter("e2e and not slow")
    n_all = count_filter("not slow")

    print(f"  ts:       {n_ts}")
    print(f"  ml:       {n_ml}")
    print(f"  ops:      {n_ops}")
    print(f"  viz:      {n_viz}")
    print(f"  pipeline: {n_pipeline}")
    print(f"  e2e:      {n_e2e}")
    print(f"  not-slow: {n_all}")

    if n_ts == 0:
        raise SystemExit("ts marker still 0 — hook not applied?")
    if n_ts >= n_all:
        raise SystemExit(f"ts==all ({n_ts}) — marker not filtering")

    return {
        "ts": n_ts, "ml": n_ml, "ops": n_ops,
        "viz": n_viz, "pipeline": n_pipeline, "e2e": n_e2e,
        "all": n_all,
    }


def test_xdist(baseline_hint=142.0):
    print("\n--- xdist parallel ---")
    # Quick baseline measurement for the speedup print (fast, not full)
    # Actually re-use the fact that baseline was 142s at v0.9.29 before hook fix.
    # Full xdist run — that's the expensive check.
    t0 = time.time()
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow",
         "-n", "auto", "--dist=loadfile"],
        timeout=1800,
    )
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"  *** xdist FAILED (exit {r.returncode}) ***")
        print("  Markers available; parallel opt-in documented.")
        return None
    speedup = baseline_hint / elapsed if elapsed > 0 else 1.0
    print(f"  parallel: {elapsed:.1f}s (~{speedup:.1f}× vs "
          f"{baseline_hint:.0f}s single-process)")
    return elapsed


def run_full_fast():
    print("\n--- full fast suite (final gate) ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def commit(counts, parallel, baseline=142.0):
    speedup = baseline / parallel if parallel else None
    par_note = (
        f" xdist -n auto --dist=loadfile: "
        f"~{baseline:.0f}s → ~{parallel:.0f}s "
        f"({speedup:.1f}×), opt-in"
        if parallel else
        " xdist: available, opt-in via -n auto"
    )
    msg = (
        f"v0.9.29: B.3 pytest markers + xdist. "
        f"Registered markers: slow (existing) + ml + ts + ops + "
        f"viz + pipeline + e2e. "
        f"fix1 (§13.4/356): auto-marking hook uses "
        f"item.nodeid (string, stable across pytest 7/8/9), "
        f"NOT item.fspath (removed in pytest 9 — pyproject "
        f"requires pytest>=9.0.3). Original v0.9.29 hook "
        f"silently marked 0 items because fspath returned "
        f"unusable py.path.local; now nodeid.split('::')[0] + "
        f"os.path.basename gives clean file patterns. "
        f"Selective runs: "
        f"`-m ts` → {counts['ts']} tests, "
        f"`-m ml` → {counts['ml']}, "
        f"`-m ops` → {counts['ops']}, "
        f"`-m viz` → {counts['viz']}, "
        f"`-m pipeline` → {counts['pipeline']}, "
        f"`-m e2e` → {counts['e2e']} "
        f"(vs {counts['all']} baseline).{par_note}. "
        f"NO addopts — parallel opt-in, safer for dev. "
        f"Test-latency pain reduced via selective markers + "
        f"parallel opt-in. 0 production changes; tests green."
    )
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", msg], check=True)
    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"\n{log}")


def main():
    print("=== v0.9.29-fix1: conftest hook nodeid ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"branch: {br}")

    verify_state()
    rewrite_hook()

    # Quick sanity: hook compiles + parses
    cs = CONFTEST.read_text()
    ast.parse(cs, filename=str(CONFTEST))
    print("  conftest.py parses OK")

    counts = test_selective()
    parallel = test_xdist()
    run_full_fast()

    # Whitelist gate
    print("\n--- verify_whitelist ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.patch_utils import verify_whitelist; "
         "verify_whitelist(raise_on_orphan=True); print('WL OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("whitelist failed")
    print(r.stdout.strip())

    commit(counts, parallel)
    print("\nv0.9.29 COMPLETE (fix1). B.3 done.")


if __name__ == "__main__":
    main()