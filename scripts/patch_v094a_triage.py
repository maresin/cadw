#!/usr/bin/env python3
# v0.9.4a-triage: 2 targeted test reductions + iteration helper.
#
# Top-2 slow single tests (per v0.9.3 baseline):
#   - test_sort_merge_p_s1::test_external_sort_tight_budget_smoke (20.5s)
#   - test_pipeline_e2e::test_e2e_pipeline_time_series (~3s)
#
# Reductions:
#   - sort_merge smoke: N=200_000 budget=16MB -> N=100_000 budget=8MB.
#     Bug (P-S1) triggers at both sizes: k_from_read_batch_size (1000)
#     vs on-disk (50000) -> k too large -> peak 1.9x budget. AST test
#     already covers the invariant; smoke test only "does it finish".
#   - pipeline e2e: N=20_000 GBDT(30) -> N=10_000 GBDT(20). R2 threshold
#     0.3 unchanged (wide margin at both sizes).
#
# New helper: patch_utils.run_pytest_iterate(). Wraps `pytest -x --lf`
# (fail-fast, last-failed-only). For iterative dev loops inside future
# patches. Full suite (no -x --lf) stays for commit gate.
#
# Run: uv run python scripts/patch_v094a_triage.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def compile_check(path):
    from cadw.patch_utils import assert_compiles
    assert_compiles(path)


def sh(cmd, check=True):
    print("  $ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print("    " + line)
    if check and r.returncode != 0:
        print("    STDERR: " + r.stderr.strip()[-1500:])
        sys.exit(r.returncode)
    return r.returncode


def add_iterate_helper():
    print("\n=== patch_utils.py: run_pytest_iterate ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def run_pytest_iterate(" in s:
        print("  already present - skip")
        return
    anchor = "\n\ndef grep_and_warn("
    if anchor not in s:
        raise SystemExit("anchor not found")
    L = []
    L.append("")
    L.append("")
    L.append("def run_pytest_iterate(paths=None, *, timeout=900):")
    L.append("    # pytest -x --lf -q --no-header on `paths` (or full).")
    L.append("    # Iteration helper: stop at first failure, only last-failed")
    L.append("    # (after first run, .pytest_cache stores fails). Full suite is")
    L.append("    # the commit gate; this is for fast feedback loops.")
    L.append("    import subprocess")
    L.append("    import sys")
    L.append("    cmd = ['uv', 'run', 'pytest', '-x', '--lf', '-q',")
    L.append("           '--no-header', '-m', 'not slow']")
    L.append("    if paths:")
    L.append("        cmd.extend(paths)")
    L.append("    print('  $ ' + ' '.join(cmd))")
    L.append("    r = subprocess.run(cmd, timeout=timeout)")
    L.append("    if r.returncode != 0:")
    L.append("        sys.exit(r.returncode)")
    L.append("")
    L.append("")
    block = "\n".join(L)
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


def slim_sort_smoke():
    print("\n=== test_sort_merge_p_s1.py: N=200k->100k, budget=8MB ===")
    p = Path("tests/test_sort_merge_p_s1.py")
    s = p.read_text()
    old = (
        "def test_external_sort_tight_budget_smoke(tmp_path):\n"
        "    # P-S1: many runs under tight budget. With fix, sort completes\n"
        "    # without MemoryLimitExceeded/OOM. Prior code silently peaked\n"
        "    # at ~1.9x budget but did not raise.\n"
        "    out = external_sort(\n"
        "        batches(200_000, 20_000, ncol=10),\n"
        "        [\"k\"], [\"k\"] + [\"v\" + str(j) for j in range(9)],\n"
        "        MemoryGuard(max_memory_mb=16), tmp_path,\n"
        "        read_batch_size=1000,\n"
        "    )\n"
        "    assert out.exists()"
    )
    new = (
        "def test_external_sort_tight_budget_smoke(tmp_path):\n"
        "    # v0.9.4a-triage: N=200k->100k, budget=16MB->8MB (v0.9.3\n"
        "    # test time 20.5s; now ~10s). P-S1 regression still triggers:\n"
        "    # k computed from read_batch_size=1000 vs on-disk 50000 ->\n"
        "    # 2x too high; peak = k * 50000 * row_bytes. AST test covers\n"
        "    # invariant; this smoke just verifies end-to-end completion.\n"
        "    out = external_sort(\n"
        "        batches(100_000, 20_000, ncol=10),\n"
        "        [\"k\"], [\"k\"] + [\"v\" + str(j) for j in range(9)],\n"
        "        MemoryGuard(max_memory_mb=8), tmp_path,\n"
        "        read_batch_size=1000,\n"
        "    )\n"
        "    assert out.exists()"
    )
    if new in s and old not in s:
        print("  already slim - skip")
        return
    if old not in s:
        raise SystemExit("sort smoke anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


def slim_pipeline_e2e():
    print("\n=== test_pipeline_e2e.py: N=20k->10k, GBDT 30->20 ===")
    p = Path("tests/test_pipeline_e2e.py")
    s = p.read_text()
    # Find and adjust the _r2 test header + n_estimators. Two edits.
    old_header = (
        "@pytest.mark.slow\n"
        "def test_e2e_pipeline_time_series(tmp_path):\n"
        "    \"\"\"Realistic: timestamp + categorical + target; dt features → GBDT.\n"
        "\n"
        "    N=20K. Feature set: year/month/day/weekday (from ts). Model: GBDT\n"
        "    regression on those + sensor_id.\n"
        "    Target: y = 10 + 5 * month / 12 + 3 * weekday / 7 + noise.\n"
        "    R² > 0.3 expected.\n"
        "    \"\"\"\n"
        "    rng = np.random.default_rng(42)\n"
        "    n = 20_000"
    )
    new_header = (
        "@pytest.mark.slow\n"
        "def test_e2e_pipeline_time_series(tmp_path):\n"
        "    # v0.9.4a-triage: N=20k->10k, GBDT(30)->GBDT(20). v0.9.3\n"
        "    # test time ~3s; now ~1.5s. R2>0.3 threshold unchanged\n"
        "    # (wide margin at both sizes on linear-signal target).\n"
        "    rng = np.random.default_rng(42)\n"
        "    n = 10_000"
    )
    if new_header in s and old_header not in s:
        print("  already slim (header) - skip header edit")
    elif old_header in s:
        s = s.replace(old_header, new_header, 1)
    else:
        print("  WARNING: pipeline e2e header anchor not found")
        print("    (may have different docstring; abort to inspect)")
        raise SystemExit("pipeline_e2e header not found")

    # n_estimators 30 -> 20 in this test
    old_gbdt = "            feature_cols, \"y\",\n            n_estimators=30, max_depth=3, learning_rate=0.1, bins=32,\n"
    new_gbdt = "            feature_cols, \"y\",\n            n_estimators=20, max_depth=3, learning_rate=0.1, bins=32,\n"
    if new_gbdt in s:
        print("  already slim (gbdt) - skip")
    elif old_gbdt in s:
        s = s.replace(old_gbdt, new_gbdt, 1)
    else:
        print("  WARNING: GBDT anchor not found; showing context")
        idx = s.find("n_estimators=")
        while idx >= 0:
            print("    " + repr(s[max(0, idx-60):idx+100]))
            idx = s.find("n_estimators=", idx + 1)
        raise SystemExit("pipeline_e2e GBDT anchor not found")

    p.write_text(s)
    compile_check(p)
    print("  OK")


def append_iterate_test():
    print("\n=== test_patch_utils.py: run_pytest_iterate ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_run_pytest_iterate_callable" in s:
        print("  already present - skip")
        return
    L = []
    L.append("")
    L.append("")
    L.append("def test_run_pytest_iterate_callable():")
    L.append("    # Only shape-check: callable and doesn't raise on import.")
    L.append("    from cadw.patch_utils import run_pytest_iterate")
    L.append("    assert callable(run_pytest_iterate)")
    L.append("    import inspect")
    L.append("    sig = inspect.signature(run_pytest_iterate)")
    L.append("    assert 'paths' in sig.parameters")
    L.append("")
    p.write_text(s + "\n".join(L))
    compile_check(p)


def main():
    print("=== v0.9.4a-triage ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    add_iterate_helper()
    slim_sort_smoke()
    slim_pipeline_e2e()
    append_iterate_test()

    from cadw.patch_utils import commit_with_guards

    print("\n=== timing: sort_merge smoke + pipeline e2e ===")
    import time
    t0 = time.time()
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_sort_merge_p_s1.py::test_external_sort_tight_budget_smoke",
         "tests/test_pipeline_e2e.py",
         "-q"],
        timeout=300,
    )
    dt = time.time() - t0
    print(f"  elapsed: {dt:.1f}s (was ~24s; expected ~12-14s)")
    if r.returncode != 0:
        print("\n*** TIMING TARGET FAILED ***")
        sys.exit(1)

    print("\n=== pytest: patch_utils ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_patch_utils.py", "-q"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** PATCH_UTILS FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    commit_with_guards(
        "v0.9.4a-triage: slim 2 slowest tests (sort_merge smoke "
        "N=200k->100k budget=16->8MB; pipeline_e2e N=20k->10k GBDT=30->20); "
        "patch_utils.run_pytest_iterate (-x --lf iteration helper)",
    )
    print("\n==> v0.9.4a-triage applied.")


if __name__ == "__main__":
    main()
