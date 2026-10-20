#!/usr/bin/env python3
# v0.9.3b: P-S1 sort merge fan-in S2 fix.
# Bug: compute_fan_in uses read_batch_size (1000 output flush) as if it
# were the reader's on-disk batch size (_WRITE_RUN_CHUNK=50000). 50x
# underestimate of merge peak. Fix: use on-disk chunk for fan-in; add
# check_batch in merge loop as defense.
# Run: uv run python scripts/patch_v093b_sort_merge_fanin.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


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


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import py_compile; py_compile.compile('" + str(path) +
         "', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("COMPILE FAIL: " + str(path))
        print(r.stderr[-800:])
        sys.exit(1)
    print("  compile OK: " + str(path))


# ============ Change 1: _WRITE_RUN_CHUNK constant ============

def add_constant():
    print("\n=== sort.py: _WRITE_RUN_CHUNK constant ===")
    p = Path("cadw/ops/sort.py")
    s = p.read_text()
    if "_WRITE_RUN_CHUNK = " in s:
        print("  already present - skip")
        return
    anchor = 'logger = get_logger("ops.sort")\n'
    if anchor not in s:
        raise SystemExit("anchor not found: logger line")
    insert = (
        '\n# On-disk run batch size (rows per IPC batch written by\n'
        '# _write_run). compute_fan_in MUST use this for the reader side\n'
        '# of the merge, not read_batch_size (output flush frequency).\n'
        '_WRITE_RUN_CHUNK = 50_000\n'
    )
    s = s.replace(anchor, anchor + insert, 1)
    p.write_text(s)
    compile_check(p)


# ============ Change 2: _write_run uses constant ============

def use_constant_in_write_run():
    print("\n=== sort.py: _write_run uses _WRITE_RUN_CHUNK ===")
    p = Path("cadw/ops/sort.py")
    s = p.read_text()
    if "    chunk = _WRITE_RUN_CHUNK\n" in s:
        print("  already uses constant")
        return
    old = "    chunk = 50_000\n"
    if old in s:
        s = s.replace(old, "    chunk = _WRITE_RUN_CHUNK\n", 1)
        p.write_text(s)
        compile_check(p)
        print("  OK")
        return
    # Fallback: find any `chunk = 50_000` (no indent match)
    if "chunk = 50_000" in s:
        s = s.replace("chunk = 50_000", "chunk = _WRITE_RUN_CHUNK", 1)
        p.write_text(s)
        compile_check(p)
        print("  OK (fallback pattern)")
        return
    raise SystemExit("_write_run chunk anchor not found")


# ============ Change 3: compute_fan_in uses _WRITE_RUN_CHUNK ============

def fix_compute_fan_in_call():
    print("\n=== sort.py: compute_fan_in uses _WRITE_RUN_CHUNK ===")
    p = Path("cadw/ops/sort.py")
    s = p.read_text()
    if "_WRITE_RUN_CHUNK, row_bytes" in s or "_WRITE_RUN_CHUNK,\n" in s:
        print("  already fixed")
        return
    # Find compute_fan_in( call, replace read_batch_size with _WRITE_RUN_CHUNK
    # within its args.
    import re
    # Match: compute_fan_in(\n\s*memory_guard.available_bytes(), read_batch_size, row_bytes\n\s*)
    pat = re.compile(
        r"(compute_fan_in\(\s*\n?\s*memory_guard\.available_bytes\(\),)\s*read_batch_size",
        re.MULTILINE,
    )
    if not pat.search(s):
        print("  WARNING: compute_fan_in pattern not found. Context:")
        idx = s.find("compute_fan_in(")
        while idx >= 0:
            print("    " + repr(s[max(0, idx-20):idx+180]))
            idx = s.find("compute_fan_in(", idx + 1)
        raise SystemExit("compute_fan_in anchor not found")
    s = pat.sub(r"\1 _WRITE_RUN_CHUNK", s, count=1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Change 4: _merge_keys memory_guard + check_batch ============

def add_memory_guard_to_merge():
    print("\n=== sort.py: _merge_keys memory_guard + check_batch ===")
    p = Path("cadw/ops/sort.py")
    s = p.read_text()

    # 4a: signature
    if 'memory_guard: "MemoryGuard | None" = None,\n) -> Path:' in s:
        print("  signature already updated")
    else:
        old_sig = "    read_batch_size: int,\n) -> Path:"
        new_sig = ('    read_batch_size: int,\n'
                   '    memory_guard: "MemoryGuard | None" = None,\n'
                   ') -> Path:')
        # Only replace the one in _merge_keys (first occurrence after
        # its `def`). Find _merge_keys def, then old_sig nearby.
        d_idx = s.find("def _merge_keys(")
        if d_idx < 0:
            raise SystemExit("_merge_keys def not found")
        sig_idx = s.find(old_sig, d_idx)
        if sig_idx < 0:
            raise SystemExit("_merge_keys signature anchor not found")
        s = s[:sig_idx] + new_sig + s[sig_idx + len(old_sig):]
        p.write_text(s)
        compile_check(p)
        print("  signature updated")

    # 4b: fetch: check_batch after next()
    if 'memory_guard.check_batch(\n                        nb,' in s:
        print("  fetch already updated")
    else:
        old_fetch = (
            "            try:\n"
            "                cur_batches[idx] = next(readers[idx])\n"
            "                cur_pos[idx] = 0\n"
            "            except StopIteration:\n"
            "                return None\n"
        )
        new_fetch = (
            "            try:\n"
            "                nb = next(readers[idx])\n"
            "                if memory_guard is not None:\n"
            "                    memory_guard.check_batch(\n"
            "                        nb, name=\"sort.merge_read[%d]\" % idx\n"
            "                    )\n"
            "                cur_batches[idx] = nb\n"
            "                cur_pos[idx] = 0\n"
            "            except StopIteration:\n"
            "                return None\n"
        )
        if old_fetch not in s:
            print("  WARNING: fetch block not found. Context:")
            idx = s.find("cur_batches[idx] = next(")
            if idx >= 0:
                print("    " + repr(s[max(0, idx-100):idx+220]))
            raise SystemExit("_merge_keys fetch anchor not found")
        s = s.replace(old_fetch, new_fetch, 1)
        p.write_text(s)
        compile_check(p)
        print("  fetch updated")

    # 4c: call site — pass memory_guard
    if "asc, nulls_first, out, read_batch_size, memory_guard," in s:
        print("  call already updated")
    else:
        old_call = (
            "                group, all_columns, key_columns, value_columns,\n"
            "                asc, nulls_first, out, read_batch_size,\n"
            "            )\n"
        )
        new_call = (
            "                group, all_columns, key_columns, value_columns,\n"
            "                asc, nulls_first, out, read_batch_size, memory_guard,\n"
            "            )\n"
        )
        if old_call not in s:
            print("  WARNING: call anchor not found. Context:")
            idx = s.find("_merge_keys(")
            while idx >= 0:
                print("    " + repr(s[max(0, idx-20):idx+200]))
                idx = s.find("_merge_keys(", idx + 1)
            raise SystemExit("_merge_keys call anchor not found")
        s = s.replace(old_call, new_call, 1)
        p.write_text(s)
        compile_check(p)
        print("  call updated")


# ============ Tests ============

TESTS_LINES = [
    '# v0.9.3b: P-S1 (S2) sort merge fan-in tests.',
    '# Prior bug: compute_fan_in read_batch_size=1000, but on-disk',
    '# batches are _WRITE_RUN_CHUNK=50000. 50x peak underestimate.',
    '',
    'from pathlib import Path',
    '',
    'import numpy as np',
    'import pyarrow as pa',
    'import pytest',
    '',
    'from cadw.core.memory_guard import MemoryGuard',
    'from cadw.ops.sort import external_sort, _WRITE_RUN_CHUNK',
    '',
    '',
    'def batches(n: int, bs: int, ncol: int = 10):',
    '    rng = np.random.default_rng(0)',
    '    for start in range(0, n, bs):',
    '        m = min(bs, n - start)',
    '        cols = {"k": pa.array(rng.integers(0, 1000, m),',
    '                              type=pa.int64())}',
    '        for j in range(ncol - 1):',
    '            cols["v" + str(j)] = pa.array(rng.random(m))',
    '        yield pa.record_batch(cols)',
    '',
    '',
    'def test_write_run_chunk_constant_exists():',
    '    assert _WRITE_RUN_CHUNK > 0',
    '    assert isinstance(_WRITE_RUN_CHUNK, int)',
    '',
    '',
    'def test_compute_fan_in_uses_write_run_chunk():',
    '    # AST: compute_fan_in second positional arg must be',
    '    # _WRITE_RUN_CHUNK (real on-disk chunk), not read_batch_size.',
    '    import ast',
    '    import inspect',
    '    import textwrap',
    '    from cadw.ops import sort as sort_mod',
    '    src = textwrap.dedent(inspect.getsource(sort_mod.external_sort))',
    '    tree = ast.parse(src)',
    '    for node in ast.walk(tree):',
    '        if not isinstance(node, ast.Call):',
    '            continue',
    '        if not (isinstance(node.func, ast.Name)',
    '                and node.func.id == "compute_fan_in"):',
    '            continue',
    '        assert len(node.args) >= 2, (',
    '            "compute_fan_in needs 2+ positional args"',
    '        )',
    '        arg1 = ast.unparse(node.args[1])',
    '        assert "_WRITE_RUN_CHUNK" in arg1, (',
    '            "compute_fan_in read_batch_size=" + arg1 +',
    '            " must be _WRITE_RUN_CHUNK (P-S1 fix)"',
    '        )',
    '        return',
    '    raise AssertionError("compute_fan_in call not found")',
    '',
    '',
    'def test_merge_keys_has_check_batch():',
    '    import inspect',
    '    from cadw.ops import sort as sort_mod',
    '    src = inspect.getsource(sort_mod._merge_keys)',
    '    assert "memory_guard" in src, (',
    '        "_merge_keys must accept memory_guard param"',
    '    )',
    '    assert "check_batch" in src, (',
    '        "_merge_keys must call check_batch on read batches"',
    '    )',
    '',
    '',
    'def test_external_sort_tight_budget_smoke(tmp_path):',
    '    # P-S1: many runs under tight budget. With fix, sort completes',
    '    # without MemoryLimitExceeded/OOM. Prior code silently peaked',
    '    # at ~1.9x budget but did not raise.',
    '    out = external_sort(',
    '        batches(200_000, 20_000, ncol=10),',
    '        ["k"], ["k"] + ["v" + str(j) for j in range(9)],',
    '        MemoryGuard(max_memory_mb=16), tmp_path,',
    '        read_batch_size=1000,',
    '    )',
    '    assert out.exists()',
    '',
    '',
    'def test_external_sort_still_correct(tmp_path):',
    '    rng = np.random.default_rng(1)',
    '    n = 1000',
    '    ks = rng.integers(0, 100, n)',
    '    vs = rng.random(n)',
    '    def iter_b():',
    '        for i in range(0, n, 100):',
    '            yield pa.record_batch({',
    '                "k": pa.array(ks[i:i+100], type=pa.int64()),',
    '                "v": pa.array(vs[i:i+100]),',
    '            })',
    '    out = external_sort(',
    '        iter_b(), ["k"], ["k", "v"],',
    '        MemoryGuard(max_memory_mb=10), tmp_path,',
    '    )',
    '    import pyarrow.ipc as ipc',
    '    r = ipc.open_file(str(out))',
    '    keys = []',
    '    for i in range(r.num_record_batches):',
    '        b = r.get_batch(i)',
    '        keys.extend(b.column("k").to_pylist())',
    '    assert keys == sorted(keys)',
    '',
]


def write_tests():
    print("\n=== tests/test_sort_merge_p_s1.py ===")
    p = Path("tests/test_sort_merge_p_s1.py")
    p.write_text("\n".join(TESTS_LINES) + "\n")
    compile_check(p)
    print("  OK")


# ============ CHANGELOG ============

CHANGELOG_LINES = [
    "### Исправлено (v0.9.3b - P-S1 sort merge fan-in, S2 fix)",
    "",
    "**P-S1 (S2, из v0.9.2 audit):** `external_sort` вызывал",
    "`compute_fan_in(read_batch_size=1000)` — это output flush",
    "frequency, не reader batch size. Реальные on-disk batches",
    "(от `_write_run`) — `_WRITE_RUN_CHUNK = 50_000` rows. Reader",
    "держит 50_000 × r на batch, не 1000 × r. 50x underestimate",
    "merge peak. Measured: 126.81 MB при 67.1 MB budget (1.9x).",
    "**MemoryLimitExceeded не поднимался.**",
    "",
    "**Fix:**",
    "",
    "- `_WRITE_RUN_CHUNK = 50_000` — module constant.",
    "- `_write_run` uses it instead of literal `chunk = 50_000`.",
    "- `compute_fan_in` in `external_sort` uses `_WRITE_RUN_CHUNK`",
    "  (on-disk batch size) instead of `read_batch_size`.",
    "- `_merge_keys` gets `memory_guard` param; `check_batch` on each",
    "  read batch (defense-in-depth).",
    "",
    "**Покрывает:** `sort_values`, `rank`, `quantile`, `resample`,",
    "`roc_auc`, `TimeSeriesSplit` — все через `external_sort`.",
    "",
    "**Trade-off:** k уменьшается ~50x (больше merge levels при tight",
    "budget). Peak теперь fits; merge slower для малого budget.",
    "Safety > speed.",
    "",
    "**Тесты (v0.9.3b):** +5 fast (`test_sort_merge_p_s1.py`):",
    "constant exists, AST compute_fan_in uses _WRITE_RUN_CHUNK, AST",
    "check_batch in _merge_keys, tight-budget smoke, regression",
    "sort correctness.",
    "",
    "- fast: **~1565** (+5). slow: **20** (unchanged).",
    "",
]


def append_changelog():
    print("\n=== CHANGELOG v0.9.3b ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.3b - P-S1 sort merge fan-in" in s:
        print("  already present - skip")
        return
    from cadw.patch_utils import append_unreleased_entry
    entry = "\n".join(CHANGELOG_LINES)
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.3b: P-S1 sort merge fan-in ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    add_constant()
    use_constant_in_write_run()
    fix_compute_fan_in_call()
    add_memory_guard_to_merge()
    write_tests()
    append_changelog()

    print("\n=== pytest: sort merge p_s1 ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_sort_merge_p_s1.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: sort/rank/quantile/resample regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_sort.py",
         "tests/test_rank.py",
         "tests/test_quantile.py",
         "tests/test_resample.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** SORT-FAMILY REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: metrics + cv regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_metrics.py",
         "tests/test_cv.py",
         "tests/test_cv_keyed_prng.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** METRICS/CV REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

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

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.3b: P-S1 sort merge fan-in S2 fix — _WRITE_RUN_CHUNK=50000 "
        "constant; compute_fan_in uses on-disk chunk (not read_batch_size); "
        "_merge_keys accepts memory_guard + check_batch defense; covers "
        "sort_values/rank/quantile/resample/roc_auc/TimeSeriesSplit"])

    print("\n==> v0.9.3b applied.")
    print("NEXT: v0.9.3c (P-D2 dedup string keys S2)")


if __name__ == "__main__":
    main()
