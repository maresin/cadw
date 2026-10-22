#!/usr/bin/env python3
# v0.9.25c-fix1 — 3 test/tooling bugs in v0.9.25c (production OK).
#
# Bug 1: test_runstream_refill_no_concat_in_loop used inspect.cleandoc
#   on method source. cleandoc strips first line then min-indent from
#   lines 2+ → for method source it produces `def ...:\n"""docstring"""`
#   at col 0 → IndentationError. Fix: textwrap.dedent (matches
#   patch_utils.parse_method_source and v0.9.25c dogfood, which worked).
#
# Bug 2: collect_project.py additions use subprocess.run() but the
#   original file has no module-level `import subprocess` (main() didn't
#   need it). Fix: add `import subprocess` at top of _additions block.
#
# Bug 3: test_collect_project_no_args_errors asserted returncode != 0.
#   But no new-mode flags → dispatch falls to legacy main() → smart
#   snapshot (README-documented default). Test was wrong about the
#   backward-compat contract. Fix: rename to _uses_legacy_snapshot,
#   assert returncode == 0 + output exists (with --output override).
#
# Dogfood noise: MemoryGuard.check_state logs ERROR before raising; in
#   dogfood this prints "check_state FAIL: ..." before VERIFY OK.
#   Suppress only that logger during dogfood verification.
#
# v0.9.25c production changes applied but NOT committed (targeted
# pytest failed). This patch re-runs targeted pytest + full suite,
# then commits v0.9.25c + fix1.
#
# Run: uv run python scripts/patch_v0925c_fix1_test_bugs.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape, run_verify_in_subprocess,
)


# ============ 0. Verify v0.9.25c changes on disk ============

def verify_state():
    print("\n=== verify v0.9.25c changes on disk ===")
    checks = [
        ("cadw/ops/sort.py", "parts: list = [self.buffer]"),
        ("cadw/ops/sort.py", "read_batch_size, row_bytes"),
        ("cadw/ops/sort.py", "max_k: int = 256"),
        ("cadw/dataframe.py", "len(uniques) * 180"),
        ("scripts/collect_project.py", "_phase_aware_main"),
        ("tests/test_v0925c_phase_a.py", "inspect.cleandoc"),
        ("tests/test_collect_project_modes.py",
         "test_collect_project_no_args_errors"),
    ]
    for path, anchor in checks:
        if anchor not in Path(path).read_text():
            raise SystemExit(
                f"{path}: missing {anchor!r}. Re-run v0.9.25c."
            )
    print("  all anchors present — OK")


# ============ 1. Fix test: cleandoc → textwrap.dedent ============

OLD_TEST_A4 = '''def test_runstream_refill_no_concat_in_loop():
    """AST: no pa.concat_tables inside while loop in refill (B1)."""
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod._RunStream.refill)
    tree = ast.parse(inspect.cleandoc(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "concat_tables"):
                    raise AssertionError(
                        f"concat_tables inside while loop at "
                        f"line {sub.lineno} — quadratic concat (B1)"
                    )
'''

NEW_TEST_A4 = '''def test_runstream_refill_no_concat_in_loop():
    """AST: no pa.concat_tables inside while loop in refill (B1)."""
    import textwrap
    from cadw.ops import sort as sort_mod
    src = textwrap.dedent(inspect.getsource(sort_mod._RunStream.refill))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "concat_tables"):
                    raise AssertionError(
                        f"concat_tables inside while loop at "
                        f"line {sub.lineno} — quadratic concat (B1)"
                    )
'''

def fix_test_cleandoc():
    print("\n=== fix: test_runstream_refill cleandoc -> textwrap.dedent ===")
    applied = apply_or_fail(
        "tests/test_v0925c_phase_a.py", OLD_TEST_A4, NEW_TEST_A4,
        label="test_a4_dedent",
    )
    if applied:
        assert_compiles("tests/test_v0925c_phase_a.py")
        check_no_invalid_escape("tests/test_v0925c_phase_a.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 2. Add import subprocess to collect_project.py ============

OLD_IMPORTS = '''import json as _json
import re as _re
import sys as _sys

_DOCS_ROADMAP = Path("docs/roadmap.json")
'''

NEW_IMPORTS = '''import json as _json
import re as _re
import subprocess
import sys as _sys

_DOCS_ROADMAP = Path("docs/roadmap.json")
'''

def fix_subprocess_import():
    print("\n=== fix: add import subprocess to collect_project.py ===")
    applied = apply_or_fail(
        "scripts/collect_project.py", OLD_IMPORTS, NEW_IMPORTS,
        label="collect_subprocess",
    )
    if applied:
        assert_compiles("scripts/collect_project.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 3. Fix no-args test ============

OLD_NOARGS_TEST = '''def test_collect_project_no_args_errors():
    r = _run()
    assert r.returncode != 0
    assert "--module" in r.stderr or "--mode" in r.stderr
'''

NEW_NOARGS_TEST = '''def test_collect_project_no_args_uses_legacy_snapshot(tmp_path):
    """v0.9.25c-fix1: no new-mode flags → legacy --mode=smart.

    Backward compat: running `collect_project.py` without --module /
    --phase / --doc / --list-phases falls through to the original
    `main()` (README-documented default = smart snapshot to
    merged_cadw_project.py). Prior test asserted an error — wrong
    about the contract. Verify by overriding --output to a tmp file
    and checking success + non-empty output.
    """
    out = tmp_path / "full_snapshot.md"
    r = _run("--output", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial snapshot
'''

def fix_noargs_test():
    print("\n=== fix: test_collect_project_no_args -> legacy snapshot ===")
    applied = apply_or_fail(
        "tests/test_collect_project_modes.py",
        OLD_NOARGS_TEST, NEW_NOARGS_TEST,
        label="test_noargs",
    )
    if applied:
        assert_compiles("tests/test_collect_project_modes.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 4. Dogfood with log noise suppression ============

def dogfood():
    print("\n=== dogfood: 4 items closed (log noise suppressed) ===")

    def _verify():
        import logging as _logging
        # Suppress MemoryGuard check_state FAIL log during expected-raise.
        # v0.9.25c-fix2: correct logger name is "cadw.memory_guard"
        # (get_logger("memory_guard") = getLogger("cadw.memory_guard")).
        # Prior v0.9.25c-fix1 used "cadw.core.memory_guard" — no-op.
        _logging.getLogger("cadw.memory_guard").setLevel(_logging.CRITICAL)

        import ast as _ast
        import inspect as _inspect
        import textwrap as _tw
        import subprocess as _sp
        import sys as _sys
        from cadw.ops import sort as _sort

        # A.4 refill AST — no concat_tables in while
        src = _tw.dedent(_inspect.getsource(_sort._RunStream.refill))
        tree = _ast.parse(src)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.While):
                for sub in _ast.walk(node):
                    if (isinstance(sub, _ast.Call)
                            and isinstance(sub.func, _ast.Attribute)
                            and sub.func.attr == "concat_tables"):
                        raise AssertionError(
                            f"A.4 not closed: concat_tables inside while"
                        )

        # A.5 encode_cat guard raises on high-card
        import pyarrow as _pa
        import cadw as _cadw
        from cadw import CadwDataFrame as _CDF
        n = 50_000
        t = _pa.table({
            "c": _pa.array([f"k{i:08d}" for i in range(n)],
                           type=_pa.string()),
            "v": list(range(n)),
        })
        df = _CDF.from_table(
            t, memory_guard=_cadw.MemoryGuard(max_memory_mb=2),
        )
        try:
            df.encode_categorical("c")
            raise AssertionError(
                "A.5 not closed: no MemoryLimitExceeded on 50k unique"
            )
        except _cadw.MemoryLimitExceeded:
            pass

        # A.7 fan_in uses read_batch_size
        src = _tw.dedent(_inspect.getsource(_sort.external_sort))
        tree = _ast.parse(src)
        found = False
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            if not (isinstance(node.func, _ast.Name)
                    and node.func.id == "compute_fan_in"):
                continue
            arg1 = _ast.unparse(node.args[1])
            assert "read_batch_size" in arg1, (
                f"A.7 not closed: compute_fan_in arg={arg1!r}"
            )
            found = True
            break
        assert found, "A.7: compute_fan_in not called"

        from cadw.ops.sort import compute_fan_in
        assert compute_fan_in(32 * 1024 * 1024, 1000, 640) == 8
        assert compute_fan_in(32 * 1024 * 1024, 1000, 80) == 69

        # B.4 collect_project --list-phases
        r = _sp.run(
            [_sys.executable, "scripts/collect_project.py",
             "--list-phases"],
            cwd=".", capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        for pid in "ABCDEFGH":
            assert f" {pid} " in r.stdout, (
                f"B.4: phase {pid!r} missing: {r.stdout}"
            )

        # B.4-fix1 smoke: --module also works
        import tempfile
        from pathlib import Path as _P
        with tempfile.TemporaryDirectory() as td:
            out = _P(td) / "mod.md"
            r2 = _sp.run(
                [_sys.executable, "scripts/collect_project.py",
                 "--module", "cadw/ts/models.py", "--out", str(out)],
                cwd=".", capture_output=True, text=True,
            )
            assert r2.returncode == 0, r2.stderr
            assert out.exists() and out.stat().st_size > 1000
            assert "class SARIMAForecaster" in out.read_text()

        print(
            "  4 items + B.4 --module smoke: A.4 refill linear, "
            "A.5 encode_cat guard, A.7 fan_in rbs, B.4 phase+module collect"
        )

    run_verify_in_subprocess(_verify)


# ============ 5. Pytest ============

def run_pytest():
    print("\n=== pre-flight: test_facade_has_no_as_py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py", "-q"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** FACADE AS_PY VIOLATION — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: targeted (v0925c + collect) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925c_phase_a.py",
         "tests/test_collect_project_modes.py",
         "tests/test_sort_merge_p_s1.py",
         "tests/test_merge_arrow.py",
         "tests/test_encode_categorical.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 6. main ============

def main():
    print("=== v0.9.25c-fix1: test/tooling bugs (production OK) ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    fix_test_cleandoc()
    fix_subprocess_import()
    fix_noargs_test()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25c: Phase A batch 2 — A.4 + A.5 + A.7 + B.4 "
        "(4 items from docs/roadmap.json). "
        "A.4 (part1_report B1): _RunStream.refill linear concat. "
        "Prior pa.concat_tables([buffer, chunk]) inside while loop "
        "→ quadratic; 12.9 GB copied @ rbs=200k. Now: list collect + "
        "single concat_tables. "
        "A.5 (part1_report B2): encode_categorical in-loop guard. "
        "Prior uniques: set accumulated Python objects unbounded; "
        "@ 10M unique strings ~600 MB invisible to MemoryGuard. "
        "Now check_state(len(uniques) * 180) per batch. "
        "A.7 (audit §6.6): compute_fan_in uses read_batch_size, "
        "NOT _WRITE_RUN_CHUNK. Root cause of 24.4s @ "
        "N=500k/8cols/32MB — v0.9.23c _merge_keys uses rbs=1000 "
        "internally, but fan_in got _WRITE_RUN_CHUNK=50000 → K=2 → "
        "3-4x more passes. Correct fix: pass rbs. Now K=8 @ "
        "32MB/8cols/640B/row. Formula: 3*K*rbs*row <= state_limit "
        "(matches _merge_keys check_state). max_k=256 FD cap. "
        "Audit's 'floor=8' was a proxy; formula gives correct K. "
        "B.4 (docs/roadmap.json): collect_project.py --module / "
        "--phase / --doc / --list-phases. Reads docs/roadmap.json. "
        "Brace-expansion for composite file specs. --module is main "
        "workflow unlock: 27 files → 5 files/session. "
        "fix1 (3 test/tooling bugs, production OK): "
        "(1) test_runstream_refill_no_concat_in_loop used "
        "inspect.cleandoc on method source → IndentationError; fix: "
        "textwrap.dedent (matches patch_utils.parse_method_source and "
        "v0.9.25c dogfood which worked). "
        "(2) collect_project.py additions use subprocess.run but the "
        "original file has no module-level `import subprocess` "
        "(main() didn't need it); added. "
        "(3) test_collect_project_no_args_errors asserted error on "
        "no args — but no new-mode flags falls to legacy main() "
        "(README-documented smart snapshot). Renamed to "
        "_uses_legacy_snapshot, asserts returncode == 0 + output. "
        "Test additions: 10 new (test_v0925c_phase_a, "
        "test_collect_project_modes) + 1 flipped "
        "(test_compute_fan_in_uses_write_run_chunk → "
        "_uses_read_batch_size — stale-test-on-refactor). "
        "Dogfood: 4 items + --module smoke verified. "
        "Fast suite: ~2275 → ~2285.",
        self_path="scripts/patch_v0925c_fix1_test_bugs.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25c+fix1 applied. Next: v0.9.25d (A.6: 18 "
          "to_pylist sites extra_factor).")


if __name__ == "__main__":
    main()