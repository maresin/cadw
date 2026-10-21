#!/usr/bin/env python3
# v0.9.6d-fix1 — two meta-bugs in v0.9.6d.
#
# (1) dogfood ImportError:
#       cannot import name 'assert_no_call_in_source' from 'cadw.patch_utils'
#     Root cause: §13.4/178 (stale module cache — the very class of bug
#     that v0.9.6d adds helpers against). Top-level
#     `from cadw.patch_utils import assert_compiles, commit_with_guards`
#     bound OLD module before add_helpers() rewrote the file on disk.
#     assert_compiles passed (disk correct), but in-memory
#     pu.__dict__ stale. Fresh pytest would pass too (it reloads).
#     Fix: importlib.reload(pu) before importing new helpers.
#
# (2) join.py cleanup anchor not found: my CLEANUPS pattern had
#     `# (v0.8.11a: ...` — actual file has `# v0.8.11a: ...` (no
#     paren). Substring-vs-block anchor issue. Fix: block-based
#     replacement — backtrack over consecutive `#` lines, replace
#     whole block.
#
# State: helpers, tests, and 2/3 cleanups already on disk (v0.9.6d
# aborted before commit). This fix completes join.py cleanup, runs
# dogfood with reload, commits everything as v0.9.6d.
#
# Run: uv run python scripts/patch_v096d_fix1.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


def verify_state():
    print("\n=== verify: helpers + tests + partial cleanup on disk ===")
    pu_path = Path("cadw/patch_utils.py")
    s = pu_path.read_text()
    for name in (
        "assert_no_call_in_source", "assert_call_order",
        "is_patched", "run_smoke_inline",
    ):
        if f"def {name}(" not in s:
            raise SystemExit(f"helper {name} missing from patch_utils.py")
    print("  4 helpers present")

    t = Path("tests/test_patch_utils.py")
    ts = t.read_text()
    if "def test_assert_no_call_in_source_clean" not in ts:
        raise SystemExit("test_assert_no_call_in_source_clean missing")
    print("  helper tests present")

    for path_str in (
        "tests/test_no_pandas_in_ops.py",
        "tests/test_hash_v081.py",
    ):
        if "_pandas_hash_column" in Path(path_str).read_text():
            raise SystemExit(
                f"{path_str}: cleanup expected but reference remains"
            )
        print(f"  {path_str}: cleanup OK")


def fix_join_py():
    print("\n=== fix join.py stale comment block ===")
    p = Path("cadw/ops/join.py")
    s = p.read_text()
    if "_pandas_hash_column" not in s:
        print("  already clean — skip")
        return
    lines = s.split("\n")
    for i, line in enumerate(lines):
        if "_pandas_hash_column" not in line:
            continue
        # Backtrack / forward over consecutive `#` comment lines.
        start = i
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        end = i
        while (end < len(lines) - 1
               and lines[end + 1].lstrip().startswith("#")):
            end += 1
        print(f"  block lines {start+1}..{end+1}:")
        for j in range(start, end + 1):
            print(f"    {lines[j]}")
        new_block = [
            "# v0.8.11a: no pandas in hot path (dead helpers removed).",
            "# v0.9.6a: shuffle.py uses own XXH64-based hash; cadw/ops/",
            "# is pandas-free.",
        ]
        lines = lines[:start] + new_block + lines[end + 1:]
        p.write_text("\n".join(lines))
        assert_compiles(p)
        print("  fixed")
        return
    print("  WARNING: _pandas_hash_column present but not in comment block")


def dogfood_helpers():
    print("\n=== dogfood: new helpers on real _per_group_apply ===")
    # v0.9.6d-fix1 (§13.4/178): reload to defeat stale module cache.
    # Top-level `from cadw.patch_utils import ...` bound OLD module;
    # add_helpers() rewrote file on disk. Reload in-memory __dict__.
    import importlib
    import cadw.patch_utils as pu
    importlib.reload(pu)
    from cadw.patch_utils import (
        assert_no_call_in_source, assert_call_order, is_patched,
    )
    import inspect
    from cadw.dataframe import CadwDataFrame

    fn = CadwDataFrame._per_group_apply
    src = inspect.getsource(fn)

    assert_no_call_in_source(src, "to_pandas", fn_name="_per_group_apply")
    print("  assert_no_call_in_source: no to_pandas in _per_group_apply")

    assert_call_order(src, "check_state", "from_table",
                      fn_name="_per_group_apply")
    print("  assert_call_order: check_state < from_table")

    assert is_patched(fn, "CadwDataFrame.from_table")
    print("  is_patched: from_table marker present")

    # Negative control
    try:
        assert_no_call_in_source(
            "def f():\n    x.to_pandas()\n", "to_pandas",
            fn_name="negative-control",
        )
    except AssertionError:
        print("  negative control: correctly rejected")
    else:
        raise SystemExit("negative control failed")
    print("  OK")


def main():
    print("=== v0.9.6d-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    verify_state()
    fix_join_py()
    dogfood_helpers()

    print("\n=== pytest: patch_utils + guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** PATCH_UTILS/GUARDS FAILED ***")
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
        "v0.9.6d-fix1: two meta-bugs in v0.9.6d. "
        "(1) dogfood ImportError 'assert_no_call_in_source' — "
        "§13.4/178 stale module cache: top-level `from cadw.patch_utils` "
        "bound OLD module; add_helpers() rewrote file on disk "
        "(assert_compiles passed, fresh pytest would pass, in-process "
        "stale). Fix: importlib.reload(pu) before importing new "
        "helpers. Note the irony: this is the exact bug class that "
        "v0.9.6d adds helpers against (§13.4/178..185) — the scaffold "
        "must itself use reload. Helpers/tests/2/3 cleanups already "
        "on disk from v0.9.6d. "
        "(2) join.py cleanup anchor 'not found': pattern had "
        "`# (v0.8.11a`, file has `# v0.8.11a` (no paren). Fix: "
        "block-based replacement (backtrack over consecutive `#` "
        "lines). Both fixes complete v0.9.6d.",
        self_path="scripts/patch_v096d_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6d complete (fix1).")


if __name__ == "__main__":
    main()