#!/usr/bin/env python3
# v0.9.6d-fix2 — verify_state active-usage (not substring); join.py
# cleanup via block replacement.
#
# (1) verify_state false-negative on historical comment:
#       tests/test_no_pandas_in_ops.py: cleanup expected but reference remains
#     v0.9.6d cleanup REPLACED the stale rationale with a historical
#     note that intentionally mentions `_pandas_hash_column`:
#         "v0.9.6a replaced `_pandas_hash_column` with own XXH64..."
#     Useful for future readers. But verify used
#     `if "_pandas_hash_column" in text: fail` — substring, catches
#     historical mentions. SAME class as §13.4/183 (substring vs AST),
#     now at verify-function level. Fix: regex for ACTIVE usage only
#     (import line or `_pandas_hash_column(...)` call).
#
# (2) join.py cleanup still stale (v0.9.6d: "anchor not found"). My
#     pattern was `# (v0.8.11a: ...`, actual file `# v0.8.11a: ...`
#     (no paren). Fix: block-based replacement — backtrack over
#     consecutive `#` lines, replace whole block.
#
# State: helpers + tests on disk (v0.9.6d). 2/3 cleanups done. This
# fix completes join.py, dogfoods helpers with reload (§13.4/178/186),
# commits everything as v0.9.6d.
#
# Run: uv run python scripts/patch_v096d_fix2.py

from __future__ import annotations
import os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


# ============ 1. active-usage check (not substring) ============

_ACTIVE_USAGE_PATTERNS = [
    re.compile(r"import[^\n]*_pandas_hash_column"),
    re.compile(r"_pandas_hash_column\s*\("),
]


def _has_active_usage(path: Path) -> bool:
    s = path.read_text()
    return any(p.search(s) for p in _ACTIVE_USAGE_PATTERNS)


def verify_state():
    print("\n=== verify: helpers + tests + cleanups (active usage) ===")
    pu = Path("cadw/patch_utils.py").read_text()
    for name in (
        "assert_no_call_in_source", "assert_call_order",
        "is_patched", "run_smoke_inline",
    ):
        if f"def {name}(" not in pu:
            raise SystemExit(f"helper {name} missing from patch_utils.py")
    print("  4 helpers present")

    ts = Path("tests/test_patch_utils.py").read_text()
    if "def test_assert_no_call_in_source_clean" not in ts:
        raise SystemExit("test_assert_no_call_in_source_clean missing")
    print("  helper tests present")

    for path_str in (
        "tests/test_no_pandas_in_ops.py",
        "tests/test_hash_v081.py",
    ):
        p = Path(path_str)
        if _has_active_usage(p):
            s = p.read_text()
            for pat in _ACTIVE_USAGE_PATTERNS:
                for m in pat.finditer(s):
                    print(f"  {path_str}: ACTIVE usage: {m.group(0)!r}")
            raise SystemExit(f"{path_str}: active _pandas_hash_column usage")
        print(f"  {path_str}: no active usage (historical mentions OK)")


# ============ 2. fix join.py via block replacement ============

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


# ============ 3. dogfood with reload (§13.4/178, /186) ============

def dogfood_helpers():
    print("\n=== dogfood: new helpers on real _per_group_apply ===")
    # §13.4/178/186: reload to defeat stale module cache. Top-level
    # `from cadw.patch_utils import assert_compiles, commit_with_guards`
    # bound OLD module object; add_helpers() rewrote file on disk.
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


# ============ main ============

def main():
    print("=== v0.9.6d-fix2 ===")
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
        "v0.9.6d-fix2: verify_state active-usage (not substring). "
        "v0.9.6d cleanup replaced stale rationale in "
        "test_no_pandas_in_ops.py with a historical note that "
        "intentionally mentions `_pandas_hash_column` "
        "(\"v0.9.6a replaced ...\"). fix1's verify checked "
        "`\"_pandas_hash_column\" not in text` — substring, same "
        "class as §13.4/183, now at verify-function level. Fix: "
        "regex for ACTIVE usage only (import line or "
        "`_pandas_hash_column(...)` call); historical comments OK. "
        "Also: join.py cleanup via block replacement (v0.9.6d "
        "anchor `# (v0.8.11a` — actual file `# v0.8.11a`, no "
        "paren). dogfood reloads cadw.patch_utils before importing "
        "new helpers (§13.4/178, /186 — the bug class that this "
        "scaffold exists to prevent, hit twice in the scaffold's "
        "own patch script). v0.9.6d complete.",
        self_path="scripts/patch_v096d_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6d complete (fix1+fix2).")


if __name__ == "__main__":
    main()