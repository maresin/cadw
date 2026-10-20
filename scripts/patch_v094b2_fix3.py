#!/usr/bin/env python3
# v0.9.4b-2-fix3: whitelist describe() — anchor by grep, not literal.
#
# fix2 anchor assumed pre-K1.1 whitelist format ("_repr_html_": ...).
# K1.1 (v0.9.2-audit merge) qualname-keyed: WHITELISTED_FUNCS keys are
# "{path}:{qualname}" — e.g., "cadw/dataframe.py:_repr_html_".
# fix3 finds the _repr_html_ line by substring, inserts "describe"
# after it (line-based, K1.1-agnostic).
#
# Run: uv run python scripts/patch_v094b2_fix3.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


def whitelist_describe():
    print("\n=== test_arrow_call_memory_guard.py: whitelist describe ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if '"describe"' in s and "whitelist" in s.lower() or (
        "describe" in s and "quantile" in s and "whitelist" in s.lower()
    ):
        # loose check: "describe" appears near "quantile" in a comment
        pass
    if "whitelisted describe" in s.lower() or (
        '"describe"' in s and "self.quantile" in s
    ):
        print("  describe already whitelisted - skip")
        return

    lines = s.split("\n")
    # Find a line containing _repr_html_ in a whitelist-key context.
    # K1.1 format: '    "cadw/dataframe.py:_repr_html_": "20-row ...",'
    # Pre-K1.1:    '    "_repr_html_": "20-row ...",'
    idx = None
    for i, line in enumerate(lines):
        if "_repr_html_" in line and line.strip().startswith('"'):
            idx = i
            break
    if idx is None:
        print("  _repr_html_ whitelist line not found. Def lines:")
        for i, line in enumerate(lines):
            if "_repr_html_" in line:
                print("    " + str(i+1) + ": " + repr(line))
        raise SystemExit("_repr_html_ anchor line not found")

    print("  anchor line " + str(idx+1) + ": " + repr(lines[idx]))

    # Determine qualname key format (K1.1: path:name; else bare name).
    # We insert a new entry in the SAME format as the anchor.
    # K1.1 file may have "cadw/dataframe.py:..." keys.
    has_path_prefix = "cadw/" in lines[idx] or ":" in lines[idx].split('"')[1] \
        if len(lines[idx].split('"')) > 1 else False

    if has_path_prefix:
        new_key = '"cadw/dataframe.py:describe": '
    else:
        new_key = '"describe": '
    new_line = (
        "    # v0.9.4b-2: describe() delegates to self.quantile "
        "(guarded via external_sort + check_batch)."
    )
    new_entry = (
        "    " + new_key
        + '"delegates to self.quantile (guarded via external_sort)",'
    )
    lines.insert(idx+1, new_line)
    lines.insert(idx+2, new_entry)

    p.write_text("\n".join(lines))
    assert_compiles(p)
    print("  OK (inserted after _repr_html_)")


def main():
    print("=== v0.9.4b-2-fix3 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    whitelist_describe()

    print("\n=== pytest: AST guard ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py", "-q"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** AST GUARD STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: v094b2 + dt/str + patch_utils regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe_v094b2.py",
         "tests/test_dt_accessor.py",
         "tests/test_str_accessor.py",
         "tests/test_patch_utils.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
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
        "v0.9.4b-2-fix1: _DtAccessor._feature_custom (4 tests); "
        "v0.9.4b-2-fix2/fix3: whitelist describe() in AST guard "
        "(self.quantile delegation; internal guards via external_sort). "
        "K1.1-aware: uses {path}:{qualname} key format",
    )
    print("\n==> v0.9.4b-2-fix3 applied. v0.9.4b-2 complete.")


if __name__ == "__main__":
    main()
