#!/usr/bin/env python3
# v0.9.4b-2-fix2: whitelist describe() in AST guard.
#
# describe() calls self.quantile() (method delegation) — quantile is
# internally guarded via external_sort + check_batch. AST guard matches
# by attribute name (quantile), can't distinguish self.quantile from
# pc.quantile. Whitelist CadwDataFrame.describe by qualname (K1.1
# mechanism) with rationale.
#
# Run: uv run python scripts/patch_v094b2_fix2.py

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
    if "CadwDataFrame.describe" in s:
        print("  already whitelisted - skip")
        return

    anchor = (
        '    # v0.9.0c Jupyter: _repr_html_ calls .to_pandas().to_html() on\n'
        '    # a 20-row sample only — bounded by design, no O(N) materialize.\n'
        '    "_repr_html_": "20-row head().to_pandas().to_html(), bounded sample",\n'
    )
    if anchor not in s:
        raise SystemExit("_repr_html_ whitelist anchor not found")

    L = []
    L.append(anchor)
    L.append('    # v0.9.4b-2: describe() calls self.quantile() (method')
    L.append('    # delegation) — quantile is internally guarded via')
    L.append('    # external_sort + check_batch. AST guard matches by attribute')
    L.append('    # name (quantile), can\'t distinguish self.quantile from')
    L.append('    # pc.quantile. Whitelisted with rationale (§13.4/170).')
    L.append('    "describe": "delegates to self.quantile (guarded via external_sort)",')
    L.append('')
    block = "\n".join(L)
    s = s.replace(anchor, block, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def main():
    print("=== v0.9.4b-2-fix2 ===")
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

    print("\n=== pytest: v094b2 + dt/str regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe_v094b2.py",
         "tests/test_dt_accessor.py",
         "tests/test_str_accessor.py",
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
        "v0.9.4b-2-fix1: _DtAccessor._feature(name) static; new "
        "_feature_custom(name, fn, out_type) for floor/ceil/round/"
        "week_of_year; v0.9.4b-2-fix2: whitelist describe() in AST "
        "guard (self.quantile delegation; internal guards)",
    )
    print("\n==> v0.9.4b-2-fix2 applied.")


if __name__ == "__main__":
    main()
