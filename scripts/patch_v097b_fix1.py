#!/usr/bin/env python3
# v0.9.7b-fix1 — dogfood np not imported (patch script).
#
# Bug: dogfood() in patch_v097b_describe.py called np.sqrt(2.5) but
# `import numpy as np` was not in scope (no top-level import; dogfood
# imports only `import pyarrow as pa` locally). NameError at runtime.
#
# State: all writes done (describe.py, dataframe.py facade, whitelist,
# test_describe_arrow_native.py) BEFORE dogfood. Only dogfood + pytest
# + commit remain. fix1 does the corrected dogfood inline + full run.
#
# §13.4/200 (same class as §13.4/192 — helper's scope missing import;
# for patch scripts: dogfood body needs own imports, cannot rely on
# module-level names that aren't imported). Diagnostic: dogfood is
# throwaway verification, but it runs inside the patch script's
# __main__ scope — needs same import discipline as production code.
#
# Run: uv run python scripts/patch_v097b_fix1.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    assert_no_call_in_source, commit_with_guards,
)


def dogfood_corrected():
    print("\n=== dogfood (fix1): fused describe + AST guards ===")
    import numpy as np
    import pyarrow as pa

    from cadw.dataframe import CadwDataFrame
    from cadw.ops import describe as desc_mod
    from cadw.core.memory_guard import MemoryGuard

    # Facade: no quantile / to_pandas.
    import inspect
    src = inspect.getsource(CadwDataFrame.describe)
    assert_no_call_in_source(src, "quantile", "to_pandas",
                              fn_name="CadwDataFrame.describe")
    print("  facade describe: no quantile / to_pandas")

    # Impl has Welford + reservoir.
    isrc = inspect.getsource(desc_mod.DescribeOp)
    assert "M2" in isrc
    assert "reservoir" in isrc
    print("  DescribeOp: Welford (M2) + reservoir present")

    # Quick correctness in-process.
    batches = [
        pa.record_batch({"x": [1.0, 2.0, 3.0]}),
        pa.record_batch({"x": [4.0, 5.0]}),
    ]
    res = desc_mod.describe_columns(
        iter(batches), ["x"], MemoryGuard(max_memory_mb=10),
    )
    assert res["x"]["count"] == 5
    assert abs(res["x"]["mean"] - 3.0) < 1e-12
    assert abs(res["x"]["std"] - np.sqrt(2.5)) < 1e-12
    print(f"  in-process describe: count={res['x']['count']}, "
          f"mean={res['x']['mean']:.3f}, std={res['x']['std']:.3f}")


def main():
    print("=== v0.9.7b-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    dogfood_corrected()

    print("\n=== pytest: describe + regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_describe_arrow_native.py",
         "tests/test_dataframe_v094b2.py",
         "tests/test_dataframe_v094b.py",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_arrow_call_memory_guard.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** DESCRIBE REGRESSION FAILED ***")
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
        "v0.9.7b-fix1: dogfood np not imported (patch script). "
        "All writes already on disk from v0.9.7b before dogfood "
        "failed — describe.py, dataframe.py facade, whitelist "
        "cleanup, test_describe_arrow_native.py. Only remaining "
        "steps (corrected dogfood + pytest + commit) here. "
        "§13.4/200 (same class as /192 — scope missing import; "
        "for dogfood: needs own imports, cannot rely on module-level "
        "names not imported). v0.9.7b complete.",
        self_path="scripts/patch_v097b_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.7b complete (fix1).")


if __name__ == "__main__":
    main()
