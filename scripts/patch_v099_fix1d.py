#!/usr/bin/env python3
# v0.9.9-fix1d — test order bug in test_merge_fanout_duplicate_new_column_raises.
# v0.9.9 COMPLETE.
#
# fix1c green everywhere except one test: pipe.fit(df) is outside
# pytest.raises, but _merge_fanout is called in Pipeline.fit (line 603),
# not transform. F5 raises correctly, test catches nothing → fail with
# unexpected RuntimeError.
#
# Fix: move pipe.fit(df) INSIDE pytest.raises. transform() never
# reached (fit raises first), which is fine — F5 contract is
# verified by the raise from fit.
#
# Run: uv run python scripts/patch_v099_fix1d.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


OLD_TEST = '''    pipe.add_step('merge', __import__('cadw.pipeline', fromlist=['DropnaStep']).DropnaStep(),
                  inputs=['base', 'b', 'c'])
    pipe.fit(df)
    with pytest.raises(RuntimeError, match='duplicate new column'):
        pipe.transform(df)'''

NEW_TEST = '''    pipe.add_step('merge', __import__('cadw.pipeline', fromlist=['DropnaStep']).DropnaStep(),
                  inputs=['base', 'b', 'c'])
    # F5 raises in Pipeline.fit (line 603: _merge_fanout for fan-in),
    # not transform. Wrap fit in raises.
    with pytest.raises(RuntimeError, match='duplicate new column'):
        pipe.fit(df)'''


def fix_test():
    print("\n=== fix pipe.fit inside pytest.raises ===")
    p = Path("tests/test_pipeline_dag.py")
    s = p.read_text()
    if NEW_TEST in s:
        print("  already fixed — skip")
        return
    if OLD_TEST not in s:
        raise SystemExit("test anchor not found")
    s = s.replace(OLD_TEST, NEW_TEST, 1)
    p.write_text(s)
    pu.assert_compiles(p)
    print("  OK")


def verify_state():
    print("\n=== verify: F1-F9 on disk ===")
    checks = [
        ("cadw/patch_utils.py", "def run_verify_in_subprocess("),
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
        ("cadw/pipeline.py", "v0.9.9 (N1): use __serialization_name__"),
        ("cadw/accessors/_dt.py", "def _validate_timestamp_only("),
        ("cadw/ops/groupby.py", "v0.9.9 (N3)"),
        ("cadw/pipeline.py", "v0.9.9 (N5)"),
        ("cadw/ml/pca.py", "if TYPE_CHECKING:  # v0.9.9 (B2)"),
        ("cadw/pipeline.py", "# v0.9.9 (B3): for `-> Path` annotation"),
        ("cadw/ml/pca.py", "v0.9.9 (B4)"),
        ("cadw/dataframe.py", "Per-group apply (v0.9.6b2: CADW-native"),
    ]
    for path, anchor in checks:
        if anchor not in Path(path).read_text():
            raise SystemExit(f"{path}: missing {anchor!r}")
    print(f"  all {len(checks)} anchors present")


def dogfood():
    print("\n=== dogfood: F1-F9 via fresh subprocess ===")
    importlib.reload(pu)

    def _verify_all():
        import inspect
        from cadw.pipeline import ColumnTransformer, _merge_fanout
        from cadw.accessors import _dt as dt_mod
        from cadw.ops import groupby as gb_mod
        from cadw.ml import pca as pca_mod
        from cadw.dataframe import _GroupBy

        assert "__serialization_name__" in inspect.getsource(
            ColumnTransformer.to_dict
        )
        for m in ("floor", "ceil", "round"):
            assert "_validate_timestamp_only" in inspect.getsource(
                getattr(dt_mod._DtAccessor, m)
            )
        assert "check_state" in inspect.getsource(gb_mod._combine_batches)
        assert "check_state" in inspect.getsource(gb_mod._merge_partials)
        assert "duplicate new column" in inspect.getsource(_merge_fanout)
        assert "if TYPE_CHECKING:" in open("cadw/ml/pca.py").read()
        assert "\nfrom pathlib import Path " in open("cadw/pipeline.py").read()
        assert "v0.9.9 (B4)" in inspect.getsource(pca_mod.PCA)
        assert "v0.9.6b2: CADW-native" in inspect.getsource(_GroupBy.apply)
        print("  F1-F9 all verified")

    pu.run_verify_in_subprocess(_verify_all)


def run_pytest():
    print("\n=== pytest: hygiene + regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_column_transformer.py",
         "tests/test_dt_accessor.py",
         "tests/test_combiner.py",
         "tests/test_pipeline_dag.py",
         "tests/test_pipeline_save_load.py",
         "tests/test_pca.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** HYGIENE REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=900,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


def main():
    print("=== v0.9.9-fix1d ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    verify_state()
    fix_test()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.9-fix1d: complete v0.9.9 (last test order bug). "
        "test_merge_fanout_duplicate_new_column_raises: pipe.fit(df) "
        "was outside pytest.raises, but _merge_fanout is called in "
        "Pipeline.fit (line 603: fan-in merge), not transform. F5 "
        "raises correctly at fit → uncaught → fail. Fix: move "
        "pipe.fit(df) INSIDE pytest.raises (one-line move). F5 "
        "contract verified by raise-from-fit. v0.9.9 (F1-F9) "
        "COMPLETE: 10/10 anchors, dogfood subprocess green, "
        "fast+slow green.",
        self_path="scripts/patch_v099_fix1d.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.9-fix1d applied. v0.9.9 COMPLETE.")


if __name__ == "__main__":
    main()
