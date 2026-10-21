#!/usr/bin/env python3
# v0.9.9-fix1b — one anchor typo in verify_state. v0.9.9 COMPLETE.
#
# fix1a applied F1 dedent + F9 docstring correctly (log: "applied"),
# then died at verify_state: anchor `textwrap.dedent(_inspect.getsource`
# — but actual code uses alias `_tw.dedent(...)` (import textwrap as _tw
# in the same function body). Substring `textwrap.dedent(` never appears
# because it's `_tw.dedent(`. Third occurrence this session (§13.4/161):
# anchor for verification of `new` must be an exact fragment of `new`,
# not the "conceptual" name.
#
# F1 + F9 already on disk (verified by fix1a log "applied"). fix1b only
# re-runs verify_state with corrected anchor, then dogfood + pytest +
# commit.
#
# Run: uv run python scripts/patch_v099_fix1b.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ verify (corrected anchor) ============

def verify_state():
    print("\n=== verify: all F1-F9 on disk ===")
    checks = [
        ("cadw/patch_utils.py", "def run_verify_in_subprocess("),
        # F1: anchor is what we ACTUALLY wrote (`_tw.dedent(...)`,
        # alias — see fix_dedent's `import textwrap as _tw`).
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


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: verify F1-F9 via fresh subprocess ===")
    importlib.reload(pu)

    def _verify_all():
        import inspect
        from cadw.pipeline import ColumnTransformer, _merge_fanout
        from cadw.accessors import _dt as dt_mod
        from cadw.ops import groupby as gb_mod
        from cadw.ml import pca as pca_mod
        from cadw.dataframe import _GroupBy

        # F1 works: we ARE in its subprocess on a nested verify_fn
        print("  F1: helper reachable (nested verify_fn, dedent OK)")

        assert "__serialization_name__" in inspect.getsource(
            ColumnTransformer.to_dict
        )
        print("  F2 (N1): OK")

        assert "is_timestamp" in inspect.getsource(
            dt_mod._validate_timestamp_only
        )
        for m in ("floor", "ceil", "round"):
            assert "_validate_timestamp_only" in inspect.getsource(
                getattr(dt_mod._DtAccessor, m)
            )
        print("  F3 (N2): OK")

        assert "check_state" in inspect.getsource(gb_mod._combine_batches)
        assert "check_state" in inspect.getsource(gb_mod._merge_partials)
        print("  F4 (N3): OK")

        assert "duplicate new column" in inspect.getsource(_merge_fanout)
        print("  F5 (N5): OK")

        assert "if TYPE_CHECKING:" in open("cadw/ml/pca.py").read()
        print("  F6 (B2): OK")

        ps = open("cadw/pipeline.py").read()
        assert (
            "\nfrom pathlib import Path " in ps
            or ps.startswith("from pathlib import Path ")
        )
        print("  F7 (B3): OK")

        assert "v0.9.9 (B4)" in inspect.getsource(pca_mod.PCA)
        print("  F8 (B4): OK")

        assert "v0.9.6b2: CADW-native" in inspect.getsource(_GroupBy.apply)
        print("  F9 (docstring): OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

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
         "tests/test_pca.py", "-v"],
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


# ============ main ============

def main():
    print("=== v0.9.9-fix1b ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.9-fix1b: one anchor typo in v0.9.9-fix1a verify_state. "
        "F1 (dedent) + F9 (docstring) already applied by fix1a "
        "(log: 'applied'). fix1a died at verify_state: anchor "
        "`textwrap.dedent(_inspect.getsource` — but actual code uses "
        "alias `_tw.dedent(...)` (`import textwrap as _tw` in the "
        "same function body), substring `textwrap.dedent(` never "
        "appears. §13.4/161, third occurrence this session: anchor "
        "for verification of `new` must be an exact fragment of "
        "`new`, not the conceptual name. F1 + F9 on disk. fix1b: "
        "corrected anchor (`_tw.dedent(_inspect.getsource`); dogfood "
        "via subprocess (F1 helper); F1-F9 verify all 10 anchors; "
        "pytest fast+slow. v0.9.9 COMPLETE.",
        self_path="scripts/patch_v099_fix1b.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.9-fix1b applied. v0.9.9 COMPLETE.")


if __name__ == "__main__":
    main()
