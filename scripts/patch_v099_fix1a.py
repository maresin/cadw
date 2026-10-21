#!/usr/bin/env python3
# v0.9.9-fix1a — complete v0.9.9 (replaces failed fix1).
#
# fix1 died at verify_state: anchor 'N1: use __serialization_name__'
# not in file (actual: 'v0.9.9 (N1): use __serialization_name__' —
# leading 'v0.9.9 (' prefix). Substring 'N1: use' does not match
# '(N1): use' because ')' precedes ':'. §13.4/161 (substring anchor
# without context).
#
# State on disk (git diff verified):
#   F1 helper: present  (run_verify_in_subprocess)
#   F2 (N1): present    (v0.9.9 (N1): use __serialization_name__)
#   F3 (N2): present    (_validate_timestamp_only)
#   F4 (N3): present    (v0.9.9 (N3): check_state in combiner)
#   F5 (N5): present    (duplicate new column RuntimeError)
#   F6 (B2): present    (if TYPE_CHECKING:  # v0.9.9 (B2))
#   F7 (B3): present    (from pathlib import Path  # v0.9.9 (B3))
#   F8 (B4): present    (PCA docstring v0.9.9 (B4))
#   F9:      MISSING    (docstring still v0.9.1e; AST silent no-op)
#   F1 dedent: MISSING  (fix1 died before fix_dedent())
#
# Remaining work in fix1a:
#   1. textwrap.dedent in run_verify_in_subprocess (nested verify_fn)
#   2. F9 docstring (exact anchor from merged snapshot)
#   3. dogfood via subprocess (fresh imports)
#   4. pytest fast+slow
#   5. commit
#
# Run: uv run python scripts/patch_v099_fix1a.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


def apply_or_fail(path: str, old: str, new: str, *, label: str) -> bool:
    """Replace old->new in path; return True if applied, False if already.

    Fail-fast (no AST fallback, no silent no-op) if anchor missing.
    Prints 10 lines of context around partial-match zone to aid fixing.
    """
    p = Path(path)
    s = p.read_text()
    if new in s:
        print(f"  {label}: already applied — skip")
        return False
    if old not in s:
        # Diagnostics: find first differing line
        print(f"  {label}: ANCHOR MISS in {path}")
        # Try find best-effort prefix match (first line of anchor)
        first_line = old.split("\n")[0].strip()
        for i, line in enumerate(s.split("\n")):
            if first_line and first_line in line:
                start = max(0, i - 1)
                for j in range(start, min(len(s.split("\n")), i + 10)):
                    print(f"    L{j+1:4d}: {s.split(chr(10))[j]}")
                break
        raise SystemExit(f"{label}: anchor missing in {path}")
    p.write_text(s.replace(old, new, 1))
    pu.assert_compiles(p)
    print(f"  {label}: applied")
    return True


# ============ F1 dedent ============

def fix_dedent():
    print("\n=== F1: textwrap.dedent in run_verify_in_subprocess ===")
    old = (
        "    source = _inspect.getsource(verify_fn)\n"
        "    with _tempfile.NamedTemporaryFile(\n"
    )
    new = (
        "    import textwrap as _tw\n"
        "    source = _tw.dedent(_inspect.getsource(verify_fn))\n"
        "    with _tempfile.NamedTemporaryFile(\n"
    )
    apply_or_fail("cadw/patch_utils.py", old, new, label="F1 dedent")


# ============ F9 docstring ============

def fix_f9():
    print("\n=== F9: _GroupBy.apply docstring (v0.9.1e -> v0.9.6b2) ===")
    old = (
        '        """Per-group apply (v0.9.1e). Prototype-validated (11 cases).\n'
        '\n'
        '        Args:\n'
        '            fn: callable(group_df: pd.DataFrame) → scalar | pd.Series |\n'
        '                pd.DataFrame. Uniform mode across groups.\n'
    )
    new = (
        '        """Per-group apply (v0.9.6b2: CADW-native, no pandas).\n'
        '\n'
        '        Args:\n'
        '            fn: callable(group_df: CadwDataFrame) → scalar |\n'
        '                dict[str, scalar] | CadwDataFrame. Uniform\n'
        '                mode across groups. For pd.Series-like output\n'
        '                use a dict; for pd.DataFrame-like output use\n'
        '                CadwDataFrame.from_table(pa_table).\n'
    )
    apply_or_fail("cadw/dataframe.py", old, new, label="F9 docstring")


# ============ verify state ============

def verify_state():
    print("\n=== verify: all F1-F9 on disk ===")
    checks = [
        ("cadw/patch_utils.py", "def run_verify_in_subprocess("),
        ("cadw/patch_utils.py", "textwrap.dedent(_inspect.getsource"),
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
    importlib.reload(pu)  # pick up dedent

    def _verify_all():
        import inspect
        from cadw.patch_utils import run_verify_in_subprocess
        from cadw.pipeline import ColumnTransformer, _merge_fanout
        from cadw.accessors import _dt as dt_mod
        from cadw.ops import groupby as gb_mod
        from cadw.ml import pca as pca_mod
        from cadw.dataframe import _GroupBy

        # F1: helper works on nested verify_fn (itself)
        print("  F1: helper reachable (running in its own subprocess)")

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
    print("=== v0.9.9-fix1a ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_dedent()
    fix_f9()
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.9-fix1a: complete v0.9.9. Replaces failed fix1 "
        "(verify_state anchor mismatch: 'N1: use' not substring of "
        "'v0.9.9 (N1): use' — ')' precedes ':', §13.4/161). State "
        "on disk: F1-F8 already applied by v0.9.9 (git diff "
        "confirmed F2/F5/F6/F7/F8; F1 helper, F3, F4 by v0.9.9 log). "
        "Remaining: (1) F1 textwrap.dedent in "
        "run_verify_in_subprocess — nested verify_fn needs dedent, "
        "source from inspect.getsource has class-method indent; "
        "(2) F9 docstring v0.9.1e -> v0.9.6b2 (v0.9.9 anchor had "
        "wrong line break: actual has 'scalar | pd.Series |' one "
        "line, prior anchor expected 'scalar |\\n pd.Series'; AST "
        "fallback gave silent no-op, §13.4/160); (3) dogfood via "
        "subprocess (8th §13.4/178: 'from cadw.patch_utils import "
        "run_verify_in_subprocess' fails in-process because module "
        "in sys.modules predates F1 write — importlib.reload + "
        "pu.run_verify_in_subprocess); (4) new apply_or_fail helper "
        "in patch_v099_fix1a.py: no AST fallback, fail-fast on "
        "missing anchor with 10-line context (prevents §13.4/160 "
        "silent no-op class).",
        self_path="scripts/patch_v099_fix1a.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.9-fix1a applied. v0.9.9 COMPLETE.")


if __name__ == "__main__":
    main()
