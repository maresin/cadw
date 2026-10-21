#!/usr/bin/env python3
# v0.9.9-fix1 — 3 fixes to complete v0.9.9.
#
# Bug 1 (F1 dedent): run_verify_in_subprocess writes inspect.getsource
#   verbatim. For nested verify_fn (dogfood uses nested `_verify_all`;
#   test_run_verify_in_subprocess_ok uses nested `dogfood_ok`), source
#   has indentation → top-level temp file → IndentationError.
#   Fix: textwrap.dedent.
#
# Bug 2 (F9 stale anchor): anchor had "groups." on own line; actual
#   disk has "Uniform mode across groups." on one line. `old in s`
#   → False → silent skip (AST found apply, printed diagnostic, no
#   WARNING, no write). §13.4/160 (stale anchor).
#   Fix: explicit anchor from actual disk content.
#
# Bug 3 (§13.4/178, 8th): dogfood called in-process → `from
#   cadw.patch_utils import run_verify_in_subprocess` fails (module
#   in sys.modules predates F1's write). 8th occurrence.
#   Fix: dogfood in subprocess (via reloaded helper — real fix, not
#   workaround).
#
# After v0.9.9-fix1: v0.9.9 COMPLETE (F1-F9 applied, verified, green).
#
# Run: uv run python scripts/patch_v099_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
# NOTE: do NOT `from cadw.patch_utils import run_verify_in_subprocess` at
# top — we'll fix dedent on disk first, then reload, then use
# pu.run_verify_in_subprocess (which reads module attribute → fresh).


# ============ Fix 1: dedent in run_verify_in_subprocess ============

def fix_dedent():
    print("\n=== fix 1: textwrap.dedent in run_verify_in_subprocess ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "textwrap.dedent(_inspect.getsource(verify_fn))" in s:
        print("  already dedented — skip")
        return
    old = (
        "    source = _inspect.getsource(verify_fn)\n"
        "    with _tempfile.NamedTemporaryFile(\n"
    )
    new = (
        "    import textwrap as _tw\n"
        "    source = _tw.dedent(_inspect.getsource(verify_fn))\n"
        "    with _tempfile.NamedTemporaryFile(\n"
    )
    if old not in s:
        raise SystemExit("run_verify_in_subprocess anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    pu.assert_compiles(p)
    print("  OK")


# ============ Fix 2: F9 explicit anchor from actual disk ============

def fix_f9_anchor():
    print("\n=== fix 2: F9 docstring (actual disk anchor) ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "Per-group apply (v0.9.6b2: CADW-native, no pandas)." in s:
        print("  already fixed — skip")
        return
    # Actual disk has `scalar | pd.Series |` on one line.
    # Prior v0.9.9 anchor had extra line break before "groups".
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
    if old not in s:
        raise SystemExit(
            "F9 anchor still not found — verify actual docstring. "
            "Expected: Per-group apply (v0.9.1e) + 'scalar | pd.Series |' "
            "on one line + 'Uniform mode across groups.' on next line."
        )
    s = s.replace(old, new, 1)
    p.write_text(s)
    pu.assert_compiles(p)
    print("  OK")


# ============ Fix 3: dogfood in subprocess (8th §13.4/178) ============

def dogfood():
    print("\n=== fix 3: dogfood in fresh subprocess (via reloaded helper) ===")
    # Reload so pu.run_verify_in_subprocess has dedent (just written).
    importlib.reload(pu)

    def _verify_all():
        import inspect
        from cadw.pipeline import ColumnTransformer, _merge_fanout
        from cadw.accessors import _dt as dt_mod
        from cadw.ops import groupby as gb_mod
        from cadw.ml import pca as pca_mod
        from cadw.dataframe import _GroupBy

        src = inspect.getsource(ColumnTransformer.to_dict)
        assert "__serialization_name__" in src
        print("  F2 (N1): OK")

        src = inspect.getsource(dt_mod._validate_timestamp_only)
        assert "is_timestamp" in src
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
        assert "\nfrom pathlib import Path " in ps or ps.startswith("from pathlib import Path ")
        print("  F7 (B3): OK")

        assert "v0.9.9 (B4)" in inspect.getsource(pca_mod.PCA)
        print("  F8 (B4): OK")

        assert "v0.9.6b2: CADW-native" in inspect.getsource(_GroupBy.apply)
        print("  F9 (docstring): OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ verify state on disk ============

def verify_state():
    print("\n=== verify: F1-F8 on disk ===")
    checks = [
        ("cadw/patch_utils.py", "def run_verify_in_subprocess("),
        ("cadw/pipeline.py", "N1: use __serialization_name__"),
        ("cadw/accessors/_dt.py", "def _validate_timestamp_only("),
        ("cadw/ops/groupby.py", "v0.9.9 (N3)"),
        ("cadw/pipeline.py", "v0.9.9 (N5)"),
        ("cadw/ml/pca.py", "if TYPE_CHECKING:  # v0.9.9 (B2)"),
        ("cadw/pipeline.py", "# v0.9.9 (B3): for `-> Path` annotation"),
        ("cadw/ml/pca.py", "v0.9.9 (B4)"),
    ]
    for path, anchor in checks:
        if anchor not in Path(path).read_text():
            raise SystemExit(f"{path}: missing {anchor!r}")
        print(f"  {path}: OK")


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


# ============ main ============

def main():
    print("=== v0.9.9-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    verify_state()
    fix_dedent()
    fix_f9_anchor()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.9-fix1: complete v0.9.9 (3 bugs in v0.9.9 patch itself). "
        "(1) F1 helper missing textwrap.dedent → nested verify_fn "
        "(dogfood's _verify_all, test's dogfood_ok) fails at subprocess "
        "with IndentationError. Fix: dedent. (2) F9 anchor had extra "
        "line break ('groups.' on own line); actual disk has 'Uniform "
        "mode across groups.' on one line. old in s = False → AST "
        "fallback printed diagnostic but no WARNING (found=True), no "
        "write — silent no-op (§13.4/160). Fix: explicit anchor from "
        "actual disk. (3) dogfood called in-process → 'from "
        "cadw.patch_utils import run_verify_in_subprocess' fails "
        "(8th §13.4/178): module loaded before F1's write. Fix: "
        "importlib.reload + pu.run_verify_in_subprocess (fresh "
        "subprocess, by construction). v0.9.9 COMPLETE: F1-F9 applied, "
        "docstring CADW-native, dogfood verified via subprocess, "
        "pytest fast/slow green.",
        self_path="scripts/patch_v099_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.9-fix1 applied. v0.9.9 COMPLETE.")


if __name__ == "__main__":
    main()
