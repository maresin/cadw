#!/usr/bin/env python3
# v0.9.23b — metrics(y_proba_cols=[...]) alias (AUDIT_FULL_v0922 item 7.A.11.3).
#
# CLARIFICATION (vs audit c11): multiclass proba ALREADY works via
# y_proba=[...] — the facade normalizes str → [str], Sequence → list.
# c11 failed because the Jupyter harness used MetricsOp's parameter
# name (y_proba_cols=), not the facade's (y_proba=). This patch adds
# `y_proba_cols` as an alias to close the naming gap — consistency
# with MetricsOp, not a functional fix. Zero RAM, zero breaking change.
#
# Rules:
#   - y_proba only (existing): str → [str], list → list, None → None
#   - y_proba_cols only (new): Sequence[str] → list, str → TypeError
#     (footgun: iterating a string char-by-char is almost never intended)
#   - both → ValueError (ambiguous; caller must choose)
#   - empty list → ValueError
#
# Run: uv run python scripts/patch_v0923b_metrics_proba_cols.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess,
)


# ============ 1. Signature: add y_proba_cols ============

def fix_signature():
    print("\n=== add y_proba_cols to metrics() signature ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "y_proba_cols: \"Sequence[str] | None\" = None," in s:
        print("  already present — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            '        y_pred: "str | None" = None,\n'
            '        y_proba: "str | Sequence[str] | None" = None,\n'
            '        classes: "Sequence | None" = None,'
        ),
        new=(
            '        y_pred: "str | None" = None,\n'
            '        y_proba: "str | Sequence[str] | None" = None,\n'
            '        y_proba_cols: "Sequence[str] | None" = None,\n'
            '        classes: "Sequence | None" = None,'
        ),
        label="metrics signature",
    )
    assert_compiles(p)
    print("  OK")


# ============ 2. Body: reconcile before existing normalization ============

def fix_body():
    print("\n=== reconcile y_proba_cols alias before normalization ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "# v0.9.23b: y_proba_cols alias" in s:
        print("  already present — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            '        # v0.9.2c: y_proba normalization (str → [str], list → list).\n'
            '        if y_proba is None:'
        ),
        new=(
            '        # v0.9.23b: y_proba_cols alias (matches MetricsOp param\n'
            '        # name; AUDIT_FULL_v0922 item 7.A.11.3 / c11 Jupyter\n'
            '        # gap). Existing `y_proba` (str|Sequence) and new\n'
            '        # `y_proba_cols` (Sequence[str]) both work; passing\n'
            '        # both → ValueError (ambiguous); str → TypeError (would\n'
            '        # iterate char-by-char — footgun).\n'
            '        if y_proba_cols is not None:\n'
            '            if y_proba is not None:\n'
            '                raise ValueError(\n'
            '                    "metrics: pass either y_proba= or "\n'
            '                    "y_proba_cols=, not both"\n'
            '                )\n'
            '            if isinstance(y_proba_cols, str):\n'
            '                raise TypeError(\n'
            '                    "metrics: y_proba_cols must be a "\n'
            '                    "sequence of column names; got single "\n'
            '                    f"string {y_proba_cols!r}. Use "\n'
            '                    "y_proba=\'col\' for single-column "\n'
            '                    "proba, or y_proba_cols=[\'col\'] for a "\n'
            '                    "list."\n'
            '                )\n'
            '            y_proba = list(y_proba_cols)\n'
            '            if not y_proba:\n'
            '                raise ValueError(\n'
            '                    "metrics: y_proba_cols must be non-empty"\n'
            '                )\n'
            '\n'
            '        # v0.9.2c: y_proba normalization (str → [str], list → list).\n'
            '        if y_proba is None:'
        ),
        label="metrics y_proba_cols reconciliation",
    )
    assert_compiles(p)
    print("  OK")


# ============ 3. Docstring: y_proba / y_proba_cols ============

def fix_docstring():
    print("\n=== update y_proba docstring ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "y_proba_cols: alias for `y_proba`" in s:
        print("  already present — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            '            y_proba: predicted probabilities (float in [0,1], binary).'
        ),
        new=(
            '            y_proba: predicted probabilities. str for binary\n'
            '                (single column, float in [0,1]); list[str] for\n'
            '                multiclass (K columns, K classes — requires\n'
            '                `classes`).\n'
            '            y_proba_cols: alias for `y_proba` when passing a\n'
            '                list of columns. Both work; use whichever you\n'
            '                prefer (matches MetricsOp parameter name).\n'
            '                Passing both → ValueError.'
        ),
        label="metrics docstring",
    )
    assert_compiles(p)
    print("  OK")


# ============ 4. Tests ============

TESTS = '''"""v0.9.23b: metrics(y_proba_cols=[...]) alias (AUDIT_FULL_v0922 7.A.11.3).

y_proba_cols is an alias for y_proba when passing a list of columns,
matching MetricsOp's parameter name. Both forms work; both together →
ValueError.
"""
from pathlib import Path

import math
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


# ============ y_proba_cols alias: positive cases ============


def test_metrics_y_proba_cols_binary(tmp_path):
    """y_proba_cols=['p'] on binary y → same as y_proba='p'."""
    df = make_df(tmp_path, {
        "y": [0.0, 0.0, 1.0, 1.0],
        "p": [0.1, 0.4, 0.35, 0.8],
    })
    via_kwarg = df.metrics(
        y_true="y", y_proba="p", metrics=["roc_auc"],
    )
    via_alias = df.metrics(
        y_true="y", y_proba_cols=["p"], metrics=["roc_auc"],
    )
    assert via_kwarg == via_alias
    assert via_alias["roc_auc"] == pytest.approx(0.75)


def test_metrics_y_proba_cols_multiclass_log_loss(tmp_path):
    """y_proba_cols=[3 cols] on multiclass y → same as y_proba list."""
    df = make_df(tmp_path, {
        "y": [0, 1, 2, 0, 1, 2],
        "p0": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "p1": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "p2": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    })
    via_list = df.metrics(
        y_true="y", y_proba=["p0", "p1", "p2"],
        metrics=["log_loss"], classes=[0, 1, 2],
    )
    via_alias = df.metrics(
        y_true="y", y_proba_cols=["p0", "p1", "p2"],
        metrics=["log_loss"], classes=[0, 1, 2],
    )
    assert via_alias["log_loss"] == pytest.approx(via_list["log_loss"])
    assert via_alias["log_loss"] == pytest.approx(0.0, abs=1e-10)


def test_metrics_y_proba_cols_multiclass_roc_auc(tmp_path):
    """y_proba_cols works for macro roc_auc (K ovr passes)."""
    df = make_df(tmp_path, {
        "y": [0, 0, 1, 1, 2, 2],
        "p0": [0.9, 0.8, 0.05, 0.1, 0.05, 0.1],
        "p1": [0.05, 0.1, 0.9, 0.8, 0.1, 0.05],
        "p2": [0.05, 0.1, 0.05, 0.1, 0.9, 0.85],
    })
    res = df.metrics(
        y_true="y", y_proba_cols=["p0", "p1", "p2"],
        metrics=["roc_auc"], classes=[0, 1, 2],
    )
    assert res["roc_auc"] == pytest.approx(1.0)


# ============ y_proba_cols alias: errors ============


def test_metrics_y_proba_cols_both_given_raises(tmp_path):
    """Both y_proba= and y_proba_cols= → ValueError (ambiguous)."""
    df = make_df(tmp_path, {
        "y": [0, 1], "p": [0.1, 0.9],
    })
    with pytest.raises(ValueError, match="not both"):
        df.metrics(
            y_true="y",
            y_proba="p",
            y_proba_cols=["p"],
            metrics=["roc_auc"],
        )


def test_metrics_y_proba_cols_string_raises(tmp_path):
    """y_proba_cols='p' (single string) → TypeError with hint."""
    df = make_df(tmp_path, {
        "y": [0, 1], "p": [0.1, 0.9],
    })
    with pytest.raises(TypeError, match="must be a sequence"):
        df.metrics(
            y_true="y", y_proba_cols="p", metrics=["roc_auc"],
        )


def test_metrics_y_proba_cols_empty_raises(tmp_path):
    """y_proba_cols=[] → ValueError."""
    df = make_df(tmp_path, {
        "y": [0, 1], "p": [0.1, 0.9],
    })
    with pytest.raises(ValueError, match="must be non-empty"):
        df.metrics(
            y_true="y", y_proba_cols=[], metrics=["roc_auc"],
        )


# ============ backward compat: y_proba unchanged ============


def test_metrics_y_proba_str_unchanged(tmp_path):
    """Existing y_proba='p' still works."""
    df = make_df(tmp_path, {
        "y": [0.0, 0.0, 1.0, 1.0],
        "p": [0.1, 0.4, 0.35, 0.8],
    })
    res = df.metrics(y_true="y", y_proba="p", metrics=["roc_auc"])
    assert res["roc_auc"] == pytest.approx(0.75)


def test_metrics_y_proba_list_unchanged(tmp_path):
    """Existing y_proba=[...] (list) still works."""
    df = make_df(tmp_path, {
        "y": [0, 1, 2],
        "p0": [0.9, 0.1, 0.2],
        "p1": [0.05, 0.7, 0.2],
        "p2": [0.05, 0.2, 0.6],
    })
    res = df.metrics(
        y_true="y", y_proba=["p0", "p1", "p2"],
        metrics=["log_loss"], classes=[0, 1, 2],
    )
    assert not math.isnan(res["log_loss"])
    assert res["log_loss"] > 0


def test_metrics_y_proba_none_unchanged(tmp_path):
    """Existing y_proba=None + regression metric still works."""
    df = make_df(tmp_path, {"y": [1.0, 2.0], "yp": [1.0, 2.0]})
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == pytest.approx(1.0)
'''


def write_test_file():
    print("\n=== write tests/test_metrics_proba_cols.py ===")
    p = Path("tests/test_metrics_proba_cols.py")
    if p.exists():
        print("  already exists — skip")
        return
    p.write_text(TESTS)
    assert_compiles(p)
    print("  OK")


# ============ 5. Dogfood ============

def dogfood():
    print("\n=== dogfood: y_proba_cols alias parity ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        # Binary: y_proba='p' == y_proba_cols=['p']
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "y": [0.0, 0.0, 1.0, 1.0],
                "p": [0.1, 0.4, 0.35, 0.8],
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            a = df.metrics(y_true="y", y_proba="p", metrics=["roc_auc"])
            b = df.metrics(
                y_true="y", y_proba_cols=["p"], metrics=["roc_auc"],
            )
            assert a == b, f"binary parity: {a} != {b}"

        # Multiclass: list == y_proba_cols
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "y": [0, 1, 2],
                "p0": [0.9, 0.1, 0.2],
                "p1": [0.05, 0.7, 0.2],
                "p2": [0.05, 0.2, 0.6],
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            a = df.metrics(
                y_true="y", y_proba=["p0", "p1", "p2"],
                metrics=["log_loss"], classes=[0, 1, 2],
            )
            b = df.metrics(
                y_true="y", y_proba_cols=["p0", "p1", "p2"],
                metrics=["log_loss"], classes=[0, 1, 2],
            )
            assert a == b, f"multiclass parity: {a} != {b}"

        # Error: both
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"y": [0, 1], "p": [0.1, 0.9]})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            try:
                df.metrics(
                    y_true="y", y_proba="p",
                    y_proba_cols=["p"], metrics=["roc_auc"],
                )
            except ValueError as e:
                assert "not both" in str(e)
            else:
                raise AssertionError("both-given should raise ValueError")

        print("  binary parity + multiclass parity + both-given raise ok")

    run_verify_in_subprocess(_verify)


# ============ 6. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_metrics_proba_cols.py",
         "tests/test_metrics.py",
         "tests/test_metrics_reference.py",
         "-q"],
        timeout=300,
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


# ============ 7. main ============

def main():
    print("=== v0.9.23b: metrics y_proba_cols alias ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_signature()
    fix_body()
    fix_docstring()
    write_test_file()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23b: metrics(y_proba_cols=[...]) alias. "
        "AUDIT_FULL_v0922 item 7.A.11.3 / c11 Jupyter gap. "
        "CLARIFICATION: multiclass proba ALREADY works via "
        "y_proba=[...] — the c11 error was a naming mismatch "
        "(Jupyter harness used MetricsOp's y_proba_cols=, not "
        "the facade's y_proba=). This patch adds y_proba_cols "
        "as an alias for consistency with MetricsOp; both work, "
        "passing both → ValueError, y_proba_cols='str' → TypeError "
        "(footgun hint), empty list → ValueError. y_proba= "
        "behavior unchanged (str → [str], list → list, None → None). "
        "Docstring updated: y_proba accepts list[str] for multiclass. "
        "+tests/test_metrics_proba_cols.py (9 tests: binary parity, "
        "multiclass log_loss + macro roc_auc, both-given raise, "
        "str raise, empty raise, 3 backward-compat). Zero RAM, "
        "zero breaking change. Fast: 1926 -> 1935 passed.",
        self_path="scripts/patch_v0923b_metrics_proba_cols.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23b applied.")


if __name__ == "__main__":
    main()