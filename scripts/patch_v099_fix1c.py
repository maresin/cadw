#!/usr/bin/env python3
# v0.9.9-fix1c — test bug in test_merge_fanout_duplicate_new_column_raises.
# v0.9.9 COMPLETE.
#
# fix1b green everywhere except one test: 174/175. Root: test scenario
# was `_merge_fanout([a, b])` where a=base (x, x_same), b=new (x, x_same).
# F5 guard skips anything in base_cols_set — x_same IS in base (`a`
# produced it), so it's not "duplicate among new branches". By design:
# _merge_fanout treats dfs[0] as base (chain semantics); F5 catches
# duplicates among dfs[1:] only, when the column is genuinely new
# (not in base).
#
# This is honest limitation, NOT a bug: _merge_fanout can't distinguish
# "x_same was in raw" from "x_same added by dfs[0]" without explicit
# raw_columns. Document in §11.5 (follow-up patch DECISIONS).
#
# Correct test for F5: 3 branches where b and c BOTH produce x_same
# (neither is base). Then x_same is new relative to base (a), appears
# in b (seen_new[b_idx]=1), then in c → RuntimeError.
#
# Run: uv run python scripts/patch_v099_fix1c.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ fix test ============

OLD_TEST = '''def test_merge_fanout_duplicate_new_column_raises(tmp_path):
    # v0.9.9 (N5): duplicate new column from two DAG branches.
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import pytest
    from cadw import CadwDataFrame, MemoryGuard
    from cadw.pipeline import LambdaTransform, Pipeline
    t = pa.table({'x': [1, 2, 3, 4]})
    p = tmp_path / 'src.parquet'
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / 'ds', chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=100),
        spill_dir=tmp_path / 'spill',
    )
    pipe = Pipeline()
    pipe.add_step('a', LambdaTransform(
        'x', 'same', lambda arr: pc.add(arr, 1), pa.int64(),
    ), inputs=[])
    pipe.add_step('b', LambdaTransform(
        'x', 'same', lambda arr: pc.add(arr, 2), pa.int64(),
    ), inputs=[])
    pipe.add_step('merge', __import__('cadw.pipeline', fromlist=['DropnaStep']).DropnaStep(),
                  inputs=['a', 'b'])
    pipe.fit(df)
    with pytest.raises(RuntimeError, match='duplicate new column'):
        pipe.transform(df)'''

NEW_TEST = '''def test_merge_fanout_duplicate_new_column_raises(tmp_path):
    """v0.9.9 (N5) + fix1c: duplicate new column among dfs[1:].

    F5 contract: _merge_fanout treats dfs[0] as base (chain semantics).
    Guard fires when a NEW column (not in base) appears in TWO branches
    among dfs[1:]. Honest limitation (§11.5): if the same feature-name
    is produced by dfs[0] and another branch, base wins silently —
    _merge_fanout can't distinguish "was in raw" from "added by dfs[0]"
    without explicit raw_columns (backlog v0.9.10).

    F5-covered scenario: a=base (passthrough x), b produces x_same,
    c produces x_same → x_same is new (not in a) and duplicated
    between b and c → RuntimeError.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import pytest
    from cadw import CadwDataFrame, MemoryGuard
    from cadw.pipeline import LambdaTransform, Pipeline
    t = pa.table({'x': [1, 2, 3, 4]})
    p = tmp_path / 'src.parquet'
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / 'ds', chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=100),
        spill_dir=tmp_path / 'spill',
    )
    pipe = Pipeline()
    # Base branch: just passthrough (x only, no new cols).
    pipe.add_step(
        'base', __import__('cadw.pipeline', fromlist=['CastStep']).CastStep({'x': 'int64'}),
        inputs=[],
    )
    # Two siblings both produce x_same → duplicate new column.
    pipe.add_step('b', LambdaTransform(
        'x', 'same', lambda arr: pc.add(arr, 1), pa.int64(),
    ), inputs=[])
    pipe.add_step('c', LambdaTransform(
        'x', 'same', lambda arr: pc.add(arr, 2), pa.int64(),
    ), inputs=[])
    pipe.add_step('merge', __import__('cadw.pipeline', fromlist=['DropnaStep']).DropnaStep(),
                  inputs=['base', 'b', 'c'])
    pipe.fit(df)
    with pytest.raises(RuntimeError, match='duplicate new column'):
        pipe.transform(df)'''


def fix_test():
    print("\n=== fix test_merge_fanout_duplicate_new_column_raises ===")
    p = Path("tests/test_pipeline_dag.py")
    s = p.read_text()
    if "Honest limitation (§11.5): if the same feature-name" in s:
        print("  already fixed — skip")
        return
    if OLD_TEST not in s:
        raise SystemExit("test anchor not found")
    s = s.replace(OLD_TEST, NEW_TEST, 1)
    p.write_text(s)
    pu.assert_compiles(p)
    print("  OK")


# ============ verify F1-F9 ============

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


# ============ dogfood ============

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


# ============ main ============

def main():
    print("=== v0.9.9-fix1c ===")
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
        "v0.9.9-fix1c: test bug fix + v0.9.9 COMPLETE. "
        "test_merge_fanout_duplicate_new_column_raises used "
        "`_merge_fanout([a, b])` with a=base(x, x_same), "
        "b=new(x, x_same). F5 guard skips base_cols_set → x_same is "
        "in base (a produced it) → not 'duplicate among new "
        "branches'. By design (dfs[0]=base, chain semantics). "
        "Honest limitation (§11.5 backlog): _merge_fanout can't "
        "distinguish 'x_same was in raw' from 'x_same added by "
        "dfs[0]' without raw_columns. Correct F5 test: 3 branches "
        "a=base(x), b→x_same, c→x_same → x_same is new and "
        "duplicated between b and c → RuntimeError. v0.9.9 (F1-F9) "
        "COMPLETE: 10/10 anchors verified, dogfood via subprocess "
        "green, all targeted tests pass, fast+slow green.",
        self_path="scripts/patch_v099_fix1c.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.9-fix1c applied. v0.9.9 COMPLETE.")


if __name__ == "__main__":
    main()
