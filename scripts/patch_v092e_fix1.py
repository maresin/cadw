#!/usr/bin/env python3
"""v0.9.2e-fix1: _attach_column nullability + TSS test data (§13.4/169).

BUG 1 (9 failures): _attach_column built out_schema with
`pa.field("__fold", type, nullable=False)`. Dataset.from_batches
reconstructs per-column singleton via
`pa.RecordBatch.from_arrays([col], names=[field_name])` — ALWAYS
creates nullable=True field. Writer target created with
`pa.schema([field])` where field nullable=False → schema mismatch →
ArrowInvalid.

CADW convention: all output fields nullable=True (matches from_arrays
behavior; existing ops shift/diff/rank all use nullable=True).
Fix: force nullable=True in _attach_column for ALL fields.

BUG 2 (1 failure): test_cv_score_timeseries used linear trend data
(`y = 2*ts`, x=y). TSS splits train=[0,50), test=[50,100) — test x
values 100..198 OUTSIDE training bin range. GBDT without lag features
does not extrapolate (all test values clip to last bin). r2 = -23.9.

This is not a CADW bug — inherent tree-model limitation (§11.5, no
lag features yet — TimeSeriesPipeline backlog). Test data was
unsuitable. Fix: use i.i.d. x ~ N(0,1), y = 2x + noise. TSS still
splits by ts column order, but x/y distributions match train vs test.
r2 well above 0.3.

Minor: SyntaxWarning in test regex (match="n=3 < n_splits\\+1") —
missing raw-string prefix. Fix in test.

GUARDS: cv fast + E2E + gbdt + pipeline + full suite + guards.

Run: uv run python scripts/patch_v092e_fix1.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ Fix 1: _attach_column nullability ============

def fix_attach_column():
    print("\n=== cadw/cv.py: _attach_column nullable=True ===")
    p = Path("cadw/cv.py")
    s = p.read_text()

    old = '''def _attach_column(
    df: CadwDataFrame, name: str, arrow_array: pa.Array,
) -> CadwDataFrame:
    """Materialize df's filtered view + attach Arrow array as column.

    Caller ensures len(arrow_array) == df.count() over filtered view.
    Errors if iterated rows != len(arrow_array).
    """
    base_cols = list(df._columns)
    out_fields = [df._dataset.schema.field(c) for c in base_cols]
    out_fields.append(pa.field(name, arrow_array.type, nullable=False))
    out_schema = pa.schema(out_fields)

    expected_n = len(arrow_array)

    def gen():
        off = 0
        for b in df._iter_filtered_batches():
            n_b = b.num_rows
            if off + n_b > expected_n:
                raise RuntimeError(
                    f"_attach_column: iterated rows exceed array length "
                    f"({off + n_b} > {expected_n})"
                )
            col = arrow_array.slice(off, n_b)
            off += n_b
            yield b.append_column(name, col)
        if off != expected_n:
            raise RuntimeError(
                f"_attach_column: iterated {off} rows != "
                f"array length {expected_n}"
            )

    new_ds = Dataset.from_batches(out_schema, gen(), df._new_dataset_dir())
    return CadwDataFrame(
        new_ds,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir,
    )'''

    new = '''def _attach_column(
    df: CadwDataFrame, name: str, arrow_array: pa.Array,
) -> CadwDataFrame:
    """Materialize df's filtered view + attach Arrow array as column.

    Caller ensures len(arrow_array) == df.count() over filtered view.
    Errors if iterated rows != len(arrow_array).

    v0.9.2e-fix1 (§13.4/169): ALL output fields forced nullable=True.
    `Dataset.from_batches` reconstructs per-column singleton via
    `pa.RecordBatch.from_arrays([col], names=[field_name])` — always
    nullable=True. If out_schema has nullable=False for ANY field,
    writer target `pa.schema([field])` (nullable=False) mismatches
    singleton's (nullable=True) → ArrowInvalid. Convention: nullable
    =True everywhere, matching from_arrays default (all CADW ops do
    this: shift/diff/rank/expanding all use nullable=True).
    """
    base_cols = list(df._columns)
    out_fields = [
        pa.field(c, df._dataset.schema.field(c).type, nullable=True)
        for c in base_cols
    ]
    out_fields.append(pa.field(name, arrow_array.type, nullable=True))
    out_schema = pa.schema(out_fields)

    expected_n = len(arrow_array)

    def gen():
        off = 0
        for b in df._iter_filtered_batches():
            n_b = b.num_rows
            if off + n_b > expected_n:
                raise RuntimeError(
                    f"_attach_column: iterated rows exceed array length "
                    f"({off + n_b} > {expected_n})"
                )
            col = arrow_array.slice(off, n_b)
            off += n_b
            arrays = [b.column(c) for c in base_cols] + [col]
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
        if off != expected_n:
            raise RuntimeError(
                f"_attach_column: iterated {off} rows != "
                f"array length {expected_n}"
            )

    new_ds = Dataset.from_batches(out_schema, gen(), df._new_dataset_dir())
    return CadwDataFrame(
        new_ds,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir,
    )'''

    s = replace_line_block(s, old, new, tag="attach-column-nullability")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2: test_cv_score_timeseries data ============

def fix_tss_test_data():
    print("\n=== tests/test_cv.py: TSS test with i.i.d. data ===")
    p = Path("tests/test_cv.py")
    s = p.read_text()

    old = '''def test_cv_score_timeseries(tmp_path):
    """TSS cv_score on linear time series — mean r2 > 0 (learnable)."""
    n = 200
    ts = list(range(n))
    y = [float(t) * 2.0 for t in ts]
    x = y  # perfect predictor
    data = {"ts": ts, "x": x, "y": y}
    df = make_df(tmp_path, data, chunk=20)

    cv = TimeSeriesSplit(n_splits=3, time_col="ts")

    def factory():
        return GBDTStep(["x"], "y", n_estimators=3, max_depth=1)

    result = cv_score(
        df, cv, factory,
        X_cols=["x"], y_col="y", metrics=["r2"],
    )
    assert len(result["fold_scores"]) == 3
    assert result["mean"]["r2"] > 0.3'''

    new = '''def test_cv_score_timeseries(tmp_path):
    """TSS cv_score on i.i.d. data — mean r2 > 0.3.

    §13.4/169: prior version used linear trend (y = 2*ts, x = y). TSS
    split train=[0,50), test=[50,100) — test x values 100..198 OUTSIDE
    training bin range → GBDT clips to last bin → r2 ≈ -23.9. Not a
    CADW bug — tree model without lag features cannot extrapolate
    (§11.5). Test used unsuitable data for TSS.

    Fix: i.i.d. x ~ N(0,1), y = 2x + noise. TSS still splits by ts
    order, but x/y distributions in train vs test are identical →
    GBDT learns well (r2 > 0.9 typically).

    For real time-series with trend, use TimeSeriesPipeline with lag
    features (backlog v0.9.4) so GBDT sees bounded training distribution.
    """
    rng = np.random.default_rng(42)
    n = 200
    ts = list(range(n))
    x = rng.normal(size=n).tolist()  # i.i.d., no trend
    y = [2.0 * xi + 0.1 * rng.normal() for xi in x]
    data = {"ts": ts, "x": x, "y": y}
    df = make_df(tmp_path, data, chunk=20)

    cv = TimeSeriesSplit(n_splits=3, time_col="ts")

    def factory():
        return GBDTStep(["x"], "y", n_estimators=10, max_depth=2)

    result = cv_score(
        df, cv, factory,
        X_cols=["x"], y_col="y", metrics=["r2"],
    )
    assert len(result["fold_scores"]) == 3
    assert result["mean"]["r2"] > 0.3'''

    s = replace_line_block(s, old, new, tag="tss-test-data")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 3: raw string in TSS regex ============

def fix_tss_regex():
    print("\n=== tests/test_cv.py: raw-string regex (n_splits) ===")
    p = Path("tests/test_cv.py")
    s = p.read_text()

    old = '    with pytest.raises(ValueError, match="n=3 < n_splits\\+1"):'
    new = '    with pytest.raises(ValueError, match=r"n=3 < n_splits\\+1"):'

    if new in s:
        print("  already raw — skip")
        return
    if old not in s:
        print("  NOT FOUND — skip")
        return
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 4: cleanup shuffle test placeholder ============

def fix_shuffle_test():
    print("\n=== tests/test_cv.py: shuffle different assignment (clean) ===")
    p = Path("tests/test_cv.py")
    s = p.read_text()

    old = '''def test_kfold_shuffle_different_assignment(tmp_path):
    """shuffle=True with different seeds → different fold assignment."""
    df = make_xy(tmp_path, n=100)
    cv1 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv2 = KFold(n_splits=5, shuffle=True, random_state=99)

    # Extract test row sums (y column) per fold — proxy for assignment
    sums1 = []
    sums2 = []
    for _, _, test in cv1.split(df):
        n_test = test.count()
        s = sum(test["y"]._pred  # type: ignore[attr-defined]
                if False else [0])  # placeholder
        # Instead: count non-null y in test
        sums1.append(n_test)
    for _, _, test in cv2.split(df):
        sums2.append(test.count())

    # Sizes equal, but assignment should differ
    assert sums1 == sums2  # sizes same
    # Verify order differs via y column first value
    # (simpler: run twice with same seed, compare)
    # Skip deep comparison — sizes + non-overlap test covers main correctness'''

    new = '''def test_kfold_shuffle_different_assignment(tmp_path):
    """shuffle=True with different seeds → different fold assignment.

    Use identity data (y == row_id) so each fold's y-set is the
    row-position set assigned to that fold. Different seeds must
    yield different per-fold content.
    """
    data = {"x": list(range(100)), "y": list(range(100))}
    df = make_df(tmp_path, data)
    cv1 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv2 = KFold(n_splits=5, shuffle=True, random_state=99)

    def _test_y_sets(cv, df):
        sets = []
        for _, _, test in cv.split(df):
            ys = []
            for b in test._iter_filtered_batches():
                ys.extend(b.column("y").to_pylist())
            sets.append(set(ys))
        return sets

    sets1 = _test_y_sets(cv1, df)
    sets2 = _test_y_sets(cv2, df)

    # Sizes equal (same n, k)
    assert [len(s) for s in sets1] == [20, 20, 20, 20, 20]
    assert [len(s) for s in sets2] == [20, 20, 20, 20, 20]

    # Assignment differs: at least one fold has different content
    differ = any(s1 != s2 for s1, s2 in zip(sets1, sets2))
    assert differ, (
        "different seeds must produce different fold assignments"
    )

    # Invariant: each row appears exactly once across test folds
    for sets in (sets1, sets2):
        union = set().union(*sets)
        assert union == set(range(100))'''

    s = replace_line_block(s, old, new, tag="shuffle-test-clean")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 5: DECISIONS §13.4/169 + §11.6 ============

def append_decisions_169():
    print("\n=== DECISIONS.md: §13.4/169 + §11.6 note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 169" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/169 (_attach_column nullable; TSS+GBDT data).** "
        s += "nullable=False in out_schema → ArrowInvalid; TSS trend "
        s += "data → GBDT extrapolation r2 = -23.9.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 169 (nullable=False violates Dataset.from_batches
convention; TSS+GBDT data design).** v0.9.2e: 10 test failures,
2 root causes.

**Bug 1 (9 failures, `ArrowInvalid: different schema`):**
`_attach_column` built `out_schema` with
`pa.field("__fold", type, nullable=False)`. `Dataset.from_batches`
(core) reconstructs per-column singleton via
`pa.RecordBatch.from_arrays([col], names=[field_name])` — **always**
creates nullable=True field. Writer target created with
`pa.schema([field])` where field nullable=False → schema mismatch
→ ArrowInvalid.

This is a **CADW convention**: all output fields must be nullable=True
(matching `from_arrays` default). Existing ops (shift/diff/rank/
expanding/resample) all use nullable=True. My `_attach_column`
violated convention by explicit nullable=False on `__fold`.

Fix: force nullable=True everywhere. Yield batches with explicit
`pa.RecordBatch.from_arrays(arrays, schema=out_schema)` — matches
writer target.

**Bug 2 (1 failure, r2 = -23.9):**
`test_cv_score_timeseries` used linear trend data
(`y = 2*ts`, `x = y`). TSS split fold 0: train ts=[0,50), test
ts=[50,100). Test x values 100..198 OUTSIDE training bin range
(0..98). GBDT without lag features **does not extrapolate** — all
test values clip to last bin (pred ≈ 95 constant). SS_res >> SS_tot
→ negative r2.

**Not a CADW bug** — inherent tree-model limitation, documented
§11.5, motivation for TimeSeriesPipeline (lag features — backlog).
**Bug in test**: unsuitable data for TSS+GBDT. Test design must
respect inherent model limitation.

Fix: i.i.d. x ~ N(0,1), y = 2x + noise. TSS splits by ts order but
x/y distributions align train vs test. r2 well above 0.3.

**Правило (§13.4/169, расширение 128, 129, 158, 166):**
- **Output schema nullability = nullable=True** для всех полей,
  produced через `Dataset.from_batches`. Core rebuilds singleton
  via `from_arrays(names=...)` (always nullable=True); writer target
  must match. Non-nullable output fields → ArrowInvalid. Latent
  backlog: align `Dataset.from_batches` to handle nullable=False by
  using `from_arrays(arrays, schema=pa.schema([field]))` for singleton
  (core change).
- **Test data must respect inherent model limitations.** TSS + GBDT
  (без lag features) cannot extrapolate. Test with trend data
  ensures failure, не от bug'а в коде. Real use case — TimeSeriesPipeline
  с lags. Test i.i.d. data, or bounded (periodic) features.

**Связанные:** §13.4/128 (test expectations authored), /129 (rolling
reference), /158 (E2E respects inherent complexity), /166 (prototype
test-authoring).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/169")


def append_changelog_fix1():
    print("\n=== CHANGELOG.md: v0.9.2e-fix1 note ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.2e-fix1" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Исправлено (v0.9.2e-fix1 — nullability + test data)

- **`cadw/cv.py::_attach_column`** — все output fields nullable=True.
  `nullable=False` на `__fold` ломал `Dataset.from_batches`:
  singleton-batch через `pa.RecordBatch.from_arrays([col],
  names=[name])` всегда nullable=True, writer target с
  `pa.schema([field])` (nullable=False) → ArrowInvalid. 9 failures.
  CADW convention: output fields always nullable=True (all existing
  ops соблюдают).
- **`test_cv_score_timeseries`** — i.i.d. data (`x ~ N(0,1)`,
  `y = 2x + noise`), не linear trend. Trend + TSS → GBDT
  extrapolation outside training bin range → r2 = -23.9 (inherent
  limitation §11.5; test design was wrong, not code). 1 failure.
- **`test_kfold_shuffle_different_assignment`** — cleaned placeholder
  (was doing nothing); proper set-of-y comparison + union invariant.
- **`test_tss_n_lt_k_plus_1_raises`** — raw-string regex (silence
  SyntaxWarning).

### §13.4/169 — nullability convention + test-respects-limitation

- **Output schema = nullable=True** для всех полей, produced через
  `Dataset.from_batches`. Core rebuilds singleton via
  `from_arrays(names=...)` (always nullable=True); writer target must
  match. Backlog: align `Dataset.from_batches` for nullable=False.
- **Test data must respect inherent model limitations.** TSS + GBDT
  (без lag features) cannot extrapolate. Use i.i.d. or bounded
  (periodic) features. Real time-series → TimeSeriesPipeline (lags,
  backlog v0.9.4).

'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.2e-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_attach_column()
    fix_tss_test_data()
    fix_tss_regex()
    fix_shuffle_test()
    append_decisions_169()
    append_changelog_fix1()

    print("\n=== pytest: cv fast (was 10 failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_cv.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** CV STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: cv E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_cv_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** CV E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: gbdt + pipeline regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_gbdt_regression.py",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

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

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.2e: cross-validation — KFold(shuffle=False|True) + "
        "TimeSeriesSplit(no leakage) + cv_score; §13.4/169 "
        "(_attach_column nullable=True; TSS+GBDT i.i.d. test data; "
        "nullability convention)"])

    print("\n==> v0.9.2e applied (v092e + fix1 combined).")


if __name__ == "__main__":
    main()