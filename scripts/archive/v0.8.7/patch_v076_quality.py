#!/usr/bin/env python3
"""v0.7.6.1: quality fixes.

Bugs / improvements:
    B2 (S2, latent): `_NumpyBackedBuilder.finish()` инвертирует null
        bitmap. `packbits(mask=True=null)` даёт `1` для null-позиций, а
        Arrow читает `1=valid`. Результат: null↔valid инвертированы.
        Диагностика (v0.7.6): `packbits([F,F,T]) → [None,None,30]` (2 nulls),
        правильно `[10,20,None]` (1 null). Fix: `packbits(~mask)`.
        Native builders (Int64Builder/Float64Builder) в текущем pyarrow 26
        есть — баг latent, но real: срабатывает для типов без native
        builder'а и после любой мутации pyarrow.

    B1: `fit_multinomial_nb` не валидирует X >= 0. Silent NaN при
        отрицательных значениях (feature_sum + alpha может стать
        отрицательным → log(negative) = nan). Fix: 1-pass check.

    I5: `_validate_binary_y` — 1 скан; `_fit_boosting` делает 2-й скан
        для y_sum/n_valid. Fix: return `(y_sum, n_valid)` из validation.

Tag: v0.7.6.1 (patch-level; не ломает API).

Run: uv run python scripts/patch_v076_quality.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
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


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


# ============ B2: _NumpyBackedBuilder.finish() ============

def patch_b2_bitmap():
    print("\n=== B2: _NumpyBackedBuilder.finish() — packbits(~mask) ===")
    p = Path("cadw/ops/sort.py")
    s = p.read_text()

    # Verify current state
    if "packed = np.packbits(mask, bitorder=\"little\")" not in s:
        if "packed = np.packbits(~mask, bitorder=\"little\")" in s:
            print("  already fixed — skip")
            return
        raise SystemExit("B2: neither 'packbits(mask)' nor 'packbits(~mask)' found in finish()")

    old = '''    def finish(self) -> pa.Array:
        import numpy as np
        values = self._buf[: self._n].copy()
        mask = self._nulls[: self._n]
        if mask.any():
            packed = np.packbits(mask, bitorder="little")
            arr = pa.Array.from_buffers(
                self._arrow_type, self._n,
                [pa.py_buffer(packed), pa.py_buffer(values)],
            )
        else:
            arr = pa.Array.from_buffers(
                self._arrow_type, self._n,
                [None, pa.py_buffer(values)],
            )
        # Reset
        self._capacity = 64
        self._buf = np.empty(self._capacity, dtype=self._dtype)
        self._nulls = np.ones(self._capacity, dtype=bool)
        self._n = 0
        return arr'''

    new = '''    def finish(self) -> pa.Array:
        import numpy as np
        values = self._buf[: self._n].copy()
        # Arrow validity bitmap: bit=1 means VALID, bit=0 means NULL.
        # _nulls[i] = True means NULL (inverted vs Arrow).
        # So packed = packbits(~_nulls).
        # Regression: v0.7.6 diagnostic showed packbits(mask) produced
        # wrong null positions (null↔valid inversion). Fix: packbits(~mask).
        mask = self._nulls[: self._n]
        if mask.any():
            packed = np.packbits(~mask, bitorder="little")
            arr = pa.Array.from_buffers(
                self._arrow_type, self._n,
                [pa.py_buffer(packed), pa.py_buffer(values)],
            )
        else:
            arr = pa.Array.from_buffers(
                self._arrow_type, self._n,
                [None, pa.py_buffer(values)],
            )
        # Reset
        self._capacity = 64
        self._buf = np.empty(self._capacity, dtype=self._dtype)
        self._nulls = np.ones(self._capacity, dtype=bool)
        self._n = 0
        return arr'''

    s = replace_exact(s, old, new, tag="b2-finish")
    p.write_text(s)
    compile_check(p)
    print("  OK: packbits(~mask)")


# ============ B2 regression test (direct, not through sort) ============

def add_b2_regression_test():
    print("\n=== B2 regression: test_numpy_backed_builder_null_bitmap_direct ===")
    p = Path("tests/test_sort.py")
    s = p.read_text()

    if "test_numpy_backed_builder_null_bitmap_direct" in s:
        print("  already present — skip")
        return

    extra = '''

def test_numpy_backed_builder_null_bitmap_direct():
    """_NumpyBackedBuilder: Arrow bitmap 1=valid (not 1=null).

    Регрессия v0.7.6.1 (B2): до фикса `packbits(mask=True=null)` давал
    1 в null-позициях, Arrow читал 1=valid → null↔valid инвертированы.
    Диагностика: packbits([F,F,T]) → pa.Array.from_buffers → [None,None,30]
    (2 nulls, неверно). После фикса: [10,20,None] (1 null).

    Тест прямой (не через external_sort) — чтобы не маскировалось
    native builder'ом, если он есть в текущем pyarrow.
    """
    from cadw.ops.sort import _NumpyBackedBuilder
    import numpy as np

    b = _NumpyBackedBuilder(pa.int64(), np.int64)
    b.append(10)
    b.append(20)
    b.append_null()
    arr = b.finish()

    assert arr.to_pylist() == [10, 20, None], (
        f"null position wrong: {arr.to_pylist()} (expected [10, 20, None])"
    )
    assert arr.null_count == 1, f"null_count={arr.null_count}, expected 1"


def test_numpy_backed_builder_null_first():
    """Bitmap: null at position 0, then valid."""
    from cadw.ops.sort import _NumpyBackedBuilder
    import numpy as np

    b = _NumpyBackedBuilder(pa.float64(), np.float64)
    b.append_null()
    b.append(1.5)
    b.append(2.5)
    arr = b.finish()

    assert arr.to_pylist() == [None, 1.5, 2.5]
    assert arr.null_count == 1


def test_numpy_backed_builder_all_valid():
    """No nulls — buffer path (no bitmap)."""
    from cadw.ops.sort import _NumpyBackedBuilder
    import numpy as np

    b = _NumpyBackedBuilder(pa.int64(), np.int64)
    b.append(1)
    b.append(2)
    arr = b.finish()

    assert arr.to_pylist() == [1, 2]
    assert arr.null_count == 0
'''
    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: 3 regression tests added")


# ============ B1: fit_multinomial_nb X >= 0 ============

def patch_b1_multinomial_nonneg():
    print("\n=== B1: fit_multinomial_nb — validate X >= 0 ===")
    p = Path("cadw/ml/naive_bayes.py")
    s = p.read_text()

    # Add helper before fit entry points
    if "_check_X_non_negative" not in s:
        anchor = "def _validate_common(df, X_cols, y_col):"
        if anchor not in s:
            raise SystemExit("_validate_common not found")

        helper = '''def _check_X_non_negative(df, X_cols) -> None:
    """Multinomial NB requires X >= 0 (counts). 1-pass, cheap.

    Raises ValueError on first negative found. Silent NaN otherwise
    (feature_sum + alpha may go negative → log → nan).
    """
    import pyarrow.compute as pc
    for b in df._iter_filtered_batches():
        for c in X_cols:
            col = b.column(c)
            min_val = pc.min(col, skip_nulls=True).as_py()
            if min_val is not None and min_val < 0:
                raise ValueError(
                    f"fit_multinomial_nb: X column {c!r} has negative "
                    f"value {min_val}; counts must be >= 0"
                )


'''
        s = s.replace(anchor, helper + anchor, 1)
        print("  OK: _check_X_non_negative helper added")

    # Call it inside fit_multinomial_nb
    old = '''    X_cols = _validate_common(df, X_cols, y_col)
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0; got {alpha}")
    mg = memory_guard or df._memory_guard
    op = _MultinomialStatsOp(X_cols, y_col)'''

    new = '''    X_cols = _validate_common(df, X_cols, y_col)
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0; got {alpha}")
    mg = memory_guard or df._memory_guard
    _check_X_non_negative(df, X_cols)
    op = _MultinomialStatsOp(X_cols, y_col)'''

    s = replace_exact(s, old, new, tag="b1-call")
    p.write_text(s)
    compile_check(p)
    print("  OK: X>=0 validation added before op")


# ============ B1 test ============

def add_b1_test():
    print("\n=== B1 test: test_multinomial_nb_negative_X_raises ===")
    p = Path("tests/test_naive_bayes.py")
    s = p.read_text()

    if "test_multinomial_nb_negative_X_raises" in s:
        print("  already present — skip")
        return

    extra = '''

def test_multinomial_nb_negative_X_raises(tmp_path):
    """Multinomial NB: X < 0 → ValueError (counts must be >= 0).

    Без валидации: feature_sum + alpha может уйти в отрицательное
    → log(negative) = nan → молчаливые nan предсказания.
    """
    data = {"x0": [-1.0, 2.0, 3.0], "x1": [0.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0]}
    df = make_df(tmp_path, data)
    with pytest.raises(ValueError, match="must be >= 0"):
        fit_multinomial_nb(df, ["x0", "x1"], "y")


def test_multinomial_nb_non_negative_X_passes(tmp_path):
    """All-zero X — валидно (sparse counts)."""
    data = {"x0": [0.0, 0.0, 0.0], "x1": [0.0, 0.0, 0.0], "y": [0.0, 0.0, 1.0]}
    df = make_df(tmp_path, data)
    model = fit_multinomial_nb(df, ["x0", "x1"], "y")
    assert model is not None
'''
    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: 2 B1 tests added")


# ============ I5: _validate_binary_y returns (y_sum, n_valid) ============

def patch_i5_single_scan():
    print("\n=== I5: _validate_binary_y → (y_sum, n_valid); _fit_boosting reuse ===")
    p = Path("cadw/ml/gbdt.py")
    s = p.read_text()

    # 1. Change signature + body of _validate_binary_y
    old_validate = '''def _validate_binary_y(df, y_col: str) -> None:
    """Ensure y has both classes {0, 1}."""
    seen: set = set()
    for b in df._iter_filtered_batches():
        col = b.column(y_col)
        try:
            uniq = col.unique()
            vals = uniq.to_pylist() if uniq is not None else col.to_pylist()
        except Exception:
            vals = col.to_pylist()
        for v in vals:
            if v is None:
                continue
            fv = float(v)
            if fv != 0.0 and fv != 1.0:
                raise ValueError(
                    f"fit_gbdt_classifier: y must be binary (0/1); got {v!r}"
                )
            seen.add(fv)
    if len(seen) < 2:
        raise ValueError(
            f"fit_gbdt_classifier: y must have both classes 0 and 1; got {seen}"
        )'''

    new_validate = '''def _validate_binary_y(df, y_col: str) -> tuple[float, int]:
    """Ensure y has both classes {0, 1}. Returns (y_sum, n_valid).

    v0.7.6.1 (I5): returns y_sum and n_valid so _fit_boosting can skip
    a second full scan. Single pass over df instead of two.
    """
    seen: set = set()
    y_sum = 0.0
    n_valid = 0
    for b in df._iter_filtered_batches():
        col = b.column(y_col)
        try:
            uniq = col.unique()
            vals = uniq.to_pylist() if uniq is not None else col.to_pylist()
        except Exception:
            vals = col.to_pylist()
        for v in vals:
            if v is None:
                continue
            fv = float(v)
            if fv != 0.0 and fv != 1.0:
                raise ValueError(
                    f"fit_gbdt_classifier: y must be binary (0/1); got {v!r}"
                )
            seen.add(fv)
        # Accumulate y_sum / n_valid in same pass
        import numpy as np
        y_np = _column_to_f64(col)
        mask = ~np.isnan(y_np)
        y_sum += float(y_np[mask].sum())
        n_valid += int(mask.sum())
    if len(seen) < 2:
        raise ValueError(
            f"fit_gbdt_classifier: y must have both classes 0 and 1; got {seen}"
        )
    return y_sum, n_valid'''

    s = replace_exact(s, old_validate, new_validate, tag="i5-validate")

    # 2. In _fit_boosting: use returned (y_sum, n_valid) instead of re-scanning
    old_scan = '''    if loss == "logloss_binary":
        _validate_binary_y(df, y_col)

    mg = memory_guard or df._memory_guard'''

    new_scan = '''    precomputed_y_sum: Optional[float] = None
    precomputed_n_valid: Optional[int] = None
    if loss == "logloss_binary":
        precomputed_y_sum, precomputed_n_valid = _validate_binary_y(df, y_col)

    mg = memory_guard or df._memory_guard'''

    s = replace_exact(s, old_scan, new_scan, tag="i5-prescan")

    # 3. In the y_sum computation block: skip re-scan if precomputed
    old_compute = '''    # 2. Base prediction (loss-specific) — needs y_sum and n_valid
    y_sum = 0.0
    n_valid = 0
    offset = 0
    for b in df._iter_filtered_batches():
        n = b.num_rows
        X = _stack_numeric(b, X_cols)
        y = _column_to_f64(b.column(y_col))
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        y_v = y[valid]
        y_sum += float(y_v.sum())
        n_valid += int(valid.sum())
        offset += n
    if n_valid == 0:
        raise ValueError("fit_gbdt: no valid rows")
    base_pred = _base_prediction(y_sum, n_valid, loss)'''

    new_compute = '''    # 2. Base prediction (loss-specific) — needs y_sum and n_valid.
    # v0.7.6.1 (I5): reuse precomputed values from _validate_binary_y
    # for logloss_binary (skip 2nd scan). For l2 — scan once here.
    if precomputed_y_sum is not None and precomputed_n_valid is not None:
        y_sum = precomputed_y_sum
        n_valid = precomputed_n_valid
    else:
        y_sum = 0.0
        n_valid = 0
        for b in df._iter_filtered_batches():
            X = _stack_numeric(b, X_cols)
            y = _column_to_f64(b.column(y_col))
            valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            y_v = y[valid]
            y_sum += float(y_v.sum())
            n_valid += int(valid.sum())
    if n_valid == 0:
        raise ValueError("fit_gbdt: no valid rows")
    base_pred = _base_prediction(y_sum, n_valid, loss)'''

    s = replace_exact(s, old_compute, new_compute, tag="i5-reuse")
    p.write_text(s)
    compile_check(p)
    print("  OK: single scan for binary y")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v076-quality"

    print("=== v0.7.6.1: quality fixes ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v076-quality-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v076-quality-{ts_str}",
        "-m", "before patch v0.7.6.1"])

    patch_b2_bitmap()
    add_b2_regression_test()

    patch_b1_multinomial_nonneg()
    add_b1_test()

    patch_i5_single_scan()

    print("\n=== pytest tests/test_sort.py -v (B2 regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_sort.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** B2 REGRESSION FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.6.1 (B2 test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_naive_bayes.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_naive_bayes.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** B1 TEST FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.6.1 (B1 test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_gbdt_classification.py -v (I5 regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_gbdt_classification.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** I5 REGRESSION FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.6.1 (I5 test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.6.1: quality fixes — "
        "B2 _NumpyBackedBuilder null bitmap (packbits(~mask)); "
        "B1 multinomial X>=0 validation; "
        "I5 _validate_binary_y returns (y_sum, n_valid) — single scan"])

    print("\n==> v0.7.6.1 applied on", branch)


if __name__ == "__main__":
    main()