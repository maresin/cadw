#!/usr/bin/env python3
"""v0.7.2-fix: inverse_transform duplicate columns + NaN→null в transform.

Баг 1: inverse_transform(df) на df, где уже есть x0..x2 (transform
сохраняет оригиналы) — duplicate KeyError 'exists 2 times'.
Fix: убрать feature columns из passthrough; reconstructed — на их место.

Баг 2: transform для null-row даёт NaN (valid float), не null.
Test ожидает None. Fix: pa.array(..., mask=isnan) — Arrow-null.
"""
from __future__ import annotations

import os
import subprocess
import sys
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


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


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


def patch_pca():
    print("\n=== cadw/ml/pca.py: transform + inverse_transform fixes ===")
    p = Path("cadw/ml/pca.py")
    s = p.read_text()

    # Fix 1: transform — NaN masked rows → null in Arrow arrays
    old_transform = '''        def _iter_transform():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, self.feature_names_)
                n = Xm.shape[0]
                # NaN row → all-NaN PCs
                proj = np.full((n, self.n_components_), np.nan, dtype=np.float64)
                mask = ~np.isnan(Xm).any(axis=1)
                if mask.any():
                    Xv = Xm[mask]
                    proj[mask] = (Xv - mean) @ comp.T
                arrays = [b.column(c) for c in X._columns]
                for i in range(self.n_components_):
                    arrays.append(pa.array(proj[:, i], type=pa.float64()))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)'''

    new_transform = '''        def _iter_transform():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, self.feature_names_)
                n = Xm.shape[0]
                # NaN row → all-NaN PCs (converted to Arrow null below)
                proj = np.full((n, self.n_components_), np.nan, dtype=np.float64)
                mask = ~np.isnan(Xm).any(axis=1)
                if mask.any():
                    Xv = Xm[mask]
                    proj[mask] = (Xv - mean) @ comp.T
                arrays = [b.column(c) for c in X._columns]
                for i in range(self.n_components_):
                    col_np = proj[:, i]
                    null_mask = np.isnan(col_np)
                    arr = pa.array(
                        col_np, type=pa.float64(),
                        mask=null_mask if null_mask.any() else None,
                    )
                    arrays.append(arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)'''

    s = replace_exact(s, old_transform, new_transform, tag="transform-nan-null")

    # Fix 2: inverse_transform — replace existing feature columns
    old_inv = '''        orig_schema = Z._dataset.schema
        out_fields = [orig_schema.field(c) for c in Z._columns]
        for name in self.feature_names_:
            out_fields.append(pa.field(name, pa.float64(), nullable=True))
        out_schema = pa.schema(out_fields)
        comp = self.components_
        mean = self.mean_

        def _iter_inv():
            for b in Z._iter_filtered_batches():
                Zm = np.column_stack(
                    [_column_to_f64(b.column(c)) for c in pc_cols]
                )
                n = Zm.shape[0]
                recon = np.full((n, self.n_features_in_), np.nan,
                                dtype=np.float64)
                mask = ~np.isnan(Zm).any(axis=1)
                if mask.any():
                    recon[mask] = Zm[mask] @ comp + mean
                arrays = [b.column(c) for c in Z._columns]
                for i in range(self.n_features_in_):
                    arrays.append(pa.array(recon[:, i], type=pa.float64()))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)'''

    new_inv = '''        orig_schema = Z._dataset.schema
        # v0.7.2-fix: replace feature columns (avoid duplicates if Z already
        # has x0..xd from a preceding transform).
        existing_features = set(self.feature_names_) & set(Z._columns)
        passthrough_cols = [
            c for c in Z._columns if c not in existing_features
        ]
        out_fields = [orig_schema.field(c) for c in passthrough_cols]
        for name in self.feature_names_:
            out_fields.append(pa.field(name, pa.float64(), nullable=True))
        out_schema = pa.schema(out_fields)
        comp = self.components_
        mean = self.mean_

        def _iter_inv():
            for b in Z._iter_filtered_batches():
                Zm = np.column_stack(
                    [_column_to_f64(b.column(c)) for c in pc_cols]
                )
                n = Zm.shape[0]
                recon = np.full((n, self.n_features_in_), np.nan,
                                dtype=np.float64)
                mask = ~np.isnan(Zm).any(axis=1)
                if mask.any():
                    recon[mask] = Zm[mask] @ comp + mean
                arrays = [b.column(c) for c in passthrough_cols]
                for i in range(self.n_features_in_):
                    col_np = recon[:, i]
                    null_mask = np.isnan(col_np)
                    arr = pa.array(
                        col_np, type=pa.float64(),
                        mask=null_mask if null_mask.any() else None,
                    )
                    arrays.append(arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)'''

    s = replace_exact(s, old_inv, new_inv, tag="inverse-transform-replace")

    p.write_text(s)
    compile_check(p)
    print("  OK: transform NaN→null; inverse_transform replaces")


def main():
    print("=== v0.7.2-fix ===")
    patch_pca()

    print("\n=== pytest tests/test_pca.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pca.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.2-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.2-fix: PCA transform NaN→null; inverse_transform replaces features"])

    print("\n==> v0.7.2 complete.")


if __name__ == "__main__":
    main()