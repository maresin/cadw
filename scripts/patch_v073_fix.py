#!/usr/bin/env python3
"""v0.7.3-fix: n_total = df.count() (filtered); +noise cleanup.

Баг: n_total = df._dataset.total_rows (unfiltered), но
_iter_filtered_batches даёт filtered rows. При filter x>0: pred=500,
cur_leaf=277 → broadcast fail.

Fix:
1. n_total = df.count() — filtered count. Free для unfiltered.
2. _AssignLeavesOp/_LeafValuesOp.estimate_state_nbytes() = 0
   (state покрыт n×32 guard в fit).
3. _find_best_split: np.errstate (divide/invalid) — warnings при λ=0.
4. _AssignLeavesOp: маска NaN X (invalid cast warning).
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


def patch_gbdt():
    print("\n=== cadw/ml/gbdt.py ===")
    p = Path("cadw/ml/gbdt.py")
    s = p.read_text()

    # 1. n_total = df.count() (filtered)
    old_guard = '''    mg = memory_guard or df._memory_guard

    # Guard: internal O(n) state (pred + cur_leaf + grad + hess ≈ 32B/row)
    n_total = df._dataset.total_rows
    est_n_state = n_total * 32
    try:
        mg.check_state(est_n_state, name="gbdt.internal_state")
    except MemoryLimitExceeded:
        raise MemoryLimitExceeded(
            f"fit_gbdt_regressor: internal state O(n) = {est_n_state} B "
            f"({n_total} rows × 32B) > available * state_safety. "
            f"GBDT requires materializing per-row gradients (like XGBoost). "
            f"Increase budget or sample. See §13.4/92."
        )'''
    new_guard = '''    mg = memory_guard or df._memory_guard

    # v0.7.3-fix: n_total MUST be filtered count, not total_rows.
    # pred/cur_leaf are sized to filtered rows, because _iter_filtered_batches
    # iterates filtered batches only.
    n_total = df.count()
    if n_total == 0:
        raise ValueError(
            "fit_gbdt_regressor: no valid (non-null) rows "
            "(filtered count is 0)"
        )
    est_n_state = n_total * 32
    try:
        mg.check_state(est_n_state, name="gbdt.internal_state")
    except MemoryLimitExceeded:
        raise MemoryLimitExceeded(
            f"fit_gbdt_regressor: internal state O(n) = {est_n_state} B "
            f"({n_total} rows × 32B) > available * state_safety. "
            f"GBDT requires materializing per-row gradients (like XGBoost). "
            f"Increase budget or sample. See §13.4/92."
        )'''
    s = replace_exact(s, old_guard, new_guard, tag="n-total")

    # 2a. _AssignLeavesOp: mask NaN X, add estimate_state_nbytes
    old_assign = '''    def update(self, state, batch):
        X = _stack_numeric(batch, self.feature_cols)
        n = X.shape[0]
        leaf = np.full(n, self.root_id, dtype=np.int64)
        # Traverse depth-by-depth. tree is a flat dict; internal nodes have
        # children with higher node_ids (children = 2*parent+1, 2*parent+2).
        max_iter = 32
        for _ in range(max_iter):
            any_change = False
            for nid, spec in self.tree.items():
                if spec is None:
                    continue
                j, thr, left, right = spec
                m = leaf == nid
                if not m.any():
                    continue
                b_idx = np.clip(
                    ((X[m, j] - self.mn[j]) / self.range[j] * self.bins)
                    .astype(np.int64),
                    0, self.bins - 1,
                )
                new_ids = np.where(b_idx <= thr, left, right)
                leaf[m] = new_ids
                any_change = True
            if not any_change:
                break
        state.append(leaf)
        return state

    def merge(self, a, b):
        return a + b

    def state_nbytes(self, state):
        return int(sum(x.nbytes for x in state) + 8)

    def finalize(self, state):
        return np.concatenate(state) if state else np.array([], dtype=np.int64)'''

    new_assign = '''    def update(self, state, batch):
        X = _stack_numeric(batch, self.feature_cols)
        n = X.shape[0]
        leaf = np.full(n, self.root_id, dtype=np.int64)
        # v0.7.3-fix: rows with NaN feature stay at root (never descend).
        nan_rows = np.isnan(X).any(axis=1)
        # Traverse depth-by-depth.
        max_iter = 32
        with np.errstate(divide="ignore", invalid="ignore"):
            for _ in range(max_iter):
                any_change = False
                for nid, spec in self.tree.items():
                    if spec is None:
                        continue
                    j, thr, left, right = spec
                    m = (leaf == nid) & ~nan_rows
                    if not m.any():
                        continue
                    b_idx = np.clip(
                        ((X[m, j] - self.mn[j]) / self.range[j] * self.bins)
                        .astype(np.int64),
                        0, self.bins - 1,
                    )
                    new_ids = np.where(b_idx <= thr, left, right)
                    leaf[m] = new_ids
                    any_change = True
                if not any_change:
                    break
        state.append(leaf)
        return state

    def merge(self, a, b):
        return a + b

    def state_nbytes(self, state):
        return int(sum(x.nbytes for x in state) + 8)

    def estimate_state_nbytes(self):
        # State is per-batch leaves; total bounded by n_total × 8 bytes,
        # already covered by the n×32 pre-check in fit_gbdt_regressor.
        return 0

    def finalize(self, state):
        return np.concatenate(state) if state else np.array([], dtype=np.int64)'''

    s = replace_exact(s, old_assign, new_assign, tag="assign-leaves")

    # 2b. _LeafValuesOp: add estimate_state_nbytes
    old_leaf_op = '''    def state_nbytes(self, state):
        return 16 * max(1, len(state))

    def finalize(self, state):
        return {
            k: -g / (h + self.reg_lambda)
            for k, (g, h) in state.items()
        }'''
    new_leaf_op = '''    def state_nbytes(self, state):
        return 16 * max(1, len(state))

    def estimate_state_nbytes(self):
        # O(leaves) bounded by 2^depth; covered by n×32 pre-check.
        return 0

    def finalize(self, state):
        return {
            k: -g / (h + self.reg_lambda)
            for k, (g, h) in state.items()
        }'''
    s = replace_exact(s, old_leaf_op, new_leaf_op, tag="leaf-values-op")

    # 3. _find_best_split: errstate
    old_find = '''        # min_child_weight: hess in each child
        valid = (h_l >= min_child_weight) & (h_r >= min_child_weight)
        if not valid.any():
            continue
        gains = np.where(
            valid,
            g_l * g_l / (h_l + reg_lambda)
            + g_r * g_r / (h_r + reg_lambda)
            - parent_gain,
            -np.inf,
        )
        k = int(np.argmax(gains))'''
    new_find = '''        # min_child_weight: hess in each child
        valid = (h_l >= min_child_weight) & (h_r >= min_child_weight)
        if not valid.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            g_l_sq = g_l * g_l / (h_l + reg_lambda)
            g_r_sq = g_r * g_r / (h_r + reg_lambda)
        gains = np.where(
            valid,
            g_l_sq + g_r_sq - parent_gain,
            -np.inf,
        )
        k = int(np.argmax(gains))'''
    s = replace_exact(s, old_find, new_find, tag="find-split-errstate")

    p.write_text(s)
    compile_check(p)
    print("  OK: gbdt.py — n_total=count(); +estimate_state_nbytes; errstate")


def main():
    print("=== v0.7.3-fix ===")
    patch_gbdt()

    print("\n=== pytest tests/test_gbdt_regression.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_gbdt_regression.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.3-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.3-fix: n_total=df.count() (filtered); +estimate_state_nbytes; "
        "errstate in _find_best_split; NaN-mask in _AssignLeavesOp"])

    print("\n==> v0.7.3 complete.")


if __name__ == "__main__":
    main()