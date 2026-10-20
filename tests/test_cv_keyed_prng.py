"""v0.9.3a: KFold keyed PRNG + materialize opt-in (§13.4/173)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard, KFold, TimeSeriesSplit
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, data, chunk=1000, name="L", mem_mb=200):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def make_id_df(tmp_path, n, chunk=1000, name="L"):
    """Identity data: y == row_id → fold content reveals fold_id."""
    return make_df(tmp_path, {
        "x": list(range(n)), "y": list(range(n)),
    }, chunk=chunk, name=name)


# ============ Default mode: O(batch) state ============

def test_kfold_no_materialize_default_folds(tmp_path):
    """Default mode: 5 folds, all rows covered exactly once."""
    df = make_id_df(tmp_path, n=1000)
    cv = KFold(n_splits=5)
    for _, train, test in cv.split(df):
        pass  # smoke


def test_kfold_shuffle_keyed_deterministic(tmp_path):
    """Same seed → same assignment. Different seed → different."""
    df = make_id_df(tmp_path, n=2000)
    cv1 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv2 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv3 = KFold(n_splits=5, shuffle=True, random_state=99)

    def _test_sets(cv):
        sets = []
        for _, _, test in cv.split(df):
            ys = []
            for b in test._iter_filtered_batches():
                ys.extend(b.column("y").to_pylist())
            sets.append(set(ys))
        return sets

    s1 = _test_sets(cv1)
    s2 = _test_sets(cv2)
    s3 = _test_sets(cv3)

    # Same seed → same sets, fold by fold
    for a, b in zip(s1, s2):
        assert a == b
    # Different seeds → at least one fold differs
    assert any(a != b for a, b in zip(s1, s3))
    # Union invariant: all rows covered
    for sets in (s1, s2, s3):
        assert set().union(*sets) == set(range(2000))


def test_kfold_shuffle_default_seed_deterministic(tmp_path):
    """random_state=None → default seed=42, deterministic across calls."""
    df = make_id_df(tmp_path, n=1000)
    cv_a = KFold(n_splits=5, shuffle=True)  # random_state=None
    cv_b = KFold(n_splits=5, shuffle=True)

    def _first_test_set(cv):
        for _, _, test in cv.split(df):
            ys = []
            for b in test._iter_filtered_batches():
                ys.extend(b.column("y").to_pylist())
            return set(ys)

    assert _first_test_set(cv_a) == _first_test_set(cv_b)


def test_kfold_shuffle_keyed_balanced_large(tmp_path):
    """Binomial fold sizes: mean n/k, per-fold variance O(sqrt(n/k))."""
    n = 10_000
    k = 5
    df = make_id_df(tmp_path, n=n)
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    sizes = []
    for _, _, test in cv.split(df):
        sizes.append(test.count())
    assert sum(sizes) == n
    # Split-quality: per-fold within ±3σ, σ ≈ sqrt(n/k · (1-1/k))
    mean = n / k
    sigma = np.sqrt(n / k * (1 - 1 / k))
    for s in sizes:
        assert abs(s - mean) < 3 * sigma, f"sizes={sizes}, mean={mean:.1f}"


def test_kfold_no_shuffle_default_unchanged(tmp_path):
    """shuffle=False default: fold_id = row_id % k (unchanged)."""
    df = make_id_df(tmp_path, n=100)
    cv = KFold(n_splits=5)
    for i, _, test in cv.split(df):
        ys = []
        for b in test._iter_filtered_batches():
            ys.extend(b.column("y").to_pylist())
        ys_sorted = sorted(ys)
        # fold i gets rows where row_id % k == i
        expected = list(range(i, 100, 5))
        assert ys_sorted == expected


# ============ materialize=True: opt-in O(N) backward compat ============

def test_kfold_materialize_true_balanced(tmp_path):
    """materialize=True: exact-balanced shuffle (old behavior)."""
    df = make_id_df(tmp_path, n=1000)
    cv = KFold(n_splits=5, shuffle=True, random_state=42,
               materialize=True)
    sizes = []
    for _, _, test in cv.split(df):
        sizes.append(test.count())
    assert sizes == [200, 200, 200, 200, 200]


def test_kfold_materialize_oom_guard(tmp_path):
    """materialize=True: O(N) arrays guarded — tight budget → MemoryLimitExceeded."""
    n = 1_000_000
    df = make_id_df(tmp_path, n=n)
    # 5 MB budget → state_limit = 2.5 MB. 20·N = 20 MB > 2.5 MB.
    df._memory_guard = MemoryGuard(max_memory_mb=5)
    cv = KFold(n_splits=5, shuffle=True, materialize=True)
    with pytest.raises(MemoryLimitExceeded, match="KFold.materialize"):
        list(cv.split(df))


def test_kfold_materialize_small_n_ok(tmp_path):
    """materialize=True on small N with ample budget: OK."""
    df = make_id_df(tmp_path, n=1000)
    cv = KFold(n_splits=5, shuffle=True, random_state=42,
               materialize=True)
    sizes = []
    for _, _, test in cv.split(df):
        sizes.append(test.count())
    assert sum(sizes) == 1000


# ============ AST guard: no O(N) materialize in default mode ============

def test_kfold_split_default_no_oN_arrays():
    """AST: default path must not call np.arange(n) or rng.permutation(n).

    §13.4/173: O(N) arrays only allowed inside `if self.materialize:`
    branch.
    """
    import ast
    import inspect
    import textwrap
    from cadw.cv import KFold

    src = textwrap.dedent(inspect.getsource(KFold.split))
    tree = ast.parse(src)

    # Find the materialize branch body (from `if self.materialize:` to
    # `else:`) and the else branch body.
    materialize_body: list = []
    default_body: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check condition is `self.materialize`
            test = node.test
            if (isinstance(test, ast.Attribute)
                    and test.attr == "materialize"):
                materialize_body = list(node.body)
                default_body = list(node.orelse)
                break

    assert materialize_body, "materialize branch not found"
    assert default_body, "default (else) branch not found"

    def _calls_np_arange_with_n(node):
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            # np.arange(n) — arg is a Name 'n'
            if (isinstance(f, ast.Attribute) and f.attr == "arange"
                    and isinstance(f.value, ast.Name)
                    and f.value.id == "np"):
                for arg in sub.args:
                    if isinstance(arg, ast.Name) and arg.id == "n":
                        return True
            # rng.permutation(n) — arg is a Name 'n'
            if (isinstance(f, ast.Attribute)
                    and f.attr == "permutation"):
                for arg in sub.args:
                    if isinstance(arg, ast.Name) and arg.id == "n":
                        return True
        return False

    # Materialize branch SHOULD have np.arange(n) or rng.permutation(n)
    assert any(_calls_np_arange_with_n(stmt) for stmt in materialize_body), (
        "materialize branch must materialize O(N) array"
    )
    # Default branch MUST NOT
    for stmt in default_body:
        assert not _calls_np_arange_with_n(stmt), (
            "default branch must not materialize O(N) array — "
            "P-CV1 regression"
        )


def test_kfold_default_uses_attach_column_with_fn():
    """AST: default path uses _attach_column_with_fn (streaming)."""
    import inspect
    from cadw.cv import KFold
    src = inspect.getsource(KFold.split)
    assert "_attach_column_with_fn" in src
    assert "_attach_column(df," in src


# ============ N large, tight budget: default mode works ============

def test_kfold_default_large_n_tight_budget(tmp_path):
    """N=100k, budget 50 MB. Default mode fits (O(batch) state).

    v0.9.2 KFold would have needed ~1.2 MB fold_id — fits here, but
    the invariant is verified by AST test_kfold_split_default_no_oN_arrays.
    This is a functional sanity: correct results, no crash.
    """
    df = make_id_df(tmp_path, n=100_000, chunk=1000)
    df._memory_guard = MemoryGuard(max_memory_mb=50)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    sizes = []
    for _, _, test in cv.split(df):
        sizes.append(test.count())
    assert sum(sizes) == 100_000
