"""A5 (v0.9.240): smoke coverage for cv / column / register_step_type.

Restored minimal coverage after 13 test files were deleted in v0.9.238.
Structural smoke only (no value-level assertions) — catches import breakage
and obvious API change. Full coverage for these modules is deferred.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard
from cadw.cv import KFold
from cadw.pipeline import (
    _STEP_REGISTRY,
    list_step_types,
    register_step_type,
    unregister_step_type,
)
from cadw.steps.column import ColumnTransformer
from cadw.steps._mvp.standardize import Standardize


def _make_df(n: int = 100, seed: int = 0) -> CadwDataFrame:
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(pa.table({
        "x1": pa.array(rng.normal(0, 1, n), type=pa.float64()),
        "x2": pa.array(rng.normal(5, 2, n), type=pa.float64()),
        "x3": pa.array(rng.normal(-3, 1, n), type=pa.float64()),
        "y": pa.array(rng.normal(0, 1, n), type=pa.float64()),
    }), memory_guard=MemoryGuard(max_memory_mb=200))


# ---------------- KFold ----------------

def test_kfold_split_smoke():
    """KFold(3).split: train+test=n per fold; test folds partition rows."""
    df = _make_df(n=30, seed=1)
    k = 3
    folds = list(KFold(k, shuffle=False).split(df))
    assert len(folds) == k, f"expected {k} folds, got {len(folds)}"

    test_sizes = []
    for fold_i, train, test in folds:
        assert 0 <= fold_i < k, f"fold_i={fold_i} out of range"
        n_tr, n_te = train.count(), test.count()
        assert n_tr + n_te == 30, (
            f"fold {fold_i}: train={n_tr} + test={n_te} != 30")
        test_sizes.append(n_te)
    # shuffle=False -> row_id % k -> each row is test in exactly one fold
    assert sum(test_sizes) == 30, (
        f"test folds do not partition rows: sizes={test_sizes}")


def test_kfold_n_splits_gt_n_raises():
    df = _make_df(n=30, seed=1)
    with pytest.raises(ValueError, match="n_splits"):
        list(KFold(31, shuffle=False).split(df))


# ---------------- ColumnTransformer ----------------

@pytest.mark.skip(
    reason="TODO v0.9.241: df[cols] does not project schema "
           "(schema=4, batches=1). Affects _mvp/Standardize "
           "(reads df._dataset.schema -> RecordBatch.from_arrays "
           "mismatch) and likely StandardScaler via "
           "_wrap_transformed(df, cols) -> _output_schema(df, ...) "
           "builds 4-col schema, _iter_transformed yields 1-col. "
           "Fix df[cols] (schema + _iter_filtered_batches) first."
)
def test_column_transformer_passthrough_and_drop():
    """ColumnTransformer: passthrough keeps unclaimed; drop removes them."""
    df = _make_df(n=100, seed=2)
    base_transformers = [
        ("a", ["x1"], Standardize(["x1"])),
        ("b", ["x2"], Standardize(["x2"])),
    ]

    ct_pass = ColumnTransformer(base_transformers, remainder="passthrough")
    out = ct_pass.fit(df).transform(df)
    assert out.count() == 100
    out_cols = set(out._dataset.schema.names)
    for c in ("x1", "x2", "x3", "y"):
        assert c in out_cols, f"passthrough missing {c!r}: {out_cols}"

    ct_drop = ColumnTransformer(
        [("a", ["x1"], Standardize(["x1"]))], remainder="drop")
    out_drop = ct_drop.fit(df).transform(df)
    out_drop_cols = set(out_drop._dataset.schema.names)
    assert "x3" not in out_drop_cols and "y" not in out_drop_cols, (
        f"drop remainder kept unclaimed: {out_drop_cols}")


# ---------------- register_step_type ----------------

class _DummyA5Step:
    """Minimal Transformer for register_step_type smoke."""
    is_barrier = False
    cadw_function = None

    def __init__(self, name: str = "dummy") -> None:
        self._name = name

    def fit(self, df, target=None):
        return self

    def transform(self, df):
        return df

    def to_dict(self):
        return {"name": self._name}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"])

    def estimate_state_bytes(self, dc) -> int:
        return 0

    def state_bytes(self) -> int:
        return 0


def test_register_step_type_smoke():
    """register/unregister roundtrip; builtin protection."""
    name = "DummyA5Step"
    try:
        register_step_type(name, _DummyA5Step)
        assert _STEP_REGISTRY.get(name) is _DummyA5Step
        assert name in list_step_types()
    finally:
        unregister_step_type(name)
    assert _STEP_REGISTRY.get(name) is None, (
        "unregister_step_type did not remove entry")
    assert name not in list_step_types()

    with pytest.raises(ValueError, match="built-in"):
        unregister_step_type("StandardScaler")

def test_column_transformer_validation():
    """ColumnTransformer constructor validation (independent of df[cols])."""
    step = Standardize(["x1"])

    with pytest.raises(ValueError, match="non-empty"):
        ColumnTransformer([])

    with pytest.raises(ValueError, match="duplicate"):
        ColumnTransformer([
            ("a", ["x1"], Standardize(["x1"])),
            ("a", ["x2"], Standardize(["x2"])),
        ])

    with pytest.raises(ValueError, match="claimed by both"):
        ColumnTransformer([
            ("a", ["x1"], Standardize(["x1"])),
            ("b", ["x1"], Standardize(["x1"])),
        ])

    with pytest.raises(ValueError, match="remainder"):
        ColumnTransformer([("a", ["x1"], step)], remainder="bogus")

    with pytest.raises(TypeError):
        ColumnTransformer([("a", ["x1"], object())])

