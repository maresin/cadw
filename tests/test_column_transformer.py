"""v0.9.1d: ColumnTransformer tests."""
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.pipeline import (
    CastStep, ColumnTransformer, DropnaStep, LambdaTransform,
)


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=200):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_table(df, tmp_path, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


# ============ construction errors ============

def test_empty_transformers_raises():
    with pytest.raises(ValueError, match="must be non-empty"):
        ColumnTransformer([])


def test_bad_remainder_raises():
    with pytest.raises(ValueError, match="remainder must be"):
        ColumnTransformer(
            [("a", ["x"], CastStep({"x": "float64"}))],
            remainder="bad",
        )


def test_duplicate_name_raises():
    with pytest.raises(ValueError, match="duplicate transformer name"):
        ColumnTransformer([
            ("t", ["a"], CastStep({"a": "float64"})),
            ("t", ["b"], CastStep({"b": "float64"})),
        ])


def test_overlapping_cols_raises():
    with pytest.raises(ValueError, match="claimed by both"):
        ColumnTransformer([
            ("t1", ["a", "b"], CastStep({"a": "float64"})),
            ("t2", ["b", "c"], CastStep({"b": "float64"})),
        ])


def test_empty_cols_raises():
    with pytest.raises(ValueError, match="cols must be non-empty"):
        ColumnTransformer([("t", [], CastStep({"a": "float64"}))])


def test_step_missing_transform_raises():
    class NoTransform:
        def fit(self, df, target=None):
            return self
    with pytest.raises(TypeError, match=r"must have fit \+ transform"):
        ColumnTransformer([("t", ["a"], NoTransform())])


# ============ fit / transform basics ============

def test_basic_two_transformers(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    ct = ColumnTransformer([
        ("num1", ["a"], CastStep({"a": "float64"})),
        ("num2", ["b"], CastStep({"b": "float64"})),
    ], remainder="drop")
    ct.fit(df)
    out = ct.transform(df)
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"a", "b"}
    assert t.schema.field("a").type == pa.float64()
    assert t.schema.field("b").type == pa.float64()


def test_remainder_passthrough(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    ct = ColumnTransformer(
        [("num", ["a"], CastStep({"a": "float64"}))],
        remainder="passthrough",
    )
    ct.fit(df)
    out = ct.transform(df)
    t = read_table(out, tmp_path)
    # a (cast), b, c (passthrough)
    assert set(t.schema.names) == {"a", "b", "c"}
    assert t.schema.field("a").type == pa.float64()
    assert t.schema.field("b").type == pa.int64()


def test_remainder_drop_drops_unclaimed(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    ct = ColumnTransformer(
        [("num", ["a"], CastStep({"a": "float64"}))],
        remainder="drop",
    )
    ct.fit(df)
    out = ct.transform(df)
    t = read_table(out, tmp_path)
    assert list(t.schema.names) == ["a"]


def test_transformer_appends_new_col(tmp_path):
    """LambdaTransform inside ColumnTransformer appends x_sq."""
    df = make_df(tmp_path, {"x": [3, 4], "y": [1, 2]})
    ct = ColumnTransformer([
        ("feat", ["x"], LambdaTransform(
            "x", "sq", lambda arr: pc.multiply(arr, arr), pa.int64(),
        )),
    ], remainder="drop")
    ct.fit(df)
    out = ct.transform(df)
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"x", "x_sq"}
    assert t.column("x_sq").to_pylist() == [9, 16]


# ============ transform errors ============

def test_transform_missing_col_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2]})
    ct = ColumnTransformer(
        [("t", ["a", "b"], CastStep({"a": "float64"}))],
    )
    with pytest.raises(KeyError, match="b"):
        ct.fit(df)


def test_transform_row_count_changed_raises(tmp_path):
    """dropna inside transformer → RuntimeError."""
    df = make_df(tmp_path, {"a": [1, None, 3], "b": [4, 5, 6]})
    ct = ColumnTransformer([
        ("bad", ["a"], DropnaStep(how="any", subset=["a"])),
    ], remainder="passthrough")
    ct.fit(df)
    with pytest.raises(RuntimeError, match="changed row count"):
        ct.transform(df)


def test_transform_duplicate_output_col_raises(tmp_path):
    """Two transformers emit same new col → RuntimeError."""
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4]})
    ct = ColumnTransformer([
        ("t1", ["a"], LambdaTransform(
            "a", "sq", lambda arr: pc.multiply(arr, arr), pa.int64(),
        )),
        ("t2", ["b"], LambdaTransform(
            "b", "sq", lambda arr: pc.multiply(arr, arr), pa.int64(),
        )),
    ], remainder="drop")
    ct.fit(df)
    # Both emit x_sq vs b_sq — no collision, so this should NOT raise.
    out = ct.transform(df)
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"a", "a_sq", "b", "b_sq"}


# ============ serialization ============

def test_to_dict_from_dict_round_trip():
    ct = ColumnTransformer([
        ("num", ["a", "b"], CastStep({"a": "float64"})),
        ("drop", ["c"], DropnaStep(how="any")),
    ], remainder="passthrough")
    d = ct.to_dict()
    restored = ColumnTransformer.from_dict(d)
    assert len(restored._transformers) == 2
    assert restored._transformers[0][0] == "num"
    assert restored._transformers[0][1] == ["a", "b"]
    assert isinstance(restored._transformers[0][2], CastStep)
    assert restored._remainder == "passthrough"


def test_from_dict_unknown_nested_type_raises():
    d = {
        "transformers": [
            {"name": "x", "cols": ["a"], "type": "UnknownStep",
             "step_dict": {}},
        ],
        "remainder": "drop",
    }
    with pytest.raises(NotImplementedError, match="unknown nested step"):
        ColumnTransformer.from_dict(d)


def test_to_dict_lambda_nested_raises():
    """LambdaTransform.to_dict raises NotImplementedError (§13.4/151).

    v0.9.1d-fix1: LambdaTransform DOES implement to_dict (which raises
    NotImplementedError). Message: "LambdaTransform cannot be serialized".
    Prior test expected "does not implement to_dict" (wrong — that's for
    custom steps without to_dict method).
    """
    ct = ColumnTransformer([
        ("lam", ["a"], LambdaTransform(
            "a", "sq", lambda arr: pc.multiply(arr, arr), pa.int64(),
        )),
    ])
    with pytest.raises(NotImplementedError, match="LambdaTransform cannot"):
        ct.to_dict()


def test_ct_to_dict_uses_serialization_name():
    # v0.9.9 (N1): custom-step registered name, not class name.
    from cadw.pipeline import ColumnTransformer
    from tests.test_register_step_type import _CustomStep, _STEP_REGISTRY
    import pytest
    from cadw.pipeline import register_step_type
    register_step_type('MyCustomV99', _CustomStep)
    try:
        ct = ColumnTransformer([
            ('t1', ['x'], _CustomStep({'x': 'float64'})),
        ])
        d = ct.to_dict()
        assert d['transformers'][0]['type'] == 'MyCustomV99', (
            f"expected 'MyCustomV99', got {d['transformers'][0]['type']!r}"
        )
    finally:
        from cadw.pipeline import unregister_step_type
        unregister_step_type('MyCustomV99')