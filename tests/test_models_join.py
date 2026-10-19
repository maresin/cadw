"""Тесты JoinMemoryModel (11-B-2c-1, §9.2.5)."""
import pyarrow as pa
import pytest

from cadw.core.models import DataChars, get_model, list_models
from cadw.core.models_join import JoinMemoryModel, JoinParams


def _chars(
    n_rows: int = 100_000,
    k_est: int = 1000,
    row_size: int = 32,
    cols: tuple = ("k", "v"),
) -> DataChars:
    schema = pa.schema([(c, pa.int64()) for c in cols])
    return DataChars(
        schema=schema,
        n_rows=n_rows,
        row_size_bytes=row_size,
        k_est=k_est,
        avg_key_size_bytes=8,
        size_of_key_entry_bytes=80,
    )


# ---------- registry ----------

def test_join_model_registered():
    m = get_model("join")
    assert isinstance(m, JoinMemoryModel)
    assert "join" in list_models()


# ---------- invariant: estimate(solve) <= budget ----------

@pytest.mark.parametrize(
    "avail_mb, how, left_k, right_k",
    [
        (10, "inner", 100, 100),
        (10, "inner", 10_000, 1_000),
        (10, "inner", 100_000, 10_000),
        (10, "left", 100, 100),
        (100, "inner", 1_000_000, 100_000),
        (100, "left", 1_000_000, 100_000),
    ],
)
def test_invariant(avail_mb, how, left_k, right_k):
    m = JoinMemoryModel()
    left = _chars(n_rows=left_k * 10, k_est=left_k)
    right = _chars(n_rows=right_k * 10, k_est=right_k)
    available = avail_mb * 1024 * 1024
    safety = 0.7
    params = m.solve(
        available, left, safety=safety,
        op_context={"right_chars": right, "how": how},
    )
    assert params is not None, (
        f"unexpected None: avail={avail_mb}MB how={how} "
        f"left_k={left_k} right_k={right_k}"
    )
    assert isinstance(params, JoinParams)
    assert params.how == how
    assert params.num_partitions >= 2
    est = m.estimate(params, left)
    assert est <= int(available * safety), (
        f"estimate {est} > safety {int(available*safety)} "
        f"for avail={avail_mb}MB how={how}"
    )


# ---------- build_side ----------

def test_build_side_right_for_left_join():
    m = JoinMemoryModel()
    left = _chars(n_rows=10_000, k_est=100)
    right = _chars(n_rows=10_000, k_est=100)
    params = m.solve(
        10 * 1024 * 1024, left,
        op_context={"right_chars": right, "how": "left"},
    )
    assert params is not None
    assert params.build_side == "right"


def test_build_side_smaller_for_inner():
    m = JoinMemoryModel()
    left_big = _chars(n_rows=1_000_000, k_est=1_000_000, row_size=32)
    right_small = _chars(n_rows=100, k_est=10, row_size=16)
    params = m.solve(
        10 * 1024 * 1024, left_big,
        op_context={"right_chars": right_small, "how": "inner"},
    )
    assert params is not None
    assert params.build_side == "right"  # right smaller


def test_build_side_left_when_left_smaller():
    m = JoinMemoryModel()
    left_small = _chars(n_rows=100, k_est=10, row_size=16)
    right_big = _chars(n_rows=1_000_000, k_est=1_000_000, row_size=32)
    params = m.solve(
        10 * 1024 * 1024, left_small,
        op_context={"right_chars": right_big, "how": "inner"},
    )
    assert params is not None
    assert params.build_side == "left"


# ---------- max_output_rows из бюджета ----------

def test_max_output_rows_from_budget():
    """max_output_rows × row_size_out × (1+c_merge) <= output_budget."""
    m = JoinMemoryModel()
    left = _chars(n_rows=100_000, k_est=1000)
    right = _chars(n_rows=100_000, k_est=1000)
    available = 10 * 1024 * 1024
    safety = 0.7
    params = m.solve(
        available, left, safety=safety,
        op_context={"right_chars": right, "how": "inner"},
    )
    assert params is not None
    output_budget = int(available * safety * m.output_ratio)
    worst = (
        params.max_output_rows
        * params.output_row_size
        * (1 + m.c_merge_ratio)
    )
    assert worst <= output_budget + 1000  # допуск на округление вниз


# ---------- None когда build не влезает ----------

def test_none_when_build_partition_too_big():
    """Огромный build (K × row_size / P) > build_budget → None."""
    m = JoinMemoryModel()
    # left огромен (P будет большой, но K/P всё равно большой)
    left = _chars(n_rows=10_000_000, k_est=10_000_000, row_size=200)
    right = _chars(n_rows=10_000_000, k_est=10_000_000, row_size=200)
    params = m.solve(
        1024, left,  # 1 KB — заведомо мало
        op_context={"right_chars": right, "how": "inner"},
    )
    assert params is None


# ---------- requires right_chars ----------

def test_solve_requires_right_chars():
    m = JoinMemoryModel()
    left = _chars()
    with pytest.raises(ValueError, match="right_chars"):
        m.solve(10 * 1024 * 1024, left, op_context=None)
    with pytest.raises(ValueError, match="right_chars"):
        m.solve(10 * 1024 * 1024, left, op_context={})


def test_solve_invalid_how():
    m = JoinMemoryModel()
    left = _chars()
    right = _chars()
    with pytest.raises(ValueError, match="how must be"):
        m.solve(
            10 * 1024 * 1024, left,
            op_context={"right_chars": right, "how": "outer"},
        )


# ---------- type check ----------

def test_estimate_type_check():
    m = JoinMemoryModel()
    left = _chars()
    with pytest.raises(TypeError, match="JoinParams"):
        m.estimate("not a params", left)


# ---------- describe ----------

def test_describe():
    m = JoinMemoryModel()
    d = m.describe()
    assert "JoinMemoryModel" in d
    assert "build_ratio" in d


# ---------- invalid ratios ----------

def test_invalid_ratios():
    with pytest.raises(ValueError, match="must be in"):
        JoinMemoryModel(build_ratio=0)
    with pytest.raises(ValueError, match="must be in"):
        JoinMemoryModel(probe_ratio=1.5)
    with pytest.raises(ValueError, match="<= 1.0"):
        JoinMemoryModel(
            build_ratio=0.5, probe_ratio=0.4, output_ratio=0.4,
        )
