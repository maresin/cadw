"""
Тесты HashShuffleMemoryModel, SortMemoryModel, GroupByMemoryModel (11-A-2).

Инвариант для каждой модели: estimate(solve(x)) <= x * safety.
Граничные случаи: None когда params не существует; композиция GroupBy.
"""
import pyarrow as pa
import pytest

from cadw.core.models import DataChars, get_model, list_models
from cadw.core.models_shuffle import HashShuffleMemoryModel, HashShuffleParams
from cadw.core.models_sort import SortMemoryModel, SortParams
from cadw.core.models_groupby import GroupByMemoryModel, GroupByParams


def _dc(
    row_size: int = 16,
    n_rows: int = 1_000_000,
    k_est: int = 1000,
    avg_key_size: int = 8,
    size_of_key_entry: int = 80,
) -> DataChars:
    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    return DataChars(
        schema=schema,
        n_rows=n_rows,
        row_size_bytes=row_size,
        k_est=k_est,
        avg_key_size_bytes=avg_key_size,
        size_of_key_entry_bytes=size_of_key_entry,
    )


# ---------- registry ----------

def test_all_models_registered():
    names = list_models()
    for n in ("reduce", "hash_shuffle", "sort", "groupby"):
        assert n in names, f"{n} not registered; got {names}"


# ---------- HashShuffleMemoryModel ----------

@pytest.mark.parametrize(
    "available_mb, k_est, row_size",
    [
        (10, 1000, 16),
        (10, 100_000, 16),
        (100, 1_000_000, 16),
        (10, 1_000_000, 16),
        (50, 10, 16),
        (100, 1000, 1000),
    ],
)
def test_shuffle_invariant(available_mb, k_est, row_size):
    m = HashShuffleMemoryModel()
    dc = _dc(row_size=row_size, k_est=k_est)
    available = available_mb * 1024 * 1024
    safety = 0.7
    params = m.solve(available, dc, safety=safety)
    assert params is not None, (
        f"unexpected None: available={available_mb}MB k_est={k_est}"
    )
    assert isinstance(params, HashShuffleParams)
    assert params.num_partitions >= 2
    # power of 2
    assert (params.num_partitions & (params.num_partitions - 1)) == 0
    assert params.batch_size >= 1
    est = m.estimate(params, dc)
    assert est <= int(available * safety), (
        f"estimate {est} > safety {int(available*safety)}"
    )


def test_shuffle_none_when_too_many_partitions():
    """P * writer_buffer_min > budget * 0.4 -> None."""
    m = HashShuffleMemoryModel(writer_buffer_min=64 * 1024)  # 64 KB
    dc = _dc(k_est=10**9, size_of_key_entry=80)
    # available очень мал, K_est огромен -> P огромно -> P*64KB > budget*0.4
    params = m.solve(64 * 1024, dc, safety=0.7)
    assert params is None


def test_shuffle_requires_k_est():
    m = HashShuffleMemoryModel()
    dc = _dc(k_est=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k_est"):
        m.solve(10 * 1024 * 1024, dc)


def test_shuffle_requires_size_of_key_entry():
    m = HashShuffleMemoryModel()
    dc = _dc(size_of_key_entry=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="size_of_key_entry"):
        m.solve(10 * 1024 * 1024, dc)


def test_shuffle_type_check():
    m = HashShuffleMemoryModel()
    dc = _dc()
    with pytest.raises(TypeError, match="HashShuffleParams"):
        m.estimate("not a params", dc)  # type: ignore[arg-type]


# ---------- SortMemoryModel ----------

@pytest.mark.parametrize(
    "available_mb, row_size",
    [(10, 16), (10, 1000), (100, 16), (1, 16)],
)
def test_sort_invariant(available_mb, row_size):
    m = SortMemoryModel()
    dc = _dc(row_size=row_size)
    available = available_mb * 1024 * 1024
    safety = 0.7
    params = m.solve(available, dc, safety=safety)
    assert params is not None
    assert isinstance(params, SortParams)
    assert params.k >= 2
    assert params.read_batch_size >= 1
    est = m.estimate(params, dc)
    assert est <= int(available * safety)


def test_sort_none_when_too_tight():
    m = SortMemoryModel()
    dc = _dc(row_size=1000)
    # available=5 KB -> budget=3.5 KB; read_avail=1.05 KB < 2 * 1 KB -> None
    params = m.solve(5 * 1024, dc, safety=0.7)
    assert params is None


def test_sort_max_k_cap():
    m = SortMemoryModel(max_k=8)
    dc = _dc(row_size=16)
    params = m.solve(100 * 1024 * 1024, dc, safety=0.7)
    assert params is not None
    assert params.k <= 8


# ---------- GroupByMemoryModel ----------

@pytest.mark.parametrize(
    "available_mb, k_est, size_of_key_entry",
    [
        (10, 10, 80),
        (10, 1000, 80),
        (10, 10_000, 80),
        (10, 100_000, 80),
        (100, 1_000_000, 80),
    ],
)
def test_groupby_invariant(available_mb, k_est, size_of_key_entry):
    m = GroupByMemoryModel()
    dc = _dc(k_est=k_est, size_of_key_entry=size_of_key_entry)
    available = available_mb * 1024 * 1024
    safety = 0.7
    params = m.solve(available, dc, safety=safety)
    assert params is not None, (
        f"unexpected None: available={available_mb}MB k_est={k_est}"
    )
    assert isinstance(params, GroupByParams)
    assert params.recurse_depth >= 0
    assert params.recurse_depth <= 5  # max_depth default
    est = m.estimate(params, dc)
    assert est <= int(available * safety), (
        f"estimate {est} > safety {int(available*safety)} "
        f"for k_est={k_est}"
    )


def test_groupby_recurse_depth_upper_bound():
    """K_est / P / p_recurse^d <= max_dict_keys — при d=recurse_depth."""
    m = GroupByMemoryModel()
    dc = _dc(k_est=1_000_000, size_of_key_entry=80, row_size=16)
    params = m.solve(10 * 1024 * 1024, dc, safety=0.7)
    assert params is not None
    k_per = dc.k_est / params.shuffle.num_partitions
    k_after = k_per / (params.p_recurse ** params.recurse_depth)
    assert k_after <= params.max_dict_keys, (
        f"after {params.recurse_depth} splits: "
        f"{k_after:.0f} > max_dict_keys={params.max_dict_keys}"
    )


def test_groupby_none_when_exceeds_max_depth():
    """K_est/P/p_recurse^MAX_DEPTH > max_dict_keys -> None."""
    m = GroupByMemoryModel(max_depth=2)
    dc = _dc(k_est=10**12, size_of_key_entry=80, row_size=16)
    params = m.solve(10 * 1024 * 1024, dc, safety=0.7)
    assert params is None


def test_groupby_invalid_ratios():
    with pytest.raises(ValueError, match="<= 1.0"):
        GroupByMemoryModel(partition_ratio=0.5, dict_ratio=0.4, c_temp_ratio=1.0)


def test_groupby_requires_k_est():
    m = GroupByMemoryModel()
    dc = _dc(k_est=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k_est"):
        m.solve(10 * 1024 * 1024, dc)


# ---------- composition check ----------

def test_groupby_estimate_is_max_of_parts():
    """f_groupby = max(f_shuffle, f_local); проверяем, что >= f_shuffle."""
    m = GroupByMemoryModel()
    dc = _dc(k_est=10_000, size_of_key_entry=80, row_size=16)
    params = m.solve(10 * 1024 * 1024, dc, safety=0.7)
    assert params is not None
    f_shuffle = m.shuffle_model.estimate(params.shuffle, dc)
    f_total = m.estimate(params, dc)
    assert f_total >= f_shuffle
