"""
E2E join (11-B-2d, v0.2 final): реальные объёмы, MemoryGuard 20 MB.

Три сценария:
  1. 100k × 10k inner (string keys, PK-FK) — pandas-equivalence + RSS/time.
  2. 100k × 100k skew (один ключ) — MemoryLimitExceeded ДО OOM.
  3. 100k × 5k left (половина left-ключей unmatched) — left preserved,
     right nulls.

String keys специально: проверяют planner-путь с avg_key_size из sample
и наличие строкового кодирования в shuffle.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _write_pair(tmp_path, left_data, right_data):
    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    pq.write_table(pa.table(left_data), left_path, row_group_size=5_000)
    pq.write_table(pa.table(right_data), right_path, row_group_size=5_000)
    return left_path, right_path


def _load_pair(tmp_path, left_path, right_path, mem_mb=20):
    left = CadwDataFrame.from_parquet(
        left_path, tmp_path / "ds_left", chunk_size=5_000,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / "spill_left",
    )
    right = CadwDataFrame.from_parquet(
        right_path, tmp_path / "ds_right", chunk_size=5_000,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / "spill_right",
    )
    return left, right


# ---------- 1. inner 100k × 10k ----------

@pytest.mark.slow
def test_e2e_join_100k_x_10k(tmp_path):
    """100k FK × 10k PK, string keys, 20 MB budget, inner join."""
    n_left, n_right = 100_000, 10_000
    rng = np.random.default_rng(42)
    left_keys = [f"k{i:05d}" for i in rng.integers(0, n_right, size=n_left)]
    right_keys = [f"k{i:05d}" for i in range(n_right)]
    left_vals = rng.integers(0, 1000, size=n_left).astype(np.int64).tolist()
    right_vals = rng.integers(0, 1000, size=n_right).astype(np.int64).tolist()

    left_path, right_path = _write_pair(
        tmp_path,
        {"k": left_keys, "lv": left_vals},
        {"k": right_keys, "rv": right_vals},
    )
    left, right = _load_pair(tmp_path, left_path, right_path)

    rss_start = _rss_mb()
    t0 = time.time()
    result = left.merge(right, on="k")
    out = tmp_path / "joined.parquet"
    result.write_parquet(out)
    elapsed = time.time() - t0
    rss_delta = _rss_mb() - rss_start

    # 1. Результат = len(left) (each left matches exactly 1 right PK)
    got = pq.read_table(out)
    assert got.num_rows == n_left, f"expected {n_left}, got {got.num_rows}"

    # 2. Pandas-equivalence
    pdf_l = pd.read_parquet(left_path)
    pdf_r = pd.read_parquet(right_path)
    expected = (
        pdf_l.merge(pdf_r, on="k", how="inner")
        .sort_values(["k", "lv"]).reset_index(drop=True)
    )
    got_sorted = (
        got.to_pandas()
        .sort_values(["k", "lv"]).reset_index(drop=True)
    )
    assert set(got_sorted.columns) == set(expected.columns)
    np.testing.assert_array_equal(
        got_sorted["lv"].to_numpy(), expected["lv"].to_numpy(),
    )
    np.testing.assert_array_equal(
        got_sorted["rv"].to_numpy(), expected["rv"].to_numpy(),
    )

    # 3. RSS delta << budget·k (проверяем, что не OOM-нуло и не залипло)
    assert rss_delta < 200, f"RSS grew by {rss_delta:.1f} MB under 20 MB budget"
    assert elapsed < 60, f"took {elapsed:.1f}s"


# ---------- 2. skew reject 100k × 100k ----------

@pytest.mark.slow
def test_e2e_join_skew_reject(tmp_path):
    """100k × 100k с одним ключом: MemoryLimitExceeded ДО OOM."""
    n = 100_000
    left_path, right_path = _write_pair(
        tmp_path,
        {"k": ["HOT"] * n, "lv": list(range(n))},
        {"k": ["HOT"] * n, "rv": list(range(n))},
    )
    left, right = _load_pair(tmp_path, left_path, right_path)

    rss_start = _rss_mb()
    with pytest.raises(MemoryLimitExceeded):
        left.merge(right, on="k")
    rss_delta = _rss_mb() - rss_start

    # Процесс выжил, RSS не взорвался до raise
    assert rss_delta < 500, (
        f"RSS grew by {rss_delta:.1f} MB before raise; "
        f"skew protection должен остановить до большого output"
    )


# ---------- 3. left preserve 100k × 5k ----------

@pytest.mark.slow
def test_e2e_join_left_preserve(tmp_path):
    """Left join: 100k left, 5k right (половина left-ключей unmatched).

    Все 100k строк left сохраняются; для unmatched right-колонки → null.
    """
    n_left, n_right = 100_000, 5_000
    rng = np.random.default_rng(7)
    left_keys = [f"k{i:05d}" for i in rng.integers(0, 10_000, size=n_left)]
    right_keys = [f"k{i:05d}" for i in range(n_right)]
    left_vals = rng.integers(0, 1000, size=n_left).astype(np.int64).tolist()
    right_vals = rng.integers(0, 1000, size=n_right).astype(np.int64).tolist()

    left_path, right_path = _write_pair(
        tmp_path,
        {"k": left_keys, "lv": left_vals},
        {"k": right_keys, "rv": right_vals},
    )
    left, right = _load_pair(tmp_path, left_path, right_path)

    result = left.merge(right, on="k", how="left")
    out = tmp_path / "joined.parquet"
    result.write_parquet(out)

    got = pq.read_table(out)
    assert got.num_rows == n_left, (
        f"left join must preserve {n_left} left rows, got {got.num_rows}"
    )

    # Pandas-equivalence
    pdf_l = pd.read_parquet(left_path)
    pdf_r = pd.read_parquet(right_path)
    expected = (
        pdf_l.merge(pdf_r, on="k", how="left")
        .sort_values(["k", "lv"]).reset_index(drop=True)
    )
    got_sorted = (
        got.to_pandas()
        .sort_values(["k", "lv"]).reset_index(drop=True)
    )

    np.testing.assert_array_equal(
        got_sorted["lv"].to_numpy(), expected["lv"].to_numpy(),
    )
    # rv: null для unmatched. Сравниваем null-mask + значения.
    got_null = got_sorted["rv"].isna().to_numpy()
    exp_null = expected["rv"].isna().to_numpy()
    np.testing.assert_array_equal(
        got_null, exp_null,
        err_msg="unmatched-null mask differs",
    )
    # Сравниваем non-null значения
    mask = ~exp_null
    np.testing.assert_array_equal(
        got_sorted.loc[mask, "rv"].to_numpy().astype(np.int64),
        expected.loc[mask, "rv"].to_numpy().astype(np.int64),
    )
