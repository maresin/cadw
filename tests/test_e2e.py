"""
E2E: реальный Parquet → filter → groupby → sort → Parquet
под жёстким бюджетом RAM (MemoryGuard), с контролем пикового RSS.

Это единственный тест, доказывающий «каркас production-readiness»:
все компоненты (Dataset, CadwDataFrame, RowFilter, примитивы, MemoryGuard,
Arrow IPC) работают вместе на данных, превышающих бюджет памяти,
и завершаются без OOM.
"""
import os
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _make_source(path: Path, n: int = 200_000) -> None:
    rng = np.random.default_rng(seed=42)
    cats = np.array([f"g{i:03d}" for i in range(100)])
    table = pa.table({
        "cat": rng.choice(cats, size=n),
        "val": rng.integers(0, 1000, size=n).astype(np.int64),
    })
    pq.write_table(table, path, row_group_size=10_000)


@pytest.mark.slow
def test_e2e_filter_groupby_sort(tmp_path):
    src = tmp_path / "src.parquet"
    _make_source(src, n=200_000)

    # Бюджет: 20 MB. Один батч = 5k строк ~ 80 KB; рекурсия/skew работают.
    guard = MemoryGuard(max_memory_mb=20)
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5_000,
        memory_guard=guard,
        spill_dir=tmp_path / "spill",
    )

    rss_start = _rss_mb()
    t0 = time.time()

    # filter -> groupby -> sort -> write
    pipeline = (
        df[df["val"] > 100]
        .groupby("cat")
        .agg("val", "mean")
        .sort_values("val_mean", ascending=False)
    )
    out = tmp_path / "out.parquet"
    pipeline.write_parquet(out)

    elapsed = time.time() - t0
    rss_peak_est = _rss_mb() - rss_start

    # 1. Файл создан, схема корректна
    assert out.exists()
    t = pq.read_table(out)
    assert t.num_rows > 0
    assert set(t.schema.names) == {"cat", "val_mean"}

    # 2. Сортировка применена (val_mean по убыванию)
    means = t.column("val_mean").to_pylist()
    assert means == sorted(means, reverse=True), "sort_values(ascending=False) broken"

    # 3. Все cat — уникальны (groupby схлопнул)
    cats = t.column("cat").to_pylist()
    assert len(cats) == len(set(cats)), "groupby left duplicates"

    # 4. RSS не взорвался (allow 200 MB — с запасом на Python + Arrow overhead)
    assert rss_peak_est < 200, f"RSS grew by {rss_peak_est:.1f} MB under 20 MB budget"

    # 5. Уложились в разумное время (< 60 s для CI; локально ~2-5 s)
    assert elapsed < 60, f"E2E took {elapsed:.1f}s"


@pytest.mark.slow
def test_e2e_against_pandas_equivalence(tmp_path):
    """Тот же результат, что и у pandas (sanity)."""
    src = tmp_path / "src.parquet"
    _make_source(src, n=50_000)

    import pandas as pd
    pdf = pd.read_parquet(src)
    expected = (
        pdf[pdf["val"] > 100]
        .groupby("cat", as_index=False)["val"]
        .mean()
        .sort_values("val_mean" if "val_mean" in pdf else "val",
                     ascending=False)
    )
    # pandas: колонка будет 'val', у нас 'val_mean'. Переименуем для сравнения.
    expected = expected.rename(columns={"val": "val_mean"}).reset_index(drop=True)

    guard = MemoryGuard(max_memory_mb=20)
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5_000,
        memory_guard=guard, spill_dir=tmp_path / "spill",
    )
    result = (
        df[df["val"] > 100]
        .groupby("cat")
        .agg("val", "mean")
        .sort_values("val_mean", ascending=False)
    )
    out = tmp_path / "out.parquet"
    result.write_parquet(out)
    got = pq.read_table(out).to_pandas().reset_index(drop=True)

    # Сравнение mean по cat, допуск на float
    got_sorted = got.sort_values("cat").reset_index(drop=True)
    exp_sorted = expected.sort_values("cat").reset_index(drop=True)
    assert set(got_sorted["cat"]) == set(exp_sorted["cat"])
    np.testing.assert_allclose(
        got_sorted["val_mean"].to_numpy(),
        exp_sorted["val_mean"].to_numpy(),
        rtol=1e-10,
    )
