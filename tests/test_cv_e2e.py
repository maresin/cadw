"""v0.9.2e: cross-validation E2E (slow)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard, KFold, cv_score
from cadw.pipeline import GBDTStep


@pytest.mark.slow
def test_cv_score_100k_kfold_gbdt(tmp_path):
    """100K rows, KFold(5), GBDT, memory 50 MB, R² > 0.5."""
    n = 100_000
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.0 * x1 + 1.5 * x2 + rng.normal(scale=0.2, size=n)

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist(),
        }),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )

    cv = KFold(n_splits=5, shuffle=False)

    def factory():
        return GBDTStep(
            ["x1", "x2"], "y",
            n_estimators=20, max_depth=2, learning_rate=0.2,
        )

    result = cv_score(
        df, cv, factory,
        X_cols=["x1", "x2"], y_col="y", metrics=["r2", "rmse"],
    )
    assert len(result["fold_scores"]) == 5
    assert result["mean"]["r2"] > 0.5, (
        f"mean r2 = {result['mean']['r2']:.3f}, expected > 0.5"
    )
    # std bounded (uniform data → consistent across folds)
    assert result["std"]["r2"] < 0.1
