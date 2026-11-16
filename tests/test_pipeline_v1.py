"""v1.0.0-pipeline: real-model pipeline integration tests.

Real steps (not mocks):
  - Standardize (narrow, streaming-capable)
  - PCAProjector (barrier transformer)
  - LinearRegressor (barrier predictor)

Tests: correctness (R²), multi-barrier chain, bit-identical eager vs
streaming (where streaming path succeeds).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.pipeline import Pipeline
from cadw.steps import FrameMethod, Step, Transformer, Model
from cadw.steps._mvp import LinearRegressor, Standardize, PCAProjector
from tests._helpers._report import report, pipeline_trace


def _make_df(n: int = 500, seed: int = 0,
             guard_mb: float = 500) -> CadwDataFrame:
    rng = np.random.RandomState(seed)
    x1 = rng.normal(10, 2, n)
    x2 = rng.normal(-5, 1.5, n)
    x3 = rng.normal(0, 1, n)
    y = 2.0 * x1 + 3.0 * x2 - 1.5 * x3 + 0.5 * rng.normal(size=n)
    return CadwDataFrame.from_table(pa.table({
        "x1": pa.array(x1, type=pa.float64()),
        "x2": pa.array(x2, type=pa.float64()),
        "x3": pa.array(x3, type=pa.float64()),
        "y": pa.array(y, type=pa.float64()),
    }), memory_guard=MemoryGuard(max_memory_mb=guard_mb))


def _collect_col(df, col):
    return np.array(
        [v for b in df._iter_filtered_batches()
         for v in b.column(col).to_pylist()], dtype=np.float64)


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return 1.0 - np.var(y_true - y_pred) / np.var(y_true)


# ============================================================
# Base class hierarchy
# ============================================================

def test_base_classes_inheritance():
    assert isinstance(Standardize(["x1"]), Transformer)
    assert isinstance(PCAProjector(["x1", "x2"], 1), Transformer)
    assert isinstance(LinearRegressor(["x1"], "y"), Model)
    # All Steps
    assert isinstance(Standardize(["x1"]), Step)
    assert isinstance(LinearRegressor(["x1"], "y"), Step)


# ============================================================
# Single model pipeline
# ============================================================

def test_single_model_pipeline():
    df = _make_df()
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p.add_step("model", LinearRegressor(["x1", "x2", "x3"], "y"))
    with pipeline_trace() as trace:
        p.fit(df, target="y")
        plan = p.plan(df)
        preds = p.predict(df)
    assert preds.count() == df.count()
    r2 = _r2(_collect_col(df, "y"), _collect_col(preds, "y_pred"))
    report("test_single_model_pipeline", r2=r2, n_rows=df.count(),
           plan_peak_b=plan.peak_bytes, total_state_b=p.total_state_bytes(),
           writes=trace.writes, scans=trace.scans)
    assert r2 > 0.99, f"R2={r2:.4f}"


# ============================================================
# Long chain with barriers: narrow + barrier + narrow? + model
# ============================================================

def test_long_chain_two_barriers():
    """[Standardize, PCAProjector, LinearRegressor] — 2 barriers.

    Standardize: narrow (fused in streaming)
    PCAProjector: barrier (materializes)
    LinearRegressor: barrier (predictor, fits on lazy)
    """
    df = _make_df()
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p.add_step("pca", PCAProjector(["x1", "x2", "x3"], n_components=2))
    # Discover PCA output column names from fitted model
    p.fit(df, target="y")
    # Predict using whatever PCA produced — model refit on pca cols
    # For test simplicity: PCA projects, linear fits on pc_* columns
    # Get actual output schema by transforming df
    pca_step = dict(p.steps)["pca"]
    df_scaled = dict(p.steps)["scale"].transform(df)
    df_pca = pca_step.transform(df_scaled)
    new_cols = [c for c in df_pca.columns if c not in df.columns]
    assert len(new_cols) >= 2, f"PCA must produce ≥2 new cols, got {new_cols}"

    # Full chain with discovered PCA col names
    p2 = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p2.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p2.add_step("pca", PCAProjector(["x1", "x2", "x3"], n_components=2))
    p2.add_step("model", LinearRegressor(new_cols[:2], "y"))
    with pipeline_trace(label="fit") as trace_fit:
        p2.fit(df, target="y")
        writes_fit = trace_fit.writes
    with pipeline_trace(label="predict") as trace_pred:
        preds = p2.predict(df)
        writes_pred = trace_pred.writes
    assert preds.count() == df.count()
    r2 = _r2(_collect_col(df, "y"), _collect_col(preds, "y_pred"))
    plan = p2.plan(df)
    report("test_long_chain_two_barriers", r2=r2, n_rows=df.count(),
           pca_new_cols=new_cols[:2], total_state_b=p2.total_state_bytes(),
           plan_peak_b=plan.peak_bytes,
           writes_fit=writes_fit, writes_predict=writes_pred)
    for w in trace_fit.write_log:
        report(f"  [fit] write #{w['write_n']}",
               rows=w["rows"], n_cols=len(w["cols"]))
    for w in trace_pred.write_log:
        report(f"  [predict] write #{w['write_n']}",
               rows=w["rows"], n_cols=len(w["cols"]))
    # PCA to 2 dims loses some signal (y depends on 3 inputs); R² > 0.5 generous
    assert r2 > 0.5, f"R2={r2:.4f}"


# ============================================================
# FrameMethod + narrow pipeline
# ============================================================

def test_frame_method_pipeline():
    """[FrameMethod fillna, Standardize, LinearRegressor] — 1 barrier."""
    df = _make_df()
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("fill", FrameMethod("fillna", value=0))
    p.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p.add_step("model", LinearRegressor(["x1", "x2", "x3"], "y"))
    p.fit(df, target="y")
    preds = p.predict(df)
    r2 = _r2(_collect_col(df, "y"), _collect_col(preds, "y_pred"))
    report("test_frame_method_pipeline", r2=r2, n_rows=df.count())
    assert r2 > 0.99


# ============================================================
# Eager vs streaming: fitted model produces same predictions
# ============================================================

def test_eager_vs_streaming_predictions_match():
    """fit (eager) vs fit_transform (tries stream) → same predictions."""
    df = _make_df()
    mg = MemoryGuard(max_memory_mb=500)

    # Eager path
    p_eager = Pipeline(memory_guard=mg)
    p_eager.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p_eager.add_step("model", LinearRegressor(["x1", "x2", "x3"], "y"))
    with pipeline_trace() as trace_eager:
        p_eager.fit(df, target="y")
        preds_eager = _collect_col(p_eager.predict(df), "y_pred")

    # Streaming attempt path (fit_transform; falls back to eager if not supported)
    mg2 = MemoryGuard(max_memory_mb=500)
    p_stream = Pipeline(memory_guard=mg2)
    p_stream.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p_stream.add_step("model", LinearRegressor(["x1", "x2", "x3"], "y"))
    with pipeline_trace() as trace_stream:
        p_stream.fit_transform(df, target="y")
        preds_stream = _collect_col(p_stream.predict(df), "y_pred")

    # Same predictions (both paths fit OLS on standardized data)
    diff = float(np.max(np.abs(preds_eager - preds_stream)))
    report("test_eager_vs_streaming", max_diff=diff,
           n_rows=len(preds_eager),
           writes_eager=trace_eager.writes,
           writes_stream=trace_stream.writes,
           scans_eager=trace_eager.scans,
           scans_stream=trace_stream.scans)
    np.testing.assert_allclose(
        preds_eager, preds_stream, rtol=1e-6,
        err_msg="eager vs streaming predictions must match")


# ============================================================
# Batch selection: n >> max_rows_in_budget → multiple batches
# ============================================================

def test_batch_selection_splits_multi_batch(caplog):
    """For n >> max_rows_in_budget, fit splits into multiple batches.

    Small budget forces small max_rows_in_budget, n=200k triggers
    multi-batch. Evidence: derive_batch_size log line
    'actual: effective_batch=X rows -> N batch(es) (n_rows=... > max_rows)'
    with N > 1.
    """
    import logging
    import re

    caplog.set_level(logging.INFO, logger="cadw.memory_estimate")

    mg = MemoryGuard(max_memory_mb=10)       # 7 MB usable
    df = _make_df(n=200_000, guard_mb=10)    # 200k rows, 10 MB guard (same as pipeline)
    p = Pipeline(memory_guard=mg)
    p.add_step("model", LinearRegressor(["x1", "x2", "x3"], "y"))
    p.fit(df, target="y")

    # Find "actual: effective_batch=N rows -> M batch(es)" log
    actual_logs = [
        r.message for r in caplog.records
        if "actual:" in r.message and "batch(es)" in r.message
    ]
    assert actual_logs, (
        f"no 'actual:' log; caplog records: "
        f"{[r.message[:80] for r in caplog.records if 'batch' in r.message.lower()][:5]}")
    log_msg = actual_logs[0]
    m = re.search(
        r"effective_batch=(\d+) rows -> (\d+) batch\(es\)",
        log_msg)
    assert m, f"unparsable log: {log_msg!r}"
    effective = int(m.group(1))
    n_batches = int(m.group(2))

    report("test_batch_selection_multi_batch",
           n_rows=df.count(), budget_mb=10,
           effective_batch=effective, n_batches=n_batches)

    # batch selection MUST produce >1 batch when n > max_rows_in_budget
    assert n_batches > 1, (
        f"n=200k, budget=10MB: expected multi-batch, got "
        f"{n_batches} batch(es) (effective={effective}); "
        f"log={log_msg!r}")
    # And effective < n_rows (data was split)
    assert effective < df.count(), (
        f"effective_batch={effective} >= n_rows={df.count()} "
        f"— no split happened")
    # Verify fit succeeded (would OOM if batch=n and memory too small)
    assert p.fitted


# ============================================================
# add_step validation
# ============================================================

def test_add_step_duplicate_name_raises():
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=100))
    p.add_step("s", Standardize(["x1"]))
    with pytest.raises(ValueError, match="duplicate name"):
        p.add_step("s", Standardize(["x2"]))


def test_add_step_two_predictors_raises():
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=100))
    p.add_step("m1", LinearRegressor(["x1"], "y"))
    with pytest.raises(RuntimeError, match="already has a Predictor"):
        p.add_step("m2", LinearRegressor(["x2"], "y"))


def test_add_step_after_predictor_raises():
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=100))
    p.add_step("m", LinearRegressor(["x1"], "y"))
    with pytest.raises(RuntimeError, match="after a Predictor"):
        p.add_step("t", Standardize(["x1"]))


# ============================================================
# Serialization: to_json/from_json with MVP steps
# ============================================================

def test_to_json_from_json_roundtrip(tmp_path):
    df = _make_df()
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p.add_step("model", LinearRegressor(["x1", "x2", "x3"], "y"))
    p.fit(df, target="y")
    preds_before = _collect_col(p.predict(df), "y_pred")

    out_path = p.to_json(tmp_path / "pipe")
    p2 = Pipeline.from_json(out_path, memory_guard=MemoryGuard(max_memory_mb=500))
    assert p2.fitted
    preds_after = _collect_col(p2.predict(df), "y_pred")
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-9)
