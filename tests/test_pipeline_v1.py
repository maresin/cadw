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

@pytest.mark.slow
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

# ============================================================
# v0.9.239: pipeline memory architecture
# ============================================================

def test_pipeline_guard_authority(caplog):
    """P-C: Pipeline mg overrides df mg during fit_linear."""
    import logging
    caplog.set_level(logging.INFO, logger="cadw.memory_estimate")

    df = _make_df(n=10_000, guard_mb=500)
    mg = MemoryGuard(max_memory_mb=10)
    p = Pipeline(memory_guard=mg)
    p.add_step("model", LinearRegressor(["x1", "x2", "x3"], "y"))
    p.fit(df, target="y")

    budget_logs = [
        r.message for r in caplog.records
        if "budget:" in r.message and "available=" in r.message
    ]
    assert budget_logs, (
        f"no budget log; records: "
        f"{[r.message[:80] for r in caplog.records if 'budget' in r.message.lower()][:3]}")
    assert "available=10485760" in budget_logs[0], (
        f"expected budget=10MB from Pipeline (P-C); got {budget_logs[0]!r}")


def test_plan_real_peak(caplog):
    """P-A: plan_peak uses effective_batch — MUST be formula_batch."""
    import logging
    caplog.set_level(logging.DEBUG, logger="cadw.pipeline")

    df = _make_df(n=500)
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p.add_step("model", LinearRegressor(["x1", "x2", "x3"], "y"))
    plan = p.plan(df, mode="eager")

    total_state_b = p.total_state_bytes()
    report("test_plan_real_peak", n_rows=df.count(),
           plan_peak_b=plan.peak_bytes,
           total_state_b=total_state_b)

    # A2: total_state_bytes() after plan() must be non-zero
    assert total_state_b > 0, (
        f"A2: total_state_bytes() after plan() returned 0 "
        f"(plan state_estimates not exposed)")

    peak_logs = [r.message for r in caplog.records
                 if "peak_source=" in r.message]
    assert any("peak_source=formula_batch" in m for m in peak_logs), (
        f"P-A not active: no peak_source=formula_batch in {peak_logs}")
    assert plan.peak_bytes < 10 * 1024 * 1024, (
        f"plan_peak={plan.peak_bytes} B (expected < 10 MB for n=500)")


def test_plan_segmented(caplog):
    """P-B: segmented plan — per-stage log lines."""
    import logging
    caplog.set_level(logging.INFO, logger="cadw.pipeline")

    df = _make_df(n=500)
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p.add_step("pca", PCAProjector(["x1", "x2", "x3"], n_components=2))
    p.add_step("model", LinearRegressor(["pc0", "pc1"], "y"))
    plan = p.plan(df, mode="eager")

    stage_logs = [r.message for r in caplog.records
                  if "plan[eager] stage" in r.message]
    assert len(stage_logs) >= 2, (
        f"expected >=2 stage logs, got {len(stage_logs)}: {stage_logs}")
    for log in stage_logs:
        assert "narrow_peak=" in log, f"missing narrow_peak: {log!r}"
        assert "barrier_peak=" in log, f"missing barrier_peak: {log!r}"
        assert "stage_total=" in log, f"missing stage_total: {log!r}"
    assert plan.peak_bytes > 0


def test_pca_serialization_roundtrip(tmp_path):
    """PCAProjector.to_dict/from_dict preserves model state (via PCA)."""
    df = _make_df(n=500)
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p.add_step("pca", PCAProjector(["x1", "x2", "x3"], n_components=2))
    p.add_step("model", LinearRegressor(["pc0", "pc1"], "y"))
    p.fit(df, target="y")
    preds_before = _collect_col(p.predict(df), "y_pred")

    out = p.to_json(tmp_path / "pipe")
    p2 = Pipeline.from_json(out, memory_guard=MemoryGuard(max_memory_mb=500))
    assert p2.fitted
    preds_after = _collect_col(p2.predict(df), "y_pred")
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-9,
        err_msg="PCA state not preserved through to_json/from_json")


def test_execute_stream_preflight(caplog):
    """_execute_stream runs plan(mode='streaming') preflight."""
    import logging
    caplog.set_level(logging.INFO, logger="cadw.pipeline")

    df = _make_df(n=500)
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=10))
    p.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p.add_step("model", LinearRegressor(["x1", "x2", "x3"], "y"))
    p.fit_transform(df, target="y")

    streaming_plan_logs = [r.message for r in caplog.records
                           if "plan[streaming]" in r.message]
    assert streaming_plan_logs, (
        f"no plan[streaming] log; records: "
        f"{[r.message[:80] for r in caplog.records if 'plan' in r.message][:5]}")

def test_streaming_guard_authority(caplog):
    """A1: Pipeline mg overrides df mg in _execute_stream (streaming path).

    Was: df(guard=500MB) leaked into streaming fit_iter/transform_batch and
    LinearRegressor.fit -> derive_batch_size saw 500MB, not pipeline's 10MB.
    Now: fit_transform overrides df._memory_guard during _execute_stream.
    """
    import logging
    caplog.set_level(logging.INFO, logger="cadw.memory_estimate")
    caplog.set_level(logging.INFO, logger="cadw.pipeline")

    df = _make_df(n=500, guard_mb=500)
    mg = MemoryGuard(max_memory_mb=10)
    p = Pipeline(memory_guard=mg)
    p.add_step("scale", Standardize(["x1", "x2", "x3"]))
    p.add_step("model", LinearRegressor(["x1", "x2", "x3"], "y"))
    p.fit_transform(df, target="y")

    # Must have actually taken streaming path (not eager fallback)
    streaming_plan = [r.message for r in caplog.records
                      if "plan[streaming]" in r.message]
    assert streaming_plan, (
        f"streaming preflight not logged; records with 'plan': "
        f"{[r.message[:80] for r in caplog.records if 'plan' in r.message][:5]}")
    completed = [r.message for r in caplog.records
                 if "_execute_stream: completed (streaming)" in r.message]
    assert completed, (
        f"streaming did not complete (fell back to eager); "
        f"FAILED logs: {[r.message for r in caplog.records if 'FAILED' in r.message][:3]}")

    budget_logs = [r.message for r in caplog.records
                   if "budget:" in r.message and "available=" in r.message]
    assert budget_logs, "no derive_batch_size budget log in streaming path"
    assert "available=10485760" in budget_logs[0], (
        f"A1 (streaming guard) not active: expected 10485760 B "
        f"(pipeline mg=10MB); got {budget_logs[0]!r}. "
        f"df(guard=500MB) leaked into streaming path.")

# ============================================================
# A4 (v0.9.240): CompositeStep regression
# ============================================================

def test_composite_step_transform_chain():
    """CompositeStep.transform == manual flat chain (narrow+narrow)."""
    from cadw.steps.composite import CompositeStep

    df = _make_df(n=500, seed=7)
    s1 = Standardize(["x1", "x2"])
    s2 = Standardize(["x2", "x3"])

    cs = CompositeStep([s1, s2])
    assert cs.is_barrier is False, (
        "narrow+narrow CompositeStep must not be barrier")
    assert cs._streamable is True, (
        "Standardize has fit_iter/transform_batch/output_schema -> streamable")

    cs.fit(df)
    out_cs = cs.transform(df)

    # Manual flat chain (same fit semantics: all children fit on same df;
    # transform chains through children).
    s1m = Standardize(["x1", "x2"])
    s2m = Standardize(["x2", "x3"])
    s1m.fit(df)
    s2m.fit(df)
    out_manual = s2m.transform(s1m.transform(df))

    for col in ["x1", "x2", "x3"]:
        np.testing.assert_allclose(
            _collect_col(out_cs, col), _collect_col(out_manual, col),
            rtol=1e-12, err_msg=f"CompositeStep chain mismatch at {col}")


def test_composite_step_barrier_flag():
    """CompositeStep.is_barrier True iff any child is barrier."""
    from cadw.steps.composite import CompositeStep

    df = _make_df(n=200, seed=3)
    narrow = Standardize(["x1", "x2"])
    barrier = PCAProjector(["x1", "x2", "x3"], n_components=2)

    cs_narrow_only = CompositeStep([narrow, Standardize(["x3"])])
    assert cs_narrow_only.is_barrier is False

    cs_with_barrier = CompositeStep([Standardize(["x1", "x2"]), barrier])
    assert cs_with_barrier.is_barrier is True, (
        "PCAProjector child is barrier -> CompositeStep barrier")

    # fit+transform works end-to-end (barrier composite)
    cs_with_barrier.fit(df)
    out = cs_with_barrier.transform(df)
    assert "pc0" in out._dataset.schema.names
    assert out.count() == df.count()

def test_multi_phase_model(caplog):
    """P-Y (B2): plan_peak = max(phase_1, phase_2), NOT sum.

    Synthetic MultiPhaseModel with real materialization between phases;
    synthetic phase_1_bytes/phase_2_bytes prove plan honours max (not sum).
    Real GBDT/LDA (per-phase peak from primitive composition) — P2b.
    """
    import logging
    caplog.set_level(logging.INFO, logger="cadw.pipeline")

    from cadw.steps._mvp.multi_phase import MultiPhaseModel

    df = _make_df(n=500)
    N1, N2 = 300, 280
    mp = MultiPhaseModel(
        Standardize(["x1", "x2"]),
        LinearRegressor(["x1", "x2", "x3"], "y"),
        phase_1_bytes=N1, phase_2_bytes=N2,
    )
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("mp", mp)

    plan = p.plan(df, mode="eager")
    report("test_multi_phase_model", n_rows=df.count(),
           phase_1_bytes=N1, phase_2_bytes=N2,
           plan_peak_b=plan.peak_bytes,
           expected_max=max(N1, N2), expected_sum=N1 + N2)

    # KEY (P-Y): plan uses max, not sum
    assert plan.peak_bytes == max(N1, N2), (
        f"P-Y violated: plan_peak_b={plan.peak_bytes}, "
        f"expected max({N1},{N2})={max(N1,N2)}; "
        f"if == sum ({N1+N2}) → multi-phase treated as sum (regression)")
    assert plan.peak_bytes != N1 + N2, (
        f"plan_peak_b == sum (={N1+N2}); "
        f"P-Y regression: multi-phase must be max, not sum")

    # Execution: both phases materialize, no OOM
    p.fit(df, target="y")
    assert p.fitted
    preds = _collect_col(p.predict(df), "y_pred")
    assert len(preds) == df.count()

    # Both phases logged (real materialization between)
    phase_logs = [r.message for r in caplog.records
                  if "fit multi_phase" in r.message]
    assert any("phase 1" in m for m in phase_logs), (
        f"no phase 1 log; records: {phase_logs[:5]}")
    assert any("phase 2" in m for m in phase_logs), (
        f"no phase 2 log; records: {phase_logs[:5]}")

def test_df_cols_view():
    """L7 (v0.9.241): df[cols] is a column-view (shared dataset); steps
    that read df._dataset.schema must use df._columns instead.
    """
    df = _make_df(n=200, seed=5)
    sub = df[["x1"]]
    assert list(sub._columns) == ["x1"]
    assert sub._dataset.schema.names == ["x1", "x2", "x3", "y"]
    batch_names = [b.schema.names for b in sub._iter_filtered_batches()]
    assert all(names == ["x1"] for names in batch_names), (
        f"df[cols] batches not projected: {batch_names[:2]}")

    s = Standardize(["x1"])
    s.fit(df)
    out_sub = s.transform(sub)
    assert out_sub.count() == 200
    assert list(out_sub._columns) == ["x1"]

    from cadw.scalers import StandardScaler
    ss = StandardScaler(["x1"])
    ss.fit(df)
    out_ss = ss.transform(sub)
    assert out_ss.count() == 200
    assert list(out_ss._columns) == ["x1"]

@pytest.mark.slow
def test_pipeline_detailed_log_and_correctness(tmp_path):
    """Detailed log + correctness: 3 batches x 3 rows, hand-computed.

    Data: y = 2*x1 + 3*x2 + 1 (no noise). OLS recovers exactly.
    Verifies: chunking preserved, fit state matches hand, predict == y,
    predict doesn't mutate state, to_json/from_json preserves state.
    """
    from cadw.dataset import Dataset
    from cadw.pipeline import (
        list_step_types, register_step_type,
    )

    # --- data: 3 batches x 3 rows ---
    x1 = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
    x2 = np.array([2., 1., 4., 3., 6., 5., 8., 7., 10.])
    y = 2 * x1 + 3 * x2 + 1  # [9, 8, 19, 18, 29, 28, 39, 38, 49]

    schema = pa.schema([
        pa.field("x1", pa.float64()),
        pa.field("x2", pa.float64()),
        pa.field("y", pa.float64()),
    ])
    batches = [
        pa.RecordBatch.from_arrays([
            pa.array(x1[i*3:(i+1)*3], type=pa.float64()),
            pa.array(x2[i*3:(i+1)*3], type=pa.float64()),
            pa.array(y[i*3:(i+1)*3], type=pa.float64()),
        ], schema=schema)
        for i in range(3)
    ]
    ds = Dataset.from_batches(schema, iter(batches), tmp_path / "raw")
    df = CadwDataFrame.from_dataset(
        ds, memory_guard=MemoryGuard(max_memory_mb=200))

    # --- verify chunking (3 batches x 3) ---
    seen = list(df._iter_filtered_batches())
    assert len(seen) == 3, f"expected 3 batches, got {len(seen)}"
    assert [b.num_rows for b in seen] == [3, 3, 3], \
        f"batch sizes: {[b.num_rows for b in seen]}"
    print("\n=== Input df: 9 rows, 3 batches ===")
    for i, b in enumerate(seen):
        print(f"  batch {i}: x1={b.column('x1').to_pylist()} "
              f"x2={b.column('x2').to_pylist()} "
              f"y={b.column('y').to_pylist()}")

    # --- hand-computed expected (numpy, independent path) ---
    mean_x1 = float(x1.mean())
    mean_x2 = float(x2.mean())
    mean_y = float(y.mean())
    std_x1 = float(x1.std(ddof=0))
    std_x2 = float(x2.std(ddof=0))
    # In z-space: y = intercept_z + z1*coef_z1 + z2*coef_z2
    #   coef_z1 = 2*std_x1, coef_z2 = 3*std_x2, intercept_z = mean_y
    expected_coef_z = np.array([2*std_x1, 3*std_x2])
    expected_intercept_z = mean_y
    print(f"\n=== Expected (numpy independent): mean_x1={mean_x1:.6f} "
          f"mean_x2={mean_x2:.6f} mean_y={mean_y:.6f}")
    print(f"  std_x1={std_x1:.6f} std_x2={std_x2:.6f}")
    print(f"  expected_coef_z={expected_coef_z} "
          f"intercept_z={expected_intercept_z}")

    # --- fit pipeline ---
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    p.add_step("scale", Standardize(["x1", "x2"]))
    p.add_step("model", LinearRegressor(["x1", "x2"], "y"))
    p.fit(df, target="y")

    # --- check Standardize state ---
    s = p._steps[0]
    print(f"\n=== Standardize fitted: mean={s._mean} std={s._std} ===")
    np.testing.assert_allclose(
        [s._mean["x1"], s._mean["x2"]], [mean_x1, mean_x2],
        rtol=1e-9, err_msg="Standardize mean mismatch")
    np.testing.assert_allclose(
        [s._std["x1"], s._std["x2"]], [std_x1, std_x2],
        rtol=1e-9, err_msg="Standardize std mismatch")

    # --- check LinearRegressor state ---
    m = p._steps[1]._model
    print(f"=== LinearModel fitted: coef_={m.coef_} "
          f"intercept_={m.intercept_} ===")
    np.testing.assert_allclose(
        m.coef_, expected_coef_z, rtol=1e-9,
        err_msg="LinearModel coef mismatch (z-space)")
    assert abs(m.intercept_ - expected_intercept_z) < 1e-9, \
        f"intercept {m.intercept_} != {expected_intercept_z}"

    # --- predict correctness ---
    pred_df = p.predict(df)
    y_pred = _collect_col(pred_df, "y_pred")
    print(f"\n=== predict: y_pred={y_pred.tolist()} "
          f"y_true={y.tolist()} ===")
    max_diff = float(np.max(np.abs(y_pred - y)))
    print(f"  max_diff={max_diff:.2e}")
    np.testing.assert_allclose(y_pred, y, rtol=1e-9,
        err_msg="predict != y (no-noise linear data)")

    # --- predict doesn't mutate state ---
    mean_snapshot = dict(s._mean)
    std_snapshot = dict(s._std)
    coef_snapshot = m.coef_.copy()
    intercept_snapshot = float(m.intercept_)
    p.predict(df)  # second call
    assert s._mean == mean_snapshot
    assert s._std == std_snapshot
    np.testing.assert_array_equal(m.coef_, coef_snapshot)
    assert m.intercept_ == intercept_snapshot
    print("=== predict does not modify state: OK ===")

    # --- persistence roundtrip ---
    for nm, cls_ in [("Standardize", Standardize),
                     ("LinearRegressor", LinearRegressor)]:
        if nm not in list_step_types():
            register_step_type(nm, cls_)
    out = p.to_json(tmp_path / "pipe")
    p2 = Pipeline.from_json(out, memory_guard=MemoryGuard(max_memory_mb=200))
    assert p2.fitted
    s2 = p2._steps[0]
    m2 = p2._steps[1]._model
    np.testing.assert_allclose(
        [s2._mean["x1"], s2._mean["x2"]], [mean_x1, mean_x2],
        rtol=1e-9)
    np.testing.assert_allclose(m2.coef_, m.coef_, rtol=1e-9)
    assert m2.intercept_ == m.intercept_
    y_pred2 = _collect_col(p2.predict(df), "y_pred")
    diff_rt = float(np.max(np.abs(y_pred2 - y_pred)))
    print(f"\n=== to_json -> from_json -> predict: max_diff vs original = "
          f"{diff_rt:.2e} ===")
    np.testing.assert_allclose(y_pred2, y_pred, rtol=1e-9)

    report("test_pipeline_detailed_log", n_rows=df.count(),
           batch_sizes=[3, 3, 3], max_diff_predict=max_diff,
           max_diff_roundtrip=diff_rt)

@pytest.mark.slow
def test_pipeline_train_test_split_no_state_leak(tmp_path):
    """Train/test split: pipe.fit(train); pipe.predict(test).

    Guards against fit-state being overwritten / lost / recomputed on test.
    - y_pred on test must match hand-computed using TRAIN statistics
      (mean/std/coef from train), NOT test statistics.
    - predict must not modify state (snapshot before/after).
    - Standalone model.fit(train); model.predict(test) works (outside Pipeline).
    """
    from cadw.dataset import Dataset
    from cadw.pipeline import list_step_types, register_step_type

    x1 = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
    x2 = np.array([2., 1., 4., 3., 6., 5., 8., 7., 10.])
    y = 2 * x1 + 3 * x2 + 1
    schema = pa.schema([
        pa.field("x1", pa.float64()),
        pa.field("x2", pa.float64()),
        pa.field("y", pa.float64()),
    ])

    def _make_batch(start, end):
        return pa.RecordBatch.from_arrays([
            pa.array(x1[start:end], type=pa.float64()),
            pa.array(x2[start:end], type=pa.float64()),
            pa.array(y[start:end], type=pa.float64()),
        ], schema=schema)

    mg = MemoryGuard(max_memory_mb=200)
    # Train: rows 0..6, 2 batches of 3
    train_ds = Dataset.from_batches(
        schema, iter([_make_batch(0, 3), _make_batch(3, 6)]),
        tmp_path / "train")
    train = CadwDataFrame.from_dataset(train_ds, memory_guard=mg)
    # Test: rows 6..9, 1 batch of 3
    test_ds = Dataset.from_batches(
        schema, iter([_make_batch(6, 9)]), tmp_path / "test")
    test = CadwDataFrame.from_dataset(test_ds, memory_guard=mg)

    # --- train statistics (hand) ---
    x1_tr, x2_tr, y_tr = x1[:6], x2[:6], y[:6]
    mean_x1_tr = float(x1_tr.mean())
    mean_x2_tr = float(x2_tr.mean())
    mean_y_tr = float(y_tr.mean())
    std_x1_tr = float(x1_tr.std(ddof=0))
    std_x2_tr = float(x2_tr.std(ddof=0))
    coef_z_exp = np.array([2*std_x1_tr, 3*std_x2_tr])
    intercept_z_exp = mean_y_tr
    print(f"\n=== TRAIN stats (hand) ===")
    print(f"  mean_x1={mean_x1_tr} mean_x2={mean_x2_tr} mean_y={mean_y_tr}")
    print(f"  std_x1={std_x1_tr:.10f} std_x2={std_x2_tr:.10f}")
    print(f"  expected coef_z={coef_z_exp} intercept_z={intercept_z_exp}")

    # --- expected y_pred on TEST using TRAIN statistics ---
    x1_te, x2_te, y_te = x1[6:], x2[6:], y[6:]
    z1_te = (x1_te - mean_x1_tr) / std_x1_tr
    z2_te = (x2_te - mean_x2_tr) / std_x2_tr
    y_pred_exp = intercept_z_exp + z1_te * coef_z_exp[0] + z2_te * coef_z_exp[1]
    print(f"\n=== TEST expected (via TRAIN stats) ===")
    print(f"  x1={x1_te.tolist()} x2={x2_te.tolist()}")
    print(f"  y_true={y_te.tolist()}")
    print(f"  y_pred_exp={y_pred_exp.tolist()}")

    # --- Pipeline: fit on train, predict on test ---
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    p.add_step("scale", Standardize(["x1", "x2"]))
    p.add_step("model", LinearRegressor(["x1", "x2"], "y"))
    p.fit(train, target="y")

    s = p._steps[0]
    m = p._steps[1]._model
    print(f"\n=== After fit(train) ===")
    print(f"  Standardize.mean={s._mean}")
    print(f"  Standardize.std={s._std}")
    print(f"  coef_={m.coef_} intercept_={m.intercept_}")
    # Fitted state MUST be train statistics, not full/test
    np.testing.assert_allclose(
        [s._mean["x1"], s._mean["x2"]], [mean_x1_tr, mean_x2_tr],
        rtol=1e-9, err_msg="Standardize mean != TRAIN mean")
    np.testing.assert_allclose(
        [s._std["x1"], s._std["x2"]], [std_x1_tr, std_x2_tr],
        rtol=1e-9, err_msg="Standardize std != TRAIN std")
    np.testing.assert_allclose(
        m.coef_, coef_z_exp, rtol=1e-9,
        err_msg="coef != TRAIN-derived")

    # Snapshot state to detect overwrite on predict
    mean_snap = dict(s._mean); std_snap = dict(s._std)
    coef_snap = m.coef_.copy(); int_snap = float(m.intercept_)

    # Predict on TEST (not train!) — no refit should happen
    pred_test = p.predict(test)
    y_pred_te = _collect_col(pred_test, "y_pred")
    print(f"\n=== predict(test) ===")
    print(f"  y_pred_te={y_pred_te.tolist()}")
    print(f"  y_pred_exp={y_pred_exp.tolist()}")
    max_diff = float(np.max(np.abs(y_pred_te - y_pred_exp)))
    print(f"  max_diff(vs TRAIN-derived expected)={max_diff:.2e}")
    np.testing.assert_allclose(
        y_pred_te, y_pred_exp, rtol=1e-9,
        err_msg="predict(test) != expected via TRAIN stats — state leak?")
    np.testing.assert_allclose(
        y_pred_te, y_te, rtol=1e-9,
        err_msg="predict(test) != y_true")

    # State must be unchanged after predict(test)
    assert s._mean == mean_snap, "predict modified Standardize._mean"
    assert s._std == std_snap, "predict modified Standardize._std"
    np.testing.assert_array_equal(m.coef_, coef_snap)
    assert m.intercept_ == int_snap
    print("=== state unchanged after predict(test): OK ===")

    # --- Sanity: refit on test SHOULD change state (expected) ---
    p.fit(test, target="y")
    assert s._mean["x1"] != mean_snap["x1"], (
        "refit(test) did not change state (unexpected)")
    print(f"=== refit(test) changed state as expected: "
          f"mean_x1={s._mean['x1']} != {mean_snap['x1']} ===")

    # --- Standalone: step.fit(train); step.transform(test) ---
    s_alone = Standardize(["x1", "x2"])
    s_alone.fit(train)
    out_te = s_alone.transform(test)
    z1_te_alone = _collect_col(out_te, "x1")
    z2_te_alone = _collect_col(out_te, "x2")
    print(f"\n=== Standalone Standardize: fit(train).transform(test) ===")
    print(f"  z1_te={z1_te_alone.tolist()}")
    print(f"  z1_exp={z1_te.tolist()}")
    np.testing.assert_allclose(z1_te_alone, z1_te, rtol=1e-9)
    np.testing.assert_allclose(z2_te_alone, z2_te, rtol=1e-9)

    # --- Standalone: model.fit(train); model.predict(test) ---
    m_alone = LinearRegressor(["x1", "x2"], "y")
    m_alone.fit(train)
    out_alone = m_alone.predict(test)
    y_alone = _collect_col(out_alone, "y_pred")
    print(f"\n=== Standalone LinearRegressor: fit(train).predict(test) ===")
    print(f"  y_alone={y_alone.tolist()}")
    print(f"  y_pred_exp={y_pred_exp.tolist()}")
    np.testing.assert_allclose(y_alone, y_pred_exp, rtol=1e-9)

    # --- Persistence: to_json on train-fit, from_json, predict(test) ---
    for nm, cls_ in [("Standardize", Standardize),
                     ("LinearRegressor", LinearRegressor)]:
        if nm not in list_step_types():
            register_step_type(nm, cls_)
    # Re-fit on train first (p was refit on test above)
    p.fit(train, target="y")
    y_before = _collect_col(p.predict(test), "y_pred")
    out = p.to_json(tmp_path / "pipe")
    p2 = Pipeline.from_json(out, memory_guard=MemoryGuard(max_memory_mb=200))
    y_after = _collect_col(p2.predict(test), "y_pred")
    diff_rt = float(np.max(np.abs(y_after - y_before)))
    print(f"\n=== to_json(train-fit) -> from_json -> predict(test): "
          f"max_diff={diff_rt:.2e} ===")
    np.testing.assert_allclose(y_after, y_before, rtol=1e-9)

    report("test_pipeline_train_test_split",
           n_train=train.count(), n_test=test.count(),
           mean_x1_train=mean_x1_tr, coef_z0=float(coef_z_exp[0]),
           max_diff_predict_test=max_diff, max_diff_roundtrip=diff_rt)

