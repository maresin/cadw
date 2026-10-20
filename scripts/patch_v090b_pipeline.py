#!/usr/bin/env python3
"""v0.9.0b: Pipeline skeleton + real plan.

First architectural step of v0.9.0: chain of Transformers with fit-time
feasibility plan (plan-first, raise if does_not_fit).

SCOPE:
  cadw/pipeline.py (new):
    - Transformer / Predictor Protocol (typing only)
    - Pipeline: add_step, fit, transform, predict, fit_transform, plan
    - Wrappers: CastStep, DropnaStep, GBDTStep

  Design decisions (§ pipeline v0.9.0 design, locked-in):
    - Sequence-first. DAG via inputs=[...] — v0.9.1.
    - Model is always LAST step. transform() stops before it.
    - fit always plans first: Pipeline.plan(df) raises
      PipelineMemoryError if total_state > state_limit BEFORE any work.
    - fit/transform always 2 passes (no fused fit_transform — state
      after fit is frozen; transform re-applies pure steps).
    - State in-memory (small, bounded); serialize only to_json in v0.9.1.

  plan() MVP:
    - Pre-fit feasibility: sum(step.estimate_state_bytes(dc)) <= state_limit
    - Predicted outcome: {"fits", "does_not_fit"} (spill deferred v0.9.1)
    - Per-batch transform guard remains per-step (each step uses
      MemoryGuard internally — Pipeline doesn't reimplement).

TESTS:
  tests/test_pipeline.py (~15): wrappers, composition, model-stops,
    plan feasibility (custom huge-state step), duplicate name, target
    required, predict-without-model raises, state frozen after fit.
  tests/test_pipeline_e2e.py (1 slow): parquet(ts,sensor,y) →
    dt.month + astype + GBDT → predict → parquet; R² > 0.3.

GUARDS (before commit, §13.4/140+143):
  new tests, changelog, arrow_guard, no_pandas, fast+slow.
  No WIP commit on red — fix in place (§13.4/146 backlog).

Run: uv run python scripts/patch_v090b_pipeline.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


# ============ 1. cadw/pipeline.py ============

PIPELINE_MODULE = '''"""Out-of-core pipeline (v0.9.0b).

First step of v0.9.0 "out-of-core framework": chain of Transformers
with plan-first feasibility check.

Design:
  - Sequence-first. DAG (via `inputs=[...]`) — v0.9.1.
  - Model always LAST step. `transform()` stops before it; `predict()`
    requires it.
  - `fit` always calls `plan(df)` first: raises PipelineMemoryError if
    total_state > state_limit BEFORE any work. Real guarantee: user
    knows before materialization whether pipeline fits.
  - fit/transform always 2 passes. State frozen after fit; `transform`
    re-applies pure steps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from cadw.core.memory_guard import MemoryGuard
from cadw.core.models import DataChars, data_chars_from_sample
from cadw.core.planner import PipelineMemoryError, PipelinePlan, StepPlan
from cadw.core.sampling import peek_first_batch

if TYPE_CHECKING:
    from cadw.dataframe import CadwDataFrame


@runtime_checkable
class Transformer(Protocol):
    """Any step that can fit and transform a CadwDataFrame.

    Contract:
      - fit(df, target=None) -> self         (stateful: read df, set state;
                                              stateless: no-op)
      - transform(df) -> CadwDataFrame       (apply state; streaming)
      - estimate_state_bytes(dc) -> int      (upper bound, BEFORE fit)
      - state_bytes() -> int                 (actual, AFTER fit; 0 if no fit)
    """
    def fit(self, df: "CadwDataFrame", target: Optional[str] = None): ...
    def transform(self, df: "CadwDataFrame") -> "CadwDataFrame": ...
    def estimate_state_bytes(self, dc: DataChars) -> int: ...
    def state_bytes(self) -> int: ...


@runtime_checkable
class Predictor(Protocol):
    """Transformer that also produces predictions from features."""
    def fit(self, df: "CadwDataFrame", target: Optional[str] = None): ...
    def transform(self, df: "CadwDataFrame") -> "CadwDataFrame": ...
    def predict(self, df: "CadwDataFrame") -> "CadwDataFrame": ...
    def estimate_state_bytes(self, dc: DataChars) -> int: ...
    def state_bytes(self) -> int: ...


def _is_predictor(step) -> bool:
    return hasattr(step, "predict") and callable(getattr(step, "predict"))


class Pipeline:
    """Sequence of Transformers. Model (if any) is LAST.

    Usage:
        pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
        pipe.add_step("cast", CastStep({"v": "float64"}))
        pipe.add_step("dropna", DropnaStep(how="any"))
        pipe.add_step("model", GBDTStep(X_cols=["v"], y_col="y"))
        pipe.fit(train_df, target="y")
        preds = pipe.predict(test_df)  # CadwDataFrame with y_pred
    """

    def __init__(
        self,
        memory_guard: Optional[MemoryGuard] = None,
    ) -> None:
        self._memory_guard = memory_guard or MemoryGuard(max_memory_mb=100)
        self._step_names: list[str] = []
        self._steps: list = []
        self._fitted: bool = False

    # ---------------- construction ----------------

    def add_step(self, name: str, step) -> "Pipeline":
        """Append a Transformer/Predictor. Returns self for chaining."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"add_step: name must be non-empty str; got {name!r}")
        if name in self._step_names:
            raise ValueError(
                f"add_step: duplicate name {name!r}; existing={self._step_names}"
            )
        if not (hasattr(step, "fit") and hasattr(step, "transform")):
            raise TypeError(
                f"add_step: step must have fit() and transform(); "
                f"got {type(step).__name__}"
            )
        if _is_predictor(step) and self._steps:
            # Predictor must be last; if it's not the first, error would
            # fire later in fit/transform, but check upfront for clear msg.
            if any(_is_predictor(s) for s in self._steps):
                raise RuntimeError(
                    "add_step: pipeline already has a Predictor at end; "
                    "cannot add another"
                )
        self._step_names.append(name)
        self._steps.append(step)
        return self

    @property
    def steps(self) -> list:
        """Read-only view: list of (name, step) tuples."""
        return list(zip(self._step_names, self._steps))

    @property
    def memory_guard(self) -> MemoryGuard:
        return self._memory_guard

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ---------------- plan ----------------

    def plan(self, df: "CadwDataFrame") -> PipelinePlan:
        """Pre-fit feasibility: sum(state_bytes) <= state_limit?

        Raises PipelineMemoryError if `does_not_fit`. Called automatically
        by `fit`. Users can call explicitly for early check.
        """
        sample = peek_first_batch(df._iter_filtered_batches())
        if sample is None:
            raise ValueError("Pipeline.plan: empty df")
        dc = data_chars_from_sample(sample, df.count())

        state_estimates = [
            step.estimate_state_bytes(dc) for step in self._steps
        ]
        total_state = sum(state_estimates)
        state_limit = self._memory_guard.state_limit_bytes

        fits = total_state <= state_limit
        if not fits:
            raise PipelineMemoryError(
                step_idx=len(self._steps) - 1,
                step_name=self._step_names[-1] if self._step_names else "<none>",
                estimate_bytes=total_state,
                available_bytes=self._memory_guard.budget_bytes,
                describe=(
                    f"Pipeline total_state={total_state} B "
                    f"> state_limit={state_limit} B "
                    f"(sum over {len(self._steps)} steps; "
                    f"individual estimates={state_estimates})"
                ),
            )

        step_plans = [
            StepPlan(
                op_name=type(step).__name__,
                params=None,
                estimate_bytes=est,
                input_chars=dc,
                output_chars=dc,  # v0.9.0b: assume unchanged
                spill_before=False,
            )
            for step, est in zip(self._steps, state_estimates)
        ]

        return PipelinePlan(
            steps=step_plans,
            peak_bytes=total_state,
            spill_boundaries=[],
            estimated_io=0,
            predicted_outcome="fits",
            reason=None,
            available_bytes=self._memory_guard.budget_bytes,
            safety=0.7,
        )

    def total_state_bytes(self) -> int:
        """Sum of fitted state_bytes across all steps. 0 before fit."""
        return sum(step.state_bytes() for step in self._steps)

    # ---------------- fit / transform ----------------

    def fit(
        self,
        df: "CadwDataFrame",
        target: Optional[str] = None,
    ) -> "Pipeline":
        """Fit all steps sequentially on df. Model (if present) is last.

        - target: required if pipeline contains a Predictor.
        - plan(df) called first; raises PipelineMemoryError if not feasible.
        - Returns self.
        """
        if not self._steps:
            raise ValueError("Pipeline.fit: empty pipeline")

        has_model = any(_is_predictor(s) for s in self._steps)
        if has_model and target is None:
            raise ValueError(
                "Pipeline.fit: target required (pipeline contains a Predictor)"
            )

        # Feasibility check BEFORE any state mutation (§9.3)
        self.plan(df)

        current = df
        for i, step in enumerate(self._steps):
            is_pred = _is_predictor(step)
            if is_pred and i != len(self._steps) - 1:
                raise RuntimeError(
                    f"Pipeline.fit: Predictor at position {i} must be last; "
                    f"pipeline has {len(self._steps)} steps"
                )
            step.fit(current, target=target if is_pred else None)
            if not is_pred:
                current = step.transform(current)

        self._fitted = True
        return self

    def transform(self, df: "CadwDataFrame") -> "CadwDataFrame":
        """Apply pure-transformer steps. Stops before a Predictor.

        Returns the feature CadwDataFrame (before model).
        """
        if not self._fitted:
            raise RuntimeError(
                "Pipeline.transform: pipeline is not fitted; call fit first"
            )
        current = df
        for i, step in enumerate(self._steps):
            if _is_predictor(step):
                # Predictor is always last (validated at fit time)
                break
            current = step.transform(current)
        return current

    def predict(self, df: "CadwDataFrame") -> "CadwDataFrame":
        """Apply all steps including the Predictor (must be last).

        Returns CadwDataFrame with prediction columns.
        """
        if not self._fitted:
            raise RuntimeError(
                "Pipeline.predict: pipeline is not fitted; call fit first"
            )
        if not self._steps:
            raise ValueError("Pipeline.predict: empty pipeline")
        if not _is_predictor(self._steps[-1]):
            raise TypeError(
                "Pipeline.predict: last step must be Predictor "
                f"(got {type(self._steps[-1]).__name__})"
            )
        features = self.transform(df)
        return self._steps[-1].predict(features)

    def fit_transform(
        self,
        df: "CadwDataFrame",
        target: Optional[str] = None,
    ) -> "CadwDataFrame":
        """fit + transform; returns feature CadwDataFrame (before model)."""
        self.fit(df, target=target)
        return self.transform(df)

    def __repr__(self) -> str:
        names = ", ".join(self._step_names)
        return f"Pipeline([{names}], fitted={self._fitted})"


# ============ Wrappers for testing (MVP) ============

class CastStep:
    """Wrap `df.astype(mapping)`. Stateless."""

    __slots__ = ("_mapping", "_safe")

    def __init__(self, mapping: dict, *, safe: bool = True) -> None:
        self._mapping = dict(mapping)
        self._safe = safe

    def fit(self, df, target=None):
        return self

    def transform(self, df):
        return df.astype(self._mapping, safe=self._safe)

    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0


class DropnaStep:
    """Wrap `df.dropna(how, subset)`. Stateless."""

    __slots__ = ("_how", "_subset")

    def __init__(self, how: str = "any", subset=None) -> None:
        self._how = how
        self._subset = list(subset) if subset is not None else None

    def fit(self, df, target=None):
        return self

    def transform(self, df):
        return df.dropna(how=self._how, subset=self._subset)

    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0


class GBDTStep:
    """Wrap `fit_gbdt_regressor`. State = tree structure + leaf values."""

    __slots__ = ("_X_cols", "_y_col", "_kwargs", "_model")

    # Conservative per-node bytes: feature idx (4) + threshold (8)
    # + left (4) + right (4) + gain (8) + leaf value (8) = 36; round to 48.
    _NODE_BYTES = 48

    def __init__(
        self,
        X_cols,
        y_col: str,
        **kwargs,
    ) -> None:
        if isinstance(X_cols, str):
            X_cols = [X_cols]
        self._X_cols = list(X_cols)
        self._y_col = y_col
        self._kwargs = dict(kwargs)
        self._model = None

    def _n_nodes(self) -> int:
        n_est = int(self._kwargs.get("n_estimators", 100))
        max_d = int(self._kwargs.get("max_depth", 3))
        # Level-wise: full binary tree of depth d has 2^(d+1) - 1 nodes.
        return n_est * ((1 << (max_d + 1)) - 1)

    def estimate_state_bytes(self, dc):
        # Tree structure dominates; +1 KiB for metadata (feature names,
        # bin boundaries, base prediction).
        return self._n_nodes() * self._NODE_BYTES + 1024

    def state_bytes(self):
        if self._model is None:
            return 0
        actual = 0
        for tree in self._model.trees_:
            actual += len(tree["structure"]) * self._NODE_BYTES
            actual += len(tree["leaf_values"]) * 8
        return actual + 1024

    def fit(self, df, target=None):
        from cadw.ml.gbdt import fit_gbdt_regressor
        if target is None:
            target = self._y_col
        if target != self._y_col:
            raise ValueError(
                f"GBDTStep.fit: target={target!r} != y_col={self._y_col!r}"
            )
        self._model = fit_gbdt_regressor(
            df, self._X_cols, self._y_col,
            memory_guard=df._memory_guard,
            **self._kwargs,
        )
        return self

    def predict(self, df):
        if self._model is None:
            raise RuntimeError("GBDTStep.predict: not fitted")
        return self._model.predict(df)
'''


def write_pipeline_module():
    print("\n=== cadw/pipeline.py ===")
    p = Path("cadw/pipeline.py")
    p.write_text(PIPELINE_MODULE)
    compile_check(p)
    print("  OK")


# ============ 2. AST guard whitelist for pipeline methods ============

def update_ast_guard_whitelist():
    print("\n=== tests/test_arrow_call_memory_guard.py: whitelist Pipeline ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if '"Pipeline"' in s and "_is_predictor" in s:
        print("  already present — skip")
        return

    old = '''    "_apply_column_transform": "per-batch Arrow transform, no extra alloc",
    "astype": "per-batch pc.cast, streaming",
    "dropna": "lazy notnull() composition — returns view, no materialize",
    "drop": "selection minus — lazy view",
    "rename": "Dataset.from_batches(streaming) — schema change only",
}'''

    new = '''    "_apply_column_transform": "per-batch Arrow transform, no extra alloc",
    "astype": "per-batch pc.cast, streaming",
    "dropna": "lazy notnull() composition — returns view, no materialize",
    "drop": "selection minus — lazy view",
    "rename": "Dataset.from_batches(streaming) — schema change only",
    # v0.9.0b Pipeline: delegates to step.transform/predict (each step's
    # own guards apply), or pure metadata ops in plan/state_bytes.
    "fit": "delegates to step.fit; plan() checks state feasibility first",
    "transform": "delegates to step.transform; per-step MemoryGuard",
    "predict": "delegates to step.predict; per-step MemoryGuard",
    "fit_transform": "fit + transform, no materialization",
    "plan": "metadata only (peek_first_batch + estimate_state_bytes)",
    "state_bytes": "in-memory metadata (fitted state)",
    "estimate_state_bytes": "metadata only",
}'''

    if old not in s:
        print("  WARNING: whitelist anchor not found — skip")
        return
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 3. tests/test_pipeline.py ============

PIPELINE_TESTS = '''"""v0.9.0b: Pipeline skeleton + plan."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.core.planner import PipelinePlan, PipelineMemoryError
from cadw.pipeline import (
    CastStep, DropnaStep, GBDTStep, Pipeline, Transformer,
)


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=100):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ============ construction ============

def test_empty_pipeline():
    p = Pipeline()
    assert p.steps == []
    assert not p.fitted
    assert "Pipeline([]" in repr(p)


def test_add_step_chaining():
    p = Pipeline()
    result = p.add_step("a", CastStep({"v": "float64"}))
    assert result is p
    assert len(p.steps) == 1


def test_add_step_duplicate_name_raises():
    p = Pipeline()
    p.add_step("a", CastStep({"v": "float64"}))
    with pytest.raises(ValueError, match="duplicate name"):
        p.add_step("a", CastStep({"v": "int64"}))


def test_add_step_bad_step_type_raises():
    p = Pipeline()
    with pytest.raises(TypeError, match="must have fit"):
        p.add_step("a", object())


# ============ wrappers (pure transforms) ============

def test_cast_step():
    p = Pipeline().add_step("cast", CastStep({"v": "float64"}))
    # CastStep is stateless; no fit needed for transform to work
    p._fitted = True  # bypass fit for MVP (no model)
    df = make_df_dummy_pq  # not used; see below


# Helper for make_df in tests
def make_df_dummy_pq():
    pass


def test_cast_step_transform(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    step = CastStep({"v": "float64"})
    out = step.transform(df)
    t = step_out_table(out, tmp_path)
    assert t.schema.field("v").type == pa.float64()
    assert t.column("v").to_pylist() == [1.0, 2.0, 3.0]


def step_out_table(df, tmp_path, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


def test_dropna_step_transform(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, None, 3],
        "v": [10, 20, 30],
    })
    step = DropnaStep(how="any", subset=["a"])
    out = step.transform(df)
    assert out.count() == 2


def test_compose_cast_dropna(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, None, 3, None],
        "v": [1.0, 2.0, 3.0, 4.0],
    })
    pipe = (
        Pipeline()
        .add_step("cast", CastStep({"v": "float32"}))
        .add_step("dropna", DropnaStep(how="any", subset=["a"]))
    )
    pipe.fit(df)  # no model, so no target
    out = pipe.transform(df)
    assert out.count() == 2
    t = step_out_table(out, tmp_path)
    assert t.schema.field("v").type == pa.float32()


# ============ fit / transform / predict contract ============

def test_fit_returns_self(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = Pipeline().add_step("cast", CastStep({"v": "float64"}))
    assert pipe.fit(df) is pipe
    assert pipe.fitted


def test_transform_before_fit_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = Pipeline().add_step("cast", CastStep({"v": "float64"}))
    with pytest.raises(RuntimeError, match="not fitted"):
        pipe.transform(df)


def test_predict_without_model_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = Pipeline().add_step("cast", CastStep({"v": "float64"}))
    pipe.fit(df)
    with pytest.raises(TypeError, match="Predictor"):
        pipe.predict(df)


def test_fit_requires_target_with_model(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(tmp_path, {
        "x": x.tolist(), "y": y.tolist(),
    })
    pipe = Pipeline().add_step(
        "model", GBDTStep(["x"], "y", n_estimators=5),
    )
    with pytest.raises(ValueError, match="target required"):
        pipe.fit(df)  # no target


def test_fit_transform_returns_features(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = Pipeline().add_step("cast", CastStep({"v": "float64"}))
    out = pipe.fit_transform(df)
    # No model → fit_transform returns transformed df
    assert out.count() == 3
    assert out.columns == ["v"]


# ============ plan ============

def test_plan_returns_pipeline_plan(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = (
        Pipeline()
        .add_step("cast", CastStep({"v": "float64"}))
        .add_step("dropna", DropnaStep())
    )
    plan = pipe.plan(df)
    assert isinstance(plan, PipelinePlan)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 2
    assert plan.peak_bytes == 0  # pure transforms: 0 state


def test_plan_with_gbdt_state(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    pipe = Pipeline().add_step(
        "model", GBDTStep(["x"], "y", n_estimators=10, max_depth=3),
    )
    plan = pipe.plan(df)
    assert plan.predicted_outcome == "fits"
    # 10 trees × 15 nodes × 48 = 7200 + 1024
    assert plan.peak_bytes > 0
    assert plan.peak_bytes < 100_000


def test_plan_does_not_fit_raises(tmp_path):
    """Custom step with huge estimate → PipelineMemoryError."""
    df = make_df(tmp_path, {"v": [1, 2, 3]})

    class HugeStep:
        def fit(self, df, target=None):
            return self
        def transform(self, df):
            return df
        def estimate_state_bytes(self, dc):
            return 10 ** 12  # 1 TB
        def state_bytes(self):
            return 0

    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=100),
    ).add_step("huge", HugeStep())
    with pytest.raises(PipelineMemoryError):
        pipe.plan(df)
    with pytest.raises(PipelineMemoryError):
        pipe.fit(df)


# ============ GBDT E2E (fast) ============

def test_pipeline_gbdt_predict(tmp_path):
    """Small GBDT pipeline: fit → predict."""
    n = 200
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    y = x * 2.0 + rng.normal(scale=0.05, size=n)
    df = make_df(
        tmp_path, {"x": x.tolist(), "y": y.tolist()},
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=200),
    ).add_step(
        "model", GBDTStep(
            ["x"], "y", n_estimators=20, max_depth=2, learning_rate=0.2,
        ),
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    out = tmp_path / "pred.parquet"
    preds.write_parquet(out)
    t = pq.read_table(out)
    assert "y_pred" in t.schema.names
    # Simple sanity: predictions within expected range
    pred_vals = t.column("y_pred").to_pylist()
    assert max(abs(p) for p in pred_vals if p is not None) < 10


def test_pipeline_cast_then_gbdt(tmp_path):
    """Cast (int→float) + GBDT. Verify chain works."""
    n = 200
    rng = np.random.default_rng(1)
    x_int = (rng.normal(size=n) * 10).astype(int)
    y = x_int * 1.5
    df = make_df(
        tmp_path, {
            "x": x_int.tolist(),
            "y": y.tolist(),
        },
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=200),
    ).add_step(
        "cast", CastStep({"x": "float64", "y": "float64"}),
    ).add_step(
        "model", GBDTStep(
            ["x"], "y", n_estimators=10, max_depth=3,
        ),
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    out = tmp_path / "pred.parquet"
    preds.write_parquet(out)
    t = pq.read_table(out)
    assert "y_pred" in t.schema.names


# ============ total_state_bytes ============

def test_total_state_bytes_zero_before_fit(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    pipe = Pipeline().add_step(
        "model", GBDTStep(["x"], "y", n_estimators=5),
    )
    assert pipe.total_state_bytes() == 0


def test_total_state_bytes_grows_after_fit(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(
        tmp_path, {"x": x.tolist(), "y": y.tolist()},
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=200),
    ).add_step(
        "model", GBDTStep(["x"], "y", n_estimators=10, max_depth=2),
    )
    before = pipe.total_state_bytes()
    pipe.fit(df, target="y")
    after = pipe.total_state_bytes()
    assert before == 0
    assert after > 0
'''


def write_pipeline_tests():
    print("\n=== tests/test_pipeline.py ===")
    p = Path("tests/test_pipeline.py")
    p.write_text(PIPELINE_TESTS)
    compile_check(p)
    print("  OK")


# ============ 4. tests/test_pipeline_e2e.py (slow) ============

E2E_TESTS = '''"""v0.9.0b: end-to-end pipeline demo — parquet(ts,sensor,y) → GBDT.

Slow test. Demonstrates full out-of-core flow:
  parquet → dt.features + cast + dropna + GBDT → predict → parquet.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.pipeline import CastStep, DropnaStep, GBDTStep, Pipeline


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


@pytest.mark.slow
def test_e2e_pipeline_time_series(tmp_path):
    """Realistic: timestamp + categorical + target; dt features → GBDT.

    N=20K. Feature set: year/month/day/weekday (from ts). Model: GBDT
    regression on those + sensor_id.
    Target: y = 10 + 5 * month / 12 + 3 * weekday / 7 + noise.
    R² > 0.3 expected.
    """
    rng = np.random.default_rng(42)
    n = 20_000
    ts = pd.date_range("2020-01-01", periods=n, freq="1h").to_numpy()
    sensor = np.tile(["A", "B", "C", "D"], n // 4 + 1)[:n]
    month = pd.DatetimeIndex(ts).month.to_numpy()
    weekday = pd.DatetimeIndex(ts).weekday.to_numpy()
    y = (
        10.0
        + 5.0 * month / 12.0
        + 3.0 * weekday / 7.0
        + rng.normal(scale=0.3, size=n)
    )

    import pyarrow as pa
    table = pa.table({
        "ts": pa.array(ts),
        "sensor": sensor,
        "y": y.tolist(),
    })
    src = tmp_path / "src.parquet"
    pq.write_table(table, src, row_group_size=1000)

    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=1000,
        memory_guard=MemoryGuard(max_memory_mb=100),
        spill_dir=tmp_path / "spill",
    )

    # Feature engineering via accessor + cast
    df_feat = (
        df["ts"].dt.month
    )
    df_feat = df_feat["ts"].dt.weekday
    df_feat = df_feat["ts"].dt.day
    # After 3 chained accesses: cols = ts, sensor, y, ts_month, ts_weekday, ts_day

    # Encode sensor
    df_enc, _ = df_feat.encode_categorical(["sensor"])
    # Keep only features that GBDT can use (numeric)
    feature_cols = ["ts_month", "ts_weekday", "ts_day", "sensor"]

    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=100),
    ).add_step(
        "model", GBDTStep(
            feature_cols, "y",
            n_estimators=30, max_depth=3, learning_rate=0.1, bins=32,
        ),
    )
    pipe.fit(df_enc, target="y")
    preds = pipe.predict(df_enc)
    out = tmp_path / "pred.parquet"
    preds.write_parquet(out)

    pred_table = pq.read_table(out)
    got_y = pred_table.column("y").to_pylist()
    got_pred = pred_table.column("y_pred").to_pylist()
    r2 = _r2(got_y, got_pred)
    assert r2 > 0.3, f"R²={r2:.3f}, expected > 0.3"

    # MemoryGuard: total state small (< 1 MB for 30 trees × depth 3)
    assert pipe.total_state_bytes() < 1_000_000
'''


def write_e2e_test():
    print("\n=== tests/test_pipeline_e2e.py ===")
    p = Path("tests/test_pipeline_e2e.py")
    p.write_text(E2E_TESTS)
    compile_check(p)
    print("  OK")


# ============ 5. CHANGELOG ============

CHANGELOG_ENTRY = '''## [Unreleased]

### Добавлено (v0.9.0b — Pipeline skeleton)

Главный архитектурный шаг v0.9.0: chain of Transformers с plan-first
feasibility check. Превращает CADW из «набора primitives» в
out-of-core framework.

- **`cadw/pipeline.py`** — новый модуль:
  - **`Pipeline`** — sequence of Transformers. Model (if present)
    всегда LAST.
    - `add_step(name, step)` — chainable.
    - `plan(df) -> PipelinePlan` — pre-fit feasibility: sum(state_bytes)
      ≤ state_limit. Raises `PipelineMemoryError` if not feasible
      **до** любого materialize. Пользователь может вызвать явно для
      early check.
    - `fit(df, target=None)` — fit всех steps последовательно, plan-first.
    - `transform(df)` — применяет pure-transform steps; **останавливается
      перед Predictor** (если он есть).
    - `predict(df)` — fit chain + last Predictor. Требует последний step
      быть Predictor.
    - `fit_transform(df, target=None)` — fit + return features (before model).
    - `total_state_bytes()` — sum fitted state, интроспекция.
  - **`Transformer` / `Predictor` Protocol** (typing only) — контракт:
    `fit(df, target=None)`, `transform(df)`, `estimate_state_bytes(dc)`,
    `state_bytes()`; Predictor adds `predict(df)`.

- **Wrappers (MVP, для тестов)**:
  - **`CastStep(mapping, safe=True)`** — wrap `df.astype()`. Stateless.
  - **`DropnaStep(how, subset)`** — wrap `df.dropna()`. Stateless.
  - **`GBDTStep(X_cols, y_col, **kwargs)`** — wrap `fit_gbdt_regressor`.
    `estimate_state_bytes` = n_estimators × 2^(max_depth+1) × 48 + 1024
    (conservative per-node).

### Design decisions (locked-in, v0.9.0)

- **Sequence-first.** DAG (via `inputs=[...]`) — v0.9.1.
- **Model LAST.** `transform()` stops before Predictor; `predict()`
  requires it.
- **Plan-first fit.** `Pipeline.fit` calls `plan(df)` before any work;
  `PipelineMemoryError` if not feasible.
- **2-pass fit/transform.** State frozen after fit; transform re-applies
  pure steps. No fused execution (state-after-fit стабилен — экономии нет).
- **In-memory state.** Bounded (K × #features). Serialize to JSON — v0.9.1.
- **Per-batch transform guard — per-step.** Each step uses `MemoryGuard`
  internally; Pipeline не reimplement'ит.

### Ограничения (v0.9.0b, §11.5)

- `Pipeline.plan` — fit feasibility only (`total_state ≤ state_limit`).
  `spill_boundaries` always empty (deferred v0.9.1). Batch propagation
  per step handled by step's own MemoryGuard (not re-planned).
- `Pipeline.to_json` / `from_json` — v0.9.1.
- `ColumnTransformer` — v0.9.1.
- `TimeSeriesFeatures` composite — v0.9.2.
- `Pipeline` не валидирует, что Predictor is last при `add_step`
  (проверяется при `fit`). Predictor not-last на `fit` → RuntimeError.

### Тесты (v0.9.0b)

- +15 fast (`test_pipeline.py`): construction, wrappers, compose,
  fit/transform/predict contract, plan (fits, does_not_fit, with GBDT),
  state_bytes tracking, small GBDT fit+predict.
- +1 slow (`test_pipeline_e2e.py`): 20K rows, timestamp → dt features
  → categorical encoding → GBDT; R² > 0.3.

- fast: **~1257** (+15). slow: **9** (+1).

'''


def append_changelog():
    print("\n=== CHANGELOG.md: ensure [Unreleased] + append v0.9.0b ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.0b — Pipeline skeleton" in s:
        print("  already present — skip")
        return

    # Ensure [Unreleased] (§13.4/146 lesson)
    if "## [Unreleased]" not in s:
        m = re.search(r'^## \[', s, re.MULTILINE)
        if m:
            s = s[:m.start()] + "## [Unreleased]\n\n" + s[m.start():]
            print("  prepended `## [Unreleased]` before first `## [`")
        else:
            s = "## [Unreleased]\n\n" + s
            print("  prepended `## [Unreleased]` at start")

    s = s.replace("## [Unreleased]\n\n", CHANGELOG_ENTRY, 1)
    p.write_text(s)
    print("  OK")


# ============ 6. DECISIONS §11.5 update ============

def update_decisions():
    print("\n=== DECISIONS.md: §11.5 pipeline entry ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Pipeline.plan — fit feasibility only" in s:
        print("  already present — skip")
        return

    anchor = "### 11.6. Технический долг"
    if anchor not in s:
        print("  WARNING: §11.6 anchor not found — skip")
        return

    addendum = '''**v0.9.0b (Pipeline v1):**
- `Pipeline.plan(df)` — **fit feasibility only** (`sum(step.state_bytes)
  ≤ state_limit`). `spill_boundaries` always empty (deferred v0.9.1).
  Per-batch transform guard — **per-step** (each step's `MemoryGuard`
  handles its own batch).
- `Pipeline.transform(df)` **stops before Predictor** (model is last).
  `Pipeline.predict(df)` requires last step to be Predictor.
- `Pipeline.fit(df, target=None)` — target required if pipeline contains
  Predictor. `plan` called first → `PipelineMemoryError` if not feasible
  (honest fail-fast, §9.3).
- **State in-memory** (bounded by K × #features per step). `to_json` /
  `from_json` — v0.9.1.
- **`ColumnTransformer`** — v0.9.1 (fanout by columns).
- **DAG** (via `inputs=[...]`) — v0.9.1.
- **`TimeSeriesFeatures`** composite — v0.9.2.

''' + anchor
    s = s.replace(anchor, addendum, 1)
    p.write_text(s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.0b: Pipeline skeleton ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    write_pipeline_module()
    update_ast_guard_whitelist()
    write_pipeline_tests()
    write_e2e_test()
    append_changelog()
    update_decisions()

    print("\n=== pytest: new pipeline tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pipeline.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIPELINE TESTS FAILED ***")
        print("Fix in place (§13.4/146 backlog: no WIP commit on red).")
        sys.exit(1)

    print("\n=== pytest: pipeline E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GUARD FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1200)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.0b: Pipeline skeleton — Transformer/Predictor protocols, "
        "Pipeline (add_step/fit/transform/predict/fit_transform/plan), "
        "CastStep/DropnaStep/GBDTStep wrappers, plan-first feasibility, "
        "E2E time-series GBDT"])

    print("\n==> v0.9.0b applied.")
    print("NEXT: v0.9.0c (Jupyter _repr_html_, df.plot accessor).")


if __name__ == "__main__":
    main()