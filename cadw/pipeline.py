"""Out-of-core pipeline (v0.9.0b).

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
        # v0.9.0b-fix1: Transformer contract = fit + (transform OR predict).
        # A Predictor is a step with fit + predict; its transform is never
        # called (Pipeline.transform stops before Predictor; Pipeline.predict
        # uses step.predict on the last step). Requiring transform on
        # Predictors broke GBDTStep (has fit+predict, no transform).
        has_fit = hasattr(step, "fit")
        has_transform = hasattr(step, "transform")
        has_predict = hasattr(step, "predict")
        if not (has_fit and (has_transform or has_predict)):
            raise TypeError(
                f"add_step: step must have fit() and (transform() or "
                f"predict()); got {type(step).__name__}"
            )
        # v0.9.0b-fix1: Predictor must be LAST. Two checks:
        #   (a) new step is Predictor, existing steps contain Predictor → error
        #   (b) existing last is Predictor, new step is non-Predictor → error
        new_is_pred = _is_predictor(step)
        if new_is_pred and any(_is_predictor(s) for s in self._steps):
            raise RuntimeError(
                "add_step: pipeline already has a Predictor; "
                "cannot add another (model must be last)"
            )
        if self._steps and _is_predictor(self._steps[-1]) and not new_is_pred:
            raise RuntimeError(
                "add_step: cannot add step after a Predictor "
                "(model must be last)"
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

    def _repr_html_(self) -> str:
        """Jupyter HTML repr — step table with per-step state_bytes."""
        if not self._steps:
            return "<div class='cadw-pipeline'>Pipeline (empty)</div>"

        total_state = self.total_state_bytes()
        rows = []
        for i, (name, step) in enumerate(zip(self._step_names, self._steps)):
            kind = "Predictor" if _is_predictor(step) else "Transformer"
            try:
                sb = step.state_bytes() if self._fitted else 0
            except Exception:
                sb = "?"
            state_str = f"{sb:,} B" if isinstance(sb, int) else str(sb)
            rows.append(
                f"<tr><td>{i + 1}</td><td>{name}</td>"
                f"<td>{type(step).__name__}</td>"
                f"<td>{kind}</td><td>{state_str}</td></tr>"
            )
        fitted_badge = (
            "<span style='color:green'>fitted</span>"
            if self._fitted
            else "<span style='color:#999'>not fitted</span>"
        )
        return (
            f"<div class='cadw-pipeline'>"
            f"<strong>Pipeline</strong> ({len(self._steps)} steps, "
            f"{fitted_badge}, total_state={total_state:,} B)"
            f"<table><tr><th>#</th><th>name</th><th>type</th>"
            f"<th>role</th><th>state</th></tr>"
            f"{''.join(rows)}</table></div>"
        )


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
