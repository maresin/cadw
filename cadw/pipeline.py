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

import json

from pathlib import Path  # v0.9.9 (B3): for `-> Path` annotation
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from cadw.core.memory_guard import MemoryGuard
from cadw.core.models import DataChars, data_chars_from_sample
from cadw.core.planner import PipelineMemoryError, PipelinePlan, StepPlan
from cadw.core.sampling import peek_first_batch

if TYPE_CHECKING:
    from cadw.dataframe import CadwDataFrame


# v0.9.1a (§13.4/147 backlog): strict Protocol separation.
# Transformer and Predictor are NOT nested (Predictor is NOT a
# Transformer subclass) — they have disjoint primary methods:
#   - Transformer: fit + transform (pure feature engineering)
#   - Predictor:   fit + predict   (model; transform never called)
# Both share state-method contract (estimate_state_bytes, state_bytes).
# Pipeline validates structure via `_is_predictor(step) = hasattr(step,
# "predict")`, not via isinstance — duck-typing keeps Protocol minimal.

@runtime_checkable
class Transformer(Protocol):
    """Feature engineering step: fit → transform.

    Contract:
      - fit(df, target=None) -> self         (read df, set state; or no-op)
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
    """Model step: fit → predict. `transform` NOT required.

    Pipeline's `transform()` stops BEFORE Predictor; `predict()` calls
    Predictor.predict() on the last step. So Predictor never needs
    `.transform()` — strictly minimal contract per §13.4/147.

    Contract:
      - fit(df, target=None) -> self
      - predict(df) -> CadwDataFrame
      - estimate_state_bytes(dc) -> int
      - state_bytes() -> int
    """
    def fit(self, df: "CadwDataFrame", target: Optional[str] = None): ...
    def predict(self, df: "CadwDataFrame") -> "CadwDataFrame": ...
    def estimate_state_bytes(self, dc: DataChars) -> int: ...
    def state_bytes(self) -> int: ...


def _is_predictor(step) -> bool:
    return hasattr(step, "predict") and callable(getattr(step, "predict"))

def _merge_fanout(dfs: list) -> "CadwDataFrame":
    """Merge DAG fan-in outputs by column union (row-aligned concat).

    Preconditions: all `dfs` have the SAME row count (positional alignment).
    Result: columns from dfs[0] (base) + unique new columns from dfs[1:]
    (in order, duplicates skipped).

    v0.9.1b MVP: row-selection fan-in (where branches drop different rows)
    is NOT supported — raises RuntimeError with count mismatch. See §11.5.
    """
    from cadw import CadwDataFrame
    from cadw.dataset import Dataset
    import pyarrow as pa

    if not dfs:
        raise ValueError("_merge_fanout: empty list")
    if len(dfs) == 1:
        return dfs[0]

    # Row-count alignment check
    n_first = dfs[0].count()
    for i, d in enumerate(dfs[1:], start=1):
        n_i = d.count()
        if n_i != n_first:
            raise RuntimeError(
                f"_merge_fanout: row count mismatch between branches: "
                f"branch 0 has {n_first} rows, branch {i} has {n_i}. "
                f"Fan-in MVP requires same row count — branches must not "
                f"drop rows (e.g. dropna) before merge. See §11.5."
            )

    # Build output schema: base (dfs[0]) + new columns from others
    base_cols = list(dfs[0].columns)
    base_cols_set = set(base_cols)
    # v0.9.9 (N5): track NEW column -> first branch index. Duplicate
    # new column from two branches → clear RuntimeError (prior
    # silent dedup led to "Schema and number of arrays unequal"
    # in RecordBatch.from_arrays, cryptic downstream).
    seen_new: dict = {}
    out_fields = [dfs[0]._dataset.schema.field(c) for c in base_cols]
    for i, d in enumerate(dfs[1:], start=1):
        for c in d.columns:
            if c in base_cols_set:
                continue
            if c in seen_new:
                raise RuntimeError(
                    f"_merge_fanout: duplicate new column {c!r} from "
                    f"DAG branches {seen_new[c]} and {i}. Rename one "
                    f"of them (or wrap in ColumnTransformer with "
                    f"distinct feature names)."
                )
            seen_new[c] = i
            out_fields.append(d._dataset.schema.field(c))
    seen = base_cols_set | set(seen_new.keys())
    out_schema = pa.schema(out_fields)

    BATCH = 10_000

    def _iter():
        iters = [d._iter_arrow_batches(batch_size=BATCH) for d in dfs]
        for batches in zip(*iters):
            n = batches[0].num_rows
            for b in batches[1:]:
                if b.num_rows != n:
                    raise RuntimeError(
                        f"_merge_fanout: batch size mismatch at streaming: "
                        f"{n} vs {b.num_rows}"
                    )
            first_b = batches[0]
            arrays = [first_b.column(c) for c in base_cols]
            # v0.9.1b-fix1: simple check — append only columns NOT in
            # base_cols. (Prior garbage: `c in seen and c not in set(
            # b.schema.names[:0])` — always True on second part.)
            for b in batches[1:]:
                for c in b.schema.names:
                    if c in base_cols_set:
                        continue
                    arrays.append(b.column(c))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

    new_ds = Dataset.from_batches(
        out_schema, _iter(), dfs[0]._new_dataset_dir(),
    )
    return CadwDataFrame(
        new_ds,
        memory_guard=dfs[0]._memory_guard,
        spill_dir=dfs[0]._spill_dir,
    )



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
        # v0.9.1b: DAG edges. None = chain (auto [prev] or [] if first).
        # list[str] = explicit inputs ([] = source, [a,b] = fan-in).
        self._step_inputs: list = []
        self._fitted: bool = False

    # ---------------- construction ----------------

    def add_step(
        self,
        name: str,
        step,
        *,
        inputs: "list[str] | None" = None,
    ) -> "Pipeline":
        """Append a Transformer/Predictor. Returns self for chaining.

        inputs: DAG edges (v0.9.1b).
          - None (default): chain semantics. First step → source ([]);
            subsequent steps → [previous step name].
          - []: explicit source (only valid for first step, redundant but
            allowed).
          - [name1, name2, ...]: DAG fan-in. All names must exist; none
            can be a Predictor.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"add_step: name must be non-empty str; got {name!r}")
        if name in self._step_names:
            raise ValueError(
                f"add_step: duplicate name {name!r}; existing={self._step_names}"
            )
        has_fit = hasattr(step, "fit")
        has_transform = hasattr(step, "transform")
        has_predict = hasattr(step, "predict")
        if not (has_fit and (has_transform or has_predict)):
            raise TypeError(
                f"add_step: step must have fit() and (transform() or "
                f"predict()); got {type(step).__name__}"
            )
        new_is_pred = _is_predictor(step)
        if new_is_pred and any(_is_predictor(s) for s in self._steps):
            raise RuntimeError(
                "add_step: pipeline already has a Predictor; "
                "cannot add another (model must be last)"
            )

        # v0.9.1b-fix1 (§13.4/148): validate inputs BEFORE Predictor-after
        # check. Specific error ("cannot depend on Predictor") should fire
        # before generic ("cannot add step after Predictor") when user
        # passes inputs=["model"].
        # Also: inputs=[] is VALID for any step (multiple sources — main
        # DAG feature). v0.9.1b originally forbade inputs=[] after first
        # step — that killed fanout (`a inputs=[]`, `b inputs=[]`).
        if inputs is not None:
            if not isinstance(inputs, (list, tuple)):
                raise TypeError(
                    f"add_step: inputs must be list or None; "
                    f"got {type(inputs).__name__}"
                )
            for inp in inputs:
                if not isinstance(inp, str):
                    raise TypeError(
                        f"add_step: inputs must contain str; "
                        f"got {type(inp).__name__}"
                    )
                if inp not in self._step_names:
                    raise ValueError(
                        f"add_step: inputs refers to unknown step {inp!r}; "
                        f"existing={self._step_names}"
                    )
                inp_step = self._steps[self._step_names.index(inp)]
                if _is_predictor(inp_step):
                    raise RuntimeError(
                        f"add_step: cannot depend on Predictor {inp!r} "
                        f"(model must be sink)"
                    )
            inputs_stored = list(inputs)
        else:
            inputs_stored = None  # chain (derived at execution)

        # Predictor-after check (moved AFTER inputs validation)
        if self._steps and _is_predictor(self._steps[-1]) and not new_is_pred:
            raise RuntimeError(
                "add_step: cannot add step after a Predictor "
                "(model must be last)"
            )

        self._step_names.append(name)
        self._steps.append(step)
        self._step_inputs.append(inputs_stored)
        return self

    def _build_graph(self) -> dict:
        """Return {name: [input_names]}. None → chain derivation."""
        if not self._steps:
            return {}
        graph: dict = {}
        prev = None
        for name, _step, inputs in zip(
            self._step_names, self._steps, self._step_inputs,
        ):
            if inputs is None:
                graph[name] = [prev] if prev else []
            else:
                graph[name] = list(inputs)
            prev = name
        return graph

    def _topological_order(self) -> list:
        """Topological sort of step names. Raises on cycle / unknown ref."""
        graph = self._build_graph()
        visited: set = set()
        order: list = []
        stack: set = set()

        def dfs(node: str) -> None:
            if node in stack:
                raise RuntimeError(
                    f"Pipeline DAG cycle detected at {node!r}"
                )
            if node in visited:
                return
            stack.add(node)
            for inp in graph.get(node, []):
                if inp not in graph:
                    raise ValueError(
                        f"Pipeline DAG: unknown input {inp!r} for {node!r}"
                    )
                dfs(inp)
            stack.discard(node)
            visited.add(node)
            order.append(node)

        for name, _step, _inputs in zip(
            self._step_names, self._steps, self._step_inputs,
        ):
            dfs(name)
        return order

    def _get_step(self, name: str):
        idx = self._step_names.index(name)
        return self._steps[idx]

    # ---------------- save / load (v0.9.1c) ----------------

    _SERIALIZATION_VERSION = "0.9.1"

    def to_json(self, path) -> "Path":
        """Save pipeline to directory: `path/pipeline.json` + `path/state/`.

        - `path` must NOT be a file inside some directory — it IS the
          pipeline dir. `to_json('mypipe')` → `mypipe/pipeline.json`.
        - Each step must implement `to_dict()` → dict. State (fitted
          model) goes into `state/<step_name>.json` if non-trivial.
        - Raises NotImplementedError for steps without to_dict (e.g.
          LambdaTransform — callable not serializable).

        Returns Path to pipeline.json.
        """
        from pathlib import Path
        path = Path(path)
        if path.exists() and path.is_file():
            raise ValueError(
                f"to_json: path {path} is a file; expected directory"
            )
        path.mkdir(parents=True, exist_ok=True)
        state_dir = path / "state"

        steps_data = []
        for name, step in zip(self._step_names, self._steps):
            if not hasattr(step, "to_dict") or not callable(step.to_dict):
                raise NotImplementedError(
                    f"to_json: step {name!r} ({type(step).__name__}) "
                    f"does not implement to_dict(); cannot serialize. "
                    f"Define custom step with to_dict/from_dict."
                )
            d = step.to_dict()
            # Separate params from state (state goes to state/<name>.json)
            state = d.pop("state", None) if isinstance(d, dict) else None
            params = d
            # v0.9.1d-fix1: use __serialization_name__ (set by
            # register_step_type / pre-registration) if present; fall
            # back to class name for backward compat.
            type_name = getattr(
                type(step), "__serialization_name__", type(step).__name__
            )
            entry = {
                "name": name,
                "type": type_name,
                "inputs": self._step_inputs[
                    self._step_names.index(name)
                ],
                "params": params,
                "has_state": state is not None,
            }
            if state is not None:
                state_dir.mkdir(parents=True, exist_ok=True)
                state_file = state_dir / f"{name}.json"
                state_file.write_text(
                    json.dumps(state, indent=1)
                )
                entry["state_file"] = f"state/{name}.json"
            steps_data.append(entry)

        payload = {
            "version": self._SERIALIZATION_VERSION,
            "fitted": self._fitted,
            "steps": steps_data,
        }
        out_file = path / "pipeline.json"
        out_file.write_text(json.dumps(payload, indent=1))
        return out_file

    @classmethod
    def from_json(
        cls,
        path,
        *,
        memory_guard: Optional[MemoryGuard] = None,
        strict_version: bool = False,
    ) -> "Pipeline":
        """Load pipeline from directory (or from pipeline.json path).

        - `path` может быть директорией (ищет `path/pipeline.json`) или
          файлом `pipeline.json`.
        - `strict_version=True` — raise on version mismatch; default
          warns and tries best-effort.
        - State files loaded from `path/state/` (relative to pipeline.json).
        """
        from pathlib import Path
        import warnings

        path = Path(path)
        # v0.9.1c-fix1: unambiguous handling. If path doesn't exist,
        # assume it's a directory (user typically passes "mypipe", not
        # "mypipe/pipeline.json"), and show expected file in error.
        if path.is_dir():
            meta_file = path / "pipeline.json"
        elif path.exists():
            meta_file = path  # file
        else:
            # Non-existent path: assume directory for clearer error
            meta_file = path / "pipeline.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"from_json: {meta_file} not found "
                f"(path can be a pipeline directory or the "
                f"pipeline.json file inside it)"
            )
        base_dir = meta_file.parent
        try:
            payload = json.loads(meta_file.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"from_json: {meta_file} is not valid JSON: {e}"
            ) from e

        file_version = payload.get("version", "unknown")
        if file_version != cls._SERIALIZATION_VERSION:
            msg = (
                f"from_json: pipeline file version {file_version!r} "
                f"!= current {cls._SERIALIZATION_VERSION!r}; "
                f"attempting best-effort restore. Set "
                f"strict_version=True to fail on mismatch."
            )
            if strict_version:
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        # v0.9.1d: use module-level _STEP_REGISTRY (extensible via
        # register_step_type). Built-in types pre-registered.
        pipe = cls(memory_guard=memory_guard)
        for step_data in payload.get("steps", []):
            type_name = step_data["type"]
            if type_name not in _STEP_REGISTRY:
                raise NotImplementedError(
                    f"from_json: unknown step type {type_name!r}. "
                    f"Register via register_step_type({type_name!r}, cls) "
                    f"or use a built-in: {list_step_types()}"
                )
            step_cls = _STEP_REGISTRY[type_name]
            params = step_data.get("params", {})
            # Merge state into params for GBDTStep (it expects dict with
            # both "params" and "state" keys)
            combined = params.copy() if isinstance(params, dict) else {}
            if step_data.get("has_state"):
                state_file = base_dir / step_data["state_file"]
                if not state_file.exists():
                    raise FileNotFoundError(
                        f"from_json: state file {state_file} not found"
                    )
                combined["state"] = json.loads(state_file.read_text())
            if type_name == "GBDTStep":
                step = step_cls.from_dict(combined)
            else:
                step = step_cls.from_dict(combined)
            pipe.add_step(
                step_data["name"], step, inputs=step_data["inputs"],
            )
        if payload.get("fitted"):
            pipe._fitted = True
        return pipe

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

        graph = self._build_graph()
        order = self._topological_order()
        outputs: dict = {}

        for name in order:
            step = self._get_step(name)
            input_names = graph[name]
            if not input_names:
                df_in = df  # source
            elif len(input_names) == 1:
                df_in = outputs[input_names[0]]
            else:
                df_in = _merge_fanout([outputs[i] for i in input_names])

            is_pred = _is_predictor(step)
            step.fit(df_in, target=target if is_pred else None)
            if not is_pred:
                outputs[name] = step.transform(df_in)

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
        graph = self._build_graph()
        order = self._topological_order()
        outputs: dict = {}

        for name in order:
            step = self._get_step(name)
            if _is_predictor(step):
                # Predictor is sink; transform stops before it.
                break
            input_names = graph[name]
            if not input_names:
                df_in = df
            elif len(input_names) == 1:
                df_in = outputs[input_names[0]]
            else:
                df_in = _merge_fanout([outputs[i] for i in input_names])
            outputs[name] = step.transform(df_in)

        # Return the last non-Predictor output. If pipeline is
        # model-only (Predictor is first), return df unchanged.
        # Find the sink of the pure-transform subgraph.
        for name in reversed(order):
            step = self._get_step(name)
            if not _is_predictor(step):
                return outputs[name]
        return df  # model-only pipeline, no feature steps

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
        if not any(_is_predictor(s) for s in self._steps):
            raise TypeError(
                "Pipeline.predict: pipeline has no Predictor; "
                "call transform() instead"
            )
        graph = self._build_graph()
        order = self._topological_order()
        outputs: dict = {}

        for name in order:
            step = self._get_step(name)
            input_names = graph[name]
            if not input_names:
                df_in = df
            elif len(input_names) == 1:
                df_in = outputs[input_names[0]]
            else:
                df_in = _merge_fanout([outputs[i] for i in input_names])

            if _is_predictor(step):
                return step.predict(df_in)
            outputs[name] = step.transform(df_in)

        raise RuntimeError(
            "Pipeline.predict: DAG finished without hitting Predictor "
            "(should be unreachable — Predictor is sink, validated)"
        )

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
        """Jupyter HTML repr — step table with DAG edges + state_bytes."""
        if not self._steps:
            return "<div class='cadw-pipeline'>Pipeline (empty)</div>"

        graph = self._build_graph()
        total_state = self.total_state_bytes()
        rows = []
        for i, (name, step) in enumerate(zip(self._step_names, self._steps)):
            kind = "Predictor" if _is_predictor(step) else "Transformer"
            try:
                sb = step.state_bytes() if self._fitted else 0
            except Exception:
                sb = "?"
            state_str = f"{sb:,} B" if isinstance(sb, int) else str(sb)
            inputs = graph.get(name, [])
            inputs_str = (
                "(source)" if not inputs
                else " &larr; " + ", ".join(inputs)
            )
            rows.append(
                f"<tr><td>{i + 1}</td><td>{name}</td>"
                f"<td>{type(step).__name__}</td>"
                f"<td>{kind}</td><td>{inputs_str}</td>"
                f"<td>{state_str}</td></tr>"
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
            f"<th>role</th><th>inputs</th><th>state</th></tr>"
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

    def to_dict(self) -> dict:
        """v0.9.1c: serialize params. pa.DataType → str alias."""
        return {
            "mapping": {
                col: str(dt) for col, dt in self._mapping.items()
            },
            "safe": self._safe,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CastStep":
        import pyarrow as pa
        mapping = {
            col: pa.type_for_alias(str(dt)) if isinstance(dt, str) else dt
            for col, dt in d["mapping"].items()
        }
        return cls(mapping, safe=bool(d.get("safe", True)))


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

    def to_dict(self) -> dict:
        return {"how": self._how, "subset": self._subset}

    @classmethod
    def from_dict(cls, d: dict) -> "DropnaStep":
        return cls(how=d.get("how", "any"), subset=d.get("subset"))


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

    def to_dict(self) -> dict:
        """v0.9.1c: serialize params + state (fitted model).

        Returns params always; state only if fitted (`model` is None →
        state=None). `from_dict` with state=None → unfitted step.
        """
        out = {
            "params": {
                "X_cols": list(self._X_cols),
                "y_col": self._y_col,
                "kwargs": dict(self._kwargs),
            },
            "state": None,
        }
        if self._model is not None:
            from cadw._serialization import gbdt_regressor_to_dict
            out["state"] = {
                "kind": "gbdt_regressor",
                "data": gbdt_regressor_to_dict(self._model),
            }
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "GBDTStep":
        params = d.get("params", {})
        step = cls(
            X_cols=params["X_cols"],
            y_col=params["y_col"],
            **params.get("kwargs", {}),
        )
        state = d.get("state")
        if state is not None:
            kind = state.get("kind")
            if kind == "gbdt_regressor":
                from cadw._serialization import gbdt_regressor_from_dict
                step._model = gbdt_regressor_from_dict(state["data"])
            else:
                raise ValueError(
                    f"GBDTStep.from_dict: unknown state kind {kind!r}"
                )
        return step


class LambdaTransform:
    """Test helper: apply Arrow-native fn to a source column, append
    `{src_col}_{feature_name}` (v0.9.1b — for DAG tests and simple
    feature engineering). Stateless."""

    __slots__ = ("_src_col", "_feature_name", "_fn", "_out_type")

    def __init__(
        self,
        src_col: str,
        feature_name: str,
        fn,
        out_type,
    ) -> None:
        self._src_col = src_col
        self._feature_name = feature_name
        self._fn = fn
        self._out_type = out_type

    def fit(self, df, target=None):
        return self

    def transform(self, df):
        return df._apply_column_transform(
            self._src_col, self._feature_name, self._fn, self._out_type,
            accessor="lambda",
        )

    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0

    def to_dict(self) -> dict:
        """v0.9.1c: LambdaTransform NOT serializable (callable fn).

        Raise with clear message: define a custom step class implementing
        to_dict / from_dict, or wrap your function in a named class.
        """
        raise NotImplementedError(
            "LambdaTransform cannot be serialized: `fn` is a callable, "
            "not JSON-serializable. Define a custom step class with "
            "to_dict()/from_dict() instead, or use only named wrappers "
            "(CastStep, DropnaStep, GBDTStep) in pipelines you intend "
            "to save."
        )

    @classmethod
    def from_dict(cls, d: dict) -> "LambdaTransform":
        raise NotImplementedError(
            "LambdaTransform cannot be deserialized. Use a custom step "
            "class with to_dict/from_dict."
        )


# ============ Step type registry (v0.9.1d) ============

_STEP_REGISTRY: dict = {}


def register_step_type(name: str, cls: type) -> None:
    """Register a custom step class for `from_json` deserialization.

    Custom step MUST implement:
      - `to_dict()` — serialize params (+ optional state)
      - `from_dict(d)` (classmethod) — deserialize
      - `fit(df, target=None)`, `transform(df)` — Transformer contract
        (or `predict` for Predictor)
      - `estimate_state_bytes(dc)`, `state_bytes()` — for planner

    Overwrites existing registration with warning (same pattern as
    `register_model` in memory_guard.py — §9.7).

    Built-in types (`CastStep`, `DropnaStep`, `GBDTStep`) are pre-registered;
    override allowed for custom implementations of standard interfaces.

    Usage:
        >>> from cadw.pipeline import register_step_type
        >>> register_step_type("MyScaler", MyScalerStep)
        >>> pipe = Pipeline.from_json("saved/")  # uses MyScaler
    """
    import warnings
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"register_step_type: name must be non-empty str; got {name!r}"
        )
    if not isinstance(cls, type):
        raise TypeError(
            f"register_step_type: cls must be class; got {type(cls).__name__}"
        )
    if not (hasattr(cls, "from_dict") and callable(cls.from_dict)):
        raise TypeError(
            f"register_step_type: {cls.__name__} must have from_dict "
            f"classmethod (required for deserialization)"
        )
    if name in _STEP_REGISTRY:
        warnings.warn(
            f"register_step_type: {name!r} already registered as "
            f"{_STEP_REGISTRY[name].__name__}; overwriting with "
            f"{cls.__name__}",
            RuntimeWarning, stacklevel=2,
        )
    _STEP_REGISTRY[name] = cls
    # v0.9.1d-fix1 (§13.4/151): set `__serialization_name__` so `to_json`
    # writes THIS name (registry key), not `type(step).__name__` (class
    # name). Resolves namespace mismatch: registration key = serialization
    # key. Registering "MyCustom" for class `_CustomStep` → to_json writes
    # "MyCustom"; from_json finds it in registry.
    setattr(cls, "__serialization_name__", name)


def unregister_step_type(name: str) -> None:
    """Remove custom step type from registry (for tests / cleanup).

    Built-in types (`CastStep`, `DropnaStep`, `GBDTStep`) cannot be
    unregistered (they're always available for from_json).
    """
    if name in ("CastStep", "DropnaStep", "GBDTStep"):
        raise ValueError(
            f"unregister_step_type: cannot unregister built-in type "
            f"{name!r}"
        )
    _STEP_REGISTRY.pop(name, None)


def list_step_types() -> list:
    """Return sorted list of registered step type names."""
    return sorted(_STEP_REGISTRY.keys())


# Pre-register built-in wrappers. `__serialization_name__` set to the
# canonical name so `to_json` writes it (and `from_json` finds it).
# `ColumnTransformer` is registered at the bottom of this module,
# AFTER its class definition (see below).
_STEP_REGISTRY["CastStep"] = CastStep
_STEP_REGISTRY["DropnaStep"] = DropnaStep
_STEP_REGISTRY["GBDTStep"] = GBDTStep
for _pre_name, _pre_cls in list(_STEP_REGISTRY.items()):
    setattr(_pre_cls, "__serialization_name__", _pre_name)


# ============ ColumnTransformer (v0.9.1d) ============

class ColumnTransformer:
    """Apply different transformers to different column subsets.

    sklearn-style column fanout. Each transformer is fit and applied
    to `df[cols]` subset; outputs are concatenated (column union).

    Args:
        transformers: list of (name, cols, step) triples.
            - name: str, unique identifier per transformer.
            - cols: str or list[str] — column subset the transformer sees.
            - step: any Transformer (fit + transform contract).
        remainder: 'drop' (default) | 'passthrough'.
            - 'drop': columns not claimed by any transformer are dropped.
            - 'passthrough': they appear unchanged at the end of output.

    Valid cases:
        - Disjoint `cols` across transformers (each column claimed by ≤1).
        - Transformers preserve row count (no dropna inside).
        - Unique transformer names.
        - `remainder='drop'` with all cols claimed.
        - `remainder='passthrough'` with some cols unclaimed.
        - Transformer emits new columns (e.g. LambdaTransform appends
          `{col}_{feat}`) — new cols appear in output.

    Invalid cases (raise at construction):
        - Empty `transformers` → ValueError.
        - Duplicate transformer `name` → ValueError.
        - Overlapping `cols` across transformers → ValueError.
        - Empty `cols` in a transformer → ValueError.
        - `step` missing `fit` or `transform` → TypeError.
        - `remainder` not in ('drop', 'passthrough') → ValueError.

    Invalid cases (raise at transform):
        - `cols` column missing from input df → KeyError.
        - Transformer changed row count (e.g. dropna inside) → RuntimeError.
        - Duplicate output column name across transformers → RuntimeError.

    Output column order:
        [t1 output cols..., t2 output cols..., ..., remainder cols...]
        Where "t_i output cols" = whatever `t_i.transform()` emitted.

    Examples:
        >>> ct = ColumnTransformer([
        ...     ('num', ['a', 'b'], CastStep({'a': 'float64', 'b': 'float64'})),
        ...     ('cat', ['c'], CastStep({'c': 'int32'})),
        ... ], remainder='drop')
        >>> ct.fit(df)
        >>> out = ct.transform(df)  # columns: a, b, c
    """

    __slots__ = ("_transformers", "_remainder")

    def __init__(
        self,
        transformers: list,
        *,
        remainder: str = "drop",
    ) -> None:
        if remainder not in ("drop", "passthrough"):
            raise ValueError(
                f"ColumnTransformer: remainder must be 'drop' or "
                f"'passthrough'; got {remainder!r}"
            )
        if not transformers:
            raise ValueError(
                "ColumnTransformer: transformers must be non-empty list"
            )

        names_seen: set = set()
        cols_claimed: dict = {}  # col → transformer name
        normalized: list = []

        for entry in transformers:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise TypeError(
                    f"ColumnTransformer: each transformer must be "
                    f"(name, cols, step) triple; got {entry!r}"
                )
            name, cols, step = entry

            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"ColumnTransformer: transformer name must be "
                    f"non-empty str; got {name!r}"
                )
            if name in names_seen:
                raise ValueError(
                    f"ColumnTransformer: duplicate transformer name "
                    f"{name!r}; existing={sorted(names_seen)}"
                )
            names_seen.add(name)

            if isinstance(cols, str):
                cols = [cols]
            else:
                cols = list(cols)
            if not cols:
                raise ValueError(
                    f"ColumnTransformer: transformer {name!r}: cols "
                    f"must be non-empty"
                )
            for c in cols:
                if not isinstance(c, str):
                    raise TypeError(
                        f"ColumnTransformer: transformer {name!r}: cols "
                        f"must be str; got {type(c).__name__}"
                    )
                if c in cols_claimed:
                    raise ValueError(
                        f"ColumnTransformer: column {c!r} claimed by "
                        f"both {cols_claimed[c]!r} and {name!r} — each "
                        f"column can belong to at most one transformer"
                    )
                cols_claimed[c] = name

            if not (hasattr(step, "fit") and hasattr(step, "transform")):
                raise TypeError(
                    f"ColumnTransformer: transformer {name!r} "
                    f"({type(step).__name__}) must have fit + transform "
                    f"(Transformer contract); Predictors cannot be "
                    f"inside ColumnTransformer (only as final Pipeline step)"
                )

            normalized.append((name, cols, step))

        self._transformers = normalized
        self._remainder = remainder

    def fit(self, df, target=None):
        """Fit each transformer on its column subset."""
        for name, cols, step in self._transformers:
            for c in cols:
                if c not in df.columns:
                    raise KeyError(
                        f"ColumnTransformer.fit: transformer {name!r}: "
                        f"column {c!r} not in {df.columns}"
                    )
            subset = df[cols]
            step.fit(subset, target=None)
        return self

    def transform(self, df):
        """Apply each transformer to its subset; column-union output."""
        n_orig = df.count()
        claimed = set()
        for _, cols, _ in self._transformers:
            claimed.update(cols)

        results = []
        out_col_owner: dict = {}  # col → transformer name (for duplicate check)

        for name, cols, step in self._transformers:
            for c in cols:
                if c not in df.columns:
                    raise KeyError(
                        f"ColumnTransformer.transform: transformer "
                        f"{name!r}: column {c!r} not in {df.columns}"
                    )
            subset = df[cols]
            out = step.transform(subset)
            n_out = out.count()
            if n_out != n_orig:
                raise RuntimeError(
                    f"ColumnTransformer: transformer {name!r} changed "
                    f"row count: {n_orig} -> {n_out}. ColumnTransformer "
                    f"requires row-preserving transformers (column-level "
                    f"only). If you need row filtering, apply it before "
                    f"ColumnTransformer."
                )
            for c in out.columns:
                if c in out_col_owner:
                    raise RuntimeError(
                        f"ColumnTransformer: duplicate output column "
                        f"{c!r} from transformers "
                        f"{out_col_owner[c]!r} and {name!r}. Use distinct "
                        f"column names or check for overlapping transforms."
                    )
                out_col_owner[c] = name
            results.append(out)

        if self._remainder == "passthrough":
            remainder_cols = [c for c in df.columns if c not in claimed]
            if remainder_cols:
                results.append(df[remainder_cols])

        return _merge_fanout(results)

    def estimate_state_bytes(self, dc) -> int:
        return sum(
            step.estimate_state_bytes(dc)
            for _, _, step in self._transformers
        )

    def state_bytes(self) -> int:
        return sum(
            step.state_bytes()
            for _, _, step in self._transformers
        )

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict (nested state inline)."""
        entries = []
        for name, cols, step in self._transformers:
            if not (hasattr(step, "to_dict") and callable(step.to_dict)):
                raise NotImplementedError(
                    f"ColumnTransformer.to_dict: nested step {name!r} "
                    f"({type(step).__name__}) does not implement to_dict()"
                )
            # v0.9.9 (N1): use __serialization_name__ (set by
            # register_step_type) so from_json finds the same key.
            entries.append({
                "name": name,
                "cols": list(cols),
                "type": getattr(
                    type(step), "__serialization_name__",
                    type(step).__name__,
                ),
                "step_dict": step.to_dict(),
            })
        return {
            "transformers": entries,
            "remainder": self._remainder,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnTransformer":
        entries = []
        for entry in d.get("transformers", []):
            type_name = entry["type"]
            if type_name not in _STEP_REGISTRY:
                raise NotImplementedError(
                    f"ColumnTransformer.from_dict: unknown nested step "
                    f"type {type_name!r}. Register via "
                    f"register_step_type({type_name!r}, ...) or use "
                    f"built-in wrappers: {list_step_types()}"
                )
            step_cls = _STEP_REGISTRY[type_name]
            step = step_cls.from_dict(entry["step_dict"])
            entries.append((entry["name"], entry["cols"], step))
        return cls(entries, remainder=d.get("remainder", "drop"))

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{name}={list(cols)}" for name, cols, _ in self._transformers
        )
        return (
            f"ColumnTransformer([{parts}], remainder={self._remainder!r})"
        )


# v0.9.1d-fix1 (§13.4/151): register ColumnTransformer AFTER its class
# definition (pre-registration block runs before class exists).
_STEP_REGISTRY["ColumnTransformer"] = ColumnTransformer
ColumnTransformer.__serialization_name__ = "ColumnTransformer"
