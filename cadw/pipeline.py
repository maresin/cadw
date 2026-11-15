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

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
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

    v0.9.10 (§11.5): F5 duplicate-new-column guard fires only
    for columns new relative to dfs[0] (base). If dfs[0]
    produces a column with the same name as a sibling's NEW
    column, dfs[0] silently wins (base_cols_set absorbs it).
    Cannot distinguish "was in raw df" from "added by dfs[0]"
    without explicit raw_columns. Fix (backlog v0.9.11):
    optional raw_columns kwarg passed by Pipeline.transform.
    Users hitting this: rename conflicting features, or wrap
    in ColumnTransformer.
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



class _LazyView:
    """v0.9.237b: lightweight df-like for streaming chain.

    Only exposes what fit/transform of downstream steps need:
    _memory_guard, _spill_dir, _iter_filtered_batches, _new_dataset_dir,
    _dataset (minimal proxy with .schema), count() (approximate).
    """
    __slots__ = ("_factory", "_schema", "_template",
                 "_memory_guard", "_spill_dir")

    def __init__(self, factory, schema, template_df):
        self._factory = factory
        self._schema = schema
        self._template = template_df
        self._memory_guard = template_df._memory_guard
        self._spill_dir = template_df._spill_dir

    def _iter_filtered_batches(self, batch_size=None):
        # batch_size ignored: factory already yields batches
        yield from self._factory()

    def _new_dataset_dir(self):
        return self._template._new_dataset_dir()

    def count(self):
        return self._template.count()

    @property
    def _dataset(self):
        return _LazyDatasetView(self._schema)


class _LazyDatasetView:
    __slots__ = ("schema",)
    def __init__(self, schema):
        self.schema = schema


def _materialize(factory, template_df, schema):
    """v0.9.237b: 1-write materialization into new Dataset."""
    from cadw.dataset import Dataset
    from cadw.dataframe import CadwDataFrame
    new_ds = Dataset.from_batches(
        schema, factory(), template_df._new_dataset_dir())
    return CadwDataFrame(
        new_ds,
        memory_guard=template_df._memory_guard,
        spill_dir=template_df._spill_dir)


class Pipeline:
    """Sequence of Transformers. Model (if any) is LAST.

    Usage:
        pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
        pipe.add_step("scale", StandardScaler(columns=["v"]))
        pipe.add_step("model", ModelStep("fit_linear", X_cols=["v"], y_col="y"))
        # (StandardScaler / ModelStep are user-defined or reference steps)
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
          arbitrary callable — not serializable).

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
            # Merge state into params: step.from_dict receives both
            # "params" and optional "state" keys (convention: from_dict
            # reads d.get("params"), d.get("state"))
            combined = params.copy() if isinstance(params, dict) else {}
            if step_data.get("has_state"):
                state_file = base_dir / step_data["state_file"]
                if not state_file.exists():
                    raise FileNotFoundError(
                        f"from_json: state file {state_file} not found"
                    )
                combined["state"] = json.loads(state_file.read_text())
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

    def _step_peak_estimate(self, step, dc) -> int:
        """v0.9.233 (P2a): per-step peak via YamlFormulaModel.

        Resolution order:
          1. step.cadw_function + registered in YAML registry:
             - batch-linear formula: solve(budget) -> estimate(params)  [full peak]
             - state-only formula (no batch term): estimate at batch=1   [state]
          2. step.estimate_state_bytes(dc)  [fallback: no formula, or scaler]
        """
        from cadw.core.yaml_formula_model import get_registered_model
        cf = getattr(step, "cadw_function", None)
        if cf is not None:
            model = get_registered_model(cf)
            if model is not None:
                budget = self._memory_guard.budget_bytes
                try:
                    params = model.solve(budget, dc, safety=0.7)
                    if params is not None:
                        return model.estimate(params, dc)
                    # state-only formula: eval at batch=1.
                    # Formula may reference step hyperparams (n_estimators,
                    # max_depth for GBDT) not present in dc -> KeyError ->
                    # fallback to step.estimate_state_bytes (honest state floor).
                    return model._eval(dc, batch=1)
                except MemoryLimitExceeded:
                    raise  # bubble up; caller wraps in PipelineMemoryError
                except (KeyError, NameError):
                    pass  # missing hyperparams -> fallback
        return step.estimate_state_bytes(dc)

    def plan(self, df: "CadwDataFrame") -> PipelinePlan:
        """v0.9.233 (P2a): pre-fit feasibility via peak, not just state.

        Checks (both must fit):
          - sum(state_bytes) <= state_limit   (state accounting)
          - sum(peak_bytes)  <= budget_bytes  (peak accounting; includes transient)

        peak_bytes comes from YamlFormulaModel for steps with cadw_function
        (batch-linear: full formula; state-only: state). Fallback to
        estimate_state_bytes for steps without formula (e.g. scalers,
        fallback). Conservative: sum of peaks across steps (no overlap
        assumption).

        Raises PipelineMemoryError with step_idx + actionable describe.
        """
        sample = peek_first_batch(df._iter_filtered_batches())
        if sample is None:
            raise ValueError("Pipeline.plan: empty df")
        dc = data_chars_from_sample(sample, df.count())

        # v0.9.235 (DEC-083): peak = max_k [cumulative_state_prior + peak_k].
        # Peaks occur sequentially, not simultaneously. For N barriers with
        # state=0 and peak=30MB: correct total = 30MB, not N*30MB.
        # For N stateful models (state=8MB, peak=8MB): correct = sum(state)
        # (all held), which equals N*8MB in step_k loop.
        #
        # state_check = sum(state_k) (all states held for predict; current impl).
        # Future (streaming, state release after barrier): max(segment_state).
        state_estimates = []
        peak_estimates = []
        cumulative_state = 0
        total_peak = 0
        first_bad_idx = None
        first_bad_reason = ""
        for i, step in enumerate(self._steps):
            try:
                peak_k = self._step_peak_estimate(step, dc)
            except MemoryLimitExceeded as e:
                raise PipelineMemoryError(
                    step_idx=i,
                    step_name=self._step_names[i],
                    estimate_bytes=0,
                    available_bytes=self._memory_guard.budget_bytes,
                    describe=f"{type(step).__name__}.cadw_function="
                             f"{getattr(step, 'cadw_function', None)!r}: {e}",
                )
            state_k = step.estimate_state_bytes(dc)
            step_total_peak = cumulative_state + peak_k
            total_peak = max(total_peak, step_total_peak)
            cumulative_state += state_k
            state_estimates.append(state_k)
            peak_estimates.append(peak_k)
            # Track first step that individually exceeds (for actionable error)
            if first_bad_idx is None:
                if step_total_peak > self._memory_guard.budget_bytes:
                    first_bad_idx = i
                    first_bad_reason = (
                        f"step_total_peak={step_total_peak} > "
                        f"budget={self._memory_guard.budget_bytes}")
                elif cumulative_state > self._memory_guard.state_limit_bytes:
                    first_bad_idx = i
                    first_bad_reason = (
                        f"cumulative_state={cumulative_state} > "
                        f"state_limit={self._memory_guard.state_limit_bytes}")

        total_state = cumulative_state
        state_limit = self._memory_guard.state_limit_bytes
        budget = self._memory_guard.budget_bytes

        state_ok = total_state <= state_limit
        peak_ok = total_peak <= budget

        if not state_ok or not peak_ok:
            bad_idx = first_bad_idx if first_bad_idx is not None else (
                len(self._steps) - 1)
            raise PipelineMemoryError(
                step_idx=bad_idx,
                step_name=(
                    self._step_names[bad_idx]
                    if self._step_names else "<none>"),
                estimate_bytes=total_peak if not peak_ok else total_state,
                available_bytes=budget,
                describe=(
                    f"Pipeline peak={total_peak} B (max sequential) "
                    f"vs budget={budget} B; "
                    f"state={total_state} B vs state_limit={state_limit} B; "
                    f"per_step_peak={peak_estimates}; "
                    f"per_step_state={state_estimates}; "
                    f"first_failing={first_bad_reason or 'last'}"
                ),
            )

        step_plans = [
            StepPlan(
                op_name=type(step).__name__,
                params=None,
                estimate_bytes=peak_estimates[i],
                input_chars=dc,
                output_chars=dc,
                spill_before=False,
            )
            for i, step in enumerate(self._steps)
        ]

        return PipelinePlan(
            steps=step_plans,
            peak_bytes=total_peak,
            spill_boundaries=[],
            estimated_io=0,
            predicted_outcome="fits",
            reason=None,
            available_bytes=budget,
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

    def _execute_stream(self, df, target=None):
        """v0.9.237b: segment-based streaming executor.

        Narrow chain fused per-batch; barrier materializes (1 write);
        final materialize (1 write). Returns features df (or None on
        fan-in DAG or missing support -> eager fallback).

        Streaming eligibility per step: has fit_iter + transform_batch +
        output_schema (v0.9.238 will add to scalers/Cast/Dropna; mocks
        already implement).
        """
        graph = self._build_graph()
        order = self._topological_order()
        # DAG with fan-in -> fallback (linear chain only)
        for name in order:
            if len(graph[name]) > 1:
                return None
        try:
            factory = df._iter_filtered_batches
            schema = df._dataset.schema
            last_features_df = None

            for name in order:
                step = self._get_step(name)
                is_pred = _is_predictor(step)
                has_stream = (
                    not getattr(step, "is_barrier", True)
                    and getattr(step, "_streamable", True)
                    and hasattr(step, "fit_iter")
                    and hasattr(step, "transform_batch")
                    and hasattr(step, "output_schema"))

                if is_pred:
                    lazy = _LazyView(factory, schema, df)
                    step.fit(lazy, target=target)
                    last_features_df = None
                    factory = None
                    break

                if has_stream:
                    step.fit_iter(factory, df)
                    prev_factory = factory
                    prev_fn = step.transform_batch
                    factory = (
                        lambda pf=prev_factory, fn=prev_fn: (
                            fn(b) for b in pf()))
                    schema = step.output_schema(schema)
                    last_features_df = None
                else:
                    # Barrier or eager-only step
                    lazy = _LazyView(factory, schema, df)
                    step.fit(lazy, target=None)
                    out_df = step.transform(lazy)
                    factory = out_df._iter_filtered_batches
                    schema = out_df._dataset.schema
                    last_features_df = out_df

            if factory is not None:
                last_features_df = _materialize(factory, df, schema)
            elif last_features_df is None:
                last_features_df = df

            self._fitted = True
            return last_features_df
        except Exception:
            # Any incompatibility -> caller falls back to eager
            return None

    def fit_transform(
        self,
        df: "CadwDataFrame",
        target: Optional[str] = None,
    ) -> "CadwDataFrame":
        """fit + transform; returns feature CadwDataFrame (before model).

        v0.9.237b: attempts streaming path first; falls back to eager.
        """
        result = self._execute_stream(df, target=target)
        if result is not None:
            return result
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

    Built-in step types (`CompositeStep`, `ColumnTransformer`,
    `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`) are
    registered at import of `cadw.steps` / `cadw.scalers`.

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

    Built-in types (`CompositeStep`, `ColumnTransformer`, scalers) cannot
    be unregistered (re-registered on import of `cadw.steps` / `cadw.scalers`).
    """
    _BUILTINS = ("CompositeStep", "ColumnTransformer",
                 "StandardScaler", "MinMaxScaler", "MaxAbsScaler",
                 "RobustScaler")
    if name in _BUILTINS:
        raise ValueError(
            f"unregister_step_type: cannot unregister built-in type "
            f"{name!r} (re-registered on import of cadw.steps / cadw.scalers)"
        )
    _STEP_REGISTRY.pop(name, None)


def list_step_types() -> list:
    """Return sorted list of registered step type names."""
    return sorted(_STEP_REGISTRY.keys())
