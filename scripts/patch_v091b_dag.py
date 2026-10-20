#!/usr/bin/env python3
"""v0.9.1b: DAG Pipeline — inputs=[...], fan-in, topo execution.

Adds DAG support to Pipeline: `add_step(name, step, inputs=[...])`.
`inputs=None` keeps sequence semantics (chain); `inputs=[]` explicit
source; `inputs=["a", "b"]` DAG fan-in.

Fan-in MVP semantics (§11.5):
  - All branches must have SAME row count (positional alignment).
  - Result = column union: base columns (source) dedup + new columns
    from each branch.
  - Dropna / row-selection in one branch → RuntimeError (v0.9.1c: fan-in
    with row selection).

Predictor = sink: can't be input to another step; adding non-Predictor
after Predictor → RuntimeError (v0.9.1a).

GUARDS (before commit, via patch_utils.assert_guards_green):
  test_pipeline (backward compat) + test_pipeline_dag (new) +
  test_pipeline_e2e + test_changelog + arrow_guard + no_pandas + fast+slow.

Run: uv run python scripts/patch_v091b_dag.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


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


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ 1. _merge_fanout helper in pipeline.py ============

MERGE_HELPER = '''

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
    seen = set(base_cols)
    out_fields = [dfs[0]._dataset.schema.field(c) for c in base_cols]
    for d in dfs[1:]:
        for c in d.columns:
            if c in seen:
                continue
            out_fields.append(d._dataset.schema.field(c))
            seen.add(c)
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
            for b in batches[1:]:
                for c in b.schema.names:
                    if c in seen and c not in set(
                        b.schema.names[:0]
                    ):  # c in base_cols → already taken
                        continue
                    # Append new column (not in base_cols)
                    if c in base_cols:
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
'''


def add_merge_fanout():
    print("\n=== cadw/pipeline.py: _merge_fanout helper ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()
    if "_merge_fanout" in s:
        print("  already present — skip")
        return

    # Insert before `def _is_predictor(step)` (module helper)
    anchor = "def _is_predictor(step) -> bool:\n    return hasattr(step, \"predict\") and callable(getattr(step, \"predict\"))"
    if anchor not in s:
        raise SystemExit("_is_predictor anchor not found")

    s = s.replace(anchor, anchor + MERGE_HELPER, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 2. Pipeline: inputs storage + _build_graph + topo ============

def patch_pipeline_init():
    print("\n=== cadw/pipeline.py: __init__ + _step_inputs ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''    def __init__(
        self,
        memory_guard: Optional[MemoryGuard] = None,
    ) -> None:
        self._memory_guard = memory_guard or MemoryGuard(max_memory_mb=100)
        self._step_names: list[str] = []
        self._steps: list = []
        self._fitted: bool = False'''

    new = '''    def __init__(
        self,
        memory_guard: Optional[MemoryGuard] = None,
    ) -> None:
        self._memory_guard = memory_guard or MemoryGuard(max_memory_mb=100)
        self._step_names: list[str] = []
        self._steps: list = []
        # v0.9.1b: DAG edges. None = chain (auto [prev] or [] if first).
        # list[str] = explicit inputs ([] = source, [a,b] = fan-in).
        self._step_inputs: list = []
        self._fitted: bool = False'''

    s = replace_line_block(s, old, new, tag="pipeline-init-dag")
    p.write_text(s)
    compile_check(p)
    print("  OK")


def patch_add_step():
    print("\n=== cadw/pipeline.py: add_step with inputs=... ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''    def add_step(self, name: str, step) -> "Pipeline":
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
        return self'''

    new = '''    def add_step(
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
        if self._steps and _is_predictor(self._steps[-1]) and not new_is_pred:
            raise RuntimeError(
                "add_step: cannot add step after a Predictor "
                "(model must be last)"
            )
        # v0.9.1b: validate inputs
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
            if len(inputs) == 0 and self._steps:
                raise ValueError(
                    "add_step: inputs=[] (explicit source) is only allowed "
                    "for the first step; leave inputs=None for chain"
                )
            inputs_stored = list(inputs)
        else:
            inputs_stored = None  # chain (derived at execution)
        self._step_names.append(name)
        self._steps.append(step)
        self._step_inputs.append(inputs_stored)
        return self'''

    s = replace_line_block(s, old, new, tag="add-step-dag")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 3. Graph + topo helper methods ============

def add_graph_helpers():
    print("\n=== cadw/pipeline.py: _build_graph + _topological_order ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()
    if "def _build_graph(" in s:
        print("  already present — skip")
        return

    # Insert before @property steps
    anchor = '''    @property
    def steps(self) -> list:
        """Read-only view: list of (name, step) tuples."""
        return list(zip(self._step_names, self._steps))'''

    new = '''    def _build_graph(self) -> dict:
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

    @property
    def steps(self) -> list:
        """Read-only view: list of (name, step) tuples."""
        return list(zip(self._step_names, self._steps))'''

    s = replace_line_block(s, anchor, new, tag="graph-helpers")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 4. fit/transform/predict use DAG ============

def patch_fit_transform_predict():
    print("\n=== cadw/pipeline.py: fit/transform/predict via DAG ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    # --- fit ---
    old_fit = '''        # Feasibility check BEFORE any state mutation (§9.3)
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
        return self'''

    new_fit = '''        # Feasibility check BEFORE any state mutation (§9.3)
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
        return self'''

    s = replace_line_block(s, old_fit, new_fit, tag="fit-dag")

    # --- transform ---
    old_tf = '''        if not self._fitted:
            raise RuntimeError(
                "Pipeline.transform: pipeline is not fitted; call fit first"
            )
        current = df
        for i, step in enumerate(self._steps):
            if _is_predictor(step):
                # Predictor is always last (validated at fit time)
                break
            current = step.transform(current)
        return current'''

    new_tf = '''        if not self._fitted:
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
        return df  # model-only pipeline, no feature steps'''

    s = replace_line_block(s, old_tf, new_tf, tag="transform-dag")

    # --- predict ---
    old_pred = '''        if not self._fitted:
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
        return self._steps[-1].predict(features)'''

    new_pred = '''        if not self._fitted:
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
        )'''

    s = replace_line_block(s, old_pred, new_pred, tag="predict-dag")
    p.write_text(s)
    compile_check(p)
    print("  OK: fit/transform/predict via topo DAG")


# ============ 5. DAG-aware _repr_html_ ============

def patch_repr_html():
    print("\n=== cadw/pipeline.py: DAG-aware _repr_html_ ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''    def _repr_html_(self) -> str:
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
        )'''

    new = '''    def _repr_html_(self) -> str:
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
        )'''

    s = replace_line_block(s, old, new, tag="repr-dag")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 6. LambdaTransform test helper ============

def add_lambda_transform():
    print("\n=== cadw/pipeline.py: LambdaTransform test helper ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()
    if "class LambdaTransform:" in s:
        print("  already present — skip")
        return

    anchor = '''    def predict(self, df):
        if self._model is None:
            raise RuntimeError("GBDTStep.predict: not fitted")
        return self._model.predict(df)'''

    new = anchor + '''


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
        return 0'''

    s = replace_line_block(s, anchor, new, tag="lambda-transform")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 7. Tests ============

DAG_TESTS = '''"""v0.9.1b: DAG Pipeline tests."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from pathlib import Path

from cadw import CadwDataFrame, MemoryGuard
from cadw.pipeline import (
    CastStep, DropnaStep, GBDTStep, LambdaTransform, Pipeline,
)


def make_df(tmp_path, data, chunk=5, name="L", mem_mb=200):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_table(df, tmp_path, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


# ============ backward compat: sequence still works ============

def test_sequence_still_works(tmp_path):
    """inputs=None → chain semantics (v0.9.0 behaviour preserved)."""
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = (
        Pipeline()
        .add_step("cast", CastStep({"v": "float64"}))
        .add_step("dropna", DropnaStep())
    )
    pipe.fit(df)
    out = pipe.transform(df)
    assert out.count() == 3


def test_inputs_none_chains_to_prev(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    pipe.add_step("b", DropnaStep())  # implicitly inputs=['a']
    graph = pipe._build_graph()
    assert graph["a"] == []
    assert graph["b"] == ["a"]


# ============ fan-in basics ============

def test_dag_two_source_branches_merge(tmp_path):
    """a and b both source → merge fan-in → union columns."""
    df = make_df(tmp_path, {"x": [1, 2, 3, 4]})

    pipe = Pipeline()
    pipe.add_step(
        "a", LambdaTransform(
            "x", "plus1", lambda arr: pc.add(arr, 1), pa.int64(),
        ), inputs=[],
    )
    pipe.add_step(
        "b", LambdaTransform(
            "x", "times2", lambda arr: pc.multiply(arr, 2), pa.int64(),
        ), inputs=[],
    )
    pipe.add_step("merge", DropnaStep(), inputs=["a", "b"])
    pipe.fit(df)
    out = pipe.transform(df)
    t = read_table(out, tmp_path)
    # Columns: x, x_plus1, x_times2 (x dedup, source col once)
    cols = t.schema.names
    assert "x" in cols
    assert "x_plus1" in cols
    assert "x_times2" in cols
    assert cols.count("x") == 1
    # Verify values
    assert t.column("x_plus1").to_pylist() == [2, 3, 4, 5]
    assert t.column("x_times2").to_pylist() == [2, 4, 6, 8]


def test_dag_three_branches(tmp_path):
    df = make_df(tmp_path, {"x": [10, 20]})
    pipe = Pipeline()
    pipe.add_step(
        "a", LambdaTransform(
            "x", "add1", lambda arr: pc.add(arr, 1), pa.int64(),
        ), inputs=[],
    )
    pipe.add_step(
        "b", LambdaTransform(
            "x", "add2", lambda arr: pc.add(arr, 2), pa.int64(),
        ), inputs=[],
    )
    pipe.add_step(
        "c", LambdaTransform(
            "x", "add3", lambda arr: pc.add(arr, 3), pa.int64(),
        ), inputs=[],
    )
    pipe.add_step("merge", DropnaStep(), inputs=["a", "b", "c"])
    pipe.fit(df)
    t = read_table(pipe.transform(df), tmp_path)
    assert t.column("x_add1").to_pylist() == [11, 21]
    assert t.column("x_add2").to_pylist() == [12, 22]
    assert t.column("x_add3").to_pylist() == [13, 23]


def test_dag_diamond(tmp_path):
    """raw → a, raw → b, merge → model. Predict via full DAG."""
    n = 100
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(
        tmp_path, {"x": x.tolist(), "y": y.tolist()},
        chunk=20, mem_mb=200,
    )

    pipe = Pipeline(
        memory_guard=MemoryGuard(max_memory_mb=200),
    )
    pipe.add_step(
        "a", LambdaTransform(
            "x", "sq", lambda arr: pc.multiply(arr, arr), pa.float64(),
        ), inputs=[],
    )
    pipe.add_step(
        "b", LambdaTransform(
            "x", "abs", lambda arr: pc.abs(arr), pa.float64(),
        ), inputs=[],
    )
    pipe.add_step(
        "model", GBDTStep(
            ["x_sq", "x_abs"], "y",
            n_estimators=10, max_depth=2,
        ),
        inputs=["a", "b"],
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    t = read_table(preds, tmp_path)
    assert "y_pred" in t.schema.names


# ============ fan-in row count mismatch ============

def test_dag_fan_in_row_mismatch_raises(tmp_path):
    """dropna in one branch drops rows → count mismatch → RuntimeError."""
    df = make_df(tmp_path, {
        "x": [1, 2, 3, 4],
        "y": [10, None, 20, None],   # dropna drops 2 rows
    })
    pipe = Pipeline()
    pipe.add_step("a", DropnaStep(how="any", subset=["y"]), inputs=[])
    pipe.add_step("b", CastStep({"x": "float64"}), inputs=[])
    pipe.add_step("merge", DropnaStep(), inputs=["a", "b"])
    with pytest.raises(RuntimeError, match="row count mismatch"):
        pipe.fit(df)


# ============ validation ============

def test_add_step_unknown_input_raises():
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    with pytest.raises(ValueError, match="unknown step"):
        pipe.add_step("b", DropnaStep(), inputs=["nope"])


def test_add_step_predictor_as_input_raises(tmp_path):
    df = make_df(
        tmp_path,
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]},
    )
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("model", GBDTStep(["x"], "y", n_estimators=5))
    with pytest.raises(RuntimeError, match="cannot depend on Predictor"):
        pipe.add_step("post", DropnaStep(), inputs=["model"])


def test_add_step_bad_inputs_type_raises():
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    with pytest.raises(TypeError, match="inputs must be list"):
        pipe.add_step("b", DropnaStep(), inputs="a")  # str, not list


def test_add_step_explicit_source_only_first():
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    with pytest.raises(ValueError, match="explicit source.*first"):
        pipe.add_step("b", DropnaStep(), inputs=[])


# ============ topology ============

def test_topological_order_chain():
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}))
    pipe.add_step("b", DropnaStep())
    pipe.add_step("c", CastStep({"v": "int64"}))
    order = pipe._topological_order()
    assert order == ["a", "b", "c"]


def test_topological_order_fan_out():
    pipe = Pipeline()
    pipe.add_step(
        "a", LambdaTransform(
            "x", "p1", lambda arr: arr, pa.int64(),
        ), inputs=[],
    )
    pipe.add_step(
        "b", LambdaTransform(
            "x", "p2", lambda arr: arr, pa.int64(),
        ), inputs=[],
    )
    pipe.add_step("merge", DropnaStep(), inputs=["a", "b"])
    order = pipe._topological_order()
    # merge must come after a and b
    assert order.index("merge") > order.index("a")
    assert order.index("merge") > order.index("b")


# ============ transform/predict DAG semantics ============

def test_transform_stops_at_model(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step(
        "feat", LambdaTransform(
            "x", "sq", lambda arr: pc.multiply(arr, arr), pa.float64(),
        ),
    )
    pipe.add_step("model", GBDTStep(["x_sq"], "y", n_estimators=5))
    pipe.fit(df, target="y")
    features = pipe.transform(df)
    # features: x, y, x_sq
    assert "x_sq" in features.columns
    assert "y_pred" not in features.columns


def test_predict_runs_full_dag(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = x * 2.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step(
        "feat", LambdaTransform(
            "x", "sq", lambda arr: pc.multiply(arr, arr), pa.float64(),
        ),
    )
    pipe.add_step("model", GBDTStep(["x_sq"], "y", n_estimators=5))
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    assert "y_pred" in preds.columns


# ============ repr_html DAG edges ============

def test_repr_html_shows_dag_edges(tmp_path):
    df = make_df(tmp_path, {"x": [1, 2, 3]})
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"v": "float64"}), inputs=[])
    pipe.add_step("b", DropnaStep(), inputs=["a"])
    html = pipe._repr_html_()
    assert "a" in html
    assert "b" in html
    # HTML contains arrow direction marker
    assert "&larr;" in html or "<-" in html
'''


def write_dag_tests():
    print("\n=== tests/test_pipeline_dag.py ===")
    p = Path("tests/test_pipeline_dag.py")
    p.write_text(DAG_TESTS)
    compile_check(p)
    print("  OK")


# ============ 8. CHANGELOG + DECISIONS ============

def append_changelog():
    print("\n=== CHANGELOG.md: append v0.9.1b via patch_utils ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.1b — DAG Pipeline" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Добавлено (v0.9.1b — DAG Pipeline)

- **`Pipeline.add_step(name, step, inputs=...)`** — DAG edges (v0.9.1b).
  - `inputs=None` (default): chain semantics — first step = source (`[]`),
    subsequent = `[previous step name]`. **Backward compat** с v0.9.0.
  - `inputs=[]`: explicit source (redundant for first step, validated).
  - `inputs=["a", "b"]`: DAG fan-in. Все names должны существовать;
    Predictor нельзя использовать как input.
- **`Pipeline._build_graph()`** — `{name: [input_names]}`. Resolves
  `inputs=None` chain derivation.
- **`Pipeline._topological_order()`** — DFS topological sort с cycle
  detection (defensive: forward refs запрещены, cycle невозможен by
  construction).
- **`_merge_fanout(dfs)`** — module helper: column-union concat.
  Precondition: same row count across branches (positional alignment).
  Streaming через `zip(*iters)` с per-batch size check.
- **`LambdaTransform(src_col, feature_name, fn, out_type)`** — test
  helper: Arrow-native fn on one source column, appends
  `{src_col}_{feature_name}`.
- **`Pipeline._repr_html_`** — DAG-aware: column "inputs" показывает
  `(source)` или `← a, b` для fan-in.

### Изменено

- **`Pipeline.fit` / `transform` / `predict`** — execution через
  topological order, не list iteration. Multi-input → `_merge_fanout`.
  `transform` возвращает sink pure-transform подграфа (последний
  non-Predictor в reversed order); если pipeline model-only — исходный df.
- **`Pipeline.predict`** — возвращает результат first (и единственного)
  Predictor в topo order (Predictor валидирован как sink на `add_step`).

### Ограничения (v0.9.1b, §11.5)

- **Fan-in MVP: все branches должны иметь same row count.** Dropna /
  row-selection в одной ветви → RuntimeError с clear message.
  Row-selection fan-in (join по `__orig_rid` с пересечением) — v0.9.1c.
- **Fan-in реализация: positional alignment** (не row_id merge).
  Гарантия корректности = все branches идут от одного source и не
  меняют row order. Документировано.
- **Predictor = sink**, нельзя быть input. Проверяется на `add_step`.
- **Forward references запрещены** (`inputs=[name]` где `name` ещё не
  добавлен) → ValueError. Cycle detection — defensive (недостижим).
- **`Pipeline.plan(df)`** — sum across all steps (не меняется под DAG).
  Per-DAG-step state estimation — v0.9.1c backlog.

### Тесты (v0.9.1b)

- +14 fast (`test_pipeline_dag.py`): chain backward compat, two/three
  branch fan-in, diamond (raw → a, raw → b, merge, model), row-mismatch
  RuntimeError, unknown input, Predictor-as-input, explicit source
  validation, topo order (chain, fan-out), transform stops at model,
  predict full DAG, repr_html DAG edges.

- fast: **~1309** (+14). slow: **9** (unchanged).
'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


def update_decisions_11_5():
    print("\n=== DECISIONS.md §11.5: DAG fan-in note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.1b (DAG fan-in MVP)" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        print("  WARNING: §11.5 not found — append via warning")
        s = s.rstrip() + "\n\n**v0.9.1b (DAG fan-in MVP):** positional "
        s += "alignment — all branches must have same row count. See "
        s += "_merge_fanout docstring.\n"
        p.write_text(s)
        return

    note = '''**v0.9.1b (DAG fan-in MVP):** `_merge_fanout` требует **одинаковое
row count** во всех branches (positional alignment). Dropna /
row-selection в одной ветви → RuntimeError. Row-selection fan-in
(join по `__orig_rid` с пересечением) — v0.9.1c backlog. Predictor =
sink; нельзя быть input. Forward references запрещены (DAG ациклический
by construction; cycle detection — defensive).

'''
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK: §11.5 DAG note")


# ============ main ============

def main():
    print("=== v0.9.1b: DAG Pipeline ===")

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

    add_merge_fanout()
    patch_pipeline_init()
    patch_add_step()
    add_graph_helpers()
    patch_fit_transform_predict()
    patch_repr_html()
    add_lambda_transform()
    write_dag_tests()
    append_changelog()
    update_decisions_11_5()

    print("\n=== pytest: new DAG tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pipeline_dag.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** DAG TESTS FAILED ***")
        print("Fix in place (§13.4/146 backlog: no WIP commit on red).")
        sys.exit(1)

    print("\n=== pytest: pipeline backward compat + E2E ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py", "tests/test_pipeline_e2e.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** BACKWARD COMPAT FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards via patch_utils ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

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
        "v0.9.1b: DAG Pipeline — add_step(inputs=[...]), fan-in via "
        "_merge_fanout (column union, same-row precondition), "
        "topological execution (fit/transform/predict), LambdaTransform "
        "test helper, DAG-aware _repr_html_; §11.5 fan-in MVP note"])

    print("\n==> v0.9.1b applied.")
    print("NEXT: v0.9.1c (save/load to_json/from_json).")


if __name__ == "__main__":
    main()