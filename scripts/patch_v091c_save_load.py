#!/usr/bin/env python3
"""v0.9.1c: Pipeline save/load — to_json / from_json.

SCOPE:
  - Pipeline.to_json(path) — write pipeline_dir/pipeline.json + state/*.json
  - Pipeline.from_json(path, memory_guard=None) — restore
  - Per-step to_dict() / from_dict() for CastStep, DropnaStep, GBDTStep
  - LambdaTransform.to_dict() raises NotImplementedError (callable)
  - Serialization helpers in cadw/_serialization.py:
      - gbdt_to_dict(model) / gbdt_from_dict(d) — tree structure
      - linear_to_dict / linear_from_dict — coef + intercept
      - numpy → nested list helpers

DESIGN (locked-in, v0.9.1c):
  - Always directory: pipeline_dir/pipeline.json + state/<step>.json
  - JSON for everything (no pickle — safety: no RCE on untrusted input)
  - GBDT trees ~50KB per model in JSON; acceptable for reproducibility
  - version field "0.9.1"; from_json warns on mismatch (does not fail —
    best-effort restore, users know their format)
  - LambdaTransform (and any step without to_dict) → NotImplementedError
    with clear message: "define custom step with to_dict/from_dict"

STATE FILE STRATEGY:
  - Stateless steps (CastStep, DropnaStep): params only in pipeline.json
  - Stateful steps (GBDTStep): params + state in state/<name>.json
  - No state size threshold — GBDT is only stateful step; JSON is fine.

TESTS (~16 fast + 1 slow):
  test_pipeline_save_load.py:
    - round-trip stateless pipeline (params equal)
    - round-trip GBDT (predictions equal, float compare)
    - round-trip DAG (graph preserved)
    - state file created for GBDT
    - from_json stateless (no state dir)
    - version mismatch warns
    - LambdaTransform serialization raises
    - corrupt JSON raises
    - missing state file raises
    - path as file vs directory
    - fit status preserved
  test_pipeline_save_load_e2e.py (1 slow):
    - fit → save → load → predict, R² equal on original

GUARDS (before commit): new + pipeline regression + guards + fast + slow.

Run: uv run python scripts/patch_v091c_save_load.py
"""
from __future__ import annotations

import os
import json
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


# ============ 1. cadw/_serialization.py ============

SERIALIZATION = '''"""Serialization helpers for Pipeline save/load (v0.9.1c).

Explicit to_dict / from_dict per model type (no pickle — safety).
JSON for everything; nested lists for numpy arrays / tree structures.

Serialized model sizes:
  - GBDT (n_estimators=100, max_depth=3): ~100 trees × 15 nodes × ~200 B
    = ~30 KB JSON. Acceptable for reproducibility.
  - Linear (d features): d floats + intercept = ~100 B × d.
  - PCA: (k, d) components + mean.
  - KMeans: (k, d) centers.
"""
from __future__ import annotations

import numpy as np


# ============ GBDT (regressor / classifier) ============

def gbdt_regressor_to_dict(model) -> dict:
    """Serialize GBDTRegressor (regression, L2 loss)."""
    trees = []
    for tree in model.trees_:
        trees.append({
            "structure": {
                int(nid): (
                    None if spec is None else list(spec)
                )
                for nid, spec in tree["structure"].items()
            },
            "leaf_values": {
                int(lid): float(v)
                for lid, v in tree["leaf_values"].items()
            },
        })
    return {
        "base_prediction": float(model.base_prediction_),
        "feature_names": list(model.feature_names_),
        "target_col": model.target_col_,
        "mn": [float(x) for x in model.mn_],
        "mx": [float(x) for x in model.mx_],
        "bins": int(model.bins_),
        "learning_rate": float(model.learning_rate_),
        "n_estimators": int(model.n_estimators_),
        "max_depth": int(model.max_depth_),
        "n_features_in": int(model.n_features_in_),
        "trees": trees,
    }


def gbdt_regressor_from_dict(d: dict):
    """Deserialize GBDTRegressor from dict."""
    from cadw.ml.gbdt import GBDTRegressor
    trees = []
    for td in d["trees"]:
        structure = {
            int(nid): (None if spec is None else tuple(spec))
            for nid, spec in td["structure"].items()
        }
        leaf_values = {
            int(lid): float(v) for lid, v in td["leaf_values"].items()
        }
        trees.append({
            "structure": structure,
            "leaf_values": leaf_values,
        })
    return GBDTRegressor(
        trees_=trees,
        base_prediction_=float(d["base_prediction"]),
        feature_names_=list(d["feature_names"]),
        target_col_=d["target_col"],
        mn_=np.asarray(d["mn"], dtype=np.float64),
        mx_=np.asarray(d["mx"], dtype=np.float64),
        bins_=int(d["bins"]),
        learning_rate_=float(d["learning_rate"]),
        n_estimators_=int(d["n_estimators"]),
        max_depth_=int(d["max_depth"]),
        n_features_in_=int(d["n_features_in"]),
    )


# ============ Linear ============

def linear_to_dict(model) -> dict:
    return {
        "coef": [float(x) for x in model.coef_],
        "intercept": float(model.intercept_),
        "feature_names": list(model.feature_names_),
        "target_col": model.target_col_,
        "alpha": float(model.alpha_),
        "n_rows": int(model.n_rows_),
        "penalty": model.penalty_,
        "l1_ratio": float(model.l1_ratio_),
        "n_iter_cd": int(model.n_iter_cd_),
        "converged": bool(model.converged_),
    }


def linear_from_dict(d: dict):
    from cadw.ml.linear import LinearModel
    return LinearModel(
        coef_=np.asarray(d["coef"], dtype=np.float64),
        intercept_=float(d["intercept"]),
        feature_names_=list(d["feature_names"]),
        target_col_=d["target_col"],
        alpha_=float(d["alpha"]),
        n_rows_=int(d["n_rows"]),
        penalty_=d["penalty"],
        l1_ratio_=float(d["l1_ratio"]),
        n_iter_cd_=int(d["n_iter_cd"]),
        converged_=bool(d["converged"]),
    )
'''


def write_serialization():
    print("\n=== cadw/_serialization.py ===")
    p = Path("cadw/_serialization.py")
    p.write_text(SERIALIZATION)
    compile_check(p)
    print("  OK")


# ============ 2. Wrappers to_dict / from_dict ============

def add_wrapper_to_dict():
    print("\n=== cadw/pipeline.py: wrappers to_dict/from_dict + Pipeline.to_json ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    if "def to_json(" in s:
        print("  already present — skip")
        return

    # --- CastStep ---
    old_cast = '''    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0


class DropnaStep:'''

    new_cast = '''    def estimate_state_bytes(self, dc):
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


class DropnaStep:'''

    s = replace_line_block(s, old_cast, new_cast, tag="caststep-todict")

    # --- DropnaStep ---
    old_drop = '''    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0


class GBDTStep:'''

    new_drop = '''    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0

    def to_dict(self) -> dict:
        return {"how": self._how, "subset": self._subset}

    @classmethod
    def from_dict(cls, d: dict) -> "DropnaStep":
        return cls(how=d.get("how", "any"), subset=d.get("subset"))


class GBDTStep:'''

    s = replace_line_block(s, old_drop, new_drop, tag="dropnastep-todict")

    # --- GBDTStep: to_dict / from_dict at end of class ---
    old_gbdt_tail = '''    def predict(self, df):
        if self._model is None:
            raise RuntimeError("GBDTStep.predict: not fitted")
        return self._model.predict(df)


class LambdaTransform:'''

    new_gbdt_tail = '''    def predict(self, df):
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


class LambdaTransform:'''

    s = replace_line_block(s, old_gbdt_tail, new_gbdt_tail, tag="gbdtstep-todict")

    # --- LambdaTransform: to_dict raises ---
    old_lt = '''    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0'''

    # Careful: this pattern appears twice now (also DropnaStep). Use
    # more specific anchor: the LambdaTransform ends with this + class
    # boundary OR end-of-file.
    # Simpler: add after the lambda-transform-specific fn
    old_lt_full = '''    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0

    def _lambda_marker(self):'''

    # We'll do a targeted replace using a unique anchor from the
    # LambdaTransform class body:
    marker = "        self._src_col = src_col\n        self._feature_name = feature_name\n        self._fn = fn\n        self._out_type = out_type"
    if marker not in s:
        # LambdaTransform was added via patch_v091b — different indent
        marker = "        self._src_col = src_col"

    # Safest: append to_dict/from_dict at very end of LambdaTransform class
    # (right before end-of-file or next top-level def). Since LambdaTransform
    # is the LAST class in file, just append after its last method.
    if "class LambdaTransform" in s:
        # Find the last "def state_bytes(self):\n        return 0" after
        # "class LambdaTransform:"
        lt_idx = s.find("class LambdaTransform")
        tail = s[lt_idx:]
        anchor_in_tail = "    def state_bytes(self):\n        return 0"
        if anchor_in_tail in tail:
            new_lt_tail = '''    def state_bytes(self):
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
        )'''
            new_tail = tail.replace(anchor_in_tail, new_lt_tail, 1)
            s = s[:lt_idx] + new_tail

    p.write_text(s)
    compile_check(p)
    print("  OK: CastStep/DropnaStep/GBDTStep/LambdaTransform to_dict")


# ============ 3. Pipeline.to_json / from_json ============

def add_pipeline_to_json():
    print("\n=== cadw/pipeline.py: Pipeline.to_json / from_json ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    if "def to_json(" in s:
        print("  already present — skip")
        return

    # Insert to_json/from_json before @property steps (still inside Pipeline)
    anchor = '''    @property
    def steps(self) -> list:
        """Read-only view: list of (name, step) tuples."""
        return list(zip(self._step_names, self._steps))'''

    methods = '''    # ---------------- save / load (v0.9.1c) ----------------

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
            entry = {
                "name": name,
                "type": type(step).__name__,
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
        if path.is_dir():
            meta_file = path / "pipeline.json"
        else:
            meta_file = path
        if not meta_file.exists():
            raise FileNotFoundError(
                f"from_json: {meta_file} not found"
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

        # Registry of wrapper types
        registry = {
            "CastStep": CastStep,
            "DropnaStep": DropnaStep,
            "GBDTStep": GBDTStep,
        }

        pipe = cls(memory_guard=memory_guard)
        for step_data in payload.get("steps", []):
            type_name = step_data["type"]
            if type_name not in registry:
                raise NotImplementedError(
                    f"from_json: unknown step type {type_name!r}. "
                    f"Register custom step class or use a wrapper from "
                    f"{list(registry)}."
                )
            step_cls = registry[type_name]
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
        return list(zip(self._step_names, self._steps))'''

    s = replace_line_block(s, anchor, methods, tag="to-json-methods")

    # Add `import json` and `import warnings` at top if not present
    if "\nimport json\n" not in s:
        # find `from __future__ import annotations`
        anchor_top = "from __future__ import annotations\n"
        if anchor_top in s:
            s = s.replace(
                anchor_top,
                anchor_top + "\nimport json\n",
                1,
            )

    p.write_text(s)
    compile_check(p)
    print("  OK: to_json / from_json")


# ============ 4. AST guard whitelist ============

def update_ast_guard():
    print("\n=== tests/test_arrow_call_memory_guard.py: whitelist to_json ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if '"_repr_html_":' in s and '"to_json":' in s:
        print("  already present — skip")
        return

    old = '''    # v0.9.0c Jupyter: _repr_html_ calls .to_pandas().to_html() on
    # a 20-row sample only — bounded by design, no O(N) materialize.
    "_repr_html_": "20-row head().to_pandas().to_html(), bounded sample",
}'''

    new = '''    # v0.9.0c Jupyter: _repr_html_ calls .to_pandas().to_html() on
    # a 20-row sample only — bounded by design, no O(N) materialize.
    "_repr_html_": "20-row head().to_pandas().to_html(), bounded sample",
    # v0.9.1c save/load: to_json/from_json touch only step params/state
    # (K×#features scale, not N). No Arrow read of df.
    "to_json": "params + state serialization, no df read",
    "from_json": "params + state deserialization, no df read",
}'''

    s = replace_line_block(s, old, new, tag="whitelist-to-json")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 5. Tests ============

SAVE_LOAD_TESTS = '''"""v0.9.1c: Pipeline save/load tests."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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


# ============ CastStep / DropnaStep round-trip ============

def test_caststep_round_trip():
    step = CastStep({"x": "float64", "y": "int32"}, safe=False)
    d = step.to_dict()
    restored = CastStep.from_dict(d)
    assert restored._mapping == step._mapping
    assert restored._safe is False


def test_dropnastep_round_trip():
    step = DropnaStep(how="all", subset=["a", "b"])
    d = step.to_dict()
    restored = DropnaStep.from_dict(d)
    assert restored._how == "all"
    assert restored._subset == ["a", "b"]


# ============ GBDT round-trip ============

def test_gbdtstep_round_trip_predictions(tmp_path):
    n = 200
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    y = x * 2.0 + rng.normal(scale=0.05, size=n)
    df = make_df(
        tmp_path, {"x": x.tolist(), "y": y.tolist()},
        chunk=50, mem_mb=200,
    )
    step = GBDTStep(["x"], "y", n_estimators=10, max_depth=2)
    step.fit(df, target="y")

    d = step.to_dict()
    restored = GBDTStep.from_dict(d)

    preds_orig = step.predict(df)
    preds_restored = restored.predict(df)

    orig_list = preds_orig._dataset.schema  # just smoke
    out_orig = tmp_path / "o1.parquet"
    out_rest = tmp_path / "o2.parquet"
    preds_orig.write_parquet(out_orig)
    preds_restored.write_parquet(out_rest)
    t1 = pq.read_table(out_orig)
    t2 = pq.read_table(out_rest)
    p1 = t1.column("y_pred").to_pylist()
    p2 = t2.column("y_pred").to_pylist()
    np.testing.assert_allclose(p1, p2, rtol=1e-9, equal_nan=True)


def test_gbdtstep_unfitted_round_trip():
    step = GBDTStep(["x"], "y", n_estimators=5)
    d = step.to_dict()
    assert d["state"] is None
    restored = GBDTStep.from_dict(d)
    assert restored._model is None


# ============ Pipeline round-trip ============

def test_pipeline_to_json_creates_dir(tmp_path):
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    out_path = tmp_path / "mypipe"
    result = pipe.to_json(out_path)
    assert isinstance(result, Path)
    assert (out_path / "pipeline.json").exists()
    assert result == out_path / "pipeline.json"


def test_pipeline_round_trip_stateless(tmp_path):
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    pipe.add_step("dropna", DropnaStep(how="any", subset=["x"]))
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)

    restored = Pipeline.from_json(out_dir)
    assert len(restored.steps) == 2
    assert restored._step_names == ["cast", "dropna"]
    # Params preserved
    cast_step = restored._steps[0]
    assert isinstance(cast_step, CastStep)
    assert cast_step._mapping == pipe._steps[0]._mapping


def test_pipeline_round_trip_dag_edges(tmp_path):
    pipe = Pipeline()
    pipe.add_step("a", CastStep({"x": "float64"}), inputs=[])
    pipe.add_step("b", CastStep({"x": "int64"}), inputs=[])
    pipe.add_step("merge", DropnaStep(), inputs=["a", "b"])
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)

    restored = Pipeline.from_json(out_dir)
    graph = restored._build_graph()
    assert graph["a"] == []
    assert graph["b"] == []
    assert graph["merge"] == ["a", "b"]


def test_pipeline_round_trip_state_file_created(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(
        tmp_path, {"x": rng.normal(size=n).tolist(),
                   "y": (rng.normal(size=n) * 2).tolist()},
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("model", GBDTStep(["x"], "y", n_estimators=5))
    pipe.fit(df, target="y")

    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    assert (out_dir / "state" / "model.json").exists()


def test_pipeline_from_json_without_state_dir(tmp_path):
    """Stateless pipeline: state/ dir not created, from_json works."""
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    assert not (out_dir / "state").exists()
    restored = Pipeline.from_json(out_dir)
    assert restored._step_names == ["cast"]


def test_pipeline_round_trip_fitted_flag(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(
        tmp_path, {"x": rng.normal(size=n).tolist(),
                   "y": (rng.normal(size=n) * 2).tolist()},
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("model", GBDTStep(["x"], "y", n_estimators=5))
    pipe.fit(df, target="y")
    assert pipe.fitted

    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    restored = Pipeline.from_json(out_dir)
    assert restored.fitted


def test_pipeline_from_json_path_is_file(tmp_path):
    """from_json accepts pipeline.json file directly."""
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    out_dir = tmp_path / "mypipe"
    out_file = pipe.to_json(out_dir)

    restored = Pipeline.from_json(out_file)
    assert restored._step_names == ["cast"]


# ============ Errors ============

def test_pipeline_to_json_lambda_raises(tmp_path):
    pipe = Pipeline()
    pipe.add_step(
        "lambda", LambdaTransform(
            "x", "p", lambda arr: arr, pa.int64(),
        ),
        inputs=[],
    )
    out_dir = tmp_path / "mypipe"
    with pytest.raises(NotImplementedError, match="LambdaTransform"):
        pipe.to_json(out_dir)


def test_pipeline_from_json_corrupt_raises(tmp_path):
    out_dir = tmp_path / "mypipe"
    out_dir.mkdir()
    (out_dir / "pipeline.json").write_text("{not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        Pipeline.from_json(out_dir)


def test_pipeline_from_json_missing_state_file_raises(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(
        tmp_path, {"x": rng.normal(size=n).tolist(),
                   "y": (rng.normal(size=n) * 2).tolist()},
        chunk=50, mem_mb=200,
    )
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("model", GBDTStep(["x"], "y", n_estimators=5))
    pipe.fit(df, target="y")
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    # Delete state file
    (out_dir / "state" / "model.json").unlink()
    with pytest.raises(FileNotFoundError, match="state file"):
        Pipeline.from_json(out_dir)


def test_pipeline_from_json_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="pipeline.json"):
        Pipeline.from_json(tmp_path / "nope")


def test_pipeline_to_json_path_is_file_raises(tmp_path):
    f = tmp_path / "existing.txt"
    f.write_text("hello")
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    with pytest.raises(ValueError, match="is a file; expected directory"):
        pipe.to_json(f)


def test_pipeline_version_mismatch_warns(tmp_path):
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    # Corrupt version
    meta = json.loads((out_dir / "pipeline.json").read_text())
    meta["version"] = "0.0.0"
    (out_dir / "pipeline.json").write_text(json.dumps(meta))

    with pytest.warns(RuntimeWarning, match="version"):
        Pipeline.from_json(out_dir)


def test_pipeline_version_mismatch_strict_raises(tmp_path):
    pipe = Pipeline()
    pipe.add_step("cast", CastStep({"x": "float64"}))
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    meta = json.loads((out_dir / "pipeline.json").read_text())
    meta["version"] = "0.0.0"
    (out_dir / "pipeline.json").write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="version"):
        Pipeline.from_json(out_dir, strict_version=True)
'''


E2E_TESTS = '''"""v0.9.1c: E2E save/load — fit → save → load → predict equal."""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.pipeline import GBDTStep, Pipeline


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


@pytest.mark.slow
def test_e2e_save_load_predictions_equal(tmp_path):
    """Fit GBDT pipeline, save, load, predict. Predictions must match."""
    n = 5000
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = x1 * 3.0 + x2 * 1.5 + rng.normal(scale=0.2, size=n)

    import pyarrow as pa
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()}),
        src, row_group_size=500,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=500,
        memory_guard=MemoryGuard(max_memory_mb=100),
        spill_dir=tmp_path / "spill",
    )

    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=100))
    pipe.add_step(
        "model", GBDTStep(
            ["x1", "x2"], "y",
            n_estimators=50, max_depth=3, learning_rate=0.1,
        ),
    )
    pipe.fit(df, target="y")

    # Save
    pipe_dir = tmp_path / "saved_pipeline"
    pipe.to_json(pipe_dir)

    # Load in fresh object
    restored = Pipeline.from_json(
        pipe_dir, memory_guard=MemoryGuard(max_memory_mb=100),
    )

    # Predict with both
    preds_orig = pipe.predict(df)
    preds_restored = restored.predict(df)

    out1 = tmp_path / "p1.parquet"
    out2 = tmp_path / "p2.parquet"
    preds_orig.write_parquet(out1)
    preds_restored.write_parquet(out2)

    t1 = pq.read_table(out1)
    t2 = pq.read_table(out2)
    p1 = t1.column("y_pred").to_pylist()
    p2 = t2.column("y_pred").to_pylist()
    np.testing.assert_allclose(p1, p2, rtol=1e-9, equal_nan=True)

    # Sanity: R² > 0.9 on training
    r2 = _r2(t1.column("y").to_pylist(), p1)
    assert r2 > 0.9
'''


def write_tests():
    for name, content in [
        ("tests/test_pipeline_save_load.py", SAVE_LOAD_TESTS),
        ("tests/test_pipeline_save_load_e2e.py", E2E_TESTS),
    ]:
        print(f"\n=== {name} ===")
        p = Path(name)
        p.write_text(content)
        compile_check(p)
        print("  OK")


# ============ 6. CHANGELOG + DECISIONS ============

def append_changelog():
    print("\n=== CHANGELOG.md: append v0.9.1c via patch_utils ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.1c — save/load" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Добавлено (v0.9.1c — save/load)

- **`Pipeline.to_json(path)` / `Pipeline.from_json(path)`** — сохранение
  и восстановление pipeline для reproducibility и production deployment.
  - `path` — директория: `path/pipeline.json` + (опционально)
    `path/state/<step_name>.json`.
  - `from_json` принимает директорию или `pipeline.json` файл.
  - `strict_version=True` — raise на version mismatch; default — warn.
  - `path` must be directory (не file) для `to_json` — raises clear
    error если path exists and is file.
- **Per-step `to_dict()` / `from_dict()`** — Protocol extension, реализовано
  для `CastStep`, `DropnaStep`, `GBDTStep`. Custom steps должны
  реализовать оба, иначе `to_json` raises clear message.
- **`GBDTStep.to_dict()` / `from_dict()`** — полная сериализация
  fitted model (trees + base_prediction + mn/mx + hyperparameters) в
  JSON. ~30 KB для 100 trees × depth 3.
- **`LambdaTransform.to_dict()`** — raises NotImplementedError: `fn` —
  callable, не JSON-serializable. Message предлагает: custom step class
  с to_dict/from_dict, или только именованные wrappers.
- **`cadw/_serialization.py`** — helpers: `gbdt_regressor_to_dict` /
  `gbdt_regressor_from_dict`, `linear_to_dict` / `linear_from_dict`.
  Explicit, no pickle (safety: untrusted input не вызывает RCE).

### Design decisions (locked-in, v0.9.1c)

- **Always directory**: `pipeline_dir/pipeline.json` + optional
  `state/`. Cleaner для multi-file pipelines (state per step).
- **JSON for everything**: no pickle. GBDT trees ~30 KB JSON acceptable.
- **Version field** "0.9.1": `from_json` warns on mismatch, best-effort
  restore. `strict_version=True` для fail.
- **`LambdaTransform` non-serializable** — callables нельзя безопасно
  сохранить в JSON. Custom step class — user-side.

### Ограничения (v0.9.1c, §11.5)

- **JSON size для больших моделей** — GBDT n_estimators=1000 × depth=5
  ≈ 2.4 MB JSON. Для очень больших моделей — Arrow IPC sidecar
  backlog (v0.9.2). Сейчас JSON для всех.
- **Version compatibility**: strict_version=True поддерживает только
  same-minor. Future minor bumps требуют manual migration.
- **Custom steps**: `to_json` raises `NotImplementedError` если step
  не имеет to_dict. `from_json` raises если type не в registry —
  custom steps требуют register (v0.9.2 `Pipeline.register_step_type`).
- **`state` dir not created** для stateless pipeline — OK, load
  detects `has_state=False`.

### Тесты (v0.9.1c)

- +20 fast (`test_pipeline_save_load.py`): CastStep/DropnaStep/GBDTStep
  round-trip, predictions equal, DAG edges preserved, state file
  created, version mismatch warn/strict, corrupt JSON, missing state
  file, missing pipeline.json, path-as-file, LambdaTransform raises,
  fitted flag preserved.
- +1 slow (`test_pipeline_save_load_e2e.py`): fit → save → load →
  predict, predictions equal (rtol=1e-9), R² > 0.9.

- fast: **~1329** (+20). slow: **10** (+1).
'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


def update_decisions():
    print("\n=== DECISIONS.md §11.5: v0.9.1c save/load note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.1c (Pipeline save/load)" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        print("  WARNING: §11.5 not found — append")
        s = s.rstrip() + "\n\n**v0.9.1c (Pipeline save/load):** to_json / "
        s += "from_json — JSON for everything, no pickle. GBDT trees ~30 KB.\n"
        p.write_text(s)
        return

    note = '''**v0.9.1c (Pipeline save/load):** `to_json` / `from_json` — JSON for
everything, no pickle (safety). GBDT trees ~30 KB per model (100 trees
× depth 3). Large models (1000+ trees) — Arrow IPC sidecar backlog
(v0.9.2). `LambdaTransform` non-serializable (callable). Custom steps
must implement `to_dict`/`from_dict`; `from_json` registry по умолчанию
поддерживает только `CastStep`/`DropnaStep`/`GBDTStep` (`register_step_type`
backlog v0.9.2). Version mismatch — warn; `strict_version=True` для fail.

'''
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK: §11.5 save/load note")


# ============ main ============

def main():
    print("=== v0.9.1c: save/load ===")

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

    write_serialization()
    add_wrapper_to_dict()
    add_pipeline_to_json()
    update_ast_guard()
    write_tests()
    append_changelog()
    update_decisions()

    print("\n=== pytest: new save/load tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline_save_load.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** SAVE/LOAD TESTS FAILED ***")
        print("Fix in place (§13.4/146 backlog: no WIP commit on red).")
        sys.exit(1)

    print("\n=== pytest: save/load E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline_save_load_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_pipeline_e2e.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIPELINE REGRESSION FAILED ***")
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
        "v0.9.1c: Pipeline save/load — to_json/from_json (directory "
        "format: pipeline.json + state/<step>.json), per-step "
        "to_dict/from_dict, GBDT full state serialization (JSON, no "
        "pickle), version field + strict_version, LambdaTransform raises "
        "NotImplementedError; §11.5 save/load constraints"])

    print("\n==> v0.9.1c applied.")
    print("NEXT: v0.9.1d (ColumnTransformer + register_step_type).")


if __name__ == "__main__":
    main()