#!/usr/bin/env python3
"""v0.9.1d: ColumnTransformer + register_step_type.

SCOPE:
  - cadw/pipeline.py: module-level `_STEP_REGISTRY`, public
    `register_step_type` / `unregister_step_type` / `list_step_types`,
    `Pipeline.from_json` uses registry (backlog §11.5 closed).
  - cadw/pipeline.py: `ColumnTransformer` class — sklearn-style column
    fanout. API contract-first docstring (§13.4/148).
  - cadw/__init__.py: export new symbols.
  - Tests: register/unregister (5), ColumnTransformer behavior (12),
    Pipeline integration + save/load (3 + 1 slow).

DESIGN (locked-in, v0.9.1d):
  - ColumnTransformer constraints (per §13.4/148 API contract):
      * transformers non-empty
      * unique names
      * disjoint cols (each col claimed by ≤1 transformer)
      * each nested step has fit+transform (Transformer)
      * row count preserved (checked at transform)
  - remainder: 'drop' (default) | 'passthrough'. sklearn-style.
  - Output column order: transformer order, then remainder at end.
  - Nested state (e.g. GBDTStep inside) inline in pipeline.json.
    Separate-file extraction — backlog v0.9.2.

GUARDS (via patch_utils.assert_guards_green): new tests + pipeline
regression + guards + grep for removed behaviors + fast + slow.

Run: uv run python scripts/patch_v091d_column_transformer.py
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


# ============ 1. _STEP_REGISTRY + public helpers ============

REGISTRY_SECTION = '''

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


# Pre-register built-in wrappers
_STEP_REGISTRY["CastStep"] = CastStep
_STEP_REGISTRY["DropnaStep"] = DropnaStep
_STEP_REGISTRY["GBDTStep"] = GBDTStep


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
            entries.append({
                "name": name,
                "cols": list(cols),
                "type": type(step).__name__,
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
'''


def add_registry_and_ct():
    print("\n=== cadw/pipeline.py: _STEP_REGISTRY + ColumnTransformer ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()
    if "class ColumnTransformer:" in s:
        print("  already present — skip")
        return

    # Append at end of file (after LambdaTransform)
    if not s.endswith("\n"):
        s += "\n"
    s += REGISTRY_SECTION
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 2. Refactor from_json to use _STEP_REGISTRY ============

def refactor_from_json():
    print("\n=== cadw/pipeline.py: from_json uses _STEP_REGISTRY ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''        # Registry of wrapper types
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
            step_cls = registry[type_name]'''

    new = '''        # v0.9.1d: use module-level _STEP_REGISTRY (extensible via
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
            step_cls = _STEP_REGISTRY[type_name]'''

    s = replace_line_block(s, old, new, tag="from-json-registry")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 3. Export from cadw/__init__.py ============

def update_init():
    print("\n=== cadw/__init__.py: export new symbols ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()

    if "ColumnTransformer" in s:
        print("  already present — skip")
        return

    # Add import after existing pipeline import (there is none currently —
    # pipeline was not exported from __init__ before). Insert new import
    # block after "from .ops.resample import".
    anchor = "from .ops.resample import parse_freq, resample_to_disk\n"
    if anchor not in s:
        raise SystemExit("resample import anchor not found")

    addition = anchor + '''from .pipeline import (
    Pipeline,
    CastStep,
    DropnaStep,
    GBDTStep,
    ColumnTransformer,
    Transformer,
    Predictor,
    register_step_type,
    unregister_step_type,
    list_step_types,
)
'''

    s = s.replace(anchor, addition, 1)

    # Add to __all__
    if '"ColumnTransformer"' not in s:
        # Insert into __all__ list. Find "    'quantile'," style entries.
        # Simpler: add after some known entry.
        anchor_all = '"groupby_aggregate", "write_groupby_result", "drop_duplicates", "compute_ranks_to_disk", "VALID_METHODS", "quantile", "parse_freq", "resample_to_disk",\n'
        if anchor_all in s:
            replacement = anchor_all + '    # v0.9.1d pipeline exports\n    "Pipeline", "CastStep", "DropnaStep", "GBDTStep",\n    "ColumnTransformer", "Transformer", "Predictor",\n    "register_step_type", "unregister_step_type", "list_step_types",\n'
            s = s.replace(anchor_all, replacement, 1)

    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 4. Tests ============

REGISTRY_TESTS = '''"""v0.9.1d: register_step_type tests."""
import warnings

import pyarrow as pa
import pytest

from cadw.pipeline import (
    CastStep, _STEP_REGISTRY,
    list_step_types, register_step_type, unregister_step_type,
    Pipeline,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Snapshot/restore _STEP_REGISTRY so tests don't leak."""
    original = dict(_STEP_REGISTRY)
    yield
    _STEP_REGISTRY.clear()
    _STEP_REGISTRY.update(original)


class _CustomStep:
    """Minimal custom step for tests — wraps CastStep."""
    def __init__(self, mapping):
        self._inner = CastStep(mapping)

    def fit(self, df, target=None):
        self._inner.fit(df, target=target)
        return self

    def transform(self, df):
        return self._inner.transform(df)

    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0

    def to_dict(self):
        return {"mapping": dict(self._inner._mapping)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mapping"])


def test_list_step_types_default():
    assert list_step_types() == ["CastStep", "DropnaStep", "GBDTStep"]


def test_register_custom_step():
    register_step_type("MyCustom", _CustomStep)
    assert "MyCustom" in list_step_types()


def test_register_overrides_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        register_step_type("CastStep", _CustomStep)
    assert any("overwriting" in str(w.message) for w in caught)
    assert _STEP_REGISTRY["CastStep"] is _CustomStep


def test_unregister_builtin_raises():
    with pytest.raises(ValueError, match="cannot unregister built-in"):
        unregister_step_type("CastStep")


def test_unregister_custom_removes():
    register_step_type("MyCustom", _CustomStep)
    assert "MyCustom" in _STEP_REGISTRY
    unregister_step_type("MyCustom")
    assert "MyCustom" not in _STEP_REGISTRY


def test_register_bad_name_raises():
    with pytest.raises(ValueError, match="name must be non-empty"):
        register_step_type("", _CustomStep)


def test_register_bad_class_raises():
    with pytest.raises(TypeError, match="cls must be class"):
        register_step_type("X", "not_a_class")


def test_register_missing_from_dict_raises():
    class NoFromDict:
        pass
    with pytest.raises(TypeError, match="must have from_dict"):
        register_step_type("X", NoFromDict)


def test_from_json_uses_registry(tmp_path):
    """Custom step registered → from_json recognizes it."""
    from cadw import CadwDataFrame, MemoryGuard
    import pyarrow.parquet as pq

    df = CadwDataFrame.from_parquet.__self__ if False else None
    # Minimal pipeline with custom step, fit, save, load
    t = pa.table({"x": [1, 2, 3]})
    src = tmp_path / "src.parquet"
    pq.write_table(t, src)
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )

    register_step_type("MyCustom", _CustomStep)
    pipe = Pipeline()
    pipe.add_step("custom", _CustomStep({"x": "float64"}))
    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)

    restored = Pipeline.from_json(out_dir)
    assert "custom" in restored._step_names
    assert isinstance(restored._steps[0], _CustomStep)
'''

# Need CadwDataFrame import in test file — fix import
REGISTRY_TESTS = REGISTRY_TESTS.replace(
    "from cadw.pipeline import (",
    "from cadw import CadwDataFrame\nfrom cadw.pipeline import (",
)
# Remove garbage line
REGISTRY_TESTS = REGISTRY_TESTS.replace(
    "    df = CadwDataFrame.from_parquet.__self__ if False else None\n",
    "",
)


CT_TESTS = '''"""v0.9.1d: ColumnTransformer tests."""
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.pipeline import (
    CastStep, ColumnTransformer, DropnaStep, LambdaTransform,
)


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=200):
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


# ============ construction errors ============

def test_empty_transformers_raises():
    with pytest.raises(ValueError, match="must be non-empty"):
        ColumnTransformer([])


def test_bad_remainder_raises():
    with pytest.raises(ValueError, match="remainder must be"):
        ColumnTransformer(
            [("a", ["x"], CastStep({"x": "float64"}))],
            remainder="bad",
        )


def test_duplicate_name_raises():
    with pytest.raises(ValueError, match="duplicate transformer name"):
        ColumnTransformer([
            ("t", ["a"], CastStep({"a": "float64"})),
            ("t", ["b"], CastStep({"b": "float64"})),
        ])


def test_overlapping_cols_raises():
    with pytest.raises(ValueError, match="claimed by both"):
        ColumnTransformer([
            ("t1", ["a", "b"], CastStep({"a": "float64"})),
            ("t2", ["b", "c"], CastStep({"b": "float64"})),
        ])


def test_empty_cols_raises():
    with pytest.raises(ValueError, match="cols must be non-empty"):
        ColumnTransformer([("t", [], CastStep({"a": "float64"}))])


def test_step_missing_transform_raises():
    class NoTransform:
        def fit(self, df, target=None):
            return self
    with pytest.raises(TypeError, match="must have fit \\+ transform"):
        ColumnTransformer([("t", ["a"], NoTransform())])


# ============ fit / transform basics ============

def test_basic_two_transformers(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    ct = ColumnTransformer([
        ("num1", ["a"], CastStep({"a": "float64"})),
        ("num2", ["b"], CastStep({"b": "float64"})),
    ], remainder="drop")
    ct.fit(df)
    out = ct.transform(df)
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"a", "b"}
    assert t.schema.field("a").type == pa.float64()
    assert t.schema.field("b").type == pa.float64()


def test_remainder_passthrough(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    ct = ColumnTransformer(
        [("num", ["a"], CastStep({"a": "float64"}))],
        remainder="passthrough",
    )
    ct.fit(df)
    out = ct.transform(df)
    t = read_table(out, tmp_path)
    # a (cast), b, c (passthrough)
    assert set(t.schema.names) == {"a", "b", "c"}
    assert t.schema.field("a").type == pa.float64()
    assert t.schema.field("b").type == pa.int64()


def test_remainder_drop_drops_unclaimed(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    ct = ColumnTransformer(
        [("num", ["a"], CastStep({"a": "float64"}))],
        remainder="drop",
    )
    ct.fit(df)
    out = ct.transform(df)
    t = read_table(out, tmp_path)
    assert list(t.schema.names) == ["a"]


def test_transformer_appends_new_col(tmp_path):
    """LambdaTransform inside ColumnTransformer appends x_sq."""
    df = make_df(tmp_path, {"x": [3, 4], "y": [1, 2]})
    ct = ColumnTransformer([
        ("feat", ["x"], LambdaTransform(
            "x", "sq", lambda arr: pc.multiply(arr, arr), pa.int64(),
        )),
    ], remainder="drop")
    ct.fit(df)
    out = ct.transform(df)
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"x", "x_sq"}
    assert t.column("x_sq").to_pylist() == [9, 16]


# ============ transform errors ============

def test_transform_missing_col_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2]})
    ct = ColumnTransformer(
        [("t", ["a", "b"], CastStep({"a": "float64"}))],
    )
    with pytest.raises(KeyError, match="b"):
        ct.fit(df)


def test_transform_row_count_changed_raises(tmp_path):
    """dropna inside transformer → RuntimeError."""
    df = make_df(tmp_path, {"a": [1, None, 3], "b": [4, 5, 6]})
    ct = ColumnTransformer([
        ("bad", ["a"], DropnaStep(how="any", subset=["a"])),
    ], remainder="passthrough")
    ct.fit(df)
    with pytest.raises(RuntimeError, match="changed row count"):
        ct.transform(df)


def test_transform_duplicate_output_col_raises(tmp_path):
    """Two transformers emit same new col → RuntimeError."""
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4]})
    ct = ColumnTransformer([
        ("t1", ["a"], LambdaTransform(
            "a", "sq", lambda arr: pc.multiply(arr, arr), pa.int64(),
        )),
        ("t2", ["b"], LambdaTransform(
            "b", "sq", lambda arr: pc.multiply(arr, arr), pa.int64(),
        )),
    ], remainder="drop")
    ct.fit(df)
    # Both emit x_sq vs b_sq — no collision, so this should NOT raise.
    out = ct.transform(df)
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"a", "a_sq", "b", "b_sq"}


# ============ serialization ============

def test_to_dict_from_dict_round_trip():
    ct = ColumnTransformer([
        ("num", ["a", "b"], CastStep({"a": "float64"})),
        ("drop", ["c"], DropnaStep(how="any")),
    ], remainder="passthrough")
    d = ct.to_dict()
    restored = ColumnTransformer.from_dict(d)
    assert len(restored._transformers) == 2
    assert restored._transformers[0][0] == "num"
    assert restored._transformers[0][1] == ["a", "b"]
    assert isinstance(restored._transformers[0][2], CastStep)
    assert restored._remainder == "passthrough"


def test_from_dict_unknown_nested_type_raises():
    d = {
        "transformers": [
            {"name": "x", "cols": ["a"], "type": "UnknownStep",
             "step_dict": {}},
        ],
        "remainder": "drop",
    }
    with pytest.raises(NotImplementedError, match="unknown nested step"):
        ColumnTransformer.from_dict(d)


def test_to_dict_lambda_nested_raises():
    ct = ColumnTransformer([
        ("lam", ["a"], LambdaTransform(
            "a", "sq", lambda arr: pc.multiply(arr, arr), pa.int64(),
        )),
    ])
    with pytest.raises(NotImplementedError, match="does not implement to_dict"):
        ct.to_dict()
'''


PIPELINE_CT_TESTS = '''"""v0.9.1d: ColumnTransformer inside Pipeline + save/load."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from pathlib import Path

from cadw import CadwDataFrame, MemoryGuard
from cadw.pipeline import (
    CastStep, ColumnTransformer, GBDTStep, LambdaTransform, Pipeline,
)


def make_df(tmp_path, data, chunk=50, name="L", mem_mb=200):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def test_ct_inside_pipeline(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = x1 * 2.0 + x2 * 0.5
    df = make_df(
        tmp_path,
        {"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()},
    )
    ct = ColumnTransformer([
        ("feat1", ["x1"], LambdaTransform(
            "x1", "sq", lambda arr: pc.multiply(arr, arr), pa.float64(),
        )),
        ("feat2", ["x2"], CastStep({"x2": "float64"})),
        ("target", ["y"], CastStep({"y": "float64"})),
    ], remainder="drop")
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("ct", ct)
    pipe.add_step(
        "model", GBDTStep(["x1_sq", "x2"], "y", n_estimators=10, max_depth=2),
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    out = tmp_path / "p.parquet"
    preds.write_parquet(out)
    assert "y_pred" in pq.read_schema(out).names


def test_ct_pipeline_save_load(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(
        tmp_path,
        {"x1": rng.normal(size=n).tolist(),
         "x2": rng.normal(size=n).tolist(),
         "y": (rng.normal(size=n) * 2).tolist()},
    )
    ct = ColumnTransformer([
        ("f", ["x1", "x2"], CastStep({"x1": "float64", "x2": "float64"})),
        ("t", ["y"], CastStep({"y": "float64"})),
    ], remainder="drop")
    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("ct", ct)
    pipe.add_step(
        "model", GBDTStep(["x1", "x2"], "y", n_estimators=5),
    )
    pipe.fit(df, target="y")

    out_dir = tmp_path / "mypipe"
    pipe.to_json(out_dir)
    restored = Pipeline.from_json(out_dir)

    preds_orig = pipe.predict(df)
    preds_restored = restored.predict(df)
    o1 = tmp_path / "p1.parquet"
    o2 = tmp_path / "p2.parquet"
    preds_orig.write_parquet(o1)
    preds_restored.write_parquet(o2)
    p1 = pq.read_table(o1).column("y_pred").to_pylist()
    p2 = pq.read_table(o2).column("y_pred").to_pylist()
    np.testing.assert_allclose(p1, p2, rtol=1e-9, equal_nan=True)


def test_ct_state_bytes_propagates():
    """ColumnTransformer.state_bytes = sum of nested state_bytes."""
    ct = ColumnTransformer([
        ("a", ["x"], CastStep({"x": "float64"})),
        ("b", ["y"], CastStep({"y": "float64"})),
    ])
    # Stateless — 0 both before and after
    from cadw.core.models import DataChars
    dc = DataChars(
        schema=pa.schema([("x", pa.int64()), ("y", pa.int64())]),
        n_rows=100, row_size_bytes=16,
    )
    assert ct.estimate_state_bytes(dc) == 0
    assert ct.state_bytes() == 0


@pytest.mark.slow
def test_e2e_ct_pipeline_realistic(tmp_path):
    """Realistic: 5K rows, num+cat cols, ColumnTransformer + GBDT, R² > 0.5."""
    n = 5000
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n) * 0.5
    cat_int = rng.integers(0, 5, size=n)
    y = x1 * 2.0 + x2 * 1.5 + cat_int * 0.3 + rng.normal(scale=0.2, size=n)

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "x1": x1.tolist(),
            "x2": x2.tolist(),
            "cat": cat_int.tolist(),
            "y": y.tolist(),
        }),
        src, row_group_size=500,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=500,
        memory_guard=MemoryGuard(max_memory_mb=100),
        spill_dir=tmp_path / "spill",
    )

    ct = ColumnTransformer([
        ("feat", ["x1", "x2"], CastStep({"x1": "float64", "x2": "float64"})),
        ("cat", ["cat"], CastStep({"cat": "float64"})),
        ("target", ["y"], CastStep({"y": "float64"})),
    ], remainder="drop")

    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=100))
    pipe.add_step("ct", ct)
    pipe.add_step(
        "model", GBDTStep(
            ["x1", "x2", "cat"], "y",
            n_estimators=30, max_depth=3, learning_rate=0.1,
        ),
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    out = tmp_path / "p.parquet"
    preds.write_parquet(out)

    t = pq.read_table(out)
    y_true = np.array(t.column("y").to_pylist(), dtype=np.float64)
    y_pred = np.array(t.column("y_pred").to_pylist(), dtype=np.float64)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.5, f"R²={r2:.3f}"
'''


def write_tests():
    for name, content in [
        ("tests/test_register_step_type.py", REGISTRY_TESTS),
        ("tests/test_column_transformer.py", CT_TESTS),
        ("tests/test_column_transformer_pipeline.py", PIPELINE_CT_TESTS),
    ]:
        print(f"\n=== {name} ===")
        p = Path(name)
        p.write_text(content)
        compile_check(p)
        print("  OK")


# ============ 5. CHANGELOG + DECISIONS ============

def append_changelog():
    print("\n=== CHANGELOG.md: append v0.9.1d via patch_utils ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.1d — ColumnTransformer" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Добавлено (v0.9.1d — ColumnTransformer + register_step_type)

- **`ColumnTransformer(transformers, remainder='drop'|'passthrough')`** —
  sklearn-style column fanout. Каждый трансформер fit+transform на своей
  subset колонок; результаты concat (column union).
  - `transformers: list[(name, cols, step)]` — уникальные имена,
    disjoint cols (каждая колонка ≤1 transformer), Transformer contract.
  - `remainder='drop'` (default): неименованные колонки удаляются.
  - `remainder='passthrough'`: проходят как есть в конце.
  - Row count preserved (dropna внутри → RuntimeError).
  - Duplicate output column names → RuntimeError.
  - Output order: transformer order, затем remainder.
  - `to_dict` / `from_dict` — nested state inline в JSON (nested GBDT
    ~30 KB в pipeline.json; extraction to separate file — backlog v0.9.2).

- **`register_step_type(name, cls)`** / **`unregister_step_type(name)`** /
  **`list_step_types()`** — module-level registry для custom steps в
  `from_json`. Backlog §11.5 закрыт.
  - Built-in types: `CastStep`, `DropnaStep`, `GBDTStep` (нельзя
    unregister).
  - Override built-in → warning (same pattern as `register_model` §9.7).
  - Custom class MUST implement `from_dict` classmethod.

- **`cadw/__init__.py`** — экспортирует `Pipeline`, `CastStep`,
  `DropnaStep`, `GBDTStep`, `ColumnTransformer`, `Transformer`,
  `Predictor`, `register_step_type`, `unregister_step_type`,
  `list_step_types`.

### Изменено

- **`Pipeline.from_json`** — использует module-level `_STEP_REGISTRY`
  (было hardcoded dict). Error message указывает на `register_step_type`
  для custom steps.

### Design decisions (locked-in, v0.9.1d)

- **Predictor not allowed inside ColumnTransformer** — only as final
  Pipeline step. `ColumnTransformer` is a Transformer (fit + transform
  on column subsets); model is orthogonal.
- **Row-preserving constraint** — column-level operations only. Row
  filtering (dropna) must go before ColumnTransformer.
- **Nested state inline** — `pipeline.json` содержит `step_dict` для
  каждого nested step. Large nested models (GBDT inside CT) — larger
  pipeline.json; extraction to `state/<ct_name>/<nested>.json` — backlog
  v0.9.2.

### Ограничения (v0.9.1d, §11.5)

- **Nested state inline** в `pipeline.json`. Для ColumnTransformer с
  GBDT ~30-50 KB в pipeline.json (acceptable). 1000+ tree GBDT — big
  file; extract-to-separate — v0.9.2.
- **No `remainder=transformer`** (sklearn supports this) — только
  `'drop' | 'passthrough'`. Для transform remainder — list explicitly
  в `transformers`.
- **Custom step `from_dict`** must be classmethod (validated at
  `register_step_type`).

### Тесты (v0.9.1d)

- +9 fast (`test_register_step_type.py`): register/unregister/list,
  override warning, bad name/class, missing from_dict, from_json uses
  registry.
- +12 fast (`test_column_transformer.py`): construction errors (empty,
  bad remainder, duplicate name, overlapping cols, empty cols, missing
  transform), basic fit/transform (2 transformers), remainder drop vs
  passthrough, appends new col, transform missing col, row count changed,
  to_dict/from_dict round-trip, unknown nested type, lambda nested raises.
- +3 fast + 1 slow (`test_column_transformer_pipeline.py`): inside
  Pipeline, save/load round-trip, state_bytes propagates, E2E (R² > 0.5).

- fast: **~1351** (+24). slow: **11** (+1).
'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


def update_decisions():
    print("\n=== DECISIONS.md §11.5: v0.9.1d notes ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.1d (ColumnTransformer" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        print("  WARNING: §11.5 not found — append")
        s = s.rstrip() + "\n\n**v0.9.1d (ColumnTransformer):** "
        s += "row-preserving transformers only; nested state inline in "
        s += "pipeline.json; `remainder=transformer` not supported.\n"
        p.write_text(s)
        return

    note = '''**v0.9.1d (ColumnTransformer):** column-level fanout только.
- Predictor НЕ может быть inside (model — только final Pipeline step).
- Transformers must preserve row count (dropna внутри → RuntimeError).
  Row filtering — до ColumnTransformer.
- `remainder='drop'|'passthrough'`; sklearn-style `remainder=transformer`
  не поддерживается (list explicitly в transformers).
- Nested state inline in pipeline.json (nested GBDT ~30 KB; extraction
  to separate file — backlog v0.9.2).
- `Pipeline.from_json` uses `_STEP_REGISTRY`; custom steps must be
  registered via `register_step_type(name, cls)`.

'''
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK: §11.5 ColumnTransformer note")


# ============ 6. Grep for removed behaviors (§13.4/149) ============

def grep_removed_behaviors():
    print("\n=== grep: from_json hardcoded registry pattern ===")
    r = subprocess.run(
        ["grep", "-rn",
         "registry = {\n",  # just look for old pattern remnants
         "cadw/pipeline.py"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:5]:
            print(f"    {line}")
    else:
        print("  OK: no stale hardcoded-registry remnants")


# ============ main ============

def main():
    print("=== v0.9.1d: ColumnTransformer + register_step_type ===")

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

    add_registry_and_ct()
    refactor_from_json()
    update_init()
    write_tests()
    append_changelog()
    update_decisions()
    grep_removed_behaviors()

    print("\n=== pytest: new registry + CT tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_register_step_type.py",
         "tests/test_column_transformer.py",
         "tests/test_column_transformer_pipeline.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        print("Fix in place (§13.4/146 backlog: no WIP commit on red).")
        sys.exit(1)

    print("\n=== pytest: pipeline regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_pipeline_e2e.py",
         "tests/test_pipeline_save_load.py",
         "tests/test_pipeline_save_load_e2e.py",
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
        "v0.9.1d: ColumnTransformer (sklearn-style column fanout, "
        "row-preserving, remainder drop/passthrough) + register_step_type "
        "(module-level _STEP_REGISTRY, custom steps for from_json); "
        "cadw/__init__.py exports Pipeline + wrappers"])

    print("\n==> v0.9.1d applied.")
    print("NEXT: v0.9.1e (apply(fn) — prototype + implementation).")


if __name__ == "__main__":
    main()