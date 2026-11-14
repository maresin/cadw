"""v0.9.231-spec: pipeline streaming behavior — baseline + target spec.

BASELINE (pass): freezes current behavior on mock steps. Drift = investigate.
SPEC (xfail): documents target after StreamPipeline refactor (design validated
  in /tmp/lazy_proto.py v2, 8/8 experiments PASS). Remove xfail when refactor
  lands.

Design (validated abstract prototype):
  - Chain = source_factory + fused list[transform_batch]
    (factory => __iter__() re-callable, fresh source each pass)
  - Narrow step: fit(chain) 1-pass, transform_batch(state, batch) per batch
  - Wide step: fit_chain(chain) -> Chain (may materialize internally)
  - Composite: recursive; narrow-only folds into parent (no extra materialize)
  - BufferRegistry: path-scoped names; collision -> RuntimeError

Factory invariant (NEW requirement, was bug in prototype v1):
  - Chain.source MUST be Callable[[], Iterator[Batch]], not a one-shot iterator.
  - In CADW: pass `df._iter_filtered_batches` (method), not `iter(...)` result.
  - Without this, second fit() of downstream step sees empty source (0 rows).

Refactor deferred until MemoryGuard unification complete (Q33 decision).
Spec frozen here so design doesn't get lost.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_pool
from cadw.dataset import Dataset
from cadw.pipeline import GBDTStep, Pipeline
from cadw.scalers import StandardScaler


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 10_000, seed: int = 0) -> CadwDataFrame:
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(pa.table({
        "x1": pa.array(rng.normal(size=n), type=pa.float64()),
        "x2": pa.array(rng.normal(size=n), type=pa.float64()),
        "y": pa.array(
            (rng.random(n) < 0.5).astype(np.float64), type=pa.float64()),
    }), memory_guard=MemoryGuard(max_memory_mb=500))


@pytest.fixture
def scan_counter(monkeypatch):
    counter = {"n": 0}
    orig = CadwDataFrame._iter_filtered_batches

    def patched(self, *a, **kw):
        counter["n"] += 1
        return orig(self, *a, **kw)

    monkeypatch.setattr(CadwDataFrame, "_iter_filtered_batches", patched)
    yield counter


@pytest.fixture
def write_counter(monkeypatch):
    """Count Dataset.from_batches calls = materializations."""
    counter = {"n": 0}
    orig = Dataset.from_batches.__func__  # classmethod unwrap

    def patched(cls, *a, **kw):
        counter["n"] += 1
        return orig(cls, *a, **kw)

    monkeypatch.setattr(Dataset, "from_batches", classmethod(patched))
    yield counter


# ---------------------------------------------------------------------------
# Mock steps — isolate pipeline mechanics from GBDT intrinsic behavior
# ---------------------------------------------------------------------------


class _NarrowMockStep:
    """Narrow mock: fit = 1 scan; transform_batch possible (target API).

    Implements both current Protocol (fit/transform returning new df) and
    target Protocol (transform_batch, is_barrier). Current pipeline calls
    transform(); refactor will call transform_batch() when all-narrow.
    """

    def __init__(self, name: str = "narrow") -> None:
        self.name = name
        self._fitted = False

    def fit(self, df, target=None):
        for _ in df._iter_filtered_batches():
            pass
        self._fitted = True
        return self

    def transform(self, df):
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted")
        out_schema = df._dataset.schema

        def _iter():
            for b in df._iter_filtered_batches():
                yield b  # identity

        new_ds = Dataset.from_batches(
            out_schema, _iter(), df._new_dataset_dir())
        return CadwDataFrame(
            new_ds,
            memory_guard=df._memory_guard,
            spill_dir=df._spill_dir,
        )

    # Target API (not yet called by Pipeline; spec uses for mock)
    def transform_batch(self, batch):
        return batch

    def is_barrier(self) -> bool:
        return False

    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0

    def to_dict(self):
        return {"name": self.name}

    @classmethod
    def from_dict(cls, d):
        return cls(name=d.get("name", "narrow"))


class _WideMockStep:
    """Wide mock: fit = 1 full scan (barrier); no transform_batch."""

    def __init__(self, name: str = "wide") -> None:
        self.name = name
        self._fitted = False
        self._snapshot: list = []

    def fit(self, df, target=None):
        self._snapshot = list(df._iter_filtered_batches())
        self._fitted = True
        return self

    def transform(self, df):
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted")
        out_schema = df._dataset.schema
        snapshot = self._snapshot

        def _iter():
            for b in snapshot:
                yield b

        new_ds = Dataset.from_batches(
            out_schema, _iter(), df._new_dataset_dir())
        return CadwDataFrame(
            new_ds,
            memory_guard=df._memory_guard,
            spill_dir=df._spill_dir,
        )

    def is_barrier(self) -> bool:
        return True

    def estimate_state_bytes(self, dc):
        return 0

    def state_bytes(self):
        return 0

    def to_dict(self):
        return {"name": self.name}

    @classmethod
    def from_dict(cls, d):
        return cls(name=d.get("name", "wide"))


# ---------------------------------------------------------------------------
# Baseline (pass today — freezes current behavior)
# ---------------------------------------------------------------------------


def test_2step_pipeline_scan_count_baseline(scan_counter):
    """BASELINE: [Scaler, GBDT(5)] n=10k = 42 scans (v0.9.231).

    Drift = investigate regression or update baseline intentionally.
    """
    df = _make_df()
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("scale", StandardScaler(columns=["x1", "x2"]))
    p.add_step("gbdt",
               GBDTStep(X_cols=["x1", "x2"], y_col="y", n_estimators=5))
    p.fit_transform(df, target="y")
    assert scan_counter["n"] == 42, (
        f"Baseline drift: {scan_counter['n']} scans (was 42 at v0.9.231)")


def test_narrow_narrow_writes_gt_0_baseline(write_counter):
    """BASELINE: [narrow, narrow] current — eager transform (from_batches
    per step between fit and predict). After refactor: 0 writes (target).
    """
    df = _make_df()
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("n1", _NarrowMockStep("n1"))
    p.add_step("n2", _NarrowMockStep("n2"))
    p.fit_transform(df)
    assert write_counter["n"] >= 1, (
        "Baseline: current pipeline should materialize between narrow steps")


def test_narrow_wide_narrow_writes_ge_2_baseline(write_counter):
    """BASELINE: [narrow, wide, narrow] current — materialize per step."""
    df = _make_df()
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("n1", _NarrowMockStep("n1"))
    p.add_step("w1", _WideMockStep("w1"))
    p.add_step("n2", _NarrowMockStep("n2"))
    p.fit_transform(df)
    assert write_counter["n"] >= 2, (
        "Baseline: current pipeline materializes every transform step")


# ---------------------------------------------------------------------------
# Spec (xfail — target after StreamPipeline refactor)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v0.9.231-spec: [narrow, narrow] target = 0 writes, 3 scans "
        "(2 fits + 1 predict, fused). Current: eager materialize between "
        "steps (baseline test_narrow_narrow_writes_gt_0_baseline). "
        "Remove xfail after StreamPipeline refactor."
    ),
)
def test_narrow_narrow_zero_writes(write_counter, scan_counter):
    """SPEC: [narrow, narrow] = 0 writes, 3 scans (fusion)."""
    df = _make_df()
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("n1", _NarrowMockStep("n1"))
    p.add_step("n2", _NarrowMockStep("n2"))
    p.fit_transform(df)
    assert write_counter["n"] == 0, (
        f"fusion: 0 materialize expected; got {write_counter['n']}")
    assert scan_counter["n"] == 3, (
        f"fusion: 2 fits + 1 predict = 3 scans; got {scan_counter['n']}")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v0.9.231-spec: [narrow, wide, narrow] target = 1 write "
        "(only wide barrier materializes); current = per-step materialize."
    ),
)
def test_narrow_wide_narrow_one_materialize(write_counter):
    """SPEC: [narrow, wide, narrow] = 1 write (wide only)."""
    df = _make_df()
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("n1", _NarrowMockStep("n1"))
    p.add_step("w1", _WideMockStep("w1"))
    p.add_step("n2", _NarrowMockStep("n2"))
    p.fit_transform(df)
    assert write_counter["n"] == 1, (
        f"only wide barrier materializes; got {write_counter['n']} writes")


def test_chain_source_factory_reiterable():
    """BASELINE: pipeline result re-iterable.

    CADW Dataset.iter_batches opens a fresh reader each call ->
    factory invariant already satisfied. Prototype v1 bug was
    unrelated (passed `iter(...)` instead of method-factory).
    This baseline protects against future regression.
    """
    df = _make_df()
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("n1", _NarrowMockStep("n1"))
    p.add_step("n2", _NarrowMockStep("n2"))
    out = p.fit_transform(df)
    first_rows = sum(b.num_rows for b in out._iter_filtered_batches())
    second_rows = sum(b.num_rows for b in out._iter_filtered_batches())
    assert first_rows == second_rows == 10_000, (
        f"re-iterable: both passes must give full data; "
        f"first={first_rows}, second={second_rows}")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v0.9.231-spec: BufferRegistry path-scoped names — collision "
        "on duplicate (path, name) must raise RuntimeError. Validated in "
        "/tmp/lazy_proto.py (experiment H). Current CADW: no BufferRegistry, "
        "no collision detection."
    ),
)
def test_buffer_collision_raises():
    """SPEC: two steps with same path-scoped buffer name -> RuntimeError."""
    from cadw.core.buffer_registry import BufferRegistry  # not yet exists
    reg = BufferRegistry()
    reg.allocate("pipeline.s0_minmax", "state", 32, kind="state")
    with pytest.raises(RuntimeError, match="collision"):
        reg.allocate("pipeline.s0_minmax", "state", 32, kind="state")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v0.9.231-spec: PipelinePlan.peak_bytes = sum(state_bytes) — "
        "does NOT include transient (GBDT fit n-sized arrays, sort "
        "buffers). Target: peak = sum(state) + max(step_transient)."
    ),
)
def test_plan_covers_transient():
    """SPEC: plan.peak_bytes >= actual pool_peak (measure_peak_pool)."""
    df = _make_df()
    p = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p.add_step("scale", StandardScaler(columns=["x1", "x2"]))
    p.add_step("gbdt",
               GBDTStep(X_cols=["x1", "x2"], y_col="y", n_estimators=5))
    plan = p.plan(df)
    result = measure_peak_pool(p.fit_transform, df, target="y")
    actual = result["pool_peak"]
    assert plan.peak_bytes >= actual, (
        f"plan.peak_bytes={plan.peak_bytes} < actual pool_peak={actual} "
        f"(plan must cover transient)")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v0.9.231-spec: composite all-narrow should fold into parent chain "
        "(same scans/writes as flat equivalent). Validated in prototype "
        "(experiment D = A). Current CADW: no CompositeStep, no fold."
    ),
)
def test_composite_narrow_folds(scan_counter, write_counter):
    """SPEC: composite[narrow,narrow] == flat[narrow,narrow]."""
    from cadw.pipeline import CompositeStep  # not yet exists

    # Flat
    df_a = _make_df()
    p_flat = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p_flat.add_step("n1", _NarrowMockStep("n1"))
    p_flat.add_step("n2", _NarrowMockStep("n2"))
    scan_counter["n"] = 0; write_counter["n"] = 0
    p_flat.fit_transform(df_a)
    flat_scans, flat_writes = scan_counter["n"], write_counter["n"]

    # Composite (all narrow)
    df_b = _make_df()
    p_comp = Pipeline(memory_guard=MemoryGuard(max_memory_mb=500))
    p_comp.add_step(
        "c1",
        CompositeStep([_NarrowMockStep("n1"), _NarrowMockStep("n2")]),
    )
    scan_counter["n"] = 0; write_counter["n"] = 0
    p_comp.fit_transform(df_b)
    comp_scans, comp_writes = scan_counter["n"], write_counter["n"]

    assert comp_scans == flat_scans, (
        f"composite narrow must fold; flat={flat_scans} comp={comp_scans}")
    assert comp_writes == flat_writes, (
        f"composite narrow must not add writes; "
        f"flat={flat_writes} comp={comp_writes}")
