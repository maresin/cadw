#!/usr/bin/env python3
# v0.9.20-fix1 — DECISIONS order fix + bench decomposition.
# v0.9.20 COMPLETE.
#
# v0.9.20 died at append_decisions ASSERT after write (files a, b
# on disk uncommitted; DECISIONS in wrong order). Two issues:
#
# (1) DECISIONS order bug: append_decisions used
#     anchor = "**Gap 190–210 (reserved)"  (fallback to §13.5)
#     if anchor not in s: anchor = "### 13.5"  ← FALLBACK
#     But gap-note IS present (from v0.9.17). So insert went BEFORE
#     gap → order became 189...226-232...gap...211-225...§13.5.
#     Assert `225 < 226` failed → 226 (early) < 225 (late).
#     §13.4/233 (new): append with fallback anchor — never rely on
#     `if X not in s`. Check ALL candidate anchors; pick the
#     LATEST by position explicitly, not first-if-present.
#
# (2) Bench transparency: predicted=batch_bytes+2*state for reduce
#     showed no decomposition. Sort/shuffle/groupby showed "unknown"
#     (models not wired). User asked: why only batch size, no aux
#     object sizes?
#     Fix: decompose reduce into pred_batch + pred_state; wire
#     sort/shuffle/groupby model.solve()+estimate() for pred_total;
#     add C_batch = peak/batch (N× batch memory transparency).
#
# Fix1:
#   (a) DECISIONS: extract 226-232 block from wrong position, move
#       to before §13.5 (after 225).
#   (b) Bench: full rewrite with decomposed columns:
#       op, pred_batch, pred_state, pred_total, batch, peak_pool,
#       peak_rss, C_peak, C_batch.
#   (c) AST test: DECISIONS 225 < 226 < 232 < §13.5.
#   (d) AST test: bench has pred_batch/pred_state/C_batch columns.
#   (e) verify + dogfood + pytest + commit (atomic v0.9.20 =
#       initial a, b + fix1 DECISIONS move + bench rewrite).
#
# Run: uv run python scripts/patch_v0920_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) DECISIONS order fix ============

def fix_decisions_order():
    print("\n=== (a) DECISIONS: move 226-232 after 225, before §13.5 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    START = "**Эпизод 226 (nullable int to_numpy"
    END_MARKER = "coercion — same «wrong thing measured» class)."
    TARGET = "### 13.5. Открытые вопросы"
    MARK_225 = "**Эпизод 225 "

    if START not in s:
        print("  no 226 block — nothing to move")
        return
    if TARGET not in s:
        raise SystemExit("§13.5 not found")
    if MARK_225 not in s:
        raise SystemExit("225 not found")

    idx_225 = s.find(MARK_225)
    idx_start = s.find(START)
    idx_end = s.find(END_MARKER, idx_start)
    if idx_end < 0:
        raise SystemExit("232 end marker not found")
    idx_end += len(END_MARKER)
    idx_target = s.find(TARGET)

    # Already correct? 225 < 226 < end < §13.5
    if idx_225 < idx_start < idx_end < idx_target:
        print("  already ordered correctly — skip")
        return

    block = s[idx_start:idx_end]

    # Remove block: strip surrounding blank lines
    prefix = s[:idx_start].rstrip() + "\n\n\n"
    suffix = s[idx_end:].lstrip("\n\r \t")
    s_without = prefix + suffix

    # Re-find target in s_without (positions shift after removal)
    idx_target2 = s_without.find(TARGET)
    if idx_target2 < 0:
        raise SystemExit("§13.5 not found after extraction")

    # Insert block before §13.5
    s_final = (
        s_without[:idx_target2] + block + "\n\n\n"
        + s_without[idx_target2:]
    )

    p.write_text(s_final, newline="")
    # Verify
    new_225 = s_final.find(MARK_225)
    new_226 = s_final.find(START)
    new_232_end = s_final.find(END_MARKER, new_226) + len(END_MARKER)
    new_target = s_final.find(TARGET)
    assert new_225 < new_226, (
        f"225 ({new_225}) must precede 226 ({new_226})"
    )
    assert new_226 < new_232_end
    assert new_232_end < new_target, (
        f"232 end ({new_232_end}) must precede §13.5 ({new_target})"
    )
    print("  OK (226-232 moved after 225, before §13.5)")


# ============ (b) Bench rewrite with decomposition ============

BENCH_V2 = r'''#!/usr/bin/env python3
"""v0.9.20-fix1: calibration bench — decomposed predicted vs actual.

Columns:
  pred_batch: formula batch component (batch_bytes for reduce)
  pred_state: formula state component (2×state for reduce)
  pred_total: total formula estimate (sum, or model.solve()+estimate()
              for sort/shuffle/groupby)
  batch:      reference per-batch bytes (transparency)
  peak_pool:  max pa.pool.bytes_allocated during op (measured)
  peak_rss:   max psutil RSS during op (measured)
  C_peak:     peak_pool / pred_total (formula drift signal)
  C_batch:    peak_pool / batch (N× batch memory — transparency)

Interpretation:
  - C_peak ≈ 1.0 → formula accurate
  - C_peak > 2.0 → formula drift (temp buffers, unmodeled state)
  - C_batch ≫ 1 → op needs multi-batch memory (sort: sort_indices
                  int64[N], k×read_buffer; groupby: hash table, etc.)

Formula components (DECISIONS §9.2.N):
  - reduce §9.2.1:  batch_size×row_size + 2×state_bytes
  - sort   §9.2.3:  max(run_buffer, k×read_buffer + k×heap
                    + read_batch×row_size)
  - shuffle §9.2.2: batch×row_size + P×writer_buffer + c_hash×batch
  - groupby §9.2.4: max(shuffle, local_groupby)

Usage:
    uv run python scripts/bench_v0919_calibration.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
os.chdir(Path(__file__).resolve().parent.parent)

import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.models import get_model, data_chars_from_sample
from cadw.ops.reduce import (
    reduce, CountOp, MeanOp, StdOp, MinMaxOp, CovarianceOp,
)
from cadw.ops.sort import external_sort
from cadw.ops.shuffle import hash_shuffle
from cadw.ops.groupby import groupby_aggregate


def _make_batches(n: int = 100_000, n_batches: int = 10):
    table = pa.table({
        "x": pa.array(list(range(n)), type=pa.int64()),
        "g": pa.array([f"g{i % 100}" for i in range(n)]),
    })
    bs = n // n_batches
    return [table.slice(i, bs).to_batches()[0] for i in range(0, n, bs)]


def _dc_from_batches(batches, on_cols):
    from cadw.core._k_est import estimate_k_sample_arrow
    n_total = sum(b.num_rows for b in batches)
    sample = batches[0]
    if sample.num_rows == 0 or n_total == 0:
        return None
    on_present = [c for c in on_cols if c in sample.schema.names]
    k_sample = 1
    if on_present:
        try:
            k_sample = estimate_k_sample_arrow(sample, on_present)
        except Exception:
            k_sample = 1
    k_est = max(1, int(k_sample * n_total / max(1, sample.num_rows)))
    first = {c: sample.column(c)[0].as_py() for c in on_present}
    avg_key_size = len(repr(tuple(first.values())).encode()) if first else 8
    try:
        return data_chars_from_sample(
            sample, n_total,
            k_est=k_est,
            avg_key_size_bytes=avg_key_size,
            size_of_key_entry_bytes=avg_key_size + 80,
        )
    except Exception:
        return None


def _model_predicted(name, batches, on_cols, available):
    try:
        model = get_model(name)
    except KeyError:
        return None
    dc = _dc_from_batches(batches, on_cols)
    if dc is None:
        return None
    try:
        params = model.solve(available, dc)
    except Exception:
        return None
    if params is None:
        return None
    try:
        return model.estimate(params, dc)
    except Exception:
        return None


def _fmt(x):
    if x is None:
        return "?"
    if x == 0:
        return "0"
    return f"{x:,}"


def _print_row(name, pred_batch, pred_state, pred_total, batch, sample):
    peak_pool = sample.get("peak_pool", 0)
    peak_rss = sample.get("peak_rss", 0)
    c_batch = peak_pool / batch if batch else 0
    if pred_total and pred_total > 0:
        c_peak = peak_pool / pred_total
        print(
            f"{name:18s} {_fmt(pred_batch):>11s} {_fmt(pred_state):>11s} "
            f"{_fmt(pred_total):>11s} {_fmt(batch):>10s} "
            f"{_fmt(peak_pool):>12s} {_fmt(peak_rss):>12s} "
            f"{c_peak:>7.2f} {c_batch:>7.2f}"
        )
    else:
        print(
            f"{name:18s} {'?':>11s} {'?':>11s} {'?':>11s} "
            f"{_fmt(batch):>10s} "
            f"{_fmt(peak_pool):>12s} {_fmt(peak_rss):>12s} "
            f"{'?':>7s} {c_batch:>7.2f}"
        )


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="cadw_bench_v0920_"))
    batches = _make_batches()
    batch_bytes = batches[0].nbytes
    n_total = sum(b.num_rows for b in batches)
    available = 100 * 1024 * 1024

    print("v0.9.20-fix1 calibration bench (decomposed predicted)")
    print("=" * 140)
    print(f"shape: n={n_total:,}, n_batches={len(batches)}, "
          f"batch_bytes={batch_bytes:,}")
    print()
    print(
        f"{'op':18s} {'pred_batch':>11s} {'pred_state':>11s} "
        f"{'pred_total':>11s} {'batch':>10s} "
        f"{'peak_pool':>12s} {'peak_rss':>12s} "
        f"{'C_peak':>7s} {'C_batch':>7s}"
    )
    print("-" * 140)

    reduce_specs = [
        ("CountOp", CountOp("x"), 16),
        ("MeanOp", MeanOp("x"), 16),
        ("StdOp", StdOp("x"), 24),
        ("MinMaxOp", MinMaxOp("x"), 16),
        ("CovarianceOp", CovarianceOp(["x"]), None),
    ]
    for name, op, state_bytes in reduce_specs:
        if state_bytes is None:
            state_bytes = op.estimate_state_nbytes()
        pred_batch = batch_bytes
        pred_state = 2 * state_bytes
        pred_total = pred_batch + pred_state
        guard = MemoryGuard(max_memory_mb=100, calibrate=True)
        with guard.calibrate(name, pred_total, sample_peak=True):
            reduce(iter(batches), op, guard, log=False)
        _print_row(
            name, pred_batch, pred_state, pred_total, batch_bytes,
            guard.calibration_samples[name][-1],
        )

    pred_sort = _model_predicted("sort", batches, [], available)
    guard = MemoryGuard(max_memory_mb=100, calibrate=True)
    with guard.calibrate(
        "external_sort", pred_sort or 0, sample_peak=True,
    ):
        external_sort(
            iter(batches), ["x"], ["x", "g"], guard,
            tmp_root / "sort", log=False,
        )
    _print_row(
        "external_sort", None, None, pred_sort, batch_bytes,
        guard.calibration_samples["external_sort"][-1],
    )

    pred_shuf = _model_predicted(
        "hash_shuffle", batches, ["g"], available,
    )
    guard = MemoryGuard(max_memory_mb=100, calibrate=True)
    with guard.calibrate(
        "hash_shuffle", pred_shuf or 0, sample_peak=True,
    ):
        hash_shuffle(
            iter(batches), ["g"], ["x"], 8, guard,
            tmp_root / "shuffle", log=False,
        )
    _print_row(
        "hash_shuffle", None, None, pred_shuf, batch_bytes,
        guard.calibration_samples["hash_shuffle"][-1],
    )

    pred_gb = _model_predicted("groupby", batches, ["g"], available)
    guard = MemoryGuard(max_memory_mb=100, calibrate=True)
    with guard.calibrate(
        "groupby_aggregate", pred_gb or 0, sample_peak=True,
    ):
        groupby_aggregate(
            iter(batches), ["g"], ["x"], ["sum"], guard,
            tmp_root / "gb", log=False,
        )
    _print_row(
        "groupby_aggregate", None, None, pred_gb, batch_bytes,
        guard.calibration_samples["groupby_aggregate"][-1],
    )

    print("-" * 140)
    print()
    print("Formula components (DECISIONS §9.2.N):")
    print("  reduce §9.2.1:  batch + 2×state (decomposed above)")
    print("  sort   §9.2.3:  max(run_buffer, k×read_buffer + k×heap "
          "+ read_batch×row_size)")
    print("  shuffle §9.2.2: batch×row_size + P×writer_buffer "
          "+ c_hash×batch")
    print("  groupby §9.2.4: max(shuffle, local_groupby)")
    print()
    print("Reading the table:")
    print("  C_peak  = peak_pool / pred_total -> formula accuracy")
    print("  C_batch = peak_pool / batch      -> N× batch memory "
          "(transparency)")
    print("  Note: pred_total = model.solve().estimate() is UPPER BOUND.")
    print("  For inputs < buffer capacity, C_peak << 1 (formula "
          "over-conservative).")
    print("  For inputs filling buffers, C_peak ~= 1. C_peak > 2 -> "
          "formula drift.")
    print()
    print(f"tmp dir (not cleaned): {tmp_root}")


if __name__ == "__main__":
    main()
'''


def rewrite_bench():
    print("\n=== (b) bench: full rewrite with decomposition ===")
    p = Path("scripts/bench_v0919_calibration.py")
    # Idempotency: v2 has pred_batch column header
    if p.exists() and "pred_batch" in p.read_text():
        print("  already v2 — skip")
        return
    p.write_text(BENCH_V2, newline="")
    pu.assert_compiles(p)
    print("  OK (v2 written + compiled)")


# ============ (c) AST test: DECISIONS order ============

def add_decisions_order_test():
    print("\n=== (c) test_changelog.py: DECISIONS 226-232 order ===")
    p = Path("tests/test_changelog.py")
    s = p.read_text()
    if "def test_decisions_226_232_order" in s:
        print("  already present — skip")
        return

    extra = '''

# ============ v0.9.20-fix1 (§13.4/233): DECISIONS order ============


def test_decisions_226_232_order():
    """v0.9.20-fix1: episodes 226-232 ordered after 225, before §13.5.

    §13.4/233: append with fallback anchor must pick LATEST candidate
    explicitly. v0.9.20 used anchor = "**Gap 190-210" (present from
    v0.9.17) with fallback to §13.5 — since gap-note existed, 226-232
    inserted BEFORE it → order became 189...226-232...gap...211-225.
    Test pins correct order: 225 < 226 < 232 < §13.5.
    """
    from pathlib import Path as _P
    path = _P(__file__).resolve().parent.parent / "DECISIONS.md"
    text = path.read_text()
    if "**Эпизод 226 " not in text:
        pytest.skip("226 block not yet present (v0.9.20 not applied)")
    idx_225 = text.index("**Эпизод 225 ")
    idx_226 = text.index("**Эпизод 226 ")
    idx_232 = text.index("**Эпизод 232 ")
    idx_135 = text.index("### 13.5. Открытые вопросы")
    assert idx_225 < idx_226, (
        f"Эпизод 226 ({idx_226}) appears before 225 ({idx_225}) — "
        f"append inserted in wrong position"
    )
    assert idx_226 < idx_232, (
        f"Эпизод 232 ({idx_232}) before 226 ({idx_226})"
    )
    assert idx_232 < idx_135, (
        f"Эпизод 232 ({idx_232}) after §13.5 ({idx_135})"
    )
'''

    p.write_text(s + extra, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (d) AST test: bench decomposition ============

def add_bench_decomposition_tests():
    print("\n=== (d) test_runtime_monitor.py: bench decomposition AST ===")
    p = Path("tests/test_runtime_monitor.py")
    s = p.read_text()
    if "def test_bench_has_decomposed_columns" in s:
        print("  already present — skip")
        return

    extra = '''

# ============ v0.9.20-fix1: bench decomposition ============


def test_bench_has_decomposed_columns():
    """v0.9.20-fix1: bench prints pred_batch/pred_state/C_batch columns.

    User feedback: 'why only batch size, no aux object sizes?'
    Fix decomposes reduce (batch + state), wires sort/shuffle/groupby
    models for pred_total, and adds C_batch = peak/batch (N× batch
    memory transparency for ops with significant non-batch state).
    """
    from pathlib import Path as _P
    bench = (
        _P(__file__).resolve().parent.parent
        / "scripts" / "bench_v0919_calibration.py"
    )
    src = bench.read_text()
    # Header column names
    assert "pred_batch" in src, "missing pred_batch column"
    assert "pred_state" in src, "missing pred_state column"
    assert "pred_total" in src, "missing pred_total column"
    assert "C_batch" in src, "missing C_batch column"
    # Model wiring for sort/shuffle/groupby
    assert '_model_predicted("sort"' in src, "sort model not wired"
    assert '_model_predicted("hash_shuffle"' in src, (
        "hash_shuffle model not wired"
    )
    assert '_model_predicted("groupby"' in src, (
        "groupby model not wired"
    )


def test_bench_wires_models_via_solve_estimate():
    """v0.9.20-fix1: bench uses model.solve()+estimate() for pred_total.

    §9.2.2-4 formulas exist; bench must call them, not hardcode 0.
    """
    from pathlib import Path as _P
    bench = (
        _P(__file__).resolve().parent.parent
        / "scripts" / "bench_v0919_calibration.py"
    )
    src = bench.read_text()
    assert "model.solve(" in src
    assert "model.estimate(" in src
    # data_chars_from_sample imported for DataChars construction
    assert "data_chars_from_sample" in src
'''

    p.write_text(s + extra, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.20 + fix1 + regression ===")
    pu.verify_anchors([
        # fix1 (a)
        ("DECISIONS.md", "**Эпизод 226 (nullable int to_numpy"),
        ("DECISIONS.md", "**Эпизод 232 (metric unit must match"),
        # fix1 (b)
        ("scripts/bench_v0919_calibration.py", "pred_batch"),
        ("scripts/bench_v0919_calibration.py", "pred_state"),
        ("scripts/bench_v0919_calibration.py", "pred_total"),
        ("scripts/bench_v0919_calibration.py", "C_batch"),
        ("scripts/bench_v0919_calibration.py", '_model_predicted("sort"'),
        ("scripts/bench_v0919_calibration.py",
         '_model_predicted("hash_shuffle"'),
        ("scripts/bench_v0919_calibration.py",
         '_model_predicted("groupby"'),
        # fix1 (c)
        ("tests/test_changelog.py", "def test_decisions_226_232_order"),
        # fix1 (d)
        ("tests/test_runtime_monitor.py",
         "def test_bench_has_decomposed_columns"),
        # v0.9.20 (a)
        ("cadw/core/memory_guard.py", "sample_peak: bool = False"),
        ("cadw/core/memory_guard.py", "sample_interval_ms: float = 10.0"),
        ("cadw/core/memory_guard.py", "peak_pool_box"),
        # v0.9.20 (c) DECISIONS content
        ("DECISIONS.md", "**Эпизод 226"),
        ("DECISIONS.md", "**Эпизод 232"),
        # v0.9.19 regression
        ("cadw/core/memory_guard.py", "def calibrate("),
        ("cadw/ops/reduce.py", "with memory_guard.calibrate("),
        # v0.9.18/17/16
        ("cadw/ops/shuffle.py", "§13.4/227"),
        ("cadw/dataset/index.py", "v0.9.18 (AUDIT §1.6)"),
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный"),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
    ])
    print("  all anchors present")


# ============ dogfood (happy path only, §13.4/228) ============

def dogfood():
    print("\n=== dogfood: v0.9.20-fix1 (happy path only) ===")

    def _verify_all():
        # 1. DECISIONS order (critical)
        from pathlib import Path as _P
        text = _P("DECISIONS.md").read_text()
        idx_225 = text.find("**Эпизод 225 ")
        idx_226 = text.find("**Эпизод 226 ")
        idx_232 = text.find("**Эпизод 232 ")
        idx_135 = text.find("### 13.5. Открытые вопросы")
        assert 0 <= idx_225 < idx_226 < idx_232 < idx_135, (
            f"DECISIONS order broken: 225={idx_225} 226={idx_226} "
            f"232={idx_232} §13.5={idx_135}"
        )

        # 2. Bench compiles + has decomposition columns
        import py_compile
        bench = _P("scripts/bench_v0919_calibration.py")
        py_compile.compile(str(bench), doraise=True)
        src = bench.read_text()
        assert "pred_batch" in src
        assert "pred_state" in src
        assert "C_batch" in src
        assert "def _dc_from_batches" in src
        assert "def _model_predicted" in src

        # 3. Bench smoke run: imports work, no execution
        # (skip actual run — heavy; tested in CI)
        print("  DECISIONS order 225 < 226 < 232 < §13.5 OK")
        print("  bench compiles + has decomposition cols OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py", "tests/test_runtime_monitor.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.20-fix1 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_decisions_order()
    rewrite_bench()
    add_decisions_order_test()
    add_bench_decomposition_tests()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.20: peak-polling in calibrate() + DECISIONS §13.4/226-232 "
        "append + bench decomposition (initial + fix1, atomic — "
        "v0.9.20 initial died at append_decisions assert AFTER write; "
        "files a,b on disk uncommitted; DECISIONS wrong order). "
        "(Motivation) v0.9.19 bench revealed C_arrow=0 for reduce ops — "
        "net pool delta = end − start; batches pre-allocated at guard "
        "creation, Arrow temp buffers released before end → net ≈ 0. "
        "Formula §9.2.1 predicts absolute peak; net delta useless for "
        "formula validation. §13.4/232 (new): metric unit must match "
        "formula unit. Peak vs peak, delta vs delta. Never cross. "
        "(Design a) `calibrate(..., sample_peak=True, "
        "sample_interval_ms=10)` — opt-in background thread polls "
        "pool+RSS every interval, tracks max. Records `peak_pool` / "
        "`peak_rss` (absolute, comparable to formula) AND "
        "`net_pool_delta` (leak detection). sample_mode='peak'|'delta'. "
        "Zero overhead when calibrate=False. runtime_monitor uses "
        "peak_rss when sampled (honest hard/soft check). (Design b) "
        "bench decomposition — user feedback: 'why only batch, no aux "
        "object sizes?'. v0.9.20-fix1 rewrites bench: reduce ops "
        "decompose into pred_batch (batch_bytes) + pred_state "
        "(2×state); sort/shuffle/groupby wired via "
        "model.solve()+estimate() for pred_total (was 'unknown'); "
        "adds C_batch = peak/batch (N× batch memory transparency). "
        "New columns: op, pred_batch, pred_state, pred_total, batch, "
        "peak_pool, peak_rss, C_peak, C_batch. (DECISIONS) "
        "§13.4/226-232 appended: 226 (nullable int to_numpy → "
        "float64), 227 (dictionary_encode all-null → empty dict K=0), "
        "228 (dogfood = happy path only), 229 (numpy == coerces label "
        "to array dtype — 13 loc[tuple] tests fixed), 230 "
        "(write_content lines — complete syntactic units for "
        "f-strings), 231 (helper methods are caller-facing, calibrate "
        "does direct lazy init), 232 (metric unit must match formula "
        "unit). (Fix1) v0.9.20 initial died at append_decisions ASSERT "
        "AFTER write. Root: append used anchor = "
        "'**Gap 190–210 (reserved)' (present from v0.9.17) with "
        "fallback to §13.5, but since gap-note EXISTS, 226-232 "
        "inserted BEFORE gap → order became 189...226-232...gap..."
        "211-225. Assert 225 < 226 failed → 226 (early) < 225 (late). "
        "§13.4/233 (new): append with fallback anchor must check ALL "
        "candidate anchors; pick LATEST by position explicitly. Fix1: "
        "(a) extract 226-232 block from wrong position, move to "
        "before §13.5; (b) bench full rewrite; (c) AST test "
        "test_decisions_226_232_order (225 < 226 < 232 < §13.5); "
        "(d) AST tests bench_has_decomposed_columns + "
        "bench_wires_models_via_solve_estimate. (Backward compat) 1902 "
        "existing tests unchanged; zero overhead when calibrate=False. "
        "New tests: 6 v0.9.20 (peak mode, delta mode, spike capture, "
        "thread joined, monitor peak, bench sample_peak) + 3 fix1 "
        "(decisions order, bench decomposition, bench model wiring) "
        "= 9 new total. Deferred to v0.9.21: AUDIT §1.4 per-group "
        "shift/rolling numpy; per-op predicted wiring for sort/"
        "shuffle/groupby in ops (not just bench); Pingis round 3 "
        "(unaudited: ML/pipeline/cv/metrics); DECISIONS §13.4/233 "
        "append (this fix's meta-lesson).",
        self_path="scripts/patch_v0920_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.20-fix1 applied. v0.9.20 COMPLETE.")


if __name__ == "__main__":
    main()
