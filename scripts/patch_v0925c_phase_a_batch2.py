#!/usr/bin/env python3
# v0.9.25c — Phase A batch 2 (4 items from consolidated roadmap).
#
# A.4 B1: _RunStream.refill linear concat (was O(n²) per call —
#   12.9 GB copied @ rbs=200k measured in part1_report).
# A.5 B2: encode_categorical guard — in-loop check_state(len(uniques)*180)
#   before sorted(uniques) copy. Prior code accumulated Python-set of
#   all uniques with no bound; @ 10M unique strings ~600 MB invisible
#   to MemoryGuard.
# A.7: compute_fan_in uses read_batch_size, NOT _WRITE_RUN_CHUNK.
#   Root cause of AUDIT §6.6 "24.4s @ N=500k/8cols/32MB": v0.9.23c
#   _merge_keys uses rbs=1000 internally (Arrow batched merge), but
#   compute_fan_in was still called with _WRITE_RUN_CHUNK=50000 →
#   K=2 → 3-4x more merge passes → 60x wall. Correct fix: pass rbs.
#   Audit's "floor=8" was a proxy for this.
# B.4: collect_project.py --module / --phase / --doc / --list.
#   Reads docs/roadmap.json (v0.9.25b) for phase-based extraction.
#   --module is the main workflow unlock: 27 files → 5 files/session.
#
# Tests: 10 new + 1 flipped (test_compute_fan_in_uses_write_run_chunk
#   in test_sort_merge_p_s1.py — asserts old buggy behavior; flip to
#   _read_batch_size).
#
# Run: uv run python scripts/patch_v0925c_phase_a_batch2.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape, run_verify_in_subprocess,
)


# ============ 0. Verify v0.9.25b committed ============

def verify_state():
    print("\n=== verify v0.9.25b committed + docs present ===")
    r = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True,
    )
    if "v0.9.25b" not in r.stdout:
        raise SystemExit(
            f"v0.9.25b not in HEAD; got {r.stdout!r}. "
            f"Re-run v0.9.25b+fix1 first."
        )
    for f in ("docs/roadmap.json", "docs/ROADMAP.md",
              "docs/index.md", "tests/test_roadmap.py"):
        if not Path(f).exists():
            raise SystemExit(f"{f} missing — v0.9.25b did not apply")
    # Anchors in cadw/
    anchors = [
        ("cadw/ops/sort.py", "def refill(self, memory_guard)"),
        ("cadw/ops/sort.py", "def compute_fan_in("),
        ("cadw/ops/sort.py", "_WRITE_RUN_CHUNK, row_bytes"),
        ("cadw/dataframe.py", "def encode_categorical("),
    ]
    for path, anchor in anchors:
        if anchor not in Path(path).read_text():
            raise SystemExit(f"{path}: missing anchor {anchor!r}")
    print(f"  HEAD OK; 4 anchors present; docs present")


# ============ 1. A.4 — _RunStream.refill linear ============

OLD_REFILL = '''    def refill(self, memory_guard) -> None:
        """Read up to rbs rows into buffer. Marks exhausted on EOF."""
        while len(self.buffer) < self.rbs:
            r = self._next_batch(memory_guard)
            if r is None:
                self.exhausted = True
                return
            b, pos = r
            need = self.rbs - len(self.buffer)
            avail = b.num_rows - pos
            take = min(need, avail)
            if take <= 0:
                continue
            chunk = pa.Table.from_batches([b.slice(pos, take)])
            if len(self.buffer) == 0:
                self.buffer = chunk
            else:
                self.buffer = pa.concat_tables([self.buffer, chunk])
            self.cur_pos += take
            if self.cur_pos >= b.num_rows:
                self.cur_batch = None
                self.cur_pos = 0
'''

NEW_REFILL = '''    def refill(self, memory_guard) -> None:
        """Read up to rbs rows into buffer. Marks exhausted on EOF.

        v0.9.25c (part1_report B1): linear concat. Prior code did
        `self.buffer = pa.concat_tables([self.buffer, chunk])` inside
        the while loop — quadratic, each iteration copies the entire
        accumulated buffer. Measured: 12.9 GB copied @ rbs=200k,
        0.078s vs 0.0003s for list-collect + single concat. Now:
        accumulate parts, one concat_tables(parts) at end.
        """
        parts: list = [self.buffer] if len(self.buffer) else []
        have = len(self.buffer)
        while have < self.rbs:
            r = self._next_batch(memory_guard)
            if r is None:
                self.exhausted = True
                break
            b, pos = r
            need = self.rbs - have
            avail = b.num_rows - pos
            take = min(need, avail)
            if take <= 0:
                continue
            parts.append(pa.Table.from_batches([b.slice(pos, take)]))
            have += take
            self.cur_pos += take
            if self.cur_pos >= b.num_rows:
                self.cur_batch = None
                self.cur_pos = 0
        if len(parts) == 1:
            self.buffer = parts[0]
        elif len(parts) > 1:
            self.buffer = pa.concat_tables(parts)
        # (parts empty + buffer empty = no-op; unchanged)
'''

def fix_refill():
    print("\n=== A.4: _RunStream.refill linear concat ===")
    applied = apply_or_fail(
        "cadw/ops/sort.py", OLD_REFILL, NEW_REFILL,
        label="refill_linear",
    )
    print("  applied" if applied else "  already applied — skip")


# ============ 2. A.7 — compute_fan_in uses rbs ============

OLD_FANIN = '''def compute_fan_in(available_bytes: int, read_batch_size: int, row_bytes: int) -> int:
    """k (fan-in) для k-way merge: сколько run'ов держать открытыми одновременно.

    Ограничение: k * read_batch_size * row_bytes <= available_bytes.
    Возвращает >= 2 (иначе merge невозможен).
    """
    denom = max(1, read_batch_size * max(1, row_bytes))
    return max(2, available_bytes // denom)
'''

NEW_FANIN = '''def compute_fan_in(
    available_bytes: int,
    read_batch_size: int,
    row_bytes: int,
    *,
    max_k: int = 256,
) -> int:
    """k (fan-in) for Arrow-native k-way merge.

    v0.9.25c (audit §6.6): must match _merge_keys.check_state(3 * K * rbs
    * row_bytes) <= available * state_safety (=0.5). The factor 3 =
    buffers + merged + sorted (see _merge_keys docstring). Prior code
    used `available // (rbs * row_bytes)` — 3x under-counted the actual
    peak; and caller passed _WRITE_RUN_CHUNK (50000) while _merge_keys
    uses rbs=1000, giving K=2 at 32 MB / 8 cols / 640 B/row — root
    cause of 24.4s @ N=500k/8 cols / 32 MB (audit §6.6).

    Now: state_limit // (3 * rbs * row_bytes), capped at max_k (file
    handle / FD limit — 256 is conservative). Min 2 (merge impossible
    with k < 2; caller (_merge_keys) will fire check_state if 2 runs
    exceed state_limit — documented for wide schemas at tight budget).
    """
    state_limit = int(available_bytes * 0.5)
    per_run = 3 * max(1, read_batch_size) * max(1, row_bytes)
    k = state_limit // per_run
    return max(2, min(max_k, k))
'''

def fix_fan_in_fn():
    print("\n=== A.7a: compute_fan_in formula (3*K*rbs*row) + max_k ===")
    applied = apply_or_fail(
        "cadw/ops/sort.py", OLD_FANIN, NEW_FANIN,
        label="fan_in_formula",
    )
    print("  applied" if applied else "  already applied — skip")


OLD_FANIN_CALL = '''    k = merge_fan_in or compute_fan_in(
        memory_guard.available_bytes(), _WRITE_RUN_CHUNK, row_bytes
    )
'''

NEW_FANIN_CALL = '''    # v0.9.25c: pass read_batch_size (the rbs that _merge_keys will
    # use), NOT _WRITE_RUN_CHUNK (on-disk chunk, unrelated to merge
    # memory). Prior mis-pass gave K=2 @ 32MB/8cols → 3-4x more merge
    # passes → 60x wall (audit §6.6). Now: K=8 @ 32MB/8cols / 640 B/row.
    k = merge_fan_in or compute_fan_in(
        memory_guard.available_bytes(), read_batch_size, row_bytes
    )
'''

def fix_fan_in_call():
    print("\n=== A.7b: external_sort passes read_batch_size to fan_in ===")
    applied = apply_or_fail(
        "cadw/ops/sort.py", OLD_FANIN_CALL, NEW_FANIN_CALL,
        label="fan_in_call",
    )
    print("  applied" if applied else "  already applied — skip")


# ============ 3. A.5 — encode_categorical guard ============

OLD_ENCCAT = '''        # Pass 1: collect unique values per column.
        reverse_maps: dict[str, list] = {}
        for col in cols:
            uniques: set = set()
            for b in self._iter_filtered_batches():
                arr = b.column(col)
                vc = pc.value_counts(arr)  # struct<values, counts>
                if isinstance(vc, pa.ChunkedArray):
                    vc = vc.combine_chunks()
                values = vc.field("values")
                for v in values.to_pylist():  # K per batch, not N
                    if v is not None:
                        uniques.add(v)
            try:
                sorted_vals = sorted(uniques)
            except TypeError:
                sorted_vals = sorted(uniques, key=lambda x: str(x))
            reverse_maps[col] = sorted_vals
'''

NEW_ENCCAT = '''        # Pass 1: collect unique values per column.
        reverse_maps: dict[str, list] = {}
        for col in cols:
            uniques: set = set()
            for b in self._iter_filtered_batches():
                arr = b.column(col)
                vc = pc.value_counts(arr)  # struct<values, counts>
                if isinstance(vc, pa.ChunkedArray):
                    vc = vc.combine_chunks()
                values = vc.field("values")
                for v in values.to_pylist():  # K per batch, not N
                    if v is not None:
                        uniques.add(v)
                # v0.9.25c (part1_report B2): in-loop guard. Prior code
                # accumulated Python-set of all uniques with no bound;
                # @ 10M unique strings ~600 MB invisible to MemoryGuard
                # (batch.nbytes ~130 KB/batch). 180 B/entry = 50 (str)
                # + ~40 (set slot) + ~90 (list copy margin for sorted()).
                # Fail-fast at state_limit (=available*0.5).
                self._memory_guard.check_state(
                    len(uniques) * 180,
                    name=f"encode_categorical({col})",
                )
            try:
                sorted_vals = sorted(uniques)
            except TypeError:
                sorted_vals = sorted(uniques, key=lambda x: str(x))
            reverse_maps[col] = sorted_vals
'''

def fix_encode_cat():
    print("\n=== A.5: encode_categorical in-loop guard ===")
    applied = apply_or_fail(
        "cadw/dataframe.py", OLD_ENCCAT, NEW_ENCCAT,
        label="encode_cat_guard",
    )
    print("  applied" if applied else "  already applied — skip")


# ============ 4. Tests for A.4/A.5/A.7 ============

NEW_TESTS = '''"""v0.9.25c: A.4 refill + A.5 encode_cat + A.7 fan_in rbs."""
import ast
import inspect
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cadw
from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


# ============ A.4: _RunStream.refill linear ============


def test_runstream_refill_no_concat_in_loop():
    """AST: no pa.concat_tables inside while loop in refill (B1)."""
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod._RunStream.refill)
    tree = ast.parse(inspect.cleandoc(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "concat_tables"):
                    raise AssertionError(
                        f"concat_tables inside while loop at "
                        f"line {sub.lineno} — quadratic concat (B1)"
                    )


def test_runstream_refill_linear_smoke():
    """1000 tiny chunks refill < 0.5s (generous CI bound; old O(n²)
    took ~0.04-0.08s, new <0.005s; but slower CI allows 0.5s)."""
    import time
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.ops.sort import _RunStream

    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "test.arrow"
        schema = pa.schema([
            ("x", pa.int64()), ("row_id", pa.uint64()),
        ])
        w = ArrowIPCBatchWriter(str(path), schema)
        try:
            for i in range(1000):
                w.write_batch(pa.RecordBatch.from_arrays(
                    [pa.array([i], type=pa.int64()),
                     pa.array([i], type=pa.uint64())],
                    schema=schema,
                ))
        finally:
            w.close()
        stream = _RunStream(path, rbs=1000, name="test")
        t0 = time.perf_counter()
        stream.refill(None)
        elapsed = time.perf_counter() - t0
        assert len(stream.buffer) == 1000
        assert elapsed < 0.5, (
            f"refill took {elapsed:.3f}s — quadratic concat regression (B1)"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============ A.5: encode_categorical guard ============


def test_encode_categorical_high_card_raises():
    """B2: 100k unique strings @ 4 MB budget → MemoryLimitExceeded
    before OOM (was silent 18 MB Python-set on top of ~1 MB Arrow)."""
    n = 100_000
    table = pa.table({
        "cat": pa.array(
            [f"s{i:08d}" for i in range(n)], type=pa.string(),
        ),
        "v": list(range(n)),
    })
    df = CadwDataFrame.from_table(
        table, memory_guard=MemoryGuard(max_memory_mb=4),
    )
    # 100k * 180 B = 18 MB >> 2 MB state_limit (4 * 0.5).
    with pytest.raises(MemoryLimitExceeded, match="encode_categorical"):
        df.encode_categorical("cat")


def test_encode_categorical_low_card_ok():
    """Small K passes guard, behavior unchanged."""
    table = pa.table({
        "cat": pa.array(["a", "b", "a", "c", "b"], type=pa.string()),
        "v": list(range(5)),
    })
    df = CadwDataFrame.from_table(
        table, memory_guard=MemoryGuard(max_memory_mb=4),
    )
    encoded, maps = df.encode_categorical("cat")
    assert maps["cat"] == ["a", "b", "c"]


# ============ A.7: compute_fan_in uses read_batch_size ============


def test_compute_fan_in_uses_read_batch_size():
    """AST: external_sort passes read_batch_size (not _WRITE_RUN_CHUNK)."""
    import textwrap
    from cadw.ops import sort as sort_mod
    src = textwrap.dedent(inspect.getsource(sort_mod.external_sort))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name)
                and node.func.id == "compute_fan_in"):
            continue
        assert len(node.args) >= 2
        arg1 = ast.unparse(node.args[1])
        assert "read_batch_size" in arg1, (
            f"compute_fan_in arg={arg1!r} — must use read_batch_size "
            f"(v0.9.25c; Arrow-native merge internal rbs)"
        )
        assert "_WRITE_RUN_CHUNK" not in arg1, (
            f"compute_fan_in arg={arg1!r} — must NOT use _WRITE_RUN_CHUNK "
            f"(was source of 24s @ 500k/8cols/32MB)"
        )
        return
    raise AssertionError("compute_fan_in not called in external_sort")


def test_compute_fan_in_formula():
    """A.7: state_limit // (3 * rbs * row_bytes); min=2, max=256."""
    from cadw.ops.sort import compute_fan_in
    # 32 MB, rbs=1000, row_bytes=640 (8 cols narrow): 16 MB / 1.92 MB = 8
    assert compute_fan_in(32 * 1024 * 1024, 1000, 640) == 8
    # 32 MB, rbs=1000, row_bytes=80 (1 col): 16 MB / 240 KB = 69
    assert compute_fan_in(32 * 1024 * 1024, 1000, 80) == 69
    # Wide schema — 2 is minimum even when 3*K*rbs*row > state_limit
    assert compute_fan_in(32 * 1024 * 1024, 1000, 5000) == 2
    # Large budget capped at max_k=256
    assert compute_fan_in(2 * 1024 * 1024 * 1024, 1000, 80) == 256


def test_external_sort_at_32mb_8cols_reasonable_time():
    """End-to-end smoke: N=50k / 8 cols / 32 MB, wall < 5s.

    Audit §6.6 measured 24.4s @ N=500k/8cols/32MB pre-fix. Post-fix
    K=8 (not 2), fewer merge passes. Smoke test uses N=50k to stay
    fast; full 500k bench in _measure/sort_scaling.py.
    """
    import time
    rng = np.random.default_rng(0)
    n = 50_000
    cols = {"k": pa.array(rng.integers(0, 1000, n), type=pa.int64())}
    for j in range(7):
        cols[f"v{j}"] = pa.array(rng.normal(size=n), type=pa.float64())
    table = pa.table(cols)
    src = Path(tempfile.mkdtemp()) / "src.parquet"
    try:
        pq.write_table(table, src, row_group_size=5000)
        df = CadwDataFrame.from_parquet(
            src, src.parent / "ds", chunk_size=5000,
            memory_guard=MemoryGuard(max_memory_mb=32),
            spill_dir=src.parent / "spill",
        )
        t0 = time.perf_counter()
        out = df.sort_values("k")
        out_path = src.parent / "out.parquet"
        out.write_parquet(out_path)
        elapsed = time.perf_counter() - t0
        got_k = pq.read_table(out_path).column("k").to_pylist()
        assert got_k == sorted(table.column("k").to_pylist())
        assert elapsed < 5.0, (
            f"sort @ 50k/8cols/32MB took {elapsed:.2f}s — fan_in "
            f"regression (was ~24s @ 500k pre-fix; post-fix should be "
            f"scaled to ~2.5s @ 500k → ~0.25s @ 50k)"
        )
    finally:
        shutil.rmtree(src.parent, ignore_errors=True)
'''

def add_tests():
    print("\n=== tests/test_v0925c_phase_a.py ===")
    p = Path("tests/test_v0925c_phase_a.py")
    if p.exists() and "test_runstream_refill_no_concat_in_loop" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(NEW_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 5. Flip stale test in test_sort_merge_p_s1.py ============

OLD_FANIN_TEST = '''def test_compute_fan_in_uses_write_run_chunk():
    # AST: compute_fan_in second positional arg must be
    # _WRITE_RUN_CHUNK (real on-disk chunk), not read_batch_size.
    import ast
    import inspect
    import textwrap
    from cadw.ops import sort as sort_mod
    src = textwrap.dedent(inspect.getsource(sort_mod.external_sort))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name)
                and node.func.id == "compute_fan_in"):
            continue
        assert len(node.args) >= 2, (
            "compute_fan_in needs 2+ positional args"
        )
        arg1 = ast.unparse(node.args[1])
        assert "_WRITE_RUN_CHUNK" in arg1, (
            "compute_fan_in read_batch_size=" + arg1 +
            " must be _WRITE_RUN_CHUNK (P-S1 fix)"
        )
        return
    raise AssertionError("compute_fan_in call not found")
'''

NEW_FANIN_TEST = '''def test_compute_fan_in_uses_read_batch_size():
    # v0.9.25c (A.7): flipped from test_compute_fan_in_uses_write_run_chunk.
    # v0.9.23c changed _merge_keys to use rbs=1000 (Arrow-native
    # batched merge). compute_fan_in must use the SAME rbs, not
    # _WRITE_RUN_CHUNK (which is on-disk chunk size, unrelated to
    # merge memory). Prior mismatch: K=2 @ 32MB/8cols → 60x wall
    # (audit §6.6). Audit "floor=8" was a proxy for this.
    import ast
    import inspect
    import textwrap
    from cadw.ops import sort as sort_mod
    src = textwrap.dedent(inspect.getsource(sort_mod.external_sort))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name)
                and node.func.id == "compute_fan_in"):
            continue
        assert len(node.args) >= 2, (
            "compute_fan_in needs 2+ positional args"
        )
        arg1 = ast.unparse(node.args[1])
        assert "read_batch_size" in arg1, (
            f"compute_fan_in arg={arg1!r} — must use read_batch_size "
            f"(v0.9.25c; _merge_keys uses rbs internally, not on-disk "
            f"_WRITE_RUN_CHUNK)"
        )
        assert "_WRITE_RUN_CHUNK" not in arg1, (
            f"compute_fan_in arg={arg1!r} — must NOT use _WRITE_RUN_CHUNK "
            f"(root cause of 24s @ 500k/8cols/32MB)"
        )
        return
    raise AssertionError("compute_fan_in call not found")
'''

def flip_stale_test():
    print("\n=== flip stale test_sort_merge_p_s1.py::test_compute_fan_in ===")
    p = Path("tests/test_sort_merge_p_s1.py")
    s = p.read_text()
    if "def test_compute_fan_in_uses_read_batch_size()" in s and \
       "def test_compute_fan_in_uses_write_run_chunk()" not in s:
        print("  already flipped — skip")
        return
    if OLD_FANIN_TEST not in s:
        raise SystemExit(
            "old test anchor not found in test_sort_merge_p_s1.py — "
            "inspect file (v0.9.25b may have not touched it, but the "
            "name/anchor may differ)."
        )
    s = s.replace(OLD_FANIN_TEST, NEW_FANIN_TEST, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  flipped")


# ============ 6. B.4 — collect_project.py phase-aware ============

COLLECT_PROJECT_ADDITIONS = '''

# ============================================================================
# v0.9.25c: phase-aware / module-aware modes (B.4)
# ============================================================================
#
# Runs AFTER main() above — new entry points that reuse the same file
# collection primitives but with a manifest-driven filter.
#
#   --module cadw/ts/models.py
#       collect module + matching tests. Default excludes core, interfaces.
#       Use --with-interfaces to inline cadw/dataframe.py signatures.
#   --phase A
#       reads docs/roadmap.json, collects all phase-A items' files
#       (best-effort glob expansion; warns on unresolvable composites).
#   --doc docs/ROADMAP.md
#       just that doc.
#   --list-phases / --list-modules
#       debug helpers.
#
# Output: ts_review_<mode>.md by default; override with --out.

import json as _json
import re as _re
import sys as _sys

_DOCS_ROADMAP = Path("docs/roadmap.json")


def _parse_file_spec(spec: str) -> list[Path]:
    """Parse file paths from a roadmap.json 'file' field.

    Supported:
      - 'cadw/ts/models.py'         → [Path]
      - 'cadw/{a,b,c}/*.py'         → brace-expansion + glob
      - 'cadw/dataframe.py, ...'    → comma-split
      - 'tests/, pyproject.toml'    → directory (rglob *.py) + file
    Unknown → [] (caller logs warning).
    """
    if not spec:
        return []
    out: list[Path] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "{" in part and "}" in part:
            prefix, rest = part.split("{", 1)
            options, suffix = rest.split("}", 1)
            for opt in options.split(","):
                for sub in _parse_file_spec(
                    prefix + opt.strip() + suffix
                ):
                    out.append(sub)
        elif "*" in part:
            out.extend(Path(".").glob(part))
        elif part.endswith("/"):
            d = Path(part)
            if d.is_dir():
                out.extend(sorted(d.rglob("*.py")))
        else:
            p = Path(part)
            if p.exists() and p.is_file():
                out.append(p)
    return out


def _find_tests_for_module(mod_path: Path) -> list[Path]:
    """Heuristic: tests/test_<parent>_<stem>*.py + tests/test_<stem>*.py."""
    tests_dir = Path("tests")
    if not tests_dir.exists():
        return []
    stem = mod_path.stem
    parent = mod_path.parent.name
    out: list[Path] = []
    if parent and parent != "tests":
        out.extend(sorted(tests_dir.glob(f"test_{parent}_{stem}*.py")))
    out.extend(sorted(tests_dir.glob(f"test_{stem}*.py")))
    # dedupe
    seen = set()
    unique = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _load_roadmap() -> dict:
    if not _DOCS_ROADMAP.exists():
        print(f"ERROR: {_DOCS_ROADMAP} missing; run v0.9.25b first",
              file=_sys.stderr)
        raise SystemExit(2)
    return _json.loads(_DOCS_ROADMAP.read_text())


def _list_phases() -> None:
    data = _load_roadmap()
    print(f"Roadmap v{data['current_version']}, "
          f"schema={data['schema_version']}")
    for phase in data["phases"]:
        n_done = sum(1 for i in phase["items"]
                     if i["status"] == "done")
        n_total = len(phase["items"])
        print(f"  {phase['id']:2s} {phase['name']:30s} "
              f"{n_done}/{n_total} done  ({phase['status']})")


def _extract_phase_files(phase_id: str) -> tuple[list[Path], list[str]]:
    """Return (files, warnings) for a phase."""
    data = _load_roadmap()
    phase = next(
        (p for p in data["phases"] if p["id"] == phase_id), None,
    )
    if phase is None:
        raise SystemExit(
            f"phase {phase_id!r} not in roadmap; "
            f"available: {[p['id'] for p in data['phases']]}"
        )
    files: set[Path] = set()
    warnings: list[str] = []
    for item in phase["items"]:
        if item["status"] == "done":
            continue  # already shipped, not current work
        spec = item.get("file", "")
        if not spec:
            continue
        parsed = _parse_file_spec(spec)
        if not parsed and spec:
            warnings.append(
                f"item {item['id']}: could not resolve {spec!r} — "
                f"manual include"
            )
            continue
        for p in parsed:
            files.add(p)
    return sorted(files), warnings


def _write_snapshot(
    out_path: Path,
    mode_label: str,
    full_files: list[Path],
    test_files: list[Path],
    notes: list[str],
) -> None:
    SEP = "# " + "=" * 76
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(SEP + "\\n")
        f.write(f"# CADW SNAPSHOT ({mode_label})\\n")
        from datetime import datetime
        f.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\\n")
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        )
        f.write(f"# HEAD: {r.stdout.strip()}\\n")
        f.write(f"# Full files: {len(full_files)}\\n")
        f.write(f"# Test files: {len(test_files)}\\n")
        for n in notes:
            f.write(f"# NOTE: {n}\\n")
        f.write(SEP + "\\n\\n")
        for p in full_files + test_files:
            f.write(SEP + f"\\n# FILE: {p}\\n" + SEP + "\\n")
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                f.write(f"# ERROR: {e}\\n\\n")
                continue
            if not content.endswith("\\n"):
                content += "\\n"
            f.write(content + "\\n")
    print(f"  wrote {out_path} "
          f"({out_path.stat().st_size / 1024:.1f} KiB)")


def _run_module_mode(args) -> int:
    mod = Path(args.module)
    if not mod.exists():
        raise SystemExit(f"module {mod} does not exist")
    tests = _find_tests_for_module(mod)
    notes = []
    if not tests:
        notes.append(f"no matching tests for {mod.stem}")
    out = Path(args.out or f"snapshot_module_{mod.stem}.md")
    _write_snapshot(out, f"module={mod}", [mod], tests, notes)
    return 0


def _run_phase_mode(args) -> int:
    files, warnings = _extract_phase_files(args.phase)
    if not files:
        print(f"ERROR: phase {args.phase!r} resolved to 0 files",
              file=_sys.stderr)
        for w in warnings:
            print(f"  warning: {w}", file=_sys.stderr)
        return 1
    tests: list[Path] = []
    for f in files:
        if f.parts[0] == "cadw":
            tests.extend(_find_tests_for_module(f))
    tests = sorted(set(tests))
    out = Path(args.out or f"snapshot_phase_{args.phase}.md")
    _write_snapshot(
        out, f"phase={args.phase}", files, tests, warnings,
    )
    return 0


def _run_doc_mode(args) -> int:
    doc = Path(args.doc)
    if not doc.exists():
        raise SystemExit(f"doc {doc} does not exist")
    out = Path(args.out or f"snapshot_doc_{doc.stem}.md")
    _write_snapshot(out, f"doc={doc}", [doc], [], [])
    return 0


# v0.9.25c entry point — recognized by scripts/patch_v0925c invocation.
def _phase_aware_main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="collect_project.py (v0.9.25c phase-aware)",
    )
    ap.add_argument("--module", help="collect one module + tests")
    ap.add_argument("--phase", help="collect phase files from roadmap.json")
    ap.add_argument("--doc", help="collect one doc")
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.list_phases:
        _list_phases()
        return 0
    if args.module:
        return _run_module_mode(args)
    if args.phase:
        return _run_phase_mode(args)
    if args.doc:
        return _run_doc_mode(args)
    print(
        "ERROR: use --module / --phase / --doc / --list-phases. "
        "Full snapshot mode is --mode=smart (see --help).",
        file=_sys.stderr,
    )
    return 2
'''

def add_collect_project_modes():
    print("\n=== B.4: collect_project.py phase-aware ===")
    p = Path("scripts/collect_project.py")
    s = p.read_text()
    if "_phase_aware_main" in s:
        print("  already present — skip")
        return
    # Append before the existing `if __name__` guard, replacing it.
    old_guard = 'if __name__ == "__main__":\n    raise SystemExit(main())\n'
    if old_guard not in s:
        raise SystemExit(
            "cannot find `if __name__ == '__main__'` guard in "
            "scripts/collect_project.py — file may have changed"
        )
    new_guard = (
        COLLECT_PROJECT_ADDITIONS + "\n\n"
        + 'if __name__ == "__main__":\n'
        + '    if any(a in _sys.argv[1:] for a in (\n'
        + '        "--module", "--phase", "--doc", "--list-phases",\n'
        + '    )):\n'
        + '        raise SystemExit(_phase_aware_main(_sys.argv[1:]))\n'
        + '    raise SystemExit(main())\n'
    )
    s = s.replace(old_guard, new_guard, 1)
    p.write_text(s)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  appended phase-aware modes + dispatch")


# ============ 7. Tests for collect_project.py ============

COLLECT_TESTS = '''"""v0.9.25c: collect_project.py --module / --phase / --doc modes."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "collect_project.py"


def _run(*args: str, timeout: int = 60):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
    )


def test_collect_project_list_phases():
    r = _run("--list-phases")
    assert r.returncode == 0, r.stderr
    for pid in "ABCDEFGH":
        assert f" {pid} " in r.stdout or f"  {pid} " in r.stdout, (
            f"phase {pid!r} missing from --list-phases: {r.stdout}"
        )
    assert "Roadmap v" in r.stdout


def test_collect_project_module_mode(tmp_path):
    out = tmp_path / "snap.md"
    r = _run("--module", "cadw/ts/models.py", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()
    text = out.read_text()
    assert "FILE: cadw/ts/models.py" in text
    # Should include its test file
    assert "FILE: tests/test_ts_models.py" in text
    # Marker: SARIMA class present
    assert "class SARIMAForecaster" in text


def test_collect_project_phase_mode(tmp_path):
    out = tmp_path / "phase_a.md"
    r = _run("--phase", "A", "--out", str(out))
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    # Phase A items include cadw/ts/models.py and cadw/ops/sort.py
    assert "FILE: cadw/ts/models.py" in text
    assert "FILE: cadw/ops/sort.py" in text
    # Dataframe.py has A.5 (encode_cat)
    assert "FILE: cadw/dataframe.py" in text


def test_collect_project_doc_mode(tmp_path):
    out = tmp_path / "doc.md"
    r = _run("--doc", "docs/ROADMAP.md", "--out", str(out))
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "FILE: docs/ROADMAP.md" in text
    assert "# CADW Roadmap" in text


def test_collect_project_no_args_errors():
    r = _run()
    assert r.returncode != 0
    assert "--module" in r.stderr or "--mode" in r.stderr
'''

def add_collect_tests():
    print("\n=== tests/test_collect_project_modes.py ===")
    p = Path("tests/test_collect_project_modes.py")
    if p.exists() and "test_collect_project_list_phases" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(COLLECT_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 8. Dogfood ============

def dogfood():
    print("\n=== dogfood: 4 items closed (fresh subprocess) ===")

    def _verify():
        import subprocess as _sp
        import sys as _sys
        import tempfile
        from pathlib import Path as _P

        # (1) A.4 refill AST — no concat_tables in while
        import ast as _ast
        import inspect as _inspect
        import textwrap as _tw
        from cadw.ops import sort as _sort
        src = _tw.dedent(_inspect.getsource(_sort._RunStream.refill))
        tree = _ast.parse(src)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.While):
                for sub in _ast.walk(node):
                    if (isinstance(sub, _ast.Call)
                            and isinstance(sub.func, _ast.Attribute)
                            and sub.func.attr == "concat_tables"):
                        raise AssertionError(
                            f"A.4 not closed: concat_tables inside while"
                        )

        # (2) A.5 encode_cat guard raises on high-card
        import pyarrow as _pa
        import cadw as _cadw
        from cadw import CadwDataFrame as _CDF
        n = 50_000
        t = _pa.table({
            "c": _pa.array([f"k{i:08d}" for i in range(n)],
                           type=_pa.string()),
            "v": list(range(n)),
        })
        df = _CDF.from_table(
            t, memory_guard=_cadw.MemoryGuard(max_memory_mb=2),
        )
        try:
            df.encode_categorical("c")
            raise AssertionError(
                "A.5 not closed: encode_categorical on 50k unique "
                "did not raise"
            )
        except _cadw.MemoryLimitExceeded:
            pass

        # (3) A.7 fan_in uses read_batch_size
        src = _tw.dedent(_inspect.getsource(_sort.external_sort))
        tree = _ast.parse(src)
        found = False
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            if not (isinstance(node.func, _ast.Name)
                    and node.func.id == "compute_fan_in"):
                continue
            arg1 = _ast.unparse(node.args[1])
            assert "read_batch_size" in arg1, (
                f"A.7 not closed: compute_fan_in arg={arg1!r}"
            )
            found = True
            break
        assert found, "A.7: compute_fan_in not called"

        # (3b) fan_in formula: 32MB/1000/640 -> 8
        from cadw.ops.sort import compute_fan_in
        assert compute_fan_in(32 * 1024 * 1024, 1000, 640) == 8
        assert compute_fan_in(32 * 1024 * 1024, 1000, 80) == 69

        # (4) B.4 collect_project --list-phases
        r = _sp.run(
            [_sys.executable, "scripts/collect_project.py",
             "--list-phases"],
            cwd=".", capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        for pid in "ABCDEFGH":
            assert f" {pid} " in r.stdout, (
                f"B.4: phase {pid!r} missing: {r.stdout}"
            )

        print(
            "  4 items closed: A.4 refill linear, A.5 encode_cat guard, "
            "A.7 fan_in rbs+formula, B.4 phase-aware collect"
        )

    run_verify_in_subprocess(_verify)


# ============ 9. Pytest ============

def run_pytest():
    print("\n=== pre-flight: test_facade_has_no_as_py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py", "-q"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** FACADE AS_PY VIOLATION — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: targeted (v0925c + sort + collect) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925c_phase_a.py",
         "tests/test_collect_project_modes.py",
         "tests/test_sort_merge_p_s1.py",
         "tests/test_merge_arrow.py",
         "tests/test_encode_categorical.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 10. main ============

def main():
    print("=== v0.9.25c: Phase A batch 2 (A.4/A.5/A.7/B.4) ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    fix_refill()                # A.4
    fix_fan_in_fn()             # A.7a
    fix_fan_in_call()           # A.7b
    fix_encode_cat()            # A.5
    assert_compiles(Path("cadw/ops/sort.py"))
    assert_compiles(Path("cadw/dataframe.py"))
    add_tests()                 # A.4/A.5/A.7 tests
    flip_stale_test()           # test_compute_fan_in
    add_collect_project_modes() # B.4
    add_collect_tests()         # B.4 tests
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25c: Phase A batch 2 — A.4 + A.5 + A.7 + B.4 "
        "(4 items from docs/roadmap.json). "
        "A.4 (part1_report B1): _RunStream.refill linear concat. "
        "Prior pa.concat_tables([buffer, chunk]) inside while loop "
        "→ quadratic; 12.9 GB copied @ rbs=200k, 0.078s vs 0.0003s. "
        "Now: list collect + single concat_tables. "
        "A.5 (part1_report B2): encode_categorical in-loop guard. "
        "Prior uniques: set accumulated Python objects unbounded; "
        "@ 10M unique strings ~600 MB invisible to MemoryGuard "
        "(batch.nbytes ~130 KB). Now check_state(len(uniques) * 180) "
        "per batch — fail-fast at 50% budget. "
        "A.7 (audit §6.6 + docs/roadmap A.7): compute_fan_in uses "
        "read_batch_size, NOT _WRITE_RUN_CHUNK. Root cause of 24.4s "
        "@ N=500k/8cols/32MB — v0.9.23c _merge_keys uses rbs=1000 "
        "internally, but fan_in got _WRITE_RUN_CHUNK=50000 → K=2 → "
        "3-4x more passes. Correct fix: pass rbs. Now K=8 @ 32MB/8cols "
        "/ 640B/row. Also: formula updated to 3*K*rbs*row <= "
        "state_limit (matches _merge_keys check_state; prior formula "
        "3x under-counted). max_k=256 cap for FD limits. "
        "Audit's 'floor=8' was a proxy; not applied as floor — "
        "formula gives correct K naturally. "
        "B.4 (docs/roadmap.json): collect_project.py --module / "
        "--phase / --doc / --list-phases. Reads docs/roadmap.json. "
        "Brace-expansion for composite file specs (cadw/{a,b,c}/*.py). "
        "--module is main workflow unlock: 27 files → 5 files/session. "
        "Test additions: 10 new (test_v0925c_phase_a.py, "
        "test_collect_project_modes.py) + 1 flipped "
        "(test_compute_fan_in_uses_write_run_chunk → "
        "test_compute_fan_in_uses_read_batch_size — old test asserted "
        "the buggy behavior, written before v0.9.23c changed internal "
        "merge to rbs; §13.4 stale-test-on-refactor). "
        "Dogfood: 4 invariants verified. "
        "Fast suite: ~2275 → ~2285.",
        self_path="scripts/patch_v0925c_phase_a_batch2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25c applied. Next: v0.9.25d (A.6: 18 to_pylist "
          "sites extra_factor across dataframe/ops/ts).")


if __name__ == "__main__":
    main()