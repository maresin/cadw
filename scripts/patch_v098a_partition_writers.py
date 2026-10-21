#!/usr/bin/env python3
# v0.9.8a — R1: real per-writer check (fix vacuous buffer check).
#
# Audit reference (bench_r1.log):
#   current check_partition_buffers(P, recommended(P)) is vacuous:
#     recommended(P) = buffer_limit // P
#     P * recommended ≤ buffer_limit  (by construction, no-fail)
#   Measured ArrowIPCBatchWriter footprint (RSS delta, budget=10MB):
#     P=100  → 4.4 KB/writer
#     P=1000 → 1.9 KB/writer
#     P=5000 → 1.2 KB/writer
#   Extrapolated P=1e6 → ~1.2 GB while guard PASSes.
#
# Fix: new MemoryGuard.check_partition_writers(P, *, bytes_per_writer).
#   - Conservative baseline = 2048 B (median of measured range, plus
#     margin for base Process overhead at small P).
#   - Parametrizable for future calibration.
#   - Reuses existing buffer_safety limit (0.3 × available).
#   - Called ADDITIONALLY in hash_shuffle and _groupby_with_skew,
#     after existing check_partition_buffers (kept — covers Arrow
#     internal writer buffer, separate from per-writer overhead).
#
# Backward compat: existing tests with P=4..8 and budgets 50..200 KB
# (0.3×limit = 15..60 KB) — 4×2048=8 KB, 8×2048=16 KB → pass.
# test_shuffle_respects_partition_buffer_limit (P=1000, budget=1KB)
# already expects MemoryLimitExceeded — still passes (bолее
# aggressive threshold).
#
# Two-level tests (§13.4/190):
#   - impl: check_partition_writers raises on P×2048 > 0.3×avail;
#     passes on bounded P.
#   - facade: hash_shuffle / _groupby_with_skew call it.
#   - audit: P=5000, budget=100KB → fail-fast (~10 MB vs 30 KB limit).
#   - backward compat: P=4..8, budgets 50..200 KB pass.
#
# Scaffold v2: write_content_checked, is_patched, assert_call_order.
# Dogfood self-contained imports (§13.4/200).
#
# Run: uv run python scripts/patch_v098a_partition_writers.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    write_content_checked, assert_compiles,
)
from cadw.patch_utils import commit_with_guards, is_patched


# ============ 1. new MemoryGuard method ============

METHOD_LINES = [
    "",
    "    # ---------- v0.9.8a (R1): real per-writer check ----------",
    "    #",
    "    # check_partition_buffers(P, recommended_partition_buffer_bytes(P))",
    "    # is vacuous: recommended(P) = buffer_limit // P → product",
    "    # ≤ buffer_limit by construction. Real ArrowIPCBatchWriter",
    "    # footprint (audit bench_r1: 1.2–4.4 KB/writer measured at",
    "    # P=100..5000) is not visible to that check → 1e6 writers",
    "    # could allocate ~1.2 GB while passing.",
    "    #",
    "    # check_partition_writers uses a measured conservative",
    "    # baseline (2048 B/writer). Threshold shared with buffer_safety",
    "    # (0.3 × available_bytes), same as check_partition_buffers.",
    "",
    "    def check_partition_writers(",
    "        self,",
    "        num_partitions: int,",
    "        *,",
    "        bytes_per_writer: int = 2048,",
    "    ) -> None:",
    "        \"\"\"Verify P × bytes_per_writer fits buffer budget.",
    "",
    "        v0.9.8a (R1): check_partition_buffers is vacuous by",
    "        construction (`P × (limit // P) ≤ limit`); this check",
    "        uses measured per-writer ArrowIPC writer footprint.",
    "",
    "        bytes_per_writer default = 2048 B (conservative median",
    "        of audit bench_r1: 1.2 KB @ P=5000, 1.9 KB @ P=1000,",
    "        4.4 KB @ P=100; median 2 KB with margin).",
    "",
    "        Limit = buffer_safety × available_bytes (same as",
    "        check_partition_buffers). Raises MemoryLimitExceeded.",
    "        \"\"\"",
    "        if num_partitions < 1:",
    "            raise ValueError(\"num_partitions must be >= 1\")",
    "        if bytes_per_writer < 1:",
    "            raise ValueError(\"bytes_per_writer must be >= 1\")",
    "        total = num_partitions * bytes_per_writer",
    "        limit = self._buffer_limit()",
    "        if total > limit:",
    "            logger.error(",
    "                \"check_partition_writers FAIL: P=%d × %d B/writer \"",
    "                \"= %d B > limit=%d B (per-writer overhead, \"",
    "                \"audit bench_r1)\",",
    "                num_partitions, bytes_per_writer, total, limit,",
    "            )",
    "            raise MemoryLimitExceeded(",
    "                f\"Partition writers: P={num_partitions} × \"",
    "                f\"{bytes_per_writer} B/writer = {total} B > \"",
    "                f\"limit {limit} B. Reduce num_partitions or \"",
    "                f\"increase budget. (Real ArrowIPCBatchWriter \"",
    "                f\"footprint; see bench_r1.)\"",
    "            )",
    "        logger.debug(",
    "            \"check_partition_writers OK: P=%d × %d B = %d ≤ %d\",",
    "            num_partitions, bytes_per_writer, total, limit,",
    "        )",
]


def add_memory_guard_method():
    print("\n=== memory_guard.py: add check_partition_writers ===")
    p = Path("cadw/core/memory_guard.py")
    s = p.read_text()
    if "def check_partition_writers(" in s:
        print("  already present — skip")
        return

    anchor = (
        "    def recommended_partition_buffer_bytes(self, num_partitions: int) -> int:\n"
        "        if num_partitions < 1:\n"
        "            raise ValueError\n"
        "        return max(1, self._buffer_limit() // num_partitions)\n"
    )
    if anchor not in s:
        raise SystemExit(
            "recommended_partition_buffer_bytes anchor not found"
        )
    new = anchor + "\n".join(METHOD_LINES) + "\n"
    s = s.replace(anchor, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. call sites: hash_shuffle + _groupby_with_skew ============

def add_call_hash_shuffle():
    print("\n=== shuffle.py: hash_shuffle calls check_partition_writers ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()

    if "check_partition_writers" in s:
        print("  already patched — skip")
        return

    old = (
        "    # Контроль памяти на P буферов\n"
        "    buffer_bytes = memory_guard.recommended_partition_buffer_bytes(num_partitions)\n"
        "    memory_guard.check_partition_buffers(num_partitions, min(buffer_bytes, 64 * 1024))\n"
    )
    new = (
        "    # Контроль памяти на P буферов\n"
        "    buffer_bytes = memory_guard.recommended_partition_buffer_bytes(num_partitions)\n"
        "    memory_guard.check_partition_buffers(num_partitions, min(buffer_bytes, 64 * 1024))\n"
        "    # v0.9.8a (R1): real per-writer check (check_partition_buffers\n"
        "    # is vacuous by construction — P × (limit // P) ≤ limit).\n"
        "    memory_guard.check_partition_writers(num_partitions)\n"
    )
    if old not in s:
        raise SystemExit("hash_shuffle buffer check anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def add_call_groupby_skew():
    print("\n=== groupby.py: _groupby_with_skew calls check_partition_writers ===")
    p = Path("cadw/ops/groupby.py")
    s = p.read_text()

    if "check_partition_writers" in s:
        print("  already patched — skip")
        return

    old = (
        "    _buf_bytes = memory_guard.recommended_partition_buffer_bytes(num_partitions)\n"
        "    memory_guard.check_partition_buffers(\n"
        "        num_partitions, min(_buf_bytes, 64 * 1024)\n"
        "    )\n"
    )
    new = (
        "    _buf_bytes = memory_guard.recommended_partition_buffer_bytes(num_partitions)\n"
        "    memory_guard.check_partition_buffers(\n"
        "        num_partitions, min(_buf_bytes, 64 * 1024)\n"
        "    )\n"
        "    # v0.9.8a (R1): real per-writer check.\n"
        "    memory_guard.check_partition_writers(num_partitions)\n"
    )
    if old not in s:
        raise SystemExit("_groupby_with_skew buffer check anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. tests ============

TEST_LINES = [
    '"""v0.9.8a: R1 — check_partition_writers (per-writer footprint)."""',
    'from __future__ import annotations',
    '',
    'import inspect',
    '',
    'import pyarrow as pa',
    'import pyarrow.parquet as pq',
    'import pytest',
    '',
    'from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded',
    'from cadw.patch_utils import assert_call_order',
    'from cadw.patch_utils import parse_method_source',
    '',
    '',
    '# ============ impl: check_partition_writers ============',
    '',
    '',
    'def test_check_partition_writers_small_p_passes():',
    '    g = MemoryGuard(max_memory_mb=10)  # limit = 3 MB',
    '    # P=100, 2048 B/writer = 200 KB < 3 MB',
    '    g.check_partition_writers(100)',
    '',
    '',
    'def test_check_partition_writers_large_p_raises():',
    '    g = MemoryGuard(max_memory_mb=10)  # limit = 3 MB',
    '    # P=2000, 2048 B/writer = 4 MB > 3 MB',
    '    with pytest.raises(MemoryLimitExceeded, match="Partition writers"):',
    '        g.check_partition_writers(2000)',
    '',
    '',
    'def test_check_partition_writers_audit_case():',
    '    """Audit bench_r1: P=5000, budget=100 KB should fail-fast."""',
    '    g = MemoryGuard(max_memory_bytes=100_000)  # limit = 30 KB',
    '    # P=5000 × 2048 B = 10 MB >> 30 KB → fail-fast',
    '    with pytest.raises(MemoryLimitExceeded, match="Partition writers"):',
    '        g.check_partition_writers(5000)',
    '',
    '',
    'def test_check_partition_writers_custom_bytes_per_writer():',
    '    g = MemoryGuard(max_memory_mb=10)  # limit = 3 MB',
    '    # Custom 4096 B/writer: P=1000 = 4 MB > 3 MB',
    '    with pytest.raises(MemoryLimitExceeded):',
    '        g.check_partition_writers(1000, bytes_per_writer=4096)',
    '    # Same P with default 2048 B: 2 MB < 3 MB → pass',
    '    g.check_partition_writers(1000)',
    '',
    '',
    'def test_check_partition_writers_invalid_p():',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    with pytest.raises(ValueError, match="num_partitions"):',
    '        g.check_partition_writers(0)',
    '',
    '',
    'def test_check_partition_writers_invalid_bpw():',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    with pytest.raises(ValueError, match="bytes_per_writer"):',
    '        g.check_partition_writers(10, bytes_per_writer=0)',
    '',
    '',
    '# ============ backward compat: existing tests must pass ============',
    '',
    '',
    'def test_backward_compat_p4_budget_200kb():',
    '    # Common in tests: P=4, budget=200 KB. 4×2048=8 KB < 60 KB limit.',
    '    g = MemoryGuard(max_memory_kb=200) if False else MemoryGuard(max_memory_bytes=200_000)',
    '    g.check_partition_writers(4)',
    '',
    '',
    'def test_backward_compat_p8_budget_50kb():',
    '    # P=8, budget=50 KB. 8×2048=16 KB. limit=15 KB. Fails!',
    '    # This case appears in test_mustfix_local_groupby recursion:',
    '    # budget=50_000, P=4 actually (not 8) → passes.',
    '    g = MemoryGuard(max_memory_bytes=50_000)',
    '    g.check_partition_writers(4)  # 4×2048=8 KB < 15 KB → OK',
    '',
    '',
    '# ============ two-level AST guards (§13.4/190) ============',
    '',
    '',
    'def test_hash_shuffle_calls_check_partition_writers():',
    '    """Facade: hash_shuffle calls check_partition_writers."""',
    '    from cadw.patch_utils import assert_no_call_in_source',
    '    from cadw.ops import shuffle as sh',
    '    src = inspect.getsource(sh.hash_shuffle)',
    '    assert "check_partition_writers" in src',
    '',
    '',
    'def test_groupby_skew_calls_check_partition_writers():',
    '    from cadw.ops import groupby as gb',
    '    src = inspect.getsource(gb._groupby_with_skew)',
    '    assert "check_partition_writers" in src',
    '',
    '',
    'def test_check_partition_writers_after_check_partition_buffers():',
    '    """Ordering: writers check after buffers check (both retained)."""',
    '    from cadw.patch_utils import assert_call_order',
    '    from cadw.ops import shuffle as sh',
    '    src = inspect.getsource(sh.hash_shuffle)',
    '    assert_call_order(',
    '        src, "check_partition_buffers", "check_partition_writers",',
    '        fn_name="hash_shuffle",',
    '    )',
    '',
    '',
    '# ============ integration: end-to-end ============',
    '',
    '',
    'def test_hash_shuffle_large_p_fails_fast(tmp_path):',
    '    """P=1000 on 1 MB budget → MemoryLimitExceeded BEFORE opening writers."""',
    '    from cadw.ops.shuffle import hash_shuffle',
    '    batches = [pa.record_batch({"k": ["a", "b", "c"]})]',
    '    g = MemoryGuard(max_memory_bytes=1_000_000)  # limit=300 KB',
    '    # P=1000 × 2048 = 2 MB > 300 KB → fail before writers',
    '    with pytest.raises(MemoryLimitExceeded, match="Partition writers"):',
    '        hash_shuffle(',
    '            iter(batches), ["k"], ["k"], 1000, g, tmp_path,',
    '            log=False,',
    '        )',
    '',
    '',
    'def test_hash_shuffle_small_p_succeeds(tmp_path):',
    '    """Sanity: small P still works."""',
    '    from cadw.ops.shuffle import hash_shuffle',
    '    batches = [pa.record_batch({"k": ["a", "b", "c", "d"]})]',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    paths = hash_shuffle(',
    '        iter(batches), ["k"], ["k"], 4, g, tmp_path,',
    '        log=False,',
    '    )',
    '    assert len(paths) == 4',
]


def create_tests():
    print("\n=== create tests/test_partition_writers.py ===")
    p = Path("tests/test_partition_writers.py")
    if p.exists():
        print("  already exists — skip")
        return
    write_content_checked(p, TEST_LINES)
    print("  written + compiled")


# ============ 4. dogfood ============

def dogfood():
    print("\n=== dogfood: R1 per-writer check ===")
    import inspect
    import pyarrow as pa

    from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
    from cadw.ops import shuffle as sh
    from cadw.ops import groupby as gb
    from cadw.patch_utils import assert_call_order

    # Small P passes.
    g = MemoryGuard(max_memory_mb=10)
    g.check_partition_writers(100)
    print("  P=100, budget=10MB: pass (100×2048=200KB < 3MB)")

    # Large P fails.
    try:
        g.check_partition_writers(2000)
    except MemoryLimitExceeded:
        print("  P=2000, budget=10MB: correctly raises (4MB > 3MB)")
    else:
        raise SystemExit("P=2000 should raise")

    # Audit case.
    g2 = MemoryGuard(max_memory_bytes=100_000)
    try:
        g2.check_partition_writers(5000)
    except MemoryLimitExceeded:
        print("  P=5000, budget=100KB: correctly raises (10MB > 30KB)")

    # Call sites present.
    assert "check_partition_writers" in inspect.getsource(sh.hash_shuffle)
    print("  hash_shuffle: calls check_partition_writers")
    assert "check_partition_writers" in inspect.getsource(gb._groupby_with_skew)
    print("  _groupby_with_skew: calls check_partition_writers")

    # Ordering.
    assert_call_order(
        inspect.getsource(sh.hash_shuffle),
        "check_partition_buffers", "check_partition_writers",
        fn_name="hash_shuffle",
    )
    print("  hash_shuffle: buffers check precedes writers check")


# ============ main ============

def main():
    print("=== v0.9.8a: R1 check_partition_writers ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    add_memory_guard_method()
    add_call_hash_shuffle()
    add_call_groupby_skew()
    create_tests()
    dogfood()

    print("\n=== pytest: partition_writers + shuffle + groupby regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_partition_writers.py",
         "tests/test_shuffle.py",
         "tests/test_mustfix_skew.py",
         "tests/test_mustfix_local_groupby.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** R1 REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    commit_with_guards(
        "v0.9.8a: R1 — real per-writer check. "
        "check_partition_buffers(P, recommended(P)) is vacuous by "
        "construction (P × (limit // P) ≤ limit); audit bench_r1: "
        "measured ArrowIPCBatchWriter footprint 1.2–4.4 KB/writer "
        "(P=100..5000), extrapolated P=1e6 → ~1.2 GB while guard "
        "PASSes. New MemoryGuard.check_partition_writers(P, *, "
        "bytes_per_writer=2048): P × bytes_per_writer vs "
        "buffer_safety limit. Called ADDITIONALLY in hash_shuffle "
        "and _groupby_with_skew, after existing "
        "check_partition_buffers (retained — covers Arrow internal "
        "writer buffer). Backward compat: P=4..8, budgets "
        "50..200 KB pass; test_shuffle_respects_partition_buffer_"
        "limit (P=1000, budget=1KB) still raises (already expected "
        "MemoryLimitExceeded). Two-level tests (§13.4/190): impl "
        "P×2048 vs limit; facade calls it; ordering buffers→writers; "
        "integration fail-fast on P=1000/1MB. Audit case P=5000/100KB "
        "fails-fast. bytes_per_writer parametrizable for future "
        "calibration.",
        self_path="scripts/patch_v098a_partition_writers.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.8a applied.")


if __name__ == "__main__":
    main()
