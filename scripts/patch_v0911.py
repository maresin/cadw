#!/usr/bin/env python3
# v0.9.11 — PF4 (deferred C1 widen, 1000x fewer allocs) + U1 (test data).
#
# PF4 (from v0.9.10 session, self-found): _maybe_upgrade_for_row is
# called per _emit_row (per group row), allocates `upgraded: list = []`
# of length K every call. For 10M groups × K=10 → ~100M list allocs
# + ~1B attr checks. Hot path of _per_group_apply on small groups.
#
# Fix: defer to _flush_buf (once per 1000 rows / BATCH_ROWS). Called
# with full buf_rows, one scan. Plus: replace silent post-flush
# truncation (writer already committed → ArrowInvalid or worse) with
# clear RuntimeError. This makes §11.5 BATCH_ROWS limitation explicit.
#
# U1: test_dt_ceil_date64_raises_init creates pa.date32() — copy-paste
# from test_dt_floor_date32_raises_init. Both types raise TypeError, so
# test passes; but it doesn't test date64. Fix: from_table (in-memory,
# no parquet round-trip which converts date64 → date32, §11.5) with
# actual pa.date64().
#
# Skipped from backlog (defer to Pingis round 2 + v0.9.12):
#   T2.2 (assert_no_call_in_source helper promotion) — tooling
#   T2.5 (pre-import restore docstring) — doc
#   U3 (pyproject version 0.1.0 vs 0.9.10) — cosmetic, needs uv lock regen
#   PF1-PF3 (heavy_keys / accumulate_heavy / pivot dict) — need benchmark
#   validation via Pingis round 2
#
# Run: uv run python scripts/patch_v0911.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ PF4: defer widen to _flush_buf ============

OLD_PF4 = (
    "        def _maybe_upgrade_for_row(row: dict) -> None:\n"
    "            \"\"\"v0.9.10 (C1): widen int->float64 on fractional float.\n"
    "\n"
    "            _build_out_schema infers from first group only. If first is\n"
    "            int, later is float, pa.Table.from_pylist(rows, schema=int64)\n"
    "            either truncates (silent corruption) or raises ArrowInvalid.\n"
    "            Widen to float64 before flush. Lossless for |v| < 2^53.\n"
    "\n"
    "            Limitation: only before first flush (BATCH_ROWS=1000).\n"
    "            If mixed appears after first flush, writer committed to\n"
    "            int64 -> ArrowInvalid. Workaround: return float(x) from fn().\n"
    "            \"\"\"\n"
    "            if not out_schema_box:\n"
    "                return\n"
    "            schema = out_schema_box[0]\n"
    "            upgraded = []\n"
    "            needs = False\n"
    "            for field in schema:\n"
    "                val = row.get(field.name)\n"
    "                if pa.types.is_integer(field.type) and isinstance(val, float):\n"
    "                    if val != int(val):\n"
    "                        upgraded.append(pa.field(\n"
    "                            field.name, pa.float64(),\n"
    "                            nullable=field.nullable,\n"
    "                        ))\n"
    "                        needs = True\n"
    "                        continue\n"
    "                upgraded.append(field)\n"
    "            if needs:\n"
    "                out_schema_box[0] = pa.schema(upgraded)\n"
    "\n"
    "        def _emit_row(row: dict) -> None:\n"
    "            \"\"\"Append a row, widening schema if C1 numeric conflict.\"\"\"\n"
    "            nonlocal buf_rows\n"
    "            _maybe_upgrade_for_row(row)\n"
    "            buf_rows.append(row)\n"
)

NEW_PF4 = (
    "        def _maybe_upgrade_schema() -> None:\n"
    "            \"\"\"v0.9.11 (PF4): widen int->float64 on fractional float.\n"
    "\n"
    "            Deferred from per-row (_maybe_upgrade_for_row) to\n"
    "            once-per-flush. Per-row allocation of `upgraded` list\n"
    "            was O(N*K) — for 10M groups x K=10, ~100M list allocs\n"
    "            + ~1B attr checks. Now: one scan of buf_rows\n"
    "            (<= BATCH_ROWS=1000 rows) per flush. ~1000x fewer allocs,\n"
    "            ~10x fewer attr checks.\n"
    "\n"
    "            Checks each integer field across all buf_rows; if any\n"
    "            fractional float found -> widen to float64. Lossless\n"
    "            for |v| < 2^53.\n"
    "\n"
    "            Limitation (§11.5): only before first flush. Once writer\n"
    "            committed (writer_box non-empty), post-flush widen raises\n"
    "            clear RuntimeError (prior: silent truncation or cryptic\n"
    "            ArrowInvalid at write_batch). Workaround: return\n"
    "            float(x) from fn() if any group might be fractional.\n"
    "            \"\"\"\n"
    "            if not out_schema_box:\n"
    "                return\n"
    "            schema = out_schema_box[0]\n"
    "            upgraded = []\n"
    "            needs = False\n"
    "            widen_fields: list = []\n"
    "            for field in schema:\n"
    "                if not pa.types.is_integer(field.type):\n"
    "                    upgraded.append(field)\n"
    "                    continue\n"
    "                widen = False\n"
    "                for row in buf_rows:\n"
    "                    val = row.get(field.name)\n"
    "                    if isinstance(val, float) and val != int(val):\n"
    "                        widen = True\n"
    "                        break\n"
    "                if widen:\n"
    "                    upgraded.append(pa.field(\n"
    "                        field.name, pa.float64(),\n"
    "                        nullable=field.nullable,\n"
    "                    ))\n"
    "                    needs = True\n"
    "                    widen_fields.append(field.name)\n"
    "                else:\n"
    "                    upgraded.append(field)\n"
    "            if needs:\n"
    "                if writer_box:\n"
    "                    raise RuntimeError(\n"
    "                        f\"_per_group_apply: post-flush widen needed \"\n"
    "                        f\"for {widen_fields}, but writer already \"\n"
    "                        f\"committed. Workaround: return float(x) \"\n"
    "                        f\"from fn() if any group might be fractional. \"\n"
    "                        f\"See §11.5.\"\n"
    "                    )\n"
    "                out_schema_box[0] = pa.schema(upgraded)\n"
    "\n"
    "        def _emit_row(row: dict) -> None:\n"
    "            \"\"\"Append row; widen check deferred to _flush_buf (PF4).\"\"\"\n"
    "            nonlocal buf_rows\n"
    "            buf_rows.append(row)\n"
)


def pf4_fix():
    print("\n=== PF4: defer widen to _flush_buf ===")
    pu.apply_or_fail(
        "cadw/dataframe.py", OLD_PF4, NEW_PF4, label="PF4 replace",
    )

    # Insert _maybe_upgrade_schema() call in _flush_buf, before writer
    # open. `_flush_buf` starts:
    #   def _flush_buf():
    #       nonlocal buf_rows
    #       if not buf_rows:
    #           return
    #       if not writer_box:
    pu.apply_or_fail(
        "cadw/dataframe.py",
        (
            "        def _flush_buf():\n"
            "            nonlocal buf_rows\n"
            "            if not buf_rows:\n"
            "                return\n"
            "            if not writer_box:\n"
        ),
        (
            "        def _flush_buf():\n"
            "            nonlocal buf_rows\n"
            "            if not buf_rows:\n"
            "                return\n"
            "            _maybe_upgrade_schema()  # v0.9.11 (PF4)\n"
            "            if not writer_box:\n"
        ),
        label="PF4 flush call",
    )
    print("  PF4 OK")


# ============ U1: fix test_dt_ceil_date64 to use date64 ============

OLD_U1 = '''def test_dt_ceil_date64_raises_init(tmp_path):
    import datetime as dt
    import pyarrow as pa
    import pyarrow.parquet as pq
    from cadw import CadwDataFrame, MemoryGuard
    t = pa.table({'d': pa.array([dt.date(2020, 1, 1)], type=pa.date32())})
    p = tmp_path / 'src.parquet'
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / 'ds', chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / 'spill',
    )
    import pytest
    with pytest.raises(TypeError, match='expected timestamp'):
        df['d'].dt.ceil('1D')'''

NEW_U1 = '''def test_dt_ceil_date64_raises_init(tmp_path):
    """v0.9.11 (U1): actual pa.date64() (prior: copy-paste from date32).

    date32/64 → TypeError in dt.ceil (N2). Uses from_table (in-memory,
    no parquet round-trip: parquet has no ms-date type, date64 → date32
    on read, §11.5). Prior test used pa.date32() despite name.
    """
    import datetime as dt
    import pyarrow as pa
    from cadw import CadwDataFrame
    t = pa.table({'d': pa.array([dt.date(2020, 1, 1)], type=pa.date64())})
    df = CadwDataFrame.from_table(t)
    import pytest
    with pytest.raises(TypeError, match='expected timestamp'):
        df['d'].dt.ceil('1D')'''


def u1_fix():
    print("\n=== U1: test_dt_ceil_date64_raises_init → pa.date64() ===")
    pu.apply_or_fail(
        "tests/test_dt_accessor.py", OLD_U1, NEW_U1, label="U1",
    )
    print("  U1 OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.11 + regression ===")
    pu.verify_anchors([
        # PF4
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/dataframe.py", "v0.9.11 (PF4)"),
        ("cadw/dataframe.py", "post-flush widen needed"),
        # U1
        ("tests/test_dt_accessor.py", "v0.9.11 (U1): actual pa.date64()"),
        # v0.9.10 (regression — C1 still functional)
        ("cadw/dataframe.py", "_emit_row(row)"),
        ("cadw/dataframe.py", "v0.9.10 (U2)"),
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
        ("cadw/pipeline.py", "raw_columns. Fix (backlog v0.9.11)"),
        # v0.9.9 (regression)
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
        ("cadw/pipeline.py", "v0.9.9 (N1): use __serialization_name__"),
        ("cadw/accessors/_dt.py", "def _validate_timestamp_only("),
        ("cadw/dataframe.py", "Per-group apply (v0.9.6b2: CADW-native"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.11 via subprocess ===")

    def _verify_all():
        import ast as _ast
        import inspect
        from cadw.dataframe import CadwDataFrame

        src = inspect.getsource(CadwDataFrame._per_group_apply)
        tree = _ast.parse(src)

        # PF4: _maybe_upgrade_schema def present
        has_new = any(
            isinstance(n, _ast.FunctionDef)
            and n.name == "_maybe_upgrade_schema"
            for n in _ast.walk(tree)
        )
        assert has_new, "no _maybe_upgrade_schema def"

        # PF4: old name gone
        has_old = any(
            isinstance(n, _ast.FunctionDef)
            and n.name == "_maybe_upgrade_for_row"
            for n in _ast.walk(tree)
        )
        assert not has_old, "_maybe_upgrade_for_row still present"

        # PF4: _flush_buf body calls _maybe_upgrade_schema()
        flush_calls = False
        for n in _ast.walk(tree):
            if not (isinstance(n, _ast.FunctionDef) and n.name == "_flush_buf"):
                continue
            for sub in _ast.walk(n):
                if (isinstance(sub, _ast.Call)
                        and isinstance(sub.func, _ast.Name)
                        and sub.func.id == "_maybe_upgrade_schema"):
                    flush_calls = True
                    break
        assert flush_calls, "_flush_buf does not call _maybe_upgrade_schema"

        # PF4: post-flush widen raises clear RuntimeError
        assert "post-flush widen needed" in src, (
            "no clear RuntimeError for post-flush widen"
        )

        # U1: date64 test uses pa.date64()
        u1_src = inspect.getsource(
            __import__("tests.test_dt_accessor", fromlist=["test_dt_ceil_date64_raises_init"])
            .test_dt_ceil_date64_raises_init
        )
        assert "pa.date64()" in u1_src, "U1 test still uses date32"
        assert "from_table" in u1_src, "U1 test should use from_table"

        print("  PF4 (deferred widen, clear post-flush error) OK")
        print("  U1 (pa.date64 + from_table) OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_apply.py",
         "tests/test_apply_e2e.py",
         "tests/test_dt_accessor.py",
         "tests/test_patch_utils.py", "-q"],
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
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=900,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.11 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    pf4_fix()
    u1_fix()
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.11: PF4 (deferred C1 widen) + U1 (test data). "
        "PF4: _maybe_upgrade_for_row called per _emit_row (per group "
        "row), allocates `upgraded: list = []` (K elements) every "
        "call. For 10M groups × K=10 → ~100M list allocs + ~1B attr "
        "checks — hot path of _per_group_apply on small groups. "
        "Deferred to _flush_buf (once per 1000 rows/BATCH_ROWS): one "
        "scan of buf_rows per flush. ~1000x fewer allocs, ~10x fewer "
        "attr checks. Plus: clear RuntimeError on post-flush widen "
        "(prior: silent truncation or cryptic ArrowInvalid at "
        "write_batch) — §11.5 BATCH_ROWS limitation now explicit. "
        "U1: test_dt_ceil_date64_raises_init created pa.date32() "
        "(copy-paste from date32 test); passes because both types "
        "raise TypeError, but doesn't test date64. Fix: from_table "
        "(in-memory, no parquet round-trip which converts date64 → "
        "date32, §11.5) with actual pa.date64(). "
        "C1 test (test_apply_mixed_int_float_widens_schema) still "
        "passes: buf_rows has 2 groups, first flush has both rows "
        "→ widen before first writer open. Post-flush case now "
        "raises clear RuntimeError. "
        "Deferred: T2.2 (assert_no_call helper), T2.5 (pre-import "
        "restore doc), U3 (pyproject version, needs uv lock), "
        "PF1-PF3 (need Pingis round 2 benchmark validation).",
        self_path="scripts/patch_v0911.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.11 applied.")


if __name__ == "__main__":
    main()
