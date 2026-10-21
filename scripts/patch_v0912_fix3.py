#!/usr/bin/env python3
# v0.9.12-fix3 — whitelist _detect_heavy_keys + fix stale rationale.
# v0.9.12 COMPLETE.
#
# v0.9.12-fix2 applied, dogfood+pytest green, but fast pytest failed
# test_no_unguarded_high_risk_arrow_calls:
#   cadw/ops/groupby.py:507: group_by() in _detect_heavy_keys
#   cadw/ops/groupby.py:499: from_batches() in _detect_heavy_keys
#   without preceding check_state/check_batch/stat
#
# F12-4 (v0.9.12) made _detect_heavy_keys Arrow-native (prior per-row
# Python loop, no Arrow API → no AST guard trigger). Now — high-risk
# Arrow calls, no explicit guard.
#
# Fix per §3.11: whitelist with rationale "helper with verified
# bounded allocation". Caller chain:
#   groupby_aggregate samples batch_iter
#     sample_cap=10000 rows or sample_cap_bytes=0.3×available_bytes
#     skip_skew=True if sample_bytes > 0.5×available_bytes (then not called)
#   → _detect_heavy_keys(sample_batches, by_columns)
# So input is bounded by caller's MemoryGuard. Same pattern as
# existing whitelist entries:
#   _accumulate_heavy: "heavy keys ≤ 20 rows (bounded by sample)"
#   _estimate_k_sample: "first batch only; to_pandas on ≤sample rows"
#
# Also (popup): _estimate_k_sample rationale stale — v0.9.6c1 replaced
# pandas path with dictionary_encode (own hash impl, no pandas), but
# whitelist still says "to_pandas on ≤sample rows". Fix in same edit
# (both groupby.py and dedup.py entries).
#
# §13.4/219 (new): whitelist with caller-bounded rationale requires
# guard on caller count. If a future caller passes unbounded input,
# whitelist is a lie — no explicit check, but sample not actually
# bounded. Meta-test test_detect_heavy_keys_single_caller ensures
# caller count stays 1 (groupby_aggregate only).
#
# State on disk (from v0.9.12-fix2, all applied + verified):
#   F12-1 topk incremental acc_bytes
#   F12-2 newline="" + docstring
#   F12-4 Arrow-native _detect_heavy_keys (but triggers AST guard)
#   fix2 _compute_heavy_mask fill_null (§13.4/218)
#   commit never reached (fast pytest failed)
#
# Run: uv run python scripts/patch_v0912_fix3.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ 1. Whitelist update ============

def fix_whitelist():
    print("\n=== whitelist: add _detect_heavy_keys, fix stale _estimate_k_sample ===")
    pu.apply_or_fail(
        "tests/test_arrow_call_memory_guard.py",
        (
            '    "cadw/ops/dedup.py:_estimate_k_sample":\n'
            '        "first batch only; to_pandas on ≤sample rows",\n'
            '    "cadw/ops/groupby.py:_estimate_k_sample":\n'
            '        "first batch only; to_pandas on ≤sample rows",\n'
        ),
        (
            '    "cadw/ops/dedup.py:_estimate_k_sample":\n'
            '        "first batch only; v0.9.6c1 Arrow-native "\n'
            '        "(dictionary_encode, no pandas); caller-bounded "\n'
            '        "sample",\n'
            '    "cadw/ops/groupby.py:_detect_heavy_keys":\n'
            '        "sample_batches bounded by groupby_aggregate "\n'
            '        "(sample_cap 10k rows / 0.3×avail; skip_skew if "\n'
            '        "> 0.5×avail); pa.concat_tables + group_by on "\n'
            '        "bounded sample; peak ~2×sample + Arrow pool, "\n'
            '        "caller MemoryGuard (§9.2.8)",\n'
            '    "cadw/ops/groupby.py:_estimate_k_sample":\n'
            '        "first batch only; v0.9.6c1 Arrow-native "\n'
            '        "(dictionary_encode, no pandas); caller-bounded "\n'
            '        "sample (10k rows / 0.3×avail)",\n'
        ),
        label="whitelist",
    )
    print("  OK")


# ============ 2. Meta-guard tests ============

NEW_TESTS = [
    "",
    "",
    "# ============ v0.9.12-fix3 (§3.11, §13.4/219): whitelist guards ============",
    "",
    "",
    "def test_detect_heavy_keys_in_whitelist():",
    "    \"\"\"_detect_heavy_keys whitelisted with caller-bounded rationale.\"\"\"",
    "    key = 'cadw/ops/groupby.py:_detect_heavy_keys'",
    "    assert key in WHITELISTED_FUNCS",
    "    rationale = WHITELISTED_FUNCS[key]",
    "    assert 'groupby_aggregate' in rationale or 'caller' in rationale.lower()",
    "",
    "",
    "def test_detect_heavy_keys_single_caller():",
    "    \"\"\"§13.4/219: whitelist rationale assumes single caller.",
    "",
    "    If a new caller appears (potentially unbounded input), the",
    "    whitelist becomes a lie — no explicit guard, sample not",
    "    actually bounded. Guard: exactly 1 call site in groupby.py.",
    "    \"\"\"",
    "    import ast",
    "    import inspect",
    "    from cadw.ops import groupby as gb_mod",
    "    src = inspect.getsource(gb_mod)",
    "    tree = ast.parse(src)",
    "    call_sites = [",
    "        n.lineno for n in ast.walk(tree)",
    "        if isinstance(n, ast.Call)",
    "        and isinstance(n.func, ast.Name)",
    "        and n.func.id == '_detect_heavy_keys'",
    "    ]",
    "    assert len(call_sites) == 1, (",
    "        f'_detect_heavy_keys called from {len(call_sites)} sites; '",
    "        f'whitelist rationale assumes single caller (groupby_aggregate). '",
    "        f'New caller: verify sample is bounded or add check_state.'",
    "    )",
    "",
    "",
    "def test_estimate_k_sample_whitelist_rationale_updated():",
    "    \"\"\"Stale 'to_pandas' rationale replaced with Arrow-native.\"\"\"",
    "    for key in ('cadw/ops/groupby.py:_estimate_k_sample',",
    "                'cadw/ops/dedup.py:_estimate_k_sample'):",
    "        assert key in WHITELISTED_FUNCS",
    "        rationale = WHITELISTED_FUNCS[key]",
    "        assert 'dictionary_encode' in rationale or 'Arrow-native' in rationale, (",
    "            f'{key}: rationale should mention Arrow-native / dictionary_encode'",
    "        )",
    "        assert 'to_pandas' not in rationale, (",
    "            f'{key}: stale to_pandas rationale (v0.9.6c1 removed pandas)'",
    "        )",
]


def add_tests():
    print("\n=== add whitelist meta-guard tests ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if "test_detect_heavy_keys_in_whitelist" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(NEW_TESTS))
    pu.assert_compiles(p)
    print("  OK")


# ============ 3. verify ============

def verify_state():
    print("\n=== verify: v0.9.12-fix3 + regression ===")
    pu.verify_anchors([
        # fix3
        ("tests/test_arrow_call_memory_guard.py",
         '"cadw/ops/groupby.py:_detect_heavy_keys":'),
        ("tests/test_arrow_call_memory_guard.py", "caller-bounded"),
        ("tests/test_arrow_call_memory_guard.py",
         "test_detect_heavy_keys_in_whitelist"),
        ("tests/test_arrow_call_memory_guard.py",
         "test_detect_heavy_keys_single_caller"),
        ("tests/test_arrow_call_memory_guard.py",
         "test_estimate_k_sample_whitelist_rationale_updated"),
        # fix2 regression
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("cadw/ops/groupby.py", "pc.fill_null(mask, pa.scalar(False))"),
        # F12-1..4 regression
        ("cadw/ops/topk.py", "v0.9.12 (F12-1)"),
        ("cadw/patch_utils.py", "newline=\"\""),
        ("cadw/patch_utils.py", "v0.9.12 (F12-3, §13.4/211)"),
        ("cadw/ops/groupby.py", "v0.9.12 (F12-4, PF1)"),
        # v0.9.11/v0.9.10/v0.9.9 regression
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/dataframe.py", "post-flush widen needed"),
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
    ])
    print("  all anchors present")


# ============ 4. dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.12-fix3 via subprocess ===")

    def _verify_all():
        import ast as _ast
        import inspect
        from cadw.ops import groupby as gb_mod
        from tests.test_arrow_call_memory_guard import WHITELISTED_FUNCS

        # Whitelist entry present
        key = "cadw/ops/groupby.py:_detect_heavy_keys"
        assert key in WHITELISTED_FUNCS, "whitelist entry missing"
        rationale = WHITELISTED_FUNCS[key]
        assert "groupby_aggregate" in rationale or "caller" in rationale.lower()

        # Stale rationale fixed
        for k in (
            "cadw/ops/groupby.py:_estimate_k_sample",
            "cadw/ops/dedup.py:_estimate_k_sample",
        ):
            assert k in WHITELISTED_FUNCS
            r = WHITELISTED_FUNCS[k]
            assert "to_pandas" not in r, f"{k}: still has stale to_pandas"
            assert "dictionary_encode" in r or "Arrow-native" in r

        # Caller count = 1
        src_gb = inspect.getsource(gb_mod)
        tree = _ast.parse(src_gb)
        call_sites = [
            n.lineno for n in _ast.walk(tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Name)
            and n.func.id == "_detect_heavy_keys"
        ]
        assert len(call_sites) == 1, (
            f"expected 1 caller, found {len(call_sites)} at {call_sites}"
        )

        print("  whitelist entry + rationale OK")
        print("  stale _estimate_k_sample rationale fixed (groupby.py + dedup.py)")
        print("  _detect_heavy_keys single caller (groupby_aggregate) OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ 5. pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_mustfix_skew.py",
         "tests/test_groupby_v2.py",
         "tests/test_topk_arrow_native.py",
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


# ============ 6. main ============

def main():
    print("=== v0.9.12-fix3 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_whitelist()
    add_tests()
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.12-fix3: complete v0.9.12 (whitelist _detect_heavy_keys "
        "+ fix stale rationale). v0.9.12-fix2 green, fast pytest "
        "failed test_no_unguarded_high_risk_arrow_calls: F12-4 made "
        "_detect_heavy_keys Arrow-native (pa.Table.from_batches + "
        "group_by), no explicit check_state — AST guard correctly "
        "flags §3.11 violation. Fix: whitelist with rationale "
        "(caller-bounded sample: groupby_aggregate sample_cap 10k "
        "rows / 0.3×avail, skip_skew if > 0.5×avail). Precedent: "
        "_accumulate_heavy (\"heavy keys ≤ 20 rows\") and "
        "_estimate_k_sample (\"first batch only\") both whitelisted. "
        "Also fixed stale _estimate_k_sample rationale "
        "(groupby.py + dedup.py): said \"to_pandas on ≤sample rows\" "
        "but v0.9.6c1 replaced pandas path with dictionary_encode. "
        "§13.4/219 (new): whitelist with caller-bounded rationale "
        "requires caller-count guard — if new caller passes "
        "unbounded input, whitelist is a lie. Meta-test "
        "test_detect_heavy_keys_single_caller (AST: exactly 1 call "
        "site in groupby.py). New tests: "
        "test_detect_heavy_keys_in_whitelist (entry + rationale), "
        "test_detect_heavy_keys_single_caller (AST caller count), "
        "test_estimate_k_sample_whitelist_rationale_updated "
        "(no stale to_pandas, mentions dictionary_encode). "
        "v0.9.12 COMPLETE: F12-1 topk incremental acc_bytes; "
        "F12-2 newline=\"\" + docstring; F12-4 Arrow-native "
        "_detect_heavy_keys; fix2 _compute_heavy_mask fill_null "
        "(§13.4/218, silent data-loss 3 months); fix3 whitelist + "
        "stale rationale. Deferred: A1 (dataframe split), T2.2/"
        "T2.5/T2.6/T2.7 (tooling), U3, N6, R3, quantile stream loop.",
        self_path="scripts/patch_v0912_fix3.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.12-fix3 applied. v0.9.12 COMPLETE.")


if __name__ == "__main__":
    main()
