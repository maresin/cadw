#!/usr/bin/env python3
# v0.9.12-fix1 — precise AST negative check for as_py. v0.9.12 COMPLETE.
#
# v0.9.12 dogfood failed on `assert not as_py_calls` — but the new
# _detect_heavy_keys legitimately contains ONE .as_py():
#   total = int(pc.sum(counts.column(count_col)).as_py() or 0)
# That's an aggregate scalar extract (Call.as_py()), not per-row
# indexing (b.column(c)[i].as_py()).
#
# §13.4/217 (new): negative AST guards must target PRECISE
# anti-pattern, not "any Call with this method name". Original
# finding was per-row .as_py() in `for i in range(num_rows)` loop.
# Correct check: .as_py() Call where receiver is a Subscript
# (X[i].as_py()) = per-row anti-pattern. Call.as_py() = aggregate
# extract = fine. Same class as §13.4/213 (negative substring must
# be AST, now precise AST).
#
# State on disk (from v0.9.12, all applied + verify_state green):
#   F12-1 topk incremental acc_bytes
#   F12-2 newline=""
#   F12-3 docstring compile-before-write
#   F12-4 _detect_heavy_keys Arrow-native + 4 tests
#   commit never reached (dogfood died)
#
# fix1: replace dogfood negative check with Subscript-receiver scan.
# v0.9.12 COMPLETE.
#
# Run: uv run python scripts/patch_v0912_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


def verify_state():
    print("\n=== verify: v0.9.12 + regression ===")
    pu.verify_anchors([
        ("cadw/ops/topk.py", "v0.9.12 (F12-1)"),
        ("cadw/ops/topk.py", "nonlocal row_id_offset, acc_bytes"),
        ("cadw/patch_utils.py", "v0.9.12 (F12-2)"),
        ("cadw/patch_utils.py", "newline=\"\""),
        ("cadw/patch_utils.py", "v0.9.12 (F12-3, §13.4/211)"),
        ("cadw/ops/groupby.py", "v0.9.12 (F12-4, PF1)"),
        ("cadw/ops/groupby.py", "pa.concat_tables(tables"),
        # v0.9.11 regression
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/dataframe.py", "post-flush widen needed"),
        # v0.9.10 regression
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
        # v0.9.9 regression
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
        # Tests
        ("tests/test_mustfix_skew.py",
         "test_detect_heavy_keys_arrow_native_correctness"),
    ])
    print("  all anchors present")


def dogfood():
    print("\n=== dogfood: v0.9.12 via subprocess (precise as_py check) ===")

    def _verify_all():
        import ast as _ast
        import inspect
        import textwrap as _tw

        # F12-1: no O(P²) sum, incremental acc_bytes
        from cadw.ops import topk as tk
        src_topk = inspect.getsource(tk.topk_to_disk)
        assert "sum(tb.nbytes for tb in per_batch)" not in src_topk
        assert "acc_bytes += top.nbytes" in src_topk

        # F12-2: newline=""
        from cadw.patch_utils import apply_or_fail
        src_aof = inspect.getsource(apply_or_fail)
        assert 'newline=""' in src_aof

        # F12-3: docstring notes compile-before-write
        assert "compile-before-write" in src_aof
        assert "Never reorder" in (apply_or_fail.__doc__ or "")

        # F12-4: precise AST — no per-row .as_py() (X[i].as_py()),
        # but aggregate .as_py() (pc.sum(...).as_py()) is fine.
        from cadw.ops.groupby import _detect_heavy_keys
        src_dhk = _tw.dedent(inspect.getsource(_detect_heavy_keys))
        tree = _ast.parse(src_dhk)
        per_row_as_py: list[int] = []
        aggregate_as_py: list[int] = []
        for n in _ast.walk(tree):
            if not isinstance(n, _ast.Call):
                continue
            f = n.func
            if not (isinstance(f, _ast.Attribute) and f.attr == "as_py"):
                continue
            if isinstance(f.value, _ast.Subscript):
                # X[i].as_py() — per-row anti-pattern (F12-4 target)
                per_row_as_py.append(n.lineno)
            else:
                # pc.sum(...).as_py() or arr.as_py() — aggregate, fine
                aggregate_as_py.append(n.lineno)
        assert not per_row_as_py, (
            f"F12-4 regression: per-row .as_py() at line(s) "
            f"{per_row_as_py} (anti-pattern: X[i].as_py() inside "
            f"for i in range(num_rows))"
        )
        # Sanity: aggregate as_py present (total = int(pc.sum(...).as_py()))
        assert aggregate_as_py, (
            "F12-4: expected one aggregate .as_py() for total count; "
            "if absent, the function may not compute total at all"
        )

        # group_by Call present
        group_by_calls = [
            n for n in _ast.walk(tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "group_by"
        ]
        assert group_by_calls, "F12-4: group_by() not called"

        # No `for i in range(<expr containing num_rows>)` loop
        # (would be the per-row pattern even without .as_py, e.g.
        # via `[i]` indexing into a pyarrow list)
        for n in _ast.walk(tree):
            if not isinstance(n, _ast.For):
                continue
            it = n.iter
            if not (isinstance(it, _ast.Call)
                    and isinstance(it.func, _ast.Name)
                    and it.func.id == "range"):
                continue
            for arg in it.args:
                if any(
                    isinstance(sub, _ast.Attribute)
                    and sub.attr == "num_rows"
                    for sub in _ast.walk(arg)
                ):
                    raise AssertionError(
                        f"F12-4 regression: 'for ... in range(...num_rows)' "
                        f"loop at line {n.lineno}"
                    )

        print("  F12-1 (no O(P²) sum, acc_bytes +=) OK")
        print("  F12-2 (newline=\"\") OK")
        print("  F12-3 (docstring) OK")
        print(
            f"  F12-4 (no per-row .as_py, 1 aggregate .as_py at "
            f"{aggregate_as_py}, group_by present) OK"
        )

    pu.run_verify_in_subprocess(_verify_all)


def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_topk_arrow_native.py",
         "tests/test_mustfix_skew.py",
         "tests/test_groupby_v2.py",
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


def main():
    print("=== v0.9.12-fix1 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.12-fix1: complete v0.9.12 (precise AST as_py check). "
        "v0.9.12 dogfood failed on `assert not as_py_calls` — but "
        "new _detect_heavy_keys legitimately contains ONE .as_py(): "
        "'total = int(pc.sum(counts.column(count_col)).as_py() or 0)' "
        "— aggregate scalar extract (Call.as_py()), not per-row "
        "indexing. §13.4/217 (new): negative AST guards must target "
        "PRECISE anti-pattern, not 'any Call with this method name'. "
        "Original F12-4 finding: per-row .as_py() in `for i in "
        "range(b.num_rows)` loop. Correct check: fail only on "
        "'X[i].as_py()' where receiver is Subscript (per-row); "
        "allow 'Call(...).as_py()' (aggregate). Same class as "
        "§13.4/213 (negative substring must be AST) but now the AST "
        "walk itself is precise. fix1 dogfood: (a) fail on Subscript-"
        "receiver .as_py() (per-row), (b) require >=1 aggregate "
        ".as_py() (positive sanity), (c) no 'for i in range(... "
        "num_rows)' loop at all (catches non-.as_py per-row too). "
        "F12-4 Arrow-native _detect_heavy_keys is CORRECT on disk "
        "(verified: pa.concat_tables + group_by().aggregate + "
        "pc.greater_equal filter; 4 correctness tests: 900/100 split, "
        "uniform no-heavy, null group, below _HEAVY_ABSOLUTE_MIN). "
        "v0.9.12 COMPLETE: F12-1 topk incremental acc_bytes (O(P²) "
        "→ O(P)); F12-2 newline=\"\" cross-platform LF; F12-3 "
        "docstring compile-before-write §13.4/211; F12-4 Arrow-"
        "native heavy detection ~10x faster. Pingis round 2 inline "
        "review: PF2 false positive (accum bounded, not O(N)); PF3 "
        "bounded by K_columns, §11.5 documented. Deferred: A1 "
        "(dataframe split 5404 lines), T2.2/T2.5/T2.6/T2.7 (tooling), "
        "U3 (pyproject version), N6, R3, quantile stream loop. "
        "Next: v0.9.13 — quantile Arrow-native stream (F12-5) or A1 "
        "split, whichever first.",
        self_path="scripts/patch_v0912_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.12-fix1 applied. v0.9.12 COMPLETE.")


if __name__ == "__main__":
    main()
