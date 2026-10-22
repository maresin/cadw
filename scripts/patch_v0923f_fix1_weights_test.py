#!/usr/bin/env python3
# v0.9.23f-fix1 — 2 dogfood/test bugs + 1 SyntaxWarning in v0.9.23f.
#
# All three are test bugs, NOT production bugs:
#   (1) dogfood used df.sample(n=110, weights="w") on N=100 with
#       replace=False -> ValueError (n>N, pandas-parity, production
#       correct). Intent was 10:1 ratio test; n must be <= N when
#       replace=False. Fix: n=50 from N=100 (50 low-w + 50 high-w)
#       -> ~45 low, ~5 high; assert low > high*3.
#   (2) test_sample_weights_proportional: same n=110 -> n=50.
#   (3) SyntaxWarning: match="weights \+ replace" — "\+" invalid
#       escape in non-raw string -> r"weights \+ replace".
#
# Production sample() correct: n>N with replace=False raises ValueError
# (matches pandas). weights+replace raises NotImplementedError before
# n>N check (ordered correctly).
#
# Run: uv run python scripts/patch_v0923f_fix1_weights_test.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess,
)


# ============ 1. Fix test_sample_weights_proportional (n=110 -> 50) ==

def fix_proportional_test():
    print("\n=== fix test_sample_weights_proportional (n=110->50) ===")
    p = Path("tests/test_sample.py")
    s = p.read_text()
    if "test_sample_weights_proportional" not in s:
        print("  test not present — skip")
        return
    if "s = df.sample(n=50, weights=\"w\", random_state=42)" in s:
        print("  already fixed — skip")
        return

    old = '''def test_sample_weights_proportional(tmp_path):
    """Weight ratio 10:1 should give ~10:1 sample ratio."""
    df = make_df(tmp_path, {
        "v": list(range(100)),
        "w": [10.0] * 50 + [1.0] * 50,
    })
    s = df.sample(n=110, weights="w", random_state=42)
    got = read_col(s, tmp_path, "v")
    low = sum(1 for v in got if v < 50)
    high = sum(1 for v in got if v >= 50)
    # Expected ratio 10:1; allow some slack
    assert low > high * 3, f"expected low>>high, got low={low} high={high}"'''

    new = '''def test_sample_weights_proportional(tmp_path):
    """Weight ratio 10:1 should give ~10:1 sample ratio.

    v0.9.23f-fix1: prior used n=110 > N=100 (replace=False), which
    correctly raised ValueError (pandas-parity). Fix: n=50 from N=100
    (50 low-w + 50 high-w). Expected ~45 low, ~5 high (ratio 10:1).
    """
    df = make_df(tmp_path, {
        "v": list(range(100)),
        "w": [10.0] * 50 + [1.0] * 50,
    })
    s = df.sample(n=50, weights="w", random_state=42)
    got = read_col(s, tmp_path, "v")
    low = sum(1 for v in got if v < 50)
    high = sum(1 for v in got if v >= 50)
    # Expected ~45 low, ~5 high; require low > 3*high (threshold safe)
    assert low > high * 3, f"expected low>>high, got low={low} high={high}"'''

    if old not in s:
        raise SystemExit(
            "test_sample_weights_proportional anchor not found"
        )
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Fix SyntaxWarning: r"weights \+ replace" ============

def fix_regex_raw_string():
    print("\n=== fix regex raw string (SyntaxWarning) ===")
    p = Path("tests/test_sample.py")
    s = p.read_text()
    # After fix1, the only remaining "\+" is in the regex
    if 'r"weights \\+ replace"' in s:
        print("  already fixed — skip")
        return
    # Locate the offending line
    old = 'with pytest.raises(NotImplementedError, match="weights \\+ replace"):'
    new = (
        'with pytest.raises('
        'NotImplementedError, match=r"weights \\+ replace"'
        '):'
    )
    if old not in s:
        # Try alternate spacing
        old = 'with pytest.raises(NotImplementedError, match="weights \\+ replace"):'
        if old not in s:
            # Fallback: raw string replace
            if 'match="weights \\+ replace"' in s:
                s = s.replace(
                    'match="weights \\+ replace"',
                    'match=r"weights \\+ replace"',
                    1,
                )
                p.write_text(s)
                assert_compiles(p)
                print("  OK (fallback)")
                return
            raise SystemExit("regex anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. Dogfood (fix n=110 -> 50) ============

def dogfood():
    print("\n=== dogfood: sample fixed weights ratio test ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        # weights: 10:1 ratio; n=50 from N=100 (replace=False, n<=N OK)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "v": list(range(100)),
                "w": [10.0] * 50 + [1.0] * 50,
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=10,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            s = df.sample(n=50, weights="w", random_state=42)
            out = td / "o.parquet"
            s.write_parquet(out)
            got = pq_.read_table(out).column("v").to_pylist()
            assert len(got) == 50
            low = sum(1 for v in got if v < 50)
            high = sum(1 for v in got if v >= 50)
            # Expected ratio ~10:1 -> low ~45, high ~5; safe threshold
            assert low > high * 3, (
                f"10:1 weights -> expected low>>high, got low={low} "
                f"high={high}"
            )

        # Sanity: n>N replace=False raises (pandas-parity)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"v": list(range(10))})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            try:
                df.sample(n=100)  # no replace, n>N
            except ValueError as e:
                assert "n=100 > N=10" in str(e)
            else:
                raise AssertionError(
                    "n>N replace=False must raise (pandas-parity)"
                )

        print("  weighted 10:1 ratio + n>N raises ok")

    run_verify_in_subprocess(_verify)


# ============ 4. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_sample.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=300,
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


# ============ 5. main ============

def main():
    print("=== v0.9.23f-fix1: 2 test bugs + SyntaxWarning ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_proportional_test()
    fix_regex_raw_string()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23f: CadwDataFrame.sample() (AUDIT_FULL_v0922 item "
        "7.A.2, Top-10 gap #1). ONE method, three modes: "
        "df.sample(n=None, frac=None, replace=False, weights=None, "
        "random_state=None). "
        "Algorithms (streaming, bounded memory): "
        "(a) replace=False, no weights: Algorithm R reservoir, "
        "vectorized per-batch decisions (probs=n/(i+1)); expected "
        "replacements ~n*ln(N/n); peak O(n). "
        "(b) replace=False, weights=col: Efraimidis-Spirakis A-Res "
        "(key=u^(1/w), top-n min-heap). Peak O(n), time O(N log n) — "
        "documented N<1e7 limit. Null/<=0 weights skipped (raises if "
        "ALL invalid). "
        "(c) replace=True: vectorized rng.integers(0, N, n) + "
        "iloc[list] (n>N allowed). "
        "Edge: n=0 -> empty; n=N no-replace -> shuffle (pandas "
        "parity); n>N no-replace -> ValueError (pandas parity); "
        "frac in (0,1]; frac=0 -> empty; frac rounds to "
        "max(1, round(N*frac)). "
        "Determinism: random_state=None -> seed=42 (matches KFold). "
        "weights+replace -> NotImplementedError (needs O(N) full-"
        "weight materialize; honest). "
        "Return: CadwDataFrame (chainable; row order = sample order). "
        "MemoryGuard: check_state(4*n*row_bytes_est) before iloc; "
        "row_bytes_est via reestimate_from_batch (v0.9.8b R2). "
        "fix1 (2 test bugs + 1 SyntaxWarning, all caught in "
        "dogfood/pytest, NOT production): "
        "(1) dogfood used n=110 from N=100 replace=False -> "
        "ValueError (pandas-parity, production correct); intent was "
        "10:1 ratio. Fix: n=50 from N=100 -> ~45 low, ~5 high. "
        "(2) test_sample_weights_proportional: same n=110 -> n=50. "
        "(3) SyntaxWarning: match='weights \\+ replace' ('\\+' invalid "
        "escape in non-raw string) -> r'weights \\+ replace'. "
        "+tests/test_sample.py (34 tests: basic n, n=0, n>N error, "
        "n=N shuffle, determinism, no dupes, within range, frac "
        "0.5/1.0/tiny/0/invalid, replace=True basic/n>N/dupes, "
        "validation neither/both/negative/replace-not-bool, weights "
        "basic/proportional/nulls/all-zero error/missing/non-numeric/"
        "weights+replace, filtered, memory guard, empty df, 4 AST). "
        "Fast: 2010 -> ~2044 passed.",
        self_path="scripts/patch_v0923f_fix1_weights_test.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23f-fix1 applied. v0.9.23f complete (7.A.2).")


if __name__ == "__main__":
    main()