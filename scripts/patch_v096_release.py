#!/usr/bin/env python3
# v0.9.6 release: CHANGELOG [Unreleased] -> [0.9.6], tag, verify.
#
# Milestone v0.9.6:
#   - pandas optional (import cadw does not pull pandas)
#   - own XXH64 hash (no pandas in shuffle)
#   - CADW-native _per_group_apply (scalar|dict|CadwDataFrame)
#   - streaming PivotOp (R6: 26.8 MB -> ~10 MB, 100k×20 passes)
#   - _estimate_k_sample Arrow-native
#   - scaffold v1: 4 helpers (assert_no_call_in_source,
#     assert_call_order, is_patched, run_smoke_inline)
#
# Run: uv run python scripts/patch_v096_release.py

from __future__ import annotations
import os, re, subprocess, sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    ensure_unreleased_header, append_unreleased_entry,
)


def sh(cmd):
    print("  $ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:15]:
            print("    " + line)
    if r.returncode != 0:
        print("    STDERR: " + r.stderr.strip()[-1200:])
        sys.exit(r.returncode)
    return r


V096_ENTRY = """### v0.9.6 — pandas removal + own XXH64 hash + streaming PivotOp + scaffold

**Milestone**: `import cadw` no longer pulls pandas. Core (`ops/`,
ML, pivot) is pandas-free; pandas is opt-in via
`cadw-framework[pandas]` for `to_batches(format="pandas")`.

#### Added

- **`CadwDataFrame.from_table(pa.Table)`** — zero-copy in-memory df;
  no disk I/O. Backed by `_InMemoryDataset` (read-only `Dataset`
  subclass).
- **`_InMemoryDataset(pa.Table)`** — `iter_batches` via
  `table.to_batches()`; `add_column` raises `NotImplementedError`.
- **`_ColumnProxy` aggregators** — `.mean()`, `.sum()`, `.min()`,
  `.max()`, `.std(ddof=1)`, `.count()` via existing `ReduceOp` (O(1)
  state, no pandas).
- **`_ColumnProxy` materializers** — `.to_list()`, `.to_numpy()`,
  `.to_arrow()` with `check_state` before.
- **`CadwDataFrame.__len__`** — same as `.count()`.
- **`CadwDataFrame.sort_values` in-memory fast path** —
  `pa.Table.sort_by` directly (avoids 1–5 ms `external_sort` per
  group; critical for per-group apply on 100k groups).
- **`cadw/ops/pivot.py::pivot_to_disk`** — streaming Arrow-native
  pivot (R6). Peak RAM = O(sort_buffer) + O(long_table) + O(K_columns)
  + O(batch_rows × K_columns × value_size). Independent of K_index.
  Replaces `to_pandas().pivot()` (26.8 MB → ~10 MB for 200k/300×300;
  100k×20 cols passes under 50 MB, previously fail-fast on dense
  estimate).
- **`cadw/core/_k_est.py::estimate_k_sample_arrow`** — Arrow-native
  K estimate (`pc.dictionary_encode`); used by groupby + dedup.
- **`cadw/patch_utils.py`** scaffold v1 (v0.9.6d):
  - `assert_no_call_in_source(src, *names)` — AST, no `.foo()` Call
    nodes; comments/docstrings invisible (§13.4/183).
  - `assert_call_order(src, first, second)` — AST Call ordering
    (§13.4/185).
  - `is_patched(fn, *markers)` — `inspect.getsource(fn)` scoped
    idempotency (§13.4/182).
  - `run_smoke_inline(script)` — `uv run python -c` (cwd on
    `sys.path[0]`; §13.4/184).

#### Changed

- **`shuffle.py` hash: own XXH64-based** (v0.9.6a) — replaces
  `_pandas_hash_column` + `_python_hash_column` with
  `_vectorized_hash_numeric` (numpy element-wise over uint64 bitcast
  for int/uint/bool/float/timestamp/date/time) +
  `_string_hash_column` (per-element `xxhash.xxh64`, no
  repr/encode wrapper). No `import pandas`. Cross-batch consistency,
  null sentinel, splitmix64 avalanche preserved.
- **`groupby.py::_estimate_k_sample`** — Arrow-native union of
  per-batch dictionaries (was `to_pandas().drop_duplicates()`).
- **`dedup.py::_estimate_k_sample`** — delegates to
  `estimate_k_sample_arrow` (was pandas).
- **`_per_group_apply` — BREAKING** for pd.Series/pd.DataFrame users.
  `fn(CadwDataFrame) → scalar | dict[str, scalar] | CadwDataFrame`.
  No pandas inside `apply`.
- **`pivot_table`** — delegates to streaming `pivot_to_disk`;
  behaviour preserved (same values/column-order/row-order); 100k ×
  20 cols passes where dense failed.
- **`pyproject.toml`**: `pandas` removed from `dependencies`;
  `[project.optional-dependencies].pandas = ["pandas>=3.0"]`;
  `[dependency-groups].dev += pandas` (test reference helpers).
- **`_convert_batch(format="pandas")`** — friendly `ImportError`
  with install hint (`pip install 'cadw-framework[pandas]'`).

#### Fixed

- `_estimate_k_sample` — no `to_pandas()` (dedup, groupby).
- `pivot_table` — no `to_pandas()` / `pd.pivot()` (R6).
- `_ColumnProxy` — aggregators work natively.
- `sort_values` — fast path for in-memory datasets.
- `test_pivot_too_large_raises` — match="pivot" (pivot.* names).
- `test_pivot_table_has_check_state_before_to_pandas` — replaced
  with two-level contract tests (`delegates_to_streaming` +
  `pivot_to_disk_guarded_before_output`).

#### Removed

- `_pandas_hash_column`, `_python_hash_column` (shuffle.py).
- `tests/test_shuffle_fallback_warning.py`.
- `pd.Series` / `pd.DataFrame` return contract in `_per_group_apply`.
"""


def add_entry_and_bump():
    print("\n=== CHANGELOG: add v0.9.6 entry + bump ===")
    p = Path("CHANGELOG.md")
    text = p.read_text()
    text = ensure_unreleased_header(text)
    text = append_unreleased_entry(text, V096_ENTRY)
    # Bump [Unreleased] -> [0.9.6]
    version = "0.9.6"
    release_date = date.today().isoformat()
    pat = re.compile(r"^## \[Unreleased\]\s*\n", re.MULTILINE)
    m = pat.search(text)
    if not m:
        raise SystemExit("[Unreleased] missing after append")
    before = text[:m.start()]
    after = text[m.end():]
    block = "## [Unreleased]\n\n## [" + version + "] — " + release_date + "\n"
    if after and not after.startswith("\n"):
        after = "\n" + after
    new_text = before + block + after
    p.write_text(new_text)
    versions = re.findall(r"^## \[(\d[^\]]*)\]", new_text, re.MULTILINE)
    if not versions or versions[0] != version:
        raise SystemExit(f"bump sanity failed: {versions[:3]}")
    print(f"  bumped to [{version}]; top 3: {versions[:3]}")


def commit_release():
    print("\n=== commit release ===")
    sh(["git", "add", "CHANGELOG.md"])
    sh(["git", "commit", "-m", "v0.9.6: release"])


def tag_release():
    print("\n=== tag -a v0.9.6 ===")
    r = subprocess.run(["git", "tag", "--list", "v0.9.6"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        raise SystemExit("tag v0.9.6 already exists")
    msg = (
        "v0.9.6: milestone — pandas optional. Milestone features:\n"
        "  - own XXH64 hash (no pandas in shuffle)\n"
        "  - CADW-native _per_group_apply\n"
        "  - streaming PivotOp (R6: 26.8 -> ~10 MB)\n"
        "  - _estimate_k_sample Arrow-native\n"
        "  - in-memory CadwDataFrame.from_table\n"
        "  - scaffold v1: 4 patch_utils helpers\n"
        "  - pyproject: pandas -> optional-dependencies + dev-group\n"
        "  - `import cadw` does not pull pandas"
    )
    sh(["git", "tag", "-a", "v0.9.6", "-m", msg])


def verify():
    print("\n=== test_changelog post-tag ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** CHANGELOG GUARD FAILED ***")
        sys.exit(1)


def main():
    print("=== v0.9.6 release ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    branch = r.stdout.strip()
    print("  branch: " + branch)
    if branch != "master":
        raise SystemExit("must be on master")

    add_entry_and_bump()
    commit_release()
    tag_release()
    verify()

    print("\n==> v0.9.6 released locally.")
    print("\nNEXT (manual):")
    print("  git --no-pager log --oneline -3")
    print("  git push origin master")
    print("  git push origin v0.9.6")


if __name__ == "__main__":
    main()