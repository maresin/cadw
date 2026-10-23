#!/usr/bin/env python3
# v0.9.25n-corrected — Phase C close-out, verify assumption fixed.
#
# PREVIOUS v0.9.25n attempt aborted in verify_state:
#   DECISIONS.md: expected §13.4/270+, got 248.
#
# ROOT: my assumption "commit messages reference §13.4/270+ → DECISIONS.md
# is at 270+" was wrong. Phase C commit messages §13.4/262–279 are
# meta-lesson NUMBERS in commit message text, but those lessons were
# NEVER appended to DECISIONS.md. DECISIONS.md canonical is at 248
# (v0.9.22b era); gap 249–279 = ~30 lessons of docs debt.
#
# This is the same speculative-assumption anti-pattern §13.4/264
# (REPL-first) prevents — but applied to file STATE instead of library
# BEHAVIOR. My verify_state did read the actual state (regex found 248)
# but then asserted a speculative minimum (>= 270) based on commit
# messages. §13.4/280 (NEW): file-state assertions need the same
# inspect-actual-first discipline as library assumptions — commit
# messages are NOT authoritative for file contents.
#
# FIX:
#   - verify_state accepts whatever last §13.4 number is present;
#     only aborts on obviously-broken state (< 200 = too old, likely
#     wrong file; > 400 = drift beyond expected).
#   - Retrospective parametrized on new_num = last + 1 (will be 249
#     given actual state).
#   - Retrospective explicitly notes docs debt: §13.4/249–{new_num-1}
#     not appended; Phase C lessons live in commit messages only.
#   - CHANGELOG paragraph "lessons consolidated in DECISIONS.md
#     §13.4/262–279" corrected to "lessons recorded in Phase C
#     commit messages (v0.9.25g–l); canonical DECISIONS.md backfill
#     deferred to v0.9.26 docs-debt sweep".
#
# §13.4/278 preserved: version bump verified via importlib.reload.
# Idempotency: each section checks own marker; re-run → all no-op.
# §13.4/264 preserved: REPL checks (env pyarrow) + file reads done
# actual-first.
#
# Run: uv run python scripts/patch_v0925n_phase_c_close_corrected.py

from __future__ import annotations
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, commit_with_guards,
    check_no_invalid_escape,
)


TARGET_VERSION = "0.9.25"


# ============ 0. verify state (lenient — accept actual) ============

def verify_state():
    print("\n=== verify: comprehensive pre-close-out inspection "
          "(lenient on actual file state) ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25l" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25l; got {log[:200]!r}"
        )
    if "v0.9.25n" in log:
        raise SystemExit("v0.9.25n already committed — patch misplaced")
    print("  HEAD = v0.9.25l ✓")

    # --- CHANGELOG.md ---
    cl = Path("CHANGELOG.md").read_text()
    if "## [Unreleased]" not in cl:
        raise SystemExit("CHANGELOG.md: '## [Unreleased]' missing")
    if "### v0.9.24: time series toolkit" not in cl:
        raise SystemExit("CHANGELOG.md: '### v0.9.24: ...' missing")
    if "### v0.9.25:" in cl:
        raise SystemExit(
            "CHANGELOG.md: v0.9.25 section already present"
        )
    if "## [Unreleased]\n\n### v0.9.24:" not in cl:
        raise SystemExit(
            "CHANGELOG.md: exact anchor not found. Inspect file."
        )
    print("  CHANGELOG.md: anchor OK, v0.9.25 not yet present ✓")

    # --- DECISIONS.md ---
    dec = Path("DECISIONS.md").read_text()
    matches = re.findall(r'§13\.4/(\d+)', dec)
    if not matches:
        raise SystemExit("DECISIONS.md: no §13.4/NNN markers found")
    last_num = max(int(m) for m in matches)
    print(f"  DECISIONS.md: last §13.4 number = {last_num}")
    # §13.4/280: only abort on obviously-broken state, accept actual
    # number otherwise. Do NOT assert a speculative minimum based on
    # commit messages — commit messages are NOT DECISIONS.md content.
    if last_num < 200:
        raise SystemExit(
            f"DECISIONS.md: last §13.4 = {last_num} < 200 — state "
            f"is unexpectedly old. Inspect git history: this file "
            f"should have §13.4/245+ from v0.9.22 close-out. "
            f"Possible wrong repo state or corrupted file."
        )
    if last_num > 400:
        raise SystemExit(
            f"DECISIONS.md: last §13.4 = {last_num} > 400 — "
            f"unexpected drift beyond project scope. Inspect file "
            f"for accidental duplication or wrong repo."
        )
    new_num = last_num + 1
    print(f"  → will append §13.4/{new_num} (actual-based, not "
          f"commit-message-based)")

    # --- pyproject.toml version ---
    pp = Path("pyproject.toml").read_text()
    ppm = re.search(r'^version\s*=\s*"([^"]+)"', pp, re.M)
    if not ppm:
        raise SystemExit("pyproject.toml: 'version = \"...\"' not found")
    cur_ver = ppm.group(1)
    print(f"  pyproject.toml version = {cur_ver}")

    # --- cadw/__init__.py __version__ ---
    init_text = Path("cadw/__init__.py").read_text()
    init_m = re.search(
        r'__version__\s*=\s*["\']([^"\']+)["\']', init_text,
    )
    init_ver = init_m.group(1) if init_m else "MISSING"
    print(f"  cadw.__version__ = {init_ver}")
    if init_m and init_ver != cur_ver:
        print(
            f"  ⚠ version drift: pyproject={cur_ver} "
            f"vs __init__={init_ver} (will align both to "
            f"{TARGET_VERSION})"
        )

    # --- roadmap.json structure ---
    try:
        rm_text = Path("docs/roadmap.json").read_text()
        rm_data = json.loads(rm_text)
        rm_keys = (
            list(rm_data.keys())
            if isinstance(rm_data, dict)
            else f"(not dict, type={type(rm_data).__name__})"
        )
        print(f"  roadmap.json top keys: {rm_keys}")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  roadmap.json: could not parse ({e}) — will skip")

    import pyarrow as pa
    print(f"  pyarrow = {pa.__version__} ✓")

    return {
        "last_13_4": last_num,
        "new_13_4": new_num,
        "pyproject_version": cur_ver,
        "init_version": init_ver,
    }


# ============ 1. CHANGELOG v0.9.25 section (corrected text) ============

CHANGELOG_ANCHOR = """## [Unreleased]

### v0.9.24: time series toolkit + temporal hierarchical forecasting"""


CHANGELOG_NEW = """## [Unreleased]

### v0.9.25: Phase C — Pandas P0 API (20 features across 6 batches)

**Milestone: Phase C closed.** All 20 Pandas P0 features implemented.
Zero production bugs after C-1 (v0.9.25g); all downstream churn was
test-tooling. Phase C-5 (`dropna`/`fillna`) was already shipped in
v0.9.23g — `hasattr` verify (§13.4/279 in commit-message numbering,
§13.4/{NEW_NUM} in DECISIONS.md canonical) returned 0 mutations,
confirming the 20/20 target was met without duplicate work.

#### Added

**C-1 (v0.9.25g) — `isin` / `between` / `select_dtypes`**

- `_ColumnProxy.isin(values)` — null-safe membership via `pc.is_in`.
  `None in values` → null row matches (Kleene). Empty values → all
  False.
- `_ColumnProxy.between(left, right, *, inclusive="both")` —
  `pc.and_(ge/gt, le/lt)`. Null rows dropped (Kleene). `inclusive`
  ∈ {`both`, `neither`, `left`, `right`}.
- `CadwDataFrame.select_dtypes(include=, exclude=)` — category names
  (`"number"`, `"int"`, `"string"`, `"bool"`, `"datetime"`, …) and
  any `pa.type_for_alias`-resolvable type (`int64`, `Int64` case-
  insensitive, `string`, `timestamp[ns]`). Bracket syntax
  (`string[pyarrow]`) → ValueError.

**C-2 (v0.9.25h) — `idxmax` / `idxmin` / `pct_change`**

- `_ColumnProxy.idxmax()` / `.idxmin()` — index label of first
  max/min (nulls skipped, first occurrence wins). RangeIndex → int;
  Materialized → label; MultiIndex → tuple. All-null / empty filter
  → ValueError.
- `CadwDataFrame.pct_change(periods=1, columns=None)` — **pandas-
  parity**: `pc.divide` uses IEEE 754 (inf for x/0, nan for 0/0).
  Nulls propagate; first `periods` rows null. Output float64.

**C-3a (v0.9.25i) — `ffill` / `bfill` / `reindex`**

- `CadwDataFrame.ffill(columns=None)` — streaming per-batch
  `pc.fill_null_forward` + carry Array[1] for cross-batch leading-
  null fix. O(#cols) state, O(N) I/O.
- `CadwDataFrame.bfill(columns=None)` — reverse-sort (`__orig_rid`
  desc) + ffill-internal + re-sort (asc). 2× `external_sort`,
  O(N log N) I/O. Heavier than ffill — documented.
- `CadwDataFrame.reindex(labels)` — filtered-RangeIndex aware.
  Labels are original positionals (not filtered enumeration).
  Missing → null row via null-safe Arrow `take`. Bounded by
  `K × row_size ≤ state_limit`.

**C-3b (v0.9.25j) — `cut` / `qcut` / `duplicated`**

- `CadwDataFrame.cut(x, bins, labels=, right=)` — external_sort +
  `pc.bucketize` + bin edges. Returns int labels.
- `CadwDataFrame.qcut(x, q, labels=)` — quantile + cut combination.
- `CadwDataFrame.duplicated(subset=, keep='first'|'last'|False)` —
  row_id-aware mask via HashShuffle + groupby, preserves order.

**C-4a (v0.9.25k) — `str.slice` / `str.cat` / `str.count`**

- `_StrAccessor.slice(start=, stop=)` — negative indices supported
  (pandas parity).
- `_StrAccessor.cat(others=, sep=)` — string concatenation; null
  propagation per pandas.
- `_StrAccessor.count(pat)` — non-overlapping matches via
  `pc.count_substring`.

**C-4b (v0.9.25l) — `dt.strftime` / `dt.tz_convert` / `str.split`**

- `_DtAccessor.strftime(fmt)` — `pc.strftime`, naive + tz-aware.
- `_DtAccessor.tz_convert(tz)` — `pc.cast` (naive → UTC first via
  `pc.assume_timezone`).
- `_StrAccessor.split(pat, max_splits=-1)` — list<string> output via
  `pc.split_pattern`.

**C-5 (pre-existing since v0.9.23g) — `dropna` / `fillna`**

- `CadwDataFrame.dropna(how="any", subset=None)` — lazy, composes
  notnull() filters. `how='any'` drops rows with any subset null;
  `how='all'` drops rows where all subset nulls.
- `CadwDataFrame.fillna(value, columns=None)` — streaming per-batch
  `pc.fill_null`, scalar value type-coerced via `pa.scalar`.
  **Not reimplemented in Phase C** — `hasattr` verify revealed
  existing implementation with 12+ tests. 0 production changes in
  v0.9.25m.

#### Fixed (Phase C fix cycles)

- **§13.4/262 (commit-message):** `select_dtypes` type matching —
  `str(t) == s` failed on `str(pa.float32()) == "float"` (actual
  `"float"`) and `str(pa.float64()) == "double"` (actual `"double"`).
  Fix: `pa.type_for_alias(s)` first, fallback str-compare.
- **§13.4/263 (commit-message):** `select_dtypes` accepts `"Int64"`
  (case-insensitive via `type_for_alias`) but rejects
  `"string[pyarrow]"` (bracket syntax, unparseable).
- **§13.4/265 (commit-message):** v0.9.25h-fix1 — stray
  `assert math.isinf(vals[2])` copy-pasted from sibling test.
  Fix: `vals[2] == 3.0`.
- **§13.4/277 (commit-message):** v0.9.25l-fix1 —
  `test_tz_convert_non_temporal_raises` expected
  `"timestamp required"` but `.dt` accessor `__init__` gate fires
  first with `"expected timestamp or date"`. Test-only fix;
  production correct.
- **§13.4/278 (commit-message):** v0.9.25l-fix1-corrected —
  `verify_state` source-text marker `"def split(self"`
  false-negative (multi-line signature). Fix:
  `hasattr(imported_class, "method")` — runtime attribute presence.

#### Process — Phase C fix-cycle tally

| Batch | Attempts | Prod bugs | Notes |
|---|---|---|---|
| C-1 (`v0.9.25g`) | 4 | 1 (`type_alias`) | 2 test-overreach |
| C-2 (`v0.9.25h`) | 2 | 0 | REPL-first abort + fix1 stray assert |
| C-3a (`v0.9.25i`) | 1 | 0 | anchor miss (0 mut) + reindex-filtered pre-ship fix |
| C-3b (`v0.9.25j`) | 1 | 0 | — |
| C-4a (`v0.9.25k`) | 1 | 0 | — |
| C-4b (`v0.9.25l`) | 3 | 0 | REPL-first abort + fix1 + fix1-corrected |
| C-5 (pre-existing) | 0 | 0 | `hasattr` reveal → skip |
| **Total** | **12** | **1** | **92% test-tooling** |

**Key learnings (§13.4/262–279 in Phase C commit messages;
canonical `DECISIONS.md` backfill deferred to v0.9.26 docs-debt
sweep):**
- §13.4/264 — REPL-first (10–12 pre-mutation checks) saved 2 cycles
  in v0.9.25h and prevented wrong `pc.divide` semantics (IEEE 754
  inf/nan, not null-out).
- §13.4/266 — post-anchor re-verify each feature; caught latent
  reindex-filtered bug (v0.9.25i-corrected) pre-ship.
- §13.4/278 — `hasattr` beats source-text for presence checks.
- §13.4/279 — `hasattr` reveal ≠ rebuild; diagnostic-only response
  when feature already exists.

#### Version

- `pyproject.toml`: `{PYPROJECT_OLD}` → `0.9.25` (was 11 versions
  stale since v0.9.14 close-out).
- `cadw.__version__`: `{INIT_OLD}` → `0.9.25` (aligned).

#### Test counts

- ~2314 fast at v0.9.25f-a
- ~2488 fast at v0.9.25l (after Phase C)
- **No new tests in v0.9.25n** (docs-only close-out)

#### Docs debt (v0.9.26 backlog)

- **§13.4 numbering drift**: DECISIONS.md canonical is at
  §13.4/{NEW_NUM-1} (actual file state); Phase C lessons
  §13.4/262–279 exist only in commit messages. Backfill §13.4/249–
  279 into DECISIONS.md as formal appends — tracked as v0.9.26
  docs-debt sweep, not blocking Phase C close-out.


### v0.9.24: time series toolkit + temporal hierarchical forecasting"""


def update_changelog(new_num, pyproject_old, init_old):
    print("\n=== CHANGELOG.md: insert v0.9.25 section ===")
    if "### v0.9.25:" in Path("CHANGELOG.md").read_text():
        print("  already present — skip")
        return
    block = (CHANGELOG_NEW
             .replace("{NEW_NUM}", str(new_num))
             .replace("{NEW_NUM-1}", str(new_num - 1))
             .replace("{PYPROJECT_OLD}", pyproject_old)
             .replace("{INIT_OLD}", init_old))
    applied = apply_or_fail(
        "CHANGELOG.md", CHANGELOG_ANCHOR, block,
        label="changelog_v0925",
    )
    print("  applied" if applied else "  already applied — skip")


# ============ 2. DECISIONS §13.4 append (parametrized) ============

def update_decisions(new_num):
    print(f"\n=== DECISIONS.md: append §13.4/{new_num} ===")
    p = Path("DECISIONS.md")
    src = p.read_text()
    if "hasattr reveal ≠ rebuild (Phase C close)" in src:
        print("  Phase C close retrospective already present — skip")
        return
    block = f"""

#### §13.4/{new_num} — hasattr reveal ≠ rebuild (Phase C close)

*Also referenced as §13.4/279 in v0.9.25m-diagnostic commit
message (Phase C lessons are recorded in commit messages
§13.4/262–279; canonical DECISIONS.md backfill deferred to
v0.9.26 docs-debt sweep).*

**Context (v0.9.25m-diagnostic):** the "final batch" C-5 patch
(`dropna` + `fillna`) had `verify_state` abort with
`CadwDataFrame.dropna already present`. Both features shipped in
v0.9.23g (EDA-basics batch, went through 6 fix-cycles — highest
in the project). Diagnostic revealed:

- `dropna(how="any", subset=None)` — lazy notnull() filter-compose.
  API matches pandas parameter order (`how` before `subset`).
- `fillna(value, columns=None)` — streaming `pc.fill_null`.
  Scalar value only; no dict `{{col: val}}` variant. Dict is P2.
- **12 tests** already cover both: `test_eda_basics.py`
  (5 dropna + 7 fillna), `test_facade_transformers.py` (6 dropna),
  `test_pipeline.py` (2 dropna).

**Response:** diagnostic-only patch, 0 mutations. Do not rewrite
working hardened code to match imagined spec. Do not add P2
`fillna(dict)` to a 6-fix-cycle-hardened module for marginal
benefit — `df.fillna(0).fillna({{"s": "?"}})` composes today.

**§13.4/{new_num} rule:** when `hasattr` verify (§13.4/278) reveals
pre-existing implementation:

1. **Diagnostic-only** first — `inspect.getsource` + signature +
   tests + git log + CHANGELOG. No mutation.
2. **Accept existing API as-is** — actual signature may differ from
   speculative spec; matching actual is correct.
3. **Assess genuine gaps** — P2 dict-fillna is real gap; compose
   works today.
4. **Don't touch hardened code** — 6 fix-cycles vs 0-mutation abort
   is the cost signal.
5. **Test-only extension** is safe middle path if coverage thin
   (not needed here — 12 existing tests).

**Cost model:** C-5 (v0.9.23g) took 6 fix-cycles + ongoing coverage
vs v0.9.25m verify (0 attempts, 0 mutation). The 20/20 target was
already met — v0.9.25m is a verification pass, not a feature patch.

**Phase C close (v0.9.25n):** 20/20 Pandas P0 features. 12
fix-cycle attempts across 6 batches. 1 production bug total
(C-1 `type_alias`, §13.4/262). 92% churn in test-tooling.
Defense layers that emerged: §13.4/264 (REPL-first),
§13.4/266 (post-anchor re-verify), §13.4/278 (hasattr vs
source-text), §13.4/{new_num} (hasattr reveal ≠ rebuild).

#### §13.4/280 — file-state assertions need REPL-first too

**Context (v0.9.25n-corrected):** my first v0.9.25n attempt
aborted in `verify_state` with "expected §13.4/270+, got 248."
My assumption: "Phase C commit messages reference §13.4/262–279,
therefore DECISIONS.md is at 270+." **Wrong.** Those numbers were
only in commit message text; DECISIONS.md canonical was at 248
(v0.9.22b era). The gap 249–279 = ~30 lessons of docs debt.

**Root:** I applied §13.4/264 (REPL-first) to **library behavior**
(`pc.divide` semantics) but not to **file state**. The same
speculative-assumption anti-pattern, different domain.

**§13.4/280 rule:** file-state assertions in `verify_state` use
**actual inspection** (regex/regex-max/getsource), not inference
from commit messages, git log, or sibling files. If actual differs
from expectation → diagnostic, not assertion. Where "expected ≥ N"
is speculative, downgrade to "abort only if < lower_bound or
> upper_bound" (loose sanity, not target). Commit messages and
git log are NOT authoritative for file contents.

**Symmetric to §13.4/279:** hasattr reveal ≠ rebuild (state
revealed is what actually runs). §13.4/280: file-state reveal ≠
commit-message state (state read from file is what's actually
there). Both require **accept actual, don't infer**.
"""
    p.write_text(src.rstrip() + block + "\n")
    print(f"  appended §13.4/{new_num} + §13.4/280 "
          f"({len(block.splitlines())} L)")


# ============ 3. version bump ============

def bump_versions(cur_pp, cur_init):
    print(f"\n=== version bump → {TARGET_VERSION} ===")
    pp_path = Path("pyproject.toml")
    pp_src = pp_path.read_text()
    pp_new = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\1"{TARGET_VERSION}"',
        pp_src, count=1, flags=re.M,
    )
    if pp_new != pp_src:
        pp_path.write_text(pp_new)
        print(f"  pyproject.toml: {cur_pp} → {TARGET_VERSION}")
    else:
        print(f"  pyproject.toml: no-op (already {TARGET_VERSION})")

    init_path = Path("cadw/__init__.py")
    init_src = init_path.read_text()
    init_new = re.sub(
        r'__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{TARGET_VERSION}"',
        init_src, count=1,
    )
    if init_new != init_src:
        init_path.write_text(init_new)
        print(f"  cadw/__init__.py: {cur_init} → {TARGET_VERSION}")
    else:
        print(f"  cadw/__init__.py: no-op (already {TARGET_VERSION})")

    import cadw as _cadw
    importlib.reload(_cadw)
    if _cadw.__version__ != TARGET_VERSION:
        raise SystemExit(
            f"version sync FAILED: cadw.__version__="
            f"{_cadw.__version__}, expected {TARGET_VERSION}"
        )
    print(f"  ✓ cadw.__version__ == {TARGET_VERSION}")


# ============ 4. roadmap.json (defensive, skip) ============

def update_roadmap():
    print("\n=== roadmap.json: skip (schema unknown) ===")
    print("  Defer to v0.9.25o-roadmap_schema follow-up.")


# ============ 5. pytest ============

def run_pytest():
    print("\n=== pre-flight: facade AST guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "tests/test_dataframe.py::test_facade_has_no_row_loops",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FACADE AST GUARD FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: changelog + roadmap docs tests ===")
    targets = ["tests/test_changelog.py"]
    if Path("tests/test_roadmap.py").exists():
        targets.append("tests/test_roadmap.py")
    r = subprocess.run(
        ["uv", "run", "pytest", *targets,
         "-q", "-W", "error::SyntaxWarning"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** DOCS TEST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 6. main ============

def main():
    print("=== v0.9.25n-corrected: Phase C close-out ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    state = verify_state()
    update_changelog(
        state["new_13_4"],
        state["pyproject_version"],
        state["init_version"],
    )
    update_decisions(state["new_13_4"])
    bump_versions(state["pyproject_version"], state["init_version"])
    update_roadmap()
    run_pytest()

    nn = state["new_13_4"]
    commit_with_guards(
        "v0.9.25n: Phase C close-out — CHANGELOG + DECISIONS "
        f"§13.4/{nn} + §13.4/280 + version sync. "
        "Docs-only + version bump; no production code changes. "
        "Phase C closed at 20/20 Pandas P0 features: "
        "C-1 (isin/between/select_dtypes, v0.9.25g), "
        "C-2 (idxmax/idxmin/pct_change, v0.9.25h), "
        "C-3a (ffill/bfill/reindex, v0.9.25i), "
        "C-3b (cut/qcut/duplicated, v0.9.25j), "
        "C-4a (str.slice/cat/count, v0.9.25k), "
        "C-4b (dt.strftime/tz_convert/str.split, v0.9.25l), "
        "C-5 (dropna/fillna, pre-existing since v0.9.23g). "
        "C-5 was not reimplemented — v0.9.25m-diagnostic via "
        "hasattr revealed both features already shipped and tested "
        "(12 tests: test_eda_basics 5+7, test_facade_transformers "
        "6 dropna, test_pipeline 2 dropna). 0 mutations, 0 "
        "duplicate work (§13.4/279). "
        "Changes: "
        "(1) CHANGELOG.md: new '### v0.9.25: Phase C — Pandas P0 "
        "API' section documenting all 20 features + Phase C fix-"
        "cycle tally + key learnings + docs-debt note. "
        "(2) DECISIONS.md: append "
        f"§13.4/{nn} — 'hasattr reveal ≠ rebuild' (Phase C-5 "
        "meta-lesson, actual DECISIONS.md numbering — commit "
        "messages use §13.4/279 alias) + Phase C close "
        "retrospective; AND §13.4/280 — 'file-state assertions "
        "need REPL-first too' (v0.9.25n-corrected meta-lesson: "
        "commit messages are NOT authoritative for file contents). "
        f"(3) pyproject.toml: version "
        f"{state['pyproject_version']} → {TARGET_VERSION}. "
        f"(4) cadw/__init__.py: __version__ "
        f"{state['init_version']} → {TARGET_VERSION} "
        "(aligned; import-time verify via importlib.reload). "
        "(5) docs/roadmap.json: SKIPPED — schema unknown; "
        "dedicated v0.9.25o-roadmap_schema follow-up. "
        "§13.4/280 (NEW): file-state assertions use actual "
        "inspection, not inference from commit messages / git log "
        "/ sibling files. First v0.9.25n attempt aborted in "
        "verify_state with 'expected §13.4/270+, got 248' — my "
        "assumption 'commit messages cite §13.4/270+ therefore "
        "DECISIONS.md is at 270+' was wrong. Commit messages use "
        "§13.4/262–279 as meta-references; DECISIONS.md canonical "
        "was at 248 (v0.9.22b era). Gap 249–279 = ~30 lessons of "
        "docs debt, tracked as v0.9.26 docs-debt sweep. "
        "§13.4/279 (v0.9.25m): hasattr reveal ≠ rebuild. "
        "§13.4/278: hasattr-vs-source-text presence checks. "
        "§13.4/277: accessor gate layer. §13.4/266: post-anchor "
        "re-verify. §13.4/264: REPL-first (applied to library "
        "behavior in Phase C; applied to file STATE in this "
        "corrected patch via §13.4/280). "
        "Idempotency: each section checks own marker; re-run → all "
        "no-op skip. "
        "Test counts: unchanged (~2488 fast) — docs-only patch. "
        "Next: v0.9.25o-roadmap_schema (roadmap.json Phase C "
        "close), then v0.9.26 docs-debt sweep + Phase D / v1.0 "
        "readiness.",
        self_path="scripts/patch_v0925n_phase_c_close_corrected.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25n COMPLETE. **Phase C 20/20 CLOSED.** "
          "Next: v0.9.25o-roadmap_schema, then Phase D / v1.0.")


if __name__ == "__main__":
    main()