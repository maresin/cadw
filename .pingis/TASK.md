# PINGIS — periodic analysis task for CADW

## Context

**CADW** — out-of-core framework for tabular data that doesn't fit in
RAM. Three primitives: `Reduce`, `HashShuffle`, `Sort`. Bounded-memory
guarantee. Current version: check `git log --oneline -1` and read
`DECISIONS.md` §11.1 (implemented), §11.5 (limitations), §13.4 (lessons).

**Read before working:**
1. `DECISIONS.md` §9 (planner), §11.4 (must-fix history), §13.4 (lessons).
2. `_FOR_PINGIS/REPORT.md` and `analysis_ram.md` — previous audit.
3. `CHANGELOG.md` — what's recently changed.

**Do NOT re-report known issues without checking:**
- Known S1 (as of v0.5.1): P1 (join `to_pandas`), P3 (`list(reader)` in
  groupby), P4 (heavy accum Python dict) — see `analysis_ram.md`.
- Known P2, P8 — fixed in v0.5.1. Verify with `tests/test_join_streaming.py`
  and `tests/test_reduce_state_bytes.py`.

## Constraints

- **Do NOT modify `cadw/`** — analysis-only.
- Work in branch `pingis/<task-id>`.
- Output goes to `_FOR_PINGIS/` (gitignored) or `.pingis/` (gitignored
  except this TASK.md).
- Do not add new dependencies. Use existing: `pyarrow`, `pandas`,
  `numpy`, `psutil`, `xxhash`.

## Goal 1: RAM correctness audit

Read each op in `cadw/ops/` (`reduce.py`, `shuffle.py`, `sort.py`,
`groupby.py`, `join.py`, `rank.py`, `quantile.py`, `dedup.py`) and look
for:

- `to_pylist()`, `to_pandas()`, `as_py()` in hot paths.
- `list(reader)`, `Table.from_batches(...)` on unbounded input.
- `out_batches.extend(...)` — output accumulation.
- Python-list buffers (`buffer[c].append(...)`).
- Python dict accumulators (`accum[key] = ...`).
- `check_state` called after materialize.

For each finding report: **P#-ID**, **severity (S1/S2/S3/S4)**,
**code location (file:line)**, **peak estimate formula**, **how to
reproduce** (benchmark idea).

Severity:
- **S1**: guaranteed OOM on realistic data (> 2× budget violation).
- **S2**: might exceed budget depending on data type / scale.
- **S3**: efficiency/correctness issue, not OOM.
- **S4**: cosmetic / docs.

## Goal 2: Algorithm suggestions

For each finding (or generally), suggest Arrow-native or classic
external-memory alternative. **Verify availability** via:

```python
import pyarrow.compute as pc
print([n for n in pc.list_functions() if 'hash' in n.lower()])
```

**Do not suggest non-existent functions.** Currently confirmed absent:
- `pc.hash_64` (PR #45001 not merged as of Arrow 26).
- `pc.compute_hash`.

Confirmed present: `Table.join`, `Table.group_by`, `pc.sort_indices`,
`pc.take`, `pc.bucketize`, `pc.dictionary_encode` (method on Array).

Report: current → suggested → feasibility (doable / needs research /
not possible).

## Goal 3: Prototype new operations

When new op is on roadmap (see `DECISIONS.md` §12.7) or requested by
user, prototype in `_FOR_PINGIS/proto/`. Each prototype:

- Working script (runnable).
- Uses CADW primitives (Reduce/Shuffle/Sort) — not reinvent the wheel.
- Measured peak RSS via `psutil` + sampler thread.
- Comparison to naive approach where relevant.

## Goal 4: ML/viz readiness

- Which ML algorithms are feasible on CADW primitives?
- Which need adapters (fine) or are out-of-scope (state > RAM)?
- Prototype 1-2 algorithms per run.
- Viz adapters: aggregate-first pattern (histogram/box/line/heatmap);
  scatter only via sampling or datashader.

## Output format

Create/update these files in `_FOR_PINGIS/`:
- `REPORT.md` — executive summary + status of previous findings.
- `analysis_ram.md` — full P1..Pn catalog with benchmarks.
- `analysis_external.md` — external solutions + applicability matrix.
- `gap_ml.md` — ML/viz readiness.
- `bench/bench_ram*.py` + `bench/RESULTS.md`.
- `proto/` — working prototypes.

At the top of `REPORT.md`:
```
**Run**: YYYY-MM-DD
**CADW version**: vX.Y.Z (git describe)
**Arrow version**: X.Y.Z
**Findings**: X new, Y resolved, Z unchanged.
```

## Re-run criteria

Trigger a new analysis when:
- **A new tag is released** (vX.Y.Z).
- **A new op is added** to `cadw/ops/`.
- **Arrow version bumps** in `pyproject.toml` (check for new `pc.*`
  functions that would fix P6/P11).
- **~2-3 months** since last analysis.

## Current focus (fill each run)

- [ ] Verify v0.5.1 fixes (P2, P8).
- [ ] Re-benchmark join (B1a) and sort (B3a) — expect less than previous.
- [ ] Next candidate fixes: P1 (join `to_pandas`), P3/P10 (groupby
      materialize), P5 (sort buffer), P6/P11 (`to_pylist`).
- [ ] Check Arrow version for `pc.rank`/`pc.quantile`/`pc.rank_quantile`
      availability (would fix P11 immediately).
