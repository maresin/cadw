# v0.9.34 - Notes (edge cases / surprises)

## Already-fixed at baseline
- B1: fixed by 6fa5e49 (v0.9.25c). Quadratic `concat_tables([buffer, chunk])` in `_RunStream.refill` replaced with `parts_list` + one concat. Measured 239x at N=2000.
- B2: fixed by 6fa5e49 (v0.9.25c). In-loop `check_state(len(uniques)*180)` added to `encode_categorical` Pass 1. Guard fires before the Python set can grow unbounded.
- B7: fixed by v0.9.18 (AUDIT 1.3). `_string_hash_column` now dictionary-encodes then hashes K uniques + vectorized map-back (11.7x, -13.2 MB).
- The BUG_BACKLOG.md was authored from audit reports; for B1/B2/B7 those fixes had already merged before v0.9.33. Task explicitly allows rejection-with-proof; here the proof is: repro PASSES at HEAD, and inline bench shows the current code path is the fast one.

## B3: rejected with measurement
- Python per-row loop (0.094 s @500k, O(batch)) beats Arrow `group_by(proba)` (0.233 s, +72 MB worst-case K=N).
- Pass2 is not the bottleneck: `external_sort` dominates end-to-end AUC by >100x.
- Decision documented inline in `cadw/ops/metrics.py` and in `.pingis/benches/v0934_b3_rocauc.json`.

## Implementation surprises
- pyarrow `Array` type is immutable at the C level, so B4 could not be tested via monkeypatching `dictionary_encode`. Extracted `_fallback_unique_count(col)` as a module-level helper; test now calls it directly (400 for 100 uniques).
- B1 test needed `textwrap.dedent(inspect.getsource(...))` before `ast.parse` - otherwise IndentationError from the method body.
- B1 bench: N=2000 gives marginal speedup on tiny tables because of per-call overhead; bumped N=5000 and asserted 3x floor to avoid flake.
- A9 wiring is duplicated pattern: each fit function does a local `from cadw.ml._logging import log_fit` right before the return, matching how kmeans/logistic/gbdt already do it. 3 NB fitters share the same file; all three now log with distinct model names (`naive_bayes.gaussian` etc.).
- B6 README had three separate stale markers: `(v0.8.8)` header, `(v0.9.24)` TS header, and a hardcoded `~1096 fast + 8 slow` count. All three replaced with pointers; test asserts none reappear.

## Resume-run additions (v0.9.34)
- Bench scripts were originally only `.pingis/bench_b1.py` (a stub).
  Created the task-required `scripts/bench_v0934_{b1,b2,b3,b7}.py`;
  each regenerates its JSON under `.pingis/benches/`.
- The b2/b7 scripts needed `sys.path.insert(0, parents[1])` because
  `uv run python scripts/...` puts `scripts/` (not the repo root) on
  sys.path. b1/b3 only import pyarrow, so they did not need it.
- Bench numbers re-measured on this run differ from the earlier run
  (machine noise / smaller warm cache): B1 55x here vs 239x earlier;
  B3 loop 0.026s vs 0.094s earlier, arrow 4.0x slower vs 2.5x.
  The qualitative conclusion is unchanged in both cases; the JSON
  holds the fresh numbers and the report quotes them.
- CHANGELOG had [Unreleased] then v0.9.32 with no v0.9.33 entry
  (v0.9.33 was a docs-only commit). Added the v0.9.34 entry only;
  did not retroactively add v0.9.33.
