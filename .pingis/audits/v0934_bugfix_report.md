# v0.9.34 - Bug fix report

Baseline HEAD `38c8eb7` (v0.9.32). Product changes in this pass: B4, A9, A10, B6. B1/B2/B7 were already fixed at baseline (proof + measurement below). B3 rejected with measurement.

| ID | Status | Fix | Before | After | Test |
|---|---|---|---|---|---|
| B1 | already-fixed-at-baseline | list `parts` + single `concat_tables(parts)` (6fa5e49, v0.9.25c) | 0.0773 s / 2,001,000 row copies (12.9 GB @rbs=200k) | 0.0014 s / 2,000 copies (0.13 GB) | tests/test_v0934_b1_refill.py pass |
| B2 | already-fixed-at-baseline | in-loop `check_state(len(uniques)*180)` (6fa5e49, v0.9.25c) | OOM ~600 MB @10M uniq (invisible) | MemoryLimitExceeded at 100k uniq / 2 MB budget | tests/test_v0934_b2_encode.py pass |
| B3 | rejected-with-measurement | keep Python loop; Arrow group_by slower (+K=N mem) | loop 0.0258 s, O(batch) | arrow 0.1036 s (4.01x), 500k groups, +72 MB | tests/test_v0934_b3_rocauc.py pass |
| B7 | already-fixed-at-baseline | `dictionary_encode` + hash uniques (v0.9.18) | 934 ms / 19.4 MB peak | 65 ms / 4.9 MB peak (14.4x, -14.5 MB) | tests/test_v0934_b7_hash.py pass |
| A9 | fixed | `log_fit` wired into linear / pca / naive_bayes (3 NB fitters) | 3/6 models logged | 6/6 logged, uniform format | tests/test_v0934_a9_ml_logs.py pass |
| B4 | fixed | `_UNIQUE_FALLBACK_FACTOR=4` + `_fallback_unique_count` helper | raw `len(set(...))` no factor | 4x factor on short-string set | tests/test_v0934_b4_mg.py pass |
| A10 | fixed | docstrings: empty-input raises ValueError (both) | contract undocumented | documented explicitly | tests/test_v0934_a10_empty.py pass |
| B6 | fixed | README: hardcoded v0.8.8/v0.9.24 -> `cadw.__version__` pointer | v0.8.8 / v0.9.24 hardcoded | pointer to `cadw.__version__` | tests/test_v0934_b6_readme.py pass |

Bench artifacts: `.pingis/benches/v0934_{b1_refill,b2_encode,b3_rocauc,b7_hash}.json`.
## Bench scripts (task-required path `scripts/`)

| Bug | Script | Artifact |
|---|---|---|
| B1 | `scripts/bench_v0934_b1.py` | `.pingis/benches/v0934_b1_refill.json` |
| B2 | `scripts/bench_v0934_b2_encode.py` | `.pingis/benches/v0934_b2_encode.json` |
| B3 | `scripts/bench_v0934_b3_rocauc.py` | `.pingis/benches/v0934_b3_rocauc.json` |
| B7 | `scripts/bench_v0934_b7_hash.py` | `.pingis/benches/v0934_b7_hash.json` |

All four scripts run from the project root and overwrite their JSON
artifact with current numbers.

## Suite status

- Fast: `uv run pytest -q -m "not slow"` → 2576 passed, 2 skipped,
  25 deselected (baseline 2557 + 19 new v0.9.34 tests), 0 failed.
- Slow: `uv run pytest -q -m slow` → 25 passed, 2578 deselected, 0 failed (103.55 s).

## Notes on already-fixed items

B1/B2/B7 were authored in `docs/BUG_BACKLOG.md` from audit reports;
the fixes had already merged (6fa5e49 / v0.9.18) before v0.9.33. The
task allows rejection-with-proof; here the proof is a repro test that
PASSES at HEAD plus a bench showing the current code path is the fast
one. No production code change was needed for those three.
