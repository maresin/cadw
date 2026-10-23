# EFFICIENCY_AUDIT_v0930 (PINGIS)

Bench: `scripts/pingis_efficiency_bench.py` (raw: `EFFICIENCY_BENCH_v0930.json`).
Reference = pandas 3.x in-memory. CADW = out-of-core with MemoryGuard(200 MiB). Peak RSS = psutil delta @5 ms.

## Table

| Algorithm | Asymptotic (CADW) | Peak RAM/row | 100k | 1M | Ref 1M | Alternative | Why better | Source |
|---|---|---|---|---|---|---|---|---|
| groupby_aggregate | O(N) IO + O(K*P) | ~2 MB @1M (2e-6 B/row); 56 MB @100k (fixed IPC-writer) | 0.48 s | 1.11 s | 0.03 s | keep (R1) | pandas is in-memory only | pandas, polars docs |
| external_sort | O(N log N) + O(N) IO | 8.2 MB @100k; 39.4 MB @1M (mmap-inclusive) | 0.11 s | 1.06 s | 0.04 s | keep; A1 fix | Arrow sort_indices is C++ | Arrow compute docs |
| hash_join | O(N) shuffle + O(N) output | 52.5 MB @100k; 158.8 MB @1M | 0.21 s | 2.81 s | 0.09 s | use hash_join_streaming (R2) | wrapper collects to Table | join.py:731 |
| reduce (Mean/Sum/Count/Std/MinMax/Cov) | O(N), O(1) state | ~0 | <0.02 s | 0.67 s | 0.005 s | keep | already optimal | DECISIONS 9.2.1 |
| hash_shuffle | O(N) | budget-sized | 0.02 s | 0.07 s | n/a | keep | vectorized XXH64 | shuffle.py docstring |
| ML (linear/logistic/kmeans/pca/gbdt/nb) | O(N*d) per epoch | small state (d^2, k*d, trees*leaves) | suite -m ml 406 | - | - | keep | inherent OOC design | sklearn docs |

## Findings & recommendations

### R1. groupby @100k peak 56 MB despite tiny output
Cause: 8 partition writers open simultaneously + mmap of partition files + Arrow IPC writer buffer per partition. Output is K=100 rows, but fixed cost dominates at small N. Not a bug (formula predicts shuffle peak ~batch*P, within budget). Documented behavior: for N << 10M choose num_partitions proportional to N/batch_rows.

### R2. hash_join (non-streaming) peak 158 MB @1M under 200 MB budget
Finding under 1.3. hash_join is a backward-compat wrapper that collects the full streaming output into a Table (join.py:731). This violates the <=1-batch constraint if used as the primary API. Action: use hash_join_streaming (returns iterator). No code change to hash_join (breaking return type rejected); doc-emphasis on streaming is the fix.

### R3. external_sort @1M peak 39 MB vs pred 3.67 MB (10.7x)
Cross-reference AUDIT A1. Formula (3*K*rbs*row_bytes) is fine; drift is mmap pages from _RunStream IPC readers. Fix in stage 5 (extra_factor=2.0 in phase-1 check_batch; close readers eagerly).

### R4. Reduce optimal
All ReduceOps are O(1) state, one pass. pandas 5-100x faster on tiny data due to startup; same asymptotic. Keep.

### R5. ML models — no replacement
O(d^2)/O(k*d)/O(trees) state is standard asymptotic. sklearn accuracy comparison confirms; no public benchmark suggests strictly better OOC algorithm.

## Replacements

No algorithm from roadmap is replaced. Every Phase D candidate is a new algorithm (IForest, permutation importance, GMM density), not a replacement of existing code. The only candidate for replacement was hash_join (R2) — rejected as a breaking API change.

Rules from 1.3 satisfied: replacement requires strictly-better asymptotic/RAM + public benchmark; none found that meets both for the audited algorithms.

## Sources consulted

- Apache Arrow C++ compute docs (sort_indices, hash kernels) — asymptotic reference for sort/join.
- pandas 3.0 docs — in-memory reference only.
- Polars docs (group_by, sort, streaming) — asymptotic reference for out-of-core.
- DECISIONS.md 9.2.1–9.2.13, 13.4/226–236 — internal formula and lesson trail.
