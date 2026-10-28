# Section J — algorithm specs (NLP)

Conventions: same as section I. Document-term data represented as a `RecordBatch` with columns
`doc_id:int64`, `term_id:int64`, `count:float64` (COO triplets) OR a dense `(b, V)` batch when V is small.

## J1.ComplementNB
- state_class: O(C*V) class-term counts (C classes, V vocab) + O(C) totals; V spilled to disk when `8*C*V > state_limit_bytes`
- state_type: aggregation (Reduce over sparse term counts)
- bounded_state: true — C small (<=100 typical), V up to 1e6 -> spill (C separate IPC files, one per class, or a single file keyed by class)
- n_scalability: O(nnz) time per pass, single pass to fit; O(1) RAM if V spilled
- primitive_reuse: `ReduceOp` `CNBClassTermOp` -> per-class `np.add.at(counts[c], term_ids, counts)`; `SpillDir` for per-class count vectors when dense C*V too large; `MemoryGuard.check_state` pre-flight
- passes: 1 to fit, then predict streaming per batch (argmin of complement log-lik)
- estimate_state: `8*C*V + 8*C + 64`; checked before `init()`
- prediction: `score[c] = -( sum_t x_t * log((N_c_total - count[c,t] + alpha) / (N_c_total + alpha*V)) )`; argmin instead of argmax (complement)
- edge cases: unseen terms at predict time -> skipped (no contribution); alpha default 1.0; classes with N_c=0 skipped with warning
- output: `ComplementNBModel(classes_, class_log_prior_, feature_log_prob_, feature_names_)`; `.predict(batch_iter)` yields labelled batches
- deviation from sklearn: sklearn's `feature_log_prob_` uses smoothing over full V; we compute it lazily from counts on disk, so we never materialize C*V. Improvement: streaming `partial_fit` variant that merges new class-term counts into the spill files.

## J3.NMF
- state_class: O(n*k) W (spilled always) + O(k*V) H (kept in RAM if `8*k*V <= state_limit_bytes`, else spilled too)
- state_type: partitioned (HashShuffle row blocks for W update) + aggregation (Reduce for XtV, XW)
- bounded_state: true — W always spilled; H bounded by k*V check
- n_scalability: O(nnz*k) per iteration, 2 passes per iteration; #iterations configurable (default 100 or until relative delta < 1e-4)
- primitive_reuse: `hash_shuffle` to partition rows by `row_block = row_id // block_rows` (gives contiguous W blocks), `read_partition_batches` for per-block W update, `ReduceOp` for `XtV` accumulation, `SpillDir` + `spill_array_writer/reader` for W, `external_sort` optional for deterministic output
- passes per iteration: 2 (update H with fixed W -> Reduce over X; update W with fixed H -> HashShuffle by row block, per-block multiplicative update)
- estimate_state: `8*k*V + 8*k*block_rows + 64`; checked at fit start
- update rules (Lee-Seung, L2 optional):
  H <- H * (Wt X) / (Wt W H + eps)
  W <- W * (X Ht) / (W H Ht + eps)
- edge cases: zeros in numerator/denominator -> eps=1e-10; k > min(n, V) -> raise; non-negative check on input (skip negatives with warning)
- output: `NMFModel(W_path, H, components_, reconstruction_error_)`; `W` stays on disk (`.transform` returns path)
- deviation from sklearn: sklearn keeps W in RAM; we spill W and update it blockwise. Improvement: deterministic block layout via HashShuffle with fixed hash_seed and row_id, so W blocks are reproducible across runs.

## SKIP rationale
- J1.LinearSVC / J2.Word2Vec / BERTopic / BERT / SentenceTransformer: see selection.md (torch or thin wrapper with no new out-of-core algo needed under 2.8 GB).
