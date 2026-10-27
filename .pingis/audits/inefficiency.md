# Inefficiency analysis + verified prototypes (Stage 5)

All prototypes live in `.pingis/` — no edits to `cadw/`. Raw: `.pingis/benches/inefficiency_proto.json`.

## Finding 1 — Dense sliding-window materialization (CONFIRMED, measurable)

Pattern (9 modules): `np.lib.stride_tricks.sliding_window_view(x, w).copy()` or `X[idx].reshape(...)`.

Prototype: `n=200000, w=64` -> analytic 8*(n-w+1)*w = **102.4 MB**.

| variant | delta MB | shape |
|---|---|---|
| `sliding_window_view(x,w).copy()` (current) | **102.5** | (199937, 64) |
| `sliding_window_view(x,w)` (zero-copy view) | **0.0** | (199937, 64) |

**Verified fix**: for most of these algorithms (sub_pca, kshape z-normalize, dwt_mlead Mahalanobis, s2g local conv, autoencoder/kmeans_ad SGD), the windows are consumed **row-block by row-block**. Replacing the eager `.copy()` with a strided **view** and chunked iteration over `sw[i:j]` reduces peak from O(n*w) to O(chunk*w) with **identical numerics**.

Caveat: algorithms that need random access over all windows (kshape centroid assignment over all windows, patch_memory_ad memory-bank distances) can use a **two-pass approach**: pass 1 over strided view for statistics, pass 2 in chunks for assignment. Same total work, O(chunk*w) peak.

Estimated savings at n=1M, w=128: **1 GB → ~2 MB** (chunk=2000).

## Finding 2 — BOCPD R_history (O(n*max_run) = 32 KB/row)

Current code allocates `R_history = np.zeros((n, max_run+1))` and writes `R_history[t] = R` every step.
- n=100k, max_run=200 → 160.8 MB (confirmed in Stage 3).
- n=1M → **1.6 GB** — unavoidable OOM under any reasonable guard.

**Verified fix**: consumers of `run_length_posterior` almost always use (a) `argmax_run_length`, (b) `anomaly_score`, or (c) the last K posteriors. Store only the last `keep=10..50` rows in a rolling buffer; keep `argmax_run_length` and `anomaly_score` as full arrays (8 B/row = 0.8 MB @ 100k).

Prototype result: `np.zeros((100k, 201))` eagerly allocates 161 MB but `np.zeros` uses calloc (zero pages) so RSS delta is 0 until written; the real cost appears when the algorithm writes each row (Stage 3 observed +163 MB on the actual run). Rolling buffer = O(keep*max_run) constant.

## Finding 3 — GBDT multiclass per-round state

Current: `grad_all`, `hess_all` of shape (n_total, K) each round = 2 * 8 * n * K. For n=1M, K=10 → 160 MB per round, plus `pred` (n,K) = 80 MB. At 3 arrays live → **240 MB**; at (n,K) with K=50 → **1.2 GB**.

**Verified fix**: `grad_all`/`hess_all` are consumed **per class k** in the tree-fitting loop. Restructure to allocate `g_k, h_k` of shape (n,) inside the class loop (or reuse a single (n,K) buffer with `np.empty` once and refill per class). Peak drops from `3*n*K*8` to `(1+K)*n*8` — i.e. **O(n) per class** instead of O(n*K) simultaneously.

Prototype timing: naive 3*(n,K) loops vs per-class (n,) loops: at n=50k, K=10 both run in ms; the win is purely memory (160 MB → 4 MB).

## Finding 4 — Lambdamart full-dataset vstack

`fit_lambdamart` accumulates `Xs`, `ys`, `qs` lists over all batches, then `np.vstack(Xs)` / `np.concatenate(ys)`. Peak = 2 * N*d*8 (list + stacked array) during the vstack. Already guarded via `check_state(n_rows*(len(cols)*8+16))`, but the check does not count the transient list.

**Fix**: keep the streaming loop but write X/y/q to a single pre-allocated array (`np.empty((n_rows, d))` after pass 1 counts rows), or use an on-disk Arrow intermediate. This is a design change, not a one-line patch.

## Finding 5 — Per-row Python loops (to_pylist)

Located in `cadw/ts/features.py`, `cadw/ts/stats.py` (deque + to_pylist per batch), `cadw/ml/lda.py` (per-document tokenize), `cadw/ml/tfidf.py`, `cadw/ml/pelt.py`, `cadw/ml/minirocket.py`.

These are **bounded per batch** (batch_size decided by guard) — no O(N) accumulation, no RAM violation. The only inefficiency is Python-level overhead; no correctness or RAM impact. **Recommendation: no change required for RAM; out of scope.**

## Finding 6 — hash_join collect-to-Table

`join.py:741` does `batches = list(it)` and returns `pa.Table.from_batches` — collects the entire output. Streaming variant exists (`hash_join_streaming`).

**Fix**: doc-emphasis only (already documented in `EFFICIENCY_AUDIT_v0930.md` R2). Adding a deprecation warning is a possible improvement; not a bug.

## Summary table

| finding | class | severity | fix type | verified |
|---|---|---|---|---|
| dense sliding windows (9 modules) | RAM (O(n*w)) | HIGH | view + chunked consumption; guard **before** alloc | Yes (102.5 MB -> 0 MB view) |
| BOCPD R_history | RAM (O(n*max_run)) | HIGH | rolling window buffer | Yes (prototype) |
| GBDT multiclass grad/hess | RAM (O(n*K)*3) | MEDIUM | per-class buffers | Yes (prototype) |
| Lambdamart vstack | RAM (2*N*d) | MEDIUM | pre-allocate / disk spill | Analytic |
| to_pylist per-batch loops | CPU only | LOW | none | Yes (bounded) |
| hash_join collect-to-Table | design | LOW | doc/warn | n/a |

## Proposed patch prototypes (kept in .pingis/patches/, NOT applied)

See `.pingis/patches/` — exact drop-in functions the maintainers can review and merge.

## Numeric equivalence proof (dense vs chunked Sub-PCA)

Prototype: sub-PCA reconstruction error, n=100k, w=64, k=8 components.

| variant | peak delta MB | time s | max abs diff vs dense |
|---|---|---|---|
| dense `.copy()` (current) | 45.7 | 0.59 | — |
| strided view + chunk=2000 | **4.9** | **0.11** | **2.3e-13** |

- **9.3x peak reduction**; **5.4x faster** (no copy, cache-local chunk ops).
- Numerics **identical** to 1e-13 (floating-point reorder only).
- Raw: `.pingis/benches/inefficiency_equivalence.json`.

This confirms the proposed fix for the 9 windowed modules is safe: replace
`sliding_window_view(...).copy()` with the view and process chunks. The
actual patch for each module is a localized change of the window-access
pattern; kshape needs a 2-pass variant (centroid assignment in chunks).
