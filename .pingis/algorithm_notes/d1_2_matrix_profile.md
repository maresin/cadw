# D.1.2 — Matrix Profile (streaming, bounded state)

**Version:** v0.9.35 · **Module:** `cadw/ml/matrix_profile.py` · **Mixin:** `df.fit_matrix_profile(...)`
**Complexity:** M · **State:** O(S·w), independent of N

## Sources

- Yeh, Zhu, Ulanova, Begum, Ding, Dau, Zimmerman, Silva, Mueen, Keogh,
  *Matrix Profile I: All Pairs Similarity Joins for Time Series*,
  ICDM 2016, arXiv:1701.02795.
- Zhu, Zimmerman, Senobari, Yeh, Funning, Mueen, Brisk, Keogh,
  *Matrix Profile II: Exploiting a Novel Algorithm and GPUs to Break the
  One Hundred Million Barrier*, ICDM 2016, arXiv:1612.00835.
- STUMPY documentation (BSD-3), `stumpy.stump` / `stumpy.stumpi` API and
  z-normalized Euclidean distance definition.
- UCR Time Series Archive — discord/novelty tasks; "finds planted anomaly"
  is the test criterion in ALGORITHM_BACKLOG.

## Algorithm

z-normalized Euclidean distance between subsequences of length `w`:

```
d(T_i, T_j) = sqrt( sum_k ((x_{i+k} - mu_i)/sd_i - (x_{j+k} - mu_j)/sd_j)^2 )
```

Matrix profile at i = `min_{j != i} d(T_i, T_j)`; the index array records
argmin. Discords = largest profile values.

### Deviation from STAMP (documented, not silent)

STAMP/STOMP keep O(N) for the profile and are exact over the full series.
The project constraint is ≤1 batch in RAM, so this implementation:

1. **Pass 1** — deterministic reservoir sample of up to `max_samples`
   subsequences (default 4096) from the streaming column.
2. **Pass 2** — exact block STOMP *over the sample*, chunked so the
   distance block stays ≈32 MB.

The result is an **approximate** matrix profile with respect to the whole
series; `sample_fraction_` (= sample / total subsequences) is reported and
stored in `to_dict()`. The exact block computation over the sample is
verified against brute force in tests (atol 1e-10).

This is a substitution of "exact but O(N)" for "approximate but O(S·w)",
which is the only shape that satisfies the project's RAM rule. It is
recorded here rather than presented as exact.

### Vectorization

Distance block via the standard dot-product expansion
`sq_i + sq_j - 2·(Z_i·Z_j)` with `np.maximum(..., 0)` for numerical
safety, self-match set to `inf` on the diagonal of each block. No Python
row loop over data; only per-block rows for the diagonal mask.

## RAM model

| Component | Bytes |
|---|---|
| Sample (pass 1) | `S·w·8` |
| Reference z-norm copy | `S·w·8` |
| Profile + index | `S·(8+8)` |
| Pass-2 block | ≤32 MB (block size derived from `S`) |

`check_state(S·w·8)` runs *before* allocation, so an oversized
`max_samples` raises `MemoryLimitExceeded` instead of OOM (test with
`max_samples=40000`, `w=64` under a 2 MB budget).

## Benchmark (measured)

`scripts/bench_matrix_profile.py` → `.pingis/benches/matrix_profile.json`.
Synthetic seasonal signal + noise, w=32, S=4096, 100k-row batches:

| N | fit | peak RSS delta | state | n_seen | sample fraction |
|---|---|---|---|---|---|
| 100 000 | 0.600 s | 32.9 MB | 1 114 112 B | 99 969 | 0.0410 |
| 1 000 000 | 3.174 s | 23.7 MB | 1 114 112 B | 999 690 | 0.0041 |

State is identical at 10× the row count. Time scales ~linearly in N because
pass 1 must still visit every subsequence; the expensive O(S²·w) part is
constant. Peak RSS does not grow with N.

No STUMPY install in this environment, so the reference comparison is an
independent NumPy brute-force over the stored sample (exact agreement) and
a separate closed-form brute force for `transform` against raw data.

## Tests (12)

smoke/shapes · brute-force reference on the sample · planted anomaly in top
k · discord score bounds · determinism · multi-batch invariance (chunk 1 vs
10 000) · state independent of N · guard on oversized sample · errors
(window, missing col, nulls, too few subsequences) · `transform` vs brute
force · source contract (citations, no pandas, every `check_batch` has
`extra_factor`).

## Edge cases

- Nulls in the value column → `ValueError` (a null would silently shift the
  subsequence grid).
- `< 8` subsequences → `ValueError` with the actual count.
- Constant subsequence (zero variance) → z-norm row set to zeros rather
  than dividing by ~0; documented in `_znorm_rows`.
- Single batch or many: reservoir indices are per-stream, so the sample is
  identical for the same seed *only if* batch boundaries do not change
  which subsequences exist — verified by the chunk-invariance test.

## Measured accuracy vs exact STAMP (reported as measured)

Two measurements, synthetic data:

1. Exact path (`sample_fraction = 1.0`): N=200, w=12, S=4096 keeps every
   subsequence; profile is **bit-identical** to brute-force STAMP over the
   full series (`maxdiff = 0.00e+00`). Covered by
   `test_exact_path_matches_bruteforce_stamp`.
2. Sampled path: N=3000, w=16, S=512, `sample_fraction = 0.1715`.
   **Metric definition:** top-k is taken over ALL 2985 subsequence
   positions of the full series, for both the exact full-series profile
   and the sampled-path distances (i.e. `m.transform(df)`, which gives
   each position's exact NN distance to the sample). This is Case A:
   overlap with the true top-k of the full series is expected to be low
   for any uniform subsample, so 0/10 is not a rank bug. The separate
   property that IS guaranteed — profile equal to brute force over the
   sampled positions — is covered by `test_reference_vs_bruteforce`
   (atol 1e-10). Distances are z-normalized Euclidean units
   (range 0..sqrt(2w) = 0..5.66 for w=16).
   `transform()` (exact NN distance from each subsequence to the sample)
   vs the exact full-series profile:

   | metric | value |
   |---|---|
   | top-10 discord overlap | 0 / 10 |
   | top-50 overlap | 1 / 50 |
   | top-100 overlap | 6 / 100 |
   | mean abs profile error (z-norm units) | 0.804 |
   | max abs profile error (z-norm units) | 3.215 |

Weak, and reported as weak: a uniform reservoir sample is not a good
approximation for discord ranking, because a subsequence's true nearest
neighbour is usually a near-duplicate the sample did not keep. The profile
is exact when the sample covers the series, and otherwise a bounded-memory
approximation with `sample_fraction_` exposed.

**Status:** D.1.2 ships as exact-on-sample, not as a drop-in STAMP
replacement. A future revision should use stride sampling (keeps
neighbourhoods) or STUMPI-style incremental updates. Recorded here, not
claimed as done.

