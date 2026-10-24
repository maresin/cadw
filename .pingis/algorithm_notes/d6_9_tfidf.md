# D.6.9 Streaming TF-IDF vectorizer

## Paper
Salton, Buckley, *Term-Weighting Approaches in Automatic Text Retrieval*,
Information Processing & Management 24(5), 1988. Reference implementation
in tests: sklearn `TfidfVectorizer` (Pedregosa et al., JMLR 12, 2011).

## Docs
- `cadw.ml.tfidf.fit_tfidf(df, text_col, vocab_size=10000,
  sublinear_tf=True, min_df=1, norm="l2")`
- `CadwDataFrame.fit_tfidf(...)`.
- Model: `transform(df, text_col)` -> per-document sparse (indices, values)
  rows; `transform_dense(...)` for small inputs/tests; `idf_`, `vocab_`.

## Design
- Pass 1 streams the document-frequency vector (O(V) state), filters by
  min_df and keeps the top `vocab_size` tokens.
- IDF uses sklearn's smoothed formula: idf = log((1+n)/(1+df)) + 1.
- Pass 2 emits sparse (indices, values) per row with sublinear TF
  (1 + log tf) and l1/l2/none normalisation, so a dense D x V matrix is
  never materialised outside the explicit `transform_dense` helper.

## Benchmark (`.pingis/benches/tfidf.json`, vocab 16)
| n | fit_s | nnz | peak_mb | state_b |
|---|---|---|---|---|
| 10 000 | 0.026 | 51 000 | 0.7 | 768 |
| 100 000 | 0.258 | 509 405 | 45.1 | 768 |
| 500 000 | 1.281 | 2 545 851 | 229.6 | 768 |
Reference: sklearn TfidfVectorizer with the identical tokenizer,
vocabulary, sublinear_tf, smooth_idf and l2 norm — dense matrices agree to
atol 1e-9 (test_reference_sklearn_tfidf_exact). Peak RSS at n=500k is
230 MB, under budget.

## RAM model
State = 8*V (idf) + 8*V (vocab) + 512 overhead (768 B measured). Fitting
holds only the O(V) document-frequency dict and one in-memory batch
(O(batch) strings); every batch passes check_batch(extra_factor=3.0).
