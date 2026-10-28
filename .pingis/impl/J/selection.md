# Section J - NLP

Benchmarks: 20Newsgroups https://scikit-learn.org/stable/datasets/real_world.html ; SST-2 https://huggingface.co/datasets/nyu-mll/glue ; Reuters https://www.nltk.org/book/ch02.html ; STS-B https://huggingface.co/datasets/sentence-transformers/stsb

## IMPLEMENT

### J1.ComplementNB [baseline, tier1_core]
- algorithm_source_url: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html (fetched 200)
- ref: Rennie et al. 2003, Tackling the Poor Assumptions of Naive Bayes Text Classifiers
- benchmark_url: https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset
- metric: accuracy
- state_class: O(C*V) class-feature weights; bounded_state: true (C small, V spilled to disk if > state_limit)
- state_type: aggregation (Reduce over sparse batch counts)
- primitive_reuse: Reduce (per-class term counts, total count), SpillDir for V-sized arrays
- algorithm: pass 1 Reduce -> per-class token counts & totals; complement weights w[c,t] = log((N_c_total + alpha) / (count[c,t] + alpha)) / normalizer. Predict argmax_c sum_t x[t]*w[c,t] per batch.

### J3.NMF [baseline, tier2 -> out-of-core tier1]
- algorithm_source_url: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html (fetched 200); Lee & Seung 1999 multiplicative updates
- ref: https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization
- benchmark_url: https://scikit-learn.org/stable/datasets/real_world.html
- metric: reconstruction_error (proxy for perplexity)
- state_class: O(n*k + V*k) W/H factors; bounded_state: true (W spilled to disk)
- state_type: partitioned (HashShuffle batches by row block) + aggregation (multiplicative update)
- primitive_reuse: Reduce (XtV, WHt stats), SpillDir for W, HashShuffle for row blocks
- algorithm: alternating multiplicative updates on W (n*k) and H (k*V). Per iteration: pass batches to update H (Reduce), then pass again (via W spilled to disk) to update W. Bounded by state_limit.

## SKIP (tier3_export, no algorithm)
- J1.BERT - torch
- J1.LinearSVC - thin wrapper, no new algo under 2.8GB needed (sklearn is O(nnz) per batch already); keep described_only
- J2.Word2Vec - DEFERRED (not torch): plausible out-of-core via HashShuffle on (target, context) pairs + negative sampling + SpillDir on embedding tables; not built in this pass
- J2.SentenceTransformer - torch
- J3.BERTopic - torch + UMAP + HDBSCAN combo; described_only
- J1.TFIDF_LogReg / J2.LSA_TFIDF_SVD / J3.LDA - verified_in_cadw; no new algorithm
