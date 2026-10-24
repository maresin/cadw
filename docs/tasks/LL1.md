# L1 — Node / edge / graph classification, link prediction

*Section:* L — Graph ML (summary — out of scope, torch required)

**Description:** Learn representations of nodes/edges/graphs for classification and link prediction.

**Real-life examples:**
- Fraud-ring detection on transaction graphs
- Citation-network node classification
- Friend / item link prediction

## Benchmarks
- **Cora_Citeseer_Pubmed** — https://arxiv.org/html/1609.02907
  - metrics: accuracy
- **BlogCatalog_PPI_Wikipedia** — https://arxiv.org/html/1607.00653
  - metrics: macro_f1
- **PPI_GraphSAGE** — https://arxiv.org/html/1706.02216
  - metrics: micro_f1

## Models
### GCN — baseline / tier3_export

**Roles:** local_structure, representation
**Status:** desc, export

**Purpose:** Kipf & Welling 2017 — graph convolution as a first-order Chebyshev approximation of spectral filtering; each layer averages neighbour features then applies a linear map. Inductive bias: homophily + locality, one-hop aggregation per layer.

**Use cases:**
- Citation-network classification
- Molecule property prediction
- Knowledge-graph completion

**Config:** `{'hidden_dim': 'int', 'num_layers': 'int', 'dropout': 'float'}`

**Benchmarks:**
- **[literature]** Cora_Citeseer_Pubmed → accuracy=81.5 (n=3)
  - algorithm: `GCN (Kipf & Welling 2017)` (paper)
  - number: https://arxiv.org/html/1609.02907
  - code/paper: https://arxiv.org/abs/1609.02907
  - note: fetched 2026-10-25 from arXiv:1609.02907 HTML Table 2: 'GCN (this paper) ... 81.5 (4s)' for Cora; Citeseer 70.3, Pubmed 79.0 in the same row. accuracy is percent. out of scope: torch required.

### GraphSAGE — sota / tier3_export

**Roles:** local_structure, representation
**Status:** desc, export

**Purpose:** Hamilton et al. 2017 — inductive node embedding by sampling and aggregating a fixed-size neighbourhood, so it generalizes to unseen nodes/graphs. Inductive bias: local sample-and-aggregate; unlike GCN, no full-graph Laplacian required.

**Use cases:**
- Inductive node classification on new graphs
- Dynamic social graphs
- Recommendation on evolving graphs

**Config:** `{'aggregator': 'string', 'num_samples': 'list', 'hidden_dim': 'int'}`

**Benchmarks:**
- **[literature]** PPI_GraphSAGE → micro_f1=null (n=1)
  - algorithm: `GraphSAGE (Hamilton et al. 2017)` (paper)
  - number: https://arxiv.org/html/1706.02216
  - code/paper: https://arxiv.org/abs/1706.02216
  - note: fetch_ok_no_number: https://arxiv.org/html/1706.02216 renders but the results table is figure-embedded, not extractable as text. out of scope: torch required.

### GAT — sota / tier3_export

**Roles:** local_structure, representation, nonlinear_interactions
**Status:** desc, export

**Purpose:** Velickovic et al. 2018 — self-attention over neighbours, learning per-edge weights instead of fixed Laplacian coefficients. Inductive bias: importance-weighted local aggregation, one head per relation type.

**Use cases:**
- Node classification with heterogeneous neighbour importance
- Knowledge-graph embedding
- Traffic forecasting

**Config:** `{'num_heads': 'int', 'hidden_dim': 'int', 'dropout': 'float'}`

**Benchmarks:**
- **[literature]** Cora_Citeseer_Pubmed → accuracy=83.0 (n=3)
  - algorithm: `GAT (Velickovic et al. 2018)` (paper)
  - number: https://arxiv.org/html/1710.10903
  - code/paper: https://arxiv.org/abs/1710.10903
  - note: fetched 2026-10-25 from arXiv:1710.10903 HTML Table 1: 'GAT (ours) 83.0 ± 0.7%' on Cora; Citeseer 72.5, Pubmed 79.0 in the same row. out of scope: torch required.

### Node2Vec — sota / tier3_export

**Roles:** representation, local_structure
**Status:** desc, export

**Purpose:** Grover & Leskovec 2016 — biased random walks (return/in-out parameters p, q) feeding Skip-gram embeddings. Inductive bias: random-walk neighbourhood similarity; transductive (fixed node set).

**Use cases:**
- Graph node embeddings
- Link prediction
- Community detection

**Config:** `{'p': 'float', 'q': 'float', 'walk_length': 'int', 'num_walks': 'int', 'dim': 'int'}`

**Benchmarks:**
- **[literature]** BlogCatalog_PPI_Wikipedia → macro_f1=0.2581 (n=3)
  - algorithm: `node2vec (Grover & Leskovec 2016)` (paper)
  - number: https://arxiv.org/html/1607.00653
  - code/paper: https://arxiv.org/abs/1607.00653
  - note: fetched 2026-10-25 from arXiv:1607.00653 HTML Table 2: 'node2vec 0.2581 0.1791 0.1552' across BlogCatalog/PPI/Wikipedia (macro-F1 at 50% labeled). out of scope: torch required.

## Metrics
- **accuracy** (anchor, higher)
  - Node classification accuracy on transductive citation benchmarks (Cora/Citeseer/Pubmed).
  - *special:* Splits differ across papers; compare only within one source table.
- **micro_f1** (diagnostic, higher)
  - Micro-F1 on inductive multi-graph benchmarks (PPI).
- **macro_f1** (diagnostic, higher)
  - Macro-F1 on multilabel node classification (BlogCatalog/PPI/Wikipedia).
