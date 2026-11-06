# S1 — Neural generative (VAE / GAN)

*Section:* S — Generative modelling

**Description:** Learn p(x) via neural encoder-decoder (VAE) or adversarial game (GAN). Synthesize realistic tabular rows.

**Real-life examples:**
- Privacy-preserving synthetic data
- Class-imbalance oversampling
- Simulation

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: downstream_accuracy, runtime_s

## Models
### TVAE — baseline / tier3_export

**Roles:** probabilistic, representation
**Status:** desc, export

**Purpose:** Xu et al. 2019: tabular VAE (CTGAN paper). Encoder-decoder with mode-specific normalization for continuous + categorical. Baseline neural synthesizer. Last priority (v0.9.130+): CPU-feasible but training is hours-days; implement after verified CADW core.

**Use cases:**
- Synthetic tabular data
- Imbalance oversampling

**Config:** `{'epochs': 'int', 'latent_dim': 'int', 'batch_size': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → downstream_accuracy=null
  - algorithm: `sdv.tabular.TVAE` (code)
  - number: https://github.com/sdv-dev/CTGAN
  - code/paper: https://github.com/sdv-dev/CTGAN
  - note: no_public_benchmark: tabular synthesis evaluated per-dataset (SDV paper). torch required.

### CTGAN — sota / tier3_export

**Roles:** probabilistic, representation
**Status:** desc, export

**Purpose:** Xu et al. 2019: conditional tabular GAN with mode-specific normalization and training-by-sampling. Handles multimodal + imbalanced columns. Last priority (v0.9.130+): CPU-feasible but training is hours-days; implement after verified CADW core.

**Use cases:**
- Synthetic tabular data
- Imbalanced categorical columns

**Config:** `{'epochs': 'int', 'generator_dim': 'tuple', 'discriminator_dim': 'tuple'}`

**Benchmarks:**
- **[literature]** cadw_internal → downstream_accuracy=null
  - algorithm: `sdv.tabular.CTGAN` (paper)
  - number: https://github.com/sdv-dev/CTGAN
  - code/paper: https://arxiv.org/abs/1907.00503
  - note: no_public_benchmark: Xu et al. 2019 per-dataset numbers. torch required.

### CTAB-GAN+ — sota / tier3_export

**Roles:** probabilistic, representation
**Status:** desc, export

**Purpose:** Zhao et al. 2022: improved CTGAN with auxiliary classifier, information loss, and conditional vectors. Better for downstream ML utility. Last priority (v0.9.130+): CPU-feasible but training is hours-days; implement after verified CADW core.

**Use cases:**
- ML utility-focused synthesis
- Mixed-type tabular

**Config:** `{'epochs': 'int', 'pac': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → downstream_accuracy=null
  - algorithm: `CTAB-GAN+` (paper)
  - number: https://github.com/Team-TUD/CTAB-GAN-Plus
  - code/paper: https://arxiv.org/abs/2109.06563
  - note: no_public_benchmark: Zhao et al. 2022. torch required.

### TabNet — sota / tier3_export

**Roles:** nonlinear_interactions
**Status:** desc, export

**Purpose:** TabNet (Arik & Pfister 2019): attentive tabular deep net; sequential feature attention.

**Use cases:**
- Tabular deep learning baseline
- Feature-attribution-friendly DL

**Config:** `{'n_steps': 'int', 'n_epochs': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → downstream_accuracy=null
  - algorithm: `TabNet` (paper)
  - number: .pingis/benches/
  - code/paper: https://arxiv.org/abs/1908.07442
  - note: no_public_benchmark: TabNet per-dataset (torch; CPU-feasible)

## Metrics
- **downstream_accuracy** (anchor, higher)
  - Train downstream model on synthetic, test on real; utility of synthetic data.
  - *special:* Depends on downstream model and train size; report 3 model families.
- **runtime_s** (diagnostic, lower)
  - Wall-clock training time.
  - *special:* GPU-dependent; not comparable to CPU runs.
