# S2 — Diffusion (tabular)

*Section:* S — Generative modelling

**Description:** Denoising diffusion probabilistic models adapted to tabular data.

**Real-life examples:**
- High-fidelity synthetic data
- Generative modeling

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: downstream_accuracy

## Models
### TabDDPM — baseline / tier3_export

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** Kotelnikov et al. 2022: diffusion for tabular data. Gaussian for continuous, multinomial diffusion for categorical. Best quality on many benchmarks.

**Use cases:**
- Synthetic tabular data
- Privacy-preserving synthesis

**Config:** `{'timesteps': 'int', 'epochs': 'int', 'batch_size': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → downstream_accuracy=null
  - algorithm: `TabDDPM` (paper)
  - number: https://github.com/yandex-research/tab-ddpm
  - code/paper: https://arxiv.org/abs/2209.15421
  - note: no_public_benchmark: Kotelnikov et al. 2022. torch required.

### TabSyn — sota / tier3_export

**Roles:** probabilistic, representation
**Status:** desc, export

**Purpose:** Zhang et al. 2023: diffusion in latent space of a VAE. Combines VAE compression with diffusion generation; better efficiency.

**Use cases:**
- Efficient synthetic data
- High-quality tabular

**Config:** `{'latent_dim': 'int', 'timesteps': 'int', 'epochs': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → downstream_accuracy=null
  - algorithm: `TabSyn` (paper)
  - number: https://github.com/amazon-science/tabsyn
  - code/paper: https://arxiv.org/abs/2310.09656
  - note: no_public_benchmark: Zhang et al. 2023. torch required.

## Metrics
- **downstream_accuracy** (anchor, higher)
  - Train-on-synthetic / test-on-real ML utility.
  - *special:* Same as S1; report 3 model families.
