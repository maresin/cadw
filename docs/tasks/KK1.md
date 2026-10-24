# K1 — Image classification

*Section:* K — Vision (summary — out of scope, torch required)

**Description:** Classify images into a fixed label set; the canonical deep-learning benchmark task.

**Real-life examples:**
- Product image categorization
- Medical image triage
- Content moderation

## Benchmarks
- **ImageNet1k** — https://paperswithcode.com/sota/image-classification-on-imagenet
  - metrics: accuracy, top1_err

## Models
### ResNet50 — baseline / tier3_export

**Roles:** representation, nonlinear_interactions
**Status:** desc, export

**Purpose:** He et al. 2016 — residual connections let very deep CNNs train by learning identity shortcuts. Inductive bias: translation-equivariant convolution + depth via skip connections; the canonical vision backbone.

**Use cases:**
- Image classification backbone
- Feature extractor for detection
- Transfer-learning baseline

**Config:** `{'depth': 'int', 'width_mult': 'float', 'num_classes': 'int'}`

**Benchmarks:**
- **[literature]** ImageNet1k → top1_err=22.85 (n=1)
  - algorithm: `ResNet-50 (He et al. 2016)` (paper)
  - number: https://arxiv.org/html/1512.03385
  - code/paper: https://arxiv.org/abs/1512.03385
  - note: fetched 2026-10-25 from arXiv:1512.03385 HTML Table 3: 'ResNet-50 22.85 6.71' (10-crop validation, top-1 error %; accuracy = 100 - 22.85 = 77.15%). out of scope: torch required.

### ViT — sota / tier3_export

**Roles:** representation, nonlinear_interactions
**Status:** desc, export

**Purpose:** Dosovitskiy et al. 2021 — split image into 16x16 patches, treat as tokens for a Transformer encoder. Inductive bias: minimal image-specific prior (patch locality only); needs large-scale pretraining to beat CNNs.

**Use cases:**
- Large-scale vision backbone
- Multimodal (CLIP-style) base
- Transfer to detection/segmentation

**Config:** `{'patch_size': 'int', 'num_layers': 'int', 'hidden_dim': 'int'}`

**Benchmarks:**
- **[literature]** ImageNet1k → accuracy=88.55 (n=1)
  - algorithm: `ViT-H/14 (Dosovitskiy et al. 2021)` (paper)
  - number: https://arxiv.org/html/2010.11929
  - code/paper: https://arxiv.org/abs/2010.11929
  - note: fetched 2026-10-25 from arXiv:2010.11929 HTML abstract: 'the best model reaches the accuracy of 88.55% on ImageNet' (ViT-H/14, ImageNet-21k pretrain). out of scope: torch required.

### ConvNeXt — sota / tier3_export

**Roles:** representation, nonlinear_interactions
**Status:** desc, export

**Purpose:** Liu et al. 2022 — modernize a ResNet with Transformer-era training recipe and design choices (depthwise conv, LayerNorm, GELU). Inductive bias: pure convolution, but matched to ViT-era scaling.

**Use cases:**
- Vision backbone competitive with ViT
- Edge-friendly (no attention)
- Detection/segmentation backbone

**Config:** `{'depths': 'list', 'dims': 'list', 'num_classes': 'int'}`

**Benchmarks:**
- **[literature]** ImageNet1k → accuracy=87.8 (n=1)
  - algorithm: `ConvNeXt (Liu et al. 2022)` (paper)
  - number: https://arxiv.org/html/2201.03545
  - code/paper: https://arxiv.org/abs/2201.03545
  - note: fetched 2026-10-25 from arXiv:2201.03545 HTML abstract: 'achieving 87.8% ImageNet top-1 accuracy' (ConvNeXt-XL). out of scope: torch required.

### EfficientNet — sota / tier3_export

**Roles:** representation, nonlinear_interactions
**Status:** desc, export

**Purpose:** Tan & Le 2019 — compound scaling of depth/width/resolution under a fixed FLOP budget, found by neural-architecture search. Inductive bias: mobile-conv blocks with SE; Pareto-optimal for a compute budget.

**Use cases:**
- Mobile / edge inference
- Compute-budgeted backbone
- Transfer-learning baseline

**Config:** `{'compound_coef': 'float', 'num_classes': 'int'}`

**Benchmarks:**
- **[literature]** ImageNet1k → accuracy=84.3 (n=1)
  - algorithm: `EfficientNet-B7 (Tan & Le 2019)` (paper)
  - number: https://arxiv.org/html/1905.11946
  - code/paper: https://arxiv.org/abs/1905.11946
  - note: fetched 2026-10-25 from arXiv:1905.11946 HTML abstract: 'EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet'. out of scope: torch required.

## Metrics
- **accuracy** (anchor, higher)
  - ImageNet-1k top-1 accuracy (percent).
  - *special:* Model size and pretraining regime differ across papers; compare only within one source table.
- **top1_err** (diagnostic, lower)
  - ImageNet-1k top-1 error rate (percent) as reported by ResNet paper.
  - *special:* Complement of accuracy; report which one the source uses.
