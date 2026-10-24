# M1 — Contextual bandit (online policy selection)

*Section:* M — Contextual bandits

**Description:** Sequentially choose an arm given context and observed reward; balance exploration vs exploitation under delayed feedback.

**Real-life examples:**
- News / content recommendation with per-article CTR
- Ad placement with per-user context
- Clinical trial adaptive randomization

## Benchmarks
- **Yahoo_R6A** — https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=1
  - metrics: ctr, ctr_lift_pct
- **Open_Bandit_Dataset** — https://github.com/st-tech/zr-obp
  - metrics: ctr, relative_ctr

## Models
### LinUCB — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** Li et al. 2010 — per-arm ridge regression on context with an upper-confidence exploration bonus. Inductive bias: reward is a linear function of context per arm, with a disjoint or hybrid parameterization; strong when contexts are low-dimensional and reward is roughly linear.

**Use cases:**
- News recommendation with article features
- Ad slot allocation with user context
- Dynamic pricing with linear demand

**Config:** `{'alpha': 'float', 'l2': 'float'}`

**CADW:** `cadw.ml.linucb.fit_linucb`

**Benchmarks:**
- **[literature]** Yahoo_R6A → ctr=1.795 (n=1)
  - algorithm: `LinUCB (Li et al. 2010, disjoint)` (code)
  - number: https://arxiv.org/html/1003.0146v2
  - code/paper: https://github.com/david-cortes/contextualbandits
  - note: fetched 2026-10-25 from arXiv:1003.0146 Table 1 (one-week eval set, 100% data): 'linucb (disjoint) 1.795' in deploy bucket. Same page also reports +12.5% lift over epsilon-greedy.
- **[literature]** Yahoo_R6A → ctr_lift_pct=12.5 (n=1)
  - algorithm: `LinUCB (Li et al. 2010, disjoint)` (code)
  - number: https://arxiv.org/html/1003.0146v2
  - code/paper: https://github.com/david-cortes/contextualbandits
  - note: fetched 2026-10-25 from arXiv:1003.0146 Table 1: 'linucb (disjoint) ... 12.5%' lift over epsilon-greedy at 100% data.
- **[cadw_measured]** cadw_internal → peak_rss_mb=51.1 (n=3)
  - algorithm: `cadw.ml.linucb.fit_linucb` (code)
  - number: .pingis/benches/linucb.json
  - code/paper: cadw/ml/linucb.py
  - note: CADW-native: peak_mb=4.7 @n=10k, 19.4 @n=50k, 51.1 @n=150k (linucb.json); fit_s=0.04 @n=150k, 5 arms. NOT comparable to external CTR numbers.

### EpsilonGreedy — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ agent, desc

**Purpose:** Epsilon-greedy exploration over per-arm reward estimates. Inductive bias: no structural assumption — uniform random exploration with rate epsilon, then exploit the empirical mean. Canonical baseline; weak when rewards are context-dependent.

**Use cases:**
- A/B test with fixed explore rate
- Cold-start content display
- Bandit baseline in production stacks

**Config:** `{'epsilon': 'float'}`

**Benchmarks:**
- **[literature]** Yahoo_R6A → ctr=1.596 (n=1)
  - algorithm: `epsilon-greedy (Li et al. 2010 baseline)` (code)
  - number: https://arxiv.org/html/1003.0146v2
  - code/paper: https://github.com/st-tech/zr-obp
  - note: fetched 2026-10-25 from arXiv:1003.0146 Table 1: 'epsilon-greedy 1.596'. Reference baseline (0% lift by construction).

### UCB1 — baseline / tier1_core

**Roles:** probabilistic
**Status:** ✓ agent, desc

**Purpose:** Auer et al. 2002 UCB1 — non-contextual optimism under uncertainty with an O(sqrt(log T / n_a)) bonus. Inductive bias: stationary arm means with no context; outperforms epsilon-greedy on stationary problems but not context-dependent ones.

**Use cases:**
- Non-contextual slot selection
- Bandit baseline without features
- Multi-armed test allocation

**Config:** `{'confidence': 'float'}`

**Benchmarks:**
- **[literature]** Yahoo_R6A → ctr=1.594 (n=1)
  - algorithm: `UCB1 (Li et al. 2010 baseline)` (code)
  - number: https://arxiv.org/html/1003.0146v2
  - code/paper: https://github.com/david-cortes/contextualbandits
  - note: fetched 2026-10-25 from arXiv:1003.0146 Table 1: 'ucb 1.594' in deploy bucket.

### ThompsonSampling — sota / tier2_wrapper

**Roles:** probabilistic
**Status:** ✓ agent, desc

**Purpose:** Chapelle & Li 2011 — Beta-Bernoulli posterior sampling per arm; plays an arm with probability proportional to its posterior probability of being optimal. Inductive bias: conjugate Bayesian per-arm model; naturally balances exploration and exploitation and is empirically strong on CTR data.

**Use cases:**
- News recommendation with Thompson sampling
- Ad CTR optimization
- Adaptive clinical allocation

**Config:** `{'prior_alpha': 'float', 'prior_beta': 'float'}`

**Benchmarks:**
- **[literature]** Open_Bandit_Dataset → relative_ctr=1.43 (n=3)
  - algorithm: `Bernoulli Thompson Sampling (ZOZO logger)` (paper)
  - number: https://arxiv.org/html/2008.07146
  - code/paper: https://papers.nips.cc/paper_files/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html
  - note: fetched 2026-10-25 from arXiv:2008.07146 Table 1 (ALL campaign): 'Bernoulli TS 12,168,084 0.50% ± 0.004 1.43' — relative CTR vs uniform-random. Same table gives 1.94 on Men's and 1.84 on Women's.
- **[literature]** Open_Bandit_Dataset → ctr=0.5 (n=3)
  - algorithm: `Bernoulli Thompson Sampling (ZOZO logger)` (paper)
  - number: https://arxiv.org/html/2008.07146
  - code/paper: https://papers.nips.cc/paper_files/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html
  - note: fetched 2026-10-25 from arXiv:2008.07146 Table 1 (ALL campaign): 'Bernoulli TS ... 0.50% ± 0.004'. Value is percent CTR (0.5% = 0.005 fraction).

### NeuralUCB — sota / tier3_export

**Roles:** nonlinear_interactions, representation
**Status:** desc, export

**Purpose:** Zhou et al. 2020 — UCB exploration on top of a neural-network reward surrogate, using the network's gradient as a feature map. Inductive bias: arbitrary nonlinear reward function learnable by an MLP; strong on high-dimensional context (image/text) but needs torch.

**Use cases:**
- Image-content recommendation
- High-dimensional ad creatives
- Neural bandits in RLHF

**Config:** `{'hidden_sizes': 'list', 'lr': 'float', 'lambda_': 'float'}`

**Benchmarks:**
- **[literature]** Mushroom → cumulative_reward=null (n=1)
  - algorithm: `NeuralUCB (Zhou et al. 2020)` (paper)
  - number: https://archive.ics.uci.edu/dataset/73/mushroom
  - code/paper: https://arxiv.org/abs/1911.04462
  - note: fetch_ok_no_number: https://archive.ics.uci.edu/dataset/73/mushroom is the dataset page only; NeuralUCB's own Mushroom results are in figures (arXiv:1911.04462) with no extractable numeric table.

## Metrics
- **ctr** (anchor, higher)
  - Click-through rate on a held-out evaluation set; the primary online-bandit metric.
  - *special:* Depends on the arm set and the logging policy; only compare within the same dataset. Report relative CTR vs baseline alongside.
- **ctr_lift_pct** (diagnostic, higher)
  - Relative CTR lift versus the epsilon-greedy baseline.
  - *special:* Baseline-specific; do not compare across papers with different baselines.
- **relative_ctr** (diagnostic, higher)
  - Relative CTR versus uniform-random logging policy (ZOZO convention).
  - *special:* Different baseline than ctr_lift_pct; do not mix.
- **cumulative_reward** (pareto_axis, higher)
  - Total reward over the horizon in simulated bandits (Mushroom).
  - *special:* Horizon- and reward-encoding-dependent.

## Ensemble rules
- **[linear_signal, probabilistic]** → LinUCB + ThompsonSampling (max_per_role=2)
  - LinUCB (linear UCB) and Thompson Sampling (Bayesian posterior sampling) are the two canonical exploration strategies; combine by mixing their action distributions when both verified in cadw.
