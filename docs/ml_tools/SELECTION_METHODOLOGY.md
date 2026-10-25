# Model Selection Methodology (ML sections)

Status: approved v0.3, 2026-10-25
Applies to: all ML sections A–N (except O — separate methodology)
Not applies to: purely infrastructural cards (cadw_internal runtime/RAM)

Companion files:
- DU_DRAFT_TEMPLATE.md — template for per-DU drafts
- scripts/audit_leakage.py — leakage gate
- schema.py — extended card fields (§12)

## §1. Principles (9)

1. Multi-benchmark by default. Min 2 per subtype; 3+ mainstream.
2. Unified evaluation. Normalize all on base benchmark + base metric.
3. Leakage-aware. Pretrained → tier3 only; leakage_risk mandatory.
4. Pareto, not sum. No weighted scalar from heterogeneous metrics.
5. Zoo control. 1–2 per cell class × mode × modality; empty = documented gap.
6. Baselines vs SOTA vs Specialists. All three mandatory per subtype.
7. Tier honesty. sota = top of modern benchmark; baseline = floor.
8. Library-aware. PyOD/Merlion/sktime/Orion/TAB — full source; source_library field.
9. Read-the-source. For every candidate: read description AND code before
   deciding heavy/testable/exclude. Formal numeric limits are NOT used —
   heaviness is judged from description (deps, reported params, algorithm),
   relative speed, and code complexity.

## §2. Three-level taxonomy (selection unit = L3)

| L1 | L2 | L3 examples |
|---|---|---|
| Supervised tabular | Classification | binary, multiclass, imbalanced, multilabel, ordinal |
| Supervised tabular | Regression | point, quantile |
| Time series | Forecast | uni, multi |
| Time series | Anomaly | point, changepoint, shape |
| Time series | Classification | — |
| Unsupervised | Anomaly | unsup, sup |
| Unsupervised | Density | — |
| Specialized | Recsys, LTR, Causal, Survival, Bandits | ... |
| Applicability-only | NLP, Vision, Graph | 1–2 per subtype |

## §3. Model classes (TAB, 5)
non-learning | ML | DL from scratch | LLM-based | pretrained-TS.
Classes 1–3 target for CADW (tier1/tier2). Classes 4–5 only tier3_export.

## §4. Learning modes (6)
unsupervised, semi-supervised, self-supervised, supervised, zero-shot, few-shot.
Zero-shot/few-shot → only tier3.

## §5. Treatment per subtype
core | applicability-check | non-predictive (see §13)

## §6. Section-specific parameters
(table as in prior discussion — anchors, benchmarks, leakage per section)

## §7. Benchmarks: levels
base | domain | scale | legacy | library-native.
Registered in benchmarks.yaml with level, model_classes_covered,
leakage_notes, split_protocol.

## §8. Leakage control
(fields and rules as before; audit_leakage.py is a separate gate)

## §9. Evaluation protocol
(primary/secondary/tertiary; Pareto; slices)

## §10. Selection strategy — funnel
See §15. Three categories per subtype: baseline / sota / specialist.

## §11. Tier system (4)
tier1_core | tier2_wrapper | tier3_export | tier4_rejected

## §12. Required card fields
(see Artifact 3)

## §13. DoD per subtype
(9 items as before)

## §14. DU-draft template
See DU_DRAFT_TEMPLATE.md

## §15. Selection funnel

### §15.1. Two channels — run in parallel

Channel B (benchmarks): B1 → B2 → B3 → B4 → B5
Channel L (libraries):    L1 → L2 → L3 → L4 → L5 → L6

### §15.2. Channel B — benchmarks

| Step | Action | Decision |
|---|---|---|
| B1 | Locate leaderboard | If no live leaderboard — channel B not usable for this task; go L-only |
| B2 | Cut bottom half by primary metric | Keep top 50% |
| B3 | Baseline exception | Old / widely used / present in 2+ libraries / sole representative of class → keep as baseline despite bottom half |
| B4 | Read description of survivors | Deps, reported params, relative speed, architecture — judge heaviness |
| B5 | Cut to 1–2 per class×mode | Keep top accuracy per cell |

### §15.3. Channel L — libraries

| Step | Action | Decision |
|---|---|---|
| L1 | Find maximum number of relevant libraries | Record each with URL |
| L2 | Extract model list from each library's docs | model × library table |
| L3 | Read description + code for each model | Deps, LOC, algorithmic complexity, relative speed |
| L4 | Search for accuracy: docs / paper / internet / own knowledge | accuracy_data: number \| none |
| L5 | If no data → mark `needs_physical_check` | — |
| L6 | Cut by class / type / accuracy | 1–2 per class×mode |

### §15.4. Merge

| Step | Action |
|---|---|
| M1 | Union of B5 and L6 |
| M2 | Dedupe by (name, class, mode). If duplicate: keep benchmark value as primary, add source_library as secondary |

### §15.5. Code review pass (only `needs_physical_check`)

| Step | Action | Result |
|---|---|---|
| C1 | Read source code | — |
| C2 | Algorithm matches claimed class? | mismatch → exclude |
| C3 | Judge complexity from code: deps, LOC, algorithmic complexity, speed signals | heaviness_signals |
| C4 | Implementable on CADW primitives? | yes → tier1 candidate |
| C5 | Verdict | testable / tier3 / exclude — with justification |

### §15.6. Physical test
P1 synthetic DGP → P2 primary metric + runtime + RAM →
P3 compare with reported; verdict consistent/divergent →
P4 pass to YAML cards (PINGIS).

### §15.7. Funnel metrics
Each step records N_before, N_after, reason_for_cut.
Final table in draft: N₀ → ... → N_final.

## §16. Workflow and artifacts

### Phases

| Phase | Owner | Input | Output |
|---|---|---|---|
| A. Funnel pass | human + assistant | libraries + benchmarks | DU draft, full candidate tables with stage_in/stage_out |
| B. Code review | assistant | draft, model sources | code_review block in draft; verdicts |
| C. Physical test | assistant | testable candidates, samples.py | .pingis/benches/reproduction/*.json |
| D. YAML cards | PINGIS | approved draft + reproduction JSONs | changes to docs/ml_tools/<Section>_*.yaml + tests |
| E. Acceptance | human | YAML diff, tests, CHANGELOG | commit + accept/reject |

### Artifacts per phase

| Phase | Artifact | Location | Committed? |
|---|---|---|---|
| A | DU draft | .pingis/drafts/DU<N>_*.md | no (working log) |
| B | code_review section | same draft | no |
| C | reproduction JSON | .pingis/benches/reproduction/<model>.json | yes |
| D | YAML card updates | docs/ml_tools/<Section>_*.yaml | yes |
| D | validator + tests | tests/test_ml_tools_*.py | yes |
| E | CHANGELOG entry | CHANGELOG.md | yes |

### Hand-off contract
- Phase A→B: draft must have `needs_physical_check` list with `source_url` per candidate
- Phase B→C: draft must have verdict per candidate (testable/tier3/exclude)
- Phase C→D: for every testable model, reproduction JSON exists
- Phase D→E: validator passes, render_matrix OK, audit_leakage OK