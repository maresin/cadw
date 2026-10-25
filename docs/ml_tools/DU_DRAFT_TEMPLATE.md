# DU-<N> — <Section> Selection Draft

Status: <draft vX | final v1.0>
Started: <date>
Owner: <human>
Section: <A|D|E|...>
Subtypes covered: <list of L3 ids>

## 1. Subtypes (L3)

| ID | Name | Treatment |
|---|---|---|
| ... | ... | core / applicability / non-predictive |

## 2. Benchmarks

| Level | Name | Metrics | Notes |
|---|---|---|---|
| base | ... | ... | ... |
| domain | ... | ... | ... |
| scale | ... | ... | ... |
| legacy | ... | ... | ... |

## 3. Libraries scanned

| Library | URL | Models listed | Leaderboard? |
|---|---|---|---|
| PyOD | ... | ... | → TSB-AD |
| ... | ... | ... | ... |

## 4. Candidate tables — per subtype

### 4.x. <Subtype>

| Model | Class | Mode | Source | Stage in | Leaderboard / accuracy | stage_out | Reason_out | Tier (draft) |
|---|---|---|---|---|---|---|---|---|
| ... | ... | ... | B/L | B1/L1 | value or "none" | B5/L6/M2/C5/P4 | ... | ... |

**Code review block** (for needs_physical_check rows):

| Model | read_desc | read_code | deps | loc_est | params_reported | speed_signal | verdict | justification |
|---|---|---|---|---|---|---|---|---|

## 5. Coverage matrix

| class × mode | <subtype 1> | <subtype 2> | ... |
|---|---|---|---|
| non-learning | ... | ... | ... |
| ML | ... | ... | ... |
| DL (from scratch) | ... | ... | ... |
| pretrained | ... | ... | ... |

**Documented gaps:** ...

## 6. Leakage flags

| Model | pretraining_corpus | overlap_risk | temporal_split | notes |
|---|---|---|---|---|

## 7. Metrics per subtype

| Subtype | Primary (anchor) | Secondary | Legacy (warning) |
|---|---|---|---|

## 8. P0-edits to existing YAML

| Current | Action | Reason |
|---|---|---|

## 9. Excluded list (rule-based)

| Model | Reason (bottom-half / heavy / no-code / mismatch) |
|---|---|

## 10. Funnel metrics

| Stage | N | Δ | Reason |
|---|---|---|---|
| N₀ long list | ... | — | — |
| N₁ after B2 | ... | ... | ... |
| ... | ... | ... | ... |
| N_final | ... | ... | — |

## 11. Open questions

1. ...

## 12. Decisions

1. ...