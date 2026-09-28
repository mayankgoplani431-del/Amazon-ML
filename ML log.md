# ML log: Amazon ML Challenge 2026, Business Entity Resolution (team amazing)

Metric: macro F0.5 per Source-1 entity (singleton: empty = 1.0, any prediction = 0.0). Test: 1,732,544 S1
(India 46.75%, US 38.27%, France 14.98%, unseen); pool 9,969,589 S2/S3. All "held-out" numbers are on labelled train data
never used for the choice being evaluated.

## 1. Leaderboard history
| # | File | What changed | Leaderboard |
|---|---|---|---|
| 1 | E01 | 4 blocking passes, ~50 features, LightGBM 2-stage, expected-F0.5 decision | 0.953 |
| 2 | E02 | + name-as-token and exact-key blocking (recall 95.37% -> 97.51%) | 0.964 |
| 3 | S3 | + group/density features (validation up, test down) | 0.956 |
| 4 | S4 | E02 ∩ S3 agreement | 0.969 |
| 5 | S5 | E02 ∩ S3 ∩ E06 (E06: postal/legal normalisation, test-density training) | 0.971 |
| 6 | S6 | S5 + France strictness calibration | 0.971 |
| 7 | 0.977 file | Other-PC pipeline "EXP-1" (learned 8-per-S1 cut, 2-stage LightGBM) | 0.977 |
| 8 | **S9** | 0.977 minus 16,642 French pairs: <=3 of 5 models agree AND house number differs | **0.9796** |
| 9 | S10 | S9 minus 13,722 French weak pairs with name similarity < 40 | 0.9786 (worse) |
| 10 | S11 | S9 with decision re-tuned on half A (delta +0.25) | 0.9796 |
| 11 | **S12R** | S9 minus 45,610 US/India pairs our independent models (E06/E07) doubt, via score-level combiner | **0.9822** |
| 12 | **S14** | combiner v2 (+E02/S3 votes) removals + 16,691 confident additions (combiner >= 0.95) | **0.9825 (best)** |
| - | S15 | S14 with US strictness re-calibrated (not submitted) | expected 0.9826-0.9828 |

## 2. Pipeline (final file = S14)
- Candidates: 0.977 pipeline blocking = per-state char-3gram name + address TF-IDF (top 40) ∪ IDF inverted-index blocker with
  learned transliteration/abbreviation substitutions -> learned LightGBM cut to 8 per S1 (recall@8 OOF India 98.85%, US 99.08%).
  13,860,352 candidate pairs.
- Matcher: 0.977 model (2-stage LightGBM, 74/89 features) -> scores for all test pairs.
- Combiner (held-out trained): LightGBM over [0.977 score, stage-1 score, ranker score, cut rank, rank/gap in S1,
  E06 prob, E07 prob, E02 vote, S3 vote]; used for REMOVALS (US/India) and confident ADDITIONS (>= 0.95).
- Decision: expected-F0.5 top-n per S1 + one owner per S2/S3 record.
- France: 0.977 decisions + validated house-number rule (never touched by our US/India-trained models).

## 3. Key validation / held-out results
| Experiment | Result |
|---|---|
| E02 OOF | 0.9776 |
| S3 OOF | 0.9806 (leaderboard 0.956: density-sensitive features overfit) |
| E06 / E07 OOF (plain / density-stressed alpha 1.9) | 0.97986 / 0.9790 ; 0.97988 / 0.97906 |
| 0.977 model half B (their held-out) | India 0.98358, US 0.98484 |
| Re-tuned decision (S11) half B | India +0.00006/+0.00012, US -0.00006/+0.00003 (neutral; confirmed on LB) |
| Combiner v1, removals only (S12R) half B | India +0.0012/+0.0015, US +0.0009/+0.0012 -> leaderboard +0.0026 (2.9x amplification) |
| Combiner v2 removals half B | India +0.00127/+0.00156, US +0.00079/+0.00126 |
| Additions by combiner score (half B) | >= 0.95: 96.6-99.3% true; 0.90-0.95: 93.6%; 0.5-0.9: 61-83% |
| US strictness re-calibrated (S15) half B | US +0.00095/+0.00138 vs +0.00079/+0.00126 |

## 4. Rejected ideas (measured, not guessed)
| Idea | Evidence | Verdict |
|---|---|---|
| France name-dissimilarity removal (S10) | leaderboard -0.0010 (~93% of removed pairs were true) | rejected |
| House-number rule for US/India | weak pairs 46-54% true regardless of number (train) | rejected |
| Off-by-one numbers are adversarial | accepted pairs with diff 1-2: US 96.1% / India 99.0% true | rejected |
| Word swap = false match ("Team Ecole" vs "Team Services") | accepted single-word swaps 99.4% true (services 99.2%) | rejected |
| MD plan R2 (core sim < 0.65 at same address -> reject) | half B -0.0706 India / -0.0154 US | rejected |
| MD plan R3 (number guard with low core sim) | half B -0.0251 / -0.0060 | rejected |
| MD plan R5 (singleton guard 0.72/0.58) | half B 0.0000 / -0.0003 | rejected |
| Soft meta-model replacing hard AND (train-fitted) | +0.0029 held-out but calibrated to LB -0.0009..+0.0001 | rejected |
| Restore best pair for S1 emptied by removals | 60% of emptied S1 truly no-match; best variant +0.00005 | rejected |
| Extra char-3gram name+address blocking pass | recovers 32% (India) / 52% (US) of blocking misses but ~80 ms per S1 (~17 h capped) | deferred (compute) |
| Dense embeddings (BGE-M3 / MiniLM) | 14M names; 4-8 h GPU + retrain | deferred (compute); est. +0.0003..+0.0015 |
| Domain/email/bracket normalisation | 0.977 model recall on domain 98.9%, email 98.6% vs 97.0% baseline | already handled |

## 5. Lessons
1. Validation on train does not rank test changes: test has ~1.9x more ownerless near-duplicate distractors and an unseen
   country; every change was checked on held-out data AND calibrated with leaderboard anchors.
2. Removing low-confidence pairs that independent models doubt is the strongest test-side lever (amplified ~2.9x on test).
3. A rule is only safe if its reference is unbiased: consensus pairs reproduced true-match rates for house numbers (worked)
   but not for name similarity (S10 failed).
4. Unknown country: never let models trained on other countries loosen France; only validated France-specific rules.

## 6. Artefacts
- Files: `S9_new_minus_fr/`, `new/S12R/`, `new/S14/`, `new/S15/` (each with matching_results.tsv + candidate_pairs.tsv).
- Scripts: `new/tune_0977.py`, `new/stack_0977.py`, `new/stack_v2.py`, `new/calib_alpha_light.py`, `new/plan_md.py`,
  `src/diag.py`, `src/setprof.py`, `src/exp_lsh*.py`, `src/exp_meta*.py`.
- Held-out results: `new/*_halfB.json`, `new/calib_alpha.json`; audits: `docs/audit_s5.md`, `docs/audit_new_methods.md`;
  experiment ledgers: `work*/experiments/log.jsonl`, `scorecards.md`.
