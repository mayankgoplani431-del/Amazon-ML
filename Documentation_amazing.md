# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** amazing
**Team Members:** [FILL: team member names]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

A two-tier, country-agnostic pipeline: (1) high-recall multi-pass blocking (sparse TF-IDF top-K passes in both directions plus exact-key passes, run per country label as an open set), pruned by a LightGBM ranker to 20 candidates per Source-1 entity; (2) a LightGBM pair matcher with a second-stage re-scorer and an expected-F0.5 subset decision under a one-owner-per-record constraint. Key innovations: training and threshold tuning **at the test set's distractor density** (test has ~1.9x more ownerless near-duplicate records per S1 than train), normalization that makes the unseen country (France) look like the training distribution (postal codes, dotted legal forms, invisible legal suffixes), a label-free strictness calibration for the unseen country validated by a leave-one-country-out simulation, and a multi-model agreement ensemble.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** train 2.21M S1 vs 10.32M pool records (S2+S3); test 1.73M S1 (US 663k, India 810k, **France 259k, unseen**) vs 9.97M pool.
- **Structure verified on train ground truth:** every S2/S3 record belongs to **at most one** S1; country always agrees within a true pair; ~26% of pool records belong to no S1 (distractors); 5.6% of S1 have no match; mean 3.46 matches per S1. US and India share the *same* match-count prior, so the generator's prior is country-independent.
- **Noise in true matches:** typos, leetspeak (C0ok, 5ervices), legal-form variants (Pvt/Private, S.A.R.L./SARL), honorifics (Smt, Shri, M/s, Dr), appended ids ((ID: 123), #71076), DBA/formerly aliases, domain-style names, Indic-script names (~7% of pool names), address abbreviations, truncated addresses (house number or components missing), empty addresses (~4.5%), state full name vs abbreviation, region vs department (France).
- **Distractors** are near-copies of an S1: house number shifted (80 vs 89), legal suffix added (Inc/Corp/Holding), last name token changed, different name at the same address.
- **Train vs test shift (decisive finding):** pool records per S1 are 4.67 on train but 5.5-5.8 on test in every country, i.e. ~1.2 vs ~2.3 ownerless distractors per S1. A model tuned on train is over-confident on test. Evidence from the leaderboard: a model with higher validation F0.5 but density-sensitive group features scored *lower* on test, while intersecting two models' predictions (removing 1% of pairs) *raised* the test score by +0.005, i.e. most disagreement pairs were false merges.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + entity-level expected-F0.5 decision + agreement ensemble.
**Core Innovation:** density-aware training/validation (ownerless negatives weighted by the measured test/train density ratio, density-sensitive features removed, thresholds tuned on a density-stressed macro F0.5), and unseen-country alignment (normalization + label-free prior-profile calibration, validated by leave-one-country-out).

---

## 3. Candidate Generation (Blocking)

All passes run **separately per country label** (whatever labels appear, open set; country is never a model feature) and are unioned:

| Pass | Representation | Direction / K |
|---|---|---|
| comb | word TF-IDF of core name + normalized address | S1 -> pool, top 50 |
| addr | word TF-IDF of normalized address | S1 -> pool, top 25 |
| name_ch | char 4-gram TF-IDF of space-less core name (typos, glued/domain names) | S1 -> pool, top 12 |
| rev | word TF-IDF name+address | pool -> S1, top 5 ("which S1 does this record resemble most") |
| nameaddr / rev_na | whole core name as one token + address tokens | both directions, top 20 / 3 |
| k_name_num, k_name | exact keys (core name + house number; near-unique core name) | size-capped joins |
| k_sorted_num, k_nosp_num | exact keys on word-sorted name / space-less name + house number | size-capped joins |

- Sparse top-K via `sparse_dot_topn`, 100k S1 per chunk; never materializes dense matrices; O(N) memory in the pool.
- **Pruning ("Model 1"):** LightGBM ranker on pass scores/ranks + six cheap similarities keeps the **top 20 per S1**. On train it keeps 99.86% of the true pairs found by the union (union recall 97.65% -> pruned 97.51%) while cutting ~92 pairs/S1 to 20.
- **Candidate pairs generated (test):** 34.6M per model (20.0 per S1); the ensemble's final candidate file (union of the models' candidate sets) has **38,871,509 pairs (22.4 per S1)**. Reduction ratio vs same-country all-pairs (6.72e12): 0.999994.
- **Blocking recall (train, E06):** 97.63% of all true pairs are in the final top-20 (E02: 97.51%); the new keys and postal-code handling recover +9,004 true pairs.
- **Ensemble candidate file:** the final `candidate_pairs.tsv` is exactly the set the models scored: the union of the candidate sets of the models in the ensemble.

---

## 4. Matching Model

**Features (~60, all country-agnostic):**
- Name: rapidfuzz ratio / partial / token-sort / token-set, Jaro-Winkler, Levenshtein, space-less ratios, best alias (DBA) match, token Jaccard/containment, per-country TF-IDF cosine, per-country relative name frequency, legal-form sets (equal / only on one side, incl. Holding/Groupe), first/last token equality, extra/missing tokens.
- Address: ratios, token sets, per-country TF-IDF cosine, house-number equality / edit distance / absolute difference (distractor shift), number-set Jaccard, **postal-code equality (postal codes separated from house numbers)**, state/region equality, street+city words without numbers, empty-address flags.
- Context: blocking pass scores/ranks, pruning score and its rank/gap within the S1 group and within the candidate's group; stage-2 adds the same competition statistics of stage-1 probabilities.
- Removed for robustness: absolute counts and candidate-vs-candidate density features (they shift with test density).

**Normalization:** Unicode to ASCII, learned Indic-script -> English word dictionary (from train ground truth only), abbreviation/legal-form canonicalization, dotted acronyms collapsed (S.A.R.L. -> sarl), honorifics/ids/aliases handled, state/region components separated (US states, Indian states, French regions and departments).

**Model type:** LightGBM (MIT), stage 1 (255 leaves, 1000 rounds) + stage-2 re-scorer; 2-fold grouped out-of-fold by S1 over all 2.2M train entities. Negatives whose record belongs to no S1 weighted x1.9 (measured test/train distractor density ratio).

**Threshold selection method:** exclusivity (each S2/S3 record assigned only to its best S1), then per S1 the top-k set maximizing plug-in expected F0.5 (empty set chosen when P(no match) wins); parameters tuned on out-of-fold predictions with a **density-stressed macro F0.5** (a false match on an ownerless record counts 1.9x). For France: one logit offset chosen label-free so its predicted empty-rate / matches-per-S1 profile moves toward the in-domain profile; the fraction was chosen on a leave-one-country-out simulation (train US -> apply to India with India's labels hidden).

**Ensemble:** agreement vote over decorrelated models (E02, S3, E06), accepted only when the changed pairs pass a label-free precision estimate (house-number-mismatch mixture of train true-pair vs train false-positive profiles; F0.5 break-even precision ~0.77).

---

## 5. Results & Error Analysis

| Version | Validation macro F0.5 (OOF, 2.2M S1) | Leaderboard |
|---|---|---|
| E01 baseline | 0.9684 | 0.953 |
| E02 + name-as-token and exact-key blocking | 0.9776 | 0.964 |
| S3 + group/density features | 0.9806 | 0.956 |
| S4 E02 ∩ S3 agreement | n/a | 0.969 |
| **E06** density-aware training + France/postal normalization (stage 1) | **0.9799** (density-stressed: 0.9790; both folds 0.97903) | used in S5 |
| **S5** E02 ∩ S3 ∩ E06 agreement | n/a | **0.971** |
| S6 = S5 + unseen-country (France) calibration | n/a | 0.971 |
| **Final submission: S5** | E06 0.9799 | **0.971** |

- **F_0.5 Score (macro):** leaderboard **0.971** (final, S5); best single-model OOF validation 0.97988 (E07) / 0.97986 (E06).
- **Common false positives:** ownerless near-copies (house number shifted, legal suffix added, one name word changed at the same address); records owned by another S1 with the same name; in France, names sharing only the city word.
- **Common false negatives:** blocking misses (~2.5% of true pairs: Indic-script names with truncated or empty addresses), empty-address candidates, number typos, address copies missing the house number.

---

### Explored, measured, not adopted
| Idea | Measured result | Decision |
|---|---|---|
| Extra char-3gram name+address blocking pass (LSH-style) | Recovers 32% (India) / 52% (US) of the true pairs blocking misses (union recall 97.75% -> 98.68%), but costs ~80 ms per S1 uncapped / 14-19 ms with a common-3gram cap (~17 h at full scale on 16 GB) | Deferred: not feasible on the available hardware before the deadline |
| Soft meta-model over model probabilities/votes instead of hard AND | +0.0029 macro F0.5 on held-out train, but it re-adds the disputed pairs whose removal gained on the leaderboard; calibrated to the leaderboard it is ~-0.0009 to +0.0001 | Rejected (train-fitted meta trusts disagreements too much at test density) |
| Unseen-country strictness calibration (France) | Leave-one-country-out (train US -> India): +0.0065 to +0.011 F0.5 on the simulated unseen country; leaderboard S6 = 0.971 (no measurable change) | Validated, neutral; not in final |
| Sentence-embedding similarity (MiniLM) | Not measurable within memory/time (21.6 GB of embeddings) | Not attempted |
| Normalization rules mined from predictions (US state codes IN/OR, French abbreviations, Indian house-number prefixes) | E07 OOF 0.97988 vs E06 0.97986 (no measurable change) | Neutral |

## 6. Conclusion

Blocking recall and precision under distribution shift decided the score more than model capacity: the largest leaderboard gains came from recall-oriented blocking passes and from making training, thresholds and the unseen country consistent with the test distribution. Lesson: validate against the *test* conditions (density, unseen country) with label-free checks, not only against a random train split.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/` (all code), `scripts/run_e06.sh` (end-to-end), `README.md`, `requirements.txt` (pinned; LightGBM, polars, scikit-learn, rapidfuzz, sparse-dot-topn, anyascii; all MIT/BSD/Apache). Entry points: `prep.py` (normalize) -> `candidates.py` (blocking) -> `run_train.py` (pruning, features, stage-1/2, OOF scorecard) -> `run_test.py` (test scoring, writes both TSVs) -> `calib.py` (unseen-country calibration) -> `vote.py` (ensemble) -> `check_output.py` + official validator. No external data, APIs or geocoding; no test labels; ~16 GB RAM, ~5 h.

### B. Additional Results
Per-country prediction profile (label-free check that the unseen country behaves like the training countries):

| Country | Empty rate S4 -> S5 | Matches/S1 S4 -> S5 |
|---|---|---|
| France | 5.04% -> 5.38% | 3.39 -> 3.29 |
| India | 5.97% -> 6.03% | 3.26 -> 3.24 |
| US | 5.84% -> 5.90% | 3.34 -> 3.31 |

Pairs where the models disagree look like distractors (house-number mismatch rate of disputed pairs 0.44-0.67 vs 0.11-0.21 for true train pairs and 0.35-0.37 for train false positives); the agreement ensemble therefore drops them. In France 61% of the dropped pairs differ only by a legal suffix ("Holding") that the earlier normalization made invisible.

Data-driven normalization mining (single-word substitutions inside predicted pairs) surfaced: US state codes IN/OR not recognized (32k pairs), Saint vs St, French address abbreviations (bis/b, quai/q, ter/t, appartement/appt), Indian house-number prefixes (plot/flat/door/shop/house), name variants (freres/frs, laxmi/lakshmi, jay/jai).
