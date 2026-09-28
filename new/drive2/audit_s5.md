# Audit: submission 5 (S5 = E02 ∩ S3 ∩ E06) (Sep 26, ~21:00 IST)

Leaderboard: E01 0.953 · E02 0.964 · S3 0.956 · S4 (E02 ∩ S3) 0.969 · **S5 = 0.971** (+0.002; my pre-submission estimate was 0.972–0.975).
File: `D:\Amazon ML\S5_all3\matching_results.tsv` (+ `candidate_pairs.tsv` = union of the E06 and S3 candidate sets).
Open item: confirm which file was uploaded. `S5_all3` (this audit) or `S5_best` (= S5_all3 ∩ the other PC's S5; 5,656,125 pairs). The conclusions in §4 are computed for S5_all3.

## 1. What S5 is
| Step | What | Result |
|---|---|---|
| E06 model | New normalization (postal codes separated from house numbers; dotted legal forms S.A.R.L. -> sarl; "Holding"/"Groupe" visible to legal-form features), new exact-key blocking passes, density-sensitive features dropped (`cvc_*`, `g_same_*`, raw counts), ownerless negatives weighted x1.9, decision tuned on density-stressed macro F0.5 (alpha 1.9) | OOF macro F0.5 **0.9799** plain / 0.9790 stressed, both folds 0.97903 (E02 0.9776, S3 0.9801). Blocking recall 97.51% -> **97.63%** (+9,004 true pairs). Stage 1 only (stage 2 crashed twice on memory). |
| Vote | Keep a pair only if E02, S3 **and** E06 all predict it | S4 minus 66,330 pairs; nothing added |
| Output | 1,732,544 rows; 5,670,143 pairs (3.273/S1); 101,890 empty (5.88%) | Official validator `--check-ids` PASS; `check_output.py` PASS (no id to >1 S1, matches ⊆ candidates, row order = test_source1) |

## 2. Why this candidate was chosen (evidence at decision time)
E06 vs S4: +67,067 added, -66,330 removed. Label-free profile of each set (share of pairs whose first house number differs; train references: true pairs US 0.112 / India 0.206, model false positives US 0.349 / India 0.371):

| Set | France | India | US |
|---|---|---|---|
| Added by E06 (not used) | 0.238 | 0.439 | 0.667 |
| **Removed from S4 (used)** | **0.637** (61% differ only by a legal suffix) | 0.515 | 0.443 |

Both disputed sets looked distractor-like, so the rule "add nothing, remove the disputed pairs" (the S4 pattern, +0.005 last time) was applied.

Per-country prediction profile S4 -> S5 (France moved in line with the training countries):
| Country | Empty rate | Matches/S1 |
|---|---|---|
| France | 5.04% -> 5.38% | 3.39 -> 3.29 |
| India | 5.97% -> 6.03% | 3.26 -> 3.24 |
| US | 5.84% -> 5.90% | 3.34 -> 3.31 |

## 3. Findings
| # | Finding | Evidence | Impact | Type |
|---|---|---|---|---|
| F1 | **The label-free precision estimator is too pessimistic on disputed pairs** | It estimated ~0% of the 66,330 removed pairs were correct; the leaderboard gain (+0.002) implies ~55–60% were correct (§4) | HIGH: it over-recommends removals. The consensus lever is nearly exhausted | Silent (fixed by recalibrating on this result) |
| F2 | True matches the models disagree on look like distractors | Hard true pairs carry number typos / missing addresses, so their mismatch rate is far above the average true pair | HIGH: a single-feature mixture cannot separate them | Structural |
| F3 | Estimate missed: 0.971 vs 0.972–0.975 forecast | §4 | MEDIUM: future forecasts use the recalibrated removal precision | Process |
| F4 | Stage 2 not used | Two native crashes (memory) on the 16 GB laptop; stage 2 was worth ~+0.0005 before | LOW | Resource |
| F5 | France calibration not applied | `calib.py loco` (US -> India simulation) could not allocate memory; no unverified change was shipped | MEDIUM: France still slightly over-matches in E06 alone (4.88% empty vs 5.8%) | Resource |
| F6 | ~1 h lost to crashes | 3 native segfaults (intermittent, retries added), 3 out-of-memory (raw text columns kept during features, fixed; diagnostics loading full tables, fixed with column-only reads), 1 own bug (density metric allocated 128 TB inside a group aggregation; fixed with a join, verified at full scale in 0.6 s), 1 path-quoting bug | Time only; no wrong output shipped | Operational |
| F7 | Normalization gaps found by substitution mining | US state codes IN/OR not recognized (32k predicted pairs affected), Saint vs St (10k), French abbreviations (bis/b 5k, quai/q, ter/t, appartement/appt), Indian house-number prefixes (plot/flat/door/shop, thousands), name variants (freres/frs, laxmi/lakshmi, jay/jai) | Recall + address similarity; fixed in E07 (running) | Silent before S5 |

## 4. What the 0.971 proves (derived)
Under F0.5, removing a false merge gains ~+0.20 for that entity; removing a true match costs ~-0.06. With 66,330 removed pairs (~1 per affected entity) and a gain of +0.002 x 1,732,544 ≈ 3,465 entity-units:
66,330 x (0.26 f - 0.06) = 3,465  ->  **f ≈ 0.43** (share of removed pairs that were false merges).
For S4 the same calculation gave f ≈ 0.81. **Each new round of agreement filtering removes a less wrong set**: pairs still disputed after two filters are close to the F0.5 break-even (f ≈ 0.23 -> neutral; removal helps only when f > ~0.23 at this per-pair cost ratio, and the marginal gain shrinks fast).

## 5. Decisions derived from this audit
1. **Stop adding removal-only filters** unless a disputed set is far more distractor-like than this one (recalibrated threshold: house-number mismatch well above the removed-set levels of §2, not just above the train false-positive reference).
2. **Next gains must come from recall / better scoring, not from agreement**: E07 (mined normalization fixes, full retrain, running) targets pairs the models currently miss.
3. Additions are judged against the recalibrated estimator; France changes only with the unseen-country simulation (`calib.py loco`, reduced sample to fit memory).
4. Keep every go/no-go check: official validator, `check_output.py`, per-country profile, profile of added/removed pairs vs the confirmed best (now S5, 0.971).
