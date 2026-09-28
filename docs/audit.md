# Pipeline audit: silent failure modes (Sep 26, ~04:00 IST)

Leaderboard: E01 = 0.953, **E02 = 0.964**, **E03/S3 = 0.956**. S3's validation was higher (0.9806 vs 0.9776) but its leaderboard score was lower. That combination only happens when validation stops representing test. Every finding below was measured, not assumed. Code paths are relative to `sub/code/business_entity_resolution/src/`.

## Ranked summary

| # | Issue | Where | Impact | Type |
|---|---|---|---|---|
| 1 | **Validation doesn't represent test's distractor density** (pool/S1 4.67 train vs 5.5–5.8 test). S3's new density-sensitive features over-accept shifted-house-number distractors on test | `features.py: group_features` (cvc_*, g_same_*), `entity.py`, `decide.py` params | **HIGH** (explains the −0.008) | **Silent** |
| 2 | Stale-cache reuse: stages skip when output files exist, even if the code or config changed | `pipeline.run_features` (skips existing feat parts), `pipeline.run_stage0` (skips if pr exists), `run_test` (skips blocking if any cand part exists), `candidates.run` (doesn't clear old parts) | **HIGH** risk (none triggered yet, verified) | **Silent** |
| 3 | Decision rule and entity-rule parameters are tuned on the same out-of-fold predictions they are scored on | `run_train.sweep` → `report`; `entity_tune.py` | MEDIUM (small optimism, ~0.0003–0.001) | Silent |
| 4 | No France (unseen country) in validation; the proxy covers only US↔India | `run_train.py`, `loco.py` | HIGH for leaderboard, can't be measured directly | Silent |
| 5 | Transliteration dictionary learned from all train ground truth, validation folds included | `translit.build` | MEDIUM-LOW (India native-script names look easier in validation) | Silent |
| 6 | Absolute-count features shift with split size (US p90 log name frequency 5.47 train vs 4.90 test; `s0_p_n`, `g_size`, `n_cand`) | `features.py` (n_freq_q/c), `stage0.cheap_features` (g_size), `pipeline.group_stats` (`*_p_n`), `entity.aggregates` (n_cand) | MEDIUM | Silent |
| 7 | Absolute `max_df=20000` in blocking TF-IDF prunes different tokens depending on country size (France pool 1.4M vs India 4.7M) | `candidates.CFG` (W) | MEDIUM (blocking recall differs by country) | Silent |
| 8 | Empty strings score as identical: `fuzz.ratio("","") = 100`, `token_sort_ratio("","") = 100` | `features.py` (n_ratio, n_tsort, a_ratio, a_tsort, cvc_*), `stage0.cheap_features` | LOW (measured: empty core name ≤0.2% of records; S1 addresses are never empty) | Silent |
| 9 | Train normalized with old code, test re-normalized with new code (state bug fix) | `work/cache/train_s*.parquet` vs `test_s*.parquet` | LOW (measured: 0.05% of records differ) | Silent |
| 10 | Test blocking ran on old normalization; test features/stage-0 ran on new normalization | S3 run order | LOW | Silent |
| 11 | Second-order leakage in stage-2 candidate-side stats (a competitor S1's p1 comes from a model trained on the current S1's fold) | `pipeline.group_stats` on OOF p1 | LOW | Silent |
| 12 | Fold assignment uses `pl.Expr.hash`, which isn't guaranteed stable across polars versions | `pipeline.fold_of` | LOW (reproducibility only) | Cosmetic |
| 13 | `FEATURES_ONLY` flag file makes `run_test` exit without writing outputs | `run_test.main` | LOW (currently removed) | Silent if left behind |

## 1. Blocking / candidate generation

**1a. Zero-candidate S1 entities: blocking bug vs genuine singleton.** Measured: test `candidate_pairs.tsv` has **0 empty rows** for E02/S3 (E01 had 131). Every S1 gets ≥1 candidate, so no S1 is silently dropped. How to tell a bug from a real singleton: for each zero-candidate S1, check (a) whether `name_core` and `addr_norm` are empty after normalization (a normalization bug), and (b) run exact-key lookups (name_core, num1) against the pool. A hit means a blocking bug; no hit on any key plus a unique name means a genuine singleton. On train, blocking misses are measured directly: 2.49% of true pairs never become candidates (E02 final set; 2.35% before pruning).

**1b. Key sensitivity (case/punctuation/encoding).** Keys use normalized fields: lowercase, anyascii, punctuation→space, `pvt/pvt./PVT`→`private`→removed as a legal word, leetspeak fixed. Verified: "Pvt." / "Pvt" / "PVT" land in the same key. Remaining sensitivity: exact keys (`k_name_num`, `k_name`) break on any typo or word swap in the name. Mitigated by fuzzy TF-IDF passes, plus the new `k_sorted_num` and `k_nosp_num` passes (coded for E04). The first-number key also breaks when the first number is a unit or flat number rather than the house number.

**1c. Country handling.** No hardcoded country list in blocking: `candidates.run` loops over `q['country_n'].unique()`, and the country is never a model feature. France is processed (3 chunks, ~90 candidates/S1 before pruning). One country-specific component: `normalize.STATE_LOOKUP` (US/India states; France regions/departments added in S3). An unknown 4th country would get no state stripping, which degrades gracefully. Risk (issue 7): absolute `max_df` interacts with country size.

**1d. Leakage between passes.** No pass uses ground truth. Reverse passes (`rev`, `rev_na`) use all S1 records of the split, which is legitimate (no labels) and the same at test time. Stage-0 pruning is a model trained out-of-fold on train labels and applied with fold-averaged models on test: no leakage.

**1e. Duplicates within a candidate list.** Each pass yields unique (qi, pi) pairs (top-K per row; key joins on unique ids). The union is a full outer join on (qi, pi), and `run_test.write_lists` also applies `.unique()`. Verified: `check_output.py` reports no duplicates.

**1f. Is candidate_pairs.tsv the exact model input?** Yes. `candidate_pairs.tsv` is written from the `pr_test` parts; features are built from exactly those parts (`run_features`); the models score only feature rows; the decision rule only removes rows. So matches ⊆ candidates always, verified by `check_output.py` (0 violations). Caveat: issue 2 (stale caches) could break this silently if `pr` and `feat` come from different runs.

**1g. Reduction ratio and pairs completeness, computed.**
- Total possible pairs (within the same country only, since positives always share a country) = Σ_c |S1_c|·|pool_c| = 663k·3.82M + 810k·4.72M + 259k·1.44M ≈ 2.53e12 + 3.82e12 + 0.37e12 = **6.72e12**. Across countries it would be 1.73M × 9.97M = 1.73e13.
- Final candidate pairs (test) = **34.6M** (20.0/S1). **Reduction ratio = 1 − 34.6e6 / 6.72e12 = 0.999995** (0.999998 against the all-pairs baseline).
- **Pairs completeness (train, E02 blocking) = 7,448,317 / 7,638,365 = 97.51%** after pruning (97.65% for the union before pruning). Computed in `pipeline.run_stage0`: `hits[N]` counts true pairs whose stage-0 rank ≤ N, and `union` counts all true pairs in the blocking union.

## 2. Matching model

**2a. Features using information not available at test time.** None use labels. But several statistics are computed on the split itself (transductive) and scale with its size, so their meaning differs train↔test (issue 6): absolute name frequencies, candidate-group sizes, competition counts. Fix: use size-normalized or rank-based versions only (e.g. `n_freq_rel_*`, group ranks, gaps), and drop the raw counts.

**2b. Identical computation for train and test.** Same functions; differences come only from the data (issue 6) and the normalization-version mismatch (issue 9, 0.05%).

**2c. Empty/null fields.** rapidfuzz returns **100 for empty vs empty** (`ratio`, `token_sort_ratio`), 0 for empty vs non-empty, and `token_set_ratio("","")=0`. The flags `a_empty_q/c` and `n_empty_c` exist, so the model can learn around it, but `cvc_name`/`cvc_addr` and the stage-0 cheap features have no flag. Fix: set similarity to NaN when either side is empty (LightGBM handles NaN natively). Impact is low: empty core names are ≤0.2% and S1 addresses are never empty.

**2d. Threshold tuned on the evaluated data.** Yes. `run_train.sweep` picks the rule on the same OOF predictions it reports, and `entity_tune.py` does the same. With 2–3 parameters over 2.2M entities, optimism is small, but it compounds with issue 1. Fix: tune on fold 0 and report on fold 1 (and vice versa), then report the mean.

**2e. Multiple true matches.** Supported. `decide.expected_f` / `entity.decide` choose the best top-k with k up to the full list. Exclusivity is per *candidate* (each S2/S3 record goes to at most one S1), not per S1, so an S1 can keep many matches. Verified: test predictions average 3.4–3.6 matches per S1.

**2f. Forced matches on singletons.** No "always return top-1" path. An empty prediction is chosen when `empty_scale · P(no match) ≥` the best expected F of any non-empty set. 5.6% of test S1 get empty predictions.

## 3. Output / format
- **Missing or duplicate S1 rows:** impossible. Rows come from the full S1 table (`ids_q`, left join), and `check_output.py` asserts that the set equals test_source1 and has no duplicates.
- **Self-matches or unknown IDs:** candidate IDs come only from the S2+S3 pool table. `validate_submission.py --check-ids` and `check_output.py` both confirm every ID exists in test S2/S3.
- **Match not among candidates** (the official validator only warns): enforced as a hard failure in `check_output.py`; 0 cases.

## 4. Evaluation
- **Held-out?** Models are held out (grouped 2-fold out-of-fold by S1). Unsupervised statistics (TF-IDF IDF, name frequency, blocking) are fit on the whole split. That uses no labels and is the same procedure as test, so it's acceptable. Label-derived artifact fit on everything: the transliteration dictionary (issue 5).
- **Macro vs micro:** `decide.macro_f05` is a true per-S1 macro average over **all** S1, including singletons (1.0 for an empty prediction, 0.0 for any prediction) and S1 with zero candidates (they count as 0 matches predicted). Checked against the README example in its self-test.
- **Representativeness:** validation = US+India, test-like country mix absent, and **distractor density absent** (issue 1). This is the main reason validation (0.9806) overstated the leaderboard (0.956).

## 5. Evidence for issue 1 (S3 vs E02 on test)
| Pairs | Country | Count | House-number mismatch | Exact core name |
|---|---|---|---|---|
| Kept (in both) | US / India / France | 2.22M / 2.64M / 0.88M | 0.106 / 0.203 / 0.029 | 0.56 / 0.71 / 0.70 |
| **Added by S3** | US / India / France | 77k / 93k / 60k | **0.899 / 0.605 / 0.745** | 0.50 / 0.32 / 0.38 |
| Removed by S3 | US / India / France | 16k / 20k / 21k | 0.50 / 0.51 / 0.56 | 0.41 / 0.50 / 0.64 |

The pairs S3 added are dominated by shifted-house-number records, the exact distractor pattern, while the pairs both models agree on look clean. **The pairs where E02 and S3 disagree are low-precision in both directions.**

## 6. Recommended actions (ordered)
1. **Consensus submission (≈10 min, no training):** keep only pairs that **both E02 and S3** predict. This removes S3's ~230k suspicious additions and E02's 58k contested pairs (50%+ house-number mismatch). Under F0.5, dropping a pair helps whenever its precision is below ~0.78. Expected: at or above E02 (0.964), roughly **0.965–0.970**.
2. **Make validation test-like:** reweight training negatives about 2× (to match test distractor density) and tune the decision rule under that weighting. Test via a density-stressed validation: score only S1 groups whose candidate lists contain ≥2 shifted-number near-copies.
3. **Remove or neutralize density-sensitive features** (`cvc_*`, `g_same_*`, raw counts `n_freq_q/c`, `g_size`, `*_p_n`, `n_cand`) or replace them with rank/relative versions. Keep the pairwise distractor features (`num_first_absdiff`, `n_last_eq`, legal-only flags).
4. **Explicit house-number guard:** a feature `num1_mismatch & name/address otherwise near-identical`, and a decision penalty on shifted-number pairs unless the name is unique in its group.
5. **Cache hygiene:** version every cache directory with a hash of (code version + config), and fail loudly instead of reusing on mismatch (issue 2). Clear old `part_*` files in `candidates.run`.
6. **Empty-string NaN fix** (issue 8), **fold-split decision tuning** (issue 3), **dictionary built on train-fold only** (issue 5) for honest validation.
