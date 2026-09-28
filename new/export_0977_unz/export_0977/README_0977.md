# Export of the 0.977 submission (team SchruteFarmCoders, "EXP-1")

## 1. Which pipeline produced the 0.977 files
Verified against the submitted files in `hv_repo/output/v2_lgbm_two_stage/`:

| Check | Expected | Found |
|---|---:|---:|
| `matching_results.tsv` rows | 1,732,544 | 1,732,544 |
| Matched pairs | 5,819,086 | 5,819,086 |
| Empty rows | 97,781 | 97,781 |
| First data line | `S1-714132312<TAB>S2-187020300,S2-637340732,S3-625880872,S3-867809779` | identical |
| `candidate_pairs.tsv` pairs | 13,860,352 | 13,860,352, exactly 8 per S1 |

| What | Path (on this PC, under `AMAZON_ML/`) |
|---|---|
| Code | `hv_repo/code/business_entity_resolution/src/` (git commit `25578de`, branch `patel`); copy in `code/` |
| Run script | `hv_repo/code/business_entity_resolution/src/run_patel.sh` (steps 0–5; EXP-1 = step 5) |
| Second candidate source | `code/business_entity_resolution/work_e04/{train,test}/{records,cands}.parquet`, produced by `hv_repo/code/patel_blocker/` (`OUR_CANDS_DIR`) |
| Work dir / models | `hv_repo/work/exp1/`: `v2/ranker.txt` (learned cut), `v2/stage1.txt`, `v2/stage2.txt` (LightGBM), `v2/config.json` (decision), `v2/word_lists.json` |
| Log | `hv_repo/work/run_patel.log` |

## 2. Files in this folder
| File | Content |
|---|---|
| `test_scores.parquet` | All 13,860,352 test candidate pairs. Columns: `source1_entity_id`, `candidate_entity_id`, `score` (final match probability = `stage2_score`, before any decision), `stage1_score`, `stage2_score`, `ranker_score` (learned-cut ranker probability), `cut_rank` (1–8 within S1), `country`. **Re-scored** with the saved models (the original run did not save scores; no retraining). |
| `train_oof_scores.parquet` | 3,200,000 training pairs (400k training S1 × 8 candidates). Columns: `source1_entity_id`, `candidate_entity_id`, `score` (= `stage2_score`), `label`, `fold` (0–4), `stage1_score`, `stage2_score`, `ranker_score`, `cut_rank`, `country`, `held_out`, `heldout_half`, `name_sim`, `addr_sim`, `via`. See the OOF notes below. |
| `decision.json` | The exact score → match rule. |
| `sanity_checks.json` | Reproduction check (section 5). |
| `pip_freeze.txt` | Library versions (Python 3.12.14, macOS arm64). |
| `code/` | `code/business_entity_resolution` + `code/patel_blocker` exactly at commit 25578de, plus `export_0977.py` (this export). |

**How out-of-fold the train scores are:**
- `score` is **OUT-OF-FOLD**: `sklearn.model_selection.GroupKFold(5)` with groups = `source1_entity_id`, for both stages.
  - No S1 is in both the training and validation part of the same model.
  - `fold` is recomputed with the identical call on the identical row order.
- **Sampling:**
  - 200k training S1 drawn uniformly at random (polars `sample(seed=42)`) from all India+US training S1: `held_out=True`, the half A/B evaluation set.
  - Plus 200k more from the rest (`seed=43`): `held_out=False`, training only.
  - Not label-dependent: S1 with no true match are included.
  - **All 8 candidates of each S1** (3,200,000 = 400,000 × 8) come from the same blocking → union → learned cut K=8 used on test.
  - The one difference: training S1 are cut with an out-of-fold ranker, test S1 with the ranker trained on all training S1.
- Stage 2 uses stage-1 OOF scores as inputs.
- `ranker_score` / `cut_rank` for training S1 come from an **out-of-fold ranker** (`OOF_CUT=1`, `v2/ranker_oof.parquet`).
- The patel blocker's token-substitution map was learned from true pairs of half the training S1 (even id bucket: `numeric id // 10000 % 10000 % 2 == 0`).
  - Leak-free subset for evaluation: `held_out & heldout_half == 1 & bucket odd`.
- Everything else is label-free.
  - Word lists (`word_lists.json`) come from house-number agreement on candidates, not from the ground truth.
  - The Indic dictionary and normalisation use no labels.
- `country` column: included.

## 3. How it was made
**Candidate generation**, per country, per S1:
1. **Harsh V v4 blocking.**
   - Per-state TF-IDF: char 3-grams of the compact core name (top-30) and address tokens (top-30), plus 10 cross-state name candidates.
   - Merged and scored (`score` = name_sim + addr_sim), keeping the **top-40**.
2. **∪ the "patel" S2/S3→S1 blocker**, flipped to S1→candidate:
   - IDF inverted index over name/address unigrams, bigrams and compact keys, with learned transliteration/abbreviation substitutions.
   - Low-confidence records get their top-200 re-ranked and keep 8; confident records keep 3; relative pruning 0.8.
   - Gives `our_rank` / `our_rr`, and `via` bit 4.
3. **Learned cut.** A LightGBM binary ranker (300 rounds, lr 0.1, 63 leaves; features: similarities, ranks, source flags, state relation, key equality, `our_rank`, `our_rr`) scores the union and keeps the **top 8 per S1**.
   - **Why 8:** `pick_k.py` picks the smallest K in {8, 10, 12, 15} whose learned-cut recall is at least the raw top-15 recall in both countries. K=8 qualified (India 0.971, US 0.981 before the union).
   - The organisers also stated that smaller candidate sets rank higher.
   - With the union, recall@8 (OOF) is India 0.9885, US 0.9908.

**Matcher: two-stage LightGBM, binary logloss.**
- Parameters: lr 0.05, 127 leaves, min_child_samples 50, feature_fraction 0.7, bagging 0.8, L2 1.0, seed 7, deterministic, **600 rounds**. No class weights: about 43% of the training pairs are positive after the cut.
- **Stage 1 (74 features):**
  - 23 base pair features (`lgbm_matcher.FEATURES`): name/address similarities, blocking score/rank/via, legal and number relations, core-name ratios, key equality, gap to best, n_close.
  - 51 `features_v2`: entity/legal type changes, compound house numbers, first-token/first-number, invented/acronym/web names, decoy/noise words, IDF-weighted shared/unshared words, state relation, namesake/crowding counts, group features.
  - Indic / USPS / AFNOR normalisation, 3 name views.
- **Stage 2 (89 features):** stage-1 features + 15 group features from the stage-1 probabilities of the same S1's other candidates (rank, max, second, gap, counts ≥0.5 / ≥0.8, sources, top candidate's flags, deviation from the group's majority).
- **Training data:** 400k training S1 (India + US only; train has no France), sampled with seeds 42 and 43. That is 200k "val" + 200k "extra", × 8 candidates = 3.2M pairs.
- **Folds:** 5-fold GroupKFold by S1 for OOF scores. The final models are trained on all 3.2M pairs.

**Decision.** See `decision.json`.
- Per S1, the expected-F0.5 top-n rule, which may predict nothing.
  - India: miss rate 0.0. US: miss rate 0.2.
- Then one owner per S2/S3 candidate (highest score wins).
- **Chosen on half A** of the 200k held-out "val" S1 (numpy `default_rng(0)` halves) and **reported on half B**, a held-out half never used for any choice:
  - India 0.98358, US 0.98484.
- The rule (threshold vs expected-F), threshold (0.30–0.95, step 0.025) and miss rate ({0, 0.05, 0.1, 0.2}) were all chosen on half A.

**France (unseen country).**
- No French labels exist.
- The same models are used; features are country-agnostic. French legal forms and AFNOR address abbreviations are normalised.
- France gets the stricter labelled setting, India's: expected-F with miss 0.0.
- No France-specific rules or post-processing in this file.

**Ensembling / post-rules:** none.

## 4. Reproduce
```bash
cd hv_repo && python3.12 -m venv .venv && .venv/bin/pip install -r code/business_entity_resolution/requirements.txt   # exact versions: pip_freeze.txt; macOS needs `brew install libomp`
# 1) patel blocker candidates (see code/patel_blocker/README.md), then:
cd code/business_entity_resolution/src
OUR_CANDS_DIR=<AMAZON_ML>/code/business_entity_resolution/work_e04 PATH=../../../.venv/bin:$PATH bash run_patel.sh 0
#   -> output/v2_lgbm_two_stage/{matching_results,candidate_pairs}.tsv   (EXP-1; v1_* is EXP-0)
# re-export scores without retraining:
WORK_DIR=../../../work/exp1 MODEL_K=8 python export_0977.py test && WORK_DIR=../../../work/exp1 MODEL_K=8 python export_0977.py check
```
Runtime is about 3 h on a 10-core / 16 GB Mac.

## 5. Sanity checks
SANITY_PLACEHOLDER
