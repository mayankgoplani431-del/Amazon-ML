# Business Entity Resolution: Amazon ML Challenge 2026

For every Source-1 (S1) business record, find all matching Source-2/Source-3 records (0..many).
Metric: **macro F0.5 per S1 entity** (singleton S1: empty prediction = 1.0, any prediction = 0.0).
Train covers US + India. Test covers US + India + **France (no training data)**.

> ⚠️ This repo contains **code + documentation only**. The competition dataset, caches, models and
> submission files are git-ignored and must never be committed. Keep this repo **private** (fair-play rules).

## Results so far

| Version | What changed | Validation F0.5 (OOF, 2.2M train S1) | Leaderboard |
|---|---|---|---|
| E01 | 4 blocking passes, ~50 features, LightGBM stage-1 + stage-2, expected-F0.5 decision | 0.9684 | 0.953 |
| **E02** | + name-as-one-token TF-IDF pass, exact-key passes (name+house no., unique names); blocking recall 95.4 → 97.5% | 0.9776 | **0.964** |
| S3 (E03a) | + per-country TF-IDF, distractor features, candidate-vs-candidate group features, entity decision model, France normalization | 0.9806 | 0.956 ❌ |
| S4 consensus | only pairs predicted by **both** E02 and S3 | n/a | pending |

**Key lesson (read `docs/audit.md`):** S3's validation went up but its leaderboard score went down. Test has
**~2× more distractors per S1** than train (5.8 vs 4.67 pool records per S1). S3's density-sensitive group features
accepted ~230k extra pairs on test, and 60–90% of them have a **different house number** than the S1 (the
distractor pattern). **Validation on train can't see this.** Treat any change that adds matches on test with suspicion.

## Pipeline (src/)

| Step | File | What it does |
|---|---|---|
| 1 | `prep.py`, `normalize.py`, `translit.py` | Normalizes names/addresses (legal forms, abbreviations, leetspeak, house numbers, US/India states, France regions/departments). Learns a native-script → English word dictionary from train GT (Devanagari/Kannada/Tamil/...). Caches parquet. |
| 2 | `candidates.py`, `blocking.py` | Multi-pass blocking **per country (open set)**, unioned: word TF-IDF name+addr, addr, char-4gram name, reverse (pool→S1), name-as-one-token+addr, exact keys (name+house no., unique name, sorted-name+no., no-space name+no.). |
| 3 | `stage0.py` | LightGBM ranker on blocking scores + cheap similarities → keep **top-20 per S1**. This is `candidate_pairs.tsv`, the exact model input. |
| 4 | `features.py` | ~60 pairwise features (rapidfuzz ratios, token overlap, TF-IDF cosine, legal form, house-number eq/diff, state eq, name frequency, alias parts) + group features. |
| 5 | `pipeline.py`, `run_train.py` | Stage-1 LightGBM (2-fold grouped OOF), stage-2 with competition stats of stage-1 probs, OOF validation + scorecard. |
| 6 | `decide.py`, `entity.py` | Exclusivity (each S2/S3 record → at most one S1, true in train GT), expected-F0.5 subset selection, entity-level empty/count model. |
| 7 | `run_test.py`, `check_output.py` | Scores test, writes `output/matching_results.tsv` + `candidate_pairs.tsv`, independent validation. |
| - | `evaluate.py`, `analyze.py`, `loco.py`, `cleanup.py` | Metric + scorecard log, FP/FN error analysis, leave-one-country-out proxy (US→India simulates unseen France), cache cleanup. |

## How to run

### Final submission (reproduces output/)
```bash
pip install -r requirements.txt                     # tested with Python 3.13 on Windows 11, 16 GB RAM
export ER_DATA=/path/to/student_resource/dataset    # folder with train/ and test/
# 1. E06 model: normalization -> blocking -> pruning -> features -> density-aware LightGBM -> test scoring
ER_WORK=../work_e06 OUT=../output_e06 EXP=E06 PY=python bash scripts/run_e06.sh
# 2. earlier models of the ensemble, same pipeline with their settings (see Results table / docs):
#    E02 -> output_E02/matching_results.tsv, S3 -> output_S3/{matching_results,candidate_pairs}.tsv
# 3. agreement ensemble: keep a pair only if all three models predict it; candidate file = union of the
#    candidate sets the models scored (written next to the output)
cd src && python vote.py ../output/matching_results.tsv 3 ../output_e06/candidate_pairs.tsv,../output_S3/candidate_pairs.tsv \
    ../output_e06/matching_results.tsv ../output_E02/matching_results.tsv ../output_S3/matching_results.tsv
python check_output.py ../output $ER_DATA/test      # + official utils/validate_submission.py --check-ids
```
Diagnostics: `diag.py <matching>` (per-country profile, data-driven substitution mining), `diffprof.py <best> <new>`
(label-free go/no-go on added/removed pairs), `calib.py loco` (unseen-country calibration check).

### Original baseline pipeline
```bash
bash scripts/run_all.sh        # or run the steps inside it one by one
```
Paths can be overridden with env vars `ER_DATA` and `ER_WORK`. Needs ~16 GB RAM and ~20 GB free disk; a full
cycle takes ~7–8 h (blocking is the slowest step). Always also run the official validator:
`python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir <test dir> --check-ids`

## Where to help (highest expected leaderboard gain first)

1. **Make training match test's distractor density.** Upweight hard negatives (~2×) or tune the decision
   rule under a density-stressed validation. Then remove or neutralize density-sensitive features
   (`cvc_*`, `g_same_*`, raw counts `n_freq_q/c`, `g_size`, `*_p_n`, `n_cand`) or make them rank/relative.
2. **House-number guard:** strong penalty (feature + decision) for pairs whose first house number differs while
   everything else is near-identical. That's exactly the distractor pattern on test.
3. **Blocking recall** (2.5% of true pairs never become candidates; recall gains transferred ~1.5× to the
   leaderboard for E02). Try relative `max_df` per country, and the new `k_sorted_num` / `k_nosp_num` passes (already coded).
4. **France (unseen country):** validate any France idea with `loco.py` (train US → evaluate India).
   Monotone constraints were tried: worse.
5. **Hygiene fixes from `docs/audit.md`:** stale-cache reuse (stages skip when files exist), empty-string
   similarity = 100 in rapidfuzz, decision rule tuned on the same OOF it's evaluated on.

## Docs
- `docs/STATE.md`: data facts, pipeline, numbers (brief for anyone joining).
- `docs/PLAN_98.md`: improvement plan + revisions after each leaderboard result.
- `docs/audit.md`: code/pipeline audit, 13 ranked silent failure modes with fixes.
- `docs/EXPERIMENT_SCORECARDS.md`, `docs/experiment_log.jsonl`: every experiment's scorecard.
- `logs/`: raw run logs (timings, blocking recall per pass, chunk sizes).

## Rules we follow
No external data, APIs or geocoding. No hand-labeling of test. No leaderboard probing. Models are LightGBM
(MIT) only. Test data is used unlabeled (normalization, blocking statistics) exactly as at inference.
