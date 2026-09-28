# Audit: proposed additions (LSH blocking, embeddings, F0.5 threshold, soft meta-ensemble) (Sep 27, ~13:45 IST)

Leaderboard: S4 0.969 · **S5 (E02 ∩ S3 ∩ E06) 0.971** · S6 (S5 + France calibration) 0.971.
Every number below was measured on real train data (labels) with the code in `src/exp_lsh.py`, `src/exp_lsh2.py`,
`src/exp_meta.py`, `src/exp_meta2.py`. Leaderboard conversions use the audit cost model: removing a false merge ≈ +0.20
per entity, losing / gaining a true match ≈ ∓0.06–0.08 per entity; break-even precision for ADDED pairs ≈ 0.74.

## 1. Already implemented vs genuinely new (checked in code)
| Proposal | Already have | Missing |
|---|---|---|
| MinHash/LSH fuzzy blocking | Blocking is already partly fuzzy: `candidates.py` `name_ch` = char-4gram TF-IDF cosine top-12; word TF-IDF passes `comb`/`addr`/`nameaddr` + reverse passes; exact keys `k_name_num`, `k_name`, `k_sorted_num`, `k_nosp_num` | A higher-recall char-level pass over name+address (degree, not capability) |
| Track A: legal-suffix normalization + exact match | Yes: `normalize.py` `NAME_CANON`/`NAME_STOP` (incl. SARL/SAS/EURL, dotted forms since E06), `features.py` `LEGAL` -> `legal_eq`, `legal_only_c/q`; exact equality via ratio = 100 | Nothing material |
| Track B: bi-encoder (MiniLM) embeddings | No embeddings anywhere | New |
| F0.5 threshold, high cutoff | Yes, and stronger: `decide.expected_f` picks the expected-F0.5-optimal subset per S1 with an empty option; `run_train.sweep` tunes it on OOF macro F0.5 (density-stressed, alpha 1.9 since E06), incl. global thresholds 0.05–0.95 | Nothing (a fixed 0.9 cutoff would be a regression) |
| Soft meta-classifier instead of hard-AND | No: `vote.py` is hard k-of-n; S5 is hard AND | New |

Missed true pairs (train, E07 blocking): 7,638,365 true pairs; **171,525 (2.25%) never reach the blocking union**,
11,017 (0.14%) reach it but are pruned from the top-20. Profile of the blocking misses: median char-3gram name Jaccard
0.48 (India) / 0.50 (US); candidate address empty 26.5% (India) / 49.3% (US); non-ASCII candidate name 13.3% / 3.7%.

## 2. (a) Extra fuzzy blocking pass (char-3gram name+address, the similarity MinHash/LSH approximates)
Sample: 3,000 random S1 with missed pairs per country (exact top-K search over the full same-country pool = upper bound
for LSH). Recall = share of that sample's missed true pairs retrieved; cost = new non-union candidates per S1.
| Variant (K = 12) | India recall of misses | US recall of misses | New cands / S1 | Compute / S1 | Full train+test run |
|---|---|---|---|---|---|
| name only, uncapped | 6.5% | 11.7% | 6.1–7.5 | ~30 ms | ~33 h |
| **name+addr, uncapped** | **31.9%** | **52.1%** | 4.9–5.0 | ~80 ms | ~87 h |
| name+addr, max_df 100k | 29.9% | 46.4% | 4.3–4.5 | 14–19 ms | ~17 h |
| name+addr, max_df 20k | 7.4% | 9.3% | 5.8–7.1 | 1.1–1.3 ms | ~1.4 h |
| name+addr, max_df 5k | 1.9% | 1.1% | 5.8–6.6 | 0.1 ms | minutes |
Blocking-union recall (train): current 97.75% -> +uncapped 98.68% -> +max_df 100k 98.59% -> +max_df 20k 97.94%.
**Finding:** the recall lives in common 3-grams, which are exactly what makes the pass expensive.

Leaderboard conversion (max_df 100k; test scale x0.785; s = survives top-20 pruning, r = model recall on these hard pairs,
π = precision of the added matches):
| Assumptions (s, r, π) | Added TP / FP on test | ΔF0.5 |
|---|---|---|
| pessimistic (0.5, 0.3, 0.7) | 7.6k / 3.2k | −0.0001 |
| central (0.7, 0.45, 0.8) | 15.9k / 4.0k | +0.0002 |
| optimistic (0.85, 0.6, 0.9) | 25.7k / 2.9k | +0.0009 |
Upper bound from the E01->E02 history (+0.0051 LB per point of blocking recall, confounded by other changes): ~+0.0036.
max_df 20k (the only variant that fits today): ≈ +0.00004 central. **Not run: ~17 h compute on 16 GB for the useful variant.**

## 3. (c) Soft meta-model vs hard-AND (deployable features: E06/E07 probabilities + E02/S3 votes)
OOF regenerated for E02 and S3 from their archived models (each fold scored by the model trained on the other fold);
meta = LightGBM, trained on fold k, evaluated on fold 1-k; its decision rule tuned on the training fold only.
| Held-out, 2.2M train S1 | Hard-AND (S5 logic) | Soft meta | Δ |
|---|---|---|---|
| macro F0.5 | 0.97839 | 0.98130 | +0.0029 |
| density-stressed F0.5 (α 1.9) | 0.97782 | 0.98038 | +0.0026 |
| pair precision | 99.81% | 99.67% | −0.14 pt |
| pair recall | 93.89% | 95.17% | +1.28 pt |
| FP / FN | 13,823 / 466,708 | 24,263 / 368,682 | +10.4k / −98.0k |
Soft vs hard: +108,637 pairs added (90.4% true on train), 171 removed.

**Calibration against the leaderboard (decisive):** the train analogue of the S4−S5 set (E02 and S3 yes, E06 no) is
38,222 pairs, **86.1% true on train**, but the same kind of set was only **~57–66% true on test** (audit_s5.md: removing
it gained +0.002). The soft model re-adds 34,066 of those 38,222 (89%). Applying the observed train->test degradation
(odds x0.21–0.31) to all soft-added pairs: 90.4% true on train -> **~67–75% true on test**, around the 0.74 break-even.
ΔF0.5 on the leaderboard ≈ 85k added x (p·0.07 − (1−p)·0.20) / 1.73M = **−0.0009 to +0.0001** (most likely negative).
**Finding:** a meta-model fit on train OOF learns train's lower distractor density and trusts disagreements too much;
hard-AND looks worse on train but is the better test-side rule (the S3 lesson again). Test file built (5,801,039 pairs,
`S8_meta/`) but **not recommended**.

## 4. (b) Bi-encoder embeddings
Not measured: torch / sentence-transformers not installed (~1 GB download + model); 14.1M unique names -> 21.6 GB of
float32 embeddings (> RAM); ~2–4 h CPU encoding + ~3 h feature rebuild and retraining. **Insufficient data to estimate**;
no number is claimed.

## 5. Summary chart (measured / derived)
```
Blocking-union recall on train (true pairs reachable by any model)
  current                        97.75% |##################################################
  + name+addr 3gram, max_df 20k  97.94% |###################################################  (fits today, ~1.4 h)
  + name+addr 3gram, max_df 100k 98.59% |########################################################  (~17 h)
  + name+addr 3gram, uncapped    98.68% |#########################################################  (~87 h)
                                        (bar = (recall - 95%) x 20)

Held-out macro F0.5 on train (density-stressed)
  hard-AND (S5 logic)   0.97782 |######################################
  soft meta             0.98038 |##################################################
                                (bar = (F - 0.9765) x 12,500)

Estimated leaderboard ΔF0.5 vs S5 (0.971), range
  (a) fuzzy blocking max_df 100k   -0.0001 ... +0.0009   (upper bound by history +0.0036); needs ~17 h
  (a) fuzzy blocking max_df 20k    ≈ +0.0000             fits today
  (c) soft meta over votes/probs   -0.0009 ... +0.0001   likely negative on test
  (b) embeddings                   insufficient data
```

## 6. Recommendation
1. **Keep S5 (0.971) as the final leaderboard file** on this laptop. No measured addition clears the bar within today's
   compute: (c) is likely negative on test once calibrated against the leaderboard; (a) helps only in variants that need
   ~17 h here; (b) cannot be measured.
2. **(a) is the one worth pursuing with more compute** (e.g. a 64-core box: max_df 100k pass ≈ 3–4 h for train+test):
   +0.9 pt blocking recall; realistic ΔF0.5 ≈ +0.0002 central, +0.0009 optimistic, higher only if the historical
   recall->leaderboard ratio holds.
3. **Do not replace hard-AND with a train-fitted soft meta**; if revisited, the meta must be trained with test-like
   density (e.g. ownerless negatives up-weighted to match the leaderboard-observed degradation) and checked against the
   S4->S5 calibration set before any submission.
