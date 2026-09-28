# Plan to push the leaderboard above 0.98 (written Sep 25, ~23:15 IST)

## 1. Where we are

| Submission | Validation macro F0.5 (all 2.2M train S1, out-of-fold) | Leaderboard |
|---|---|---|
| E01 | 0.9684 | **0.950** (actual) |
| E02 (ready, validated) | 0.9776 | ~0.958–0.962 expected |

Validation is out-of-fold and grouped by S1, and it has been checked for leakage. The only leak is small: the transliteration dictionary was learned from all train labels.

## 2. Where we lose points: validation (E02, total loss 0.0224)

| Error type | S1 entities affected | F0.5 lost | Share |
|---|---|---|---|
| True match reached the model but was rejected (model/decision) | 181k | **0.0082** | 37% |
| True match never reached the model (blocking) | 148k | **0.0072** | 32% |
| Entity got a false merge | 35k | 0.0039 | 18% |
| Missed by both | 17k | 0.0016 | 7% |
| Singleton S1 got any prediction (score 0) | 3.4k | 0.0016 | 7% |

Extra facts: 9,038 non-singleton S1 got an empty prediction, and 3,629 entities got only wrong matches.
**Conclusion:** recall is the bigger lever on validation (0.0154 of 0.0224). The model is very precise (99.45%) but conservative.

## 3. Where we lose points: validation → leaderboard gap (0.968 → 0.950 = 0.018)

| Hypothesis | Evidence | Share of gap (estimate) |
|---|---|---|
| **H1: France is an unseen country** (15% of test S1) | Unseen-country proxy (train US only, score India): 0.9739 → **0.9512**, false positives 3×, singleton acc 96.6 → 93.8%. France predictions show more matches (3.47/S1) and fewer empties (4.2%) than US/India. Training shows both countries share the same prior (5.58% singletons, 3.46 matches/S1), so France very likely has the same prior and is over-matching. | **largest, ~0.008–0.015** |
| **H2: test is denser than train** | Pool records per S1: test 5.5–5.8 vs train 4.67, in every country. Only ~58% of test pool records get matched vs ~70% on train, so test has more unowned records. A naive simulation (drop 19–30% of train S1) shows almost no loss (0.9776 → 0.9775), but it's biased because the features still "saw" the dropped owners. | unknown, 0–0.008; **must test faithfully** |
| H3: transliteration dictionary leak | Test token coverage 96.4% vs ~100% on train; India native-script names are ~23% of pool names | small, ~0.001 |
| H4: public-subset noise | Large test set | ±0.002 |
| Format / ID issues | Official validator strict mode + independent check pass | 0 |

Rejected so far: monotone constraints (unseen-country proxy 0.9512 → 0.9489, worse).

## 4. What it takes to reach 0.98 on the leaderboard
If the gap stays at ~0.018, validation would need ~0.998, which is not realistic. **The only realistic route is to close the gap (France + density) and add ~+0.004–0.006 of validation gain.** For example: validation 0.982 with the gap cut to 0.004 gives 0.978; with the gap cut to 0.002 it gives 0.980.

## 5. Phase 0: cheap, decisive experiments (~1.5 h, no retraining, cached E02 data)
Every item is judged on per-country out-of-fold scores **and** the unseen-country proxy (US→India and India→US).

| ID | Experiment | Why | Keep if |
|---|---|---|---|
| P0.1 | **Label-free prior matching:** on an unseen country, pick decision strictness (empty-bias / temperature) so predicted empty rate ≈ 5.6% and mean predicted matches ≈ what the model predicts in-domain | Directly targets France over-matching; uses only the known, country-independent prior | Proxy F recovers ≥ +0.005 |
| P0.2 | **Per-country rank normalization** of features (interrupted run L03) | Aligns feature distributions across countries | Proxy F ≥ +0.003 and in-domain not worse |
| P0.3 | **Self-training:** add high-confidence unseen-country pairs (p>0.98 positive, p<0.02 negative) as pseudo-labels and retrain | Classic domain adaptation for an unlabeled country | Proxy F ≥ +0.004 |
| P0.4 | **Faithful density simulation:** drop 20% of train S1, recompute competition features (s0 ranks/gaps, candidate-side ranks) without the dropped owners, re-score with E02 models | Tests H2 properly | If the loss is ≥0.004, add owner-dropout augmentation to E03 |
| P0.5 | **Model-miss deep dive** on the 181k entities: missed candidates' probabilities, how many were taken by exclusivity vs rejected by the threshold, empty-address share | Largest loss bucket | Chooses the decision/feature fixes below |

## 6. Phase 1: E03 full cycle (~4.5 h train + test) → submission 3
Batched because each full cycle is expensive. Every block is measured separately where possible (blocking recall, per-country out-of-fold, proxy).

1. **Normalization fixes:** "817 38ND" state bug (already coded); France address normalization (R./Bd/Av/All./Ch/Imp/Pl/Qu/Rte/Fbg, St/Ste→saint, `N°`, `bis/ter`; region and department names such as Nord, Gironde, Loire-Atlantique and the regions treated like US/India states); French articles (du/de/la/le/des/d'/l') dropped from core names. *Expected: validation +0.000–0.001; France +0.005–0.02 (address and name features become meaningful).*
2. **Blocking** (blocking loss 0.0072 + 0.0016): replace absolute `max_df=20000` with a per-country relative cap (it prunes common name words in large countries); add exact keys (sorted name tokens + first number; space-less transliterated name + first number); `top_n` 20 → 25. *Target blocking recall ≥98.2% (now 97.51%); validation +0.002–0.004.*
3. **Features** (model-miss 0.0082 + FP 0.0039): distractor features (already coded: house-number difference, last token changed, legal suffix only on one side, extra/missing words); candidate-vs-candidate features (similarity of this candidate to the best other candidate of the same S1; number of candidates sharing the same address or name); IDF weight of the differing name tokens. *Validation +0.002–0.004.*
4. **Model:** stage-1 trained on 60% of S1 groups (now 30%), 1500 rounds; stage-2 re-scorer on. *Validation +0.001–0.002.*
5. **Phase-0 winners** (prior matching for new countries, rank normalization, self-training, owner-dropout).

*E03 expected: validation 0.981–0.984; leaderboard 0.966–0.976, depending on how much France/density is fixed.*

## 7. Phase 2: E04 (~4.5 h) → submission 4
- Second model family (XGBoost on GPU) averaged with LightGBM; kept only if out-of-fold improves.
- Exact expected-F0.5 subset selection (dynamic programming over candidate probabilities) instead of the plug-in approximation; kept only if out-of-fold improves.
- France self-training on the real test France records, if P0.3 proved the method on the proxy.
- Decision re-tuned under the density simulation, if P0.4 showed it matters.
*Expected: leaderboard +0.002–0.006 over E03.*

## 8. Schedule (IST)
| When | What |
|---|---|
| Sep 25 23:30 – Sep 26 01:00 | Phase 0 (P0.1–P0.5) |
| 01:00 – 04:30 | E03 train cycle (re-preprocess, blocking, features, stage-1/2, validation) |
| 04:30 – 06:30 | E03 test run + double validation → **submission 3 ready ~06:30** |
| 07:00 – 13:30 | E04 train + test → **submission 4 ready ~13:30** |
| 14:00 – 22:00 | E05: targeted fixes from E03/E04 results → **submission 5 ready ~22:00** |
| Sep 27 12:00 | **Cutoff for the last model change** |
| Sep 27 12:00 – 18:00 | Final package: output/, code/ (src, README, requirements.txt pinned), Documentation_template.md filled, clean reproduction check |
| Sep 27 18:00 – 23:59 | Buffer |

## 9. Go/no-go checklist before every submission
1. Official validator with `--check-ids` passes; independent check passes (row count, IDs exist, no duplicates, matches ⊆ candidates, each S2/S3 at most one S1).
2. Validation macro F0.5 ≥ previous best on **both** folds and on **both** countries.
3. Unseen-country proxy not worse than the previous best.
4. Per-country test prediction stats are plausible: empty rate ~5–6%, mean matches ~3.3–3.45, France not an outlier.
5. Difference vs previous submission reviewed: pairs added and removed per country, with a sample inspected by eye.
6. Only one submission per validated improvement; no leaderboard probing.

## 10. Honest outlook
- Most likely final leaderboard: **0.968–0.976**.
- Probability of **> 0.97**: ~60%. Probability of **> 0.98**: ~20%. That needs the France/density gap to shrink to ≤0.003 and validation to reach ≥0.982.
- Biggest risks: France can't be measured directly (mitigated by the unseen-country proxy and prior matching); each full cycle costs ~4.5 h on 16 GB RAM (mitigated by batching and cutoffs); overfitting the public leaderboard (mitigated by deciding only on validation and proxy results).

---
# v2: corrections after adversarial review (Sep 25, 22:50)
The review was 4 independent critics plus a verdict; the claims were checked against the code. These corrections override sections 4–8 where they conflict.

**A. Leaderboard model:** LB ≈ 0.85·F(US+India) + 0.15·F(France). To reach 0.98 you need, e.g., US+India 0.985 with France 0.947, or US+India 0.983 with France 0.963. Every experiment is judged by which of the two terms it moves.

**B. Submit E02 tonight (before 23:59).** It banks a better fallback, uses today's slot, and is diagnostic: E02 should gain about +0.008 over E01 if the gap is France-only. If it gains much less, test density is also hurting US/India. This is a real improvement submission, not a probe.

**C. P0.1 target fixed:** France's predicted 3.47 matches/S1 already equals the train prior, so the train prior is the wrong target. The new target is the **US+India TEST prediction histogram** (empty rate and counts k=0..11), after checking that US test statistics agree with US out-of-fold statistics. Move at most halfway toward the target. Realistic gain ≤ +0.006 on the leaderboard.

**D. P0.4 replaced:** dropping S1 owners only creates easy orphans. Instead, measure **near-copy distractors per S1** (same name or address, different house number/suffix) on test vs train. If test has more, inject synthetic distractors into train.

**E. P0.3 (self-training) dropped:** pseudo-labels at p>0.98 reinforce the confident false positives that are the problem. P0.2 (rank normalization) ships only if it passes both proxy directions and US/India out-of-fold.

**F. New, cheapest levers first, no retrain (decision layer on cached probabilities, ~1.5 h):**
  1. Exclusivity with **reassignment**: 2–3 rounds that give a released candidate back to the next S1, plus an explicit "belongs to no S1" option.
  2. **Per-country prior-shift correction** (EM / Saerens-style, label-free).
  3. **Entity-level empty/count model** trained on out-of-fold data (replaces prod(1−p)). It targets the 9k non-singletons predicted empty and the singleton loss.
  4. Per-country `empty_bias`/`miss`.
  Expected: +0.003–0.008 on the leaderboard. **Submitted first on Sep 26 if validation and proxy improve.**

**G. E03 additions and guards:**
  - **Per-country TF-IDF and name frequency** in `features.py` (they're currently fitted on all countries together; name frequency gets normalized by country pool size). Label-free; France +0.01–0.03 expected.
  - **Every French rule guarded by `country=='france'`**, e.g. St→saint and dropping de/la/le would hurt US names.
  - Memory: keep 20 candidates per S1 (top_n=25 only to measure recall); stage-1 on 30–40% of groups (not 60%); test the relaxed max_df on a 10% sample first; checkpoint every stage.

**H. XGBoost-GPU dropped** (a ~10 GB matrix doesn't fit on a 4 GB GPU). Use a second LightGBM with a different seed and parameters.

**I. Gains are not added up:** blocking ≈ +0.0015–0.003 and features ≈ +0.001–0.0025 on validation. The France levers overlap, so they get one combined estimate.

**J. No hand-labeling of test records** (fair-play). France is judged by the unseen-country proxy plus prediction-statistics sanity checks.

**K. Revised schedule (IST):**
| When | What |
|---|---|
| Sep 25 before 23:59 | Submit E02 |
| Sep 25 23:00 – Sep 26 01:30 | Decision-layer work (F) + P0.1/P0.2/P0.5 + near-copy density measurement (D), on cached data |
| Sep 26 ~02:30 | Submission 3 = E02 models + best decision layer, **if validation and proxy improve** (test run ~50 min, no retrain) |
| Sep 26 02:30 – 09:30 | E03 full cycle (normalization, per-country IDF, blocking, features, stage-2), checkpointed → **submission 4 ~09:30** |
| Sep 26 10:00 – 20:00 | E04: second model + remaining winners → **submission 5 ~20:00** |
| Sep 27 06:00 | **Model freeze**; clean reproduction run overnight/morning |
| Sep 27 06:00 – 18:00 | Final package + documentation; up to 5 more submissions available on Sep 27 |

**L. Revised outlook:** most likely final leaderboard **0.970–0.978**. P(≥0.98) is about **10–15%**. It needs France ≈0.95+ without labels and near-zero density loss. 0.98 stays the stretch goal; the final pick is made by validation + proxy, not by the public score.

---
# v3: after E02 leaderboard = 0.963 (E01 0.950)
- Leaderboard gain +0.013 vs ~+0.008 expected from validation (+0.0092 × 0.85). The gap shrank 0.018 → 0.0146.
- E02 barely changed France (97.5% of France pairs identical), so the extra gain came from US/India on the denser test data, where recall gains transfer more strongly.
- **Reprioritized:** (1) blocking recall + per-country statistics + French normalization; (2) decision layer (reassignment, "no match" option, empty/count model); (3) distractor features + stage-2; (4) France calibration only if the proxy proves it.
- Outlook: E03 expected leaderboard 0.971–0.978; P(≥0.98) ≈ 20–25%.
