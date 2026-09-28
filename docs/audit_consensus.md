# Audit: consensus submission (S4) (Sep 26, ~11:00 IST)

Leaderboard: E01 0.953 · **E02 0.964** · S3 0.956 · **S4 consensus (E02 ∩ S3) = 0.969** (best so far; my guess was 0.966).

## 1. What the 0.969 proves (measured / derived)
| Finding | Evidence | Impact | Type |
|---|---|---|---|
| **~81% of the 57,673 pairs removed from E02 were false merges** | Leaderboard +0.005 from removing only 1% of pairs. Under F0.5, removing an FP gains ≈+0.20 for that entity and removing a TP costs ≈−0.06, which solves to ≈19% correct | HIGH: test precision is far below validation precision (99.45%) | Silent before this result |
| **Model agreement is a strong precision signal on test** | Pairs where two differently-trained models disagree are mostly wrong on test, even though each model scores ≥0.977 on validation | HIGH: use multi-model agreement / ensembles in every future submission | Strategy |
| Validation can't rank test-time precision changes | S3 validation +0.003 → leaderboard −0.008; consensus has no validation score → leaderboard +0.005 | HIGH: validation is blind to test's ~2× distractor density | Silent |

## 2. What is left in the consensus predictions
| Check | US | India | France |
|---|---|---|---|
| Pairs | 2.22M | 2.64M | 0.88M |
| House-number mismatch, consensus predictions | 10.6% | 20.3% | 2.9% |
| House-number mismatch, **true train pairs** | 10.8% | 20.6% | n/a |
| Mismatch while a sibling match has the right number, consensus | 8.1% | 12.9% | 2.5% |
| Same, **true train pairs** | 8.3% | 13.3% | n/a |

**Conclusion:** the remaining predictions have the same house-number profile as genuine matches. **A blunt house-number veto would delete mostly true pairs.** Rejected (it would have looked attractive without this check). The remaining errors can't be separated by a single rule; they need a better model or more model agreement.

## 3. Loopholes / risks in the consensus approach
| # | Issue | Impact | Type |
|---|---|---|---|
| C1 | Consensus can only **remove** pairs: recall is capped at E02's (≈94–95% on validation; lower on test) | MEDIUM: limits the upside to about +0.005 per round | Structural |
| C2 | No validation score exists for consensus (built on test only). Future ensembles must be built on train OOF too, so their rule can be validated | MEDIUM | Silent |
| C3 | E02 and S3 share blocking, stage-0 and most features, so their errors are correlated; agreement filters only the errors they *don't* share | MEDIUM: a more different third model adds more | Structural |
| C4 | France (unseen) is still unmeasured; consensus changed France more (−2.4% of pairs) than US/India (−0.7%) | MEDIUM | Silent |
| C5 | `candidate_pairs.tsv` is S3's candidate file (same blocking as E02), and consensus matches ⊆ it (validated). Fine, but the final package must document this | LOW | Cosmetic |

## 4. Next submission (S5) design, derived from this audit
1. **Third, deliberately different model (E04)**, trained for test-like density: negatives weighted ~2×; density-sensitive group features removed (`cvc_*`, `g_same_*`, raw counts); normalization re-applied to train (consistency); new blocking keys (`k_sorted_num`, `k_nosp_num`) for recall.
2. **Agreement ensemble built on train OOF as well as test.** Final rule chosen on validation among: E04 alone, majority vote of {E02, S3-style, E04}, and E04 ∩ (E02 ∪ S3). Preference goes to the rule that raises precision without losing validation recall.
3. Keep every go/no-go check (both validators, per-country stats, diff vs S4 profiled by house-number agreement).
