
### E01_baseline_stage1  (2026-09-25 15:51)  [INCONCLUSIVE]
 | per-fold F0.5 {0: 0.9679467450582795, 1: 0.9679338961104174}

config: `{"top_n": 20, "blocking": "{'comb': {'kind': 'comb', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 50}, 'addr': {'kind': 'addr', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 25}, 'name_ch': {'kind': 'name_nospace', 'vkw': {'analyzer': 'char', 'ngram_range': (4, 4), 'max_df': 20000, 'min_df': 2}, 'K': 12},", "model": "lgb stage1", "decision": {"rule": "expected_f", "miss": 0.2, "empty_bias": 1.2, "exclusive": true}}`

```
┌────────────────────────────────────────┐
│ EXPERIMENT SCORECARD: E01_baseline_stag│
├────────────────────────────────────────┤
│ Blocking recall                95.37 % │
│ Avg candidates / S1               20.0 │
│ P95 candidates / S1                 20 │
│ Pair precision                 99.48 % │
│ Pair recall                    92.82 % │
│ Macro F0.5                      0.9679 │
│ Singleton accuracy             97.19 % │
│ False positives                  37063 │
│ False negatives                 548195 │
└────────────────────────────────────────┘
```

### E01_baseline_stage2  (2026-09-25 16:18)  [INCONCLUSIVE]
 | per-fold F0.5 {0: 0.9684682727326785, 1: 0.968389068318502}

config: `{"top_n": 20, "blocking": "{'comb': {'kind': 'comb', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 50}, 'addr': {'kind': 'addr', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 25}, 'name_ch': {'kind': 'name_nospace', 'vkw': {'analyzer': 'char', 'ngram_range': (4, 4), 'max_df': 20000, 'min_df': 2}, 'K': 12},", "model": "lgb stage1+stage2", "decision": {"rule": "expected_f", "miss": 0.0, "empty_bias": 1.2, "exclusive": true}}`

```
┌────────────────────────────────────────┐
│ EXPERIMENT SCORECARD: E01_baseline_stag│
├────────────────────────────────────────┤
│ Blocking recall                95.37 % │
│ Avg candidates / S1               20.0 │
│ P95 candidates / S1                 20 │
│ Pair precision                 99.48 % │
│ Pair recall                    93.03 % │
│ Macro F0.5                      0.9684 │
│ Singleton accuracy             97.45 % │
│ False positives                  37419 │
│ False negatives                 532186 │
└────────────────────────────────────────┘
```

### E02_blocking_stage1  (2026-09-25 20:18)  [INCONCLUSIVE]
 | per-fold F0.5 {0: 0.9776179629991506, 1: 0.9775251881754786}

config: `{"top_n": 20, "blocking": "{'comb': {'kind': 'comb', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 50}, 'addr': {'kind': 'addr', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 25}, 'name_ch': {'kind': 'name_nospace', 'vkw': {'analyzer': 'char', 'ngram_range': (4, 4), 'max_df': 20000, 'min_df': 2}, 'K': 12},", "model": "lgb stage1", "decision": {"rule": "expected_f", "miss": 0.5, "empty_bias": 1.2, "exclusive": true}}`

```
┌────────────────────────────────────────┐
│ EXPERIMENT SCORECARD: E02_blocking_stag│
├────────────────────────────────────────┤
│ Blocking recall                97.51 % │
│ Avg candidates / S1               20.0 │
│ P95 candidates / S1                 20 │
│ Pair precision                 99.45 % │
│ Pair recall                    94.61 % │
│ Macro F0.5                      0.9776 │
│ Singleton accuracy             97.22 % │
│ False positives                  39686 │
│ False negatives                 411965 │
└────────────────────────────────────────┘
```

### L01_us_to_india  (2026-09-25 22:10)  [INCONCLUSIVE]
LOCO proxy for unseen country; top gain features: s0, s0_p_gap_2nd, s0_p_rank, s0_q_gap_max, num_first_pfx, s0_q_sum, legal_jacc, nfull_tsort, s0_p_gap_max, ntok_len_c, n_tfidf, legal_eq, s0_q_gap_2nd, num_cont_c, n_jw

config: `{"train": "us", "eval": "india", "drop": [], "decision": {"rule": "threshold", "t": 0.9, "exclusive": true}}`

```
┌────────────────────────────────────────┐
│ EXPERIMENT SCORECARD: L01_us_to_india  │
├────────────────────────────────────────┤
│ Blocking recall                  nan % │
│ Avg candidates / S1                nan │
│ P95 candidates / S1                nan │
│ Pair precision                 98.03 % │
│ Pair recall                    90.67 % │
│ Macro F0.5                      0.9512 │
│ Singleton accuracy             93.80 % │
│ False positives                  55769 │
│ False negatives                 285558 │
└────────────────────────────────────────┘
```

### L02_mono  (2026-09-25 22:25)  [INCONCLUSIVE]
LOCO proxy for unseen country; top gain features: s0, s0_p_gap_2nd, s0_p_rank, s0_q_gap_max, num_first_pfx, s0_q_sum, s0_q_gap_2nd, legal_jacc, nfull_tsort, legal_eq, num_cont_c, ntok_cont_c, ntok_len_c, g_sim_rank, n_freq_c

config: `{"train": "us", "eval": "india", "drop": [], "decision": {"rule": "threshold", "t": 0.87, "rel": 0.0, "exclusive": true}, "ranknorm": null, "mono": "1"}`

```
┌────────────────────────────────────────┐
│ EXPERIMENT SCORECARD: L02_mono         │
├────────────────────────────────────────┤
│ Blocking recall                  nan % │
│ Avg candidates / S1                nan │
│ P95 candidates / S1                nan │
│ Pair precision                 97.77 % │
│ Pair recall                    90.56 % │
│ Macro F0.5                      0.9489 │
│ Singleton accuracy             92.86 % │
│ False positives                  63110 │
│ False negatives                 288929 │
└────────────────────────────────────────┘
```

### E03a_features_stage1  (2026-09-26 02:00)  [INCONCLUSIVE]
 | per-fold F0.5 {0: 0.9801207717084434, 1: 0.9800286378634588}

config: `{"top_n": 20, "blocking": "{'comb': {'kind': 'comb', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 50}, 'addr': {'kind': 'addr', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 25}, 'name_ch': {'kind': 'name_nospace', 'vkw': {'analyzer': 'char', 'ngram_range': (4, 4), 'max_df': 20000, 'min_df': 2}, 'K': 12},", "model": "lgb stage1", "decision": {"rule": "expected_f", "miss": 0.5, "empty_bias": 1.2, "exclusive": true}}`

```
┌────────────────────────────────────────┐
│ EXPERIMENT SCORECARD: E03a_features_sta│
├────────────────────────────────────────┤
│ Blocking recall                97.51 % │
│ Avg candidates / S1               20.0 │
│ P95 candidates / S1                 20 │
│ Pair precision                 99.60 % │
│ Pair recall                    95.01 % │
│ Macro F0.5                      0.9801 │
│ Singleton accuracy             97.72 % │
│ False positives                  28784 │
│ False negatives                 381072 │
└────────────────────────────────────────┘
```

### E03a_features_stage2  (2026-09-26 02:30)  [INCONCLUSIVE]
 | per-fold F0.5 {0: 0.9805636324951484, 1: 0.9805927813018959}

config: `{"top_n": 20, "blocking": "{'comb': {'kind': 'comb', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 50}, 'addr': {'kind': 'addr', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 25}, 'name_ch': {'kind': 'name_nospace', 'vkw': {'analyzer': 'char', 'ngram_range': (4, 4), 'max_df': 20000, 'min_df': 2}, 'K': 12},", "model": "lgb stage1+stage2", "decision": {"rule": "expected_f", "miss": 0.0, "empty_bias": 1.2, "exclusive": true}}`

```
┌────────────────────────────────────────┐
│ EXPERIMENT SCORECARD: E03a_features_sta│
├────────────────────────────────────────┤
│ Blocking recall                97.51 % │
│ Avg candidates / S1               20.0 │
│ P95 candidates / S1                 20 │
│ Pair precision                 99.60 % │
│ Pair recall                    95.19 % │
│ Macro F0.5                      0.9806 │
│ Singleton accuracy             97.92 % │
│ False positives                  28891 │
│ False negatives                 367603 │
└────────────────────────────────────────┘
```

### E03a_stage2_entity  (2026-09-26 02:32)  [INCONCLUSIVE]


config: `{"decision": {"empty_scale": 1.3, "n_blend": 1.0}, "model": "stage1+stage2+entity"}`

```
┌────────────────────────────────────────┐
│ EXPERIMENT SCORECARD: E03a_stage2_entit│
├────────────────────────────────────────┤
│ Blocking recall                97.51 % │
│ Avg candidates / S1               20.0 │
│ P95 candidates / S1                 20 │
│ Pair precision                 99.60 % │
│ Pair recall                    95.20 % │
│ Macro F0.5                      0.9806 │
│ Singleton accuracy             98.14 % │
│ False positives                  28901 │
│ False negatives                 366496 │
└────────────────────────────────────────┘
```
