
### E07_stage1  (2026-09-26 22:00)  [INCONCLUSIVE]
 | plain F0.5 0.97988 (density-stressed alpha=1.9 above) | per-fold F0.5 {0: 0.9790824570093625, 1: 0.9790438647367463}

config: `{"top_n": 20, "blocking": "{'comb': {'kind': 'comb', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 50}, 'addr': {'kind': 'addr', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 25}, 'name_ch': {'kind': 'name_nospace', 'vkw': {'analyzer': 'char', 'ngram_range': (4, 4), 'max_df': 20000, 'min_df': 2}, 'K': 12},", "drop": ["cvc_name", "cvc_addr", "g_same_addr", "g_same_name", "n_freq_q", "n_freq_c", "g_size", "s0_q_n", "s0_p_n", "s0_q_sum"], "neg_w": 1.9, "alpha": 1.9, "model": "lgb stage1", "decision": {"rule": "expected_f", "miss": 0.2, "empty_bias": 1.2, "exclusive": true}}`

```
┌────────────────────────────────────────┐
│ EXPERIMENT SCORECARD: E07_stage1       │
├────────────────────────────────────────┤
│ Blocking recall                97.61 % │
│ Avg candidates / S1               20.0 │
│ P95 candidates / S1                 20 │
│ Pair precision                 99.44 % │
│ Pair recall                    94.71 % │
│ Macro F0.5                      0.9791 │
│ Singleton accuracy             98.00 % │
│ False positives                  40996 │
│ False negatives                 404434 │
└────────────────────────────────────────┘
```
