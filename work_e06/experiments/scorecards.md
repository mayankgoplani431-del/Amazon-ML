
### E06_stage1  (2026-09-26 18:38)  [INCONCLUSIVE]
 | plain F0.5 0.97986 (density-stressed alpha=1.9 above) | per-fold F0.5 {0: 0.9790292684184817, 1: 0.9790271760592533}

config: `{"top_n": 20, "blocking": "{'comb': {'kind': 'comb', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 50}, 'addr': {'kind': 'addr', 'vkw': {'token_pattern': '\\\\S+', 'max_df': 20000}, 'K': 25}, 'name_ch': {'kind': 'name_nospace', 'vkw': {'analyzer': 'char', 'ngram_range': (4, 4), 'max_df': 20000, 'min_df': 2}, 'K': 12},", "drop": ["cvc_name", "cvc_addr", "g_same_addr", "g_same_name", "n_freq_q", "n_freq_c", "g_size", "s0_q_n", "s0_p_n", "s0_q_sum"], "neg_w": 1.9, "alpha": 1.9, "model": "lgb stage1", "decision": {"rule": "expected_f", "miss": 0.2, "empty_bias": 1.2, "exclusive": true}}`

```
┌────────────────────────────────────────┐
│ EXPERIMENT SCORECARD: E06_stage1       │
├────────────────────────────────────────┤
│ Blocking recall                97.63 % │
│ Avg candidates / S1               20.0 │
│ P95 candidates / S1                 20 │
│ Pair precision                 99.43 % │
│ Pair recall                    94.71 % │
│ Macro F0.5                      0.9790 │
│ Singleton accuracy             97.97 % │
│ False positives                  41178 │
│ False negatives                 403863 │
└────────────────────────────────────────┘
```
