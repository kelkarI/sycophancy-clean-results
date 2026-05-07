# Normalized effect sizes (fig11)

Behavioral efficiency = |∆logit| / ε, where ε = |α| / ‖h̄^baseline_TARGET‖.

∆logit averaged over 3 test seeds (seed_42, seed_7, seed_123) at the layerwise experiment's locked α (sourced from each repo's `results/best_coefs_test.json`). `ratio_vs_random_0` is behavioral_efficiency / random_0_efficiency on the same model.

| model | family | condition | α | ‖h̄‖ | ε | Δlogit | Δlogit CI lo | Δlogit CI hi | efficiency | ratio vs random | argmax cos | argmax L |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma-2-27b-it | targeted | caa | -2000 | 19750 | 0.1013 | -0.8795 | -0.8801 | -0.8788 | 8.685 | 11.65× | +1.000 | 22 |
| gemma-2-27b-it | critical | skeptic | +2000 | 19750 | 0.1013 | -0.7108 | -0.7259 | -0.6957 | 7.019 | 9.41× | +0.337 | 44 |
| gemma-2-27b-it | critical | devils_advocate | +2000 | 19750 | 0.1013 | -0.5207 | -0.5388 | -0.5026 | 5.142 | 6.90× | +0.362 | 44 |
| gemma-2-27b-it | critical | judge | +2000 | 19750 | 0.1013 | -0.5564 | -0.5599 | -0.5529 | 5.495 | 7.37× | +0.244 | 44 |
| gemma-2-27b-it | conformist | peacekeeper | +5000 | 19750 | 0.2532 | -0.7137 | -0.7286 | -0.6989 | 2.819 | 3.78× | +0.274 | 44 |
| gemma-2-27b-it | conformist | pacifist | +2000 | 19750 | 0.1013 | +0.0999 | +0.0984 | +0.1015 | 0.987 | 1.32× | +0.245 | 44 |
| gemma-2-27b-it | conformist | collaborator | +500 | 19750 | 0.0253 | +0.0451 | +0.0385 | +0.0517 | 1.782 | 2.39× | -0.165 | 22 |
| gemma-2-27b-it | conformist | facilitator | -5000 | 19750 | 0.2532 | -0.7270 | -0.7360 | -0.7180 | 2.872 | 3.85× | +0.284 | 38 |
| gemma-2-27b-it | null | random_0 | +2000 | 19750 | 0.1013 | -0.0755 | -0.0847 | -0.0663 | 0.746 | 1.00× | +0.135 | 44 |
| qwen3-32b | targeted | caa | -200 | 408 | 0.4898 | -1.9651 | -2.1079 | -1.8223 | 4.012 | 0.95× | +1.000 | 32 |
| qwen3-32b | critical | skeptic | +200 | 408 | 0.4898 | -1.8231 | -1.8889 | -1.7573 | 3.722 | 0.88× | +0.359 | 49 |
| qwen3-32b | critical | devils_advocate | +200 | 408 | 0.4898 | -2.2721 | -2.4933 | -2.0509 | 4.638 | 1.10× | +0.402 | 49 |
| qwen3-32b | critical | judge | +200 | 408 | 0.4898 | -1.6989 | -1.7838 | -1.6140 | 3.468 | 0.82× | +0.261 | 47 |
| qwen3-32b | conformist | peacekeeper | -200 | 408 | 0.4898 | -0.7091 | -0.8319 | -0.5863 | 1.448 | 0.34× | +0.356 | 46 |
| qwen3-32b | conformist | pacifist | +500 | 408 | 1.2246 | -2.9786 | -3.1750 | -2.7822 | 2.432 | 0.58× | +0.191 | 51 |
| qwen3-32b | conformist | collaborator | -100 | 408 | 0.2449 | -0.0294 | -0.0479 | -0.0108 | 0.120 | 0.03× | +0.186 | 46 |
| qwen3-32b | conformist | facilitator | -200 | 408 | 0.4898 | -0.4686 | -0.5804 | -0.3569 | 0.957 | 0.23× | +0.335 | 46 |
| qwen3-32b | null | random_0 | +200 | 408 | 0.4898 | -2.0680 | -2.1277 | -2.0083 | 4.222 | 1.00× | +0.343 | 47 |
