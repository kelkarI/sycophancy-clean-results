# Main table (degradation-filtered)

Same as `main_table.md` but with any row whose locked coefficient is flagged as degraded in at least one test seed removed, so the table reflects only conditions where the steered forward pass is behaving normally. The dropped row(s) appear in the unfiltered `main_table.md` with a † marker.

| Model | Family | Condition | Best coef | Δlogit (mean ± std) [95% CI] | Δrate (pp) | Holm-sig seeds | Degraded |
|---|---|---|---|---|---|---|---|
| gemma-2-27b-it | caa | CAA (targeted) | -2000.0 | -0.879 ± 0.001 [-0.881, -0.878] | -8.89 [-9.37, -8.41] | 3/3 *** | no |
| gemma-2-27b-it | critical | Skeptic | 2000.0 | -0.711 ± 0.013 [-0.744, -0.678] | -9.56 [-10.51, -8.60] | 3/3 *** | no |
| gemma-2-27b-it | critical | Devil's Advocate | 2000.0 | -0.521 ± 0.016 [-0.560, -0.481] | -8.67 [-10.86, -6.48] | 3/3 *** | no |
| gemma-2-27b-it | critical | Judge | 2000.0 | -0.556 ± 0.003 [-0.564, -0.549] | -9.33 [-10.77, -7.90] | 3/3 *** | no |
| gemma-2-27b-it | conformist | Peacekeeper | 2000.0 | -0.052 ± 0.004 [-0.062, -0.041] | +1.78 [-2.31, +5.86] | 0/3 ns | no |
| gemma-2-27b-it | conformist | Pacifist | 2000.0 | +0.100 ± 0.001 [+0.097, +0.103] | +0.33 [-4.05, +4.71] | 1/3 * | no |
| gemma-2-27b-it | conformist | Collaborator | 500.0 | +0.045 ± 0.006 [+0.031, +0.060] | +1.67 [-1.94, +5.28] | 2/3 ** | no |
| gemma-2-27b-it | null_control | Random (n=10) | — | -0.254 ± 0.006 [-0.269, -0.240] | -2.07 [-3.96, -0.18] | — | no |
| qwen3-32b | caa | CAA (targeted) | -200.0 | -1.965 ± 0.126 [-2.279, -1.652] | -20.89 [-25.88, -15.90] | 3/3 *** | no |
| qwen3-32b | critical | Skeptic | 200.0 | -1.823 ± 0.058 [-1.968, -1.679] | -18.11 [-22.02, -14.20] | 3/3 *** | no |
| qwen3-32b | critical | Devil's Advocate | 200.0 | -2.272 ± 0.195 [-2.758, -1.787] | -16.56 [-20.47, -12.64] | 3/3 *** | no |
| qwen3-32b | critical | Judge | 200.0 | -1.699 ± 0.075 [-1.885, -1.512] | -4.44 [-7.11, -1.78] | 3/3 *** | no |
| qwen3-32b | conformist | Peacekeeper | -200.0 | -0.709 ± 0.108 [-0.979, -0.440] | +0.11 [-0.37, +0.59] | 0/3 ns | no |
| qwen3-32b | conformist | Collaborator | -100.0 | -0.029 ± 0.016 [-0.070, +0.011] | -1.67 [-3.32, -0.01] | 0/3 ns | no |
| qwen3-32b | null_control | Random (n=10) | — | -1.058 ± 0.077 [-1.249, -0.867] | -10.29 [-13.09, -7.49] | — | no |
