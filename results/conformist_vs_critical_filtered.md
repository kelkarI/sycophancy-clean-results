# Family-level summary (degradation-filtered)

Same as the core table but any family member whose locked coefficient is flagged as degraded in at least one test seed is excluded from the family mean/min/max. This isolates on-manifold steering responses from model-collapse artefacts.

| Model | Family | n cond | n used | mean Δlogit | range Δlogit | mean Δrate (pp) | Holm-sig 3/3 count | Degraded excluded |
|---|---|---|---|---|---|---|---|---|
| gemma-2-27b-it | critical | 3 | 3 | -0.596 | [-0.711, -0.521] | -9.19 | 3 | — |
| gemma-2-27b-it | caa | 1 | 1 | -0.879 | [-0.879, -0.879] | -8.89 | 1 | — |
| gemma-2-27b-it | conformist | 3 | 3 | +0.031 | [-0.052, +0.100] | +1.26 | 0 | — |
| gemma-2-27b-it | null_control | 1 | 1 | -0.254 | [-0.254, -0.254] | -2.07 | 0 | — |
| qwen3-32b | critical | 3 | 3 | -1.931 | [-2.272, -1.699] | -13.04 | 3 | — |
| qwen3-32b | caa | 1 | 1 | -1.965 | [-1.965, -1.965] | -20.89 | 1 | — |
| qwen3-32b | conformist | 3 | 2 | -0.369 | [-0.709, -0.029] | -0.78 | 0 | pacifist |
| qwen3-32b | null_control | 1 | 1 | -1.058 | [-1.058, -1.058] | -10.29 | 0 | — |
