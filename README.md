# Clean results — sycophancy steering on Gemma 2 27B and Qwen 3 32B

A standalone post-hoc package of the main paper result only: critical
role directions reduce sycophancy, conformist role directions do not,
and a general persona direction approaches the effect of the targeted
CAA direction that was trained on sycophancy labels. Everything in
this repo is derived from existing experiment outputs (no re-running
of model inference).

Source repositories (read only, not modified):

- Gemma pipeline: `../sycophancy-gemma/experiment-main/`
  (multi-seed aggregate, per-seed JSONs, tune-locked best coefficients,
  steering-vector cosine matrix).
- Qwen pipeline:  `../sycophancy-qwen/`
  (same structure; target layer 32, coefficient grid rescaled 10×
  because of Qwen's smaller activation norms — see the parent paper
  repo for details).

## Scope

**Kept conditions** (both models):

- Critical roles: `skeptic`, `devils_advocate`, `judge`.
- Conformist roles: `peacekeeper`, `pacifist`, `collaborator`.
- Targeted CAA: `caa`.
- Null control: aggregate over `random_0` through `random_9`
  (unit-Gaussian steering vectors, 10 per seed).
- Baseline (coef = 0 on any real condition).

**Dropped conditions**:
`assistant_axis`, `contrarian`, `scientist`, `facilitator`, all
`*_residual` conditions, all `*_caa_component` conditions.

Dropping these leaves a single-question focus: does a role vector
reduce sycophancy at its tune-locked steering coefficient, and is the
effect in the expected direction per role family?

## Directory layout

```
sycophancy-clean-results/
├── README.md                                  (this file)
├── data/
│   ├── gemma-2-27b-it_clean.json              per-seed + aggregate for kept conds
│   ├── qwen3-32b_clean.json                   same, Qwen
│   ├── gemma-2-27b-it_cosines.json            6×7 cosine matrix (kept role vectors + CAA)
│   └── qwen3-32b_cosines.json                 same, Qwen
├── scripts/
│   ├── build_data.py                          rebuilds data/ from source repos
│   ├── make_figures.py                        rebuilds figures/ from data/
│   ├── make_tables.py                         rebuilds results/*.csv and *.md
│   └── _style.py                              shared matplotlib + palette + labels
├── figures/
│   ├── fig1_delta_logit.{pdf,png}             Δ sycophancy logit, paired bar
│   ├── fig2_delta_rate.{pdf,png}              Δ sycophancy rate (pp), paired bar
│   ├── fig3_per_seed.{pdf,png}                per-seed dot plot (consistency check)
│   └── fig4_cosines.{pdf,png}                 6+1 cosine heatmap per model
└── results/
    ├── main_table.{csv,md}                    condition × model table
    └── conformist_vs_critical.{csv,md}        family-level summary
```

## Methods (summary)

- **Benchmark.** `sycophancy_on_philpapers2020` (Perez et al. 2023 A/B
  preferences), 300 base questions × 2 orderings = 600 rows per seed;
  held-out test half (150 base × 2 = 300 rows, seed-99 split).
- **Seeds.** 3 test seeds (42, 7, 123). Tune uses 5 seeds
  (42, 7, 123, 456, 789) for coefficient selection; this repo reports
  only the test split.
- **Coefficients** are locked on the tune split per condition (mode
  across seeds, tie-break by count then proximity to median). The
  numbers here are evaluated at the locked coefficient on held-out test
  seeds.
- **Null control (`random`).** For each test seed, we pool all
  `random_{0..9}` at every non-zero coefficient in the sweep (80
  samples per seed) and take the mean Δ. Cross-seed CI is then the
  t-interval on 3 per-seed means (df=2, t*≈4.30).
- **Metrics.**
  - `delta_logit = post_steer_mean_syc_logit − baseline_mean_syc_logit`
    where `syc_logit = logp(syc_token) − logp(hon_token)` at the last
    prompt token. Lower = less sycophantic.
  - `delta_rate_pp = (post_steer_rate − baseline_rate) × 100`, rate =
    fraction of rows where `argmax(logp_A, logp_B)` matches the
    sycophantic label.
- **Significance.** Paired one-sided Wilcoxon on base-level Δlogit
  (n=150 bases per seed). Each seed's Wilcoxon is Holm-corrected
  across the source pipeline's 14-condition primary family. Kept
  conditions are a subset of that family, so Holm significance carries
  over directly. We report how many of the 3 test seeds crossed
  α=0.05 after correction.

## How to reproduce

```bash
cd sycophancy-clean-results
python3 scripts/build_data.py       # rebuilds data/ from source repos
python3 scripts/make_tables.py      # rebuilds results/
python3 scripts/make_figures.py     # rebuilds figures/
```

All three scripts are CPU-only and deterministic. `build_data.py`
requires the two source repos to exist at
`../sycophancy-gemma/experiment-main/` and `../sycophancy-qwen/`;
adjust the hard-coded paths in `build_data.py` if they live elsewhere.

## Citations

- CAA — Rimsky et al. 2024, *Steering Llama 2 via Contrastive
  Activation Addition*, arXiv:2312.06681.
- Persona / assistant-axis vectors — `lu-christina/assistant-axis-vectors`
  on HuggingFace, built by the `safety-research/assistant-axis`
  pipeline.
- Benchmark — Perez et al. 2023, *Discovering Language Model Behaviors
  with Model-Written Evaluations*, arXiv:2212.09251.
- Models — `google/gemma-2-27b-it`, `Qwen/Qwen3-32B`.
