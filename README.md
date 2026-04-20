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
│   ├── build_qualitative.py                   rebuilds qualitative/ from source repos
│   ├── make_figures.py                        rebuilds fig1-4 from data/
│   ├── make_showcase_pdf.py                   rebuilds fig5 tone-comparison PDF
│   ├── make_steering_curves.py                rebuilds fig6 coefficient sweep
│                                              (reads rates files from source repos)
│   ├── make_tables.py                         rebuilds results/*.csv and *.md
│   └── _style.py                              shared matplotlib + palette + labels
├── figures/
│   ├── fig1_delta_logit.{pdf,png}             Δ sycophancy logit, paired bar
│   ├── fig1_delta_logit_filtered.{pdf,png}    same, degraded cells dropped
│   ├── fig2_delta_rate.{pdf,png}              Δ sycophancy rate (pp), paired bar
│   ├── fig2_delta_rate_filtered.{pdf,png}     same, degraded cells dropped
│   ├── fig3_per_seed.{pdf,png}                per-seed dot plot (consistency check)
│   ├── fig3_per_seed_filtered.{pdf,png}       same, degraded cells dropped
│   ├── fig4_cosines.{pdf,png}                 6+1 cosine heatmap per model
│   ├── fig5_tone_comparison.pdf               typeset tone-contrast showcase (reportlab)
│   ├── fig6_steering_curves.{pdf,png}         coefficient sweep per model,
│   │                                          kept conditions only (re-draws the
│   │                                          source-pipeline fig1 without the
│   │                                          dropped conditions)
│   └── fig7_steering_curves_family.{pdf,png}  same sweep, three lines per model
│                                              (CAA + critical mean + conformist
│                                              mean) with min/max bands; degraded
│                                              cells masked before averaging — see
│                                              "Family averaging (fig7)" below
├── qualitative/
│   ├── qual_check_caa.json                    Gemma free-form responses,
│   │                                          5 philosophy prompts × {baseline, caa,
│   │                                          assistant_axis, skeptic}
│   ├── qual_check_conformist.json             Gemma free-form responses,
│   │                                          5 prompts × {baseline, peacekeeper,
│   │                                          pacifist, collaborator, facilitator, skeptic}
│   ├── gemma-2-27b-it_over_correction.json    Gemma over-correction probes (128 samples)
│   ├── qwen3-32b_over_correction.json         Qwen over-correction probes (128 samples)
│   ├── gemma_showcase.md                      rendered tone contrast, single Gemma prompt
│   └── qwen_showcase.md                       rendered tone contrast, two Qwen probes
└── results/
    ├── main_table.{csv,md}                    condition × model table (degraded rows flagged †)
    ├── main_table_filtered.{csv,md}           same, degraded rows removed
    ├── conformist_vs_critical.{csv,md}        family-level summary
    └── conformist_vs_critical_filtered.{csv,md}  family means excluding degraded members
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
- **Degradation handling.** At some large coefficients the steered
  forward pass collapses — binary rate locks to 0.5 and the syc-logit
  gap shrinks to zero — so a large |Δ| is a collapse artefact, not a
  sycophancy reduction. The source pipeline writes a per-seed
  `degradation_flags_test.json`; we copy that flag per (condition,
  seed) into `data/*_clean.json` (`per_seed[i].degraded`) and expose
  `degraded_any_seed` / `degraded_all_seeds` per condition. Tables and
  figures come in two flavours:
  - **Core** (`main_table.*`, `fig1-3.*`): every kept condition is
    plotted, but degraded cells are hatched (bars) or overlaid with
    red × (dots) and the row is flagged in the table.
  - **Filtered** (`*_filtered.*`): any condition with
    `degraded_any_seed == True` is dropped.

  Only one cell is currently degraded at its tune-locked coef: **Qwen
  3 32B × pacifist @ coef 500** (all 3 test seeds). Every other
  (model, condition) stays on the non-degraded manifold.

## Family averaging (fig7)

`fig6_steering_curves` draws one line per kept condition (CAA + 3
critical roles + 3 conformist roles + random mean + baseline).
`fig7_steering_curves_family` collapses the two role families to one
line each, giving three lines per panel (CAA, critical mean,
conformist mean) plus the usual random band and baseline. The
aggregation rule is:

1. **Metric.** Plotted values are the already-aggregated multi-seed
   means in `results/sycophancy_rates_test.json` (same numbers fig6
   plots). No re-averaging across seeds happens here.
2. **Inputs per family.**
   - `critical`   = {`skeptic`, `devils_advocate`, `judge`}
   - `conformist` = {`peacekeeper`, `pacifist`, `collaborator`}
   - `caa` is not a family — it is a single vector, plotted as-is with
     no band (its own degraded cells are still dropped).
3. **Degradation mask.** At each coefficient `c`, role `r` is excluded
   from the family mean / min / max if
   `degradation_flags_test.json[r][c] == True` (OR across test seeds,
   matching the `degraded_any_seed` field used elsewhere). This
   prevents a single collapsed forward pass from dragging the family
   curve toward binary rate ≈ 50 % / syc-logit ≈ 0. Example: on
   Qwen, `pacifist @ ±500` and several ±500 cells collapse; on Gemma,
   the ±5000 endpoints collapse for most conditions. Those points are
   omitted from the mean.
4. **Aggregation.** After masking, at each coefficient the family
   *mean* line is the arithmetic mean of the surviving role values
   (`1/k ∑ rate_r` or `1/k ∑ syc_logit_r`, k ≤ 3). The shaded band is
   the (min, max) across surviving members at that coefficient. If
   zero members survive, the point is NaN and the line breaks there.
5. **No within-family sign alignment.** Values are averaged at the raw
   signed coefficient — we do not re-orient per-role curves onto a
   shared "dose" axis. Within the Qwen conformist family the tune-
   locked coefficients are {peacekeeper −200, pacifist +500,
   collaborator −100}, so the three roles push sycophancy in opposite
   coefficient directions; the min/max band therefore widens
   noticeably on that panel. Readers who want each role's own dose-
   response should consult fig6.
6. **Random and baseline.** The random mean (n = 10 vectors) ± std
   band and the baseline (coef = 0) line are drawn identically to
   fig6 — they are not re-averaged at the family level.

The code path is `scripts/make_steering_curves.py:_family_series` (mean
and min/max) and `_plot_family` (rendering).

## Qualitative samples

The A/B logit numbers are the primary signal, but stored decoded
responses show what the tone shift actually looks like. `qualitative/`
collects two kinds of samples:

- **Free-form responses on philosophy prompts** (Gemma only, from
  `sycophancy-final`). Each prompt is a philosopher introducing
  themselves and asking the model to take a side; the response is a
  full open-ended continuation under steering. Stored as
  `qual_check_{caa,conformist}.json`.
- **Over-correction probes** (both models, from
  `sycophancy-{final,qwen}`). Each probe is a domain expert asserting
  either a true or false claim and asking if the model agrees. The
  pipeline auto-categorises each response as
  `AGREE_CORRECT` / `AGREE_INCORRECT` / `REFUSE` / `HEDGE`. Stored as
  `{gemma-2-27b-it,qwen3-32b}_over_correction.json` (trimmed to kept
  conditions).

`gemma_showcase.md` and `qwen_showcase.md` render one representative
prompt per model across the kept conditions so the tone shift is
readable without opening a JSON. `figures/fig5_tone_comparison.pdf`
is a typeset (reportlab) tone-comparison document: John Locke's
empiricism prompt on Gemma and a chemistry professor's false claim on
Qwen, each decoded at baseline / CAA / skeptic / pacifist. Signature
opening sentences are bolded — both skeptics open with "I must
respectfully disagree", both baselines with flattery. Qwen × pacifist
@ +500 is the one cell flagged `degraded`; you can see the collapsed
forward pass as a repetition loop in free text.

## How to reproduce

```bash
cd sycophancy-clean-results
python3 scripts/build_data.py         # rebuilds data/ from source repos
python3 scripts/build_qualitative.py  # rebuilds qualitative/ from source repos
python3 scripts/make_showcase_pdf.py  # rebuilds figures/fig5_tone_comparison.pdf
python3 scripts/make_tables.py      # rebuilds results/
python3 scripts/make_figures.py     # rebuilds figures/ (fig1-4)
python3 scripts/make_steering_curves.py  # rebuilds figures/fig6_steering_curves
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
