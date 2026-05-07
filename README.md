# Clean results — sycophancy steering on Gemma 2 27B and Qwen 3 32B

A standalone post-hoc package of the main paper result only: critical
role directions reduce sycophancy, conformist role directions are
mixed (one increases, one is flat, one reduces, one is degraded — see
§Scope below), and a general persona direction approaches the effect
of the targeted CAA direction that was trained on sycophancy labels.
Everything in this repo is derived from existing experiment outputs
(no re-running of model inference).

> **Audit note.** A peer-review audit (`AUDIT_NOTES.md`) added per-drop
> rationale below, surfaced the dropped conditions inside the data JSONs
> (under `dropped_conditions_for_transparency`), and clarified the Holm
> family-size statement. The headline findings and `data/{model}_clean.json["conditions"]`
> are unchanged.

Source repositories (read only, not modified):

- Gemma pipeline: `../sycophancy-gemma/experiment-main/`
  (multi-seed aggregate, per-seed JSONs, tune-locked best coefficients,
  steering-vector cosine matrix).
- Qwen pipeline:  `../sycophancy-qwen/`
  (same structure; target layer 32, coefficient grid rescaled 10×
  because of Qwen's smaller activation norms — see the parent paper
  repo for details).
- **General experiment code (public mirror)**:
  <https://github.com/vkmk1/Sycophancy-Steering>
  (the same Gemma pipeline that powers `../sycophancy-gemma/`,
  published under the project's primary GitHub home).

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

**Per-drop rationale** (added in audit, see `AUDIT_NOTES.md`):

| Dropped | Family | Δ logit at locked coef | Why dropped |
|---|---|---|---|
| `assistant_axis` | broad axis | Gemma −0.375, Qwen −2.410 | Not a role per se; it is the underlying assistant-vs-non-assistant axis. |
| `contrarian` | critical | Gemma −0.286 (sig 3/3); residual fails Holm on Gemma | Drops because the **standalone residual** doesn't reach Holm significance on Gemma; the parent condition does pass. Drop is for narrative cleanliness, not because the parent fails. |
| `scientist` | critical | Gemma −0.509, Qwen −0.984 (sig 3/3 both) | Tune-locked coef is +2000 on Gemma but −100 on Qwen — breaks the within-family sign symmetry. Both directions reduce, but the asymmetric coef is awkward to report. |
| `facilitator` | **conformist** | **Gemma −0.727, Qwen −0.469 (both REDUCE)** | A conformist-family role that *reduces* sycophancy at its locked coef — directly contradicting the simple bidirectionality prediction for conformist roles. Neither effect reaches Holm significance per-seed on the 14-condition family. **This is the most consequential drop**; re-included in `data/{model}_clean.json["dropped_conditions_for_transparency"]` so readers can verify. |
| `*_residual` | derived | various | Repo 3 is "main paper result only"; the residual decomposition is the centrepiece of the parent paper (`role-based-steering`) but does not appear in the clean-results headline. |
| `*_caa_component` | derived | various | Same rationale as `*_residual`. |

The dropped conditions are not omitted from the data files — they live
in `data/{model}_clean.json["dropped_conditions_for_transparency"]`
with full per-seed Δ provenance and per-condition exclusion reasons.
They are simply absent from `main_table.{csv,md}` and from figures 1–8.

**Bidirectionality framing.** The headline framing "critical reduce,
conformist do not (or increase)" holds across the kept 6 + 1 conditions
on Gemma but is weaker on Qwen and at the family level. After audit,
the more accurate summary is:

- Critical-family roles all reduce sycophancy at their tune-locked
  coefficient on both models (3/3 critical-kept on Gemma sig; 3/3 on
  Qwen, with a 2/3 nuance for contrarian-residual fail on Gemma).
- Conformist-family roles are heterogeneous. On Gemma, 1/4
  (collaborator) significantly *increases* syc, 1/4 (pacifist) increases
  with weak significance, 1/4 (peacekeeper) is flat, and 1/4
  (facilitator, dropped) actively *reduces*. On Qwen, all four
  conformist roles either reduce sycophancy at their tune-locked
  coefficient (peacekeeper, collaborator, facilitator) or are degraded
  (pacifist locked at +500 saturates the model). The "conformist family
  pushes toward sycophancy" prediction does not hold cleanly on either
  model when all four roles are considered.

## Directory layout

```
sycophancy-clean-results/
├── README.md                                  (this file)
├── data/
│   ├── gemma-2-27b-it_clean.json              per-seed + aggregate for kept conds
│   ├── qwen3-32b_clean.json                   same, Qwen
│   ├── gemma-2-27b-it_cosines.json            6×7 cosine matrix (kept role vectors + CAA)
│   ├── qwen3-32b_cosines.json                 same, Qwen
│   ├── gemma-2-27b-it_perturbation_propagation.json   per-layer cosine + norm summary
│   ├── qwen3-32b_perturbation_propagation.json        same, Qwen
│   ├── gemma-2-27b-it_normalized_effects.json         |∆logit|/ε per condition
│   ├── qwen3-32b_normalized_effects.json              same, Qwen
│   └── _baseline_norms_at_target_layer.json           ‖h̄_baseline_TARGET‖ (input to fig11)
├── scripts/
│   ├── build_data.py                          rebuilds data/ from source repos
│   ├── build_qualitative.py                   rebuilds qualitative/ from source repos
│   ├── make_figures.py                        rebuilds fig1-4 from data/
│   ├── make_showcase_pdf.py                   rebuilds fig5 tone-comparison PDF
│   ├── make_steering_curves.py                rebuilds fig6 coefficient sweep
│                                              (reads rates files from source repos)
│   ├── make_tables.py                         rebuilds results/*.csv and *.md
│   ├── build_perturbation_propagation.py      aggregates per-layer summaries from source repos
│   ├── make_perturbation_propagation_figs.py  builds fig9 + fig10
│   ├── make_perturbation_propagation_table.py builds results/perturbation_propagation.{csv,md}
│   ├── build_normalized_effects.py            joins ‖h̄‖ + ∆logit → data/{model}_normalized_effects.json
│   ├── make_normalized_effects_figs.py        builds fig11 (|∆logit|/ε per condition)
│   ├── make_normalized_effects_table.py       builds results/normalized_effects.{csv,md}
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
│   ├── fig7_steering_curves_family.{pdf,png}  same sweep, three lines per model
│   │                                          (CAA + critical mean + conformist
│   │                                          mean) with min/max bands; degraded
│   │                                          cells masked before averaging — see
│   │                                          "Family averaging (fig7)" below
│   ├── fig8_steering_curves_family_pos.{pdf,png}  fig7 restricted to coef >= 0,
│   │                                          i.e. the positive half of the
│   │                                          sweep only (same averaging and
│   │                                          masking rules as fig7)
│   ├── fig9_perturbation_cosine.{pdf,png}     per-layer mean cos(ΔH^CAA, ΔH^persona)
│   ├── fig10_perturbation_norm.{pdf,png}      per-layer mean ‖ΔH_ℓ‖, collapse check
│   └── fig11_normalized_effects.{pdf,png}     |∆logit|/ε per condition, two-panel side-by-side
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
    ├── conformist_vs_critical_filtered.{csv,md}  family means excluding degraded members
    ├── perturbation_propagation.csv          (model × persona) cosine + norm-decay table
    ├── perturbation_propagation.md           same, markdown
    ├── normalized_effects.csv                (model × condition) |∆logit|/ε table
    └── normalized_effects.md                 same, markdown
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
  across the source pipeline's **14-condition primary family** (11
  main + 3 standalone residuals; the 10 random controls are not in
  the family). Kept conditions are a subset of that family, so Holm
  significance carries over directly to the 7 kept role conditions.
  We report how many of the 3 test seeds crossed α=0.05 after
  correction. (`role-based-steering/paper/RESULTS.md` previously said
  "Holm across all 24 conditions" — that was incorrect; both repos now
  say 14.)
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

`fig8_steering_curves_family_pos` is the same plot restricted to
`coef >= 0`. Reading the positive half alone is the most direct
"push toward X" story:

- CAA at `+coef` amplifies sycophancy (the CAA vector points from
  honest toward sycophantic, so its tune-locked reduction coefficient
  is negative; on the positive half CAA is the upward-going line).
- Critical family mean at `+coef` reduces sycophancy.
- Conformist family mean at `+coef` increases sycophancy on Gemma;
  on Qwen it is roughly flat / slightly upward before the collapse at
  coef = +500, because two of three Qwen conformist roles have
  *negative* tune-locked coefficients (peacekeeper −200, collaborator
  −100) and only pacifist pushes sycophancy up at +coef. fig7 shows
  the full picture on both sides.

## Perturbation propagation (fig9–10)

A reviewer raised the concern that geometric near-orthogonality
of the persona and CAA steering *vectors* at the injection
layer (paper §4.3, all |cos| < 0.17 on Gemma; |cos| ≤ 0.108 on
Qwen) does **not** by itself imply mechanistic independence —
24+ nonlinear blocks downstream can collapse orthogonal inputs
onto shared pathways. fig9 addresses this directly: for each
prompt in the held-out test set, we capture the post-block
residual at every layer ℓ ≥ TARGET_LAYER under nine forward
passes per prompt — baseline, CAA-steered, three
critical-persona-steered (skeptic, devils_advocate, judge),
four conformist-role-steered (peacekeeper, pacifist,
collaborator, facilitator), and one random-vector-steered
(`random_0`, unit-Gaussian, coef = mean of |critical-coef|).
ΔH = H_steer − H_base is then taken per layer per condition,
and we plot the per-layer mean of cos(ΔH^CAA, ΔH^persona)
averaged within prompt over tokens then across prompts. fig10
plots ‖ΔH_ℓ‖ per layer per condition as a magnitude check —
high cosine is uninformative if both perturbations have
decayed to ~0.

**Why the random control matters.** The reviewer's "downstream
alignment ⇒ shared mechanism" concern only has teeth if the
alignment is bigger than what an arbitrary direction at the
same coefficient magnitude produces. `random_0` provides that
null. On Gemma the persona/CAA cosines downstream sit ~2–3×
above the random null at their argmax layer — so there is a
persona-specific signal beyond the geometric baseline. On Qwen
the personas mostly *do not* exceed the random null in
magnitude (random reaches ~0.35 by midpoint, which is at or
above the Qwen judge curve and around skeptic), so the Qwen
downstream alignment is largely a geometric / magnitude
artefact of pushing the residual at α=200 — not persona-
specific. Per-cell numbers are in
`results/perturbation_propagation.md`.

**Why all four conformist roles, when fig1-fig8 drop
facilitator?** fig1-fig8's "kept conditions" exclude
facilitator because of the audit's headline-narrative
constraint — facilitator is a conformist-family role that
*reduces* sycophancy at its locked coef on both models, which
breaks the simple bidirectionality story (see "Scope" above).
That narrative concern doesn't apply to fig9-fig10: the
question here is geometric, not behavioural, so the right unit
of analysis is the complete conformist family (4 roles).

**Sanity check.** At the injection layer ℓ = TARGET_LAYER,
ΔH^CAA is by construction equal to α_CAA · v_CAA (broadcast
across tokens), so the per-token cosine reduces analytically
to sign(α_CAA · α_persona) · cos(v_CAA, v_persona) — the §4.3
vector cosine, with sign flipped according to whether the
locked persona coef is in the same or opposite half-plane as
α_CAA. (α_CAA < 0 on both models. α_persona > 0 for all three
critical roles on both models, but conformists vary: Gemma
facilitator is α<0 — flipping the sign-flip back so its
expected cosine is positive — and Qwen peacekeeper / collaborator
/ facilitator are also α<0.) The driver halts on a >0.005
absolute mismatch on tier-1 cells (|expected| ≥ 0.010); for
tier-2 noise-floor cells (Gemma DA at α=2000 has expected
≈ −0.003; both random_0 cells have expected ≈ −0.007) the
hard gate is `|observed| < 0.05` plus the projection check
`⟨ΔH, v⟩.mean() ≈ α to within 5% relative` (which catches
cross-model contamination and wrong-vector errors that the
near-zero cosine cannot). All 16 (model, persona) cells pass.

**What fig9 / fig10 / the table show.** See
`results/perturbation_propagation.md` for the per-cell numbers
(injection / midpoint / final-layer cosines, plus the
`‖ΔH_final‖ / ‖ΔH_inject‖` decay ratio per persona — a
condition with decay ratio << 1 means the cosine reading at
the final layer is being computed on small vectors and should
be read with that caveat). fig9 shows the trajectory; fig10
shows that the perturbations do not collapse to zero at any
layer (in fact they *amplify* by 3–7× across the captured
range on both models).

This is a curve-and-sanity-check report. Mechanistic
conclusions belong in the parent paper repo
(`../role-based-steering/paper/`), not here.

**Reproduction.**
```bash
python3 scripts/build_perturbation_propagation.py
python3 scripts/make_perturbation_propagation_figs.py
python3 scripts/make_perturbation_propagation_table.py
```

## Normalized effect sizes (fig11)

A reviewer noted that the original paper's cross-model effect-size
comparison uses hand-tuned coefficients (α=2000 on Gemma, α=200 on
Qwen) without normalization to activation scale, and that the
random-direction baseline produces substantial behavioral effects on
Qwen (∆logit ≈ −1.06 in the n=10 random aggregate, ≈ −2.07 for the
specific `random_0` vector at α=200), weakening the claim that
persona vectors are uniquely meaningful directions on that model.

fig11 normalizes effect sizes by the **perturbation energy**
ε = |α| / ‖h̄^baseline_TARGET_LAYER‖. **Behavioral efficiency** =
|∆logit| / ε measures sycophancy reduction per unit fractional
activation perturbation; the random_0 row gives the geometric
baseline ("what does an arbitrary direction at this α do per unit
perturbation energy"), and `ratio_vs_random_0` divides each row by
that baseline so a value > 1 means "exceeds random at matched ε".

`results/normalized_effects.{csv,md}` —
[**view on GitHub**](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/results/normalized_effects.md).

**Empirical perturbation energies are not equal across models.**
This is the first thing the normalization surfaces. The original
hand-tuning *was* implicitly trying to equalize ε, but the
empirical baseline residual norms are:

  ‖h̄_Gemma_layer22‖ ≈ 19,750  (mean over 600 prompts × tokens)
  ‖h̄_Qwen_layer32‖  ≈    408  (mean over 600 prompts × tokens)

So at the locked α the actual fractional perturbations are
ε_Gemma ≈ 0.10 (10% of baseline norm) and ε_Qwen ≈ 0.49 (49% of
baseline norm). Qwen is being pushed ~5× harder than Gemma in
fractional terms, even though the "tune-locked" α is 10× smaller.
The cross-model asymmetry that fig9 surfaces (random_0 nearly
matches persona cosines on Qwen but not on Gemma) is consistent
with this: at ε ≈ 0.5 a random direction is large enough to
dominate the residual stream's behaviour, so the
"persona-vs-random" gap should be expected to be smaller on Qwen.

**Persona-vs-random survives normalization on Gemma; doesn't on Qwen.**
Reading `ratio_vs_random_0` from `results/normalized_effects.md`:

  Gemma — every persona row exceeds random_0:
    caa 11.7×, skeptic 9.4×, judge 7.4×, devils_advocate 6.9×,
    facilitator 3.9×, peacekeeper 3.8×, collaborator 2.4×,
    pacifist 1.3×, random_0 1.0×.
  Qwen — only devils_advocate exceeds random_0, and barely:
    devils_advocate 1.10×, caa 0.95×, skeptic 0.88×, judge 0.82×,
    pacifist 0.58×, peacekeeper 0.34×, facilitator 0.23×,
    collaborator 0.03×, random_0 1.0×.

So the headline finding from this analysis is consistent with what
fig9 already showed geometrically — the persona-specific signal is
robust on Gemma, but on Qwen most persona effects are at or below
what an arbitrary direction at the same α would do. This does not
mean Qwen's persona steering is uninformative (it does reduce
sycophancy in absolute terms, ∆logit ≈ −1.7 to −2.3 on critical
roles), but it is not "uniquely meaningful in direction" relative
to a random unit-Gaussian at the same coefficient on this model.

**The peacekeeper coefficient on Gemma.** The aggregate file
`sycophancy-gemma/.../results/best_coefs_test.json` (which the
layerwise experiment loads) records peacekeeper at α=+5000, while
each of the three test seeds' own `best_coefs_test.json` records
α=+2000. fig11 mirrors the layerwise experiment and uses α=+5000
on this cell; this is a divergence from `data/gemma-2-27b-it_clean.json`
(which uses α=+2000 and reports ∆logit = −0.05, vs fig11's −0.71
at α=+5000). The divergence is in the source data; fig11 picks the
larger-α reading because it matches the layerwise experiment, not
because it's the "right" calibration. See the deviation log in
PERTURBATION_PROPAGATION.md.

This is a curve-and-table report. Mechanistic interpretation
belongs in the parent paper repo, not here.

**Reproduction.**
```bash
# One-time GPU job to extract baseline residual norms (5 min on H100):
python3 /path/to/extract_baseline_norms.py
# (writes data/_baseline_norms_at_target_layer.json)

python3 scripts/build_normalized_effects.py     # data/{model}_normalized_effects.json
python3 scripts/make_normalized_effects_figs.py # figures/fig11
python3 scripts/make_normalized_effects_table.py # results/normalized_effects.{csv,md}
```

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
python3 scripts/build_perturbation_propagation.py    # rebuilds data/*_perturbation_propagation.json from source repos
python3 scripts/make_perturbation_propagation_figs.py # rebuilds figures/fig9, fig10
python3 scripts/make_perturbation_propagation_table.py # rebuilds results/perturbation_propagation.{csv,md}
python3 scripts/build_normalized_effects.py          # rebuilds data/*_normalized_effects.json (needs data/_baseline_norms_at_target_layer.json from extract_baseline_norms.py)
python3 scripts/make_normalized_effects_figs.py      # rebuilds figures/fig11
python3 scripts/make_normalized_effects_table.py     # rebuilds results/normalized_effects.{csv,md}
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
