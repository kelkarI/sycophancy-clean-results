# Revised experimental methodology — `sycophancy-clean-results`

## 0. Relation to the prior experiment

`role-based-steering` (the "full" experiment) ran a **24-condition steering protocol** on two models (Gemma 2 27B, Qwen 3 32B) to answer three stacked questions: (Q1) does a general persona direction match a targeted CAA direction; (Q2) does the reduction live in the CAA-aligned or CAA-orthogonal component; (Q3) does the CAA-orthogonal residual reduce sycophancy as a standalone steering direction, with a sub-claim about high-|cos(role, CAA)| residuals.

`sycophancy-clean-results` is **a deliberately reduced, post-hoc package** of the same underlying inference runs, restricted to the primary finding only: *critical role directions reduce sycophancy at rates comparable to a targeted CAA direction; conformist directions do not.* The decomposition questions (Q2, Q3) are removed from scope, as are all their associated conditions. Every number in this repo is derived from outputs already written to disk by the upstream Gemma and Qwen pipelines — no model is re-loaded and no inference is re-run.

The revision is therefore a **narrowing**, not a re-experiment. The methodology below specifies exactly what was kept, what was dropped, how reporting changed, and why.

---

## 1. Research questions

The revised experiment answers a **single** primary question, with one secondary check:

- **RQ1 (primary).** At each condition's tune-locked best steering coefficient on the held-out `philpapers2020` test split, does steering with a critical-role persona direction (Skeptic / Devil's Advocate / Judge) reduce sycophancy as measured by (a) A/B syc-logit and (b) binary sycophancy rate, and how does the size of that reduction compare to (i) the targeted CAA vector and (ii) a conformist-role direction (Peacekeeper / Pacifist / Collaborator)?
- **RQ2 (bidirectionality).** Do conformist directions fail to reduce sycophancy, and where they move it at all, do they increase it? This is reported as a family-level contrast (critical vs conformist), with Qwen explicitly flagged as ceiling-constrained.

Dropped from scope: the residual-vs-aligned decomposition (Q2 of the prior paper); the standalone-residual Holm test (Q3); and the high-|cos| residual sub-claim (Q3a). All three belong in a follow-on paper, not in this package.

---

## 2. Condition set (8 per model, vs 24 in the prior experiment)

| Family | Kept | Dropped |
|---|---|---|
| Targeted | `caa` | — |
| Critical role | `skeptic`, `devils_advocate`, `judge` | `contrarian`, `scientist` |
| Conformist role | `peacekeeper`, `pacifist`, `collaborator` | `facilitator` |
| Axis | — | `assistant_axis` (dropped) |
| Decomposition | — | all `*_residual`, all `*_caa_component` |
| Null control | `random` (aggregate over 10 unit-Gaussian seeds) | — |

**Rationale for drops.**

- `assistant_axis` is the full persona-space axis — relevant for the decomposition story but redundant for the family-contrast story, where a single cleanly labelled "Skeptic" persona carries the evidence.
- `*_caa_component` is mathematically degenerate after unit-normalisation: the projection of any role onto `v_CAA` reduces to `sign(v_role · v_CAA) · v_CAA`, i.e. ±CAA. Reporting per-role aligned components reports the same intervention multiple times. Keeping them was only defensible in the decomposition paper.
- `*_residual` conditions are Q3 material; they also have a seed asymmetry on Gemma (single-seed) that the clean package refuses to paper over.
- `contrarian` and `scientist` overlap heavily with `skeptic` and `devils_advocate` in cosine space on both models; `facilitator` overlaps with `peacekeeper`. The reduction keeps three members per family so family means are stable without over-representing near-duplicates.

**Result of the narrowing.** The single-question focus is: *does a role vector reduce sycophancy at its tune-locked coefficient, and is the direction the expected one for its family?*

---

## 3. Models, target layers, and infrastructure (inherited unchanged)

| | Gemma 2 27B | Qwen 3 32B |
|---|---|---|
| HF id | `google/gemma-2-27b-it` | `Qwen/Qwen3-32B` |
| Decoder layers | 46 | 64 |
| Hidden size | 4608 | 5120 |
| Target steering layer | 22 | 32 |
| Single-token A/B ids | 235280 / 235305 | 32 / 33 |
| Chat-template flag | — | `enable_thinking=False` (required so the A/B logit is measured post-`</think>`) |
| Precision | bfloat16 | bfloat16 |
| Steering library | `assistant_axis.steering.ActivationSteering`, `addition` mode, applied at every prompt+template token |

Steering applies `coef × v` to the residual stream at the target layer, where `v` is unit-normalised. The target layers are the canonical mid-stack layers from `assistant_axis.models.MODEL_CONFIGS`.

All of the above is inherited verbatim from the upstream pipelines; the clean-results package does no model loading. See `scripts/build_data.py:44-52` for the source paths.

---

## 4. Vectors (provenance, kept-set only)

- **Role vectors** (`skeptic`, `devils_advocate`, `judge`, `peacekeeper`, `pacifist`, `collaborator`). Downloaded from `lu-christina/assistant-axis-vectors` (`gemma-2-27b/` and `qwen-3-32b/` subdirs), produced by the `safety-research/assistant-axis` generation-plus-LLM-judge pipeline. For each role `R`: `v_R = unit(role_mean_activation[L] − default_mean_activation[L])`.
- **CAA vector.** Extracted per-model from 2,000 A/B pairs from `sycophancy_on_nlp_survey` + `sycophancy_on_political_typology_quiz` (disjoint from the philpapers2020 eval set), at the target layer. `v_CAA = unit(mean_syc_activation − mean_hon_activation)`. Metadata in `{model}/caa_metadata.json` upstream.
- **Random controls.** Ten unit-Gaussian vectors per model, seeded `RANDOM_SEED_BASE + i` for `i ∈ [0, 10)`. `torch.randn(d)` then unit-normalised.

**Cosine summary (kept vectors only).** The clean package ships a 7×7 cosine matrix per model in `data/{model}_cosines.json` covering the six kept roles and CAA. All role↔CAA cosines are < 0.17 on Gemma and < 0.11 on Qwen; every role is within ~10° of orthogonal to CAA.

---

## 5. Benchmark and splits (inherited)

- **Evaluation set.** `sycophancy_on_philpapers2020.jsonl` from `anthropics/evals` (commit `5525210614d4f26b1732042e7dcb7210d23fe5aa`). 300 base questions, each evaluated with counterbalanced A/B orderings → 600 rows per seed.
- **Tune/test split.** 50/50 at the base-question level under `TUNE_TEST_SEED = 99`; A-first and B-first orderings of the same base stay together. Held-out test: 150 base × 2 orderings = 300 rows.
- **Question-level disjointness from CAA training.** The CAA training sets (`nlp_survey`, `political_typology_quiz`) share no question text with `philpapers2020`; this is a transfer eval, not an in-domain fit.

---

## 6. Sampling seeds

| Split | Seeds | Used for |
|---|---|---|
| Tune | 42, 7, 123, 456, 789 | Best-coefficient locking only; never reported as a headline number |
| Test | 42, 7, 123 | Every number reported in this package |

All three test seeds are reported for all 7 real conditions and for the random pool on both models. This removes the single-seed asymmetry that affected the prior experiment's Gemma residual conditions — since residual conditions are dropped from scope, **every kept condition × model cell has 3 test seeds**.

`build_data.py` enforces this symmetry by reading only `results/multiseed_aggregate_test.json` from each upstream pipeline, which is the 3-test-seed aggregate.

---

## 7. Coefficient sweep and locking

### 7.1 Sweep (model-specific, inherited)

- **Gemma:** `[−5000, −2000, −1000, −500, 0, +500, +1000, +2000, +5000]`
- **Qwen:**  `[−500, −200, −100, −50, 0, +50, +100, +200, +500]`

The 10× rescale is informed by a ~116× raw-norm ratio between the two models' CAA vectors (Gemma L22 raw norm ≈ 740.85; Qwen L32 raw norm ≈ 6.38). The rescale is hand-tuned, not calibrated: chosen so that locked best coefficients land in the interior of each sweep (never at ±max) and so the next-token distribution remains interpretable. This is explicitly flagged as a limitation; a formal KL- or Mahalanobis-matched rescale is out of scope.

### 7.2 Best-coefficient locking (direction-aware)

For each (seed, condition), the selector picks the coefficient maximising the excess over the random-mean logit at that coefficient:

- `decrease` conditions (`caa`, `skeptic`, `devils_advocate`, `judge`): largest positive excess `rand_mean_logit(c) − cond_logit(c)`.
- `increase` conditions (`peacekeeper`, `pacifist`, `collaborator`): largest negative excess.
- `unsigned` (`random_*`, only used for the null pool): largest `|excess|` — but the random selector does *not* feed into the aggregate null band; see §8.

The per-seed selections are aggregated across **tune** seeds by mode (tie-break: count, then proximity to median), producing `best_coefs_tune_aggregate.json`. This aggregate coefficient is what the test runs are evaluated at, via `--locked-coefs-from` upstream.

The clean package reads the aggregate locked coefficient per (model, condition) from upstream (`results/best_coefs_tune_aggregate.json`), copies per-seed locked coefficients into `data/{model}_clean.json:conditions.{cond}.per_seed[i].best_coef`, and reports the aggregate locked coefficient in all tables.

---

## 8. Metrics

Let `b_i = baseline_logit(seed_i)` and `b_r_i = baseline_rate(seed_i)`, measured at `coef=0` of any real condition (all real conditions give identical baseline forward passes by construction).

For a real condition `C` at its locked coefficient `c*`:

- **Per-seed sycophancy logit:** `post_steer_logit(C, i) = mean_{rows in test} (logp(syc_token) − logp(hon_token))` at the last prompt token.
- **Per-seed sycophancy rate:** `post_steer_rate(C, i) = frac_rows where argmax(logp_A, logp_B) = syc_label`.
- **Per-seed Δlogit:** `post_steer_logit − b_i`.
- **Per-seed Δrate (pp):** `(post_steer_rate − b_r_i) × 100`.
- **Cross-seed aggregate:** mean, std (`ddof=1`), and 95% confidence interval using a Student-t interval at `df = n−1`. For `n=3`, `t* = 4.3027` (hard-coded as `T975_DF2` in `scripts/build_data.py:60`). This is reported for both Δlogit and Δrate.

**Null-control aggregation (new, revised from the prior paper).** For each test seed `i`, pool all `random_{0..9}` rows at every non-zero sweep coefficient (10 vectors × 8 coefficients = 80 paired samples per seed) and compute the mean Δ. Report the cross-seed mean of these three per-seed means with the same `t, df=2` interval. This is framed as "any random steering, any coef" null — a looser and less self-serving null than the direction-aware best-coef null, deliberately chosen to avoid a selector that can *find* a significant random intervention.

Rationale: in the prior paper, the direction-aware random selector would sometimes lock a specific random seed onto a sweep extreme and produce a misleadingly large random null. Aggregating across coefficients defuses that.

---

## 9. Statistical tests

Inherited from the upstream pipelines, with no re-computation in the clean package:

- **Primary test.** Paired one-sided Wilcoxon signed-rank on base-level Δlogit (steered − baseline), `n = 150 bases` per test seed. Direction set by `EXPECTED_DIRECTION`: `less` for `caa` and critical roles, `greater` for conformist roles.
- **Multiple-comparison correction.** Holm–Bonferroni across the source pipeline's 14-condition primary family, per seed. The kept conditions in this package are a **subset** of that family, so Holm-adjusted p-values carry over directly and are already conservative relative to a re-computed Holm on just the 8 kept conditions. We **do not** re-compute Holm on the narrower family — doing so would silently increase significance by shrinking the correction factor.
- **Reporting convention.** For each (model, condition) we report how many of the 3 test seeds crossed α = 0.05 after Holm correction in the full-family sense:

  | Marker | Meaning |
  |---|---|
  | `***` | 3/3 |
  | `**`  | 2/3 |
  | `*`   | 1/3 |
  | `ns`  | 0/3 |

- **Secondary tests.** McNemar on per-row correct/incorrect, and bootstrap 95% CIs on the mean of per-base paired differences (2000 iterations, percentile method), are computed upstream but not surfaced in clean-results tables; they remain in the per-model `statistical_tests_test.json` referenced from upstream.

The random null is not given a paired Wilcoxon because it is a pooled null distribution rather than a single operating point.

---

## 10. Degradation handling (promoted to a first-class reporting axis)

At large steering coefficients the steered forward pass can collapse: binary rate drifts to 0.5, the A/B logit gap shrinks to ~0, and the next-token distribution becomes near-uniform over whitespace and control tokens. In the prior experiment this was documented but handled ad-hoc; in the clean-results package it is promoted to **a first-class dimension of every table and figure**.

### 10.1 Flag definition (inherited)

A (condition, coefficient) cell is `degraded = True` when **both**:

- `|post_steer_rate − 0.5| < 0.03`, **and**
- `|post_steer_logit − random_mean_logit(c)| < 0.10`

i.e. the model is near-uniform on A/B and its logit is indistinguishable from a random steering vector at the same magnitude. Flags come from the upstream `degradation_flags_test.json`, per seed where available with an aggregate-level fallback.

### 10.2 Per-cell propagation

For each (model, condition, seed), `data/{model}_clean.json:conditions.{cond}.per_seed[i].degraded` is the upstream flag at the locked coefficient for that seed. Two derived booleans are surfaced:

- `degraded_any_seed` — the cell is degraded in at least one test seed.
- `degraded_all_seeds` — the cell is degraded in all three.

### 10.3 Dual reporting (core and filtered)

Every headline artefact ships in two flavours:

- **Core** (`main_table.*`, `fig1-3.*`, `conformist_vs_critical.*`): every kept condition is plotted, but degraded cells are **hatched** on bars (`hatch='//'`), **red-× overlaid** on per-seed dots, and **daggered (`†`)** in tables with a footnote pointing to the degradation-flag file.
- **Filtered** (`*_filtered.*`): any condition with `degraded_any_seed == True` is dropped entirely — no hatching, no dagger, no cell.

This design answers two questions simultaneously: *"What does the coefficient-locking procedure pick for every condition?"* (core) and *"What happens if we restrict to conditions where the forward pass is on-manifold?"* (filtered).

**Empirical scope of degradation.** With the current locking, exactly one cell is currently degraded at its locked coefficient: **Qwen × pacifist at coef +500** (all 3 test seeds). No other (model, condition) is degraded at its locked coef. Pacifist moves into the filtered family summary with a flag indicating that 1 of 3 family members was excluded.

---

## 11. Qualitative evidence

Preserved from the prior experiment with two levels of slimming:

### 11.1 Free-form philosophy responses (Gemma only)

Copied from `sycophancy-final/experiment-main/results/qual_check_{caa,conformist}.json`. Five philosopher-voiced prompts (e.g. John Locke introducing himself and asking the model to take a side on empiricism vs rationalism) × {baseline, caa @ −2000, skeptic @ +2000, collaborator @ +500, peacekeeper @ +2000, pacifist @ +2000}. Each response is a full greedy continuation under steering, with `syc_logit` scored on the A/B completion.

Rendered in `qualitative/gemma_showcase.md` as a single prompt × 6 conditions tone contrast.

### 11.2 Over-correction probes (both models)

16 probes × {baseline, caa, skeptic, devils_advocate, judge, peacekeeper, pacifist, collaborator, random_0} per model. Each probe is a domain expert asserting a true or false claim and asking whether the model agrees. Auto-classified by the upstream pipeline into `{AGREE_CORRECT, AGREE_INCORRECT, REFUSE, HEDGE}`. Stored in `qualitative/{model}_over_correction.json`, trimmed from the full 14-condition upstream set to the 9 conditions in the clean scope.

Rendered in `qualitative/qwen_showcase.md` as two probes (arithmetic, chemistry-false) × 8 conditions.

### 11.3 Typeset tone comparison

`figures/fig5_tone_comparison.pdf` is a reportlab-typeset, single-prompt-per-model showcase: Locke/empiricism on Gemma and a chemistry professor's false claim on Qwen, each decoded at {baseline, CAA, skeptic, pacifist}. Signature opening sentences are bolded — both skeptics open with "I must respectfully disagree", both baselines with flattery. Qwen × pacifist @ +500 is the one degraded cell; its repetition-loop text is included as a worked example of what model-collapse looks like.

---

## 12. Figure inventory

| File | Contents | Core / Filtered |
|---|---|---|
| `fig1_delta_logit.{pdf,png}` | Paired bar chart, 2 panels (Gemma / Qwen), 8 conditions each. Bars = Δlogit; errorbars = 95% CI across 3 test seeds; asterisks = Holm-sig per seed count. | both |
| `fig2_delta_rate.{pdf,png}` | Same layout, Δrate (percentage points). | both |
| `fig3_per_seed.{pdf,png}` | Per-seed consistency dot plot; 3 dots per condition, horizontal bar at per-condition mean; red × overlays on degraded dots. | both |
| `fig4_cosines.{pdf,png}` | 7×7 cosine heatmap per model (6 roles + CAA). | core only |
| `fig5_tone_comparison.pdf` | Reportlab typeset qualitative showcase. | core only |

Fig 4 is core-only because cosine geometry is independent of degradation. Fig 5 is core-only because it includes Qwen × pacifist @ +500 precisely to exhibit what degradation looks like in text.

---

## 13. Software stack and determinism

- `torch 2.7`, `transformers 5.5`, `accelerate 1.13`, `statsmodels 0.14`; the `assistant_axis` library pinned to the HEAD of `safety-research/assistant-axis` at session time. These are inherited via the upstream pipelines.
- All clean-package scripts are CPU-only and deterministic — no RNG draws at analysis time. Sort order is fixed in `_style.py:DEFAULT_ORDER = ["caa"] + CRITICAL + CONFORMIST + ["random"]`.
- Plots use a colorblind-safe palette (critical = blue family, conformist = orange family, CAA = neutral gray, random = light gray), serif text, `pdf.fonttype=42` for embedded TrueType.

---

## 14. Reproducibility

`build_data.py` requires the Gemma and Qwen source pipelines at
`../sycophancy-gemma/experiment-main/` and `../sycophancy-qwen/` (paths hard-coded, easily edited). Given those, the full pipeline is:

```bash
python3 scripts/build_data.py         # data/ from source repos
python3 scripts/build_qualitative.py  # qualitative/ from source repos
python3 scripts/make_tables.py        # results/{core, filtered}
python3 scripts/make_figures.py       # figures/fig1-4
python3 scripts/make_showcase_pdf.py  # figures/fig5
```

The upstream Gemma and Qwen pipelines are themselves fully reproducible from their respective repos, including raw steering vectors, per-seed checkpoints, and per-row eval outputs. The clean-results package intentionally **does not** ship those — they are large and rebuildable.

---

## 15. Scope and limitations deliberately inherited or newly adopted

1. **One benchmark.** All numbers are on `philpapers2020`. The two other A/B sycophancy benchmarks in the same Perez et al. series (`nlp_survey`, `political_typology_quiz`) are used for CAA training and cannot be used as eval without leakage.
2. **Two models at similar scale.** Gemma 2 27B and Qwen 3 32B; no Llama 3.3 70B replication even though `lu-christina/assistant-axis-vectors` ships for it.
3. **Single-layer, rank-1 steering.** No layer sweep, no multi-layer composition.
4. **Coefficient-scale calibration is hand-tuned, not principled.** The 10× Qwen rescale is explicitly documented as informed-by-norm, not formally matched.
5. **Qwen bidirectionality is ceiling-constrained.** Baseline rate 84% leaves ~16 pp of headroom before saturation; Pacifist-at-+500 degrades exactly where the direction-aware selector wants to push it. **Any bidirectionality claim in this package should cite Gemma (baseline 59%) as primary evidence and Qwen as ceiling-constrained, not a clean test.**
6. **Qualitative labels are keyword-rule classified**, not LLM-judged or human-rated. Over-correction categories are indicative, not authoritative.
7. **Random null is a pooled mean across coefficients**, which systematically under-states the tails — if readers interpret the null band as "what a single random steering vector at a good coefficient looks like," they will over-estimate the null. The null is explicitly framed as "typical random intervention" to avoid this.
8. **What is NOT tested here (but was in the parent paper).** Residual steering as a standalone condition; the CAA-aligned component; the high-|cos(role, CAA)| residual contrast; the `assistant_axis` full-space axis; the Gemma-only response-token projection (`02d`). These remain in `role-based-steering` and should be cited there.

---

## 16. Summary of differences from the prior experiment (one-page view)

| Dimension | `role-based-steering` (prior) | `sycophancy-clean-results` (this) |
|---|---|---|
| Conditions per model | 24 | 8 |
| Research questions | 3 (persona-vs-CAA, decomposition, standalone residual) | 1 + family-level bidirectionality |
| Test seeds per condition | 3 — except Gemma residuals (1) | 3 for every kept cell |
| Residual / aligned-component eval | Yes (Q2, Q3) | No (out of scope) |
| Bidirectionality framing | Gemma & Qwen pooled | Gemma primary; Qwen flagged ceiling-constrained |
| Degradation handling | Documented, ad-hoc | First-class: hatch / ×-overlay / dagger in core; full drop in filtered |
| Random null | Direction-aware best-coef selector per seed | Pooled over 10 vectors × 8 non-zero coefs (n=80) per seed |
| Multiple-comparison correction | Holm over 24 | Inherited Holm over source pipeline's 14-condition primary family (kept conditions are a subset) |
| Qualitative artefacts | `qual_check_*`, `over_correction_eval_test`, per-model figures | Same inputs, slimmed to kept conditions, plus typeset fig5 |
| Code footprint | Full pipeline incl. inference | Analysis-only; no model loading |
| Deliverable | 8-page workshop paper + full tables | Clean-results package suitable for a single headline figure + supporting material |

---

This is the full revised methodology. The revision's operating principle is: **narrow the question, keep the inference outputs untouched, promote degradation from a footnote to a first-class axis, and refuse to paper over the Gemma/Qwen asymmetry (seed asymmetry, ceiling effect, coefficient-scale calibration) that the prior experiment could only gesture at.**
