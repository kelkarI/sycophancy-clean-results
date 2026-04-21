# Audit notes — branch `claude/audit-repo-consistency-vGCJJ`

This branch resolves a peer-review audit of the clean-results repo
against `kelkari/role-based-steering` (Repo 2) and the upstream source
pipelines (`kelkarI/sycophancy-final/experiment-main/` for Gemma and
`kelkarI/sycophancy-qwen` for Qwen). All changes here are documentation
and metadata — no model re-runs, no changes to the kept-condition Δ
values in `data/{model}_clean.json["conditions"]`.

## What changed

### 1. Dropped conditions are now first-class data, not silently absent

The audit's most consequential finding was that `facilitator` — a
"conformist" role whose tune-locked best coefficient *reduces*
sycophancy on both models (Gemma Δ = −0.727, Qwen Δ = −0.469) — is
silently dropped from the headline tables and figures. Combined with
dropping `contrarian` (the only critical residual that fails Holm on
Gemma) and `scientist` (asymmetric coef sign across models), the drops
disproportionately remove counter-examples to the headline narrative.

**Resolution.** All four dropped conditions (`assistant_axis`,
`contrarian`, `scientist`, `facilitator`) are now included in
`data/{model}_clean.json` under a new top-level key
`"dropped_conditions_for_transparency"`. Each entry has:

- The full per-seed structure (`per_seed[].seed`, `.best_coef`,
  `.post_steer_logit`, `.post_steer_rate`, `.delta_logit`,
  `.delta_rate_pp`, `.wilcoxon_p_adj`).
- Cross-seed `delta_logit_mean` ± `delta_logit_std` and 95% CI
  (t-distribution, df=2) — same aggregation rule as the kept
  conditions.
- A per-condition `exclusion_reason` field explaining why this
  condition was dropped from the headline.

The headline tables (`results/main_table.{csv,md}`) and figures are
unchanged — readers who want the curated view get the curated view;
readers who want to test the curation choices can read the dropped
block directly.

### 2. Per-drop rationale added to README §Scope

The README now contains a per-drop rationale table and a fuller
"Bidirectionality framing" paragraph that distinguishes the
6+1-condition headline (where the framing holds on Gemma) from the
all-conditions picture (where it weakens on Gemma and breaks on Qwen).

### 3. Holm family-size statement clarified

The README §Significance previously said "Holm-corrected across the
source pipeline's 14-condition primary family". That statement is
correct — but `kelkari/role-based-steering/paper/RESULTS.md` said
"Holm across all 24 conditions per seed", which contradicted it. Both
repos now consistently say **14-condition family** (11 main + 3
standalone residuals; the 10 random controls are not in the family).
A note has been added to README §Significance to flag this and to
point readers at the corrected `role-based-steering` text.

### 4. Top-of-README framing softened

The opening sentence used to assert "conformist role directions do not"
[reduce sycophancy]. Updated to acknowledge the heterogeneous picture
that the dropped-conditions reveal.

## What is *not* changed

- All Δ values, CIs, and significance counts in
  `data/{model}_clean.json["conditions"]` (the kept 7 conditions per
  model) are unchanged. Verified byte-identical for those keys.
- All cosine values in `data/{model}_cosines.json` are unchanged.
- All figures, qualitative samples, and `results/main_table.{csv,md}`
  are unchanged. The "main" table remains the curated view; only the
  data-file additions and README prose changed.

## Audit findings still open

These cannot be fixed from this repo alone:

- **HF model revisions for Gemma 2 27B / Qwen 3 32B and the persona
  vectors are not pinned.** The producing pipelines used the floating
  HF "main" revision at run time. To make these results bit-reproducible
  the next run should pin via `from_pretrained(..., revision=SHA)` and
  record the SHA. Tracked on the matching branch in
  `kelkari/role-based-steering` as `_metadata.json`'s `TODO_PIN_AT_RUNTIME`.
- **`build_data.py` hard-codes paths** to `/lambda/nfs/filesystem/...`.
  This is intentionally unportable and noted in the README, but a
  cleaner version would take paths as CLI args.

## Files changed in this commit

- `README.md` — softened headline framing; added per-drop rationale
  table; added bidirectionality nuance paragraph; clarified Holm
  family size; added pointer to AUDIT_NOTES.md.
- `data/gemma-2-27b-it_clean.json` — added top-level
  `dropped_conditions_for_transparency` block with full per-seed Δ
  for `assistant_axis`, `contrarian`, `scientist`, `facilitator`.
- `data/qwen3-32b_clean.json` — same.
- `AUDIT_NOTES.md` — this file.
