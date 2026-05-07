# Diagnostic: the 0.88× Qwen-skeptic-vs-random ratio

This document diagnoses a specific arithmetic discrepancy raised against
`results/normalized_effects.md`: the claim that on Qwen,
`skeptic / random_0 = 0.88×` is at odds with paper Table 1, which would
predict `1.82 / 1.058 = 1.72×`.

**One-line answer.** Cause (b) in the prompt's enumeration: the random
∆logit value used in fig11 is *not* the paper's headline `random`
aggregate (−1.058 on Qwen / −0.254 on Gemma). It is a different,
narrower quantity — `random_0` at α=±|critical-coef| specifically —
which produces −2.068 on Qwen and −0.075 on Gemma. Both numbers are
correctly loaded from the source data, both are well-defined, but they
are *different baselines* and the user's expected 1.72× corresponds to
the paper's broader baseline. fig11's 0.88× corresponds to the
narrower, ε-matched baseline. No bug in the code or the formula.
Methodological choice that should have been explicit in the README.

---

## Section 1 — Code (literal lines from `scripts/build_normalized_effects.py`)

The two arithmetic lines that produce `behavioral_efficiency`:

```python
# scripts/build_normalized_effects.py:178
h_norm = float(norm_block["h_baseline_norm_at_target_layer"]["mean"])

# scripts/build_normalized_effects.py:194
epsilon = abs(alpha) / h_norm if alpha is not None else None

# scripts/build_normalized_effects.py:198
eff = (abs(dlm) / epsilon) if (dlm is not None and epsilon and epsilon > 0) else None
```

So `behavioral_efficiency = |delta_logit_mean| / (|alpha| / h_baseline_norm)`
= `|delta_logit_mean| * h_baseline_norm / |alpha|` — exactly `|∆logit|/ε`
as documented. **Formula is correct; cause (d) is ruled out.**

The `delta_logit_mean` itself is built by `_aggregate_delta_logit`
(lines 103-143):

```python
# scripts/build_normalized_effects.py:112
alpha = random_alpha if cond == RANDOM_CONTROL else float(locked[cond])
# scripts/build_normalized_effects.py:114-126
for seed in TEST_SEEDS:
    p = repo_results / seed / "sycophancy_rates_test.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    if cond not in d:
        continue
    key = f"{float(alpha):.1f}"
    if key not in d[cond] or "0.0" not in d[cond]:
        continue
    steered = d[cond][key]["mean_syc_logit"]
    baseline = d[cond]["0.0"]["mean_syc_logit"]
    per_seed.append(steered - baseline)
```

Where `random_alpha` (line 184 in `_build_one`) is

```python
# scripts/build_normalized_effects.py:184
random_alpha = float(np.mean([abs(locked[r]) for r in CRITICAL_PERSONAS]))
```

i.e. the random_0 vector is evaluated at the **same** α magnitude the
layerwise experiment's driver used:

  `mean(|locked_skeptic|, |locked_devils_advocate|, |locked_judge|)`
  = mean(2000, 2000, 2000) = **2000 on Gemma**;
  = mean(200, 200, 200) = **200 on Qwen**.

This is the layerwise-driver-aligned random null. **Critical:** this is
*not* the same baseline as the paper's `random` aggregate
(`data/{model}_clean.json["conditions"]["random"]`), which averages
over 10 random vectors × 8 nonzero coefs (n=80 cells per seed).

---

## Section 2 — Output JSON values (verbatim)

### Qwen — `data/qwen3-32b_normalized_effects.json["conditions"]`

```json
"skeptic": {
  "alpha": 200.0,
  "perturbation_energy": 0.48983110655186657,
  "delta_logit_mean": -1.823124987946616,
  "delta_logit_std": 0.058169051441661006,
  "delta_logit_ci95_lo": -1.8889494662629784,
  "delta_logit_ci95_hi": -1.7573005096302536,
  "delta_logit_per_seed": [-1.7995, -1.8894, -1.7804],
  "n_seeds": 3,
  "behavioral_efficiency": 3.721946122981823,
  "argmax_cosine_with_caa": 0.3586561679840088,
  "argmax_layer": 49
}
"random_0": {
  "alpha": 200.0,
  "perturbation_energy": 0.48983110655186657,
  "delta_logit_mean": -2.0679861117733846,
  "delta_logit_std": 0.052776384029809414,
  "delta_logit_ci95_lo": -2.127708212445257,
  "delta_logit_ci95_hi": -2.008264011101512,
  "delta_logit_per_seed": [-2.0087, -2.1100, -2.0852],
  "n_seeds": 3,
  "behavioral_efficiency": 4.221835004172837,
  "argmax_cosine_with_caa": 0.34346622228622437,
  "argmax_layer": 47
}
"caa": {
  "alpha": -200.0,
  "perturbation_energy": 0.48983110655186657,
  "delta_logit_mean": -1.9650694509678415,
  "delta_logit_std": 0.12618266847086596,
  "delta_logit_per_seed": [-1.8271, -2.0746, -1.9935],
  "n_seeds": 3,
  "behavioral_efficiency": 4.0117285829412035
}
"judge": {
  "alpha": 200.0,
  "perturbation_energy": 0.48983110655186657,
  "delta_logit_mean": -1.698888888756434,
  "delta_logit_std": 0.07502673557508553,
  "delta_logit_per_seed": [-1.6148, -1.7590, -1.7229],
  "n_seeds": 3,
  "behavioral_efficiency": 3.4683156419273757
}
```

### Gemma — `data/gemma-2-27b-it_normalized_effects.json["conditions"]`

```json
"skeptic": {
  "alpha": 2000.0,
  "perturbation_energy": 0.10126481272371579,
  "delta_logit_mean": -0.7108159750699997,
  "delta_logit_per_seed": [-0.7179, -0.6954, -0.7192],
  "n_seeds": 3,
  "behavioral_efficiency": 7.019377767570093
}
"random_0": {
  "alpha": 2000.0,
  "perturbation_energy": 0.10126481272371579,
  "delta_logit_mean": -0.07550347619586524,
  "delta_logit_per_seed": [-0.0840, -0.0678, -0.0747],
  "n_seeds": 3,
  "behavioral_efficiency": 0.7456042643545288
}
```

---

## Section 3 — Hand-computed ratios from the JSON values

`behavioral_efficiency / random_0_efficiency`, computed in plain
Python from the JSON fields above:

  Qwen `random_0` efficiency   = 3.7219 / 0.4898 = 4.2218
  Qwen `skeptic`  efficiency   = 1.8231 / 0.4898 = 3.7219
  Qwen `judge`    efficiency   = 1.6989 / 0.4898 = 3.4683
  Qwen `caa`      efficiency   = 1.9651 / 0.4898 = 4.0117

  Qwen skeptic / random_0 = 3.7219 / 4.2218 = **0.8816**  ✓ matches `0.88×` in summary
  Qwen judge   / random_0 = 3.4683 / 4.2218 = **0.8215**  ✓ matches `0.82×` in summary
  Qwen caa     / random_0 = 4.0117 / 4.2218 = **0.9502**  ✓ matches `0.95×` in summary

  Gemma random_0 efficiency = 0.0755 / 0.1013 = 0.7456
  Gemma skeptic  efficiency = 0.7108 / 0.1013 = 7.0194
  Gemma skeptic / random_0  = 7.0194 / 0.7456 = **9.4143**  ✓ matches `9.41×` in summary

The reported summary numbers are arithmetically consistent with the
JSON. **No bug; cause (e) is ruled out.**

---

## Section 4 — Source ∆logit verification

For each cell, I re-load the per-seed `sycophancy_rates_test.json` and
recompute (steered − baseline) by hand. Compared against:
  - what `data/{model}_normalized_effects.json` stores (column "JSON")
  - what paper Table 1 / `data/{model}_clean.json` reports (column "paper")
  - the discrepancy with the "expected" value in the user's prompt

**Source file path** (per model):
- Qwen:  `sycophancy-qwen/results/seed_{42,7,123}/sycophancy_rates_test.json`
- Gemma: `sycophancy-gemma/experiment-main/results/seed_{42,7,123}/sycophancy_rates_test.json`

**Recomputed ∆logit values** (mean over 3 test seeds of
`mean_syc_logit[cond][α] − mean_syc_logit[cond]["0.0"]`):

| cell | α | per_seed | mean | JSON | paper expects |
|---|---:|---|---:|---:|---:|
| Qwen skeptic | +200 | [-1.7996, -1.8894, -1.7804] | **−1.8231** | −1.8231 ✓ | −1.82 ✓ |
| Qwen judge | +200 | [-1.6148, -1.7590, -1.7229] | **−1.6989** | −1.6989 ✓ | −1.70 ✓ |
| Qwen caa | −200 | [-1.8271, -2.0746, -1.9935] | **−1.9651** | −1.9651 ✓ | −1.97 ✓ |
| Qwen random_0 | +200 | [-2.0087, -2.1100, -2.0852] | **−2.0680** | −2.0680 ✓ | (paper has −1.058 for `random` *aggregate*; see below) |
| Gemma skeptic | +2000 | [-0.7179, -0.6954, -0.7192] | **−0.7108** | −0.7108 ✓ | −0.711 ✓ |
| Gemma random_0 | +2000 | [-0.0840, -0.0678, -0.0747] | **−0.0755** | −0.0755 ✓ | (paper has −0.254 for `random` *aggregate*) |

**The paper's `random` is a different quantity.** From
`data/qwen3-32b_clean.json["conditions"]["random"]`:

```
delta_logit_mean = -1.0581  (CI95 [-1.2493, -0.8669])
description = "per seed: mean Δ over 10 random unit-Gaussian
              steering vectors at all 8 non-zero coefs (n=80 per
              seed); aggregated across seeds."
```

So the paper aggregate averages **10 random vectors × 8 nonzero coefs
(±50, ±100, ±200, ±500 on Qwen) = 80 cells**, while fig11's
`random_0` is **a single specific random vector at α=+200 on Qwen**.

These are different averages. The paper's averages dilute the larger-α
cells (which produce big ∆logits) with smaller-α cells (which produce
small ∆logits), pulling toward zero on Gemma and producing a moderate
value on Qwen. fig11's `random_0` is ε-matched to the personas it's
being compared to, so all comparisons in fig11 are "at the same α
magnitude" — which is what `|∆logit|/ε` requires for the ratio to be
interpretable.

**Cause (a) is ruled out for skeptic/judge/caa** — those values match
the paper. **Cause (b) is confirmed for random_0** — a different, more
specific, larger-magnitude random number is loaded.

---

## Section 5 — Source α verification

| cell | α used (JSON) | α from `best_coefs_test.json` | Match? |
|---|---:|---:|---|
| Qwen skeptic   | +200  | +200  | ✓ |
| Qwen judge     | +200  | +200  | ✓ |
| Qwen caa       | −200  | −200  | ✓ |
| Qwen random_0  | +200  | (derived: mean(|+200, +200, +200|) = +200) | ✓ |
| Gemma skeptic  | +2000 | +2000 | ✓ |
| Gemma random_0 | +2000 | (derived: mean(|+2000, +2000, +2000|) = +2000) | ✓ |

All four Qwen cells in question (skeptic, random_0, caa, judge) are at
**|α| = 200 → ε = 0.4898** (identical). Same for Gemma at |α| = 2000
→ ε = 0.1013. So the ratio
`behavioral_efficiency_persona / behavioral_efficiency_random` reduces
*algebraically* to `|∆logit_persona| / |∆logit_random_0|`, since the ε's
cancel. The full calculation is:

  Qwen: 1.8231 / 2.0680 = 0.8816 ✓ matches summary
  Gemma: 0.7108 / 0.0755 = 9.4143 ✓ matches summary

**Cause (c) is ruled out** — both cells in each ratio are at the same α.

---

## Section 6 — Diagnosis

| Cause | Status | Evidence |
|---|---|---|
| (a) skeptic ∆logit not −1.82 | **Ruled out**. JSON has −1.8231; paper has −1.82. Match. |
| (b) random ∆logit not −1.058 | **CONFIRMED**. JSON has random_0 = −2.068 on Qwen, −0.0755 on Gemma. Paper's −1.058 / −0.254 is a different quantity (`random` aggregate over 10 vectors × 8 coefs in `clean.json`). |
| (c) Different α for skeptic vs random_0 | **Ruled out**. Both at +200 on Qwen / +2000 on Gemma. Confirmed by `best_coefs_test.json` and the JSON's `alpha` field. |
| (d) Formula isn't `\|∆logit\|/ε` | **Ruled out**. Code at line 198 is exactly `abs(dlm) / epsilon`. |
| (e) Bug in script | **Ruled out**. Section 3's hand-computed ratios match summary to 4 decimals. |

The 0.88× / 0.82× / 0.95× numbers are correct *given the inputs the
script uses*. The mismatch with the user's expected 1.72× / 1.61× /
1.86× comes entirely from a difference in how the random control is
defined:

  fig11 random null    = single vector `random_0` at α=mean(|critical|)
                         (matches the layerwise experiment's fig9
                         random null; ε-matched to each persona it's
                         compared to)
  paper random aggregate = mean over 10 random vectors × 8 nonzero
                          coefs from the full sweep
                          (n=80 cells; not ε-matched to any specific
                          persona at the locked α)

Both are valid baselines; they answer different questions.

---

## Section 7 — "Correct numbers" table for both definitions

To make the difference concrete, here is each Qwen cell's behavioral
efficiency under both random definitions. (Note: the "paper-aggregate
ε" cannot be cleanly defined since the paper aggregate spans multiple
α; we report the ratio `|∆logit_persona| / |∆logit_random|` directly,
which is what the user's 1.72× expectation actually is.)

### Qwen

| condition | α | ‖h̄‖ | ε | ∆logit | eff = \|∆\|/ε | ratio vs **fig11 random_0** | ratio vs **paper random aggregate** |
|---|---:|---:|---:|---:|---:|---:|---:|
| caa             | −200 | 408 | 0.4898 | −1.9651 | 4.012 | 0.95× | 1.86× |
| skeptic         | +200 | 408 | 0.4898 | −1.8231 | 3.722 | **0.88×** | **1.72×** |
| devils_advocate | +200 | 408 | 0.4898 | −2.2721 | 4.638 | 1.10× | 2.15× |
| judge           | +200 | 408 | 0.4898 | −1.6989 | 3.468 | **0.82×** | **1.61×** |
| peacekeeper     | −200 | 408 | 0.4898 | −0.7091 | 1.448 | 0.34× | 0.67× |
| pacifist        | +500 | 408 | 1.2246 | −2.9786 | 2.432 | 0.58× | 1.13× |
| collaborator    | −100 | 408 | 0.2449 | −0.0294 | 0.120 | 0.03× | 0.06× |
| facilitator    | −200 | 408 | 0.4898 | −0.4686 | 0.957 | 0.23× | 0.44× |
| random_0       | +200 | 408 | 0.4898 | −2.0680 | 4.222 | 1.00× | 1.95× |
| (paper-aggregate "random", reference only) | mixed | 408 | mixed | −1.058 | (n/a, ε mixed) | — | 1.00× |

**Reading both columns**: under fig11's ε-matched random_0 baseline,
only DA exceeds the random null on Qwen, and barely. Under the
paper's aggregate random baseline, all four critical conditions and
two conformist conditions clear it. The ε-matched comparison is
stricter — that's exactly the point of the per-unit-norm framing —
because fig11's random_0 happens to *also* produce a large ∆logit at
α=200 specifically (random vectors at high coefficient magnitudes
move the logits a lot on Qwen).

### Gemma

| condition | α | ε | ∆logit | eff | ratio vs **fig11 random_0** | ratio vs **paper random aggregate** |
|---|---:|---:|---:|---:|---:|---:|
| caa             | −2000 | 0.1013 | −0.8795 | 8.685 | 11.65× | 3.46× |
| skeptic         | +2000 | 0.1013 | −0.7108 | 7.019 | **9.41×** | **2.80×** |
| devils_advocate | +2000 | 0.1013 | −0.5207 | 5.142 | 6.90× | 2.05× |
| judge           | +2000 | 0.1013 | −0.5564 | 5.495 | 7.37× | 2.19× |
| peacekeeper     | +5000 | 0.2532 | −0.7137 | 2.819 | 3.78× | 1.12× |
| pacifist        | +2000 | 0.1013 | +0.0999 | 0.987 | 1.32× | 0.39× |
| collaborator    | +500  | 0.0253 | +0.0451 | 1.782 | 2.39× | 0.71× |
| facilitator     | −5000 | 0.2532 | −0.7270 | 2.872 | 3.85× | 1.14× |
| random_0        | +2000 | 0.1013 | −0.0755 | 0.746 | 1.00× | 0.30× |
| (paper-aggregate "random", reference only) | mixed | 0.1013 | −0.254 | (n/a) | — | 1.00× |

**On Gemma the asymmetry is reversed**: fig11's random_0 produces a
very small ∆logit at α=+2000 specifically (−0.075), so the
ε-matched ratio of skeptic/random is *9.4×*, while against the
larger paper aggregate (−0.254) it's only 2.8×.

So **fig11's choice of random null is the harsher comparison on Qwen**
(makes personas look worse) and **the more lenient comparison on
Gemma** (makes personas look better). It's not a uniformly biased
choice; it's a choice that puts every comparison on a single
α-matched footing, and the consequences happen to flip directions
between models.

---

## Section 8 — Plain-language summary

`behavioral_efficiency = |∆logit| / ε` exactly, where
`ε = |α| / ‖h̄^baseline_TARGET‖`. The formula is correct; both factors
load correctly from the source data; the per-cell numbers in
`results/normalized_effects.{csv,md}` are arithmetically faithful to
those inputs.

The 0.88× number for Qwen skeptic is the ratio of skeptic's
behavioral efficiency to the **fig11-defined** random null —
specifically, **a single random unit-Gaussian vector `random_0`
evaluated at α=+200 on Qwen and α=+2000 on Gemma** (the same α the
layerwise experiment used for the random control). It is *not* the
ratio against the paper's headline `random` quantity, which averages
over 10 random vectors × 8 different coefficients (n=80 cells per
seed) and gives a much smaller magnitude on Qwen and a much larger
magnitude on Gemma at the same effective comparison.

If the analysis used the paper's `random` aggregate instead, the
Qwen skeptic ratio becomes 1.72× (matches the user's expectation)
and the Gemma skeptic ratio becomes 2.80× (also matches the user's
expectation). Neither rendering is wrong; they answer different
questions:

- *Is the persona vector different from a random direction at the
  **same** activation perturbation magnitude?* (fig11's choice)
- *Is the persona vector's behavioral effect bigger than the
  expected behavioral effect of a random direction at **any**
  coefficient sweep point?* (paper's choice)

The first is the more rigorous form of "do persona vectors do
something specific to direction" because it isolates direction from
magnitude; but it requires the random control to be evaluated at
exactly the comparison persona's α, which the paper's aggregate is
not.

The fix for the user's writeup is therefore not a code change. It is
a clarification in the README about which random null fig11 uses, and
a sentence about how the choice differs from the paper's headline
random. fig11's results aren't wrong, but readers familiar only with
the paper's `random = −1.058 / −0.254` numbers will compute the
"wrong" ratio if they try.

The actual code (`scripts/build_normalized_effects.py`) is correct
and consistent with the layerwise experiment's `random_0` choice for
fig9/fig10. No file modifications are required to fix the
discrepancy; only documentation needs to be tightened.
