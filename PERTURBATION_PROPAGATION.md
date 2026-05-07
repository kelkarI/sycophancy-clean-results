# Per-layer ΔH-cosine experiment — full summary

A reviewer-prompted follow-up to **§4.3 of the role-based-steering paper**,
which observed that persona steering vectors and the targeted CAA vector are
nearly orthogonal at the injection layer (|cos| < 0.17 on Gemma 2 27B,
|cos| ≤ 0.108 on Qwen 3 32B) and inferred from this that the two reductions
arise via **different mechanisms**. The reviewer's concern: orthogonality
of *steering vectors* at the injection layer doesn't imply *mechanistic*
independence — 24+ nonlinear blocks downstream can collapse orthogonal
inputs onto shared pathways.

This experiment captures the post-block residual at every layer
ℓ ∈ [TARGET_LAYER, num_hidden_layers−1] under nine forward passes per
prompt — baseline, CAA, three critical personas, four conformist roles, and
one random-vector control — and tracks **cos(ΔH^CAA, ΔH^persona)** and
**‖ΔH_ℓ‖** as a function of depth, on both Gemma 2 27B and Qwen 3 32B,
across the held-out 600-prompt philpapers2020 test set.

---

## TL;DR — what we observed

1. **Paper §4.3 reproduced.** At the injection layer, all (model, persona)
   cosines land within 0.005 of the analytical sign-flipped paper value on
   the 14 tier-1 cells, and within the bf16 noise floor (|observed| < 0.05)
   on the 2 tier-2 cells. All sixteen (model, persona) cells pass.

2. **The cosines do not stay near zero downstream.** Across both models,
   cos(ΔH^CAA, ΔH^persona) climbs through the residual stack and reaches
   peaks of **+0.18 to +0.40** at mid-stack, despite starting near-orthogonal
   at the injection layer.

3. **‖ΔH‖ does not collapse.** Every condition's perturbation magnitude
   *grows* 3–7× from injection to final layer on both models, so the high
   downstream cosines are not artifacts of two near-zero vectors trivially
   aligning.

4. **Random-control null breaks the two models apart.** A unit-Gaussian
   `random_0` direction at α = mean(|critical-coef|) is the geometric
   baseline for "what does an arbitrary direction look like at this
   coefficient":

   | | Gemma random_0 (argmax @ L44) | Qwen random_0 (argmax @ L47) |
   |---|---|---|
   | argmax cos | **+0.135** | **+0.343** |

   - **Gemma**: persona-CAA argmax cosines (+0.245 to +0.362) sit cleanly
     ~2–3× above the random null. Conformist roles look like a weaker
     version of critical roles, not like random. There is a persona-specific
     signal beyond the geometric baseline.
   - **Qwen**: persona-CAA argmax cosines (+0.186 to +0.402) **straddle**
     the random null. DA and skeptic clearly exceed it; peacekeeper and
     facilitator roughly match it; judge / pacifist / collaborator fall
     below it. So most of what fig9 looks like on Qwen is reproducible by
     an arbitrary direction at α=200; only DA and skeptic are clearly
     above the geometric baseline.

5. **The α-sign matrix is correctly handled across the whole 16-cell grid.**
   Eight of the sixteen cells have a negative locked persona coefficient
   that cancels α_caa<0's sign-flip, yielding a positive expected cosine
   (or vice versa). Notable example: Gemma facilitator at α=−5000 gives
   expected +0.146; observed +0.146 (abs_err 0.0003).

This is a **descriptive curve-and-sanity-check report**. Mechanistic
interpretation belongs in `role-based-steering/paper/`, not here.

---

## Headline 16-row table

`results/perturbation_propagation.{csv,md}` —
[**view on GitHub**](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/results/perturbation_propagation.md).

### Gemma 2 27B (TARGET_LAYER = 22, capture 22 → 45)

| persona | family | inj cos | argmax L | argmax cos | final cos | ‖ΔH‖ growth |
|---|---|---:|---:|---:|---:|---:|
| skeptic         | critical    | −0.064 | 44 | **+0.337** | +0.300 | 4.72× |
| devils_advocate | critical    | +0.003 | 44 | **+0.362** | +0.331 | 4.87× |
| judge           | critical    | −0.082 | 44 | **+0.245** | +0.194 | 5.01× |
| peacekeeper     | conformist  | −0.064 | 44 | **+0.274** | +0.236 | 3.36× |
| pacifist        | conformist  | −0.072 | 44 | **+0.245** | +0.184 | 4.24× |
| collaborator    | conformist  | **−0.165** | 22 | −0.165 | +0.121 | 4.26× |
| facilitator     | conformist  | +0.146 | 38 | **+0.285** | +0.179 | 3.30× |
| **random_0**    | null        | −0.006 | 44 | **+0.135** | +0.125 | 2.93× |

(`collaborator` argmax is at the injection layer itself: its initial cos
with CAA is unusually large in magnitude (-0.165) and the downstream growth
in cos doesn't surpass that absolute value.)

### Qwen 3 32B (TARGET_LAYER = 32, capture 32 → 63)

| persona | family | inj cos | argmax L | argmax cos | final cos | ‖ΔH‖ growth |
|---|---|---:|---:|---:|---:|---:|
| skeptic         | critical    | +0.104 | 49 | **+0.359** | +0.265 | 6.53× |
| devils_advocate | critical    | +0.108 | 49 | **+0.402** | +0.255 | 6.58× |
| judge           | critical    | +0.041 | 47 | **+0.261** | +0.208 | 7.13× |
| peacekeeper     | conformist  | −0.016 | 46 | **+0.356** | +0.298 | 5.06× |
| pacifist        | conformist  | −0.033 | 51 | **+0.191** | +0.135 | 6.00× |
| collaborator    | conformist  | −0.020 | 46 | **+0.186** | +0.177 | 5.76× |
| facilitator     | conformist  | −0.054 | 46 | **+0.335** | +0.282 | 5.12× |
| **random_0**    | null        | −0.007 | 47 | **+0.343** | +0.288 | 5.59× |

---

## Injection-layer sanity check (all 16 cells)

Per RUN_PROMPT INV-3: at the injection layer, ΔH = α·v exactly (modulo
bf16 storage), so per-token cosine reduces analytically to
`sign(α_caa · α_persona) · cos(v_caa, v_persona)`. The driver halts on a
>5e-3 mismatch on tier-1 cells (|expected| ≥ 0.010); for tier-2 noise-floor
cells the hard gate is `|observed| < 0.05` plus a projection check
(`⟨ΔH, v⟩.mean() ≈ α to 5% relative`) to catch cross-model contamination
and wrong-vector errors that the near-zero cosine itself cannot.

| model | persona | observed | expected | abs_err | tier | gate |
|---|---|---:|---:|---:|---|---|
| gemma-2-27b-it | skeptic         | −0.0644 | −0.0640 | 0.0004 | 1 | OK |
| gemma-2-27b-it | devils_advocate | +0.0028 | −0.0030 | 0.0058 | **2** | OK (\|obs\|<0.05; sign WARN) |
| gemma-2-27b-it | judge           | −0.0816 | −0.0854 | 0.0038 | 1 | OK |
| gemma-2-27b-it | peacekeeper     | −0.0638 | −0.0651 | 0.0013 | 1 | OK |
| gemma-2-27b-it | pacifist        | −0.0716 | −0.0757 | 0.0041 | 1 | OK |
| gemma-2-27b-it | collaborator    | −0.1651 | −0.1648 | 0.0003 | 1 | OK |
| gemma-2-27b-it | facilitator     | +0.1462 | +0.1459 | 0.0003 | 1 | OK (α<0 → sign-flip cancelled) |
| gemma-2-27b-it | random_0        | −0.0063 | −0.0075 | 0.0012 | **2** | OK (\|obs\|<0.05) |
| qwen3-32b      | skeptic         | +0.1036 | +0.1049 | 0.0013 | 1 | OK |
| qwen3-32b      | devils_advocate | +0.1075 | +0.1078 | 0.0003 | 1 | OK |
| qwen3-32b      | judge           | +0.0408 | +0.0423 | 0.0015 | 1 | OK |
| qwen3-32b      | peacekeeper     | −0.0155 | −0.0171 | 0.0016 | 1 | OK (α<0) |
| qwen3-32b      | pacifist        | −0.0328 | −0.0320 | 0.0008 | 1 | OK |
| qwen3-32b      | collaborator    | −0.0204 | −0.0216 | 0.0012 | 1 | OK (α<0) |
| qwen3-32b      | facilitator     | −0.0540 | −0.0556 | 0.0016 | 1 | OK (α<0) |
| qwen3-32b      | random_0        | −0.0066 | −0.0065 | 0.0001 | **2** | OK (\|obs\|<0.05) |

All 16 cells pass.

---

## Locations of every artifact

Branch on all three repos: **`claude/design-implementation-spec-EWNnw`**.

### Source repos (each has its own driver, byte-identical between them)

#### kelkarI/sycophancy-gemma — Gemma 2 27B (TARGET_LAYER=22, |α_caa|=2000)

| Artifact | Path | Link |
|---|---|---|
| Implementation spec | `IMPLEMENTATION_SPEC.md` | [view](https://github.com/kelkarI/sycophancy-gemma/blob/claude/design-implementation-spec-EWNnw/IMPLEMENTATION_SPEC.md) |
| Driver | `experiment-main/scripts/04_perturbation_propagation.py` | [view](https://github.com/kelkarI/sycophancy-gemma/blob/claude/design-implementation-spec-EWNnw/experiment-main/scripts/04_perturbation_propagation.py) |
| CPU unit tests | `experiment-main/scripts/test_perturbation_propagation.py` | [view](https://github.com/kelkarI/sycophancy-gemma/blob/claude/design-implementation-spec-EWNnw/experiment-main/scripts/test_perturbation_propagation.py) |
| Per-layer summary | `experiment-main/results/perturbation_propagation/per_layer_summary.json` | [view](https://github.com/kelkarI/sycophancy-gemma/blob/claude/design-implementation-spec-EWNnw/experiment-main/results/perturbation_propagation/per_layer_summary.json) |
| Run config snapshot | `experiment-main/results/perturbation_propagation/meta.json` | [view](https://github.com/kelkarI/sycophancy-gemma/blob/claude/design-implementation-spec-EWNnw/experiment-main/results/perturbation_propagation/meta.json) |
| Run log | `experiment-main/results/perturbation_propagation/run_log_gemma.txt` | [view](https://github.com/kelkarI/sycophancy-gemma/blob/claude/design-implementation-spec-EWNnw/experiment-main/results/perturbation_propagation/run_log_gemma.txt) |
| Per-repo cosine fig | `experiment-main/figures/fig_perturbation_cosine.png` | [view](https://github.com/kelkarI/sycophancy-gemma/blob/claude/design-implementation-spec-EWNnw/experiment-main/figures/fig_perturbation_cosine.png) |
| Per-repo norm fig | `experiment-main/figures/fig_perturbation_norm.png` | [view](https://github.com/kelkarI/sycophancy-gemma/blob/claude/design-implementation-spec-EWNnw/experiment-main/figures/fig_perturbation_norm.png) |

Per-prompt cosine / norm `.npz` files and the per-prompt `checkpoints/`
directory are deliberately gitignored (rebuildable, ~MB-scale).

#### kelkarI/sycophancy-qwen — Qwen 3 32B (TARGET_LAYER=32, |α_caa|=200)

| Artifact | Path | Link |
|---|---|---|
| Implementation spec | `IMPLEMENTATION_SPEC.md` | [view](https://github.com/kelkarI/sycophancy-qwen/blob/claude/design-implementation-spec-EWNnw/IMPLEMENTATION_SPEC.md) |
| Driver (byte-identical) | `scripts/04_perturbation_propagation.py` | [view](https://github.com/kelkarI/sycophancy-qwen/blob/claude/design-implementation-spec-EWNnw/scripts/04_perturbation_propagation.py) |
| CPU unit tests | `scripts/test_perturbation_propagation.py` | [view](https://github.com/kelkarI/sycophancy-qwen/blob/claude/design-implementation-spec-EWNnw/scripts/test_perturbation_propagation.py) |
| Per-layer summary | `results/perturbation_propagation/per_layer_summary.json` | [view](https://github.com/kelkarI/sycophancy-qwen/blob/claude/design-implementation-spec-EWNnw/results/perturbation_propagation/per_layer_summary.json) |
| Run config snapshot | `results/perturbation_propagation/meta.json` | [view](https://github.com/kelkarI/sycophancy-qwen/blob/claude/design-implementation-spec-EWNnw/results/perturbation_propagation/meta.json) |
| Run log | `results/perturbation_propagation/run_log_qwen.txt` | [view](https://github.com/kelkarI/sycophancy-qwen/blob/claude/design-implementation-spec-EWNnw/results/perturbation_propagation/run_log_qwen.txt) |
| Per-repo cosine fig | `figures/fig_perturbation_cosine.png` | [view](https://github.com/kelkarI/sycophancy-qwen/blob/claude/design-implementation-spec-EWNnw/figures/fig_perturbation_cosine.png) |
| Per-repo norm fig | `figures/fig_perturbation_norm.png` | [view](https://github.com/kelkarI/sycophancy-qwen/blob/claude/design-implementation-spec-EWNnw/figures/fig_perturbation_norm.png) |

### Aggregated repo (this one): kelkarI/sycophancy-clean-results

| Artifact | Path | Link |
|---|---|---|
| **This summary** | `PERTURBATION_PROPAGATION.md` | [view](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/PERTURBATION_PROPAGATION.md) |
| README ("Perturbation propagation" section) | `README.md` | [view](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/README.md#perturbation-propagation-fig910) |
| Cross-model fig9 (cosine, 8 lines per panel) | `figures/fig9_perturbation_cosine.{pdf,png}` | [view PNG](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/figures/fig9_perturbation_cosine.png) · [view PDF](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/figures/fig9_perturbation_cosine.pdf) |
| Cross-model fig10 (‖ΔH‖, log scale) | `figures/fig10_perturbation_norm.{pdf,png}` | [view PNG](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/figures/fig10_perturbation_norm.png) · [view PDF](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/figures/fig10_perturbation_norm.pdf) |
| 16-row table (markdown) | `results/perturbation_propagation.md` | [view](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/results/perturbation_propagation.md) |
| 16-row table (CSV) | `results/perturbation_propagation.csv` | [view](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/results/perturbation_propagation.csv) |
| Aggregated data (Gemma) | `data/gemma-2-27b-it_perturbation_propagation.json` | [view](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/data/gemma-2-27b-it_perturbation_propagation.json) |
| Aggregated data (Qwen) | `data/qwen3-32b_perturbation_propagation.json` | [view](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/data/qwen3-32b_perturbation_propagation.json) |
| Build script | `scripts/build_perturbation_propagation.py` | [view](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/scripts/build_perturbation_propagation.py) |
| Figure script | `scripts/make_perturbation_propagation_figs.py` | [view](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/scripts/make_perturbation_propagation_figs.py) |
| Table script | `scripts/make_perturbation_propagation_table.py` | [view](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/scripts/make_perturbation_propagation_table.py) |
| Style helper (PALETTE/LABELS) | `scripts/_style.py` | [view](https://github.com/kelkarI/sycophancy-clean-results/blob/claude/design-implementation-spec-EWNnw/scripts/_style.py) |

---

## Commit chain (all on `claude/design-implementation-spec-EWNnw`)

| # | Repo | SHA | Description |
|---|---|---|---|
| 1 | sycophancy-gemma | [`3d49df3`](https://github.com/kelkarI/sycophancy-gemma/commit/3d49df3) | Driver + tests + Gemma 5-prompt smoke (initial Phase 3) |
| 2 | sycophancy-gemma | [`2752fbb`](https://github.com/kelkarI/sycophancy-gemma/commit/2752fbb) | Gemma 600-prompt full run, 3 critical personas |
| 3 | sycophancy-qwen  | [`35e5c16`](https://github.com/kelkarI/sycophancy-qwen/commit/35e5c16)  | Qwen 600-prompt full run, byte-identical driver, 3 critical personas |
| 4 | sycophancy-clean-results | [`1be645f`](https://github.com/kelkarI/sycophancy-clean-results/commit/1be645f) | First aggregation: data/*.json, fig9, fig10, table, README ×4 surgical edits |
| 5 | sycophancy-gemma | [`263a242`](https://github.com/kelkarI/sycophancy-gemma/commit/263a242) | Gemma README "See also" |
| 6 | sycophancy-qwen  | [`7f49c65`](https://github.com/kelkarI/sycophancy-qwen/commit/7f49c65)  | Qwen README "See also" |
| 7 | sycophancy-gemma | [`b749a3e`](https://github.com/kelkarI/sycophancy-gemma/commit/b749a3e) | **Extension**: Gemma rerun with all 8 conditions (3 critical + 4 conformist + random_0) |
| 8 | sycophancy-qwen  | [`498cc25`](https://github.com/kelkarI/sycophancy-qwen/commit/498cc25)  | **Extension**: Qwen byte-identical 8-condition rerun |
| 9 | sycophancy-clean-results | [`079e011`](https://github.com/kelkarI/sycophancy-clean-results/commit/079e011) | **Extension**: 8-line fig9/fig10, 16-row table, expanded README, +facilitator in PALETTE |

(Commits 7–9 are the "extend to all conformist roles + random null" rerun
in response to the methodological question of whether downstream alignment
exceeds the random-direction baseline. The first six commits cover the
3-critical-persona version that landed Phases 0–7 of the executor's RUN_PROMPT.)

---

## Reproduction

Hardware: 1× H100 PCIe 80GB (single-GPU, bf16, sdpa attention,
`do_sample=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`).
Determinism is bit-perfect (G3.3 max diff = 0.0).

```bash
# Source repos (one-time setup)
git clone -b claude/design-implementation-spec-EWNnw \
    https://github.com/kelkarI/sycophancy-gemma.git
git clone -b claude/design-implementation-spec-EWNnw \
    https://github.com/kelkarI/sycophancy-qwen.git
git clone -b claude/design-implementation-spec-EWNnw \
    https://github.com/kelkarI/sycophancy-clean-results.git
git clone https://github.com/safety-research/assistant-axis.git ~/assistant-axis

# Vector regen (Qwen only — gitignored on that repo by license)
cd sycophancy-qwen/scripts
python fetch_external.py --skip-vectors
python extract_all_vectors.py --skip-personas      # Qwen CAA, ~5 min H100
python build_vectors_from_official.py              # CPU, persona vectors

# Run: each is ~10-11 min on a single H100
cd /path/to/sycophancy-gemma/experiment-main/scripts
EXPERIMENT_ROOT=/path/to/sycophancy-gemma/experiment-main \
ASSISTANT_AXIS_PATH=~/assistant-axis \
python 04_perturbation_propagation.py

cd /path/to/sycophancy-qwen/scripts
EXPERIMENT_ROOT=/path/to/sycophancy-qwen \
ASSISTANT_AXIS_PATH=~/assistant-axis \
python 04_perturbation_propagation.py

# Aggregate
cd /path/to/sycophancy-clean-results/scripts
python build_perturbation_propagation.py
python make_perturbation_propagation_figs.py
python make_perturbation_propagation_table.py
```

CPU-only smoke (no model load): `python test_perturbation_propagation.py`
on either source-repo's scripts dir. 15 unit tests covering the cosine
helper (U1–U4), the INV-3 sign-flip arithmetic (U5), `load_unit` (U6 ×2),
the Qwen/Gemma chat-template wrapper (U7 ×2), the EXPECTED_COS_AT_INJECTION
shape (U8), and the tier-1/tier-2 gate edge cases (4 cases including
cross-model contamination).

---

## Caveats and deviations from the original spec

These are flagged in the relevant commit messages; collected here for
reference.

1. **§6.1 gate semantics swapped** (Phase 3 commit `3d49df3`). The spec
   wrote `max|ΔH − α·v|_∞ < 5e-2` at the injection layer. On Gemma 2 27B
   layer 22, `|h|_∞ ≈ 1.6e5` at BOS-like tokens, which forces bf16 storage
   ULP ≈ 1024 even when the implementation is bit-correct. The driver
   instead gates on `⟨ΔH, v⟩.mean() ≈ α to within 5% relative` (bf16 noise
   ε is approximately orthogonal to v so `⟨ε, v⟩` averages out). The
   element-wise max-abs-err is kept as a `[DIAG]` annotation. The §6.3
   norm gate (‖ΔH^CAA‖ ≈ |α| to 0.22%) and §6.2 cosine gate independently
   confirm INV-1 / INV-2 are working.

2. **INV-3 tier-2 sign check softened to WARN**. The spec hard-gated on
   sign match. Empirically, at α=2000 on Gemma the DA cell's analytical
   cosine of −0.003 is at the bf16 noise floor, so its sign is randomized
   even with correct vectors and coefs. The driver still hard-gates on
   `|observed| < 0.05` (catches cross-model contamination — verified by
   `test_tier2_fail_on_cross_model_magnitude`); sign flip and magnitude
   drift surface as `[WARN ...]` annotations on the same line.

3. **G5.5 string-match expectation doesn't match current Qwen3 chat
   template**. The prompt's `assert '<think>' not in p` was written
   assuming an older Qwen3 tokenizer. The current tokenizer (cached at HF)
   emits an explicit empty `<think>\n\n</think>\n\n` block when
   `enable_thinking=False`. The driver's `build_prompt` is byte-identical
   to the canonical `sycophancy-qwen/scripts/02_evaluate_steering.py`
   output (which also uses `enable_thinking=False` and produces the same
   prompt); cosines compare to the §4.3 reference correctly.

4. **Qwen `vector_cosine_similarities.json` left at HEAD**. The R3 vector
   recovery (`build_vectors_from_official.py`) overwrote this file as a
   side-effect with values that drifted ~0.001 from the paper-shipped
   reference. T0.5b passes on the original; the working-tree change was
   reverted before commit. Locally-extracted Qwen vectors used by the
   experiment have ~0.001 cosine drift vs the paper-shipped values; that
   is well within the 5e-3 tier-1 tolerance.

5. **Environment fixes**: `pip install --upgrade Pillow` (PIL.Image.Resampling
   required by transformers 5.x); `pip install --upgrade jinja2` (3.1.0+
   required by `apply_chat_template`).

---

## Where the science of this lives

This document and the curves are *descriptive*. Mechanistic interpretation
of whether persona steering and CAA "really" share a downstream pathway
belongs in the parent paper repo:

  <https://github.com/kelkarI/role-based-steering>

…specifically the `paper/` subdirectory there. The clean-results repo's
job is to surface the curves and the sanity-check evidence, not to argue
for a particular reading of them.

## See also

- **Public mirror of the general experiment code**:
  <https://github.com/vkmk1/Sycophancy-Steering>
- **Cross-model paper repo**:
  <https://github.com/kelkarI/role-based-steering>
- **safety-research/assistant-axis** (steering library):
  <https://github.com/safety-research/assistant-axis>
- **lu-christina/assistant-axis-vectors** (HF dataset, source for the
  persona vectors on both models): <https://huggingface.co/datasets/lu-christina/assistant-axis-vectors>
