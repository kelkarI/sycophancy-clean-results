# RUN_PROMPT — per-layer ΔH-cosine experiment, end-to-end

This is a copy-paste prompt for a fresh Claude Code session. It executes
the experiment specified in `IMPLEMENTATION_SPEC.md` (committed in both
`sycophancy-gemma` and `sycophancy-qwen` on the
`claude/design-implementation-spec-EWNnw` branch), runs it on Gemma 2
27B and Qwen 3 32B in turn, and aggregates everything into
`sycophancy-clean-results`.

To use: paste everything between the `===BEGIN PROMPT===` and
`===END PROMPT===` markers below into a new Claude Code session.

```text
===BEGIN PROMPT===

You are executing a multi-repo follow-up experiment. The detailed
experimental spec is committed at IMPLEMENTATION_SPEC.md in both
source repos; pull it from either:

  https://github.com/kelkarI/sycophancy-gemma/raw/claude/design-implementation-spec-EWNnw/IMPLEMENTATION_SPEC.md
  https://github.com/kelkarI/sycophancy-qwen/raw/claude/design-implementation-spec-EWNnw/IMPLEMENTATION_SPEC.md

Read the full spec before writing any code. The spec is the source of
truth for the algorithm. This prompt is the source of truth for the
execution plan, code drops, and integration into sycophancy-clean-results.
If the two disagree, STOP and surface the disagreement before continuing.

────────────────────────────────────────────────────────────────────────
Repos in scope (all on branch claude/design-implementation-spec-EWNnw):
────────────────────────────────────────────────────────────────────────

  1. kelkarI/sycophancy-gemma          (run experiment, push results)
  2. kelkarI/sycophancy-qwen           (run experiment, push results)
  3. kelkarI/sycophancy-clean-results  (aggregate, build figures, README)

Read-only / link-only repo (do NOT modify, but DO link from the
clean-results README):

  vkmk1/Sycophancy-Steering — public mirror of the general experiment
  code. https://github.com/vkmk1/Sycophancy-Steering

The two model runs are independent. Different repos, different models,
different vectors, different coefficients. Do NOT mix parameters or
paths between them. The "DO NOT CONFUSE" block exists because everything
else about the two pipelines is deliberately identical, which makes it
dangerously easy to use the wrong constant.

If you ever feel uncertain whether you are operating on Gemma or Qwen,
run `python -c "from config import MODEL_NAME, TARGET_LAYER; print(MODEL_NAME, TARGET_LAYER)"`
in the working directory before writing or running anything.

════════════════════════════════════════════════════════════════════════
DO NOT CONFUSE — per-model constants
════════════════════════════════════════════════════════════════════════

                                Gemma                  Qwen
  ----------------------------- ---------------------- ----------------------
  Repo working dir              sycophancy-gemma/      sycophancy-qwen/
                                experiment-main        (root)

  config.py MODEL_NAME          google/gemma-2-27b-it  Qwen/Qwen3-32B
  config.py TARGET_LAYER        22                     32
  num_hidden_layers             46                     64
  hidden_size                   4608                   5120
  Layers to capture (inclusive) 22..45  (L = 24)       32..63  (L = 32)
  A/B token IDs                 235280 / 235305        32 / 33

  Locked CAA coef               -2000.0                -200.0
  Locked critical-persona coefs +2000.0 (all four)     +200.0 (all four)
  Random-control coef (default) 2000.0                 200.0

  Chat template flag            (default)              enable_thinking=False
                                same try/except wrapper used in build_prompt
                                so identical script runs on both

  Persona-vector build script   01_prepare_steering_   build_vectors_from_
                                vectors.py             official.py
                                (DO NOT use Qwen's 01_prepare_steering_vectors.py;
                                 it hardcodes vectors/gemma-2-27b paths — known
                                 stale from the Gemma fork)

  Eval data — IMPORTANT         Use each repo's existing data/eval_data.json
                                AS IS. Only 21 of 300 base questions overlap
                                between the two repos despite identical seed.
                                Do NOT regenerate or harmonise.

  Stale files in Qwen — IGNORE  data/setup_info.json    (says model=gemma)
                                data/target_layer.txt   (says 31; config says 32)
                                Trust scripts/config.py only.

  Expected cosines at the       Sign-flipped because locked CAA coef is
  injection layer (§6.2 of      negative while persona coefs are positive:
  the spec) — these are the
  single invariant that          Persona            Gemma     Qwen
  catches a Gemma↔Qwen           ----------------- --------- ---------
  mix-up at the earliest         skeptic           -0.0640   +0.1049
  moment:                        devils_advocate   -0.0030   +0.1078
                                 judge             -0.0854   +0.0423

  Tolerance: ±0.005 absolute. If the observed cosine has the WRONG SIGN
  on either model, you have crossed the wires — STOP and re-check which
  repo's coefficients and vectors are loaded.

════════════════════════════════════════════════════════════════════════
PHASE 1 — Gemma smoke test
════════════════════════════════════════════════════════════════════════

  cd sycophancy-gemma/experiment-main/scripts
  git checkout claude/design-implementation-spec-EWNnw    # already there

Verify pre-conditions:

  ls ../vectors/steering/{caa,skeptic,devils_advocate,judge,random_0}_unit.pt
  python -c "
  import torch
  for n in ['caa','skeptic','devils_advocate','judge','random_0']:
      v = torch.load(f'../vectors/steering/{n}_unit.pt', weights_only=False)
      assert v.shape[0] == 4608, f'{n}: got {v.shape}'
      assert abs(v.float().norm().item() - 1.0) < 1e-3, f'{n} not unit-norm'
  print('Gemma pre-conditions OK')
  "

  python -c "
  import json
  d = json.load(open('../results/best_coefs_test.json'))['best_coefs']
  assert d['caa'] == -2000.0 and d['skeptic'] == 2000.0, d
  print('Gemma coefs OK', d)
  "

Drop the new script `scripts/04_perturbation_propagation.py` exactly as
given in the CODE BLOCK A below. Do not paraphrase, edit, or "improve"
it during the smoke phase. Then run:

  python 04_perturbation_propagation.py --max-base 5 --smoke-only

Expect ALL [OK] in the printed sanity table, with the GEMMA expected
values from the table above. If any cosine has the wrong sign, or if
ΔH^CAA exactness exceeds the printed tolerance, STOP and report.

════════════════════════════════════════════════════════════════════════
PHASE 2 — Gemma full run
════════════════════════════════════════════════════════════════════════

Once smoke passes:

  mkdir -p ../results/perturbation_propagation
  python 04_perturbation_propagation.py 2>&1 | tee \
      ../results/perturbation_propagation/run_log_gemma.txt

If this dies mid-run (OOM, SSH disconnect, etc.):

  python 04_perturbation_propagation.py --resume 2>&1 | tee -a \
      ../results/perturbation_propagation/run_log_gemma.txt

When done, verify outputs exist:

  ls ../results/perturbation_propagation/{deltaH_cosine_per_prompt.npz,deltaH_norm_per_prompt.npz,per_layer_summary.json,meta.json}
  ls ../results/perturbation_propagation/checkpoints/ | wc -l   # should equal 600

Commit and push:

  git add scripts/04_perturbation_propagation.py \
          ../results/perturbation_propagation/per_layer_summary.json \
          ../results/perturbation_propagation/meta.json \
          ../results/perturbation_propagation/run_log_gemma.txt
  # NB: deltaH_*.npz and checkpoints/ are usually large; gitignore them
  # via the existing .gitignore patterns or add them explicitly here if
  # the repo .gitignore does not already exclude *.npz.
  git commit -m "Add per-layer ΔH cosine experiment (CAA vs persona) on Gemma

  Implements IMPLEMENTATION_SPEC.md §5: 600-prompt run captures the
  post-block residual at every layer ℓ ∈ [22, 45] under baseline,
  caa, skeptic, devils_advocate, judge, random_0 steering, then
  computes per-token cos(ΔH^CAA, ΔH^persona) and ‖ΔH_ℓ‖ per layer.

  All §6 sanity checks pass (injection-layer cosines match the
  paper's §4.3 numbers to ±0.005)."
  git push -u origin claude/design-implementation-spec-EWNnw

If push fails on a network error, retry with exponential backoff
(2 s, 4 s, 8 s, 16 s); do not skip hooks or force-push.

════════════════════════════════════════════════════════════════════════
PHASE 3 — Qwen smoke test
════════════════════════════════════════════════════════════════════════

  cd sycophancy-qwen/scripts
  git checkout claude/design-implementation-spec-EWNnw

Verify pre-conditions:

  ls ../vectors/steering/{caa,skeptic,devils_advocate,judge,random_0}_unit.pt

  If any of those files are MISSING, vectors/steering/ is gitignored
  in this repo. Re-extracting them is OUT OF SCOPE for this experiment.
  Stop and ask the human to populate vectors/steering/ — do not run
  build_vectors_from_official.py or extract_all_vectors.py yourself
  unless the human explicitly authorises it.

  python -c "
  import torch
  for n in ['caa','skeptic','devils_advocate','judge','random_0']:
      v = torch.load(f'../vectors/steering/{n}_unit.pt', weights_only=False)
      assert v.shape[0] == 5120, f'{n}: got {v.shape}'
      assert abs(v.float().norm().item() - 1.0) < 1e-3, f'{n} not unit-norm'
  print('Qwen pre-conditions OK')
  "

  python -c "
  import json
  d = json.load(open('../results/best_coefs_test.json'))['best_coefs']
  assert d['caa'] == -200.0 and d['skeptic'] == 200.0, d
  print('Qwen coefs OK', d)
  "

Copy the EXACT same 04_perturbation_propagation.py from the Gemma run:

  cp ../../sycophancy-gemma/experiment-main/scripts/04_perturbation_propagation.py \
     scripts/04_perturbation_propagation.py
  diff ../../sycophancy-gemma/experiment-main/scripts/04_perturbation_propagation.py \
       scripts/04_perturbation_propagation.py
  # diff must produce zero output. The script is byte-identical between
  # the two repos. Per-model variation is resolved by config.py and the
  # on-disk vectors / coefs, NOT by editing this script.

Run:

  python 04_perturbation_propagation.py --max-base 5 --smoke-only

Expect ALL [OK] in the printed sanity table, with the QWEN expected
values. Note signs are FLIPPED relative to Gemma. If you see Gemma's
signs on Qwen or vice versa, you have loaded the wrong vectors or
coefs — STOP and inspect.

════════════════════════════════════════════════════════════════════════
PHASE 4 — Qwen full run
════════════════════════════════════════════════════════════════════════

  mkdir -p ../results/perturbation_propagation
  python 04_perturbation_propagation.py 2>&1 | tee \
      ../results/perturbation_propagation/run_log_qwen.txt

  ls ../results/perturbation_propagation/{deltaH_cosine_per_prompt.npz,deltaH_norm_per_prompt.npz,per_layer_summary.json,meta.json}

Commit and push to claude/design-implementation-spec-EWNnw with
commit message "Add per-layer ΔH cosine experiment (CAA vs persona) on
Qwen" plus the same body paragraph as Gemma but with layers [32,63].

════════════════════════════════════════════════════════════════════════
PHASE 5 — Aggregate into sycophancy-clean-results
════════════════════════════════════════════════════════════════════════

CPU-only phase. No model inference. Reads from the two source repos'
results/perturbation_propagation/per_layer_summary.json files.

  cd sycophancy-clean-results
  git checkout claude/design-implementation-spec-EWNnw

Add THREE new files using the verbatim CODE BLOCKS B, C, D below.

  Path                                                       From CODE BLOCK
  ---------------------------------------------------------- ---------------
  scripts/build_perturbation_propagation.py                  B
  scripts/make_perturbation_propagation_figs.py              C
  scripts/make_perturbation_propagation_table.py             D

(Three small scripts is cleaner than one omnibus. Each has a single
responsibility — data, figures, table — and matches the existing
build_data.py / make_figures.py / make_tables.py separation.)

Run them in order:

  python3 scripts/build_perturbation_propagation.py
  python3 scripts/make_perturbation_propagation_figs.py
  python3 scripts/make_perturbation_propagation_table.py

This produces:

  data/gemma-2-27b-it_perturbation_propagation.json
  data/qwen3-32b_perturbation_propagation.json
  figures/fig9_perturbation_cosine.{pdf,png}
  figures/fig10_perturbation_norm.{pdf,png}
  results/perturbation_propagation.csv
  results/perturbation_propagation.md

Then update README.md per CODE BLOCK E (two surgical edits).

Finally, update the source repos' READMEs per CODE BLOCK F (one edit
each — append/extend a "See also" section).

Commit clean-results:

  git add data/gemma-2-27b-it_perturbation_propagation.json \
          data/qwen3-32b_perturbation_propagation.json \
          scripts/build_perturbation_propagation.py \
          scripts/make_perturbation_propagation_figs.py \
          scripts/make_perturbation_propagation_table.py \
          figures/fig9_perturbation_cosine.pdf \
          figures/fig9_perturbation_cosine.png \
          figures/fig10_perturbation_norm.pdf \
          figures/fig10_perturbation_norm.png \
          results/perturbation_propagation.csv \
          results/perturbation_propagation.md \
          README.md
  git commit -m "Add fig9/fig10 perturbation propagation: per-layer ΔH cosine and norm

  Aggregates the Phase 2 / Phase 4 outputs from sycophancy-gemma and
  sycophancy-qwen into clean-results. New cross-model figures answer the
  reviewer concern that geometric near-orthogonality of CAA vs persona
  steering vectors at the injection layer (paper §4.3) does not by itself
  imply mechanistic independence: cos(ΔH^CAA, ΔH^persona) is now reported
  at every layer downstream, with a magnitude check (‖ΔH_ℓ‖) to rule out
  the collapse-to-zero failure mode.

  Source provenance (model, repo, branch, commit, n_prompts) is recorded
  in data/{model}_perturbation_propagation.json. Sanity check: at the
  injection layer, observed cosine matches the paper's §4.3 number on
  all 6 (model, persona) cells to ±0.005."
  git push -u origin claude/design-implementation-spec-EWNnw

Then go back and commit the source-repo README edits (CODE BLOCK F)
on each source repo's branch.

════════════════════════════════════════════════════════════════════════
PHASE 6 — Source-repo README touch-ups
════════════════════════════════════════════════════════════════════════

In each source repo, append or extend a "See also" section per CODE
BLOCK F. These are small commits — one file each.

  cd sycophancy-gemma/experiment-main
  # apply the README.md edit per CODE BLOCK F
  git add README.md
  git commit -m "README: add See-also links to clean-results, role-based-steering, public mirror"
  git push origin claude/design-implementation-spec-EWNnw

  cd sycophancy-qwen
  # apply the README.md edit per CODE BLOCK F
  git add README.md
  git commit -m "README: add See-also links to clean-results, role-based-steering, public mirror"
  git push origin claude/design-implementation-spec-EWNnw

════════════════════════════════════════════════════════════════════════
PHASE 7 — Final summary back to the human
════════════════════════════════════════════════════════════════════════

In your final message, report:

  - n_prompts actually run on each model (should be 600 each unless
    --max-base or interruption-resume-trim)
  - The injection-layer sanity-check table (observed vs expected) for
    all 6 (model, persona) cells
  - The headline numerical finding from results/perturbation_propagation.md
  - Links to the 5 commits you pushed (one Gemma, one Qwen, one clean-results,
    plus two README touch-ups). Format as:
        kelkarI/sycophancy-gemma          @ <sha>
        kelkarI/sycophancy-qwen           @ <sha>
        kelkarI/sycophancy-clean-results  @ <sha>

Do NOT draw conclusions about whether persona steering and CAA "really"
share a mechanism. Curves and sanity checks only — interpretation
belongs in role-based-steering/paper/, not here.

════════════════════════════════════════════════════════════════════════
What NOT to do
════════════════════════════════════════════════════════════════════════

- Do NOT copy results/best_coefs_test.json or vectors/ between source repos.
- Do NOT regenerate data/eval_data.json. Use the committed version.
- Do NOT touch sycophancy-qwen/scripts/01_prepare_steering_vectors.py.
- Do NOT batch prompts. One prompt at a time, per spec §5.5.
- Do NOT skip the §6 sanity checks. They run automatically inside the
  driver script and halt the run on [FAIL].
- Do NOT push to main on any of the three repos; the working branch
  on all three is claude/design-implementation-spec-EWNnw.
- Do NOT modify vkmk1/Sycophancy-Steering. Link-only.
- Do NOT amend or force-push.
- Do NOT use --no-verify or skip hooks.
- Do NOT draw mechanistic conclusions in the clean-results README.

════════════════════════════════════════════════════════════════════════
CODE BLOCK A — scripts/04_perturbation_propagation.py
(byte-identical between sycophancy-gemma and sycophancy-qwen)
════════════════════════════════════════════════════════════════════════

```python
"""
Per-layer ΔH cosine and norm under CAA-steering vs persona-steering.

For each prompt in data/eval_data.json, runs K+1 = 5–6 forward passes
(baseline + caa + 3 critical personas + optional random control),
captures post-block residual at every layer ℓ in
[TARGET_LAYER, num_hidden_layers - 1], and computes:

  - per-token cos(ΔH^CAA_ℓ, ΔH^persona_ℓ) for ℓ in layers_to_capture,
    averaged over tokens within the prompt
  - ‖ΔH^cond_ℓ‖_2 for each condition, averaged over tokens

Aggregates per-prompt then across prompts (mean, sem, 95% CI) and writes:

  results/perturbation_propagation/
    deltaH_cosine_per_prompt.npz
    deltaH_norm_per_prompt.npz
    per_layer_summary.json
    meta.json
    checkpoints/prompt_*.npz

This script is byte-identical between sycophancy-gemma and
sycophancy-qwen. All per-model variation is resolved by config.py
(MODEL_NAME, TARGET_LAYER) and the on-disk vectors / coefficients.

Hook semantics (verified by scripts/pilot_debug.py:60-87 in the source
repos): ActivationSteering registers a forward hook on
model.model.layers[TARGET_LAYER] that mutates out[0]. PyTorch fires
forward hooks in registration order, so capture hooks must be
registered *inside* the with-block to read the post-steered residual at
TARGET_LAYER. For ℓ > TARGET_LAYER, the order is moot — those blocks
consume the steered residual on input regardless.

Usage:
  python 04_perturbation_propagation.py --max-base 5 --smoke-only
  python 04_perturbation_propagation.py --max-base 5      # smoke run
  python 04_perturbation_propagation.py                   # full run
  python 04_perturbation_propagation.py --resume          # resume
"""
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ROOT, ASSISTANT_AXIS_PATH, MODEL_NAME, TARGET_LAYER

sys.path.insert(0, ASSISTANT_AXIS_PATH)
from assistant_axis.steering import ActivationSteering  # noqa: E402


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
CRITICAL_PERSONAS = ["skeptic", "devils_advocate", "judge"]
RANDOM_VEC_NAME = "random_0"

# Sanity-check expected cosines at the injection layer. These are the
# §4.3 numbers from the paper (caa_decomposition.json on Gemma,
# vector_cosine_similarities.json on Qwen), sign-flipped because the
# locked CAA coefficient is negative while persona coefficients are
# positive. cosine of α·v with β·u is sign(α·β)·cos(v, u).
EXPECTED_COS_AT_INJECTION = {
    "google/gemma-2-27b-it": {
        "skeptic":          -0.0640,
        "devils_advocate":  -0.0030,
        "judge":            -0.0854,
    },
    "Qwen/Qwen3-32B": {
        "skeptic":          +0.1049,
        "devils_advocate":  +0.1078,
        "judge":            +0.0423,
    },
}
COSINE_TOL = 5e-3
DELTA_H_EXACTNESS_TOL = 5e-2  # bf16 round-off in c·v at coef=2000


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def build_prompt(tokenizer, question_text: str) -> str:
    """Chat-template wrapper. Qwen 3 needs enable_thinking=False; Gemma
    silently rejects the kwarg with TypeError, so we try/except. This is
    the same pattern as sycophancy-qwen/scripts/02_evaluate_steering.py.
    """
    chat = [{"role": "user", "content": question_text}]
    try:
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )


def load_unit(name: str) -> torch.Tensor:
    path = f"{ROOT}/vectors/steering/{name}_unit.pt"
    v = torch.load(path, map_location="cpu", weights_only=False).float()
    if abs(float(v.norm().item()) - 1.0) > 1e-3:
        raise RuntimeError(f"{name}_unit.pt has non-unit norm {float(v.norm()):.6f}")
    return v


@torch.no_grad()
def _capture_pass(model, inputs, layers_to_capture, steering_vec, coef, device):
    """Run one forward pass; return {layer_idx: [T, H] float32 cpu tensor}."""
    store: Dict[int, torch.Tensor] = {}

    def make_hook(ell: int):
        def hook(module, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            # tensor: [batch=1, seq=T, hidden=H]; cast in-hook so GPU
            # memory does not accumulate across hooks.
            store[ell] = tensor[0].detach().to(torch.float32).cpu()
            return None
        return hook

    handles = []
    use_steering = (steering_vec is not None) and (abs(float(coef)) > 1e-9)

    if use_steering:
        with ActivationSteering(
            model,
            steering_vectors=[steering_vec.to(device)],
            coefficients=[float(coef)],
            layer_indices=[TARGET_LAYER],
            intervention_type="addition",
            positions="all",
        ):
            # Register capture hooks INSIDE the context so they fire
            # AFTER ActivationSteering's hook at TARGET_LAYER.
            for ell in layers_to_capture:
                handles.append(model.model.layers[ell].register_forward_hook(make_hook(ell)))
            model(**inputs)
            for h in handles:
                h.remove()
    else:
        for ell in layers_to_capture:
            handles.append(model.model.layers[ell].register_forward_hook(make_hook(ell)))
        model(**inputs)
        for h in handles:
            h.remove()
    return store


def per_token_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: [T, H] float32 → [T] float32 cosine per token."""
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-12
    return num / den


def run_sanity_checks(model, tokenizer, eval_data, device,
                      layers_to_capture, conditions, locked_coefs):
    """Spec §6 sanity battery on the first prompt. Halts on [FAIL]."""
    print("\n" + "=" * 60)
    print(f"=== SANITY CHECKS  model={MODEL_NAME}  TARGET_LAYER={TARGET_LAYER}")
    print("=" * 60)
    failed: List[str] = []

    expected_map = EXPECTED_COS_AT_INJECTION.get(MODEL_NAME, {})
    if not expected_map:
        print(f"[WARN] No expected cosines registered for {MODEL_NAME}; "
              f"sign-check is skipped. Inspect output manually.")

    row = eval_data[0]
    prompt = build_prompt(tokenizer, row["question_text"])
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    baseline_store = _capture_pass(model, inputs, layers_to_capture, None, 0.0, device)
    cond_stores = {
        name: _capture_pass(model, inputs, layers_to_capture, vec, coef, device)
        for (name, coef, vec) in conditions
    }

    # 6.1 ΔH^CAA at injection layer == α_caa · v_caa exactly (within bf16 tol)
    caa_coef = float(locked_coefs["caa"])
    caa_vec = next(v for (n, _, v) in conditions if n == "caa")
    delta = (cond_stores["caa"][TARGET_LAYER] - baseline_store[TARGET_LAYER]).float()
    expected_delta = (caa_coef * caa_vec).float().unsqueeze(0).expand_as(delta)
    abs_err = float((delta - expected_delta).abs().max().item())
    tag = "OK" if abs_err < DELTA_H_EXACTNESS_TOL else "FAIL"
    print(f"[{tag}] inject ΔH^CAA exactness: max abs err = {abs_err:.2e}  "
          f"(tol {DELTA_H_EXACTNESS_TOL:.0e})")
    if tag == "FAIL":
        failed.append("ΔH^CAA exactness")

    # 6.2 cos(ΔH^CAA, ΔH^persona) at injection layer matches expected
    for persona in CRITICAL_PERSONAS:
        if persona not in cond_stores:
            continue
        cos_t = per_token_cosine(
            (cond_stores["caa"][TARGET_LAYER] - baseline_store[TARGET_LAYER]).float(),
            (cond_stores[persona][TARGET_LAYER] - baseline_store[TARGET_LAYER]).float(),
        )
        observed = float(cos_t.mean().item())
        expected = expected_map.get(persona)
        if expected is None:
            print(f"[--] inject cos(CAA, {persona:18s}) observed={observed:+.4f}  (no expected)")
            continue
        ok = abs(observed - expected) < COSINE_TOL
        tag = "OK" if ok else "FAIL"
        print(f"[{tag}] inject cos(CAA, {persona:18s}) observed={observed:+.4f}  "
              f"expected={expected:+.4f}  err={observed-expected:+.4f}")
        if not ok:
            failed.append(f"cos(CAA,{persona}) sign/value")

    # 6.3 ‖ΔH_ℓ‖ > 1e-3 at every layer for every condition
    min_seen = (None, None, math.inf)
    any_collapse = False
    for name in cond_stores:
        for ell in layers_to_capture:
            d = (cond_stores[name][ell] - baseline_store[ell]).float()
            n = float(d.norm(dim=-1).mean().item())
            if n < min_seen[2]:
                min_seen = (name, ell, n)
            if n < 1e-3:
                any_collapse = True
                failed.append(f"‖ΔH‖ collapse {name}@layer{ell} = {n:.2e}")
    print(f"[{'FAIL' if any_collapse else 'OK'}] ‖ΔH‖ > 1e-3 everywhere; "
          f"min {min_seen[0]}@layer{min_seen[1]} = {min_seen[2]:.3f}")

    if failed:
        msg = ["", "Sanity-check FAILURES:"]
        for f in failed:
            msg.append(f"  - {f}")
        msg += [
            "",
            "Diagnostic — did you load the wrong repo's vectors / coefs?",
            f"  MODEL_NAME       = {MODEL_NAME}",
            f"  TARGET_LAYER     = {TARGET_LAYER}",
            f"  expected cos     = {expected_map}",
            f"  observed CAA coef= {locked_coefs.get('caa')}",
        ]
        raise SystemExit("\n".join(msg))
    print("=" * 60)
    print("All sanity checks passed.")
    print("=" * 60)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-base", type=int, default=None)
    ap.add_argument("--start-base", type=int, default=0)
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke-only", action="store_true",
                    help="Run sanity checks on prompt 0, then exit.")
    args = ap.parse_args()

    out_dir = f"{ROOT}/results/perturbation_propagation"
    ckpt_dir = f"{out_dir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(f"{ROOT}/figures", exist_ok=True)

    print(f"Loading {MODEL_NAME} (bf16, device_map=auto)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = model.device
    print(f"Loaded. hidden={model.config.hidden_size}  "
          f"layers={model.config.num_hidden_layers}", flush=True)

    locked = json.load(open(f"{ROOT}/results/best_coefs_test.json"))["best_coefs"]
    conditions = [("caa", float(locked["caa"]), load_unit("caa"))]
    for r in CRITICAL_PERSONAS:
        conditions.append((r, float(locked[r]), load_unit(r)))
    if not args.no_random:
        rand_coef = float(np.mean([abs(float(locked[r])) for r in CRITICAL_PERSONAS]))
        conditions.append((RANDOM_VEC_NAME, rand_coef, load_unit(RANDOM_VEC_NAME)))

    eval_data = json.load(open(f"{ROOT}/data/eval_data.json"))
    if args.start_base or args.max_base:
        bases = sorted(set(r["base_id"] for r in eval_data))
        end = (args.start_base + args.max_base) if args.max_base else len(bases)
        keep = set(bases[args.start_base:end])
        eval_data = [r for r in eval_data if r["base_id"] in keep]
    print(f"eval_data: {len(eval_data)} rows ({len(set(r['base_id'] for r in eval_data))} bases)")

    n_layers = model.config.num_hidden_layers
    layers_to_capture = list(range(TARGET_LAYER, n_layers))
    L = len(layers_to_capture)
    cond_names = [c[0] for c in conditions]
    persona_compare_names = [n for (n, _, _) in conditions
                             if n != "caa" and n != RANDOM_VEC_NAME]

    # ---- Always run sanity battery on prompt 0 first ----
    run_sanity_checks(model, tokenizer, eval_data, device,
                      layers_to_capture, conditions, locked)

    if args.smoke_only:
        print("\nSmoke-only mode; exiting.")
        return

    # ---- Persist meta ----
    meta = {
        "model": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "num_hidden_layers": int(n_layers),
        "hidden_size": int(model.config.hidden_size),
        "layers_to_capture": layers_to_capture,
        "conditions": cond_names,
        "personas_compared_to_caa": persona_compare_names,
        "coefficients": {n: c for (n, c, _) in conditions},
        "n_eval_rows": len(eval_data),
        "n_base": len(set(r["base_id"] for r in eval_data)),
        "include_random_control": not args.no_random,
        "torch_dtype": "bfloat16",
        "intervention_type": "addition",
        "positions": "all",
        "locked_coefs_source": "results/best_coefs_test.json",
        "argv": sys.argv,
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Per-prompt loop ----
    cos_per_prompt: List[np.ndarray] = []
    norm_per_prompt: Dict[str, List[np.ndarray]] = {c: [] for c in cond_names}
    prompt_ids: List[int] = []

    t_start = time.time()
    for i, row in enumerate(eval_data):
        qid = int(row["question_id"])
        ckpt = f"{ckpt_dir}/prompt_{qid:05d}.npz"
        if args.resume and os.path.exists(ckpt):
            d = np.load(ckpt, allow_pickle=False)
            cos_per_prompt.append(d["cos"])
            for cn in cond_names:
                norm_per_prompt[cn].append(d[f"norm_{cn}"])
            prompt_ids.append(qid)
            continue

        prompt = build_prompt(tokenizer, row["question_text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        baseline_store = _capture_pass(model, inputs, layers_to_capture, None, 0.0, device)
        per_cond_store = {
            name: _capture_pass(model, inputs, layers_to_capture, vec, coef, device)
            for (name, coef, vec) in conditions
        }

        # Per-condition per-layer ΔH: [T, H] float32 cpu
        delta = {
            name: {ell: (per_cond_store[name][ell] - baseline_store[ell])
                   for ell in layers_to_capture}
            for name in cond_names
        }

        # Per-condition per-layer mean ‖ΔH‖_2 over tokens: [L]
        norm_arr = {}
        for name in cond_names:
            arr = np.zeros(L, dtype=np.float32)
            for li, ell in enumerate(layers_to_capture):
                arr[li] = float(delta[name][ell].norm(dim=-1).mean().item())
            norm_arr[name] = arr
            norm_per_prompt[name].append(arr)

        # Per-persona per-layer cosine: [n_personas_compared, L]
        cos_p = np.zeros((len(persona_compare_names), L), dtype=np.float32)
        for ri, name in enumerate(persona_compare_names):
            for li, ell in enumerate(layers_to_capture):
                c_t = per_token_cosine(delta["caa"][ell], delta[name][ell])
                cos_p[ri, li] = float(c_t.mean().item())
        cos_per_prompt.append(cos_p)
        prompt_ids.append(qid)

        np.savez(
            ckpt, cos=cos_p,
            **{f"norm_{cn}": norm_arr[cn] for cn in cond_names},
        )

        # Free transient memory before the next prompt
        del baseline_store, per_cond_store, delta, norm_arr, cos_p

        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t_start
            rate = (i + 1) / max(dt, 1e-6)
            eta = (len(eval_data) - i - 1) / max(rate, 1e-6) / 60.0
            print(f"  prompt {i+1}/{len(eval_data)}  rate={rate:.2f}/s  ETA={eta:.1f}m", flush=True)

    # ---- Aggregate ----
    cos_arr = np.stack(cos_per_prompt, axis=1)  # [n_personas, n_prompts, L]
    norm_stk = {cn: np.stack(norm_per_prompt[cn], axis=0) for cn in cond_names}

    np.savez(
        f"{out_dir}/deltaH_cosine_per_prompt.npz",
        cos=cos_arr,
        persona_names=np.array(persona_compare_names),
        layers=np.array(layers_to_capture, dtype=np.int32),
        prompt_ids=np.array(prompt_ids, dtype=np.int32),
    )
    np.savez(
        f"{out_dir}/deltaH_norm_per_prompt.npz",
        layers=np.array(layers_to_capture, dtype=np.int32),
        prompt_ids=np.array(prompt_ids, dtype=np.int32),
        **{f"norm_{cn}": norm_stk[cn] for cn in cond_names},
    )

    summary = {
        "model": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "num_hidden_layers": int(n_layers),
        "layers": layers_to_capture,
        "personas": persona_compare_names,
        "conditions_for_norm": cond_names,
        "n_prompts": int(cos_arr.shape[1]),
        "cosine": {},
        "norm": {},
    }
    n = cos_arr.shape[1]
    for ri, persona in enumerate(persona_compare_names):
        m = cos_arr[ri].mean(axis=0)
        sd = cos_arr[ri].std(axis=0, ddof=1) if n > 1 else np.zeros_like(m)
        sem = sd / max(math.sqrt(n), 1.0)
        summary["cosine"][persona] = {
            "mean": m.tolist(),
            "sem": sem.tolist(),
            "ci_lo": (m - 1.96 * sem).tolist(),
            "ci_hi": (m + 1.96 * sem).tolist(),
            "n_prompts": int(n),
        }
    for cn in cond_names:
        nn = norm_stk[cn].shape[0]
        m = norm_stk[cn].mean(axis=0)
        sd = norm_stk[cn].std(axis=0, ddof=1) if nn > 1 else np.zeros_like(m)
        sem = sd / max(math.sqrt(nn), 1.0)
        summary["norm"][cn] = {
            "mean": m.tolist(),
            "sem": sem.tolist(),
            "ci_lo": (m - 1.96 * sem).tolist(),
            "ci_hi": (m + 1.96 * sem).tolist(),
            "n_prompts": int(nn),
        }
    with open(f"{out_dir}/per_layer_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. n_prompts={cos_arr.shape[1]}  outputs in {out_dir}/")


if __name__ == "__main__":
    main()
```

════════════════════════════════════════════════════════════════════════
CODE BLOCK B — sycophancy-clean-results/scripts/build_perturbation_propagation.py
════════════════════════════════════════════════════════════════════════

```python
"""Aggregate per-layer perturbation-propagation results from both source repos.

Reads:
  ../sycophancy-gemma/experiment-main/results/perturbation_propagation/per_layer_summary.json
  ../sycophancy-gemma/experiment-main/results/perturbation_propagation/deltaH_cosine_per_prompt.npz
  ../sycophancy-qwen/results/perturbation_propagation/per_layer_summary.json
  ../sycophancy-qwen/results/perturbation_propagation/deltaH_cosine_per_prompt.npz

Writes:
  data/gemma-2-27b-it_perturbation_propagation.json
  data/qwen3-32b_perturbation_propagation.json

Each output records full provenance (source repo + commit SHA + n_prompts)
and re-derives the injection-layer sanity-check from the per-prompt npz so
we can independently confirm the §4.3 numbers from the paper.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    {
        "model": "google/gemma-2-27b-it",
        "slug":  "gemma-2-27b-it",
        "src":   ROOT.parent / "sycophancy-gemma" / "experiment-main",
        "gh":    "kelkarI/sycophancy-gemma",
        "expected_cos": {"skeptic": -0.0640, "devils_advocate": -0.0030, "judge": -0.0854},
    },
    {
        "model": "Qwen/Qwen3-32B",
        "slug":  "qwen3-32b",
        "src":   ROOT.parent / "sycophancy-qwen",
        "gh":    "kelkarI/sycophancy-qwen",
        "expected_cos": {"skeptic": +0.1049, "devils_advocate": +0.1078, "judge": +0.0423},
    },
]


def git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo)
        ).decode().strip()
    except Exception as e:
        return f"<unavailable: {e}>"


def main():
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    for s in SOURCES:
        prop_dir = s["src"] / "results" / "perturbation_propagation"
        sj = prop_dir / "per_layer_summary.json"
        cj = prop_dir / "deltaH_cosine_per_prompt.npz"
        if not sj.exists() or not cj.exists():
            print(f"SKIP {s['slug']}: required outputs missing under {prop_dir}")
            continue
        summary = json.load(open(sj))
        cos_npz = np.load(cj, allow_pickle=False)

        layers = cos_npz["layers"].tolist()
        injection_idx = layers.index(summary["target_layer"])
        persona_names = [p.decode() if isinstance(p, bytes) else p
                         for p in cos_npz["persona_names"].tolist()]
        cos = cos_npz["cos"]   # [n_personas, n_prompts, L]

        sanity = {}
        for ri, p in enumerate(persona_names):
            observed = float(cos[ri, :, injection_idx].mean())
            expected = s["expected_cos"].get(p)
            sanity[p] = {
                "observed": observed,
                "expected": expected,
                "abs_err":  (None if expected is None else float(abs(observed - expected))),
            }

        out = {
            "model": s["model"],
            "target_layer": summary["target_layer"],
            "num_hidden_layers": summary["num_hidden_layers"],
            "source_repo":   s["gh"],
            "source_branch": "claude/design-implementation-spec-EWNnw",
            "source_commit": git_head(s["src"]),
            "n_prompts":     summary["n_prompts"],
            "personas_compared_to_caa": summary["personas"],
            "conditions":    summary["conditions_for_norm"],
            "layers":        summary["layers"],
            "cosine":        summary["cosine"],
            "norm":          summary["norm"],
            "injection_layer_sanity_check": sanity,
        }
        out_path = out_dir / f"{s['slug']}_perturbation_propagation.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"WROTE {out_path}  "
              f"(n_prompts={out['n_prompts']}, "
              f"layers {out['layers'][0]}..{out['layers'][-1]})")


if __name__ == "__main__":
    main()
```

════════════════════════════════════════════════════════════════════════
CODE BLOCK C — sycophancy-clean-results/scripts/make_perturbation_propagation_figs.py
════════════════════════════════════════════════════════════════════════

```python
"""Build fig9 (per-layer cosine) and fig10 (per-layer norm) from
data/{gemma-2-27b-it,qwen3-32b}_perturbation_propagation.json.

Two-panel side-by-side layout per figure (Gemma left, Qwen right). Reuses
PALETTE/LABELS/save() from _style.py to match the rest of the repo.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import PALETTE, LABELS, save  # noqa: F401  (set_style applied on import)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("Gemma 2 27B", "gemma-2-27b-it"),
    ("Qwen 3 32B",  "qwen3-32b"),
]

# Random control is named random_0 in the source npz; map to the existing
# "random" palette/label entry so the figure legend reads naturally.
def _color(name): return PALETTE.get(name, PALETTE.get(name.split("_")[0], "#444444"))
def _label(name):
    if name == "random_0":
        return LABELS.get("random", "Random")
    return LABELS.get(name, name)


def _load(slug):
    path = DATA / f"{slug}_perturbation_propagation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def make_cosine():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (pretty, slug) in zip(axes, MODELS):
        d = _load(slug)
        if d is None:
            ax.set_title(f"{pretty}\n(missing)")
            continue
        layers = np.array(d["layers"])
        for persona in d["personas_compared_to_caa"]:
            stats = d["cosine"][persona]
            m = np.array(stats["mean"])
            lo = np.array(stats["ci_lo"]); hi = np.array(stats["ci_hi"])
            ax.plot(layers, m, color=_color(persona), label=_label(persona), lw=1.5)
            ax.fill_between(layers, lo, hi, color=_color(persona), alpha=0.15, lw=0)
        ax.axhline(0.0, color="k", lw=0.5, ls=":")
        ax.axvline(d["target_layer"], color="k", lw=0.5, ls="--")
        ax.set_xlabel("Layer ℓ (post-block residual)")
        ax.set_title(f"{pretty}  (TARGET={d['target_layer']}, "
                     f"n={d['n_prompts']} prompts)")
    axes[0].set_ylabel(r"mean $\cos(\Delta H^{\mathrm{CAA}}_\ell, "
                       r"\Delta H^{\mathrm{persona}}_\ell)$")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle("Per-layer perturbation cosine: CAA vs persona steering",
                 y=1.02)
    fig.tight_layout()
    save(fig, str(FIG / "fig9_perturbation_cosine"))
    plt.close(fig)
    print("WROTE figures/fig9_perturbation_cosine.{pdf,png}")


def make_norm():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (pretty, slug) in zip(axes, MODELS):
        d = _load(slug)
        if d is None:
            ax.set_title(f"{pretty}\n(missing)")
            continue
        layers = np.array(d["layers"])
        for cn in d["conditions"]:
            stats = d["norm"][cn]
            m = np.array(stats["mean"])
            lo = np.array(stats["ci_lo"]); hi = np.array(stats["ci_hi"])
            ax.plot(layers, m, color=_color(cn), label=_label(cn), lw=1.5)
            ax.fill_between(layers, lo, hi, color=_color(cn), alpha=0.15, lw=0)
        ax.axvline(d["target_layer"], color="k", lw=0.5, ls="--")
        ax.set_xlabel("Layer ℓ (post-block residual)")
        ax.set_ylabel(r"mean $\|\Delta H_\ell\|_2$")
        ax.set_title(f"{pretty}  (n={d['n_prompts']} prompts)")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle("Per-layer perturbation magnitude (collapse check)",
                 y=1.02)
    fig.tight_layout()
    save(fig, str(FIG / "fig10_perturbation_norm"))
    plt.close(fig)
    print("WROTE figures/fig10_perturbation_norm.{pdf,png}")


def main():
    make_cosine()
    make_norm()


if __name__ == "__main__":
    main()
```

════════════════════════════════════════════════════════════════════════
CODE BLOCK D — sycophancy-clean-results/scripts/make_perturbation_propagation_table.py
════════════════════════════════════════════════════════════════════════

```python
"""Build results/perturbation_propagation.{csv,md} — one row per
(model, persona) pair with injection-layer / midpoint / final-layer
cosines and norm-decay diagnostics."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("Gemma 2 27B", "gemma-2-27b-it"),
    ("Qwen 3 32B",  "qwen3-32b"),
]

COLUMNS = [
    "model", "persona",
    "n_prompts",
    "inject_layer", "inject_cos_observed", "inject_cos_expected", "inject_cos_abs_err",
    "midpoint_layer", "midpoint_cos_mean", "midpoint_cos_ci_lo", "midpoint_cos_ci_hi",
    "final_layer", "final_cos_mean", "final_cos_ci_lo", "final_cos_ci_hi",
    "argmax_layer", "argmax_cos_mean",
    "norm_persona_inject", "norm_persona_final", "norm_persona_decay",
    "norm_caa_inject",     "norm_caa_final",     "norm_caa_decay",
]


def _row(model_pretty, slug):
    p = DATA / f"{slug}_perturbation_propagation.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    layers = d["layers"]
    inject = d["target_layer"]
    final = layers[-1]
    midpoint = layers[(len(layers) - 1) // 2]
    inj_idx, mid_idx, fin_idx = layers.index(inject), layers.index(midpoint), layers.index(final)
    san = d["injection_layer_sanity_check"]
    rows = []
    for persona in d["personas_compared_to_caa"]:
        cos_stats = d["cosine"][persona]
        means = cos_stats["mean"]
        argmax_i = max(range(len(means)), key=lambda i: means[i])
        rows.append({
            "model": model_pretty,
            "persona": persona,
            "n_prompts": d["n_prompts"],
            "inject_layer": inject,
            "inject_cos_observed": round(san[persona]["observed"], 4),
            "inject_cos_expected": (None if san[persona]["expected"] is None
                                    else round(san[persona]["expected"], 4)),
            "inject_cos_abs_err":  (None if san[persona]["abs_err"] is None
                                    else round(san[persona]["abs_err"], 4)),
            "midpoint_layer":      midpoint,
            "midpoint_cos_mean":   round(means[mid_idx], 4),
            "midpoint_cos_ci_lo":  round(cos_stats["ci_lo"][mid_idx], 4),
            "midpoint_cos_ci_hi":  round(cos_stats["ci_hi"][mid_idx], 4),
            "final_layer":         final,
            "final_cos_mean":      round(means[fin_idx], 4),
            "final_cos_ci_lo":     round(cos_stats["ci_lo"][fin_idx], 4),
            "final_cos_ci_hi":     round(cos_stats["ci_hi"][fin_idx], 4),
            "argmax_layer":        layers[argmax_i],
            "argmax_cos_mean":     round(means[argmax_i], 4),
            "norm_persona_inject": round(d["norm"][persona]["mean"][inj_idx], 3),
            "norm_persona_final":  round(d["norm"][persona]["mean"][fin_idx], 3),
            "norm_persona_decay":  round(d["norm"][persona]["mean"][fin_idx]
                                         / max(d["norm"][persona]["mean"][inj_idx], 1e-9), 3),
            "norm_caa_inject":     round(d["norm"]["caa"]["mean"][inj_idx], 3),
            "norm_caa_final":      round(d["norm"]["caa"]["mean"][fin_idx], 3),
            "norm_caa_decay":      round(d["norm"]["caa"]["mean"][fin_idx]
                                         / max(d["norm"]["caa"]["mean"][inj_idx], 1e-9), 3),
        })
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(rows, path):
    lines = ["# Perturbation propagation — summary table",
             "",
             "Per (model, persona) row, the cosine of ΔH^CAA with ΔH^persona "
             "is reported at the injection layer, at the midpoint of the "
             "captured stack, and at the final layer. `argmax_layer` is "
             "where the mean cosine is maximised across captured layers.",
             "",
             "Norm decay = ‖ΔH_final‖ / ‖ΔH_inject‖. Values << 1 signal "
             "that the perturbation has shrunk significantly downstream "
             "and that the cosine reading at the final layer is being "
             "computed on small vectors (interpret with care). Values "
             "near 1 mean the perturbation has propagated with stable "
             "magnitude.",
             "",
             "| Model | Persona | n | inject | inject cos (obs / exp) | midpoint cos (95% CI) | final cos (95% CI) | argmax (layer / value) | persona norm decay | CAA norm decay |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        exp = "" if r["inject_cos_expected"] is None else f" / {r['inject_cos_expected']:+.4f}"
        lines.append(
            f"| {r['model']} | {r['persona']} | {r['n_prompts']} "
            f"| L{r['inject_layer']} "
            f"| {r['inject_cos_observed']:+.4f}{exp} "
            f"| {r['midpoint_cos_mean']:+.4f} [{r['midpoint_cos_ci_lo']:+.4f}, {r['midpoint_cos_ci_hi']:+.4f}] "
            f"| {r['final_cos_mean']:+.4f} [{r['final_cos_ci_lo']:+.4f}, {r['final_cos_ci_hi']:+.4f}] "
            f"| L{r['argmax_layer']} / {r['argmax_cos_mean']:+.4f} "
            f"| {r['norm_persona_decay']:.3f} "
            f"| {r['norm_caa_decay']:.3f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    all_rows = []
    for pretty, slug in MODELS:
        all_rows.extend(_row(pretty, slug))
    write_csv(all_rows, RES / "perturbation_propagation.csv")
    write_md(all_rows, RES / "perturbation_propagation.md")
    print(f"WROTE {RES / 'perturbation_propagation.csv'}")
    print(f"WROTE {RES / 'perturbation_propagation.md'}")


if __name__ == "__main__":
    main()
```

════════════════════════════════════════════════════════════════════════
CODE BLOCK E — sycophancy-clean-results/README.md edits
════════════════════════════════════════════════════════════════════════

EDIT 1: Update the "Source repositories" list near the top of the
README (currently around line 17–25). Use the Edit tool with:

old_string:
    Source repositories (read only, not modified):

    - Gemma pipeline: `../sycophancy-gemma/experiment-main/`
      (multi-seed aggregate, per-seed JSONs, tune-locked best coefficients,
      steering-vector cosine matrix).
    - Qwen pipeline:  `../sycophancy-qwen/`
      (same structure; target layer 32, coefficient grid rescaled 10×
      because of Qwen's smaller activation norms — see the parent paper
      repo for details).

new_string:
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

EDIT 2: Insert a brand-new section "Perturbation propagation (fig9–10)"
between the existing "Family averaging (fig7)" section and the
"Qualitative samples" section. Use the Edit tool with:

old_string:
    The code path is `scripts/make_steering_curves.py:_family_series` (mean
    and min/max) and `_plot_family` (rendering).

new_string:
    The code path is `scripts/make_steering_curves.py:_family_series` (mean
    and min/max) and `_plot_family` (rendering).

    ## Perturbation propagation (fig9–10)

    A reviewer raised the concern that geometric near-orthogonality of the
    persona and CAA steering *vectors* at the injection layer (paper §4.3,
    all |cos| < 0.17 on Gemma; |cos| ≤ 0.108 on Qwen) does **not** by
    itself imply mechanistic independence — 24+ nonlinear blocks
    downstream can collapse orthogonal inputs onto shared pathways. fig9
    addresses this directly: for each prompt in the held-out test set, we
    capture the post-block residual at every layer ℓ ≥ TARGET_LAYER under
    baseline, CAA-steered, and three persona-steered (skeptic,
    devils_advocate, judge) forward passes, take ΔH = H_steer − H_base,
    and plot the per-layer mean cosine between ΔH^CAA and ΔH^persona,
    averaged within prompt over tokens then across prompts. fig10 plots
    ‖ΔH_ℓ‖_2 per layer per condition as a magnitude check — high cosine
    is uninformative if both perturbations have decayed to ~0.

    **Sanity check.** At the injection layer ℓ = TARGET_LAYER, ΔH^CAA is
    by construction equal to α_CAA · v_CAA (broadcast across tokens), so
    the per-token cosine reduces to sign(α_CAA · α_persona) · cos(v_CAA,
    v_persona) — the §4.3 vector cosine, with sign flipped because the
    locked CAA coefficient is negative while persona coefficients are
    positive. The driver script halts on a >0.005 absolute mismatch with
    the paper-reported number, on either model. The two source-repo runs
    pass this check on all 6 (model, persona) cells (see
    `data/{model}_perturbation_propagation.json["injection_layer_sanity_check"]`
    and `results/perturbation_propagation.md` for the per-cell numbers).

    **What fig9 shows (numerical headlines, populated from the data
    JSONs).** See `results/perturbation_propagation.md` for the full
    (model × persona) table. The headline values are the cosines at the
    injection layer (matches paper §4.3 to ±0.005 by construction), at
    the midpoint of the captured stack, and at the final layer of the
    decoder — read those off the `cosine[persona].mean[i]` arrays in the
    JSON or directly from `results/perturbation_propagation.{csv,md}`.

    **What fig10 shows.** ‖ΔH_ℓ‖ per layer per condition. At the
    injection layer the norm equals the absolute coefficient (~2000 on
    Gemma, ~200 on Qwen) by construction. Downstream layers can compress
    or amplify the perturbation freely; the relevant invariant for
    interpreting fig9 is that ‖ΔH_ℓ‖ does not collapse to zero. The
    `results/perturbation_propagation.md` table reports the
    `‖ΔH_final‖ / ‖ΔH_inject‖` ratio per persona; values near 1 mean
    fig9's downstream reading is on stably-sized perturbations, values
    << 1 mean the cosine reading is being taken on small vectors and
    should be interpreted accordingly.

    This is a curve-and-sanity-check report. Mechanistic conclusions
    belong in the parent paper repo
    (`../role-based-steering/paper/`), not here.

    **Provenance.** fig9 / fig10 / `data/{model}_perturbation_propagation.json`
    are aggregated from the per-prompt outputs in
    `../sycophancy-gemma/experiment-main/results/perturbation_propagation/`
    and `../sycophancy-qwen/results/perturbation_propagation/`, with the
    source-repo commit SHA recorded inside each data JSON's
    `source_commit` field.

    **Reproduction.**
    ```bash
    python3 scripts/build_perturbation_propagation.py
    python3 scripts/make_perturbation_propagation_figs.py
    python3 scripts/make_perturbation_propagation_table.py
    ```

EDIT 3: Update the Directory layout tree to add the new files. Find
the section starting "├── figures/" and extend its block, then find
"├── results/" and extend its block, then find "├── scripts/" and
extend its block, and add the new data files under "├── data/". Use
the Edit tool surgically — match the smallest unique snippet around
the insertion point.

  Add to data/:
    │   ├── gemma-2-27b-it_perturbation_propagation.json   per-layer cosine + norm summary
    │   └── qwen3-32b_perturbation_propagation.json        same, Qwen

  Add to scripts/:
    │   ├── build_perturbation_propagation.py     aggregates per-layer summaries from source repos
    │   ├── make_perturbation_propagation_figs.py builds fig9 + fig10
    │   └── make_perturbation_propagation_table.py builds results/perturbation_propagation.{csv,md}

  Add to figures/:
    │   ├── fig9_perturbation_cosine.{pdf,png}    per-layer mean cos(ΔH^CAA, ΔH^persona)
    │   └── fig10_perturbation_norm.{pdf,png}     per-layer mean ‖ΔH_ℓ‖, collapse check

  Add to results/:
    │   ├── perturbation_propagation.csv          (model × persona) cosine + norm-decay table
    │   └── perturbation_propagation.md           same, markdown

EDIT 4: Update the "How to reproduce" section at the bottom — append
the three new commands to the existing block.

old_string:
    python3 scripts/make_steering_curves.py  # rebuilds figures/fig6_steering_curves
    ```

    All three scripts are CPU-only and deterministic.

new_string:
    python3 scripts/make_steering_curves.py  # rebuilds figures/fig6_steering_curves
    python3 scripts/build_perturbation_propagation.py    # rebuilds data/*_perturbation_propagation.json from source repos
    python3 scripts/make_perturbation_propagation_figs.py # rebuilds figures/fig9, fig10
    python3 scripts/make_perturbation_propagation_table.py # rebuilds results/perturbation_propagation.{csv,md}
    ```

    All scripts are CPU-only and deterministic.

════════════════════════════════════════════════════════════════════════
CODE BLOCK F — source-repo README "See also" sections
════════════════════════════════════════════════════════════════════════

For BOTH `sycophancy-gemma/experiment-main/README.md` and
`sycophancy-qwen/README.md`: append the following section at the END
of the file, after the last existing line. If a "See also" section
already exists, add the missing bullets without duplicating any.

```markdown

## See also

- **Public experiment code (general pipeline)**:
  <https://github.com/vkmk1/Sycophancy-Steering>
- **Aggregated clean results (cross-model)**:
  <https://github.com/kelkarI/sycophancy-clean-results>
- **Cross-model paper repo**:
  <https://github.com/kelkarI/role-based-steering>
- **Per-layer perturbation-propagation experiment** (this branch's
  follow-up to the paper's §4.3 geometric-orthogonality claim):
  see `IMPLEMENTATION_SPEC.md` in this repo.
```

════════════════════════════════════════════════════════════════════════
Final deliverables checklist
════════════════════════════════════════════════════════════════════════

Per repo, on branch claude/design-implementation-spec-EWNnw, pushed:

  sycophancy-gemma:
    scripts/04_perturbation_propagation.py
    results/perturbation_propagation/
      ├── per_layer_summary.json
      ├── meta.json
      ├── run_log_gemma.txt
      ├── deltaH_cosine_per_prompt.npz   (gitignored; not pushed)
      ├── deltaH_norm_per_prompt.npz     (gitignored; not pushed)
      └── checkpoints/                   (gitignored; not pushed)
    README.md                            (See also section appended)

  sycophancy-qwen:
    scripts/04_perturbation_propagation.py   (byte-identical to Gemma's)
    results/perturbation_propagation/        (same set; run_log_qwen.txt)
    README.md                                (See also section appended)

  sycophancy-clean-results:
    data/gemma-2-27b-it_perturbation_propagation.json
    data/qwen3-32b_perturbation_propagation.json
    scripts/build_perturbation_propagation.py
    scripts/make_perturbation_propagation_figs.py
    scripts/make_perturbation_propagation_table.py
    figures/fig9_perturbation_cosine.{pdf,png}
    figures/fig10_perturbation_norm.{pdf,png}
    results/perturbation_propagation.{csv,md}
    README.md                                 (4 surgical edits per BLOCK E)

If anything in IMPLEMENTATION_SPEC.md contradicts what you observe in
the source files, trust the source files and report the contradiction
in the relevant commit message — do not silently work around it.

If you have any uncertainty about whether to proceed at any phase
boundary, surface the question to the human BEFORE running the next
phase. The cost of pausing is low; the cost of running the wrong
experiment for 10 GPU-hours is high.

===END PROMPT===
```

---

## Notes on this prompt (for your reference, not for the executor)

A few decisions worth flagging in case you want to redirect:

- **`.npz` and `checkpoints/` are gitignored, not committed.** The
  per-prompt `.npz` files are large (≈ 100 MB total per model at 600
  prompts) and the aggregated `per_layer_summary.json` carries the
  full signal needed for clean-results. If you'd rather commit the
  `.npz` files too (e.g. via Git LFS), say so and I'll amend the
  commit-set list and the source-repo `.gitignore` patterns.

- **`vkmk1/Sycophancy-Steering` link**: placed in (a) the
  clean-results "Source repositories" list at the top of the README,
  and (b) a new "See also" section at the bottom of both source-repo
  READMEs. Tell me if you also want it in `role-based-steering/`.

- **No mechanistic interpretation in clean-results.** I have the
  executor write only curves + sanity-check confirmations. The
  paragraph that draws conclusions ("does this support or undermine
  the §4.3 claim?") belongs in `role-based-steering/paper/RESULTS.md`,
  which the executor does not touch. Easy to relax — I can add a
  cautious one-paragraph interpretation in the clean-results README
  if you'd like.

- **Three small build scripts** (data, figures, table) instead of one
  omnibus, mirroring the existing `build_data.py` /
  `make_figures.py` / `make_tables.py` separation.

Want me to commit and push `RUN_PROMPT.md` to `sycophancy-clean-results`
on `claude/design-implementation-spec-EWNnw` so you have a permanent
GitHub-hosted copy?
