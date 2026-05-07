# RUN_PROMPT — per-layer ΔH-cosine experiment, end-to-end

This is a copy-paste prompt for a fresh Claude Code session. It executes
the experiment specified in `IMPLEMENTATION_SPEC.md` (committed in both
source repos on branch `claude/design-implementation-spec-EWNnw`), runs
it on Gemma 2 27B and Qwen 3 32B in turn, and aggregates everything
into `sycophancy-clean-results`.

The executor writes all code itself. This prompt pins only the
**non-obvious invariants** that, if got wrong, produce plausible-looking
but silently wrong output — and a battery of **gate tests** the executor
must run between phases to verify its implementation stays faithful to
the spec.

To use: paste everything between the `===BEGIN PROMPT===` and
`===END PROMPT===` markers below into a new Claude Code session.

```text
===BEGIN PROMPT===

You are executing a multi-repo follow-up experiment. The detailed
experimental spec is committed at IMPLEMENTATION_SPEC.md in both
source repos; pull it from either:

  https://github.com/kelkarI/sycophancy-gemma/raw/claude/design-implementation-spec-EWNnw/IMPLEMENTATION_SPEC.md
  https://github.com/kelkarI/sycophancy-qwen/raw/claude/design-implementation-spec-EWNnw/IMPLEMENTATION_SPEC.md

Read the full spec FIRST. The spec is the source of truth for the
algorithm — what to capture, at which layers, with which steering
context, and the §6 sanity tests. This prompt is the source of truth
for the execution plan, the non-obvious invariants you must respect,
and the gate tests between phases. You write all the implementation
code yourself, faithful to the spec. If the two disagree, STOP and
surface the disagreement before continuing.

────────────────────────────────────────────────────────────────────────
Repos in scope (all on branch claude/design-implementation-spec-EWNnw):
────────────────────────────────────────────────────────────────────────

  1. kelkarI/sycophancy-gemma          (run experiment, push results)
  2. kelkarI/sycophancy-qwen           (run experiment, push results)
  3. kelkarI/sycophancy-clean-results  (aggregate, build figures, README)

Read-only / link-only repo (do NOT modify, but DO link from
clean-results README):

  vkmk1/Sycophancy-Steering — public mirror of the general experiment
  code. https://github.com/vkmk1/Sycophancy-Steering

The two model runs are independent. Different repos, different models,
different vectors, different coefficients. Do NOT mix parameters or
paths between them.

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
                                use a try/except TypeError wrapper so the
                                same code runs on both

  Persona-vector build script   01_prepare_steering_   build_vectors_from_
                                vectors.py             official.py
                                (DO NOT use Qwen's 01_prepare_steering_vectors.py;
                                 it hardcodes vectors/gemma-2-27b paths — known
                                 stale from the Gemma fork)

  Eval data — IMPORTANT         Use each repo's existing data/eval_data.json
                                AS IS. Only 21 of 300 base questions overlap
                                between the two repos despite identical seed.
                                Do NOT regenerate or harmonise.

  Stale files in Qwen — IGNORE  data/setup_info.json   (says model=gemma)
                                data/target_layer.txt  (says 31; config says 32)
                                Trust scripts/config.py only.

════════════════════════════════════════════════════════════════════════
FOUR NON-OBVIOUS INVARIANTS (re-deriving these from the spec is
expensive and easy to get wrong; pin them in your implementation):
════════════════════════════════════════════════════════════════════════

  INV-1  HOOK ORDER. ActivationSteering registers a forward hook on
         model.model.layers[TARGET_LAYER] that mutates out[0] to
         h + coef·v. PyTorch fires forward hooks in registration order.
         To capture the POST-STEERED residual at TARGET_LAYER, your
         capture hooks must be registered AFTER entering the
         `with ActivationSteering(...)` context. Hooks registered
         before the context fire BEFORE steering and capture
         pre-steered residuals. For ℓ > TARGET_LAYER the order is
         moot — those blocks consume the steered residual on input
         regardless.

         Verification reference: scripts/pilot_debug.py:60-87 in the
         Gemma source repo, which shows ‖h_pre − h_base‖ ≈ 0 vs
         ‖h_post − h_base‖ ≈ 5000 at coef=5000 with a unit vector.

  INV-2  IN-HOOK FP32 CAST. Captured tensors are bf16 on GPU. Compute
         ΔH and per-token cosine in float32, otherwise bf16 round-off
         dominates the cosine signal at small magnitudes. Do the cast
         INSIDE the hook (not after collecting all hooks) so GPU
         memory does not accumulate across the 24/32 captured
         tensors per pass:

             store[ell] = tensor[0].detach().to(torch.float32).cpu()

  INV-3  SIGN OF EXPECTED COSINE AT INJECTION LAYER (TIERED RULE).
         Per-token cosine at layer TARGET_LAYER reduces analytically
         to
              cos(ΔH^CAA, ΔH^persona)
            = cos(α_caa · v_caa,  α_persona · v_persona)
            = sign(α_caa · α_persona) · cos(v_caa, v_persona).
         Because locked CAA coef is NEGATIVE while persona coefs are
         POSITIVE, the §4.3 vector-cosine numbers from the paper
         appear sign-flipped in the observed cosines. Expected table
         (from caa_decomposition.json on Gemma and
         vector_cosine_similarities.json on Qwen):

                                  Gemma     Qwen
            skeptic              -0.0640   +0.1049
            devils_advocate      -0.0030   +0.1078
            judge                -0.0854   +0.0423

         GATE RULE — applied per (model, persona):

           if |expected| >= 0.010:
               require |observed - expected| < 0.005
           else:                                    # devils_advocate-class
               require sign(observed) == sign(expected)   # catches sign errors
                 AND  |observed| < 0.05                   # catches cross-model mix-ups
                 AND  |observed - expected| < 0.005       # WARN-only diagnostic
                                                          # (do NOT halt on this alone)

         Rationale: the §6.2 expected magnitude for Gemma DA is 0.0030
         — the same order as bf16 round-off in α·v at α=2000. A uniform
         ±0.005 magnitude check would pass on noise alone for that
         cell. The sign-based check catches the failure mode that
         actually matters (wrong vectors / wrong coefs) without
         chasing the bf16 noise floor.

         Wrong sign on either model means you have crossed the wires
         (loaded the wrong repo's vectors or coefs). STOP — do not
         self-correct.

  INV-4  CHAT-TEMPLATE WRAPPER. Qwen 3's chat template injects a
         <think>...</think> block by default; passing
         enable_thinking=False suppresses it. Gemma's template raises
         TypeError on the same kwarg. The same driver script must run
         on both repos (Phase 5 byte-identity requirement), so
         build_prompt MUST use a try/except wrapper. This is the
         canonical pattern from
         sycophancy-qwen/scripts/02_evaluate_steering.py:42-56:

             def build_prompt(tokenizer, question_text):
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

         Failure mode this closes: if the executor writes the
         no-kwarg form on Qwen, the model receives a <think>...
         </think>-augmented prompt, the residuals at every layer
         shift, and *the §6.2 sanity check still passes* (at the
         injection layer ΔH = α·v independent of the prompt). A
         wrong template is therefore SILENT for our other gates.
         INV-4 + the mandatory U7 unit test + G2.5 / G5.5 tokenizer
         behaviour gates close the hole.

════════════════════════════════════════════════════════════════════════
HARDWARE ASSUMPTIONS (the determinism rationale behind INV-3 tier 1)
════════════════════════════════════════════════════════════════════════

The INV-3 ±0.005 magnitude tolerance and the sign-based DA check both
assume the forward pass is bit-deterministic across runs. That holds
on the documented hardware:

  - 1× H100 80GB SXM (or 1× A100 80GB) per the README. Gemma 27B in
    bf16 ≈ 54 GB and Qwen 32B in bf16 ≈ 64 GB; both fit on one card,
    so device_map="auto" places the model on a single GPU. Single-GPU
    forward = no tensor-parallel all-reduce non-determinism.
  - torch_dtype=torch.bfloat16; default attention impl = sdpa
    (deterministic in current PyTorch for forward); model.eval() +
    torch.no_grad() (dropout off, no autograd).
  - No sampling (we run model.__call__, not model.generate).
  - pilot_debug.py:85 ("||pre - base|| = ... should be ~0") confirms
    two-pass bit-identity in the existing pipeline.

If running on a sharded multi-GPU setup (e.g. 2× A100 40GB), the
all-reduce reduction order is non-deterministic and the gate
tolerances may fire spuriously. In that case, before model load:

    import os, torch
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

OR force single-GPU placement:

    CUDA_VISIBLE_DEVICES=0 python 04_perturbation_propagation.py ...

Phase 2's G2.0 precheck visualises the actual GPU layout at runtime.

════════════════════════════════════════════════════════════════════════
PHASE 0 — Pre-implementation reconnaissance and verification
════════════════════════════════════════════════════════════════════════

DO NOT WRITE ANY CODE in this phase. Verify, in BOTH source repos, that
the spec's assumptions still match the current source. Run each check
and confirm it passes before continuing.

  T0.1  IMPLEMENTATION_SPEC.md is committed on the branch in each
        source repo:

          test -f sycophancy-gemma/IMPLEMENTATION_SPEC.md
          test -f sycophancy-qwen/IMPLEMENTATION_SPEC.md

  T0.2  config.py constants match the DO-NOT-CONFUSE table:

          (cd sycophancy-gemma/experiment-main/scripts && \
           python -c "from config import MODEL_NAME, TARGET_LAYER, A_TOKEN_ID, B_TOKEN_ID; \
             assert MODEL_NAME == 'google/gemma-2-27b-it', MODEL_NAME; \
             assert TARGET_LAYER == 22, TARGET_LAYER; \
             assert A_TOKEN_ID == 235280 and B_TOKEN_ID == 235305; \
             print('Gemma config OK')")
          (cd sycophancy-qwen/scripts && \
           python -c "from config import MODEL_NAME, TARGET_LAYER, A_TOKEN_ID, B_TOKEN_ID; \
             assert MODEL_NAME == 'Qwen/Qwen3-32B', MODEL_NAME; \
             assert TARGET_LAYER == 32, TARGET_LAYER; \
             assert A_TOKEN_ID == 32 and B_TOKEN_ID == 33; \
             print('Qwen config OK')")

  T0.3  best_coefs_test.json has the expected locked values:

          python -c "import json; \
            d=json.load(open('sycophancy-gemma/experiment-main/results/best_coefs_test.json'))['best_coefs']; \
            assert d['caa']==-2000.0 and d['skeptic']==2000.0 and d['judge']==2000.0 and d['devils_advocate']==2000.0, d; \
            print('Gemma coefs OK')"
          python -c "import json; \
            d=json.load(open('sycophancy-qwen/results/best_coefs_test.json'))['best_coefs']; \
            assert d['caa']==-200.0 and d['skeptic']==200.0 and d['judge']==200.0 and d['devils_advocate']==200.0, d; \
            print('Qwen coefs OK')"

  T0.4  Vector files exist with the expected shape and unit norm.
        (If Qwen vectors are missing — gitignored in that repo — STOP
        and ask the human to populate vectors/steering/. Do not run
        build_vectors_from_official.py yourself unless authorised.)

          for repo in sycophancy-gemma/experiment-main sycophancy-qwen; do
            python - <<EOF
          import torch
          dim = 4608 if "gemma" in "$repo" else 5120
          for n in ["caa","skeptic","devils_advocate","judge","random_0"]:
              v = torch.load(f"$repo/vectors/steering/{n}_unit.pt", weights_only=False)
              assert v.shape[0] == dim, (n, v.shape)
              assert abs(float(v.float().norm().item()) - 1.0) < 1e-3, (n, v.float().norm())
          print("$repo vectors OK")
          EOF
          done

  T0.5  The §4.3 cosine numbers are still in caa_decomposition.json
        (Gemma) and vector_cosine_similarities.json (Qwen). These are
        what your INV-3 expected table is derived from:

          python -c "
          import json
          d = json.load(open('sycophancy-gemma/experiment-main/vectors/steering/caa_decomposition.json'))
          for r, exp in [('skeptic', 0.0640), ('devils_advocate', 0.0030), ('judge', 0.0854)]:
              got = d[r]['cosine_with_caa']
              assert abs(got - exp) < 1e-3, (r, got, exp)
          print('Gemma §4.3 cosines verified')
          "
          python -c "
          import json
          d = json.load(open('sycophancy-qwen/results/vector_cosine_similarities.json'))
          names = d['names']; mat = d['matrix']; ci = names.index('caa')
          for r, exp in [('skeptic', -0.1049), ('devils_advocate', -0.1078), ('judge', -0.0423)]:
              got = mat[names.index(r)][ci]
              assert abs(got - exp) < 1e-3, (r, got, exp)
          print('Qwen §4.3 cosines verified')
          "

  T0.6  Confirm the hook-ordering reference exists. (You will not run
        pilot_debug.py — it loads the model — but the file should
        exist as supporting evidence for INV-1.)

          test -f sycophancy-gemma/experiment-main/scripts/pilot_debug.py

  T0.7  Confirm eval_data.json schema:

          python -c "
          import json
          for p in ['sycophancy-gemma/experiment-main/data/eval_data.json',
                   'sycophancy-qwen/data/eval_data.json']:
              d = json.load(open(p))
              assert len(d) == 600, (p, len(d))
              assert set(d[0].keys()) >= {'question_id','base_id','variant','question_text','sycophantic_answer'}, d[0].keys()
          print('eval_data schemas OK')
          "

If any of T0.1–T0.7 fail, STOP and report which check failed. Do not
attempt to repair the source repos.

════════════════════════════════════════════════════════════════════════
PHASE 1 — Implement the driver, with unit tests, NO GPU
════════════════════════════════════════════════════════════════════════

Write `scripts/04_perturbation_propagation.py` in
`sycophancy-gemma/experiment-main/scripts/`. The script must satisfy
the spec §5.3 algorithm, respect INV-1 / INV-2 / INV-3, and run a §6
sanity battery on prompt 0 of every invocation (smoke or full) that
halts with a clear diagnostic on any failure.

Write `scripts/test_perturbation_propagation.py` in the same directory
with the unit tests described below. These run on CPU only — no model
loading. The intent is to catch implementation bugs before the smoke
test burns GPU time.

UNIT TESTS the executor must include and run (`pytest scripts/test_perturbation_propagation.py`,
or `python scripts/test_perturbation_propagation.py` if pytest is unavailable):

  U1  per_token_cosine on identical vectors returns 1.0.
        a = torch.randn(5, 8); b = a.clone()
        assert torch.allclose(per_token_cosine(a, b), torch.ones(5), atol=1e-6)

  U2  per_token_cosine on negated vectors returns -1.0.
        a = torch.randn(5, 8); b = -a
        assert torch.allclose(per_token_cosine(a, b), -torch.ones(5), atol=1e-6)

  U3  per_token_cosine on orthogonal pairs returns 0.0.
        e1 = torch.tensor([[1.,0.,0.]]); e2 = torch.tensor([[0.,1.,0.]])
        assert abs(float(per_token_cosine(e1, e2).item())) < 1e-6

  U4  per_token_cosine handles zero vectors safely (eps-floor in
      denominator); should not raise.
        z = torch.zeros(3, 4); v = torch.randn(3, 4)
        _ = per_token_cosine(z, v)  # must not raise

  U5  Sign-flip arithmetic for INV-3: given a=−2000, b=+2000,
      cos(v_caa, v_skeptic)=+0.0640, the analytic injection-layer
      cosine is −0.0640.
        import math
        a, b, c = -2000.0, +2000.0, 0.0640
        assert math.copysign(1.0, a*b) * c == -0.0640

  U6  load_unit fails loudly on a non-unit-norm tensor. Mock by
      saving a temp tensor of norm 0.5 and asserting load_unit raises.

  U7  build_prompt wrapper handles BOTH branches of INV-4. Mandatory.
      Use a tiny mock tokenizer — no real tokenizer / model load:

          class _MockTok:
              def __init__(self, accepts_enable_thinking):
                  self.accepts = accepts_enable_thinking
              def apply_chat_template(self, chat, **kw):
                  if "enable_thinking" in kw and not self.accepts:
                      raise TypeError("unexpected kwarg 'enable_thinking'")
                  prefix = "THINK_OFF:" if (kw.get("enable_thinking") is False
                                            and self.accepts) else ""
                  return prefix + chat[-1]["content"]

          # Qwen-like path: kwarg consumed
          out = build_prompt(_MockTok(accepts_enable_thinking=True), "x")
          assert out.startswith("THINK_OFF:"), out

          # Gemma-like path: TypeError caught, fallback used
          out = build_prompt(_MockTok(accepts_enable_thinking=False), "x")
          assert out == "x" and "THINK_OFF" not in out, out

      This is the only test that proves the wrapper actually
      exercises both code paths. Do not skip it.

  U8  EXPECTED_COS_AT_INJECTION map keys exactly equal the two
      MODEL_NAME strings; values exactly equal the table in INV-3
      to 4 decimals.

If any unit test fails, fix the implementation. Do not weaken the test
to make it pass.

INTEGRATION GATE before moving to Phase 2:

  - All U1..U8 pass.
  - The driver script imports without error from
    `sycophancy-gemma/experiment-main/scripts/`.
  - The driver script exposes a `--smoke-only` flag that runs the §6
    sanity battery on prompt 0 of eval_data.json and exits.

════════════════════════════════════════════════════════════════════════
PHASE 2 — Gemma smoke test (1 prompt, full sanity battery)
════════════════════════════════════════════════════════════════════════

  cd sycophancy-gemma/experiment-main/scripts
  python 04_perturbation_propagation.py --smoke-only --max-base 1

EXPECTED OUTPUT (the §6 sanity battery printout):

  [OK] inject ΔH^CAA exactness: max abs err = <ε>  (tol 5e-2)
        ε is bf16 round-off; expect ~1e-3 to 5e-3 in practice.
  [OK] inject cos(CAA, skeptic         ) observed=-0.0640±0.005
  [OK] inject cos(CAA, devils_advocate ) observed=-0.0030±0.005
  [OK] inject cos(CAA, judge           ) observed=-0.0854±0.005
  [OK] ‖ΔH‖ > 1e-3 everywhere

GATE TESTS the executor must run before / after the smoke test:

  G2.0  Hardware precheck (run BEFORE the smoke test, ~1 s; surfaces
        the determinism assumption from "Hardware assumptions" above):

          python -c "
          import torch
          assert torch.cuda.is_available(), 'no CUDA'
          n = torch.cuda.device_count()
          memgb = torch.cuda.get_device_properties(0).total_memory / 2**30
          print(f'{n} GPU(s); primary {memgb:.1f} GB')
          if n > 1:
              print('WARN: multi-GPU detected. device_map=auto will shard 27B/32B'
                    ' if the primary card cannot fit the model in bf16.')
              print('      Either set CUDA_VISIBLE_DEVICES=0 or enable'
                    ' torch.use_deterministic_algorithms(True, warn_only=True).')
          if memgb < 70:
              print(f'WARN: primary GPU has {memgb:.1f} GB; bf16 27B/32B may shard or OOM.')
          "

        Document the printed n / memgb values in your final summary
        so the determinism assumption is auditable.

  G2.1  All [OK] in the printed sanity table — no [FAIL].
  G2.2  No [WARN] indicating MODEL_NAME is not in the
        EXPECTED_COS_AT_INJECTION map.
  G2.3  Inject-layer abs error per persona is ≤ 0.005 (sign included).
  G2.4  ‖ΔH‖ at TARGET_LAYER for the CAA condition should be
        approximately |α_caa| · 1.0 = 2000 ± 1% (Gemma) or
        200 ± 1% (Qwen). Read this off the printed minimum-norm line
        OR re-compute it from the captured tensors. Document the
        observed value.

  G2.5  Chat-template behaviour check (no model load, ~5 s; closes
        the INV-4 silent-failure hole):

          python -c "
          import importlib.util, pathlib
          from transformers import AutoTokenizer
          from config import MODEL_NAME
          spec = importlib.util.spec_from_file_location(
              'driver', pathlib.Path('04_perturbation_propagation.py'))
          mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
          tok = AutoTokenizer.from_pretrained(MODEL_NAME)
          if tok.pad_token is None: tok.pad_token = tok.eos_token
          p = mod.build_prompt(tok, 'What is your view on the following topic?')
          assert isinstance(p, str) and len(p) > 0, p
          if 'qwen' in MODEL_NAME.lower():
              assert '<think>' not in p, ('Qwen prompt contains <think> — '
                                          'INV-4 wrapper not applied!')
          print(f'chat-template OK on {MODEL_NAME}; len={len(p)}')
          "

If G2.1 / G2.3 fail with WRONG SIGN on any cosine: STOP. Do not
self-correct. Wrong sign almost always means INV-3 is misapplied or
the wrong vector or coef was loaded.

If G2.4 fails (‖ΔH^CAA‖ at TARGET_LAYER is far from |α_caa|): STOP.
Almost always means INV-1 (hook-order) is wrong — capture is reading
PRE-steered residual instead of POST.

════════════════════════════════════════════════════════════════════════
PHASE 3 — Gemma extended smoke (5 prompts) + determinism test
════════════════════════════════════════════════════════════════════════

  python 04_perturbation_propagation.py --max-base 5

GATE TESTS:

  G3.1  Output files exist:
          ls ../results/perturbation_propagation/{deltaH_cosine_per_prompt.npz,deltaH_norm_per_prompt.npz,per_layer_summary.json,meta.json}
          ls ../results/perturbation_propagation/checkpoints/ | wc -l   # == 10 (5 base × 2 orderings)

  G3.2  Output schema. Read per_layer_summary.json and confirm:
          - "model" == "google/gemma-2-27b-it"
          - "target_layer" == 22
          - len(layers_to_capture) == 24
          - personas_compared_to_caa == ["skeptic","devils_advocate","judge"]
          - For each persona, len(cosine[persona]["mean"]) == 24

  G3.3  Determinism. Re-run prompt 0 once more by deleting its
        checkpoint and resuming:
          rm ../results/perturbation_propagation/checkpoints/prompt_00000.npz
          python 04_perturbation_propagation.py --max-base 5 --resume
          # Compare the new prompt_00000.npz cosine to the value the
          # other 9 prompts saw on the second pass (deterministic
          # forward should produce bit-equal results).
        Implement this as a small inline test (eg a Python one-liner
        comparing the cos array across two runs). Document the
        observed max abs diff (should be 0.0 exactly; non-zero
        indicates non-determinism in the model forward, which is a
        known multi-GPU artifact — flag it but do not block).

  G3.4  Norm-decay sanity: read deltaH_norm_per_prompt.npz; for the
        CAA condition the norm at layers_to_capture[0] should be
        |α_caa| ≈ 2000 within ±1% on Gemma. The same condition's
        norm at the final layer is unconstrained (can be larger or
        smaller). Document the ratio.

  G3.5  No NaN / Inf in any output array.

If G3.1–G3.5 all pass, commit the smoke artifacts:

  git add scripts/04_perturbation_propagation.py \
          scripts/test_perturbation_propagation.py \
          ../results/perturbation_propagation/per_layer_summary.json \
          ../results/perturbation_propagation/meta.json
  git commit -m "Add per-layer ΔH-cosine driver + tests; Gemma smoke pass

  Implements IMPLEMENTATION_SPEC.md §5; respects INV-1/2/3.
  Smoke test on 5 base questions (10 rows) passes the §6 sanity
  battery on all 3 critical personas to ±0.005."

(Don't push yet — Phase 4 will produce the full-run results to push
together.)

════════════════════════════════════════════════════════════════════════
PHASE 4 — Gemma full run + push
════════════════════════════════════════════════════════════════════════

  python 04_perturbation_propagation.py 2>&1 | tee \
      ../results/perturbation_propagation/run_log_gemma.txt

If interrupted: `python 04_perturbation_propagation.py --resume`.
On network errors during push: retry with exponential backoff
(2 s, 4 s, 8 s, 16 s); never `--no-verify`, never force-push.

GATE TESTS:

  G4.1  per_layer_summary.json["n_prompts"] == 600.

  G4.2  injection-layer cosine, recomputed from the npz, still
        matches the §4.3 expected values to ±0.005 — re-run the
        tighter check with all 600 prompts averaged in instead of
        just prompt 0:

          python -c "
          import numpy as np, json
          d = np.load('../results/perturbation_propagation/deltaH_cosine_per_prompt.npz')
          layers = d['layers'].tolist(); inj = layers.index(22)
          summary = json.load(open('../results/perturbation_propagation/per_layer_summary.json'))
          expected = {'skeptic':-0.0640,'devils_advocate':-0.0030,'judge':-0.0854}
          for ri,p in enumerate([n.decode() if isinstance(n,bytes) else n for n in d['persona_names']]):
              obs = float(d['cos'][ri,:,inj].mean())
              err = abs(obs - expected[p])
              print(f'{p:18s} observed={obs:+.4f}  expected={expected[p]:+.4f}  err={err:.4f}')
              assert err < 5e-3, (p, obs, expected[p])
          "

  G4.3  ‖ΔH‖ does not collapse on any layer × condition (mean over
        all 600 prompts must be > 1e-3 everywhere).

  G4.4  Cross-prompt CI is non-degenerate (some layers should have
        nonzero sem; if every sem is 0.0 across 600 prompts, the
        aggregation is bugged).

Commit and push:

  git add ../results/perturbation_propagation/run_log_gemma.txt \
          ../results/perturbation_propagation/per_layer_summary.json \
          ../results/perturbation_propagation/meta.json
  # NB: deltaH_*.npz and checkpoints/ are large; rely on existing
  # .gitignore to exclude *.npz, or add to .gitignore explicitly.
  git commit -m "Add per-layer ΔH cosine experiment (CAA vs persona) on Gemma — full run

  600-prompt run, all 24 layers, 5 conditions per prompt
  (baseline + caa + skeptic + devils_advocate + judge + random_0).
  All §6 sanity checks pass on the full aggregate (injection-layer
  cosine matches §4.3 to <5e-3 absolute on all 3 personas)."
  git push -u origin claude/design-implementation-spec-EWNnw

════════════════════════════════════════════════════════════════════════
PHASE 5 — Mirror to Qwen
════════════════════════════════════════════════════════════════════════

  cd sycophancy-qwen/scripts

Pre-flight:

  T5.1  Vectors present (T0.4 already covered this; re-confirm):
          python - <<EOF
          import torch
          for n in ['caa','skeptic','devils_advocate','judge','random_0']:
              v = torch.load(f'../vectors/steering/{n}_unit.pt', weights_only=False)
              assert v.shape[0] == 5120 and abs(float(v.float().norm().item())-1.0) < 1e-3, (n, v.shape, v.float().norm())
          print('Qwen vectors OK')
          EOF

  T5.2  Locked coefs:
          python -c "import json; d=json.load(open('../results/best_coefs_test.json'))['best_coefs']; \
            assert d['caa']==-200.0 and d['skeptic']==200.0; print('Qwen coefs OK', d)"

Copy the EXACT same 04_perturbation_propagation.py and
test_perturbation_propagation.py from the Gemma run:

  cp ../../sycophancy-gemma/experiment-main/scripts/04_perturbation_propagation.py \
     scripts/04_perturbation_propagation.py
  cp ../../sycophancy-gemma/experiment-main/scripts/test_perturbation_propagation.py \
     scripts/test_perturbation_propagation.py

  diff ../../sycophancy-gemma/experiment-main/scripts/04_perturbation_propagation.py \
       scripts/04_perturbation_propagation.py
  # diff must produce zero output. The driver is byte-identical.
  # Per-model variation comes from config.py and the on-disk vectors/coefs.

Re-run unit tests on Qwen too (still no GPU):

  python -m pytest scripts/test_perturbation_propagation.py -q
  # or: python scripts/test_perturbation_propagation.py
  # Gate: all U1..U8 pass.

Hardware + chat-template gates on Qwen (mirror of G2.0 / G2.5):

  G5.0  Hardware precheck — same command as G2.0. Document n / memgb.

  G5.5  Chat-template behaviour — same command as G2.5. CRITICAL on
        Qwen: the assert '<think>' not in p line MUST pass. Failure
        here means INV-4's enable_thinking=False is not being
        applied, the model is seeing a thinking-augmented prompt,
        and your downstream cosines are silently incomparable to
        the paper's setup. STOP.

Smoke (1 prompt) — focus on the SIGN-FLIP MIX-UP CHECK:

  python 04_perturbation_propagation.py --smoke-only --max-base 1

  EXPECTED ON QWEN (note positive signs — opposite of Gemma):

    [OK] inject ΔH^CAA exactness: max abs err < 5e-2
    [OK] inject cos(CAA, skeptic         ) observed=+0.1049±0.005
    [OK] inject cos(CAA, devils_advocate ) observed=+0.1078±0.005
    [OK] inject cos(CAA, judge           ) observed=+0.0423±0.005
    [OK] ‖ΔH‖ > 1e-3 everywhere

  GATE: if any cosine has Gemma's sign on Qwen (negative for skeptic
  / devils_advocate / judge), STOP. You have loaded the wrong
  vectors or coefs.

Extended smoke (5 prompts) + same gates as G3.1–G3.5, with the
appropriate Qwen-side substitutions:
  - layers_to_capture has 32 entries (32..63)
  - ‖ΔH^CAA‖ at TARGET_LAYER ≈ |α_caa| ≈ 200 ± 1%
  - target_layer == 32, model == "Qwen/Qwen3-32B"

Full run + push (mirror of Phase 4 with run_log_qwen.txt and the
Qwen-specific G4.2 expected map {skeptic:+0.1049, devils_advocate:+0.1078,
judge:+0.0423}):

  python 04_perturbation_propagation.py 2>&1 | tee \
      ../results/perturbation_propagation/run_log_qwen.txt
  # ... gate tests ...
  git add scripts/04_perturbation_propagation.py \
          scripts/test_perturbation_propagation.py \
          ../results/perturbation_propagation/run_log_qwen.txt \
          ../results/perturbation_propagation/per_layer_summary.json \
          ../results/perturbation_propagation/meta.json
  git commit -m "Add per-layer ΔH cosine experiment (CAA vs persona) on Qwen — full run

  Same driver as the Gemma run (byte-identical script). Qwen-side
  expected cosines are sign-flipped relative to Gemma because the
  underlying vector cosines have opposite sign, but the analytic
  identity sign(α_caa·α_persona)·cos(v_caa,v_persona) holds on both.
  All §6 sanity checks pass."
  git push -u origin claude/design-implementation-spec-EWNnw

════════════════════════════════════════════════════════════════════════
PHASE 6 — Aggregate into sycophancy-clean-results
════════════════════════════════════════════════════════════════════════

CPU-only phase. Reads from the two source repos'
results/perturbation_propagation/ directories. Writes:

  data/gemma-2-27b-it_perturbation_propagation.json
  data/qwen3-32b_perturbation_propagation.json
  scripts/build_perturbation_propagation.py     (you write)
  scripts/make_perturbation_propagation_figs.py (you write)
  scripts/make_perturbation_propagation_table.py (you write)
  figures/fig9_perturbation_cosine.{pdf,png}
  figures/fig10_perturbation_norm.{pdf,png}
  results/perturbation_propagation.csv
  results/perturbation_propagation.md

The three new scripts mirror the existing build_data.py / make_figures.py /
make_tables.py separation. STUDY THE EXISTING SCRIPTS BEFORE WRITING:

  cd sycophancy-clean-results
  cat scripts/build_data.py     # path conventions, json schema
  cat scripts/_style.py         # PALETTE, LABELS, save() helper
  cat scripts/make_figures.py   # matplotlib pattern, side-by-side panels
  cat scripts/make_tables.py    # csv + md emission

Behavioural requirements for the three scripts:

  build_perturbation_propagation.py
    Reads:  ../sycophancy-gemma/experiment-main/results/perturbation_propagation/
              {per_layer_summary.json, deltaH_cosine_per_prompt.npz}
            ../sycophancy-qwen/results/perturbation_propagation/
              {per_layer_summary.json, deltaH_cosine_per_prompt.npz}
    Writes: data/{gemma-2-27b-it, qwen3-32b}_perturbation_propagation.json
    Each output JSON must include:
      - the full per_layer_summary content (cosine, norm, layers, ...)
      - a "provenance" subset: {model, source_repo, source_branch,
        source_commit (from `git rev-parse HEAD` in the source repo),
        n_prompts}
      - "injection_layer_sanity_check": {persona: {observed, expected,
        abs_err}} re-derived from the npz so we can independently
        confirm the §4.3 numbers survive end-to-end. Use the
        sign-flipped expected values from INV-3.

  make_perturbation_propagation_figs.py
    Reads data/{model}_perturbation_propagation.json and emits
    fig9 (cosine vs layer, two-panel side-by-side, Gemma left/Qwen
    right) and fig10 (‖ΔH‖ vs layer, same layout). Reuse PALETTE,
    LABELS, save() from _style.py. The "random_0" condition should
    map to PALETTE["random"] / LABELS["random"]. Vertical dashed
    line at TARGET_LAYER on each panel.

  make_perturbation_propagation_table.py
    Reads data/{model}_perturbation_propagation.json and emits
    results/perturbation_propagation.{csv,md} — one row per
    (model, persona) pair with columns:
      model, persona, n_prompts,
      inject_layer, inject_cos_observed, inject_cos_expected, inject_cos_abs_err,
      midpoint_layer, midpoint_cos_mean, midpoint_cos_ci_lo, midpoint_cos_ci_hi,
      final_layer, final_cos_mean, final_cos_ci_lo, final_cos_ci_hi,
      argmax_layer, argmax_cos_mean,
      norm_persona_inject, norm_persona_final, norm_persona_decay,
      norm_caa_inject, norm_caa_final, norm_caa_decay
    `*_decay` = norm_final / norm_inject; values << 1 mean fig9's
    final-layer cosine is being read off small vectors and should be
    interpreted accordingly.

GATE TESTS for Phase 6 (run after the three scripts):

  G6.1  All four data files / two figures / two tables exist:
          ls data/{gemma-2-27b-it,qwen3-32b}_perturbation_propagation.json
          ls figures/fig9_perturbation_cosine.{pdf,png}
          ls figures/fig10_perturbation_norm.{pdf,png}
          ls results/perturbation_propagation.{csv,md}

  G6.2  injection_layer_sanity_check passes on all 6 cells:
          python -c "
          import json
          for slug,exp in [('gemma-2-27b-it', {'skeptic':-0.0640,'devils_advocate':-0.0030,'judge':-0.0854}),
                           ('qwen3-32b',      {'skeptic':+0.1049,'devils_advocate':+0.1078,'judge':+0.0423})]:
              d = json.load(open(f'data/{slug}_perturbation_propagation.json'))
              for p,e in exp.items():
                  san = d['injection_layer_sanity_check'][p]
                  assert abs(san['observed'] - e) < 5e-3, (slug, p, san)
                  print(f'{slug} {p:18s} OK  observed={san[\"observed\"]:+.4f}  expected={e:+.4f}')
          "

  G6.3  Provenance fields are populated (no <unavailable>):
          python -c "
          import json
          for slug in ['gemma-2-27b-it','qwen3-32b']:
              d = json.load(open(f'data/{slug}_perturbation_propagation.json'))
              for k in ['source_repo','source_branch','source_commit','n_prompts']:
                  v = d.get(k)
                  assert v and 'unavailable' not in str(v), (slug, k, v)
                  print(f'{slug} {k}: {v}')
          "

  G6.4  Figures rendered without error and are non-empty
        (size > a few KB on disk).

  G6.5  results/perturbation_propagation.md is well-formed Markdown
        and has 6 data rows (2 models × 3 personas) plus header rows.

────────────────────────────────────────────────────────────────────────
README updates (sycophancy-clean-results/README.md)
────────────────────────────────────────────────────────────────────────

Apply FOUR surgical edits to README.md:

  EDIT 1 — Source repositories list. Update the bullet list near the
  top to include the public mirror:

    old (verbatim, currently around line 17-25):
      Source repositories (read only, not modified):

      - Gemma pipeline: `../sycophancy-gemma/experiment-main/`
        (multi-seed aggregate, per-seed JSONs, tune-locked best coefficients,
        steering-vector cosine matrix).
      - Qwen pipeline:  `../sycophancy-qwen/`
        (same structure; target layer 32, coefficient grid rescaled 10×
        because of Qwen's smaller activation norms — see the parent paper
        repo for details).

    new — append a third bullet to that list:
      - **General experiment code (public mirror)**:
        <https://github.com/vkmk1/Sycophancy-Steering>
        (the same Gemma pipeline that powers `../sycophancy-gemma/`,
        published under the project's primary GitHub home).

  EDIT 2 — New section "Perturbation propagation (fig9–10)". Insert
  AFTER the "Family averaging (fig7)" section's last paragraph
  (ending in "scripts/make_steering_curves.py:_family_series (mean
  and min/max) and _plot_family (rendering).") and BEFORE the
  "Qualitative samples" section.

  Section content (write this faithfully — it is the single place
  in clean-results that summarises the new experiment):

      ## Perturbation propagation (fig9–10)

      A reviewer raised the concern that geometric near-orthogonality
      of the persona and CAA steering *vectors* at the injection
      layer (paper §4.3, all |cos| < 0.17 on Gemma; |cos| ≤ 0.108 on
      Qwen) does **not** by itself imply mechanistic independence —
      24+ nonlinear blocks downstream can collapse orthogonal inputs
      onto shared pathways. fig9 addresses this directly: for each
      prompt in the held-out test set, we capture the post-block
      residual at every layer ℓ ≥ TARGET_LAYER under baseline,
      CAA-steered, and three persona-steered (skeptic,
      devils_advocate, judge) forward passes, take ΔH = H_steer −
      H_base, and plot the per-layer mean cosine between ΔH^CAA and
      ΔH^persona, averaged within prompt over tokens then across
      prompts. fig10 plots ‖ΔH_ℓ‖ per layer per condition as a
      magnitude check — high cosine is uninformative if both
      perturbations have decayed to ~0.

      **Sanity check.** At the injection layer ℓ = TARGET_LAYER,
      ΔH^CAA is by construction equal to α_CAA · v_CAA (broadcast
      across tokens), so the per-token cosine reduces analytically
      to sign(α_CAA · α_persona) · cos(v_CAA, v_persona) — the §4.3
      vector cosine, with sign flipped because the locked CAA
      coefficient is negative while persona coefficients are
      positive. The driver halts on a >0.005 absolute mismatch with
      the paper-reported number on either model. The two source-repo
      runs pass this check on all 6 (model, persona) cells; per-cell
      `observed`, `expected`, and `abs_err` values are stored in
      `data/{model}_perturbation_propagation.json["injection_layer_sanity_check"]`
      and reproduced in `results/perturbation_propagation.md`.

      **What fig9 / fig10 / the table show.** See
      `results/perturbation_propagation.md` for the per-cell numbers
      (injection / midpoint / final-layer cosines, plus the
      `‖ΔH_final‖ / ‖ΔH_inject‖` decay ratio per persona — a
      condition with decay ratio << 1 means the cosine reading at
      the final layer is being computed on small vectors and should
      be read with that caveat). fig9 shows the trajectory; fig10
      shows that the perturbations do not collapse to zero at any
      layer.

      This is a curve-and-sanity-check report. Mechanistic
      conclusions belong in the parent paper repo
      (`../role-based-steering/paper/`), not here.

      **Reproduction.**
      ```bash
      python3 scripts/build_perturbation_propagation.py
      python3 scripts/make_perturbation_propagation_figs.py
      python3 scripts/make_perturbation_propagation_table.py
      ```

  EDIT 3 — Directory layout tree. Extend the existing tree to
  include the new files. Insert the new lines under the appropriate
  existing subdirectory blocks (data/, scripts/, figures/, results/).
  Use surgical Edit-tool patches — find the smallest unique snippet
  around each insertion point.

      data/   add:
        ├── gemma-2-27b-it_perturbation_propagation.json   per-layer cosine + norm summary
        └── qwen3-32b_perturbation_propagation.json        same, Qwen

      scripts/   add:
        ├── build_perturbation_propagation.py     aggregates per-layer summaries from source repos
        ├── make_perturbation_propagation_figs.py builds fig9 + fig10
        └── make_perturbation_propagation_table.py builds results/perturbation_propagation.{csv,md}

      figures/   add:
        ├── fig9_perturbation_cosine.{pdf,png}    per-layer mean cos(ΔH^CAA, ΔH^persona)
        └── fig10_perturbation_norm.{pdf,png}     per-layer mean ‖ΔH_ℓ‖, collapse check

      results/   add:
        ├── perturbation_propagation.csv          (model × persona) cosine + norm-decay table
        └── perturbation_propagation.md           same, markdown

  EDIT 4 — "How to reproduce" block. Append the three new commands
  to the existing `bash` codeblock at the bottom of that section.
  Find the existing line `python3 scripts/make_steering_curves.py`
  and add three new lines after it:

      python3 scripts/build_perturbation_propagation.py    # rebuilds data/*_perturbation_propagation.json from source repos
      python3 scripts/make_perturbation_propagation_figs.py # rebuilds figures/fig9, fig10
      python3 scripts/make_perturbation_propagation_table.py # rebuilds results/perturbation_propagation.{csv,md}

GATE TESTS for the README edits:

  G6.6  README still parses as Markdown (`grep -c '^##' README.md`
        produces a higher count than before, confirming the new
        section landed).
  G6.7  `git diff README.md` shows exactly the four edits intended,
        no others.

Commit and push:

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

  Aggregates the Phase 4 / Phase 5 outputs from sycophancy-gemma and
  sycophancy-qwen into clean-results. New cross-model figures address
  the reviewer concern that geometric near-orthogonality of CAA vs
  persona steering vectors at the injection layer (paper §4.3) does
  not by itself imply mechanistic independence: cos(ΔH^CAA, ΔH^persona)
  is now reported at every layer downstream, with a magnitude check
  (‖ΔH_ℓ‖) to rule out collapse-to-zero. Source provenance (model,
  repo, branch, commit, n_prompts) is recorded in
  data/{model}_perturbation_propagation.json. Sanity check passes on
  all 6 (model, persona) cells to <5e-3 absolute."
  git push -u origin claude/design-implementation-spec-EWNnw

════════════════════════════════════════════════════════════════════════
PHASE 7 — Source-repo README "See also" sections
════════════════════════════════════════════════════════════════════════

For BOTH `sycophancy-gemma/experiment-main/README.md` and
`sycophancy-qwen/README.md`: append the following section at the END
of the file. If a "See also" section already exists, add only the
missing bullets without duplicating any.

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

GATE TEST:

  G7.1  Both READMEs end with the new "See also" block; no duplicate
        bullets if the section already existed.

Commit and push (one commit per repo):

  cd sycophancy-gemma/experiment-main
  git add README.md
  git commit -m "README: add See-also links to clean-results, role-based-steering, public mirror"
  git push origin claude/design-implementation-spec-EWNnw

  cd ../../sycophancy-qwen
  git add README.md
  git commit -m "README: add See-also links to clean-results, role-based-steering, public mirror"
  git push origin claude/design-implementation-spec-EWNnw

════════════════════════════════════════════════════════════════════════
PHASE 8 — Final report back to the human
════════════════════════════════════════════════════════════════════════

In your final message, report:

  - n_prompts actually run on each model (should be 600 each unless
    --max-base or interrupt-trim).
  - The injection-layer sanity-check table for all 6 (model, persona)
    cells: observed / expected / abs_err.
  - The 6-row headline table from results/perturbation_propagation.md.
  - The 5 commits pushed (Gemma full run, Qwen full run, clean-results
    aggregation, Gemma README, Qwen README) with their SHAs.
  - Any deviation from this prompt or from IMPLEMENTATION_SPEC.md
    that you made and why.

DO NOT draw conclusions about whether persona steering and CAA
"really" share a mechanism. Curves and sanity checks only.
Interpretation belongs in role-based-steering/paper/, not here.

════════════════════════════════════════════════════════════════════════
What NOT to do
════════════════════════════════════════════════════════════════════════

- Do NOT copy results/best_coefs_test.json or vectors/ between source
  repos.
- Do NOT regenerate data/eval_data.json. Use the committed version.
- Do NOT touch sycophancy-qwen/scripts/01_prepare_steering_vectors.py.
- Do NOT batch prompts. One prompt at a time, per spec §5.5.
- Do NOT skip the §6 / Phase-2 / Phase-3 / Phase-5 sanity checks.
- Do NOT push to main on any of the three repos; the working branch
  on all three is claude/design-implementation-spec-EWNnw.
- Do NOT modify vkmk1/Sycophancy-Steering. Link-only.
- Do NOT amend or force-push.
- Do NOT use --no-verify or skip hooks.
- Do NOT self-correct on a wrong-sign cosine. STOP and report.
- Do NOT draw mechanistic conclusions in the clean-results README.
- Do NOT widen any tolerance to make a failing test pass. Fix the
  bug instead.

════════════════════════════════════════════════════════════════════════
Test inventory (recap)
════════════════════════════════════════════════════════════════════════

If you finished the experiment, you have run all of these:

  Phase 0: T0.1 .. T0.7   (pre-flight verification)
  Phase 1: U1 .. U8       (CPU unit tests on the driver; U7 mandatory)
  Phase 2: G2.0 .. G2.5   (hardware precheck + 1-prompt smoke + sanity
                           battery + chat-template behaviour, Gemma)
  Phase 3: G3.1 .. G3.5   (5-prompt extended smoke + determinism, Gemma)
  Phase 4: G4.1 .. G4.4   (full-run aggregate gates, Gemma)
  Phase 5: G5.0, G5.5,
           G2.x/G3.x/G4.x  (mirror of Phase 1–4 tests on Qwen, with
            mirrors        sign-flipped expected cosines + the
                           Qwen-critical "<think> not in prompt" check)
  Phase 6: G6.1 .. G6.7   (clean-results aggregation gates + README)
  Phase 7: G7.1           (source-repo README touch-ups)

Including the persona-by-persona sanity tables, this run involves
≈ 50 explicit gate tests. None of them are optional. If a gate is
ambiguous, default to the stricter interpretation.

If anything in IMPLEMENTATION_SPEC.md contradicts what you observe in
the source files, trust the source files and report the contradiction
in the relevant commit message — do not silently work around it.

If you have any uncertainty at any phase boundary, surface the
question to the human BEFORE running the next phase. The cost of
pausing is low; the cost of running the wrong experiment for 10
GPU-hours is high.

===END PROMPT===
```

---

## Notes on this prompt (for your reference, not for the executor)

What I changed vs the previous version:

- **Removed all four code blocks** (the 430-line driver, the build/figures/table
  scripts, ~1100 lines of verbatim code). The executor now writes those
  itself, faithful to `IMPLEMENTATION_SPEC.md` and the three pinned
  invariants.
- **Pinned the three non-obvious invariants** (hook-order, in-hook fp32
  cast, sign-flip arithmetic). These are the things that, if got wrong,
  produce silently plausible output. Re-deriving them from the spec is
  expensive and easy to miss.
- **Added a Phase 0 pre-flight battery** — 7 read-only checks the
  executor runs against the existing repos before writing any code.
  Confirms the spec's assumptions still hold and the §4.3 source
  numbers are still where I said they are. Catches "the source repos
  drifted" failures fast.
- **Added 8 CPU-only unit tests** for the cosine helper + sign-flip
  arithmetic + module-loading. The executor must include these in a
  `test_perturbation_propagation.py` deliverable.
- **Added gate tests at every phase boundary** (G2.x, G3.x, G4.x,
  G6.x, G7.x) — ≈50 explicit checks total. The executor cannot
  advance to the next phase without them passing, and is told NOT to
  weaken a test to make it pass.
- **Kept the README content** (the section text and the four edit
  hooks) because it is pure prose, doesn't benefit from regeneration
  by the executor, and pinning it ensures the public-mirror link
  to vkmk1/Sycophancy-Steering lands in the right place.
- **The §6.2 expected-cosine table is still verbatim** in the prompt
  (under INV-3) — re-deriving it requires reading
  `caa_decomposition.json` and `vector_cosine_similarities.json`,
  which Phase 0 verifies but doesn't replicate the sign-flip logic for.

Net effect: the prompt is roughly half the size of the previous version
(~750 lines vs 1459) but does more of the heavy lifting that prevents
silent wrong answers. The executor writes the code; the prompt makes
sure the code can't pass its own tests if it has the kind of bug that
would produce a plausible-but-wrong figure.

Want me to commit and push this `RUN_PROMPT.md` (overwrites the previous
hand-coded version) to `sycophancy-clean-results` on
`claude/design-implementation-spec-EWNnw`?
