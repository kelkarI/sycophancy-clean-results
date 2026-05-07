"""Extract baseline residual stream norms at TARGET_LAYER.

For each (model, prompt) computes the mean over tokens of ‖h_TARGET_LAYER[t]‖
where h is the baseline (no steering) residual stream output of
model.model.layers[TARGET_LAYER]. Averages across all prompts in
data/eval_data.json. Uses the same forward-pass setup as the layerwise
experiment so the resulting norm is what ε = |α| / ‖h̄‖ should be
divided by.

Output: a JSON dict per model with mean / std / sem / per-prompt array.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tok, q):
    chat = [{"role": "user", "content": q}]
    try:
        return tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)


def extract(model_name, root, target_layer):
    print(f"=== {model_name}  ROOT={root}  TARGET={target_layer} ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"loaded in {time.time()-t0:.1f}s")

    eval_data = sorted(json.load(open(f"{root}/data/eval_data.json")),
                       key=lambda r: int(r["question_id"]))

    # Single-layer hook on TARGET_LAYER, capture mean per-token L2 norm
    # in-hook (no need to keep [T, H] tensors around).
    captured = {"mean_norm": None, "T": None}

    def hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        # t: [batch=1, seq=T, hidden=H]
        per_tok = t[0].detach().to(torch.float32).norm(dim=-1)  # [T]
        captured["mean_norm"] = float(per_tok.mean().item())
        captured["T"] = int(per_tok.shape[0])
        return None

    h = model.model.layers[target_layer].register_forward_hook(hook)

    per_prompt_means = np.zeros(len(eval_data), dtype=np.float64)
    per_prompt_T = np.zeros(len(eval_data), dtype=np.int32)

    t1 = time.time()
    with torch.no_grad():
        for i, row in enumerate(eval_data):
            p = build_prompt(tok, row["question_text"])
            inputs = tok(p, return_tensors="pt").to(device)
            model(**inputs)
            per_prompt_means[i] = captured["mean_norm"]
            per_prompt_T[i] = captured["T"]
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(eval_data)}  elapsed={time.time()-t1:.1f}s",
                      flush=True)

    h.remove()
    print(f"forward pass loop: {time.time()-t1:.1f}s")

    n = len(per_prompt_means)
    mean = float(per_prompt_means.mean())
    std = float(per_prompt_means.std(ddof=1))
    sem = std / np.sqrt(n)
    print(f"\n‖h_baseline_TARGET={target_layer}‖ across {n} prompts:")
    print(f"  mean = {mean:.2f}   std = {std:.2f}   sem = {sem:.2f}")
    print(f"  CI95 = [{mean - 1.96*sem:.2f}, {mean + 1.96*sem:.2f}]")
    print(f"  T (token count) range: [{int(per_prompt_T.min())}, {int(per_prompt_T.max())}]")

    return {
        "model": model_name,
        "target_layer": int(target_layer),
        "n_prompts": int(n),
        "h_baseline_norm_at_target_layer": {
            "mean": mean,
            "std": std,
            "sem": sem,
            "ci95_lo": mean - 1.96 * sem,
            "ci95_hi": mean + 1.96 * sem,
            "definition": "E_prompt[E_token[‖h[t]‖_2]] at layer TARGET_LAYER, baseline (no steering) forward pass on data/eval_data.json (600 prompts)",
        },
        "per_prompt_mean_norm": per_prompt_means.tolist(),
        "per_prompt_T": per_prompt_T.tolist(),
    }


def main():
    out = {}
    for slug, model, root, target in [
        ("gemma-2-27b-it", "google/gemma-2-27b-it",
         "/home/ubuntu/sycophancy-gemma/experiment-main", 22),
        ("qwen3-32b", "Qwen/Qwen3-32B",
         "/home/ubuntu/sycophancy-qwen", 32),
    ]:
        out[slug] = extract(model, root, target)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write to clean-results data dir for build_normalized_effects.py to pick up
    out_path = "/home/ubuntu/sycophancy-clean-results/data/_baseline_norms_at_target_layer.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
