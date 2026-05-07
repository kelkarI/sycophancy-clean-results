"""Aggregate the per-layer ΔH-cosine experiment outputs from the two
source repos into clean-results/data/{model}_perturbation_propagation.json.

Source files (read-only):
  ../sycophancy-gemma/experiment-main/results/perturbation_propagation/{per_layer_summary.json, deltaH_cosine_per_prompt.npz, meta.json}
  ../sycophancy-qwen/results/perturbation_propagation/                {per_layer_summary.json, deltaH_cosine_per_prompt.npz, meta.json}

For each model we emit:
  - the full per_layer_summary content (cosine + norm time series with CIs)
  - "provenance": model, source_repo, source_branch, source_commit, n_prompts
  - "injection_layer_sanity_check": per persona, the observed cosine at
    TARGET_LAYER (re-derived from the npz), the analytical expected value
    from INV-3 (sign(α_caa·α_persona) · paper §4.3 cosine), and abs_err.

Deterministic: pure aggregation, no RNG, no model loads.
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


CRITICAL_PERSONAS = ["skeptic", "devils_advocate", "judge"]

# Sign-flipped paper-§4.3 cosines (mirrors driver's EXPECTED_COS_AT_INJECTION).
# These are the analytical expected values at the injection layer given
# the locked coef signs (α_caa<0, α_persona>0) — see RUN_PROMPT INV-3.
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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Try absolute Lambda paths first (production), fall back to local sibling
# clones (this repo on a dev box). Mirrors the convention in
# scripts/make_steering_curves.py.
GEMMA_RES_ABS = Path("/lambda/nfs/filesystem/sycophancy-gemma/experiment-main/results/perturbation_propagation")
QWEN_RES_ABS  = Path("/lambda/nfs/filesystem/sycophancy-qwen/results/perturbation_propagation")
GEMMA_RES_REL = ROOT.parent / "sycophancy-gemma/experiment-main/results/perturbation_propagation"
QWEN_RES_REL  = ROOT.parent / "sycophancy-qwen/results/perturbation_propagation"
GEMMA_REPO_ABS = Path("/lambda/nfs/filesystem/sycophancy-gemma")
QWEN_REPO_ABS  = Path("/lambda/nfs/filesystem/sycophancy-qwen")
GEMMA_REPO_REL = ROOT.parent / "sycophancy-gemma"
QWEN_REPO_REL  = ROOT.parent / "sycophancy-qwen"


def _resolve(abs_path: Path, rel_path: Path) -> Path:
    if abs_path.exists():
        return abs_path
    if rel_path.exists():
        return rel_path
    raise FileNotFoundError(
        f"Neither {abs_path} nor {rel_path} exists. The Phase 4/5 source-repo "
        "runs must complete (and write per_layer_summary.json) before this "
        "script is useful.")


def _git_info(repo: Path):
    """Best-effort (commit, branch) from a local clone. Returns
    ('<unavailable>', '<unavailable>') if .git doesn't exist or git fails."""
    if not (repo / ".git").exists():
        return "<unavailable>", "<unavailable>"
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "<unavailable>"
    try:
        branch = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        branch = "<unavailable>"
    return commit, branch


def _build_one(slug: str, model_name: str, res_dir: Path, repo: Path) -> dict:
    summary_path = res_dir / "per_layer_summary.json"
    npz_path = res_dir / "deltaH_cosine_per_prompt.npz"
    meta_path = res_dir / "meta.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} missing. The {slug} source-repo run hasn't completed.")

    summary = json.loads(summary_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Re-derive injection-layer cosine from npz so the sanity check is
    # independent of summary's own number — catches drift between summary
    # and per-prompt arrays.
    inj_check = {}
    expected_map = EXPECTED_COS_AT_INJECTION.get(model_name, {})
    if npz_path.exists() and expected_map:
        d = np.load(npz_path)
        layers = d["layers"].tolist()
        inj_idx = layers.index(int(summary["target_layer"]))
        persona_names = [
            (n.decode() if isinstance(n, (bytes, np.bytes_)) else str(n))
            for n in d["persona_names"]
        ]
        for persona, exp in expected_map.items():
            if persona not in persona_names:
                continue
            ri = persona_names.index(persona)
            obs = float(d["cos"][ri, :, inj_idx].mean())
            inj_check[persona] = {
                "observed": obs,
                "expected": float(exp),
                "abs_err":  float(abs(obs - exp)),
            }

    commit, branch = _git_info(repo)

    out = {
        # ---- provenance ----
        "model": model_name,
        "source_repo": str(repo),
        "source_branch": branch,
        "source_commit": commit,
        "n_prompts": int(summary.get("n_prompts", 0)),
        # ---- per-layer summary ----
        "target_layer": int(summary["target_layer"]),
        "layers": list(summary["layers"]),
        "personas_compared_to_caa": list(summary["personas_compared_to_caa"]),
        "conditions_for_norm": list(summary["conditions_for_norm"]),
        "cosine": summary["cosine"],
        "norm":   summary["norm"],
        # ---- sanity check ----
        "injection_layer_sanity_check": inj_check,
        # ---- meta passthrough (subset, for reproducibility) ----
        "coefficients": meta.get("coefficients", {}),
        "torch_dtype": meta.get("torch_dtype", "<unavailable>"),
        "intervention_type": meta.get("intervention_type", "<unavailable>"),
        "positions": meta.get("positions", "<unavailable>"),
    }
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    targets = [
        ("gemma-2-27b-it", "google/gemma-2-27b-it",
         _resolve(GEMMA_RES_ABS, GEMMA_RES_REL),
         GEMMA_REPO_ABS if GEMMA_REPO_ABS.exists() else GEMMA_REPO_REL),
        ("qwen3-32b", "Qwen/Qwen3-32B",
         _resolve(QWEN_RES_ABS, QWEN_RES_REL),
         QWEN_REPO_ABS if QWEN_REPO_ABS.exists() else QWEN_REPO_REL),
    ]
    for slug, model_name, res_dir, repo in targets:
        out = _build_one(slug, model_name, res_dir, repo)
        out_path = DATA / f"{slug}_perturbation_propagation.json"
        out_path.write_text(json.dumps(out, indent=2))
        n = out["n_prompts"]
        sanity = out["injection_layer_sanity_check"]
        print(f"  wrote {out_path}  n_prompts={n}  sanity_personas="
              f"{list(sanity)}")
        for p, v in sanity.items():
            print(f"    {p:<18s}  observed={v['observed']:+.4f}  "
                  f"expected={v['expected']:+.4f}  "
                  f"abs_err={v['abs_err']:.4f}")


if __name__ == "__main__":
    main()
