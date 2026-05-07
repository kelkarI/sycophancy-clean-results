"""Build per-(model, condition) normalized effect-size data.

Joins two existing data sources per model:

  SOURCE A — baseline residual stream norm at TARGET_LAYER
             read from data/_baseline_norms_at_target_layer.json,
             extracted by `extract_baseline_norms.py` (a one-off
             standalone forward-pass-only run on the held-out
             eval set, see PERTURBATION_PROPAGATION.md). 600 prompts
             × per-token L2 norm × averaged over (tokens, prompts).

  SOURCE B — ∆logit at the layerwise experiment's locked α
             read from each source repo's per-seed
             results/seed_{42,7,123}/sycophancy_rates_test.json,
             aggregated as: mean over 3 test seeds of
             (mean_syc_logit[cond][α] - mean_syc_logit[cond]["0.0"]).
             The locked α per condition is from
             sycophancy-{gemma,qwen}/results/best_coefs_test.json
             (top-level aggregate file). NOTE: for Gemma
             peacekeeper the aggregate file says α=+5000 but each
             of the 3 test seeds has α=+2000 in its own
             best_coefs_test.json — the layerwise experiment
             uses the aggregate (5000), so we mirror that here.
             This divergence is documented in
             PERTURBATION_PROPAGATION.md.

Output: data/{slug}_normalized_effects.json with the schema in the
extension prompt. Q1 = perturbation_energy ε = |α| / ‖h̄‖, Q2 =
behavioral_efficiency = |∆logit| / ε, Q3 = argmax_cosine_with_caa
+ argmax_layer (re-used from the layerwise experiment's outputs).

Pure post-processing. No GPU forward passes, no recomputation of
∆logit values from scratch.
"""
import json
import statistics as st
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

GEMMA_REPO_REL = ROOT.parent / "sycophancy-gemma"
QWEN_REPO_REL  = ROOT.parent / "sycophancy-qwen"
GEMMA_RES = GEMMA_REPO_REL / "experiment-main/results"
QWEN_RES  = QWEN_REPO_REL / "results"

# Test-seed dirs used by the original paper run (each contains a
# sycophancy_rates_test.json with mean_syc_logit per condition per coef).
TEST_SEEDS = ["seed_42", "seed_7", "seed_123"]

# Same condition list as the layerwise experiment.
CRITICAL_PERSONAS = ["skeptic", "devils_advocate", "judge"]
CONFORMIST_ROLES  = ["peacekeeper", "pacifist", "collaborator", "facilitator"]
RANDOM_CONTROL    = "random_0"
ALL_PERSONAS = CRITICAL_PERSONAS + CONFORMIST_ROLES + [RANDOM_CONTROL]
ALL_CONDITIONS = ["caa"] + ALL_PERSONAS  # 9 cells per model


def _git_info(repo: Path):
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


def _load_baseline_norm(slug: str) -> Dict:
    """SOURCE A. Returns the per-prompt-mean-of-per-token-L2-norms aggregate."""
    p = DATA / "_baseline_norms_at_target_layer.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Run /home/ubuntu/extract_baseline_norms.py first "
            "(see PERTURBATION_PROPAGATION.md, deviation log: this required "
            "an additional 5-min forward-pass-only GPU job because the "
            "layerwise experiment captured ‖ΔH‖ but not ‖h^baseline‖)."
        )
    d = json.loads(p.read_text())
    if slug not in d:
        raise KeyError(f"{slug} not in {p}")
    return d[slug]


def _load_locked_alpha(repo_results: Path) -> Dict[str, float]:
    """The locked α used by the layerwise experiment, per condition."""
    return json.loads((repo_results / "best_coefs_test.json").read_text())["best_coefs"]


def _aggregate_delta_logit(repo_results: Path, conditions: List[str],
                           locked: Dict[str, float],
                           random_alpha: float) -> Dict[str, Dict]:
    """SOURCE B. Per condition, mean over test seeds of
       (mean_syc_logit[cond][α] - mean_syc_logit[cond]["0.0"])."""
    out = {}
    for cond in conditions:
        # Random_0's coef is computed by the layerwise driver; everything
        # else uses each condition's own locked value.
        alpha = random_alpha if cond == RANDOM_CONTROL else float(locked[cond])
        per_seed = []
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
        if not per_seed:
            out[cond] = {"alpha": alpha, "delta_logit_mean": None,
                         "delta_logit_std": None, "n_seeds": 0,
                         "per_seed": []}
            continue
        m = st.mean(per_seed)
        sd = st.stdev(per_seed) if len(per_seed) > 1 else 0.0
        out[cond] = {
            "alpha": alpha,
            "delta_logit_mean": m,
            "delta_logit_std": sd,
            "delta_logit_ci95_lo": m - 1.96 * (sd / max(np.sqrt(len(per_seed)), 1)),
            "delta_logit_ci95_hi": m + 1.96 * (sd / max(np.sqrt(len(per_seed)), 1)),
            "n_seeds": len(per_seed),
            "per_seed": per_seed,
        }
    return out


def _argmax_cos(slug: str, condition: str):
    """Q3: from existing layerwise data."""
    p = DATA / f"{slug}_perturbation_propagation.json"
    if not p.exists():
        return {"argmax_cosine_with_caa": None, "argmax_layer": None,
                "inj_cos": None, "final_cos": None}
    d = json.loads(p.read_text())
    if condition == "caa":
        # CAA's cosine with itself is trivially +1; doesn't go in fig11
        # but record for symmetry.
        return {"argmax_cosine_with_caa": 1.0,
                "argmax_layer": int(d["target_layer"]),
                "inj_cos": 1.0, "final_cos": 1.0}
    if condition not in d.get("cosine", {}):
        return {"argmax_cosine_with_caa": None, "argmax_layer": None,
                "inj_cos": None, "final_cos": None}
    cs = d["cosine"][condition]
    layers = d["layers"]
    means = np.array(cs["mean"])
    argmax_idx = int(np.argmax(np.abs(means)))
    inj_idx = layers.index(int(d["target_layer"]))
    return {
        "argmax_cosine_with_caa": float(means[argmax_idx]),
        "argmax_layer": int(layers[argmax_idx]),
        "inj_cos": float(means[inj_idx]),
        "final_cos": float(means[-1]),
    }


def _build_one(slug: str, model_name: str, repo: Path,
               repo_results: Path) -> Dict:
    norm_block = _load_baseline_norm(slug)
    h_norm = float(norm_block["h_baseline_norm_at_target_layer"]["mean"])
    h_norm_sem = float(norm_block["h_baseline_norm_at_target_layer"]["sem"])
    target_layer = int(norm_block["target_layer"])

    locked = _load_locked_alpha(repo_results)
    # random_0 α = mean(|critical-coef|) — must match the layerwise driver.
    random_alpha = float(np.mean([abs(locked[r]) for r in CRITICAL_PERSONAS]))

    delta_logit = _aggregate_delta_logit(
        repo_results, ALL_CONDITIONS, locked, random_alpha)

    conditions = {}
    for cond in ALL_CONDITIONS:
        dl = delta_logit.get(cond, {})
        alpha = dl.get("alpha")
        dlm = dl.get("delta_logit_mean")
        epsilon = abs(alpha) / h_norm if alpha is not None else None
        # Q2: |∆logit| / ε. Sign of ∆logit is also informative; we save
        # the absolute value as the metric (sycophancy reduction is
        # always |∆logit|>0; sign records direction).
        eff = (abs(dlm) / epsilon) if (dlm is not None and epsilon and epsilon > 0) else None
        cos = _argmax_cos(slug, cond)
        conditions[cond] = {
            "alpha": alpha,
            "perturbation_energy": epsilon,
            "delta_logit_mean": dlm,
            "delta_logit_std": dl.get("delta_logit_std"),
            "delta_logit_ci95_lo": dl.get("delta_logit_ci95_lo"),
            "delta_logit_ci95_hi": dl.get("delta_logit_ci95_hi"),
            "delta_logit_per_seed": dl.get("per_seed"),
            "n_seeds": dl.get("n_seeds"),
            "behavioral_efficiency": eff,
            **cos,
        }

    commit, branch = _git_info(repo)

    return {
        "model": model_name,
        "target_layer": target_layer,
        "h_baseline_norm_at_target_layer": h_norm,
        "h_baseline_norm_sem": h_norm_sem,
        "n_prompts_for_norm": int(norm_block["n_prompts"]),
        "random_control_alpha": random_alpha,
        "test_seeds": list(TEST_SEEDS),
        "n_test_seeds": len(TEST_SEEDS),
        "conditions": conditions,
        # provenance
        "source_repo": str(repo),
        "source_branch": branch,
        "source_commit": commit,
        "layerwise_data_source":
            f"data/{slug}_perturbation_propagation.json",
    }


def main():
    out_paths = []
    targets = [
        ("gemma-2-27b-it", "google/gemma-2-27b-it",
         GEMMA_REPO_REL, GEMMA_RES),
        ("qwen3-32b", "Qwen/Qwen3-32B",
         QWEN_REPO_REL, QWEN_RES),
    ]
    for slug, model_name, repo, results in targets:
        out = _build_one(slug, model_name, repo, results)
        out_path = DATA / f"{slug}_normalized_effects.json"
        out_path.write_text(json.dumps(out, indent=2))
        out_paths.append(out_path)
        print(f"  wrote {out_path}")
        eps = out["h_baseline_norm_at_target_layer"]
        print(f"    ‖h̄_baseline_TARGET={out['target_layer']}‖ = {eps:.2f}")
        for cond, c in out["conditions"].items():
            if c["delta_logit_mean"] is None:
                continue
            print(f"    {cond:18s}  α={c['alpha']:+7.1f}  "
                  f"ε={c['perturbation_energy']:.4f}  "
                  f"Δlogit={c['delta_logit_mean']:+7.4f}  "
                  f"|Δlogit|/ε={c['behavioral_efficiency']:7.3f}  "
                  f"argmax_cos@L{c.get('argmax_layer')}={c.get('argmax_cosine_with_caa'):+.3f}"
                  if c.get("argmax_cosine_with_caa") is not None else
                  f"    {cond:18s}  α={c['alpha']:+7.1f}  ε={c['perturbation_energy']:.4f}  Δlogit={c['delta_logit_mean']:+7.4f}  |Δlogit|/ε={c['behavioral_efficiency']:7.3f}")


if __name__ == "__main__":
    main()
