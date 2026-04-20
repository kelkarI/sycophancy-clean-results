"""Filter source experiment outputs down to the main-result subset.

Reads from the Gemma and Qwen pipelines in sibling directories on this
filesystem (kelkarI/sycophancy-gemma and kelkarI/sycophancy-qwen) and
writes cleaned JSONs to data/. No re-running, no model loading -- pure
aggregation over existing outputs.

Conditions kept (both models):
  critical roles : skeptic, devils_advocate, judge
  conformist     : peacekeeper, pacifist, collaborator
  targeted CAA   : caa
  null control   : random (mean over random_0..9, per seed)
  baseline       : coef=0 on any real condition (identical across conds
                   because the baseline is the same forward pass)

Conditions dropped: assistant_axis, contrarian, scientist, facilitator,
all *_residual, all *_caa_component.

For each (model, kept condition) we emit per-seed:
  best_coef (tune-locked from best_coefs_tune_aggregate.json)
  post_steer_logit, post_steer_rate
  delta_logit = post_steer_logit - baseline_logit (same seed)
  delta_rate_pp = (post_steer_rate - baseline_rate) * 100
and aggregate across seeds:
  mean_delta_logit, std, 95% CI (t-distribution, df=2)
  mean_delta_rate_pp, std, CI
  n_significant (Holm-corrected across the 14-condition primary family
                 of the original pipeline; kept conditions are a subset
                 of that family, so Holm significance carries over
                 directly)
  per_seed_wilcoxon_p_adj (Holm-adjusted per seed)
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


GEMMA_ROOT = Path("/lambda/nfs/filesystem/sycophancy-gemma/experiment-main")
QWEN_ROOT  = Path("/lambda/nfs/filesystem/sycophancy-qwen")
OUT_ROOT   = Path(__file__).resolve().parent.parent / "data"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("gemma-2-27b-it", GEMMA_ROOT, 22),
    ("qwen3-32b",      QWEN_ROOT,  32),
]

CRITICAL_KEPT   = ["skeptic", "devils_advocate", "judge"]
CONFORMIST_KEPT = ["peacekeeper", "pacifist", "collaborator"]
REAL_KEPT       = CRITICAL_KEPT + CONFORMIST_KEPT + ["caa"]
N_RANDOM        = 10

# Student-t 97.5 percentile at df=2 (for n=3 samples)
T975_DF2 = 4.302652729911275


def ci95_small_n(vals):
    """Two-sided 95% t-interval for the mean of a small sample.
    Returns (mean, low, high, std, n). With n=3, df=2, t*=4.303."""
    a = np.asarray(vals, dtype=float)
    n = a.size
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    m = float(a.mean())
    if n == 1:
        return m, m, m, 0.0, 1
    s = float(a.std(ddof=1))
    se = s / np.sqrt(n)
    # For n=3 use t at df=2; for larger n a lookup is fine, but we'll
    # fall back to the normal z=1.96 at n>=30 as a practical rule.
    if n < 30:
        from math import sqrt
        # df-2 t for n>3 would need a lookup; this package has only n=3
        # and n=30 realistic cases, so branch on those two.
        if n == 3:
            tcrit = T975_DF2
        else:
            # conservative: use t at df=n-1 via a couple of hard-coded
            # percentiles if needed; for n in the 4..9 range this would
            # over-narrow the CI if we used z. But our only relevant n
            # values are 3 (per-condition) and 30 (random pool). Fall
            # through to normal for anything >= 30.
            tcrit = 1.96
    else:
        tcrit = 1.96
    return m, m - tcrit * se, m + tcrit * se, s, n


def load_model(name, root):
    agg = json.loads((root / "results/multiseed_aggregate_test.json").read_text())
    seeds = agg["seeds"]
    per_cond = agg["per_condition"]

    # tune-locked best coefs (aggregate — same schema on both)
    bc_path = root / "results/best_coefs_tune_aggregate.json"
    bc = json.loads(bc_path.read_text())["best_coefs"]

    # per-seed rates (needed for baseline + random pool)
    per_seed_rates = {}
    for s in seeds:
        p = root / f"results/seed_{s}/sycophancy_rates_test.json"
        per_seed_rates[s] = json.loads(p.read_text())

    # baseline per seed: coef=0 of any real condition (all identical by
    # construction — the baseline forward pass is the same)
    baselines_logit = []
    baselines_rate = []
    for s in seeds:
        r = per_seed_rates[s]
        first_real = next(c for c in r if not c.startswith("random_"))
        baselines_logit.append(float(r[first_real]["0.0"]["mean_syc_logit"]))
        baselines_rate.append(float(r[first_real]["0.0"]["binary_rate"]))

    conds = {}
    # Real kept conditions from the aggregate (uses tune-locked coef per seed)
    for cond in REAL_KEPT:
        if cond not in per_cond:
            raise KeyError(f"{name}: missing {cond} in aggregate")
        c = per_cond[cond]
        per_seed = []
        for i, seed in enumerate(seeds):
            per_seed.append({
                "seed": seed,
                "best_coef": c["best_coefs"][i],
                "post_steer_logit": c["logits"][i],
                "post_steer_rate": c["rates"][i],
                "delta_logit": c["logits"][i] - baselines_logit[i],
                "delta_rate_pp": (c["rates"][i] - baselines_rate[i]) * 100,
                "wilcoxon_p_adj": c["wilcoxon_p_adj"][i],
            })
        conds[cond] = {
            "condition": cond,
            "family": ("critical" if cond in CRITICAL_KEPT
                       else "conformist" if cond in CONFORMIST_KEPT
                       else "caa"),
            "locked_best_coef_aggregate": float(bc.get(cond)),
            "per_seed": per_seed,
            "n_significant_after_holm_per_seed": c["n_significant"],
        }

    # Random-control aggregate: per seed, take mean over random_0..9 at
    # the random-mean operating point (coef with best mean |excess|).
    # Simpler and closer to the paper convention: for each seed compute
    # the mean post-steer logit/rate of random_0..9 at their SHARED
    # best-absolute-excess coefficient, then average across seeds.
    # Since random selector is "unsigned", each random's best is its
    # own large-|excess|. For a clean null band we just average at the
    # coefficient whose |excess| is largest on each seed -- but simpler
    # still: report mean across coefficients and randoms per seed to
    # give a "typical random intervention" null distribution. Concretely:
    # for each seed, pool all random_{0..9} at every non-zero coef and
    # take the mean of the Δ values. That is the null "any random
    # steering, any coef" distribution.
    per_seed_rand = []
    for i, s in enumerate(seeds):
        r = per_seed_rates[s]
        bl_logit = baselines_logit[i]
        bl_rate  = baselines_rate[i]
        deltas_logit = []
        deltas_rate = []
        for ri in range(N_RANDOM):
            rc = r.get(f"random_{ri}", {})
            for coef_k, cell in rc.items():
                if coef_k == "0.0":
                    continue
                deltas_logit.append(float(cell["mean_syc_logit"]) - bl_logit)
                deltas_rate.append((float(cell["binary_rate"]) - bl_rate) * 100)
        per_seed_rand.append({
            "seed": s,
            "n_random_x_coef": len(deltas_logit),
            "mean_delta_logit": float(np.mean(deltas_logit)),
            "mean_delta_rate_pp": float(np.mean(deltas_rate)),
        })
    conds["random"] = {
        "condition": "random",
        "family": "null_control",
        "description": "per seed: mean Δ over 10 random unit-Gaussian "
                       "steering vectors at all 8 non-zero coefs (n=80 "
                       "per seed); aggregated across seeds.",
        "per_seed": per_seed_rand,
    }

    return {
        "model": name,
        "target_layer": MODELS_BY_NAME[name]["layer"],
        "seeds": seeds,
        "baselines_per_seed": [
            {"seed": s, "logit": bl, "rate": br}
            for s, bl, br in zip(seeds, baselines_logit, baselines_rate)
        ],
        "baseline_mean_logit": float(np.mean(baselines_logit)),
        "baseline_mean_rate":  float(np.mean(baselines_rate)),
        "conditions": conds,
    }


MODELS_BY_NAME = {m[0]: {"layer": m[2]} for m in MODELS}


def enrich_with_aggregates(block):
    """Compute cross-seed mean/std/CI for each condition."""
    for cond_key, cd in block["conditions"].items():
        if cond_key == "random":
            logits = [p["mean_delta_logit"] for p in cd["per_seed"]]
            rates  = [p["mean_delta_rate_pp"] for p in cd["per_seed"]]
        else:
            logits = [p["delta_logit"] for p in cd["per_seed"]]
            rates  = [p["delta_rate_pp"] for p in cd["per_seed"]]
        m, lo, hi, s, n = ci95_small_n(logits)
        cd["delta_logit_mean"] = m
        cd["delta_logit_std"]  = s
        cd["delta_logit_ci95_lo"] = lo
        cd["delta_logit_ci95_hi"] = hi
        m, lo, hi, s, n = ci95_small_n(rates)
        cd["delta_rate_mean_pp"] = m
        cd["delta_rate_std_pp"]  = s
        cd["delta_rate_ci95_lo_pp"] = lo
        cd["delta_rate_ci95_hi_pp"] = hi
        cd["n_seeds"] = n
    return block


def main():
    for name, root, _ in MODELS:
        print(f"=== {name} ===")
        block = load_model(name, root)
        block = enrich_with_aggregates(block)
        out = OUT_ROOT / f"{name.replace('/', '-').replace(':','-')}_clean.json"
        out.write_text(json.dumps(block, indent=2))
        print(f"wrote {out}")
        for c, cd in block["conditions"].items():
            if c == "random":
                print(f"  {c:18s}  family={cd['family']:15s}  "
                      f"Δlogit={cd['delta_logit_mean']:+.3f} [{cd['delta_logit_ci95_lo']:+.3f}, {cd['delta_logit_ci95_hi']:+.3f}]  "
                      f"Δrate={cd['delta_rate_mean_pp']:+.2f}pp")
            else:
                print(f"  {c:18s}  family={cd['family']:15s}  "
                      f"Δlogit={cd['delta_logit_mean']:+.3f} [{cd['delta_logit_ci95_lo']:+.3f}, {cd['delta_logit_ci95_hi']:+.3f}]  "
                      f"Δrate={cd['delta_rate_mean_pp']:+.2f}pp  "
                      f"sig={cd['n_significant_after_holm_per_seed']}/{cd['n_seeds']}")

    # Cosine similarity — also filter to the kept 6 roles + caa
    for name, root, _ in MODELS:
        cos_path = root / "results/vector_cosine_similarities.json"
        if not cos_path.exists():
            print(f"{name}: no cosine file")
            continue
        cos = json.loads(cos_path.read_text())
        names = cos["names"]
        mat = np.array(cos["matrix"])
        kept = REAL_KEPT
        idx = [names.index(k) for k in kept if k in names]
        kept_names = [names[i] for i in idx]
        sub = mat[np.ix_(idx, idx)]
        out = OUT_ROOT / f"{name.replace('/', '-').replace(':','-')}_cosines.json"
        out.write_text(json.dumps({
            "model": name,
            "names": kept_names,
            "matrix": sub.tolist(),
        }, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
