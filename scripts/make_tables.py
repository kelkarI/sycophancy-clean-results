"""Emit the two paper-ready tables:

  results/main_table.csv            one row per (model, condition)
  results/conformist_vs_critical.csv family-level summary

Both accompanied by .md siblings for easy pasting.
"""
import csv
import json
from pathlib import Path

import numpy as np

from _style import DEFAULT_ORDER, LABELS, CRITICAL, CONFORMIST

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "gemma-2-27b-it": DATA / "gemma-2-27b-it_clean.json",
    "qwen3-32b":      DATA / "qwen3-32b_clean.json",
}


def load(p):
    return json.loads(Path(p).read_text())


def main_table():
    fieldnames = [
        "model", "family", "condition", "display_name",
        "locked_best_coef",
        "delta_logit_mean", "delta_logit_std",
        "delta_logit_ci95_lo", "delta_logit_ci95_hi",
        "delta_rate_mean_pp", "delta_rate_std_pp",
        "delta_rate_ci95_lo_pp", "delta_rate_ci95_hi_pp",
        "per_seed_delta_logit",
        "per_seed_wilcoxon_p_holm_adj",
        "n_significant_after_holm_seeds",
        "n_seeds",
    ]
    rows = []
    for model, p in MODELS.items():
        block = load(p)
        for c in DEFAULT_ORDER:
            cd = block["conditions"][c]
            if c == "random":
                per_seed_d = [s["mean_delta_logit"] for s in cd["per_seed"]]
                per_seed_p = []
                n_sig = None
                coef = ""
            else:
                per_seed_d = [s["delta_logit"] for s in cd["per_seed"]]
                per_seed_p = [s["wilcoxon_p_adj"] for s in cd["per_seed"]]
                n_sig = cd.get("n_significant_after_holm_per_seed")
                coef = cd.get("locked_best_coef_aggregate", "")
            rows.append({
                "model": model,
                "family": cd["family"],
                "condition": c,
                "display_name": LABELS[c],
                "locked_best_coef": coef,
                "delta_logit_mean": round(cd["delta_logit_mean"], 4),
                "delta_logit_std":  round(cd["delta_logit_std"], 4),
                "delta_logit_ci95_lo": round(cd["delta_logit_ci95_lo"], 4),
                "delta_logit_ci95_hi": round(cd["delta_logit_ci95_hi"], 4),
                "delta_rate_mean_pp": round(cd["delta_rate_mean_pp"], 3),
                "delta_rate_std_pp":  round(cd["delta_rate_std_pp"], 3),
                "delta_rate_ci95_lo_pp": round(cd["delta_rate_ci95_lo_pp"], 3),
                "delta_rate_ci95_hi_pp": round(cd["delta_rate_ci95_hi_pp"], 3),
                "per_seed_delta_logit": ";".join(f"{v:+.4f}" for v in per_seed_d),
                "per_seed_wilcoxon_p_holm_adj": ";".join(
                    (f"{p:.3e}" if p is not None else "") for p in per_seed_p
                ),
                "n_significant_after_holm_seeds": n_sig if n_sig is not None else "",
                "n_seeds": cd["n_seeds"],
            })
    csv_path = OUT / "main_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}  ({len(rows)} rows)")

    # Compact markdown too
    md_path = OUT / "main_table.md"
    def fmt_sig(nsig, nseeds):
        if nsig == "" or nsig is None:
            return ""
        if nsig == nseeds: return "***"
        if nsig == nseeds - 1: return "**"
        if nsig >= 1: return "*"
        return "ns"
    lines = [
        "# Main table — condition × model",
        "",
        "Negative Δ = steering reduces sycophancy. Error bars are 95% CIs "
        "across 3 test seeds (t-distribution, df=2). Holm correction is "
        "applied within each seed's 14-condition primary family from the "
        "source pipeline; columns show how many of the 3 seeds crossed "
        "α=0.05 after correction. *** = 3/3, ** = 2/3, * = 1/3, ns = 0/3.",
        "",
        "| Model | Family | Condition | Best coef | Δlogit (mean ± std) "
        "[95% CI] | Δrate (pp) | Holm-sig seeds |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        dl = (f"{r['delta_logit_mean']:+.3f} ± {r['delta_logit_std']:.3f} "
              f"[{r['delta_logit_ci95_lo']:+.3f}, {r['delta_logit_ci95_hi']:+.3f}]")
        dr = (f"{r['delta_rate_mean_pp']:+.2f} "
              f"[{r['delta_rate_ci95_lo_pp']:+.2f}, "
              f"{r['delta_rate_ci95_hi_pp']:+.2f}]")
        sig = fmt_sig(r['n_significant_after_holm_seeds'], r['n_seeds'])
        n_over = (f"{r['n_significant_after_holm_seeds']}/3 {sig}"
                  if r['n_significant_after_holm_seeds'] != "" else "—")
        coef = r['locked_best_coef'] if r['locked_best_coef'] != "" else "—"
        lines.append(
            f"| {r['model']} | {r['family']} | {r['display_name']} | "
            f"{coef} | {dl} | {dr} | {n_over} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


def conformist_vs_critical_table():
    """Family-level summary: mean (and min/max) Δlogit within each family,
    plus a pooled test of the family mean against zero."""
    fieldnames = [
        "model", "family", "n_conditions", "n_seeds",
        "family_mean_delta_logit", "family_mean_delta_rate_pp",
        "family_min_delta_logit", "family_max_delta_logit",
        "members_holm_sig_3of3",
    ]
    rows = []
    for model, p in MODELS.items():
        block = load(p)
        conds = block["conditions"]
        for fam, members in [("critical",   CRITICAL),
                             ("caa",        ["caa"]),
                             ("conformist", CONFORMIST),
                             ("null_control", ["random"])]:
            logits = [conds[c]["delta_logit_mean"] for c in members]
            rates  = [conds[c]["delta_rate_mean_pp"] for c in members]
            sig3 = 0
            for c in members:
                if c == "random":
                    continue
                if conds[c].get("n_significant_after_holm_per_seed") == 3:
                    sig3 += 1
            rows.append({
                "model": model,
                "family": fam,
                "n_conditions": len(members),
                "n_seeds": 3,
                "family_mean_delta_logit": round(float(np.mean(logits)), 4),
                "family_mean_delta_rate_pp": round(float(np.mean(rates)), 3),
                "family_min_delta_logit": round(float(np.min(logits)), 4),
                "family_max_delta_logit": round(float(np.max(logits)), 4),
                "members_holm_sig_3of3": sig3,
            })
    csv_path = OUT / "conformist_vs_critical.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}  ({len(rows)} rows)")

    md_path = OUT / "conformist_vs_critical.md"
    lines = [
        "# Family-level summary — critical vs conformist roles",
        "",
        "Aggregated across the kept members of each family at their own "
        "tune-locked coefficients. `members_holm_sig_3of3` counts how many "
        "of the family's members reached Holm significance in all 3 test "
        "seeds.",
        "",
        "| Model | Family | n cond | mean Δlogit | range Δlogit | mean Δrate (pp) | Holm-sig 3/3 count |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['family']} | {r['n_conditions']} "
            f"| {r['family_mean_delta_logit']:+.3f} "
            f"| [{r['family_min_delta_logit']:+.3f}, "
            f"{r['family_max_delta_logit']:+.3f}] "
            f"| {r['family_mean_delta_rate_pp']:+.2f} "
            f"| {r['members_holm_sig_3of3']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


def main():
    main_table()
    conformist_vs_critical_table()


if __name__ == "__main__":
    main()
